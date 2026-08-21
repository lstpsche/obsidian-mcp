//! End-to-end upgrade orchestration.

use std::collections::BTreeSet;
use std::fs::OpenOptions;
use std::path::{Path, PathBuf};

use fs2::FileExt;

use crate::daemon::bootstrap::{self, BootstrapConfig};

use super::cargo_install::{self, binary_file_name};
use super::{
    BuildIdentity, InstallRecord, RealCommandExecutor, SemanticReconcileResult, ServiceOwner,
    ServiceTarget, UpgradeError, UpgradeReport,
};

#[derive(Debug, Clone)]
pub struct UpgradeOptions {
    pub dry_run: bool,
    pub executable: Option<PathBuf>,
    #[cfg(windows)]
    pub windows_parent_pid: Option<u32>,
}

impl UpgradeOptions {
    pub fn normal(dry_run: bool) -> Self {
        Self {
            dry_run,
            executable: None,
            #[cfg(windows)]
            windows_parent_pid: None,
        }
    }

    #[cfg(windows)]
    pub fn windows_helper(executable: PathBuf, parent_pid: u32) -> Self {
        Self {
            dry_run: false,
            executable: Some(executable),
            windows_parent_pid: Some(parent_pid),
        }
    }
}

#[derive(Debug, Clone)]
pub struct UpgradePlan {
    pub identity: BuildIdentity,
    pub install: InstallRecord,
    pub cargo_args: Vec<std::ffi::OsString>,
    pub services: Vec<ServiceTarget>,
}

#[derive(Debug, Clone)]
pub enum UpgradeRunOutcome {
    DryRun(UpgradePlan),
    Completed(UpgradeReport),
    #[cfg(windows)]
    WindowsHandoff {
        helper: PathBuf,
    },
}

pub async fn run(options: UpgradeOptions) -> Result<UpgradeRunOutcome, UpgradeError> {
    let identity = BuildIdentity::embedded();
    let executable = options
        .executable
        .clone()
        .map(Ok)
        .unwrap_or_else(std::env::current_exe)?;

    #[cfg(windows)]
    if let Some(parent_pid) = options.windows_parent_pid {
        super::helper::await_windows_replacement_window(parent_pid)?;
    }

    let install = cargo_install::discover_install(&executable, &identity)?;
    let executor = RealCommandExecutor;
    let services = discover_services(&executor, &executable)?;
    let cargo_args = cargo_install::cargo_install_args(&install, &identity);
    ensure_no_unmanaged_http(&executable, &services)?;
    let mut plan = UpgradePlan {
        identity: identity.clone(),
        install: install.clone(),
        cargo_args,
        services,
    };
    if options.dry_run {
        return Ok(UpgradeRunOutcome::DryRun(plan));
    }

    #[cfg(windows)]
    if options.windows_parent_pid.is_none() {
        let helper = super::helper::spawn_windows_handoff(&executable)?;
        return Ok(UpgradeRunOutcome::WindowsHandoff { helper });
    }

    let _upgrade_lock = UpgradeLock::acquire(&install.root)?;
    // Revalidate after taking the lock so an interrupted or concurrent Cargo
    // operation cannot invalidate the preflight result.
    let install = cargo_install::discover_install(&executable, &identity)?;
    plan.services = discover_services(&executor, &executable)?;
    ensure_no_unmanaged_http(&executable, &plan.services)?;
    let semantic_path = install
        .root
        .join("bin")
        .join(binary_file_name("obsidian-semanticd"));
    let bootstrap_config = BootstrapConfig::from_env();

    #[cfg(windows)]
    let semantic_stopped = bootstrap::prepare_daemon_for_upgrade(&bootstrap_config, &semantic_path)
        .await
        .map_err(|err| UpgradeError::Activation(err.to_string()))?;

    let outcome = match cargo_install::install_latest(&executor, &install, &identity) {
        Ok(outcome) => outcome,
        Err(err) => {
            #[cfg(windows)]
            if semantic_stopped {
                let restoration =
                    bootstrap::ensure_daemon_from_sibling(&bootstrap_config, &semantic_path).await;
                return Err(UpgradeError::Install(match restoration {
                    Ok(_) => format!(
                        "{err}; the semantic daemon was restored after the failed installation"
                    ),
                    Err(restore_err) => format!(
                        "{err}; additionally, the semantic daemon could not be restored: {restore_err}"
                    ),
                }));
            }
            return Err(err);
        }
    };
    let new_identity = outcome.after.main_identity.clone();
    let current_services = discover_services(&executor, &executable).map_err(|err| {
        UpgradeError::Activation(format!(
            "binaries were installed, but running services could not be rediscovered: {err}"
        ))
    })?;
    merge_service_targets(&mut plan.services, current_services);
    ensure_no_unmanaged_http(&executable, &plan.services).map_err(|err| {
        UpgradeError::Activation(format!(
            "binaries were installed, but runtime activation is incomplete: {err}"
        ))
    })?;

    #[cfg(windows)]
    let semantic = if semantic_stopped {
        match bootstrap::ensure_daemon_from_sibling(&bootstrap_config, &semantic_path).await {
            Ok(result) if result.manifest.daemon_version == new_identity.version => {
                Some(SemanticReconcileResult {
                    success: true,
                    ownership: result.manifest.binary_origin.as_str().into(),
                    was_running: true,
                    restarted: true,
                    observed_version: Some(result.manifest.daemon_version),
                    diagnostic:
                        "locally owned semantic daemon restarted after Windows binary replacement"
                            .into(),
                })
            }
            Ok(result) => Some(SemanticReconcileResult {
                success: false,
                ownership: result.manifest.binary_origin.as_str().into(),
                was_running: true,
                restarted: true,
                observed_version: Some(result.manifest.daemon_version.clone()),
                diagnostic: format!(
                    "semantic daemon reports version '{}', expected '{}'",
                    result.manifest.daemon_version, new_identity.version
                ),
            }),
            Err(err) => Some(failed_semantic_result(err.to_string())),
        }
    } else {
        reconcile_semantic(&bootstrap_config, &semantic_path, &new_identity.version).await
    };

    #[cfg(not(windows))]
    let semantic =
        reconcile_semantic(&bootstrap_config, &semantic_path, &new_identity.version).await;

    let mut service_activation_attempted = false;
    let services = plan
        .services
        .iter()
        .map(|target| {
            if !service_needs_activation(target, &new_identity.version) {
                return super::ActivationResult {
                    target_id: target.id.clone(),
                    success: true,
                    observed_version: target.observed_version.clone(),
                    diagnostic: "already reports the installed version; not restarted".into(),
                };
            }
            service_activation_attempted = true;
            match target.owner {
                ServiceOwner::Launchd => {
                    super::launchd::restart(&executor, target, &new_identity.version)
                }
                ServiceOwner::SystemdUser => {
                    super::systemd::restart(&executor, target, &new_identity.version)
                }
            }
        })
        .collect();
    let semantic_activation_attempted = semantic
        .as_ref()
        .is_some_and(|result| result.restarted || !result.success);

    Ok(UpgradeRunOutcome::Completed(UpgradeReport {
        old_identity: identity,
        new_identity,
        binaries_changed: outcome.changed,
        semantic,
        services,
        stdio_reconnect_required: outcome.changed,
        activation_attempted: service_activation_attempted || semantic_activation_attempted,
    }))
}

fn service_needs_activation(target: &ServiceTarget, expected_version: &str) -> bool {
    target.observed_version.as_deref() != Some(expected_version)
}

fn merge_service_targets(existing: &mut Vec<ServiceTarget>, current: Vec<ServiceTarget>) {
    for target in current {
        if let Some(previous) = existing
            .iter_mut()
            .find(|previous| previous.owner == target.owner && previous.id == target.id)
        {
            *previous = target;
        } else {
            existing.push(target);
        }
    }
    existing.sort_by(|left, right| left.id.cmp(&right.id));
}

fn ensure_no_unmanaged_http(
    executable: &Path,
    services: &[ServiceTarget],
) -> Result<(), UpgradeError> {
    let managed_pids = services
        .iter()
        .filter_map(|target| target.previous_pid)
        .collect::<BTreeSet<_>>();
    let unmanaged = super::processes::unmanaged_http_processes(executable, &managed_pids)?;
    if unmanaged.is_empty() {
        return Ok(());
    }
    Err(UpgradeError::UnsupportedRuntime(format!(
        "ad-hoc HTTP process{} {} use{} this binary, but the upgrader cannot reconstruct inherited settings; stop {}, retry the upgrade, then restart {} with the same launch command",
        if unmanaged.len() == 1 { "" } else { "es" },
        unmanaged
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(", "),
        if unmanaged.len() == 1 { "s" } else { "" },
        if unmanaged.len() == 1 { "it" } else { "them" },
        if unmanaged.len() == 1 { "it" } else { "them" }
    )))
}

async fn reconcile_semantic(
    config: &BootstrapConfig,
    semantic_path: &Path,
    expected_version: &str,
) -> Option<SemanticReconcileResult> {
    match bootstrap::reconcile_daemon_after_upgrade(config, semantic_path, expected_version).await {
        Ok(result) => Some(SemanticReconcileResult {
            success: true,
            ownership: result.origin.as_str().into(),
            was_running: result.was_running,
            restarted: result.restarted,
            observed_version: result.observed_version,
            diagnostic: result.diagnostic,
        }),
        Err(err) => Some(failed_semantic_result(err.to_string())),
    }
}

fn failed_semantic_result(diagnostic: String) -> SemanticReconcileResult {
    SemanticReconcileResult {
        success: false,
        ownership: "unknown".into(),
        was_running: false,
        restarted: false,
        observed_version: None,
        diagnostic,
    }
}

fn discover_services(
    executor: &RealCommandExecutor,
    executable: &Path,
) -> Result<Vec<ServiceTarget>, UpgradeError> {
    let mut services = super::launchd::discover(executor, executable)?;
    services.extend(super::systemd::discover(executor, executable)?);
    Ok(services)
}

struct UpgradeLock {
    file: std::fs::File,
}

impl UpgradeLock {
    fn acquire(root: &Path) -> Result<Self, UpgradeError> {
        let path = root.join(".obsidian-mcp-upgrade.lock");
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&path)?;
        file.try_lock_exclusive().map_err(|err| {
            UpgradeError::Install(format!(
                "another obsidian-mcp upgrade is already using '{}': {err}",
                root.display()
            ))
        })?;
        Ok(Self { file })
    }
}

impl Drop for UpgradeLock {
    fn drop(&mut self) {
        let _ = self.file.unlock();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn service(observed_version: Option<&str>) -> ServiceTarget {
        ServiceTarget {
            owner: ServiceOwner::Launchd,
            id: "test.service".into(),
            definition_path: None,
            executable: "/tmp/obsidian-mcp".into(),
            host: "127.0.0.1".into(),
            port: 37842,
            previous_pid: Some(1),
            observed_version: observed_version.map(str::to_owned),
        }
    }

    #[test]
    fn unchanged_healthy_service_is_not_activated() {
        assert!(!service_needs_activation(&service(Some("2.5.0")), "2.5.0"));
    }

    #[test]
    fn only_old_or_unhealthy_service_is_activated() {
        assert!(!service_needs_activation(&service(Some("2.5.0")), "2.5.0"));
        assert!(service_needs_activation(&service(Some("2.4.0")), "2.5.0"));
        assert!(service_needs_activation(&service(None), "2.5.0"));
    }

    #[test]
    fn refreshed_service_state_replaces_snapshot_and_adds_new_targets() {
        let mut existing = vec![service(Some("2.4.0"))];
        existing[0].id = "existing.service".into();
        let mut refreshed = service(Some("2.5.0"));
        refreshed.id = "existing.service".into();
        refreshed.previous_pid = Some(2);
        let mut added = service(Some("2.5.0"));
        added.id = "new.service".into();

        merge_service_targets(&mut existing, vec![refreshed, added]);

        assert_eq!(existing.len(), 2);
        assert_eq!(existing[0].id, "existing.service");
        assert_eq!(existing[0].previous_pid, Some(2));
        assert_eq!(existing[1].id, "new.service");
    }
}
