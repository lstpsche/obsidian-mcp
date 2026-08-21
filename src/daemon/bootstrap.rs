//! Daemon installer/bootstrap manager for shared semantic runtime.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use semver::Version;
use serde_json::json;
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader};

#[cfg(unix)]
use tokio::net::UnixStream;
#[cfg(windows)]
use tokio::net::windows::named_pipe::ClientOptions;

use crate::client::semantic_daemon::{DaemonConnectPolicy, SemanticDaemonClient};
use crate::config::SemanticRuntimeConfig;
use crate::error::{VaultError, VaultResult};

use super::home::{self, InstallLock, SemanticHomePaths};
use super::manifest::{self, BinaryOrigin, ManifestIpc, RuntimeManifest, RuntimeManifestInput};
use super::protocol::{self, DAEMON_API_VERSION, ERR_INCOMPATIBLE_API_VERSION};
use super::server::IpcEndpoint;

const DEFAULT_DOWNLOAD_BASE_URL: &str =
    "https://github.com/lstpsche/obsidian-mcp/releases/download";

#[derive(Debug, Clone)]
pub struct BootstrapConfig {
    pub semantic_home_override: Option<PathBuf>,
    pub daemon_path_override: Option<PathBuf>,
    pub model_name: String,
    pub download_url_override: Option<String>,
    pub bootstrap_client_name: String,
    pub bootstrap_client_version: String,
}

impl BootstrapConfig {
    pub fn from_env() -> Self {
        let runtime = SemanticRuntimeConfig::load_from_env();
        Self {
            semantic_home_override: runtime.semantic_home_override,
            daemon_path_override: runtime.daemon_path_override,
            model_name: runtime.model_name,
            download_url_override: runtime.daemon_download_url,
            bootstrap_client_name: "obsidian-mcp".to_string(),
            bootstrap_client_version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
}

impl Default for BootstrapConfig {
    fn default() -> Self {
        Self::from_env()
    }
}

#[derive(Debug, Clone)]
pub struct BootstrapResult {
    pub semantic_home: PathBuf,
    pub endpoint: IpcEndpoint,
    pub daemon_binary_path: PathBuf,
    pub manifest: RuntimeManifest,
    pub reused_existing: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DaemonReconcileOutcome {
    pub origin: BinaryOrigin,
    pub was_running: bool,
    pub restarted: bool,
    pub observed_version: Option<String>,
    pub diagnostic: String,
}

#[derive(Debug)]
enum HealthProbeOutcome {
    Healthy(protocol::HealthResult),
    Unreachable,
    Incompatible(String),
    Invalid(String),
}

/// Reconcile a running, locally owned semantic daemon after the Cargo package
/// binaries have been replaced. Explicit overrides and PATH-owned daemons are
/// reported but never mutated.
pub async fn reconcile_daemon_after_upgrade(
    config: &BootstrapConfig,
    installed_daemon: &Path,
    expected_version: &str,
) -> VaultResult<DaemonReconcileOutcome> {
    let semantic_home =
        home::resolve_semantic_home_with_override(config.semantic_home_override.as_deref(), None)?;
    let paths = home::semantic_home_paths(&semantic_home);
    if !paths.manifest_path.is_file() {
        return Ok(DaemonReconcileOutcome {
            origin: BinaryOrigin::Unknown,
            was_running: false,
            restarted: false,
            observed_version: None,
            diagnostic: "no semantic daemon manifest; nothing to restart".into(),
        });
    }
    home::ensure_home_layout(&paths)?;
    let _install_lock = InstallLock::acquire_async(&paths).await?;
    let Some(current_manifest) = manifest::load(&paths.manifest_path)? else {
        return Ok(DaemonReconcileOutcome {
            origin: BinaryOrigin::Unknown,
            was_running: false,
            restarted: false,
            observed_version: None,
            diagnostic: "no semantic daemon manifest; nothing to restart".into(),
        });
    };
    let origin = if config.daemon_path_override.is_some() {
        BinaryOrigin::Override
    } else {
        effective_origin(&current_manifest, &paths, installed_daemon)
    };
    let external = matches!(
        origin,
        BinaryOrigin::Override | BinaryOrigin::Path | BinaryOrigin::Unknown
    );
    let Some(endpoint) = endpoint_from_manifest(&current_manifest) else {
        if external {
            return Ok(DaemonReconcileOutcome {
                origin,
                was_running: false,
                restarted: false,
                observed_version: None,
                diagnostic: "external semantic daemon ownership preserved; manifest endpoint was not inspected".into(),
            });
        }
        return Err(VaultError::DaemonBootstrap(
            "semantic manifest contains an unsupported IPC endpoint".into(),
        ));
    };
    let health = match probe_health(&endpoint).await? {
        HealthProbeOutcome::Healthy(health) => health,
        HealthProbeOutcome::Unreachable => {
            return Ok(DaemonReconcileOutcome {
                origin,
                was_running: false,
                restarted: false,
                observed_version: None,
                diagnostic: "semantic daemon is not running; preserved its manifest and cache"
                    .into(),
            });
        }
        HealthProbeOutcome::Incompatible(message) | HealthProbeOutcome::Invalid(message)
            if external =>
        {
            return Ok(DaemonReconcileOutcome {
                origin,
                was_running: true,
                restarted: false,
                observed_version: None,
                diagnostic: format!(
                    "external semantic daemon ownership preserved; health was not compatible: {message}"
                ),
            });
        }
        HealthProbeOutcome::Incompatible(message) | HealthProbeOutcome::Invalid(message) => {
            return Err(VaultError::DaemonBootstrap(format!(
                "running semantic daemon could not be safely reconciled: {message}"
            )));
        }
    };

    if external {
        return Ok(DaemonReconcileOutcome {
            origin,
            was_running: true,
            restarted: false,
            observed_version: Some(health.daemon_version),
            diagnostic: "external semantic daemon ownership preserved; no restart attempted".into(),
        });
    }
    match compare_daemon_versions(&health.daemon_version, expected_version)? {
        std::cmp::Ordering::Greater => {
            return Ok(DaemonReconcileOutcome {
                origin,
                was_running: true,
                restarted: false,
                observed_version: Some(health.daemon_version),
                diagnostic:
                    "semantic daemon is newer than this package; preserved without downgrade".into(),
            });
        }
        std::cmp::Ordering::Equal => {
            return Ok(DaemonReconcileOutcome {
                origin,
                was_running: true,
                restarted: false,
                observed_version: Some(health.daemon_version),
                diagnostic: "semantic daemon already reports the installed version".into(),
            });
        }
        std::cmp::Ordering::Less => {}
    }
    if !installed_daemon.is_file() {
        return Err(VaultError::DaemonBootstrap(format!(
            "installed sibling daemon '{}' is missing",
            installed_daemon.display()
        )));
    }

    shutdown_owned_daemon(&endpoint, &current_manifest, health.pid).await?;

    let mut child = start_daemon_process(
        installed_daemon,
        &paths,
        &endpoint,
        &current_manifest.model_name,
    )?;
    let pid = child.id().unwrap_or_default();
    tokio::time::sleep(Duration::from_millis(50)).await;
    if let Some(status) = child.try_wait()? {
        return Err(VaultError::DaemonBootstrap(format!(
            "updated semantic daemon exited immediately with {status} (check '{}')",
            paths.daemon_stderr_log_path.display()
        )));
    }
    drop(child);

    let health = wait_for_health(&endpoint, Duration::from_secs(10)).await?;
    if health.daemon_version != expected_version {
        return Err(VaultError::DaemonBootstrap(format!(
            "updated semantic daemon reports version '{}', expected '{expected_version}'",
            health.daemon_version
        )));
    }
    let binary_bytes = std::fs::read(installed_daemon)?;
    let updated_manifest = RuntimeManifest::from_input(RuntimeManifestInput {
        daemon_api_version: health.daemon_api_version,
        daemon_version: health.daemon_version.clone(),
        binary_path: installed_daemon.display().to_string(),
        binary_origin: BinaryOrigin::Sibling,
        binary_sha256: Some(home::sha256_hex(&binary_bytes)),
        ipc: current_manifest.ipc,
        pid,
        semantic_home: current_manifest.semantic_home,
        fastembed_cache_dir: current_manifest.fastembed_cache_dir,
        model_name: current_manifest.model_name,
        bootstrap_client_name: config.bootstrap_client_name.clone(),
        bootstrap_client_version: config.bootstrap_client_version.clone(),
    });
    manifest::save_atomic(&paths.manifest_path, &updated_manifest)?;

    Ok(DaemonReconcileOutcome {
        origin: BinaryOrigin::Sibling,
        was_running: true,
        restarted: true,
        observed_version: Some(health.daemon_version),
        diagnostic: "locally owned semantic daemon restarted with its existing home and model"
            .into(),
    })
}

/// Stop a running local semantic daemon before replacing a locked executable
/// (needed on Windows). Returns whether this call stopped the daemon so the
/// caller can restore it after installation; external daemons are never stopped.
pub async fn prepare_daemon_for_upgrade(
    config: &BootstrapConfig,
    installed_daemon: &Path,
) -> VaultResult<bool> {
    let semantic_home =
        home::resolve_semantic_home_with_override(config.semantic_home_override.as_deref(), None)?;
    let paths = home::semantic_home_paths(&semantic_home);
    if !paths.manifest_path.is_file() {
        return Ok(false);
    }
    home::ensure_home_layout(&paths)?;
    let _install_lock = InstallLock::acquire_async(&paths).await?;
    let Some(current_manifest) = manifest::load(&paths.manifest_path)? else {
        return Ok(false);
    };
    let origin = if config.daemon_path_override.is_some() {
        BinaryOrigin::Override
    } else {
        effective_origin(&current_manifest, &paths, installed_daemon)
    };
    let external = matches!(
        origin,
        BinaryOrigin::Override | BinaryOrigin::Path | BinaryOrigin::Unknown
    );
    let Some(endpoint) = endpoint_from_manifest(&current_manifest) else {
        if external {
            return Ok(false);
        }
        return Err(VaultError::DaemonBootstrap(
            "semantic manifest contains an unsupported IPC endpoint".into(),
        ));
    };
    let health = match probe_health(&endpoint).await? {
        HealthProbeOutcome::Healthy(health) => health,
        HealthProbeOutcome::Unreachable => return Ok(false),
        HealthProbeOutcome::Incompatible(_) | HealthProbeOutcome::Invalid(_) if external => {
            return Ok(false);
        }
        HealthProbeOutcome::Incompatible(message) | HealthProbeOutcome::Invalid(message) => {
            return Err(VaultError::DaemonBootstrap(format!(
                "running semantic daemon could not be safely prepared: {message}"
            )));
        }
    };
    if external {
        return Ok(false);
    }
    shutdown_owned_daemon(&endpoint, &current_manifest, health.pid).await?;
    Ok(true)
}

/// Ensure a shared daemon exists and is healthy.
pub async fn ensure_daemon(config: &BootstrapConfig) -> VaultResult<BootstrapResult> {
    ensure_daemon_inner(config, None).await
}

/// Ensure the daemon using a verified package sibling when a temporary
/// upgrader process cannot discover that sibling beside its own executable.
pub async fn ensure_daemon_from_sibling(
    config: &BootstrapConfig,
    sibling: &Path,
) -> VaultResult<BootstrapResult> {
    ensure_daemon_inner(config, Some(sibling)).await
}

async fn ensure_daemon_inner(
    config: &BootstrapConfig,
    preferred_sibling: Option<&Path>,
) -> VaultResult<BootstrapResult> {
    let semantic_home =
        home::resolve_semantic_home_with_override(config.semantic_home_override.as_deref(), None)?;
    let paths = home::semantic_home_paths(&semantic_home);
    home::ensure_home_layout(&paths)?;

    let _install_lock = InstallLock::acquire_async(&paths).await?;
    let mut existing_manifest = manifest::load(&paths.manifest_path)?;
    let default_endpoint = home::default_ipc_endpoint(&paths);
    let mut preserve_manifest_model = preferred_sibling.is_some();

    if let Some(current_manifest) = existing_manifest.as_mut() {
        let endpoint =
            endpoint_from_manifest(current_manifest).unwrap_or_else(|| default_endpoint.clone());
        match probe_health(&endpoint).await? {
            HealthProbeOutcome::Healthy(health) => {
                let sibling = preferred_sibling
                    .filter(|path| path.is_file())
                    .map(Path::to_path_buf)
                    .or_else(sibling_daemon_path);
                let origin = if config.daemon_path_override.is_some() {
                    BinaryOrigin::Override
                } else if let Some(sibling) = sibling.as_deref() {
                    effective_origin(current_manifest, &paths, sibling)
                } else {
                    current_manifest.binary_origin
                };
                let should_reconcile = if matches!(
                    origin,
                    BinaryOrigin::Sibling | BinaryOrigin::ManagedDownload
                ) {
                    if let Some(sibling) = sibling.as_deref() {
                        let sibling_version = daemon_binary_version(sibling)?;
                        let running_version = parse_daemon_version(
                            "running semantic daemon",
                            &health.daemon_version,
                        )?;
                        sibling_version > running_version
                    } else {
                        false
                    }
                } else {
                    false
                };
                if should_reconcile {
                    preserve_manifest_model = true;
                    tracing::info!(
                        running_version = %health.daemon_version,
                        "newer owned semantic daemon sibling detected; completing activation"
                    );
                    shutdown_owned_daemon(&endpoint, current_manifest, health.pid).await?;
                } else {
                    current_manifest.daemon_version = health.daemon_version;
                    current_manifest.daemon_api_version = health.daemon_api_version;
                    current_manifest.binary_origin = origin;
                    current_manifest.touch_health();
                    manifest::save_atomic(&paths.manifest_path, current_manifest)?;
                    let daemon_binary_path = PathBuf::from(&current_manifest.binary_path);
                    return Ok(BootstrapResult {
                        semantic_home,
                        endpoint,
                        daemon_binary_path,
                        manifest: current_manifest.clone(),
                        reused_existing: true,
                    });
                }
            }
            HealthProbeOutcome::Incompatible(message) => {
                return Err(VaultError::DaemonBootstrap(format!(
                    "existing daemon is incompatible with API v{DAEMON_API_VERSION}: {message}"
                )));
            }
            HealthProbeOutcome::Invalid(message) => {
                tracing::warn!(error = %message, "manifest endpoint responded but was invalid; daemon will be restarted");
            }
            HealthProbeOutcome::Unreachable => {
                tracing::info!("manifest endpoint unreachable; daemon will be started");
            }
        }
    }

    let endpoint = existing_manifest
        .as_ref()
        .and_then(endpoint_from_manifest)
        .unwrap_or(default_endpoint);

    let (mut daemon_binary_path, mut binary_origin) = resolve_daemon_binary(
        config,
        &paths,
        existing_manifest.as_ref(),
        preferred_sibling,
    )?;
    let mut binary_sha256 = existing_manifest.as_ref().and_then(|manifest| {
        (Path::new(&manifest.binary_path) == daemon_binary_path)
            .then(|| manifest.binary_sha256.clone())
            .flatten()
    });

    if !daemon_binary_path.exists() {
        if binary_origin == BinaryOrigin::Override {
            return Err(VaultError::DaemonBootstrap(format!(
                "daemon override path does not exist: {}",
                daemon_binary_path.display()
            )));
        }
        if let Some(path_binary) = find_daemon_on_path() {
            tracing::info!(path = %path_binary.display(), "using daemon binary found on $PATH");
            daemon_binary_path = path_binary;
            binary_origin = BinaryOrigin::Path;
        } else {
            daemon_binary_path = paths.daemon_binary_path.clone();
            binary_origin = BinaryOrigin::ManagedDownload;
            let download_url = resolve_download_url(config)?;
            tracing::info!(url = %download_url, "downloading semantic daemon binary");
            let checksum = download_and_install(&download_url, &daemon_binary_path).await?;
            binary_sha256 = Some(checksum);
        }
    }

    let expected_started_version = if binary_origin == BinaryOrigin::Sibling {
        Some(daemon_binary_version(&daemon_binary_path)?.to_string())
    } else {
        None
    };

    let daemon_model_name = if preserve_manifest_model {
        existing_manifest
            .as_ref()
            .map(|manifest| manifest.model_name.clone())
            .unwrap_or_else(|| config.model_name.clone())
    } else {
        config.model_name.clone()
    };
    let mut child =
        start_daemon_process(&daemon_binary_path, &paths, &endpoint, &daemon_model_name)?;
    let pid = child.id().unwrap_or_default();

    tokio::time::sleep(Duration::from_millis(50)).await;
    match child.try_wait() {
        Ok(Some(status)) => {
            return Err(VaultError::DaemonBootstrap(format!(
                "daemon process exited immediately with {status} \
                 (binary: '{}', check {})",
                daemon_binary_path.display(),
                paths.daemon_stderr_log_path.display()
            )));
        }
        Ok(None) => {}
        Err(err) => {
            tracing::warn!(error = %err, "failed to check daemon process status after spawn");
        }
    }
    drop(child);

    let health = wait_for_health(&endpoint, Duration::from_secs(10)).await?;
    if let Some(expected_version) = expected_started_version
        && health.daemon_version != expected_version
    {
        return Err(VaultError::DaemonBootstrap(format!(
            "started sibling semantic daemon reports version '{}', expected '{expected_version}'",
            health.daemon_version
        )));
    }
    let ipc = ManifestIpc {
        transport: endpoint_transport(&endpoint).to_string(),
        endpoint: endpoint.endpoint_string(),
    };

    let runtime_manifest = RuntimeManifest::from_input(RuntimeManifestInput {
        daemon_api_version: health.daemon_api_version,
        daemon_version: health.daemon_version,
        binary_path: daemon_binary_path.display().to_string(),
        binary_origin,
        binary_sha256,
        ipc,
        pid,
        semantic_home: semantic_home.display().to_string(),
        fastembed_cache_dir: paths.fastembed_cache_dir.display().to_string(),
        model_name: daemon_model_name,
        bootstrap_client_name: config.bootstrap_client_name.clone(),
        bootstrap_client_version: config.bootstrap_client_version.clone(),
    });
    manifest::save_atomic(&paths.manifest_path, &runtime_manifest)?;

    Ok(BootstrapResult {
        semantic_home,
        endpoint,
        daemon_binary_path,
        manifest: runtime_manifest,
        reused_existing: false,
    })
}

fn resolve_daemon_binary(
    config: &BootstrapConfig,
    paths: &SemanticHomePaths,
    existing_manifest: Option<&RuntimeManifest>,
    preferred_sibling: Option<&Path>,
) -> VaultResult<(PathBuf, BinaryOrigin)> {
    if let Some(path) = config.daemon_path_override.as_ref() {
        return Ok((path.clone(), BinaryOrigin::Override));
    }

    if let Some(path) = preferred_sibling.filter(|path| path.is_file()) {
        return Ok((path.to_path_buf(), BinaryOrigin::Sibling));
    }

    if let Some(path) = sibling_daemon_path() {
        return Ok((path, BinaryOrigin::Sibling));
    }

    if let Some(manifest) = existing_manifest {
        let manifest_path = PathBuf::from(&manifest.binary_path);
        if !manifest_path.as_os_str().is_empty() {
            return Ok((manifest_path, manifest.binary_origin));
        }
    }

    Ok((
        paths.daemon_binary_path.clone(),
        BinaryOrigin::ManagedDownload,
    ))
}

fn sibling_daemon_path() -> Option<PathBuf> {
    let current = std::env::current_exe().ok()?;
    let sibling = current.parent()?.join(home::daemon_binary_name());
    sibling.is_file().then_some(sibling)
}

fn daemon_binary_version(binary: &Path) -> VaultResult<Version> {
    let output = std::process::Command::new(binary)
        .arg("--version")
        .output()
        .map_err(|err| {
            VaultError::DaemonBootstrap(format!(
                "failed to inspect semantic daemon binary '{}': {err}",
                binary.display()
            ))
        })?;
    if !output.status.success() {
        return Err(VaultError::DaemonBootstrap(format!(
            "'{} --version' exited unsuccessfully",
            binary.display()
        )));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let version = stdout
        .trim()
        .strip_prefix("obsidian-semanticd ")
        .ok_or_else(|| {
            VaultError::DaemonBootstrap(format!(
                "'{} --version' returned an unexpected identity",
                binary.display()
            ))
        })?;
    parse_daemon_version("semantic daemon binary", version)
}

fn compare_daemon_versions(running: &str, installed: &str) -> VaultResult<std::cmp::Ordering> {
    Ok(
        parse_daemon_version("running semantic daemon", running)?.cmp(&parse_daemon_version(
            "installed semantic daemon",
            installed,
        )?),
    )
}

fn parse_daemon_version(label: &str, version: &str) -> VaultResult<Version> {
    Version::parse(version).map_err(|err| {
        VaultError::DaemonBootstrap(format!(
            "{label} reported invalid semantic version '{version}': {err}"
        ))
    })
}

fn find_daemon_on_path() -> Option<PathBuf> {
    let binary_name = home::daemon_binary_name();
    let path_var = std::env::var_os("PATH")?;
    std::env::split_paths(&path_var)
        .map(|dir| dir.join(binary_name))
        .find(|candidate| candidate.is_file())
}

fn resolve_download_url(config: &BootstrapConfig) -> VaultResult<String> {
    if let Some(url) = config.download_url_override.as_ref()
        && !url.trim().is_empty()
    {
        return Ok(url.trim().to_string());
    }

    let version = env!("CARGO_PKG_VERSION");
    let tag = format!("v{version}");
    let target = target_triple()?;
    let asset = if cfg!(windows) {
        format!("obsidian-semanticd-{version}-{target}.zip")
    } else {
        format!("obsidian-semanticd-{version}-{target}.tar.gz")
    };
    Ok(format!("{DEFAULT_DOWNLOAD_BASE_URL}/{tag}/{asset}"))
}

fn target_triple() -> VaultResult<&'static str> {
    if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        Ok("x86_64-unknown-linux-gnu")
    } else if cfg!(all(target_os = "linux", target_arch = "aarch64")) {
        Ok("aarch64-unknown-linux-gnu")
    } else if cfg!(all(target_os = "macos", target_arch = "x86_64")) {
        Ok("x86_64-apple-darwin")
    } else if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        Ok("aarch64-apple-darwin")
    } else if cfg!(all(target_os = "windows", target_arch = "x86_64")) {
        Ok("x86_64-pc-windows-msvc")
    } else {
        Err(VaultError::DaemonBootstrap(
            "unsupported target for daemon auto-download".to_string(),
        ))
    }
}

async fn download_and_install(url: &str, destination: &Path) -> VaultResult<String> {
    let response = reqwest::get(url).await.map_err(|err| {
        VaultError::DaemonBootstrap(format!("failed to download daemon binary '{url}': {err}"))
    })?;
    if !response.status().is_success() {
        return Err(VaultError::DaemonBootstrap(format!(
            "failed to download daemon binary '{url}': HTTP {}",
            response.status()
        )));
    }

    let bytes = response.bytes().await.map_err(|err| {
        VaultError::DaemonBootstrap(format!("failed to read daemon download bytes: {err}"))
    })?;

    if let Some(parent) = destination.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let checksum = home::sha256_hex(&bytes);
    if url.ends_with(".zip") {
        extract_zip_binary(bytes.as_ref(), destination)?;
    } else if url.ends_with(".tar.gz") {
        extract_tar_gz_binary(bytes.as_ref(), destination)?;
    } else {
        std::fs::write(destination, bytes.as_ref())?;
    }

    #[cfg(unix)]
    make_executable(destination)?;

    Ok(checksum)
}

fn extract_tar_gz_binary(archive_bytes: &[u8], destination: &Path) -> VaultResult<()> {
    let cursor = std::io::Cursor::new(archive_bytes);
    let decoder = flate2::read::GzDecoder::new(cursor);
    let mut archive = tar::Archive::new(decoder);
    let wanted_name = home::daemon_binary_name();
    let mut found = false;

    for entry in archive.entries().map_err(|err| {
        VaultError::DaemonBootstrap(format!("failed to read tar.gz entries: {err}"))
    })? {
        let mut entry = entry.map_err(|err| {
            VaultError::DaemonBootstrap(format!("failed to read tar.gz entry: {err}"))
        })?;
        let path = entry.path().map_err(|err| {
            VaultError::DaemonBootstrap(format!("failed to inspect tar.gz path: {err}"))
        })?;
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if file_name == wanted_name {
            if !entry.header().entry_type().is_file() {
                return Err(VaultError::DaemonBootstrap(format!(
                    "tar entry '{}' is not a regular file (type: {:?}); refusing to extract",
                    file_name,
                    entry.header().entry_type()
                )));
            }
            entry.unpack(destination).map_err(|err| {
                VaultError::DaemonBootstrap(format!(
                    "failed to unpack daemon binary to '{}': {err}",
                    destination.display()
                ))
            })?;
            found = true;
            break;
        }
    }

    if !found {
        return Err(VaultError::DaemonBootstrap(format!(
            "downloaded tar.gz archive does not contain '{}'",
            wanted_name
        )));
    }
    Ok(())
}

fn extract_zip_binary(archive_bytes: &[u8], destination: &Path) -> VaultResult<()> {
    let cursor = std::io::Cursor::new(archive_bytes);
    let mut archive = zip::ZipArchive::new(cursor)
        .map_err(|err| VaultError::DaemonBootstrap(format!("failed to open zip archive: {err}")))?;

    let wanted_name = home::daemon_binary_name();
    let mut found = false;
    for idx in 0..archive.len() {
        let mut file = archive.by_index(idx).map_err(|err| {
            VaultError::DaemonBootstrap(format!("failed to read zip entry: {err}"))
        })?;
        let Some(file_name) = std::path::Path::new(file.name())
            .file_name()
            .and_then(|name| name.to_str())
        else {
            continue;
        };
        if file_name == wanted_name {
            if file.is_dir() {
                return Err(VaultError::DaemonBootstrap(format!(
                    "zip entry '{}' is a directory; refusing to extract",
                    file.name()
                )));
            }
            if file.name().contains("..") {
                return Err(VaultError::DaemonBootstrap(format!(
                    "zip entry '{}' contains path traversal sequence; refusing to extract",
                    file.name()
                )));
            }
            let mut out = std::fs::File::create(destination)?;
            std::io::copy(&mut file, &mut out).map_err(|err| {
                VaultError::DaemonBootstrap(format!(
                    "failed to extract daemon binary to '{}': {err}",
                    destination.display()
                ))
            })?;
            found = true;
            break;
        }
    }

    if !found {
        return Err(VaultError::DaemonBootstrap(format!(
            "downloaded zip archive does not contain '{}'",
            wanted_name
        )));
    }
    Ok(())
}

#[cfg(unix)]
fn make_executable(path: &Path) -> VaultResult<()> {
    use std::os::unix::fs::PermissionsExt;

    let mut permissions = std::fs::metadata(path)?.permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(path, permissions)?;
    Ok(())
}

fn start_daemon_process(
    binary_path: &Path,
    paths: &SemanticHomePaths,
    endpoint: &IpcEndpoint,
    model_name: &str,
) -> VaultResult<tokio::process::Child> {
    let stderr_log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&paths.daemon_stderr_log_path)
        .map_err(|err| {
            VaultError::DaemonBootstrap(format!(
                "failed to open daemon stderr log file '{}': {err}",
                paths.daemon_stderr_log_path.display()
            ))
        })?;

    let mut command = tokio::process::Command::new(binary_path);
    command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::from(stderr_log))
        .env("OBSIDIAN_SEMANTIC_HOME", &paths.root)
        .env("OBSIDIAN_SEMANTIC_ENDPOINT", endpoint.endpoint_string())
        .env("OBSIDIAN_SEMANTIC_MODEL", model_name)
        .env("FASTEMBED_CACHE_DIR", &paths.fastembed_cache_dir);

    if let Ok(log_level) = std::env::var("OBSIDIAN_LOG_LEVEL") {
        command.env("OBSIDIAN_LOG_LEVEL", log_level);
    }

    command.spawn().map_err(|err| {
        VaultError::DaemonBootstrap(format!(
            "failed to spawn semantic daemon '{}': {err}",
            binary_path.display()
        ))
    })
}

async fn wait_for_health(
    endpoint: &IpcEndpoint,
    timeout: Duration,
) -> VaultResult<protocol::HealthResult> {
    let started = tokio::time::Instant::now();
    loop {
        match probe_health(endpoint).await? {
            HealthProbeOutcome::Healthy(health) => return Ok(health),
            HealthProbeOutcome::Incompatible(message) => {
                return Err(VaultError::DaemonBootstrap(format!(
                    "daemon API incompatibility detected: {message}"
                )));
            }
            HealthProbeOutcome::Invalid(message) => {
                return Err(VaultError::DaemonBootstrap(format!(
                    "daemon health probe returned invalid response: {message}"
                )));
            }
            HealthProbeOutcome::Unreachable => {}
        }

        if started.elapsed() >= timeout {
            return Err(VaultError::DaemonBootstrap(format!(
                "timed out waiting for daemon health on endpoint '{}'",
                endpoint.endpoint_string()
            )));
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}

async fn shutdown_owned_daemon(
    endpoint: &IpcEndpoint,
    manifest: &RuntimeManifest,
    health_pid: u32,
) -> VaultResult<()> {
    let pid = if health_pid == 0 {
        manifest.pid
    } else {
        health_pid
    };
    let expected_binary = Path::new(&manifest.binary_path);
    if pid == 0 || pid == std::process::id() {
        return Err(VaultError::DaemonBootstrap(
            "semantic daemon reported an unsafe PID".into(),
        ));
    }
    if !process_matches_binary(pid, expected_binary) {
        return Err(VaultError::DaemonBootstrap(format!(
            "refusing to stop semantic daemon PID {pid} because its executable identity does not match '{}'",
            manifest.binary_path
        )));
    }
    let policy = DaemonConnectPolicy {
        timeout: Duration::from_secs(2),
        retries: 0,
        retry_backoff: Duration::ZERO,
    };
    let client = SemanticDaemonClient::new(endpoint.clone(), policy);
    match client.shutdown().await {
        Ok(result) if result.accepted => {
            wait_for_unreachable(endpoint, Duration::from_secs(5)).await
        }
        Ok(_) => Err(VaultError::DaemonBootstrap(
            "semantic daemon rejected graceful shutdown".into(),
        )),
        Err(VaultError::DaemonRpc { code, .. }) if code == protocol::ERR_METHOD_NOT_FOUND => {
            stop_legacy_owned_process(pid, expected_binary)?;
            wait_for_unreachable(endpoint, Duration::from_secs(5)).await
        }
        Err(err) => Err(VaultError::DaemonBootstrap(format!(
            "semantic daemon graceful shutdown failed: {err}"
        ))),
    }
}

async fn wait_for_unreachable(endpoint: &IpcEndpoint, timeout: Duration) -> VaultResult<()> {
    let started = tokio::time::Instant::now();
    loop {
        if matches!(
            probe_health(endpoint).await?,
            HealthProbeOutcome::Unreachable
        ) {
            return Ok(());
        }
        if started.elapsed() >= timeout {
            return Err(VaultError::DaemonBootstrap(format!(
                "timed out waiting for semantic daemon '{}' to stop",
                endpoint.endpoint_string()
            )));
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

fn effective_origin(
    manifest: &RuntimeManifest,
    paths: &SemanticHomePaths,
    installed_daemon: &Path,
) -> BinaryOrigin {
    if manifest.binary_origin != BinaryOrigin::Unknown {
        return manifest.binary_origin;
    }
    let manifest_path = Path::new(&manifest.binary_path);
    if paths_equal(manifest_path, installed_daemon) {
        BinaryOrigin::Sibling
    } else if paths_equal(manifest_path, &paths.daemon_binary_path) {
        BinaryOrigin::ManagedDownload
    } else {
        BinaryOrigin::Unknown
    }
}

fn paths_equal(left: &Path, right: &Path) -> bool {
    match (left.canonicalize(), right.canonicalize()) {
        (Ok(left), Ok(right)) => left == right,
        _ => left == right,
    }
}

fn stop_legacy_owned_process(pid: u32, expected_binary: &Path) -> VaultResult<()> {
    if !process_matches_binary(pid, expected_binary) {
        return Err(VaultError::DaemonBootstrap(format!(
            "refusing to stop PID {pid} because its executable identity does not match '{}'",
            expected_binary.display()
        )));
    }
    terminate_process(pid, false)?;
    if wait_for_process_exit(pid, Duration::from_secs(5)) {
        return Ok(());
    }
    terminate_process(pid, true)?;
    if wait_for_process_exit(pid, Duration::from_secs(2)) {
        Ok(())
    } else {
        Err(VaultError::DaemonBootstrap(format!(
            "legacy semantic daemon PID {pid} did not stop"
        )))
    }
}

#[cfg(target_os = "linux")]
fn process_matches_binary(pid: u32, expected: &Path) -> bool {
    let Ok(actual) = std::fs::read_link(format!("/proc/{pid}/exe")) else {
        return false;
    };
    paths_equal(&without_deleted_suffix(actual), expected)
}

#[cfg(target_os = "macos")]
fn process_matches_binary(pid: u32, expected: &Path) -> bool {
    let Ok(output) = std::process::Command::new("lsof")
        .args(["-a", "-p", &pid.to_string(), "-d", "txt", "-Fn"])
        .output()
    else {
        return false;
    };
    output.status.success()
        && String::from_utf8_lossy(&output.stdout)
            .lines()
            .filter_map(|line| line.strip_prefix('n'))
            .map(PathBuf::from)
            .map(without_deleted_suffix)
            .any(|path| paths_equal(&path, expected))
}

#[cfg(unix)]
fn without_deleted_suffix(path: PathBuf) -> PathBuf {
    let rendered = path.to_string_lossy();
    rendered
        .strip_suffix(" (deleted)")
        .map(PathBuf::from)
        .unwrap_or(path)
}

#[cfg(windows)]
fn process_matches_binary(pid: u32, expected: &Path) -> bool {
    let filter = format!("ProcessId = {pid}");
    let script = format!(
        "(Get-CimInstance Win32_Process -Filter '{}').ExecutablePath",
        filter.replace('\'', "''")
    );
    let Ok(output) = std::process::Command::new("powershell.exe")
        .args(["-NoProfile", "-NonInteractive", "-Command", &script])
        .output()
    else {
        return false;
    };
    output.status.success()
        && paths_equal(
            Path::new(String::from_utf8_lossy(&output.stdout).trim()),
            expected,
        )
}

#[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
fn process_matches_binary(_pid: u32, _expected: &Path) -> bool {
    false
}

#[cfg(unix)]
fn terminate_process(pid: u32, force: bool) -> VaultResult<()> {
    let signal = if force { "-KILL" } else { "-INT" };
    let status = std::process::Command::new("kill")
        .args([signal, &pid.to_string()])
        .status()?;
    if status.success() {
        Ok(())
    } else {
        Err(VaultError::DaemonBootstrap(format!(
            "failed to send {signal} to legacy semantic daemon PID {pid}"
        )))
    }
}

#[cfg(windows)]
fn terminate_process(pid: u32, force: bool) -> VaultResult<()> {
    let mut args = vec!["/PID".to_string(), pid.to_string()];
    if force {
        args.push("/F".to_string());
    }
    let status = std::process::Command::new("taskkill").args(args).status()?;
    if status.success() {
        Ok(())
    } else {
        Err(VaultError::DaemonBootstrap(format!(
            "failed to stop legacy semantic daemon PID {pid}"
        )))
    }
}

fn wait_for_process_exit(pid: u32, timeout: Duration) -> bool {
    let started = std::time::Instant::now();
    while started.elapsed() < timeout {
        if !process_is_alive(pid) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    !process_is_alive(pid)
}

#[cfg(unix)]
fn process_is_alive(pid: u32) -> bool {
    std::process::Command::new("kill")
        .args(["-0", &pid.to_string()])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

#[cfg(windows)]
fn process_is_alive(pid: u32) -> bool {
    let filter = format!("PID eq {pid}");
    std::process::Command::new("tasklist")
        .args(["/FI", &filter, "/FO", "CSV", "/NH"])
        .output()
        .is_ok_and(|output| {
            output.status.success()
                && String::from_utf8_lossy(&output.stdout).lines().any(|line| {
                    line.split("\",\"").nth(1).is_some_and(|value| {
                        value.trim_matches('"').parse::<u32>().ok() == Some(pid)
                    })
                })
        })
}

#[cfg(unix)]
async fn probe_health(endpoint: &IpcEndpoint) -> VaultResult<HealthProbeOutcome> {
    let IpcEndpoint::UnixSocket(path) = endpoint;
    let stream = match UnixStream::connect(path).await {
        Ok(stream) => stream,
        Err(err)
            if matches!(
                err.kind(),
                std::io::ErrorKind::NotFound
                    | std::io::ErrorKind::ConnectionRefused
                    | std::io::ErrorKind::ConnectionReset
            ) =>
        {
            return Ok(HealthProbeOutcome::Unreachable);
        }
        Err(err) => {
            return Err(VaultError::DaemonBootstrap(format!(
                "failed to connect to daemon endpoint '{}': {err}",
                path.display()
            )));
        }
    };

    request_health(stream).await
}

#[cfg(windows)]
async fn probe_health(endpoint: &IpcEndpoint) -> VaultResult<HealthProbeOutcome> {
    let IpcEndpoint::NamedPipe(name) = endpoint;
    let stream = match ClientOptions::new().open(name) {
        Ok(stream) => stream,
        Err(err)
            if err.kind() == std::io::ErrorKind::NotFound
                || err.kind() == std::io::ErrorKind::ConnectionRefused
                || err.raw_os_error() == Some(231) =>
        {
            return Ok(HealthProbeOutcome::Unreachable);
        }
        Err(err) => {
            return Err(VaultError::DaemonBootstrap(format!(
                "failed to connect to daemon named pipe '{}': {err}",
                name
            )));
        }
    };

    request_health(stream).await
}

#[cfg(not(any(unix, windows)))]
async fn probe_health(_endpoint: &IpcEndpoint) -> VaultResult<HealthProbeOutcome> {
    Ok(HealthProbeOutcome::Unreachable)
}

async fn request_health<S>(mut stream: S) -> VaultResult<HealthProbeOutcome>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    let request = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "health",
        "params": {
            "client_name": "obsidian-mcp",
            "client_version": env!("CARGO_PKG_VERSION"),
            "min_api_version": DAEMON_API_VERSION,
            "max_api_version": DAEMON_API_VERSION
        }
    });
    let request_str = serde_json::to_string(&request).map_err(|err| {
        VaultError::DaemonBootstrap(format!("failed to serialize daemon health request: {err}"))
    })?;

    stream
        .write_all(request_str.as_bytes())
        .await
        .map_err(|err| {
            VaultError::DaemonBootstrap(format!("failed to write daemon health request: {err}"))
        })?;
    stream.write_all(b"\n").await.map_err(|err| {
        VaultError::DaemonBootstrap(format!("failed to finish daemon health request: {err}"))
    })?;
    stream.flush().await.map_err(|err| {
        VaultError::DaemonBootstrap(format!("failed to flush daemon health request: {err}"))
    })?;

    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    let read = reader.read_line(&mut line).await.map_err(|err| {
        VaultError::DaemonBootstrap(format!("failed to read daemon health response: {err}"))
    })?;
    if read == 0 {
        return Ok(HealthProbeOutcome::Unreachable);
    }

    let response: serde_json::Value = serde_json::from_str(&line).map_err(|err| {
        VaultError::DaemonBootstrap(format!(
            "failed to parse daemon health response JSON: {err}"
        ))
    })?;

    if let Some(error) = response.get("error") {
        let code = error
            .get("code")
            .and_then(serde_json::Value::as_i64)
            .unwrap_or_default();
        let message = error
            .get("message")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("daemon health error")
            .to_string();
        if code == ERR_INCOMPATIBLE_API_VERSION {
            return Ok(HealthProbeOutcome::Incompatible(message));
        }
        return Ok(HealthProbeOutcome::Invalid(message));
    }

    let Some(result) = response.get("result") else {
        return Ok(HealthProbeOutcome::Invalid(
            "missing result in daemon health response".to_string(),
        ));
    };
    let health =
        serde_json::from_value::<protocol::HealthResult>(result.clone()).map_err(|err| {
            VaultError::DaemonBootstrap(format!("failed to decode daemon health result: {err}"))
        })?;
    Ok(HealthProbeOutcome::Healthy(health))
}

fn endpoint_transport(endpoint: &IpcEndpoint) -> &'static str {
    match endpoint {
        #[cfg(unix)]
        IpcEndpoint::UnixSocket(_) => "unix_socket",
        #[cfg(windows)]
        IpcEndpoint::NamedPipe(_) => "named_pipe",
    }
}

fn endpoint_from_manifest(manifest: &RuntimeManifest) -> Option<IpcEndpoint> {
    if manifest.ipc.transport == "unix_socket" {
        #[cfg(unix)]
        {
            return Some(IpcEndpoint::UnixSocket(PathBuf::from(
                &manifest.ipc.endpoint,
            )));
        }
    }

    if manifest.ipc.transport == "named_pipe" {
        #[cfg(windows)]
        {
            return Some(IpcEndpoint::NamedPipe(manifest.ipc.endpoint.clone()));
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::daemon::server;

    #[test]
    fn resolve_download_url_uses_override() {
        let config = BootstrapConfig {
            download_url_override: Some("https://example.com/custom.tar.gz".to_string()),
            ..Default::default()
        };
        let url = resolve_download_url(&config).expect("download URL should resolve");
        assert_eq!(url, "https://example.com/custom.tar.gz");
    }

    #[test]
    fn resolve_download_url_uses_versioned_semanticd_asset_name() {
        let url = resolve_download_url(&BootstrapConfig::default()).expect("resolve default URL");
        let version = env!("CARGO_PKG_VERSION");
        assert!(
            url.contains(&format!("/releases/download/v{version}/")),
            "url should target release tag path, got: {url}"
        );
        assert!(
            url.contains(&format!("obsidian-semanticd-{version}-")),
            "url should include versioned semantic daemon asset, got: {url}"
        );
        if cfg!(windows) {
            assert!(url.ends_with(".zip"), "windows URL should end with .zip");
        } else {
            assert!(url.ends_with(".tar.gz"), "unix URL should end with .tar.gz");
        }
    }

    #[test]
    fn endpoint_from_manifest_rejects_unknown_transport() {
        let manifest = RuntimeManifest::from_input(RuntimeManifestInput {
            daemon_api_version: 1,
            daemon_version: "1.0.1".to_string(),
            binary_path: "/tmp/semanticd".to_string(),
            binary_origin: BinaryOrigin::Unknown,
            binary_sha256: None,
            ipc: ManifestIpc {
                transport: "tcp".to_string(),
                endpoint: "127.0.0.1:1234".to_string(),
            },
            pid: 10,
            semantic_home: "/tmp/home".to_string(),
            fastembed_cache_dir: "/tmp/home/model/fastembed-cache".to_string(),
            model_name: "BAAI/bge-small-en-v1.5".to_string(),
            bootstrap_client_name: "obsidian-mcp".to_string(),
            bootstrap_client_version: "1.0.1".to_string(),
        });
        assert!(endpoint_from_manifest(&manifest).is_none());
    }

    #[test]
    fn find_daemon_on_path_discovers_binary_in_temp_dir() {
        let dir = tempfile::tempdir().expect("tempdir");
        let binary_name = home::daemon_binary_name();
        let fake_binary = dir.path().join(binary_name);
        std::fs::write(&fake_binary, b"fake").expect("write fake binary");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&fake_binary).unwrap().permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&fake_binary, perms).unwrap();
        }

        let original_path = std::env::var_os("PATH").unwrap_or_default();
        let mut new_path = std::env::split_paths(&original_path).collect::<Vec<_>>();
        new_path.insert(0, dir.path().to_path_buf());
        let joined = std::env::join_paths(&new_path).expect("join paths");
        // SAFETY: test-only; mutating env is acceptable in single-threaded test context
        unsafe { std::env::set_var("PATH", &joined) };

        let result = find_daemon_on_path();
        unsafe { std::env::set_var("PATH", &original_path) };

        assert_eq!(result, Some(fake_binary));
    }

    #[test]
    fn find_daemon_on_path_returns_none_when_absent() {
        let dir = tempfile::tempdir().expect("tempdir");
        let original_path = std::env::var_os("PATH").unwrap_or_default();
        // SAFETY: test-only; mutating env is acceptable in single-threaded test context
        unsafe { std::env::set_var("PATH", dir.path()) };

        let result = find_daemon_on_path();
        unsafe { std::env::set_var("PATH", &original_path) };

        assert!(result.is_none());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn ensure_daemon_reuses_healthy_manifest_endpoint() {
        let dir = tempfile::tempdir().expect("tempdir should be created");
        let paths = SemanticHomePaths::new(dir.path().join("semantic-home"));
        home::ensure_home_layout(&paths).expect("home layout should be created");
        let endpoint = IpcEndpoint::UnixSocket(paths.ipc_dir.join("semanticd.sock"));
        let daemon_config = server::DaemonServerConfig {
            endpoint: endpoint.clone(),
            model_name: "BAAI/bge-small-en-v1.5".to_string(),
            semantic_home: paths.root.clone(),
        };

        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
        let server_task = tokio::spawn(async move {
            server::run_with_shutdown(daemon_config, async move {
                let _ = shutdown_rx.await;
            })
            .await
            .expect("daemon server should run");
        });

        let mut ready = false;
        let IpcEndpoint::UnixSocket(socket_path) = &endpoint;
        for _ in 0..50 {
            if UnixStream::connect(socket_path).await.is_ok() {
                ready = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        assert!(ready, "daemon endpoint did not become ready");

        let manifest = RuntimeManifest::from_input(RuntimeManifestInput {
            daemon_api_version: DAEMON_API_VERSION,
            daemon_version: env!("CARGO_PKG_VERSION").to_string(),
            binary_path: "/tmp/nonexistent-semanticd".to_string(),
            binary_origin: BinaryOrigin::Unknown,
            binary_sha256: None,
            ipc: ManifestIpc {
                transport: "unix_socket".to_string(),
                endpoint: endpoint.endpoint_string(),
            },
            pid: 4242,
            semantic_home: paths.root.display().to_string(),
            fastembed_cache_dir: paths.fastembed_cache_dir.display().to_string(),
            model_name: "BAAI/bge-small-en-v1.5".to_string(),
            bootstrap_client_name: "obsidian-mcp".to_string(),
            bootstrap_client_version: "1.0.1".to_string(),
        });
        manifest::save_atomic(&paths.manifest_path, &manifest).expect("manifest should persist");

        let config = BootstrapConfig {
            semantic_home_override: Some(paths.root.clone()),
            daemon_path_override: None,
            model_name: "BAAI/bge-small-en-v1.5".to_string(),
            download_url_override: None,
            bootstrap_client_name: "obsidian-mcp".to_string(),
            bootstrap_client_version: "1.0.1".to_string(),
        };
        let result = ensure_daemon(&config)
            .await
            .expect("bootstrap should reuse daemon");
        assert!(result.reused_existing, "expected reuse-existing path");

        shutdown_tx.send(()).expect("shutdown signal should send");
        server_task.await.expect("server task should join");
    }
}
