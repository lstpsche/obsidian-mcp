//! Cargo installation provenance, reconstruction, and verification.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{OsStr, OsString};
use std::fs;
use std::io::Read;
use std::path::Path;

use serde::Deserialize;
use sha2::{Digest, Sha256};

use super::{BuildIdentity, CommandExecutor, InstallRecord, UpgradeError};

const PACKAGE_NAME: &str = "obsidian-mcp";
const MAIN_BINARY: &str = "obsidian-mcp";
const SEMANTIC_BINARY: &str = "obsidian-semanticd";
const TRACKER_FILE: &str = ".crates2.json";
const OFFICIAL_GIT_INDEX: &str = "registry+https://github.com/rust-lang/crates.io-index";
const OFFICIAL_SPARSE_INDEX: &str = "sparse+https://index.crates.io/";

#[derive(Debug, Deserialize)]
struct CargoTracker {
    #[serde(default)]
    installs: BTreeMap<String, TrackedInstall>,
}

#[derive(Debug, Deserialize)]
struct TrackedInstall {
    #[serde(default)]
    bins: BTreeSet<String>,
    #[serde(default)]
    features: BTreeSet<String>,
    profile: String,
    target: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstallSnapshot {
    pub main_identity: BuildIdentity,
    pub semantic_identity: BuildIdentity,
    pub main_hash: [u8; 32],
    pub semantic_hash: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstallOutcome {
    pub after: InstallSnapshot,
    pub changed: bool,
}

/// Verify that `executable` is the Cargo-installed package described by its
/// install-root tracker. No alternative source is accepted implicitly.
pub fn discover_install(
    executable: &Path,
    identity: &BuildIdentity,
) -> Result<InstallRecord, UpgradeError> {
    let executable = executable.canonicalize().map_err(|err| {
        UpgradeError::UnsupportedInstall(format!(
            "cannot resolve running executable '{}': {err}",
            executable.display()
        ))
    })?;
    let expected_name = binary_file_name(MAIN_BINARY);
    if executable.file_name() != Some(OsStr::new(&expected_name)) {
        return Err(UpgradeError::UnsupportedInstall(format!(
            "running executable '{}' is not {expected_name}",
            executable.display()
        )));
    }

    let bin_dir = executable.parent().ok_or_else(|| {
        UpgradeError::UnsupportedInstall("running executable has no parent directory".into())
    })?;
    if bin_dir.file_name() != Some(OsStr::new("bin")) {
        return Err(UpgradeError::UnsupportedInstall(format!(
            "'{}' is not inside a Cargo install root bin directory",
            executable.display()
        )));
    }
    let root = bin_dir.parent().ok_or_else(|| {
        UpgradeError::UnsupportedInstall("Cargo bin directory has no install root".into())
    })?;
    let tracker_path = root.join(TRACKER_FILE);
    let tracker_raw = fs::read(&tracker_path).map_err(|err| {
        UpgradeError::UnsupportedInstall(format!(
            "Cargo ownership tracker '{}' is unavailable: {err}",
            tracker_path.display()
        ))
    })?;
    let tracker: CargoTracker = serde_json::from_slice(&tracker_raw).map_err(|err| {
        UpgradeError::InvalidInstall(format!(
            "cannot parse Cargo ownership tracker '{}': {err}",
            tracker_path.display()
        ))
    })?;

    let prefix = format!("{PACKAGE_NAME} {} (", identity.version);
    let mut matches = tracker
        .installs
        .into_iter()
        .filter(|(package_id, _)| package_id.starts_with(&prefix));
    let (package_id, tracked) = matches.next().ok_or_else(|| {
        UpgradeError::UnsupportedInstall(format!(
            "Cargo does not track {PACKAGE_NAME} {} in '{}'",
            identity.version,
            root.display()
        ))
    })?;
    if matches.next().is_some() {
        return Err(UpgradeError::InvalidInstall(format!(
            "Cargo tracker contains multiple records for {PACKAGE_NAME} {}",
            identity.version
        )));
    }
    let source = package_source(&package_id)
        .ok_or_else(|| {
            UpgradeError::InvalidInstall(format!("malformed Cargo package id '{package_id}'"))
        })?
        .to_string();
    if !is_official_crates_io(&source) {
        return Err(UpgradeError::UnsupportedInstall(format!(
            "Cargo source '{source}' is not the official crates.io registry"
        )));
    }

    let tracked_bins = tracked
        .bins
        .iter()
        .map(|bin| bin.trim_end_matches(".exe"))
        .collect::<BTreeSet<_>>();
    for required in [MAIN_BINARY, SEMANTIC_BINARY] {
        if !tracked_bins.contains(required) {
            return Err(UpgradeError::InvalidInstall(format!(
                "Cargo record does not own required binary '{required}'"
            )));
        }
        let path = bin_dir.join(binary_file_name(required));
        if !path.is_file() {
            return Err(UpgradeError::InvalidInstall(format!(
                "Cargo-owned binary '{}' is missing",
                path.display()
            )));
        }
    }

    if tracked.profile.trim().is_empty() {
        return Err(UpgradeError::InvalidInstall(
            "Cargo record has an empty build profile".into(),
        ));
    }
    let target = tracked.target.ok_or_else(|| {
        UpgradeError::InvalidInstall("Cargo record does not include its target triple".into())
    })?;
    if target != identity.target {
        return Err(UpgradeError::InvalidInstall(format!(
            "Cargo target '{target}' does not match running binary target '{}'",
            identity.target
        )));
    }

    let tracked_features = tracked
        .features
        .iter()
        .filter(|feature| feature.as_str() != "default")
        .cloned()
        .collect::<BTreeSet<_>>();
    if !tracked_features.is_subset(&identity.features) {
        return Err(UpgradeError::InvalidInstall(format!(
            "Cargo-tracked features [{}] are not present in the running binary [{}]",
            tracked_features
                .iter()
                .cloned()
                .collect::<Vec<_>>()
                .join(","),
            identity.feature_list()
        )));
    }
    Ok(InstallRecord {
        root: root.to_path_buf(),
        profile: tracked.profile,
        target,
    })
}

pub fn cargo_install_args(record: &InstallRecord, identity: &BuildIdentity) -> Vec<OsString> {
    let mut args = vec![
        OsString::from("install"),
        OsString::from(PACKAGE_NAME),
        OsString::from("--root"),
        record.root.as_os_str().to_owned(),
        OsString::from("--registry"),
        OsString::from("crates-io"),
        OsString::from("--locked"),
        OsString::from("--bins"),
        OsString::from("--profile"),
        OsString::from(&record.profile),
        OsString::from("--target"),
        OsString::from(&record.target),
        OsString::from("--no-default-features"),
    ];
    if !identity.features.is_empty() {
        args.push(OsString::from("--features"));
        args.push(OsString::from(identity.feature_list()));
    }
    args
}

pub fn snapshot_install<E: CommandExecutor>(
    executor: &E,
    record: &InstallRecord,
) -> Result<InstallSnapshot, UpgradeError> {
    let main = record.root.join("bin").join(binary_file_name(MAIN_BINARY));
    let semantic = record
        .root
        .join("bin")
        .join(binary_file_name(SEMANTIC_BINARY));
    Ok(InstallSnapshot {
        main_identity: probe_identity(executor, &main)?,
        semantic_identity: probe_identity(executor, &semantic)?,
        main_hash: hash_file(&main)?,
        semantic_hash: hash_file(&semantic)?,
    })
}

pub fn install_latest<E: CommandExecutor>(
    executor: &E,
    record: &InstallRecord,
    identity: &BuildIdentity,
) -> Result<InstallOutcome, UpgradeError> {
    let before = snapshot_install(executor, record)?;
    validate_snapshot(&before, identity)?;

    let args = cargo_install_args(record, identity);
    let status = executor.status_inherit(OsStr::new("cargo"), &args)?;
    if !status.success {
        return Err(UpgradeError::Install(format!(
            "cargo install exited with status {}",
            status.code.map_or_else(
                || "terminated by signal".to_string(),
                |code| code.to_string()
            )
        )));
    }

    let after = snapshot_install(executor, record)
        .map_err(|err| UpgradeError::PostInstall(err.to_string()))?;
    validate_post_install(&after, identity)
        .map_err(|err| UpgradeError::PostInstall(err.to_string()))?;
    let changed =
        before.main_hash != after.main_hash || before.semantic_hash != after.semantic_hash;
    Ok(InstallOutcome { after, changed })
}

fn validate_snapshot(
    snapshot: &InstallSnapshot,
    expected: &BuildIdentity,
) -> Result<(), UpgradeError> {
    if &snapshot.main_identity != expected || &snapshot.semantic_identity != expected {
        return Err(UpgradeError::InvalidInstall(
            "installed sibling binaries do not match the running build identity".into(),
        ));
    }
    Ok(())
}

fn validate_post_install(
    snapshot: &InstallSnapshot,
    previous: &BuildIdentity,
) -> Result<(), UpgradeError> {
    if snapshot.main_identity != snapshot.semantic_identity {
        return Err(UpgradeError::Install(
            "updated obsidian-mcp and obsidian-semanticd identities differ".into(),
        ));
    }
    let current = &snapshot.main_identity;
    if current.name != PACKAGE_NAME {
        return Err(UpgradeError::Install(format!(
            "updated package identity is '{}', expected '{PACKAGE_NAME}'",
            current.name
        )));
    }
    if current.target != previous.target {
        return Err(UpgradeError::Install(format!(
            "updated target '{}' does not preserve '{}'",
            current.target, previous.target
        )));
    }
    if current.features != previous.features {
        return Err(UpgradeError::Install(format!(
            "updated features [{}] do not preserve [{}]",
            current.feature_list(),
            previous.feature_list()
        )));
    }
    let previous_version = semver::Version::parse(&previous.version).map_err(|err| {
        UpgradeError::Install(format!(
            "running binary has invalid semantic version '{}': {err}",
            previous.version
        ))
    })?;
    let current_version = semver::Version::parse(&current.version).map_err(|err| {
        UpgradeError::Install(format!(
            "updated binary has invalid semantic version '{}': {err}",
            current.version
        ))
    })?;
    if current_version < previous_version {
        return Err(UpgradeError::Install(format!(
            "Cargo installed older version '{}' over '{}'",
            current.version, previous.version
        )));
    }
    Ok(())
}

fn probe_identity<E: CommandExecutor>(
    executor: &E,
    binary: &Path,
) -> Result<BuildIdentity, UpgradeError> {
    let output = executor.output(binary.as_os_str(), &[OsString::from("--__build-info")])?;
    if !output.success {
        return Err(UpgradeError::InvalidInstall(format!(
            "'{} --__build-info' failed: {}",
            binary.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    BuildIdentity::from_json(&output.stdout).map_err(|err| {
        UpgradeError::InvalidInstall(format!(
            "'{}' returned invalid build identity: {err}",
            binary.display()
        ))
    })
}

fn hash_file(path: &Path) -> Result<[u8; 32], UpgradeError> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher.finalize().into())
}

fn package_source(package_id: &str) -> Option<&str> {
    package_id
        .strip_suffix(')')?
        .rsplit_once(" (")
        .map(|(_, source)| source)
}

fn is_official_crates_io(source: &str) -> bool {
    matches!(source, OFFICIAL_GIT_INDEX | OFFICIAL_SPARSE_INDEX)
}

pub fn binary_file_name(stem: &str) -> String {
    format!("{stem}{}", std::env::consts::EXE_SUFFIX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::upgrade::{CommandOutput, CommandStatus};
    use std::collections::VecDeque;
    use std::path::PathBuf;
    use std::sync::Mutex;

    fn identity() -> BuildIdentity {
        BuildIdentity {
            name: PACKAGE_NAME.into(),
            version: "2.4.0".into(),
            target: "test-target".into(),
            features: BTreeSet::from(["embeddings-api".into()]),
        }
    }

    fn fixture(source: &str) -> (tempfile::TempDir, PathBuf) {
        let root = tempfile::tempdir().expect("tempdir should be created");
        let bin = root.path().join("bin");
        fs::create_dir(&bin).expect("bin dir should be created");
        let main = bin.join(binary_file_name(MAIN_BINARY));
        fs::write(&main, b"main").expect("main binary should be written");
        fs::write(bin.join(binary_file_name(SEMANTIC_BINARY)), b"semantic")
            .expect("semantic binary should be written");
        let package_id = format!("obsidian-mcp 2.4.0 ({source})");
        let tracker = serde_json::json!({
            "installs": {
                package_id: {
                    "version_req": "=2.4.0",
                    "bins": [MAIN_BINARY, SEMANTIC_BINARY],
                    "features": ["embeddings-api"],
                    "all_features": false,
                    "no_default_features": false,
                    "profile": "release",
                    "target": "test-target",
                    "future_cargo_field": true
                }
            }
        });
        fs::write(
            root.path().join(TRACKER_FILE),
            serde_json::to_vec(&tracker).expect("tracker should serialize"),
        )
        .expect("tracker should be written");
        (root, main)
    }

    #[test]
    fn discovers_official_cargo_install_and_ignores_unknown_metadata() {
        let (root, main) = fixture(OFFICIAL_GIT_INDEX);
        let record = discover_install(&main, &identity()).expect("install should be supported");
        assert_eq!(
            record.root,
            root.path()
                .canonicalize()
                .expect("root should canonicalize")
        );
        assert_eq!(record.profile, "release");
        assert_eq!(record.target, "test-target");
    }

    #[test]
    fn rejects_non_registry_install_before_any_command() {
        let (_root, main) = fixture("git+https://github.com/lstpsche/obsidian-mcp");
        let error = discover_install(&main, &identity()).expect_err("git install must be rejected");
        assert!(
            error
                .to_string()
                .contains("not the official crates.io registry")
        );
    }

    #[test]
    fn reconstructed_command_preserves_exact_compiled_capabilities() {
        let (root, main) = fixture(OFFICIAL_SPARSE_INDEX);
        let record = discover_install(&main, &identity()).expect("install should be supported");
        let args = cargo_install_args(&record, &identity());
        let canonical_root = root
            .path()
            .canonicalize()
            .expect("root should canonicalize");
        let rendered = args
            .iter()
            .map(|arg| arg.to_string_lossy())
            .collect::<Vec<_>>();
        assert_eq!(
            rendered,
            vec![
                "install",
                PACKAGE_NAME,
                "--root",
                canonical_root.to_string_lossy().as_ref(),
                "--registry",
                "crates-io",
                "--locked",
                "--bins",
                "--profile",
                "release",
                "--target",
                "test-target",
                "--no-default-features",
                "--features",
                "embeddings-api"
            ]
        );
        assert!(!rendered.contains(&std::borrow::Cow::Borrowed("--force")));
        assert!(!rendered.contains(&std::borrow::Cow::Borrowed("--no-track")));
    }

    #[test]
    fn rejects_target_and_feature_mismatches() {
        let (_root, main) = fixture(OFFICIAL_GIT_INDEX);
        let mut wrong_target = identity();
        wrong_target.target = "different-target".into();
        assert!(discover_install(&main, &wrong_target).is_err());

        let mut missing_feature = identity();
        missing_feature.features.clear();
        assert!(discover_install(&main, &missing_feature).is_err());
    }

    struct FakeExecutor {
        identities: Mutex<VecDeque<BuildIdentity>>,
        replacement: Option<(PathBuf, PathBuf)>,
        install_success: bool,
    }

    impl CommandExecutor for FakeExecutor {
        fn output(
            &self,
            _program: &OsStr,
            args: &[OsString],
        ) -> Result<CommandOutput, UpgradeError> {
            assert_eq!(args, &[OsString::from("--__build-info")]);
            let identity = self
                .identities
                .lock()
                .expect("identity queue should lock")
                .pop_front()
                .expect("identity response should exist");
            Ok(CommandOutput {
                success: true,
                code: Some(0),
                stdout: identity.to_json()?.into_bytes(),
                stderr: Vec::new(),
            })
        }

        fn status_inherit(
            &self,
            program: &OsStr,
            args: &[OsString],
        ) -> Result<CommandStatus, UpgradeError> {
            assert_eq!(program, OsStr::new("cargo"));
            assert!(!args.iter().any(|arg| arg == "--force"));
            if self.install_success
                && let Some((main, semantic)) = &self.replacement
            {
                fs::write(main, b"new-main")?;
                fs::write(semantic, b"new-semantic")?;
            }
            Ok(CommandStatus {
                success: self.install_success,
                code: Some(if self.install_success { 0 } else { 101 }),
            })
        }
    }

    #[test]
    fn install_verifies_both_replaced_binaries_and_detects_hash_change() {
        let (root, main) = fixture(OFFICIAL_GIT_INDEX);
        let current = identity();
        let mut updated = current.clone();
        updated.version = "2.5.0".into();
        let record = discover_install(&main, &current).expect("install should be supported");
        let semantic = root
            .path()
            .join("bin")
            .join(binary_file_name(SEMANTIC_BINARY));
        let executor = FakeExecutor {
            identities: Mutex::new(VecDeque::from([
                current.clone(),
                current.clone(),
                updated.clone(),
                updated.clone(),
            ])),
            replacement: Some((main, semantic)),
            install_success: true,
        };

        let outcome = install_latest(&executor, &record, &current).expect("install should pass");
        assert!(outcome.changed);
        assert_eq!(outcome.after.main_identity, updated);
    }

    #[test]
    fn same_version_and_hashes_are_a_no_op() {
        let (_root, main) = fixture(OFFICIAL_GIT_INDEX);
        let current = identity();
        let record = discover_install(&main, &current).expect("install should be supported");
        let executor = FakeExecutor {
            identities: Mutex::new(VecDeque::from([
                current.clone(),
                current.clone(),
                current.clone(),
                current.clone(),
            ])),
            replacement: None,
            install_success: true,
        };

        let outcome = install_latest(&executor, &record, &current).expect("install should pass");
        assert!(!outcome.changed);
        assert_eq!(outcome.after.main_identity, current);
    }

    #[test]
    fn install_rejects_post_install_feature_drift_after_cargo_completes() {
        let (root, main) = fixture(OFFICIAL_GIT_INDEX);
        let current = identity();
        let mut drifted = current.clone();
        drifted.version = "2.5.0".into();
        drifted.features.clear();
        let record = discover_install(&main, &current).expect("install should be supported");
        let semantic = root
            .path()
            .join("bin")
            .join(binary_file_name(SEMANTIC_BINARY));
        let executor = FakeExecutor {
            identities: Mutex::new(VecDeque::from([
                current.clone(),
                current.clone(),
                drifted.clone(),
                drifted,
            ])),
            replacement: Some((main, semantic)),
            install_success: true,
        };

        let error = install_latest(&executor, &record, &current)
            .expect_err("feature drift must fail verification");
        assert!(matches!(error, UpgradeError::PostInstall(_)));
    }

    #[test]
    fn install_rejects_post_install_version_downgrade() {
        let (root, main) = fixture(OFFICIAL_GIT_INDEX);
        let current = identity();
        let mut older = current.clone();
        older.version = "2.3.9".into();
        let record = discover_install(&main, &current).expect("install should be supported");
        let semantic = root
            .path()
            .join("bin")
            .join(binary_file_name(SEMANTIC_BINARY));
        let executor = FakeExecutor {
            identities: Mutex::new(VecDeque::from([
                current.clone(),
                current.clone(),
                older.clone(),
                older,
            ])),
            replacement: Some((main, semantic)),
            install_success: true,
        };

        let error = install_latest(&executor, &record, &current)
            .expect_err("a downgrade must fail verification");
        assert!(matches!(error, UpgradeError::PostInstall(_)));
        assert!(error.to_string().contains("older version"));
    }

    #[test]
    fn cargo_failure_stops_before_post_install_probes() {
        let (_root, main) = fixture(OFFICIAL_GIT_INDEX);
        let current = identity();
        let record = discover_install(&main, &current).expect("install should be supported");
        let executor = FakeExecutor {
            identities: Mutex::new(VecDeque::from([current.clone(), current.clone()])),
            replacement: None,
            install_success: false,
        };
        let error = install_latest(&executor, &record, &current)
            .expect_err("Cargo failure should propagate");
        assert!(matches!(error, UpgradeError::Install(_)));
        assert!(
            executor
                .identities
                .lock()
                .expect("identity queue should lock")
                .is_empty()
        );
    }
}
