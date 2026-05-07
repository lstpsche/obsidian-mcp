//! Shared semantic-home path, namespace, and lock helpers.

use std::fmt::Write as _;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};

use fs2::FileExt;
use sha2::{Digest, Sha256};

use crate::error::{VaultError, VaultResult};

use super::server::IpcEndpoint;

#[derive(Debug, Clone)]
pub struct SemanticHomePaths {
    pub root: PathBuf,
    pub manifest_path: PathBuf,
    pub lock_dir: PathBuf,
    pub install_lock_path: PathBuf,
    pub logs_dir: PathBuf,
    pub daemon_stderr_log_path: PathBuf,
    pub bin_dir: PathBuf,
    pub daemon_binary_path: PathBuf,
    pub model_dir: PathBuf,
    pub fastembed_cache_dir: PathBuf,
    pub ipc_dir: PathBuf,
    pub vaults_dir: PathBuf,
}

impl SemanticHomePaths {
    pub fn new(root: PathBuf) -> Self {
        let lock_dir = root.join("lock");
        let logs_dir = root.join("logs");
        let bin_dir = root.join("bin");
        let model_dir = root.join("model");
        Self {
            manifest_path: root.join("manifest.json"),
            install_lock_path: lock_dir.join("install.lock"),
            daemon_stderr_log_path: logs_dir.join("obsidian-semanticd.stderr.log"),
            daemon_binary_path: bin_dir.join(daemon_binary_name()),
            fastembed_cache_dir: model_dir.join("fastembed-cache"),
            ipc_dir: root.join("ipc"),
            vaults_dir: root.join("vaults"),
            root,
            lock_dir,
            logs_dir,
            bin_dir,
            model_dir,
        }
    }
}

pub fn daemon_binary_name() -> &'static str {
    if cfg!(windows) {
        "obsidian-semanticd.exe"
    } else {
        "obsidian-semanticd"
    }
}

pub fn resolve_semantic_home() -> VaultResult<PathBuf> {
    if let Ok(raw) = std::env::var("OBSIDIAN_SEMANTIC_HOME") {
        let normalized = normalize_path_value(&raw);
        if !normalized.as_os_str().is_empty() {
            return Ok(normalized);
        }
    }
    resolve_default_semantic_home()
}

pub fn resolve_semantic_home_with_override(
    explicit_override: Option<&Path>,
    env_override: Option<&Path>,
) -> VaultResult<PathBuf> {
    if let Some(path) = explicit_override
        && !path.as_os_str().is_empty()
    {
        return Ok(path.to_path_buf());
    }
    if let Some(path) = env_override
        && !path.as_os_str().is_empty()
    {
        return Ok(path.to_path_buf());
    }
    resolve_semantic_home()
}

pub fn semantic_home_paths(semantic_home: &Path) -> SemanticHomePaths {
    SemanticHomePaths::new(semantic_home.to_path_buf())
}

pub fn ensure_home_layout(paths: &SemanticHomePaths) -> VaultResult<()> {
    std::fs::create_dir_all(&paths.root)?;
    std::fs::create_dir_all(&paths.lock_dir)?;
    std::fs::create_dir_all(&paths.logs_dir)?;
    std::fs::create_dir_all(&paths.bin_dir)?;
    std::fs::create_dir_all(&paths.model_dir)?;
    std::fs::create_dir_all(&paths.fastembed_cache_dir)?;
    std::fs::create_dir_all(&paths.ipc_dir)?;
    std::fs::create_dir_all(&paths.vaults_dir)?;
    Ok(())
}

pub fn default_ipc_endpoint(paths: &SemanticHomePaths) -> IpcEndpoint {
    #[cfg(unix)]
    {
        IpcEndpoint::UnixSocket(paths.ipc_dir.join("semanticd.sock"))
    }
    #[cfg(windows)]
    {
        let digest = sha256_hex(paths.root.to_string_lossy().as_bytes());
        let suffix = &digest[..12];
        IpcEndpoint::NamedPipe(format!(r"\\.\pipe\obsidian-semanticd-{suffix}"))
    }
}

pub fn compute_vault_id(vault_root: &Path) -> VaultResult<String> {
    let canonical = vault_root.canonicalize().map_err(|err| {
        VaultError::InvalidPath(format!("failed to canonicalize vault root: {err}"))
    })?;

    let mut text = canonical.to_string_lossy().to_string();
    if cfg!(windows) {
        text.make_ascii_lowercase();
    }
    Ok(sha256_hex(text.as_bytes()))
}

pub struct InstallLock {
    file: File,
}

impl InstallLock {
    /// Blocking acquire with retry loop. Suitable for synchronous contexts
    /// or when already inside `spawn_blocking`.
    pub fn acquire(paths: &SemanticHomePaths) -> VaultResult<Self> {
        use std::time::{Duration, Instant};

        std::fs::create_dir_all(&paths.lock_dir)?;
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&paths.install_lock_path)?;

        let timeout = Duration::from_secs(30);
        let retry_interval = Duration::from_millis(200);
        let start = Instant::now();
        let mut warned = false;

        loop {
            match file.try_lock_exclusive() {
                Ok(()) => return Ok(Self { file }),
                Err(err) if is_lock_contention(&err) => {
                    let elapsed = start.elapsed();
                    if elapsed >= timeout {
                        return Err(VaultError::DaemonBootstrap(format!(
                            "timed out ({}s) waiting for install lock '{}'",
                            timeout.as_secs(),
                            paths.install_lock_path.display()
                        )));
                    }
                    if !warned && elapsed >= Duration::from_secs(5) {
                        tracing::warn!(
                            "waiting for install lock '{}' (held by another process)...",
                            paths.install_lock_path.display()
                        );
                        warned = true;
                    }
                    std::thread::sleep(retry_interval);
                }
                Err(err) => {
                    return Err(VaultError::DaemonBootstrap(format!(
                        "failed to acquire install lock '{}': {err}",
                        paths.install_lock_path.display()
                    )));
                }
            }
        }
    }

    /// Async-friendly acquire that offloads the blocking retry loop to a
    /// dedicated thread, avoiding stalling the tokio worker pool.
    pub async fn acquire_async(paths: &SemanticHomePaths) -> VaultResult<Self> {
        let paths = paths.clone();
        tokio::task::spawn_blocking(move || Self::acquire(&paths))
            .await
            .map_err(|e| VaultError::DaemonBootstrap(format!("install lock task panicked: {e}")))?
    }

    pub fn try_acquire(paths: &SemanticHomePaths) -> VaultResult<Option<Self>> {
        std::fs::create_dir_all(&paths.lock_dir)?;
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&paths.install_lock_path)?;

        match file.try_lock_exclusive() {
            Ok(()) => Ok(Some(Self { file })),
            Err(err) if is_lock_contention(&err) => Ok(None),
            Err(err) => Err(VaultError::DaemonBootstrap(format!(
                "failed to acquire install lock '{}': {err}",
                paths.install_lock_path.display()
            ))),
        }
    }
}

fn is_lock_contention(err: &std::io::Error) -> bool {
    if err.kind() == std::io::ErrorKind::WouldBlock {
        return true;
    }

    #[cfg(windows)]
    {
        // Windows can report lock conflicts as sharing/lock-violation OS codes.
        matches!(err.raw_os_error(), Some(32 | 33))
    }
    #[cfg(not(windows))]
    {
        false
    }
}

impl Drop for InstallLock {
    fn drop(&mut self) {
        if let Err(err) = self.file.unlock() {
            tracing::warn!(error = %err, "failed to release semantic install lock");
        }
    }
}

fn resolve_default_semantic_home() -> VaultResult<PathBuf> {
    if cfg!(target_os = "macos") {
        let home = std::env::var_os("HOME")
            .map(PathBuf::from)
            .ok_or_else(|| VaultError::InvalidPath("HOME is not set".to_string()))?;
        return Ok(home
            .join("Library")
            .join("Application Support")
            .join("obsidian-semantic"));
    }

    if cfg!(windows) {
        let appdata = std::env::var_os("APPDATA")
            .map(PathBuf::from)
            .ok_or_else(|| VaultError::InvalidPath("APPDATA is not set".to_string()))?;
        return Ok(appdata.join("obsidian-semantic"));
    }

    let base = match std::env::var_os("XDG_DATA_HOME") {
        Some(path) => PathBuf::from(path),
        None => {
            let home = std::env::var_os("HOME")
                .map(PathBuf::from)
                .ok_or_else(|| VaultError::InvalidPath("HOME is not set".to_string()))?;
            home.join(".local").join("share")
        }
    };
    Ok(base.join("obsidian-semantic"))
}

pub(crate) fn normalize_path_value(raw: &str) -> PathBuf {
    let trimmed = raw.trim();
    let unquoted = strip_matching_outer_quotes(trimmed).trim();
    if unquoted.is_empty() {
        PathBuf::from(trimmed)
    } else {
        PathBuf::from(unquoted)
    }
}

fn strip_matching_outer_quotes(mut value: &str) -> &str {
    loop {
        let double = value.starts_with('"') && value.ends_with('"');
        let single = value.starts_with('\'') && value.ends_with('\'');
        if (double || single) && value.len() >= 2 {
            value = &value[1..value.len() - 1];
            continue;
        }
        return value;
    }
}

pub(crate) fn sha256_hex(input: &[u8]) -> String {
    let digest = Sha256::digest(input);
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_vault_id_is_stable() {
        let dir = tempfile::tempdir().expect("tempdir should be created");
        std::fs::create_dir_all(dir.path().join(".obsidian")).expect("create .obsidian");

        let first = compute_vault_id(dir.path()).expect("first id");
        let second = compute_vault_id(dir.path()).expect("second id");
        assert_eq!(first, second);
        assert_eq!(first.len(), 64);
    }

    #[test]
    fn install_lock_prevents_double_acquire() {
        let dir = tempfile::tempdir().expect("tempdir should be created");
        let paths = SemanticHomePaths::new(dir.path().join("semantic-home"));
        ensure_home_layout(&paths).expect("home layout should be created");

        let lock = InstallLock::acquire(&paths).expect("primary lock acquire should succeed");
        let second =
            InstallLock::try_acquire(&paths).expect("secondary lock attempt should not fail");
        assert!(second.is_none(), "secondary lock should be blocked");

        drop(lock);
        let third = InstallLock::try_acquire(&paths).expect("third lock attempt should not fail");
        assert!(third.is_some(), "lock should be acquirable after release");
    }

    #[test]
    fn resolve_semantic_home_with_explicit_override_wins() {
        let explicit = PathBuf::from("/tmp/explicit-home");
        let env = PathBuf::from("/tmp/env-home");
        let resolved = resolve_semantic_home_with_override(Some(&explicit), Some(&env))
            .expect("resolve semantic home with override");
        assert_eq!(resolved, explicit);
    }
}
