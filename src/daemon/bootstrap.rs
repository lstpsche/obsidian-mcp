//! Daemon installer/bootstrap manager for shared semantic runtime.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use serde_json::json;
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader};

#[cfg(unix)]
use tokio::net::UnixStream;
#[cfg(windows)]
use tokio::net::windows::named_pipe::ClientOptions;

use crate::config::SemanticRuntimeConfig;
use crate::error::{VaultError, VaultResult};

use super::home::{self, InstallLock, SemanticHomePaths};
use super::manifest::{self, ManifestIpc, RuntimeManifest, RuntimeManifestInput};
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

#[derive(Debug)]
enum HealthProbeOutcome {
    Healthy(protocol::HealthResult),
    Unreachable,
    Incompatible(String),
    Invalid(String),
}

/// Ensure a shared daemon exists and is healthy.
pub async fn ensure_daemon(config: &BootstrapConfig) -> VaultResult<BootstrapResult> {
    let semantic_home =
        home::resolve_semantic_home_with_override(config.semantic_home_override.as_deref(), None)?;
    let paths = home::semantic_home_paths(&semantic_home);
    home::ensure_home_layout(&paths)?;

    let _install_lock = InstallLock::acquire_async(&paths).await?;
    let mut existing_manifest = manifest::load(&paths.manifest_path)?;
    let default_endpoint = home::default_ipc_endpoint(&paths);

    if let Some(current_manifest) = existing_manifest.as_mut() {
        let endpoint =
            endpoint_from_manifest(current_manifest).unwrap_or_else(|| default_endpoint.clone());
        match probe_health(&endpoint).await? {
            HealthProbeOutcome::Healthy(_) => {
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

    let mut daemon_binary_path =
        resolve_daemon_binary_path(config, &paths, existing_manifest.as_ref())?;
    let mut binary_sha256 = existing_manifest
        .as_ref()
        .and_then(|manifest| manifest.binary_sha256.clone());

    if !daemon_binary_path.exists() {
        if config.daemon_path_override.is_some() {
            return Err(VaultError::DaemonBootstrap(format!(
                "daemon override path does not exist: {}",
                daemon_binary_path.display()
            )));
        }
        if let Some(path_binary) = find_daemon_on_path() {
            tracing::info!(path = %path_binary.display(), "using daemon binary found on $PATH");
            daemon_binary_path = path_binary;
        } else {
            let download_url = resolve_download_url(config)?;
            tracing::info!(url = %download_url, "downloading semantic daemon binary");
            let checksum = download_and_install(&download_url, &daemon_binary_path).await?;
            binary_sha256 = Some(checksum);
        }
    }

    let mut child =
        start_daemon_process(&daemon_binary_path, &paths, &endpoint, &config.model_name)?;
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
    let ipc = ManifestIpc {
        transport: endpoint_transport(&endpoint).to_string(),
        endpoint: endpoint.endpoint_string(),
    };

    let runtime_manifest = RuntimeManifest::from_input(RuntimeManifestInput {
        daemon_api_version: health.daemon_api_version,
        daemon_version: health.daemon_version,
        binary_path: daemon_binary_path.display().to_string(),
        binary_sha256,
        ipc,
        pid,
        semantic_home: semantic_home.display().to_string(),
        fastembed_cache_dir: paths.fastembed_cache_dir.display().to_string(),
        model_name: config.model_name.clone(),
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

fn resolve_daemon_binary_path(
    config: &BootstrapConfig,
    paths: &SemanticHomePaths,
    existing_manifest: Option<&RuntimeManifest>,
) -> VaultResult<PathBuf> {
    if let Some(path) = config.daemon_path_override.as_ref() {
        return Ok(path.clone());
    }

    if let Some(manifest) = existing_manifest {
        let manifest_path = PathBuf::from(&manifest.binary_path);
        if !manifest_path.as_os_str().is_empty() {
            return Ok(manifest_path);
        }
    }

    Ok(paths.daemon_binary_path.clone())
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
