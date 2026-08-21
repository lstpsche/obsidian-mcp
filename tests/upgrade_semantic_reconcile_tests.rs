#![cfg(unix)]

use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use obsidian_mcp::client::semantic_daemon::{DaemonConnectPolicy, SemanticDaemonClient};
use obsidian_mcp::daemon::bootstrap::{self, BootstrapConfig};
use obsidian_mcp::daemon::home::{self, SemanticHomePaths};
use obsidian_mcp::daemon::manifest::{
    self, BinaryOrigin, ManifestIpc, RuntimeManifest, RuntimeManifestInput,
};
use obsidian_mcp::daemon::protocol::{DAEMON_API_VERSION, JSONRPC_VERSION};
use obsidian_mcp::daemon::server::IpcEndpoint;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixListener;

async fn run_fake_daemon(socket: &Path, version: &str) {
    if socket.exists() {
        let _ = std::fs::remove_file(socket);
    }
    let listener = UnixListener::bind(socket).expect("old daemon socket should bind");
    loop {
        let (stream, _) = listener.accept().await.expect("old daemon should accept");
        let (reader, mut writer) = tokio::io::split(stream);
        let mut line = String::new();
        BufReader::new(reader)
            .read_line(&mut line)
            .await
            .expect("old daemon should read request");
        let request: Value = serde_json::from_str(&line).expect("request should be JSON");
        let id = request["id"].clone();
        let method = request["method"].as_str().unwrap_or_default();
        let response = match method {
            "health" => json!({
                "jsonrpc": JSONRPC_VERSION,
                "id": id,
                "result": {
                    "daemon_version": version,
                    "daemon_api_version": DAEMON_API_VERSION,
                    "status": "ok",
                    "uptime_ms": 1,
                    "model_name": "preserved-model",
                    "semantic_home": socket.parent().unwrap().display().to_string()
                }
            }),
            "shutdown" => json!({
                "jsonrpc": JSONRPC_VERSION,
                "id": id,
                "result": { "accepted": true }
            }),
            _ => panic!("unexpected old-daemon method: {method}"),
        };
        writer
            .write_all(format!("{}\n", serde_json::to_string(&response).unwrap()).as_bytes())
            .await
            .expect("old daemon should write response");
        writer.flush().await.expect("old daemon should flush");
        if method == "shutdown" {
            drop(writer);
            drop(listener);
            let _ = std::fs::remove_file(socket);
            return;
        }
    }
}

#[tokio::test]
async fn reconcile_restarts_owned_daemon_and_preserves_runtime_state() {
    let temp = tempfile::tempdir().expect("tempdir should be created");
    let paths = SemanticHomePaths::new(temp.path().join("semantic-home"));
    home::ensure_home_layout(&paths).expect("semantic home should be created");
    let cache_marker = paths.fastembed_cache_dir.join("keep.marker");
    std::fs::write(&cache_marker, "keep").expect("cache marker should be written");
    let socket = paths.ipc_dir.join("semanticd.sock");
    let endpoint = IpcEndpoint::UnixSocket(socket.clone());
    let installed_daemon = Path::new(env!("CARGO_BIN_EXE_obsidian-semanticd"));
    let sentinel = ProcessGuard::sleep();

    let old_server = tokio::spawn({
        let socket = socket.clone();
        async move { run_fake_daemon(&socket, "0.0.1").await }
    });
    for _ in 0..50 {
        if socket.exists() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(socket.exists(), "old daemon socket should become ready");

    let old_manifest = RuntimeManifest::from_input(RuntimeManifestInput {
        daemon_api_version: DAEMON_API_VERSION,
        daemon_version: "0.0.1".into(),
        binary_path: "/bin/sleep".into(),
        binary_origin: BinaryOrigin::Sibling,
        binary_sha256: None,
        ipc: ManifestIpc {
            transport: "unix_socket".into(),
            endpoint: endpoint.endpoint_string(),
        },
        pid: sentinel.pid(),
        semantic_home: paths.root.display().to_string(),
        fastembed_cache_dir: paths.fastembed_cache_dir.display().to_string(),
        model_name: "preserved-model".into(),
        bootstrap_client_name: "old-client".into(),
        bootstrap_client_version: "0.0.1".into(),
    });
    manifest::save_atomic(&paths.manifest_path, &old_manifest).expect("old manifest should save");
    let config = BootstrapConfig {
        semantic_home_override: Some(paths.root.clone()),
        daemon_path_override: None,
        model_name: "different-config-model".into(),
        download_url_override: None,
        bootstrap_client_name: "upgrade-test".into(),
        bootstrap_client_version: env!("CARGO_PKG_VERSION").into(),
    };

    let outcome = bootstrap::reconcile_daemon_after_upgrade(
        &config,
        installed_daemon,
        env!("CARGO_PKG_VERSION"),
    )
    .await
    .expect("owned daemon should reconcile");
    old_server.await.expect("old daemon task should join");

    assert!(outcome.was_running);
    assert!(outcome.restarted);
    assert_eq!(outcome.origin, BinaryOrigin::Sibling);
    assert_eq!(
        outcome.observed_version.as_deref(),
        Some(env!("CARGO_PKG_VERSION"))
    );
    let updated = manifest::load(&paths.manifest_path)
        .expect("manifest should load")
        .expect("manifest should exist");
    assert_eq!(updated.binary_origin, BinaryOrigin::Sibling);
    assert_eq!(updated.model_name, "preserved-model");
    assert_eq!(std::fs::read_to_string(cache_marker).unwrap(), "keep");

    let client = SemanticDaemonClient::new(
        endpoint,
        DaemonConnectPolicy {
            timeout: Duration::from_secs(2),
            retries: 0,
            retry_backoff: Duration::ZERO,
        },
    );
    assert!(
        client
            .shutdown()
            .await
            .expect("updated daemon should stop")
            .accepted
    );
    for _ in 0..50 {
        if !socket.exists() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(!socket.exists(), "updated daemon should remove its socket");
}

#[tokio::test]
async fn reconcile_leaves_explicit_override_daemon_running() {
    let temp = tempfile::tempdir().expect("tempdir should be created");
    let paths = SemanticHomePaths::new(temp.path().join("semantic-home"));
    home::ensure_home_layout(&paths).expect("semantic home should be created");
    let socket = paths.ipc_dir.join("external.sock");
    let endpoint = IpcEndpoint::UnixSocket(socket.clone());
    let old_server = tokio::spawn({
        let socket = socket.clone();
        async move { run_fake_daemon(&socket, "0.0.1").await }
    });
    for _ in 0..50 {
        if socket.exists() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    let manifest = RuntimeManifest::from_input(RuntimeManifestInput {
        daemon_api_version: DAEMON_API_VERSION,
        daemon_version: "0.0.1".into(),
        binary_path: "/external/obsidian-semanticd".into(),
        binary_origin: BinaryOrigin::Override,
        binary_sha256: None,
        ipc: ManifestIpc {
            transport: "unix_socket".into(),
            endpoint: endpoint.endpoint_string(),
        },
        pid: 4242,
        semantic_home: paths.root.display().to_string(),
        fastembed_cache_dir: paths.fastembed_cache_dir.display().to_string(),
        model_name: "preserved-model".into(),
        bootstrap_client_name: "old-client".into(),
        bootstrap_client_version: "0.0.1".into(),
    });
    manifest::save_atomic(&paths.manifest_path, &manifest).expect("manifest should save");
    let config = BootstrapConfig {
        semantic_home_override: Some(paths.root.clone()),
        daemon_path_override: Some("/external/obsidian-semanticd".into()),
        model_name: "model".into(),
        download_url_override: None,
        bootstrap_client_name: "upgrade-test".into(),
        bootstrap_client_version: env!("CARGO_PKG_VERSION").into(),
    };

    let outcome = bootstrap::reconcile_daemon_after_upgrade(
        &config,
        Path::new(env!("CARGO_BIN_EXE_obsidian-semanticd")),
        env!("CARGO_PKG_VERSION"),
    )
    .await
    .expect("external daemon should be left alone");
    assert!(outcome.was_running);
    assert!(!outcome.restarted);
    assert_eq!(outcome.origin, BinaryOrigin::Override);
    assert!(socket.exists(), "external daemon must remain running");

    let client = SemanticDaemonClient::new(
        endpoint,
        DaemonConnectPolicy {
            timeout: Duration::from_secs(2),
            retries: 0,
            retry_backoff: Duration::ZERO,
        },
    );
    client
        .shutdown()
        .await
        .expect("test daemon should stop after assertion");
    old_server.await.expect("external daemon task should join");
}

#[tokio::test]
async fn reconcile_refuses_to_stop_a_responder_with_mismatched_executable_identity() {
    let temp = tempfile::tempdir().expect("tempdir should be created");
    let paths = SemanticHomePaths::new(temp.path().join("semantic-home"));
    home::ensure_home_layout(&paths).expect("semantic home should be created");
    let socket = paths.ipc_dir.join("mismatched.sock");
    let endpoint = IpcEndpoint::UnixSocket(socket.clone());
    let server = tokio::spawn({
        let socket = socket.clone();
        async move { run_fake_daemon(&socket, "0.0.1").await }
    });
    wait_for_socket(&socket).await;

    let sentinel = ProcessGuard::sleep();
    save_manifest(
        &paths,
        &endpoint,
        "0.0.1",
        BinaryOrigin::Sibling,
        Path::new(env!("CARGO_BIN_EXE_obsidian-semanticd")),
        sentinel.pid(),
    );

    let error = bootstrap::reconcile_daemon_after_upgrade(
        &test_config(&paths),
        Path::new(env!("CARGO_BIN_EXE_obsidian-semanticd")),
        env!("CARGO_PKG_VERSION"),
    )
    .await
    .expect_err("mismatched live executable must not be stopped");
    assert!(
        error
            .to_string()
            .contains("executable identity does not match")
    );
    assert!(socket.exists(), "mismatched responder must remain running");

    shutdown(endpoint).await;
    server.await.expect("fake daemon task should join");
}

#[tokio::test]
async fn reconcile_never_downgrades_a_newer_owned_daemon() {
    let temp = tempfile::tempdir().expect("tempdir should be created");
    let paths = SemanticHomePaths::new(temp.path().join("semantic-home"));
    home::ensure_home_layout(&paths).expect("semantic home should be created");
    let socket = paths.ipc_dir.join("newer.sock");
    let endpoint = IpcEndpoint::UnixSocket(socket.clone());
    let server = tokio::spawn({
        let socket = socket.clone();
        async move { run_fake_daemon(&socket, "999.0.0").await }
    });
    wait_for_socket(&socket).await;

    save_manifest(
        &paths,
        &endpoint,
        "999.0.0",
        BinaryOrigin::Sibling,
        Path::new(env!("CARGO_BIN_EXE_obsidian-semanticd")),
        4242,
    );
    let config = test_config(&paths);
    let outcome = bootstrap::reconcile_daemon_after_upgrade(
        &config,
        Path::new(env!("CARGO_BIN_EXE_obsidian-semanticd")),
        env!("CARGO_PKG_VERSION"),
    )
    .await
    .expect("newer daemon should be preserved");

    assert!(!outcome.restarted);
    assert_eq!(outcome.observed_version.as_deref(), Some("999.0.0"));
    assert!(socket.exists());
    shutdown(endpoint).await;
    server.await.expect("fake daemon task should join");
}

#[tokio::test]
async fn normal_bootstrap_completes_interrupted_owned_daemon_activation() {
    let temp = tempfile::tempdir().expect("tempdir should be created");
    let paths = SemanticHomePaths::new(temp.path().join("semantic-home"));
    home::ensure_home_layout(&paths).expect("semantic home should be created");
    let socket = paths.ipc_dir.join("interrupted.sock");
    let endpoint = IpcEndpoint::UnixSocket(socket.clone());
    let server = tokio::spawn({
        let socket = socket.clone();
        async move { run_fake_daemon(&socket, "0.0.1").await }
    });
    wait_for_socket(&socket).await;

    let sentinel = ProcessGuard::sleep();
    save_manifest(
        &paths,
        &endpoint,
        "0.0.1",
        BinaryOrigin::Sibling,
        Path::new("/bin/sleep"),
        sentinel.pid(),
    );
    let result = bootstrap::ensure_daemon_from_sibling(
        &test_config(&paths),
        Path::new(env!("CARGO_BIN_EXE_obsidian-semanticd")),
    )
    .await
    .expect("normal bootstrap should activate the newer sibling");
    server.await.expect("old daemon task should join");

    assert!(!result.reused_existing);
    assert_eq!(result.manifest.daemon_version, env!("CARGO_PKG_VERSION"));
    assert_eq!(result.manifest.model_name, "preserved-model");
    shutdown(result.endpoint).await;
}

async fn wait_for_socket(socket: &Path) {
    for _ in 0..50 {
        if socket.exists() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("fake daemon socket did not become ready");
}

fn test_config(paths: &SemanticHomePaths) -> BootstrapConfig {
    BootstrapConfig {
        semantic_home_override: Some(paths.root.clone()),
        daemon_path_override: None,
        model_name: "different-config-model".into(),
        download_url_override: None,
        bootstrap_client_name: "upgrade-test".into(),
        bootstrap_client_version: env!("CARGO_PKG_VERSION").into(),
    }
}

fn save_manifest(
    paths: &SemanticHomePaths,
    endpoint: &IpcEndpoint,
    version: &str,
    origin: BinaryOrigin,
    binary_path: &Path,
    pid: u32,
) {
    let manifest = RuntimeManifest::from_input(RuntimeManifestInput {
        daemon_api_version: DAEMON_API_VERSION,
        daemon_version: version.into(),
        binary_path: binary_path.display().to_string(),
        binary_origin: origin,
        binary_sha256: None,
        ipc: ManifestIpc {
            transport: "unix_socket".into(),
            endpoint: endpoint.endpoint_string(),
        },
        pid,
        semantic_home: paths.root.display().to_string(),
        fastembed_cache_dir: paths.fastembed_cache_dir.display().to_string(),
        model_name: "preserved-model".into(),
        bootstrap_client_name: "old-client".into(),
        bootstrap_client_version: version.into(),
    });
    manifest::save_atomic(&paths.manifest_path, &manifest).expect("manifest should save");
}

struct ProcessGuard(Child);

impl ProcessGuard {
    fn sleep() -> Self {
        Self(
            Command::new("/bin/sleep")
                .arg("30")
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .expect("sentinel process should start"),
        )
    }

    fn pid(&self) -> u32 {
        self.0.id()
    }
}

impl Drop for ProcessGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

async fn shutdown(endpoint: IpcEndpoint) {
    let client = SemanticDaemonClient::new(
        endpoint,
        DaemonConnectPolicy {
            timeout: Duration::from_secs(2),
            retries: 0,
            retry_backoff: Duration::ZERO,
        },
    );
    client
        .shutdown()
        .await
        .expect("test daemon should stop cleanly");
}
