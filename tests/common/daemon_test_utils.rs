#![cfg(all(unix, feature = "embeddings"))]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use obsidian_mcp::daemon::protocol::{
    DAEMON_API_VERSION, EnsureVaultResult, OpenHintResult, SearchResult,
};
use obsidian_mcp::daemon::server::{self, DaemonServerConfig, IpcEndpoint};
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

static REQUEST_ID: AtomicU64 = AtomicU64::new(1);

pub struct DaemonTestServer {
    _semantic_home: tempfile::TempDir,
    endpoint_path: PathBuf,
    shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
    task: tokio::task::JoinHandle<obsidian_mcp::error::VaultResult<()>>,
}

impl DaemonTestServer {
    pub async fn start(model_name: &str) -> Self {
        let semantic_home = tempfile::tempdir().expect("semantic home tempdir should be created");
        let endpoint_path = semantic_home.path().join("ipc").join("semanticd.sock");

        let config = DaemonServerConfig {
            endpoint: IpcEndpoint::UnixSocket(endpoint_path.clone()),
            model_name: model_name.to_string(),
            semantic_home: semantic_home.path().to_path_buf(),
        };

        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
        let task = tokio::spawn(async move {
            server::run_with_shutdown(config, async move {
                let _ = shutdown_rx.await;
            })
            .await
        });

        let mut ready = false;
        for _ in 0..150 {
            if tokio::net::UnixStream::connect(&endpoint_path)
                .await
                .is_ok()
            {
                ready = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert!(ready, "daemon test server did not become ready in time");

        Self {
            _semantic_home: semantic_home,
            endpoint_path,
            shutdown_tx: Some(shutdown_tx),
            task,
        }
    }

    pub fn endpoint_path(&self) -> &Path {
        &self.endpoint_path
    }

    pub async fn shutdown(mut self) {
        if let Some(shutdown_tx) = self.shutdown_tx.take() {
            let _ = shutdown_tx.send(());
        }
        let joined = self
            .task
            .await
            .expect("daemon test server task should join");
        joined.expect("daemon test server should exit cleanly");
    }

    pub async fn request_value(&self, method: &str, params: Value) -> Value {
        rpc_request(&self.endpoint_path, method, params).await
    }

    pub async fn request_typed<T: DeserializeOwned>(&self, method: &str, params: Value) -> T {
        let response = self.request_value(method, params).await;
        if let Some(error) = response.get("error")
            && !error.is_null()
        {
            panic!("expected successful response, got error: {error}");
        }
        serde_json::from_value(response["result"].clone())
            .expect("result payload should deserialize")
    }

    pub async fn health_api_version(&self) -> u32 {
        let response = self
            .request_value(
                "health",
                json!({
                    "client_name": "daemon-test",
                    "client_version": "0.0.0",
                    "min_api_version": DAEMON_API_VERSION,
                    "max_api_version": DAEMON_API_VERSION,
                }),
            )
            .await;
        assert!(response["error"].is_null(), "health should succeed");
        response["result"]["daemon_api_version"]
            .as_u64()
            .expect("daemon_api_version should be present") as u32
    }

    pub async fn ensure_vault(&self, vault_root: &Path, watch: bool) -> EnsureVaultResult {
        self.request_typed(
            "ensure_vault",
            json!({
                "vault_root": vault_root.display().to_string(),
                "watch": watch
            }),
        )
        .await
    }

    pub async fn search_semantic(
        &self,
        vault_root: &Path,
        query: &str,
        top_k: usize,
        include_content: bool,
    ) -> SearchResult {
        self.request_typed(
            "search_semantic",
            json!({
                "vault_root": vault_root.display().to_string(),
                "query": query,
                "top_k": top_k,
                "include_content": include_content
            }),
        )
        .await
    }

    pub async fn search_hybrid(
        &self,
        vault_root: &Path,
        query: &str,
        top_k: usize,
        prefetch: usize,
        alpha: f32,
        include_content: bool,
    ) -> SearchResult {
        self.request_typed(
            "search_hybrid",
            json!({
                "vault_root": vault_root.display().to_string(),
                "query": query,
                "top_k": top_k,
                "prefetch": prefetch,
                "alpha": alpha,
                "include_content": include_content
            }),
        )
        .await
    }

    pub async fn open_hint(&self, vault_root: &Path, path: &str) -> OpenHintResult {
        self.request_typed(
            "open_hint",
            json!({
                "vault_root": vault_root.display().to_string(),
                "path": path
            }),
        )
        .await
    }
}

pub async fn rpc_request(endpoint_path: &Path, method: &str, params: Value) -> Value {
    let mut stream = tokio::net::UnixStream::connect(endpoint_path)
        .await
        .expect("request stream should connect");

    let request_id = REQUEST_ID.fetch_add(1, Ordering::Relaxed);
    let request = json!({
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": params
    });

    let encoded = serde_json::to_string(&request).expect("request should serialize");
    stream
        .write_all(encoded.as_bytes())
        .await
        .expect("request write should succeed");
    stream
        .write_all(b"\n")
        .await
        .expect("request newline write should succeed");
    stream.flush().await.expect("request flush should succeed");

    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader
        .read_line(&mut line)
        .await
        .expect("response should be readable");
    serde_json::from_str(&line).expect("response JSON should parse")
}

pub fn create_temp_vault() -> tempfile::TempDir {
    let vault = tempfile::tempdir().expect("vault tempdir should be created");
    std::fs::create_dir_all(vault.path().join(".obsidian")).expect("create .obsidian");
    vault
}

pub fn write_note(vault_root: &Path, relative: &str, content: &str) {
    let path = vault_root.join(relative);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create parent dirs");
    }
    std::fs::write(path, content).expect("write note");
}

pub fn write_note_bytes(vault_root: &Path, relative: &str, content: &[u8]) {
    let path = vault_root.join(relative);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create parent dirs");
    }
    std::fs::write(path, content).expect("write note bytes");
}
