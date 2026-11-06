#![cfg(all(unix, feature = "embeddings"))]

#[path = "isolated_command.rs"]
mod command;

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use obsidian_mcp::client::semantic_daemon::{
    DaemonConnectPolicy, SemanticDaemonClient, VaultAttachment,
};
use obsidian_mcp::daemon::protocol::{
    DAEMON_API_VERSION, IndexScope, OpenHintResult, SearchResult,
};
use obsidian_mcp::daemon::server::IpcEndpoint;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

static REQUEST_ID: AtomicU64 = AtomicU64::new(1);

pub struct DaemonTestServer {
    _semantic_home: tempfile::TempDir,
    endpoint_path: PathBuf,
    client: SemanticDaemonClient,
    child: tokio::process::Child,
}

impl DaemonTestServer {
    pub async fn start(model_name: &str) -> Self {
        let semantic_home = tempfile::tempdir().expect("semantic home tempdir should be created");
        let endpoint_path = semantic_home.path().join("ipc").join("semanticd.sock");

        let binary = std::env::var_os("OBSIDIAN_TEST_DAEMON_BINARY")
            .unwrap_or_else(|| env!("CARGO_BIN_EXE_obsidian-semanticd").into());
        let mut child = command::isolated_command(binary)
            .env("OBSIDIAN_SEMANTIC_HOME", semantic_home.path())
            .env("OBSIDIAN_SEMANTIC_ENDPOINT", &endpoint_path)
            .env("OBSIDIAN_SEMANTIC_MODEL", model_name)
            .env("OBSIDIAN_EMBEDDING_PROVIDER", "local")
            .env("OBSIDIAN_LOG_LEVEL", "error")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .kill_on_drop(true)
            .spawn()
            .expect("start local-model daemon");

        let client = SemanticDaemonClient::new(
            IpcEndpoint::UnixSocket(endpoint_path.clone()),
            DaemonConnectPolicy::default(),
        );
        tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                assert!(
                    child.try_wait().unwrap().is_none(),
                    "local-model daemon exited during startup"
                );
                if client.health("daemon-test", "1").await.is_ok() {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("local-model daemon must serve health");

        Self {
            _semantic_home: semantic_home,
            client,
            endpoint_path,
            child,
        }
    }

    pub fn endpoint_path(&self) -> &Path {
        &self.endpoint_path
    }

    pub async fn shutdown(mut self) {
        assert!(
            self.client
                .shutdown()
                .await
                .expect("request daemon shutdown")
                .accepted
        );
        let status = tokio::time::timeout(Duration::from_secs(10), self.child.wait())
            .await
            .expect("daemon should shut down")
            .expect("wait for local-model daemon");
        assert!(status.success(), "daemon shutdown failed: {status}");
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

    pub async fn ensure_vault(&self, vault_root: &Path, watch: bool) -> VaultAttachment {
        self.client
            .attach_vault(vault_root, &IndexScope::default(), watch, None)
            .await
            .expect("scoped attachment should be established")
    }

    pub async fn ensure_vault_ready(
        &self,
        vault_root: &Path,
        watch: bool,
        timeout: Duration,
    ) -> VaultAttachment {
        let attachment = self.ensure_vault(vault_root, watch).await;
        let scope = IndexScope::default();
        let deadline = std::time::Instant::now() + timeout;
        loop {
            let status = self
                .client
                .vault_status(vault_root, &scope)
                .await
                .expect("read scope status");
            if status.ready {
                return attachment;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "timed out waiting for semantic readiness: {status:?}"
            );
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    pub async fn search_semantic(
        &self,
        attachment: &VaultAttachment,
        query: &str,
        top_k: usize,
        include_content: bool,
    ) -> SearchResult {
        self.client
            .search_semantic(attachment, query, top_k, include_content)
            .await
            .expect("semantic search should succeed")
    }

    pub async fn search_hybrid(
        &self,
        attachment: &VaultAttachment,
        query: &str,
        top_k: usize,
        prefetch: usize,
        alpha: f32,
        include_content: bool,
    ) -> SearchResult {
        self.client
            .search_hybrid(attachment, query, top_k, prefetch, alpha, include_content)
            .await
            .expect("hybrid search should succeed")
    }

    pub async fn open_hint(
        &self,
        vault_root: &Path,
        attachment: &VaultAttachment,
        path: &str,
    ) -> OpenHintResult {
        self.request_typed(
            "open_hint",
            json!({
                "vault_root": vault_root.display().to_string(),
                "path": path,
                "attachment_id": attachment.result.attachment_id
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
