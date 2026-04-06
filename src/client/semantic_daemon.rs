//! MCP-side JSON-RPC client for the local semantic daemon.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader};

#[cfg(unix)]
use tokio::net::UnixStream;
#[cfg(windows)]
use tokio::net::windows::named_pipe::ClientOptions;

use crate::config::SemanticRuntimeConfig;
use crate::daemon::protocol::{
    self, DAEMON_API_VERSION, EnsureVaultResult, HealthResult, SearchResult,
};
use crate::daemon::server::IpcEndpoint;
use crate::error::{VaultError, VaultResult};

trait AsyncReadWrite: AsyncRead + AsyncWrite + Send + Unpin {}
impl<T> AsyncReadWrite for T where T: AsyncRead + AsyncWrite + Send + Unpin {}
type DaemonStream = Box<dyn AsyncReadWrite>;

#[derive(Debug, Clone, Copy)]
pub struct DaemonConnectPolicy {
    pub timeout: Duration,
    pub retries: u32,
    pub retry_backoff: Duration,
}

impl DaemonConnectPolicy {
    pub fn from_runtime_config(runtime: &SemanticRuntimeConfig) -> Self {
        Self {
            timeout: Duration::from_millis(runtime.connect_timeout_ms),
            retries: runtime.connect_retries,
            retry_backoff: Duration::from_millis(runtime.retry_backoff_ms),
        }
    }
}

impl Default for DaemonConnectPolicy {
    fn default() -> Self {
        Self {
            timeout: Duration::from_millis(2_000),
            retries: 2,
            retry_backoff: Duration::from_millis(250),
        }
    }
}

#[derive(Clone)]
pub struct SemanticDaemonClient {
    inner: Arc<ClientInner>,
}

struct ClientInner {
    endpoint: IpcEndpoint,
    policy: DaemonConnectPolicy,
    request_id: AtomicU64,
}

impl SemanticDaemonClient {
    pub fn new(endpoint: IpcEndpoint, policy: DaemonConnectPolicy) -> Self {
        Self {
            inner: Arc::new(ClientInner {
                endpoint,
                policy,
                request_id: AtomicU64::new(1),
            }),
        }
    }

    pub fn endpoint(&self) -> &IpcEndpoint {
        &self.inner.endpoint
    }

    pub fn policy(&self) -> DaemonConnectPolicy {
        self.inner.policy
    }

    pub async fn health(
        &self,
        client_name: &str,
        client_version: &str,
    ) -> VaultResult<HealthResult> {
        self.call(
            "health",
            protocol::HealthParams {
                client_name: Some(client_name.to_string()),
                client_version: Some(client_version.to_string()),
                min_api_version: Some(DAEMON_API_VERSION),
                max_api_version: Some(DAEMON_API_VERSION),
            },
        )
        .await
    }

    pub async fn ensure_vault(
        &self,
        vault_root: &Path,
        watch: bool,
        model_name: Option<&str>,
    ) -> VaultResult<EnsureVaultResult> {
        self.call(
            "ensure_vault",
            protocol::EnsureVaultParams {
                vault_root: vault_root.display().to_string(),
                watch: Some(watch),
                model_name: model_name.map(|value| value.to_string()),
            },
        )
        .await
    }

    pub async fn search_semantic(
        &self,
        vault_root: &Path,
        query: &str,
        top_k: usize,
        include_content: bool,
    ) -> VaultResult<SearchResult> {
        self.call(
            "search_semantic",
            protocol::SearchSemanticParams {
                vault_root: vault_root.display().to_string(),
                query: query.to_string(),
                top_k: Some(top_k),
                include_content: Some(include_content),
            },
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
    ) -> VaultResult<SearchResult> {
        self.call(
            "search_hybrid",
            protocol::SearchHybridParams {
                vault_root: vault_root.display().to_string(),
                query: query.to_string(),
                top_k: Some(top_k),
                prefetch: Some(prefetch),
                alpha: Some(alpha),
                include_content: Some(include_content),
            },
        )
        .await
    }

    async fn call<P, R>(&self, method: &str, params: P) -> VaultResult<R>
    where
        P: Serialize,
        R: DeserializeOwned,
    {
        let params = serde_json::to_value(params).map_err(|err| {
            VaultError::DaemonProtocol(format!(
                "failed to serialize daemon params for '{method}': {err}"
            ))
        })?;
        let attempts = self.inner.policy.retries.saturating_add(1);

        let mut last_error = None;
        for attempt_idx in 0..attempts {
            match self.call_once(method, params.clone()).await {
                Ok(result) => return Ok(result),
                Err(err) => {
                    let is_last = attempt_idx + 1 >= attempts;
                    let retryable = is_retryable_error(&err);
                    if is_last || !retryable {
                        return Err(err);
                    }

                    let backoff_ms = self
                        .inner
                        .policy
                        .retry_backoff
                        .as_millis()
                        .saturating_mul((attempt_idx + 1) as u128)
                        .min(u64::MAX as u128) as u64;

                    tracing::warn!(
                        method,
                        attempt = attempt_idx + 1,
                        retries = attempts - 1,
                        backoff_ms,
                        error = %err,
                        "daemon call failed; retrying"
                    );

                    last_error = Some(err);
                    tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            VaultError::DaemonProtocol(format!("daemon call '{method}' exhausted attempts"))
        }))
    }

    async fn call_once<R>(&self, method: &str, params: Value) -> VaultResult<R>
    where
        R: DeserializeOwned,
    {
        let request_id = self.inner.request_id.fetch_add(1, Ordering::Relaxed);
        let request = json!({
            "jsonrpc": protocol::JSONRPC_VERSION,
            "id": request_id,
            "method": method,
            "params": params,
        });
        let encoded = serde_json::to_string(&request).map_err(|err| {
            VaultError::DaemonProtocol(format!(
                "failed to serialize daemon request '{method}': {err}"
            ))
        })?;

        let timeout_ms = self.inner.policy.timeout.as_millis().min(u64::MAX as u128) as u64;
        let operation = method.to_string();
        let future = async {
            let mut stream = self.connect().await?;
            stream.write_all(encoded.as_bytes()).await.map_err(|err| {
                VaultError::DaemonIpc(format!(
                    "failed to write daemon request '{method}' to '{}': {err}",
                    self.inner.endpoint.endpoint_string()
                ))
            })?;
            stream.write_all(b"\n").await.map_err(|err| {
                VaultError::DaemonIpc(format!(
                    "failed to finish daemon request '{method}' to '{}': {err}",
                    self.inner.endpoint.endpoint_string()
                ))
            })?;
            stream.flush().await.map_err(|err| {
                VaultError::DaemonIpc(format!(
                    "failed to flush daemon request '{method}' to '{}': {err}",
                    self.inner.endpoint.endpoint_string()
                ))
            })?;

            let mut reader = BufReader::new(stream);
            let mut line = String::new();
            let read = reader.read_line(&mut line).await.map_err(|err| {
                VaultError::DaemonIpc(format!(
                    "failed to read daemon response for '{method}' from '{}': {err}",
                    self.inner.endpoint.endpoint_string()
                ))
            })?;
            if read == 0 {
                return Err(VaultError::DaemonIpc(format!(
                    "daemon endpoint '{}' closed connection for '{method}'",
                    self.inner.endpoint.endpoint_string()
                )));
            }

            parse_response(method, request_id, &line)
        };

        match tokio::time::timeout(self.inner.policy.timeout, future).await {
            Ok(result) => result,
            Err(_) => Err(VaultError::DaemonTimeout {
                operation,
                timeout_ms,
            }),
        }
    }

    async fn connect(&self) -> VaultResult<DaemonStream> {
        match &self.inner.endpoint {
            #[cfg(unix)]
            IpcEndpoint::UnixSocket(path) => {
                let stream = UnixStream::connect(path).await.map_err(|err| {
                    VaultError::DaemonIpc(format!(
                        "failed to connect to daemon socket '{}': {err}",
                        path.display()
                    ))
                })?;
                Ok(Box::new(stream))
            }
            #[cfg(windows)]
            IpcEndpoint::NamedPipe(name) => {
                let stream = ClientOptions::new().open(name).map_err(|err| {
                    VaultError::DaemonIpc(format!(
                        "failed to connect to daemon named pipe '{}': {err}",
                        name
                    ))
                })?;
                Ok(Box::new(stream))
            }
        }
    }
}

fn parse_response<R>(method: &str, request_id: u64, line: &str) -> VaultResult<R>
where
    R: DeserializeOwned,
{
    let response: Value = serde_json::from_str(line).map_err(|err| {
        VaultError::DaemonProtocol(format!(
            "failed to parse daemon response JSON for '{method}': {err}"
        ))
    })?;

    let Some(jsonrpc) = response.get("jsonrpc").and_then(Value::as_str) else {
        return Err(VaultError::DaemonProtocol(format!(
            "daemon response for '{method}' missing jsonrpc version"
        )));
    };
    if jsonrpc != protocol::JSONRPC_VERSION {
        return Err(VaultError::DaemonProtocol(format!(
            "daemon response for '{method}' had invalid jsonrpc version"
        )));
    }

    let expected_id = Value::from(request_id);
    let Some(response_id) = response.get("id") else {
        return Err(VaultError::DaemonProtocol(format!(
            "daemon response for '{method}' missing id"
        )));
    };
    if *response_id != expected_id {
        return Err(VaultError::DaemonProtocol(format!(
            "daemon response for '{method}' had mismatched id"
        )));
    }

    if let Some(error) = response.get("error").filter(|value| !value.is_null()) {
        let code = error
            .get("code")
            .and_then(Value::as_i64)
            .unwrap_or(protocol::ERR_INTERNAL);
        let message = error
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("daemon RPC error")
            .to_string();
        let data = error.get("data").cloned();
        return Err(VaultError::DaemonRpc {
            code,
            message,
            data,
        });
    }

    let Some(result) = response.get("result") else {
        return Err(VaultError::DaemonProtocol(format!(
            "daemon response for '{method}' missing result"
        )));
    };

    serde_json::from_value(result.clone()).map_err(|err| {
        VaultError::DaemonProtocol(format!(
            "failed to decode daemon result for '{method}': {err}"
        ))
    })
}

fn is_retryable_error(err: &VaultError) -> bool {
    matches!(
        err,
        VaultError::DaemonIpc(_) | VaultError::DaemonTimeout { .. }
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    use std::path::PathBuf;
    #[cfg(unix)]
    use tokio::net::UnixListener;

    #[cfg(unix)]
    async fn start_unix_server_once(
        socket_path: PathBuf,
        response_fn: impl FnOnce(Value) -> Value + Send + 'static,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            if socket_path.exists() {
                let _ = std::fs::remove_file(&socket_path);
            }
            let listener = UnixListener::bind(&socket_path).expect("bind unix socket");
            let (stream, _) = listener.accept().await.expect("accept client");
            let (reader, mut writer) = tokio::io::split(stream);
            let mut reader = BufReader::new(reader);
            let mut line = String::new();
            reader.read_line(&mut line).await.expect("read request");
            let request: Value = serde_json::from_str(&line).expect("parse request");

            let response = response_fn(request);
            let encoded = serde_json::to_string(&response).expect("serialize response");
            writer
                .write_all(format!("{encoded}\n").as_bytes())
                .await
                .expect("write response");
            writer.flush().await.expect("flush response");
        })
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn health_call_decodes_success_response() {
        let temp = tempfile::tempdir().expect("tempdir");
        let socket_path = temp.path().join("semanticd.sock");
        let server = start_unix_server_once(socket_path.clone(), |request| {
            let id = request
                .get("id")
                .cloned()
                .expect("request id should be present");
            json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "daemon_version": "1.0.1",
                    "daemon_api_version": 1,
                    "status": "ok",
                    "uptime_ms": 10,
                    "model_name": "BAAI/bge-small-en-v1.5",
                    "semantic_home": "/tmp/semantic"
                }
            })
        })
        .await;

        let client = SemanticDaemonClient::new(
            IpcEndpoint::UnixSocket(socket_path),
            DaemonConnectPolicy::default(),
        );
        let health = client
            .health("obsidian-mcp-test", "1.0.1")
            .await
            .expect("health should succeed");
        assert_eq!(health.daemon_api_version, DAEMON_API_VERSION);

        server.await.expect("server task should complete");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn call_retries_when_endpoint_is_temporarily_missing() {
        let temp = tempfile::tempdir().expect("tempdir");
        let socket_path = temp.path().join("semanticd.sock");
        let delayed_socket = socket_path.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(200)).await;
            let server = start_unix_server_once(delayed_socket, |request| {
                let id = request
                    .get("id")
                    .cloned()
                    .expect("request id should be present");
                json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {
                        "daemon_version": "1.0.1",
                        "daemon_api_version": 1,
                        "status": "ok",
                        "uptime_ms": 12,
                        "model_name": "BAAI/bge-small-en-v1.5",
                        "semantic_home": "/tmp/semantic"
                    }
                })
            })
            .await;
            server.await.expect("delayed server task should complete");
        });

        let client = SemanticDaemonClient::new(
            IpcEndpoint::UnixSocket(socket_path),
            DaemonConnectPolicy {
                timeout: Duration::from_secs(2),
                retries: 4,
                retry_backoff: Duration::from_millis(150),
            },
        );
        let health = client.health("obsidian-mcp-test", "1.0.1").await;
        assert!(
            health.is_ok(),
            "call should eventually succeed with retries"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn rpc_errors_are_reported_with_code_and_message() {
        let temp = tempfile::tempdir().expect("tempdir");
        let socket_path = temp.path().join("semanticd.sock");
        let server = start_unix_server_once(socket_path.clone(), |request| {
            let id = request
                .get("id")
                .cloned()
                .expect("request id should be present");
            json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": {
                    "code": -32030,
                    "message": "vault not ready; call ensure_vault first",
                    "data": { "vault_root": "/tmp/vault" }
                }
            })
        })
        .await;

        let client = SemanticDaemonClient::new(
            IpcEndpoint::UnixSocket(socket_path),
            DaemonConnectPolicy::default(),
        );
        let result = client.health("obsidian-mcp-test", "1.0.1").await;
        match result {
            Err(VaultError::DaemonRpc {
                code,
                message,
                data,
            }) => {
                assert_eq!(code, -32030);
                assert!(message.contains("vault not ready"));
                assert!(data.is_some());
            }
            other => panic!("expected daemon RPC error, got: {other:?}"),
        }

        server.await.expect("server task should complete");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn response_with_mismatched_id_is_rejected() {
        let temp = tempfile::tempdir().expect("tempdir");
        let socket_path = temp.path().join("semanticd.sock");
        let server = start_unix_server_once(socket_path.clone(), |_request| {
            json!({
                "jsonrpc": "2.0",
                "id": 999,
                "result": {
                    "daemon_version": "1.0.1",
                    "daemon_api_version": 1,
                    "status": "ok",
                    "uptime_ms": 10,
                    "model_name": "BAAI/bge-small-en-v1.5",
                    "semantic_home": "/tmp/semantic"
                }
            })
        })
        .await;

        let client = SemanticDaemonClient::new(
            IpcEndpoint::UnixSocket(socket_path),
            DaemonConnectPolicy::default(),
        );
        let result = client.health("obsidian-mcp-test", "1.0.1").await;
        match result {
            Err(VaultError::DaemonProtocol(message)) => {
                assert!(message.contains("mismatched id"));
            }
            other => panic!("expected daemon protocol error, got: {other:?}"),
        }

        server.await.expect("server task should complete");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn response_missing_jsonrpc_is_rejected() {
        let temp = tempfile::tempdir().expect("tempdir");
        let socket_path = temp.path().join("semanticd.sock");
        let server = start_unix_server_once(socket_path.clone(), |request| {
            let id = request
                .get("id")
                .cloned()
                .expect("request id should be present");
            json!({
                "id": id,
                "result": {
                    "daemon_version": "1.0.1",
                    "daemon_api_version": 1,
                    "status": "ok",
                    "uptime_ms": 10,
                    "model_name": "BAAI/bge-small-en-v1.5",
                    "semantic_home": "/tmp/semantic"
                }
            })
        })
        .await;

        let client = SemanticDaemonClient::new(
            IpcEndpoint::UnixSocket(socket_path),
            DaemonConnectPolicy::default(),
        );
        let result = client.health("obsidian-mcp-test", "1.0.1").await;
        match result {
            Err(VaultError::DaemonProtocol(message)) => {
                assert!(message.contains("missing jsonrpc version"));
            }
            other => panic!("expected daemon protocol error, got: {other:?}"),
        }

        server.await.expect("server task should complete");
    }
}
