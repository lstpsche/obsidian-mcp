//! MCP-side JSON-RPC client for the local semantic daemon.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::sync::Mutex;

#[cfg(unix)]
use tokio::net::UnixStream;
#[cfg(windows)]
use tokio::net::windows::named_pipe::ClientOptions;

use crate::config::SemanticRuntimeConfig;
use crate::daemon::protocol::{
    self, DAEMON_API_VERSION, EnsureVaultResult, HealthResult, SearchResult, ShutdownResult,
};
use crate::daemon::server::IpcEndpoint;
use crate::error::{VaultError, VaultResult};

trait AsyncReadWrite: AsyncRead + AsyncWrite + Send + Sync + Unpin {}
impl<T> AsyncReadWrite for T where T: AsyncRead + AsyncWrite + Send + Sync + Unpin {}
type DaemonStream = Box<dyn AsyncReadWrite>;

/// A daemon-side holder lives until this connection is dropped.
#[must_use = "retain the attachment while this scope should index or serve queries"]
pub struct VaultAttachment {
    pub result: EnsureVaultResult,
    root: String,
    _connection: DaemonStream,
}

impl std::fmt::Debug for VaultAttachment {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("VaultAttachment")
            .field("root", &self.root)
            .finish_non_exhaustive()
    }
}

fn wire_root(root: &Path) -> VaultResult<String> {
    root.to_str()
        .map(str::to_owned)
        .ok_or_else(|| VaultError::DaemonProtocol("the daemon requires a UTF-8 vault root".into()))
}

fn validate_semantics(health: &HealthResult) -> VaultResult<()> {
    if health.daemon_api_version != DAEMON_API_VERSION {
        return Err(VaultError::DaemonRpc {
            code: protocol::ERR_INCOMPATIBLE_API_VERSION,
            message: format!(
                "semantic daemon API version {} is incompatible with v{DAEMON_API_VERSION}",
                health.daemon_api_version
            ),
            data: None,
        });
    }
    if let Some(required) = protocol::missing_semantic_capability(&health.capabilities) {
        return Err(VaultError::DaemonRpc {
            code: protocol::ERR_UNSUPPORTED_CAPABILITY,
            message: format!(
                "semantic daemon requires capability '{required}'; upgrade or restart its owner with a compatible daemon"
            ),
            data: Some(json!({
                "required_capability": required,
                "reported_capabilities": health.capabilities,
                "daemon_version": health.daemon_version,
            })),
        });
    }
    Ok(())
}

fn health_params(client_name: &str, client_version: &str) -> protocol::HealthParams {
    protocol::HealthParams {
        client_name: Some(client_name.into()),
        client_version: Some(client_version.into()),
        min_api_version: Some(DAEMON_API_VERSION),
        max_api_version: Some(DAEMON_API_VERSION),
    }
}

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
    health: Mutex<Option<HealthResult>>,
}

impl SemanticDaemonClient {
    pub fn new(endpoint: IpcEndpoint, policy: DaemonConnectPolicy) -> Self {
        Self {
            inner: Arc::new(ClientInner {
                endpoint,
                policy,
                request_id: AtomicU64::new(1),
                health: Mutex::new(None),
            }),
        }
    }

    /// Reuse the authoritative handshake completed by owned-daemon bootstrap.
    pub fn with_health(
        endpoint: IpcEndpoint,
        policy: DaemonConnectPolicy,
        health: HealthResult,
    ) -> Self {
        Self {
            inner: Arc::new(ClientInner {
                endpoint,
                policy,
                request_id: AtomicU64::new(1),
                health: Mutex::new(Some(health)),
            }),
        }
    }

    /// Refresh a negative observation so a compatible replacement can recover.
    pub async fn require_semantic_capabilities(&self) -> VaultResult<()> {
        let mut observation = self.inner.health.lock().await;
        if observation
            .as_ref()
            .is_some_and(|health| validate_semantics(health).is_ok())
        {
            return Ok(());
        }
        let health = self
            .fetch_health("obsidian-mcp", env!("CARGO_PKG_VERSION"))
            .await?;
        let result = validate_semantics(&health);
        *observation = Some(health);
        result
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
        let mut observation = self.inner.health.lock().await;
        let health = self.fetch_health(client_name, client_version).await?;
        // Explicit reads replace the previous observation, including a daemon
        // that lacked a capability before its owner was upgraded.
        *observation = Some(health.clone());
        Ok(health)
    }

    async fn fetch_health(
        &self,
        client_name: &str,
        client_version: &str,
    ) -> VaultResult<HealthResult> {
        self.call("health", health_params(client_name, client_version))
            .await
    }

    pub async fn shutdown(&self) -> VaultResult<ShutdownResult> {
        self.call("shutdown", protocol::ShutdownParams::default())
            .await
    }

    /// Connect and dispatch within the RPC deadline, then await the initial scan.
    /// Dropping this future or its returned attachment withdraws the holder.
    pub async fn attach_vault(
        &self,
        vault_root: &Path,
        scope: &protocol::IndexScope,
        watch: bool,
        model_name: Option<&str>,
    ) -> VaultResult<VaultAttachment> {
        scope.validate().map_err(VaultError::InvalidPath)?;
        let root = wire_root(vault_root)?;
        let (result, stream): (EnsureVaultResult, _) = self
            .request(
                "ensure_vault",
                protocol::EnsureVaultParams {
                    vault_root: root.clone(),
                    index_scope: scope.clone(),
                    watch: Some(watch),
                    model_name: model_name.map(str::to_owned),
                },
            )
            .await?;
        if result.scope_id.is_empty() || result.attachment_id.is_empty() {
            return Err(VaultError::DaemonProtocol(
                "daemon returned an empty indexing scope or attachment handle".into(),
            ));
        }
        Ok(VaultAttachment {
            result,
            root,
            _connection: stream,
        })
    }

    /// Inspect a matching active scope without creating a holder.
    pub async fn vault_status(
        &self,
        vault_root: &Path,
        scope: &protocol::IndexScope,
    ) -> VaultResult<protocol::SemanticStatus> {
        scope.validate().map_err(VaultError::InvalidPath)?;
        self.call(
            "vault_status",
            protocol::VaultStatusParams {
                vault_root: wire_root(vault_root)?,
                attachment_id: None,
                index_scope: Some(scope.clone()),
            },
        )
        .await
    }

    pub async fn search_semantic(
        &self,
        attachment: &VaultAttachment,
        query: &str,
        top_k: usize,
        include_content: bool,
    ) -> VaultResult<SearchResult> {
        self.search(attachment, query, top_k, include_content, None, None)
            .await
    }

    pub async fn search_hybrid(
        &self,
        attachment: &VaultAttachment,
        query: &str,
        top_k: usize,
        prefetch: usize,
        alpha: f32,
        include_content: bool,
    ) -> VaultResult<SearchResult> {
        self.search(
            attachment,
            query,
            top_k,
            include_content,
            None,
            Some((prefetch, alpha)),
        )
        .await
    }

    /// Both direct calls and MCP queries route to the held view. MCP also
    /// supplies its current visible paths before either ranking strategy runs.
    pub(crate) async fn search(
        &self,
        attachment: &VaultAttachment,
        query: &str,
        top_k: usize,
        include_content: bool,
        allowed_paths: Option<&[String]>,
        hybrid: Option<(usize, f32)>,
    ) -> VaultResult<SearchResult> {
        let attachment_id = attachment.result.attachment_id.clone();
        let vault_root = attachment.root.clone();
        let query = query.to_owned();
        let top_k = Some(top_k);
        let include_content = Some(include_content);
        let allowed_paths = allowed_paths.map(<[String]>::to_vec);
        if let Some((prefetch, alpha)) = hybrid {
            self.call(
                "search_hybrid",
                protocol::SearchHybridParams {
                    attachment_id,
                    vault_root,
                    query,
                    top_k,
                    include_content,
                    allowed_paths,
                    prefetch: Some(prefetch),
                    alpha: Some(alpha),
                },
            )
            .await
        } else {
            self.call(
                "search_semantic",
                protocol::SearchSemanticParams {
                    attachment_id,
                    vault_root,
                    query,
                    top_k,
                    include_content,
                    allowed_paths,
                },
            )
            .await
        }
    }

    async fn call<P: Serialize, R: DeserializeOwned>(
        &self,
        method: &str,
        params: P,
    ) -> VaultResult<R> {
        self.request(method, params)
            .await
            .map(|(result, _connection)| result)
    }

    async fn request<P: Serialize, R: DeserializeOwned>(
        &self,
        method: &str,
        params: P,
    ) -> VaultResult<(R, DaemonStream)> {
        let mut retries = 0;
        loop {
            let result = self.call_once(method, &params).await;
            match result {
                Err(error) if retries < self.inner.policy.retries && is_retryable_error(&error) => {
                    retries += 1;
                    tracing::warn!(method, retries, error = %error, "retrying daemon request");
                    tokio::time::sleep(self.inner.policy.retry_backoff.saturating_mul(retries))
                        .await;
                }
                result => return result,
            }
        }
    }

    async fn call_once<R: DeserializeOwned>(
        &self,
        method: &str,
        params: &impl Serialize,
    ) -> VaultResult<(R, DaemonStream)> {
        let connect_and_send = async {
            let mut stream = self.connect().await?;
            if !matches!(method, "health" | "shutdown") {
                // Older v2 peers ignore unknown fields. Inspect the actual connection before
                // submitting a policy, query text or path to the process that accepted it.
                let probe = self
                    .send_request(
                        &mut stream,
                        "health",
                        &health_params("obsidian-mcp", env!("CARGO_PKG_VERSION")),
                    )
                    .await?;
                let health = self.read_response(&mut stream, "health", probe).await?;
                let accepted = validate_semantics(&health);
                *self.inner.health.lock().await = Some(health);
                accepted?;
            }
            let id = self.send_request(&mut stream, method, params).await?;
            Ok((stream, id))
        };
        if method == "ensure_vault" {
            // Once delivered, the live connection owns the cold scan. Its caller
            // controls the lifetime; an RPC deadline must not restart that scan.
            let (mut stream, id) = self.with_deadline(method, connect_and_send).await?;
            let result = self.read_response(&mut stream, method, id).await?;
            Ok((result, stream))
        } else {
            self.with_deadline(method, async {
                let (mut stream, id) = connect_and_send.await?;
                let result = self.read_response(&mut stream, method, id).await?;
                Ok((result, stream))
            })
            .await
        }
    }

    pub(crate) async fn with_deadline<T>(
        &self,
        method: &str,
        future: impl Future<Output = VaultResult<T>>,
    ) -> VaultResult<T> {
        tokio::time::timeout(self.inner.policy.timeout, future)
            .await
            .map_err(|_| VaultError::DaemonTimeout {
                operation: method.to_owned(),
                timeout_ms: self.inner.policy.timeout.as_millis().min(u64::MAX as u128) as u64,
            })?
    }

    async fn send_request(
        &self,
        stream: &mut DaemonStream,
        method: &str,
        params: &impl Serialize,
    ) -> VaultResult<u64> {
        #[derive(Serialize)]
        struct Request<'a, P> {
            jsonrpc: &'static str,
            id: u64,
            method: &'a str,
            params: &'a P,
        }

        let request_id = self.inner.request_id.fetch_add(1, Ordering::Relaxed);
        let mut encoded = serde_json::to_vec(&Request {
            jsonrpc: protocol::JSONRPC_VERSION,
            id: request_id,
            method,
            params,
        })
        .map_err(|err| {
            VaultError::DaemonProtocol(format!(
                "failed to serialize daemon request '{method}': {err}"
            ))
        })?;
        encoded.push(b'\n');
        stream
            .write_all(&encoded)
            .await
            .map_err(|error| self.ipc_error(method, error))?;
        stream
            .flush()
            .await
            .map_err(|error| self.ipc_error(method, error))?;
        Ok(request_id)
    }

    fn ipc_error(&self, method: &str, error: std::io::Error) -> VaultError {
        VaultError::DaemonIpc(format!(
            "daemon request '{method}' at '{}': {error}",
            self.inner.endpoint.endpoint_string()
        ))
    }

    async fn read_response<R: DeserializeOwned>(
        &self,
        stream: &mut DaemonStream,
        method: &str,
        request_id: u64,
    ) -> VaultResult<R> {
        let mut line = String::new();
        if BufReader::new(stream)
            .read_line(&mut line)
            .await
            .map_err(|error| self.ipc_error(method, error))?
            == 0
        {
            return Err(VaultError::DaemonIpc(format!(
                "daemon endpoint '{}' closed connection for '{method}'",
                self.inner.endpoint.endpoint_string()
            )));
        }
        parse_response(method, request_id, &line)
    }

    async fn connect(&self) -> VaultResult<DaemonStream> {
        let failure = |err: std::io::Error| {
            let message = format!(
                "failed to connect to daemon endpoint '{}': {err}",
                self.inner.endpoint.endpoint_string()
            );
            match err.kind() {
                std::io::ErrorKind::NotFound | std::io::ErrorKind::ConnectionRefused => {
                    VaultError::DaemonUnavailable(message)
                }
                _ => VaultError::DaemonIpc(message),
            }
        };
        match &self.inner.endpoint {
            #[cfg(unix)]
            IpcEndpoint::UnixSocket(path) => {
                Ok(Box::new(UnixStream::connect(path).await.map_err(failure)?))
            }
            #[cfg(windows)]
            IpcEndpoint::NamedPipe(name) => {
                Ok(Box::new(ClientOptions::new().open(name).map_err(failure)?))
            }
        }
    }
}

pub(crate) fn parse_response<R>(method: &str, request_id: u64, line: &str) -> VaultResult<R>
where
    R: DeserializeOwned,
{
    let mut response: Value = serde_json::from_str(line).map_err(|err| {
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

    let Some(result) = response.get_mut("result") else {
        return Err(VaultError::DaemonProtocol(format!(
            "daemon response for '{method}' missing result"
        )));
    };

    serde_json::from_value(result.take()).map_err(|err| {
        VaultError::DaemonProtocol(format!(
            "failed to decode daemon result for '{method}': {err}"
        ))
    })
}

fn is_retryable_error(err: &VaultError) -> bool {
    matches!(
        err,
        VaultError::DaemonIpc(_)
            | VaultError::DaemonUnavailable(_)
            | VaultError::DaemonTimeout { .. }
    )
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    use std::path::PathBuf;
    use tokio::net::UnixListener;

    fn capable_health() -> HealthResult {
        HealthResult {
            daemon_version: env!("CARGO_PKG_VERSION").into(),
            daemon_api_version: DAEMON_API_VERSION,
            pid: 1,
            status: "ok".into(),
            uptime_ms: 1,
            model_name: "test-model".into(),
            semantic_home: "/tmp/semantic".into(),
            capabilities: protocol::semantic_capabilities(),
        }
    }

    fn attachment_result() -> Value {
        json!({"scope_id": "scope","attachment_id":"holder","vault_id":"vault",
            "ready":true,"watch_enabled":true,"model_name":"model"})
    }

    async fn attach(client: &SemanticDaemonClient) -> VaultResult<VaultAttachment> {
        client
            .attach_vault(
                Path::new("/tmp/vault"),
                &protocol::IndexScope::default(),
                true,
                None,
            )
            .await
    }

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

    fn start_unix_server_responses(
        socket_path: &Path,
        responses: Vec<(&'static str, Value)>,
    ) -> tokio::task::JoinHandle<()> {
        let listener = UnixListener::bind(socket_path).unwrap();
        tokio::spawn(async move {
            for (method, result) in responses {
                let (stream, _) = listener.accept().await.unwrap();
                let (reader, mut writer) = stream.into_split();
                let mut reader = BufReader::new(reader);
                let request = if method == "health" {
                    crate::test_daemon_rpc::request(&mut reader).await
                } else {
                    crate::test_daemon_rpc::scoped_request(&mut reader, &mut writer).await
                };
                assert_eq!(request["method"], method);
                let response = json!({
                    "jsonrpc": "2.0", "id": request["id"], "result": result
                });
                writer
                    .write_all(format!("{response}\n").as_bytes())
                    .await
                    .unwrap();
            }
        })
    }

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
                    "daemon_api_version": DAEMON_API_VERSION,
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
        assert!(!health.supports_source_evidence());

        server.await.expect("server task should complete");
    }

    #[tokio::test]
    async fn cached_capabilities_cannot_authorize_attachment_to_a_replaced_owner() {
        let temp = tempfile::tempdir().unwrap();
        let socket = temp.path().join("semantic.sock");
        let mut health = capable_health();
        health.capabilities = vec![protocol::SEMANTIC_SOURCE_EVIDENCE_CAPABILITY.into()];
        let server = start_unix_server_responses(
            &socket,
            vec![("health", serde_json::to_value(&health).unwrap())],
        );
        let client = SemanticDaemonClient::with_health(
            IpcEndpoint::UnixSocket(socket),
            DaemonConnectPolicy::default(),
            capable_health(),
        );
        let error = attach(&client).await.unwrap_err();
        assert!(!is_retryable_error(&error));
        match error {
            VaultError::DaemonRpc { code, data, .. } => {
                assert_eq!(code, protocol::ERR_UNSUPPORTED_CAPABILITY);
                assert_eq!(
                    data.unwrap()["required_capability"],
                    protocol::SCOPED_INDEXING_CAPABILITY
                );
            }
            other => panic!("expected unsupported capability, got {other:?}"),
        }
        tokio::time::timeout(Duration::from_secs(5), server)
            .await
            .expect("capability server must receive the expected requests")
            .unwrap();
    }

    #[tokio::test]
    async fn semantic_client_recovers_after_daemon_gains_required_capability() {
        let temp = tempfile::tempdir().unwrap();
        let socket = temp.path().join("semantic.sock");
        let capable = capable_health();
        let mut unsupported = capable.clone();
        unsupported.capabilities.clear();
        let server = start_unix_server_responses(
            &socket,
            vec![
                ("health", serde_json::to_value(unsupported).unwrap()),
                ("ensure_vault", attachment_result()),
                ("search_semantic", json!({"results": []})),
            ],
        );
        let client = SemanticDaemonClient::new(
            IpcEndpoint::UnixSocket(socket),
            DaemonConnectPolicy::default(),
        );
        let error = attach(&client).await.unwrap_err();
        assert!(matches!(
            error,
            VaultError::DaemonRpc {
                code: protocol::ERR_UNSUPPORTED_CAPABILITY,
                ..
            }
        ));

        let attachment = attach(&client).await.unwrap();
        let result = client
            .clone()
            .search_semantic(&attachment, "query", 10, false)
            .await
            .unwrap();
        assert!(result.results.is_empty());
        tokio::time::timeout(Duration::from_secs(5), server)
            .await
            .expect("capability server must receive the expected requests")
            .unwrap();
        client.require_semantic_capabilities().await.unwrap();
    }

    #[tokio::test]
    async fn explicit_health_replaces_an_unsupported_observation() {
        let temp = tempfile::tempdir().unwrap();
        let socket = temp.path().join("semantic.sock");
        let capable = capable_health();
        let mut unsupported = capable.clone();
        unsupported.capabilities.clear();
        let server = start_unix_server_responses(
            &socket,
            vec![("health", serde_json::to_value(capable).unwrap())],
        );
        let client = SemanticDaemonClient::with_health(
            IpcEndpoint::UnixSocket(socket),
            DaemonConnectPolicy::default(),
            unsupported,
        );

        assert!(
            client
                .health("obsidian-mcp-test", "1.0.1")
                .await
                .unwrap()
                .supports_source_evidence()
        );
        tokio::time::timeout(Duration::from_secs(5), server)
            .await
            .expect("capability server must receive the expected requests")
            .unwrap();
        client.require_semantic_capabilities().await.unwrap();
    }

    #[tokio::test]
    async fn concurrent_semantic_capability_checks_share_one_handshake() {
        let temp = tempfile::tempdir().unwrap();
        let socket = temp.path().join("semantic.sock");
        let server = start_unix_server_once(socket.clone(), |request| {
            assert_eq!(request["method"], "health");
            json!({
                "jsonrpc": "2.0",
                "id": request["id"],
                "result": capable_health()
            })
        })
        .await;
        let client = SemanticDaemonClient::new(
            IpcEndpoint::UnixSocket(socket),
            DaemonConnectPolicy::default(),
        );
        let cloned = client.clone();
        let (left, right) = tokio::join!(
            client.require_semantic_capabilities(),
            cloned.require_semantic_capabilities()
        );
        left.unwrap();
        right.unwrap();
        server.await.unwrap();
        // The server is gone; the established contract remains shared by clones.
        client.require_semantic_capabilities().await.unwrap();
    }

    #[tokio::test]
    async fn capable_daemon_response_still_requires_source_evidence_fields() {
        let temp = tempfile::tempdir().unwrap();
        let socket = temp.path().join("semantic.sock");
        let server = start_unix_server_responses(
            &socket,
            vec![
                ("ensure_vault", attachment_result()),
                (
                    "search_semantic",
                    json!({"results": [{"path":"note.md","score":0.8,"title":"note","tags":[]}]}),
                ),
            ],
        );
        let client = SemanticDaemonClient::new(
            IpcEndpoint::UnixSocket(socket),
            DaemonConnectPolicy::default(),
        );
        let attachment = attach(&client).await.unwrap();
        let error = client
            .search_semantic(&attachment, "query", 10, false)
            .await
            .unwrap_err();
        assert!(matches!(error, VaultError::DaemonProtocol(_)));
        server.await.unwrap();
    }

    #[tokio::test]
    async fn attachment_requires_both_handles_and_preserves_progress() {
        let temp = tempfile::tempdir().unwrap();
        let socket = temp.path().join("semantic.sock");
        let legacy =
            json!({"vault_id": "vault","watch_enabled":true,"model_name":"model","ready":true});
        let mut current = attachment_result();
        current["phase"] = json!("warming");
        current["ready"] = json!(false);
        current["indexed_notes"] = json!(3);
        current["total_notes"] = json!(9);
        current["pending_notes"] = json!(6);
        let server = start_unix_server_responses(
            &socket,
            vec![("ensure_vault", legacy), ("ensure_vault", current)],
        );
        let client = SemanticDaemonClient::new(
            IpcEndpoint::UnixSocket(socket),
            DaemonConnectPolicy::default(),
        );
        assert!(matches!(
            attach(&client).await.unwrap_err(),
            VaultError::DaemonProtocol(_)
        ));
        let holder = attach(&client).await.unwrap();
        let result = &holder.result;
        assert!(!result.ready);
        assert_eq!(result.phase, Some(protocol::SemanticPhase::Warming));
        assert_eq!(result.indexed_notes, Some(3));
        assert_eq!(result.total_notes, Some(9));
        assert_eq!(result.pending_notes, Some(6));
        assert_eq!(result.last_error, None);
        server.await.unwrap();
    }

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
                        "daemon_api_version": DAEMON_API_VERSION,
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

    #[tokio::test]
    async fn rpc_errors_preserve_semantic_status_data() {
        let temp = tempfile::tempdir().expect("tempdir");
        let socket_path = temp.path().join("semanticd.sock");
        let listener = UnixListener::bind(&socket_path).unwrap();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let (reader, mut writer) = stream.into_split();
            let request =
                crate::test_daemon_rpc::scoped_request(&mut BufReader::new(reader), &mut writer)
                    .await;
            let response = json!({
                "jsonrpc":"2.0", "id":request["id"],
                "error":{"code":-32030,"message":"semantic index is warming; retry the query shortly",
                    "data":{"phase":"warming","ready":false,"indexed_notes":2,"total_notes":5,"pending_notes":3}}
            });
            writer
                .write_all(format!("{response}\n").as_bytes())
                .await
                .unwrap();
        });
        let client = SemanticDaemonClient::new(
            IpcEndpoint::UnixSocket(socket_path),
            DaemonConnectPolicy::default(),
        );
        let result = attach(&client).await;
        match result {
            Err(VaultError::DaemonRpc {
                code,
                message,
                data,
            }) => {
                assert_eq!(code, -32030);
                assert!(message.contains("warming"));
                let data = data.expect("warming status data should be preserved");
                assert_eq!(data["phase"], "warming");
                assert_eq!(data["ready"], false);
                assert_eq!(data["indexed_notes"], 2);
                assert_eq!(data["total_notes"], 5);
                assert_eq!(data["pending_notes"], 3);
            }
            other => panic!("expected daemon RPC error, got: {other:?}"),
        }

        server.await.expect("server task should complete");
    }

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
                    "daemon_api_version": DAEMON_API_VERSION,
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
                    "daemon_api_version": DAEMON_API_VERSION,
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
