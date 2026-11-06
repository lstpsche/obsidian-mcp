//! IPC server skeleton for `obsidian-semanticd`.

use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::json;
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::sync::watch;

#[cfg(unix)]
mod unix_endpoint;
#[cfg(windows)]
use tokio::net::windows::named_pipe::{NamedPipeServer, ServerOptions};

use crate::error::{VaultError, VaultResult};

use super::protocol::{
    self, DAEMON_API_VERSION, ERR_INCOMPATIBLE_API_VERSION, ERR_INVALID_PARAMS,
    ERR_INVALID_REQUEST, ERR_METHOD_NOT_FOUND, ERR_PARSE, EnsureVaultParams, HealthParams,
    HealthResult, JSONRPC_VERSION, OpenHintParams, RpcRequest, RpcResponse, SearchHybridParams,
    SearchSemanticParams, ShutdownParams, ShutdownResult,
};
use super::query;
use super::vault_registry::{VaultAttachment, VaultRegistry};

type ShutdownSignal = Pin<Box<dyn Future<Output = ()> + Send>>;

#[derive(Debug, Clone)]
pub struct DaemonServerConfig {
    pub endpoint: IpcEndpoint,
    pub model_name: String,
    pub semantic_home: PathBuf,
}

#[derive(Debug, Clone)]
pub enum IpcEndpoint {
    #[cfg(unix)]
    UnixSocket(PathBuf),
    #[cfg(windows)]
    NamedPipe(String),
}

impl IpcEndpoint {
    pub fn endpoint_string(&self) -> String {
        match self {
            #[cfg(unix)]
            Self::UnixSocket(path) => path.display().to_string(),
            #[cfg(windows)]
            Self::NamedPipe(name) => name.clone(),
        }
    }
}

struct ServerState {
    daemon_version: String,
    model_name: String,
    semantic_home: PathBuf,
    started_at: Instant,
    registry: Arc<VaultRegistry>,
    shutdown_tx: watch::Sender<bool>,
}

impl ServerState {
    fn new(
        model_name: String,
        semantic_home: PathBuf,
        registry: Arc<VaultRegistry>,
        shutdown_tx: watch::Sender<bool>,
    ) -> Self {
        Self {
            daemon_version: env!("CARGO_PKG_VERSION").to_string(),
            model_name,
            semantic_home,
            started_at: Instant::now(),
            registry,
            shutdown_tx,
        }
    }
}

/// Run daemon IPC server until Ctrl-C.
pub async fn run(config: DaemonServerConfig) -> VaultResult<()> {
    run_with_shutdown(config, async {
        if let Err(err) = tokio::signal::ctrl_c().await {
            tracing::warn!(error = %err, "failed to install Ctrl-C handler");
        }
    })
    .await
}

/// Run daemon IPC server until the provided shutdown signal resolves.
pub async fn run_with_shutdown<F>(config: DaemonServerConfig, shutdown: F) -> VaultResult<()>
where
    F: Future<Output = ()> + Send + 'static,
{
    let registry = Arc::new(VaultRegistry::new(
        config.semantic_home.clone(),
        config.model_name.clone(),
    )?);
    let (shutdown_tx, mut shutdown_rx) = watch::channel(false);
    let state = Arc::new(ServerState::new(
        config.model_name,
        config.semantic_home,
        registry,
        shutdown_tx,
    ));
    let shutdown: ShutdownSignal = Box::pin(async move {
        tokio::select! {
            _ = shutdown => {}
            _ = async move {
                loop {
                    if *shutdown_rx.borrow() {
                        break;
                    }
                    if shutdown_rx.changed().await.is_err() {
                        break;
                    }
                }
            } => {}
        }
    });

    let result = match config.endpoint {
        #[cfg(unix)]
        IpcEndpoint::UnixSocket(path) => run_unix_socket(path, Arc::clone(&state), shutdown).await,
        #[cfg(windows)]
        IpcEndpoint::NamedPipe(name) => run_named_pipe(name, Arc::clone(&state), shutdown).await,
    };
    state.registry.close();
    state.shutdown_tx.send_replace(true);
    result
}

#[cfg(unix)]
async fn run_unix_socket(
    path: PathBuf,
    state: Arc<ServerState>,
    mut shutdown: ShutdownSignal,
) -> VaultResult<()> {
    let (owner, listener) = unix_endpoint::UnixEndpoint::bind(&path)?;
    tracing::info!(endpoint = %path.display(), "semantic daemon IPC listening");

    loop {
        tokio::select! {
            _ = &mut shutdown => {
                tracing::info!("semantic daemon shutdown requested");
                break;
            }
            accepted = listener.accept() => {
                match accepted {
                    Ok((stream, _)) => {
                        let state = Arc::clone(&state);
                        tokio::spawn(async move {
                            if let Err(err) = handle_connection(stream, state).await {
                                tracing::warn!(error = %err, "daemon connection handler failed");
                            }
                        });
                    }
                    Err(err) => {
                        tracing::warn!(error = %err, "failed to accept daemon IPC connection");
                    }
                }
            }
        }
    }

    drop(listener);
    owner.close()?;
    Ok(())
}

#[cfg(windows)]
async fn run_named_pipe(
    name: String,
    state: Arc<ServerState>,
    mut shutdown: ShutdownSignal,
) -> VaultResult<()> {
    let mut current = create_pipe_server(&name, true)?;
    tracing::info!(endpoint = %name, "semantic daemon IPC listening");

    loop {
        let next = create_pipe_server(&name, false)?;
        tokio::select! {
            _ = &mut shutdown => {
                tracing::info!("semantic daemon shutdown requested");
                break;
            }
            connected = current.connect() => {
                if let Err(err) = connected {
                    return Err(VaultError::DaemonIpc(format!("failed to accept named pipe client: {err}")));
                }

                let stream = current;
                current = next;
                let state = Arc::clone(&state);
                tokio::spawn(async move {
                    if let Err(err) = handle_connection(stream, state).await {
                        tracing::warn!(error = %err, "daemon named pipe handler failed");
                    }
                });
            }
        }
    }

    Ok(())
}

#[cfg(windows)]
fn create_pipe_server(name: &str, first: bool) -> VaultResult<NamedPipeServer> {
    ServerOptions::new()
        .first_pipe_instance(first)
        .create(name)
        .map_err(|err| VaultError::DaemonIpc(format!("failed to create named pipe server: {err}")))
}

async fn handle_connection<S>(stream: S, state: Arc<ServerState>) -> VaultResult<()>
where
    S: AsyncRead + AsyncWrite + Send + Unpin + 'static,
{
    let mut shutdown = state.shutdown_tx.subscribe();
    tokio::select! {
        _ = shutdown.wait_for(|stopped| *stopped) => Ok(()),
        result = handle_requests(stream, state) => result,
    }
}

#[cfg(all(test, has_embeddings))]
pub(super) async fn serve_test_connection(
    stream: tokio::io::DuplexStream,
    semantic_home: PathBuf,
    registry: Arc<VaultRegistry>,
) -> VaultResult<()> {
    let (shutdown_tx, _) = watch::channel(false);
    let state = ServerState::new(
        registry.model_name().into(),
        semantic_home,
        registry,
        shutdown_tx,
    );
    handle_connection(stream, Arc::new(state)).await
}

async fn handle_requests<S>(stream: S, state: Arc<ServerState>) -> VaultResult<()>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    let (reader, mut writer) = tokio::io::split(stream);
    let mut lines = BufReader::new(reader).lines();
    let mut attachment = None;
    let mut pending = std::collections::VecDeque::new();

    loop {
        let line = match pending.pop_front() {
            Some(line) => line,
            None => match lines.next_line().await? {
                Some(line) => line,
                None => return Ok(()),
            },
        };
        if line.trim().is_empty() {
            continue;
        }
        let request: RpcRequest = match serde_json::from_str(&line) {
            Ok(request) => request,
            Err(error) => {
                write_response(
                    &mut writer,
                    &RpcResponse::error(None, ERR_PARSE, format!("parse error: {error}")),
                )
                .await?;
                continue;
            }
        };
        let owns_scope = attachment.is_some() || request.method == "ensure_vault";
        let dispatch = route_request_with_control(request, Arc::clone(&state), &mut attachment);
        let outcome = if owns_scope {
            // Keep observing the holder while its index initializes or another request runs.
            // Dropping dispatch releases its provisional grant before inference can activate.
            tokio::pin!(dispatch);
            loop {
                tokio::select! {
                    biased;
                    incoming = lines.next_line() => match incoming? {
                        None => return Ok(()),
                        Some(line) if pending.len() < 16 => pending.push_back(line),
                        Some(_) => return Err(VaultError::DaemonProtocol("too many pipelined attachment requests".into())),
                    },
                    outcome = &mut dispatch => break outcome,
                }
            }
        } else {
            dispatch.await
        };

        write_response(&mut writer, &outcome.response).await?;
        if outcome.shutdown_after_response {
            state.shutdown_tx.send_replace(true);
            return Ok(());
        }
    }
}

async fn write_response<W>(writer: &mut W, response: &RpcResponse) -> VaultResult<()>
where
    W: AsyncWrite + Unpin,
{
    let encoded = serde_json::to_string(response).map_err(|err| {
        VaultError::DaemonProtocol(format!("response serialization failed: {err}"))
    })?;
    writer.write_all(encoded.as_bytes()).await?;
    writer.write_all(b"\n").await?;
    writer.flush().await?;
    Ok(())
}

#[cfg(test)]
async fn route_request(request: RpcRequest, state: Arc<ServerState>) -> RpcResponse {
    route_request_with_control(request, state, &mut None)
        .await
        .response
}

struct RouteOutcome {
    response: RpcResponse,
    shutdown_after_response: bool,
}

impl RouteOutcome {
    fn response(response: RpcResponse) -> Self {
        Self {
            response,
            shutdown_after_response: false,
        }
    }
}

async fn route_request_with_control(
    request: RpcRequest,
    state: Arc<ServerState>,
    attachment: &mut Option<VaultAttachment>,
) -> RouteOutcome {
    if request.jsonrpc != JSONRPC_VERSION {
        return RouteOutcome::response(RpcResponse::error(
            request.id,
            ERR_INVALID_REQUEST,
            "jsonrpc must be '2.0'",
        ));
    }

    let response = match request.method.as_str() {
        "health" => {
            let params: HealthParams = match parse_params(request.params, request.id.clone()) {
                Ok(params) => params,
                Err(err) => return RouteOutcome::response(*err),
            };
            route_health(request.id, params, &state)
        }
        "vault_status" => {
            let params: protocol::VaultStatusParams =
                match parse_params(request.params, request.id.clone()) {
                    Ok(params) => params,
                    Err(err) => return RouteOutcome::response(*err),
                };
            match query::vault_status(&state.registry, params).await {
                Ok(result) => response_from_result(request.id, &result),
                Err(err) => response_from_query_error(request.id, err),
            }
        }
        "ensure_vault" => {
            let params: EnsureVaultParams = match parse_params(request.params, request.id.clone()) {
                Ok(params) => params,
                Err(err) => return RouteOutcome::response(*err),
            };
            match query::ensure_vault(&state.registry, params, attachment).await {
                Ok(result) => response_from_result(request.id, &result),
                Err(err) => response_from_query_error(request.id, err),
            }
        }
        "search_semantic" => {
            let params: SearchSemanticParams =
                match parse_params(request.params, request.id.clone()) {
                    Ok(params) => params,
                    Err(err) => return RouteOutcome::response(*err),
                };
            match query::search_semantic(&state.registry, params).await {
                Ok(result) => response_from_result(request.id, &result),
                Err(err) => response_from_query_error(request.id, err),
            }
        }
        "search_hybrid" => {
            let params: SearchHybridParams = match parse_params(request.params, request.id.clone())
            {
                Ok(params) => params,
                Err(err) => return RouteOutcome::response(*err),
            };
            match query::search_hybrid(&state.registry, params).await {
                Ok(result) => response_from_result(request.id, &result),
                Err(err) => response_from_query_error(request.id, err),
            }
        }
        "open_hint" => {
            let params: OpenHintParams = match parse_params(request.params, request.id.clone()) {
                Ok(params) => params,
                Err(err) => return RouteOutcome::response(*err),
            };
            match query::open_hint(&state.registry, params).await {
                Ok(result) => response_from_result(request.id, &result),
                Err(err) => response_from_query_error(request.id, err),
            }
        }
        "shutdown" => {
            let _: ShutdownParams = match parse_params(request.params, request.id.clone()) {
                Ok(params) => params,
                Err(err) => return RouteOutcome::response(*err),
            };
            return RouteOutcome {
                response: response_from_result(request.id, &ShutdownResult { accepted: true }),
                shutdown_after_response: true,
            };
        }
        _ => RpcResponse::error(
            request.id,
            ERR_METHOD_NOT_FOUND,
            format!("unknown method '{}'", request.method),
        ),
    };

    RouteOutcome::response(response)
}

fn parse_params<T: DeserializeOwned>(
    value: serde_json::Value,
    id: Option<serde_json::Value>,
) -> Result<T, Box<RpcResponse>> {
    serde_json::from_value(value).map_err(|err| {
        Box::new(RpcResponse::error_with_data(
            id,
            ERR_INVALID_PARAMS,
            "invalid params",
            json!({ "details": err.to_string() }),
        ))
    })
}

fn route_health(
    id: Option<serde_json::Value>,
    params: HealthParams,
    state: &ServerState,
) -> RpcResponse {
    let min = params.min_api_version.unwrap_or(DAEMON_API_VERSION);
    let max = params.max_api_version.unwrap_or(DAEMON_API_VERSION);
    if min > max {
        return RpcResponse::error(
            id,
            ERR_INVALID_PARAMS,
            "min_api_version must be <= max_api_version",
        );
    }
    if DAEMON_API_VERSION < min || DAEMON_API_VERSION > max {
        return RpcResponse::error_with_data(
            id,
            ERR_INCOMPATIBLE_API_VERSION,
            "incompatible daemon API version",
            json!({
                "daemon_api_version": DAEMON_API_VERSION,
                "client_min_api_version": min,
                "client_max_api_version": max
            }),
        );
    }

    let uptime_ms = state.started_at.elapsed().as_millis().min(u64::MAX as u128) as u64;
    let result = HealthResult {
        daemon_version: state.daemon_version.clone(),
        daemon_api_version: DAEMON_API_VERSION,
        pid: std::process::id(),
        status: "ok".to_string(),
        uptime_ms,
        model_name: state.model_name.clone(),
        semantic_home: state.semantic_home.display().to_string(),
        capabilities: protocol::semantic_capabilities(),
    };

    match serde_json::to_value(result) {
        Ok(value) => RpcResponse::success(id, value),
        Err(err) => RpcResponse::error(
            id,
            protocol::ERR_INTERNAL,
            format!("failed to serialize health result: {err}"),
        ),
    }
}

fn response_from_result<T: Serialize>(id: Option<serde_json::Value>, result: &T) -> RpcResponse {
    match serde_json::to_value(result) {
        Ok(value) => RpcResponse::success(id, value),
        Err(err) => RpcResponse::error(
            id,
            protocol::ERR_INTERNAL,
            format!("failed to serialize daemon result: {err}"),
        ),
    }
}

fn response_from_query_error(id: Option<serde_json::Value>, err: query::QueryError) -> RpcResponse {
    match err.data {
        Some(data) => RpcResponse::error_with_data(id, err.code, err.message, data),
        None => RpcResponse::error(id, err.code, err.message),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn make_state(semantic_home: &Path) -> Arc<ServerState> {
        let model_name = "BAAI/bge-small-en-v1.5".to_string();
        let registry = Arc::new(
            VaultRegistry::new(semantic_home.to_path_buf(), model_name.clone())
                .expect("registry should be created"),
        );
        let (shutdown_tx, _) = watch::channel(false);
        Arc::new(ServerState::new(
            model_name,
            semantic_home.to_path_buf(),
            registry,
            shutdown_tx,
        ))
    }

    #[tokio::test]
    async fn route_health_rejects_incompatible_api() {
        let tmp = tempfile::tempdir().expect("tempdir should be created");
        let state = make_state(tmp.path());
        let response = route_request(
            RpcRequest {
                jsonrpc: JSONRPC_VERSION.to_string(),
                id: Some(json!(1)),
                method: "health".to_string(),
                params: json!({
                    "min_api_version": DAEMON_API_VERSION + 1,
                    "max_api_version": DAEMON_API_VERSION + 2
                }),
            },
            state,
        )
        .await;

        assert_eq!(
            response.error.expect("error expected").code,
            ERR_INCOMPATIBLE_API_VERSION
        );
    }

    #[tokio::test]
    async fn old_exact_v2_health_request_accepts_source_evidence_capability() {
        let tmp = tempfile::tempdir().expect("tempdir should be created");
        let state = make_state(tmp.path());
        let response = route_request(
            RpcRequest {
                jsonrpc: JSONRPC_VERSION.to_string(),
                id: Some(json!(1)),
                method: "health".to_string(),
                params: json!({"min_api_version": 2, "max_api_version": 2}),
            },
            state,
        )
        .await;

        assert!(response.error.is_none());
        let result = response.result.unwrap();
        assert_eq!(result["daemon_api_version"], 2);
        assert_eq!(
            result["capabilities"],
            json!(protocol::SEMANTIC_CAPABILITIES)
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn unix_socket_handles_parse_errors_and_health() {
        let dir = tempfile::tempdir().expect("tempdir should be created");
        let socket_path = dir.path().join("semanticd.sock");
        let endpoint = IpcEndpoint::UnixSocket(socket_path.clone());
        let config = DaemonServerConfig {
            endpoint,
            model_name: "BAAI/bge-small-en-v1.5".to_string(),
            semantic_home: dir.path().to_path_buf(),
        };

        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
        let server = tokio::spawn(async move {
            run_with_shutdown(config, async move {
                let _ = shutdown_rx.await;
            })
            .await
            .expect("server should run cleanly");
        });

        let mut connected = false;
        for _ in 0..50 {
            if tokio::net::UnixStream::connect(&socket_path).await.is_ok() {
                connected = true;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        assert!(connected, "daemon socket did not become ready in time");

        let mut stream = tokio::net::UnixStream::connect(&socket_path)
            .await
            .expect("client should connect");

        stream
            .write_all(b"not-json\n")
            .await
            .expect("write malformed request");

        let mut reader = BufReader::new(stream);
        let mut line = String::new();
        reader
            .read_line(&mut line)
            .await
            .expect("read parse-error response");
        let parse_response: serde_json::Value =
            serde_json::from_str(&line).expect("parse-error response should be valid JSON");
        assert_eq!(parse_response["error"]["code"], json!(ERR_PARSE));

        let mut stream = reader.into_inner();
        stream
            .write_all(
                br#"{"jsonrpc":"2.0","id":7,"method":"health","params":{"min_api_version":2,"max_api_version":2}}"#
            )
            .await
            .expect("write health request");
        stream.write_all(b"\n").await.expect("write newline");

        let mut reader = BufReader::new(stream);
        line.clear();
        reader
            .read_line(&mut line)
            .await
            .expect("read health response");
        let health_response: serde_json::Value =
            serde_json::from_str(&line).expect("health response should be valid JSON");
        assert_eq!(
            health_response["result"]["daemon_api_version"],
            json!(DAEMON_API_VERSION)
        );
        assert_eq!(health_response["result"]["pid"], json!(std::process::id()));
        assert_eq!(
            health_response["result"]["capabilities"],
            json!(protocol::SEMANTIC_CAPABILITIES)
        );

        shutdown_tx
            .send(())
            .expect("shutdown signal should be delivered");
        server.await.expect("server task should join");
    }

    #[tokio::test]
    async fn route_vault_status_does_not_attach_an_unseen_vault() {
        let tmp = tempfile::tempdir().expect("tempdir should be created");
        let vault_root = tmp.path().join("vault");
        std::fs::create_dir_all(&vault_root).unwrap();
        let state = make_state(&tmp.path().join("semantic-home"));
        let response = route_request(
            RpcRequest {
                jsonrpc: JSONRPC_VERSION.to_string(),
                id: Some(json!(1)),
                method: "vault_status".to_string(),
                params: json!({"vault_root": vault_root, "index_scope": protocol::IndexScope::default()}),
            },
            Arc::clone(&state),
        )
        .await;
        assert_eq!(
            response.error.expect("unattached error expected").code,
            protocol::ERR_VAULT_NOT_ATTACHED
        );
        let invalid = route_request(
            RpcRequest {
                jsonrpc: JSONRPC_VERSION.to_string(),
                id: Some(json!(2)),
                method: "vault_status".to_string(),
                params: json!({"vault_root": vault_root, "index_scope": {"version": 1, "exclude_patterns": ["[unclosed"]}}),
            },
            Arc::clone(&state),
        ).await;
        assert_eq!(
            invalid.error.expect("invalid exclusion").code,
            protocol::ERR_INVALID_PARAMS
        );
        assert!(
            state
                .registry
                .context_for_policy(&vault_root, protocol::IndexScope::default())
                .await
                .unwrap()
                .is_none()
        );
        assert!(
            tmp.path()
                .join("semantic-home/vaults")
                .read_dir()
                .unwrap()
                .next()
                .is_none()
        );
    }

    #[tokio::test]
    async fn route_ensure_vault_rejects_relative_path() {
        let tmp = tempfile::tempdir().expect("tempdir should be created");
        let state = make_state(tmp.path());
        let response = route_request(
            RpcRequest {
                jsonrpc: JSONRPC_VERSION.to_string(),
                id: Some(json!(1)),
                method: "ensure_vault".to_string(),
                params: json!({
                    "vault_root": "relative/path",
                    "index_scope": protocol::IndexScope::default(),
                    "watch": true
                }),
            },
            state,
        )
        .await;
        assert_eq!(
            response.error.expect("error expected").code,
            ERR_INVALID_PARAMS
        );
    }

    #[tokio::test]
    async fn route_unrecognized_method_returns_method_not_found() {
        let tmp = tempfile::tempdir().expect("tempdir should be created");
        let state = make_state(tmp.path());
        let response = route_request(
            RpcRequest {
                jsonrpc: JSONRPC_VERSION.to_string(),
                id: Some(json!(1)),
                method: "does_not_exist".to_string(),
                params: json!({}),
            },
            state,
        )
        .await;
        assert_eq!(
            response.error.expect("error expected").code,
            ERR_METHOD_NOT_FOUND
        );
    }

    #[tokio::test]
    async fn route_shutdown_requests_exit_only_for_valid_params() {
        let tmp = tempfile::tempdir().expect("tempdir should be created");
        let state = make_state(tmp.path());
        let outcome = route_request_with_control(
            RpcRequest {
                jsonrpc: JSONRPC_VERSION.to_string(),
                id: Some(json!(1)),
                method: "shutdown".to_string(),
                params: json!({}),
            },
            state,
            &mut None,
        )
        .await;

        assert!(outcome.shutdown_after_response);
        assert_eq!(outcome.response.result, Some(json!({ "accepted": true })));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn shutdown_response_is_flushed_before_server_exits() {
        let dir = tempfile::tempdir().expect("tempdir should be created");
        let socket_path = dir.path().join("semanticd-shutdown.sock");
        let config = DaemonServerConfig {
            endpoint: IpcEndpoint::UnixSocket(socket_path.clone()),
            model_name: "BAAI/bge-small-en-v1.5".to_string(),
            semantic_home: dir.path().to_path_buf(),
        };
        let server = tokio::spawn(async move {
            run_with_shutdown(config, std::future::pending())
                .await
                .expect("server should stop cleanly");
        });

        for _ in 0..50 {
            if socket_path.exists() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        let mut stream = tokio::net::UnixStream::connect(&socket_path)
            .await
            .expect("client should connect");
        stream
            .write_all(b"{\"jsonrpc\":\"2.0\",\"id\":9,\"method\":\"shutdown\",\"params\":{}}\n")
            .await
            .expect("shutdown request should write");

        let mut response = String::new();
        BufReader::new(stream)
            .read_line(&mut response)
            .await
            .expect("shutdown response should read");
        let response: serde_json::Value =
            serde_json::from_str(&response).expect("shutdown response should be JSON");
        assert_eq!(response["result"]["accepted"], json!(true));
        tokio::time::timeout(std::time::Duration::from_secs(2), server)
            .await
            .expect("server should exit after responding")
            .expect("server task should join");
        assert!(!socket_path.exists());
    }
}
