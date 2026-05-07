//! IPC server skeleton for `obsidian-semanticd`.

use std::future::Future;
#[cfg(unix)]
use std::io::ErrorKind;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::json;
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader};

#[cfg(unix)]
use tokio::net::UnixListener;
#[cfg(windows)]
use tokio::net::windows::named_pipe::{NamedPipeServer, ServerOptions};

use crate::error::{VaultError, VaultResult};

use super::protocol::{
    self, DAEMON_API_VERSION, ERR_INCOMPATIBLE_API_VERSION, ERR_INVALID_PARAMS,
    ERR_INVALID_REQUEST, ERR_METHOD_NOT_FOUND, ERR_PARSE, EnsureVaultParams, HealthParams,
    HealthResult, JSONRPC_VERSION, OpenHintParams, RpcRequest, RpcResponse, SearchHybridParams,
    SearchSemanticParams,
};
use super::query;
use super::vault_registry::VaultRegistry;

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
}

impl ServerState {
    fn new(model_name: String, semantic_home: PathBuf, registry: Arc<VaultRegistry>) -> Self {
        Self {
            daemon_version: env!("CARGO_PKG_VERSION").to_string(),
            model_name,
            semantic_home,
            started_at: Instant::now(),
            registry,
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
    let state = Arc::new(ServerState::new(
        config.model_name,
        config.semantic_home,
        registry,
    ));
    let shutdown: ShutdownSignal = Box::pin(shutdown);

    match config.endpoint {
        #[cfg(unix)]
        IpcEndpoint::UnixSocket(path) => run_unix_socket(path, state, shutdown).await,
        #[cfg(windows)]
        IpcEndpoint::NamedPipe(name) => run_named_pipe(name, state, shutdown).await,
    }
}

#[cfg(unix)]
async fn run_unix_socket(
    path: PathBuf,
    state: Arc<ServerState>,
    mut shutdown: ShutdownSignal,
) -> VaultResult<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if path.exists() {
        match std::fs::remove_file(&path) {
            Ok(()) => {}
            Err(err) if err.kind() == ErrorKind::NotFound => {}
            Err(err) => return Err(VaultError::Io(err)),
        }
    }

    let listener = UnixListener::bind(&path).map_err(VaultError::Io)?;
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = std::fs::Permissions::from_mode(0o600);
        if let Err(err) = std::fs::set_permissions(&path, perms) {
            tracing::warn!(error = %err, "failed to restrict socket permissions to 0600");
        }
    }
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

    match std::fs::remove_file(&path) {
        Ok(()) => {}
        Err(err) if err.kind() == ErrorKind::NotFound => {}
        Err(err) => tracing::warn!(
            error = %err,
            endpoint = %path.display(),
            "failed to remove daemon socket during shutdown"
        ),
    }

    Ok(())
}

#[cfg(windows)]
async fn run_named_pipe(
    name: String,
    state: Arc<ServerState>,
    mut shutdown: ShutdownSignal,
) -> VaultResult<()> {
    let mut current = create_pipe_server(&name)?;
    tracing::info!(endpoint = %name, "semantic daemon IPC listening");

    loop {
        let next = create_pipe_server(&name)?;
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
fn create_pipe_server(name: &str) -> VaultResult<NamedPipeServer> {
    ServerOptions::new()
        .create(name)
        .map_err(|err| VaultError::DaemonIpc(format!("failed to create named pipe server: {err}")))
}

async fn handle_connection<S>(stream: S, state: Arc<ServerState>) -> VaultResult<()>
where
    S: AsyncRead + AsyncWrite + Send + Unpin + 'static,
{
    let (reader, mut writer) = tokio::io::split(stream);
    let mut lines = BufReader::new(reader).lines();

    while let Some(line) = lines.next_line().await? {
        if line.trim().is_empty() {
            continue;
        }

        let response = match serde_json::from_str::<RpcRequest>(&line) {
            Ok(request) => route_request(request, Arc::clone(&state)).await,
            Err(err) => RpcResponse::error(None, ERR_PARSE, format!("parse error: {err}")),
        };

        write_response(&mut writer, &response).await?;
    }

    Ok(())
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

async fn route_request(request: RpcRequest, state: Arc<ServerState>) -> RpcResponse {
    if request.jsonrpc != JSONRPC_VERSION {
        return RpcResponse::error(request.id, ERR_INVALID_REQUEST, "jsonrpc must be '2.0'");
    }

    match request.method.as_str() {
        "health" => {
            let params: HealthParams = match parse_params(request.params, request.id.clone()) {
                Ok(params) => params,
                Err(err) => return *err,
            };
            route_health(request.id, params, &state)
        }
        "ensure_vault" => {
            let params: EnsureVaultParams = match parse_params(request.params, request.id.clone()) {
                Ok(params) => params,
                Err(err) => return *err,
            };
            match query::ensure_vault(&state.registry, params).await {
                Ok(result) => response_from_result(request.id, &result),
                Err(err) => response_from_query_error(request.id, err),
            }
        }
        "search_semantic" => {
            let params: SearchSemanticParams =
                match parse_params(request.params, request.id.clone()) {
                    Ok(params) => params,
                    Err(err) => return *err,
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
                Err(err) => return *err,
            };
            match query::search_hybrid(&state.registry, params).await {
                Ok(result) => response_from_result(request.id, &result),
                Err(err) => response_from_query_error(request.id, err),
            }
        }
        "open_hint" => {
            let params: OpenHintParams = match parse_params(request.params, request.id.clone()) {
                Ok(params) => params,
                Err(err) => return *err,
            };
            match query::open_hint(&state.registry, params).await {
                Ok(result) => response_from_result(request.id, &result),
                Err(err) => response_from_query_error(request.id, err),
            }
        }
        _ => RpcResponse::error(
            request.id,
            ERR_METHOD_NOT_FOUND,
            format!("unknown method '{}'", request.method),
        ),
    }
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
        status: "ok".to_string(),
        uptime_ms,
        model_name: state.model_name.clone(),
        semantic_home: state.semantic_home.display().to_string(),
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
        Arc::new(ServerState::new(
            model_name,
            semantic_home.to_path_buf(),
            registry,
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
                br#"{"jsonrpc":"2.0","id":7,"method":"health","params":{"min_api_version":1,"max_api_version":1}}"#
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
        assert_eq!(health_response["result"]["daemon_api_version"], json!(1));

        shutdown_tx
            .send(())
            .expect("shutdown signal should be delivered");
        server.await.expect("server task should join");
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
}
