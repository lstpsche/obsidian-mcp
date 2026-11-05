//! Exercise the daemon executable with an isolated, deterministic embedding provider.

#![cfg(all(feature = "embeddings-api", any(unix, windows)))]

#[path = "common/embedding_profiles.rs"]
mod embedding_profiles;
#[path = "common/http_listening.rs"]
mod http_listening;
#[path = "common/http_rpc.rs"]
mod http_rpc;

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex, atomic::AtomicBool};
use std::time::Duration;

use axum::{Json, Router, extract::State, routing::post};
use obsidian_mcp::client::semantic_daemon::{DaemonConnectPolicy, SemanticDaemonClient};
use obsidian_mcp::config::{Config, SemanticMode, ToolFilter, Transport};
use obsidian_mcp::daemon::protocol::{DAEMON_API_VERSION, ERR_VAULT_NOT_ATTACHED, IndexScope};
use obsidian_mcp::daemon::server::IpcEndpoint;
use obsidian_mcp::tools::{
    SemanticRuntime,
    search::{SearchSemanticParams, search_semantic},
};
use obsidian_mcp::vault::Vault;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::{Notify, watch};
use tokio::time::{sleep, timeout};

async fn embeddings(Json(input): Json<Value>) -> Json<Value> {
    let data: Vec<_> = input["input"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(index, text)| {
            let text = text.as_str().unwrap();
            let vector = if text.contains("TAIL_SIGNAL") {
                [0.0, 1.0]
            } else if text.contains("VISIBLE") {
                [0.8, 0.6]
            } else {
                [1.0, 0.0]
            };
            json!({"index": index, "embedding": vector})
        })
        .collect();
    Json(json!({"data": data}))
}

fn spawn_daemon(home: &Path, endpoint: &IpcEndpoint, api_port: u16) -> Child {
    spawn_daemon_with_limits(home, endpoint, api_port, 128, 32)
}

fn spawn_daemon_with_limits(
    home: &Path,
    endpoint: &IpcEndpoint,
    api_port: u16,
    max_input_tokens: usize,
    max_batch_inputs: usize,
) -> Child {
    daemon_command(
        home,
        endpoint,
        api_port,
        max_input_tokens,
        max_batch_inputs,
        "",
    )
    .spawn()
    .unwrap()
}

fn daemon_command(
    home: &Path,
    endpoint: &IpcEndpoint,
    api_port: u16,
    max_input_tokens: usize,
    max_batch_inputs: usize,
    exclusions: &str,
) -> Command {
    let profile = embedding_profiles::write_api_profile(
        home,
        api_port,
        "probe",
        max_input_tokens,
        max_batch_inputs,
    );
    let binary = std::env::var_os("OBSIDIAN_TEST_DAEMON_BINARY")
        .unwrap_or_else(|| env!("CARGO_BIN_EXE_obsidian-semanticd").into());
    let mut command = http_listening::isolated_command(binary);
    command
        .env("HOME", home)
        .env("USERPROFILE", home)
        .env("OBSIDIAN_SEMANTIC_HOME", home)
        .env("OBSIDIAN_SEMANTIC_ENDPOINT", endpoint.endpoint_string())
        .env("OBSIDIAN_SEMANTIC_MODEL", "probe")
        .env("OBSIDIAN_EXCLUDE_PATHS", exclusions)
        .env("OBSIDIAN_EMBEDDING_PROVIDER", "api")
        .env(
            "OBSIDIAN_EMBEDDING_API_BASE",
            format!("http://127.0.0.1:{api_port}/v1"),
        )
        .env("OBSIDIAN_EMBEDDING_API_KEY", "synthetic-test-key")
        .env("OBSIDIAN_EMBEDDING_API_MODEL", "probe")
        .env("OBSIDIAN_EMBEDDING_API_PROFILE", profile)
        .env("OBSIDIAN_EMBEDDING_DIM", "2")
        .env("OBSIDIAN_LOG_LEVEL", "error")
        .env("TOKIO_WORKER_THREADS", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .kill_on_drop(true);
    command
}

fn isolated_endpoint(directory: &Path) -> IpcEndpoint {
    #[cfg(unix)]
    {
        IpcEndpoint::UnixSocket(directory.join("daemon.sock"))
    }
    #[cfg(windows)]
    {
        IpcEndpoint::NamedPipe(format!(
            r"\\.\pipe\obsidian-api-test-{}-{}",
            std::process::id(),
            directory.file_name().unwrap().to_string_lossy()
        ))
    }
}

#[derive(Clone, Copy)]
enum InvalidEmbeddingResponse {
    WrongDimension,
    MissingEmbedding,
    RateLimited,
}

#[derive(Clone)]
struct ProviderState {
    requests: Arc<Mutex<Vec<Value>>>,
    received: Arc<Notify>,
    hold: watch::Receiver<bool>,
    invalid_response: Arc<Mutex<Option<InvalidEmbeddingResponse>>>,
}

struct ControlledProvider {
    port: u16,
    state: ProviderState,
    hold: watch::Sender<bool>,
    task: tokio::task::JoinHandle<()>,
}

impl ControlledProvider {
    async fn start(hold_requests: bool) -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let (hold, receiver) = watch::channel(hold_requests);
        let state = ProviderState {
            requests: Arc::new(Mutex::new(Vec::new())),
            received: Arc::new(Notify::new()),
            hold: receiver,
            invalid_response: Arc::new(Mutex::new(None)),
        };
        let app = Router::new()
            .route("/v1/embeddings", post(controlled_embeddings))
            .with_state(state.clone());
        let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        Self {
            port,
            state,
            hold,
            task,
        }
    }

    fn requests(&self) -> Vec<Value> {
        self.state.requests.lock().unwrap().clone()
    }

    fn count_inputs_containing(&self, marker: &str) -> usize {
        self.state
            .requests
            .lock()
            .unwrap()
            .iter()
            .flat_map(|request| request["input"].as_array().unwrap())
            .filter(|text| text.as_str().unwrap().contains(marker))
            .count()
    }

    async fn wait_for_input(&self, marker: &str) {
        timeout(Duration::from_secs(15), async {
            loop {
                let notified = self.state.received.notified();
                if self.count_inputs_containing(marker) > 0 {
                    return;
                }
                notified.await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("missing provider input marker {marker}"));
    }

    fn hold_requests(&self, hold: bool) {
        self.hold.send(hold).unwrap();
    }

    fn invalid_response(&self, response: Option<InvalidEmbeddingResponse>) {
        *self.state.invalid_response.lock().unwrap() = response;
    }

    async fn wait_for_requests(&self, count: usize) {
        timeout(Duration::from_secs(15), async {
            loop {
                let notified = self.state.received.notified();
                if self.state.requests.lock().unwrap().len() >= count {
                    return;
                }
                notified.await;
            }
        })
        .await
        .expect("embedding request did not arrive");
    }
}

impl Drop for ControlledProvider {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn controlled_embeddings(
    State(mut state): State<ProviderState>,
    Json(input): Json<Value>,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    state.requests.lock().unwrap().push(input.clone());
    state.received.notify_one();
    state.hold.wait_for(|hold| !*hold).await.unwrap();
    let failure = *state.invalid_response.lock().unwrap();
    match failure {
        Some(InvalidEmbeddingResponse::WrongDimension) => {
            Json(json!({"data": [{"index": 0, "embedding": [1.0]}]})).into_response()
        }
        Some(InvalidEmbeddingResponse::MissingEmbedding) => {
            Json(json!({"data": []})).into_response()
        }
        Some(InvalidEmbeddingResponse::RateLimited) => (
            axum::http::StatusCode::TOO_MANY_REQUESTS,
            [(axum::http::header::RETRY_AFTER, "2")],
        )
            .into_response(),
        None => embeddings(Json(input)).await.into_response(),
    }
}

struct ClientIndex<'a> {
    max_input_tokens: usize,
    watch: bool,
    exclude_paths: &'a str,
}

fn mcp_command(
    root: &Path,
    home: &Path,
    endpoint: &IpcEndpoint,
    mode: &str,
    api_port: u16,
    policy: ClientIndex<'_>,
) -> Command {
    let profile = embedding_profiles::write_api_profile(
        home,
        api_port,
        "mcp-local",
        policy.max_input_tokens,
        2,
    );
    let binary = std::env::var_os("OBSIDIAN_TEST_MCP_BINARY")
        .unwrap_or_else(|| env!("CARGO_BIN_EXE_obsidian-mcp").into());
    let mut command = http_listening::isolated_command(binary);
    command
        .arg(root)
        .env("HOME", home)
        .env("USERPROFILE", home)
        .env("OBSIDIAN_MCP_DATA", home.join("mcp-data"))
        .env("OBSIDIAN_SEMANTIC_HOME", home.join("semantic"))
        .env("OBSIDIAN_SEMANTIC_ENDPOINT", endpoint.endpoint_string())
        .env("OBSIDIAN_SEMANTIC_MODE", mode)
        .env("OBSIDIAN_SEMANTIC_CONNECT_TIMEOUT_MS", "1500")
        .env("OBSIDIAN_SEMANTIC_CONNECT_RETRIES", "0")
        .env("OBSIDIAN_EMBEDDINGS", "true")
        .env("OBSIDIAN_EMBEDDING_PROVIDER", "api")
        .env(
            "OBSIDIAN_EMBEDDING_API_BASE",
            format!("http://127.0.0.1:{api_port}/v1"),
        )
        .env("OBSIDIAN_EMBEDDING_API_KEY", "synthetic-test-key")
        .env("OBSIDIAN_EMBEDDING_API_MODEL", "mcp-local")
        .env("OBSIDIAN_EMBEDDING_API_PROFILE", profile)
        .env("OBSIDIAN_EMBEDDING_DIM", "2")
        .env("OBSIDIAN_WATCH", policy.watch.to_string())
        .env("OBSIDIAN_EXCLUDE_PATHS", policy.exclude_paths)
        .env("OBSIDIAN_LOG_LEVEL", "error")
        .env("TOKIO_WORKER_THREADS", "1")
        .kill_on_drop(true);
    command
}

struct StdioServer {
    child: Child,
    input: ChildStdin,
    output: Lines<BufReader<ChildStdout>>,
    next_id: u64,
}

impl StdioServer {
    async fn start(
        root: &Path,
        home: &Path,
        endpoint: &IpcEndpoint,
        mode: &str,
        api_port: u16,
    ) -> Self {
        Self::start_with_options(root, home, endpoint, mode, api_port, 128, true).await
    }

    async fn start_with_options(
        root: &Path,
        home: &Path,
        endpoint: &IpcEndpoint,
        mode: &str,
        api_port: u16,
        max_input_tokens: usize,
        watch: bool,
    ) -> Self {
        Self::start_with_policy(
            root,
            home,
            endpoint,
            mode,
            api_port,
            ClientIndex {
                max_input_tokens,
                watch,
                exclude_paths: "",
            },
        )
        .await
    }

    async fn start_with_policy(
        root: &Path,
        home: &Path,
        endpoint: &IpcEndpoint,
        mode: &str,
        api_port: u16,
        policy: ClientIndex<'_>,
    ) -> Self {
        let mut child = mcp_command(root, home, endpoint, mode, api_port, policy)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap();
        let mut server = Self {
            input: child.stdin.take().unwrap(),
            output: BufReader::new(child.stdout.take().unwrap()).lines(),
            child,
            next_id: 1,
        };
        let initialized = server
            .call(
                "initialize",
                json!({
                    "protocolVersion": "2025-11-25", "capabilities": {},
                    "clientInfo": {"name": "embedding-test", "version": "1"}
                }),
            )
            .await;
        assert!(initialized.get("error").is_none(), "{initialized}");
        server
            .input
            .write_all(b"{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\n")
            .await
            .unwrap();
        server.input.flush().await.unwrap();
        server
    }

    async fn call(&mut self, method: &str, params: Value) -> Value {
        let id = self.send_request(method, params).await;
        self.receive_response(id).await
    }

    async fn send_request(&mut self, method: &str, params: Value) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let mut request =
            json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}).to_string();
        request.push('\n');
        self.input.write_all(request.as_bytes()).await.unwrap();
        self.input.flush().await.unwrap();
        id
    }

    async fn receive_response(&mut self, id: u64) -> Value {
        timeout(Duration::from_secs(15), async {
            loop {
                let line = self.output.next_line().await.unwrap().expect("MCP exited");
                let response: Value = serde_json::from_str(&line).unwrap();
                if response["id"] == id {
                    return response;
                }
            }
        })
        .await
        .expect("MCP response did not arrive")
    }

    async fn tool(&mut self, name: &str, arguments: Value) -> Value {
        self.call("tools/call", json!({"name": name, "arguments": arguments}))
            .await
    }

    async fn embedding_info(&mut self) -> Value {
        let response = self.tool("vault_info", json!({})).await;
        assert!(response.get("error").is_none(), "{response}");
        response["result"]["structuredContent"]["embeddings"].clone()
    }

    async fn wait_for_ready(&mut self, source: &str) {
        self.wait_for_ready_notes(source, 1).await;
    }

    async fn wait_for_ready_notes(&mut self, source: &str, notes: usize) {
        timeout(Duration::from_secs(15), async {
            loop {
                let info = self.embedding_info().await;
                assert_eq!(info["source"], source, "{info}");
                if info["status"]["phase"] == "ready" && info["status"]["indexed_notes"] == notes {
                    assert_eq!(info["status"]["pending_notes"], 0);
                    return;
                }
                sleep(Duration::from_millis(25)).await;
            }
        })
        .await
        .expect("embedding index did not become ready");
    }

    async fn wait_for_local_note_count(&mut self, expected: usize) {
        // Each MCP process and its daemon consume filesystem notifications independently.
        // A ready daemon can see an incoming file before this client's allowed_paths does.
        let mut local_notes = Value::Null;
        timeout(Duration::from_secs(10), async {
            loop {
                let response = self.tool("vault_info", json!({})).await;
                local_notes = structured_result(&response)["total_notes"].clone();
                if local_notes == expected {
                    return;
                }
                sleep(Duration::from_millis(25)).await;
            }
        })
        .await
        .unwrap_or_else(|_| {
            panic!("local vault reported {local_notes} notes, expected {expected}")
        });
    }

    async fn attach_and_check_paths(&mut self, mut expected: Vec<&str>) {
        let query = json!({"query":"quokka","top_k":10});
        let first = self.tool("search_semantic", query.clone()).await;
        if first.get("error").is_some() {
            assert_warming(&first);
        }
        self.wait_for_ready_notes("daemon", expected.len()).await;
        let tools = self.call("tools/list", json!({})).await;
        let schema = tool_output_schema(&tools, "search_semantic");
        expected.sort_unstable();
        for hybrid in [false, true] {
            let mut request = query.clone();
            request["lexical_prefetch"] = json!(hybrid);
            let response = self.tool("search_semantic", request).await;
            let mut found: Vec<_> = validated_output(&schema, &response)["results"]
                .as_array()
                .unwrap()
                .iter()
                .map(|hit| hit["path"].as_str().unwrap().to_owned())
                .collect();
            found.sort_unstable();
            assert_eq!(found, expected, "{response}");
        }
    }

    async fn stop(mut self) {
        self.child.kill().await.unwrap();
        self.child.wait().await.unwrap();
    }
}

async fn wait_for_health(child: &mut Child, client: &SemanticDaemonClient) {
    timeout(Duration::from_secs(15), async {
        loop {
            assert!(
                child.try_wait().unwrap().is_none(),
                "daemon exited during startup"
            );
            if let Ok(health) = client.health("test", "1").await {
                assert_eq!(health.daemon_api_version, DAEMON_API_VERSION);
                return;
            }
            sleep(Duration::from_millis(25)).await;
        }
    })
    .await
    .unwrap();
}

async fn tool_search(vault: &Vault, runtime: &SemanticRuntime, hybrid: bool) -> Value {
    timeout(Duration::from_secs(15), async {
        loop {
            let result = search_semantic(
                vault,
                SearchSemanticParams {
                    query: "quokka".into(),
                    top_k: Some(1),
                    lexical_prefetch: Some(hybrid),
                    ..Default::default()
                },
                0.25,
                runtime,
            )
            .await;
            match result {
                Ok(result) => return result.structured_content.unwrap(),
                Err(error) if error.message.contains("warming") => {
                    sleep(Duration::from_millis(25)).await
                }
                Err(error) => panic!("semantic query failed: {error}"),
            }
        }
    })
    .await
    .unwrap()
}

fn assert_warming(response: &Value) {
    assert!(
        response["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("warming")),
        "{response}"
    );
}

fn assert_note_found(response: &Value) {
    assert!(response.get("error").is_none(), "{response}");
    let results = response["result"]["structuredContent"]["results"]
        .as_array()
        .unwrap();
    assert_eq!(results.len(), 1, "{response}");
    assert_eq!(results[0]["path"], "note.md");
}

async fn verify_daemon_indexing_ownership(mode: &str) {
    let daemon_api = ControlledProvider::start(true).await;
    let local_api = ControlledProvider::start(true).await;
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("vault");
    std::fs::create_dir_all(root.join("Private")).unwrap();
    std::fs::write(root.join("Private/hidden.md"), "quokka EXCLUDED_BY_CLIENT").unwrap();
    std::fs::write(root.join("note.md"), "# Marker\nVISIBLE quokka\n").unwrap();
    let endpoint = isolated_endpoint(directory.path());
    let client = SemanticDaemonClient::new(
        endpoint.clone(),
        DaemonConnectPolicy {
            timeout: Duration::from_secs(1),
            retries: 0,
            ..Default::default()
        },
    );
    let home = directory.path().join("semantic");
    let mut daemon = spawn_daemon(&home, &endpoint, daemon_api.port);
    wait_for_health(&mut daemon, &client).await;
    let mut server = StdioServer::start_with_policy(
        &root,
        directory.path(),
        &endpoint,
        mode,
        local_api.port,
        ClientIndex {
            max_input_tokens: 128,
            watch: true,
            exclude_paths: "Private/",
        },
    )
    .await;

    // Initialization and a passive tool response form a startup barrier.
    let info = server.embedding_info().await;
    assert_eq!(info["source"], "daemon", "{info}");
    assert_eq!(info["availability"], "unattached", "{info}");
    assert!(info.get("status").is_none(), "{info}");
    assert!(daemon_api.requests().is_empty());
    assert!(local_api.requests().is_empty());

    let query = json!({"query": "quokka", "top_k": 1});
    assert_warming(&server.tool("search_semantic", query.clone()).await);
    daemon_api.wait_for_requests(1).await;
    let info = server.embedding_info().await;
    assert_eq!(info["source"], "daemon", "{info}");
    assert_eq!(info["availability"], "available", "{info}");
    assert_eq!(info["status"]["phase"], "warming", "{info}");
    assert_eq!(info["status"]["pending_notes"], 1, "{info}");
    assert_eq!(info["status"]["queryable"], false, "{info}");
    assert_warming(&server.tool("search_semantic", query.clone()).await);
    assert_eq!(daemon_api.requests().len(), 1);
    assert!(local_api.requests().is_empty());

    daemon_api.hold_requests(false);
    server.wait_for_ready("daemon").await;
    assert_note_found(&server.tool("search_semantic", query.clone()).await);
    let requests = daemon_api.requests();
    assert_eq!(requests.len(), 2);
    assert!(requests.iter().all(|request| request["model"] == "probe"));
    assert!(
        requests[0]["input"][0]
            .as_str()
            .unwrap()
            .contains("VISIBLE")
    );
    assert_eq!(requests[1]["input"][0], "quokka");
    assert!(local_api.requests().is_empty());

    if mode == "auto" {
        // An in-flight daemon query timeout is not evidence of an unavailable daemon.
        daemon_api.hold_requests(true);
        let response = server.tool("search_semantic", query.clone()).await;
        daemon_api.wait_for_requests(3).await;
        assert!(
            response["error"]["message"]
                .as_str()
                .is_some_and(|message| message.contains("timed out")),
            "{response}"
        );
        assert!(local_api.requests().is_empty());
        let info = server.embedding_info().await;
        assert_eq!(info["source"], "daemon", "{info}");
        assert_eq!(info["status"]["phase"], "ready", "{info}");
        daemon_api.hold_requests(false);
        assert_note_found(&server.tool("search_semantic", query.clone()).await);
        assert_eq!(daemon_api.requests().len(), 4);
        assert!(local_api.requests().is_empty());

        daemon.kill().await.unwrap();
        daemon.wait().await.unwrap();
        assert_warming(&server.tool("search_semantic", query.clone()).await);
        local_api.wait_for_requests(1).await;
        let info = server.embedding_info().await;
        assert_eq!(info["source"], "local", "{info}");
        assert_eq!(info["status"]["phase"], "warming", "{info}");
        assert_warming(&server.tool("search_semantic", query.clone()).await);
        assert_eq!(local_api.requests().len(), 1);
        local_api.hold_requests(false);
        server.wait_for_ready("local").await;
        assert_note_found(&server.tool("search_semantic", query.clone()).await);
        assert_eq!(local_api.requests().len(), 2);

        // Restoring the daemon does not create a second owner during this MCP lifetime.
        daemon = spawn_daemon(&home, &endpoint, daemon_api.port);
        wait_for_health(&mut daemon, &client).await;
        assert_note_found(&server.tool("search_semantic", query).await);
        assert_eq!(server.embedding_info().await["source"], "local");
        let requests = local_api.requests();
        assert_eq!(requests.len(), 3);
        assert!(
            requests
                .iter()
                .all(|request| request["model"] == "mcp-local")
        );
        assert_eq!(daemon_api.requests().len(), 4);
    }

    assert_eq!(daemon_api.count_inputs_containing("EXCLUDED_BY_CLIENT"), 0);
    assert_eq!(local_api.count_inputs_containing("EXCLUDED_BY_CLIENT"), 0);
    server.stop().await;
    daemon.kill().await.unwrap();
    daemon.wait().await.unwrap();
}

#[tokio::test]
async fn daemon_mode_defers_local_indexing_and_exposes_passive_progress() {
    verify_daemon_indexing_ownership("daemon").await;
}

#[tokio::test]
async fn auto_mode_retains_daemon_ownership_until_unavailable_then_keeps_local_fallback() {
    verify_daemon_indexing_ownership("auto").await;
}

#[tokio::test]
async fn invalid_query_vectors_preserve_daemon_ownership_in_auto_mode() {
    let daemon_api = ControlledProvider::start(false).await;
    let local_api = ControlledProvider::start(false).await;
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("vault");
    std::fs::create_dir(&root).unwrap();
    std::fs::write(root.join("note.md"), "# Marker\nVISIBLE quokka\n").unwrap();
    let endpoint = isolated_endpoint(directory.path());
    let client = SemanticDaemonClient::new(endpoint.clone(), DaemonConnectPolicy::default());
    let mut daemon = spawn_daemon(
        &directory.path().join("semantic"),
        &endpoint,
        daemon_api.port,
    );
    wait_for_health(&mut daemon, &client).await;
    let mut server =
        StdioServer::start(&root, directory.path(), &endpoint, "auto", local_api.port).await;
    let first = server
        .tool("search_semantic", json!({"query":"quokka","top_k":1}))
        .await;
    if first.get("error").is_some() {
        assert_warming(&first);
    }
    server.wait_for_ready("daemon").await;
    assert!(local_api.requests().is_empty());
    let mut expected_requests = daemon_api.requests().len();

    for invalid in [
        InvalidEmbeddingResponse::WrongDimension,
        InvalidEmbeddingResponse::MissingEmbedding,
    ] {
        daemon_api.invalid_response(Some(invalid));
        for lexical_prefetch in [false, true] {
            let response = server
                .tool(
                    "search_semantic",
                    json!({"query":"quokka","top_k":1,"lexical_prefetch":lexical_prefetch}),
                )
                .await;
            assert_eq!(response["error"]["code"], -32603, "{response}");
            expected_requests += 1;
            assert_eq!(daemon_api.requests().len(), expected_requests);
            assert!(local_api.requests().is_empty());
            let info = server.embedding_info().await;
            assert_eq!(info["source"], "daemon", "{info}");
            assert_eq!(info["status"]["phase"], "ready", "{info}");
        }
    }
    daemon_api.invalid_response(None);
    assert_note_found(
        &server
            .tool("search_semantic", json!({"query":"quokka","top_k":1}))
            .await,
    );
    assert!(local_api.requests().is_empty());
    assert!(
        daemon_api
            .requests()
            .iter()
            .all(|request| request["model"] == "probe")
    );
    assert_eq!(server.embedding_info().await["source"], "daemon");
    server.stop().await;
    daemon.kill().await.unwrap();
    daemon.wait().await.unwrap();
}

#[tokio::test]
async fn local_semantic_queries_keep_progress_responsive_on_one_worker() {
    for lexical_prefetch in [false, true] {
        let provider = ControlledProvider::start(false).await;
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("vault");
        std::fs::create_dir(&root).unwrap();
        std::fs::write(root.join("note.md"), "# Marker\nVISIBLE quokka\n").unwrap();
        let endpoint = isolated_endpoint(directory.path());
        let mut server =
            StdioServer::start(&root, directory.path(), &endpoint, "local", provider.port).await;
        server.wait_for_ready("local").await;
        assert_eq!(provider.requests().len(), 1);

        provider.hold_requests(true);
        let query_id = server
            .send_request(
                "tools/call",
                json!({
                    "name": "search_semantic",
                    "arguments": {"query": "quokka", "top_k": 1, "lexical_prefetch": lexical_prefetch}
                }),
            )
            .await;
        provider.wait_for_requests(2).await;
        let info = server.embedding_info().await;
        assert_eq!(info["source"], "local", "{info}");
        assert_eq!(info["status"]["phase"], "ready", "{info}");
        provider.hold_requests(false);
        assert_note_found(&server.receive_response(query_id).await);
        server.stop().await;
    }
}

#[tokio::test]
async fn daemon_binary_filters_before_ranking_and_recovers_existing_clients() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let fixture = tokio::spawn(async move {
        axum::serve(
            listener,
            Router::new().route("/v1/embeddings", post(embeddings)),
        )
        .await
        .unwrap();
    });
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("vault");
    std::fs::create_dir_all(root.join("Archive")).unwrap();
    for id in 0..30 {
        std::fs::write(
            root.join(format!("Archive/{id}.md")),
            "quokka quokka quokka",
        )
        .unwrap();
    }
    std::fs::write(root.join("visible.md"), "VISIBLE quokka").unwrap();
    let config = Config {
        vault_path: root,
        watch: false,
        log_level: "error".into(),
        transport: Transport::Stdio,
        http_host: "127.0.0.1".parse().unwrap(),
        http_port: 37842,
        tantivy: true,
        embeddings: false,
        embeddings_model: "probe".into(),
        hybrid_alpha: 0.25,
        embedding_provider: None,
        tool_filter: ToolFilter::Full,
        mcp_data_dir: None,
        exclude_patterns: vec!["Archive/**".into()],
    };
    let vault = Vault::open(&config).await.unwrap();
    let endpoint = isolated_endpoint(directory.path());
    let client = SemanticDaemonClient::new(
        endpoint.clone(),
        DaemonConnectPolicy {
            retries: 0,
            ..Default::default()
        },
    );
    let runtime = SemanticRuntime {
        mode: SemanticMode::Daemon,
        pending_daemon_bootstrap: None,
        daemon_client: Some(client.clone()),
        daemon_unavailable_reason: None,
        prefetch_count: 1,
        attachments: Default::default(),
        local_fallback_active: Arc::new(AtomicBool::new(false)),
        daemon_health_pending: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
    };
    let home = directory.path().join("semantic");
    let mut daemon = spawn_daemon(&home, &endpoint, port);
    wait_for_health(&mut daemon, &client).await;
    for hybrid in [false, true] {
        let result = tool_search(&vault, &runtime, hybrid).await;
        assert_eq!(result["results"].as_array().unwrap().len(), 1);
        assert_eq!(result["results"][0]["path"], "visible.md");
    }
    let mut excluded_config = config.clone();
    excluded_config.exclude_patterns = vec!["**".into()];
    let excluded_vault = Vault::open(&excluded_config).await.unwrap();
    for hybrid in [false, true] {
        assert!(
            tool_search(&excluded_vault, &runtime, hybrid).await["results"]
                .as_array()
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            tool_search(&vault, &runtime, hybrid).await["results"][0]["path"],
            "visible.md"
        );
    }
    daemon.kill().await.unwrap();
    daemon.wait().await.unwrap();
    daemon = spawn_daemon(&home, &endpoint, port);
    wait_for_health(&mut daemon, &client).await;
    assert_eq!(
        tool_search(&vault, &runtime, false).await["results"][0]["path"],
        "visible.md"
    );
    daemon.kill().await.unwrap();
    daemon.wait().await.unwrap();
    fixture.abort();
    let _ = fixture.await;
}

async fn wait_for_detachment(client: &SemanticDaemonClient, root: &Path, scope: &IndexScope) {
    timeout(Duration::from_secs(10), async {
        loop {
            match client.vault_status(root, scope).await {
                Err(obsidian_mcp::error::VaultError::DaemonRpc {
                    code: ERR_VAULT_NOT_ATTACHED,
                    ..
                }) => return,
                Ok(_) => sleep(Duration::from_millis(20)).await,
                Err(error) => panic!("cannot observe attachment retirement: {error}"),
            }
        }
    })
    .await
    .expect("last holder did not release the scope");
}

async fn check_shared_vault_scopes(broader_first: bool) {
    let provider = ControlledProvider::start(false).await;
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("vault");
    let home = directory.path().join("semantic");
    for folder in [".obsidian-mcp", "Global", "Private"] {
        std::fs::create_dir_all(root.join(folder)).unwrap();
    }
    std::fs::write(root.join(".obsidian-mcp/ignore"), "Global/\n").unwrap();
    std::fs::write(root.join("Global/marker.md"), "quokka GLOBAL_NEVER_SENT").unwrap();
    std::fs::write(root.join("Private/marker.md"), "quokka PRIVATE_AT_START").unwrap();
    std::fs::write(root.join("shared.md"), "quokka SHARED_AT_START").unwrap();
    #[cfg(unix)]
    std::os::unix::fs::symlink(root.join("Global/marker.md"), root.join("alias.md")).unwrap();
    let endpoint = isolated_endpoint(directory.path());
    let client = SemanticDaemonClient::new(endpoint.clone(), DaemonConnectPolicy::default());
    let mut daemon = spawn_daemon(&home, &endpoint, provider.port);
    wait_for_health(&mut daemon, &client).await;

    // These processes share a vault, daemon and provider. Only their own policy differs.
    let mut narrow = StdioServer::start_with_policy(
        &root,
        directory.path(),
        &endpoint,
        "daemon",
        provider.port,
        ClientIndex {
            max_input_tokens: 128,
            watch: true,
            exclude_paths: "Private/",
        },
    )
    .await;
    let mut broad =
        StdioServer::start(&root, directory.path(), &endpoint, "daemon", provider.port).await;
    assert_eq!(narrow.embedding_info().await["availability"], "unattached");
    assert_eq!(broad.embedding_info().await["availability"], "unattached");
    assert!(provider.requests().is_empty());

    let narrow_paths = vec!["shared.md"];
    let broad_paths = vec!["Private/marker.md", "shared.md"];
    if broader_first {
        broad.attach_and_check_paths(broad_paths.clone()).await;
        assert_eq!(narrow.embedding_info().await["availability"], "unattached");
        narrow.attach_and_check_paths(narrow_paths.clone()).await;
    } else {
        narrow.attach_and_check_paths(narrow_paths.clone()).await;
        assert_eq!(provider.count_inputs_containing("PRIVATE_AT_START"), 0);
        assert_eq!(broad.embedding_info().await["availability"], "unattached");
        broad.attach_and_check_paths(broad_paths.clone()).await;
    }
    assert_eq!(provider.count_inputs_containing("SHARED_AT_START"), 2);
    assert_eq!(provider.count_inputs_containing("PRIVATE_AT_START"), 1);
    assert_eq!(provider.count_inputs_containing("GLOBAL_NEVER_SENT"), 0);

    // Equivalent exclusions attach to the live index and add no document work.
    let narrow_scope = IndexScope::new(&["Private/**".into()]);
    let mut twin = StdioServer::start_with_policy(
        &root,
        directory.path(),
        &endpoint,
        "daemon",
        provider.port,
        ClientIndex {
            max_input_tokens: 128,
            watch: false,
            exclude_paths: "Private/**",
        },
    )
    .await;
    twin.attach_and_check_paths(narrow_paths.clone()).await;
    assert_eq!(provider.count_inputs_containing("SHARED_AT_START"), 2);
    wait_for_checkpoints(&home, 2).await;

    broad.stop().await;
    wait_for_detachment(&client, &root, &IndexScope::default()).await;
    std::fs::write(root.join("Private/marker.md"), "quokka PRIVATE_AFTER_EXIT").unwrap();
    std::fs::write(root.join("Global/marker.md"), "quokka GLOBAL_AFTER_EDIT").unwrap();
    let created = root.join("Private/new.md");
    std::fs::write(&created, "quokka CREATED_IN_PRIVATE").unwrap();
    std::fs::write(root.join("shared.md"), "quokka SHARED_AFTER_EXIT").unwrap();
    provider.wait_for_input("SHARED_AFTER_EXIT").await;
    narrow.attach_and_check_paths(narrow_paths.clone()).await;
    assert_eq!(provider.count_inputs_containing("PRIVATE_AFTER_EXIT"), 0);
    assert_eq!(provider.count_inputs_containing("GLOBAL_AFTER_EDIT"), 0);
    assert_eq!(provider.count_inputs_containing("CREATED_IN_PRIVATE"), 0);

    let moved = root.join("moved.md");
    std::fs::rename(&created, &moved).unwrap();
    provider.wait_for_input("CREATED_IN_PRIVATE").await;
    narrow.wait_for_local_note_count(2).await;
    narrow
        .attach_and_check_paths(vec!["moved.md", "shared.md"])
        .await;
    std::fs::rename(&moved, &created).unwrap();
    std::fs::write(&created, "quokka RENAMED_BACK_TO_PRIVATE").unwrap();
    std::fs::write(root.join("shared.md"), "quokka VISIBLE_AFTER_RENAME").unwrap();
    provider.wait_for_input("VISIBLE_AFTER_RENAME").await;
    narrow.wait_for_local_note_count(1).await;
    narrow.attach_and_check_paths(narrow_paths.clone()).await;
    assert_eq!(provider.count_inputs_containing("CREATED_IN_PRIVATE"), 1);
    assert_eq!(
        provider.count_inputs_containing("RENAMED_BACK_TO_PRIVATE"),
        0
    );
    std::fs::remove_file(&created).unwrap();

    narrow.stop().await;
    timeout(Duration::from_secs(10), async {
        loop {
            let probe = client
                .attach_vault(&root, &narrow_scope, false, None)
                .await
                .unwrap();
            if !probe.result.watch_enabled {
                break;
            }
            drop(probe);
            sleep(Duration::from_millis(25)).await;
        }
    })
    .await
    .expect("last watch-enabled holder must release watcher");
    twin.attach_and_check_paths(narrow_paths.clone()).await;
    twin.stop().await;
    wait_for_detachment(&client, &root, &narrow_scope).await;

    daemon.kill().await.unwrap();
    daemon.wait().await.unwrap();
    let mut daemon = spawn_daemon(&home, &endpoint, provider.port);
    wait_for_health(&mut daemon, &client).await;
    wait_for_detachment(&client, &root, &IndexScope::default()).await;
    let mut narrow = StdioServer::start_with_policy(
        &root,
        directory.path(),
        &endpoint,
        "daemon",
        provider.port,
        ClientIndex {
            max_input_tokens: 128,
            watch: false,
            exclude_paths: "Private/**",
        },
    )
    .await;
    narrow.attach_and_check_paths(narrow_paths).await;
    assert_eq!(provider.count_inputs_containing("PRIVATE_AFTER_EXIT"), 0);
    let mut broad =
        StdioServer::start(&root, directory.path(), &endpoint, "daemon", provider.port).await;
    broad.attach_and_check_paths(broad_paths).await;
    assert_eq!(provider.count_inputs_containing("PRIVATE_AFTER_EXIT"), 1);
    assert_eq!(provider.count_inputs_containing("GLOBAL_AFTER_EDIT"), 0);
    narrow.stop().await;
    broad.stop().await;
    daemon.kill().await.unwrap();
    daemon.wait().await.unwrap();
}

#[tokio::test]
async fn daemon_limits_provider_inputs_to_independently_attached_scopes() {
    for broader_first in [false, true] {
        check_shared_vault_scopes(broader_first).await;
    }
}

fn tool_output_schema(tools: &Value, name: &str) -> jsonschema::Validator {
    let descriptor = tools["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .find(|tool| tool["name"] == name)
        .expect("tool is advertised");
    jsonschema::validator_for(&descriptor["outputSchema"]).unwrap()
}

fn validated_output<'a>(schema: &jsonschema::Validator, response: &'a Value) -> &'a Value {
    let result = structured_result(response);
    schema
        .validate(result)
        .expect("result matches advertised schema");
    result
}

#[tokio::test]
async fn stateless_http_filters_share_one_configured_scope() {
    let provider = ControlledProvider::start(true).await;
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("vault");
    std::fs::create_dir_all(root.join("Private")).unwrap();
    std::fs::write(root.join("shared.md"), "quokka HTTP_VISIBLE_DOCUMENT").unwrap();
    std::fs::write(
        root.join("Private/hidden.md"),
        "quokka HTTP_PRIVATE_DOCUMENT",
    )
    .unwrap();
    let endpoint = isolated_endpoint(directory.path());
    let client = SemanticDaemonClient::new(endpoint.clone(), DaemonConnectPolicy::default());
    let mut daemon = spawn_daemon(&directory.path().join("semantic"), &endpoint, provider.port);
    wait_for_health(&mut daemon, &client).await;
    let mut server = mcp_command(
        &root,
        directory.path(),
        &endpoint,
        "daemon",
        provider.port,
        ClientIndex {
            max_input_tokens: 128,
            watch: true,
            exclude_paths: "Private/",
        },
    )
    .args(["--http", "--host", "127.0.0.1", "--port", "0"])
    .env("OBSIDIAN_LOG_LEVEL", "info")
    .env("NO_COLOR", "1")
    .stdin(Stdio::null())
    .stdout(Stdio::null())
    .stderr(Stdio::piped())
    .spawn()
    .unwrap();
    let address = http_listening::ready(&mut server).await.unwrap();
    let url = format!("http://{address}");
    let http = reqwest::Client::builder().no_proxy().build().unwrap();
    let request = async |filter: &str, method: &str, params: Value| {
        let response = http_rpc::request(&http, &url, method, params, http_rpc::MODERN)
            .header("X-Obsidian-Tools", filter)
            .send()
            .await
            .unwrap();
        http_rpc::rpc_response(response).await
    };
    let tools = request("full", "tools/list", json!({})).await;
    let status_schema = tool_output_schema(&tools, "vault_info");
    let search_schema = tool_output_schema(&tools, "search_semantic");
    let status = json!({"name": "vault_info", "arguments": {}});
    let passive = request("vault_info", "tools/call", status.clone()).await;
    assert_eq!(
        validated_output(&status_schema, &passive)["embeddings"]["availability"],
        "unattached"
    );
    assert!(provider.requests().is_empty());

    let search = json!({"name": "search_semantic", "arguments": {"query":"quokka"}});
    assert_warming(&request("search_semantic", "tools/call", search.clone()).await);
    provider.wait_for_input("HTTP_VISIBLE_DOCUMENT").await;
    let progress = request("vault_info", "tools/call", status.clone()).await;
    assert_eq!(
        validated_output(&status_schema, &progress)["embeddings"]["status"]["total_notes"],
        1,
    );
    provider.hold_requests(false);
    timeout(Duration::from_secs(10), async {
        loop {
            let info = request("vault_info", "tools/call", status.clone()).await;
            if validated_output(&status_schema, &info)["embeddings"]["status"]["phase"] == "ready" {
                break;
            }
            sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .unwrap();
    let (full, filtered) = tokio::join!(
        request("full", "tools/call", search.clone()),
        request("search_semantic", "tools/call", search),
    );
    for response in [full, filtered] {
        let hits = validated_output(&search_schema, &response)["results"]
            .as_array()
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0]["path"], "shared.md");
    }
    assert_eq!(provider.count_inputs_containing("HTTP_VISIBLE_DOCUMENT"), 1);
    assert_eq!(provider.count_inputs_containing("HTTP_PRIVATE_DOCUMENT"), 0);
    server.kill().await.unwrap();
    server.wait().await.unwrap();
    wait_for_detachment(&client, &root, &IndexScope::new(&["Private/".into()])).await;
    daemon.kill().await.unwrap();
    daemon.wait().await.unwrap();
}

#[tokio::test]
async fn detached_scope_cannot_dispatch_the_next_document_batch() {
    let provider = ControlledProvider::start(false).await;
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("vault");
    std::fs::create_dir_all(root.join("Private")).unwrap();
    std::fs::write(root.join("Private/a.md"), "quokka IN_FLIGHT_PRIVATE").unwrap();
    std::fs::write(root.join("Private/b.md"), "quokka NEXT_PRIVATE_BATCH").unwrap();
    std::fs::write(root.join("shared.md"), "quokka visible").unwrap();
    let endpoint = isolated_endpoint(directory.path());
    let client = SemanticDaemonClient::new(endpoint.clone(), DaemonConnectPolicy::default());
    let mut daemon = spawn_daemon_with_limits(
        &directory.path().join("semantic"),
        &endpoint,
        provider.port,
        128,
        1,
    );
    wait_for_health(&mut daemon, &client).await;
    let mut narrow = StdioServer::start_with_policy(
        &root,
        directory.path(),
        &endpoint,
        "daemon",
        provider.port,
        ClientIndex {
            max_input_tokens: 128,
            watch: true,
            exclude_paths: "Private/",
        },
    )
    .await;
    narrow.attach_and_check_paths(vec!["shared.md"]).await;
    provider.hold_requests(true);
    let mut broad =
        StdioServer::start(&root, directory.path(), &endpoint, "daemon", provider.port).await;
    assert_warming(
        &broad
            .tool("search_semantic", json!({"query":"quokka"}))
            .await,
    );
    provider.wait_for_input("IN_FLIGHT_PRIVATE").await;
    broad.stop().await;
    wait_for_detachment(&client, &root, &IndexScope::default()).await;
    assert_eq!(narrow.embedding_info().await["status"]["phase"], "ready");
    let health = client.health("scope-test", "1").await.unwrap();
    assert_eq!(health.pid, daemon.id().unwrap());

    std::fs::write(root.join("shared.md"), "quokka NARROW_COMPLETED").unwrap();
    provider.hold_requests(false);
    provider.wait_for_input("NARROW_COMPLETED").await;
    narrow.attach_and_check_paths(vec!["shared.md"]).await;
    assert_eq!(provider.count_inputs_containing("NEXT_PRIVATE_BATCH"), 0);
    assert_eq!(provider.count_inputs_containing("IN_FLIGHT_PRIVATE"), 1);
    narrow.stop().await;
    daemon.kill().await.unwrap();
    daemon.wait().await.unwrap();
}

#[tokio::test]
async fn a_provider_retry_queued_by_a_disconnected_scope_cannot_send_documents() {
    let api = ControlledProvider::start(false).await;
    api.invalid_response(Some(InvalidEmbeddingResponse::RateLimited));
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("vault");
    std::fs::create_dir_all(root.join("Private")).unwrap();
    std::fs::write(root.join("Private/note.md"), "PRIVATE_RETRY").unwrap();
    std::fs::write(root.join("public.md"), "LIVE_SCOPE_RETRY").unwrap();
    let endpoint = isolated_endpoint(directory.path());
    let client = SemanticDaemonClient::new(endpoint.clone(), DaemonConnectPolicy::default());
    let mut daemon = spawn_daemon(&directory.path().join("semantic"), &endpoint, api.port);
    wait_for_health(&mut daemon, &client).await;

    let broad_scope = IndexScope::default();
    let narrow_scope = IndexScope::new(&["Private/".into()]);
    let broad = client
        .attach_vault(&root, &broad_scope, false, None)
        .await
        .unwrap();
    let narrow = client
        .attach_vault(&root, &narrow_scope, false, None)
        .await
        .unwrap();
    for scope in [&broad_scope, &narrow_scope] {
        timeout(Duration::from_secs(10), async {
            loop {
                let status = client.vault_status(&root, scope).await.unwrap();
                if status
                    .last_error
                    .as_deref()
                    .is_some_and(|message| message.contains("429"))
                {
                    break;
                }
                sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("each scope must observe the retryable provider response");
    }
    drop(broad);
    wait_for_detachment(&client, &root, &broad_scope).await;
    let private_attempts = api.count_inputs_containing("PRIVATE_RETRY");
    let public_attempts = api.count_inputs_containing("LIVE_SCOPE_RETRY");
    assert!(private_attempts > 0 && public_attempts >= 2);
    let after_retry = tokio::time::Instant::now() + Duration::from_millis(2300);
    api.invalid_response(None);
    timeout(Duration::from_secs(10), async {
        loop {
            if client
                .vault_status(&root, &narrow_scope)
                .await
                .unwrap()
                .ready
            {
                break;
            }
            sleep(Duration::from_millis(25)).await;
        }
    })
    .await
    .expect("the still-live scope must retry and recover");
    tokio::time::sleep_until(after_retry).await;
    assert!(api.count_inputs_containing("LIVE_SCOPE_RETRY") > public_attempts);
    assert_eq!(
        api.count_inputs_containing("PRIVATE_RETRY"),
        private_attempts
    );
    drop(narrow);
    daemon.kill().await.unwrap();
    daemon.wait().await.unwrap();
}

#[tokio::test]
async fn standalone_daemon_policy_prevents_wider_auto_fallback() {
    let owner_provider = ControlledProvider::start(false).await;
    let local_provider = ControlledProvider::start(false).await;
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("vault");
    std::fs::create_dir_all(root.join("Private")).unwrap();
    std::fs::write(root.join("shared.md"), "quokka visible").unwrap();
    std::fs::write(root.join("Private/one.md"), "quokka OPERATOR_PRIVATE").unwrap();
    let endpoint = isolated_endpoint(directory.path());
    let client = SemanticDaemonClient::new(endpoint.clone(), DaemonConnectPolicy::default());
    let mut daemon = daemon_command(
        &directory.path().join("semantic"),
        &endpoint,
        owner_provider.port,
        128,
        2,
        "Private/",
    )
    .spawn()
    .unwrap();
    wait_for_health(&mut daemon, &client).await;
    let mut server = StdioServer::start(
        &root,
        directory.path(),
        &endpoint,
        "auto",
        local_provider.port,
    )
    .await;
    server.attach_and_check_paths(vec!["shared.md"]).await;
    assert_eq!(
        owner_provider.count_inputs_containing("OPERATOR_PRIVATE"),
        0
    );

    daemon.kill().await.unwrap();
    daemon.wait().await.unwrap();
    let unavailable = server
        .tool("search_semantic", json!({"query":"quokka"}))
        .await;
    assert!(unavailable.get("error").is_some(), "{unavailable}");
    assert_eq!(server.embedding_info().await["source"], "daemon");
    assert!(
        local_provider.requests().is_empty(),
        "an unavailable stricter owner cannot authorize local document processing"
    );
    server.stop().await;
}

#[tokio::test]
async fn local_embedding_checkpoints_are_isolated_by_policy() {
    let api = ControlledProvider::start(false).await;
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("vault");
    std::fs::create_dir_all(root.join("Private")).unwrap();
    std::fs::write(root.join("shared.md"), "quokka LOCAL_SHARED").unwrap();
    std::fs::write(root.join("Private/one.md"), "quokka LOCAL_PRIVATE").unwrap();
    let endpoint = isolated_endpoint(directory.path());
    let mut broad = StdioServer::start(&root, directory.path(), &endpoint, "local", api.port).await;
    broad.wait_for_ready_notes("local", 2).await;
    let broad_cache = wait_for_checkpoints(directory.path(), 1)
        .await
        .pop()
        .unwrap();
    let legacy = directory
        .path()
        .join("mcp-data/vaults")
        .join(obsidian_mcp::config::vault_slug(
            &root.canonicalize().unwrap(),
        ))
        .join("embeddings/embeddings.bin");
    std::fs::create_dir_all(legacy.parent().unwrap()).unwrap();
    let mixed_cache = std::fs::read(broad_cache).unwrap();
    std::fs::write(&legacy, &mixed_cache).unwrap();

    let mut narrow = StdioServer::start_with_policy(
        &root,
        directory.path(),
        &endpoint,
        "local",
        api.port,
        ClientIndex {
            max_input_tokens: 128,
            watch: false,
            exclude_paths: "Private/",
        },
    )
    .await;
    narrow.wait_for_ready_notes("local", 1).await;
    wait_for_checkpoints(directory.path(), 2).await;
    assert_eq!(std::fs::read(&legacy).unwrap(), mixed_cache);
    assert_eq!(api.count_inputs_containing("LOCAL_SHARED"), 2);
    assert_eq!(api.count_inputs_containing("LOCAL_PRIVATE"), 1);
    narrow.stop().await;
    broad.stop().await;

    let submissions = api.requests().len();
    let mut narrow = StdioServer::start_with_policy(
        &root,
        directory.path(),
        &endpoint,
        "local",
        api.port,
        ClientIndex {
            max_input_tokens: 128,
            watch: false,
            exclude_paths: "Private/**",
        },
    )
    .await;
    narrow.wait_for_ready_notes("local", 1).await;
    assert_eq!(api.requests().len(), submissions);
    narrow.stop().await;
}

#[tokio::test]
async fn prepared_index_keys_cannot_read_notes_renamed_into_case_sensitive_exclusions() {
    let api = ControlledProvider::start(true).await;
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("vault");
    std::fs::create_dir_all(root.join("private")).unwrap();
    for (path, body) in [
        ("0-first.md", "HOLD_BEFORE_SOURCE_READ"),
        ("private/child.md", "formerly permitted directory"),
        ("case-note.md", "formerly permitted file"),
        ("z-last.md", "VISIBLE_AFTER_CASE_RENAME"),
    ] {
        std::fs::write(root.join(path), body).unwrap();
    }
    let endpoint = isolated_endpoint(directory.path());
    let client = SemanticDaemonClient::new(endpoint.clone(), DaemonConnectPolicy::default());
    let mut daemon = daemon_command(
        &directory.path().join("semantic"),
        &endpoint,
        api.port,
        128,
        1,
        "",
    )
    .env("OBSIDIAN_EMBEDDING_BATCH_SIZE", "1")
    .spawn()
    .unwrap();
    wait_for_health(&mut daemon, &client).await;
    let scope = IndexScope::new(&["Private/".into(), "CASE-NOTE.md".into()]);
    let attachment = client
        .attach_vault(&root, &scope, false, None)
        .await
        .unwrap();
    api.wait_for_input("HOLD_BEFORE_SOURCE_READ").await;
    assert_eq!(attachment.result.total_notes, Some(4));
    // The metadata contains both original spellings. The provider has blocked
    // note 0, and the coordinator prepares only one source per batch.
    for (before, after) in [("private", "Private"), ("case-note.md", "CASE-NOTE.md")] {
        std::fs::rename(root.join(before), root.join("intermediate-name")).unwrap();
        std::fs::rename(root.join("intermediate-name"), root.join(after)).unwrap();
    }
    std::fs::write(root.join("Private/child.md"), "EXCLUDED_PARENT_SPELLING").unwrap();
    std::fs::write(root.join("CASE-NOTE.md"), "EXCLUDED_FILE_SPELLING").unwrap();
    api.hold_requests(false);
    api.wait_for_input("VISIBLE_AFTER_CASE_RENAME").await;
    assert_eq!(api.count_inputs_containing("EXCLUDED_PARENT_SPELLING"), 0);
    assert_eq!(api.count_inputs_containing("EXCLUDED_FILE_SPELLING"), 0);
    assert_eq!(api.count_inputs_containing("formerly permitted"), 0);
    drop(attachment);
    daemon.kill().await.unwrap();
    daemon.wait().await.unwrap();
}

#[tokio::test]
async fn refreshing_the_portable_floor_expires_previous_attachments() {
    let api = ControlledProvider::start(false).await;
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("vault");
    std::fs::create_dir_all(root.join("Private")).unwrap();
    std::fs::create_dir_all(root.join(".obsidian-mcp")).unwrap();
    let ignore = root.join(".obsidian-mcp/ignore");
    std::fs::write(&ignore, "").unwrap();
    std::fs::write(root.join("shared.md"), "quokka visible").unwrap();
    std::fs::write(root.join("Private/one.md"), "quokka PREVIOUS_FLOOR").unwrap();
    let endpoint = isolated_endpoint(directory.path());
    let client = SemanticDaemonClient::new(endpoint.clone(), DaemonConnectPolicy::default());
    let mut daemon = spawn_daemon(&directory.path().join("semantic"), &endpoint, api.port);
    wait_for_health(&mut daemon, &client).await;
    let scope = IndexScope::default();
    let broad = client
        .attach_vault(&root, &scope, true, None)
        .await
        .unwrap();
    api.wait_for_input("PREVIOUS_FLOOR").await;

    std::fs::write(&ignore, "Private/\n").unwrap();
    let narrow = client
        .attach_vault(&root, &scope, true, None)
        .await
        .unwrap();
    assert_ne!(broad.result.scope_id, narrow.result.scope_id);
    let stale = client.search_semantic(&broad, "quokka", 10, true).await;
    assert!(
        matches!(
            stale,
            Err(obsidian_mcp::error::VaultError::DaemonRpc {
                code: ERR_VAULT_NOT_ATTACHED,
                ..
            })
        ),
        "{stale:?}"
    );
    std::fs::write(root.join("Private/one.md"), "quokka WITHDRAWN_FLOOR").unwrap();
    std::fs::write(root.join("shared.md"), "quokka CURRENT_FLOOR").unwrap();
    api.wait_for_input("CURRENT_FLOOR").await;
    assert_eq!(api.count_inputs_containing("WITHDRAWN_FLOOR"), 0);

    std::fs::write(&ignore, "[invalid-pattern").unwrap();
    assert!(
        client
            .attach_vault(&root, &scope, false, None)
            .await
            .is_err()
    );
    let stale = client.search_semantic(&narrow, "quokka", 10, true).await;
    assert!(
        matches!(
            stale,
            Err(obsidian_mcp::error::VaultError::DaemonRpc {
                code: ERR_VAULT_NOT_ATTACHED,
                ..
            })
        ),
        "{stale:?}"
    );
    drop((narrow, broad));
    daemon.kill().await.unwrap();
    daemon.wait().await.unwrap();
}

fn structured_result(response: &Value) -> &Value {
    assert!(response.get("error").is_none(), "{response}");
    &response["result"]["structuredContent"]
}

fn oversized_note() -> String {
    format!(
        "---\ntags: [retention]\n---\n# Operations\n\n## Weather\n\n{}\n\n## Recovery procedure\n\nTAIL_SIGNAL Keep the encrypted backup until an incident is closed.\n",
        "The coastal station records ordinary weather observations. ".repeat(120)
    )
}

#[tokio::test]
async fn hybrid_query_capacity_is_checked_before_empty_lexical_results() {
    let query = "missing ".repeat(200);
    let measured_tokens = tiktoken_rs::cl100k_base()
        .unwrap()
        .encode_ordinary(&query)
        .len();
    let max_input_tokens = 96;
    assert!(measured_tokens > max_input_tokens);

    for mode in ["local", "daemon"] {
        let provider = ControlledProvider::start(false).await;
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("vault");
        std::fs::create_dir(&root).unwrap();
        std::fs::write(
            root.join("policy.md"),
            "# Policy\nKeep backups for thirty days.\n",
        )
        .unwrap();
        let endpoint = isolated_endpoint(directory.path());
        let mut daemon = if mode == "daemon" {
            let client =
                SemanticDaemonClient::new(endpoint.clone(), DaemonConnectPolicy::default());
            let mut child = spawn_daemon_with_limits(
                &directory.path().join("semantic"),
                &endpoint,
                provider.port,
                max_input_tokens,
                2,
            );
            wait_for_health(&mut child, &client).await;
            Some(child)
        } else {
            None
        };
        let mut server = StdioServer::start_with_options(
            &root,
            directory.path(),
            &endpoint,
            mode,
            provider.port,
            max_input_tokens,
            true,
        )
        .await;
        let first = server
            .tool("search_semantic", json!({"query":"Policy","top_k":1}))
            .await;
        if first.get("error").is_some() {
            assert_warming(&first);
        }
        server.wait_for_ready(mode).await;
        let initial_requests = provider.requests().len();

        for lexical_prefetch in [false, true] {
            let response = server
                .tool(
                    "search_semantic",
                    json!({"query":query,"top_k":1,"lexical_prefetch":lexical_prefetch}),
                )
                .await;
            assert_eq!(response["error"]["code"], -32602, "{mode}: {response}");
            assert!(
                response["error"]["message"]
                    .as_str()
                    .unwrap()
                    .contains(&format!(
                        "{measured_tokens} tokens (maximum {max_input_tokens})"
                    )),
                "{mode}: {response}"
            );
        }
        let valid = server
            .tool(
                "search_semantic",
                json!({"query":"missing","top_k":1,"lexical_prefetch":true}),
            )
            .await;
        assert!(
            structured_result(&valid)["results"]
                .as_array()
                .unwrap()
                .is_empty(),
            "{mode}: {valid}"
        );

        let removed = server
            .tool("note_delete", json!({"path":"policy.md", "confirm":true}))
            .await;
        assert!(removed.get("error").is_none(), "{mode}: {removed}");
        for lexical_prefetch in [false, true] {
            let empty_scope = server
                .tool(
                    "search_semantic",
                    json!({"query":"missing","top_k":1,"lexical_prefetch":lexical_prefetch}),
                )
                .await;
            assert!(
                structured_result(&empty_scope)["results"]
                    .as_array()
                    .unwrap()
                    .is_empty(),
                "{mode}: {empty_scope}"
            );
            let overlong = server
                .tool(
                    "search_semantic",
                    json!({"query":query,"top_k":1,"lexical_prefetch":lexical_prefetch}),
                )
                .await;
            assert_eq!(overlong["error"]["code"], -32602, "{mode}: {overlong}");
        }
        assert_eq!(
            provider.requests().len(),
            initial_requests,
            "{mode}: invalid queries and empty semantic or lexical scopes must not reach the provider"
        );
        server.stop().await;
        if let Some(child) = daemon.as_mut() {
            child.kill().await.unwrap();
            child.wait().await.unwrap();
        }
    }
}

/// This suite is also run against both extracted native release binaries.
#[tokio::test]
async fn automatic_chunking_preserves_tail_evidence_and_revision_checked_context() {
    for mode in ["local", "daemon"] {
        let provider = ControlledProvider::start(false).await;
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("vault");
        std::fs::create_dir(&root).unwrap();
        let long_note = oversized_note();
        std::fs::write(root.join("policy.md"), &long_note).unwrap();
        std::fs::write(root.join("brief.md"), "# Brief\nA short unrelated note.\n").unwrap();
        let endpoint = isolated_endpoint(directory.path());
        let mut daemon = if mode == "daemon" {
            let client =
                SemanticDaemonClient::new(endpoint.clone(), DaemonConnectPolicy::default());
            let mut child = spawn_daemon_with_limits(
                &directory.path().join("semantic"),
                &endpoint,
                provider.port,
                96,
                2,
            );
            wait_for_health(&mut child, &client).await;
            Some(child)
        } else {
            None
        };
        let mut server = StdioServer::start_with_options(
            &root,
            directory.path(),
            &endpoint,
            mode,
            provider.port,
            96,
            true,
        )
        .await;
        let first = server
            .tool("search_semantic", json!({"query":"TAIL_SIGNAL","top_k":2}))
            .await;
        if first.get("error").is_some() {
            assert_warming(&first);
        }
        server.wait_for_ready_notes(mode, 2).await;
        let response = server
            .tool("search_semantic", json!({"query":"TAIL_SIGNAL","top_k":2}))
            .await;
        let hits = structured_result(&response)["results"].as_array().unwrap();
        assert_eq!(hits.len(), 2, "top_k counts notes: {response}");
        let policy = &hits[0];
        assert_eq!(policy["path"], "policy.md", "{response}");
        assert_eq!(policy["source_status"], "current");
        assert_eq!(policy["evidence"]["scope"], "passage");
        assert_eq!(hits[1]["evidence"]["scope"], "note");
        assert!(
            hits[1]["evidence"]["heading_path"]
                .as_array()
                .unwrap()
                .is_empty()
        );
        let start = policy["evidence"]["range"]["start_byte"].as_u64().unwrap() as usize;
        let end = policy["evidence"]["range"]["end_byte"].as_u64().unwrap() as usize;
        assert!(long_note[start..end].contains("TAIL_SIGNAL"));
        assert!(
            start > 400,
            "the selected evidence must be beyond the old prefix"
        );
        let revision = policy["source_revision"].as_str().unwrap().to_owned();
        let mut next = start;
        let mut reconstructed = String::new();
        while next < end {
            let page = server
                .tool(
                    "note_read",
                    json!({
                        "path":"policy.md", "expected_revision":revision,
                        "range":{"start_byte":next,"end_byte":end}, "max_bytes":17
                    }),
                )
                .await;
            let page = structured_result(&page);
            assert_eq!(page["source_revision"], revision);
            assert_eq!(page["returned_range"]["start_byte"], next);
            reconstructed.push_str(page["content"].as_str().unwrap());
            let returned_end = page["returned_range"]["end_byte"].as_u64().unwrap() as usize;
            assert!(returned_end > next, "a bounded page must make progress");
            next = returned_end;
        }
        assert_eq!(reconstructed, long_note[start..end]);

        let complete = server
            .tool(
                "search_semantic",
                json!({"query":"TAIL_SIGNAL","top_k":1,"include_content":true}),
            )
            .await;
        let full_hit = &structured_result(&complete)["results"][0];
        assert_eq!(full_hit["content"], long_note);
        assert!(full_hit.get("snippet").is_none());

        let tokenizer = tiktoken_rs::cl100k_base().unwrap();
        let requests = provider.requests();
        assert!(
            requests.len() > 2,
            "the note should require several bounded requests"
        );
        for request in requests {
            let inputs = request["input"].as_array().unwrap();
            assert!(inputs.len() <= 2);
            let token_count: usize = inputs
                .iter()
                .map(|text| {
                    let count = tokenizer.encode_ordinary(text.as_str().unwrap()).len();
                    assert!(count <= 96, "provider received an oversized input");
                    count
                })
                .sum();
            assert!(token_count <= 192);
        }
        std::fs::write(root.join("policy.md"), "# Revised\nA different policy.\n").unwrap();
        let stale = server
            .tool(
                "note_read",
                json!({"path":"policy.md","expected_revision":revision}),
            )
            .await;
        assert_eq!(stale["error"]["code"], -32600, "{stale}");
        assert!(stale.get("result").is_none());
        server.stop().await;
        if let Some(child) = daemon.as_mut() {
            child.kill().await.unwrap();
            child.wait().await.unwrap();
        }
    }
}

#[tokio::test]
async fn changed_source_keeps_ranked_hit_without_mixed_revision_context() {
    let provider = ControlledProvider::start(false).await;
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("vault");
    std::fs::create_dir(&root).unwrap();
    std::fs::write(root.join("note.md"), oversized_note()).unwrap();
    let endpoint = isolated_endpoint(directory.path());
    let mut server = StdioServer::start_with_options(
        &root,
        directory.path(),
        &endpoint,
        "local",
        provider.port,
        96,
        false,
    )
    .await;
    server.wait_for_ready("local").await;
    let first = server
        .tool("search_semantic", json!({"query":"TAIL_SIGNAL","top_k":1}))
        .await;
    let previous_revision = structured_result(&first)["results"][0]["source_revision"].clone();
    std::fs::write(
        root.join("note.md"),
        "# Replacement\nDifferent live content.\n",
    )
    .unwrap();
    let changed = server
        .tool(
            "search_semantic",
            json!({"query":"TAIL_SIGNAL","top_k":1,"include_content":true}),
        )
        .await;
    let hit = &structured_result(&changed)["results"][0];
    assert_eq!(hit["path"], "note.md");
    assert_eq!(hit["source_revision"], previous_revision);
    assert_eq!(hit["source_status"], "changed");
    assert_eq!(hit["context_error"]["code"], "source_changed");
    for field in ["snippet", "content", "preview"] {
        assert!(
            hit.get(field).is_none(),
            "{field} must not cross revisions: {changed}"
        );
    }
    server.stop().await;
}

async fn wait_for_checkpoints(directory: &Path, expected_scopes: usize) -> Vec<PathBuf> {
    timeout(Duration::from_secs(15), async {
        loop {
            let paths: Vec<_> = walkdir::WalkDir::new(directory)
                .into_iter()
                .map(Result::unwrap)
                .filter(|e| {
                    e.file_type().is_file()
                        && e.file_name() == "embeddings.bin"
                        && e.path()
                            .parent()
                            .and_then(Path::parent)
                            .and_then(Path::file_name)
                            == Some(std::ffi::OsStr::new("scopes"))
                })
                .map(|entry| entry.into_path())
                .collect();
            assert!(
                paths.len() <= expected_scopes,
                "more scoped checkpoints than active policies"
            );
            if paths.len() == expected_scopes {
                return paths;
            }
            sleep(Duration::from_millis(25)).await;
        }
    })
    .await
    .expect("scoped embedding checkpoints did not appear")
}

#[tokio::test]
async fn chunk_cache_reuses_unchanged_inputs_and_refreshes_only_the_edited_tail() {
    let provider = ControlledProvider::start(false).await;
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("vault");
    std::fs::create_dir(&root).unwrap();
    let original = oversized_note();
    std::fs::write(root.join("note.md"), &original).unwrap();
    let endpoint = isolated_endpoint(directory.path());
    let mut server = StdioServer::start_with_options(
        &root,
        directory.path(),
        &endpoint,
        "local",
        provider.port,
        96,
        false,
    )
    .await;
    server.wait_for_ready("local").await;
    wait_for_checkpoints(directory.path(), 1).await;
    let initial_requests = provider.requests().len();
    let initial_inputs: usize = provider
        .requests()
        .iter()
        .map(|request| request["input"].as_array().unwrap().len())
        .sum();
    assert!(
        initial_inputs > 2,
        "the note must have several embedded chunks"
    );

    let invalid = server
        .tool(
            "search_semantic",
            json!({"query":"query ".repeat(200),"top_k":1}),
        )
        .await;
    assert_eq!(invalid["error"]["code"], -32602, "{invalid}");
    assert_eq!(
        provider.requests().len(),
        initial_requests,
        "invalid input must not reach provider"
    );
    server.stop().await;

    let mut server = StdioServer::start_with_options(
        &root,
        directory.path(),
        &endpoint,
        "local",
        provider.port,
        96,
        false,
    )
    .await;
    server.wait_for_ready("local").await;
    assert_eq!(
        provider.requests().len(),
        initial_requests,
        "compatible cache must avoid inference"
    );

    let updated = original.replace(
        "until an incident is closed",
        "until the incident review is complete",
    );
    let response = server
        .tool("note_write", json!({"path":"note.md","content":updated}))
        .await;
    assert!(response.get("error").is_none(), "{response}");
    let expected_revision = obsidian_mcp::vault::source::source_revision(&updated);
    timeout(Duration::from_secs(15), async {
        loop {
            let response = server
                .tool("search_semantic", json!({"query":"TAIL_SIGNAL","top_k":1}))
                .await;
            if response.get("error").is_none() {
                let hit = &structured_result(&response)["results"][0];
                if hit["source_revision"] == expected_revision && hit["source_status"] == "current"
                {
                    break;
                }
            }
            sleep(Duration::from_millis(25)).await;
        }
    })
    .await
    .expect("edited tail did not become searchable");
    let new_note_inputs: Vec<_> = provider.requests()[initial_requests..]
        .iter()
        .flat_map(|request| request["input"].as_array().unwrap().iter())
        .filter_map(Value::as_str)
        .filter(|text| *text != "TAIL_SIGNAL")
        .map(str::to_owned)
        .collect();
    assert_eq!(
        new_note_inputs.len(),
        1,
        "unchanged chunks should reuse vectors: {new_note_inputs:?}"
    );
    assert!(new_note_inputs[0].contains("incident review is complete"));
    server.stop().await;
}
