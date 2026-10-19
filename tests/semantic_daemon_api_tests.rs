//! Exercise the daemon executable with an isolated, deterministic embedding provider.

#![cfg(all(feature = "embeddings-api", any(unix, windows)))]

use std::path::Path;
use std::process::Stdio;
use std::sync::{Arc, Mutex, atomic::AtomicBool};
use std::time::Duration;

use axum::{Json, Router, extract::State, routing::post};
use obsidian_mcp::client::semantic_daemon::{DaemonConnectPolicy, SemanticDaemonClient};
use obsidian_mcp::config::{Config, SemanticMode, ToolFilter, Transport};
use obsidian_mcp::daemon::protocol::DAEMON_API_VERSION;
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
            let vector = if text.as_str().unwrap().contains("VISIBLE") {
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
    let binary = std::env::var_os("OBSIDIAN_TEST_DAEMON_BINARY")
        .unwrap_or_else(|| env!("CARGO_BIN_EXE_obsidian-semanticd").into());
    let mut command = Command::new(binary);
    for (key, _) in std::env::vars_os() {
        if key.to_string_lossy().starts_with("OBSIDIAN_")
            || key.to_string_lossy().starts_with("OPENAI_")
        {
            command.env_remove(key);
        }
    }
    command
        .env("HOME", home)
        .env("USERPROFILE", home)
        .env("OBSIDIAN_SEMANTIC_HOME", home)
        .env("OBSIDIAN_SEMANTIC_ENDPOINT", endpoint.endpoint_string())
        .env("OBSIDIAN_SEMANTIC_MODEL", "probe")
        .env("OBSIDIAN_EMBEDDING_PROVIDER", "api")
        .env(
            "OBSIDIAN_EMBEDDING_API_BASE",
            format!("http://127.0.0.1:{api_port}/v1"),
        )
        .env("OBSIDIAN_EMBEDDING_API_KEY", "synthetic-test-key")
        .env("OBSIDIAN_EMBEDDING_API_MODEL", "probe")
        .env("OBSIDIAN_EMBEDDING_DIM", "2")
        .env("OBSIDIAN_LOG_LEVEL", "error")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .kill_on_drop(true)
        .spawn()
        .unwrap()
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

#[derive(Clone)]
struct ProviderState {
    requests: Arc<Mutex<Vec<Value>>>,
    received: Arc<Notify>,
    hold: watch::Receiver<bool>,
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

    fn hold_requests(&self, hold: bool) {
        self.hold.send(hold).unwrap();
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
) -> Json<Value> {
    state.requests.lock().unwrap().push(input.clone());
    state.received.notify_one();
    state.hold.wait_for(|hold| !*hold).await.unwrap();
    embeddings(Json(input)).await
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
        let mut command = Command::new(env!("CARGO_BIN_EXE_obsidian-mcp"));
        for (key, _) in std::env::vars_os() {
            if key.to_string_lossy().starts_with("OBSIDIAN_")
                || key.to_string_lossy().starts_with("OPENAI_")
            {
                command.env_remove(key);
            }
        }
        let mut child = command
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
            .env("OBSIDIAN_EMBEDDING_DIM", "2")
            .env("OBSIDIAN_WATCH", "true")
            .env("OBSIDIAN_LOG_LEVEL", "error")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true)
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
        let id = self.next_id;
        self.next_id += 1;
        let mut request =
            json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}).to_string();
        request.push('\n');
        self.input.write_all(request.as_bytes()).await.unwrap();
        self.input.flush().await.unwrap();
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
        timeout(Duration::from_secs(15), async {
            loop {
                let info = self.embedding_info().await;
                assert_eq!(info["source"], source, "{info}");
                if info["status"]["phase"] == "ready" {
                    assert_eq!(info["status"]["indexed_notes"], 1);
                    assert_eq!(info["status"]["pending_notes"], 0);
                    return;
                }
                sleep(Duration::from_millis(25)).await;
            }
        })
        .await
        .expect("embedding index did not become ready");
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
    std::fs::create_dir(&root).unwrap();
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
    let mut server =
        StdioServer::start(&root, directory.path(), &endpoint, mode, local_api.port).await;

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
            timeout: Duration::from_millis(500),
            retries: 0,
            ..Default::default()
        },
    );
    let runtime = SemanticRuntime {
        mode: SemanticMode::Daemon,
        daemon_client: Some(client.clone()),
        daemon_unavailable_reason: None,
        prefetch_count: 1,
        vault_ensured: Arc::new(AtomicBool::new(false)),
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
