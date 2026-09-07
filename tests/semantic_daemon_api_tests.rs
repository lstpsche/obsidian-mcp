//! Exercise the daemon executable with an isolated, deterministic embedding provider.

#![cfg(all(feature = "embeddings-api", any(unix, windows)))]

use std::path::Path;
use std::process::Stdio;
use std::sync::{Arc, atomic::AtomicBool};
use std::time::Duration;

use axum::{Json, Router, routing::post};
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
use tokio::process::{Child, Command};
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
        if key.to_string_lossy().starts_with("OBSIDIAN_") {
            command.env_remove(key);
        }
    }
    command
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
    #[cfg(unix)]
    let endpoint = IpcEndpoint::UnixSocket(directory.path().join("daemon.sock"));
    #[cfg(windows)]
    let endpoint = IpcEndpoint::NamedPipe(format!(
        r"\\.\pipe\obsidian-api-test-{}-{}",
        std::process::id(),
        directory.path().file_name().unwrap().to_string_lossy()
    ));
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
