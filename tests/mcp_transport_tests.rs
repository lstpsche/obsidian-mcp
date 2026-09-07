//! Wire-level coverage through the shipped HTTP and stdio entry points.

use std::net::TcpListener;
use std::process::Stdio;
use std::time::Duration;

use reqwest::{Client, RequestBuilder, Response, StatusCode};
use serde_json::{Value, json};
use tempfile::TempDir;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::time::{Instant, sleep, timeout};

const MODERN: &str = "2026-07-28";
const LEGACY: &str = "2025-11-25";
const NOTE: &str = "# Transport test\nA note served through MCP.\n";

fn server_command(vault: &TempDir) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_obsidian-mcp"));
    for (key, _) in std::env::vars_os() {
        if key.to_string_lossy().starts_with("OBSIDIAN_") {
            command.env_remove(key);
        }
    }
    command
        .arg(vault.path())
        .env("OBSIDIAN_WATCH", "false")
        .env("OBSIDIAN_EMBEDDINGS", "false")
        .env("OBSIDIAN_SEMANTIC_MODE", "local")
        .env("OBSIDIAN_LOG_LEVEL", "error")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .kill_on_drop(true);
    command
}

fn temporary_vault() -> TempDir {
    let vault = tempfile::tempdir().unwrap();
    std::fs::write(vault.path().join("note.md"), NOTE).unwrap();
    vault
}

struct HttpServer {
    child: Child,
    vault: TempDir,
    client: Client,
    url: String,
}

impl HttpServer {
    async fn start(filter: &str) -> Self {
        Self::start_on_host(filter, "127.0.0.1").await
    }

    async fn start_on_host(filter: &str, host: &str) -> Self {
        let vault = temporary_vault();
        let port = TcpListener::bind((host, 0))
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let child = server_command(&vault)
            .args(["--http", "--host", host, "--port", &port.to_string()])
            .env("OBSIDIAN_TOOLS", filter)
            .spawn()
            .unwrap();
        let mut server = Self {
            child,
            vault,
            client: Client::builder()
                .no_proxy()
                .timeout(Duration::from_secs(10))
                .build()
                .unwrap(),
            url: format!(
                "http://{}",
                std::net::SocketAddr::new(host.parse().unwrap(), port)
            ),
        };
        let deadline = Instant::now() + Duration::from_secs(15);
        loop {
            assert!(server.child.try_wait().unwrap().is_none(), "server exited");
            match server
                .client
                .get(format!("{}/health", server.url))
                .send()
                .await
            {
                Ok(response) => {
                    assert_eq!(response.status(), StatusCode::OK);
                    let health: Value =
                        serde_json::from_str(&response.text().await.unwrap()).unwrap();
                    assert_eq!(health["server"], "obsidian-mcp");
                    assert_eq!(health["version"], env!("CARGO_PKG_VERSION"));
                    break;
                }
                Err(error) if error.is_connect() && Instant::now() < deadline => {
                    sleep(Duration::from_millis(25)).await;
                }
                Err(error) => panic!("server health probe failed: {error}"),
            }
        }
        server
    }

    fn request(&self, method: &str, mut params: Value, version: &str) -> RequestBuilder {
        if version == MODERN {
            params["_meta"] = json!({
                "io.modelcontextprotocol/protocolVersion": MODERN,
                "io.modelcontextprotocol/clientInfo": {"name": "openai-mcp", "version": "1.0.0"},
                "io.modelcontextprotocol/clientCapabilities": {
                    "experimental": {"openai/visibility": {"enabled": true}},
                    "extensions": {"io.modelcontextprotocol/ui": {"mimeTypes": ["text/html;profile=mcp-app"]}}
                }
            });
        }
        let mut request = self
            .client
            .post(format!("{}/mcp", self.url))
            .header("Content-Type", "application/json")
            .header("Accept", "application/json, text/event-stream")
            .header("Mcp-Protocol-Version", version)
            .header("User-Agent", "openai-mcp/1.0.0");
        if version == MODERN {
            request = request.header("Mcp-Method", method);
            if method == "tools/call" {
                request = request.header("Mcp-Name", params["name"].as_str().unwrap());
            }
        }
        request.body(
            json!({"jsonrpc": "2.0", "id": "transport-test", "method": method, "params": params})
                .to_string(),
        )
    }

    async fn initialize(&self, filter: &str) -> String {
        let response = self
            .request(
                "initialize",
                json!({
                    "protocolVersion": LEGACY,
                    "capabilities": {},
                    "clientInfo": {"name": "legacy-client", "version": "1"}
                }),
                LEGACY,
            )
            .header("X-Obsidian-Tools", filter)
            .send()
            .await
            .unwrap();
        let session = response.headers()["mcp-session-id"]
            .to_str()
            .unwrap()
            .to_owned();
        let result = rpc_response(response).await;
        assert_eq!(result["result"]["protocolVersion"], LEGACY);
        let response = self
            .client
            .post(format!("{}/mcp", self.url))
            .header("Content-Type", "application/json")
            .header("Accept", "application/json, text/event-stream")
            .header("Mcp-Protocol-Version", LEGACY)
            .header("Mcp-Session-Id", &session)
            .body(json!({"jsonrpc": "2.0", "method": "notifications/initialized"}).to_string())
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::ACCEPTED);
        session
    }

    async fn stop(mut self) {
        self.child.kill().await.unwrap();
        self.child.wait().await.unwrap();
    }
}

async fn rpc_response(response: Response) -> Value {
    let status = response.status();
    let content_type = response
        .headers()
        .get("content-type")
        .map(|value| value.to_str().unwrap().to_owned());
    let body = response.text().await.unwrap();
    assert_eq!(status, StatusCode::OK, "{body}");
    let value = match content_type.as_deref() {
        Some("application/json") => serde_json::from_str::<Value>(&body).unwrap(),
        Some("text/event-stream") => {
            let messages: Vec<Value> = body
                .lines()
                .filter_map(|line| line.strip_prefix("data: "))
                .filter(|data| !data.is_empty())
                .map(|data| serde_json::from_str(data).unwrap())
                .collect();
            assert_eq!(messages.len(), 1, "{body}");
            messages.into_iter().next().unwrap()
        }
        other => panic!("unexpected content type {other:?}: {body}"),
    };
    assert_eq!(value["id"], "transport-test");
    value
}

fn tool_names(response: &Value) -> Vec<&str> {
    let mut names: Vec<_> = response["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|tool| tool["name"].as_str().unwrap())
        .collect();
    names.sort_unstable();
    names
}

#[tokio::test]
async fn discovery_and_stateless_tools_work_without_initialization() {
    let server = HttpServer::start("full").await;
    let response = server
        .request("server/discover", json!({}), MODERN)
        .send()
        .await
        .unwrap();
    assert!(!response.headers().contains_key("mcp-session-id"));
    let discovery = rpc_response(response).await;
    let result = &discovery["result"];
    assert_eq!(result["resultType"], "complete");
    assert!(
        result["supportedVersions"]
            .as_array()
            .unwrap()
            .contains(&json!(MODERN))
    );
    assert!(
        result["supportedVersions"]
            .as_array()
            .unwrap()
            .contains(&json!(LEGACY))
    );
    assert!(result["capabilities"]["tools"].is_object());
    assert_eq!(
        result["_meta"]["io.modelcontextprotocol/serverInfo"]["name"],
        "obsidian-mcp"
    );

    let listing = rpc_response(
        server
            .request("tools/list", json!({}), MODERN)
            .send()
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(listing["result"]["resultType"], "complete");
    assert_eq!(
        tool_names(&listing).len(),
        obsidian_mcp::config::ALL_TOOL_NAMES.len()
    );
    let created = rpc_response(server.request("tools/call", json!({
        "name": "note_create", "arguments": {"path": "created.md", "content": "Created over HTTP"}
    }), MODERN).send().await.unwrap()).await;
    assert_eq!(created["result"]["resultType"], "complete");
    assert_ne!(created["result"]["isError"], true);
    let read = rpc_response(
        server
            .request(
                "tools/call",
                json!({
                    "name": "note_read", "arguments": {"path": "created.md"}
                }),
                MODERN,
            )
            .send()
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(read["result"]["content"][0]["text"], "Created over HTTP");
    assert_eq!(
        std::fs::read_to_string(server.vault.path().join("created.md")).unwrap(),
        "Created over HTTP"
    );
    server.stop().await;
}

#[tokio::test]
async fn legacy_http_preserves_sessions_and_result_shape() {
    let server = HttpServer::start("full").await;
    let session = server.initialize("note_read").await;
    let listing = rpc_response(
        server
            .request("tools/list", json!({}), LEGACY)
            .header("Mcp-Session-Id", &session)
            .send()
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(tool_names(&listing), ["note_read"]);
    assert!(listing["result"].get("resultType").is_none());
    let read = rpc_response(
        server
            .request(
                "tools/call",
                json!({
                    "name": "note_read", "arguments": {"path": "note.md"}
                }),
                LEGACY,
            )
            .header("Mcp-Session-Id", &session)
            .send()
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(read["result"]["content"][0]["text"], NOTE);
    assert!(read["result"].get("resultType").is_none());
    let denied = rpc_response(
        server
            .request(
                "tools/call",
                json!({
                    "name": "note_delete", "arguments": {"path": "note.md", "confirm": true}
                }),
                LEGACY,
            )
            .header("Mcp-Session-Id", &session)
            .header("X-Obsidian-Tools", "full")
            .send()
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(denied["error"]["code"], -32602);
    assert_eq!(
        std::fs::read_to_string(server.vault.path().join("note.md")).unwrap(),
        NOTE
    );
    server.stop().await;
}

#[tokio::test]
async fn stateless_filters_are_isolated_and_cannot_enable_server_disabled_tools() {
    let server = HttpServer::start("!note_delete").await;
    let (restricted, full) = tokio::join!(
        server
            .request("tools/list", json!({}), MODERN)
            .header("X-Obsidian-Tools", "note_read")
            .send(),
        server
            .request("tools/list", json!({}), MODERN)
            .header("X-Obsidian-Tools", "full")
            .send()
    );
    assert_eq!(
        tool_names(&rpc_response(restricted.unwrap()).await),
        ["note_read"]
    );
    let full = rpc_response(full.unwrap()).await;
    assert!(!tool_names(&full).contains(&"note_delete"));
    assert!(tool_names(&full).contains(&"note_create"));
    for (filter, name, arguments) in [
        (
            "full",
            "note_delete",
            json!({"path": "note.md", "confirm": true}),
        ),
        (
            "note_read",
            "note_create",
            json!({"path": "forbidden.md", "content": "no"}),
        ),
    ] {
        let response = server
            .request(
                "tools/call",
                json!({"name": name, "arguments": arguments}),
                MODERN,
            )
            .header("X-Obsidian-Tools", filter)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let denied: Value = serde_json::from_str(&response.text().await.unwrap()).unwrap();
        assert_eq!(denied["error"]["code"], -32602);
    }
    let response = server
        .request(
            "tools/call",
            json!({
                "name": "note_create", "arguments": {"path": "forbidden.md", "content": "no"}
            }),
            MODERN,
        )
        .header("X-Obsidian-Tools", "invalid-profile")
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert!(response.text().await.unwrap().contains("Unknown profile"));
    assert!(server.vault.path().join("note.md").exists());
    assert!(!server.vault.path().join("forbidden.md").exists());
    server.stop().await;
}

#[tokio::test]
async fn http_rejects_invalid_routing_headers_and_untrusted_hosts() {
    let server = HttpServer::start("full").await;
    let mut request = server
        .request("tools/list", json!({}), MODERN)
        .build()
        .unwrap();
    request
        .headers_mut()
        .insert("Mcp-Method", "tools/call".parse().unwrap());
    let response = server.client.execute(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let response = server
        .request("server/discover", json!({}), MODERN)
        .header("Host", "untrusted.example")
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::FORBIDDEN);
    server.stop().await;
}

#[tokio::test]
async fn stdio_keeps_the_legacy_initialize_and_tool_call_flow() {
    let vault = temporary_vault();
    let mut child = server_command(&vault)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut input = child.stdin.take().unwrap();
    let mut output = BufReader::new(child.stdout.take().unwrap()).lines();
    for message in [
        json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": LEGACY, "capabilities": {}, "clientInfo": {"name": "stdio-test", "version": "1"}
        }}),
        json!({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        json!({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
            "name": "note_read", "arguments": {"path": "note.md"}
        }}),
    ] {
        input
            .write_all(format!("{message}\n").as_bytes())
            .await
            .unwrap();
        if message.get("id").is_some() {
            let line = timeout(Duration::from_secs(10), output.next_line())
                .await
                .unwrap()
                .unwrap()
                .unwrap();
            let response: Value = serde_json::from_str(&line).unwrap();
            assert_eq!(response["id"], message["id"]);
            assert!(response.get("error").is_none(), "{response}");
            assert!(response["result"].get("resultType").is_none());
            if message["id"] == 1 {
                assert_eq!(response["result"]["protocolVersion"], LEGACY);
            } else {
                assert_eq!(response["result"]["content"][0]["text"], NOTE);
            }
        }
    }
    drop(input);
    assert!(
        timeout(Duration::from_secs(10), child.wait())
            .await
            .unwrap()
            .unwrap()
            .success()
    );
}

fn validate_output(tool: &Value, result: &Value) {
    assert_ne!(result["isError"], true, "{result}");
    let structured = result.get("structuredContent").expect("structured result");
    let validator = jsonschema::validator_for(&tool["outputSchema"]).unwrap();
    let errors: Vec<_> = validator
        .iter_errors(structured)
        .map(|e| e.to_string())
        .collect();
    assert!(
        errors.is_empty(),
        "{}: {errors:?}; {structured}",
        tool["name"]
    );
    assert!(
        !validator.is_valid(&json!({})),
        "{} must describe required result fields",
        tool["name"]
    );

    let text = result["content"][0]["text"].as_str().unwrap();
    if tool["name"] == "frontmatter" && structured["action"] == "get" {
        assert_eq!(
            serde_json::from_str::<Value>(text).unwrap(),
            structured["frontmatter"]
        );
    } else if let Some(message) = structured.get("message") {
        assert_eq!(text, message.as_str().unwrap());
    } else if let Some(content) = structured.get("content") {
        assert_eq!(text, content.as_str().unwrap());
    } else if let Some(results) = structured.get("results") {
        assert_eq!(serde_json::from_str::<Value>(text).unwrap(), *results);
    } else {
        assert_eq!(serde_json::from_str::<Value>(text).unwrap(), *structured);
    }
}

#[tokio::test]
async fn advertised_schemas_match_results_and_annotations_over_http() {
    let server = HttpServer::start("full").await;
    std::fs::create_dir_all(server.vault.path().join(".obsidian")).unwrap();
    std::fs::write(
        server.vault.path().join(".obsidian/daily-notes.json"),
        r#"{"format":"YYYY-MM-DD","folder":"Daily"}"#,
    )
    .unwrap();
    let listing = rpc_response(
        server
            .request("tools/list", json!({}), MODERN)
            .send()
            .await
            .unwrap(),
    )
    .await;
    let tools = listing["result"]["tools"].as_array().unwrap();
    assert_eq!(tools.len(), 19);
    let readonly = [
        "vault_list",
        "vault_info",
        "note_read",
        "note_read_many",
        "note_inspect",
        "search_text",
        "search_regex",
        "search_metadata",
        "search_semantic",
        "wikilinks",
    ];
    let destructive = [
        "note_write",
        "note_patch",
        "note_delete",
        "note_move",
        "frontmatter",
    ];
    for tool in tools {
        let name = tool["name"].as_str().unwrap();
        assert_eq!(tool["outputSchema"]["type"], "object", "{name}");
        jsonschema::meta::validate(&tool["inputSchema"]).unwrap();
        jsonschema::meta::validate(&tool["outputSchema"]).unwrap();
        jsonschema::validator_for(&tool["inputSchema"]).unwrap();
        jsonschema::validator_for(&tool["outputSchema"]).unwrap();
        assert_eq!(
            tool["annotations"]["readOnlyHint"],
            readonly.contains(&name),
            "{name}"
        );
        assert_eq!(
            tool["annotations"]["destructiveHint"],
            destructive.contains(&name),
            "{name}"
        );
        assert_eq!(
            tool["annotations"]["idempotentHint"],
            readonly.contains(&name) || name == "note_write",
            "{name}"
        );
        assert_eq!(
            tool["annotations"]["openWorldHint"],
            name == "search_semantic",
            "{name}"
        );
    }
    let mut covered = std::collections::HashSet::new();
    for (name, arguments) in [
        ("vault_info", json!({})),
        ("vault_list", json!({"glob":"absent*"})),
        ("vault_list", json!({})),
        ("vault_list", json!({"include_metadata":true})),
        ("vault_list", json!({"format":"tree"})),
        ("note_read", json!({"path":"note.md"})),
        ("note_read_many", json!({"paths":["note.md"]})),
        (
            "note_create",
            json!({"path":"schema.md","content":"# Schema\nhello #schema [[note]]\n"}),
        ),
        ("note_inspect", json!({"path":"schema.md"})),
        ("note_inspect", json!({"path":"schema.md","view":"targets"})),
        ("frontmatter", json!({"action":"get","path":"schema.md"})),
        (
            "frontmatter",
            json!({"action":"set","path":"schema.md","key":"value","value":null}),
        ),
        (
            "frontmatter",
            json!({"action":"set","path":"schema.md","key":"tags","value":["schema"]}),
        ),
        ("frontmatter", json!({"action":"GET","path":"schema.md"})),
        (
            "frontmatter",
            json!({"action":"remove","path":"schema.md","key":"value"}),
        ),
        ("search_text", json!({"query":"Schema"})),
        ("search_text", json!({"query":"doesnotexist"})),
        ("search_regex", json!({"pattern":"Schema"})),
        ("search_metadata", json!({"type":"tag","tag":"schema"})),
        (
            "search_metadata",
            json!({"type":"frontmatter","field":"tags","value":"schema"}),
        ),
        ("wikilinks", json!({"query":"backlinks","path":"note.md"})),
        ("wikilinks", json!({"query":"outgoing","path":"schema.md"})),
        ("wikilinks", json!({"query":"broken"})),
        ("wikilinks", json!({"query":"orphans"})),
        ("periodic", json!({"action":"list","period":"daily"})),
        (
            "periodic",
            json!({"action":"create","period":"daily","date":"2026-01-01","content":"Daily content"}),
        ),
        (
            "periodic",
            json!({"action":"get","period":"daily","date":"2026-01-01"}),
        ),
        ("periodic", json!({"action":"list","period":"daily"})),
        (
            "note_write",
            json!({"path":"schema.md","content":"# Schema\nbody\n"}),
        ),
        ("note_insert", json!({"path":"schema.md","content":"more"})),
        (
            "note_patch",
            json!({"path":"schema.md","operation":"append","target_type":"heading","target":"Schema","content":"patched"}),
        ),
        ("note_move", json!({"from":"schema.md","to":"moved.md"})),
        ("note_delete", json!({"path":"moved.md","confirm":true})),
    ] {
        let response = rpc_response(
            server
                .request(
                    "tools/call",
                    json!({"name":name,"arguments":arguments}),
                    MODERN,
                )
                .send()
                .await
                .unwrap(),
        )
        .await;
        assert!(response.get("error").is_none(), "{name}: {response}");
        let tool = tools.iter().find(|tool| tool["name"] == name).unwrap();
        validate_output(tool, &response["result"]);
        covered.insert(name);
    }
    assert_eq!(covered.len(), 17); // Desktop launch and semantic inference require separate fixtures.
    for (name, arguments) in [
        ("note_read", json!({"path":"missing.md"})),
        (
            "frontmatter",
            json!({"path":"note.md","action":"set","key":"x"}),
        ),
        ("search_regex", json!({"pattern":"["})),
        ("search_semantic", json!({"query":"Schema"})),
        ("open_in_obsidian", json!({"path":"../../outside.md"})),
    ] {
        let response = server
            .request(
                "tools/call",
                json!({"name":name,"arguments":arguments}),
                MODERN,
            )
            .send()
            .await
            .unwrap();
        assert!(matches!(response.status().as_u16(), 200 | 400));
        let response: Value = serde_json::from_str(&response.text().await.unwrap()).unwrap();
        assert!(
            response.get("error").is_some() || response["result"]["isError"] == true,
            "{name}: {response}"
        );
        assert!(
            response["result"].get("structuredContent").is_none(),
            "errors must not masquerade as successful output"
        );
    }
    server.stop().await;
}

#[tokio::test]
async fn zero_search_limits_return_structured_empty_results() {
    let server = HttpServer::start("full").await;
    for options in [json!({"fuzzy": true}), json!({"fields": ["body"]})] {
        let mut arguments = options;
        arguments["query"] = json!("Transport");
        arguments["max_results"] = json!(0);
        let response = server
            .request(
                "tools/call",
                json!({"name":"search_text", "arguments":arguments}),
                MODERN,
            )
            .send()
            .await
            .unwrap();
        let body = response.text().await.unwrap();
        assert!(body.contains("structuredContent"), "{body}");
        assert!(body.contains("\"results\":[]"), "{body}");
    }
}

#[tokio::test]
async fn unknown_daemon_argument_does_not_start_a_runtime() {
    let directory = tempfile::tempdir().unwrap();
    let home = directory.path().join("semantic");
    let mut command = Command::new(env!("CARGO_BIN_EXE_obsidian-semanticd"));
    command
        .arg("--build-info")
        .env("OBSIDIAN_SEMANTIC_HOME", &home)
        .env_remove("OBSIDIAN_SEMANTIC_ENDPOINT")
        .kill_on_drop(true);
    let output = timeout(Duration::from_secs(5), command.output())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("unknown argument"));
    assert!(!home.exists());
}

#[tokio::test]
async fn stop_uses_the_configured_ipv6_host_from_cli_and_environment() {
    for use_env in [false, true] {
        let mut server = HttpServer::start_on_host("full", "::1").await;
        let port = reqwest::Url::parse(&server.url).unwrap().port().unwrap();
        let mut command = Command::new(env!("CARGO_BIN_EXE_obsidian-mcp"));
        command
            .args(["stop", "--port", &port.to_string()])
            .kill_on_drop(true);
        if use_env {
            command.env("OBSIDIAN_HTTP_HOST", "::1");
        } else {
            command.args(["--host", "::1"]);
        }
        let (_, stopped) = timeout(Duration::from_secs(60), async {
            tokio::join!(
                async {
                    let output = command.output().await.unwrap();
                    assert!(
                        output.status.success(),
                        "stop failed: stdout={} stderr={}",
                        String::from_utf8_lossy(&output.stdout),
                        String::from_utf8_lossy(&output.stderr)
                    );
                    assert!(
                        String::from_utf8_lossy(&output.stdout).contains("stopped"),
                        "{}",
                        String::from_utf8_lossy(&output.stdout)
                    );
                },
                server.child.wait()
            )
        })
        .await
        .unwrap();
        stopped.unwrap();
        assert!(
            server
                .client
                .get(format!("{}/health", server.url))
                .send()
                .await
                .is_err()
        );
    }
}
