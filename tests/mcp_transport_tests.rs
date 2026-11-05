//! Wire-level coverage through the shipped HTTP and stdio entry points.

#[path = "common/http_listening.rs"]
mod http_listening;
#[path = "common/http_rpc.rs"]
mod http_rpc;
use http_rpc::{MODERN, rpc_response};
use std::process::Stdio;
use std::time::Duration;

use reqwest::{Client, RequestBuilder, StatusCode};
use serde_json::{Value, json};
use tempfile::TempDir;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::time::timeout;

const LEGACY: &str = "2025-11-25";
const NOTE: &str = "# Transport test\nA note served through MCP.\n";

fn server_command(vault: &TempDir) -> Command {
    let mut command = http_listening::isolated_command(env!("CARGO_BIN_EXE_obsidian-mcp"));
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
        Self::start_with_vault(filter, host, temporary_vault()).await
    }

    async fn start_with_vault(filter: &str, host: &str, vault: TempDir) -> Self {
        let mut child = server_command(&vault)
            .args(["--http", "--host", host, "--port", "0"])
            .env("OBSIDIAN_TOOLS", filter)
            .env("OBSIDIAN_LOG_LEVEL", "info")
            .env("NO_COLOR", "1")
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let address = http_listening::ready(&mut child).await.unwrap();
        Self {
            child,
            vault,
            client: Client::builder()
                .no_proxy()
                .timeout(Duration::from_secs(10))
                .build()
                .unwrap(),
            url: format!("http://{address}"),
        }
    }

    fn request(&self, method: &str, params: Value, version: &str) -> RequestBuilder {
        http_rpc::request(&self.client, &self.url, method, params, version)
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
async fn simultaneous_http_children_serve_their_respective_vaults() {
    async fn check(host: &str, marker: &str) {
        let vault = temporary_vault();
        std::fs::write(vault.path().join("identity.md"), marker).unwrap();
        let server = HttpServer::start_with_vault("read", host, vault).await;
        let read = rpc_response(
            server
                .request(
                    "tools/call",
                    json!({
                        "name": "note_read", "arguments": {"path": "identity.md"}
                    }),
                    MODERN,
                )
                .send()
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(read["result"]["content"][0]["text"], marker);
        server.stop().await;
    }

    tokio::join!(
        check("127.0.0.1", "first"),
        check("127.0.0.1", "second"),
        check("::1", "third")
    );
}

#[tokio::test]
async fn http_child_exit_reports_startup_diagnostics() {
    let vault = temporary_vault();
    let mut child = server_command(&vault)
        .args(["--http"])
        .env("OBSIDIAN_HTTP_PORT", "invalid-port")
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let error = http_listening::ready(&mut child)
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("OBSIDIAN_HTTP_PORT"), "{error}");
    assert!(!child.wait().await.unwrap().success());
}

#[tokio::test]
async fn invalid_lifecycle_options_leave_the_configured_listener_running() {
    let mut server = HttpServer::start("read").await;
    let endpoint = reqwest::Url::parse(&server.url).unwrap();
    let port = endpoint.port_or_known_default().unwrap().to_string();
    for (command, flags) in [
        ("stop", &["--port", "invalid-port"][..]),
        ("restart", &["--host", "localhost"][..]),
        ("serve", &["--port"][..]),
        ("stop", &["--prot", "8080"][..]),
    ] {
        let output = http_listening::isolated_command(env!("CARGO_BIN_EXE_obsidian-mcp"))
            .arg(command)
            .args(flags)
            .env("OBSIDIAN_HTTP_PORT", &port)
            .env("OBSIDIAN_VAULT_PATH", server.vault.path())
            .output()
            .await
            .unwrap();
        assert!(!output.status.success(), "{command} {flags:?}");
        assert!(String::from_utf8_lossy(&output.stderr).contains(flags[0]));
        assert!(server.child.try_wait().unwrap().is_none());
    }
    let health = server
        .client
        .get(format!("{}/health", server.url))
        .send()
        .await
        .unwrap();
    assert_eq!(health.status(), StatusCode::OK);
    server.stop().await;
}

#[cfg(target_os = "linux")]
#[tokio::test]
async fn local_startup_preserves_native_vault_paths_from_arguments_and_environment() {
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt;

    let temp = tempfile::tempdir().unwrap();
    let data = temp.path().join("data");
    for (name, positional) in [(b"vault-\xff", true), (b"vault-\xfe", false)] {
        let root = temp.path().join(OsStr::from_bytes(name));
        std::fs::create_dir(&root).unwrap();
        let mut command = http_listening::isolated_command(env!("CARGO_BIN_EXE_obsidian-mcp"));
        if positional {
            command.arg(&root);
        } else {
            command.env("OBSIDIAN_VAULT_PATH", &root);
        }
        let mut child = command
            .args(["--http", "--host", "127.0.0.1", "--port", "0"])
            .env("OBSIDIAN_MCP_DATA", &data)
            .env("OBSIDIAN_SEMANTIC_MODE", "local")
            .env("OBSIDIAN_WATCH", "false")
            .env("NO_COLOR", "1")
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .unwrap();
        http_listening::ready(&mut child).await.unwrap();
        assert!(root.join(".obsidian-mcp/ignore").is_file());
        child.kill().await.unwrap();
    }
    assert_eq!(
        std::fs::read_dir(data.join("vaults-v2")).unwrap().count(),
        2
    );
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
        assert_eq!(response.status(), StatusCode::OK);
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

async fn exchange_stdio_messages(messages: &[Value], pipelined: bool) -> Vec<Value> {
    let vault = temporary_vault();
    let mut child = server_command(&vault)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut input = child.stdin.take().unwrap();
    let mut output = BufReader::new(child.stdout.take().unwrap()).lines();
    let mut responses = Vec::new();
    let group_size = if pipelined { messages.len() } else { 1 };
    for group in messages.chunks(group_size) {
        for message in group {
            input
                .write_all(format!("{message}\n").as_bytes())
                .await
                .unwrap();
        }
        for _ in group.iter().filter(|message| message.get("id").is_some()) {
            let line = timeout(Duration::from_secs(10), output.next_line())
                .await
                .unwrap()
                .unwrap()
                .unwrap();
            responses.push(serde_json::from_str(&line).unwrap());
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
    responses
}

fn legacy_stdio_messages() -> [Value; 4] {
    [
        json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": LEGACY, "capabilities": {}, "clientInfo": {"name": "stdio-test", "version": "1"}
        }}),
        json!({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
        json!({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
            "name": "note_read", "arguments": {"path": "note.md"}
        }}),
    ]
}

fn modern_stdio_request(id: u64, method: &str, mut params: Value) -> Value {
    params["_meta"] = json!({
        "io.modelcontextprotocol/protocolVersion": MODERN,
        "io.modelcontextprotocol/clientInfo": {"name": "stdio-test", "version": "1"},
        "io.modelcontextprotocol/clientCapabilities": {}
    });
    json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params})
}

fn stdio_response(responses: &[Value], id: u64) -> &Value {
    let response = responses
        .iter()
        .find(|response| response["id"] == id)
        .unwrap_or_else(|| panic!("missing response {id}: {responses:?}"));
    assert_eq!(response["jsonrpc"], "2.0");
    response
}

fn assert_legacy_stdio_results(responses: &[Value]) {
    for id in 1..=3 {
        let response = stdio_response(responses, id);
        assert!(response.get("error").is_none(), "{response}");
        assert!(response["result"].get("resultType").is_none());
    }
    assert_eq!(
        stdio_response(responses, 1)["result"]["protocolVersion"],
        LEGACY
    );
    assert_eq!(
        tool_names(stdio_response(responses, 2)).len(),
        obsidian_mcp::config::ALL_TOOL_NAMES.len()
    );
    assert_eq!(
        stdio_response(responses, 3)["result"]["content"][0]["text"],
        NOTE
    );
}

#[tokio::test]
async fn stdio_keeps_the_legacy_initialize_and_tool_call_flow() {
    let responses = exchange_stdio_messages(&legacy_stdio_messages(), false).await;
    assert_legacy_stdio_results(&responses);
}

async fn assert_stdio_discovery_then_legacy(pipelined: bool) {
    let mut messages = vec![modern_stdio_request(0, "server/discover", json!({}))];
    messages.extend(legacy_stdio_messages());
    let responses = exchange_stdio_messages(&messages, pipelined).await;
    assert_eq!(
        stdio_response(&responses, 0)["result"]["resultType"],
        "complete"
    );
    assert_legacy_stdio_results(&responses);
}

#[tokio::test]
async fn stdio_discovery_then_legacy_handshake_works_sequentially() {
    assert_stdio_discovery_then_legacy(false).await;
}

#[tokio::test]
async fn stdio_discovery_then_legacy_handshake_works_when_pipelined() {
    assert_stdio_discovery_then_legacy(true).await;
}

#[tokio::test]
async fn stdio_discovery_and_inline_tools_work_without_initialization() {
    let responses = exchange_stdio_messages(
        &[
            modern_stdio_request(0, "server/discover", json!({})),
            modern_stdio_request(2, "tools/list", json!({})),
            modern_stdio_request(
                3,
                "tools/call",
                json!({
                    "name": "note_read", "arguments": {"path": "note.md"}
                }),
            ),
        ],
        false,
    )
    .await;
    for response in &responses {
        assert!(response.get("error").is_none(), "{response}");
        assert_eq!(response["result"]["resultType"], "complete");
    }
    assert_eq!(
        tool_names(stdio_response(&responses, 2)).len(),
        obsidian_mcp::config::ALL_TOOL_NAMES.len()
    );
    assert_eq!(
        stdio_response(&responses, 3)["result"]["content"][0]["text"],
        NOTE
    );
}

#[tokio::test]
async fn stdio_inline_lifecycle_keeps_requiring_request_metadata() {
    let responses = exchange_stdio_messages(
        &[
            modern_stdio_request(0, "server/discover", json!({})),
            modern_stdio_request(2, "tools/list", json!({})),
            json!({"jsonrpc": "2.0", "id": 3, "method": "tools/list"}),
        ],
        false,
    )
    .await;
    assert_eq!(
        stdio_response(&responses, 0)["result"]["resultType"],
        "complete"
    );
    assert!(stdio_response(&responses, 2)["result"]["tools"].is_array());
    let rejected = stdio_response(&responses, 3);
    assert_eq!(rejected["error"]["code"], -32602);
    assert!(
        rejected["error"]["message"]
            .as_str()
            .unwrap()
            .contains("request _meta")
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

#[test]
fn foreign_http_listener_fixture() {
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;

    let Ok(bind) = std::env::var("OBSIDIAN_TEST_FOREIGN_HTTP_BIND") else {
        return;
    };
    let listener = TcpListener::bind(bind).unwrap();
    eprintln!(
        "HTTP MCP server listening addr={}",
        listener.local_addr().unwrap()
    );
    let body = json!({
        "status":"ok", "server":"obsidian-mcp", "version":env!("CARGO_PKG_VERSION")
    })
    .to_string();
    for mut stream in listener.incoming().map(Result::unwrap) {
        let reader = BufReader::new(&mut stream);
        for line in reader.lines() {
            if line.unwrap().is_empty() {
                break;
            }
        }
        let _ = write!(
            stream,
            "HTTP/1.0 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
    }
}

#[tokio::test]
async fn serve_and_restart_replace_the_selected_http_instance() {
    struct StopOnDrop(String);
    impl Drop for StopOnDrop {
        fn drop(&mut self) {
            let _ = std::process::Command::new(env!("CARGO_BIN_EXE_obsidian-mcp"))
                .args(["stop", "--host", "127.0.0.1", "--port", &self.0])
                .output();
        }
    }

    let mut initial = HttpServer::start("full").await;
    let port = reqwest::Url::parse(&initial.url)
        .unwrap()
        .port()
        .unwrap()
        .to_string();
    let _cleanup = StopOnDrop(port.clone());
    let mut other = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "foreign_http_listener_fixture", "--nocapture"])
        .env("OBSIDIAN_TEST_FOREIGN_HTTP_BIND", format!("[::1]:{port}"))
        .stderr(Stdio::piped())
        .stdout(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    let other_addr = http_listening::ready(&mut other).await.unwrap();
    let log_home = tempfile::tempdir().unwrap();
    let vaults = [temporary_vault(), temporary_vault()];
    for (cmd, vault) in ["serve", "restart"].into_iter().zip(&vaults) {
        std::fs::write(vault.path().join("identity.md"), cmd).unwrap();
        let output = http_listening::isolated_command(env!("CARGO_BIN_EXE_obsidian-mcp"))
            .args([cmd, "--host", "127.0.0.1", "--port", &port])
            .arg(vault.path())
            .env("HOME", log_home.path())
            .env("XDG_STATE_HOME", log_home.path())
            .env("LOCALAPPDATA", log_home.path())
            .env("OBSIDIAN_WATCH", "false")
            .env("OBSIDIAN_EMBEDDINGS", "false")
            .env("OBSIDIAN_SEMANTIC_MODE", "local")
            .output()
            .await
            .unwrap();
        assert!(output.status.success(), "{output:?}");
        let read = rpc_response(
            initial
                .request(
                    "tools/call",
                    json!({
                        "name": "note_read", "arguments": {"path": "identity.md"}
                    }),
                    MODERN,
                )
                .send()
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(read["result"]["content"][0]["text"], cmd);
        assert!(other.try_wait().unwrap().is_none());
        assert_eq!(
            initial
                .client
                .get(format!("http://{other_addr}/health"))
                .send()
                .await
                .unwrap()
                .status(),
            StatusCode::OK
        );
    }
    initial.child.wait().await.unwrap();
    for expected in ["stopped", "no server running"] {
        let output = http_listening::isolated_command(env!("CARGO_BIN_EXE_obsidian-mcp"))
            .args(["stop", "--host", "127.0.0.1", "--port", &port])
            .output()
            .await
            .unwrap();
        assert!(output.status.success(), "{output:?}");
        assert!(
            String::from_utf8_lossy(&output.stdout).contains(expected),
            "{output:?}"
        );
    }
    assert!(other.try_wait().unwrap().is_none());
    other.kill().await.unwrap();
}

#[tokio::test]
async fn lifecycle_commands_leave_a_foreign_health_listener_running() {
    let mut foreign = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "foreign_http_listener_fixture", "--nocapture"])
        .env("OBSIDIAN_TEST_FOREIGN_HTTP_BIND", "127.0.0.1:0")
        .stderr(Stdio::piped())
        .stdout(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    let bind = http_listening::ready(&mut foreign).await.unwrap();
    let vault = temporary_vault();
    let client = Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap();
    for (command, host) in [
        ("stop", "127.0.0.1"),
        ("serve", "127.0.0.1"),
        ("restart", "127.0.0.1"),
        ("stop", "0.0.0.0"),
    ] {
        let output = http_listening::isolated_command(env!("CARGO_BIN_EXE_obsidian-mcp"))
            .args([command, "--host", host, "--port", &bind.port().to_string()])
            .arg(vault.path())
            .env("HOME", vault.path())
            .output()
            .await
            .unwrap();
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(!output.status.success(), "{command}: {output:?}");
        assert!(
            error.contains("executable")
                || error.contains("inspect")
                || error.contains("access")
                || error.contains("exact bind address"),
            "{command}: {error}"
        );
        assert!(foreign.try_wait().unwrap().is_none());
        assert_eq!(
            client
                .get(format!("http://{bind}/health"))
                .send()
                .await
                .unwrap()
                .status(),
            StatusCode::OK
        );
    }
    foreign.kill().await.unwrap();
}

#[cfg(any(unix, windows))]
#[tokio::test]
async fn note_mutations_cannot_reach_outside_through_vault_links() {
    let server = HttpServer::start("full").await;
    let outside = tempfile::tempdir().unwrap();
    let marker = outside.path().join("secret.md");
    let missing = outside.path().join("missing.md");
    std::fs::write(&marker, "external marker").unwrap();
    let alias = server.vault.path().join("External.md");
    let dangling = server.vault.path().join("Dangling.md");
    let directory = server.vault.path().join("Linked");
    #[cfg(unix)]
    {
        use std::os::unix::fs::symlink;
        symlink(&marker, &alias).unwrap();
        symlink(&missing, &dangling).unwrap();
        symlink(outside.path(), &directory).unwrap();
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::symlink_file;
        // Junction creation is available to standard users on NTFS.
        let status = std::process::Command::new("cmd")
            .args(["/D", "/C", "mklink", "/J"])
            .arg(&directory)
            .arg(outside.path())
            .status()
            .unwrap();
        assert!(status.success());
        match symlink_file(&marker, &alias) {
            Ok(()) => symlink_file(&missing, &dangling).unwrap(),
            Err(error) if error.raw_os_error() == Some(1314) => (),
            Err(error) => panic!("cannot create symlink fixture: {error}"),
        }
    }
    let mut cases = vec![
        (
            "note_write",
            json!({"path":"Linked/secret.md","content":"outside write"}),
        ),
        (
            "note_create",
            json!({"path":"Linked/child/new.md","content":"outside create"}),
        ),
        (
            "note_move",
            json!({"from":"note.md","to":"Linked/moved.md"}),
        ),
        (
            "note_delete",
            json!({"path":"Linked/secret.md","confirm":true}),
        ),
    ];
    if alias.is_symlink() {
        cases.extend([
            ("note_read", json!({"path":"External.md"})),
            (
                "note_insert",
                json!({"path":"External.md","content":"outside append"}),
            ),
            (
                "note_write",
                json!({"path":"Dangling.md","content":"outside create"}),
            ),
        ]);
        if server.vault.path().join("external.MD").is_symlink() {
            cases.push((
                "note_write",
                json!({"path":"external.MD","content":"wrong case"}),
            ));
        }
    }
    for (name, arguments) in cases {
        let reply = rpc_response(
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
        assert!(
            reply["error"].is_object() || reply["result"]["isError"] == true,
            "{name}: {reply}"
        );
        assert!(!reply.to_string().contains("external marker"));
        assert_eq!(std::fs::read_to_string(&marker).unwrap(), "external marker");
        assert!(!missing.exists());
        assert!(!outside.path().join("child").exists());
        assert!(!outside.path().join("moved.md").exists());
    }
    assert!(server.vault.path().join("note.md").is_file());
    server.stop().await;
}
