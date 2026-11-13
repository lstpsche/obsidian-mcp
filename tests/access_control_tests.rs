//! Wire-level authorization coverage through the shipped server entry points.

#[cfg(feature = "embeddings-api")]
#[path = "common/embedding_profiles.rs"]
mod embedding_profiles;
#[path = "common/http_listening.rs"]
mod http_listening;
#[path = "common/http_rpc.rs"]
mod http_rpc;

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use obsidian_mcp::access::clients::{ClientPolicy, Deny, Grant, Role};
use obsidian_mcp::access::store::{AccessStore, read_token_file};
use reqwest::{Client, RequestBuilder, StatusCode};
use serde_json::{Value, json};
use tempfile::TempDir;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::time::timeout;

fn server_binary() -> std::ffi::OsString {
    std::env::var_os("OBSIDIAN_TEST_MCP_BINARY")
        .unwrap_or_else(|| env!("CARGO_BIN_EXE_obsidian-mcp").into())
}

const LEGACY: &str = "2025-11-25";
const SHARED: &str =
    "---\ntags: [shared]\nkind: budget\n---\n# Shared budget\nA budget for the household.\n";
const PRIVATE: &str = "# Private memory\nprivate-source-marker\n";

struct AccessFixture {
    _directory: TempDir,
    vault: PathBuf,
    access_file: PathBuf,
    store: AccessStore,
    tokens: BTreeMap<String, String>,
    credentials: BTreeMap<String, String>,
    token_files: BTreeMap<String, PathBuf>,
}

impl AccessFixture {
    fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let vault = directory.path().join("vault");
        for path in [
            "Shared/Empty",
            "Private",
            "Personal",
            ".obsidian/plugins/sample",
            ".git/hooks",
        ] {
            std::fs::create_dir_all(vault.join(path)).unwrap();
        }
        for (path, content) in [
            ("Shared/budget.md", SHARED),
            ("Shared/reference.md", "# Reference\nVisible reference.\n"),
            (
                "Shared/graph.md",
                "# Graph\n[[secret]] and [[reference]] are literal readable links.\n",
            ),
            ("Shared/image.txt", "attachment-marker"),
            ("Private/memory.md", PRIVATE),
            (
                "Private/template.md",
                "# Template\nprivate-template-marker\n",
            ),
            ("Personal/owner.md", "# Owner\nowner-source-marker\n"),
            (".obsidian/plugins/sample/main.js", "owner-plugin-marker"),
            (".git/hooks/pre-commit", "owner-hook-marker"),
            (
                ".obsidian/daily-notes.json",
                r#"{"format":"YYYY-MM-DD","folder":"Shared/Daily","template":"Private/template"}"#,
            ),
        ] {
            std::fs::write(vault.join(path), content).unwrap();
        }
        let access_file = directory.path().join("access/access.json");
        let data = directory.path().join("data");
        let store = AccessStore::initialize(&access_file, &vault, &data).unwrap();
        store
            .update(None, |policy| {
                policy.clients.insert(
                    "editor".into(),
                    ClientPolicy {
                        grants: vec![Grant::directory("root", "", Role::Editor)],
                        denies: vec![Deny::directory("private", "Private/")],
                        ..Default::default()
                    },
                );
                policy.clients.insert(
                    "assistant".into(),
                    ClientPolicy {
                        grants: vec![
                            Grant::directory("shared", "Shared/", Role::Editor),
                            Grant::directory("private", "Private/", Role::Editor),
                        ],
                        ..Default::default()
                    },
                );
                policy.clients.insert(
                    "reader".into(),
                    ClientPolicy {
                        grants: vec![Grant::directory("shared", "Shared/", Role::Reader)],
                        ..Default::default()
                    },
                );
                Ok(())
            })
            .unwrap();
        let mut tokens = BTreeMap::new();
        let mut credentials = BTreeMap::new();
        let mut token_files = BTreeMap::new();
        for client in ["editor", "assistant", "reader"] {
            let file = directory.path().join(format!("access/{client}.token"));
            let receipt = store.issue_token(client, &file, None).unwrap();
            tokens.insert(
                client.to_owned(),
                read_token_file(&file).unwrap().as_str().to_owned(),
            );
            credentials.insert(client.to_owned(), receipt.credential_id);
            token_files.insert(client.to_owned(), file);
        }
        Self {
            _directory: directory,
            vault,
            access_file,
            store,
            tokens,
            credentials,
            token_files,
        }
    }

    fn command(&self, authenticated: bool) -> Command {
        let mut command = http_listening::isolated_command(server_binary());
        command
            .arg(&self.vault)
            .env("OBSIDIAN_WATCH", "false")
            .env("OBSIDIAN_EMBEDDINGS", "false")
            .env("OBSIDIAN_SEMANTIC_MODE", "local")
            .env("OBSIDIAN_LOG_LEVEL", "info")
            .env("TOKIO_WORKER_THREADS", "1")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        if authenticated {
            command.arg("--access-file").arg(&self.access_file);
        }
        command
    }

    fn revoke(&self, client: &str) {
        let credential = &self.credentials[client];
        self.store
            .update(None, |policy| {
                policy
                    .clients
                    .get_mut(client)
                    .unwrap()
                    .credentials
                    .iter_mut()
                    .find(|entry| entry.id == *credential)
                    .unwrap()
                    .revoked = true;
                Ok(())
            })
            .unwrap();
    }
}

struct Server {
    child: Child,
    fixture: AccessFixture,
    client: Client,
    url: String,
}

impl Server {
    async fn start() -> Self {
        Self::start_fixture(AccessFixture::new(), true).await
    }

    async fn start_fixture(fixture: AccessFixture, authenticated: bool) -> Self {
        let command = fixture.command(authenticated);
        Self::start_command(fixture, command).await
    }

    async fn start_command(fixture: AccessFixture, mut command: Command) -> Self {
        let mut child = command
            .args(["--http", "--host", "127.0.0.1", "--port", "0"])
            .spawn()
            .unwrap();
        let address = http_listening::ready(&mut child).await.unwrap();
        Self {
            child,
            fixture,
            client: Client::builder()
                .no_proxy()
                .timeout(Duration::from_secs(10))
                .build()
                .unwrap(),
            url: format!("http://{address}"),
        }
    }

    fn request(&self, client: &str, method: &str, params: Value, version: &str) -> RequestBuilder {
        self.request_token(&self.fixture.tokens[client], method, params, version)
    }

    fn request_token(
        &self,
        token: &str,
        method: &str,
        params: Value,
        version: &str,
    ) -> RequestBuilder {
        http_rpc::request(&self.client, &self.url, method, params, version).bearer_auth(token)
    }

    async fn call(&self, client: &str, name: &str, arguments: Value) -> Value {
        http_rpc::rpc_response(
            self.request(
                client,
                "tools/call",
                json!({"name": name, "arguments": arguments}),
                http_rpc::MODERN,
            )
            .send()
            .await
            .unwrap(),
        )
        .await
    }

    async fn initialize(&self, client: &str) -> String {
        let response = self
            .request(
                client,
                "initialize",
                json!({
                    "protocolVersion": LEGACY,
                    "capabilities": {},
                    "clientInfo": {"name": "access-client", "version": "1"}
                }),
                LEGACY,
            )
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let session = response.headers()["mcp-session-id"]
            .to_str()
            .unwrap()
            .to_owned();
        let initialized = http_rpc::rpc_response(response).await;
        assert_eq!(initialized["result"]["protocolVersion"], LEGACY);
        let response = self
            .notification(Some(&self.fixture.tokens[client]), Some(&session))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::ACCEPTED);
        session
    }

    fn notification(&self, token: Option<&str>, session: Option<&str>) -> RequestBuilder {
        let mut request = self
            .client
            .post(format!("{}/mcp", self.url))
            .header("Content-Type", "application/json")
            .header("Accept", "application/json, text/event-stream")
            .header("Mcp-Protocol-Version", LEGACY)
            .body(json!({"jsonrpc":"2.0","method":"notifications/initialized"}).to_string());
        if let Some(token) = token {
            request = request.bearer_auth(token);
        }
        if let Some(session) = session {
            request = request.header("Mcp-Session-Id", session);
        }
        request
    }

    async fn stop(mut self) {
        self.child.kill().await.unwrap();
        self.child.wait().await.unwrap();
    }
}

fn data(response: &Value) -> &Value {
    assert!(response.get("error").is_none(), "{response}");
    assert_ne!(response["result"]["isError"], true, "{response}");
    response["result"]
        .get("structuredContent")
        .expect("structured tool result")
}

fn assert_rejected(response: &Value) {
    assert!(
        response.get("error").is_some() || response["result"]["isError"] == true,
        "unexpected success: {response}"
    );
    for marker in [
        "private-source-marker",
        "owner-source-marker",
        "owner-plugin-marker",
        "outside-private-target-marker",
        "outside-private-content-marker",
    ] {
        assert!(!response.to_string().contains(marker), "{response}");
    }
}

fn assert_not_found(response: &Value) {
    assert_eq!(response["error"]["code"], -32002, "{response}");
    assert_rejected(response);
}

#[tokio::test]
async fn an_unselected_profile_does_not_change_ordinary_http_access() {
    let fixture = AccessFixture::new();
    std::fs::write(&fixture.access_file, "malformed unselected policy").unwrap();
    let server = Server::start_fixture(fixture, false).await;
    for (path, marker) in [
        ("Private/memory.md", "private-source-marker"),
        (".obsidian/plugins/sample/main.js", "owner-plugin-marker"),
    ] {
        let result = http_rpc::rpc_response(
            http_rpc::request(
                &server.client,
                &server.url,
                "tools/call",
                json!({"name":"note_read","arguments":{"path":path}}),
                http_rpc::MODERN,
            )
            .send()
            .await
            .unwrap(),
        )
        .await;
        assert!(data(&result)["content"].as_str().unwrap().contains(marker));
    }
    server.stop().await;
}

#[tokio::test]
async fn every_http_lifecycle_requires_one_current_credential() {
    let server = Server::start().await;
    for method in ["server/discover", "tools/list"] {
        let response = http_rpc::request(
            &server.client,
            &server.url,
            method,
            json!({}),
            http_rpc::MODERN,
        )
        .send()
        .await
        .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(response.headers()["cache-control"], "no-store");
    }
    for token in ["", "not-an-issued-token"] {
        let response = server
            .request_token(token, "tools/list", json!({}), http_rpc::MODERN)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }
    let response = server
        .request("reader", "tools/list", json!({}), http_rpc::MODERN)
        .header(
            "Authorization",
            format!("Bearer {}", server.fixture.tokens["editor"]),
        )
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    let response = http_rpc::request(
        &server.client,
        &server.url,
        "tools/list",
        json!({}),
        http_rpc::MODERN,
    )
    .header(
        "Authorization",
        format!("bEaReR {}", server.fixture.tokens["reader"]),
    )
    .send()
    .await
    .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["cache-control"], "no-store");
    let catalog = http_rpc::rpc_response(response).await;
    assert_eq!(catalog["result"]["ttlMs"], 0);
    assert_eq!(catalog["result"]["cacheScope"], "private");
    let names: Vec<_> = catalog["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|tool| tool["name"].as_str().unwrap())
        .collect();
    assert!(names.contains(&"frontmatter"));
    assert!(names.contains(&"periodic"));
    assert!(!names.contains(&"note_write"));
    assert!(!names.contains(&"open_in_obsidian"));
    let health = server
        .client
        .get(format!("{}/health", server.url))
        .send()
        .await
        .unwrap();
    let health: Value = serde_json::from_str(&health.text().await.unwrap()).unwrap();
    for field in [
        "notes",
        "vault_path",
        "mcp_data_dir",
        "embeddings_total_notes",
        "embeddings_error",
    ] {
        assert!(
            health.get(field).is_none(),
            "health disclosed {field}: {health}"
        );
    }
    for origin in ["https://untrusted.example", "null"] {
        let response = server
            .request("reader", "tools/list", json!({}), http_rpc::MODERN)
            .header("Origin", origin)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
    }
    server.stop().await;
}

#[tokio::test]
async fn mixed_tools_and_direct_calls_preserve_reader_authority() {
    let server = Server::start().await;
    for (tool, args) in [
        ("note_read", json!({"path":"Shared/budget.md"})),
        ("note_inspect", json!({"path":"Shared/budget.md"})),
        (
            "frontmatter",
            json!({"action":"get","path":"Shared/budget.md"}),
        ),
        ("periodic", json!({"action":"list","period":"daily"})),
    ] {
        data(&server.call("reader", tool, args).await);
    }
    for (tool, args) in [
        (
            "note_create",
            json!({"path":"Shared/new.md","content":"changed"}),
        ),
        (
            "note_write",
            json!({"path":"Shared/budget.md","content":"changed"}),
        ),
        (
            "note_insert",
            json!({"path":"Shared/budget.md","content":"changed"}),
        ),
        (
            "note_patch",
            json!({"path":"Shared/budget.md","operation":"append","target_type":"heading","target":"Shared budget","content":"changed"}),
        ),
        (
            "note_delete",
            json!({"path":"Shared/budget.md","confirm":true}),
        ),
        (
            "note_move",
            json!({"from":"Shared/budget.md","to":"Shared/moved.md"}),
        ),
        (
            "frontmatter",
            json!({"action":"set","path":"Shared/budget.md","key":"kind","value":"changed"}),
        ),
        (
            "frontmatter",
            json!({"action":"remove","path":"Shared/budget.md","key":"kind"}),
        ),
        (
            "periodic",
            json!({"action":"create","period":"daily","date":"2026-11-13","content":"changed"}),
        ),
    ] {
        assert_rejected(&server.call("reader", tool, args).await);
    }
    assert_eq!(
        std::fs::read_to_string(server.fixture.vault.join("Shared/budget.md")).unwrap(),
        SHARED
    );
    assert!(!server.fixture.vault.join("Shared/new.md").exists());
    assert!(!server.fixture.vault.join("Shared/Daily").exists());
    let response = server.request("reader", "tools/call", json!({"name":"note_write","arguments":{"path":"Shared/budget.md","content":"changed"}}), http_rpc::MODERN)
        .header("X-Obsidian-Tools", "full").send().await.unwrap();
    assert_rejected(&http_rpc::rpc_response(response).await);
    server.stop().await;
}

#[tokio::test]
async fn editors_share_permitted_mutations_and_cannot_cross_denials() {
    let server = Server::start().await;
    for client in ["editor", "assistant"] {
        data(&server.call(client, "note_create", json!({"path":format!("Shared/{client}.md"),"content":format!("created by {client}")})).await);
    }
    let response = server
        .call(
            "assistant",
            "note_insert",
            json!({"path":"Shared/editor.md","content":" plus assistant"}),
        )
        .await;
    data(&response);
    let read = server
        .call("reader", "note_read", json!({"path":"Shared/editor.md"}))
        .await;
    assert!(
        data(&read)["content"]
            .as_str()
            .unwrap()
            .contains("plus assistant")
    );
    for (client, path) in [
        ("editor", "Private/memory.md"),
        ("assistant", "Personal/owner.md"),
    ] {
        for (tool, args) in [
            ("note_read", json!({"path":path})),
            ("note_write", json!({"path":path,"content":"changed"})),
            ("note_delete", json!({"path":path,"confirm":true})),
            ("note_move", json!({"from":path,"to":"Shared/stolen.md"})),
        ] {
            assert_not_found(&server.call(client, tool, args).await);
        }
    }
    assert_not_found(
        &server
            .call(
                "editor",
                "note_move",
                json!({"from":"Shared/editor.md","to":"Private/stolen.md"}),
            )
            .await,
    );
    assert_eq!(
        std::fs::read_to_string(server.fixture.vault.join("Private/memory.md")).unwrap(),
        PRIVATE
    );
    assert!(!server.fixture.vault.join("Private/stolen.md").exists());
    assert!(!server.fixture.vault.join("Shared/stolen.md").exists());
    data(
        &server
            .call(
                "assistant",
                "note_insert",
                json!({"path":"Shared/new-from-insert.md","content":"append creates"}),
            )
            .await,
    );
    assert_eq!(
        std::fs::read_to_string(server.fixture.vault.join("Shared/new-from-insert.md")).unwrap(),
        "append creates"
    );
    server.stop().await;
}

#[tokio::test]
async fn denied_and_absent_resources_are_equivalent_across_read_surfaces() {
    let server = Server::start().await;
    for tool in ["note_read", "note_inspect", "frontmatter"] {
        let mut denied_args = json!({"path":"Private/memory.md"});
        let mut absent_args = json!({"path":"Private/absent.md"});
        if tool == "frontmatter" {
            denied_args["action"] = json!("get");
            absent_args["action"] = json!("get");
        }
        let denied = server.call("reader", tool, denied_args).await;
        let absent = server.call("reader", tool, absent_args).await;
        assert_not_found(&denied);
        assert_not_found(&absent);
        assert_eq!(denied["error"]["code"], absent["error"]["code"]);
    }
    for format in ["list", "tree"] {
        let denied = server
            .call(
                "reader",
                "vault_list",
                json!({"path":"Private","format":format}),
            )
            .await;
        let absent = server
            .call(
                "reader",
                "vault_list",
                json!({"path":"Absent","format":format}),
            )
            .await;
        assert_not_found(&denied);
        assert_not_found(&absent);
    }
    let bulk = server
        .call(
            "reader",
            "note_read_many",
            json!({"paths":["Private/memory.md","Private/absent.md","Shared/budget.md"]}),
        )
        .await;
    let bulk = data(&bulk);
    assert_eq!(bulk["notes"].as_array().unwrap().len(), 1);
    assert_eq!(bulk["notes"][0]["path"], "Shared/budget.md");
    let reasons: Vec<_> = bulk["skipped"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| &entry["reason"])
        .collect();
    assert_eq!(reasons, vec![&json!("not_found"), &json!("not_found")]);
    server.stop().await;
}

#[tokio::test]
async fn root_editor_cannot_read_or_modify_owner_configuration_or_caches() {
    let server = Server::start().await;
    for path in [
        ".obsidian/plugins/sample/main.js",
        ".git/hooks/pre-commit",
        ".obsidian-mcp/ignore",
        ".obsidian-mcp/embeddings/embeddings.bin",
    ] {
        assert_not_found(
            &server
                .call("editor", "note_read", json!({"path":path}))
                .await,
        );
        assert_not_found(
            &server
                .call(
                    "editor",
                    "note_write",
                    json!({"path":path,"content":"changed"}),
                )
                .await,
        );
    }
    assert_eq!(
        std::fs::read_to_string(
            server
                .fixture
                .vault
                .join(".obsidian/plugins/sample/main.js")
        )
        .unwrap(),
        "owner-plugin-marker"
    );
    assert_eq!(
        std::fs::read_to_string(server.fixture.vault.join(".git/hooks/pre-commit")).unwrap(),
        "owner-hook-marker"
    );
    assert_rejected(
        &server
            .call("editor", "note_move", json!({"from":"Shared","to":"Moved"}))
            .await,
    );
    assert_rejected(
        &server
            .call(
                "editor",
                "note_delete",
                json!({"path":"Shared/Empty","confirm":true}),
            )
            .await,
    );
    assert!(server.fixture.vault.join("Shared/Empty").is_dir());
    assert!(!server.fixture.vault.join("Moved").exists());
    server.stop().await;
}

#[tokio::test]
async fn denied_periodic_template_is_not_treated_as_a_missing_template() {
    let server = Server::start().await;
    let result = server
        .call(
            "editor",
            "periodic",
            json!({"action":"create","period":"daily","date":"2026-11-13"}),
        )
        .await;
    assert_rejected(&result);
    assert!(!result.to_string().contains("private-template-marker"));
    assert!(!server.fixture.vault.join("Shared/Daily").exists());
    data(&server.call("editor", "periodic", json!({"action":"create","period":"daily","date":"2026-11-13","content":"explicit periodic"})).await);
    let read = server
        .call(
            "reader",
            "periodic",
            json!({"action":"get","period":"daily","date":"2026-11-13"}),
        )
        .await;
    assert_eq!(data(&read)["content"], "explicit periodic");
    server.stop().await;
}

#[tokio::test]
async fn hidden_edits_do_not_change_visible_graph_scores_or_counts() {
    let server = Server::start().await;
    let queries = [
        ("search_text", json!({"query":"budget","max_results":1})),
        (
            "search_text",
            json!({"query":"budget","fuzzy":true,"fields":["body"],"max_results":1}),
        ),
        ("search_regex", json!({"pattern":"budget|source-marker"})),
        ("search_metadata", json!({"type":"tag","tag":"shared"})),
        (
            "search_metadata",
            json!({"type":"frontmatter","field":"kind","value":"budget"}),
        ),
        (
            "wikilinks",
            json!({"query":"outgoing","path":"Shared/graph.md"}),
        ),
        (
            "wikilinks",
            json!({"query":"broken","path":"Shared/graph.md"}),
        ),
        ("wikilinks", json!({"query":"orphans"})),
        ("note_inspect", json!({"path":"Shared/graph.md"})),
        ("vault_info", json!({})),
    ];
    let mut before = Vec::new();
    for (tool, args) in &queries {
        before.push(data(&server.call("reader", tool, args.clone()).await).clone());
    }
    for (path, content) in [
        (
            "Private/secret.md",
            "# Secret\nA budget hidden from the reader.",
        ),
        (
            "Private/reference.md",
            "# Reference\nA hidden duplicate basename budget.",
        ),
        (
            "Private/extra.md",
            "# Budget\nbudget budget budget budget budget",
        ),
        (
            "Private/tagged.md",
            "---\ntags: [shared]\nkind: budget\n---\n# Hidden budget\nprivate-source-marker",
        ),
    ] {
        data(
            &server
                .call(
                    "assistant",
                    "note_create",
                    json!({"path":path,"content":content}),
                )
                .await,
        );
    }
    for ((tool, args), before) in queries.iter().zip(before) {
        let after = server.call("reader", tool, args.clone()).await;
        assert_eq!(data(&after), &before, "hidden edit changed {tool}: {after}");
    }
    let listing = server
        .call(
            "reader",
            "vault_list",
            json!({"recursive":true,"include_metadata":true}),
        )
        .await;
    let listing = data(&listing).to_string();
    assert!(listing.contains("Shared/image.txt"));
    assert!(!listing.contains("Private/"));
    assert!(!listing.contains("Personal/"));
    let empty = server
        .call("reader", "vault_list", json!({"path":"Shared/Empty"}))
        .await;
    assert_eq!(data(&empty)["results"], json!([]));
    server.stop().await;
}

#[tokio::test]
async fn legacy_sessions_cannot_substitute_for_credentials_or_survive_grant_changes() {
    let server = Server::start().await;
    let session = server.initialize("editor").await;
    let call = json!({"name":"note_read","arguments":{"path":"Shared/budget.md"}});
    let response = server
        .request("editor", "tools/call", call.clone(), LEGACY)
        .header("Mcp-Session-Id", &session)
        .send()
        .await
        .unwrap();
    data(&http_rpc::rpc_response(response).await);
    let response = http_rpc::request(
        &server.client,
        &server.url,
        "tools/call",
        call.clone(),
        LEGACY,
    )
    .header("Mcp-Session-Id", &session)
    .send()
    .await
    .unwrap();
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    let response = server
        .request("reader", "tools/call", call.clone(), LEGACY)
        .header("Mcp-Session-Id", &session)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    for method in [reqwest::Method::GET, reqwest::Method::DELETE] {
        let response = server
            .client
            .request(method, format!("{}/mcp", server.url))
            .header("Accept", "text/event-stream")
            .header("Mcp-Protocol-Version", LEGACY)
            .header("Mcp-Session-Id", &session)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }
    let response = server
        .notification(None, Some(&session))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    server
        .fixture
        .store
        .update(None, |policy| {
            policy.clients.get_mut("editor").unwrap().grants =
                vec![Grant::directory("shared", "Shared/", Role::Reader)];
            Ok(())
        })
        .unwrap();
    let response = server
        .request("editor", "tools/call", call, LEGACY)
        .header("Mcp-Session-Id", &session)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let read = server
        .call("editor", "note_read", json!({"path":"Shared/budget.md"}))
        .await;
    assert_eq!(data(&read)["content"], SHARED);
    server.stop().await;
}

#[tokio::test]
async fn credential_revocation_and_policy_restoration_are_observed_without_restart() {
    let server = Server::start().await;
    data(
        &server
            .call("reader", "note_read", json!({"path":"Shared/budget.md"}))
            .await,
    );
    server.fixture.revoke("reader");
    let response = server
        .request("reader", "tools/list", json!({}), http_rpc::MODERN)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    data(
        &server
            .call("editor", "note_read", json!({"path":"Shared/budget.md"}))
            .await,
    );
    let valid = std::fs::read(&server.fixture.access_file).unwrap();
    std::fs::write(&server.fixture.access_file, b"{invalid").unwrap();
    let response = server
        .request("editor", "tools/list", json!({}), http_rpc::MODERN)
        .send()
        .await
        .unwrap();
    assert!(response.status().is_server_error() || response.status() == StatusCode::UNAUTHORIZED);
    std::fs::write(&server.fixture.access_file, valid).unwrap();
    data(
        &server
            .call("editor", "note_read", json!({"path":"Shared/budget.md"}))
            .await,
    );
    server.stop().await;
}

#[cfg(unix)]
#[tokio::test]
async fn aliases_and_hard_links_cannot_bypass_requested_or_physical_grants() {
    let fixture = AccessFixture::new();
    let outside = fixture
        ._directory
        .path()
        .join("outside-private-target-marker.md");
    std::fs::write(&outside, "outside-private-content-marker").unwrap();
    std::os::unix::fs::symlink(&outside, fixture.vault.join("Shared/outside-alias.md")).unwrap();
    std::os::unix::fs::symlink(
        "../Private/memory.md",
        fixture.vault.join("Shared/private-alias.md"),
    )
    .unwrap();
    std::os::unix::fs::symlink(
        "../Shared/budget.md",
        fixture.vault.join("Private/shared-alias.md"),
    )
    .unwrap();
    std::fs::hard_link(
        fixture.vault.join("Private/memory.md"),
        fixture.vault.join("Shared/private-hardlink.md"),
    )
    .unwrap();
    let server = Server::start_fixture(fixture, true).await;
    for path in [
        "Shared/outside-alias.md",
        "Shared/private-alias.md",
        "Private/shared-alias.md",
        "Shared/private-hardlink.md",
    ] {
        assert_not_found(
            &server
                .call("editor", "note_read", json!({"path":path}))
                .await,
        );
        assert_not_found(
            &server
                .call(
                    "editor",
                    "note_write",
                    json!({"path":path,"content":"changed"}),
                )
                .await,
        );
    }
    assert_eq!(
        std::fs::read_to_string(server.fixture.vault.join("Private/memory.md")).unwrap(),
        PRIVATE
    );
    assert_eq!(
        std::fs::read_to_string(outside).unwrap(),
        "outside-private-content-marker"
    );
    server.stop().await;
}

#[tokio::test]
async fn configured_access_fails_before_source_state_on_invalid_or_incompatible_startup() {
    for kind in [
        "missing",
        "malformed",
        "wrong-vault",
        "nonloopback",
        "daemon",
        "stdio-without-token",
        "http-with-token",
        "duplicate-backends",
        "remote-source",
    ] {
        let fixture = AccessFixture::new();
        let mut command = fixture.command(true);
        if kind != "stdio-without-token" {
            command.args(["--http", "--host", "127.0.0.1", "--port", "0"]);
        }
        match kind {
            "missing" => {
                std::fs::remove_file(&fixture.access_file).unwrap();
            }
            "malformed" => {
                std::fs::write(&fixture.access_file, b"{malformed").unwrap();
            }
            "wrong-vault" => {
                let other = fixture._directory.path().join("other");
                std::fs::create_dir(&other).unwrap();
                command = http_listening::isolated_command(server_binary());
                command
                    .arg(other)
                    .args(["--http", "--port", "0", "--access-file"])
                    .arg(&fixture.access_file);
            }
            "nonloopback" => {
                command.args(["--host", "0.0.0.0"]);
            }
            "daemon" => {
                command.env("OBSIDIAN_SEMANTIC_MODE", "daemon");
            }
            "http-with-token" => {
                command.env("OBSIDIAN_ACCESS_TOKEN_FILE", &fixture.token_files["reader"]);
            }
            "duplicate-backends" => {
                command.env(
                    "OBSIDIAN_MEMORY_AUTHORIZER_URL",
                    "http://127.0.0.1:1/vault/memory/authorize",
                );
            }
            "remote-source" => {
                command = http_listening::isolated_command(server_binary());
                command
                    .arg("sftp://unreachable-access-fixture.invalid/vault")
                    .args(["--http", "--port", "0", "--access-file"])
                    .arg(&fixture.access_file);
            }
            _ => {}
        }
        command
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        let output = timeout(Duration::from_secs(10), command.output())
            .await
            .unwrap()
            .unwrap();
        assert!(!output.status.success(), "{kind}");
        assert!(
            !fixture.vault.join(".obsidian-mcp").exists(),
            "{kind} opened source state"
        );
    }
}

fn modern_stdio(id: u64, method: &str, mut params: Value) -> Value {
    params["_meta"] = json!({
        "io.modelcontextprotocol/protocolVersion": http_rpc::MODERN,
        "io.modelcontextprotocol/clientInfo": {"name":"access-stdio","version":"1"},
        "io.modelcontextprotocol/clientCapabilities": {}
    });
    json!({"jsonrpc":"2.0","id":id,"method":method,"params":params})
}

#[tokio::test]
async fn restricted_stdio_preserves_discovery_then_legacy_selection_and_rechecks_authority() {
    let fixture = AccessFixture::new();
    let mut child = fixture
        .command(true)
        .env("OBSIDIAN_ACCESS_TOKEN_FILE", &fixture.token_files["reader"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut input = child.stdin.take().unwrap();
    let mut output = BufReader::new(child.stdout.take().unwrap()).lines();
    for request in [
        modern_stdio(0, "server/discover", json!({})),
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":LEGACY,"capabilities":{},"clientInfo":{"name":"access-stdio","version":"1"}}}),
    ] {
        input
            .write_all(format!("{request}\n").as_bytes())
            .await
            .unwrap();
        let response = timeout(Duration::from_secs(10), output.next_line())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        let response: Value = serde_json::from_str(&response).unwrap();
        assert!(response.get("error").is_none(), "{response}");
    }
    input
        .write_all(b"{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\n")
        .await
        .unwrap();
    let request = json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"note_read","arguments":{"path":"Shared/budget.md"}}});
    input
        .write_all(format!("{request}\n").as_bytes())
        .await
        .unwrap();
    let response = timeout(Duration::from_secs(10), output.next_line())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    let response: Value = serde_json::from_str(&response).unwrap();
    assert_eq!(data(&response)["content"], SHARED);
    fixture.revoke("reader");
    let request = json!({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"note_read","arguments":{"path":"Shared/budget.md"}}});
    input
        .write_all(format!("{request}\n").as_bytes())
        .await
        .unwrap();
    let response = timeout(Duration::from_secs(10), output.next_line())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    let response: Value = serde_json::from_str(&response).unwrap();
    assert_rejected(&response);
    assert!(!response.to_string().contains("household"));
    drop(input);
    timeout(Duration::from_secs(10), child.wait())
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn credential_rotation_overlaps_without_sharing_legacy_sessions() {
    let server = Server::start().await;
    let session = server.initialize("reader").await;
    let replacement_file = server
        .fixture
        ._directory
        .path()
        .join("access/replacement.token");
    server
        .fixture
        .store
        .issue_token("reader", &replacement_file, None)
        .unwrap();
    let replacement = read_token_file(&replacement_file).unwrap();
    for token in [
        server.fixture.tokens["reader"].as_str(),
        replacement.as_str(),
    ] {
        let response = server
            .request_token(token, "tools/list", json!({}), http_rpc::MODERN)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }
    let response = server
        .request_token(replacement.as_str(), "tools/list", json!({}), LEGACY)
        .header("Mcp-Session-Id", &session)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    server.fixture.revoke("reader");
    let response = server
        .request_token(
            replacement.as_str(),
            "tools/list",
            json!({}),
            http_rpc::MODERN,
        )
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    server
        .fixture
        .store
        .update(None, |policy| {
            policy.clients.get_mut("reader").unwrap().enabled = false;
            Ok(())
        })
        .unwrap();
    let response = server
        .request_token(
            replacement.as_str(),
            "tools/list",
            json!({}),
            http_rpc::MODERN,
        )
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    server.stop().await;
}

#[tokio::test]
async fn exact_file_grants_allow_only_their_required_namespace() {
    let mut fixture = AccessFixture::new();
    fixture
        .store
        .update(None, |policy| {
            policy.clients.insert(
                "exact".into(),
                ClientPolicy {
                    grants: vec![Grant::file(
                        "single-note",
                        "Shared/New/deep/note.md",
                        Role::Editor,
                    )],
                    ..Default::default()
                },
            );
            Ok(())
        })
        .unwrap();
    let token_file = fixture._directory.path().join("access/exact.token");
    fixture
        .store
        .issue_token("exact", &token_file, None)
        .unwrap();
    fixture.tokens.insert(
        "exact".into(),
        read_token_file(&token_file).unwrap().as_str().to_owned(),
    );
    let server = Server::start_fixture(fixture, true).await;
    data(
        &server
            .call(
                "exact",
                "note_create",
                json!({"path":"Shared/New/deep/note.md","content":"exact-grant-marker"}),
            )
            .await,
    );
    let listing = server
        .call("exact", "vault_list", json!({"recursive":true}))
        .await;
    let results = data(&listing)["results"].as_array().unwrap();
    assert!(results.iter().any(|path| path == "Shared/New/deep/note.md"));
    assert!(!results.iter().any(|path| path == "Shared/budget.md"));
    let nested = server
        .call(
            "exact",
            "vault_list",
            json!({"path":"Shared/New","format":"tree"}),
        )
        .await;
    assert!(
        data(&nested)["content"]
            .as_str()
            .unwrap()
            .contains("note.md")
    );
    assert_not_found(
        &server
            .call(
                "exact",
                "note_create",
                json!({"path":"Shared/New/deep/sibling.md","content":"changed"}),
            )
            .await,
    );
    assert!(
        !server
            .fixture
            .vault
            .join("Shared/New/deep/sibling.md")
            .exists()
    );
    server.stop().await;
}

#[tokio::test]
async fn authenticated_outputs_match_the_advertised_object_schemas() {
    let server = Server::start().await;
    // Prime the shared transport with the narrower principal before the editor.
    for client in ["reader", "editor", "reader"] {
        let catalog = http_rpc::rpc_response(
            server
                .request(client, "tools/list", json!({}), http_rpc::MODERN)
                .send()
                .await
                .unwrap(),
        )
        .await;
        for (name, arguments) in [
            ("note_read", json!({"path":"Shared/budget.md"})),
            ("note_read_many", json!({"dir":"Shared","recursive":true})),
            (
                "vault_list",
                json!({"recursive":true,"include_metadata":true}),
            ),
            ("vault_info", json!({})),
            (
                "frontmatter",
                json!({"action":"get","path":"Shared/budget.md"}),
            ),
            ("periodic", json!({"action":"list","period":"daily"})),
            (
                "wikilinks",
                json!({"query":"outgoing","path":"Shared/graph.md"}),
            ),
        ] {
            let tool = catalog["result"]["tools"]
                .as_array()
                .unwrap()
                .iter()
                .find(|tool| tool["name"] == name)
                .unwrap();
            let response = server.call(client, name, arguments).await;
            let structured = data(&response);
            let schema = jsonschema::validator_for(&tool["outputSchema"]).unwrap();
            let errors: Vec<_> = schema
                .iter_errors(structured)
                .map(|error| error.to_string())
                .collect();
            assert!(errors.is_empty(), "{name}: {errors:?}; {structured}");
            assert!(
                response["result"]["content"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|item| item["type"] == "text")
            );
        }
    }
    server.stop().await;
}

#[cfg(feature = "embeddings-api")]
#[tokio::test]
async fn invalid_access_configuration_cannot_probe_or_submit_to_a_provider() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let fixture = AccessFixture::new();
    std::fs::write(&fixture.access_file, b"{invalid").unwrap();
    let requests = Arc::new(AtomicUsize::new(0));
    let observed = requests.clone();
    let app = axum::Router::new().fallback(move || {
        let observed = observed.clone();
        async move {
            observed.fetch_add(1, Ordering::SeqCst);
            axum::Json(json!({"data":[{"index":0,"embedding":[1.0,0.0]}]}))
        }
    });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let provider = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let profile = embedding_profiles::write_api_profile(
        fixture._directory.path(),
        port,
        "access-fixture",
        8192,
        32,
    );
    let output = fixture
        .command(true)
        .args(["--http", "--port", "0"])
        .env("OBSIDIAN_EMBEDDINGS", "true")
        .env("OBSIDIAN_EMBEDDING_PROVIDER", "api")
        .env(
            "OBSIDIAN_EMBEDDING_API_BASE",
            format!("http://127.0.0.1:{port}/v1"),
        )
        .env("OBSIDIAN_EMBEDDING_API_KEY", "synthetic-provider-key")
        .env("OBSIDIAN_EMBEDDING_API_MODEL", "access-fixture")
        .env("OBSIDIAN_EMBEDDING_API_PROFILE", profile)
        .env("OBSIDIAN_EMBEDDING_DIM", "2")
        .output()
        .await
        .unwrap();
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .to_ascii_lowercase()
            .contains("access")
    );
    assert_eq!(requests.load(Ordering::SeqCst), 0);
    assert!(!fixture.vault.join(".obsidian-mcp").exists());
    provider.abort();
}

#[tokio::test]
async fn absent_denied_names_cannot_be_created_under_equivalent_spellings() {
    let fixture = AccessFixture::new();
    fixture
        .store
        .update(None, |policy| {
            let client = policy.clients.get_mut("editor").unwrap();
            client
                .denies
                .push(Deny::directory("future-private", "FuturePrivate/"));
            client
                .denies
                .push(Deny::file("future-secret", "future-secret.md"));
            Ok(())
        })
        .unwrap();
    let server = Server::start_fixture(fixture, true).await;
    for path in [
        "FuturePrivate/new.md",
        "FUTUREPRIVATE/new.md",
        "ＦｕｔｕｒｅＰｒｉｖａｔｅ/new.md",
        "FUTURE-SECRET.MD",
    ] {
        assert_not_found(
            &server
                .call(
                    "editor",
                    "note_create",
                    json!({"path":path,"content":"changed"}),
                )
                .await,
        );
        assert!(!server.fixture.vault.join(path).exists());
    }
    assert!(!server.fixture.vault.join("FuturePrivate").exists());
    assert!(!server.fixture.vault.join("FUTUREPRIVATE").exists());
    assert_not_found(
        &server
            .call(
                "reader",
                "note_read",
                json!({"path":"Shared-old/budget.md"}),
            )
            .await,
    );
    server.stop().await;
}

#[tokio::test]
async fn expired_credentials_are_rejected_while_other_clients_remain_available() {
    let server = Server::start().await;
    server
        .fixture
        .store
        .update(None, |policy| {
            let credential = &mut policy.clients.get_mut("reader").unwrap().credentials[0];
            credential.created_at = 0;
            credential.expires_at = Some(1);
            Ok(())
        })
        .unwrap();
    let response = server
        .request("reader", "tools/list", json!({}), http_rpc::MODERN)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    data(
        &server
            .call("editor", "note_read", json!({"path":"Shared/budget.md"}))
            .await,
    );
    server.stop().await;
}

#[cfg(feature = "embeddings-api")]
#[tokio::test]
async fn revoking_a_held_semantic_query_never_releases_its_old_authorized_result() {
    use std::sync::{Arc, Mutex};
    use tokio::sync::Notify;

    #[derive(Default)]
    struct ProviderState {
        inputs: Mutex<Vec<String>>,
        entered: Notify,
        release: Notify,
    }

    let fixture = AccessFixture::new();
    let replacement_file = fixture
        ._directory
        .path()
        .join("access/semantic-replacement.token");
    let replacement_receipt = fixture
        .store
        .issue_token("reader", &replacement_file, None)
        .unwrap();
    let replacement = read_token_file(&replacement_file).unwrap();
    let state = Arc::new(ProviderState::default());
    let observed = state.clone();
    let app = axum::Router::new().route(
        "/v1/embeddings",
        axum::routing::post(move |axum::Json(body): axum::Json<Value>| {
            let state = observed.clone();
            async move {
                let inputs: Vec<String> = body["input"].as_array().unwrap().iter()
                    .map(|value| value.as_str().unwrap().to_owned()).collect();
                state.inputs.lock().unwrap().extend(inputs.iter().cloned());
                if inputs.iter().any(|input| input == "held-query") {
                    state.entered.notify_one();
                    state.release.notified().await;
                }
                axum::Json(json!({"data":inputs.iter().enumerate().map(|(index,_)| json!({"index":index,"embedding":[1.0,0.0]})).collect::<Vec<_>>()}))
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let provider = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let profile = embedding_profiles::write_api_profile(
        fixture._directory.path(),
        port,
        "access-fixture",
        8192,
        32,
    );
    let mut command = fixture.command(true);
    command
        .env("OBSIDIAN_EMBEDDINGS", "true")
        .env("OBSIDIAN_EMBEDDING_PROVIDER", "api")
        .env(
            "OBSIDIAN_EMBEDDING_API_BASE",
            format!("http://127.0.0.1:{port}/v1"),
        )
        .env("OBSIDIAN_EMBEDDING_API_KEY", "synthetic-provider-key")
        .env("OBSIDIAN_EMBEDDING_API_MODEL", "access-fixture")
        .env("OBSIDIAN_EMBEDDING_API_PROFILE", profile)
        .env("OBSIDIAN_EMBEDDING_DIM", "2");
    let server = Server::start_command(fixture, command).await;
    let warm_deadline = tokio::time::Instant::now() + Duration::from_secs(20);
    loop {
        let result = server
            .call(
                "reader",
                "search_semantic",
                json!({"query":"warmup","top_k":1}),
            )
            .await;
        if result.get("error").is_none() && result["result"]["isError"] != true {
            assert!(
                !data(&result)["results"].as_array().unwrap().is_empty(),
                "semantic fixture has no visible evidence: {result}"
            );
            break;
        }
        assert!(
            tokio::time::Instant::now() < warm_deadline,
            "semantic warmup failed: {result}"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    for input in state.inputs.lock().unwrap().iter() {
        for marker in [
            "private-source-marker",
            "private-template-marker",
            "owner-source-marker",
            "owner-plugin-marker",
        ] {
            assert!(!input.contains(marker), "provider received a denied marker");
        }
    }
    let request = server.request("reader", "tools/call", json!({"name":"search_semantic","arguments":{"query":"held-query","top_k":3,"include_content":true}}), http_rpc::MODERN);
    let result = tokio::spawn(async move {
        match request.send().await {
            Ok(mut response) => {
                let status = response.status();
                let mut bytes = Vec::new();
                while let Ok(Some(chunk)) = response.chunk().await {
                    bytes.extend_from_slice(&chunk);
                }
                Ok((status, String::from_utf8(bytes).unwrap()))
            }
            Err(error) => Err(error.to_string()),
        }
    });
    timeout(Duration::from_secs(10), state.entered.notified())
        .await
        .unwrap();
    let store = server.fixture.store.clone();
    let credential = server.fixture.credentials["reader"].clone();
    let revocation = tokio::task::spawn_blocking(move || {
        store.update(None, |policy| {
            policy
                .clients
                .get_mut("reader")
                .unwrap()
                .credentials
                .iter_mut()
                .find(|entry| entry.id == credential)
                .unwrap()
                .revoked = true;
            Ok(())
        })
    });
    let revoked = timeout(Duration::from_secs(3), revocation).await;
    state.release.notify_one();
    revoked
        .expect("revocation waited for held provider work")
        .unwrap()
        .unwrap();
    let result = timeout(Duration::from_secs(10), result)
        .await
        .unwrap()
        .unwrap();
    if let Ok((status, body)) = result {
        for forbidden in [
            "A budget for the household",
            "Visible reference.",
            "literal readable links",
            "Shared/budget.md",
            "Shared/reference.md",
            "Shared/graph.md",
        ] {
            assert!(
                !body.contains(forbidden),
                "revoked query disclosed {forbidden}: {body}"
            );
        }
        if status.is_success() && !body.is_empty() {
            let response: Value = serde_json::from_str(&body)
                .expect("a revoked response must not release partial result data");
            assert!(
                response.get("error").is_some() || response["result"]["isError"] == true,
                "revoked query succeeded: {response}"
            );
        }
    }
    // The replacement credential keeps the same service-owned readable corpus
    // eligible after the credential that activated it has been revoked.
    data(&server.call("editor", "note_write", json!({"path":"Shared/budget.md","content":"# Shared budget\nreader-rotation-update"})).await);
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    loop {
        if state
            .inputs
            .lock()
            .unwrap()
            .iter()
            .any(|input| input.contains("reader-rotation-update"))
        {
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "replacement credential did not retain document ingestion"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let response = server
        .request_token(
            replacement.as_str(),
            "tools/call",
            json!({"name":"search_semantic","arguments":{"query":"replacement-query","top_k":3}}),
            http_rpc::MODERN,
        )
        .send()
        .await
        .unwrap();
    data(&http_rpc::rpc_response(response).await);
    server
        .fixture
        .store
        .update(None, |policy| {
            policy
                .clients
                .get_mut("reader")
                .unwrap()
                .credentials
                .iter_mut()
                .find(|entry| entry.id == replacement_receipt.credential_id)
                .unwrap()
                .revoked = true;
            Ok(())
        })
        .unwrap();
    data(&server.call("editor", "note_write", json!({"path":"Shared/budget.md","content":"# Shared budget\nreader-retired-update"})).await);
    // The source mutation has published its generation before returning. Retain
    // a silence interval across several service reconciliation cycles as well.
    let settled = tokio::time::Instant::now() + Duration::from_secs(1);
    while tokio::time::Instant::now() < settled {
        assert!(
            !state
                .inputs
                .lock()
                .unwrap()
                .iter()
                .any(|input| input.contains("reader-retired-update")),
            "withdrawn readable policy submitted a later document"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    server.stop().await;
    provider.abort();
}

#[tokio::test]
async fn periodic_configuration_failures_do_not_disclose_owner_paths_or_format_values() {
    let fixture = AccessFixture::new();
    let private_folder = fixture._directory.path().join("owner-configuration-marker");
    std::fs::write(
        fixture.vault.join(".obsidian/daily-notes.json"),
        json!({"folder":private_folder,"format":"YYYY-MM-DD"}).to_string(),
    )
    .unwrap();
    let server = Server::start_fixture(fixture, true).await;
    let result = server
        .call(
            "reader",
            "periodic",
            json!({"action":"list","period":"daily"}),
        )
        .await;
    assert_rejected(&result);
    assert!(
        !result.to_string().contains("owner-configuration-marker"),
        "owner path disclosed: {result}"
    );
    assert!(
        !result
            .to_string()
            .contains(&private_folder.display().to_string()),
        "owner path disclosed: {result}"
    );
    std::fs::write(
        server.fixture.vault.join(".obsidian/daily-notes.json"),
        json!({"folder":"Shared/Daily","format":"[owner-format-marker]-HH"}).to_string(),
    )
    .unwrap();
    let result = server
        .call(
            "editor",
            "periodic",
            json!({"action":"create","period":"daily","date":"2026-11-13","content":"explicit"}),
        )
        .await;
    assert_rejected(&result);
    assert!(
        !result.to_string().contains("owner-format-marker"),
        "owner format disclosed: {result}"
    );
    assert!(!server.fixture.vault.join("Shared/Daily").exists());
    server.stop().await;
}

#[tokio::test]
async fn replacing_a_credential_digest_under_the_same_id_cannot_borrow_its_session() {
    use sha2::{Digest, Sha256};

    let server = Server::start().await;
    let session = server.initialize("reader").await;
    let credential_id = &server.fixture.credentials["reader"];
    let replacement = format!("omcp1_{credential_id}_{}", "a".repeat(64));
    let mut hasher = Sha256::new();
    hasher.update(b"obsidian-mcp/access-token/v1\0");
    hasher.update(replacement.as_bytes());
    let digest = hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    server
        .fixture
        .store
        .update(None, |policy| {
            policy
                .clients
                .get_mut("reader")
                .unwrap()
                .credentials
                .iter_mut()
                .find(|credential| &credential.id == credential_id)
                .unwrap()
                .digest = digest;
            Ok(())
        })
        .unwrap();
    let response = server
        .request_token(&replacement, "tools/list", json!({}), http_rpc::MODERN)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let response = server
        .request_token(&replacement, "tools/list", json!({}), LEGACY)
        .header("Mcp-Session-Id", &session)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    server.stop().await;
}

struct CapturedServer {
    server: Server,
    stderr: PathBuf,
}

impl CapturedServer {
    async fn start(fixture: AccessFixture, mut command: Command) -> Self {
        let stderr = fixture._directory.path().join("server-stderr.log");
        let log = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&stderr)
            .unwrap();
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        drop(listener);
        let mut child = command
            .args(["--http", "--host", "127.0.0.1", "--port"])
            .arg(address.port().to_string())
            .env(
                "OBSIDIAN_LOG_LEVEL",
                "trace,rmcp=trace,rmcp::transport=trace,reqwest=trace,hyper=trace",
            )
            .stderr(Stdio::from(log))
            .spawn()
            .unwrap();
        let client = Client::builder()
            .no_proxy()
            .timeout(Duration::from_secs(10))
            .build()
            .unwrap();
        let url = format!("http://{address}");
        let deadline = tokio::time::Instant::now() + Duration::from_secs(15);
        loop {
            assert!(
                child.try_wait().unwrap().is_none(),
                "captured-log server exited before health became available"
            );
            if let Ok(response) = client
                .get(format!("{url}/health"))
                .timeout(Duration::from_millis(200))
                .send()
                .await
                && response.status() == StatusCode::OK
            {
                let body: Value = serde_json::from_str(&response.text().await.unwrap()).unwrap();
                assert_eq!(body["server"], "obsidian-mcp");
                assert_eq!(body["version"], env!("CARGO_PKG_VERSION"));
                break;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "captured-log server did not become healthy"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        Self {
            server: Server {
                child,
                fixture,
                client,
                url,
            },
            stderr,
        }
    }

    async fn finish(mut self) -> String {
        self.server.child.kill().await.unwrap();
        self.server.child.wait().await.unwrap();
        std::fs::read_to_string(&self.stderr).unwrap()
    }
}

#[tokio::test]
async fn authenticated_trace_logging_omits_tokens_note_bodies_and_denied_parse_details() {
    let fixture = AccessFixture::new();
    std::fs::write(
        fixture.vault.join("Shared/visible-path-marker.md"),
        "# Visible\ndelegated-response-marker\n",
    )
    .unwrap();
    std::fs::write(
        fixture.vault.join("Private/denied-path-marker.md"),
        "---\nprivate: [unterminated-yaml-marker\n---\ndenied-parse-source-marker\n",
    )
    .unwrap();
    let token = fixture.tokens["reader"].clone();
    let command = fixture.command(true);
    let captured = CapturedServer::start(fixture, command).await;
    let response = captured
        .server
        .call(
            "reader",
            "note_read",
            json!({"path":"Shared/visible-path-marker.md"}),
        )
        .await;
    assert!(
        data(&response)["content"]
            .as_str()
            .unwrap()
            .contains("delegated-response-marker")
    );
    let session = captured.server.initialize("reader").await;
    let notification = captured.server.client
        .post(format!("{}/mcp", captured.server.url))
        .header("Content-Type", "application/json")
        .header("Accept", "application/json, text/event-stream")
        .header("Mcp-Protocol-Version", LEGACY)
        .header("Mcp-Session-Id", &session)
        .bearer_auth(&captured.server.fixture.tokens["reader"])
        .body(json!({"jsonrpc":"2.0","method":"notifications/cancelled","params":{"requestId":"not-running","reason":"notification-source-marker"}}).to_string())
        .send().await.unwrap();
    assert_eq!(notification.status(), StatusCode::ACCEPTED);
    let response = captured
        .server
        .request("reader", "tools/list", json!({}), LEGACY)
        .header("Mcp-Session-Id", &session)
        .send()
        .await
        .unwrap();
    let response = http_rpc::rpc_response(response).await;
    assert!(response.get("error").is_none());
    let stderr = captured.finish().await;
    assert!(stderr.contains("authenticated invocation finished"));
    assert!(stderr.contains("protected index build"));
    for forbidden in [
        token.as_str(),
        "visible-path-marker",
        "delegated-response-marker",
        "denied-path-marker",
        "unterminated-yaml-marker",
        "denied-parse-source-marker",
        "notification-source-marker",
    ] {
        assert!(
            !stderr.contains(forbidden),
            "authenticated trace logging disclosed a synthetic credential or source marker"
        );
    }
}

#[cfg(feature = "embeddings-api")]
#[tokio::test]
async fn authenticated_provider_error_transitions_omit_echoed_source_from_stderr_and_mcp() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let fixture = AccessFixture::new();
    let token = fixture.tokens["reader"].clone();
    let requests = Arc::new(AtomicUsize::new(0));
    let observed = requests.clone();
    let app = axum::Router::new().route(
        "/v1/embeddings",
        axum::routing::post(move || {
            let observed = observed.clone();
            async move {
                observed.fetch_add(1, Ordering::SeqCst);
                (
                    StatusCode::BAD_REQUEST,
                    axum::Json(json!({"error":{"message":"provider-error-echo-marker Private/provider-secret-path-marker.md provider-document-echo-marker"}})),
                )
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let provider = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let profile = embedding_profiles::write_api_profile(
        fixture._directory.path(),
        port,
        "access-error-fixture",
        8192,
        32,
    );
    let mut command = fixture.command(true);
    command
        .env("OBSIDIAN_EMBEDDINGS", "true")
        .env("OBSIDIAN_EMBEDDING_PROVIDER", "api")
        .env(
            "OBSIDIAN_EMBEDDING_API_BASE",
            format!("http://127.0.0.1:{port}/v1"),
        )
        .env("OBSIDIAN_EMBEDDING_API_KEY", "provider-credential-marker")
        .env("OBSIDIAN_EMBEDDING_API_MODEL", "access-error-fixture")
        .env("OBSIDIAN_EMBEDDING_API_PROFILE", profile)
        .env("OBSIDIAN_EMBEDDING_DIM", "2");
    let captured = CapturedServer::start(fixture, command).await;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(15);
    loop {
        let response = captured
            .server
            .call(
                "reader",
                "search_semantic",
                json!({"query":"safe-query","top_k":1}),
            )
            .await;
        assert_rejected(&response);
        let status = captured
            .server
            .call("reader", "vault_info", json!({}))
            .await;
        for output in [&response, &status] {
            let text = output.to_string();
            for forbidden in [
                "provider-error-echo-marker",
                "provider-secret-path-marker",
                "provider-document-echo-marker",
                "provider-credential-marker",
                token.as_str(),
            ] {
                assert!(
                    !text.contains(forbidden),
                    "MCP error/status disclosed a synthetic provider or credential marker"
                );
            }
        }
        if data(&status)["embeddings"]["status"]["phase"] == "degraded" {
            let stderr = std::fs::read_to_string(&captured.stderr).unwrap();
            if stderr.contains("provider_error") || stderr.contains("semantic_runtime") {
                break;
            }
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "provider failure did not reach a degraded state with a safe diagnostic"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(requests.load(Ordering::SeqCst) > 0);
    let stderr = captured.finish().await;
    assert!(stderr.contains("authenticated invocation finished"));
    assert!(stderr.contains("provider_error") || stderr.contains("semantic_runtime"));
    for forbidden in [
        "provider-error-echo-marker",
        "provider-secret-path-marker",
        "provider-document-echo-marker",
        "provider-credential-marker",
        token.as_str(),
    ] {
        assert!(
            !stderr.contains(forbidden),
            "authenticated provider transition logging disclosed a synthetic source or credential marker"
        );
    }
    provider.abort();
}

#[cfg(feature = "embeddings")]
#[tokio::test]
async fn authenticated_local_model_search_returns_only_current_shared_notes() {
    let fixture = AccessFixture::new();
    std::fs::write(
        fixture.vault.join("Shared/local-model.md"),
        "# Household shopping\nlocal-model-visible-marker\nWeekly groceries share the household budget.\n",
    )
    .unwrap();
    let mut command = fixture.command(true);
    command
        .env("OBSIDIAN_EMBEDDINGS", "true")
        .env("OBSIDIAN_EMBEDDING_PROVIDER", "local")
        .env("OBSIDIAN_EMBEDDINGS_MODEL", "BAAI/bge-small-en-v1.5");
    let server = Server::start_command(fixture, command).await;
    let status = server.call("reader", "vault_info", json!({})).await;
    assert!(
        data(&status)["embeddings"].get("status").is_none(),
        "passive status activated the local model"
    );
    let deadline = tokio::time::Instant::now() + Duration::from_secs(180);
    let result = loop {
        let response = server
            .call(
                "reader",
                "search_semantic",
                json!({"query":"household shared budget","top_k":10,"include_content":true}),
            )
            .await;
        if response.get("error").is_none() && response["result"]["isError"] != true {
            break response;
        }
        assert_eq!(
            response["error"]["data"]["code"], "semantic_warming",
            "local model initialization failed instead of warming: {response}"
        );
        assert!(
            tokio::time::Instant::now() < deadline,
            "local model did not finish initializing"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    };
    let hits = data(&result)["results"].as_array().unwrap();
    assert_eq!(hits.len(), 4, "local model omitted shared source evidence");
    for hit in hits {
        let path = hit["path"].as_str().unwrap();
        assert!(path.starts_with("Shared/"));
        assert_eq!(hit["source_status"], "current");
        assert!(hit.get("context_error").is_none_or(Value::is_null));
        assert_eq!(
            hit["content"],
            std::fs::read_to_string(server.fixture.vault.join(path)).unwrap()
        );
    }
    let text = result.to_string();
    assert!(text.contains("local-model-visible-marker"));
    for forbidden in [
        "Private/",
        "Personal/",
        "private-source-marker",
        "private-template-marker",
        "owner-source-marker",
    ] {
        assert!(
            !text.contains(forbidden),
            "local search disclosed a denied marker"
        );
    }
    server.fixture.revoke("reader");
    let response = server
        .request(
            "reader",
            "tools/call",
            json!({"name":"search_semantic","arguments":{"query":"household budget"}}),
            http_rpc::MODERN,
        )
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    server.stop().await;
}
