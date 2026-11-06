//! Real HTTP reader coverage for audience-bound private memory grants.

#[path = "common/embedding_profiles.rs"]
mod embedding_profiles;
#[path = "common/http_listening.rs"]
mod http_listening;
#[path = "common/http_rpc.rs"]
mod http_rpc;

use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::routing::post;
use axum::{Json, Router};
use reqwest::Client;
use serde_json::{Value, json};
use tempfile::TempDir;
use tokio::process::{Child, Command};

const ALICE: &str = "slack-t1-channel-ca";
const BOB: &str = "slack-t1-dm-ub";
const CAROL: &str = "slack-t1-dm-uc";
const DAN: &str = "slack-t1-channel-cd";

#[tokio::test]
async fn private_reader_rejects_shared_semantic_ownership_before_opening_source() {
    for mode in ["auto", "daemon"] {
        let vault = tempfile::tempdir().unwrap();
        let mut command = http_listening::isolated_command(env!("CARGO_BIN_EXE_obsidian-mcp"));
        let output = command
            .arg(vault.path())
            .arg("--http")
            .env(
                "OBSIDIAN_MEMORY_AUTHORIZER_URL",
                "http://127.0.0.1:1/vault/memory/authorize",
            )
            .env("OBSIDIAN_SEMANTIC_MODE", mode)
            .output()
            .await
            .unwrap();
        assert!(!output.status.success());
        assert!(
            String::from_utf8_lossy(&output.stderr)
                .contains("requires OBSIDIAN_SEMANTIC_MODE=local")
        );
        assert!(!vault.path().join(".obsidian-mcp").exists());
    }
}

#[tokio::test]
async fn private_reader_rejects_remote_source_before_connection_or_local_state() {
    let data = tempfile::tempdir().unwrap();
    let mut command = http_listening::isolated_command(env!("CARGO_BIN_EXE_obsidian-mcp"));
    let output = command
        .arg("sftp://unreachable-private-reader.invalid/vault")
        .arg("--http")
        .env(
            "OBSIDIAN_MEMORY_AUTHORIZER_URL",
            "http://127.0.0.1:1/vault/memory/authorize",
        )
        .env("OBSIDIAN_MEMORY_KEY_FILE", data.path().join("missing-key"))
        .env("OBSIDIAN_MCP_DATA", data.path())
        .env("OBSIDIAN_SEMANTIC_MODE", "local")
        .output()
        .await
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("requires a local vault source"));
    assert_eq!(std::fs::read_dir(data.path()).unwrap().count(), 0);
}

fn policy(token: &str) -> Value {
    let (scope, kind, audience) = match token {
        "alice" | "reflection" => (ALICE, "channel", "CA"),
        "carol" => (CAROL, "dm", "UC"),
        "dan" => (DAN, "channel", "CD"),
        _ => (BOB, "dm", "UB"),
    };
    json!({"version":1,"tenant_id":"bloomer","star_label":"bloom","workspace_id":"T1",
        "audience_kind":kind,"audience_id":audience,"purpose":"conversation","scope_id":scope,
        "allowed_prefixes":[format!("Organize/Work/{scope}/"),format!("Distill/{scope}/index.md")],
        "expires_at":4_000_000_000.0})
}

#[derive(Default)]
struct Authorization {
    revoked: Mutex<HashSet<String>>,
}

async fn authorize(
    State(state): State<Arc<Authorization>>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> (StatusCode, Json<Value>) {
    assert_eq!(headers.get("X-Vault-Key").unwrap(), "local-service-key");
    assert_eq!(body["version"], 1);
    let token = body["token"].as_str().unwrap();
    if !matches!(
        token,
        "alice" | "bob" | "carol" | "dan" | "expired" | "reflection"
    ) || state.revoked.lock().unwrap().contains(token)
    {
        (StatusCode::FORBIDDEN, Json(json!({"error":"denied"})))
    } else {
        let mut scope = policy(token);
        if token == "expired" {
            scope["expires_at"] = json!(0.0);
        }
        if token == "reflection" {
            scope["purpose"] = json!("reflection");
            scope["allowed_prefixes"]
                .as_array_mut()
                .unwrap()
                .push(json!(format!("Capture/chats/{ALICE}/")));
        }
        (StatusCode::OK, Json(scope))
    }
}

struct Reader {
    child: Child,
    authorizer: tokio::task::JoinHandle<()>,
    authorization: Arc<Authorization>,
    vault: TempDir,
    client: Client,
    url: String,
    embedding_api: Option<tokio::task::JoinHandle<()>>,
    provider_inputs: Arc<Mutex<Vec<String>>>,
}

impl Reader {
    async fn start() -> Self {
        Self::start_with_embeddings(false).await
    }

    async fn start_with_embeddings(embeddings: bool) -> Self {
        let vault = tempfile::tempdir().unwrap();
        for (scope, kind, id, secret) in [
            (ALICE, "channel", "CA", "violet-origin"),
            (BOB, "dm", "UB", "saffron-private"),
            (CAROL, "dm", "UC", "carol-private"),
            (DAN, "channel", "CD", "dan-private"),
        ] {
            let dir = vault.path().join(format!("Organize/Work/{scope}"));
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("decision.md"), content(kind, id, secret)).unwrap();
        }
        std::fs::write(
            vault.path().join("shared.md"),
            "# tenant-shared-secret\nShared content requires the shared service.",
        )
        .unwrap();
        let hot = vault.path().join(format!("Distill/{ALICE}"));
        std::fs::create_dir_all(&hot).unwrap();
        std::fs::write(
            hot.join("index.md"),
            content("channel", "CA", "authorized-index"),
        )
        .unwrap();
        for (name, body) in [
            (
                "sensitive",
                content("channel", "CA", "sensitive-provider-secret")
                    .replace("sensitivity: normal", "sensitivity: sensitive"),
            ),
            (
                "inferred",
                content(
                    "channel",
                    "CA",
                    "inferred-provider-secret\n  *   [inferred 2026-11-06] uncertain",
                ),
            ),
            (
                "legacy",
                "# legacy-provider-secret\nMoved without origin metadata.".into(),
            ),
        ] {
            std::fs::write(
                vault
                    .path()
                    .join(format!("Organize/Work/{ALICE}/{name}.md")),
                body,
            )
            .unwrap();
        }
        std::fs::write(
            hot.join("index.md-extra.md"),
            content("channel", "CA", "exact-path-secret"),
        )
        .unwrap();
        let raw = vault.path().join(format!("Capture/chats/{ALICE}"));
        std::fs::create_dir_all(&raw).unwrap();
        std::fs::write(
            raw.join("raw.md"),
            content("channel", "CA", "raw-capture-secret"),
        )
        .unwrap();
        std::fs::write(
            vault
                .path()
                .join(format!("Organize/Work/{ALICE}/mismatched.md")),
            content("dm", "UB", "wrong-origin-secret"),
        )
        .unwrap();
        let key = vault.path().join("service-key");
        std::fs::write(&key, "local-service-key\n").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&key, std::fs::Permissions::from_mode(0o600)).unwrap();
        }
        let authorization = Arc::new(Authorization::default());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = Router::new()
            .route("/vault/memory/authorize", post(authorize))
            .with_state(authorization.clone());
        let authorizer = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let mut command: Command =
            http_listening::isolated_command(env!("CARGO_BIN_EXE_obsidian-mcp"));
        command
            .arg(vault.path())
            .args(["--http", "--host", "127.0.0.1", "--port", "0"])
            .env("OBSIDIAN_WATCH", "false")
            .env("OBSIDIAN_EMBEDDINGS", "false")
            .env("OBSIDIAN_SEMANTIC_MODE", "local")
            .env("OBSIDIAN_LOG_LEVEL", "info")
            .env("NO_COLOR", "1")
            .env(
                "OBSIDIAN_MEMORY_AUTHORIZER_URL",
                format!("http://{address}/vault/memory/authorize"),
            )
            .env("OBSIDIAN_MEMORY_KEY_FILE", key)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);
        let provider_inputs = Arc::new(Mutex::new(Vec::new()));
        let embedding_api = if embeddings {
            async fn embed(
                State(inputs): State<Arc<Mutex<Vec<String>>>>,
                Json(body): Json<Value>,
            ) -> Json<Value> {
                let text = body["input"].as_array().unwrap();
                inputs
                    .lock()
                    .unwrap()
                    .extend(text.iter().map(|value| value.as_str().unwrap().to_owned()));
                Json(
                    json!({"data":text.iter().enumerate().map(|(index,_)| json!({"index":index,"embedding":[1.0,0.0]})).collect::<Vec<_>>()}),
                )
            }
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let port = listener.local_addr().unwrap().port();
            let profile = embedding_profiles::write_api_profile(
                vault.path(),
                port,
                "memory-fixture",
                8192,
                32,
            );
            command
                .env("OBSIDIAN_EMBEDDINGS", "true")
                .env("OBSIDIAN_WATCH", "true")
                .env("OBSIDIAN_EMBEDDING_PROVIDER", "api")
                .env(
                    "OBSIDIAN_EMBEDDING_API_BASE",
                    format!("http://127.0.0.1:{port}/v1"),
                )
                .env("OBSIDIAN_EMBEDDING_API_KEY", "synthetic-provider-key")
                .env("OBSIDIAN_EMBEDDING_API_MODEL", "memory-fixture")
                .env("OBSIDIAN_EMBEDDING_API_PROFILE", profile)
                .env("OBSIDIAN_EMBEDDING_DIM", "2");
            let app = Router::new()
                .route("/v1/embeddings", post(embed))
                .with_state(provider_inputs.clone());
            Some(tokio::spawn(async move {
                axum::serve(listener, app).await.unwrap();
            }))
        } else {
            None
        };
        let mut child = command.spawn().unwrap();
        let address = http_listening::ready(&mut child).await.unwrap();
        Self {
            child,
            authorizer,
            authorization,
            vault,
            client: Client::builder()
                .no_proxy()
                .timeout(Duration::from_secs(10))
                .build()
                .unwrap(),
            url: format!("http://{address}"),
            embedding_api,
            provider_inputs,
        }
    }

    fn request(&self, token: &str, method: &str, params: Value) -> reqwest::RequestBuilder {
        http_rpc::request(&self.client, &self.url, method, params, http_rpc::MODERN)
            .bearer_auth(token)
    }
    async fn call(&self, token: &str, name: &str, arguments: Value) -> Value {
        http_rpc::rpc_response(
            self.request(
                token,
                "tools/call",
                json!({"name":name,"arguments":arguments}),
            )
            .send()
            .await
            .unwrap(),
        )
        .await
    }
    async fn stop(mut self) {
        self.child.kill().await.unwrap();
        self.child.wait().await.unwrap();
        self.authorizer.abort();
        if let Some(api) = self.embedding_api {
            api.abort();
        }
    }
}

fn content(kind: &str, id: &str, body: &str) -> String {
    format!(
        "---\nmemory_schema: slack-code/1\nsensitivity: normal\ntenant_id: bloomer\nstar_label: bloom\nworkspace_id: T1\naudience_kind: {kind}\naudience_id: {id}\n---\n# {body}\n\nShared search term launch {body}.\n"
    )
}

#[tokio::test]
async fn every_private_read_surface_is_scoped_and_revocation_is_immediate() {
    let server = Reader::start().await;
    let health: Value = serde_json::from_str(
        &server
            .client
            .get(format!("{}/health", server.url))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(health["memory_scope_v1"], true);
    assert_eq!(health["memory_authorization_required"], true);
    assert!(server.provider_inputs.lock().unwrap().is_empty());
    for token in ["", "unknown", "expired"] {
        assert_eq!(
            server
                .request(token, "tools/list", json!({}))
                .send()
                .await
                .unwrap()
                .status(),
            StatusCode::UNAUTHORIZED
        );
    }
    let own = server
        .call(
            "alice",
            "note_read",
            json!({"path":format!("Organize/Work/{ALICE}/decision.md")}),
        )
        .await;
    assert!(own.to_string().contains("violet-origin"), "{own}");
    let bulk = server
        .call(
            "alice",
            "note_read_many",
            json!({"paths":[
                format!("Organize/Work/{BOB}/decision.md"),
                format!("Organize/Work/{ALICE}/sensitive.md"),
                format!("Organize/Work/{ALICE}/decision.md")
            ]}),
        )
        .await;
    assert!(bulk.get("error").is_none(), "{bulk}");
    let bulk_text = bulk.to_string();
    assert!(bulk_text.contains("violet-origin"), "{bulk}");
    for forbidden in ["saffron-private", "sensitive-provider-secret"] {
        assert!(!bulk_text.contains(forbidden), "{bulk}");
    }
    let capture_path = format!("Capture/chats/{ALICE}/raw.md");
    let denied_capture = server
        .call("alice", "note_read", json!({"path":capture_path}))
        .await;
    assert_eq!(denied_capture["error"]["code"], -32002);
    let reflection_capture = server
        .call("reflection", "note_read", json!({"path":capture_path}))
        .await;
    assert!(
        reflection_capture
            .to_string()
            .contains("raw-capture-secret")
    );
    for (tool, args) in [
        (
            "vault_list",
            json!({"recursive":true,"include_metadata":true}),
        ),
        ("search_text", json!({"query":"launch","max_results":1})),
        (
            "search_text",
            json!({"query":"launch","max_results":1,"fuzzy":true,"fields":["body"]}),
        ),
        ("search_regex", json!({"pattern":"launch"})),
        (
            "search_metadata",
            json!({"type":"frontmatter","field":"tenant_id","value":"bloomer"}),
        ),
        ("note_read_many", json!({"dir":"","recursive":true})),
        ("wikilinks", json!({"query":"orphans"})),
        ("vault_info", json!({})),
    ] {
        let result = server.call("alice", tool, args).await;
        assert!(result.get("error").is_none(), "{tool}: {result}");
        assert_ne!(result["result"]["isError"], true, "{tool}: {result}");
        let text = result.to_string();
        for forbidden in [
            BOB,
            "saffron-private",
            "raw-capture-secret",
            "wrong-origin-secret",
            "exact-path-secret",
            "sensitive-provider-secret",
            "inferred-provider-secret",
            "legacy-provider-secret",
            "carol-private",
            "dan-private",
            "tenant-shared-secret",
        ] {
            assert!(
                !text.contains(forbidden),
                "{tool} leaked {forbidden}: {text}"
            );
        }
    }
    for path in [
        format!("Organize/Work/{ALICE}/decision.md"),
        format!("Distill/{ALICE}/index.md"),
    ] {
        let result = server.call("bob", "note_read", json!({"path":path})).await;
        assert_eq!(result["error"]["code"], -32002, "{result}");
        assert!(!result.to_string().contains("violet-origin"));
    }
    let sensitive = server
        .call(
            "alice",
            "note_read",
            json!({"path":format!("Organize/Work/{ALICE}/sensitive.md")}),
        )
        .await;
    assert_eq!(sensitive["error"]["code"], -32002);
    assert!(!sensitive.to_string().contains("sensitive-provider-secret"));
    for (token, own_scope) in [
        ("alice", ALICE),
        ("bob", BOB),
        ("carol", CAROL),
        ("dan", DAN),
    ] {
        for scope in [ALICE, BOB, CAROL, DAN] {
            let result = server
                .call(
                    token,
                    "note_read",
                    json!({"path":format!("Organize/Work/{scope}/decision.md")}),
                )
                .await;
            assert_eq!(
                result.get("error").is_some(),
                scope != own_scope,
                "{token}: {result}"
            );
        }
    }
    let tools = http_rpc::rpc_response(
        server
            .request("alice", "tools/list", json!({}))
            .send()
            .await
            .unwrap(),
    )
    .await;
    assert!(
        !tools.to_string().contains("note_write"),
        "scoped reader must reject mutations even with default-full server config"
    );
    let mutation = server.call("alice", "note_write", json!({"path":format!("Organize/Work/{ALICE}/decision.md"),"content":"unexpected mutation"})).await;
    assert!(
        mutation.get("error").is_some() || mutation["result"]["isError"] == true,
        "{mutation}"
    );
    assert!(
        std::fs::read_to_string(
            server
                .vault
                .path()
                .join(format!("Organize/Work/{ALICE}/decision.md"))
        )
        .unwrap()
        .contains("violet-origin")
    );
    server
        .authorization
        .revoked
        .lock()
        .unwrap()
        .insert("alice".into());
    assert_eq!(
        server
            .request("alice", "tools/list", json!({}))
            .send()
            .await
            .unwrap()
            .status(),
        StatusCode::UNAUTHORIZED
    );
    let other = server
        .call(
            "bob",
            "note_read",
            json!({"path":format!("Organize/Work/{BOB}/decision.md")}),
        )
        .await;
    assert!(other.to_string().contains("saffron-private"));
    server.authorizer.abort();
    assert_eq!(
        server
            .request("bob", "tools/list", json!({}))
            .send()
            .await
            .unwrap()
            .status(),
        StatusCode::SERVICE_UNAVAILABLE
    );
    server.stop().await;
}

#[cfg(feature = "embeddings-api")]
#[tokio::test]
async fn private_provider_never_receives_raw_sensitive_inferred_or_unclassified_documents() {
    let server = Reader::start_with_embeddings(true).await;
    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            let response = server
                .client
                .get(format!("{}/health", server.url))
                .send()
                .await
                .unwrap();
            let health: Value = serde_json::from_str(&response.text().await.unwrap()).unwrap();
            if health["embeddings_phase"] == "ready" {
                assert_eq!(health["notes"], 5, "{health}");
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    let inputs = server.provider_inputs.lock().unwrap().join("\n");
    for allowed in ["violet-origin", "saffron-private", "authorized-index"] {
        assert!(inputs.contains(allowed), "missing {allowed}: {inputs}");
    }
    for forbidden in [
        "raw-capture-secret",
        "wrong-origin-secret",
        "exact-path-secret",
        "sensitive-provider-secret",
        "inferred-provider-secret",
        "legacy-provider-secret",
        "tenant-shared-secret",
    ] {
        assert!(
            !inputs.contains(forbidden),
            "provider received {forbidden}: {inputs}"
        );
    }
    let result = server
        .call(
            "alice",
            "search_semantic",
            json!({"query":"launch","top_k":1,"include_content":true}),
        )
        .await;
    assert!(result.get("error").is_none(), "{result}");
    assert!(!result.to_string().contains("saffron-private"), "{result}");
    let own_path = server
        .vault
        .path()
        .join(format!("Organize/Work/{ALICE}/decision.md"));
    std::fs::write(
        &own_path,
        content("channel", "CA", "transition-provider-secret")
            .replace("sensitivity: normal", "sensitivity: sensitive"),
    )
    .unwrap();
    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            let response = server
                .client
                .get(format!("{}/health", server.url))
                .send()
                .await
                .unwrap();
            let health: Value = serde_json::from_str(&response.text().await.unwrap()).unwrap();
            if health["notes"] == 4 && health["embeddings_pending_notes"] == 0 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    assert!(
        !server
            .provider_inputs
            .lock()
            .unwrap()
            .join("\n")
            .contains("transition-provider-secret")
    );
    std::fs::write(
        &own_path,
        content("channel", "CA", "permitted-provider-update"),
    )
    .unwrap();
    tokio::time::timeout(Duration::from_secs(10), async {
        while !server
            .provider_inputs
            .lock()
            .unwrap()
            .join("\n")
            .contains("permitted-provider-update")
        {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    server.stop().await;
}

#[tokio::test]
async fn source_metadata_changes_and_legacy_sessions_cannot_cross_audiences() {
    let server = Reader::start().await;
    let request = http_rpc::request(
        &server.client,
        &server.url,
        "initialize",
        json!({"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"scope-fixture","version":"1"}}),
        "2025-11-25",
    );
    let response = request.bearer_auth("alice").send().await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let session = response.headers()["Mcp-Session-Id"]
        .to_str()
        .unwrap()
        .to_owned();
    let _ = http_rpc::rpc_response(response).await;
    let initialized = server
        .client
        .post(format!("{}/mcp", server.url))
        .header("Content-Type", "application/json")
        .header("Accept", "application/json, text/event-stream")
        .header("Mcp-Protocol-Version", "2025-11-25")
        .header("Mcp-Session-Id", &session)
        .bearer_auth("alice")
        .body(json!({"jsonrpc":"2.0","method":"notifications/initialized"}).to_string())
        .send()
        .await
        .unwrap();
    assert_eq!(initialized.status(), StatusCode::ACCEPTED);
    let own_session_read = http_rpc::rpc_response(
        http_rpc::request(
            &server.client,
            &server.url,
            "tools/call",
            json!({"name":"note_read","arguments":{"path":format!("Organize/Work/{ALICE}/decision.md")}}),
            "2025-11-25",
        )
        .header("Mcp-Session-Id", &session)
        .bearer_auth("alice")
        .send()
        .await
        .unwrap(),
    )
    .await;
    assert!(
        own_session_read.to_string().contains("violet-origin"),
        "{own_session_read}"
    );
    let borrowed = http_rpc::request(
        &server.client,
        &server.url,
        "tools/list",
        json!({}),
        "2025-11-25",
    )
    .header("Mcp-Session-Id", &session)
    .bearer_auth("bob")
    .send()
    .await
    .unwrap();
    assert_eq!(borrowed.status(), StatusCode::NOT_FOUND);
    std::fs::write(
        server
            .vault
            .path()
            .join(format!("Organize/Work/{ALICE}/decision.md")),
        content("dm", "UB", "changed-origin-secret"),
    )
    .unwrap();
    for tool in ["note_read", "note_inspect"] {
        let denied = server
            .call(
                "alice",
                tool,
                json!({"path":format!("Organize/Work/{ALICE}/decision.md")}),
            )
            .await;
        assert_eq!(denied["error"]["code"], -32002, "{denied}");
        assert!(!denied.to_string().contains("changed-origin-secret"));
    }
    let listing = server
        .call("alice", "vault_list", json!({"recursive":true}))
        .await;
    assert!(!listing.to_string().contains("decision.md"));
    server
        .authorization
        .revoked
        .lock()
        .unwrap()
        .insert("alice".into());
    let revoked = http_rpc::request(
        &server.client,
        &server.url,
        "tools/list",
        json!({}),
        "2025-11-25",
    )
    .header("Mcp-Session-Id", &session)
    .bearer_auth("alice")
    .send()
    .await
    .unwrap();
    assert_eq!(revoked.status(), StatusCode::UNAUTHORIZED);
    server.stop().await;
}
