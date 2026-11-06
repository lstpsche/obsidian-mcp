//! Remote MCP behavior using a disposable launcher and real SFTP server packets.
//! Authentication is intentionally separate from this filesystem/protocol fixture.
#![cfg(unix)]

#[cfg(feature = "embeddings-api")]
#[path = "common/embedding_profiles.rs"]
mod embedding_profiles;
#[path = "common/http_listening.rs"]
mod http_listening;
#[path = "common/http_rpc.rs"]
mod http_rpc;

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use reqwest::Client;
use serde_json::{Value, json};
use tempfile::TempDir;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};

struct RemoteMcp {
    child: Child,
    directory: TempDir,
    vault: PathBuf,
    client: Client,
    endpoint: String,
}

impl RemoteMcp {
    fn command(read_only: bool) -> (TempDir, PathBuf, Command) {
        let server = [
            "/usr/libexec/sftp-server",
            "/usr/lib/openssh/sftp-server",
            "/usr/lib/ssh/sftp-server",
        ]
        .into_iter()
        .find(|p| Path::new(p).is_file())
        .expect("SFTP fixture needs the system OpenSSH sftp-server (install openssh-sftp-server)");
        let directory = tempfile::tempdir().unwrap();
        let vault = directory.path().join("remote");
        let machine = directory.path().join("machine");
        let bin = directory.path().join("bin");
        std::fs::create_dir_all(&vault).unwrap();
        std::fs::create_dir(&bin).unwrap();
        std::fs::write(vault.join("note.md"), "# Remote\nSFTP source content.\n").unwrap();
        let launcher = bin.join("ssh");
        let read_flags = if read_only { "-R" } else { "" };
        std::fs::write(
            &launcher,
            format!(
                r#"#!/bin/sh
for arg in "$@"; do
    if [ "$arg" = '-G' ]; then
        printf '%s\n' 'hostname fixture' 'user fixture' 'port 22'
        exit 0
    fi
done
if [ -n "$SFTP_FIXTURE_OFFLINE" ] && [ -f "$SFTP_FIXTURE_OFFLINE" ]; then
    echo 'synthetic SFTP connection outage' >&2
    exit 255
fi
if [ -n "$SFTP_FIXTURE_SESSION_PID" ]; then
    printf '%s\n' "$$" > "$SFTP_FIXTURE_SESSION_PID"
fi
exec '{server}' {read_flags}
"#
            ),
        )
        .unwrap();
        std::fs::set_permissions(&launcher, std::fs::Permissions::from_mode(0o700)).unwrap();
        let path = std::env::join_paths(std::iter::once(bin).chain(std::env::split_paths(
            &std::env::var_os("PATH").unwrap_or_default(),
        )))
        .unwrap();
        let binary = std::env::var_os("OBSIDIAN_TEST_MCP_BINARY")
            .unwrap_or_else(|| env!("CARGO_BIN_EXE_obsidian-mcp").into());
        let mut command = http_listening::isolated_command(binary);
        command
            .arg(format!("sftp://fixture{}", vault.display()))
            .env("PATH", path)
            .env("OBSIDIAN_MCP_DATA", &machine)
            .env("OBSIDIAN_WATCH", "false")
            .env("OBSIDIAN_EMBEDDINGS", "false")
            .env("OBSIDIAN_SEMANTIC_MODE", "auto")
            .env("OBSIDIAN_LOG_LEVEL", "info")
            .env("OBSIDIAN_SFTP_TIMEOUT_SECS", "5")
            .env(
                "SFTP_FIXTURE_SESSION_PID",
                directory.path().join("source-session.pid"),
            )
            .env("TOKIO_WORKER_THREADS", "1")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        (directory, vault, command)
    }

    async fn start(read_only: bool) -> Self {
        Self::start_configured(read_only, |_, _, _| {}).await
    }

    async fn start_configured(
        read_only: bool,
        configure: impl FnOnce(&Path, &Path, &mut Command),
    ) -> Self {
        let (directory, vault, mut command) = Self::command(read_only);
        configure(directory.path(), &vault, &mut command);
        let mut child = command.args(["--http", "--port", "0"]).spawn().unwrap();
        let address = http_listening::ready(&mut child).await.unwrap();
        Self {
            child,
            directory,
            vault,
            client: Client::builder()
                .no_proxy()
                .timeout(Duration::from_secs(15))
                .build()
                .unwrap(),
            endpoint: format!("http://{address}"),
        }
    }

    async fn rpc(&self, method: &str, params: Value) -> Value {
        http_rpc::rpc_response(
            http_rpc::request(
                &self.client,
                &self.endpoint,
                method,
                params,
                http_rpc::MODERN,
            )
            .send()
            .await
            .unwrap(),
        )
        .await
    }

    async fn tool(&self, name: &str, arguments: Value) -> Value {
        self.rpc("tools/call", json!({"name":name,"arguments":arguments}))
            .await
    }

    async fn ready(&self) {
        let mut last_search = Value::Null;
        for _ in 0..100 {
            let result = self.tool("search_text", json!({"query":"SFTP"})).await;
            if result.get("error").is_none() && result["result"]["isError"] != true {
                return;
            }
            last_search = result;
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        let status = self.tool("vault_info", json!({})).await;
        panic!("remote catalog never became queryable; search: {last_search}; status: {status}");
    }

    async fn disconnect_source_session(&self) {
        let session_pid = std::fs::read_to_string(self.directory.path().join("source-session.pid"))
            .unwrap()
            .trim()
            .parse::<u32>()
            .unwrap();
        let owner = Command::new("ps")
            .args(["-p", &session_pid.to_string(), "-o", "ppid="])
            .output()
            .await
            .unwrap();
        assert!(owner.status.success());
        assert_eq!(
            String::from_utf8(owner.stdout)
                .unwrap()
                .trim()
                .parse::<u32>()
                .unwrap(),
            self.child.id().unwrap(),
            "only this fixture's active SSH child may be disconnected"
        );
        assert!(
            Command::new("kill")
                .args(["-TERM", &session_pid.to_string()])
                .status()
                .await
                .unwrap()
                .success()
        );
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                let process = Command::new("ps")
                    .args(["-p", &session_pid.to_string(), "-o", "ppid="])
                    .output()
                    .await
                    .unwrap();
                if !process.status.success() || process.stdout.is_empty() {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("the disconnected fixture SSH child must be reaped");
    }

    async fn stop(mut self) {
        self.child.kill().await.unwrap();
        self.child.wait().await.unwrap();
    }
}

fn assert_success(result: &Value) {
    assert!(result.get("error").is_none(), "{result}");
    assert_ne!(result["result"]["isError"], true, "{result}");
}

#[tokio::test]
async fn remote_startup_keeps_ssh_configuration_diagnostics_without_profile_output() {
    let (directory, _vault, mut command) = RemoteMcp::command(false);
    std::fs::write(
        directory.path().join("bin/ssh"),
        r#"#!/bin/sh
for arg in "$@"; do
    if [ "$arg" = '-G' ]; then
        echo 'sensitive-profile-fixture'
        echo 'Bad configuration option: broken_fixture_option' >&2
        exit 255
    fi
done
exit 1
"#,
    )
    .unwrap();
    let mut child = command.args(["--http", "--port", "0"]).spawn().unwrap();
    let mut lines = BufReader::new(child.stderr.take().unwrap()).lines();
    let mut diagnostics = String::new();
    let observed = tokio::time::timeout(Duration::from_secs(5), async {
        while let Some(line) = lines.next_line().await.unwrap() {
            diagnostics.push_str(&line);
            diagnostics.push('\n');
            if diagnostics.contains("Bad configuration option: broken_fixture_option")
                && diagnostics.contains("255")
            {
                return;
            }
        }
        panic!("remote startup exited before reporting the SSH configuration failure");
    })
    .await;
    child.kill().await.unwrap();
    child.wait().await.unwrap();
    observed.unwrap_or_else(|_| panic!("SSH configuration diagnostics were lost: {diagnostics}"));
    assert!(
        !diagnostics.contains("sensitive-profile-fixture"),
        "{diagnostics}"
    );
}

#[tokio::test]
async fn remote_reads_search_and_mutations_keep_generated_data_local() {
    let server = RemoteMcp::start(false).await;
    let read = server.tool("note_read", json!({"path":"note.md"})).await;
    assert_success(&read);
    assert!(read.to_string().contains("SFTP source content"));
    server.ready().await;
    let list = server.rpc("tools/list", json!({})).await;
    assert!(
        !list["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .any(|t| t["name"] == "open_in_obsidian")
    );
    let filtered = http_rpc::rpc_response(
        http_rpc::request(
            &server.client,
            &server.endpoint,
            "tools/list",
            json!({}),
            http_rpc::MODERN,
        )
        .header("X-Obsidian-Tools", "open_in_obsidian")
        .send()
        .await
        .unwrap(),
    )
    .await;
    assert!(
        filtered["result"]["tools"].as_array().unwrap().is_empty(),
        "{filtered}"
    );
    let unavailable = server
        .tool("open_in_obsidian", json!({"path":"note.md"}))
        .await;
    assert!(
        unavailable.get("error").is_some() || unavailable["result"]["isError"] == true,
        "{unavailable}"
    );
    for (tool, args) in [
        (
            "note_create",
            json!({"path":"nested/new.md","content":"# New\ncontent\n"}),
        ),
        (
            "note_insert",
            json!({"path":"nested/new.md","content":"appended\n"}),
        ),
        (
            "frontmatter",
            json!({"path":"nested/new.md","action":"set","key":"status","value":"active"}),
        ),
        ("note_move", json!({"from":"nested/new.md","to":"moved.md"})),
    ] {
        assert_success(&server.tool(tool, args).await);
    }
    let content = std::fs::read_to_string(server.vault.join("moved.md")).unwrap();
    assert!(content.contains("appended"));
    assert!(content.contains("status: active"));
    assert!(!server.vault.join("nested/new.md").exists());
    std::fs::write(server.vault.join("occupied.md"), "preserved").unwrap();
    let conflict = server
        .tool("note_move", json!({"from":"moved.md","to":"occupied.md"}))
        .await;
    assert!(
        conflict.get("error").is_some() || conflict["result"]["isError"] == true,
        "{conflict}"
    );
    assert_eq!(
        std::fs::read_to_string(server.vault.join("occupied.md")).unwrap(),
        "preserved"
    );
    assert!(server.vault.join("moved.md").exists());
    assert_success(
        &server
            .tool("note_delete", json!({"path":"moved.md","confirm":true}))
            .await,
    );
    assert!(!server.vault.join("moved.md").exists());
    assert!(!server.vault.join(".obsidian-mcp").exists());
    assert!(server.directory.path().join("machine").exists());
    let outside = server.directory.path().join("outside.md");
    std::fs::write(&outside, "outside canary").unwrap();
    std::os::unix::fs::symlink(&outside, server.vault.join("escape.md")).unwrap();
    let escape = server.tool("note_read", json!({"path":"escape.md"})).await;
    assert!(
        escape.get("error").is_some() || escape["result"]["isError"] == true,
        "{escape}"
    );
    assert!(!escape.to_string().contains("outside canary"));
    assert_eq!(std::fs::read_to_string(outside).unwrap(), "outside canary");
    server.stop().await;
}

#[tokio::test]
async fn read_only_remote_account_supports_reads_without_setup_writes() {
    let server = RemoteMcp::start(true).await;
    assert_success(&server.tool("note_read", json!({"path":"note.md"})).await);
    server.ready().await;
    let result = server
        .tool(
            "note_write",
            json!({"path":"note.md","content":"replacement"}),
        )
        .await;
    assert!(
        result.get("error").is_some() || result["result"]["isError"] == true,
        "{result}"
    );
    assert_eq!(
        std::fs::read_to_string(server.vault.join("note.md")).unwrap(),
        "# Remote\nSFTP source content.\n"
    );
    assert!(!server.vault.join(".obsidian-mcp").exists());
    server.stop().await;
}

#[tokio::test]
async fn remote_bulk_reads_preserve_aliases_spelling_order_and_limits() {
    let server = RemoteMcp::start_configured(false, |_, vault, _| {
        let batch = vault.join("Batch");
        std::fs::create_dir(&batch).unwrap();
        for index in 0..12 {
            std::fs::write(
                batch.join(format!("item-{index:02}.md")),
                format!("note {index:02}\n"),
            )
            .unwrap();
        }
        std::fs::write(batch.join("cafe\u{301}.md"), "unicode\n").unwrap();
        std::fs::write(batch.join("raw:name.md"), "raw\n").unwrap();
        std::fs::write(batch.join("ignored.txt"), "not a note\n").unwrap();
        std::fs::create_dir(batch.join("directory.md")).unwrap();
        std::fs::write(vault.join("large.md"), [0xff; 32]).unwrap();
        std::fs::write(vault.join("not-inspected.md"), [0xff; 2]).unwrap();
        std::os::unix::fs::symlink("Batch/item-03.md", vault.join("file-alias.md")).unwrap();
        std::os::unix::fs::symlink("Batch", vault.join("folder-alias")).unwrap();
    })
    .await;
    server.ready().await;

    let unicode_name = std::fs::read_dir(server.vault.join("Batch"))
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .find(|name| matches!(name.as_str(), "caf\u{e9}.md" | "cafe\u{301}.md"))
        .unwrap();
    let mut entries = vec![
        (
            "folder-alias/caf\u{e9}.md".to_owned(),
            format!("Batch/{unicode_name}"),
            "unicode\n".to_owned(),
        ),
        (
            "file-alias.md".to_owned(),
            "Batch/item-03.md".to_owned(),
            "note 03\n".to_owned(),
        ),
        (
            "folder-alias/item-04.md".to_owned(),
            "Batch/item-04.md".to_owned(),
            "note 04\n".to_owned(),
        ),
        (
            "Batch/raw:name.md".to_owned(),
            "Batch/raw:name.md".to_owned(),
            "raw\n".to_owned(),
        ),
    ];
    entries.extend((0..12).rev().map(|index| {
        let path = format!("Batch/item-{index:02}.md");
        (path.clone(), path, format!("note {index:02}\n"))
    }));
    let mut paths = entries
        .iter()
        .map(|entry| entry.0.clone())
        .collect::<Vec<_>>();
    paths.insert(2, "Batch/missing.md".to_owned());
    let response = server.tool("note_read_many", json!({"paths":paths})).await;
    assert_success(&response);
    let output = &response["result"]["structuredContent"];
    let expected_notes = entries
        .iter()
        .map(|(_, path, content)| json!({"path":path,"content":content}))
        .collect::<Vec<_>>();
    assert_eq!(output["notes"], json!(expected_notes));
    assert_eq!(
        output["content_bytes"],
        entries.iter().map(|entry| entry.2.len()).sum::<usize>()
    );
    assert_eq!(output["skipped_count"], 1);
    assert_eq!(
        output["skipped"],
        json!([{"path":"Batch/missing.md","reason":"not_found"}])
    );

    let response = server
        .tool(
            "note_read_many",
            json!({"dir":"folder-alias","max_files":100}),
        )
        .await;
    assert_success(&response);
    let mut expected_paths = std::fs::read_dir(server.vault.join("Batch"))
        .unwrap()
        .map(|entry| entry.unwrap())
        .filter(|entry| entry.file_type().unwrap().is_file())
        .map(|entry| format!("Batch/{}", entry.file_name().into_string().unwrap()))
        .filter(|path| path.ends_with(".md"))
        .collect::<Vec<_>>();
    expected_paths.sort();
    let expected_notes = expected_paths
        .iter()
        .map(|path| {
            let content = std::fs::read_to_string(server.vault.join(path)).unwrap();
            json!({"path":path,"content":content})
        })
        .collect::<Vec<_>>();
    let output = &response["result"]["structuredContent"];
    assert_eq!(output["notes"], json!(expected_notes));
    assert_eq!(output["skipped_count"], 0);

    let response = server
        .tool(
            "note_read_many",
            json!({
                "paths":["large.md","Batch/item-00.md","not-inspected.md"],
                "max_files":2,
                "max_bytes":8
            }),
        )
        .await;
    assert_success(&response);
    let output = &response["result"]["structuredContent"];
    assert_eq!(
        output["notes"],
        json!([{"path":"Batch/item-00.md","content":"note 00\n"}])
    );
    assert_eq!(output["content_bytes"], 8);
    assert_eq!(output["skipped_count"], 2);
    assert_eq!(
        output["skipped"],
        json!([
            {"path":"large.md","reason":"byte_limit"},
            {"path":"not-inspected.md","reason":"file_limit"}
        ])
    );
    server.stop().await;
}

#[tokio::test]
async fn remote_tool_outputs_cover_metadata_graph_bulk_reads_and_periodic_templates() {
    let server = RemoteMcp::start_configured(false, |_, vault, _| {
        std::fs::create_dir_all(vault.join(".obsidian")).unwrap();
        std::fs::create_dir(vault.join("Templates")).unwrap();
        std::fs::write(
            vault.join(".obsidian/daily-notes.json"),
            r#"{"format":"YYYY-MM-DD","folder":"Daily","template":"Templates/Daily"}"#,
        )
        .unwrap();
        std::fs::write(
            vault.join("Templates/Daily.md"),
            "# {{date:YYYY-MM-DD}}\nRemote template\n",
        )
        .unwrap();
    })
    .await;
    server.ready().await;
    let tools = server.rpc("tools/list", json!({})).await;
    for (name, arguments) in [
        (
            "note_create",
            json!({"path":"linked.md","content":"# Heading\noriginal\n[[note]] #remote\n","frontmatter":{"status":"draft"}}),
        ),
        (
            "note_insert",
            json!({"path":"linked.md","position":"beginning","content":"intro\n"}),
        ),
        (
            "note_patch",
            json!({"path":"linked.md","operation":"append","target_type":"heading","target":"Heading","content":"patched\n"}),
        ),
        ("frontmatter", json!({"path":"linked.md","action":"get"})),
        (
            "frontmatter",
            json!({"path":"linked.md","action":"set","key":"active","value":true}),
        ),
        (
            "frontmatter",
            json!({"path":"linked.md","action":"remove","key":"active"}),
        ),
        ("note_inspect", json!({"path":"linked.md"})),
        ("note_inspect", json!({"path":"linked.md","view":"targets"})),
        (
            "note_inspect",
            json!({"path":"linked.md","fields":["headings"]}),
        ),
        (
            "note_read_many",
            json!({"paths":["linked.md","note.md","missing.md"]}),
        ),
        (
            "note_read_many",
            json!({"dir":"","recursive":true,"glob":"*.md"}),
        ),
        (
            "vault_list",
            json!({"recursive":true,"include_metadata":true}),
        ),
        ("vault_list", json!({"format":"tree"})),
        ("search_regex", json!({"pattern":"patched"})),
        ("search_metadata", json!({"type":"tag","tag":"remote"})),
        (
            "search_metadata",
            json!({"type":"frontmatter","field":"status","value":"draft"}),
        ),
        ("wikilinks", json!({"query":"backlinks","path":"note.md"})),
        ("wikilinks", json!({"query":"outgoing","path":"linked.md"})),
        ("wikilinks", json!({"query":"broken"})),
        ("wikilinks", json!({"query":"orphans"})),
        (
            "periodic",
            json!({"action":"create","period":"daily","date":"2026-11-06"}),
        ),
        (
            "periodic",
            json!({"action":"get","period":"daily","date":"2026-11-06"}),
        ),
        ("periodic", json!({"action":"list","period":"daily"})),
        ("vault_info", json!({})),
    ] {
        let response = server.tool(name, arguments).await;
        assert_success(&response);
        let tool = tools["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .find(|tool| tool["name"] == name)
            .unwrap();
        let validator = jsonschema::validator_for(&tool["outputSchema"]).unwrap();
        assert!(
            validator.is_valid(&response["result"]["structuredContent"]),
            "{name}: {response}"
        );
    }
    let linked = std::fs::read_to_string(server.vault.join("linked.md")).unwrap();
    assert!(linked.contains("intro"));
    assert!(linked.contains("patched"));
    assert!(linked.contains("status: draft"));
    assert!(!linked.contains("active:"));
    assert_eq!(
        std::fs::read_to_string(server.vault.join("Daily/2026-11-06.md")).unwrap(),
        "# 2026-11-06\nRemote template\n"
    );
    let read = server
        .tool("note_read", json!({"path":"linked.md","max_bytes":8}))
        .await;
    assert_success(&read);
    let revision = read["result"]["structuredContent"]["source_revision"]
        .as_str()
        .unwrap();
    assert_success(
        &server
            .tool(
                "note_write",
                json!({"path":"linked.md","content":"replacement"}),
            )
            .await,
    );
    let changed = server
        .tool(
            "note_read",
            json!({"path":"linked.md","expected_revision":revision,"max_bytes":8}),
        )
        .await;
    assert!(
        changed.get("error").is_some() || changed["result"]["isError"] == true,
        "{changed}"
    );
    assert_eq!(
        std::fs::read_to_string(server.vault.join("linked.md")).unwrap(),
        "replacement"
    );
    assert!(!server.vault.join(".obsidian-mcp").exists());
    server.stop().await;
}

#[tokio::test]
async fn remote_stdio_supports_discovery_then_legacy_initialization_and_reads() {
    let (_directory, _vault, mut command) = RemoteMcp::command(true);
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let stderr = child.stderr.take().unwrap();
    tokio::spawn(async move {
        let _ = tokio::io::copy(&mut BufReader::new(stderr), &mut tokio::io::sink()).await;
    });
    let mut input = child.stdin.take().unwrap();
    let mut lines = BufReader::new(child.stdout.take().unwrap()).lines();
    let messages = [
        json!({"jsonrpc":"2.0","id":0,"method":"server/discover","params":{"_meta":{
            "io.modelcontextprotocol/protocolVersion":http_rpc::MODERN,
            "io.modelcontextprotocol/clientInfo":{"name":"sftp-fixture","version":"1"},
            "io.modelcontextprotocol/clientCapabilities":{}
        }}}),
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"sftp-fixture","version":"1"}}}),
        json!({"jsonrpc":"2.0","method":"notifications/initialized"}),
        json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"note_read","arguments":{"path":"note.md"}}}),
    ];
    let mut responses = Vec::new();
    for message in messages {
        input
            .write_all(format!("{message}\n").as_bytes())
            .await
            .unwrap();
        if message.get("id").is_some() {
            let line = tokio::time::timeout(Duration::from_secs(15), lines.next_line())
                .await
                .unwrap()
                .unwrap()
                .unwrap();
            let response: Value = serde_json::from_str(&line).unwrap();
            assert!(response.get("error").is_none(), "{response}");
            responses.push(response);
        }
    }
    assert_eq!(responses[0]["result"]["resultType"], "complete");
    assert_eq!(responses[1]["result"]["protocolVersion"], "2025-11-25");
    assert_success(&responses[2]);
    assert!(responses[2].to_string().contains("SFTP source content"));
    drop(input);
    assert!(
        tokio::time::timeout(Duration::from_secs(15), child.wait())
            .await
            .unwrap()
            .unwrap()
            .success()
    );
}

#[cfg(feature = "embeddings-api")]
#[tokio::test]
async fn remote_documents_embed_locally_with_exclusions_and_revision_checked_context() {
    use axum::{Json, Router, extract::State, routing::post};
    use std::sync::{Arc, Mutex};

    async fn embeddings(
        State(requests): State<Arc<Mutex<Vec<Value>>>>,
        Json(input): Json<Value>,
    ) -> Json<Value> {
        requests.lock().unwrap().push(input.clone());
        let data: Vec<_> = input["input"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(index, _)| json!({"index":index,"embedding":[1.0,0.0]}))
            .collect();
        Json(json!({"data":data}))
    }
    let requests = Arc::new(Mutex::new(Vec::<Value>::new()));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let app = Router::new()
        .route("/v1/embeddings", post(embeddings))
        .with_state(Arc::clone(&requests));
    let provider = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let server = RemoteMcp::start_configured(false, |directory, vault, command| {
        std::fs::create_dir(vault.join("private")).unwrap();
        std::fs::write(vault.join("private/excluded.md"), "SECRET_EXCLUDED").unwrap();
        std::fs::write(vault.join(".hidden.md"), "SECRET_HIDDEN").unwrap();
        std::fs::write(
            vault.join("semantic.md"),
            "# REMOTE_VISIBLE\nOriginal source observation\n",
        )
        .unwrap();
        let launcher = directory.join("bin/ssh");
        let script = std::fs::read_to_string(&launcher).unwrap().replace(
            "\nexec ",
            "\nwhile [ ! -f \"$SFTP_FIXTURE_READY\" ]; do sleep 0.05; done\nexec ",
        );
        std::fs::write(launcher, script).unwrap();
        command
            .env("SFTP_FIXTURE_READY", directory.join("source-ready"))
            .env("SFTP_FIXTURE_OFFLINE", directory.join("source-offline"))
            .env(
                "SFTP_FIXTURE_SESSION_PID",
                directory.join("source-session.pid"),
            );
        let profile = embedding_profiles::write_api_profile(directory, port, "probe", 128, 32);
        command
            .env("OBSIDIAN_EMBEDDINGS", "true")
            .env("OBSIDIAN_WATCH", "true")
            .env("OBSIDIAN_SFTP_POLL_INTERVAL_SECS", "3600")
            .env("OBSIDIAN_SFTP_VERIFY_INTERVAL_SECS", "3600")
            .env("OBSIDIAN_EMBEDDING_PROVIDER", "api")
            .env(
                "OBSIDIAN_EMBEDDING_API_BASE",
                format!("http://127.0.0.1:{port}/v1"),
            )
            .env("OBSIDIAN_EMBEDDING_API_KEY", "synthetic-test-key")
            .env("OBSIDIAN_EMBEDDING_API_MODEL", "probe")
            .env("OBSIDIAN_EMBEDDING_API_PROFILE", profile)
            .env("OBSIDIAN_EMBEDDING_DIM", "2")
            .env("OBSIDIAN_EXCLUDE_PATHS", "private/**")
            .env("OBSIDIAN_SEMANTIC_HOME", directory.join("semantic-owner"));
    })
    .await;
    let warming = server.tool("vault_info", json!({})).await;
    assert_success(&warming);
    assert_eq!(
        warming["result"]["structuredContent"]["source"]["catalog_phase"],
        "warming"
    );
    assert_eq!(
        warming["result"]["structuredContent"]["embeddings"]["availability"],
        "unavailable"
    );
    std::fs::write(server.directory.path().join("source-ready"), "").unwrap();
    server.ready().await;
    let initial = tokio::time::timeout(Duration::from_secs(20), async {
        loop {
            let result = server
                .tool(
                    "search_semantic",
                    json!({"query":"REMOTE_VISIBLE","top_k":10,"include_content":true}),
                )
                .await;
            if result.get("error").is_none()
                && result["result"]["isError"] != true
                && result["result"]["structuredContent"]["results"]
                    .as_array()
                    .is_some_and(|results| results.iter().any(|hit| hit["path"] == "semantic.md"))
            {
                break result;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await
    .unwrap();
    let hit = initial["result"]["structuredContent"]["results"]
        .as_array()
        .unwrap()
        .iter()
        .find(|hit| hit["path"] == "semantic.md")
        .unwrap();
    assert_eq!(hit["source_status"], "current");
    assert!(
        hit["content"]
            .as_str()
            .unwrap()
            .contains("Original source observation")
    );
    let revision = hit["source_revision"].clone();
    let inputs = requests
        .lock()
        .unwrap()
        .iter()
        .flat_map(|request| request["input"].as_array().unwrap().clone())
        .collect::<Vec<_>>();
    assert!(inputs.iter().any(|text| {
        text.as_str()
            .unwrap()
            .contains("Original source observation")
    }));
    assert!(
        !inputs
            .iter()
            .any(|text| text.as_str().unwrap().contains("SECRET_")),
        "{inputs:?}"
    );
    assert!(
        !server.directory.path().join("semantic-owner").exists(),
        "remote auto mode must not bootstrap a daemon"
    );
    std::fs::write(
        server.vault.join("semantic.md"),
        "# REMOTE_VISIBLE\nChanged source observation\n",
    )
    .unwrap();
    let changed = server
        .tool(
            "search_semantic",
            json!({"query":"REMOTE_VISIBLE","top_k":10,"include_content":true}),
        )
        .await;
    assert_success(&changed);
    let hit = changed["result"]["structuredContent"]["results"]
        .as_array()
        .unwrap()
        .iter()
        .find(|hit| hit["path"] == "semantic.md")
        .unwrap();
    assert_eq!(hit["source_revision"], revision);
    assert_eq!(hit["source_status"], "changed");
    assert!(hit.get("content").is_none());
    assert!(hit.get("snippet").is_none());
    assert!(hit.get("context_error").is_some());
    std::fs::remove_file(server.vault.join("semantic.md")).unwrap();
    let unavailable = server
        .tool(
            "search_semantic",
            json!({"query":"REMOTE_VISIBLE","top_k":10,"include_content":true}),
        )
        .await;
    assert_success(&unavailable);
    let hit = unavailable["result"]["structuredContent"]["results"]
        .as_array()
        .unwrap()
        .iter()
        .find(|hit| hit["path"] == "semantic.md")
        .unwrap();
    assert_eq!(hit["source_revision"], revision);
    assert_eq!(hit["source_status"], "unavailable");
    assert!(hit.get("content").is_none());
    assert!(hit.get("snippet").is_none());
    assert!(hit.get("context_error").is_some());

    std::fs::write(server.directory.path().join("source-offline"), "").unwrap();
    server.disconnect_source_session().await;
    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            let disconnected = server.tool("note_read", json!({"path":"note.md"})).await;
            if disconnected.get("error").is_some() || disconnected["result"]["isError"] == true {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    let status = server.tool("vault_info", json!({})).await;
    assert_success(&status);
    assert_eq!(
        status["result"]["structuredContent"]["source"]["availability"], "unavailable",
        "{status}"
    );
    let outage = server
        .tool(
            "search_semantic",
            json!({"query":"REMOTE_VISIBLE","top_k":10,"include_content":true}),
        )
        .await;
    assert_success(&outage);
    let hit = outage["result"]["structuredContent"]["results"]
        .as_array()
        .unwrap()
        .iter()
        .find(|hit| hit["path"] == "semantic.md")
        .expect("an ordinary source outage must retain the ranked semantic hit");
    assert_eq!(hit["source_revision"], revision);
    assert_eq!(hit["source_status"], "unavailable");
    assert!(hit.get("content").is_none());
    assert!(hit.get("snippet").is_none());
    assert!(hit.get("context_error").is_some());
    assert!(!server.vault.join(".obsidian-mcp").exists());
    server.stop().await;
    provider.abort();
}

#[cfg(feature = "embeddings-api")]
#[tokio::test]
async fn remote_semantic_queries_fail_when_source_policy_is_withdrawn_in_flight() {
    use axum::{Json, Router, extract::State, routing::post};
    use std::sync::Arc;
    use tokio::sync::{Notify, watch};

    const HELD_QUERY: &str = "heldauthoritymarker";

    #[derive(Clone)]
    struct ProviderState {
        received: Arc<Notify>,
        hold: watch::Receiver<bool>,
    }

    async fn embeddings(
        State(mut state): State<ProviderState>,
        Json(input): Json<Value>,
    ) -> Json<Value> {
        let inputs = input["input"].as_array().unwrap();
        if inputs
            .iter()
            .any(|value| value.as_str() == Some(HELD_QUERY))
        {
            state.received.notify_one();
            state.hold.wait_for(|hold| !*hold).await.unwrap();
        }
        let data: Vec<_> = inputs
            .iter()
            .enumerate()
            .map(|(index, _)| json!({"index":index,"embedding":[1.0,0.0]}))
            .collect();
        Json(json!({"data":data}))
    }

    for (lexical_prefetch, during_render) in [(false, false), (true, false), (false, true)] {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let (hold, receiver) = watch::channel(false);
        let received = Arc::new(Notify::new());
        let app = Router::new()
            .route("/v1/embeddings", post(embeddings))
            .with_state(ProviderState {
                received: Arc::clone(&received),
                hold: receiver,
            });
        let provider = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let server = RemoteMcp::start_configured(false, |directory, vault, command| {
            std::fs::write(
                vault.join("semantic.md"),
                format!("# Semantic source\n{HELD_QUERY} source observation.\n"),
            )
            .unwrap();
            let profile = embedding_profiles::write_api_profile(directory, port, "probe", 128, 32);
            command
                .env("SFTP_FIXTURE_OFFLINE", directory.join("source-offline"))
                .env("OBSIDIAN_EMBEDDINGS", "true")
                .env("OBSIDIAN_EMBEDDING_PROVIDER", "api")
                .env(
                    "OBSIDIAN_EMBEDDING_API_BASE",
                    format!("http://127.0.0.1:{port}/v1"),
                )
                .env("OBSIDIAN_EMBEDDING_API_KEY", "synthetic-test-key")
                .env("OBSIDIAN_EMBEDDING_API_MODEL", "probe")
                .env("OBSIDIAN_EMBEDDING_API_PROFILE", profile)
                .env("OBSIDIAN_EMBEDDING_DIM", "2");
        })
        .await;
        server.ready().await;
        tokio::time::timeout(Duration::from_secs(20), async {
            loop {
                let result = server
                    .tool("search_semantic", json!({"query":"source observation"}))
                    .await;
                if result.get("error").is_none()
                    && result["result"]["isError"] != true
                    && result["result"]["structuredContent"]["results"]
                        .as_array()
                        .is_some_and(|results| {
                            results.iter().any(|hit| hit["path"] == "semantic.md")
                        })
                {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
        })
        .await
        .expect("the remote semantic observation must be queryable before holding inference");

        hold.send(true).unwrap();
        let request = http_rpc::request(
            &server.client,
            &server.endpoint,
            "tools/call",
            json!({
                "name":"search_semantic",
                "arguments":{
                    "query":HELD_QUERY,
                    "include_content":true,
                    "lexical_prefetch":lexical_prefetch
                }
            }),
            http_rpc::MODERN,
        );
        let query =
            tokio::spawn(
                async move { http_rpc::rpc_response(request.send().await.unwrap()).await },
            );
        tokio::time::timeout(Duration::from_secs(5), received.notified())
            .await
            .expect("query inference must reach the held provider");
        assert!(!query.is_finished());

        let offline = server.directory.path().join("source-offline");
        if during_render {
            std::fs::write(&offline, "").unwrap();
        } else {
            std::fs::create_dir(server.vault.join(".obsidian-mcp")).unwrap();
            std::fs::write(server.vault.join(".obsidian-mcp/ignore"), "semantic.md\n").unwrap();
        }
        server.disconnect_source_session().await;
        tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                let read = server.tool("note_read", json!({"path":"note.md"})).await;
                if read.get("error").is_some() || read["result"]["isError"] == true {
                    let message = read.to_string();
                    if during_render {
                        assert!(
                            !message.contains("portable indexing policy changed"),
                            "{read}"
                        );
                        break;
                    }
                    if message.contains("portable indexing policy changed") {
                        break;
                    }
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("the disconnected source must report the expected authority state");
        if during_render {
            let status = server.tool("vault_info", json!({})).await;
            assert_success(&status);
            assert_eq!(
                status["result"]["structuredContent"]["source"]["availability"], "unavailable",
                "{status}"
            );
            std::fs::create_dir(server.vault.join(".obsidian-mcp")).unwrap();
            std::fs::write(server.vault.join(".obsidian-mcp/ignore"), "semantic.md\n").unwrap();
            std::fs::remove_file(offline).unwrap();
        }

        hold.send(false).unwrap();
        let response = query.await.unwrap();
        assert!(
            response.get("error").is_some() || response["result"]["isError"] == true,
            "withdrawn source evidence must fail (hybrid={lexical_prefetch}, during_render={during_render}): {response}"
        );
        assert!(
            response
                .to_string()
                .contains("portable indexing policy changed"),
            "{response}"
        );
        assert!(
            response["result"].get("structuredContent").is_none(),
            "{response}"
        );
        server.stop().await;
        provider.abort();
    }
}

#[tokio::test]
async fn source_warming_status_stays_passive_and_keeps_aggregate_counts_unknown() {
    let server = RemoteMcp::start_configured(false, |directory, _, command| {
        let launcher = directory.join("bin/ssh");
        let script = std::fs::read_to_string(&launcher).unwrap().replace(
            "\nexec ",
            "\nwhile [ ! -f \"$SFTP_FIXTURE_READY\" ]; do sleep 0.05; done\nexec ",
        );
        std::fs::write(launcher, script).unwrap();
        command.env("SFTP_FIXTURE_READY", directory.join("source-ready"));
    })
    .await;
    let response =
        tokio::time::timeout(Duration::from_secs(2), server.tool("vault_info", json!({})))
            .await
            .unwrap();
    assert_success(&response);
    let info = &response["result"]["structuredContent"];
    assert_eq!(info["source"]["kind"], "sftp");
    assert_eq!(info["source"]["catalog_phase"], "warming");
    for field in [
        "total_notes",
        "total_files",
        "total_tags",
        "total_links",
        "vault_size_bytes",
    ] {
        assert!(
            info.get(field).is_none(),
            "unknown {field} became a value: {info}"
        );
    }
    assert!(!server.directory.path().join("source-ready").exists());
    std::fs::write(server.directory.path().join("source-ready"), "").unwrap();
    server.ready().await;
    let response = server.tool("vault_info", json!({})).await;
    assert_success(&response);
    let info = &response["result"]["structuredContent"];
    assert_eq!(info["source"]["catalog_phase"], "ready");
    assert_eq!(info["total_notes"], 1);
    server.stop().await;
}

#[tokio::test]
async fn startup_connection_failure_recovers_without_waiting_for_external_polling() {
    let server = RemoteMcp::start_configured(false, |directory, _, command| {
        let offline = directory.join("source-offline");
        std::fs::write(&offline, "").unwrap();
        command
            .env("SFTP_FIXTURE_OFFLINE", offline)
            .env("OBSIDIAN_WATCH", "false")
            .env("OBSIDIAN_SFTP_POLL_INTERVAL_SECS", "86400")
            .env("OBSIDIAN_SFTP_VERIFY_INTERVAL_SECS", "86400");
    })
    .await;
    let unavailable = tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            let status = server.tool("vault_info", json!({})).await;
            assert_success(&status);
            if status["result"]["structuredContent"]["source"]["availability"] == "unavailable" {
                break status;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("the initial connection failure must remain visible through passive status");
    assert!(
        unavailable["result"]["structuredContent"]["source"]["error"]
            .as_str()
            .is_some_and(|error| !error.is_empty()),
        "{unavailable}"
    );
    for _ in 0..2 {
        let status =
            tokio::time::timeout(Duration::from_secs(1), server.tool("vault_info", json!({})))
                .await
                .expect("passive status must not wait for a new source connection");
        assert_success(&status);
        let info = &status["result"]["structuredContent"];
        assert_eq!(info["source"]["availability"], "unavailable", "{info}");
        assert_eq!(info["source"]["catalog_phase"], "degraded", "{info}");
        assert!(info["source"]["error"].is_string(), "{info}");
        assert!(info["source"].get("last_scan").is_none(), "{info}");
        assert!(info.get("total_notes").is_none(), "{info}");
    }

    std::fs::remove_file(server.directory.path().join("source-offline")).unwrap();
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let read = server.tool("note_read", json!({"path":"note.md"})).await;
            if read.get("error").is_none() && read["result"]["isError"] != true {
                assert!(read.to_string().contains("SFTP source content"), "{read}");
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        loop {
            let status = server.tool("vault_info", json!({})).await;
            assert_success(&status);
            let info = &status["result"]["structuredContent"];
            if info["source"]["availability"] == "available"
                && info["source"]["catalog_phase"] == "ready"
            {
                assert_eq!(info["total_notes"], 1, "{info}");
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("startup recovery must use the retained initializer without the one-day poll interval");
    let search = server.tool("search_text", json!({"query":"SFTP"})).await;
    assert_success(&search);
    assert_eq!(
        search["result"]["structuredContent"]["results"]
            .as_array()
            .unwrap()
            .len(),
        1,
        "{search}"
    );
    server.stop().await;
}

#[tokio::test]
async fn source_outage_after_indexing_remains_visible_through_passive_status() {
    let server = RemoteMcp::start_configured(false, |_, _, command| {
        command
            .env("OBSIDIAN_WATCH", "true")
            .env("OBSIDIAN_SFTP_POLL_INTERVAL_SECS", "1")
            .env("OBSIDIAN_SFTP_VERIFY_INTERVAL_SECS", "1");
    })
    .await;
    server.ready().await;
    let before = server.tool("vault_info", json!({})).await;
    assert_success(&before);
    let scan = before["result"]["structuredContent"]["source"]["last_scan"].clone();
    assert!(!scan.is_null(), "{before}");
    std::fs::rename(&server.vault, server.directory.path().join("offline")).unwrap();
    let unavailable = tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            let response = server.tool("vault_info", json!({})).await;
            assert_success(&response);
            if response["result"]["structuredContent"]["source"]["availability"] == "unavailable" {
                break response;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await
    .unwrap();
    let info = &unavailable["result"]["structuredContent"];
    assert_eq!(info["source"]["last_scan"], scan);
    assert!(info["source"]["error"].is_string(), "{info}");
    assert!(info.get("total_notes").is_none(), "{info}");
    server.stop().await;
}

#[tokio::test]
async fn successful_reconnect_refreshes_catalog_without_external_polling() {
    let server = RemoteMcp::start_configured(false, |_, vault, command| {
        std::fs::write(vault.join("note.md"), "# Remote\nSFTP oldreconnectmarker\n").unwrap();
        std::fs::create_dir(vault.join(".obsidian-mcp")).unwrap();
        command.env("OBSIDIAN_WATCH", "false");
    })
    .await;
    server.ready().await;
    let original = server
        .tool("search_text", json!({"query":"oldreconnectmarker"}))
        .await;
    assert_success(&original);
    assert_eq!(
        original["result"]["structuredContent"]["results"]
            .as_array()
            .unwrap()
            .len(),
        1
    );

    let note = server.vault.join("note.md");
    let before = std::fs::metadata(&note).unwrap();
    let modified = before.modified().unwrap();
    std::fs::write(&note, "# Remote\nSFTP newreconnectmarker\n").unwrap();
    std::fs::OpenOptions::new()
        .write(true)
        .open(&note)
        .unwrap()
        .set_modified(modified)
        .unwrap();
    assert_eq!(std::fs::metadata(&note).unwrap().len(), before.len());
    server.disconnect_source_session().await;
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let read = server.tool("note_read", json!({"path":"note.md"})).await;
            if read.get("error").is_none() && read["result"]["isError"] != true {
                assert!(read.to_string().contains("newreconnectmarker"), "{read}");
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        loop {
            let result = server
                .tool("search_text", json!({"query":"newreconnectmarker"}))
                .await;
            let status = server.tool("vault_info", json!({})).await;
            assert_success(&status);
            if result.get("error").is_none()
                && result["result"]["isError"] != true
                && result["result"]["structuredContent"]["results"]
                    .as_array()
                    .is_some_and(|hits| hits.len() == 1)
                && status["result"]["structuredContent"]["source"]["availability"] == "available"
                && status["result"]["structuredContent"]["source"]["catalog_phase"] == "ready"
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("a successful reconnect must refresh the catalog even with watching disabled");
    let old = server
        .tool("search_text", json!({"query":"oldreconnectmarker"}))
        .await;
    assert_success(&old);
    assert_eq!(old["result"]["structuredContent"]["results"], json!([]));
    assert!(!server.vault.join(".obsidian-mcp/ignore").exists());
    server.stop().await;
}

#[tokio::test]
async fn malformed_remote_notes_do_not_block_valid_catalog_entries_and_recover_on_poll() {
    let server = RemoteMcp::start_configured(false, |_, vault, command| {
        std::fs::write(
            vault.join("malformed.md"),
            "---\ntags: [unterminated\n---\ninvalid projection marker\n",
        )
        .unwrap();
        std::fs::write(vault.join("invalid-utf8.md"), [0xff, 0xfe]).unwrap();
        command
            .env("OBSIDIAN_WATCH", "true")
            .env("OBSIDIAN_SFTP_POLL_INTERVAL_SECS", "1")
            .env("OBSIDIAN_SFTP_VERIFY_INTERVAL_SECS", "1");
    })
    .await;
    server.ready().await;
    let info = server.tool("vault_info", json!({})).await;
    assert_success(&info);
    assert_eq!(
        info["result"]["structuredContent"]["total_notes"], 1,
        "{info}"
    );
    assert_eq!(
        info["result"]["structuredContent"]["source"]["availability"],
        "available"
    );
    assert_eq!(
        info["result"]["structuredContent"]["source"]["catalog_phase"],
        "degraded"
    );
    assert!(info["result"]["structuredContent"]["source"]["error"].is_string());
    let bulk = server
        .tool(
            "note_read_many",
            json!({"paths":["note.md","invalid-utf8.md"]}),
        )
        .await;
    assert_success(&bulk);
    assert_eq!(
        bulk["result"]["structuredContent"]["notes"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    assert_eq!(bulk["result"]["structuredContent"]["skipped_count"], 1);
    assert_eq!(
        bulk["result"]["structuredContent"]["skipped"][0]["reason"],
        "unreadable"
    );
    let invalid = server
        .tool("search_text", json!({"query":"invalid projection marker"}))
        .await;
    assert_success(&invalid);
    assert_eq!(invalid["result"]["structuredContent"]["results"], json!([]));
    std::fs::write(
        server.vault.join("malformed.md"),
        "# Corrected\nrecovered projection marker\n",
    )
    .unwrap();
    std::fs::write(
        server.vault.join("invalid-utf8.md"),
        "# Corrected UTF8\nrecovered projection marker\n",
    )
    .unwrap();
    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            let result = server
                .tool(
                    "search_text",
                    json!({"query":"recovered projection marker"}),
                )
                .await;
            assert_success(&result);
            if result["result"]["structuredContent"]["results"]
                .as_array()
                .unwrap()
                .len()
                == 2
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await
    .unwrap();
    let info = tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            let response = server.tool("vault_info", json!({})).await;
            assert_success(&response);
            if response["result"]["structuredContent"]["source"]["catalog_phase"] == "ready" {
                break response;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await
    .unwrap();
    assert_eq!(
        info["result"]["structuredContent"]["total_notes"], 3,
        "{info}"
    );
    assert_eq!(
        info["result"]["structuredContent"]["source"]["catalog_phase"],
        "ready"
    );
    assert!(
        info["result"]["structuredContent"]["source"]
            .get("error")
            .is_none()
    );
    server.stop().await;
}
