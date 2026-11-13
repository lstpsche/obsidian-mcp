//! Owner administration and explicit connection checks through the shipped binary.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use obsidian_mcp::access::store::{AccessStore, read_token_file};
use serde_json::Value;

fn binary() -> PathBuf {
    std::env::var_os("OBSIDIAN_TEST_MCP_BINARY")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_BIN_EXE_obsidian-mcp")))
}

fn command() -> Command {
    command_at(&binary())
}

fn command_at(executable: &Path) -> Command {
    let mut command = Command::new(executable);
    for (key, _) in std::env::vars_os() {
        let name = key.to_string_lossy();
        if name.starts_with("OBSIDIAN_") || name.starts_with("OPENAI_") {
            command.env_remove(key);
        }
    }
    command.env("NO_COLOR", "1").arg("access");
    command
}

fn success(output: Output) -> Value {
    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("JSON command should return JSON")
}

struct Fixture {
    _directory: tempfile::TempDir,
    vault: PathBuf,
    profile: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let vault = directory.path().join("vault");
        std::fs::create_dir_all(vault.join("Shared")).unwrap();
        std::fs::create_dir_all(vault.join("Private")).unwrap();
        std::fs::write(vault.join("Shared/note.md"), "shared fixture").unwrap();
        std::fs::write(vault.join("Private/secret.md"), "private fixture").unwrap();
        let profile = directory.path().join("access/access.json");
        Self {
            _directory: directory,
            vault,
            profile,
        }
    }

    fn initialize(&self) -> Value {
        success(
            command()
                .arg("init")
                .arg(&self.vault)
                .arg("--access-file")
                .arg(&self.profile)
                .arg("--json")
                .output()
                .unwrap(),
        )
    }

    fn command(&self) -> Command {
        let mut command = command();
        command.arg("--access-file").arg(&self.profile);
        command
    }

    fn add(&self, name: &str, editor: bool) -> Value {
        let mut command = self.command();
        command
            .args(["client", "add", name, "--directory", "Shared/"])
            .args(["--role", if editor { "editor" } else { "reader" }])
            .arg("--json");
        success(command.output().unwrap())
    }

    fn check(&self, client: &str, action: &str, path: &str) -> Value {
        success(
            self.command()
                .args([
                    "check", client, "--action", action, "--path", path, "--json",
                ])
                .output()
                .unwrap(),
        )
    }
}

#[test]
fn help_does_not_require_a_vault_or_valid_profile() {
    let output = command()
        .arg("--help")
        .env("OBSIDIAN_ACCESS_FILE", "/missing/unsafe/profile")
        .env("OBSIDIAN_EMBEDDINGS", "true")
        .output()
        .unwrap();
    assert!(output.status.success());
    let help = String::from_utf8(output.stdout).unwrap();
    assert!(help.contains("access setup VAULT"));
    assert!(help.contains("Creating a profile does not change existing servers"));
    assert!(help.contains("headers --token-file PATH"));
}

#[test]
fn initialization_creates_external_state_without_touching_the_source() {
    let fixture = Fixture::new();
    let result = fixture.initialize();
    assert!(fixture.profile.exists());
    assert_eq!(result["revision"], 1);
    assert!(!fixture.vault.join(".obsidian-mcp").exists());
    assert!(!fixture.vault.join(".obsidian").exists());
    assert_eq!(std::fs::read_dir(&fixture.vault).unwrap().count(), 2);
    let second = fixture
        .command()
        .arg("init")
        .arg(&fixture.vault)
        .output()
        .unwrap();
    assert!(!second.status.success());
    assert!(String::from_utf8_lossy(&second.stderr).contains("access client add NAME"));
    assert_eq!(
        AccessStore::open(&fixture.profile)
            .unwrap()
            .snapshot()
            .unwrap()
            .policy
            .revision,
        1
    );
}

#[test]
fn initialization_dry_run_creates_no_profile_or_machine_data() {
    let fixture = Fixture::new();
    let output = success(
        command()
            .arg("init")
            .arg(&fixture.vault)
            .arg("--access-file")
            .arg(&fixture.profile)
            .args(["--dry-run", "--json"])
            .output()
            .unwrap(),
    );
    assert_eq!(output["dry_run"], true);
    assert!(!fixture.profile.parent().unwrap().exists());
    assert!(!fixture.vault.join(".obsidian-mcp").exists());
}

#[test]
fn reader_default_and_explicit_denial_are_enforced_by_owner_checks() {
    let fixture = Fixture::new();
    fixture.initialize();
    fixture.add("reader", false);
    assert_eq!(
        fixture.check("reader", "read", "Shared/note.md")["decision"]["allowed"],
        true
    );
    assert_eq!(
        fixture.check("reader", "update", "Shared/note.md")["decision"]["allowed"],
        false
    );
    assert_eq!(
        fixture.check("reader", "read", "Private/secret.md")["decision"]["allowed"],
        false
    );
    success(
        fixture
            .command()
            .args([
                "grant",
                "reader",
                "--directory",
                "",
                "--role",
                "editor",
                "--json",
            ])
            .output()
            .unwrap(),
    );
    let denied = success(
        fixture
            .command()
            .args(["deny", "reader", "--directory", "Private/", "--json"])
            .output()
            .unwrap(),
    );
    assert!(denied["revision"].as_u64().unwrap() > 1);
    let check = fixture.check("reader", "read", "Private/secret.md");
    assert_eq!(check["decision"]["allowed"], false);
    assert_eq!(check["decision"]["reason"], "denied");
    assert!(check["decision"]["rule_id"].as_str().is_some());
}

#[test]
fn dry_run_and_revision_conflict_do_not_publish_permissions_or_tokens() {
    let fixture = Fixture::new();
    fixture.initialize();
    fixture.add("reader", false);
    let before = std::fs::read(&fixture.profile).unwrap();
    let token = fixture.profile.parent().unwrap().join("not-created.token");
    success(
        fixture
            .command()
            .args([
                "client",
                "add",
                "preview",
                "--directory",
                "",
                "--role",
                "editor",
                "--dry-run",
                "--json",
                "--token-out",
            ])
            .arg(&token)
            .output()
            .unwrap(),
    );
    assert_eq!(std::fs::read(&fixture.profile).unwrap(), before);
    assert!(!token.exists());
    let conflict = fixture
        .command()
        .args([
            "grant",
            "reader",
            "--directory",
            "",
            "--role",
            "editor",
            "--expected-revision",
            "1",
        ])
        .output()
        .unwrap();
    assert!(!conflict.status.success());
    assert!(String::from_utf8_lossy(&conflict.stderr).contains("revision conflict"));
    assert_eq!(std::fs::read(&fixture.profile).unwrap(), before);
}

#[test]
fn credentials_overlap_until_exact_revocation_without_secret_output() {
    let fixture = Fixture::new();
    fixture.initialize();
    let first = fixture.add("editor", true);
    let first_path = Path::new(first["token_file"].as_str().unwrap());
    let first_token = read_token_file(first_path).unwrap();
    let second = success(
        fixture
            .command()
            .args(["token", "create", "editor", "--json"])
            .output()
            .unwrap(),
    );
    let second_token = read_token_file(Path::new(second["token_file"].as_str().unwrap())).unwrap();
    let store = AccessStore::open(&fixture.profile).unwrap();
    assert!(store.authenticate(first_token.expose()).is_ok());
    assert!(store.authenticate(second_token.expose()).is_ok());
    let list = fixture
        .command()
        .args(["client", "list", "--json"])
        .output()
        .unwrap();
    assert!(list.status.success());
    let displayed = String::from_utf8(list.stdout).unwrap();
    assert!(!displayed.contains(first_token.expose()));
    assert!(!displayed.contains(second_token.expose()));
    assert!(!displayed.contains("digest"));
    assert!(
        !std::fs::read_to_string(&fixture.profile)
            .unwrap()
            .contains(first_token.expose())
    );
    success(
        fixture
            .command()
            .args([
                "token",
                "revoke",
                "editor",
                first["credential_id"].as_str().unwrap(),
                "--json",
            ])
            .output()
            .unwrap(),
    );
    assert!(store.authenticate(first_token.expose()).is_err());
    assert!(store.authenticate(second_token.expose()).is_ok());
}

#[test]
fn helper_is_the_explicit_secret_output_and_needs_no_owner_profile() {
    let fixture = Fixture::new();
    fixture.initialize();
    let added = fixture.add("reader", false);
    let token_path = Path::new(added["token_file"].as_str().unwrap());
    let token = read_token_file(token_path).unwrap();
    let result = success(
        command()
            .args(["headers", "--token-file"])
            .arg(token_path)
            .env("OBSIDIAN_ACCESS_FILE", "/missing/profile")
            .output()
            .unwrap(),
    );
    assert_eq!(result.as_object().unwrap().len(), 1);
    assert_eq!(
        result["Authorization"],
        format!("Bearer {}", token.expose())
    );
}

#[test]
fn generated_configuration_contains_file_helper_not_plaintext_and_never_overwrites() {
    let fixture = Fixture::new();
    fixture.initialize();
    let added = fixture.add("reader", false);
    let token_path = Path::new(added["token_file"].as_str().unwrap());
    let token = read_token_file(token_path).unwrap();
    let out = fixture.profile.parent().unwrap().join("client.json");
    let result = success(
        fixture
            .command()
            .args([
                "config",
                "reader",
                "--client",
                "claude-code",
                "--token-file",
            ])
            .arg(token_path)
            .arg("--out")
            .arg(&out)
            .arg("--json")
            .output()
            .unwrap(),
    );
    assert_eq!(result["adapter"], "claude-code");
    let bytes = std::fs::read(&out).unwrap();
    let configuration: Value = serde_json::from_slice(&bytes).unwrap();
    let helper = configuration["mcpServers"]["reader"]["headersHelper"]
        .as_str()
        .unwrap();
    assert!(helper.contains("access headers --token-file"));
    assert!(!String::from_utf8_lossy(&bytes).contains(token.expose()));
    let repeat = fixture
        .command()
        .args([
            "config",
            "reader",
            "--client",
            "claude-code",
            "--token-file",
        ])
        .arg(token_path)
        .arg("--out")
        .arg(&out)
        .output()
        .unwrap();
    assert!(!repeat.status.success());
    assert_eq!(std::fs::read(&out).unwrap(), bytes);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            std::fs::metadata(&out).unwrap().permissions().mode() & 0o077,
            0
        );
    }
}

#[test]
fn setup_generates_explicit_optional_launch_and_client_configuration() {
    let fixture = Fixture::new();
    let out = fixture.profile.parent().unwrap().join("assistant.toml");
    let result = success(
        command()
            .arg("setup")
            .arg(&fixture.vault)
            .arg("--access-file")
            .arg(&fixture.profile)
            .args([
                "--name",
                "assistant",
                "--directory",
                "Shared",
                "--client",
                "codex",
                "--out",
            ])
            .arg(&out)
            .arg("--json")
            .output()
            .unwrap(),
    );
    assert_eq!(result["client_id"], "assistant");
    assert_eq!(result["transport"], "http");
    assert!(out.exists());
    assert!(!fixture.vault.join(".obsidian-mcp").exists());
    let config = std::fs::read_to_string(&out).unwrap();
    assert!(config.contains("http_headers_helper"));
    let check = fixture.check("assistant", "update", "Shared/note.md");
    assert_eq!(check["decision"]["allowed"], false);
}

#[test]
fn stdio_configuration_selects_the_profile_and_protected_token_explicitly() {
    let fixture = Fixture::new();
    fixture.initialize();
    let added = fixture.add("reader", false);
    let out = fixture.profile.parent().unwrap().join("stdio.json");
    success(
        fixture
            .command()
            .args([
                "config",
                "reader",
                "--client",
                "claude-code",
                "--transport",
                "stdio",
                "--token-file",
            ])
            .arg(added["token_file"].as_str().unwrap())
            .arg("--out")
            .arg(&out)
            .arg("--json")
            .output()
            .unwrap(),
    );
    let config: Value = serde_json::from_slice(&std::fs::read(out).unwrap()).unwrap();
    let arguments = config["mcpServers"]["reader"]["args"].as_array().unwrap();
    assert!(arguments.contains(&Value::String("--access-file".into())));
    assert!(arguments.contains(&Value::String("--access-token-file".into())));
    assert!(config["mcpServers"]["reader"]["headersHelper"].is_null());
}

#[test]
fn unsupported_or_misplaced_options_fail_before_changing_policy() {
    let fixture = Fixture::new();
    fixture.initialize();
    fixture.add("reader", false);
    let before = std::fs::read(&fixture.profile).unwrap();
    let output = fixture
        .command()
        .args([
            "token",
            "revoke",
            "reader",
            "does-not-exist",
            "--directory",
            "Shared",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("not valid"));
    assert_eq!(std::fs::read(&fixture.profile).unwrap(), before);
}

#[path = "common/http_listening.rs"]
mod http_listening;

#[tokio::test]
async fn doctor_checks_authenticated_mcp_and_explicit_disposable_writes() {
    use std::process::Stdio;

    let fixture = Fixture::new();
    fixture.initialize();
    let added = fixture.add("editor", true);
    let token_file = Path::new(added["token_file"].as_str().unwrap());
    let mut server = http_listening::isolated_command(binary())
        .arg(&fixture.vault)
        .args([
            "--http",
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--access-file",
        ])
        .arg(&fixture.profile)
        .env("OBSIDIAN_EMBEDDINGS", "false")
        .env("OBSIDIAN_WATCH", "false")
        .env("OBSIDIAN_LOG_LEVEL", "info")
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    let address = http_listening::ready(&mut server).await.unwrap();
    let output = http_listening::isolated_command(binary())
        .args(["access", "doctor", "--access-file"])
        .arg(&fixture.profile)
        .arg("--token-file")
        .arg(token_file)
        .args([
            "--url",
            &format!("http://{address}/mcp"),
            "--path",
            "Shared/note.md",
            "--write-check",
            "Shared/disposable-check.md",
            "--json",
        ])
        .output()
        .await
        .unwrap();
    server.kill().await.unwrap();
    server.wait().await.unwrap();
    assert!(!String::from_utf8_lossy(&output.stdout).contains("shared fixture"));
    let result = success(output);
    assert_eq!(result["ok"], true);
    assert_eq!(result["real_client_verified"], false);
    assert!(!fixture.vault.join("Shared/disposable-check.md").exists());
    assert!(
        result["checks"]
            .as_array()
            .unwrap()
            .iter()
            .any(|check| check == "explicit disposable create/read/delete passed")
    );
}

#[test]
fn dry_run_rejects_unsafe_state_and_configuration_stays_outside_the_vault() {
    let fixture = Fixture::new();
    let unsafe_profile = fixture.vault.join("Shared/access.json");
    let output = command()
        .arg("init")
        .arg(&fixture.vault)
        .arg("--access-file")
        .arg(&unsafe_profile)
        .arg("--dry-run")
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(!unsafe_profile.exists());
    fixture.initialize();
    let added = fixture.add("reader", false);
    let unsafe_config = fixture.vault.join("Shared/client.json");
    let output = fixture
        .command()
        .args(["config", "reader", "--token-file"])
        .arg(added["token_file"].as_str().unwrap())
        .arg("--out")
        .arg(&unsafe_config)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(!unsafe_config.exists());
}

#[test]
fn remote_windows_helper_uses_explicit_shell_and_client_local_paths() {
    let fixture = Fixture::new();
    fixture.initialize();
    let added = fixture.add("reader", false);
    let out = fixture
        .profile
        .parent()
        .unwrap()
        .join("windows-client.json");
    success(
        fixture
            .command()
            .args([
                "config",
                "reader",
                "--client",
                "claude-code",
                "--token-file",
            ])
            .arg(added["token_file"].as_str().unwrap())
            .args([
                "--binary",
                "C:\\Program Files\\Obsidian MCP\\obsidian-mcp.exe",
                "--client-token-file",
                "C:\\Users\\Owner\\Private\\reader.token",
                "--out",
            ])
            .arg(&out)
            .arg("--json")
            .output()
            .unwrap(),
    );
    let value: Value = serde_json::from_slice(&std::fs::read(out).unwrap()).unwrap();
    let helper = value["mcpServers"]["reader"]["headersHelper"]
        .as_str()
        .unwrap();
    assert!(helper.starts_with("powershell.exe -NoLogo -NoProfile -NonInteractive -Command"));
    assert!(helper.contains("& 'C:\\Program Files\\Obsidian MCP\\obsidian-mcp.exe'"));
    assert!(helper.contains("'C:\\Users\\Owner\\Private\\reader.token'"));
    assert!(!helper.contains(added["token_file"].as_str().unwrap()));
}

#[cfg(unix)]
#[test]
fn owner_check_does_not_approve_an_allowed_alias_to_a_denied_target() {
    let fixture = Fixture::new();
    std::os::unix::fs::symlink(
        "../Private/secret.md",
        fixture.vault.join("Shared/alias.md"),
    )
    .unwrap();
    fixture.initialize();
    fixture.add("reader", false);
    let decision = fixture.check("reader", "read", "Shared/alias.md");
    assert_eq!(decision["decision"]["allowed"], false);
    assert_eq!(decision["resolved_path"], "Private/secret.md");
}

#[test]
fn setup_rejects_an_ephemeral_client_url_before_creating_state() {
    let fixture = Fixture::new();
    let output = command()
        .arg("setup")
        .arg(&fixture.vault)
        .arg("--access-file")
        .arg(&fixture.profile)
        .args([
            "--name",
            "assistant",
            "--directory",
            "Shared",
            "--url",
            "http://127.0.0.1:0/mcp",
            "--dry-run",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("fixed nonzero port"));
    assert!(!fixture.profile.parent().unwrap().exists());
}

#[test]
fn ipv6_setup_keeps_the_generated_server_and_client_endpoint_consistent() {
    let fixture = Fixture::new();
    // A released :0 port can be reused by parallel outgoing connections before
    // setup probes it. Keep this fixture below automatic client-port ranges.
    let listener = (10000u16..10100)
        .find_map(
            |port| match std::net::TcpListener::bind((std::net::Ipv6Addr::LOCALHOST, port)) {
                Ok(listener) => Some(listener),
                Err(error) if error.kind() == std::io::ErrorKind::AddrInUse => None,
                Err(error) => panic!("could not reserve IPv6 setup fixture port: {error}"),
            },
        )
        .expect("a private IPv6 fixture port should be available");
    let port = listener.local_addr().unwrap().port();
    let mut process = command();
    process
        .arg("setup")
        .arg(&fixture.vault)
        .arg("--access-file")
        .arg(&fixture.profile)
        .args([
            "--name",
            "assistant",
            "--directory",
            "Shared",
            "--url",
            &format!("http://[::1]:{port}/mcp"),
            "--dry-run",
        ]);
    drop(listener);
    let output = process.output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains(&format!("--host ::1 --port {port}")));
    assert!(!fixture.profile.parent().unwrap().exists());
}

#[test]
fn malformed_policy_diagnostics_do_not_echo_pasted_values_or_field_names() {
    const CANARY: &str = "omcp1_synthetic_secret_must_never_be_logged";
    let fixture = Fixture::new();
    fixture.initialize();
    fixture.add("reader", false);
    let original: Value =
        serde_json::from_slice(&std::fs::read(&fixture.profile).unwrap()).unwrap();
    for variant in 0..3 {
        let mut policy = original.clone();
        match variant {
            0 => policy["clients"]["reader"]["grants"][0]["role"] = Value::String(CANARY.into()),
            1 => policy["clients"]["reader"]["enabled"] = Value::String(CANARY.into()),
            _ => policy[CANARY] = Value::Bool(true),
        }
        std::fs::write(&fixture.profile, serde_json::to_vec(&policy).unwrap()).unwrap();
        let output = fixture.command().arg("validate").output().unwrap();
        assert!(!output.status.success());
        assert!(!String::from_utf8_lossy(&output.stdout).contains(CANARY));
        assert!(!String::from_utf8_lossy(&output.stderr).contains(CANARY));
    }
}

#[test]
fn stdio_setup_preview_uses_client_started_process_instructions() {
    let fixture = Fixture::new();
    let output = command()
        .arg("setup")
        .arg(&fixture.vault)
        .arg("--access-file")
        .arg(&fixture.profile)
        .args([
            "--name",
            "assistant",
            "--directory",
            "Shared",
            "--transport",
            "stdio",
            "--dry-run",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(text.contains("Would create"));
    assert!(text.contains("The client starts the restricted server process."));
    assert!(!text.contains("--http"));
    assert!(!text.contains("http://"));
    assert!(!fixture.profile.parent().unwrap().exists());
}

fn preview_paths(fixture: &Fixture, client: &str, provider: &str) -> Vec<String> {
    let result = success(
        fixture
            .command()
            .args(["preview", client, "--json"])
            .env("OBSIDIAN_EMBEDDINGS", "true")
            .env("OBSIDIAN_EMBEDDING_PROVIDER", "api")
            .env("OBSIDIAN_EMBEDDING_API_BASE", provider)
            .env("OBSIDIAN_EMBEDDING_API_MODEL", "text-embedding-3-small")
            .env("OBSIDIAN_EMBEDDING_API_KEY", "synthetic-provider-key")
            .output()
            .unwrap(),
    );
    result["paths"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_str().unwrap().to_owned())
        .collect()
}

#[test]
fn preview_lists_regular_files_for_directory_whole_vault_and_exact_file_grants() {
    let fixture = Fixture::new();
    std::fs::create_dir_all(fixture.vault.join("Shared/nested/empty")).unwrap();
    std::fs::write(fixture.vault.join("Shared/nested/other.md"), "another note").unwrap();
    std::fs::write(fixture.vault.join("Shared/attachment.bin"), b"attachment").unwrap();
    std::fs::write(fixture.vault.join("root.md"), "root note").unwrap();
    fixture.initialize();
    fixture.add("shared", false);
    success(
        fixture
            .command()
            .args([
                "client",
                "add",
                "whole",
                "--directory",
                "",
                "--deny-directory",
                "Private",
                "--json",
            ])
            .output()
            .unwrap(),
    );
    success(
        fixture
            .command()
            .args([
                "client",
                "add",
                "exact",
                "--file",
                "Shared/note.md",
                "--json",
            ])
            .output()
            .unwrap(),
    );
    let provider = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    provider.set_nonblocking(true).unwrap();
    let url = format!("http://{}/v1", provider.local_addr().unwrap());
    let shared = vec![
        "Shared/attachment.bin",
        "Shared/nested/other.md",
        "Shared/note.md",
    ];
    assert_eq!(preview_paths(&fixture, "shared", &url), shared);
    let mut whole = shared.clone();
    whole.push("root.md");
    assert_eq!(preview_paths(&fixture, "whole", &url), whole);
    assert_eq!(preview_paths(&fixture, "exact", &url), ["Shared/note.md"]);
    assert!(!fixture.vault.join(".obsidian-mcp").exists());
    assert_eq!(
        provider.accept().unwrap_err().kind(),
        std::io::ErrorKind::WouldBlock
    );
}

#[cfg(unix)]
#[test]
fn preview_omits_unsafe_aliases_and_hardlinks_without_losing_permitted_files() {
    let fixture = Fixture::new();
    let external = fixture._directory.path().join("outside.md");
    std::fs::write(&external, "outside source").unwrap();
    std::fs::create_dir_all(fixture.vault.join("Shared/nested")).unwrap();
    std::os::unix::fs::symlink("note.md", fixture.vault.join("Shared/allowed.md")).unwrap();
    std::os::unix::fs::symlink(
        "../Private/secret.md",
        fixture.vault.join("Shared/denied.md"),
    )
    .unwrap();
    std::os::unix::fs::symlink("missing.md", fixture.vault.join("Shared/dangling.md")).unwrap();
    std::os::unix::fs::symlink(&external, fixture.vault.join("Shared/outside.md")).unwrap();
    std::os::unix::fs::symlink("nested", fixture.vault.join("Shared/directory-alias")).unwrap();
    std::fs::hard_link(&external, fixture.vault.join("Shared/hardlink.md")).unwrap();
    fixture.initialize();
    fixture.add("reader", false);
    let paths = preview_paths(&fixture, "reader", "http://127.0.0.1:1/v1");
    assert_eq!(paths, ["Shared/allowed.md", "Shared/note.md"]);
    assert!(!fixture.vault.join(".obsidian-mcp").exists());
}

#[cfg(unix)]
#[test]
fn preview_preserves_a_real_read_permission_failure() {
    use std::os::unix::fs::PermissionsExt;
    if unsafe { libc::geteuid() } == 0 {
        return;
    }
    let fixture = Fixture::new();
    fixture.initialize();
    fixture.add("reader", false);
    let note = fixture.vault.join("Shared/note.md");
    std::fs::set_permissions(&note, std::fs::Permissions::from_mode(0)).unwrap();
    let output = fixture
        .command()
        .args(["preview", "reader", "--json"])
        .output()
        .unwrap();
    std::fs::set_permissions(&note, std::fs::Permissions::from_mode(0o600)).unwrap();
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .to_ascii_lowercase()
            .contains("permission denied")
    );
}

struct OccupiedDefaultBackend {
    _serial: std::sync::MutexGuard<'static, ()>,
    listener: Option<std::net::TcpListener>,
}

fn occupy_default_backend() -> OccupiedDefaultBackend {
    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let serial = SERIAL.lock().unwrap();
    let listener = match std::net::TcpListener::bind("127.0.0.1:37842") {
        Ok(listener) => Some(listener),
        Err(error) if error.kind() == std::io::ErrorKind::AddrInUse => None,
        Err(error) => panic!("could not qualify the default loopback port: {error}"),
    };
    OccupiedDefaultBackend {
        _serial: serial,
        listener,
    }
}

#[test]
fn https_setup_preserves_public_route_and_selects_a_separate_free_backend() {
    let listener = occupy_default_backend();
    let fixture = Fixture::new();
    let out = fixture.profile.parent().unwrap().join("https-client.json");
    let public = "https://notes.example.test/assistant/mcp";
    let result = success(
        command()
            .arg("setup")
            .arg(&fixture.vault)
            .arg("--access-file")
            .arg(&fixture.profile)
            .args([
                "--name",
                "assistant",
                "--directory",
                "Shared",
                "--client",
                "claude-code",
                "--url",
                public,
                "--out",
            ])
            .arg(&out)
            .arg("--json")
            .output()
            .unwrap(),
    );
    let backend = reqwest::Url::parse(result["backend_url"].as_str().unwrap()).unwrap();
    let port = backend.port().unwrap();
    assert_ne!(port, 37842);
    assert_eq!(backend.scheme(), "http");
    assert_eq!(backend.host_str(), Some("127.0.0.1"));
    assert_eq!(backend.path(), "/mcp");
    assert_eq!(result["url"], public);
    assert_eq!(result["proxy_upstream"], backend.as_str());
    assert_eq!(result["proxy_host_header"], format!("127.0.0.1:{port}"));
    assert!(
        result["server_command"]
            .as_str()
            .unwrap()
            .contains(&format!("--host 127.0.0.1 --port {port}"))
    );
    assert!(
        result["doctor_command"]
            .as_str()
            .unwrap()
            .contains(backend.as_str())
    );
    assert!(
        result["public_doctor_command"]
            .as_str()
            .unwrap()
            .contains(public)
    );
    let configuration: Value = serde_json::from_slice(&std::fs::read(out).unwrap()).unwrap();
    assert_eq!(configuration["mcpServers"]["assistant"]["url"], public);
    if listener.listener.is_some() {
        assert_eq!(
            std::net::TcpListener::bind("127.0.0.1:37842")
                .unwrap_err()
                .kind(),
            std::io::ErrorKind::AddrInUse
        );
    }
}

#[test]
fn https_setup_preview_prints_its_selected_upstream_and_both_connection_checks() {
    let _listener = occupy_default_backend();
    let fixture = Fixture::new();
    let public = "https://notes.example.test/assistant/mcp";
    let output = command()
        .arg("setup")
        .arg(&fixture.vault)
        .arg("--access-file")
        .arg(&fixture.profile)
        .args([
            "--name",
            "assistant",
            "--directory",
            "Shared",
            "--url",
            public,
            "--dry-run",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    let launch = text
        .lines()
        .find(|line| line.contains(" --http --host "))
        .unwrap();
    let port = launch
        .split_once("--port ")
        .unwrap()
        .1
        .split_whitespace()
        .next()
        .unwrap();
    assert_ne!(port, "37842");
    let backend = format!("http://127.0.0.1:{port}/mcp");
    assert!(text.contains(&format!(
        "Route your HTTPS proxy's {public} endpoint to {backend}"
    )));
    assert!(text.contains(&format!("forward Host as 127.0.0.1:{port}")));
    assert!(
        text.lines()
            .any(|line| line.starts_with("Check the backend:") && line.contains(&backend))
    );
    assert!(
        text.lines()
            .any(|line| line.contains("check its public endpoint:") && line.contains(public))
    );
    assert!(!fixture.profile.parent().unwrap().exists());
}

#[test]
fn explicit_http_setup_does_not_reuse_an_occupied_listener() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}/mcp", listener.local_addr().unwrap());
    let fixture = Fixture::new();
    let output = command()
        .arg("setup")
        .arg(&fixture.vault)
        .arg("--access-file")
        .arg(&fixture.profile)
        .args([
            "--name",
            "assistant",
            "--directory",
            "Shared",
            "--url",
            &endpoint,
            "--dry-run",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("already in use"));
    assert!(!fixture.profile.parent().unwrap().exists());
    assert_eq!(
        std::net::TcpListener::bind(listener.local_addr().unwrap())
            .unwrap_err()
            .kind(),
        std::io::ErrorKind::AddrInUse
    );
}

#[test]
fn direct_http_setup_rejects_a_path_that_the_server_does_not_serve() {
    let fixture = Fixture::new();
    let output = command()
        .arg("setup")
        .arg(&fixture.vault)
        .arg("--access-file")
        .arg(&fixture.profile)
        .args([
            "--name",
            "assistant",
            "--directory",
            "Shared",
            "--url",
            "http://127.0.0.1:37842/not-mcp",
            "--dry-run",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("server's /mcp path"));
    assert!(!fixture.profile.parent().unwrap().exists());
}

#[test]
fn printed_launch_invokes_an_executable_path_with_spaces_for_the_native_shell() {
    let fixture = Fixture::new();
    let directory = fixture._directory.path().join("binary with spaces");
    std::fs::create_dir(&directory).unwrap();
    let executable = directory.join(format!("obsidian-mcp{}", std::env::consts::EXE_SUFFIX));
    std::fs::copy(binary(), &executable).unwrap();
    let output = command_at(&executable)
        .arg("init")
        .arg(&fixture.vault)
        .arg("--access-file")
        .arg(&fixture.profile)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    let launch = text
        .lines()
        .find(|line| line.contains(" --http --host "))
        .unwrap();
    assert!(launch.contains("binary with spaces"));
    #[cfg(windows)]
    assert!(launch.starts_with("& '"), "{launch}");
    #[cfg(not(windows))]
    assert!(launch.starts_with('\''), "{launch}");
}
