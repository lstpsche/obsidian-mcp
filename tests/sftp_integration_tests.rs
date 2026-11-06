//! Prepared native SSH acceptance, including authentication and host-key checks.
//! All keys, configuration and server state belong to this disposable fixture.

#[cfg(feature = "embeddings-api")]
#[path = "common/embedding_profiles.rs"]
mod embedding_profiles;
#[path = "common/http_listening.rs"]
mod http_listening;
#[path = "common/http_rpc.rs"]
mod http_rpc;

use std::ffi::OsString;
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
#[cfg(unix)]
use std::process::Child;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use reqwest::Client;
use serde_json::{Value, json};
use tempfile::TempDir;

enum SshServer {
    #[cfg(unix)]
    Process(Child),
    #[cfg(windows)]
    Task { name: String },
}

impl SshServer {
    fn start(sshd: &Path, config: &Path, log: &Path, root: &Path) -> Self {
        #[cfg(unix)]
        {
            let _ = root;
            Self::Process(
                Command::new(sshd)
                    .args(["-D", "-e", "-f"])
                    .arg(config)
                    .stdin(Stdio::null())
                    .stdout(Stdio::null())
                    .stderr(std::fs::File::create(log).unwrap())
                    .spawn()
                    .unwrap(),
            )
        }
        #[cfg(windows)]
        {
            // Windows needs SYSTEM's authentication privileges. Running sshd as
            // a child action avoids OpenSSH's service-time system configuration.
            let name = format!(
                "obsidian-mcp-ssh-{}-{}",
                std::process::id(),
                root.file_name().unwrap().to_string_lossy()
            );
            let server = Self::Task { name: name.clone() };
            let script = root.join("start-sshd.ps1");
            std::fs::write(
                &script,
                r#"$ErrorActionPreference = 'Stop'
$arguments = '-D -E "' + $env:OBSIDIAN_TEST_SSH_LOG + '" -f "' + $env:OBSIDIAN_TEST_SSH_CONFIG + '"'
$action = New-ScheduledTaskAction -Execute $env:OBSIDIAN_TEST_SSHD -Argument $arguments
$principal = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Minutes 15) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $env:OBSIDIAN_TEST_SSH_TASK -Action $action -Principal $principal -Settings $settings | Out-Null
Start-ScheduledTask -TaskName $env:OBSIDIAN_TEST_SSH_TASK
"#,
            ).unwrap();
            checked(
                Command::new("powershell.exe")
                    .args([
                        "-NoProfile",
                        "-NonInteractive",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-File",
                    ])
                    .arg(&script)
                    .env("OBSIDIAN_TEST_SSH_TASK", &name)
                    .env("OBSIDIAN_TEST_SSHD", sshd)
                    .env("OBSIDIAN_TEST_SSH_CONFIG", config)
                    .env("OBSIDIAN_TEST_SSH_LOG", log),
            );
            server
        }
    }

    fn assert_running(&mut self, log: &Path) {
        #[cfg(unix)]
        match self {
            Self::Process(child) => {
                if let Some(status) = child.try_wait().unwrap() {
                    panic!(
                        "disposable sshd exited {status}: {}",
                        std::fs::read_to_string(log).unwrap_or_default()
                    );
                }
            }
        }
        #[cfg(windows)]
        let _ = log;
    }
}

impl Drop for SshServer {
    fn drop(&mut self) {
        match self {
            #[cfg(unix)]
            Self::Process(child) => {
                let _ = child.kill();
                let _ = child.wait();
            }
            #[cfg(windows)]
            Self::Task { name } => {
                let cleanup = r#"
$ErrorActionPreference = 'Stop'
$task = Get-ScheduledTask -TaskName $env:OBSIDIAN_TEST_SSH_TASK -ErrorAction SilentlyContinue
if ($task) {
    Stop-ScheduledTask -InputObject $task
    $deadline = (Get-Date).AddSeconds(10)
    while ((Get-ScheduledTask -TaskName $env:OBSIDIAN_TEST_SSH_TASK).State -eq 'Running') {
        if ((Get-Date) -ge $deadline) { throw 'Disposable SSH task did not stop' }
        Start-Sleep -Milliseconds 50
    }
    Unregister-ScheduledTask -InputObject $task -Confirm:$false
}
"#;
                let output = Command::new("powershell.exe")
                    .args(["-NoProfile", "-NonInteractive", "-Command", cleanup])
                    .env("OBSIDIAN_TEST_SSH_TASK", name.as_str())
                    .output();
                let error = match output {
                    Ok(output) if output.status.success() => None,
                    Ok(output) => Some(String::from_utf8_lossy(&output.stderr).into_owned()),
                    Err(error) => Some(error.to_string()),
                };
                if let Some(error) = error {
                    if std::thread::panicking() {
                        eprintln!("failed to remove disposable SSH task {name}: {error}");
                    } else {
                        panic!("failed to remove disposable SSH task {name}: {error}");
                    }
                }
            }
        }
    }
}

struct SshFixture {
    _server: SshServer,
    directory: TempDir,
    vault: PathBuf,
    client_config: PathBuf,
    known_hosts: PathBuf,
    client_key: PathBuf,
    rejected_key: PathBuf,
    host_key: PathBuf,
    other_host_key: PathBuf,
    user: String,
    port: u16,
    path: OsString,
}

fn executable(name: &str, additional: &[&str]) -> PathBuf {
    let filename = format!("{name}{}", std::env::consts::EXE_SUFFIX);
    let native = std::env::var_os("WINDIR").map(|directory| {
        PathBuf::from(directory)
            .join("System32/OpenSSH")
            .join(&filename)
    });
    native
        .into_iter()
        .chain(
            std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default())
                .map(|directory| directory.join(&filename)),
        )
        .chain(additional.iter().map(PathBuf::from))
        .find(|path| path.is_file())
        .unwrap_or_else(|| panic!("prepared SSH test requires {name} on PATH"))
}

fn checked(command: &mut Command) {
    let output = command.output().unwrap();
    assert!(
        output.status.success(),
        "fixture command {command:?} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn config_path(path: &Path) -> String {
    let value = path.to_str().unwrap().replace('\\', "/");
    assert!(!value.contains(['"', '\n', '\r']));
    format!("\"{value}\"")
}

fn key(ssh_keygen: &Path, path: &Path) {
    checked(
        Command::new(ssh_keygen)
            .args(["-q", "-t", "ed25519", "-N", "", "-f"])
            .arg(path),
    );
}

impl SshFixture {
    fn start() -> Self {
        assert_eq!(
            std::env::var("OBSIDIAN_TEST_SSH_INTEGRATION").as_deref(),
            Ok("1"),
            "set OBSIDIAN_TEST_SSH_INTEGRATION=1 after preparing local OpenSSH client/server"
        );
        let ssh = executable("ssh", &["/usr/bin/ssh"]);
        let ssh_keygen = executable("ssh-keygen", &["/usr/bin/ssh-keygen"]);
        let sshd = executable("sshd", &["/usr/sbin/sshd"]);
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        let vault = root.join("vault");
        let bin = root.join("bin");
        std::fs::create_dir(&vault).unwrap();
        std::fs::create_dir(&bin).unwrap();
        std::fs::write(
            vault.join("note.md"),
            "# SSH fixture\nAuthenticated remote note.\n",
        )
        .unwrap();
        let client_key = root.join("client_key");
        let rejected_key = root.join("rejected_key");
        let host_key = root.join("host_key");
        let other_host_key = root.join("other_host_key");
        for path in [&client_key, &rejected_key, &host_key, &other_host_key] {
            key(&ssh_keygen, path);
        }
        let authorized = root.join("authorized_keys");
        std::fs::copy(client_key.with_extension("pub"), &authorized).unwrap();
        #[cfg(unix)]
        let user = {
            let output = Command::new("id").arg("-un").output().unwrap();
            assert!(output.status.success());
            String::from_utf8(output.stdout).unwrap().trim().to_owned()
        };
        #[cfg(windows)]
        let user = std::env::var("USERNAME").unwrap().to_lowercase();
        assert!(!user.is_empty() && !user.contains(['\n', '\r', '"']));
        let port = TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let server_config = root.join("sshd_config");
        let settings = format!(
            "Port {port}\nListenAddress 127.0.0.1\nHostKey {}\nPidFile {}\nAuthorizedKeysFile {}\nPubkeyAuthentication yes\nPasswordAuthentication no\nKbdInteractiveAuthentication no\nAuthenticationMethods publickey\nStrictModes no\nPermitRootLogin yes\nAllowUsers \"{user}\"\nSubsystem sftp internal-sftp\nForceCommand internal-sftp\nPermitUserRC no\nLogLevel VERBOSE\n{}",
            config_path(&host_key),
            config_path(&root.join("sshd.pid")),
            config_path(&authorized),
            if cfg!(unix) { "UsePAM no\n" } else { "" },
        );
        std::fs::write(&server_config, settings).unwrap();
        checked(Command::new(&sshd).arg("-t").arg("-f").arg(&server_config));
        let log = root.join("sshd.log");
        let mut server = SshServer::start(&sshd, &server_config, &log, root);
        let deadline = Instant::now() + Duration::from_secs(15);
        loop {
            if TcpStream::connect(("127.0.0.1", port)).is_ok() {
                break;
            }
            server.assert_running(&log);
            assert!(
                Instant::now() < deadline,
                "disposable sshd did not listen: {}",
                std::fs::read_to_string(&log).unwrap_or_default()
            );
            std::thread::sleep(Duration::from_millis(25));
        }
        let client_config = root.join("ssh_config");
        let known_hosts = root.join("known_hosts");
        // The product resolves `ssh` through PATH. This native launcher only injects
        // the isolated config; real OpenSSH performs all protocol and authentication.
        let launcher_source = root.join("ssh_launcher.rs");
        std::fs::write(
            &launcher_source,
            format!(
                "fn main() {{\n let status = std::process::Command::new({ssh:?}).arg(\"-F\").arg({client_config:?}).args(std::env::args_os().skip(1)).status().expect(\"launch real SSH\");\n std::process::exit(status.code().unwrap_or(1));\n}}\n",
            ),
        ).unwrap();
        checked(
            Command::new("rustc")
                .arg("--edition=2024")
                .arg(&launcher_source)
                .arg("-o")
                .arg(bin.join(format!("ssh{}", std::env::consts::EXE_SUFFIX))),
        );
        let path = std::env::join_paths(std::iter::once(bin).chain(std::env::split_paths(
            &std::env::var_os("PATH").unwrap_or_default(),
        )))
        .unwrap();
        Self {
            _server: server,
            directory,
            vault,
            client_config,
            known_hosts,
            client_key,
            rejected_key,
            host_key,
            other_host_key,
            user,
            port,
            path,
        }
    }

    fn configure(&self, trusted_host_key: Option<&Path>, authorized_key: bool) {
        let hosts = trusted_host_key.map_or_else(String::new, |key| {
            let public = std::fs::read_to_string(key.with_extension("pub")).unwrap();
            let parts: Vec<_> = public.split_whitespace().take(2).collect();
            format!("[127.0.0.1]:{} {} {}\n", self.port, parts[0], parts[1])
        });
        std::fs::write(&self.known_hosts, hosts).unwrap();
        std::fs::write(
            &self.client_config,
            format!(
                "Host fixture-vault\n HostName 127.0.0.1\n Port {}\n User \"{}\"\n IdentityFile {}\n IdentitiesOnly yes\n IdentityAgent none\n UserKnownHostsFile {}\n GlobalKnownHostsFile none\n LogLevel ERROR\n",
                self.port,
                self.user,
                config_path(if authorized_key { &self.client_key } else { &self.rejected_key }),
                config_path(&self.known_hosts),
            ),
        ).unwrap();
    }

    fn command(&self) -> tokio::process::Command {
        let binary = std::env::var_os("OBSIDIAN_TEST_MCP_BINARY")
            .unwrap_or_else(|| env!("CARGO_BIN_EXE_obsidian-mcp").into());
        let mut command = http_listening::isolated_command(binary);
        let mut root = self.vault.to_str().unwrap().replace('\\', "/");
        if cfg!(windows) {
            root = format!("/{}", root.trim_start_matches("//?/"));
        }
        let mut source = reqwest::Url::parse("sftp://fixture-vault/").unwrap();
        source.set_path(&root);
        command
            .arg(source.as_str())
            .args(["--http", "--port", "0"])
            .env("PATH", &self.path)
            .env_remove("SSH_AUTH_SOCK")
            .env("OBSIDIAN_MCP_DATA", self.directory.path().join("machine"))
            .env(
                "OBSIDIAN_SEMANTIC_HOME",
                self.directory.path().join("semantic"),
            )
            .env("OBSIDIAN_WATCH", "false")
            .env("OBSIDIAN_EMBEDDINGS", "false")
            .env("OBSIDIAN_SFTP_TIMEOUT_SECS", "5")
            .env("OBSIDIAN_LOG_LEVEL", "info")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        command
    }

    async fn read_note(&self) -> Value {
        let mut child = self.command().spawn().unwrap();
        let address = http_listening::ready(&mut child).await.unwrap();
        let client = Client::builder()
            .no_proxy()
            .timeout(Duration::from_secs(15))
            .build()
            .unwrap();
        let result = http_rpc::rpc_response(
            http_rpc::request(
                &client,
                &format!("http://{address}"),
                "tools/call",
                json!({"name":"note_read","arguments":{"path":"note.md"}}),
                http_rpc::MODERN,
            )
            .send()
            .await
            .expect("SSH source read must finish without prompting"),
        )
        .await;
        child.kill().await.unwrap();
        child.wait().await.unwrap();
        result
    }

    fn server_log(&self) -> String {
        std::fs::read_to_string(self.directory.path().join("sshd.log")).unwrap()
    }

    async fn rejects(&self, reason: &str) {
        let result = self.read_note().await;
        assert!(
            result.get("error").is_some() || result["result"]["isError"] == true,
            "rejected SSH configuration exposed source data: {result}"
        );
        let diagnostic = result.to_string();
        assert!(
            diagnostic.to_ascii_lowercase().contains(reason),
            "expected SSH rejection {reason:?}, got: {diagnostic}; server: {}",
            self.server_log(),
        );
        assert!(!diagnostic.contains("Authenticated remote note"));
    }

    #[cfg(feature = "embeddings-api")]
    async fn semantic_read(&self) {
        use axum::{Json, Router, extract::State, routing::post};
        use std::sync::{Arc, Mutex};

        async fn embeddings(
            State(inputs): State<Arc<Mutex<Vec<String>>>>,
            Json(request): Json<Value>,
        ) -> Json<Value> {
            let input = request["input"].as_array().unwrap();
            inputs
                .lock()
                .unwrap()
                .extend(input.iter().map(|text| text.as_str().unwrap().to_owned()));
            let data: Vec<_> = input
                .iter()
                .enumerate()
                .map(|(index, _)| json!({"index":index,"embedding":[1.0,0.0]}))
                .collect();
            Json(json!({"data":data}))
        }
        let inputs = Arc::new(Mutex::new(Vec::new()));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let app = Router::new()
            .route("/v1/embeddings", post(embeddings))
            .with_state(Arc::clone(&inputs));
        let provider = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let profile =
            embedding_profiles::write_api_profile(self.directory.path(), port, "probe", 128, 32);
        let mut child = self
            .command()
            .env("OBSIDIAN_EMBEDDINGS", "true")
            .env("OBSIDIAN_EMBEDDING_PROVIDER", "api")
            .env(
                "OBSIDIAN_EMBEDDING_API_BASE",
                format!("http://127.0.0.1:{port}/v1"),
            )
            .env("OBSIDIAN_EMBEDDING_API_KEY", "synthetic-test-key")
            .env("OBSIDIAN_EMBEDDING_API_MODEL", "probe")
            .env("OBSIDIAN_EMBEDDING_API_PROFILE", profile)
            .env("OBSIDIAN_EMBEDDING_DIM", "2")
            .spawn()
            .unwrap();
        let address = http_listening::ready(&mut child).await.unwrap();
        let client = Client::builder().no_proxy().build().unwrap();
        let result = tokio::time::timeout(Duration::from_secs(20), async {
            loop {
                let result = http_rpc::rpc_response(
                    http_rpc::request(
                        &client, &format!("http://{address}"), "tools/call",
                        json!({"name":"search_semantic","arguments":{"query":"Authenticated remote note","top_k":1,"include_content":true}}),
                        http_rpc::MODERN,
                    ).send().await.unwrap(),
                ).await;
                if result["result"]["structuredContent"]["results"]
                    .as_array().is_some_and(|hits| !hits.is_empty()) {
                    break result;
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        }).await.expect("remote semantic indexing must finish");
        child.kill().await.unwrap();
        child.wait().await.unwrap();
        provider.abort();
        let hit = &result["result"]["structuredContent"]["results"][0];
        assert_eq!(hit["path"], "note.md");
        assert_eq!(hit["source_status"], "current");
        assert!(
            hit["content"]
                .as_str()
                .unwrap()
                .contains("Authenticated remote note")
        );
        assert!(
            inputs
                .lock()
                .unwrap()
                .iter()
                .any(|input| input.contains("# SSH fixture"))
        );
        assert!(
            !self.directory.path().join("semantic").exists(),
            "SFTP must not bootstrap the local daemon"
        );
    }
}

#[tokio::test]
#[ignore = "requires prepared local OpenSSH client/server; see docs/remote-vaults.md"]
async fn native_ssh_enforces_host_keys_and_key_authentication() {
    let fixture = SshFixture::start();
    fixture.configure(None, true);
    fixture.rejects("host key verification failed").await;
    fixture.configure(Some(&fixture.other_host_key), true);
    fixture.rejects("host key verification failed").await;
    fixture.configure(Some(&fixture.host_key), false);
    fixture.rejects("permission denied").await;
    fixture.configure(Some(&fixture.host_key), true);
    let result = fixture.read_note().await;
    assert!(
        result.get("error").is_none() && result["result"]["isError"] != true,
        "{result}; server: {}",
        fixture.server_log(),
    );
    assert!(
        result.to_string().contains("Authenticated remote note"),
        "{result}"
    );
    assert!(!fixture.vault.join(".obsidian-mcp").exists());
    #[cfg(feature = "embeddings-api")]
    fixture.semantic_read().await;
}
