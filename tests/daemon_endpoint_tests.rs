//! Native listener ownership and recovery of a dead daemon's endpoint.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use obsidian_mcp::client::semantic_daemon::{DaemonConnectPolicy, SemanticDaemonClient};
use obsidian_mcp::daemon::server::IpcEndpoint;
use tokio::process::{Child, Command};
use tokio::time::{sleep, timeout};

fn endpoint(directory: &Path) -> IpcEndpoint {
    #[cfg(unix)]
    {
        IpcEndpoint::UnixSocket(directory.join("daemon.sock"))
    }
    #[cfg(windows)]
    {
        IpcEndpoint::NamedPipe(format!(
            r"\\.\pipe\obsidian-endpoint-{}-{}",
            std::process::id(),
            directory.file_name().unwrap().to_string_lossy()
        ))
    }
}

fn client(endpoint: &IpcEndpoint) -> SemanticDaemonClient {
    SemanticDaemonClient::new(
        endpoint.clone(),
        DaemonConnectPolicy {
            timeout: Duration::from_secs(1),
            retries: 0,
            ..Default::default()
        },
    )
}

fn spawn(home: &Path, endpoint: &IpcEndpoint) -> Child {
    let binary = std::env::var_os("OBSIDIAN_TEST_DAEMON_BINARY")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_BIN_EXE_obsidian-semanticd")));
    let mut cmd = Command::new(binary);
    for (key, _) in std::env::vars_os() {
        if key.to_string_lossy().starts_with("OBSIDIAN_") {
            cmd.env_remove(key);
        }
    }
    cmd.env("OBSIDIAN_SEMANTIC_HOME", home)
        .env("OBSIDIAN_SEMANTIC_ENDPOINT", endpoint.endpoint_string())
        .env("OBSIDIAN_LOG_LEVEL", "error")
        .env("TOKIO_WORKER_THREADS", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .unwrap()
}

async fn assert_refused(child: Child) {
    let output = timeout(Duration::from_secs(5), child.wait_with_output())
        .await
        .expect("competing daemon took over the endpoint or stalled")
        .unwrap();
    assert!(!output.status.success());
    assert!(
        !output.stderr.is_empty(),
        "owner conflict should report a cause"
    );
}

async fn shutdown(client: &SemanticDaemonClient, mut owner: Child) {
    assert!(client.shutdown().await.unwrap().accepted);
    assert!(
        timeout(Duration::from_secs(5), owner.wait())
            .await
            .expect("daemon failed to exit")
            .unwrap()
            .success()
    );
}

async fn health_pid(client: &SemanticDaemonClient) -> u32 {
    timeout(Duration::from_secs(5), async {
        loop {
            if let Ok(health) = client.health("endpoint-test", "1").await {
                break health.pid;
            }
            sleep(Duration::from_millis(15)).await;
        }
    })
    .await
    .expect("daemon failed to serve health")
}

#[tokio::test]
async fn concurrent_binaries_share_exactly_one_listener() {
    for separate_home in [false, true] {
        let directory = tempfile::tempdir().unwrap();
        let endpoint = endpoint(directory.path());
        let first_client = client(&endpoint);
        let home_a = directory.path().join("a");
        let home_b = directory.path().join(if separate_home { "b" } else { "a" });
        let start = std::sync::Arc::new(std::sync::Barrier::new(2));
        let [a, b] = [home_a.clone(), home_b].map(|home| {
            let start = start.clone();
            let endpoint = endpoint.clone();
            tokio::task::spawn_blocking(move || {
                start.wait();
                spawn(&home, &endpoint)
            })
        });
        let (a, b) = tokio::try_join!(a, b).unwrap();
        let winner = health_pid(&first_client).await;
        let (owner, loser) = if a.id() == Some(winner) {
            (a, b)
        } else {
            assert_eq!(b.id(), Some(winner));
            (b, a)
        };
        assert_refused(loser).await;
        let other_client = client(&endpoint);
        let (left, right) = tokio::join!(health_pid(&first_client), health_pid(&other_client));
        assert_eq!((left, right), (winner, winner));
        shutdown(&first_client, owner).await;
        let next = spawn(&home_a, &endpoint);
        assert_eq!(Some(health_pid(&first_client).await), next.id());
        shutdown(&first_client, next).await;
    }
}

#[cfg(unix)]
#[tokio::test]
async fn stale_socket_is_reclaimed_but_live_or_foreign_entries_survive() {
    use std::os::unix::fs::{MetadataExt, PermissionsExt, symlink};
    use std::os::unix::net::UnixDatagram;
    use tokio::net::UnixSocket;
    let directory = tempfile::tempdir().unwrap();
    let home = directory.path().join("home");
    let endpoint = endpoint(directory.path());
    let IpcEndpoint::UnixSocket(path) = &endpoint;
    let client = client(&endpoint);
    let target = directory.path().join("target");
    std::fs::write(&target, "keep").unwrap();

    async fn preserved(home: &Path, endpoint: &IpcEndpoint, path: &Path) {
        let original = std::fs::symlink_metadata(path).unwrap();
        assert_refused(spawn(home, endpoint)).await;
        assert_eq!(
            std::fs::symlink_metadata(path).unwrap().ino(),
            original.ino()
        );
    }
    // A lock symlink must never be opened through its target.
    let lock = path.with_extension("sock.lock");
    symlink(&target, &lock).unwrap();
    preserved(&home, &endpoint, &lock).await;
    std::fs::remove_file(&lock).unwrap();
    assert!(!path.exists());

    std::fs::write(path, "ordinary file").unwrap();
    preserved(&home, &endpoint, path).await;
    assert_eq!(std::fs::read(path).unwrap(), b"ordinary file");
    std::fs::remove_file(path).unwrap();

    symlink(&target, path).unwrap();
    preserved(&home, &endpoint, path).await;
    std::fs::remove_file(path).unwrap();

    std::fs::create_dir(path).unwrap();
    preserved(&home, &endpoint, path).await;
    std::fs::remove_dir(path).unwrap();
    assert_eq!(std::fs::read(&target).unwrap(), b"keep");
    let datagram = UnixDatagram::bind(path).unwrap();
    preserved(&home, &endpoint, path).await;
    drop(datagram);
    std::fs::remove_file(path).unwrap();

    let legacy = UnixSocket::new_stream().unwrap();
    legacy.bind(path).unwrap();
    preserved(&home, &endpoint, path).await;
    let legacy = legacy.listen(1).unwrap();
    preserved(&home, &endpoint, path).await;

    let addr = rustix::net::SocketAddrUnix::new(path).unwrap();
    let mut queued = Vec::new();
    for _ in 0..8 {
        let client = UnixSocket::new_stream().unwrap();
        if rustix::net::connect(&client, &addr).is_err() {
            break;
        }
        queued.push(client);
    }
    assert!(
        !queued.is_empty() && queued.len() < 8,
        "backlog must be full"
    );
    preserved(&home, &endpoint, path).await;
    timeout(Duration::from_secs(1), legacy.accept())
        .await
        .unwrap()
        .unwrap();
    drop(queued);
    drop(legacy);

    let mut daemon = spawn(&home, &endpoint);
    assert_eq!(Some(health_pid(&client).await), daemon.id());
    assert_eq!(
        std::fs::metadata(path).unwrap().permissions().mode() & 0o777,
        0o600
    );
    let lock_id = std::fs::metadata(&lock).unwrap().ino();
    daemon.kill().await.unwrap();
    assert!(path.exists());
    let next = spawn(&home, &endpoint);
    assert_eq!(Some(health_pid(&client).await), next.id());
    shutdown(&client, next).await;
    assert!(!path.exists());
    assert_eq!(std::fs::metadata(&lock).unwrap().ino(), lock_id);
}

#[cfg(unix)]
#[tokio::test]
async fn shutting_down_a_displaced_owner_preserves_the_replacement() {
    use std::os::unix::fs::MetadataExt;
    use std::os::unix::net::UnixListener;
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    let directory = tempfile::tempdir().unwrap();
    let endpoint = endpoint(directory.path());
    let IpcEndpoint::UnixSocket(path) = &endpoint;
    let mut owner = spawn(&directory.path().join("home"), &endpoint);
    assert_eq!(Some(health_pid(&client(&endpoint)).await), owner.id());
    let mut original = tokio::net::UnixStream::connect(path).await.unwrap();
    // Keep the first socket inode alive, then replace the published address.
    std::fs::rename(path, directory.path().join("old.sock")).unwrap();
    let _replacement = UnixListener::bind(path).unwrap();
    let id = std::fs::metadata(path).unwrap().ino();
    original
        .write_all(b"{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"shutdown\",\"params\":{}}\n")
        .await
        .unwrap();
    let mut response = String::new();
    BufReader::new(original)
        .read_line(&mut response)
        .await
        .unwrap();
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&response).unwrap()["result"]["accepted"],
        true
    );
    assert!(
        timeout(Duration::from_secs(5), owner.wait())
            .await
            .unwrap()
            .unwrap()
            .success()
    );
    assert_eq!(std::fs::metadata(path).unwrap().ino(), id);
    tokio::net::UnixStream::connect(path).await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn unsafe_ipc_parent_is_rejected_before_provisioning() {
    use std::os::unix::fs::PermissionsExt;
    let directory = tempfile::tempdir().unwrap();
    let shared = directory.path().join("shared");
    std::fs::create_dir(&shared).unwrap();
    std::fs::set_permissions(&shared, std::fs::Permissions::from_mode(0o777)).unwrap();
    let endpoint = IpcEndpoint::UnixSocket(shared.join("child").join("daemon.sock"));
    assert_refused(spawn(&directory.path().join("home"), &endpoint)).await;
    assert!(!shared.join("child").exists());
}

#[cfg(target_os = "macos")]
#[tokio::test]
async fn extended_grants_cannot_expose_a_private_ipc_path() {
    use std::os::unix::fs::{PermissionsExt, symlink};
    fn chmod(path: &Path, args: &[&str]) {
        assert!(
            std::process::Command::new("/bin/chmod")
                .args(args)
                .arg(path)
                .status()
                .unwrap()
                .success()
        );
    }

    let directory = tempfile::tempdir().unwrap();
    let home = directory.path().join("home");
    let private = directory.path().join("private");
    std::fs::create_dir(&private).unwrap();
    chmod(&private, &["700"]);
    chmod(&private, &["+a", "everyone allow add_file,delete_child"]);
    assert_eq!(
        std::fs::metadata(&private).unwrap().permissions().mode() & 0o777,
        0o700
    );
    let endpoint = endpoint(&private);
    let IpcEndpoint::UnixSocket(path) = &endpoint;
    assert_refused(spawn(&home, &endpoint)).await;
    assert!(!path.exists());
    chmod(&private, &["-N"]);

    let alias = directory.path().join("alias");
    symlink(&private, &alias).unwrap();
    chmod(&alias, &["-h", "+a", "everyone allow delete"]);
    assert_refused(spawn(
        &home,
        &IpcEndpoint::UnixSocket(alias.join("daemon.sock")),
    ))
    .await;
    assert!(!path.exists());

    let lock = path.with_extension("sock.lock");
    std::fs::write(&lock, "").unwrap();
    chmod(&lock, &["600"]);
    chmod(&lock, &["+a", "everyone allow write,delete"]);
    assert_refused(spawn(&home, &endpoint)).await;
    assert!(!path.exists());
    chmod(&lock, &["-N"]);

    // The default macOS home commonly has an explicit deny-delete entry.
    chmod(&private, &["+a", "everyone deny delete"]);
    let client = client(&endpoint);
    let owner = spawn(&home, &endpoint);
    let winner = health_pid(&client).await;
    chmod(&private, &["-N"]);
    assert_eq!(Some(winner), owner.id());
    shutdown(&client, owner).await;
}

#[cfg(any(target_os = "linux", target_vendor = "apple"))]
#[tokio::test]
async fn owner_can_use_a_private_directory_without_listing_access() {
    use std::os::unix::fs::PermissionsExt;
    let directory = tempfile::tempdir().unwrap();
    let private = directory.path().join("private");
    std::fs::create_dir(&private).unwrap();
    std::fs::set_permissions(&private, std::fs::Permissions::from_mode(0o300)).unwrap();
    let endpoint = endpoint(&private);
    let client = client(&endpoint);
    let owner = spawn(&directory.path().join("home"), &endpoint);
    assert_eq!(Some(health_pid(&client).await), owner.id());
    shutdown(&client, owner).await;
    std::fs::set_permissions(private, std::fs::Permissions::from_mode(0o700)).unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn short_socket_address_can_name_a_long_private_directory() {
    use std::os::unix::fs::symlink;
    use std::os::unix::net::UnixListener;
    let dir = tempfile::tempdir().unwrap();
    let long = dir.path().join("long".repeat(30));
    std::fs::create_dir(&long).unwrap();
    let alias = dir.path().join("s");
    symlink(&long, &alias).unwrap();
    let address = alias.join("d.sock");
    // Reclaim through the same short address a previous owner could bind.
    drop(UnixListener::bind(&address).unwrap());
    let endpoint = IpcEndpoint::UnixSocket(address);
    let client = client(&endpoint);
    let owner = spawn(&dir.path().join("home"), &endpoint);
    assert_eq!(Some(health_pid(&client).await), owner.id());
    shutdown(&client, owner).await;
    assert!(!long.join("d.sock").exists());
}
