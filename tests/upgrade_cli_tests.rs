use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
#[cfg(unix)]
use std::process::Stdio;

use obsidian_mcp::upgrade::BuildIdentity;

fn staged_cargo_install(source: &str) -> (tempfile::TempDir, PathBuf, PathBuf) {
    let root = tempfile::tempdir().expect("tempdir should be created");
    let bin_dir = root.path().join("bin");
    fs::create_dir(&bin_dir).expect("bin dir should be created");
    let main = copy_binary(
        Path::new(env!("CARGO_BIN_EXE_obsidian-mcp")),
        &bin_dir,
        "obsidian-mcp",
    );
    copy_binary(
        Path::new(env!("CARGO_BIN_EXE_obsidian-semanticd")),
        &bin_dir,
        "obsidian-semanticd",
    );

    let identity = probe_identity(&main);
    let package_id = format!("obsidian-mcp {} ({source})", identity.version);
    let tracker = serde_json::json!({
        "installs": {
            package_id: {
                "version_req": format!("={}", identity.version),
                "bins": ["obsidian-mcp", "obsidian-semanticd"],
                "features": identity.features,
                "all_features": false,
                "no_default_features": true,
                "profile": "release",
                "target": identity.target
            }
        }
    });
    fs::write(
        root.path().join(".crates2.json"),
        serde_json::to_vec(&tracker).expect("tracker should serialize"),
    )
    .expect("tracker should be written");
    let fake_home = root.path().join("home");
    fs::create_dir(&fake_home).expect("fake home should be created");
    (root, main, fake_home)
}

fn copy_binary(source: &Path, bin_dir: &Path, stem: &str) -> PathBuf {
    let destination = bin_dir.join(format!("{stem}{}", std::env::consts::EXE_SUFFIX));
    fs::copy(source, &destination).expect("test binary should be copied");
    destination
}

fn probe_identity(binary: &Path) -> BuildIdentity {
    let output = Command::new(binary)
        .arg("--__build-info")
        .output()
        .expect("build identity command should run");
    assert!(output.status.success());
    BuildIdentity::from_json(&output.stdout).expect("build identity should decode")
}

#[test]
fn both_binaries_publish_the_same_feature_and_target_identity() {
    let main = probe_identity(Path::new(env!("CARGO_BIN_EXE_obsidian-mcp")));
    let semantic = probe_identity(Path::new(env!("CARGO_BIN_EXE_obsidian-semanticd")));
    assert_eq!(main, semantic);
}

#[test]
fn top_level_help_lists_upgrade_with_the_other_commands() {
    let output = Command::new(env!("CARGO_BIN_EXE_obsidian-mcp"))
        .arg("--help")
        .output()
        .expect("help command should execute");

    assert!(output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("\n    obsidian-mcp upgrade [--dry-run]")
    );
}

#[test]
fn dry_run_accepts_official_cargo_install_without_mutating_it() {
    let (root, main, fake_home) =
        staged_cargo_install("registry+https://github.com/rust-lang/crates.io-index");
    let marker = root.path().join("settings.marker");
    fs::write(&marker, "unchanged").expect("marker should be written");
    let identity = probe_identity(&main);

    let output = Command::new(&main)
        .args(["upgrade", "--dry-run"])
        .env("HOME", fake_home)
        .output()
        .expect("upgrade dry run should execute");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Upgrade preflight passed"));
    assert!(stdout.contains("--registry crates-io"));
    assert!(stdout.contains("--no-default-features"));
    assert!(stdout.contains(&format!("--target {}", identity.target)));
    assert!(stdout.contains("--profile release"));
    if identity.features.is_empty() {
        assert!(!stdout.contains(" --features "));
    } else {
        assert!(stdout.contains(&format!("--features {}", identity.feature_list())));
    }
    assert!(stdout.contains("no files or processes were changed"));
    assert!(stdout.contains("Next: run 'obsidian-mcp upgrade'"));
    assert_eq!(fs::read_to_string(marker).unwrap(), "unchanged");
    assert!(!root.path().join(".obsidian-mcp-upgrade.lock").exists());
}

#[test]
fn dry_run_rejects_non_crates_io_provenance() {
    let (_root, main, fake_home) =
        staged_cargo_install("git+https://github.com/lstpsche/obsidian-mcp");
    let output = Command::new(main)
        .args(["upgrade", "--dry-run"])
        .env("HOME", fake_home)
        .output()
        .expect("upgrade dry run should execute");

    assert_eq!(
        output.status.code(),
        Some(1),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("not the official crates.io registry")
    );
}

#[test]
fn upgrade_rejects_unknown_options_before_loading_vault_config() {
    let output = Command::new(env!("CARGO_BIN_EXE_obsidian-mcp"))
        .args(["upgrade", "--force"])
        .env_remove("OBSIDIAN_VAULT_PATH")
        .output()
        .expect("upgrade command should execute");
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("usage:"));
    assert!(stderr.contains("upgrade --help"));
}

#[cfg(unix)]
#[test]
fn active_ad_hoc_http_runtime_blocks_upgrade_before_mutation() {
    use std::net::{TcpListener, TcpStream};
    use std::time::{Duration, Instant};

    let (root, main, fake_home) =
        staged_cargo_install("registry+https://github.com/rust-lang/crates.io-index");
    let vault = tempfile::tempdir().expect("temporary vault should be created");
    let port = TcpListener::bind(("127.0.0.1", 0))
        .expect("temporary port should bind")
        .local_addr()
        .expect("temporary address should resolve")
        .port();
    let mut server = Command::new(&main)
        .arg(vault.path())
        .env("HOME", &fake_home)
        .env("OBSIDIAN_TRANSPORT", "http")
        .env("OBSIDIAN_HTTP_HOST", "127.0.0.1")
        .env("OBSIDIAN_HTTP_PORT", port.to_string())
        .env("OBSIDIAN_WATCH", "false")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("ad-hoc HTTP server should start");

    let deadline = Instant::now() + Duration::from_secs(10);
    while TcpStream::connect(("127.0.0.1", port)).is_err() {
        if Instant::now() >= deadline {
            let _ = server.kill();
            let _ = server.wait();
            panic!("ad-hoc HTTP server did not become ready");
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    let marker = root.path().join("settings.marker");
    fs::write(&marker, "unchanged").expect("marker should be written");
    let output = Command::new(&main)
        .args(["upgrade", "--dry-run"])
        .env("HOME", &fake_home)
        .output()
        .expect("upgrade dry run should execute");
    let _ = server.kill();
    let _ = server.wait();

    assert_eq!(
        output.status.code(),
        Some(1),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("unsupported active runtime"),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("restart it with the same launch command"),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(fs::read_to_string(marker).unwrap(), "unchanged");
}

#[cfg(unix)]
#[test]
fn up_to_date_upgrade_reports_that_nothing_was_restarted() {
    let (root, main, fake_home) =
        staged_cargo_install("registry+https://github.com/rust-lang/crates.io-index");
    let fake_bin = install_fake_cargo(root.path(), 0);
    let output = Command::new(&main)
        .arg("upgrade")
        .env("HOME", &fake_home)
        .env("PATH", prefixed_path(&fake_bin))
        .output()
        .expect("upgrade should execute");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("already up to date; nothing was restarted"));
    assert!(!stdout.contains("Reconnect stdio MCP clients"));
}

#[cfg(unix)]
#[test]
fn cargo_failure_does_not_replace_the_staged_binaries() {
    let (root, main, fake_home) =
        staged_cargo_install("registry+https://github.com/rust-lang/crates.io-index");
    let semantic = root.path().join("bin/obsidian-semanticd");
    let main_before = fs::read(&main).expect("main binary should read");
    let semantic_before = fs::read(&semantic).expect("semantic binary should read");
    let fake_bin = install_fake_cargo(root.path(), 101);
    let output = Command::new(&main)
        .arg("upgrade")
        .env("HOME", &fake_home)
        .env("PATH", prefixed_path(&fake_bin))
        .output()
        .expect("upgrade should execute");

    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("status 101"));
    assert_eq!(fs::read(main).unwrap(), main_before);
    assert_eq!(fs::read(semantic).unwrap(), semantic_before);
}

#[cfg(unix)]
fn install_fake_cargo(root: &Path, exit_code: i32) -> PathBuf {
    use std::os::unix::fs::PermissionsExt;

    let bin = root.join("fake-bin");
    fs::create_dir(&bin).expect("fake bin directory should be created");
    let cargo = bin.join("cargo");
    fs::write(&cargo, format!("#!/bin/sh\nexit {exit_code}\n"))
        .expect("fake cargo should be written");
    let mut permissions = fs::metadata(&cargo)
        .expect("fake cargo metadata should read")
        .permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(cargo, permissions).expect("fake cargo should be executable");
    bin
}

#[cfg(unix)]
fn prefixed_path(directory: &Path) -> std::ffi::OsString {
    let mut paths = vec![directory.to_path_buf()];
    paths.extend(std::env::split_paths(
        &std::env::var_os("PATH").unwrap_or_default(),
    ));
    std::env::join_paths(paths).expect("test PATH should join")
}
