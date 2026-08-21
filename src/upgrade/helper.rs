//! Windows replacement handoff for the currently running executable.

#[cfg(windows)]
use std::ffi::OsString;
#[cfg(windows)]
use std::path::{Path, PathBuf};
#[cfg(windows)]
use std::process::{Command, Stdio};
#[cfg(windows)]
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[cfg(windows)]
use super::UpgradeError;

#[cfg(windows)]
const HELPER_PREFIX: &str = "obsidian-mcp-upgrade-";
#[cfg(windows)]
const HELPER_BINARY: &str = "obsidian-mcp-upgrade-helper.exe";

/// Copy the running executable to a temporary path and launch the hidden
/// replacement phase. The caller must exit immediately after this succeeds.
#[cfg(windows)]
pub fn spawn_windows_handoff(original_exe: &Path) -> Result<PathBuf, UpgradeError> {
    let temp_root = std::env::temp_dir();
    remove_stale_helpers(&temp_root);
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let helper_dir = temp_root.join(format!("{HELPER_PREFIX}{}-{nonce}", std::process::id()));
    std::fs::create_dir(&helper_dir).map_err(|err| {
        UpgradeError::Install(format!(
            "cannot create Windows upgrade helper directory '{}': {err}",
            helper_dir.display()
        ))
    })?;
    let helper = helper_dir.join(HELPER_BINARY);
    std::fs::copy(original_exe, &helper).map_err(|err| {
        let _ = std::fs::remove_dir(&helper_dir);
        UpgradeError::Install(format!(
            "cannot create Windows upgrade helper '{}': {err}",
            helper.display()
        ))
    })?;

    Command::new(&helper)
        .arg("__apply-upgrade")
        .arg("--parent-pid")
        .arg(std::process::id().to_string())
        .arg("--original-exe")
        .arg(original_exe)
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|err| {
            let _ = std::fs::remove_dir_all(&helper_dir);
            UpgradeError::Install(format!(
                "cannot start Windows upgrade helper '{}': {err}",
                helper.display()
            ))
        })?;
    Ok(helper)
}

/// Wait for the original process to release the installed executable and
/// reject replacement while any other installed obsidian-mcp process is live.
#[cfg(windows)]
pub fn await_windows_replacement_window(parent_pid: u32) -> Result<(), UpgradeError> {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let pids = installed_process_ids()?;
        let parent_alive = pids.contains(&parent_pid);
        let others = pids
            .iter()
            .copied()
            .filter(|pid| *pid != parent_pid)
            .collect::<Vec<_>>();
        if !others.is_empty() {
            return Err(UpgradeError::Install(format!(
                "cannot replace obsidian-mcp.exe while other instances are running (PIDs: {})",
                others
                    .iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
        if !parent_alive {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(UpgradeError::Install(format!(
                "timed out waiting for parent process {parent_pid} to exit"
            )));
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

#[cfg(windows)]
fn installed_process_ids() -> Result<Vec<u32>, UpgradeError> {
    let args = [
        OsString::from("/FI"),
        OsString::from("IMAGENAME eq obsidian-mcp.exe"),
        OsString::from("/FO"),
        OsString::from("CSV"),
        OsString::from("/NH"),
    ];
    let output = Command::new("tasklist")
        .args(args)
        .output()
        .map_err(|err| UpgradeError::Command {
            program: "tasklist".into(),
            message: err.to_string(),
        })?;
    if !output.status.success() {
        return Err(UpgradeError::Command {
            program: "tasklist".into(),
            message: String::from_utf8_lossy(&output.stderr).trim().to_string(),
        });
    }
    Ok(parse_tasklist_pids(&String::from_utf8_lossy(
        &output.stdout,
    )))
}

#[cfg(windows)]
fn parse_tasklist_pids(raw: &str) -> Vec<u32> {
    raw.lines()
        .filter_map(|line| {
            let mut fields = line.split("\",\"");
            let image = fields.next()?.trim_matches('"');
            let pid = fields.next()?.trim_matches('"');
            image
                .eq_ignore_ascii_case("obsidian-mcp.exe")
                .then(|| pid.parse::<u32>().ok())
                .flatten()
        })
        .collect()
}

#[cfg(windows)]
fn remove_stale_helpers(temp_root: &Path) {
    let Ok(entries) = std::fs::read_dir(temp_root) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let path = entry.path();
        if name.starts_with(HELPER_PREFIX) && path.is_dir() && path.join(HELPER_BINARY).is_file() {
            let _ = std::fs::remove_dir_all(path);
        }
    }
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;

    #[test]
    fn tasklist_parser_ignores_headers_and_other_images() {
        let raw = "\"obsidian-mcp.exe\",\"123\",\"Console\",\"1\",\"1,024 K\"\n\
                   \"other.exe\",\"456\",\"Console\",\"1\",\"1,024 K\"\n\
                   INFO: No tasks are running";
        assert_eq!(parse_tasklist_pids(raw), vec![123]);
    }
}
