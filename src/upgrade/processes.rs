//! Read-only discovery of running `obsidian-mcp` processes outside supported managers.

use std::collections::BTreeSet;
use std::path::Path;

use super::UpgradeError;

#[cfg(unix)]
struct ProcessCandidate {
    pid: u32,
    arguments: Vec<String>,
    transport: Option<String>,
    listens_tcp: bool,
}

/// Return matching process IDs that appear to own an unmanaged HTTP runtime.
/// Client-owned stdio processes are intentionally ignored because replacing the
/// on-disk Unix binary does not invalidate their existing file descriptor.
#[cfg(unix)]
pub fn unmanaged_http_processes(
    executable: &Path,
    managed_pids: &BTreeSet<u32>,
) -> Result<Vec<u32>, UpgradeError> {
    #[cfg(target_os = "linux")]
    let candidates = linux_candidates(executable)?;
    #[cfg(target_os = "macos")]
    let candidates = macos_candidates(executable)?;
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    let candidates = Vec::new();

    let current_pid = std::process::id();
    let mut unmanaged = candidates
        .into_iter()
        .filter(|candidate| {
            candidate.pid != current_pid
                && !managed_pids.contains(&candidate.pid)
                && is_http_runtime(
                    &candidate.arguments,
                    candidate.transport.as_deref(),
                    candidate.listens_tcp,
                )
        })
        .map(|candidate| candidate.pid)
        .collect::<Vec<_>>();
    unmanaged.sort_unstable();
    unmanaged.dedup();
    Ok(unmanaged)
}

#[cfg(not(unix))]
pub fn unmanaged_http_processes(
    _executable: &Path,
    _managed_pids: &BTreeSet<u32>,
) -> Result<Vec<u32>, UpgradeError> {
    Ok(Vec::new())
}

#[cfg(any(unix, test))]
fn is_http_runtime(arguments: &[String], transport: Option<&str>, listens_tcp: bool) -> bool {
    arguments.iter().any(|argument| argument == "--http")
        || transport.is_some_and(|value| value.eq_ignore_ascii_case("http"))
        || listens_tcp
}

#[cfg(target_os = "linux")]
fn linux_candidates(executable: &Path) -> Result<Vec<ProcessCandidate>, UpgradeError> {
    let executable = executable.canonicalize()?;
    let mut candidates = Vec::new();
    for entry in std::fs::read_dir("/proc")? {
        let entry = match entry {
            Ok(entry) => entry,
            Err(_) => continue,
        };
        let Some(pid) = entry
            .file_name()
            .to_str()
            .and_then(|value| value.parse::<u32>().ok())
        else {
            continue;
        };
        let process_executable = match proc_executable(pid) {
            Ok(path) => path,
            Err(_) => continue,
        };
        if process_executable.canonicalize().ok().as_deref() != Some(executable.as_path()) {
            continue;
        }
        let arguments = read_nul_strings(&entry.path().join("cmdline"))
            .unwrap_or_default()
            .into_iter()
            .skip(1)
            .collect();
        let transport = read_nul_strings(&entry.path().join("environ"))
            .unwrap_or_default()
            .into_iter()
            .find_map(|entry| entry.strip_prefix("OBSIDIAN_TRANSPORT=").map(str::to_owned));
        candidates.push(ProcessCandidate {
            pid,
            arguments,
            transport,
            listens_tcp: false,
        });
    }
    Ok(candidates)
}

#[cfg(target_os = "linux")]
pub(crate) fn proc_executable(pid: u32) -> std::io::Result<std::path::PathBuf> {
    Ok(strip_deleted_suffix(std::fs::read_link(format!(
        "/proc/{pid}/exe"
    ))?))
}

#[cfg(any(target_os = "linux", target_os = "macos", test))]
pub(crate) fn strip_deleted_suffix(path: std::path::PathBuf) -> std::path::PathBuf {
    let rendered = path.to_string_lossy();
    rendered
        .strip_suffix(" (deleted)")
        .map(std::path::PathBuf::from)
        .unwrap_or(path)
}

#[cfg(target_os = "linux")]
fn read_nul_strings(path: &Path) -> std::io::Result<Vec<String>> {
    Ok(std::fs::read(path)?
        .split(|byte| *byte == 0)
        .filter(|value| !value.is_empty())
        .map(|value| String::from_utf8_lossy(value).into_owned())
        .collect())
}

#[cfg(target_os = "macos")]
fn macos_candidates(executable: &Path) -> Result<Vec<ProcessCandidate>, UpgradeError> {
    use std::process::Command;

    let executable = executable.canonicalize()?;
    let pgrep = Command::new("pgrep")
        .args(["-x", "obsidian-mcp"])
        .output()
        .map_err(|err| UpgradeError::Command {
            program: "pgrep".into(),
            message: err.to_string(),
        })?;
    if !pgrep.status.success() && pgrep.status.code() != Some(1) {
        return Err(UpgradeError::Command {
            program: "pgrep".into(),
            message: String::from_utf8_lossy(&pgrep.stderr).trim().to_string(),
        });
    }

    let mut candidates = Vec::new();
    for pid in String::from_utf8_lossy(&pgrep.stdout)
        .lines()
        .filter_map(|line| line.trim().parse::<u32>().ok())
    {
        if !macos_process_matches(pid, &executable) {
            continue;
        }
        let output = match Command::new("ps")
            .args(["-p", &pid.to_string(), "-o", "command="])
            .output()
        {
            Ok(output) if output.status.success() => output,
            _ => continue,
        };
        let command = String::from_utf8_lossy(&output.stdout);
        let arguments = command
            .split_whitespace()
            .skip(1)
            .map(str::to_owned)
            .collect();
        candidates.push(ProcessCandidate {
            pid,
            arguments,
            transport: None,
            listens_tcp: macos_listens_tcp(pid)?,
        });
    }
    Ok(candidates)
}

#[cfg(target_os = "macos")]
fn macos_listens_tcp(pid: u32) -> Result<bool, UpgradeError> {
    let output = std::process::Command::new("lsof")
        .args(["-a", "-p", &pid.to_string(), "-iTCP", "-sTCP:LISTEN", "-Fn"])
        .output()
        .map_err(|err| UpgradeError::Command {
            program: "lsof".into(),
            message: err.to_string(),
        })?;
    if output.status.success() {
        return Ok(true);
    }
    if output.status.code() == Some(1) {
        return Ok(false);
    }
    Err(UpgradeError::Command {
        program: "lsof".into(),
        message: String::from_utf8_lossy(&output.stderr).trim().to_string(),
    })
}

#[cfg(target_os = "macos")]
fn macos_process_matches(pid: u32, expected: &Path) -> bool {
    let Ok(output) = std::process::Command::new("lsof")
        .args(["-a", "-p", &pid.to_string(), "-d", "txt", "-Fn"])
        .output()
    else {
        return false;
    };
    output.status.success() && lsof_text_matches(&output.stdout, expected)
}

#[cfg(any(target_os = "macos", test))]
pub(crate) fn lsof_text_matches(raw: &[u8], expected: &Path) -> bool {
    String::from_utf8_lossy(raw)
        .lines()
        .filter_map(|line| line.strip_prefix('n'))
        .map(std::path::PathBuf::from)
        .map(strip_deleted_suffix)
        .any(|path| path == expected || path.canonicalize().ok().as_deref() == Some(expected))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_only_http_runtime_arguments_or_transport() {
        assert!(is_http_runtime(&["--http".into()], None, false));
        assert!(is_http_runtime(&[], Some("HTTP"), false));
        assert!(is_http_runtime(&[], None, true));
        assert!(!is_http_runtime(&[], None, false));
        assert!(!is_http_runtime(&["serve".into()], Some("stdio"), false));
    }

    #[test]
    fn normalizes_linux_deleted_executable_suffix_for_retry_discovery() {
        assert_eq!(
            strip_deleted_suffix("/home/user/.cargo/bin/obsidian-mcp (deleted)".into()),
            std::path::PathBuf::from("/home/user/.cargo/bin/obsidian-mcp")
        );
        assert_eq!(
            strip_deleted_suffix("/tmp/not-deleted".into()),
            std::path::PathBuf::from("/tmp/not-deleted")
        );
    }

    #[test]
    fn lsof_matching_ignores_non_text_fields_and_deleted_suffix() {
        let expected = Path::new("/tmp/obsidian-mcp");
        assert!(lsof_text_matches(
            b"p123\nfcwd\nn/tmp\nftxt\nn/tmp/obsidian-mcp (deleted)\n",
            expected
        ));
        assert!(!lsof_text_matches(
            b"p123\nftxt\nn/tmp/other-binary\n",
            expected
        ));
    }
}
