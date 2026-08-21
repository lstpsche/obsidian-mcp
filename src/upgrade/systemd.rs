//! Discovery and activation of directly managed Linux systemd user services.

#[cfg(target_os = "linux")]
use std::ffi::OsString;
use std::path::Path;

#[cfg(target_os = "linux")]
use super::ServiceOwner;
use super::{ActivationResult, CommandExecutor, ServiceTarget, UpgradeError};

#[cfg(target_os = "linux")]
pub fn discover<E: CommandExecutor>(
    executor: &E,
    executable: &Path,
) -> Result<Vec<ServiceTarget>, UpgradeError> {
    discover_with(executor, executable, &ProcInspector)
}

#[cfg(target_os = "linux")]
fn discover_with<E: CommandExecutor, I: ProcessInspector>(
    executor: &E,
    executable: &Path,
    inspector: &I,
) -> Result<Vec<ServiceTarget>, UpgradeError> {
    let executable = executable.canonicalize()?;
    let output = match executor.output(
        std::ffi::OsStr::new("systemctl"),
        &[
            OsString::from("--user"),
            OsString::from("list-units"),
            OsString::from("--type=service"),
            OsString::from("--state=running"),
            OsString::from("--no-legend"),
            OsString::from("--plain"),
            OsString::from("--no-pager"),
        ],
    ) {
        Ok(output) => output,
        Err(UpgradeError::Command { message, .. }) => {
            tracing::debug!(error = %message, "systemd user manager is unavailable");
            return Ok(Vec::new());
        }
        Err(err) => return Err(err),
    };
    if !output.success {
        tracing::debug!(
            diagnostic = %String::from_utf8_lossy(&output.stderr).trim(),
            "systemd user manager is unavailable"
        );
        return Ok(Vec::new());
    }

    let mut targets = Vec::new();
    for unit in parse_unit_names(&String::from_utf8_lossy(&output.stdout)) {
        let Some(pid) = main_pid(executor, &unit)? else {
            continue;
        };
        let process_exe = match inspector.executable(pid) {
            Ok(path) => path,
            Err(_) => continue,
        };
        if process_exe.canonicalize().ok().as_deref() != Some(executable.as_path()) {
            continue;
        }
        let arguments = inspector.arguments(pid)?;
        let environment = inspector.environment(pid)?;
        let Some((host, port)) = super::http_service_config(
            &arguments,
            environment.transport.as_deref(),
            environment.host.as_deref(),
            environment.port.as_deref(),
        ) else {
            continue;
        };
        let observed_version = super::health::probe(&host, port)
            .ok()
            .map(|observation| observation.version);
        targets.push(ServiceTarget {
            owner: ServiceOwner::SystemdUser,
            id: unit,
            definition_path: None,
            executable: executable.clone(),
            host,
            port,
            previous_pid: Some(pid),
            observed_version,
        });
    }
    targets.sort_by(|left, right| left.id.cmp(&right.id));
    Ok(targets)
}

#[cfg(not(target_os = "linux"))]
pub fn discover<E: CommandExecutor>(
    _executor: &E,
    _executable: &Path,
) -> Result<Vec<ServiceTarget>, UpgradeError> {
    Ok(Vec::new())
}

#[cfg(target_os = "linux")]
pub fn restart<E: CommandExecutor>(
    executor: &E,
    target: &ServiceTarget,
    expected_version: &str,
) -> ActivationResult {
    match restart_with(executor, target, expected_version, &ProcInspector) {
        Ok(version) => ActivationResult {
            target_id: target.id.clone(),
            success: true,
            observed_version: Some(version),
            diagnostic: "systemd user service restarted and passed exact health verification"
                .into(),
        },
        Err(err) => ActivationResult {
            target_id: target.id.clone(),
            success: false,
            observed_version: super::health::probe(&target.host, target.port)
                .ok()
                .map(|observation| observation.version),
            diagnostic: err.to_string(),
        },
    }
}

#[cfg(target_os = "linux")]
fn restart_with<E: CommandExecutor, I: ProcessInspector>(
    executor: &E,
    target: &ServiceTarget,
    expected_version: &str,
    inspector: &I,
) -> Result<String, UpgradeError> {
    let active_pid = main_pid(executor, &target.id)?.ok_or_else(|| {
        UpgradeError::Activation(format!(
            "systemd user service '{}' became inactive before activation; it was left inactive",
            target.id
        ))
    })?;
    let active_executable = inspector.executable(active_pid)?;
    if active_executable.canonicalize().ok().as_deref() != Some(target.executable.as_path()) {
        return Err(UpgradeError::Activation(format!(
            "systemd user service '{}' changed executable before activation",
            target.id
        )));
    }
    checked_systemctl(executor, &["--user", "restart", &target.id])?;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
    let pid = loop {
        if let Some(pid) = main_pid(executor, &target.id)?
            && pid != active_pid
        {
            break pid;
        }
        if std::time::Instant::now() >= deadline {
            return Err(UpgradeError::Activation(format!(
                "systemd user service '{}' did not report a new running process",
                target.id
            )));
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
    };
    checked_systemctl(executor, &["--user", "is-active", "--quiet", &target.id])?;
    let process_exe = inspector.executable(pid).map_err(|err| {
        UpgradeError::Activation(format!(
            "cannot inspect restarted systemd service '{}' PID {pid}: {err}",
            target.id
        ))
    })?;
    if process_exe.canonicalize().ok().as_deref() != Some(target.executable.as_path()) {
        return Err(UpgradeError::Activation(format!(
            "restarted systemd service '{}' no longer runs '{}'",
            target.id,
            target.executable.display()
        )));
    }
    let arguments = inspector.arguments(pid)?;
    let environment = inspector.environment(pid)?;
    let (host, port) = super::http_service_config(
        &arguments,
        environment.transport.as_deref(),
        environment.host.as_deref(),
        environment.port.as_deref(),
    )
    .ok_or_else(|| {
        UpgradeError::Activation(format!(
            "restarted systemd service '{}' is no longer configured for HTTP transport",
            target.id
        ))
    })?;
    let observation = super::health::wait_for_version(
        &host,
        port,
        expected_version,
        std::time::Duration::from_secs(15),
    )?;
    Ok(observation.version)
}

#[cfg(not(target_os = "linux"))]
pub fn restart<E: CommandExecutor>(
    _executor: &E,
    target: &ServiceTarget,
    _expected_version: &str,
) -> ActivationResult {
    ActivationResult {
        target_id: target.id.clone(),
        success: false,
        observed_version: None,
        diagnostic: "systemd user-service activation is only available on Linux".into(),
    }
}

#[cfg(target_os = "linux")]
#[derive(Debug, Default)]
struct AllowedEnvironment {
    transport: Option<String>,
    host: Option<String>,
    port: Option<String>,
}

#[cfg(target_os = "linux")]
trait ProcessInspector {
    fn executable(&self, pid: u32) -> Result<std::path::PathBuf, UpgradeError>;
    fn arguments(&self, pid: u32) -> Result<Vec<String>, UpgradeError>;
    fn environment(&self, pid: u32) -> Result<AllowedEnvironment, UpgradeError>;
}

#[cfg(target_os = "linux")]
struct ProcInspector;

#[cfg(target_os = "linux")]
impl ProcessInspector for ProcInspector {
    fn executable(&self, pid: u32) -> Result<std::path::PathBuf, UpgradeError> {
        Ok(super::processes::proc_executable(pid)?)
    }

    fn arguments(&self, pid: u32) -> Result<Vec<String>, UpgradeError> {
        let raw = std::fs::read(format!("/proc/{pid}/cmdline"))?;
        Ok(raw
            .split(|byte| *byte == 0)
            .filter(|argument| !argument.is_empty())
            .skip(1)
            .map(|argument| String::from_utf8_lossy(argument).into_owned())
            .collect())
    }

    fn environment(&self, pid: u32) -> Result<AllowedEnvironment, UpgradeError> {
        let raw = std::fs::read(format!("/proc/{pid}/environ"))?;
        let mut allowed = AllowedEnvironment::default();
        for entry in raw.split(|byte| *byte == 0) {
            if let Some(value) = entry.strip_prefix(b"OBSIDIAN_TRANSPORT=") {
                allowed.transport = Some(String::from_utf8_lossy(value).into_owned());
            } else if let Some(value) = entry.strip_prefix(b"OBSIDIAN_HTTP_HOST=") {
                allowed.host = Some(String::from_utf8_lossy(value).into_owned());
            } else if let Some(value) = entry.strip_prefix(b"OBSIDIAN_HTTP_PORT=") {
                allowed.port = Some(String::from_utf8_lossy(value).into_owned());
            }
        }
        Ok(allowed)
    }
}

#[cfg(target_os = "linux")]
fn main_pid<E: CommandExecutor>(executor: &E, unit: &str) -> Result<Option<u32>, UpgradeError> {
    let output = executor.output(
        std::ffi::OsStr::new("systemctl"),
        &[
            OsString::from("--user"),
            OsString::from("show"),
            OsString::from(unit),
            OsString::from("--property=MainPID"),
            OsString::from("--value"),
        ],
    )?;
    if !output.success {
        return Ok(None);
    }
    Ok(String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse::<u32>()
        .ok()
        .filter(|pid| *pid != 0))
}

#[cfg(target_os = "linux")]
fn checked_systemctl<E: CommandExecutor>(executor: &E, args: &[&str]) -> Result<(), UpgradeError> {
    let output = executor.output(
        std::ffi::OsStr::new("systemctl"),
        &args.iter().map(OsString::from).collect::<Vec<_>>(),
    )?;
    if output.success {
        Ok(())
    } else {
        Err(command_failure(
            &format!("systemctl {}", args.join(" ")),
            &output.stderr,
        ))
    }
}

#[cfg(target_os = "linux")]
fn command_failure(command: &str, stderr: &[u8]) -> UpgradeError {
    let diagnostic = String::from_utf8_lossy(stderr);
    let diagnostic = diagnostic.trim();
    UpgradeError::Activation(if diagnostic.is_empty() {
        format!("'{command}' failed")
    } else {
        format!("'{command}' failed: {diagnostic}")
    })
}

#[cfg(any(target_os = "linux", test))]
fn parse_unit_names(raw: &str) -> Vec<String> {
    raw.lines()
        .filter_map(|line| line.split_whitespace().next())
        .filter(|unit| unit.ends_with(".service"))
        .map(str::to_string)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "linux")]
    use crate::upgrade::{CommandOutput, CommandStatus};
    #[cfg(target_os = "linux")]
    use std::ffi::{OsStr, OsString};
    #[cfg(target_os = "linux")]
    use std::sync::Mutex;

    #[test]
    fn unit_parser_ignores_footer_and_non_services() {
        let raw = "obsidian.service loaded active running Obsidian MCP\n\
                   user.slice loaded active active User Slice\n\
                   1 loaded units listed.\n";
        assert_eq!(parse_unit_names(raw), vec!["obsidian.service"]);
    }

    #[cfg(target_os = "linux")]
    struct FakeSystemd {
        commands: Mutex<Vec<String>>,
    }

    #[cfg(target_os = "linux")]
    impl CommandExecutor for FakeSystemd {
        fn output(
            &self,
            program: &OsStr,
            args: &[OsString],
        ) -> Result<CommandOutput, UpgradeError> {
            let command = format!(
                "{} {}",
                program.to_string_lossy(),
                args.iter()
                    .map(|arg| arg.to_string_lossy())
                    .collect::<Vec<_>>()
                    .join(" ")
            );
            let pid_queries = {
                let mut commands = self.commands.lock().expect("commands should lock");
                commands.push(command);
                commands
                    .iter()
                    .filter(|command| command.contains("--property=MainPID"))
                    .count()
            };
            let stdout = args
                .iter()
                .any(|arg| arg == "--property=MainPID")
                .then(|| format!("{}\n", if pid_queries == 1 { 321 } else { 654 }).into_bytes())
                .unwrap_or_default();
            Ok(CommandOutput {
                success: true,
                code: Some(0),
                stdout,
                stderr: Vec::new(),
            })
        }

        fn status_inherit(
            &self,
            _program: &OsStr,
            _args: &[OsString],
        ) -> Result<CommandStatus, UpgradeError> {
            panic!("restart must use captured systemctl commands")
        }
    }

    #[cfg(target_os = "linux")]
    struct FakeInspector {
        executable: std::path::PathBuf,
        port: u16,
    }

    #[cfg(target_os = "linux")]
    impl ProcessInspector for FakeInspector {
        fn executable(&self, _pid: u32) -> Result<std::path::PathBuf, UpgradeError> {
            Ok(self.executable.clone())
        }

        fn arguments(&self, _pid: u32) -> Result<Vec<String>, UpgradeError> {
            Ok(vec![
                "--http".into(),
                "--port".into(),
                self.port.to_string(),
            ])
        }

        fn environment(&self, _pid: u32) -> Result<AllowedEnvironment, UpgradeError> {
            Ok(AllowedEnvironment::default())
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn restart_uses_user_restart_without_daemon_reload() {
        use std::io::{Read, Write};
        use std::net::TcpListener;

        let listener = TcpListener::bind(("127.0.0.1", 0)).expect("listener should bind");
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("health should accept");
            let mut request = [0_u8; 512];
            let _ = stream.read(&mut request);
            stream
                .write_all(
                    b"HTTP/1.0 200 OK\r\n\r\n{\"status\":\"ok\",\"server\":\"obsidian-mcp\",\"version\":\"2.5.0\"}",
                )
                .expect("health response should write");
        });
        let fake = FakeSystemd {
            commands: Mutex::new(Vec::new()),
        };
        let temp = tempfile::tempdir().expect("tempdir should be created");
        let executable = temp.path().join("obsidian-mcp");
        std::fs::write(&executable, "binary").expect("fake executable should be written");
        let executable = executable
            .canonicalize()
            .expect("fake executable should canonicalize");
        let inspector = FakeInspector {
            executable: executable.clone(),
            port,
        };
        let target = ServiceTarget {
            owner: ServiceOwner::SystemdUser,
            id: "obsidian.service".into(),
            definition_path: None,
            executable,
            host: "127.0.0.1".into(),
            port,
            previous_pid: Some(321),
            observed_version: Some("2.4.0".into()),
        };

        let version =
            restart_with(&fake, &target, "2.5.0", &inspector).expect("service should restart");
        assert_eq!(version, "2.5.0");
        let commands = fake.commands.lock().expect("commands should lock");
        assert!(
            commands
                .iter()
                .any(|command| command == "systemctl --user restart obsidian.service")
        );
        assert!(
            commands
                .iter()
                .any(|command| command == "systemctl --user is-active --quiet obsidian.service")
        );
        assert!(
            !commands
                .iter()
                .any(|command| command.contains("daemon-reload"))
        );
    }
}
