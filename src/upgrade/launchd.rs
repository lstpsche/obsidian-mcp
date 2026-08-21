//! Discovery and activation of directly managed macOS user LaunchAgents.

#[cfg(target_os = "macos")]
use std::ffi::{OsStr, OsString};
use std::path::Path;
#[cfg(target_os = "macos")]
use std::time::Duration;

use super::{ActivationResult, CommandExecutor, ServiceTarget, UpgradeError};
#[cfg(target_os = "macos")]
use super::{ServiceOwner, health};

#[cfg(target_os = "macos")]
pub fn discover<E: CommandExecutor>(
    executor: &E,
    executable: &Path,
) -> Result<Vec<ServiceTarget>, UpgradeError> {
    let executable = executable.canonicalize()?;
    let home = std::env::var_os("HOME")
        .ok_or_else(|| UpgradeError::Activation("HOME is not set".into()))?;
    let agents_dir = Path::new(&home).join("Library/LaunchAgents");
    if !agents_dir.is_dir() {
        return Ok(Vec::new());
    }
    discover_in_directory(executor, &executable, &agents_dir)
}

#[cfg(target_os = "macos")]
fn discover_in_directory<E: CommandExecutor>(
    executor: &E,
    executable: &Path,
    agents_dir: &Path,
) -> Result<Vec<ServiceTarget>, UpgradeError> {
    let running = running_agents(executor)?;
    let mut targets = Vec::new();
    for entry in std::fs::read_dir(agents_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension() != Some(OsStr::new("plist")) {
            continue;
        }
        let Some(label) = plist_value(executor, &path, ":Label") else {
            continue;
        };
        let Some(pid) = running
            .iter()
            .find_map(|(candidate, pid)| (candidate == &label).then_some(*pid))
        else {
            continue;
        };

        let mut arguments = plist_array(executor, &path, ":ProgramArguments");
        let program =
            plist_value(executor, &path, ":Program").or_else(|| arguments.first().cloned());
        let Some(program) = program else {
            continue;
        };
        let program = Path::new(&program);
        if !program.is_absolute() || program.canonicalize().ok().as_deref() != Some(executable) {
            continue;
        }
        if !process_matches(executor, pid, executable)? {
            continue;
        }
        if arguments
            .first()
            .is_some_and(|arg| Path::new(arg) == program)
        {
            arguments.remove(0);
        }

        let transport = plist_value(executor, &path, ":EnvironmentVariables:OBSIDIAN_TRANSPORT");
        let env_host = plist_value(executor, &path, ":EnvironmentVariables:OBSIDIAN_HTTP_HOST");
        let env_port = plist_value(executor, &path, ":EnvironmentVariables:OBSIDIAN_HTTP_PORT");
        let Some((host, port)) = super::http_service_config(
            &arguments,
            transport.as_deref(),
            env_host.as_deref(),
            env_port.as_deref(),
        ) else {
            continue;
        };
        let observed_version = health::probe(&host, port)
            .ok()
            .map(|observation| observation.version);
        targets.push(ServiceTarget {
            owner: ServiceOwner::Launchd,
            id: label,
            definition_path: Some(path),
            executable: executable.to_path_buf(),
            host,
            port,
            previous_pid: Some(pid),
            observed_version,
        });
    }
    targets.sort_by(|left, right| left.id.cmp(&right.id));
    Ok(targets)
}

#[cfg(not(target_os = "macos"))]
pub fn discover<E: CommandExecutor>(
    _executor: &E,
    _executable: &Path,
) -> Result<Vec<ServiceTarget>, UpgradeError> {
    Ok(Vec::new())
}

#[cfg(target_os = "macos")]
pub fn restart<E: CommandExecutor>(
    executor: &E,
    target: &ServiceTarget,
    expected_version: &str,
) -> ActivationResult {
    match restart_inner(executor, target, expected_version) {
        Ok(version) => ActivationResult {
            target_id: target.id.clone(),
            success: true,
            observed_version: Some(version),
            diagnostic: "launchd agent restarted and passed exact health verification".into(),
        },
        Err(err) => ActivationResult {
            target_id: target.id.clone(),
            success: false,
            observed_version: health::probe(&target.host, target.port)
                .ok()
                .map(|observation| observation.version),
            diagnostic: err.to_string(),
        },
    }
}

#[cfg(target_os = "macos")]
fn restart_inner<E: CommandExecutor>(
    executor: &E,
    target: &ServiceTarget,
    expected_version: &str,
) -> Result<String, UpgradeError> {
    let definition = target.definition_path.as_deref().ok_or_else(|| {
        UpgradeError::Activation(format!("LaunchAgent '{}' has no plist path", target.id))
    })?;
    let uid = user_id(executor)?;
    let service = format!("gui/{uid}/{}", target.id);
    let domain = format!("gui/{uid}");
    let active_pid = running_agents(executor)?
        .into_iter()
        .find_map(|(label, pid)| (label == target.id).then_some(pid))
        .ok_or_else(|| {
            UpgradeError::Activation(format!(
                "LaunchAgent '{}' became inactive before activation; it was left inactive",
                target.id
            ))
        })?;
    if !process_matches(executor, active_pid, &target.executable)? {
        return Err(UpgradeError::Activation(format!(
            "LaunchAgent '{}' changed executable before activation",
            target.id
        )));
    }
    checked_output(executor, "launchctl", &["bootout", &service])?;
    checked_output_os(
        executor,
        OsStr::new("launchctl"),
        &[
            OsString::from("bootstrap"),
            OsString::from(&domain),
            definition.as_os_str().to_owned(),
        ],
    )?;

    let deadline = std::time::Instant::now() + Duration::from_secs(15);
    let restarted_pid = loop {
        let running = running_agents(executor)?;
        if let Some(pid) = running
            .iter()
            .find_map(|(label, pid)| (label == &target.id).then_some(*pid))
            && pid != active_pid
        {
            break pid;
        }
        if std::time::Instant::now() >= deadline {
            return Err(UpgradeError::Activation(format!(
                "LaunchAgent '{}' did not report a new running process",
                target.id
            )));
        }
        std::thread::sleep(Duration::from_millis(200));
    };
    if !process_matches(executor, restarted_pid, &target.executable)? {
        return Err(UpgradeError::Activation(format!(
            "restarted LaunchAgent '{}' no longer runs '{}'",
            target.id,
            target.executable.display()
        )));
    }
    let observation = health::wait_for_version(
        &target.host,
        target.port,
        expected_version,
        Duration::from_secs(15),
    )?;
    Ok(observation.version)
}

#[cfg(target_os = "macos")]
fn process_matches<E: CommandExecutor>(
    executor: &E,
    pid: u32,
    expected: &Path,
) -> Result<bool, UpgradeError> {
    let output = executor.output(
        OsStr::new("lsof"),
        &[
            OsString::from("-a"),
            OsString::from("-p"),
            OsString::from(pid.to_string()),
            OsString::from("-d"),
            OsString::from("txt"),
            OsString::from("-Fn"),
        ],
    )?;
    if !output.success {
        return Err(command_failure(
            "lsof process executable inspection",
            &output.stderr,
        ));
    }
    Ok(super::processes::lsof_text_matches(
        &output.stdout,
        expected,
    ))
}

#[cfg(not(target_os = "macos"))]
pub fn restart<E: CommandExecutor>(
    _executor: &E,
    target: &ServiceTarget,
    _expected_version: &str,
) -> ActivationResult {
    ActivationResult {
        target_id: target.id.clone(),
        success: false,
        observed_version: None,
        diagnostic: "launchd activation is only available on macOS".into(),
    }
}

#[cfg(target_os = "macos")]
fn running_agents<E: CommandExecutor>(executor: &E) -> Result<Vec<(String, u32)>, UpgradeError> {
    let output = executor.output(OsStr::new("launchctl"), &[OsString::from("list")])?;
    if !output.success {
        return Err(command_failure("launchctl list", &output.stderr));
    }
    Ok(parse_launchctl_list(&String::from_utf8_lossy(
        &output.stdout,
    )))
}

#[cfg(target_os = "macos")]
fn user_id<E: CommandExecutor>(executor: &E) -> Result<u32, UpgradeError> {
    let output = executor.output(OsStr::new("id"), &[OsString::from("-u")])?;
    if !output.success {
        return Err(command_failure("id -u", &output.stderr));
    }
    String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse::<u32>()
        .map_err(|err| UpgradeError::Activation(format!("invalid user id from 'id -u': {err}")))
}

#[cfg(target_os = "macos")]
fn plist_value<E: CommandExecutor>(executor: &E, path: &Path, key: &str) -> Option<String> {
    let output = executor
        .output(
            OsStr::new("/usr/libexec/PlistBuddy"),
            &[
                OsString::from("-c"),
                OsString::from(format!("Print {key}")),
                path.as_os_str().to_owned(),
            ],
        )
        .ok()?;
    output
        .success
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .filter(|value| !value.is_empty())
}

#[cfg(target_os = "macos")]
fn plist_array<E: CommandExecutor>(executor: &E, path: &Path, key: &str) -> Vec<String> {
    let mut values = Vec::new();
    for index in 0..64 {
        let indexed = format!("{key}:{index}");
        let Some(value) = plist_value(executor, path, &indexed) else {
            break;
        };
        values.push(value);
    }
    values
}

#[cfg(target_os = "macos")]
fn checked_output<E: CommandExecutor>(
    executor: &E,
    program: &str,
    args: &[&str],
) -> Result<(), UpgradeError> {
    checked_output_os(
        executor,
        OsStr::new(program),
        &args.iter().map(OsString::from).collect::<Vec<_>>(),
    )
}

#[cfg(target_os = "macos")]
fn checked_output_os<E: CommandExecutor>(
    executor: &E,
    program: &OsStr,
    args: &[OsString],
) -> Result<(), UpgradeError> {
    let output = executor.output(program, args)?;
    if output.success {
        Ok(())
    } else {
        let command = format!(
            "{} {}",
            program.to_string_lossy(),
            args.iter()
                .map(|arg| arg.to_string_lossy())
                .collect::<Vec<_>>()
                .join(" ")
        );
        Err(command_failure(&command, &output.stderr))
    }
}

#[cfg(target_os = "macos")]
fn command_failure(command: &str, stderr: &[u8]) -> UpgradeError {
    let diagnostic = String::from_utf8_lossy(stderr);
    let diagnostic = diagnostic.trim();
    UpgradeError::Activation(if diagnostic.is_empty() {
        format!("'{command}' failed")
    } else {
        format!("'{command}' failed: {diagnostic}")
    })
}

#[cfg(any(target_os = "macos", test))]
fn parse_launchctl_list(raw: &str) -> Vec<(String, u32)> {
    raw.lines()
        .filter_map(|line| {
            let mut fields = line.split_whitespace();
            let pid = fields.next()?.parse::<u32>().ok()?;
            let _status = fields.next()?;
            let label = fields.next()?.to_string();
            Some((label, pid))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "macos")]
    use crate::upgrade::{CommandOutput, CommandStatus};
    #[cfg(target_os = "macos")]
    use std::sync::Mutex;

    #[test]
    fn launchctl_parser_keeps_only_running_jobs() {
        let raw = "PID\tStatus\tLabel\n123\t0\tio.example.running\n-\t0\tio.example.stopped\n";
        assert_eq!(
            parse_launchctl_list(raw),
            vec![("io.example.running".to_string(), 123)]
        );
    }

    #[test]
    fn http_config_applies_cli_over_allowlisted_environment() {
        let arguments = vec![
            "--http".into(),
            "--host".into(),
            "0.0.0.0".into(),
            "--port".into(),
            "4010".into(),
        ];
        assert_eq!(
            super::super::http_service_config(&arguments, None, Some("127.0.0.1"), Some("4000")),
            Some(("0.0.0.0".into(), 4010))
        );
        assert_eq!(
            super::super::http_service_config(&[], Some("stdio"), None, None),
            None
        );
    }

    #[cfg(target_os = "macos")]
    struct FakeLaunchd {
        executable: String,
        queries: Mutex<Vec<String>>,
    }

    #[cfg(target_os = "macos")]
    impl CommandExecutor for FakeLaunchd {
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
            self.queries
                .lock()
                .expect("queries should lock")
                .push(command);
            let stdout = if program == "launchctl" {
                "PID\tStatus\tLabel\n321\t0\tio.obsidian.mcp\n".to_string()
            } else if program == "lsof" {
                format!("p321\nftxt\nn{}\n", self.executable)
            } else {
                let key = args.get(1).and_then(|arg| arg.to_str()).unwrap_or_default();
                match key {
                    "Print :Label" => "io.obsidian.mcp".into(),
                    "Print :ProgramArguments:0" => self.executable.clone(),
                    "Print :ProgramArguments:1" => "--http".into(),
                    "Print :ProgramArguments:2" => "--port".into(),
                    "Print :ProgramArguments:3" => "4567".into(),
                    _ => {
                        return Ok(CommandOutput {
                            success: false,
                            code: Some(1),
                            stdout: Vec::new(),
                            stderr: Vec::new(),
                        });
                    }
                }
            };
            Ok(CommandOutput {
                success: true,
                code: Some(0),
                stdout: stdout.into_bytes(),
                stderr: Vec::new(),
            })
        }

        fn status_inherit(
            &self,
            _program: &OsStr,
            _args: &[OsString],
        ) -> Result<CommandStatus, UpgradeError> {
            panic!("discovery must not run inherited commands")
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn discovery_reads_only_allowlisted_environment_keys() {
        let temp = tempfile::tempdir().expect("tempdir should be created");
        let executable = temp.path().join("obsidian-mcp");
        std::fs::write(&executable, "binary").expect("executable should be written");
        let agents = temp.path().join("LaunchAgents");
        std::fs::create_dir(&agents).expect("agents dir should be created");
        std::fs::write(agents.join("io.obsidian.mcp.plist"), "plist")
            .expect("plist should be written");
        let fake = FakeLaunchd {
            executable: executable.display().to_string(),
            queries: Mutex::new(Vec::new()),
        };

        let targets = discover_in_directory(&fake, &executable.canonicalize().unwrap(), &agents)
            .expect("agent should be discovered");
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].port, 4567);
        let queries = fake.queries.lock().expect("queries should lock");
        assert!(
            queries
                .iter()
                .any(|query| { query.contains(":EnvironmentVariables:OBSIDIAN_HTTP_PORT") })
        );
        assert!(
            !queries
                .iter()
                .any(|query| query.contains("Print :EnvironmentVariables "))
        );
    }

    #[cfg(target_os = "macos")]
    struct FakeRestart {
        commands: Mutex<Vec<String>>,
        executable: String,
    }

    #[cfg(target_os = "macos")]
    impl CommandExecutor for FakeRestart {
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
            let list_calls = {
                let mut commands = self.commands.lock().expect("commands should lock");
                commands.push(command);
                commands
                    .iter()
                    .filter(|command| *command == "launchctl list")
                    .count()
            };
            let stdout = if program == "id" {
                b"501\n".to_vec()
            } else if program == "lsof" {
                format!("p654\nftxt\nn{}\n", self.executable).into_bytes()
            } else if args == [OsString::from("list")] {
                format!(
                    "PID\tStatus\tLabel\n{}\t0\tio.obsidian.mcp\n",
                    if list_calls == 1 { 321 } else { 654 }
                )
                .into_bytes()
            } else {
                Vec::new()
            };
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
            panic!("restart must use captured launchctl commands")
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn restart_uses_bootout_then_bootstrap_and_verifies_version() {
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
        let fake = FakeRestart {
            commands: Mutex::new(Vec::new()),
            executable: "/tmp/obsidian-mcp".into(),
        };
        let target = ServiceTarget {
            owner: ServiceOwner::Launchd,
            id: "io.obsidian.mcp".into(),
            definition_path: Some("/tmp/io.obsidian.mcp.plist".into()),
            executable: "/tmp/obsidian-mcp".into(),
            host: "127.0.0.1".into(),
            port,
            previous_pid: Some(321),
            observed_version: Some("2.4.0".into()),
        };

        let result = restart(&fake, &target, "2.5.0");
        assert!(result.success, "{}", result.diagnostic);
        let commands = fake.commands.lock().expect("commands should lock");
        let bootout = commands
            .iter()
            .position(|command| command.contains("bootout gui/501/io.obsidian.mcp"))
            .expect("bootout should run");
        let bootstrap = commands
            .iter()
            .position(|command| command.contains("bootstrap gui/501 /tmp/io.obsidian.mcp.plist"))
            .expect("bootstrap should run");
        assert!(bootout < bootstrap);
        assert_eq!(
            commands
                .iter()
                .filter(|command| command.starts_with("lsof "))
                .count(),
            2,
            "live executable identity must be checked before and after restart"
        );
        assert!(!commands.iter().any(|command| command.contains("kickstart")));
    }
}
