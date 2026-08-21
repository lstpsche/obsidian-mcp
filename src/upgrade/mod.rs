//! Safe package-upgrade orchestration.

use std::ffi::{OsStr, OsString};
use std::io;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub mod build_info;
pub mod cargo_install;
pub mod health;
pub mod helper;
pub mod launchd;
pub mod orchestrator;
pub mod processes;
pub mod systemd;

pub use build_info::BuildIdentity;

#[derive(Debug, Error)]
pub enum UpgradeError {
    #[error("unsupported installation: {0}")]
    UnsupportedInstall(String),
    #[error("invalid installation metadata: {0}")]
    InvalidInstall(String),
    #[error("command '{program}' failed: {message}")]
    Command { program: String, message: String },
    #[error("upgrade failed: {0}")]
    Install(String),
    #[error("Cargo completed, but the installed binaries failed verification: {0}")]
    PostInstall(String),
    #[error("activation failed: {0}")]
    Activation(String),
    #[error("unsupported active runtime: {0}")]
    UnsupportedRuntime(String),
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandOutput {
    pub success: bool,
    pub code: Option<i32>,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommandStatus {
    pub success: bool,
    pub code: Option<i32>,
}

pub trait CommandExecutor: Send + Sync {
    fn output(&self, program: &OsStr, args: &[OsString]) -> Result<CommandOutput, UpgradeError>;

    fn status_inherit(
        &self,
        program: &OsStr,
        args: &[OsString],
    ) -> Result<CommandStatus, UpgradeError>;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct RealCommandExecutor;

impl CommandExecutor for RealCommandExecutor {
    fn output(&self, program: &OsStr, args: &[OsString]) -> Result<CommandOutput, UpgradeError> {
        let output =
            Command::new(program)
                .args(args)
                .output()
                .map_err(|err| UpgradeError::Command {
                    program: program.to_string_lossy().into_owned(),
                    message: err.to_string(),
                })?;
        Ok(CommandOutput {
            success: output.status.success(),
            code: output.status.code(),
            stdout: output.stdout,
            stderr: output.stderr,
        })
    }

    fn status_inherit(
        &self,
        program: &OsStr,
        args: &[OsString],
    ) -> Result<CommandStatus, UpgradeError> {
        let status = Command::new(program)
            .args(args)
            .stdin(Stdio::inherit())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .status()
            .map_err(|err| UpgradeError::Command {
                program: program.to_string_lossy().into_owned(),
                message: err.to_string(),
            })?;
        Ok(CommandStatus {
            success: status.success(),
            code: status.code(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstallRecord {
    pub root: PathBuf,
    pub profile: String,
    pub target: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ServiceOwner {
    Launchd,
    SystemdUser,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServiceTarget {
    pub owner: ServiceOwner,
    pub id: String,
    pub definition_path: Option<PathBuf>,
    pub executable: PathBuf,
    pub host: String,
    pub port: u16,
    pub previous_pid: Option<u32>,
    pub observed_version: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActivationResult {
    pub target_id: String,
    pub success: bool,
    pub observed_version: Option<String>,
    pub diagnostic: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticReconcileResult {
    pub success: bool,
    pub ownership: String,
    pub was_running: bool,
    pub restarted: bool,
    pub observed_version: Option<String>,
    pub diagnostic: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpgradeReport {
    pub old_identity: BuildIdentity,
    pub new_identity: BuildIdentity,
    pub binaries_changed: bool,
    pub semantic: Option<SemanticReconcileResult>,
    pub services: Vec<ActivationResult>,
    pub stdio_reconnect_required: bool,
    pub activation_attempted: bool,
}

impl UpgradeReport {
    pub fn activation_succeeded(&self) -> bool {
        self.semantic.as_ref().is_none_or(|result| result.success)
            && self.services.iter().all(|result| result.success)
    }
}

#[cfg(any(unix, test))]
pub(crate) fn http_service_config(
    arguments: &[String],
    environment_transport: Option<&str>,
    environment_host: Option<&str>,
    environment_port: Option<&str>,
) -> Option<(String, u16)> {
    const DEFAULT_HOST: &str = "127.0.0.1";
    const DEFAULT_PORT: u16 = 37842;

    let is_http = arguments.iter().any(|argument| argument == "--http")
        || environment_transport.is_some_and(|value| value.eq_ignore_ascii_case("http"));
    if !is_http {
        return None;
    }
    let mut host = environment_host.unwrap_or(DEFAULT_HOST).to_string();
    let mut port = environment_port
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_PORT);
    let mut index = 0;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--host" if index + 1 < arguments.len() => {
                host.clone_from(&arguments[index + 1]);
                index += 1;
            }
            "--port" if index + 1 < arguments.len() => {
                port = arguments[index + 1].parse().ok()?;
                index += 1;
            }
            _ => {}
        }
        index += 1;
    }
    Some((host, port))
}
