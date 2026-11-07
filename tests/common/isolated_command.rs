use tokio::process::Command;

pub fn isolated_command(executable: impl AsRef<std::ffi::OsStr>) -> Command {
    let mut command = Command::new(executable);
    for (key, _) in std::env::vars_os() {
        let name = key.to_string_lossy();
        if name.starts_with("OBSIDIAN_") || name.starts_with("OPENAI_") {
            command.env_remove(key);
        }
    }
    command.env("NO_COLOR", "1");
    command
}

#[cfg(all(test, unix))]
mod tests {
    #[tokio::test]
    async fn subprocesses_receive_plain_log_configuration() {
        let output = super::isolated_command("/bin/sh")
            .args(["-c", "printf '%s' \"$NO_COLOR\""])
            .output()
            .await
            .unwrap();
        assert!(output.status.success());
        assert_eq!(output.stdout, b"1");
    }
}
