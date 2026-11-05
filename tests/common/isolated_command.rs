use tokio::process::Command;

pub fn isolated_command(executable: impl AsRef<std::ffi::OsStr>) -> Command {
    let mut command = Command::new(executable);
    for (key, _) in std::env::vars_os() {
        let name = key.to_string_lossy();
        if name.starts_with("OBSIDIAN_") || name.starts_with("OPENAI_") {
            command.env_remove(key);
        }
    }
    command
}
