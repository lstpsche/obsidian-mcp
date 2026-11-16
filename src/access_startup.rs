//! Explicit opt-in startup options, without changing ordinary public config types.

use std::ffi::OsString;
use std::path::PathBuf;

use obsidian_mcp::access::store::{AccessStore, AuthorizedClient, read_token_file};
use obsidian_mcp::config::{Config, SemanticMode, SemanticRuntimeConfig, Transport};

#[derive(Default, Debug)]
pub struct AccessOptions {
    file: Option<PathBuf>,
    token_file: Option<PathBuf>,
}

pub struct ConfiguredAccess {
    pub store: AccessStore,
    pub client: Option<AuthorizedClient>,
}

impl AccessOptions {
    pub fn parse(
        args: impl IntoIterator<Item = OsString>,
    ) -> Result<(Vec<OsString>, Self), String> {
        let mut args = args.into_iter();
        let mut remaining = Vec::new();
        let mut file = None;
        let mut token_file = None;
        while let Some(arg) = args.next() {
            let setting = match arg.to_str() {
                Some("--access-file") => Some((&mut file, "--access-file")),
                Some("--access-token-file") => Some((&mut token_file, "--access-token-file")),
                _ => None,
            };
            if let Some((slot, name)) = setting {
                if slot.is_some() {
                    return Err(format!("{name} must be supplied only once"));
                }
                let value = args
                    .next()
                    .filter(|value| !value.is_empty())
                    .ok_or_else(|| format!("{name} requires a file path"))?;
                if value.as_encoded_bytes().starts_with(b"-") {
                    return Err(format!("{name} requires a file path"));
                }
                *slot = Some(PathBuf::from(value));
            } else {
                remaining.push(arg);
            }
        }
        Ok((
            remaining,
            Self {
                file: select_setting(file, "OBSIDIAN_ACCESS_FILE")?,
                token_file: select_setting(token_file, "OBSIDIAN_ACCESS_TOKEN_FILE")?,
            },
        ))
    }

    pub fn configure(
        self,
        config: &mut Config,
        semantic: &mut SemanticRuntimeConfig,
    ) -> Result<Option<ConfiguredAccess>, Box<dyn std::error::Error>> {
        let Some(file) = self.file else {
            if self.token_file.is_some() {
                return Err(
                    "an access token file also requires --access-file or OBSIDIAN_ACCESS_FILE"
                        .into(),
                );
            }
            return Ok(None);
        };
        if std::env::var_os("OBSIDIAN_MEMORY_AUTHORIZER_URL").is_some()
            || std::env::var_os("OBSIDIAN_MEMORY_KEY_FILE").is_some()
        {
            return Err(
                "client access and private-memory authorization cannot be configured together"
                    .into(),
            );
        }
        if config.sftp_location()?.is_some() {
            return Err("authenticated clients currently require a local vault; ordinary SFTP access is unchanged".into());
        }
        if matches!(config.transport, Transport::Http) {
            if !config.http_host.is_loopback() {
                return Err("authenticated HTTP must bind to loopback; use HTTPS through a reverse proxy or a protected SSH forward for remote access".into());
            }
            if self.token_file.is_some() {
                return Err("an HTTP server cannot use a shared stdio access token; configure each client's bearer header instead".into());
            }
        }
        if let Some(explicit) = std::env::var_os("OBSIDIAN_SEMANTIC_MODE")
            && !explicit
                .to_str()
                .is_some_and(|value| value.trim().eq_ignore_ascii_case("local"))
        {
            return Err("authenticated clients use MCP-owned semantics; remove OBSIDIAN_SEMANTIC_MODE or set it to local".into());
        }
        let store = AccessStore::open(&file)?;
        store.validate_source(&config.vault_path)?;
        let snapshot = store.snapshot()?;
        if let Some(explicit) = &config.mcp_data_dir {
            if explicit.canonicalize()? != snapshot.policy.data_dir.canonicalize()? {
                return Err("OBSIDIAN_MCP_DATA conflicts with the access profile; remove the override to use the profile's protected data directory".into());
            }
        } else if std::env::var_os("OBSIDIAN_MCP_DATA").is_some() {
            return Err("OBSIDIAN_MCP_DATA must not be empty when access is configured".into());
        }
        config.mcp_data_dir = Some(snapshot.policy.data_dir);
        semantic.mode = SemanticMode::Local;
        let client = if matches!(config.transport, Transport::Stdio) {
            let token_file = self.token_file.ok_or(
                "restricted stdio requires --access-token-file or OBSIDIAN_ACCESS_TOKEN_FILE",
            )?;
            let token = read_token_file(&token_file)?;
            Some(store.authenticate(token.expose())?)
        } else {
            None
        };
        Ok(Some(ConfiguredAccess { store, client }))
    }
}

fn select_setting(argument: Option<PathBuf>, name: &str) -> Result<Option<PathBuf>, String> {
    let environment = std::env::var_os(name).map(PathBuf::from);
    if environment
        .as_ref()
        .is_some_and(|path| path.as_os_str().is_empty())
    {
        return Err(format!("{name} must name a protected file or be unset"));
    }
    if let (Some(argument), Some(environment)) = (&argument, &environment)
        && argument != environment
    {
        return Err(format!(
            "command-line access file conflicts with {name}; remove the stale setting"
        ));
    }
    Ok(argument.or(environment))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn access_options_are_removed_without_changing_vault_or_endpoint_arguments() {
        // Parse extraction directly so unrelated host environment is not changed.
        let args = ["--http", "--port", "1234", "/vault"].map(OsString::from);
        let (remaining, _) = AccessOptions::parse(args.clone()).unwrap();
        assert_eq!(remaining, args);
    }
}
