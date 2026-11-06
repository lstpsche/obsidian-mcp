//! Operator-selected SFTP source locations and non-interactive SSH invocation.

use std::ffi::OsString;
use std::path::Path;
use std::time::Duration;

use crate::error::{VaultError, VaultResult};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SftpLocation {
    pub host: String,
    pub user: Option<String>,
    pub port: Option<u16>,
    pub root: String,
}

impl SftpLocation {
    pub fn from_path(path: &Path) -> VaultResult<Option<Self>> {
        let Some(text) = path.to_str() else {
            return Ok(None);
        };
        if text
            .get(..7)
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case("sftp://"))
        {
            Self::parse(text).map(Some)
        } else {
            Ok(None)
        }
    }

    pub fn parse(text: &str) -> VaultResult<Self> {
        let invalid = || {
            VaultError::InvalidPath("expected sftp://[user@]host[:port]/absolute/root without credentials, query or fragment".into())
        };
        if !text
            .get(..7)
            .is_some_and(|s| s.eq_ignore_ascii_case("sftp://"))
            || text.chars().any(|c| c.is_control() || c.is_whitespace())
            || text.contains(['?', '#', '\\'])
        {
            return Err(invalid());
        }
        let slash = text[7..].find('/').map(|i| i + 7).ok_or_else(invalid)?;
        let authority = &text[7..slash];
        let url = reqwest::Url::parse(text).map_err(|_| invalid())?;
        if url.password().is_some()
            || url.port() == Some(0)
            || authority.is_empty()
            || authority.matches('@').count() > 1
        {
            return Err(invalid());
        }
        let host = url
            .host_str()
            .ok_or_else(invalid)?
            .trim_matches(['[', ']'])
            .to_owned();
        if host.is_empty() || host.starts_with('-') || host.contains('%') {
            return Err(invalid());
        }
        let user = if url.username().is_empty() {
            None
        } else {
            let user = decode_component(url.username())?;
            if user.contains([':', '/', '\\']) {
                return Err(invalid());
            }
            Some(user)
        };
        let mut parts = Vec::new();
        for part in text[slash + 1..].split('/') {
            if part.is_empty() {
                continue;
            }
            let decoded = decode_component(part)?;
            if matches!(decoded.as_str(), "." | "..") {
                return Err(invalid());
            }
            parts.push(decoded);
        }
        let root = format!("/{}", parts.join("/"));
        Ok(Self {
            host,
            user,
            port: url.port(),
            root,
        })
    }

    pub fn display_uri(&self) -> String {
        let mut result = "sftp://".to_owned();
        if let Some(user) = &self.user {
            result.push_str(&encode_component(user));
            result.push('@');
        }
        if self.host.contains(':') {
            result.push('[')
        }
        result.push_str(&self.host);
        if self.host.contains(':') {
            result.push(']')
        }
        if let Some(port) = self.port {
            result.push_str(&format!(":{port}"))
        }
        result.push('/');
        result.push_str(
            &self
                .root
                .trim_start_matches('/')
                .split('/')
                .map(encode_component)
                .collect::<Vec<_>>()
                .join("/"),
        );
        result
    }

    fn controlled_args(&self) -> Vec<OsString> {
        let mut args = vec![OsString::from("-T")];
        for option in [
            "BatchMode=yes",
            "StrictHostKeyChecking=yes",
            "ClearAllForwardings=yes",
            "ForwardAgent=no",
            "ForwardX11=no",
            "PermitLocalCommand=no",
            "RemoteCommand=none",
            "ControlPath=none",
        ] {
            args.push("-o".into());
            args.push(option.into());
        }
        if let Some(user) = &self.user {
            args.push("-l".into());
            args.push(user.into())
        }
        if let Some(port) = self.port {
            args.push("-p".into());
            args.push(port.to_string().into())
        }
        args
    }

    /// Arguments are passed directly to OpenSSH; vault names never become shell code.
    pub fn ssh_args(&self) -> Vec<OsString> {
        let mut args = self.controlled_args();
        args.extend([
            "-s".into(),
            "--".into(),
            self.host.clone().into(),
            "sftp".into(),
        ]);
        args
    }

    pub fn ssh_config_args(&self) -> Vec<OsString> {
        let mut args = self.controlled_args();
        args.extend([
            "-G".into(),
            "-s".into(),
            "--".into(),
            self.host.clone().into(),
            "sftp".into(),
        ]);
        args
    }
}

fn decode_component(text: &str) -> VaultResult<String> {
    let invalid =
        || VaultError::InvalidPath("invalid encoded SFTP path or username component".into());
    let mut bytes = Vec::with_capacity(text.len());
    let mut iter = text.bytes();
    while let Some(byte) = iter.next() {
        let byte = if byte == b'%' {
            let high = iter
                .next()
                .and_then(|b| (b as char).to_digit(16))
                .ok_or_else(invalid)?;
            let low = iter
                .next()
                .and_then(|b| (b as char).to_digit(16))
                .ok_or_else(invalid)?;
            (high * 16 + low) as u8
        } else {
            byte
        };
        if byte == b'/' || byte == b'\\' || byte.is_ascii_control() {
            return Err(invalid());
        }
        bytes.push(byte);
    }
    let value = String::from_utf8(bytes).map_err(|_| invalid())?;
    if value.chars().any(char::is_control) {
        return Err(invalid());
    }
    Ok(value)
}

fn encode_component(text: &str) -> String {
    use std::fmt::Write;
    let mut encoded = String::new();
    for byte in text.bytes() {
        if byte.is_ascii_alphanumeric() || b"-._~".contains(&byte) {
            encoded.push(byte as char)
        } else {
            let _ = write!(encoded, "%{byte:02X}");
        }
    }
    encoded
}

#[derive(Debug, Clone, Copy)]
pub struct SftpSettings {
    pub timeout: Duration,
    pub poll_interval: Duration,
    pub verify_interval: Duration,
}

impl Default for SftpSettings {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            poll_interval: Duration::from_secs(300),
            verify_interval: Duration::from_secs(3600),
        }
    }
}

impl SftpSettings {
    pub fn from_env() -> VaultResult<Self> {
        let timeout = setting("OBSIDIAN_SFTP_TIMEOUT_SECS", 30)?;
        let poll_interval = setting("OBSIDIAN_SFTP_POLL_INTERVAL_SECS", 300)?;
        let verify_interval = setting("OBSIDIAN_SFTP_VERIFY_INTERVAL_SECS", 3600)?;
        if verify_interval < poll_interval {
            return Err(VaultError::InvalidPath(
                "SFTP verification interval must be at least the polling interval".into(),
            ));
        }
        Ok(Self {
            timeout,
            poll_interval,
            verify_interval,
        })
    }
}

fn setting(name: &str, default: u64) -> VaultResult<Duration> {
    let value = match std::env::var(name) {
        Ok(value) => value
            .parse::<u64>()
            .ok()
            .filter(|v| (1..=86400).contains(v))
            .ok_or_else(|| {
                VaultError::InvalidPath(format!("{name} must be an integer from 1 to 86400"))
            })?,
        Err(std::env::VarError::NotPresent) => default,
        Err(_) => return Err(VaultError::InvalidPath(format!("{name} must be UTF-8"))),
    };
    Ok(Duration::from_secs(value))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remote_source_round_trips_encoded_names_and_explicit_connection() {
        let uri = "sftp://alice@storage:2222/Notes/caf%C3%A9%20files";
        let parsed = SftpLocation::parse(uri).unwrap();
        assert_eq!(parsed.root, "/Notes/café files");
        assert_eq!(parsed.user.as_deref(), Some("alice"));
        assert_eq!(parsed.port, Some(2222));
        assert_eq!(parsed.display_uri(), uri);
        let ipv6 = SftpLocation::parse("sftp://[::1]:2222/notes").unwrap();
        assert_eq!(ipv6.host, "::1");
        assert_eq!(ipv6.display_uri(), "sftp://[::1]:2222/notes");
    }

    #[test]
    fn rejects_credentials_and_ambiguous_or_unsafe_paths_without_echoing_secrets() {
        for uri in [
            "sftp://user:secret@host/notes",
            "sftp://host",
            "sftp:///notes",
            "sftp://host/a?x=1",
            "sftp://host/a#fragment",
            "sftp://host/a%2fb",
            "sftp://host/a%5Cb",
            "sftp://host/%00",
            "sftp://host/%zz",
            "sftp://host/a/../b",
            "sftp://-oProxyCommand=x/notes",
            "sftp://host:99999/notes",
            "sftp://host:0/notes",
        ] {
            let error = SftpLocation::parse(uri).unwrap_err().to_string();
            assert!(!error.contains("secret"));
        }
    }

    #[test]
    fn subsystem_invocation_preserves_only_explicit_connection_arguments() {
        let source = SftpLocation::parse("sftp://alice@vault-host:2222/notes").unwrap();
        let args = source.ssh_args();
        assert_eq!(
            &args[args.len() - 4..],
            ["-s", "--", "vault-host", "sftp"].map(OsString::from)
        );
        assert!(args.iter().any(|a| a == "StrictHostKeyChecking=yes"));
        assert!(!args.iter().any(|a| a.to_string_lossy().contains("/notes")));
        let identity_args = source.ssh_config_args();
        let same_connection: Vec<_> = identity_args
            .into_iter()
            .filter(|arg| arg != "-G")
            .collect();
        assert_eq!(
            same_connection, args,
            "Match command and sessiontype must resolve the same connection"
        );
        assert!(
            SftpLocation::from_path(Path::new("/tmp/local-vault"))
                .unwrap()
                .is_none()
        );
    }
}
