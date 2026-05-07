//! Utility tools: vault info and open-in-Obsidian.

use std::fmt::Write as _;
use std::path::Path;

use rmcp::model::{CallToolResult, Content};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::VaultError;
use crate::models::VaultStats;
use crate::vault::Vault;

// ── vault_info ──────────────────────────────────────────────────────────

#[derive(Deserialize, JsonSchema, Default)]
pub struct VaultInfoParams {}

#[derive(Serialize, Deserialize, JsonSchema)]
pub struct VaultInfo {
    #[serde(flatten)]
    pub stats: VaultStats,
    pub vault_name: String,
    pub vault_path: String,
}

/// Return aggregate vault statistics.
pub async fn vault_info(
    vault: &Vault,
    _params: VaultInfoParams,
) -> Result<CallToolResult, rmcp::ErrorData> {
    let stats = vault.vault_stats()?;
    let info = VaultInfo {
        stats,
        vault_name: vault
            .root()
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default(),
        vault_path: vault.root().display().to_string(),
    };
    let json = serde_json::to_string_pretty(&info).map_err(|e| VaultError::Other(e.to_string()))?;
    Ok(CallToolResult::success(vec![Content::text(json)]))
}

// ── open_in_obsidian ────────────────────────────────────────────────────

#[derive(Deserialize, JsonSchema, Default)]
pub struct OpenInObsidianParams {
    /// Note path relative to vault root.
    pub path: String,
    /// Open in a new split pane (requires Obsidian Advanced URI plugin).
    #[serde(default)]
    pub new_leaf: bool,
}

/// Open a note in the Obsidian app via the `obsidian://` URI scheme.
pub async fn open_in_obsidian(
    vault: &Vault,
    params: OpenInObsidianParams,
) -> Result<CallToolResult, rmcp::ErrorData> {
    vault.validate_path(Path::new(&params.path))?;

    let vault_name = vault
        .root()
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();

    let mut uri = format!(
        "obsidian://open?vault={}&file={}",
        percent_encode(&vault_name),
        percent_encode(&params.path),
    );

    if params.new_leaf {
        uri.push_str("&openmode=split");
    }

    launch_uri(&uri)?;

    Ok(CallToolResult::success(vec![Content::text(format!(
        "Opened {} in Obsidian",
        params.path
    ))]))
}

/// Percent-encode a string for use in URI query parameters.
fn percent_encode(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for &b in input.as_bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~' | b'/') {
            out.push(b as char);
        } else {
            let _ = write!(out, "%{b:02X}");
        }
    }
    out
}

/// Launch a URI using the platform's default handler.
fn launch_uri(uri: &str) -> Result<(), VaultError> {
    let result = if cfg!(target_os = "macos") {
        std::process::Command::new("open").arg(uri).spawn()
    } else if cfg!(target_os = "linux") {
        std::process::Command::new("xdg-open").arg(uri).spawn()
    } else if cfg!(target_os = "windows") {
        std::process::Command::new("explorer").arg(uri).spawn()
    } else {
        return Err(VaultError::Other(
            "unsupported platform for opening URIs".into(),
        ));
    };

    result
        .map(|_| ())
        .map_err(|e| VaultError::Other(format!("failed to launch Obsidian URI: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{create_test_vault, test_config};

    #[tokio::test]
    async fn vault_info_returns_stats() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        vault.write_note(Path::new("note.md"), "# Hello").unwrap();

        let result = vault_info(&vault, VaultInfoParams {}).await;
        assert!(result.is_ok());

        let call_result = result.unwrap();
        let json_str = call_result.content[0]
            .as_text()
            .expect("expected text content")
            .text
            .as_str();
        let info: VaultInfo = serde_json::from_str(json_str).unwrap();
        assert_eq!(info.stats.total_notes, 1);
        assert!(!info.vault_path.is_empty());
    }

    #[tokio::test]
    async fn open_in_obsidian_rejects_path_traversal() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        let result = open_in_obsidian(
            &vault,
            OpenInObsidianParams {
                path: "../../etc/passwd".into(),
                new_leaf: false,
            },
        )
        .await;

        assert!(result.is_err());
    }

    #[test]
    fn percent_encode_preserves_safe_chars() {
        assert_eq!(percent_encode("hello"), "hello");
        assert_eq!(percent_encode("path/to/note"), "path/to/note");
        assert_eq!(percent_encode("my-note_v2.md"), "my-note_v2.md");
    }

    #[test]
    fn percent_encode_encodes_special_chars() {
        assert_eq!(percent_encode("hello world"), "hello%20world");
        assert_eq!(percent_encode("a&b=c"), "a%26b%3Dc");
        assert_eq!(percent_encode("100%"), "100%25");
    }

    #[test]
    fn percent_encode_handles_unicode() {
        let encoded = percent_encode("café");
        assert!(encoded.contains('%'));
        assert!(encoded.starts_with("caf"));
    }
}
