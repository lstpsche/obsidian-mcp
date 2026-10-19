//! Utility tools: vault info and open-in-Obsidian.

use std::fmt::Write as _;
use std::path::Path;

use rmcp::model::CallToolResult;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::daemon::protocol::{self, SemanticPhase, SemanticStatus};
use crate::error::VaultError;
use crate::models::VaultStats;
use crate::vault::Vault;

use super::SemanticRuntime;

// ── vault_info ──────────────────────────────────────────────────────────

#[derive(Deserialize, JsonSchema, Default)]
pub struct VaultInfoParams {}

#[derive(Serialize, Deserialize, JsonSchema)]
pub struct VaultInfo {
    #[serde(flatten)]
    pub stats: VaultStats,
    pub vault_name: String,
    pub vault_path: String,
    /// Active exclusion glob patterns (merged from all sources).
    pub exclude_patterns: Vec<String>,
    /// Resolved external data directory, or `null` when using default (vault-root `.obsidian-mcp/`).
    pub mcp_data_dir: Option<String>,
    /// Passive status of the selected semantic backend; inspection never activates indexing.
    pub embeddings: EmbeddingInfo,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingSource {
    Local,
    Daemon,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingAvailability {
    Available,
    Disabled,
    Unavailable,
    Unattached,
    Unsupported,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct EmbeddingSnapshot {
    pub phase: SemanticPhase,
    /// A compatible snapshot can remain queryable while warming or degraded.
    pub queryable: bool,
    pub indexed_notes: usize,
    pub total_notes: usize,
    pub pending_notes: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_error: Option<String>,
}

impl From<SemanticStatus> for EmbeddingSnapshot {
    fn from(status: SemanticStatus) -> Self {
        Self {
            phase: status.phase,
            queryable: status.ready,
            indexed_notes: status.indexed_notes,
            total_notes: status.total_notes,
            pending_notes: status.pending_notes,
            last_error: status.last_error,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct EmbeddingInfo {
    pub source: EmbeddingSource,
    pub availability: EmbeddingAvailability,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<EmbeddingSnapshot>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

async fn embedding_info(vault: &Vault, runtime: &SemanticRuntime) -> EmbeddingInfo {
    if runtime.uses_local_backend() {
        #[cfg(has_embeddings)]
        {
            let status = vault.embedding_status().map(|status| EmbeddingSnapshot {
                phase: match status.phase {
                    crate::vault::embedding_runtime::EmbeddingPhase::Warming => {
                        SemanticPhase::Warming
                    }
                    crate::vault::embedding_runtime::EmbeddingPhase::Ready => SemanticPhase::Ready,
                    crate::vault::embedding_runtime::EmbeddingPhase::Degraded => {
                        SemanticPhase::Degraded
                    }
                },
                queryable: status.queryable,
                indexed_notes: status.indexed_notes,
                total_notes: status.total_notes,
                pending_notes: status.pending_notes,
                last_error: status.last_error,
            });
            return EmbeddingInfo {
                source: EmbeddingSource::Local,
                availability: if status.is_some() {
                    EmbeddingAvailability::Available
                } else {
                    EmbeddingAvailability::Disabled
                },
                status,
                error: None,
            };
        }
        #[cfg(not(has_embeddings))]
        return EmbeddingInfo {
            source: EmbeddingSource::Local,
            availability: EmbeddingAvailability::Unsupported,
            status: None,
            error: Some("this binary was compiled without a local embedding backend".into()),
        };
    }

    let Some(client) = &runtime.daemon_client else {
        return EmbeddingInfo {
            source: EmbeddingSource::Daemon,
            availability: EmbeddingAvailability::Unavailable,
            status: None,
            error: Some(
                runtime
                    .daemon_unavailable_reason
                    .clone()
                    .unwrap_or_else(|| "semantic daemon client is not initialized".into()),
            ),
        };
    };
    let status = async {
        runtime.ensure_daemon_health().await?;
        client.vault_status(vault.root()).await
    }
    .await;
    match status {
        Ok(status) => EmbeddingInfo {
            source: EmbeddingSource::Daemon,
            availability: EmbeddingAvailability::Available,
            status: Some(status.into()),
            error: None,
        },
        Err(error) => EmbeddingInfo {
            source: EmbeddingSource::Daemon,
            availability: match &error {
                VaultError::DaemonRpc { code, .. } if *code == protocol::ERR_VAULT_NOT_ATTACHED => {
                    EmbeddingAvailability::Unattached
                }
                VaultError::DaemonRpc { code, .. } if *code == protocol::ERR_METHOD_NOT_FOUND => {
                    EmbeddingAvailability::Unsupported
                }
                _ => EmbeddingAvailability::Unavailable,
            },
            status: None,
            error: Some(error.to_string()),
        },
    }
}

/// Return aggregate vault statistics.
pub async fn vault_info(
    vault: &Vault,
    runtime: &SemanticRuntime,
    _params: VaultInfoParams,
) -> Result<CallToolResult, rmcp::ErrorData> {
    let stats = vault.vault_stats()?;
    let mcp_data_dir = if vault.mcp_data() != vault.mcp_home() {
        Some(vault.mcp_data().display().to_string())
    } else {
        None
    };
    let info = VaultInfo {
        stats,
        vault_name: vault
            .root()
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default(),
        vault_path: vault.root().display().to_string(),
        exclude_patterns: vault.exclude().patterns().to_vec(),
        mcp_data_dir,
        embeddings: embedding_info(vault, runtime).await,
    };
    super::output::json(&info)
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
    let requested_path = Path::new(&params.path);
    let uri_path = match vault.canonical_existing_relative_path(requested_path) {
        Ok(path) => path.to_string_lossy().into_owned(),
        Err(VaultError::NoteNotFound(_)) => {
            vault.validate_path(requested_path)?;
            params.path.clone()
        }
        Err(err) => return Err(err.into()),
    };

    let vault_name = vault
        .root()
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();

    let mut uri = format!(
        "obsidian://open?vault={}&file={}",
        percent_encode(&vault_name),
        percent_encode(&uri_path),
    );

    if params.new_leaf {
        uri.push_str("&openmode=split");
    }

    launch_uri(&uri)?;

    super::output::message(format!("Opened {} in Obsidian", uri_path))
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
    use crate::config::SemanticMode;
    use crate::test_helpers::{create_test_vault, test_config};
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    fn test_runtime(mode: SemanticMode) -> SemanticRuntime {
        SemanticRuntime {
            mode,
            daemon_client: None,
            daemon_unavailable_reason: None,
            prefetch_count: 50,
            vault_ensured: Arc::new(AtomicBool::new(false)),
            local_fallback_active: Arc::new(AtomicBool::new(false)),
            daemon_health_pending: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    fn assert_info_schema(result: &CallToolResult) -> serde_json::Value {
        let structured = result.structured_content.as_ref().unwrap();
        let schema = serde_json::to_value(schemars::schema_for!(VaultInfo)).unwrap();
        let validator = jsonschema::validator_for(&schema).unwrap();
        validator.validate(structured).unwrap();
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(crate::test_helpers::extract_text(result))
                .unwrap(),
            *structured
        );
        structured.clone()
    }

    #[tokio::test]
    async fn vault_info_returns_stats() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        vault.write_note(Path::new("note.md"), "# Hello").unwrap();

        let result = vault_info(
            &vault,
            &test_runtime(SemanticMode::Local),
            VaultInfoParams {},
        )
        .await;
        assert!(result.is_ok());

        let call_result = result.unwrap();
        assert_info_schema(&call_result);
        let json_str = call_result.content[0]
            .as_text()
            .expect("expected text content")
            .text
            .as_str();
        let info: VaultInfo = serde_json::from_str(json_str).unwrap();
        assert_eq!(info.stats.total_notes, 1);
        assert_eq!(info.stats.excluded_notes, 0);
        assert!(info.exclude_patterns.is_empty());
        assert!(info.mcp_data_dir.is_none());
        assert!(!info.vault_path.is_empty());
        assert_eq!(info.embeddings.source, EmbeddingSource::Local);
        assert!(info.embeddings.status.is_none());
        #[cfg(has_embeddings)]
        assert_eq!(
            info.embeddings.availability,
            EmbeddingAvailability::Disabled
        );
        #[cfg(not(has_embeddings))]
        assert_eq!(
            info.embeddings.availability,
            EmbeddingAvailability::Unsupported
        );
    }

    #[tokio::test]
    async fn vault_info_reports_daemon_unavailable_without_changing_owner() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();
        let mut runtime = test_runtime(SemanticMode::Auto);
        runtime.daemon_unavailable_reason = Some("daemon connection timed out".into());
        let result = vault_info(&vault, &runtime, VaultInfoParams {})
            .await
            .unwrap();
        let info = assert_info_schema(&result);
        assert_eq!(info["embeddings"]["source"], "daemon");
        assert_eq!(info["embeddings"]["availability"], "unavailable");
        assert_eq!(info["embeddings"]["error"], "daemon connection timed out");
        assert!(info["embeddings"].get("status").is_none());
        assert!(!runtime.uses_local_backend());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn vault_info_uses_only_passive_daemon_status_and_preserves_provenance() {
        use crate::client::semantic_daemon::{DaemonConnectPolicy, SemanticDaemonClient};
        use crate::daemon::server::IpcEndpoint;
        use serde_json::json;
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

        let cases = [
            (
                json!({"result": {
                    "phase": "degraded", "ready": true, "indexed_notes": 2,
                    "total_notes": 2, "pending_notes": 2, "last_error": "embedding API request timed out"
                }}),
                "available",
            ),
            (
                json!({"error": {"code": protocol::ERR_VAULT_NOT_ATTACHED, "message": "vault not attached"}}),
                "unattached",
            ),
            (
                json!({"error": {"code": protocol::ERR_METHOD_NOT_FOUND, "message": "unknown method"}}),
                "unsupported",
            ),
            (
                json!({"error": {"code": protocol::ERR_INTERNAL, "message": "status unavailable"}}),
                "unavailable",
            ),
        ];
        for (mut response, availability) in cases {
            let dir = tempfile::tempdir().unwrap();
            create_test_vault(dir.path());
            let vault = Vault::open(&test_config(dir.path())).await.unwrap();
            let socket_path = dir.path().join("semanticd.sock");
            let listener = tokio::net::UnixListener::bind(&socket_path).unwrap();
            let expected_root = vault.root().display().to_string();
            let server = tokio::spawn(async move {
                let (stream, _) = listener.accept().await.unwrap();
                let (reader, mut writer) = stream.into_split();
                let mut line = String::new();
                BufReader::new(reader).read_line(&mut line).await.unwrap();
                let request: serde_json::Value = serde_json::from_str(&line).unwrap();
                assert_eq!(request["method"], "vault_status");
                assert_eq!(request["params"], json!({"vault_root": expected_root}));
                response["jsonrpc"] = json!("2.0");
                response["id"] = request["id"].clone();
                writer
                    .write_all(format!("{response}\n").as_bytes())
                    .await
                    .unwrap();
            });
            let mut runtime = test_runtime(SemanticMode::Auto);
            runtime.daemon_client = Some(SemanticDaemonClient::new(
                IpcEndpoint::UnixSocket(socket_path),
                DaemonConnectPolicy {
                    retries: 0,
                    ..Default::default()
                },
            ));
            let result = vault_info(&vault, &runtime, VaultInfoParams {})
                .await
                .unwrap();
            let info = assert_info_schema(&result);
            assert_eq!(info["embeddings"]["source"], "daemon");
            assert_eq!(info["embeddings"]["availability"], availability);
            if availability == "available" {
                assert_eq!(info["embeddings"]["status"]["phase"], "degraded");
                assert_eq!(info["embeddings"]["status"]["queryable"], true);
                assert_eq!(info["embeddings"]["status"]["indexed_notes"], 2);
                assert_eq!(info["embeddings"]["status"]["pending_notes"], 2);
            } else {
                assert!(info["embeddings"].get("status").is_none());
                assert!(info["embeddings"]["error"].is_string());
            }
            assert!(
                !runtime
                    .vault_ensured
                    .load(std::sync::atomic::Ordering::Relaxed)
            );
            assert!(!runtime.uses_local_backend());
            server.await.unwrap();
        }
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
