//! Note metadata inspection and frontmatter manipulation tools.

use std::path::{Path, PathBuf};

use rmcp::model::{CallToolResult, ErrorCode};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::VaultError;
use crate::models::{FileStat, Heading, WikiLink};
use crate::vault::Vault;

// ── note_inspect ───────────────────────────────────────────────────────

/// Parameters for the `note_inspect` tool.
#[derive(Deserialize, JsonSchema, Default)]
pub struct NoteInspectParams {
    /// Path to the note, relative to vault root.
    pub path: String,
    /// View to return: `"metadata"` (default) for rich note metadata, or `"targets"` for patchable headings/blocks/frontmatter fields.
    pub view: Option<String>,
}

#[derive(Serialize, JsonSchema)]
pub(super) struct NoteMetadataOutput {
    path: PathBuf,
    title: String,
    tags: Vec<String>,
    frontmatter: Option<serde_json::Value>,
    headings: Vec<Heading>,
    outgoing_links: Vec<WikiLink>,
    block_refs: Vec<String>,
    backlinks_count: usize,
    stat: FileStat,
}

#[derive(Serialize, JsonSchema)]
#[serde(untagged)]
#[schemars(extend("type" = "object"))]
pub(super) enum NoteInspectOutput {
    Metadata(NoteMetadataOutput),
    Targets(crate::models::DocumentMap),
}

#[derive(Serialize, JsonSchema)]
#[serde(tag = "action", rename_all = "snake_case")]
#[schemars(extend("type" = "object"))]
pub(super) enum FrontmatterOutput {
    Get {
        /// The note's properties, or null when no frontmatter is present.
        #[schemars(required, schema_with = "super::json_value_schema")]
        frontmatter: Option<serde_json::Value>,
    },
    Set {
        message: String,
    },
    Remove {
        message: String,
    },
}

/// Inspect a note's metadata or patch targets.
pub async fn note_inspect(
    vault: &Vault,
    params: NoteInspectParams,
) -> Result<CallToolResult, rmcp::ErrorData> {
    let view = params.view.as_deref().unwrap_or("metadata");

    if view.eq_ignore_ascii_case("metadata") {
        note_inspect_metadata(vault, &params.path).await
    } else if view.eq_ignore_ascii_case("targets") {
        note_inspect_targets(vault, &params.path).await
    } else {
        Err(rmcp::ErrorData::new(
            ErrorCode::INVALID_PARAMS,
            format!("Unknown view '{view}'. Valid values: \"metadata\", \"targets\""),
            None::<serde_json::Value>,
        ))
    }
}

async fn note_inspect_metadata(
    vault: &Vault,
    note_path: &str,
) -> Result<CallToolResult, rmcp::ErrorData> {
    let path = Path::new(note_path);
    let meta = vault.get_note_metadata(path)?;
    let backlinks = vault.backlinks(path)?;

    let output = NoteMetadataOutput {
        path: meta.path,
        title: meta.title,
        tags: meta.tags,
        frontmatter: meta.frontmatter,
        headings: meta.headings,
        outgoing_links: meta.links,
        block_refs: meta.block_refs,
        backlinks_count: backlinks.len(),
        stat: meta.stat,
    };

    let value = serde_json::to_value(NoteInspectOutput::Metadata(output))
        .map_err(|e| VaultError::Other(format!("serialization error: {e}")))?;
    Ok(CallToolResult::structured(value))
}

async fn note_inspect_targets(
    vault: &Vault,
    note_path: &str,
) -> Result<CallToolResult, rmcp::ErrorData> {
    let path = Path::new(note_path);
    let map = vault.get_document_map(path)?;

    let value = serde_json::to_value(NoteInspectOutput::Targets(map))
        .map_err(|e| VaultError::Other(format!("serialization error: {e}")))?;
    Ok(CallToolResult::structured(value))
}

// ── frontmatter ────────────────────────────────────────────────────────

/// Parameters for the `frontmatter` tool.
#[derive(Deserialize, JsonSchema, Default)]
pub struct FrontmatterParams {
    /// Action to perform: `"get"` (return all frontmatter), `"set"` (upsert a field), or `"remove"` (delete a field).
    pub action: String,
    /// Path to the note, relative to vault root.
    pub path: String,
    /// Frontmatter key. Required for `"set"` and `"remove"` actions.
    pub key: Option<String>,
    /// JSON value to assign. Required for `"set"` action. Pass arrays and objects
    /// directly; a JSON-encoded string is stored as a literal string.
    #[serde(
        default,
        deserialize_with = "crate::tools::deserialize_optional_json_value"
    )]
    #[schemars(schema_with = "crate::tools::json_value_schema")]
    pub value: Option<serde_json::Value>,
}

/// Read, set, or remove frontmatter fields on a note.
pub async fn frontmatter(
    vault: &Vault,
    params: FrontmatterParams,
) -> Result<CallToolResult, rmcp::ErrorData> {
    let path = Path::new(&params.path);

    if params.action.eq_ignore_ascii_case("get") {
        let fm = vault.get_frontmatter(path)?;
        let text = serde_json::to_string(&fm)
            .map_err(|error| VaultError::Other(format!("JSON serialization failed: {error}")))?;
        super::output::with_text(&FrontmatterOutput::Get { frontmatter: fm }, text)
    } else if params.action.eq_ignore_ascii_case("set") {
        let key = params.key.as_deref().ok_or_else(|| {
            rmcp::ErrorData::new(
                ErrorCode::INVALID_PARAMS,
                "'key' is required for action 'set'",
                None::<serde_json::Value>,
            )
        })?;
        let value = params.value.ok_or_else(|| {
            rmcp::ErrorData::new(
                ErrorCode::INVALID_PARAMS,
                "'value' is required for action 'set'",
                None::<serde_json::Value>,
            )
        })?;
        vault.set_frontmatter_field(path, key, value)?;
        let message = format!("Set frontmatter field '{key}' on '{}'", params.path);
        super::output::with_text(
            &FrontmatterOutput::Set {
                message: message.clone(),
            },
            message,
        )
    } else if params.action.eq_ignore_ascii_case("remove") {
        let key = params.key.as_deref().ok_or_else(|| {
            rmcp::ErrorData::new(
                ErrorCode::INVALID_PARAMS,
                "'key' is required for action 'remove'",
                None::<serde_json::Value>,
            )
        })?;
        vault.remove_frontmatter_field(path, key)?;
        let message = format!("Removed frontmatter field '{key}' from '{}'", params.path);
        super::output::with_text(
            &FrontmatterOutput::Remove {
                message: message.clone(),
            },
            message,
        )
    } else {
        Err(rmcp::ErrorData::new(
            ErrorCode::INVALID_PARAMS,
            format!(
                "Unknown action '{}'. Valid values: \"get\", \"set\", \"remove\"",
                params.action
            ),
            None::<serde_json::Value>,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{PatchOperation, PatchTargetType};
    use crate::test_helpers::{create_test_vault, test_config};
    use crate::tools::notes::{NotePatchParams, note_patch};

    #[tokio::test]
    async fn note_inspect_metadata_returns_all_fields() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        vault
            .write_note(
                Path::new("test.md"),
                "---\ntags: [rust]\nstatus: draft\n---\n# Heading\n## Sub\n[[other]] #inline\n^block1\n",
            )
            .unwrap();
        vault
            .write_note(Path::new("other.md"), "# Other\n[[test]]\n")
            .unwrap();

        let result = note_inspect(
            &vault,
            NoteInspectParams {
                path: "test.md".into(),
                view: None,
            },
        )
        .await
        .unwrap();

        let v = result.structured_content.unwrap();
        assert_eq!(v["title"], "test");
        assert!(
            v["tags"]
                .as_array()
                .unwrap()
                .contains(&serde_json::json!("rust"))
        );
        assert!(v["frontmatter"].is_object());
        assert!(!v["headings"].as_array().unwrap().is_empty());
        assert!(!v["outgoing_links"].as_array().unwrap().is_empty());
        assert!(
            v["block_refs"]
                .as_array()
                .unwrap()
                .contains(&serde_json::json!("block1"))
        );
        assert_eq!(v["backlinks_count"], 1);
        assert!(v["stat"].is_object());
    }

    #[tokio::test]
    async fn note_inspect_not_found() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        let result = note_inspect(
            &vault,
            NoteInspectParams {
                path: "nonexistent.md".into(),
                view: None,
            },
        )
        .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn note_inspect_targets_lists_targets() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        vault
            .write_note(
                Path::new("mapped.md"),
                "---\ntags: [rust]\ndate: 2026-01-01\n---\n# Heading\n## Sub\nText ^block1\n",
            )
            .unwrap();

        let result = note_inspect(
            &vault,
            NoteInspectParams {
                path: "mapped.md".into(),
                view: Some("targets".into()),
            },
        )
        .await
        .unwrap();

        let v = result.structured_content.unwrap();
        let headings = v["headings"].as_array().unwrap();
        assert!(
            headings
                .iter()
                .any(|h| h.as_str().unwrap().contains("Heading"))
        );
        assert!(headings.iter().any(|h| h.as_str().unwrap().contains("Sub")));
        assert!(
            v["block_refs"]
                .as_array()
                .unwrap()
                .contains(&serde_json::json!("block1"))
        );
        assert!(
            v["frontmatter_fields"]
                .as_array()
                .unwrap()
                .contains(&serde_json::json!("tags"))
        );
        assert!(
            v["frontmatter_fields"]
                .as_array()
                .unwrap()
                .contains(&serde_json::json!("date"))
        );
    }

    #[tokio::test]
    async fn note_inspect_targets_heading_can_be_used_for_note_patch() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        vault
            .write_note(
                Path::new("scratch.md"),
                "# Scratch\n\n## Log\n\n| Date | Update |\n| ---- | ------ |\n",
            )
            .unwrap();

        let result = note_inspect(
            &vault,
            NoteInspectParams {
                path: "scratch.md".into(),
                view: Some("targets".into()),
            },
        )
        .await
        .unwrap();

        let v = result.structured_content.unwrap();
        let headings = v["headings"].as_array().unwrap();
        let target = headings
            .iter()
            .find_map(|h| {
                let heading = h.as_str().unwrap();
                (heading == "## Log").then(|| heading.to_string())
            })
            .expect("targets view should return marker-prefixed heading");

        note_patch(
            &vault,
            NotePatchParams {
                path: "scratch.md".into(),
                operation: PatchOperation::Append,
                target_type: PatchTargetType::Heading,
                target,
                content: "| 2026-02-02 | x |".into(),
            },
        )
        .await
        .unwrap();

        let content = vault.read_note(Path::new("scratch.md")).unwrap();
        let log_idx = content.find("## Log").unwrap();
        let appended_idx = content.find("| 2026-02-02 | x |").unwrap();
        assert!(appended_idx > log_idx);
    }

    #[tokio::test]
    async fn note_inspect_invalid_view() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        vault.write_note(Path::new("test.md"), "# Note\n").unwrap();

        let result = note_inspect(
            &vault,
            NoteInspectParams {
                path: "test.md".into(),
                view: Some("invalid".into()),
            },
        )
        .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn frontmatter_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        vault
            .write_note(Path::new("fm.md"), "# Note\nBody\n")
            .unwrap();

        let result = frontmatter(
            &vault,
            FrontmatterParams {
                action: "get".into(),
                path: "fm.md".into(),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(
            result.structured_content,
            Some(serde_json::json!({"action": "get", "frontmatter": null}))
        );
        let text = result.content[0].as_text().expect("expected text content");
        assert_eq!(text.text, "null");

        frontmatter(
            &vault,
            FrontmatterParams {
                action: "set".into(),
                path: "fm.md".into(),
                key: Some("status".into()),
                value: Some(serde_json::json!("draft")),
            },
        )
        .await
        .unwrap();

        let result = frontmatter(
            &vault,
            FrontmatterParams {
                action: "get".into(),
                path: "fm.md".into(),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let fm = result.structured_content.unwrap()["frontmatter"].clone();
        assert_eq!(fm["status"], "draft");

        frontmatter(
            &vault,
            FrontmatterParams {
                action: "set".into(),
                path: "fm.md".into(),
                key: Some("tags".into()),
                value: Some(serde_json::json!(["rust", "mcp"])),
            },
        )
        .await
        .unwrap();

        let result = frontmatter(
            &vault,
            FrontmatterParams {
                action: "GET".into(),
                path: "fm.md".into(),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let fm = result.structured_content.unwrap()["frontmatter"].clone();
        assert_eq!(fm["status"], "draft");
        assert_eq!(fm["tags"], serde_json::json!(["rust", "mcp"]));

        for (key, value) in [
            ("empty", serde_json::Value::Null),
            ("literal_json", serde_json::json!("[\"rust\",\"mcp\"]")),
        ] {
            frontmatter(
                &vault,
                FrontmatterParams {
                    action: "set".into(),
                    path: "fm.md".into(),
                    key: Some(key.into()),
                    value: Some(value),
                },
            )
            .await
            .unwrap();
        }

        let result = frontmatter(
            &vault,
            FrontmatterParams {
                action: "get".into(),
                path: "fm.md".into(),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let fm = result.structured_content.unwrap()["frontmatter"].clone();
        assert_eq!(fm["empty"], serde_json::Value::Null);
        assert_eq!(fm["literal_json"], "[\"rust\",\"mcp\"]");

        frontmatter(
            &vault,
            FrontmatterParams {
                action: "remove".into(),
                path: "fm.md".into(),
                key: Some("status".into()),
                value: None,
            },
        )
        .await
        .unwrap();

        let result = frontmatter(
            &vault,
            FrontmatterParams {
                action: "get".into(),
                path: "fm.md".into(),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let fm = result.structured_content.unwrap()["frontmatter"].clone();
        assert!(fm.get("status").is_none());
        assert_eq!(fm["tags"], serde_json::json!(["rust", "mcp"]));
    }

    #[test]
    fn frontmatter_params_preserve_missing_null_and_literal_strings() {
        let missing: FrontmatterParams = serde_json::from_value(serde_json::json!({
            "action": "set",
            "path": "fm.md",
            "key": "value"
        }))
        .unwrap();
        assert!(missing.value.is_none());

        let explicit_null: FrontmatterParams = serde_json::from_value(serde_json::json!({
            "action": "set",
            "path": "fm.md",
            "key": "value",
            "value": null
        }))
        .unwrap();
        assert_eq!(explicit_null.value, Some(serde_json::Value::Null));

        let array: FrontmatterParams = serde_json::from_value(serde_json::json!({
            "action": "set",
            "path": "fm.md",
            "key": "value",
            "value": ["rust", "mcp"]
        }))
        .unwrap();
        assert_eq!(array.value, Some(serde_json::json!(["rust", "mcp"])));

        let literal: FrontmatterParams = serde_json::from_value(serde_json::json!({
            "action": "set",
            "path": "fm.md",
            "key": "value",
            "value": "[\"rust\",\"mcp\"]"
        }))
        .unwrap();
        assert_eq!(literal.value, Some(serde_json::json!("[\"rust\",\"mcp\"]")));
    }

    #[tokio::test]
    async fn frontmatter_invalid_action() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();
        vault.write_note(Path::new("fm.md"), "# Note\n").unwrap();

        let result = frontmatter(
            &vault,
            FrontmatterParams {
                action: "invalid".into(),
                path: "fm.md".into(),
                ..Default::default()
            },
        )
        .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn frontmatter_set_missing_key() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();
        vault.write_note(Path::new("fm.md"), "# Note\n").unwrap();

        let result = frontmatter(
            &vault,
            FrontmatterParams {
                action: "set".into(),
                path: "fm.md".into(),
                key: None,
                value: Some(serde_json::json!("val")),
            },
        )
        .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn frontmatter_set_missing_value() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();
        vault.write_note(Path::new("fm.md"), "# Note\n").unwrap();

        let result = frontmatter(
            &vault,
            FrontmatterParams {
                action: "set".into(),
                path: "fm.md".into(),
                key: Some("k".into()),
                value: None,
            },
        )
        .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn frontmatter_remove_missing_key() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();
        vault.write_note(Path::new("fm.md"), "# Note\n").unwrap();

        let result = frontmatter(
            &vault,
            FrontmatterParams {
                action: "remove".into(),
                path: "fm.md".into(),
                key: None,
                value: None,
            },
        )
        .await;
        assert!(result.is_err());
    }
}
