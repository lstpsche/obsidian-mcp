//! Note metadata inspection and frontmatter manipulation tools.

use std::path::{Path, PathBuf};

use rmcp::model::{CallToolResult, Content};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::VaultError;
use crate::models::{FileStat, Heading, WikiLink};
use crate::vault::Vault;

// ── note_metadata ──────────────────────────────────────────────────────

#[derive(Deserialize, JsonSchema, Default)]
pub struct NoteMetadataParams {
    /// Path to the note, relative to vault root.
    pub path: String,
}

/// Enriched note metadata including backlinks count.
#[derive(Serialize, JsonSchema)]
struct NoteMetadataOutput {
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

/// Get rich metadata about a note: tags, headings, links, backlinks count,
/// frontmatter, and file stats.
pub async fn note_metadata(
    vault: &Vault,
    params: NoteMetadataParams,
) -> Result<CallToolResult, rmcp::ErrorData> {
    let path = Path::new(&params.path);
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

    let value = serde_json::to_value(output)
        .map_err(|e| VaultError::Other(format!("serialization error: {e}")))?;
    Ok(CallToolResult::structured(value))
}

// ── note_document_map ──────────────────────────────────────────────────

#[derive(Deserialize, JsonSchema, Default)]
pub struct NoteDocumentMapParams {
    /// Path to the note, relative to vault root.
    pub path: String,
}

/// List all patch targets in a note: headings (with hierarchy),
/// block refs, and frontmatter fields.
pub async fn note_document_map(
    vault: &Vault,
    params: NoteDocumentMapParams,
) -> Result<CallToolResult, rmcp::ErrorData> {
    let path = Path::new(&params.path);
    let map = vault.get_document_map(path)?;

    let value = serde_json::to_value(map)
        .map_err(|e| VaultError::Other(format!("serialization error: {e}")))?;
    Ok(CallToolResult::structured(value))
}

// ── frontmatter_get ────────────────────────────────────────────────────

#[derive(Deserialize, JsonSchema, Default)]
pub struct FrontmatterGetParams {
    /// Path to the note, relative to vault root.
    pub path: String,
}

/// Get a note's frontmatter as a JSON object, or null if absent.
pub async fn frontmatter_get(
    vault: &Vault,
    params: FrontmatterGetParams,
) -> Result<CallToolResult, rmcp::ErrorData> {
    let path = Path::new(&params.path);
    let fm = vault.get_frontmatter(path)?;

    match fm {
        Some(value) => Ok(CallToolResult::structured(value)),
        None => Ok(CallToolResult::success(vec![Content::text("null")])),
    }
}

// ── frontmatter_set ────────────────────────────────────────────────────

#[derive(Deserialize, JsonSchema, Default)]
pub struct FrontmatterSetParams {
    /// Path to the note, relative to vault root.
    pub path: String,
    /// Frontmatter key to set.
    pub key: String,
    /// JSON value to assign to the key.
    pub value: serde_json::Value,
}

/// Set a single frontmatter field on a note (upsert).
pub async fn frontmatter_set(
    vault: &Vault,
    params: FrontmatterSetParams,
) -> Result<CallToolResult, rmcp::ErrorData> {
    let path = Path::new(&params.path);
    vault.set_frontmatter_field(path, &params.key, params.value)?;

    Ok(CallToolResult::success(vec![Content::text(format!(
        "Set frontmatter field '{}' on '{}'",
        params.key, params.path
    ))]))
}

// ── frontmatter_remove ─────────────────────────────────────────────────

#[derive(Deserialize, JsonSchema, Default)]
pub struct FrontmatterRemoveParams {
    /// Path to the note, relative to vault root.
    pub path: String,
    /// Frontmatter key to remove.
    pub key: String,
}

/// Remove a single frontmatter field from a note.
pub async fn frontmatter_remove(
    vault: &Vault,
    params: FrontmatterRemoveParams,
) -> Result<CallToolResult, rmcp::ErrorData> {
    let path = Path::new(&params.path);
    vault.remove_frontmatter_field(path, &params.key)?;

    Ok(CallToolResult::success(vec![Content::text(format!(
        "Removed frontmatter field '{}' from '{}'",
        params.key, params.path
    ))]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    fn test_config(vault_root: &Path) -> Config {
        Config {
            vault_path: vault_root.to_path_buf(),
            watch: false,
            log_level: "error".into(),
            tantivy: false,
            embeddings: false,
            embeddings_model: String::new(),
        }
    }

    fn create_test_vault(dir: &Path) {
        std::fs::create_dir_all(dir.join(".obsidian")).unwrap();
    }

    #[tokio::test]
    async fn note_metadata_returns_all_fields() {
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

        let result = note_metadata(
            &vault,
            NoteMetadataParams {
                path: "test.md".into(),
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
    async fn note_metadata_not_found() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        let result = note_metadata(
            &vault,
            NoteMetadataParams {
                path: "nonexistent.md".into(),
            },
        )
        .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn document_map_lists_targets() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        vault
            .write_note(
                Path::new("mapped.md"),
                "---\ntags: [rust]\ndate: 2026-01-01\n---\n# Heading\n## Sub\nText ^block1\n",
            )
            .unwrap();

        let result = note_document_map(
            &vault,
            NoteDocumentMapParams {
                path: "mapped.md".into(),
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
    async fn frontmatter_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        vault
            .write_note(Path::new("fm.md"), "# Note\nBody\n")
            .unwrap();

        let result = frontmatter_get(
            &vault,
            FrontmatterGetParams {
                path: "fm.md".into(),
            },
        )
        .await
        .unwrap();
        assert!(result.structured_content.is_none());
        let text = result.content[0].as_text().expect("expected text content");
        assert_eq!(text.text, "null");

        frontmatter_set(
            &vault,
            FrontmatterSetParams {
                path: "fm.md".into(),
                key: "status".into(),
                value: serde_json::json!("draft"),
            },
        )
        .await
        .unwrap();

        let result = frontmatter_get(
            &vault,
            FrontmatterGetParams {
                path: "fm.md".into(),
            },
        )
        .await
        .unwrap();
        let fm = result.structured_content.unwrap();
        assert_eq!(fm["status"], "draft");

        frontmatter_set(
            &vault,
            FrontmatterSetParams {
                path: "fm.md".into(),
                key: "tags".into(),
                value: serde_json::json!(["rust", "mcp"]),
            },
        )
        .await
        .unwrap();

        let result = frontmatter_get(
            &vault,
            FrontmatterGetParams {
                path: "fm.md".into(),
            },
        )
        .await
        .unwrap();
        let fm = result.structured_content.unwrap();
        assert_eq!(fm["status"], "draft");
        assert_eq!(fm["tags"], serde_json::json!(["rust", "mcp"]));

        frontmatter_remove(
            &vault,
            FrontmatterRemoveParams {
                path: "fm.md".into(),
                key: "status".into(),
            },
        )
        .await
        .unwrap();

        let result = frontmatter_get(
            &vault,
            FrontmatterGetParams {
                path: "fm.md".into(),
            },
        )
        .await
        .unwrap();
        let fm = result.structured_content.unwrap();
        assert!(fm.get("status").is_none());
        assert_eq!(fm["tags"], serde_json::json!(["rust", "mcp"]));
    }
}
