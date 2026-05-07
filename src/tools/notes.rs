//! Note CRUD tools: read, create, edit, delete, rename, move.

use std::path::Path;

use rmcp::ErrorData;
use rmcp::model::ErrorCode;
use schemars::JsonSchema;
use serde::Deserialize;

use crate::models::{PatchOperation, PatchRequest, PatchTargetType};
use crate::vault::Vault;

// ── Parameter structs ───────────────────────────────────────────────

#[derive(Deserialize, JsonSchema, Default)]
pub struct NoteReadParams {
    /// Path to the note, relative to vault root (e.g. "folder/note.md").
    pub path: String,
}

#[derive(Deserialize, JsonSchema, Default)]
pub struct NoteCreateParams {
    /// Path for the new note, relative to vault root. Parent dirs are created automatically.
    pub path: String,
    /// Initial body content. Defaults to empty.
    #[serde(default)]
    pub content: Option<String>,
    /// Optional YAML frontmatter as a JSON object (e.g. `{"tags": ["rust"], "draft": true}`).
    #[serde(default)]
    pub frontmatter: Option<serde_json::Value>,
}

#[derive(Deserialize, JsonSchema, Default)]
pub struct NoteWriteParams {
    /// Path to the note, relative to vault root.
    pub path: String,
    /// New content that replaces the entire note.
    pub content: String,
}

#[derive(Deserialize, JsonSchema, Default)]
pub struct NoteAppendParams {
    /// Path to the note, relative to vault root.
    pub path: String,
    /// Content to append at the end of the note.
    pub content: String,
}

#[derive(Deserialize, JsonSchema, Default)]
pub struct NotePrependParams {
    /// Path to the note, relative to vault root.
    pub path: String,
    /// Content to insert after frontmatter (or at the very start if no frontmatter).
    pub content: String,
}

#[derive(Deserialize, JsonSchema)]
pub struct NotePatchParams {
    /// Path to the note, relative to vault root.
    pub path: String,
    /// Patch operation: `append`, `prepend`, or `replace`.
    pub operation: PatchOperation,
    /// Target type: `heading`, `block`, or `frontmatter`.
    pub target_type: PatchTargetType,
    /// Target identifier — heading text, block ID, or frontmatter field name.
    pub target: String,
    /// Content to insert or replace with.
    pub content: String,
}

impl Default for NotePatchParams {
    fn default() -> Self {
        Self {
            path: String::new(),
            operation: PatchOperation::Append,
            target_type: PatchTargetType::Heading,
            target: String::new(),
            content: String::new(),
        }
    }
}

#[derive(Deserialize, JsonSchema, Default)]
pub struct NoteDeleteParams {
    /// Path to the note, relative to vault root.
    pub path: String,
    /// Must be `true` to confirm deletion — a safety check to prevent accidental data loss.
    pub confirm: bool,
}

#[derive(Deserialize, JsonSchema, Default)]
pub struct NoteMoveParams {
    /// Current path of the note, relative to vault root.
    pub from: String,
    /// Destination path, relative to vault root.
    pub to: String,
}

// ── Handler functions ───────────────────────────────────────────────

pub async fn note_read(vault: &Vault, params: NoteReadParams) -> Result<String, ErrorData> {
    Ok(vault.read_note(Path::new(&params.path))?)
}

pub async fn note_create(vault: &Vault, params: NoteCreateParams) -> Result<String, ErrorData> {
    vault.create_note(
        Path::new(&params.path),
        params.content.as_deref().unwrap_or(""),
        params.frontmatter.as_ref(),
    )?;
    Ok(format!("Created note: {}", params.path))
}

pub async fn note_write(vault: &Vault, params: NoteWriteParams) -> Result<String, ErrorData> {
    vault.write_note(Path::new(&params.path), &params.content)?;
    Ok(format!("Written to: {}", params.path))
}

pub async fn note_append(vault: &Vault, params: NoteAppendParams) -> Result<String, ErrorData> {
    vault.append_note(Path::new(&params.path), &params.content)?;
    Ok(format!("Appended to: {}", params.path))
}

pub async fn note_prepend(vault: &Vault, params: NotePrependParams) -> Result<String, ErrorData> {
    vault.prepend_note(Path::new(&params.path), &params.content)?;
    Ok(format!("Prepended to: {}", params.path))
}

pub async fn note_patch(vault: &Vault, params: NotePatchParams) -> Result<String, ErrorData> {
    let request = PatchRequest {
        operation: params.operation,
        target_type: params.target_type,
        target: params.target,
        content: params.content,
    };
    vault.patch_note(Path::new(&params.path), &request)?;
    Ok(format!("Patched: {}", params.path))
}

pub async fn note_delete(vault: &Vault, params: NoteDeleteParams) -> Result<String, ErrorData> {
    if !params.confirm {
        return Err(ErrorData::new(
            ErrorCode::INVALID_PARAMS,
            "Deletion requires `confirm: true` as a safety check",
            None::<serde_json::Value>,
        ));
    }
    vault.delete_note(Path::new(&params.path))?;
    Ok(format!("Deleted: {}", params.path))
}

pub async fn note_move(vault: &Vault, params: NoteMoveParams) -> Result<String, ErrorData> {
    let new_path = vault.move_note(Path::new(&params.from), Path::new(&params.to))?;
    Ok(format!("Moved to: {}", new_path.display()))
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::test_helpers::{create_test_vault, test_config};
    use crate::vault::Vault;

    // ── note_read ───────────────────────────────────────────────────

    #[tokio::test]
    async fn read_existing_note() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();
        vault
            .write_note(Path::new("hello.md"), "# Hello\nWorld")
            .unwrap();

        let content = note_read(
            &vault,
            NoteReadParams {
                path: "hello.md".into(),
            },
        )
        .await
        .unwrap();
        assert_eq!(content, "# Hello\nWorld");
    }

    #[tokio::test]
    async fn read_nonexistent_note_errors() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        let result = note_read(
            &vault,
            NoteReadParams {
                path: "missing.md".into(),
            },
        )
        .await;
        assert!(result.is_err());
    }

    // ── note_create ─────────────────────────────────────────────────

    #[tokio::test]
    async fn create_new_note() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        let msg = note_create(
            &vault,
            NoteCreateParams {
                path: "new.md".into(),
                content: Some("body".into()),
                frontmatter: Some(serde_json::json!({"status": "draft"})),
            },
        )
        .await
        .unwrap();
        assert!(msg.contains("new.md"));

        let content = vault.read_note(Path::new("new.md")).unwrap();
        assert!(content.contains("body"));
        assert!(content.contains("status"));
    }

    #[tokio::test]
    async fn create_duplicate_note_errors() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        note_create(
            &vault,
            NoteCreateParams {
                path: "dup.md".into(),
                ..Default::default()
            },
        )
        .await
        .unwrap();

        let result = note_create(
            &vault,
            NoteCreateParams {
                path: "dup.md".into(),
                ..Default::default()
            },
        )
        .await;
        assert!(result.is_err());
    }

    // ── note_write ──────────────────────────────────────────────────

    #[tokio::test]
    async fn write_overwrites_content() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();
        vault
            .write_note(Path::new("note.md"), "old content")
            .unwrap();

        note_write(
            &vault,
            NoteWriteParams {
                path: "note.md".into(),
                content: "new content".into(),
            },
        )
        .await
        .unwrap();

        let content = vault.read_note(Path::new("note.md")).unwrap();
        assert_eq!(content, "new content");
    }

    // ── note_append ─────────────────────────────────────────────────

    #[tokio::test]
    async fn append_adds_to_end() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();
        vault.write_note(Path::new("note.md"), "start").unwrap();

        note_append(
            &vault,
            NoteAppendParams {
                path: "note.md".into(),
                content: "\nmore".into(),
            },
        )
        .await
        .unwrap();

        let content = vault.read_note(Path::new("note.md")).unwrap();
        assert!(content.ends_with("more"));
        assert!(content.starts_with("start"));
    }

    // ── note_prepend ────────────────────────────────────────────────

    #[tokio::test]
    async fn prepend_inserts_after_frontmatter() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();
        vault
            .write_note(Path::new("note.md"), "---\ntags: [a]\n---\n# Heading\n")
            .unwrap();

        note_prepend(
            &vault,
            NotePrependParams {
                path: "note.md".into(),
                content: "injected\n".into(),
            },
        )
        .await
        .unwrap();

        let content = vault.read_note(Path::new("note.md")).unwrap();
        assert!(content.starts_with("---\ntags:"));
        assert!(content.contains("injected"));
        assert!(content.contains("# Heading"));
    }

    // ── note_patch ──────────────────────────────────────────────────

    #[tokio::test]
    async fn patch_heading_append() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();
        vault
            .write_note(Path::new("patched.md"), "# Title\nBody\n## Sub\nSub body\n")
            .unwrap();

        note_patch(
            &vault,
            NotePatchParams {
                path: "patched.md".into(),
                operation: PatchOperation::Append,
                target_type: PatchTargetType::Heading,
                target: "Sub".into(),
                content: "appended\n".into(),
            },
        )
        .await
        .unwrap();

        let content = vault.read_note(Path::new("patched.md")).unwrap();
        assert!(content.contains("Sub body"));
        assert!(content.contains("appended"));
    }

    // ── note_delete ─────────────────────────────────────────────────

    #[tokio::test]
    async fn delete_requires_confirm() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();
        vault.write_note(Path::new("note.md"), "content").unwrap();

        let result = note_delete(
            &vault,
            NoteDeleteParams {
                path: "note.md".into(),
                confirm: false,
            },
        )
        .await;
        assert!(result.is_err());
        assert!(vault.read_note(Path::new("note.md")).is_ok());
    }

    #[tokio::test]
    async fn delete_with_confirm_succeeds() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();
        vault.write_note(Path::new("note.md"), "content").unwrap();

        note_delete(
            &vault,
            NoteDeleteParams {
                path: "note.md".into(),
                confirm: true,
            },
        )
        .await
        .unwrap();
        assert!(vault.read_note(Path::new("note.md")).is_err());
    }

    // ── note_move ───────────────────────────────────────────────────

    #[tokio::test]
    async fn move_renames_note() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();
        vault.write_note(Path::new("old.md"), "content").unwrap();

        let msg = note_move(
            &vault,
            NoteMoveParams {
                from: "old.md".into(),
                to: "new.md".into(),
            },
        )
        .await
        .unwrap();
        assert!(msg.contains("new.md"));

        assert!(vault.read_note(Path::new("old.md")).is_err());
        assert_eq!(vault.read_note(Path::new("new.md")).unwrap(), "content");
    }
}
