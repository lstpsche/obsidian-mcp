//! Note CRUD tools: read, create, edit, delete, rename, move.

use std::path::{Path, PathBuf};

use rmcp::ErrorData;
use rmcp::handler::server::wrapper::Json;
use rmcp::model::ErrorCode;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::VaultError;
use crate::models::{PatchOperation, PatchRequest, PatchTargetType};
use crate::vault::Vault;

const DEFAULT_MAX_FILES: usize = 20;
const MAX_FILES_CAP: usize = 100;
const DEFAULT_MAX_BYTES: usize = 64 * 1024;
const MAX_BYTES_CAP: usize = 256 * 1024;
const MAX_SKIPPED_DETAILS: usize = 100;

// ── Parameter structs ───────────────────────────────────────────────

#[derive(Deserialize, JsonSchema, Default)]
pub struct NoteReadParams {
    /// Path to the note, relative to vault root (e.g. "folder/note.md").
    pub path: String,
}

#[derive(Deserialize, JsonSchema, Default)]
pub struct NoteReadManyParams {
    /// Explicit note paths, in the order they should be returned. Mutually exclusive with `dir`.
    #[schemars(length(min = 1, max = 100))]
    pub paths: Option<Vec<String>>,
    /// Directory path relative to vault root. Use an empty string for vault root. Mutually exclusive with `paths`.
    pub dir: Option<String>,
    /// Include note files in nested directories. Defaults to false and is only valid with `dir`.
    pub recursive: Option<bool>,
    /// Glob pattern applied to vault-relative paths. Only valid with `dir`.
    pub glob: Option<String>,
    /// Maximum candidate files inspected. Defaults to 20 and is capped at 100.
    #[schemars(range(min = 1, max = 100))]
    pub max_files: Option<usize>,
    /// Maximum combined UTF-8 content bytes returned. Defaults to 65536 and is capped at 262144.
    #[schemars(range(min = 1, max = 262144))]
    pub max_bytes: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, JsonSchema)]
pub struct NoteReadManyItem {
    pub path: String,
    pub content: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum NoteReadManySkipReason {
    FileLimit,
    ByteLimit,
    NotFound,
    Unreadable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, JsonSchema)]
pub struct NoteReadManySkipped {
    pub path: String,
    pub reason: NoteReadManySkipReason,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, JsonSchema)]
pub struct NoteReadManyOutput {
    pub notes: Vec<NoteReadManyItem>,
    pub skipped: Vec<NoteReadManySkipped>,
    /// Total skipped candidates. May exceed `skipped.len()` when details hit the hard cap.
    pub skipped_count: usize,
    /// Exact sum of UTF-8 bytes in the returned `content` fields.
    pub content_bytes: usize,
}

struct ReadCandidate {
    path: PathBuf,
    display_path: String,
}

#[derive(Deserialize, JsonSchema, Default)]
pub struct NoteCreateParams {
    /// Path for the new note, relative to vault root. Parent dirs are created automatically.
    pub path: String,
    /// Initial body content. Defaults to empty.
    #[serde(default)]
    pub content: Option<String>,
    /// Optional YAML frontmatter as a JSON object (e.g. `{"tags": ["rust"], "draft": true}`).
    #[serde(
        default,
        deserialize_with = "crate::tools::deserialize_optional_json_object"
    )]
    #[schemars(schema_with = "crate::tools::json_object_schema")]
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
pub struct NoteInsertParams {
    /// Path to the note, relative to vault root.
    pub path: String,
    /// Content to insert.
    pub content: String,
    /// Where to insert: `"end"` (default) appends after existing content; `"beginning"` inserts after frontmatter (or at the very start if no frontmatter).
    #[serde(default)]
    pub position: Option<String>,
}

#[derive(Deserialize, JsonSchema)]
pub struct NotePatchParams {
    /// Path to the note, relative to vault root.
    pub path: String,
    /// Patch operation: `append`, `prepend`, or `replace`.
    pub operation: PatchOperation,
    /// Target type: `heading`, `block`, or `frontmatter`.
    pub target_type: PatchTargetType,
    /// Target identifier — heading text, block ID, or frontmatter field name. For headings, bare text such as `"Log"` is canonical; ATX marker-prefixed targets such as `"## Log"` are also accepted.
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

pub async fn note_read_many(
    vault: &Vault,
    params: NoteReadManyParams,
) -> Result<Json<NoteReadManyOutput>, ErrorData> {
    let max_files = validated_limit(
        params.max_files,
        DEFAULT_MAX_FILES,
        MAX_FILES_CAP,
        "max_files",
    )?;
    let max_bytes = validated_limit(
        params.max_bytes,
        DEFAULT_MAX_BYTES,
        MAX_BYTES_CAP,
        "max_bytes",
    )?;

    if params.paths.is_some() && (params.recursive.is_some() || params.glob.is_some()) {
        return Err(invalid_params(
            "`recursive` and `glob` are only valid with the `dir` selector",
        ));
    }

    let candidates: Vec<ReadCandidate> = match (params.paths, params.dir) {
        (Some(_), Some(_)) | (None, None) => {
            return Err(invalid_params(
                "Exactly one of `paths` or `dir` must be provided",
            ));
        }
        (Some(paths), None) => {
            if paths.is_empty() {
                return Err(invalid_params("`paths` must contain at least one path"));
            }
            if paths.len() > MAX_FILES_CAP {
                return Err(invalid_params(format!(
                    "`paths` accepts at most {MAX_FILES_CAP} entries"
                )));
            }

            for path in &paths {
                vault.validate_path(Path::new(path))?;
            }

            paths
                .into_iter()
                .map(|path| ReadCandidate {
                    path: PathBuf::from(&path),
                    display_path: path,
                })
                .collect()
        }
        (None, Some(dir)) => vault
            .list_files(
                Path::new(&dir),
                params.recursive.unwrap_or(false),
                params.glob.as_deref(),
            )?
            .into_iter()
            .filter(|path| {
                path.extension()
                    .and_then(|extension| extension.to_str())
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("md"))
                    && vault.root().join(path).is_file()
            })
            .filter_map(|path| {
                let display_path = path.to_str()?.to_owned();
                Some(ReadCandidate { path, display_path })
            })
            .collect(),
    };

    let inspected_count = candidates.len().min(max_files);
    let mut notes = Vec::with_capacity(inspected_count);
    let mut skipped = Vec::new();
    let mut skipped_count = 0;
    let mut content_bytes = 0;

    for candidate in candidates.iter().take(inspected_count) {
        let actual_path = match vault.canonical_existing_relative_path(&candidate.path) {
            Ok(path) => path,
            Err(error) => {
                record_file_error(
                    error,
                    &candidate.display_path,
                    &mut skipped,
                    &mut skipped_count,
                )?;
                continue;
            }
        };
        let display_path = actual_path
            .to_str()
            .map(str::to_owned)
            .unwrap_or_else(|| candidate.display_path.clone());
        let remaining_bytes = max_bytes - content_bytes;

        let stat = match vault.file_stat(&actual_path) {
            Ok(stat) => stat,
            Err(error) => {
                record_file_error(error, &display_path, &mut skipped, &mut skipped_count)?;
                continue;
            }
        };
        if stat.size > remaining_bytes as u64 {
            record_skipped(
                &mut skipped,
                &mut skipped_count,
                display_path,
                NoteReadManySkipReason::ByteLimit,
            );
            continue;
        }

        let content = match vault.read_note(&actual_path) {
            Ok(content) => content,
            Err(error) => {
                record_file_error(error, &display_path, &mut skipped, &mut skipped_count)?;
                continue;
            }
        };
        let note_bytes = content.len();
        if note_bytes > remaining_bytes {
            record_skipped(
                &mut skipped,
                &mut skipped_count,
                display_path,
                NoteReadManySkipReason::ByteLimit,
            );
            continue;
        }

        content_bytes += note_bytes;
        notes.push(NoteReadManyItem {
            path: display_path,
            content,
        });
    }

    let file_limit_count = candidates.len() - inspected_count;
    skipped_count += file_limit_count;
    let detail_slots = MAX_SKIPPED_DETAILS.saturating_sub(skipped.len());
    skipped.extend(
        candidates
            .iter()
            .skip(inspected_count)
            .take(detail_slots)
            .map(|candidate| NoteReadManySkipped {
                path: candidate.display_path.clone(),
                reason: NoteReadManySkipReason::FileLimit,
            }),
    );

    Ok(Json(NoteReadManyOutput {
        notes,
        skipped,
        skipped_count,
        content_bytes,
    }))
}

fn validated_limit(
    requested: Option<usize>,
    default: usize,
    hard_cap: usize,
    name: &str,
) -> Result<usize, ErrorData> {
    let value = requested.unwrap_or(default);
    if value == 0 {
        return Err(invalid_params(format!(
            "`{name}` must be greater than zero"
        )));
    }
    Ok(value.min(hard_cap))
}

fn invalid_params(message: impl Into<String>) -> ErrorData {
    ErrorData::new(
        ErrorCode::INVALID_PARAMS,
        message.into(),
        None::<serde_json::Value>,
    )
}

fn record_file_error(
    error: VaultError,
    path: &str,
    skipped: &mut Vec<NoteReadManySkipped>,
    skipped_count: &mut usize,
) -> Result<(), ErrorData> {
    let reason = match error {
        VaultError::NoteNotFound(_) => NoteReadManySkipReason::NotFound,
        VaultError::Io(_) => NoteReadManySkipReason::Unreadable,
        other => return Err(other.into()),
    };
    record_skipped(skipped, skipped_count, path.to_owned(), reason);
    Ok(())
}

fn record_skipped(
    skipped: &mut Vec<NoteReadManySkipped>,
    skipped_count: &mut usize,
    path: String,
    reason: NoteReadManySkipReason,
) {
    *skipped_count += 1;
    if skipped.len() < MAX_SKIPPED_DETAILS {
        skipped.push(NoteReadManySkipped { path, reason });
    }
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

pub async fn note_insert(vault: &Vault, params: NoteInsertParams) -> Result<String, ErrorData> {
    let position = params.position.as_deref().unwrap_or("end");

    if position.eq_ignore_ascii_case("end") {
        vault.append_note(Path::new(&params.path), &params.content)?;
        Ok(format!("Inserted into: {}", params.path))
    } else if position.eq_ignore_ascii_case("beginning") {
        vault.prepend_note(Path::new(&params.path), &params.content)?;
        Ok(format!("Inserted into: {}", params.path))
    } else {
        Err(ErrorData::new(
            ErrorCode::INVALID_PARAMS,
            format!("Unknown position '{position}'. Valid values: \"end\", \"beginning\""),
            None::<serde_json::Value>,
        ))
    }
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
    use std::path::{Path, PathBuf};

    use super::*;
    use crate::test_helpers::{create_test_vault, test_config};
    use crate::vault::Vault;
    use unicode_normalization::UnicodeNormalization;

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

    // ── note_read_many ──────────────────────────────────────────────

    #[tokio::test]
    async fn read_many_validates_selectors_limits_and_paths() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        for params in [
            NoteReadManyParams::default(),
            NoteReadManyParams {
                paths: Some(vec!["a.md".into()]),
                dir: Some(String::new()),
                ..Default::default()
            },
            NoteReadManyParams {
                paths: Some(Vec::new()),
                ..Default::default()
            },
            NoteReadManyParams {
                paths: Some(vec!["a.md".into()]),
                recursive: Some(false),
                ..Default::default()
            },
            NoteReadManyParams {
                dir: Some(String::new()),
                max_files: Some(0),
                ..Default::default()
            },
            NoteReadManyParams {
                dir: Some(String::new()),
                max_bytes: Some(0),
                ..Default::default()
            },
            NoteReadManyParams {
                paths: Some(vec!["../outside.md".into()]),
                ..Default::default()
            },
        ] {
            let error = note_read_many(&vault, params)
                .await
                .err()
                .expect("expected validation error");
            assert_eq!(error.code, ErrorCode::INVALID_PARAMS);
        }

        let too_many_paths = (0..=MAX_FILES_CAP)
            .map(|index| format!("{index}.md"))
            .collect();
        let error = note_read_many(
            &vault,
            NoteReadManyParams {
                paths: Some(too_many_paths),
                ..Default::default()
            },
        )
        .await
        .err()
        .expect("expected validation error");
        assert_eq!(error.code, ErrorCode::INVALID_PARAMS);

        let absolute = if cfg!(windows) {
            r"C:\outside.md"
        } else {
            "/outside.md"
        };
        let error = note_read_many(
            &vault,
            NoteReadManyParams {
                paths: Some(vec![absolute.into()]),
                ..Default::default()
            },
        )
        .await
        .err()
        .expect("expected validation error");
        assert_eq!(error.code, ErrorCode::INVALID_PARAMS);
    }

    #[tokio::test]
    async fn read_many_explicit_paths_preserve_order_and_partial_success() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        std::fs::write(dir.path().join("a.md"), "A").unwrap();
        std::fs::write(dir.path().join("b.md"), "BB").unwrap();
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        let output = note_read_many(
            &vault,
            NoteReadManyParams {
                paths: Some(vec!["b.md".into(), "missing.md".into(), "a.md".into()]),
                ..Default::default()
            },
        )
        .await
        .unwrap()
        .0;

        assert_eq!(
            output
                .notes
                .iter()
                .map(|note| note.path.as_str())
                .collect::<Vec<_>>(),
            vec!["b.md", "a.md"]
        );
        assert_eq!(output.content_bytes, 3);
        assert_eq!(output.skipped_count, 1);
        assert_eq!(
            output.skipped,
            vec![NoteReadManySkipped {
                path: "missing.md".into(),
                reason: NoteReadManySkipReason::NotFound,
            }]
        );
    }

    #[tokio::test]
    async fn read_many_directory_defaults_non_recursive_and_filters_non_notes() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        std::fs::write(dir.path().join("a.md"), "A").unwrap();
        std::fs::write(dir.path().join("b.MD"), "B").unwrap();
        std::fs::write(dir.path().join("ignore.txt"), "text").unwrap();
        std::fs::create_dir_all(dir.path().join("nested/fake.md")).unwrap();
        std::fs::write(dir.path().join("nested/c.md"), "C").unwrap();
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        let root = note_read_many(
            &vault,
            NoteReadManyParams {
                dir: Some(String::new()),
                ..Default::default()
            },
        )
        .await
        .unwrap()
        .0;
        assert_eq!(
            root.notes
                .iter()
                .map(|note| note.path.as_str())
                .collect::<Vec<_>>(),
            vec!["a.md", "b.MD"]
        );

        let recursive = note_read_many(
            &vault,
            NoteReadManyParams {
                dir: Some(String::new()),
                recursive: Some(true),
                glob: Some("nested/**".into()),
                ..Default::default()
            },
        )
        .await
        .unwrap()
        .0;
        assert_eq!(recursive.notes.len(), 1);
        assert_eq!(recursive.notes[0].path, "nested/c.md");
    }

    #[tokio::test]
    async fn read_many_never_reads_or_returns_an_oversized_first_file() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        std::fs::write(dir.path().join("large.md"), vec![0xff; 32]).unwrap();
        std::fs::write(dir.path().join("small.md"), "ok").unwrap();
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        let output = note_read_many(
            &vault,
            NoteReadManyParams {
                paths: Some(vec!["large.md".into(), "small.md".into()]),
                max_bytes: Some(2),
                ..Default::default()
            },
        )
        .await
        .unwrap()
        .0;

        assert_eq!(output.notes.len(), 1);
        assert_eq!(output.notes[0].path, "small.md");
        assert_eq!(output.notes[0].content, "ok");
        assert_eq!(output.content_bytes, 2);
        assert_eq!(output.skipped_count, 1);
        assert_eq!(output.skipped[0].reason, NoteReadManySkipReason::ByteLimit);
    }

    #[tokio::test]
    async fn read_many_classifies_invalid_utf8_within_budget_as_unreadable() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        std::fs::write(dir.path().join("invalid.md"), [0xff, 0xfe]).unwrap();
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        let output = note_read_many(
            &vault,
            NoteReadManyParams {
                paths: Some(vec!["invalid.md".into()]),
                max_bytes: Some(2),
                ..Default::default()
            },
        )
        .await
        .unwrap()
        .0;

        assert!(output.notes.is_empty());
        assert_eq!(output.skipped_count, 1);
        assert_eq!(output.skipped[0].reason, NoteReadManySkipReason::Unreadable);
    }

    #[tokio::test]
    async fn read_many_clamps_file_limit_and_bounds_skipped_details() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        for index in 0..205 {
            std::fs::write(dir.path().join(format!("{index:03}.md")), "").unwrap();
        }
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        let output = note_read_many(
            &vault,
            NoteReadManyParams {
                dir: Some(String::new()),
                max_files: Some(usize::MAX),
                max_bytes: Some(usize::MAX),
                ..Default::default()
            },
        )
        .await
        .unwrap()
        .0;

        assert_eq!(output.notes.len(), MAX_FILES_CAP);
        assert_eq!(output.skipped_count, 105);
        assert_eq!(output.skipped.len(), MAX_SKIPPED_DETAILS);
        assert!(
            output
                .skipped
                .iter()
                .all(|entry| entry.reason == NoteReadManySkipReason::FileLimit)
        );
    }

    #[tokio::test]
    async fn read_many_returns_actual_unicode_spelling() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let composed = "Knowledge/é.md";
        let decomposed: String = composed.nfd().collect();
        let disk_path = PathBuf::from(&decomposed);
        std::fs::create_dir_all(dir.path().join(disk_path.parent().unwrap())).unwrap();
        std::fs::write(dir.path().join(&disk_path), "unicode").unwrap();
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        let output = note_read_many(
            &vault,
            NoteReadManyParams {
                paths: Some(vec![composed.into()]),
                ..Default::default()
            },
        )
        .await
        .unwrap()
        .0;

        assert_eq!(output.notes.len(), 1);
        assert_eq!(output.notes[0].path, decomposed);
    }

    #[tokio::test]
    async fn read_many_preserves_direct_access_to_excluded_notes() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        std::fs::create_dir_all(dir.path().join("Archive")).unwrap();
        std::fs::write(dir.path().join("Archive/secret.md"), "secret").unwrap();
        let config = crate::config::Config {
            exclude_patterns: vec!["Archive/**".into()],
            ..test_config(dir.path())
        };
        let vault = Vault::open(&config).await.unwrap();

        let output = note_read_many(
            &vault,
            NoteReadManyParams {
                paths: Some(vec!["Archive/secret.md".into()]),
                ..Default::default()
            },
        )
        .await
        .unwrap()
        .0;

        assert_eq!(output.notes.len(), 1);
        assert_eq!(output.notes[0].content, "secret");
        assert_eq!(output.skipped_count, 0);
    }

    // ── note_create ─────────────────────────────────────────────────

    #[tokio::test]
    async fn create_new_note() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        let params: NoteCreateParams = serde_json::from_value(serde_json::json!({
            "path": "new.md",
            "content": "body",
            "frontmatter": {
                "status": "draft",
                "tags": ["rust", "mcp"],
                "published": false
            }
        }))
        .unwrap();
        let msg = note_create(&vault, params).await.unwrap();
        assert!(msg.contains("new.md"));

        let content = vault.read_note(Path::new("new.md")).unwrap();
        let frontmatter = crate::vault::frontmatter::parse_frontmatter(&content)
            .unwrap()
            .unwrap();
        assert_eq!(
            frontmatter,
            serde_json::json!({
                "status": "draft",
                "tags": ["rust", "mcp"],
                "published": false
            })
        );
        assert_eq!(crate::vault::frontmatter::get_body(&content), "body");
    }

    #[test]
    fn create_params_reject_non_object_frontmatter() {
        for frontmatter in [
            serde_json::json!(["rust", "mcp"]),
            serde_json::json!("{\"tags\":[\"rust\",\"mcp\"]}"),
            serde_json::json!("[\"rust\",\"mcp\"]"),
        ] {
            let result = serde_json::from_value::<NoteCreateParams>(serde_json::json!({
                "path": "new.md",
                "frontmatter": frontmatter
            }));
            assert!(result.is_err());
        }

        let params = serde_json::from_value::<NoteCreateParams>(serde_json::json!({
            "path": "new.md",
            "frontmatter": null
        }))
        .unwrap();
        assert!(params.frontmatter.is_none());
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

    // ── note_insert ────────────────────────────────────────────────

    #[tokio::test]
    async fn insert_end_appends_to_note() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();
        vault.write_note(Path::new("note.md"), "start").unwrap();

        note_insert(
            &vault,
            NoteInsertParams {
                path: "note.md".into(),
                content: "\nmore".into(),
                position: Some("end".into()),
            },
        )
        .await
        .unwrap();

        let content = vault.read_note(Path::new("note.md")).unwrap();
        assert!(content.ends_with("more"));
        assert!(content.starts_with("start"));
    }

    #[tokio::test]
    async fn insert_default_position_appends() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();
        vault.write_note(Path::new("note.md"), "start").unwrap();

        note_insert(
            &vault,
            NoteInsertParams {
                path: "note.md".into(),
                content: "\nmore".into(),
                position: None,
            },
        )
        .await
        .unwrap();

        let content = vault.read_note(Path::new("note.md")).unwrap();
        assert!(content.ends_with("more"));
        assert!(content.starts_with("start"));
    }

    #[tokio::test]
    async fn insert_beginning_after_frontmatter() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();
        vault
            .write_note(Path::new("note.md"), "---\ntags: [a]\n---\n# Heading\n")
            .unwrap();

        note_insert(
            &vault,
            NoteInsertParams {
                path: "note.md".into(),
                content: "injected\n".into(),
                position: Some("beginning".into()),
            },
        )
        .await
        .unwrap();

        let content = vault.read_note(Path::new("note.md")).unwrap();
        assert!(content.starts_with("---\ntags:"));
        assert!(content.contains("injected"));
        assert!(content.contains("# Heading"));
    }

    #[tokio::test]
    async fn insert_invalid_position_errors() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();
        vault.write_note(Path::new("note.md"), "content").unwrap();

        let result = note_insert(
            &vault,
            NoteInsertParams {
                path: "note.md".into(),
                content: "text".into(),
                position: Some("middle".into()),
            },
        )
        .await;
        let err = result.unwrap_err();
        assert!(err.message.contains("Unknown position"));
        assert!(err.message.contains("middle"));
    }

    #[tokio::test]
    async fn insert_position_is_case_insensitive() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();
        vault.write_note(Path::new("note.md"), "start").unwrap();

        note_insert(
            &vault,
            NoteInsertParams {
                path: "note.md".into(),
                content: "\nmore".into(),
                position: Some("END".into()),
            },
        )
        .await
        .unwrap();

        let content = vault.read_note(Path::new("note.md")).unwrap();
        assert!(content.ends_with("more"));
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
