//! Shared types used by both the vault layer and MCP tool handlers.

use std::path::PathBuf;

use chrono::{DateTime, Utc};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Half-open UTF-8 byte offsets into an exact raw note revision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SourceRange {
    pub start_byte: usize,
    pub end_byte: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceScope {
    Note,
    Passage,
}

/// The exact source region represented by the winning semantic vector.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SemanticEvidence {
    pub scope: EvidenceScope,
    pub cosine_similarity: f32,
    pub range: SourceRange,
    pub heading_path: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub section_range: Option<SourceRange>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct SemanticPreview {
    pub range: SourceRange,
    pub complete_evidence: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SourceStatus {
    Current,
    Changed,
    Unavailable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct SemanticContextError {
    pub code: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current_revision: Option<String>,
}

/// A ranked note and the revision-bound semantic evidence behind its score.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SemanticNoteResult {
    pub path: PathBuf,
    pub title: String,
    pub score: f32,
    pub tags: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snippet: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    pub source_revision: String,
    pub source_bytes: usize,
    pub source_status: SourceStatus,
    pub evidence: SemanticEvidence,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preview: Option<SemanticPreview>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_error: Option<SemanticContextError>,
}

/// Metadata about a single note in the vault.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct NoteMetadata {
    /// Path relative to vault root.
    pub path: PathBuf,
    /// Filename without `.md` extension.
    pub title: String,
    /// Both inline `#tags` and frontmatter tags, deduplicated.
    pub tags: Vec<String>,
    /// Parsed YAML frontmatter as a JSON value.
    pub frontmatter: Option<serde_json::Value>,
    pub headings: Vec<Heading>,
    /// Outgoing wikilinks found in the note.
    pub links: Vec<WikiLink>,
    /// `^blockid` identifiers defined in the note.
    pub block_refs: Vec<String>,
    pub stat: FileStat,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct FileStat {
    pub size: u64,
    pub created: Option<DateTime<Utc>>,
    pub modified: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Heading {
    /// Heading level (1–6).
    pub level: u8,
    /// Heading text without the `#` prefix.
    pub text: String,
    /// 0-based line number in the file.
    pub line: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct WikiLink {
    /// Full raw text, e.g. `[[note#heading|alias]]`.
    pub raw: String,
    /// Link target (note name or path).
    pub target: String,
    /// Heading fragment, if `[[note#heading]]`.
    pub heading: Option<String>,
    /// Block reference, if `[[note#^blockid]]`.
    pub block_ref: Option<String>,
    /// Display alias, if `[[note|alias]]`.
    pub alias: Option<String>,
    /// 0-based line number in the file.
    pub line: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SearchResult {
    pub path: PathBuf,
    pub matches: Vec<SearchMatch>,
    pub score: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SearchMatch {
    /// 0-based line number.
    pub line: usize,
    /// Surrounding text context.
    pub context: String,
    /// Character offset of match start within `context`.
    pub match_start: usize,
    /// Character offset of match end within `context`.
    pub match_end: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct VaultStats {
    pub total_notes: usize,
    /// All files including non-`.md`.
    pub total_files: usize,
    pub total_tags: usize,
    pub total_links: usize,
    pub vault_size_bytes: u64,
    /// Number of `.md` files skipped by exclusion patterns.
    pub excluded_notes: usize,
}

/// Available PATCH targets in a note.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DocumentMap {
    /// e.g. `["# Heading 1", "## Sub"]`
    pub headings: Vec<String>,
    /// e.g. `["^abc123"]`
    pub block_refs: Vec<String>,
    /// e.g. `["tags", "date"]`
    pub frontmatter_fields: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PatchOperation {
    Append,
    Prepend,
    Replace,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PatchTargetType {
    Heading,
    Block,
    Frontmatter,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct PatchRequest {
    pub operation: PatchOperation,
    pub target_type: PatchTargetType,
    pub target: String,
    pub content: String,
}

/// Fields available for targeted full-text search via Tantivy.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SearchField {
    Title,
    Headings,
    Tags,
    Body,
    Frontmatter,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum NotePeriod {
    #[default]
    Daily,
    Weekly,
    Monthly,
    Quarterly,
    Yearly,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct PeriodicNoteConfig {
    /// Moment.js date format string.
    pub format: String,
    /// Folder relative to vault root.
    pub folder: Option<String>,
    /// Template file path.
    pub template: Option<String>,
}
