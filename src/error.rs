//! Unified `VaultError` type and conversion to `rmcp::ErrorData`.

use std::path::PathBuf;

use rmcp::model::ErrorCode;

#[derive(Debug, thiserror::Error)]
pub enum VaultError {
    #[error("Note not found: {0}")]
    NoteNotFound(PathBuf),

    #[error("Directory not found: {0}")]
    DirectoryNotFound(PathBuf),

    #[error("Invalid path: {0}")]
    InvalidPath(String),

    #[error("Invalid source range: {0}")]
    InvalidSourceRange(String),

    #[error("Note changed since the requested revision: {path}")]
    SourceChanged {
        path: PathBuf,
        expected_revision: String,
        current_revision: String,
    },

    #[error("Semantic resource limit exceeded for {resource}: {actual} (maximum {limit})")]
    SemanticResourceLimit {
        resource: String,
        actual: usize,
        limit: usize,
    },

    #[error("Embedding input has {tokens} tokens (maximum {max_tokens})")]
    EmbeddingInputTooLong { tokens: usize, max_tokens: usize },

    #[error("Embedding configuration error: {0}")]
    EmbeddingConfiguration(String),

    #[error("Embedding cache requires rebuilding: {0}")]
    EmbeddingCacheIncompatible(String),

    #[error("Embedding request failed: {message}")]
    EmbeddingRequest {
        message: String,
        retryable: bool,
        retry_after: Option<std::time::Duration>,
    },

    #[error("Frontmatter parse error in {path}: {source}")]
    FrontmatterParse {
        path: PathBuf,
        source: serde_yaml::Error,
    },

    #[error("Invalid frontmatter for {path}: expected a JSON object, got {actual}")]
    InvalidFrontmatter { path: PathBuf, actual: &'static str },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("File already exists: {0}")]
    AlreadyExists(PathBuf),

    #[error("Patch target not found: {target_type} '{target}' in {path}")]
    PatchTargetNotFound {
        path: PathBuf,
        target_type: String,
        target: String,
    },

    #[error("Invalid vault path: {0} is outside vault root")]
    OutsideVault(PathBuf),

    #[error("Invalid regex pattern '{pattern}': {source}")]
    InvalidRegex {
        pattern: String,
        source: regex::Error,
    },

    #[error("Invalid date format '{format}' for a date without a time: {source}")]
    InvalidDateFormat {
        format: String,
        source: std::fmt::Error,
    },

    #[error("Watcher error: {0}")]
    Watcher(String),

    #[error("Tantivy index error: {0}")]
    Tantivy(#[from] tantivy::TantivyError),

    #[error("Embedding error: {0}")]
    Embedding(String),

    #[error("Daemon IPC error: {0}")]
    DaemonIpc(String),

    #[error("Daemon protocol error: {0}")]
    DaemonProtocol(String),

    #[error("Daemon unavailable: {0}")]
    DaemonUnavailable(String),

    #[error("Daemon RPC error ({code}): {message}")]
    DaemonRpc {
        code: i64,
        message: String,
        data: Option<serde_json::Value>,
    },

    #[error("Daemon request timed out after {timeout_ms}ms: {operation}")]
    DaemonTimeout { operation: String, timeout_ms: u64 },

    #[error("Daemon bootstrap error: {0}")]
    DaemonBootstrap(String),

    #[error("{0}")]
    Other(String),
}

pub type VaultResult<T> = Result<T, VaultError>;

impl From<VaultError> for rmcp::ErrorData {
    fn from(err: VaultError) -> Self {
        let code = match &err {
            VaultError::NoteNotFound(_) | VaultError::DirectoryNotFound(_) => {
                ErrorCode::RESOURCE_NOT_FOUND
            }
            VaultError::InvalidPath(_)
            | VaultError::InvalidSourceRange(_)
            | VaultError::EmbeddingInputTooLong { .. }
            | VaultError::OutsideVault(_)
            | VaultError::AlreadyExists(_)
            | VaultError::InvalidFrontmatter { .. }
            | VaultError::PatchTargetNotFound { .. }
            | VaultError::InvalidDateFormat { .. }
            | VaultError::InvalidRegex { .. } => ErrorCode::INVALID_PARAMS,
            VaultError::FrontmatterParse { .. } => ErrorCode::PARSE_ERROR,
            VaultError::Io(_)
            | VaultError::SemanticResourceLimit { .. }
            | VaultError::EmbeddingConfiguration(_)
            | VaultError::EmbeddingCacheIncompatible(_)
            | VaultError::EmbeddingRequest { .. }
            | VaultError::Watcher(_)
            | VaultError::Tantivy(_)
            | VaultError::Embedding(_)
            | VaultError::DaemonIpc(_)
            | VaultError::DaemonProtocol(_)
            | VaultError::DaemonTimeout { .. }
            | VaultError::DaemonBootstrap(_)
            | VaultError::Other(_) => ErrorCode::INTERNAL_ERROR,
            VaultError::DaemonUnavailable(_) | VaultError::SourceChanged { .. } => {
                ErrorCode::INVALID_REQUEST
            }
            VaultError::DaemonRpc { code, .. } => match *code {
                -32700 => ErrorCode::PARSE_ERROR,
                -32602 => ErrorCode::INVALID_PARAMS,
                -32600 | -32601 | -32010 | -32011 | -32020 | -32030 | -32031 | -32040 => {
                    ErrorCode::INVALID_REQUEST
                }
                _ => ErrorCode::INTERNAL_ERROR,
            },
        };
        let data = match &err {
            VaultError::DaemonRpc { data, .. } => data.clone(),
            VaultError::SourceChanged {
                path,
                expected_revision,
                current_revision,
            } => Some(serde_json::json!({
                "code": "source_changed",
                "path": path,
                "expected_revision": expected_revision,
                "current_revision": current_revision,
            })),
            VaultError::EmbeddingInputTooLong { tokens, max_tokens } => {
                Some(serde_json::json!({"tokens": tokens, "max_tokens": max_tokens}))
            }
            _ => None,
        };
        rmcp::ErrorData::new(code, err.to_string(), data)
    }
}
