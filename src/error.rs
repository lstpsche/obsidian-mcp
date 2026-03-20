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

    #[error("Frontmatter parse error in {path}: {source}")]
    FrontmatterParse {
        path: PathBuf,
        source: serde_yaml::Error,
    },

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

    #[error("Watcher error: {0}")]
    Watcher(String),

    #[error("Tantivy index error: {0}")]
    Tantivy(#[from] tantivy::TantivyError),

    #[error("Embedding error: {0}")]
    Embedding(String),

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
            | VaultError::OutsideVault(_)
            | VaultError::AlreadyExists(_)
            | VaultError::PatchTargetNotFound { .. }
            | VaultError::InvalidRegex { .. } => ErrorCode::INVALID_PARAMS,
            VaultError::FrontmatterParse { .. } => ErrorCode::PARSE_ERROR,
            VaultError::Io(_)
            | VaultError::Watcher(_)
            | VaultError::Tantivy(_)
            | VaultError::Embedding(_)
            | VaultError::Other(_) => ErrorCode::INTERNAL_ERROR,
        };

        rmcp::ErrorData::new(code, err.to_string(), None)
    }
}
