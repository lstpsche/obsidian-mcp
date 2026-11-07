//! Source I/O, separate from machine-owned index and embedding data.

pub mod sftp;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::fs::{MoveResult, VaultRoot};
use super::path::ResolvedPath;
use crate::error::{VaultError, VaultResult};
use crate::models::FileStat;

tokio::task_local! {
    static SOURCE_DEADLINE: tokio::time::Instant;
}

/// Keep successive source operations within the mutation owner's admitted budget.
pub(crate) async fn with_source_deadline<T>(
    deadline: Option<tokio::time::Instant>,
    operation: impl std::future::Future<Output = T>,
) -> T {
    match deadline {
        Some(deadline) => SOURCE_DEADLINE.scope(deadline, operation).await,
        None => operation.await,
    }
}

pub(crate) fn source_deadline(timeout: std::time::Duration) -> tokio::time::Instant {
    let deadline = tokio::time::Instant::now() + timeout;
    current_source_deadline()
        .map(|owner| deadline.min(owner))
        .unwrap_or(deadline)
}

pub(crate) fn current_source_deadline() -> Option<tokio::time::Instant> {
    SOURCE_DEADLINE.try_with(|deadline| *deadline).ok()
}

#[derive(Clone)]
pub enum Storage {
    Local(VaultRoot),
    Sftp(Arc<sftp::SftpStorage>),
}

/// One operation's raw-path resolution and bounded source reads.
pub(crate) enum ReadBatch<'a> {
    Local(&'a VaultRoot),
    Sftp(sftp::ReadBatch<'a>),
}

pub(crate) struct ResolvedRead {
    pub relative: PathBuf,
    absolute: PathBuf,
    deadline: Option<tokio::time::Instant>,
}

impl ReadBatch<'_> {
    pub async fn validate(&mut self, path: &Path) -> VaultResult<()> {
        match self {
            Self::Local(root) => {
                let root = (*root).clone();
                let path = path.to_owned();
                native(move || root.resolve_for_write(&path).map(|_| ())).await
            }
            Self::Sftp(batch) => batch.validate(path).await,
        }
    }

    pub async fn prepare(&mut self, path: &Path) -> VaultResult<ResolvedRead> {
        match self {
            Self::Local(root) => {
                let root = (*root).clone();
                let path = path.to_owned();
                native(move || {
                    let target = root.resolve_existing(&path)?;
                    Ok(ResolvedRead {
                        relative: target.relative,
                        absolute: target.absolute,
                        deadline: None,
                    })
                })
                .await
            }
            Self::Sftp(batch) => batch.prepare(path).await,
        }
    }

    pub async fn stat(&self, target: &ResolvedRead) -> VaultResult<FileStat> {
        match self {
            Self::Local(root) => {
                let root = (*root).clone();
                let path = target.relative.clone();
                native(move || root.file_stat(&path)).await
            }
            Self::Sftp(batch) => batch.stat(target).await,
        }
    }

    pub async fn read(&self, target: &ResolvedRead, limit: usize) -> VaultResult<String> {
        match self {
            Self::Local(root) => {
                let root = (*root).clone();
                let path = target.relative.clone();
                native(move || root.read_file_bounded(&path, limit)).await
            }
            Self::Sftp(batch) => batch.read(target, limit).await,
        }
    }
}

/// Exact-name directory observations retained only for one bounded source batch.
pub enum ObservedBatch<'a> {
    Local(&'a VaultRoot),
    Sftp(sftp::ObservedBatch<'a>),
}

impl ObservedBatch<'_> {
    pub async fn read_with_stat_bounded(
        &self,
        path: &Path,
        limit: usize,
    ) -> VaultResult<(String, FileStat)> {
        match self {
            Self::Local(root) => {
                let root = (*root).clone();
                let path = path.to_owned();
                native(move || root.read_exact_bounded_with_stat(&path, limit)).await
            }
            Self::Sftp(batch) => batch.read(path, limit).await,
        }
    }

    pub async fn read_observed(&self, path: &Path) -> VaultResult<String> {
        match self {
            Self::Local(root) => {
                let root = (*root).clone();
                let path = path.to_owned();
                native(move || root.read_observed(&path)).await
            }
            Self::Sftp(batch) => batch
                .read(path, sftp::MAX_READ_BYTES)
                .await
                .map(|(raw, _)| raw),
        }
    }

    pub async fn read_exact_bounded(&self, path: &Path, limit: usize) -> VaultResult<String> {
        match self {
            Self::Local(root) => {
                let root = (*root).clone();
                let path = path.to_owned();
                native(move || root.read_exact_bounded(&path, limit)).await
            }
            Self::Sftp(batch) => batch.read(path, limit).await.map(|(raw, _)| raw),
        }
    }
}

/// Passive observation of the remote source. Unknown timestamps remain absent.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct SourceStatus {
    pub kind: String,
    pub location: String,
    pub availability: String,
    pub catalog_phase: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_scan: Option<chrono::DateTime<chrono::Utc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_verification: Option<chrono::DateTime<chrono::Utc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl Storage {
    pub(crate) async fn read_batch(&self) -> VaultResult<ReadBatch<'_>> {
        match self {
            Self::Local(root) => Ok(ReadBatch::Local(root)),
            Self::Sftp(root) => sftp::ReadBatch::new(root).await.map(ReadBatch::Sftp),
        }
    }
    pub async fn observe_batch<'a>(&'a self, paths: &[PathBuf]) -> VaultResult<ObservedBatch<'a>> {
        match self {
            Self::Local(root) => Ok(ObservedBatch::Local(root)),
            Self::Sftp(root) => root.observe_batch(paths).await.map(ObservedBatch::Sftp),
        }
    }
    pub fn is_remote(&self) -> bool {
        matches!(self, Self::Sftp(_))
    }
    pub fn path(&self) -> &Path {
        match self {
            Self::Local(root) => root.path(),
            Self::Sftp(root) => root.path(),
        }
    }
    pub fn local_root(&self) -> Option<&VaultRoot> {
        match self {
            Self::Local(root) => Some(root),
            Self::Sftp(_) => None,
        }
    }
    pub async fn read_file(&self, path: &Path) -> VaultResult<String> {
        match self {
            Self::Local(root) => {
                let root = root.clone();
                let path = path.to_owned();
                native(move || root.read_file(&path)).await
            }
            Self::Sftp(root) => root.read_file(path).await,
        }
    }
    pub async fn read_file_bounded(&self, path: &Path, limit: usize) -> VaultResult<String> {
        match self {
            Self::Local(root) => {
                let root = root.clone();
                let path = path.to_owned();
                native(move || root.read_file_bounded(&path, limit)).await
            }
            Self::Sftp(root) => root.read_file_bounded(path, limit).await,
        }
    }
    pub async fn read_exact_bounded(&self, path: &Path, limit: usize) -> VaultResult<String> {
        match self {
            Self::Local(root) => {
                let root = root.clone();
                let path = path.to_owned();
                native(move || root.read_exact_bounded(&path, limit)).await
            }
            Self::Sftp(root) => root.read_exact_bounded(path, limit).await,
        }
    }
    pub async fn read_observed_with_stat(&self, path: &Path) -> VaultResult<(String, FileStat)> {
        match self {
            Self::Local(root) => {
                let root = root.clone();
                let path = path.to_owned();
                native(move || root.read_observed_with_stat(&path)).await
            }
            Self::Sftp(root) => root.read_observed_with_stat(path).await,
        }
    }
    pub async fn file_stat(&self, path: &Path) -> VaultResult<FileStat> {
        match self {
            Self::Local(root) => {
                let root = root.clone();
                let path = path.to_owned();
                native(move || root.file_stat(&path)).await
            }
            Self::Sftp(root) => root.file_stat(path).await,
        }
    }
    pub async fn is_file(&self, path: &Path) -> VaultResult<bool> {
        match self {
            Self::Local(root) => {
                let root = root.clone();
                let path = path.to_owned();
                native(move || match root.resolve_existing(&path) {
                    Ok(resolved) => Ok(root.entry_metadata_exact(&resolved.relative)?.is_file()),
                    Err(e) if e.is_not_found() => Ok(false),
                    Err(e) => Err(e),
                })
                .await
            }
            Self::Sftp(root) => root.is_file(path).await,
        }
    }
    pub async fn list_files(
        &self,
        path: &Path,
        recursive: bool,
        glob: Option<&str>,
    ) -> VaultResult<Vec<PathBuf>> {
        match self {
            Self::Local(root) => {
                let root = root.clone();
                let path = path.to_owned();
                let glob = glob.map(str::to_owned);
                native(move || root.list_files(&path, recursive, glob.as_deref())).await
            }
            Self::Sftp(root) => root.list_files(path, recursive, glob).await,
        }
    }
    pub async fn list_regular_files(
        &self,
        path: &Path,
        recursive: bool,
        glob: Option<&str>,
    ) -> VaultResult<Vec<PathBuf>> {
        match self {
            Self::Local(root) => {
                let root = root.clone();
                let path = path.to_owned();
                let glob = glob.map(str::to_owned);
                native(move || root.list_regular_files(&path, recursive, glob.as_deref())).await
            }
            Self::Sftp(root) => root.list_regular_files(path, recursive, glob).await,
        }
    }
    pub async fn resolve_existing(&self, path: &Path) -> VaultResult<ResolvedPath> {
        match self {
            Self::Local(root) => {
                let root = root.clone();
                let path = path.to_owned();
                native(move || root.resolve_existing(&path)).await
            }
            Self::Sftp(root) => root.resolve_existing(path).await,
        }
    }
    pub async fn resolve_for_write(&self, path: &Path) -> VaultResult<ResolvedPath> {
        match self {
            Self::Local(root) => {
                let root = root.clone();
                let path = path.to_owned();
                native(move || root.resolve_for_write(&path)).await
            }
            Self::Sftp(root) => root.resolve_for_write(path).await,
        }
    }
    pub async fn write_file(&self, path: &Path, content: &str) -> VaultResult<PathBuf> {
        match self {
            Self::Local(root) => {
                let root = root.clone();
                let path = path.to_owned();
                let content = content.to_owned();
                native(move || root.write_file(&path, &content)).await
            }
            Self::Sftp(root) => root.write_file(path, content).await,
        }
    }
    pub async fn write_file_if_revision(
        &self,
        path: &Path,
        content: &str,
        expected_revision: &str,
    ) -> VaultResult<PathBuf> {
        match self {
            Self::Local(root) => {
                let root = root.clone();
                let path = path.to_owned();
                let content = content.to_owned();
                let expected_revision = expected_revision.to_owned();
                native(move || {
                    let observed = root.read_file(&path)?;
                    let current_revision = super::source::source_revision(&observed);
                    if current_revision != expected_revision {
                        return Err(VaultError::SourceChanged {
                            path,
                            expected_revision,
                            current_revision,
                        });
                    }
                    root.write_file(&path, &content)
                })
                .await
            }
            Self::Sftp(root) => {
                root.write_file_if_revision(path, content, expected_revision)
                    .await
            }
        }
    }
    pub async fn create_file(&self, path: &Path, content: &str) -> VaultResult<PathBuf> {
        match self {
            Self::Local(root) => {
                let root = root.clone();
                let path = path.to_owned();
                let content = content.to_owned();
                native(move || root.create_file(&path, &content)).await
            }
            Self::Sftp(root) => root.create_file(path, content).await,
        }
    }
    pub async fn delete_file(&self, path: &Path) -> VaultResult<PathBuf> {
        match self {
            Self::Local(root) => {
                let root = root.clone();
                let path = path.to_owned();
                native(move || root.delete_file(&path)).await
            }
            Self::Sftp(root) => root.delete_file(path).await,
        }
    }
    pub async fn move_file(&self, from: &Path, to: &Path) -> VaultResult<MoveResult> {
        match self {
            Self::Local(root) => {
                let root = root.clone();
                let from = from.to_owned();
                let to = to.to_owned();
                native(move || root.move_file(&from, &to)).await
            }
            Self::Sftp(root) => root.move_file(from, to).await,
        }
    }
}

async fn native<T: Send + 'static>(
    operation: impl FnOnce() -> VaultResult<T> + Send + 'static,
) -> VaultResult<T> {
    let result = tokio::task::spawn_blocking(move || {
        #[cfg(test)]
        let before = io_counts();
        let result = operation();
        #[cfg(test)]
        {
            let after = io_counts();
            (
                result,
                [
                    after[0] - before[0],
                    after[1] - before[1],
                    after[2] - before[2],
                ],
            )
        }
        #[cfg(not(test))]
        {
            result
        }
    })
    .await
    .map_err(|error| VaultError::Other(format!("source operation task failed: {error}")))?;
    #[cfg(test)]
    {
        use crate::test_helpers::{EXACT_FILE_OPENS, INDEX_DIRECTORY_VISITS, NAME_LOOKUP_SCANS};
        for (counter, count) in [
            (&EXACT_FILE_OPENS, result.1[0]),
            (&NAME_LOOKUP_SCANS, result.1[1]),
            (&INDEX_DIRECTORY_VISITS, result.1[2]),
        ] {
            counter.set(counter.get() + count)
        }
        result.0
    }
    #[cfg(not(test))]
    {
        result
    }
}

#[cfg(test)]
fn io_counts() -> [usize; 3] {
    use crate::test_helpers::{EXACT_FILE_OPENS, INDEX_DIRECTORY_VISITS, NAME_LOOKUP_SCANS};
    [
        EXACT_FILE_OPENS.get(),
        NAME_LOOKUP_SCANS.get(),
        INDEX_DIRECTORY_VISITS.get(),
    ]
}
