//! SFTP-backed source files. Paths are interpreted by the server, never the client OS.

mod read_batch;
mod session;

pub(crate) use read_batch::ReadBatch;

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use globset::Glob;
use openssh_sftp_client_lowlevel::{FileAttrs, FileType, Permissions};
use sha2::{Digest, Sha256};
use tokio::process::Command;
use tokio::sync::{Mutex as AsyncMutex, Semaphore};
use unicode_normalization::UnicodeNormalization;

use crate::config::source::SftpLocation;
use crate::error::{MutationOutcome, StorageErrorKind, VaultError, VaultResult};
use crate::models::FileStat;
use crate::vault::fs::MoveResult;
use crate::vault::path::{ResolvedPath, VaultPath};

use session::{HandleLifetime, Session};

pub(crate) const MAX_READ_BYTES: usize = 64 * 1024 * 1024;
const CHUNK_BYTES: u32 = 32 * 1024;

#[derive(Clone, Debug)]
pub struct RemoteEntry {
    pub relative: PathBuf,
    pub stat: FileStat,
}

#[derive(Default)]
struct PublicationProgress {
    destination_dispatched: AtomicBool,
    temporary: Mutex<Option<String>>,
}

impl PublicationProgress {
    fn set_temporary(&self, path: Option<&str>) {
        *self.temporary.lock().unwrap_or_else(|e| e.into_inner()) = path.map(str::to_owned);
    }

    fn report_residual<T>(&self, result: VaultResult<T>) -> VaultResult<T> {
        let temporary = self.temporary.lock().unwrap_or_else(|e| e.into_inner());
        result.map_err(|mut error| {
            if let Some(temporary) = temporary.as_ref()
                && let VaultError::Storage { path, message, .. } = &mut error
            {
                message.push_str("; temporary upload may remain");
                if path != temporary {
                    message.push_str(" at ");
                    message.push_str(temporary);
                }
            }
            error
        })
    }
}

/// A persistent, authenticated remote source with a pinned connection namespace.
pub struct SftpStorage {
    location: SftpLocation,
    canonical: String,
    path: PathBuf,
    source_id: String,
    effective: String,
    timeout: Duration,
    session: AsyncMutex<Arc<Session>>,
    policy_revision: Mutex<Option<Option<String>>>,
    read_budget: Semaphore,
    read_admission: Semaphore,
    last_failure: Mutex<Option<(StorageErrorKind, String)>>,
    reconciliation_owner: OnceLock<Arc<tokio::sync::Notify>>,
}

impl std::fmt::Debug for SftpStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SftpStorage")
            .field("location", &self.location.display_uri())
            .field("canonical", &self.canonical)
            .field("source_id", &self.source_id)
            .finish_non_exhaustive()
    }
}

impl SftpStorage {
    pub async fn open(location: &SftpLocation) -> VaultResult<Self> {
        let timeout = timeout_from_env()?;
        tokio::time::timeout_at(
            super::source_deadline(timeout),
            Self::open_inner(location, timeout),
        )
        .await
        .map_err(|_| {
            storage_error(
                "connect",
                &location.root,
                StorageErrorKind::Unavailable,
                "SSH/SFTP initialization timed out",
            )
        })?
    }

    async fn open_inner(location: &SftpLocation, timeout: Duration) -> VaultResult<Self> {
        let effective = effective_connection(location, timeout).await?;
        let session = Arc::new(Session::connect(location, timeout).await?);
        let canonical = session.realpath(&location.root).await?;
        validate_absolute(&canonical)?;
        require_type(
            &session.stat(&canonical).await?,
            FileType::Directory,
            &canonical,
        )?;
        let source_id = source_identity(&location.display_uri(), &effective, &canonical);
        Ok(Self {
            location: location.clone(),
            path: PathBuf::from(&canonical),
            canonical,
            source_id,
            effective,
            timeout,
            session: AsyncMutex::new(session),
            policy_revision: Mutex::new(None),
            read_budget: Semaphore::new(MAX_READ_BYTES / 1024),
            read_admission: Semaphore::new(16),
            last_failure: Mutex::new(None),
            reconciliation_owner: OnceLock::new(),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
    pub fn canonical_root(&self) -> &str {
        &self.canonical
    }
    pub fn location(&self) -> &SftpLocation {
        &self.location
    }
    pub fn source_id(&self) -> &str {
        &self.source_id
    }

    pub(crate) fn set_reconciliation_owner(
        &self,
        wakeup: Arc<tokio::sync::Notify>,
    ) -> VaultResult<()> {
        self.reconciliation_owner.set(wakeup).map_err(|_| {
            storage_error(
                "configure source",
                "",
                StorageErrorKind::Protocol,
                "source reconciliation owner is already configured",
            )
        })
    }

    /// Pin the exact portable policy observation used to authorize indexing.
    pub fn pin_policy_revision(&self, revision: Option<String>) {
        *self
            .policy_revision
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Some(revision);
    }

    /// The most recent source-wide failure; reading this never contacts the server.
    pub fn last_failure(&self) -> Option<(StorageErrorKind, String)> {
        let mut failure = self.last_failure.lock().unwrap_or_else(|e| e.into_inner());
        if failure
            .as_ref()
            .is_some_and(|(kind, _)| *kind == StorageErrorKind::IdentityChanged)
        {
            return failure.clone();
        }
        if let Ok(session) = self.session.try_lock() {
            match session.failure_snapshot() {
                Some((StorageErrorKind::Unavailable, _))
                    if failure
                        .as_ref()
                        .is_some_and(|(kind, _)| *kind == StorageErrorKind::Protocol) => {}
                Some(observed) => *failure = Some(observed),
                None if failure
                    .as_ref()
                    .is_some_and(|(kind, _)| *kind == StorageErrorKind::Unavailable) =>
                {
                    return None;
                }
                None => (),
            }
        }
        failure.clone()
    }

    fn record_result<T>(&self, result: &VaultResult<T>) {
        let mut failure = self.last_failure.lock().unwrap_or_else(|e| e.into_inner());
        if failure
            .as_ref()
            .is_some_and(|(kind, _)| *kind == StorageErrorKind::IdentityChanged)
            || (failure
                .as_ref()
                .is_some_and(|(kind, _)| *kind == StorageErrorKind::Protocol)
                && matches!(
                    result,
                    Err(VaultError::Storage {
                        kind: StorageErrorKind::Unavailable,
                        ..
                    })
                ))
        {
            return;
        }
        match result {
            Err(VaultError::Storage {
                kind: StorageErrorKind::Unavailable,
                ..
            }) if self
                .session
                .try_lock()
                .is_ok_and(|session| session.is_healthy()) => {}
            Err(error @ VaultError::Storage { kind, .. })
                if matches!(
                    kind,
                    StorageErrorKind::Unavailable
                        | StorageErrorKind::IdentityChanged
                        | StorageErrorKind::Protocol
                ) =>
            {
                *failure = Some((*kind, error.to_string()));
            }
            Ok(_)
                if self
                    .session
                    .try_lock()
                    .is_ok_and(|session| session.is_healthy()) =>
            {
                *failure = None;
            }
            _ => (),
        }
    }

    async fn operation<T>(
        &self,
        operation: &str,
        path: &Path,
        work: impl std::future::Future<Output = VaultResult<T>>,
    ) -> VaultResult<T> {
        self.timed_operation(operation, path, None, work).await
    }

    async fn timed_operation<T>(
        &self,
        operation: &str,
        path: &Path,
        destination_dispatch: Option<&AtomicBool>,
        work: impl std::future::Future<Output = VaultResult<T>>,
    ) -> VaultResult<T> {
        if let Some((StorageErrorKind::IdentityChanged, message)) = self.last_failure() {
            return Err(storage_error(
                operation,
                &path.display().to_string(),
                StorageErrorKind::IdentityChanged,
                &message,
            ));
        }
        let result = match tokio::time::timeout_at(super::source_deadline(self.timeout), work).await
        {
            Ok(result) => result,
            Err(_) => Err(storage_error(
                operation,
                &path.display().to_string(),
                StorageErrorKind::Unavailable,
                "SFTP source operation exceeded its total deadline",
            )
            .with_mutation_outcome(match destination_dispatch {
                Some(dispatched) if dispatched.load(Ordering::Acquire) => MutationOutcome::Unknown,
                Some(_) => MutationOutcome::NotCommitted,
                None => MutationOutcome::NotDispatched,
            })),
        };
        self.record_result(&result);
        result
    }

    async fn current_session(&self) -> VaultResult<Arc<Session>> {
        let mut current = self.session.lock().await;
        if current.is_healthy() {
            return Ok(Arc::clone(&current));
        }
        let effective = effective_connection(&self.location, self.timeout).await?;
        if effective != self.effective {
            return Err(storage_error(
                "reconnect",
                "",
                StorageErrorKind::IdentityChanged,
                "SSH connection profile changed; reopen the source",
            ));
        }
        let next = Arc::new(Session::connect(&self.location, self.timeout).await?);
        let canonical = next.realpath(&self.location.root).await?;
        if canonical != self.canonical {
            return Err(storage_error(
                "reconnect",
                "",
                StorageErrorKind::IdentityChanged,
                "remote root changed; reopen the source",
            ));
        }
        self.validate_policy_revision(&next).await?;
        *current = Arc::clone(&next);
        self.notify_reconnected();
        Ok(next)
    }

    async fn validate_policy_revision(&self, session: &Session) -> VaultResult<()> {
        let expected = self
            .policy_revision
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        if let Some(expected) = expected {
            let absolute = absolute(&self.canonical, ".obsidian-mcp/ignore");
            let observed = match session.realpath(&absolute).await {
                Ok(path) => {
                    relative_under(&self.canonical, &path).map_err(policy_validation_error)?;
                    match read_source(
                        session,
                        &path,
                        256 * 1024,
                        &self.read_budget,
                        &self.read_admission,
                    )
                    .await
                    {
                        Ok(raw) => Some(crate::vault::source::source_revision(&raw)),
                        Err(error) if error.is_not_found() => None,
                        Err(error) => return Err(policy_validation_error(error)),
                    }
                }
                Err(error) if error.is_not_found() => None,
                Err(error) => return Err(error),
            };
            if observed != expected {
                return Err(storage_error(
                    "reconnect",
                    ".obsidian-mcp/ignore",
                    StorageErrorKind::IdentityChanged,
                    "portable indexing policy changed; restart to authorize the new policy",
                ));
            }
        }
        Ok(())
    }

    fn notify_reconnected(&self) {
        if let Some(wakeup) = self.reconciliation_owner.get() {
            wakeup.notify_one();
        }
    }

    pub async fn resolve_existing(&self, path: &Path) -> VaultResult<ResolvedPath> {
        self.operation("resolve", path, async { self.resolve(path, false).await })
            .await
    }

    pub async fn resolve_for_write(&self, path: &Path) -> VaultResult<ResolvedPath> {
        self.operation("resolve", path, async { self.resolve(path, true).await })
            .await
    }

    async fn resolve(&self, path: &Path, allow_missing: bool) -> VaultResult<ResolvedPath> {
        let session = self.current_session().await?;
        self.resolve_with(&session, path, allow_missing).await
    }

    async fn resolve_with(
        &self,
        session: &Arc<Session>,
        path: &Path,
        allow_missing: bool,
    ) -> VaultResult<ResolvedPath> {
        ReadBatch::with_session(self, Arc::clone(session))
            .resolve(path, allow_missing)
            .await
    }

    pub async fn read_file(&self, path: &Path) -> VaultResult<String> {
        self.read_file_bounded(path, MAX_READ_BYTES).await
    }

    pub async fn read_file_bounded(&self, path: &Path, max_bytes: usize) -> VaultResult<String> {
        self.operation("read", path, async {
            let session = self.current_session().await?;
            let resolved = self.resolve_with(&session, path, false).await?;
            let absolute = path_text(&resolved.absolute)?;
            read_source(
                &session,
                absolute,
                max_bytes.min(MAX_READ_BYTES),
                &self.read_budget,
                &self.read_admission,
            )
            .await
        })
        .await
    }

    pub async fn read_exact_bounded(&self, path: &Path, max_bytes: usize) -> VaultResult<String> {
        self.operation("read", path, async {
            let session = self.current_session().await?;
            let absolute = self.resolve_exact(&session, path).await?;
            read_source(
                &session,
                &absolute,
                max_bytes.min(MAX_READ_BYTES),
                &self.read_budget,
                &self.read_admission,
            )
            .await
        })
        .await
    }

    async fn resolve_exact(&self, session: &Session, path: &Path) -> VaultResult<String> {
        let raw = path_text(path)?;
        if normalize_relative(path)? != raw {
            return Err(VaultError::InvalidPath(
                "expected an exact observed remote key".into(),
            ));
        }
        if session.realpath(&self.canonical).await? != self.canonical {
            return Err(storage_error(
                "read",
                raw,
                StorageErrorKind::IdentityChanged,
                "remote root changed; reopen the source",
            ));
        }
        let mut parent = self.canonical.clone();
        for name in raw.split('/').filter(|s| !s.is_empty()) {
            let mut found = false;
            visit_directory(session, &parent, |entry, _| {
                found |= entry == name;
                Ok(())
            })
            .await?;
            if !found {
                return Err(VaultError::NoteNotFound(path.to_owned()));
            }
            parent = absolute(&parent, name);
            let attrs = session.lstat(&parent).await?;
            if attrs.get_filetype() == Some(FileType::Symlink) {
                return Err(VaultError::InvalidPath(
                    "observed remote entry changed to a symlink".into(),
                ));
            }
        }
        let canonical = session.realpath(&parent).await?;
        if canonical != parent {
            return Err(VaultError::InvalidPath(
                "observed remote path spelling changed".into(),
            ));
        }
        relative_under(&self.canonical, &canonical)?;
        Ok(parent)
    }

    pub async fn read_observed_with_stat(&self, path: &Path) -> VaultResult<(String, FileStat)> {
        self.operation("read", path, async {
            let session = self.current_session().await?;
            let absolute = self.resolve_exact(&session, path).await?;
            read_source_with_stat(
                &session,
                &absolute,
                MAX_READ_BYTES,
                &self.read_budget,
                &self.read_admission,
            )
            .await
        })
        .await
    }

    /// Validate exact names once for a caller-owned group of source reads.
    /// The observation is intentionally scoped to the batch and is never cached.
    pub async fn observe_batch(&self, paths: &[PathBuf]) -> VaultResult<ObservedBatch<'_>> {
        self.operation("observe", Path::new(""), async {
            let session = self.current_session().await?;
            if session.realpath(&self.canonical).await? != self.canonical {
                return Err(storage_error(
                    "observe",
                    "",
                    StorageErrorKind::IdentityChanged,
                    "remote root changed; reopen the source",
                ));
            }
            let mut children: HashMap<String, HashSet<String>> = HashMap::new();
            let mut requested = HashSet::new();
            let mut name_bytes = 0usize;
            for path in paths {
                let raw = path_text(path)?;
                if normalize_relative(path)? != raw || raw.is_empty() {
                    return Err(VaultError::InvalidPath(
                        "expected an exact observed remote key".into(),
                    ));
                }
                requested.insert(raw.to_owned());
                let mut parent = String::new();
                for name in raw.split('/') {
                    let names = children.entry(parent.clone()).or_default();
                    if names.insert(name.to_owned()) {
                        name_bytes = name_bytes.saturating_add(parent.len() + name.len());
                        if name_bytes > MAX_READ_BYTES {
                            return Err(storage_error(
                                "observe",
                                "",
                                StorageErrorKind::ResourceLimit,
                                "batch path observation budget exceeded",
                            ));
                        }
                    }
                    parent = absolute(&parent, name);
                }
            }
            let requested = Arc::new(requested);
            let mut pending = vec![String::new()];
            let mut jobs = tokio::task::JoinSet::new();
            let mut observed = HashSet::new();
            let result = async {
                while !pending.is_empty() || !jobs.is_empty() {
                    while jobs.len() < 8 {
                        let Some(directory) = pending.pop() else {
                            break;
                        };
                        let Some(needed) = children.remove(&directory) else {
                            continue;
                        };
                        let session = Arc::clone(&session);
                        let remote = absolute(&self.canonical, &directory);
                        let requested = Arc::clone(&requested);
                        jobs.spawn(async move {
                            let mut hits = Vec::new();
                            let result = visit_directory(&session, &remote, |name, attrs| {
                                if needed.contains(&name) {
                                    let relative = absolute(&directory, &name);
                                    match attrs.get_filetype() {
                                        Some(FileType::Directory) => hits.push((relative, true)),
                                        Some(FileType::RegularFile)
                                            if requested.contains(&relative) =>
                                        {
                                            hits.push((relative, false))
                                        }
                                        _ => (),
                                    }
                                }
                                Ok(())
                            })
                            .await;
                            match result {
                                Ok(()) => (),
                                Err(VaultError::DirectoryNotFound(_)) if !directory.is_empty() => {}
                                Err(error) => return Err(error),
                            }
                            Ok::<_, VaultError>(hits)
                        });
                    }
                    if let Some(result) = jobs.join_next().await {
                        for (path, directory) in result.map_err(|error| {
                            storage_error(
                                "observe",
                                "",
                                StorageErrorKind::Unavailable,
                                &error.to_string(),
                            )
                        })?? {
                            if directory {
                                pending.push(path);
                            } else {
                                observed.insert(path);
                            }
                        }
                    }
                }
                Ok::<_, VaultError>(())
            }
            .await;
            while jobs.join_next().await.is_some() {}
            result?;
            Ok(ObservedBatch {
                storage: self,
                session,
                requested,
                observed,
            })
        })
        .await
    }

    pub async fn file_stat(&self, path: &Path) -> VaultResult<FileStat> {
        self.operation("stat", path, async {
            let session = self.current_session().await?;
            let resolved = self.resolve_with(&session, path, false).await?;
            file_stat(&session.stat(path_text(&resolved.absolute)?).await?, path)
        })
        .await
    }

    pub async fn is_file(&self, path: &Path) -> VaultResult<bool> {
        self.operation("stat", path, async {
            self.has_type(path, FileType::RegularFile).await
        })
        .await
    }

    pub async fn is_dir(&self, path: &Path) -> VaultResult<bool> {
        self.operation("stat", path, async {
            self.has_type(path, FileType::Directory).await
        })
        .await
    }

    async fn has_type(&self, path: &Path, kind: FileType) -> VaultResult<bool> {
        let session = self.current_session().await?;
        let resolved = match self.resolve_with(&session, path, false).await {
            Ok(value) => value,
            Err(error) if error.is_not_found() => return Ok(false),
            Err(error) => return Err(error),
        };
        let attrs = session.stat(path_text(&resolved.absolute)?).await?;
        match attrs.get_filetype() {
            Some(actual) => Ok(actual == kind),
            None => Err(storage_error(
                "stat",
                &path.display().to_string(),
                StorageErrorKind::Unsupported,
                "server omitted the file type",
            )),
        }
    }

    /// Enumerate visible regular files without following discovered symlinks.
    pub async fn entries(&self, path: &Path, recursive: bool) -> VaultResult<Vec<RemoteEntry>> {
        self.operation("list", path, async {
            let session = self.current_session().await?;
            let resolved = self.resolve_with(&session, path, false).await?;
            require_type(
                &session.stat(path_text(&resolved.absolute)?).await?,
                FileType::Directory,
                path_text(path)?,
            )?;
            let mut entries = Vec::new();
            for (relative, attrs) in walk_directories(
                Arc::clone(&session),
                &self.canonical,
                path_text(&resolved.relative)?.to_owned(),
                recursive,
            )
            .await?
            {
                if attrs.get_filetype() == Some(FileType::RegularFile) {
                    let path = PathBuf::from(relative);
                    let stat = file_stat(&attrs, &path)?;
                    entries.push(RemoteEntry {
                        relative: path,
                        stat,
                    });
                }
            }
            entries.sort_by(|a, b| a.relative.as_os_str().cmp(b.relative.as_os_str()));
            Ok(entries)
        })
        .await
    }

    pub async fn list_files(
        &self,
        path: &Path,
        recursive: bool,
        glob: Option<&str>,
    ) -> VaultResult<Vec<PathBuf>> {
        self.list_entries(path, recursive, glob, false).await
    }

    pub async fn list_regular_files(
        &self,
        path: &Path,
        recursive: bool,
        glob: Option<&str>,
    ) -> VaultResult<Vec<PathBuf>> {
        self.list_entries(path, recursive, glob, true).await
    }

    async fn list_entries(
        &self,
        path: &Path,
        recursive: bool,
        glob: Option<&str>,
        regular_only: bool,
    ) -> VaultResult<Vec<PathBuf>> {
        self.operation("list", path, async {
            let session = self.current_session().await?;
            let resolved = self.resolve_with(&session, path, false).await?;
            require_type(
                &session.stat(path_text(&resolved.absolute)?).await?,
                FileType::Directory,
                path_text(path)?,
            )?;
            let matcher = glob
                .map(|pattern| Glob::new(pattern).map(|g| g.compile_matcher()))
                .transpose()
                .map_err(|e| VaultError::InvalidPath(format!("invalid glob: {e}")))?;
            let mut results = Vec::new();
            for (relative, attrs) in walk_directories(
                Arc::clone(&session),
                &self.canonical,
                path_text(&resolved.relative)?.to_owned(),
                recursive,
            )
            .await?
            {
                if matcher
                    .as_ref()
                    .is_some_and(|matcher| !matcher.is_match(&relative))
                {
                    continue;
                }
                if regular_only {
                    match attrs.get_filetype() {
                        Some(FileType::RegularFile) => (),
                        Some(FileType::Symlink) => {
                            let resolved = match self
                                .resolve_with(&session, Path::new(&relative), false)
                                .await
                            {
                                Ok(resolved) => resolved,
                                Err(error) if error.is_not_found() => continue,
                                Err(error) => return Err(error),
                            };
                            let attrs = session.stat(path_text(&resolved.absolute)?).await?;
                            match attrs.get_filetype() {
                                Some(FileType::RegularFile) => (),
                                Some(_) => continue,
                                None => {
                                    return Err(storage_error(
                                        "list",
                                        &relative,
                                        StorageErrorKind::Unsupported,
                                        "server omitted the file type",
                                    ));
                                }
                            }
                        }
                        _ => continue,
                    }
                }
                results.push(PathBuf::from(relative));
            }
            results.sort_by(|a, b| a.as_os_str().cmp(b.as_os_str()));
            Ok(results)
        })
        .await
    }

    pub async fn create_dir_all(&self, path: &Path) -> VaultResult<()> {
        let dispatched = AtomicBool::new(false);
        self.timed_operation("mkdir", path, Some(&dispatched), async {
            let session = self.current_session().await?;
            self.ensure_parent_with_stage(
                &session,
                &format!("{}/unused", normalize_relative(path)?),
                Some(&dispatched),
            )
            .await
        })
        .await
    }

    async fn ensure_parent(&self, session: &Session, relative: &str) -> VaultResult<()> {
        self.ensure_parent_with_stage(session, relative, None).await
    }

    async fn ensure_parent_with_stage(
        &self,
        session: &Session,
        relative: &str,
        dispatched: Option<&AtomicBool>,
    ) -> VaultResult<()> {
        if session.realpath(&self.canonical).await? != self.canonical {
            return Err(storage_error(
                "mkdir",
                "",
                StorageErrorKind::IdentityChanged,
                "remote root changed; reopen the source",
            ));
        }
        let (parent, _) = split_parent(relative)?;
        let mut current = String::new();
        for name in parent.split('/').filter(|s| !s.is_empty()) {
            current = if current.is_empty() {
                name.into()
            } else {
                format!("{current}/{name}")
            };
            let remote = absolute(&self.canonical, &current);
            match session.lstat(&remote).await {
                Ok(attrs) => {
                    require_type(&attrs, FileType::Directory, &current)?;
                    relative_under(&self.canonical, &session.realpath(&remote).await?)?;
                }
                Err(error) if error.is_not_found() => {
                    if let Some(dispatched) = dispatched {
                        dispatched.store(true, Ordering::Release);
                    }
                    session.mkdir(&remote).await?;
                }
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }

    pub async fn create_file(&self, path: &Path, content: &str) -> VaultResult<PathBuf> {
        self.publish(path, content, true).await
    }

    pub async fn write_file(&self, path: &Path, content: &str) -> VaultResult<PathBuf> {
        self.publish(path, content, false).await
    }

    /// Publish a transformation only while its original raw revision still exists.
    pub async fn write_file_if_revision(
        &self,
        path: &Path,
        content: &str,
        expected: &str,
    ) -> VaultResult<PathBuf> {
        let progress = PublicationProgress::default();
        let result = self
            .timed_operation(
                "write",
                path,
                Some(&progress.destination_dispatched),
                async {
                    self.publish_inner(path, content, false, Some(expected), &progress)
                        .await
                        .map_err(|error| publication_error(path, error))
                },
            )
            .await;
        progress.report_residual(result)
    }

    async fn publish(&self, path: &Path, content: &str, create: bool) -> VaultResult<PathBuf> {
        let progress = PublicationProgress::default();
        let result = self
            .timed_operation(
                "write",
                path,
                Some(&progress.destination_dispatched),
                async {
                    self.publish_inner(path, content, create, None, &progress)
                        .await
                        .map_err(|error| publication_error(path, error))
                },
            )
            .await;
        progress.report_residual(result)
    }

    async fn publish_inner(
        &self,
        path: &Path,
        content: &str,
        create: bool,
        expected_revision: Option<&str>,
        progress: &PublicationProgress,
    ) -> VaultResult<PathBuf> {
        let session = self.current_session().await?;
        let target = self.resolve_with(&session, path, true).await?;
        let relative = path_text(&target.relative)?.to_owned();
        let destination = path_text(&target.absolute)?.to_owned();
        let previous = match session.lstat(&destination).await {
            Ok(_) if create => return Err(VaultError::AlreadyExists(path.to_owned())),
            Ok(attrs) => {
                require_type(&attrs, FileType::RegularFile, &relative)?;
                if !session.has_posix_rename() {
                    return Err(storage_error(
                        "replace",
                        &relative,
                        StorageErrorKind::Unsupported,
                        "server does not support atomic POSIX replacement",
                    ));
                }
                Some((
                    attrs,
                    crate::vault::source::source_revision(
                        &read_source(
                            &session,
                            &destination,
                            MAX_READ_BYTES,
                            &self.read_budget,
                            &self.read_admission,
                        )
                        .await?,
                    ),
                ))
            }
            Err(error) if error.is_not_found() => None,
            Err(error) => return Err(error),
        };
        if let Some(expected) = expected_revision {
            let current = previous
                .as_ref()
                .map(|(_, revision)| revision.as_str())
                .unwrap_or("missing");
            if current != expected {
                return Err(VaultError::SourceChanged {
                    path: path.to_owned(),
                    expected_revision: expected.into(),
                    current_revision: current.into(),
                });
            }
        }
        self.ensure_parent(&session, &relative).await?;
        let (parent, _) = split_parent(&relative)?;
        let mut entropy = [0u8; 16];
        getrandom::fill(&mut entropy).map_err(|e| {
            storage_error(
                "create upload",
                &relative,
                StorageErrorKind::Unavailable,
                &e.to_string(),
            )
        })?;
        let nonce: String = entropy.iter().map(|b| format!("{b:02x}")).collect();
        let temp_relative = if parent.is_empty() {
            format!(".obsidian-mcp-upload-{nonce}")
        } else {
            format!("{parent}/.obsidian-mcp-upload-{nonce}")
        };
        let temporary = absolute(&self.canonical, &temp_relative);
        let mut temporary_attrs = FileAttrs::new();
        temporary_attrs.set_permissions(Permissions::READ_BY_OWNER | Permissions::WRITE_BY_OWNER);
        progress.set_temporary(Some(&temporary));
        let handle = match session.open_create(&temporary, temporary_attrs).await {
            Ok(handle) => session.track_handle(handle),
            Err(error) => {
                if error.is_not_found()
                    || matches!(
                        &error,
                        VaultError::Storage {
                            outcome: MutationOutcome::NotCommitted | MutationOutcome::NotDispatched,
                            ..
                        }
                    )
                {
                    progress.set_temporary(None);
                }
                return Err(error.with_mutation_outcome(MutationOutcome::NotCommitted));
            }
        };
        let upload = self
            .upload(
                &session,
                &temporary,
                handle,
                content.as_bytes(),
                previous.as_ref().map(|(attrs, _)| *attrs),
            )
            .await;
        if let Err(error) = upload {
            return Err(cleanup_error(
                &session,
                &temporary,
                progress,
                path,
                error.with_mutation_outcome(MutationOutcome::NotCommitted),
            )
            .await);
        }
        if let Some((_, raw)) = &previous {
            let current = match read_source(
                &session,
                &destination,
                MAX_READ_BYTES,
                &self.read_budget,
                &self.read_admission,
            )
            .await
            {
                Ok(raw) => raw,
                Err(error) => {
                    return Err(cleanup_error(
                        &session,
                        &temporary,
                        progress,
                        path,
                        error.with_mutation_outcome(MutationOutcome::NotCommitted),
                    )
                    .await);
                }
            };
            if crate::vault::source::source_revision(&current) != *raw {
                let error = VaultError::SourceChanged {
                    path: path.to_owned(),
                    expected_revision: raw.clone(),
                    current_revision: crate::vault::source::source_revision(&current),
                };
                return Err(cleanup_error(&session, &temporary, progress, path, error).await);
            }
        }
        let replacing = previous.is_some();
        progress
            .destination_dispatched
            .store(true, Ordering::Release);
        if let Err(error) = session.rename(&temporary, &destination, replacing).await {
            return Err(cleanup_error(&session, &temporary, progress, path, error).await);
        }
        progress.set_temporary(None);
        Ok(target.relative)
    }

    async fn upload(
        &self,
        session: &Session,
        path: &str,
        handle: HandleLifetime<'_>,
        bytes: &[u8],
        previous: Option<FileAttrs>,
    ) -> VaultResult<()> {
        let result = async {
            for (index, chunk) in bytes.chunks(CHUNK_BYTES as usize).enumerate() {
                session
                    .write(
                        handle.handle(),
                        (index * CHUNK_BYTES as usize) as u64,
                        chunk,
                    )
                    .await?;
            }
            if let Some(previous) = previous {
                let existing = session.lstat(path).await?;
                let mut attrs = FileAttrs::new();
                if let Some(mode) = previous.get_permissions() {
                    attrs.set_permissions(mode);
                }
                if let Some((uid, gid)) = previous.get_id()
                    && existing.get_id() != Some((uid, gid))
                {
                    attrs.set_id(uid, gid);
                }
                session.fsetstat(handle.handle(), attrs).await?;
            }
            if session.has_fsync() {
                session.fsync(handle.handle()).await?;
            }
            Ok(())
        }
        .await;
        let close = handle.close().await;
        result.and(close)
    }

    pub async fn delete_file(&self, path: &Path) -> VaultResult<PathBuf> {
        let dispatched = AtomicBool::new(false);
        self.timed_operation("delete", path, Some(&dispatched), async {
            let session = self.current_session().await?;
            let (relative, attrs) = self.resolve_entry(&session, path, false).await?;
            let remote = absolute(&self.canonical, &relative);
            if attrs.get_filetype() == Some(FileType::Symlink) {
                match session.realpath(&remote).await {
                    Ok(target) => {
                        relative_under(&self.canonical, &target)?;
                    }
                    Err(error) if error.is_not_found() => (),
                    Err(error) => return Err(error),
                }
            }
            dispatched.store(true, Ordering::Release);
            if attrs.get_filetype() == Some(FileType::Directory) {
                session.rmdir(&remote).await?;
            } else {
                session.remove(&remote).await?;
            }
            Ok(relative.into())
        })
        .await
    }

    pub async fn move_file(&self, from: &Path, to: &Path) -> VaultResult<MoveResult> {
        let dispatched = AtomicBool::new(false);
        self.timed_operation("move", from, Some(&dispatched), async {
            let session = self.current_session().await?;
            let resolved = self.resolve_with(&session, from, false).await?;
            require_type(
                &session.stat(path_text(&resolved.absolute)?).await?,
                FileType::RegularFile,
                path_text(from)?,
            )?;
            let (source, source_attrs) = self.resolve_entry(&session, from, false).await?;
            let destination = normalize_relative(to)?;
            let (destination_parent, destination_name) = split_parent(&destination)?;
            let parent = self
                .resolve_with(&session, Path::new(destination_parent), true)
                .await?;
            let destination = absolute(path_text(&parent.relative)?, destination_name)
                .trim_start_matches('/')
                .to_owned();
            let remote_destination = absolute(&self.canonical, &destination);
            match find_entry(&session, path_text(&parent.absolute)?, destination_name).await {
                Ok(Some(_)) => return Err(VaultError::AlreadyExists(to.to_owned())),
                Ok(None) => (),
                Err(error) if error.is_not_found() => (),
                Err(error) => return Err(error),
            }
            if source_attrs.get_filetype() == Some(FileType::Symlink)
                && split_parent(&source)?.0 != split_parent(&destination)?.0
                && !session
                    .readlink(&absolute(&self.canonical, &source))
                    .await?
                    .starts_with('/')
            {
                return Err(VaultError::InvalidPath(
                    "cannot move a relative remote symlink to another directory".into(),
                ));
            }
            self.ensure_parent(&session, &destination).await?;
            dispatched.store(true, Ordering::Release);
            session
                .rename(
                    &absolute(&self.canonical, &source),
                    &remote_destination,
                    false,
                )
                .await?;
            Ok(MoveResult {
                from: source.into(),
                to: destination.into(),
            })
        })
        .await
    }

    async fn resolve_entry(
        &self,
        session: &Arc<Session>,
        path: &Path,
        allow_missing: bool,
    ) -> VaultResult<(String, FileAttrs)> {
        let normalized = normalize_relative(path)?;
        let (parent, name) = split_parent(&normalized)?;
        let resolved = self
            .resolve_with(session, Path::new(parent), allow_missing)
            .await?;
        let parent = path_text(&resolved.relative)?;
        let (name, attrs) = find_entry(session, &absolute(&self.canonical, parent), name)
            .await?
            .ok_or_else(|| VaultError::NoteNotFound(path.to_owned()))?;
        let relative = absolute(parent, &name);
        Ok((relative, attrs))
    }
}

/// A short-lived observation of exact names for an explicit source batch.
pub struct ObservedBatch<'a> {
    storage: &'a SftpStorage,
    session: Arc<Session>,
    requested: Arc<HashSet<String>>,
    observed: HashSet<String>,
}

impl ObservedBatch<'_> {
    pub async fn read(&self, path: &Path, max_bytes: usize) -> VaultResult<(String, FileStat)> {
        self.storage
            .operation("read", path, async {
                let relative = path_text(path)?;
                if !self.requested.contains(relative) {
                    return Err(VaultError::InvalidPath(
                        "path is outside the observed batch".into(),
                    ));
                }
                if !self.observed.contains(relative) {
                    return Err(VaultError::NoteNotFound(path.to_owned()));
                }
                let remote = absolute(&self.storage.canonical, relative);
                require_type(
                    &self.session.lstat(&remote).await?,
                    FileType::RegularFile,
                    relative,
                )?;
                let canonical = self.session.realpath(&remote).await?;
                if canonical != remote {
                    return Err(VaultError::InvalidPath(
                        "observed remote path changed".into(),
                    ));
                }
                relative_under(&self.storage.canonical, &canonical)?;
                read_source_with_stat(
                    &self.session,
                    &remote,
                    max_bytes.min(MAX_READ_BYTES),
                    &self.storage.read_budget,
                    &self.storage.read_admission,
                )
                .await
            })
            .await
    }
}

async fn cleanup_error(
    session: &Session,
    temporary: &str,
    progress: &PublicationProgress,
    destination: &Path,
    error: VaultError,
) -> VaultError {
    match session.remove(temporary).await {
        Err(cleanup) if !cleanup.is_not_found() => {
            let detail = match cleanup {
                VaultError::Storage {
                    operation,
                    kind,
                    message,
                    ..
                } => format!("{operation} ({kind:?}): {message}"),
                other => other.to_string(),
            };
            let mut error = match error {
                error @ VaultError::Storage { .. } => error,
                error => {
                    let kind = if matches!(error, VaultError::SourceChanged { .. }) {
                        StorageErrorKind::Conflict
                    } else {
                        StorageErrorKind::Unavailable
                    };
                    storage_error(
                        "write",
                        &destination.display().to_string(),
                        kind,
                        &error.to_string(),
                    )
                    .with_mutation_outcome(MutationOutcome::NotCommitted)
                }
            };
            if let VaultError::Storage { message, .. } = &mut error {
                message.push_str("; temporary upload cleanup failed: ");
                message.push_str(&detail);
            }
            error
        }
        _ => {
            progress.set_temporary(None);
            error
        }
    }
}

fn publication_error(destination: &Path, error: VaultError) -> VaultError {
    match error {
        VaultError::Storage {
            operation,
            path,
            kind,
            message,
            outcome,
        } => {
            let destination = destination.display().to_string();
            let message = if operation == "write" && path == destination {
                message
            } else {
                format!("{operation}: {message}")
            };
            VaultError::Storage {
                operation: "write".into(),
                path: destination,
                kind,
                message,
                outcome,
            }
        }
        other => other,
    }
}

async fn read_source(
    session: &Session,
    path: &str,
    max_bytes: usize,
    budget: &Semaphore,
    admission: &Semaphore,
) -> VaultResult<String> {
    read_source_with_stat(session, path, max_bytes, budget, admission)
        .await
        .map(|(content, _)| content)
}

async fn read_source_with_stat(
    session: &Session,
    path: &str,
    max_bytes: usize,
    budget: &Semaphore,
    admission: &Semaphore,
) -> VaultResult<(String, FileStat)> {
    let _reader = admission.acquire().await.map_err(|_| {
        storage_error(
            "read",
            path,
            StorageErrorKind::Unavailable,
            "source read admission closed",
        )
    })?;
    let handle = session.track_handle(session.open_read(path).await?);
    let result = async {
        let attrs = session.fstat(handle.handle()).await?;
        require_type(&attrs, FileType::RegularFile, path)?;
        if let Some(size) = attrs.get_size()
            && size > max_bytes as u64
        {
            return Err(limit_error(
                path,
                usize::try_from(size).unwrap_or(usize::MAX),
                max_bytes,
            ));
        }
        let hint = attrs
            .get_size()
            .unwrap_or(CHUNK_BYTES as u64)
            .min(max_bytes.saturating_add(1) as u64) as usize;
        let initial_kib = hint.max(1).div_ceil(1024).min(MAX_READ_BYTES / 1024) as u32;
        let mut reserved = budget.acquire_many(initial_kib).await.map_err(|_| {
            storage_error(
                "read",
                path,
                StorageErrorKind::Unavailable,
                "source content admission closed",
            )
        })?;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(hint).map_err(|e| {
            storage_error(
                "read",
                path,
                StorageErrorKind::ResourceLimit,
                &e.to_string(),
            )
        })?;
        let mut offset = 0u64;
        loop {
            let allowed = max_bytes
                .saturating_add(1)
                .saturating_sub(bytes.len())
                .min(CHUNK_BYTES as usize);
            if allowed == 0 {
                return Err(limit_error(path, bytes.len(), max_bytes));
            }
            let Some(chunk) = session
                .read(handle.handle(), offset, allowed as u32)
                .await?
            else {
                break;
            };
            if chunk.is_empty() || chunk.len() > allowed {
                return Err(storage_error(
                    "read",
                    path,
                    StorageErrorKind::Protocol,
                    "invalid SFTP read length",
                ));
            }
            let needed_kib = (bytes.len() + chunk.len())
                .max(1)
                .div_ceil(1024)
                .min(MAX_READ_BYTES / 1024);
            if needed_kib > reserved.num_permits() {
                let extra = budget
                    .try_acquire_many((needed_kib - reserved.num_permits()) as u32)
                    .map_err(|_| {
                        storage_error(
                            "read",
                            path,
                            StorageErrorKind::ResourceLimit,
                            "source content memory budget is in use",
                        )
                    })?;
                reserved.merge(extra);
            }
            bytes.try_reserve_exact(chunk.len()).map_err(|e| {
                storage_error(
                    "read",
                    path,
                    StorageErrorKind::ResourceLimit,
                    &e.to_string(),
                )
            })?;
            offset += chunk.len() as u64;
            bytes.extend_from_slice(&chunk);
            if bytes.len() > max_bytes {
                return Err(limit_error(path, bytes.len(), max_bytes));
            }
        }
        let content = String::from_utf8(bytes).map_err(|e| {
            storage_error(
                "read",
                path,
                StorageErrorKind::InvalidSource,
                &format!("remote note is not UTF-8: {e}"),
            )
        })?;
        let stat = file_stat(&session.fstat(handle.handle()).await?, Path::new(path))?;
        Ok((content, stat))
    }
    .await;
    let close = handle.close().await;
    match (result, close) {
        (Err(error), _) | (_, Err(error)) => Err(error),
        (Ok(text), Ok(())) => Ok(text),
    }
}

// LSTAT may accept a case variant while REALPATH preserves the requested spelling.
// Return the enumerated entry so mutations refresh the existing logical key.
async fn find_entry(
    session: &Session,
    parent: &str,
    name: &str,
) -> VaultResult<Option<(String, FileAttrs)>> {
    let native_match = match session.lstat(&absolute(parent, name)).await {
        Ok(_) => true,
        Err(error) if error.is_not_found() => false,
        Err(error) => return Err(error),
    };
    let mut matcher = EntryMatcher::new(name, native_match);
    visit_directory(session, parent, |name, attrs| {
        matcher.observe(name, attrs);
        Ok(())
    })
    .await?;
    matcher.finish()
}

struct EntryMatcher<'a> {
    requested: &'a str,
    key: String,
    folded: String,
    native_match: bool,
    exact: Option<(String, FileAttrs)>,
    found: Option<(String, FileAttrs)>,
    ambiguous: bool,
}

impl<'a> EntryMatcher<'a> {
    fn new(requested: &'a str, native_match: bool) -> Self {
        let key: String = requested.nfc().collect();
        let folded = key.to_lowercase();
        Self {
            requested,
            key,
            folded,
            native_match,
            exact: None,
            found: None,
            ambiguous: false,
        }
    }

    fn observe(&mut self, name: String, attrs: FileAttrs) {
        let normalized: String = name.nfc().collect();
        if name == self.requested {
            self.exact = Some((name, attrs));
        } else if normalized == self.key
            || (self.native_match && normalized.to_lowercase() == self.folded)
        {
            self.ambiguous |= self.found.is_some();
            self.found = Some((name, attrs));
        }
    }

    fn finish(self) -> VaultResult<Option<(String, FileAttrs)>> {
        if self.exact.is_some() {
            return Ok(self.exact);
        }
        if self.ambiguous {
            return Err(VaultError::InvalidPath(
                "ambiguous Unicode-equivalent remote path".into(),
            ));
        }
        if self.native_match && self.found.is_none() {
            return Err(VaultError::InvalidPath(
                "cannot identify the exact spelling of the native remote match".into(),
            ));
        }
        Ok(self.found)
    }
}

async fn walk_directories(
    session: Arc<Session>,
    canonical: &str,
    start: String,
    recursive: bool,
) -> VaultResult<Vec<(String, FileAttrs)>> {
    let mut pending = vec![start.clone()];
    let mut jobs = tokio::task::JoinSet::new();
    let mut entries = Vec::new();
    let mut name_bytes = 0usize;
    let result = async {
        while !pending.is_empty() || !jobs.is_empty() {
            while jobs.len() < 8 {
                let Some(directory) = pending.pop() else {
                    break;
                };
                let session = Arc::clone(&session);
                let remote = absolute(canonical, &directory);
                let starting_directory = directory == start;
                jobs.spawn(async move {
                    let mut entries = Vec::new();
                    let result = visit_directory(&session, &remote, |name, attrs| {
                        if !name.starts_with('.') {
                            entries.push((absolute(&directory, &name), attrs));
                        }
                        Ok(())
                    })
                    .await;
                    match result {
                        Ok(()) => (),
                        Err(VaultError::DirectoryNotFound(_)) if !starting_directory => (),
                        Err(error) => return Err(error),
                    }
                    Ok::<_, VaultError>(entries)
                });
            }
            if let Some(result) = jobs.join_next().await {
                for (path, attrs) in result.map_err(|error| {
                    storage_error(
                        "list",
                        "",
                        StorageErrorKind::Unavailable,
                        &error.to_string(),
                    )
                })?? {
                    name_bytes = name_bytes.saturating_add(path.len());
                    if entries.len() >= 1_000_000 || name_bytes > MAX_READ_BYTES {
                        return Err(storage_error(
                            "list",
                            "",
                            StorageErrorKind::ResourceLimit,
                            "source directory entry budget exceeded",
                        ));
                    }
                    if recursive && attrs.get_filetype() == Some(FileType::Directory) {
                        pending.push(path.clone());
                    }
                    entries.push((path, attrs));
                }
            }
        }
        Ok::<_, VaultError>(())
    }
    .await;
    while jobs.join_next().await.is_some() {}
    result?;
    Ok(entries)
}

async fn visit_directory(
    session: &Session,
    path: &str,
    mut visit: impl FnMut(String, FileAttrs) -> VaultResult<()>,
) -> VaultResult<()> {
    let handle = session.track_handle(session.opendir(path).await.map_err(|error| {
        if error.is_not_found() {
            VaultError::DirectoryNotFound(PathBuf::from(path))
        } else {
            error
        }
    })?);
    let mut skipped = 0usize;
    let result = async {
        let mut seen = HashSet::new();
        let mut name_bytes = 0usize;
        let mut entry_count = 0usize;
        loop {
            let page = session.readdir(handle.handle()).await?;
            skipped = skipped.saturating_add(page.skipped_count);
            name_bytes = name_bytes.saturating_add(page.skipped_bytes);
            entry_count = entry_count.saturating_add(page.skipped_count);
            if entry_count > 1_000_000 || name_bytes > MAX_READ_BYTES {
                return Err(storage_error(
                    "list",
                    path,
                    StorageErrorKind::ResourceLimit,
                    "directory entry budget exceeded",
                ));
            }
            if page.entries.is_empty() && page.skipped_count == 0 {
                break;
            }
            for entry in page.entries {
                let name = entry
                    .filename
                    .to_str()
                    .ok_or_else(|| {
                        storage_error(
                            "list",
                            path,
                            StorageErrorKind::Protocol,
                            "directory response contains an undecoded name",
                        )
                    })?
                    .to_owned();
                name_bytes = name_bytes.saturating_add(name.len());
                entry_count = entry_count.saturating_add(1);
                if entry_count > 1_000_000 || name_bytes > MAX_READ_BYTES {
                    return Err(storage_error(
                        "list",
                        path,
                        StorageErrorKind::ResourceLimit,
                        "directory entry budget exceeded",
                    ));
                }
                if name.is_empty() || name.contains('/') || name.contains('\0') {
                    return Err(storage_error(
                        "list",
                        path,
                        StorageErrorKind::Protocol,
                        "server returned an invalid directory basename",
                    ));
                }
                if name == "." || name == ".." {
                    continue;
                }
                if name.contains('\\') || name.chars().any(char::is_control) {
                    skipped = skipped.saturating_add(1);
                    continue;
                }
                if !seen.insert(name.clone()) {
                    return Err(storage_error(
                        "list",
                        path,
                        StorageErrorKind::Protocol,
                        "server repeated a directory entry",
                    ));
                }
                let attrs = if entry.attrs.get_filetype().is_none()
                    || (entry.attrs.get_filetype() == Some(FileType::RegularFile)
                        && entry.attrs.get_size().is_none())
                {
                    session.lstat(&absolute(path, &name)).await?
                } else {
                    entry.attrs
                };
                if attrs.get_filetype().is_none() {
                    return Err(storage_error(
                        "list",
                        path,
                        StorageErrorKind::Unsupported,
                        "server omitted the directory entry type",
                    ));
                }
                visit(name, attrs)?;
            }
        }
        Ok(())
    }
    .await;
    if skipped > 0 {
        tracing::warn!(path, skipped, "skipping unrepresentable vault entries");
    }
    let close = handle.close().await;
    result.and(close)
}

fn normalize_relative(path: &Path) -> VaultResult<String> {
    Ok(VaultPath::parse(path_text(path)?)?.as_str().to_owned())
}

fn path_text(path: &Path) -> VaultResult<&str> {
    path.to_str()
        .ok_or_else(|| VaultError::InvalidPath("remote paths must be UTF-8".into()))
}

fn validate_absolute(path: &str) -> VaultResult<()> {
    if !path.starts_with('/')
        || path.chars().any(char::is_control)
        || path.contains('\\')
        || path.split('/').any(|s| matches!(s, "." | ".."))
    {
        return Err(VaultError::InvalidPath(
            "server returned an invalid canonical remote path".into(),
        ));
    }
    Ok(())
}

fn relative_under(root: &str, path: &str) -> VaultResult<String> {
    validate_absolute(path)?;
    if path == root {
        return Ok(String::new());
    }
    path.strip_prefix(if root == "/" {
        root
    } else {
        root.trim_end_matches('/')
    })
    .and_then(|s| {
        if root == "/" {
            Some(s)
        } else {
            s.strip_prefix('/')
        }
    })
    .map(str::to_owned)
    .ok_or_else(|| VaultError::OutsideVault(PathBuf::from(path)))
}

fn resolved(root: &str, relative: String) -> ResolvedPath {
    ResolvedPath {
        absolute: PathBuf::from(absolute(root, &relative)),
        relative: PathBuf::from(relative),
    }
}

fn absolute(root: &str, relative: &str) -> String {
    if relative.is_empty() {
        root.to_owned()
    } else if root.is_empty() {
        relative.to_owned()
    } else {
        format!("{}/{}", root.trim_end_matches('/'), relative)
    }
}

fn split_parent(relative: &str) -> VaultResult<(&str, &str)> {
    match relative.rsplit_once('/') {
        Some((parent, name)) if !name.is_empty() => Ok((parent, name)),
        None if !relative.is_empty() => Ok(("", relative)),
        _ => Err(VaultError::InvalidPath(
            "a remote file path is required".into(),
        )),
    }
}

fn file_stat(attrs: &FileAttrs, path: &Path) -> VaultResult<FileStat> {
    let size = attrs.get_size().ok_or_else(|| {
        storage_error(
            "stat",
            &path.display().to_string(),
            StorageErrorKind::Unsupported,
            "server omitted the file size",
        )
    })?;
    Ok(FileStat {
        size,
        created: None,
        modified: attrs
            .get_time()
            .map(|(_, m)| chrono::DateTime::from(m.as_system_time())),
    })
}

fn require_type(attrs: &FileAttrs, expected: FileType, path: &str) -> VaultResult<()> {
    match attrs.get_filetype() {
        Some(actual) if actual == expected => Ok(()),
        Some(_) => Err(VaultError::InvalidPath(format!(
            "unexpected remote file type: {path}"
        ))),
        None => Err(storage_error(
            "stat",
            path,
            StorageErrorKind::Unsupported,
            "server omitted the file type",
        )),
    }
}

fn limit_error(path: &str, actual: usize, limit: usize) -> VaultError {
    VaultError::SemanticResourceLimit {
        resource: format!("source bytes for {path}"),
        actual,
        limit,
    }
}

fn policy_validation_error(error: VaultError) -> VaultError {
    match error {
        error @ (VaultError::OutsideVault(_)
        | VaultError::InvalidPath(_)
        | VaultError::SemanticResourceLimit { .. }
        | VaultError::Storage {
            kind: StorageErrorKind::InvalidSource,
            ..
        }) => storage_error(
            "reconnect",
            ".obsidian-mcp/ignore",
            StorageErrorKind::IdentityChanged,
            &format!("authorized portable policy is no longer valid; reopen the source: {error}"),
        ),
        other => other,
    }
}

fn storage_error(operation: &str, path: &str, kind: StorageErrorKind, message: &str) -> VaultError {
    VaultError::storage(operation, path, kind, message)
}

fn source_identity(profile: &str, effective: &str, canonical: &str) -> String {
    let mut hash = Sha256::new();
    hash.update(b"obsidian-mcp/sftp-source/v1\0");
    for value in [profile, effective, canonical] {
        hash.update((value.len() as u64).to_be_bytes());
        hash.update(value.as_bytes());
    }
    hash.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

async fn effective_connection(location: &SftpLocation, timeout: Duration) -> VaultResult<String> {
    let args = location.ssh_config_args();
    let output = tokio::time::timeout_at(
        super::source_deadline(timeout),
        Command::new("ssh").args(args).kill_on_drop(true).output(),
    )
    .await
    .map_err(|_| {
        storage_error(
            "resolve SSH profile",
            "",
            StorageErrorKind::Unavailable,
            "SSH configuration resolution timed out",
        )
    })?
    .map_err(|error| {
        storage_error(
            "resolve SSH profile",
            "",
            if error.kind() == std::io::ErrorKind::NotFound {
                StorageErrorKind::Unsupported
            } else {
                StorageErrorKind::Unavailable
            },
            &error.to_string(),
        )
    })?;
    if !output.status.success() {
        return Err(storage_error(
            "resolve SSH profile",
            "",
            StorageErrorKind::PermissionDenied,
            &ssh_profile_failure(output.status, &output.stderr),
        ));
    }
    let output = String::from_utf8(output.stdout).map_err(|_| {
        storage_error(
            "resolve SSH profile",
            "",
            StorageErrorKind::Protocol,
            "SSH configuration output is not UTF-8",
        )
    })?;
    let mut identity = Vec::new();
    for field in [
        "hostname",
        "user",
        "port",
        "hostkeyalias",
        "proxyjump",
        "proxycommand",
    ] {
        if let Some(line) = output
            .lines()
            .find(|line| line.starts_with(&format!("{field} ")))
        {
            identity.push(line.to_owned());
        }
    }
    if identity
        .iter()
        .filter(|l| l.starts_with("hostname ") || l.starts_with("user ") || l.starts_with("port "))
        .count()
        != 3
    {
        return Err(storage_error(
            "resolve SSH profile",
            "",
            StorageErrorKind::Protocol,
            "SSH configuration omitted connection identity",
        ));
    }
    Ok(identity.join("\n"))
}

fn ssh_profile_failure(status: impl std::fmt::Display, stderr: &[u8]) -> String {
    let stderr = &stderr[stderr.len().saturating_sub(1024)..];
    let detail: String = String::from_utf8_lossy(stderr)
        .chars()
        .filter(|c| !c.is_control() || *c == '\n')
        .collect();
    let detail = detail.trim();
    if detail.is_empty() {
        format!("SSH configuration could not be resolved ({status})")
    } else {
        format!("SSH configuration could not be resolved ({status}): {detail}")
    }
}

fn timeout_from_env() -> VaultResult<Duration> {
    Ok(crate::config::source::SftpSettings::from_env()?.timeout)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ssh_profile_failure_retains_bounded_sanitized_cause() {
        let mut stderr = vec![b'x'; 32 * 1024];
        stderr.extend_from_slice(b"\x1b[31mBad configuration option: broken_fixture_option\x07\n");
        let message = ssh_profile_failure("exit status: 255", &stderr);
        assert!(message.contains("exit status: 255"));
        assert!(message.contains("Bad configuration option: broken_fixture_option"));
        assert!(!message.contains('\x1b'));
        assert!(!message.contains('\x07'));
        assert!(message.len() < 1200);
        let message = ssh_profile_failure(
            "exit status: 255",
            b"Bad configuration option: broken_fixture_option\x07\n",
        );
        assert!(message.contains("Bad configuration option: broken_fixture_option"));
        assert!(!message.contains('\x07'));
        assert_eq!(
            ssh_profile_failure("exit status: 255", b"\x07\r\n"),
            "SSH configuration could not be resolved (exit status: 255)"
        );
    }

    #[test]
    fn remote_paths_have_server_semantics() {
        assert_eq!(normalize_relative(Path::new("CON.md")).unwrap(), "CON.md");
        assert_eq!(normalize_relative(Path::new("a:b.md")).unwrap(), "a:b.md");
        assert_eq!(
            normalize_relative(Path::new("a\\b/../c.md")).unwrap(),
            "a/c.md"
        );
        assert!(normalize_relative(Path::new("../../outside")).is_err());
        assert!(normalize_relative(Path::new("/outside")).is_err());
    }

    #[test]
    fn server_paths_are_component_confined() {
        assert_eq!(
            relative_under("/vault", "/vault/notes/a.md").unwrap(),
            "notes/a.md"
        );
        assert!(relative_under("/vault", "/vault-other/a.md").is_err());
        assert!(relative_under("/vault", "/vault/../outside").is_err());
        assert_eq!(relative_under("/", "/a.md").unwrap(), "a.md");
    }

    #[test]
    fn source_identity_is_delimited_and_native_independent() {
        assert_ne!(
            source_identity("a", "bc", "/root"),
            source_identity("ab", "c", "/root")
        );
        assert_ne!(
            source_identity("a", "host=one", "/root"),
            source_identity("a", "host=two", "/root")
        );
    }

    #[cfg(unix)]
    async fn local_server_fixture()
    -> Option<(SftpStorage, tempfile::TempDir, tokio::process::Child)> {
        let executable = [
            "/usr/libexec/sftp-server",
            "/usr/lib/openssh/sftp-server",
            "/usr/lib/ssh/sftp-server",
        ]
        .into_iter()
        .find(|path| Path::new(path).is_file())?;
        let root = tempfile::tempdir().unwrap();
        let canonical = root
            .path()
            .canonicalize()
            .unwrap()
            .to_str()
            .unwrap()
            .to_owned();
        let mut child = Command::new(executable)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .unwrap();
        let session = Session::from_io(
            child.stdout.take().unwrap(),
            child.stdin.take().unwrap(),
            Duration::from_secs(5),
        )
        .await
        .unwrap();
        let location = SftpLocation::parse(&format!("sftp://fixture{canonical}")).unwrap();
        let storage = SftpStorage {
            location,
            canonical: canonical.clone(),
            path: canonical.into(),
            source_id: "fixture".into(),
            effective: "fixture".into(),
            timeout: Duration::from_secs(5),
            session: AsyncMutex::new(Arc::new(session)),
            policy_revision: Mutex::new(None),
            read_budget: Semaphore::new(MAX_READ_BYTES / 1024),
            read_admission: Semaphore::new(16),
            last_failure: Mutex::new(None),
            reconciliation_owner: OnceLock::new(),
        };
        Some((storage, root, child))
    }

    #[cfg(unix)]
    #[tokio::test]
    #[ignore = "requires a local OpenSSH sftp-server executable"]
    async fn local_openssh_file_operations_and_no_clobber() {
        let (storage, root, mut child) = local_server_fixture()
            .await
            .expect("install OpenSSH sftp-server for this integration test");
        storage
            .create_file(Path::new("folder/note.md"), "first")
            .await
            .unwrap();
        assert!(matches!(
            storage
                .create_file(Path::new("folder/note.md"), "bad")
                .await,
            Err(VaultError::AlreadyExists(_))
        ));
        storage
            .write_file(Path::new("folder/note.md"), "second")
            .await
            .unwrap();
        storage
            .write_file_if_revision(
                Path::new("folder/note.md"),
                "second third",
                &crate::vault::source::source_revision("second"),
            )
            .await
            .unwrap();
        assert_eq!(
            storage
                .read_file(Path::new("folder/note.md"))
                .await
                .unwrap(),
            "second third"
        );
        storage
            .create_file(Path::new("other.md"), "other")
            .await
            .unwrap();
        assert!(
            storage
                .move_file(Path::new("folder/note.md"), Path::new("other.md"))
                .await
                .is_err()
        );
        assert_eq!(
            std::fs::read_to_string(root.path().join("other.md")).unwrap(),
            "other"
        );
        storage
            .move_file(Path::new("folder/note.md"), Path::new("moved.md"))
            .await
            .unwrap();
        storage.delete_file(Path::new("moved.md")).await.unwrap();
        assert!(!root.path().join("moved.md").exists());
        assert!(
            !storage
                .entries(Path::new(""), true)
                .await
                .unwrap()
                .iter()
                .any(|e| e.relative.to_string_lossy().contains("upload"))
        );
        child.kill().await.unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    #[ignore = "requires a local OpenSSH sftp-server executable"]
    async fn missing_portable_policy_in_existing_metadata_directory_is_valid() {
        let (storage, root, mut child) = local_server_fixture()
            .await
            .expect("install OpenSSH sftp-server for this integration test");
        std::fs::create_dir(root.path().join(".obsidian-mcp")).unwrap();
        std::fs::write(root.path().join("note.md"), "available").unwrap();
        storage.pin_policy_revision(None);
        let session = Arc::clone(&*storage.session.lock().await);
        storage.validate_policy_revision(&session).await.unwrap();
        assert_eq!(
            storage.read_file(Path::new("note.md")).await.unwrap(),
            "available"
        );
        storage.pin_policy_revision(Some(crate::vault::source::source_revision(
            "previous policy",
        )));
        assert!(matches!(
            storage.validate_policy_revision(&session).await,
            Err(VaultError::Storage {
                kind: StorageErrorKind::IdentityChanged,
                ..
            })
        ));
        child.kill().await.unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    #[ignore = "requires a local OpenSSH sftp-server executable"]
    async fn observed_reads_do_not_adopt_a_unicode_alias() {
        let (storage, root, mut child) = local_server_fixture()
            .await
            .expect("install OpenSSH sftp-server for this integration test");
        let allowed = "caf\u{e9}.md";
        let other = "cafe\u{301}.md";
        std::fs::write(root.path().join(allowed), "allowed").unwrap();
        let observed = storage
            .entries(Path::new(""), true)
            .await
            .unwrap()
            .remove(0)
            .relative;
        std::fs::remove_file(root.path().join(allowed)).unwrap();
        std::fs::write(root.path().join(other), "excluded-replacement").unwrap();
        if observed.to_str().unwrap() == other {
            // A server volume may expose only one canonical Unicode spelling.
            // Exercise the exact-name distinction with a case variant instead.
            std::fs::remove_file(root.path().join(other)).unwrap();
            std::fs::write(root.path().join("other.md"), "replacement").unwrap();
        }
        assert!(storage.read_exact_bounded(&observed, 4096).await.is_err());
        let batch = storage
            .observe_batch(std::slice::from_ref(&observed))
            .await
            .unwrap();
        assert!(batch.read(&observed, 4096).await.is_err());
        assert!(matches!(
            batch.read(Path::new("unrequested.md"), 4096).await,
            Err(VaultError::InvalidPath(_))
        ));
        child.kill().await.unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    #[ignore = "requires a local OpenSSH sftp-server executable"]
    async fn native_case_matching_returns_the_discovered_source_spelling() {
        let (storage, root, mut child) = local_server_fixture()
            .await
            .expect("install OpenSSH sftp-server for this integration test");
        std::fs::write(root.path().join("Existing.md"), "before").unwrap();
        if root.path().join("existing.md").is_file() {
            let written = storage
                .write_file(Path::new("existing.md"), "after")
                .await
                .unwrap();
            assert_eq!(written.as_os_str(), "Existing.md");
            let listed = storage.entries(Path::new(""), true).await.unwrap();
            assert_eq!(listed.len(), 1);
            assert_eq!(listed[0].relative.as_os_str(), written.as_os_str());
            assert_eq!(
                std::fs::read_to_string(root.path().join("Existing.md")).unwrap(),
                "after"
            );
        }
        child.kill().await.unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    #[ignore = "requires a local OpenSSH sftp-server executable"]
    async fn confined_aliases_limits_and_private_replacement() {
        use std::os::unix::fs::{PermissionsExt, symlink};
        let (storage, root, mut child) = local_server_fixture()
            .await
            .expect("install OpenSSH sftp-server for this integration test");
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("secret.md"), "outside").unwrap();
        symlink(outside.path(), root.path().join("escape")).unwrap();
        assert!(matches!(
            storage.read_file(Path::new("escape/secret.md")).await,
            Err(VaultError::OutsideVault(_))
        ));
        assert!(
            storage
                .entries(Path::new(""), true)
                .await
                .unwrap()
                .is_empty()
        );
        storage
            .create_file(Path::new("private.md"), "one")
            .await
            .unwrap();
        assert_eq!(
            std::fs::metadata(root.path().join("private.md"))
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
        symlink("private.md", root.path().join("alias.md")).unwrap();
        storage
            .write_file(Path::new("alias.md"), "second")
            .await
            .unwrap();
        assert_eq!(
            std::fs::read_to_string(root.path().join("private.md")).unwrap(),
            "second"
        );
        assert_eq!(
            std::fs::metadata(root.path().join("private.md"))
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
        assert!(matches!(
            storage.read_file_bounded(Path::new("private.md"), 2).await,
            Err(VaultError::SemanticResourceLimit { limit: 2, .. })
        ));
        storage.delete_file(Path::new("alias.md")).await.unwrap();
        assert!(root.path().join("private.md").is_file());
        child.kill().await.unwrap();
    }
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(super) enum Fault {
        Normal,
        NoPosix,
        Collision,
        DenyAttrs,
        Conflict,
        HoldOpen,
        HoldWrite,
        HoldRename,
        SparseAttrs,
        UnrepresentableNames,
        EscapingName,
        AbsoluteName,
        EmptyName,
        NulName,
        UnicodeDestination,
        UnicodeAliasDestination,
        NonemptyDirectory,
        StatFailure,
        ContainedAliases,
        EscapingAlias,
        MissingPolicy,
        DeniedPolicy,
        EscapingPolicy,
        InvalidUtf8Policy,
        OversizedPolicy,
        DirectoryPolicy,
        MissingChild,
        MissingReadDir,
        DeniedChild,
        FailedReadDir,
        DisconnectWrite,
        ParallelDeniedChild,
        ParallelProtocolChild,
        GrowingRead,
        UnknownReadSize,
        RenameCleanupFailure,
        AttrsCleanupFailure,
        ConflictCleanupFailure,
        HeldRead,
    }

    #[derive(Default)]
    pub(super) struct FakeFiles {
        pub(super) target: Vec<u8>,
        temporary: Option<Vec<u8>>,
        removed: usize,
        renames: usize,
        writes: usize,
        pub(super) listed: HashMap<String, usize>,
        pub(super) read_paths: Vec<String>,
        pub(super) opened_directories: usize,
        fstats: usize,
        reads: usize,
        closed_handles: usize,
        pub(super) paths: Vec<String>,
        temporary_path: Option<String>,
        temporary_mode: Option<u32>,
        parallel_rejected: bool,
        parallel_release: Arc<tokio::sync::Notify>,
        read_blocked: bool,
        read_release: Arc<tokio::sync::Notify>,
    }

    fn fake_string(out: &mut Vec<u8>, value: &[u8]) {
        out.extend_from_slice(&(value.len() as u32).to_be_bytes());
        out.extend_from_slice(value);
    }

    fn fake_attrs(out: &mut Vec<u8>, size: usize, directory: bool) {
        out.extend_from_slice(&5u32.to_be_bytes());
        out.extend_from_slice(&(size as u64).to_be_bytes());
        out.extend_from_slice(&(if directory { 0o040700u32 } else { 0o100600u32 }).to_be_bytes());
    }

    fn fake_path(packet: &[u8]) -> String {
        let len = u32::from_be_bytes(packet[5..9].try_into().unwrap()) as usize;
        String::from_utf8(packet[9..9 + len].to_vec()).unwrap()
    }

    pub(super) async fn fake_storage(
        fault: Fault,
    ) -> (
        SftpStorage,
        Arc<Mutex<FakeFiles>>,
        tokio::task::JoinHandle<()>,
    ) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let files = Arc::new(Mutex::new(FakeFiles {
            target: match fault {
                Fault::InvalidUtf8Policy => vec![0xff],
                Fault::OversizedPolicy => vec![b'x'; 256 * 1024 + 1],
                _ => b"old".to_vec(),
            },
            ..Default::default()
        }));
        let server_files = Arc::clone(&files);
        let (client, mut server) = tokio::io::duplex(131072);
        let task = tokio::spawn(async move {
            let hello_len = server.read_u32().await.unwrap();
            let mut hello = vec![0; hello_len as usize];
            server.read_exact(&mut hello).await.unwrap();
            let mut version = vec![2, 0, 0, 0, 3];
            if fault != Fault::NoPosix {
                fake_string(&mut version, b"posix-rename@openssh.com");
                fake_string(&mut version, b"1");
            }
            server.write_u32(version.len() as u32).await.unwrap();
            server.write_all(&version).await.unwrap();
            let mut held_directory = None;
            let mut denied_directory = None;
            let mut parallel_done = false;
            let mut read_released = false;
            while let Ok(len) = server.read_u32().await {
                let mut request = vec![0; len as usize];
                if server.read_exact(&mut request).await.is_err() {
                    break;
                }
                let mut response = vec![101];
                response.extend_from_slice(&request[1..5]);
                let mut status = 0u32;
                let mut hold = false;
                let mut disconnect = false;
                let mut deferred = 0;
                {
                    let mut files = server_files.lock().unwrap();
                    match request[0] {
                        16 => {
                            let path = fake_path(&request);
                            files.paths.push(path.clone());
                            let canonical = match (fault, path.as_str()) {
                                (Fault::ContainedAliases, "/vault/alias.md") => "/vault/note.md",
                                (Fault::ContainedAliases, "/vault/alias-dir") => "/vault",
                                (Fault::EscapingAlias, "/vault/alias.md") => "/outside/secret.md",
                                (Fault::EscapingPolicy, "/vault/.obsidian-mcp/ignore") => {
                                    "/outside/ignore"
                                }
                                _ => &path,
                            };
                            response[0] = 104;
                            response.extend_from_slice(&1u32.to_be_bytes());
                            fake_string(&mut response, canonical.as_bytes());
                            fake_string(&mut response, b"");
                            response.extend_from_slice(&0u32.to_be_bytes());
                        }
                        7 | 17 => {
                            let path = fake_path(&request);
                            files.paths.push(path.clone());
                            if fault == Fault::StatFailure && path == "/vault/failing.md" {
                                status = 4;
                            } else if path == "/vault"
                                || path == "/vault/note.md"
                                || (matches!(
                                    fault,
                                    Fault::UnicodeDestination | Fault::UnicodeAliasDestination
                                ) && path == "/vault/cafe\u{301}.md")
                                || (fault == Fault::NonemptyDirectory && path == "/vault/folder")
                                || (matches!(fault, Fault::ContainedAliases | Fault::EscapingAlias)
                                    && (path == "/vault/alias.md" || path == "/vault/alias-dir"))
                                || (fault == Fault::SparseAttrs
                                    && (path == "/vault/nested" || path == "/vault/nested/note.md"))
                                || (matches!(fault, Fault::MissingPolicy | Fault::DeniedPolicy)
                                    && path == "/vault/.obsidian-mcp")
                                || (matches!(
                                    fault,
                                    Fault::MissingChild
                                        | Fault::MissingReadDir
                                        | Fault::DeniedChild
                                        | Fault::FailedReadDir
                                        | Fault::ParallelDeniedChild
                                        | Fault::ParallelProtocolChild
                                ) && matches!(
                                    path.as_str(),
                                    "/vault/kept" | "/vault/gone" | "/vault/kept/note.md"
                                ))
                                || (path.contains("upload-") && files.temporary.is_some())
                            {
                                response[0] = 105;
                                fake_attrs(
                                    &mut response,
                                    files.target.len(),
                                    path == "/vault"
                                        || path == "/vault/nested"
                                        || path == "/vault/folder"
                                        || path == "/vault/.obsidian-mcp"
                                        || path == "/vault/kept"
                                        || path == "/vault/gone",
                                );
                                if (fault == Fault::UnicodeAliasDestination
                                    && path == "/vault/cafe\u{301}.md")
                                    || (matches!(
                                        fault,
                                        Fault::ContainedAliases | Fault::EscapingAlias
                                    ) && (path == "/vault/alias.md"
                                        || path == "/vault/alias-dir"))
                                {
                                    let end = response.len();
                                    response[end - 4..].copy_from_slice(&0o120777u32.to_be_bytes());
                                }
                            } else {
                                status = 2;
                            }
                        }
                        11 => {
                            files.opened_directories += 1;
                            let path = fake_path(&request);
                            if matches!(
                                fault,
                                Fault::ParallelDeniedChild | Fault::ParallelProtocolChild
                            ) && !parallel_done
                                && path == "/vault/gone"
                            {
                                status = 3;
                                deferred = 2;
                            } else if path == "/vault/gone"
                                && matches!(fault, Fault::MissingChild | Fault::DeniedChild)
                            {
                                status = if fault == Fault::MissingChild { 2 } else { 3 };
                            } else {
                                files.listed.insert(path.clone(), 0);
                                response[0] = 102;
                                fake_string(&mut response, path.as_bytes());
                                if matches!(
                                    fault,
                                    Fault::ParallelDeniedChild | Fault::ParallelProtocolChild
                                ) && !parallel_done
                                    && path == "/vault/kept"
                                {
                                    deferred = 1;
                                }
                            }
                        }
                        12 => {
                            let path = fake_path(&request);
                            let page = files.listed.entry(path.clone()).or_default();
                            let current_page = *page;
                            *page += 1;
                            let names: &[&[u8]] = match (fault, current_page) {
                                (Fault::UnrepresentableNames, 0) => &[b"latin-\xff.md"],
                                (Fault::UnrepresentableNames, 1) => {
                                    &[b"literal\\name.md", b".Icon\r", b"note.md"]
                                }
                                (Fault::EscapingName, 0) => &[b"../outside-\xff.md"],
                                (Fault::AbsoluteName, 0) => &[b"/outside.md"],
                                (Fault::EmptyName, 0) => &[b""],
                                (Fault::NulName, 0) => &[b"nul\0.md"],
                                (Fault::UnicodeDestination | Fault::UnicodeAliasDestination, 0) => {
                                    &[b"note.md", "cafe\u{301}.md".as_bytes()]
                                }
                                (Fault::NonemptyDirectory, 0) => &[b"note.md", b"folder"],
                                (Fault::ContainedAliases, 0) => {
                                    &[b"note.md", b"alias.md", b"alias-dir"]
                                }
                                (Fault::EscapingAlias, 0) => &[b"note.md", b"alias.md"],
                                (Fault::SparseAttrs, 0) if path == "/vault" => &[b"nested"],
                                (
                                    Fault::MissingChild
                                    | Fault::MissingReadDir
                                    | Fault::DeniedChild
                                    | Fault::FailedReadDir
                                    | Fault::ParallelDeniedChild
                                    | Fault::ParallelProtocolChild,
                                    0,
                                ) if path == "/vault" => &[b"kept", b"gone"],
                                (Fault::MissingReadDir | Fault::FailedReadDir, _)
                                    if path == "/vault/gone" =>
                                {
                                    &[]
                                }
                                (_, 0) => &[b"note.md"],
                                _ => &[],
                            };
                            if !names.is_empty() {
                                response[0] = 104;
                                response.extend_from_slice(&(names.len() as u32).to_be_bytes());
                                for name in names {
                                    fake_string(&mut response, name);
                                    fake_string(&mut response, b"");
                                    response.extend_from_slice(&0u32.to_be_bytes());
                                }
                            } else {
                                status = if path == "/vault/gone" && fault == Fault::MissingReadDir
                                {
                                    2
                                } else if path == "/vault/gone" && fault == Fault::FailedReadDir {
                                    4
                                } else {
                                    1
                                };
                            }
                        }
                        3 => {
                            let path = fake_path(&request);
                            if path == "/vault/.obsidian-mcp/ignore"
                                && matches!(fault, Fault::MissingPolicy | Fault::DeniedPolicy)
                            {
                                status = if fault == Fault::MissingPolicy { 2 } else { 3 };
                            } else if path.contains("upload-") {
                                files.temporary_path = Some(path.clone());
                                let permissions = 17 + path.len();
                                files.temporary_mode = Some(u32::from_be_bytes(
                                    request[permissions..permissions + 4].try_into().unwrap(),
                                ));
                                if fault == Fault::Collision {
                                    files.temporary = Some(b"unowned".to_vec());
                                    status = 4;
                                } else {
                                    files.temporary = Some(Vec::new());
                                    response[0] = 102;
                                    fake_string(&mut response, &[0xfe, 0]);
                                    hold = fault == Fault::HoldOpen;
                                }
                            } else {
                                files.read_paths.push(path);
                                response[0] = 102;
                                fake_string(&mut response, &[0xff, 0]);
                            }
                        }
                        8 => {
                            files.fstats += 1;
                            response[0] = 105;
                            if fault == Fault::UnknownReadSize {
                                response.extend_from_slice(&4u32.to_be_bytes());
                                response.extend_from_slice(&0o100600u32.to_be_bytes());
                            } else {
                                fake_attrs(
                                    &mut response,
                                    if fault == Fault::GrowingRead {
                                        1
                                    } else {
                                        files.target.len()
                                    },
                                    fault == Fault::DirectoryPolicy,
                                );
                            }
                        }
                        5 => {
                            files.reads += 1;
                            let offset =
                                u64::from_be_bytes(request[11..19].try_into().unwrap()) as usize;
                            let len =
                                u32::from_be_bytes(request[19..23].try_into().unwrap()) as usize;
                            if offset >= files.target.len() {
                                status = 1;
                            } else {
                                response[0] = 103;
                                fake_string(
                                    &mut response,
                                    &files.target[offset..files.target.len().min(offset + len)],
                                );
                            }
                        }
                        6 => {
                            files.writes += 1;
                            let len =
                                u32::from_be_bytes(request[19..23].try_into().unwrap()) as usize;
                            files
                                .temporary
                                .as_mut()
                                .unwrap()
                                .extend_from_slice(&request[23..23 + len]);
                            if matches!(fault, Fault::Conflict | Fault::ConflictCleanupFailure) {
                                files.target = b"external edit".to_vec();
                            }
                            if fault == Fault::DisconnectWrite {
                                files.temporary = Some(b"unowned replacement".to_vec());
                                disconnect = true;
                            }
                            hold = fault == Fault::HoldWrite;
                        }
                        10 if matches!(fault, Fault::DenyAttrs | Fault::AttrsCleanupFailure) => {
                            status = 3
                        }
                        13 => {
                            files.removed += 1;
                            if matches!(
                                fault,
                                Fault::RenameCleanupFailure
                                    | Fault::AttrsCleanupFailure
                                    | Fault::ConflictCleanupFailure
                            ) {
                                status = 3;
                            } else {
                                files.temporary = None;
                            }
                        }
                        15 if fault == Fault::NonemptyDirectory => status = 4,
                        18 | 200 => {
                            files.renames += 1;
                            if fault == Fault::RenameCleanupFailure {
                                status = 4;
                            } else {
                                files.target = files.temporary.take().unwrap();
                            }
                            hold = fault == Fault::HoldRename;
                        }
                        4 => files.closed_handles += 1,
                        10 => (),
                        other => panic!("unexpected fake SFTP request {other}"),
                    }
                }
                if disconnect {
                    break;
                }
                if hold {
                    std::future::pending::<()>().await;
                }
                if response[0] == 101 {
                    response.extend_from_slice(&status.to_be_bytes());
                    fake_string(&mut response, b"injected server status");
                    fake_string(&mut response, b"");
                }
                if fault == Fault::HeldRead && request[0] == 5 && !read_released {
                    let release = {
                        let mut files = server_files.lock().unwrap();
                        files.read_blocked = true;
                        Arc::clone(&files.read_release)
                    };
                    release.notified().await;
                    read_released = true;
                }
                if deferred != 0 {
                    if deferred == 1 {
                        held_directory = Some(response);
                    } else {
                        denied_directory = Some(response);
                    }
                    if held_directory.is_none() || denied_directory.is_none() {
                        continue;
                    }
                    let denied = denied_directory.take().unwrap();
                    server.write_u32(denied.len() as u32).await.unwrap();
                    server.write_all(&denied).await.unwrap();
                    let release = {
                        let mut files = server_files.lock().unwrap();
                        files.parallel_rejected = true;
                        Arc::clone(&files.parallel_release)
                    };
                    release.notified().await;
                    parallel_done = true;
                    response = held_directory.take().unwrap();
                    if fault == Fault::ParallelProtocolChild {
                        response[0] = 104;
                        response.truncate(5);
                        response.extend_from_slice(&u32::MAX.to_be_bytes());
                    }
                }
                if server.write_u32(response.len() as u32).await.is_err() {
                    break;
                }
                if server.write_all(&response).await.is_err() {
                    break;
                }
            }
        });
        let (reader, writer) = tokio::io::split(client);
        let session = Session::from_io(reader, writer, Duration::from_secs(2))
            .await
            .unwrap();
        let storage = SftpStorage {
            location: SftpLocation::parse("sftp://fixture/vault").unwrap(),
            canonical: "/vault".into(),
            path: PathBuf::from("/vault"),
            source_id: "fixture".into(),
            effective: "fixture".into(),
            timeout: Duration::from_millis(200),
            session: AsyncMutex::new(Arc::new(session)),
            policy_revision: Mutex::new(None),
            read_budget: Semaphore::new(MAX_READ_BYTES / 1024),
            read_admission: Semaphore::new(16),
            last_failure: Mutex::new(None),
            reconciliation_owner: OnceLock::new(),
        };
        (storage, files, task)
    }

    #[tokio::test]
    async fn identity_withdrawal_survives_late_operation_results() {
        let (storage, _, server) = fake_storage(Fault::Normal).await;
        storage.record_result::<()>(&Err(storage_error(
            "reconnect",
            "",
            StorageErrorKind::IdentityChanged,
            "portable policy changed",
        )));
        let withdrawn = storage.last_failure().unwrap();
        for kind in [StorageErrorKind::Unavailable, StorageErrorKind::Protocol] {
            storage.record_result::<()>(&Err(storage_error(
                "read",
                "note.md",
                kind,
                "late failure",
            )));
            assert_eq!(storage.last_failure().as_ref(), Some(&withdrawn));
        }
        storage.record_result(&Ok(()));
        assert_eq!(storage.last_failure().as_ref(), Some(&withdrawn));
        assert!(matches!(
            storage.read_file(Path::new("note.md")).await,
            Err(VaultError::Storage {
                kind: StorageErrorKind::IdentityChanged,
                ..
            })
        ));
        server.abort();
    }

    #[tokio::test]
    async fn reconnect_notification_wakes_without_withdrawing_source_authority() {
        let (storage, _, server) = fake_storage(Fault::Normal).await;
        let wakeup = Arc::new(tokio::sync::Notify::new());
        storage
            .set_reconciliation_owner(Arc::clone(&wakeup))
            .unwrap();
        storage.notify_reconnected();
        tokio::time::timeout(Duration::from_millis(100), wakeup.notified())
            .await
            .unwrap();
        assert!(storage.last_failure().is_none());
        assert!(storage.session.lock().await.is_healthy());
        server.abort();
    }

    #[tokio::test]
    async fn reconnect_policy_check_treats_missing_read_as_absence() {
        let (storage, _, server) = fake_storage(Fault::MissingPolicy).await;
        let session = Arc::clone(&*storage.session.lock().await);
        storage.pin_policy_revision(None);
        storage.validate_policy_revision(&session).await.unwrap();
        assert_eq!(
            storage.read_file(Path::new("note.md")).await.unwrap(),
            "old"
        );
        storage.pin_policy_revision(Some(crate::vault::source::source_revision(
            "previous policy",
        )));
        assert!(matches!(
            storage.validate_policy_revision(&session).await,
            Err(VaultError::Storage {
                kind: StorageErrorKind::IdentityChanged,
                ..
            })
        ));
        server.abort();

        let (storage, _, server) = fake_storage(Fault::DeniedPolicy).await;
        let session = Arc::clone(&*storage.session.lock().await);
        storage.pin_policy_revision(None);
        assert!(matches!(
            storage.validate_policy_revision(&session).await,
            Err(VaultError::Storage {
                kind: StorageErrorKind::PermissionDenied,
                ..
            })
        ));
        server.abort();
    }

    #[tokio::test]
    async fn escaped_portable_policy_withdraws_source_authority_without_reading_it() {
        let (storage, files, server) = fake_storage(Fault::EscapingPolicy).await;
        let session = Arc::clone(&*storage.session.lock().await);
        storage.pin_policy_revision(None);
        let error = storage
            .operation(
                "reconnect",
                Path::new(""),
                storage.validate_policy_revision(&session),
            )
            .await
            .unwrap_err();
        assert!(
            matches!(
                error,
                VaultError::Storage {
                    kind: StorageErrorKind::IdentityChanged,
                    ..
                }
            ),
            "{error}"
        );
        let withdrawn = storage.last_failure().unwrap();
        assert_eq!(withdrawn.0, StorageErrorKind::IdentityChanged);
        storage.record_result(&Ok(()));
        assert_eq!(storage.last_failure().as_ref(), Some(&withdrawn));
        assert!(files.lock().unwrap().read_paths.is_empty());
        server.abort();
    }

    #[tokio::test]
    async fn invalid_portable_policy_withdraws_source_authority_with_its_cause() {
        for (fault, cause) in [
            (Fault::InvalidUtf8Policy, "remote note is not UTF-8"),
            (Fault::OversizedPolicy, "maximum 262144"),
            (Fault::DirectoryPolicy, "unexpected remote file type"),
        ] {
            let (storage, _, server) = fake_storage(fault).await;
            let session = Arc::clone(&*storage.session.lock().await);
            storage.pin_policy_revision(None);
            let error = storage
                .operation(
                    "reconnect",
                    Path::new(""),
                    storage.validate_policy_revision(&session),
                )
                .await
                .unwrap_err();
            assert!(
                matches!(
                    error,
                    VaultError::Storage {
                        kind: StorageErrorKind::IdentityChanged,
                        ..
                    }
                ),
                "{fault:?}: {error}"
            );
            assert!(error.to_string().contains(cause), "{fault:?}: {error}");
            assert_eq!(
                storage.last_failure().unwrap().0,
                StorageErrorKind::IdentityChanged
            );
            server.abort();
        }
    }

    #[test]
    fn unknown_policy_read_failures_preserve_their_classification() {
        for kind in [
            StorageErrorKind::PermissionDenied,
            StorageErrorKind::Unavailable,
            StorageErrorKind::RemoteFailure,
            StorageErrorKind::Protocol,
            StorageErrorKind::Unsupported,
            StorageErrorKind::ResourceLimit,
        ] {
            let error = policy_validation_error(storage_error(
                "read",
                ".obsidian-mcp/ignore",
                kind,
                "original failure",
            ));
            assert!(matches!(error, VaultError::Storage { kind: actual, .. } if actual == kind));
            assert!(error.to_string().contains("original failure"));
        }
    }

    #[tokio::test]
    async fn parallel_directory_errors_settle_siblings_and_preserve_shared_failures() {
        for fault in [Fault::ParallelDeniedChild, Fault::ParallelProtocolChild] {
            for observe in [false, true] {
                let (storage, files, server) = fake_storage(fault).await;
                let storage = Arc::new(storage);
                let session = Arc::clone(&*storage.session.lock().await);
                let release = Arc::clone(&files.lock().unwrap().parallel_release);
                let source = Arc::clone(&storage);
                let mut request = tokio::spawn(async move {
                    if observe {
                        source
                            .observe_batch(&[
                                PathBuf::from("kept/note.md"),
                                PathBuf::from("gone/note.md"),
                            ])
                            .await
                            .map(|_| ())
                    } else {
                        source.entries(Path::new(""), true).await.map(|_| ())
                    }
                });
                tokio::time::timeout(Duration::from_secs(1), async {
                    loop {
                        if files.lock().unwrap().parallel_rejected {
                            break;
                        }
                        tokio::task::yield_now().await;
                    }
                })
                .await
                .unwrap();
                assert!(
                    tokio::time::timeout(Duration::from_millis(30), &mut request)
                        .await
                        .is_err()
                );
                assert!(session.is_healthy());
                release.notify_one();
                let error = request.await.unwrap().unwrap_err();
                assert!(
                    matches!(
                        error,
                        VaultError::Storage {
                            kind: StorageErrorKind::PermissionDenied,
                            ..
                        }
                    ),
                    "{error}"
                );
                if fault == Fault::ParallelProtocolChild {
                    assert!(!session.is_healthy());
                    assert_eq!(
                        storage.last_failure().unwrap().0,
                        StorageErrorKind::Protocol
                    );
                } else {
                    assert!(session.is_healthy());
                    assert!(storage.last_failure().is_none());
                    assert_eq!(
                        storage.read_file(Path::new("kept/note.md")).await.unwrap(),
                        "old"
                    );
                    assert!(Arc::ptr_eq(&session, &*storage.session.lock().await));
                }
                server.abort();
            }
        }
    }

    #[tokio::test]
    async fn missing_walk_child_is_absent_but_missing_start_and_mid_read_failures_propagate() {
        let (storage, _, server) = fake_storage(Fault::MissingChild).await;
        let entries = storage.entries(Path::new(""), true).await.unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].relative, Path::new("kept/note.md"));
        assert!(matches!(
            storage.entries(Path::new("gone"), true).await,
            Err(VaultError::DirectoryNotFound(_))
        ));
        server.abort();

        let (storage, _, server) = fake_storage(Fault::MissingReadDir).await;
        assert!(matches!(
            storage.entries(Path::new(""), true).await,
            Err(VaultError::NoteNotFound(_))
        ));
        assert!(storage.session.lock().await.is_healthy());
        server.abort();
    }

    #[tokio::test]
    #[cfg(has_embeddings)]
    async fn semantic_context_cancellation_and_deadline_leave_launched_reads_healthy() {
        use crate::models::{EvidenceScope, SemanticEvidence, SourceRange, SourceStatus};
        use crate::vault::embeddings::{ChunkEmbedding, NoteEmbeddingEntry, prepared_text_hash};
        use crate::vault::semantic_results::{RankedNote, render_many_source};

        for cancel_caller in [false, true] {
            let (mut storage, files, server) = fake_storage(Fault::HeldRead).await;
            storage.timeout = Duration::from_secs(2);
            let session = Arc::clone(&*storage.session.lock().await);
            let release = Arc::clone(&files.lock().unwrap().read_release);
            let storage = crate::vault::storage::Storage::Sftp(Arc::new(storage));
            let range = SourceRange {
                start_byte: 0,
                end_byte: 3,
            };
            let entry = Arc::new(NoteEmbeddingEntry {
                source_revision: crate::vault::source::source_revision("old"),
                source_bytes: 3,
                body_range: range,
                title: "note".into(),
                tags: Vec::new(),
                chunks: vec![ChunkEmbedding {
                    input_hash: prepared_text_hash("old"),
                    vector: vec![1.0],
                    scope: EvidenceScope::Note,
                    range,
                    overlap_range: None,
                    heading_path: Vec::new(),
                    section_range: None,
                    block_kind: None,
                }],
            });
            let hit = RankedNote {
                path: PathBuf::from("note.md"),
                score: 1.0,
                entry,
                evidence: SemanticEvidence {
                    scope: EvidenceScope::Note,
                    cosine_similarity: 1.0,
                    range,
                    heading_path: Vec::new(),
                    section_range: None,
                },
            };
            if cancel_caller {
                let owned_storage = storage.clone();
                let request = tokio::spawn(async move {
                    render_many_source(&owned_storage, vec![hit; 5], "old", true, None).await
                });
                tokio::time::timeout(Duration::from_secs(1), async {
                    loop {
                        if files.lock().unwrap().read_blocked {
                            break;
                        }
                        tokio::task::yield_now().await;
                    }
                })
                .await
                .unwrap();
                request.abort();
                assert!(request.await.unwrap_err().is_cancelled());
            } else {
                let results = tokio::time::timeout(
                    Duration::from_secs(1),
                    render_many_source(
                        &storage,
                        vec![hit; 5],
                        "old",
                        true,
                        Some(Duration::from_millis(50)),
                    ),
                )
                .await
                .unwrap()
                .unwrap();
                assert_eq!(results.len(), 5);
                assert!(results.iter().all(|result| {
                    result.source_status == SourceStatus::Unavailable
                        && result
                            .context_error
                            .as_ref()
                            .is_some_and(|error| error.message.contains("deadline"))
                }));
            }
            assert!(session.is_healthy());
            assert!(files.lock().unwrap().read_blocked);
            release.notify_one();
            tokio::time::timeout(Duration::from_secs(1), async {
                loop {
                    let drained = {
                        let files = files.lock().unwrap();
                        assert!(files.read_paths.len() <= 4);
                        files.read_paths.len() == 4 && files.closed_handles == 5
                    };
                    if drained {
                        break;
                    }
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();
            assert!(session.is_healthy());
            assert!(files.lock().unwrap().read_paths.len() <= 4);
            server.abort();
        }
    }

    #[tokio::test]
    async fn known_oversized_read_closes_without_reserving_content_budget_or_reading() {
        let (storage, files, server) = fake_storage(Fault::Normal).await;
        let session = Arc::clone(&*storage.session.lock().await);
        let _budget = storage
            .read_budget
            .acquire_many((MAX_READ_BYTES / 1024) as u32)
            .await
            .unwrap();
        let error = storage
            .operation(
                "read",
                Path::new("note.md"),
                read_source_with_stat(
                    &session,
                    "/vault/note.md",
                    2,
                    &storage.read_budget,
                    &storage.read_admission,
                ),
            )
            .await
            .unwrap_err();
        assert!(
            matches!(
                error,
                VaultError::SemanticResourceLimit {
                    actual: 3,
                    limit: 2,
                    ..
                }
            ),
            "{error}"
        );
        let files = files.lock().unwrap();
        assert_eq!(files.reads, 0);
        assert_eq!(files.closed_handles, 1);
        assert!(session.is_healthy());
        server.abort();
    }

    #[tokio::test]
    async fn read_limit_still_checks_content_after_growth_or_unknown_size() {
        for fault in [Fault::GrowingRead, Fault::UnknownReadSize] {
            let (storage, files, server) = fake_storage(fault).await;
            let session = Arc::clone(&*storage.session.lock().await);
            let error = read_source_with_stat(
                &session,
                "/vault/note.md",
                2,
                &storage.read_budget,
                &storage.read_admission,
            )
            .await
            .unwrap_err();
            assert!(
                matches!(
                    error,
                    VaultError::SemanticResourceLimit {
                        actual: 3,
                        limit: 2,
                        ..
                    }
                ),
                "{fault:?}: {error}"
            );
            let files = files.lock().unwrap();
            assert_eq!(files.reads, 1, "{fault:?}");
            assert_eq!(files.closed_handles, 1, "{fault:?}");
            assert!(session.is_healthy(), "{fault:?}");
            server.abort();
        }
    }

    #[tokio::test]
    async fn cancelling_a_budget_wait_retires_the_session_with_an_open_handle() {
        let (storage, files, server) = fake_storage(Fault::Normal).await;
        let budget = storage
            .read_budget
            .acquire_many((MAX_READ_BYTES / 1024) as u32)
            .await
            .unwrap();
        let session = Arc::clone(&*storage.session.lock().await);
        let mut read = Box::pin(read_source_with_stat(
            &session,
            "/vault/note.md",
            MAX_READ_BYTES,
            &storage.read_budget,
            &storage.read_admission,
        ));
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                assert!(futures_util::poll!(read.as_mut()).is_pending());
                let stat_requested = files.lock().unwrap().fstats > 0;
                if stat_requested {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        // This later reply proves the earlier FSTAT reply has been decoded.
        // The next explicit poll reaches the read-budget wait with no RPC pending.
        session.stat("/vault/note.md").await.unwrap();
        assert!(futures_util::poll!(read.as_mut()).is_pending());
        assert!(session.is_healthy());
        drop(read);
        assert!(!session.is_healthy());
        assert_eq!(
            storage.last_failure().unwrap().0,
            StorageErrorKind::Unavailable
        );
        tokio::time::timeout(Duration::from_secs(1), server)
            .await
            .unwrap()
            .unwrap();
        drop(budget);
    }

    #[tokio::test]
    async fn admission_timeout_does_not_withdraw_a_healthy_source() {
        let (storage, _, server) = fake_storage(Fault::Normal).await;
        let admission = storage.read_admission.acquire_many(16).await.unwrap();
        let error = storage.read_file(Path::new("note.md")).await.unwrap_err();
        assert!(matches!(
            error,
            VaultError::Storage {
                kind: StorageErrorKind::Unavailable,
                ..
            }
        ));
        assert!(storage.session.lock().await.is_healthy());
        assert!(storage.last_failure().is_none());
        drop(admission);

        storage.record_result::<()>(&Err(storage_error(
            "read",
            "note.md",
            StorageErrorKind::Protocol,
            "invalid response",
        )));
        storage.record_result::<()>(&Err(storage_error(
            "read",
            "note.md",
            StorageErrorKind::Unavailable,
            "queued request expired",
        )));
        assert_eq!(
            storage.last_failure().unwrap().0,
            StorageErrorKind::Protocol
        );
        server.abort();
    }

    #[tokio::test]
    async fn passive_source_health_distinguishes_a_busy_lock_from_a_dead_session() {
        let (storage, _, server) = fake_storage(Fault::Normal).await;
        let session = storage.session.lock().await;
        storage.record_result::<()>(&Err(storage_error(
            "read",
            "note.md",
            StorageErrorKind::Unavailable,
            "queued request expired",
        )));
        assert!(storage.last_failure().is_some());
        drop(session);
        assert!(storage.last_failure().is_none());

        server.abort();
        let _ = server.await;
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if !storage.session.lock().await.is_healthy() {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        storage.record_result::<()>(&Err(storage_error(
            "read",
            "note.md",
            StorageErrorKind::Unavailable,
            "connection closed",
        )));
        assert_eq!(
            storage.last_failure().unwrap().0,
            StorageErrorKind::Unavailable
        );
        storage.record_result::<()>(&Err(storage_error(
            "read",
            "note.md",
            StorageErrorKind::Protocol,
            "invalid response",
        )));
        storage.record_result::<()>(&Err(storage_error(
            "read",
            "note.md",
            StorageErrorKind::Unavailable,
            "connection is still closed",
        )));
        assert_eq!(
            storage.last_failure().unwrap().0,
            StorageErrorKind::Protocol
        );
    }

    #[tokio::test]
    async fn missing_observed_child_does_not_hide_unrelated_notes() {
        let (storage, files, server) = fake_storage(Fault::MissingChild).await;
        let kept = PathBuf::from("kept/note.md");
        let gone = PathBuf::from("gone/note.md");
        let batch = storage
            .observe_batch(&[kept.clone(), gone.clone()])
            .await
            .unwrap();
        assert!(matches!(
            batch.read(&gone, 4096).await,
            Err(VaultError::NoteNotFound(_))
        ));
        assert_eq!(batch.read(&kept, 4096).await.unwrap().0, "old");
        assert_eq!(files.lock().unwrap().read_paths, ["/vault/kept/note.md"]);
        assert!(storage.session.lock().await.is_healthy());
        server.abort();
    }

    #[tokio::test]
    async fn observed_batch_propagates_read_directory_failures_after_open() {
        for fault in [
            Fault::MissingReadDir,
            Fault::DeniedChild,
            Fault::FailedReadDir,
        ] {
            let (storage, _, server) = fake_storage(fault).await;
            let error = match storage
                .observe_batch(&[PathBuf::from("kept/note.md"), PathBuf::from("gone/note.md")])
                .await
            {
                Err(error) => error,
                Ok(_) => panic!("directory failure was hidden: {fault:?}"),
            };
            match fault {
                Fault::MissingReadDir => assert!(matches!(error, VaultError::NoteNotFound(_))),
                Fault::DeniedChild => assert!(matches!(
                    error,
                    VaultError::Storage {
                        kind: StorageErrorKind::PermissionDenied,
                        ..
                    }
                )),
                Fault::FailedReadDir => assert!(matches!(
                    error,
                    VaultError::Storage {
                        kind: StorageErrorKind::RemoteFailure,
                        ..
                    }
                )),
                _ => unreachable!(),
            }
            assert!(storage.session.lock().await.is_healthy(), "{fault:?}");
            server.abort();
        }
    }

    #[tokio::test]
    async fn unsupported_remote_names_do_not_block_valid_notes_or_end_pagination() {
        let (storage, files, server) = fake_storage(Fault::UnrepresentableNames).await;
        let entries = storage.entries(Path::new(""), true).await.unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].relative, Path::new("note.md"));
        assert_eq!(
            storage.read_file(Path::new("note.md")).await.unwrap(),
            "old"
        );
        let files = files.lock().unwrap();
        assert_eq!(files.read_paths, ["/vault/note.md"]);
        assert!(
            files
                .paths
                .iter()
                .all(|path| matches!(path.as_str(), "/vault" | "/vault/note.md"))
        );
        assert!(storage.last_failure().is_none());
        server.abort();
    }

    #[tokio::test]
    async fn malicious_directory_basenames_reject_the_listing_without_source_reads() {
        for fault in [
            Fault::EscapingName,
            Fault::AbsoluteName,
            Fault::EmptyName,
            Fault::NulName,
        ] {
            let (storage, files, server) = fake_storage(fault).await;
            assert!(
                matches!(
                    storage.entries(Path::new(""), true).await,
                    Err(VaultError::Storage {
                        kind: StorageErrorKind::Protocol,
                        ..
                    })
                ),
                "{fault:?}"
            );
            let files = files.lock().unwrap();
            assert!(files.read_paths.is_empty(), "{fault:?}");
            assert!(files.paths.iter().all(|path| path == "/vault"), "{fault:?}");
            server.abort();
        }
    }

    #[tokio::test]
    async fn move_preserves_unicode_equivalent_destination_entries_without_following_aliases() {
        for fault in [Fault::UnicodeDestination, Fault::UnicodeAliasDestination] {
            let (storage, files, server) = fake_storage(fault).await;
            assert!(
                matches!(
                    storage
                        .move_file(Path::new("note.md"), Path::new("caf\u{e9}.md"))
                        .await,
                    Err(VaultError::AlreadyExists(_))
                ),
                "{fault:?}"
            );
            let files = files.lock().unwrap();
            assert_eq!(files.renames, 0, "{fault:?}");
            assert_eq!(files.removed, 0, "{fault:?}");
            assert_eq!(files.target, b"old", "{fault:?}");
            assert!(files.read_paths.is_empty(), "{fault:?}");
            server.abort();
        }
    }

    #[tokio::test]
    async fn ordinary_remote_statuses_leave_the_source_and_session_available() {
        for fault in [Fault::NonemptyDirectory, Fault::StatFailure] {
            let (storage, _, server) = fake_storage(fault).await;
            let error = if fault == Fault::NonemptyDirectory {
                storage.delete_file(Path::new("folder")).await.unwrap_err()
            } else {
                storage
                    .read_file(Path::new("failing.md"))
                    .await
                    .unwrap_err()
            };
            let expected_outcome = if fault == Fault::NonemptyDirectory {
                MutationOutcome::NotCommitted
            } else {
                MutationOutcome::NotDispatched
            };
            assert!(
                matches!(
                    error,
                    VaultError::Storage {
                        kind: StorageErrorKind::RemoteFailure,
                        outcome,
                        ..
                    } if outcome == expected_outcome
                ),
                "{fault:?}: {error}"
            );
            assert!(storage.last_failure().is_none(), "{fault:?}");
            assert!(storage.session.lock().await.is_healthy(), "{fault:?}");
            assert_eq!(
                storage.read_file(Path::new("note.md")).await.unwrap(),
                "old",
                "{fault:?}"
            );
            server.abort();
        }
    }

    #[tokio::test]
    async fn regular_file_listing_keeps_confined_file_aliases_and_rejects_escapes() {
        let (storage, _, server) = fake_storage(Fault::ContainedAliases).await;
        let files = storage
            .list_regular_files(Path::new(""), true, None)
            .await
            .unwrap();
        assert_eq!(files, [PathBuf::from("alias.md"), PathBuf::from("note.md")]);
        let entries = storage.entries(Path::new(""), true).await.unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].relative, Path::new("note.md"));
        server.abort();

        let (storage, _, server) = fake_storage(Fault::EscapingAlias).await;
        assert!(matches!(
            storage.list_regular_files(Path::new(""), true, None).await,
            Err(VaultError::OutsideVault(_))
        ));
        assert_eq!(
            storage
                .list_regular_files(Path::new(""), true, Some("note.md"))
                .await
                .unwrap(),
            [PathBuf::from("note.md")]
        );
        server.abort();
    }

    #[tokio::test]
    async fn failed_publication_preserves_destination_and_cleanup_ownership() {
        for fault in [
            Fault::NoPosix,
            Fault::Collision,
            Fault::DenyAttrs,
            Fault::Conflict,
        ] {
            let (storage, files, task) = fake_storage(fault).await;
            let error = storage
                .write_file(Path::new("note.md"), "replacement")
                .await
                .unwrap_err();
            assert!(
                !error.to_string().contains("temporary upload may remain"),
                "{fault:?}: {error}"
            );
            let files = files.lock().unwrap();
            assert_eq!(files.renames, 0, "{fault:?}");
            assert_eq!(
                files.target,
                if fault == Fault::Conflict {
                    b"external edit".as_slice()
                } else {
                    b"old".as_slice()
                },
                "{fault:?}"
            );
            if fault == Fault::Collision {
                assert_eq!(files.removed, 0);
                assert_eq!(files.temporary.as_deref(), Some(b"unowned".as_slice()));
            }
            if fault == Fault::NoPosix {
                assert_eq!(files.writes, 0);
                assert!(files.temporary.is_none());
            }
            task.abort();
        }
    }

    #[tokio::test]
    async fn cleanup_failure_preserves_the_structured_publication_destination() {
        for (fault, expected_kind) in [
            (Fault::RenameCleanupFailure, StorageErrorKind::RemoteFailure),
            (
                Fault::AttrsCleanupFailure,
                StorageErrorKind::PermissionDenied,
            ),
            (Fault::ConflictCleanupFailure, StorageErrorKind::Conflict),
        ] {
            let (storage, files, server) = fake_storage(fault).await;
            let error = storage
                .write_file(Path::new("note.md"), "replacement")
                .await
                .unwrap_err();
            assert!(
                matches!(&error, VaultError::Storage {
                operation, path, kind, outcome: MutationOutcome::NotCommitted, ..
            } if operation == "write" && path == "note.md" && *kind == expected_kind),
                "{fault:?}: {error}"
            );
            let temporary = {
                let files = files.lock().unwrap();
                assert_eq!(
                    files.target,
                    if fault == Fault::ConflictCleanupFailure {
                        b"external edit".as_slice()
                    } else {
                        b"old".as_slice()
                    }
                );
                assert!(files.temporary.is_some());
                files.temporary_path.clone().unwrap()
            };
            let message = error.to_string();
            assert!(
                message.contains("temporary upload cleanup failed"),
                "{fault:?}: {message}"
            );
            assert!(
                message.contains("temporary upload may remain"),
                "{fault:?}: {message}"
            );
            assert_eq!(
                message.matches(&temporary).count(),
                1,
                "{fault:?}: {message}"
            );
            let data = rmcp::model::ErrorData::from(error).data.unwrap();
            assert_eq!(data["operation"], "write");
            assert_eq!(data["path"], "note.md");
            assert_eq!(data["outcome"], "not_committed");
            server.abort();
        }
    }

    #[tokio::test]
    async fn publication_deadlines_distinguish_upload_from_destination_dispatch() {
        for fault in [Fault::HoldOpen, Fault::HoldWrite, Fault::HoldRename] {
            let (storage, files, task) = fake_storage(fault).await;
            let error = storage
                .write_file(Path::new("note.md"), "replacement")
                .await
                .unwrap_err();
            let expected = if fault == Fault::HoldRename {
                MutationOutcome::Unknown
            } else {
                MutationOutcome::NotCommitted
            };
            assert!(
                matches!(error,VaultError::Storage { outcome,.. } if outcome == expected),
                "{fault:?}: {error}"
            );
            let files = files.lock().unwrap();
            let temporary = files.temporary_path.as_ref().unwrap();
            assert!(error.to_string().contains(temporary), "{fault:?}: {error}");
            assert!(
                error.to_string().contains("temporary upload may remain"),
                "{fault:?}: {error}"
            );
            assert_eq!(files.temporary_mode, Some(0o600), "{fault:?}");
            assert_eq!(files.renames, usize::from(fault == Fault::HoldRename));
            assert_eq!(
                files.target,
                if fault == Fault::HoldRename {
                    b"replacement".as_slice()
                } else {
                    b"old".as_slice()
                }
            );
            task.abort();
        }
    }

    #[tokio::test]
    async fn disconnected_upload_reports_candidate_without_reclaiming_a_replacement() {
        let (storage, files, server) = fake_storage(Fault::DisconnectWrite).await;
        let error = storage
            .write_file(Path::new("note.md"), "replacement")
            .await
            .unwrap_err();
        assert!(
            matches!(
                error,
                VaultError::Storage {
                    kind: StorageErrorKind::Unavailable,
                    outcome: MutationOutcome::NotCommitted,
                    ..
                }
            ),
            "{error}"
        );
        {
            let files = files.lock().unwrap();
            let temporary = files.temporary_path.as_ref().unwrap();
            assert!(error.to_string().contains(temporary), "{error}");
            assert!(
                error.to_string().contains("temporary upload may remain"),
                "{error}"
            );
            assert_eq!(files.temporary_mode, Some(0o600));
            assert_eq!(
                files.temporary.as_deref(),
                Some(b"unowned replacement".as_slice())
            );
            assert_eq!(files.removed, 0);
            assert_eq!(files.renames, 0);
            assert_eq!(files.target, b"old");
        }
        server.await.unwrap();
    }

    #[tokio::test]
    async fn successive_source_operations_share_the_retained_owner_deadline() {
        let (storage, _, task) = fake_storage(Fault::Normal).await;
        let deadline = tokio::time::Instant::now() + Duration::from_millis(40);
        let error = super::super::with_source_deadline(Some(deadline), async {
            tokio::time::sleep(Duration::from_millis(10)).await;
            storage
                .operation("read", Path::new("note.md"), async {
                    tokio::time::sleep(Duration::from_millis(60)).await;
                    Ok::<_, VaultError>(())
                })
                .await
        })
        .await
        .unwrap_err();
        assert!(matches!(
            error,
            VaultError::Storage {
                kind: StorageErrorKind::Unavailable,
                outcome: MutationOutcome::NotDispatched,
                ..
            }
        ));
        task.abort();
    }

    #[tokio::test]
    async fn sparse_readdir_attributes_are_classified_before_recursing() {
        let (storage, _, task) = fake_storage(Fault::SparseAttrs).await;
        let entries = storage.entries(Path::new(""), true).await.unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].relative, Path::new("nested/note.md"));
        task.abort();
    }
    #[tokio::test]
    async fn transformation_revision_is_checked_before_upload_creation() {
        let (storage, files, task) = fake_storage(Fault::Normal).await;
        let expected = crate::vault::source::source_revision("earlier revision");
        assert!(matches!(
            storage
                .write_file_if_revision(Path::new("note.md"), "transformed", &expected)
                .await,
            Err(VaultError::SourceChanged { .. })
        ));
        let files = files.lock().unwrap();
        assert_eq!(files.target, b"old");
        assert_eq!(files.writes, 0);
        assert!(files.temporary.is_none());
        task.abort();
    }
}
