//! SFTP-backed source files. Paths are interpreted by the server, never the client OS.

mod session;

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use globset::Glob;
use openssh_sftp_client_lowlevel::{FileAttrs, FileType, HandleOwned, Permissions};
use sha2::{Digest, Sha256};
use tokio::process::Command;
use tokio::sync::{Mutex as AsyncMutex, Semaphore};
use unicode_normalization::UnicodeNormalization;

use crate::config::source::SftpLocation;
use crate::error::{MutationOutcome, StorageErrorKind, VaultError, VaultResult};
use crate::models::FileStat;
use crate::vault::fs::MoveResult;
use crate::vault::path::{ResolvedPath, VaultPath};

use session::Session;

const MAX_READ_BYTES: usize = 64 * 1024 * 1024;
const CHUNK_BYTES: u32 = 32 * 1024;

#[derive(Clone, Debug)]
pub struct RemoteEntry {
    pub relative: PathBuf,
    pub stat: FileStat,
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
    reconciliation_owner: OnceLock<(Arc<AtomicBool>, Arc<tokio::sync::Notify>)>,
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
        needed: Arc<AtomicBool>,
        wakeup: Arc<tokio::sync::Notify>,
    ) -> VaultResult<()> {
        self.reconciliation_owner
            .set((needed, wakeup))
            .map_err(|_| {
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
        self.last_failure
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    fn record_result<T>(&self, result: &VaultResult<T>) {
        let mut failure = self.last_failure.lock().unwrap_or_else(|e| e.into_inner());
        if failure
            .as_ref()
            .is_some_and(|(kind, _)| *kind == StorageErrorKind::IdentityChanged)
        {
            return;
        }
        match result {
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
        let expected = self
            .policy_revision
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        if let Some(expected) = expected {
            let absolute = absolute(&self.canonical, ".obsidian-mcp/ignore");
            let observed = match next.realpath(&absolute).await {
                Ok(path) => {
                    relative_under(&self.canonical, &path)?;
                    Some(crate::vault::source::source_revision(
                        &read_source(
                            &next,
                            &path,
                            256 * 1024,
                            &self.read_budget,
                            &self.read_admission,
                        )
                        .await?,
                    ))
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
        *current = Arc::clone(&next);
        if let Some((needed, wakeup)) = self.reconciliation_owner.get() {
            needed.store(true, Ordering::Release);
            wakeup.notify_one();
        }
        Ok(next)
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
        session: &Session,
        path: &Path,
        allow_missing: bool,
    ) -> VaultResult<ResolvedPath> {
        if session.realpath(&self.canonical).await? != self.canonical {
            return Err(storage_error(
                "resolve",
                "",
                StorageErrorKind::IdentityChanged,
                "remote root changed; reopen the source",
            ));
        }
        let requested = normalize_relative(path)?;
        let mut actual = String::new();
        let components: Vec<&str> = requested.split('/').filter(|s| !s.is_empty()).collect();
        for (index, name) in components.iter().enumerate() {
            let parent = absolute(&self.canonical, &actual);
            let candidate = match find_entry(session, &parent, name).await? {
                Some((name, _)) => absolute(&parent, &name),
                None if allow_missing => {
                    actual = [actual.as_str(), &components[index..].join("/")]
                        .into_iter()
                        .filter(|s| !s.is_empty())
                        .collect::<Vec<_>>()
                        .join("/");
                    return Ok(resolved(&self.canonical, actual));
                }
                None => return Err(VaultError::NoteNotFound(path.to_owned())),
            };
            let canonical = session.realpath(&candidate).await?;
            actual = relative_under(&self.canonical, &canonical)?;
        }
        Ok(resolved(&self.canonical, actual))
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

    pub async fn read_observed(&self, path: &Path) -> VaultResult<String> {
        self.read_exact_bounded(path, MAX_READ_BYTES).await
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
                        visit_directory(&session, &remote, |name, attrs| {
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
                        .await?;
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
            let mut results = walk_directories(
                Arc::clone(&session),
                &self.canonical,
                path_text(&resolved.relative)?.to_owned(),
                recursive,
            )
            .await?
            .into_iter()
            .filter(|(path, _)| matcher.as_ref().is_none_or(|m| m.is_match(path)))
            .map(|(path, _)| PathBuf::from(path))
            .collect::<Vec<_>>();
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

    pub async fn append_file(&self, path: &Path, content: &str) -> VaultResult<PathBuf> {
        let dispatched = AtomicBool::new(false);
        self.timed_operation("append", path, Some(&dispatched), async {
            let (mut complete, create) = match self.read_file(path).await {
                Ok(text) => (text, false),
                Err(error) if error.is_not_found() => (String::new(), true),
                Err(error) => return Err(error),
            };
            let expected = (!create).then(|| crate::vault::source::source_revision(&complete));
            complete.push_str(content);
            self.publish_inner(path, &complete, create, expected.as_deref(), &dispatched)
                .await
        })
        .await
    }

    /// Publish a transformation only while its original raw revision still exists.
    pub async fn write_file_if_revision(
        &self,
        path: &Path,
        content: &str,
        expected: &str,
    ) -> VaultResult<PathBuf> {
        let dispatched = AtomicBool::new(false);
        self.timed_operation(
            "write",
            path,
            Some(&dispatched),
            self.publish_inner(path, content, false, Some(expected), &dispatched),
        )
        .await
    }

    async fn publish(&self, path: &Path, content: &str, create: bool) -> VaultResult<PathBuf> {
        let dispatched = AtomicBool::new(false);
        self.timed_operation(
            "write",
            path,
            Some(&dispatched),
            self.publish_inner(path, content, create, None, &dispatched),
        )
        .await
    }

    async fn publish_inner(
        &self,
        path: &Path,
        content: &str,
        create: bool,
        expected_revision: Option<&str>,
        dispatched: &AtomicBool,
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
        let handle = session
            .open_create(&temporary, temporary_attrs)
            .await
            .map_err(|error| error.with_mutation_outcome(MutationOutcome::NotCommitted))?;
        let upload = self
            .upload(
                &session,
                &temporary,
                &handle,
                content.as_bytes(),
                previous.as_ref().map(|(attrs, _)| *attrs),
            )
            .await;
        if let Err(error) = upload {
            return Err(cleanup_error(
                &session,
                &temporary,
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
                return Err(cleanup_error(&session, &temporary, error).await);
            }
        }
        let replacing = previous.is_some();
        dispatched.store(true, Ordering::Release);
        if let Err(error) = session.rename(&temporary, &destination, replacing).await {
            return Err(cleanup_error(&session, &temporary, error).await);
        }
        Ok(target.relative)
    }

    async fn upload(
        &self,
        session: &Session,
        path: &str,
        handle: &HandleOwned,
        bytes: &[u8],
        previous: Option<FileAttrs>,
    ) -> VaultResult<()> {
        let result = async {
            for (index, chunk) in bytes.chunks(CHUNK_BYTES as usize).enumerate() {
                session
                    .write(handle, (index * CHUNK_BYTES as usize) as u64, chunk)
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
                session.fsetstat(handle, attrs).await?;
            }
            if session.has_fsync() {
                session.fsync(handle).await?;
            }
            Ok(())
        }
        .await;
        let close = session.close(handle).await;
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
            match session.lstat(&remote_destination).await {
                Ok(_) => return Err(VaultError::AlreadyExists(to.to_owned())),
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
        session: &Session,
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

async fn cleanup_error(session: &Session, temporary: &str, error: VaultError) -> VaultError {
    if let Err(cleanup) = session.remove(temporary).await {
        let (kind, outcome) = match &error {
            VaultError::Storage { kind, outcome, .. } => (*kind, *outcome),
            VaultError::SourceChanged { .. } => {
                (StorageErrorKind::Conflict, MutationOutcome::NotCommitted)
            }
            _ => (StorageErrorKind::Unavailable, MutationOutcome::NotCommitted),
        };
        storage_error(
            "cleanup upload",
            temporary,
            kind,
            &format!("{error}; temporary upload cleanup failed: {cleanup}"),
        )
        .with_mutation_outcome(outcome)
    } else {
        error
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
    let handle = session.open_read(path).await?;
    let result = async {
        let attrs = session.fstat(&handle).await?;
        require_type(&attrs, FileType::RegularFile, path)?;
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
            let Some(chunk) = session.read(&handle, offset, allowed as u32).await? else {
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
        let stat = file_stat(&session.fstat(&handle).await?, Path::new(path))?;
        Ok((content, stat))
    }
    .await;
    let close = session.close(&handle).await;
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
    let requested = name;
    let key: String = name.nfc().collect();
    let folded = key.to_lowercase();
    let mut exact = None;
    let mut found = None;
    let mut ambiguous = false;
    visit_directory(session, parent, |name, attrs| {
        let normalized: String = name.nfc().collect();
        if name == requested {
            exact = Some((name, attrs));
        } else if normalized == key || (native_match && normalized.to_lowercase() == folded) {
            ambiguous |= found.is_some();
            found = Some((name, attrs));
        }
        Ok(())
    })
    .await?;
    if exact.is_some() {
        return Ok(exact);
    }
    if ambiguous {
        return Err(VaultError::InvalidPath(
            "ambiguous Unicode-equivalent remote path".into(),
        ));
    }
    if native_match && found.is_none() {
        return Err(VaultError::InvalidPath(
            "cannot identify the exact spelling of the native remote match".into(),
        ));
    }
    Ok(found)
}

async fn walk_directories(
    session: Arc<Session>,
    canonical: &str,
    start: String,
    recursive: bool,
) -> VaultResult<Vec<(String, FileAttrs)>> {
    let mut pending = vec![start];
    let mut jobs = tokio::task::JoinSet::new();
    let mut entries = Vec::new();
    let mut name_bytes = 0usize;
    while !pending.is_empty() || !jobs.is_empty() {
        while jobs.len() < 8 {
            let Some(directory) = pending.pop() else {
                break;
            };
            let session = Arc::clone(&session);
            let remote = absolute(canonical, &directory);
            jobs.spawn(async move {
                let mut entries = Vec::new();
                visit_directory(&session, &remote, |name, attrs| {
                    if !name.starts_with('.') {
                        entries.push((absolute(&directory, &name), attrs));
                    }
                    Ok(())
                })
                .await?;
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
    Ok(entries)
}

async fn visit_directory(
    session: &Session,
    path: &str,
    mut visit: impl FnMut(String, FileAttrs) -> VaultResult<()>,
) -> VaultResult<()> {
    let handle = session.opendir(path).await?;
    let result = async {
        let mut seen = HashSet::new();
        let mut name_bytes = 0usize;
        loop {
            let page = session.readdir(&handle).await?;
            if page.is_empty() {
                break;
            }
            for entry in page {
                let name = entry
                    .filename
                    .to_str()
                    .ok_or_else(|| {
                        VaultError::InvalidPath("remote directory name is not UTF-8".into())
                    })?
                    .to_owned();
                if name == "." || name == ".." {
                    continue;
                }
                validate_entry(&name)?;
                name_bytes = name_bytes.saturating_add(name.len());
                if seen.len() >= 1_000_000 || name_bytes > MAX_READ_BYTES {
                    return Err(storage_error(
                        "list",
                        path,
                        StorageErrorKind::ResourceLimit,
                        "directory entry budget exceeded",
                    ));
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
    let close = session.close(&handle).await;
    result.and(close)
}

fn normalize_relative(path: &Path) -> VaultResult<String> {
    Ok(VaultPath::parse(path_text(path)?)?.as_str().to_owned())
}

fn path_text(path: &Path) -> VaultResult<&str> {
    path.to_str()
        .ok_or_else(|| VaultError::InvalidPath("remote paths must be UTF-8".into()))
}

fn validate_entry(name: &str) -> VaultResult<()> {
    VaultPath::entry(name).map(|_| ())
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
            "SSH configuration could not be resolved",
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

fn timeout_from_env() -> VaultResult<Duration> {
    Ok(crate::config::source::SftpSettings::from_env()?.timeout)
}

#[cfg(test)]
mod tests {
    use super::*;

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
            .append_file(Path::new("folder/note.md"), " third")
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
    enum Fault {
        Normal,
        NoPosix,
        Collision,
        DenyAttrs,
        Conflict,
        HoldOpen,
        HoldWrite,
        HoldRename,
        SparseAttrs,
    }

    #[derive(Default)]
    struct FakeFiles {
        target: Vec<u8>,
        temporary: Option<Vec<u8>>,
        removed: usize,
        renames: usize,
        writes: usize,
        listed: HashSet<String>,
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

    async fn fake_storage(
        fault: Fault,
    ) -> (
        SftpStorage,
        Arc<Mutex<FakeFiles>>,
        tokio::task::JoinHandle<()>,
    ) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let files = Arc::new(Mutex::new(FakeFiles {
            target: b"old".to_vec(),
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
            while let Ok(len) = server.read_u32().await {
                let mut request = vec![0; len as usize];
                if server.read_exact(&mut request).await.is_err() {
                    break;
                }
                let mut response = vec![101];
                response.extend_from_slice(&request[1..5]);
                let mut status = 0u32;
                let mut hold = false;
                {
                    let mut files = server_files.lock().unwrap();
                    match request[0] {
                        16 => {
                            response[0] = 104;
                            response.extend_from_slice(&1u32.to_be_bytes());
                            fake_string(&mut response, fake_path(&request).as_bytes());
                            fake_string(&mut response, b"");
                            response.extend_from_slice(&0u32.to_be_bytes());
                        }
                        7 | 17 => {
                            let path = fake_path(&request);
                            if path == "/vault"
                                || path == "/vault/note.md"
                                || (fault == Fault::SparseAttrs
                                    && (path == "/vault/nested" || path == "/vault/nested/note.md"))
                                || (path.contains("upload-") && files.temporary.is_some())
                            {
                                response[0] = 105;
                                fake_attrs(
                                    &mut response,
                                    files.target.len(),
                                    path == "/vault" || path == "/vault/nested",
                                );
                            } else {
                                status = 2;
                            }
                        }
                        11 => {
                            files.listed.remove(&fake_path(&request));
                            response[0] = 102;
                            fake_string(&mut response, fake_path(&request).as_bytes());
                        }
                        12 => {
                            let path = fake_path(&request);
                            if files.listed.insert(path.clone()) {
                                response[0] = 104;
                                response.extend_from_slice(&1u32.to_be_bytes());
                                fake_string(
                                    &mut response,
                                    if path == "/vault" && fault == Fault::SparseAttrs {
                                        b"nested"
                                    } else {
                                        b"note.md"
                                    },
                                );
                                fake_string(&mut response, b"");
                                response.extend_from_slice(&0u32.to_be_bytes());
                            } else {
                                status = 1;
                            }
                        }
                        3 => {
                            let path = fake_path(&request);
                            if path.contains("upload-") {
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
                                response[0] = 102;
                                fake_string(&mut response, &[0xff, 0]);
                            }
                        }
                        8 => {
                            response[0] = 105;
                            fake_attrs(&mut response, files.target.len(), false);
                        }
                        5 => {
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
                            if fault == Fault::Conflict {
                                files.target = b"external edit".to_vec();
                            }
                            hold = fault == Fault::HoldWrite;
                        }
                        10 if fault == Fault::DenyAttrs => status = 3,
                        13 => {
                            files.removed += 1;
                            files.temporary = None;
                        }
                        18 | 200 => {
                            files.renames += 1;
                            files.target = files.temporary.take().unwrap();
                            hold = fault == Fault::HoldRename;
                        }
                        4 | 10 => (),
                        other => panic!("unexpected fake SFTP request {other}"),
                    }
                }
                if hold {
                    std::future::pending::<()>().await;
                }
                if response[0] == 101 {
                    response.extend_from_slice(&status.to_be_bytes());
                    fake_string(&mut response, b"injected server status");
                    fake_string(&mut response, b"");
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
    async fn failed_publication_preserves_destination_and_cleanup_ownership() {
        for fault in [
            Fault::NoPosix,
            Fault::Collision,
            Fault::DenyAttrs,
            Fault::Conflict,
        ] {
            let (storage, files, task) = fake_storage(fault).await;
            assert!(
                storage
                    .write_file(Path::new("note.md"), "replacement")
                    .await
                    .is_err(),
                "{fault:?}"
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
