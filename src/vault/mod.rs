//! Vault layer — pure filesystem operations on an Obsidian vault.
//!
//! Knows nothing about MCP; provides the data model and I/O primitives
//! that tool handlers delegate to.

mod data;
pub mod exclude;
pub mod frontmatter;
pub mod fs;
pub mod index;
pub mod parser;
pub mod patch;
pub mod path;
pub mod periodic;
pub mod search_utils;
pub mod source;
pub mod storage;
pub mod tantivy_index;
pub mod watcher;
pub mod wikilink;

#[cfg(has_embeddings)]
pub(crate) mod embedding_chunks;
#[cfg(has_embeddings)]
pub(crate) mod embedding_input;
#[cfg(has_embeddings)]
pub mod embedding_runtime;
#[cfg(has_embeddings)]
pub mod embeddings;
#[cfg(has_embeddings)]
pub(crate) mod semantic_results;

use std::collections::{HashMap, HashSet};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use storage::{SourceStatus, Storage};

use chrono::{Local, NaiveDate};
use futures_util::{FutureExt, StreamExt, stream};
use notify_debouncer_mini::Debouncer;

use crate::access::ReadScope;
use crate::config::Config;
use crate::error::{VaultError, VaultResult};
use crate::models::{
    DocumentMap, FileStat, NoteMetadata, NotePeriod, PatchRequest, SearchField, SearchMatch,
    SearchResult, VaultStats, WikiLink,
};

use self::exclude::ExcludeSet;
use self::index::VaultIndex;
use self::tantivy_index::TantivyIndex;
use self::watcher::ChangeWatcher;

#[cfg(has_embeddings)]
type EmbeddingLoaderFuture = std::pin::Pin<
    Box<
        dyn std::future::Future<Output = VaultResult<Arc<dyn embeddings::Embedder>>>
            + Send
            + 'static,
    >,
>;
#[cfg(not(has_embeddings))]
type EmbeddingLoaderFuture = ();

async fn source_work<T: Send + 'static>(
    work: impl FnOnce() -> VaultResult<T> + Send + 'static,
) -> VaultResult<T> {
    tokio::task::spawn_blocking(work)
        .await
        .map_err(|error| VaultError::Other(format!("source processing task failed: {error}")))?
}

async fn read_search_matches(
    batch: &storage::ObservedBatch<'_>,
    path: &Path,
    max_bytes: Option<usize>,
    regex: regex::Regex,
    context_len: usize,
    reservation: Option<tokio::sync::OwnedSemaphorePermit>,
) -> VaultResult<Vec<SearchMatch>> {
    let read = match max_bytes {
        Some(max_bytes) => batch.read_exact_bounded(path, max_bytes).await,
        None => batch.read_observed(path).await,
    };
    let content = match read {
        Ok(content) => content,
        Err(error) if error.is_not_found() || matches!(batch, storage::ObservedBatch::Local(_)) => {
            return Ok(Vec::new());
        }
        Err(error) => return Err(error),
    };
    source_work(move || {
        let _reservation = reservation;
        Ok(match_source_content(&content, &regex, context_len))
    })
    .await
}

fn match_source_content(
    content: &str,
    regex: &regex::Regex,
    context_len: usize,
) -> Vec<SearchMatch> {
    regex
        .find_iter(content)
        .map(|matched| {
            let (context, match_start, match_end, line) =
                index::extract_match_context(content, matched.start(), matched.end(), context_len);
            SearchMatch {
                line,
                context,
                match_start,
                match_end,
            }
        })
        .collect()
}

fn listing_bound_exceeded<T>(read: &VaultResult<T>) -> bool {
    matches!(
        read,
        Err(VaultError::SemanticResourceLimit { limit, .. })
            if *limit < storage::sftp::MAX_READ_BYTES
    )
}

fn invalid_catalog_document(error: &VaultError) -> bool {
    match error {
        VaultError::Storage {
            kind: crate::error::StorageErrorKind::InvalidSource,
            ..
        }
        | VaultError::SemanticResourceLimit { .. } => true,
        VaultError::Storage {
            kind: crate::error::StorageErrorKind::PermissionDenied,
            operation,
            ..
        } => matches!(
            operation.as_str(),
            "open" | "read" | "stat" | "lstat" | "fstat"
        ),
        _ => false,
    }
}

fn terminal_source_error(error: &VaultError, attached: bool) -> bool {
    use crate::error::StorageErrorKind;
    if matches!(
        error,
        VaultError::Storage {
            kind: StorageErrorKind::IdentityChanged,
            ..
        }
    ) {
        return true;
    }
    !attached
        && !matches!(
            error,
            VaultError::Storage {
                kind: StorageErrorKind::Unavailable
                    | StorageErrorKind::RemoteFailure
                    | StorageErrorKind::ResourceLimit,
                ..
            }
        )
}

async fn validate_source(content: String) -> VaultResult<String> {
    source_work(move || {
        frontmatter::parse_frontmatter(&content)?;
        Ok(content)
    })
    .await
}

/// Internal shared state wrapped in `Arc` for cheap cloning.
struct VaultInner {
    filesystem: Option<fs::VaultRoot>,
    source: OnceLock<Storage>,
    remote: Option<crate::config::source::SftpLocation>,
    reported_root: PathBuf,
    remote_status: RwLock<Option<SourceStatus>>,
    source_ready: tokio::sync::Notify,
    source_terminal: AtomicBool,
    reconcile_needed: Arc<AtomicBool>,
    reconcile_wakeup: Arc<tokio::sync::Notify>,
    catalog_ready: AtomicBool,
    generation: AtomicU64,
    scan_active: AtomicBool,
    changed: Mutex<HashMap<OsString, u64>>,
    source_buffers: Arc<tokio::sync::Semaphore>,
    invalid_sources: Mutex<HashMap<OsString, String>>,
    watch_enabled: bool,
    mcp_home: PathBuf,
    mcp_data: OnceLock<PathBuf>,
    exclude: OnceLock<Arc<ExcludeSet>>,
    initial_exclude: Arc<ExcludeSet>,
    index: Arc<RwLock<VaultIndex>>,
    mutations: tokio::sync::Mutex<()>,
    tantivy: Option<Arc<TantivyIndex>>,
    #[cfg(has_embeddings)]
    embeddings_requested: bool,
    #[cfg(has_embeddings)]
    embedding_runtime: OnceLock<embedding_runtime::EmbeddingRuntime>,
    /// Kept alive to sustain filesystem watching; never accessed after construction.
    /// Wrapped in `Mutex` to guarantee `Sync` (`Debouncer` contains a `mpsc::Sender`
    /// which is `Send` but not `Sync`).
    _watcher: Mutex<Option<Debouncer<ChangeWatcher>>>,
}

/// High-level facade over the vault filesystem, index, and watcher.
///
/// `Vault` is `Clone + Send + Sync` — cloning increments an internal `Arc`.
/// Mutations through shared handles are serialized from reading through indexing.
/// Metadata queries acquire a shared index lock. External editors are independent.
#[derive(Clone)]
pub struct Vault {
    inner: Arc<VaultInner>,
    background: Arc<Background>,
    read_scope: Option<Arc<ReadScope>>,
}

enum IndexRead<'a> {
    Shared(std::sync::RwLockReadGuard<'a, VaultIndex>),
    Scoped(Box<VaultIndex>),
}

impl std::ops::Deref for IndexRead<'_> {
    type Target = VaultIndex;
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Shared(index) => index,
            Self::Scoped(index) => index,
        }
    }
}

struct Background {
    worker: Mutex<Option<tokio::task::JoinHandle<()>>>,
    mutations: Mutex<Vec<tokio::task::AbortHandle>>,
    admission: Arc<tokio::sync::Semaphore>,
    owner: Mutex<std::sync::Weak<VaultInner>>,
}

enum CatalogOutcome {
    Retry(PathBuf),
    Observed {
        path: PathBuf,
        stat: Option<FileStat>,
        missing: bool,
        changed: bool,
    },
}

struct CatalogTurn {
    previous: Option<tokio::sync::oneshot::Receiver<()>>,
    release: Option<tokio::sync::oneshot::Sender<()>>,
}

impl CatalogTurn {
    fn next(previous: &mut Option<tokio::sync::oneshot::Receiver<()>>) -> Self {
        let (release, next) = tokio::sync::oneshot::channel();
        Self {
            previous: previous.replace(next),
            release: Some(release),
        }
    }

    async fn wait(&mut self) {
        if let Some(previous) = self.previous.take() {
            let _ = previous.await;
        }
    }

    fn release(&mut self) {
        self.release.take();
    }
}

impl CatalogOutcome {
    fn record(
        self,
        present: &mut HashSet<OsString>,
        next: &mut HashMap<OsString, FileStat>,
    ) -> (bool, Option<PathBuf>) {
        match self {
            Self::Retry(path) => (false, Some(path)),
            Self::Observed {
                path,
                stat,
                missing,
                changed,
            } => {
                if missing {
                    present.remove(path.as_os_str());
                    next.remove(path.as_os_str());
                } else if let Some(stat) = stat {
                    next.insert(path.into_os_string(), stat);
                }
                (changed, None)
            }
        }
    }
}

struct SourceScan {
    inner: Arc<VaultInner>,
    start: u64,
}

impl Drop for SourceScan {
    fn drop(&mut self) {
        let mut changed = self.inner.changed.lock().unwrap_or_else(|e| e.into_inner());
        self.inner.scan_active.store(false, Ordering::Release);
        changed.clear();
    }
}
impl Default for Background {
    fn default() -> Self {
        Self {
            worker: Mutex::new(None),
            mutations: Mutex::new(Vec::new()),
            admission: Arc::new(tokio::sync::Semaphore::new(16)),
            owner: Mutex::new(std::sync::Weak::new()),
        }
    }
}
impl Drop for Background {
    fn drop(&mut self) {
        #[cfg(has_embeddings)]
        if let Some(owner) = self
            .owner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .upgrade()
            && let Some(runtime) = owner.embedding_runtime.get()
        {
            runtime.retire();
        }
        if let Some(worker) = self.worker.lock().unwrap_or_else(|e| e.into_inner()).take() {
            worker.abort();
        }
        for mutation in self
            .mutations
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .drain(..)
        {
            mutation.abort();
        }
    }
}
fn cell<T>(value: T) -> OnceLock<T> {
    OnceLock::from(value)
}
#[cfg(has_embeddings)]
fn optional_cell<T>(value: Option<T>) -> OnceLock<T> {
    match value {
        Some(value) => cell(value),
        None => OnceLock::new(),
    }
}

/// Indexed fields used by bulk vault listings.
pub(crate) struct NoteListMetadata {
    pub(crate) title: String,
    pub(crate) tags: Vec<String>,
    pub(crate) stat: FileStat,
}

impl Vault {
    /// Open a vault: validate the path, build the index, and optionally start the watcher.
    pub async fn open(config: &Config) -> VaultResult<Self> {
        Self::open_inner(config, None, true).await
    }

    /// Open a vault with a dormant local embedding coordinator.
    /// Mutations remain queued until `activate_embeddings` selects local ownership.
    pub async fn open_deferred_embeddings(config: &Config) -> VaultResult<Self> {
        Self::open_inner(config, None, false).await
    }

    #[cfg(all(test, has_embeddings))]
    async fn open_with_embedding_loader(
        config: &Config,
        embedding_loader: Option<EmbeddingLoaderFuture>,
    ) -> VaultResult<Self> {
        Self::open_inner(config, embedding_loader, true).await
    }

    async fn open_inner(
        config: &Config,
        _embedding_loader: Option<EmbeddingLoaderFuture>,
        _embeddings_active: bool,
    ) -> VaultResult<Self> {
        if let Some(location) = config.sftp_location()? {
            return Self::open_remote(config, location).await;
        }
        let filesystem = fs::VaultRoot::open(&config.vault_path)?;
        let root = filesystem.path().to_path_buf();
        let (data, exclude) = data::initialize(
            &filesystem,
            config.mcp_data_dir.as_deref(),
            &config.exclude_patterns,
        )?;
        let mcp_home = root.join(".obsidian-mcp");
        let mcp_data = data.path();
        let exclude = Arc::new(exclude);
        if !exclude.is_empty() {
            tracing::info!(rules = exclude.patterns().len(), "path exclusion active");
        }

        let pending_watcher = config
            .watch
            .then(|| watcher::PendingWatcher::register(&root))
            .transpose()?;
        let (vi, tantivy) = if config.tantivy {
            let (vi, tv) =
                VaultIndex::build_ranked(filesystem.clone(), Arc::clone(&exclude)).await?;
            tracing::info!(notes = vi.notes().len(), "tantivy BM25 index built");
            (vi, Some(Arc::new(tv)))
        } else {
            (
                VaultIndex::build_rooted(filesystem.clone(), Arc::clone(&exclude)).await?,
                None,
            )
        };

        let index = Arc::new(RwLock::new(vi));

        #[cfg(has_embeddings)]
        let embedding_runtime = if config.embeddings {
            let batch_size =
                crate::config::embedding_batch_size_from_env().map_err(VaultError::Embedding)?;
            let model_name = config.embeddings_model.clone();
            let provider = config.embedding_provider;
            let relative = data
                .directory
                .join("embeddings/scopes")
                .join(exclude.scope_id(&root))
                .join("embeddings.bin");
            let cache = embeddings::CacheFile::new(data.root, relative);
            let loader = _embedding_loader.unwrap_or_else(|| {
                Box::pin(async move {
                    let model = embeddings::EmbeddingModel::load(&model_name, provider).await?;
                    Ok(Arc::new(model) as Arc<dyn embeddings::Embedder>)
                })
            });
            Some(
                embedding_runtime::EmbeddingRuntime::spawn_with_scoped_cache(
                    filesystem.clone(),
                    Arc::clone(&index),
                    cache,
                    batch_size,
                    _embeddings_active,
                    loader,
                ),
            )
        } else {
            None
        };

        let watcher_handle = pending_watcher.map(|watcher| {
            watcher.start(
                Arc::clone(&index),
                tantivy.clone(),
                #[cfg(has_embeddings)]
                embedding_runtime.clone(),
                Arc::clone(&exclude),
            )
        });

        let vault = Self {
            inner: Arc::new(VaultInner {
                source: cell(Storage::Local(filesystem.clone())),
                reported_root: root.clone(),
                filesystem: Some(filesystem),
                remote: None,
                remote_status: RwLock::new(None),
                source_ready: tokio::sync::Notify::new(),
                source_terminal: AtomicBool::new(false),
                reconcile_needed: Arc::new(AtomicBool::new(false)),
                reconcile_wakeup: Arc::new(tokio::sync::Notify::new()),
                catalog_ready: AtomicBool::new(true),
                generation: AtomicU64::new(0),
                scan_active: AtomicBool::new(false),
                changed: Mutex::new(HashMap::new()),
                source_buffers: Arc::new(tokio::sync::Semaphore::new(64 * 1024)),
                invalid_sources: Mutex::new(HashMap::new()),
                watch_enabled: config.watch,
                mcp_home,
                mcp_data: cell(mcp_data),
                exclude: cell(Arc::clone(&exclude)),
                initial_exclude: exclude,
                index,
                mutations: tokio::sync::Mutex::new(()),
                tantivy,
                #[cfg(has_embeddings)]
                embedding_runtime: optional_cell(embedding_runtime),
                #[cfg(has_embeddings)]
                embeddings_requested: config.embeddings,
                _watcher: Mutex::new(watcher_handle),
            }),
            background: Arc::new(Background::default()),
            read_scope: None,
        };
        *vault
            .background
            .owner
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Arc::downgrade(&vault.inner);
        Ok(vault)
    }

    async fn run_mutation<T, F, Fut>(&self, operation: F) -> VaultResult<T>
    where
        T: Send + 'static,
        F: FnOnce(Self) -> Fut + Send + 'static,
        Fut: std::future::Future<Output = VaultResult<T>> + Send + 'static,
    {
        let deadline = self
            .is_remote()
            .then(|| {
                crate::config::source::SftpSettings::from_env()
                    .map(|settings| tokio::time::Instant::now() + settings.timeout)
            })
            .transpose()?;
        self.run_mutation_before(deadline, operation).await
    }

    async fn run_mutation_before<T, F, Fut>(
        &self,
        deadline: Option<tokio::time::Instant>,
        operation: F,
    ) -> VaultResult<T>
    where
        T: Send + 'static,
        F: FnOnce(Self) -> Fut + Send + 'static,
        Fut: std::future::Future<Output = VaultResult<T>> + Send + 'static,
    {
        self.check_mutation_scope()?;
        let admission = Arc::clone(&self.background.admission).acquire_owned();
        let permit = match deadline {
            Some(deadline) => tokio::time::timeout_at(deadline, admission)
                .await
                .map_err(|_| {
                    self.source_error(
                        crate::error::StorageErrorKind::Unavailable,
                        "mutation deadline expired before admission",
                    )
                })?,
            None => admission.await,
        }
        .map_err(|_| VaultError::Other("mutation service closed".into()))?;
        let vault = self.clone();
        let task = tokio::spawn(async move {
            let _permit = permit;
            let lock = vault.lock_mutations();
            let _mutation = match deadline {
                Some(deadline) => tokio::time::timeout_at(deadline, lock).await.map_err(|_| {
                    vault.source_error(
                        crate::error::StorageErrorKind::Unavailable,
                        "mutation deadline expired before dispatch",
                    )
                })?,
                None => lock.await,
            }?;
            let retained = vault.clone();
            let result = storage::with_source_deadline(deadline, operation(vault.clone())).await;
            if let Err(error) = &result {
                retained.reconcile_after_mutation_error(error);
            }
            result
        });
        {
            let mut retained = self
                .background
                .mutations
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            retained.retain(|task| !task.is_finished());
            retained.push(task.abort_handle());
        }
        task.await
            .map_err(|error| VaultError::Other(format!("mutation task failed: {error}")))?
    }
    fn committed_error(&self, path: &Path, error: VaultError) -> VaultError {
        if !self.is_remote() {
            return error;
        }
        let kind = match &error {
            VaultError::Storage { kind, .. } => *kind,
            VaultError::FrontmatterParse { .. }
            | VaultError::InvalidFrontmatter { .. }
            | VaultError::SemanticResourceLimit { .. } => {
                crate::error::StorageErrorKind::InvalidSource
            }
            VaultError::Io(error) if error.kind() == std::io::ErrorKind::InvalidData => {
                crate::error::StorageErrorKind::InvalidSource
            }
            _ => crate::error::StorageErrorKind::ProjectionFailure,
        };
        VaultError::Storage {
            operation: "index publication".into(),
            path: path.display().to_string(),
            kind,
            message: error.to_string(),
            outcome: crate::error::MutationOutcome::Committed,
        }
    }

    fn reconcile_after_mutation_error(&self, error: &VaultError) {
        if !self.is_remote()
            || !matches!(
                error,
                VaultError::Storage {
                    kind,
                    outcome: crate::error::MutationOutcome::Unknown
                        | crate::error::MutationOutcome::Committed,
                    ..
                } if *kind != crate::error::StorageErrorKind::IdentityChanged
            )
        {
            return;
        }
        self.inner.reconcile_needed.store(true, Ordering::Release);
        self.update_source_status(|status| {
            self.inner.catalog_ready.store(false, Ordering::Release);
            status.availability = "unavailable".into();
            status.catalog_phase = "degraded".into();
            status.error = Some(error.to_string());
        });
        self.inner.reconcile_wakeup.notify_one();
    }
    fn source_error(
        &self,
        kind: crate::error::StorageErrorKind,
        message: impl Into<String>,
    ) -> VaultError {
        VaultError::Storage {
            operation: "source".into(),
            path: self.source_location(),
            kind,
            message: message.into(),
            outcome: crate::error::MutationOutcome::NotDispatched,
        }
    }

    pub fn is_remote(&self) -> bool {
        self.inner.remote.is_some()
    }
    pub fn source_location(&self) -> String {
        self.inner
            .remote
            .as_ref()
            .map(|v| v.display_uri())
            .unwrap_or_else(|| self.root().display().to_string())
    }
    pub fn source_status(&self) -> Option<SourceStatus> {
        let mut status = self
            .inner
            .remote_status
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()?;
        if self.inner.reconcile_needed.load(Ordering::Acquire) {
            status.availability = "unavailable".into();
            status.catalog_phase = "degraded".into();
            status
                .error
                .get_or_insert_with(|| "source catalog requires reconciliation".into());
        }
        if let Some(Storage::Sftp(source)) = self.inner.source.get()
            && let Some((_, message)) = source.last_failure()
        {
            status.availability = "unavailable".into();
            status.catalog_phase = "degraded".into();
            status.error = Some(message);
        }
        Some(status)
    }
    pub fn catalog_ready(&self) -> bool {
        self.inner.catalog_ready.load(Ordering::Acquire)
    }
    pub fn resolved_mcp_data(&self) -> Option<&Path> {
        self.inner.mcp_data.get().map(PathBuf::as_path)
    }
    pub async fn storage(&self) -> VaultResult<Storage> {
        if let Some(source) = self.inner.source.get() {
            return Ok(source.clone());
        }
        let deadline =
            storage::source_deadline(crate::config::source::SftpSettings::from_env()?.timeout);
        let mut wake_requested = false;
        loop {
            let notified = self.inner.source_ready.notified();
            if let Some(source) = self.inner.source.get() {
                return Ok(source.clone());
            }
            if let Some(status) = self.source_status()
                && status.availability == "unavailable"
            {
                if self.inner.source_terminal.load(Ordering::Acquire) {
                    return Err(self.source_error(
                        crate::error::StorageErrorKind::Unavailable,
                        status
                            .error
                            .unwrap_or_else(|| "remote source unavailable".into()),
                    ));
                }
                if !wake_requested {
                    self.inner.reconcile_wakeup.notify_one();
                    wake_requested = true;
                }
            }
            tokio::time::timeout_at(deadline, notified)
                .await
                .map_err(|_| {
                    self.source_error(
                        crate::error::StorageErrorKind::Unavailable,
                        self.source_status()
                            .and_then(|status| status.error)
                            .unwrap_or_else(|| "remote source initialization timed out".into()),
                    )
                })?;
        }
    }
    pub async fn is_file(&self, path: &Path) -> VaultResult<bool> {
        self.storage().await?.is_file(path).await
    }
    fn require_catalog(&self) -> VaultResult<()> {
        if let Some(status) = self.source_status()
            && status.availability == "unavailable"
        {
            return Err(self.source_error(
                crate::error::StorageErrorKind::Unavailable,
                status
                    .error
                    .unwrap_or_else(|| "remote source unavailable".into()),
            ));
        }
        if !self.catalog_ready() {
            return Err(self.source_error(
                crate::error::StorageErrorKind::Warming,
                self.source_status()
                    .and_then(|s| s.error)
                    .unwrap_or_else(|| "remote catalog is warming".into()),
            ));
        }
        Ok(())
    }

    #[cfg(has_embeddings)]
    pub(crate) fn require_semantic_catalog(&self) -> VaultResult<()> {
        // A completed catalog can rank retained evidence during a connection
        // outage; context reads still report their own source errors. An uncertain
        // mutation withdraws catalog readiness until a complete repair scan.
        if self.inner.reconcile_needed.load(Ordering::Acquire) {
            return Err(self.source_error(
                crate::error::StorageErrorKind::Unavailable,
                "source catalog requires reconciliation after an uncertain mutation",
            ));
        }
        if !self.catalog_ready() {
            return self.require_catalog();
        }
        if let Some(Storage::Sftp(source)) = self.inner.source.get()
            && let Some((kind, message)) = source.last_failure()
            && matches!(
                kind,
                crate::error::StorageErrorKind::IdentityChanged
                    | crate::error::StorageErrorKind::Protocol
                    | crate::error::StorageErrorKind::Unsupported
            )
        {
            return Err(self.source_error(kind, message));
        }
        Ok(())
    }
    fn update_source_status(&self, update: impl FnOnce(&mut SourceStatus)) {
        if let Some(status) = self
            .inner
            .remote_status
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .as_mut()
        {
            update(status)
        }
        self.inner.source_ready.notify_waiters();
    }

    fn begin_reconciliation(&self, notified: bool) -> bool {
        let mut uncertain = false;
        self.update_source_status(|status| {
            uncertain = self.inner.reconcile_needed.swap(false, Ordering::AcqRel);
            if uncertain {
                self.inner.catalog_ready.store(false, Ordering::Release);
                status.availability = "unavailable".into();
                status.catalog_phase = "degraded".into();
                status
                    .error
                    .get_or_insert_with(|| "source catalog requires reconciliation".into());
            } else if notified && self.catalog_ready() {
                status.availability = "available".into();
                status.catalog_phase = "reconciling".into();
                status.error = None;
            }
        });
        uncertain
    }
    async fn open_remote(
        config: &Config,
        location: crate::config::source::SftpLocation,
    ) -> VaultResult<Self> {
        let initial = Arc::new(ExcludeSet::build(config.exclude_patterns.clone())?);
        let source = Self {
            inner: Arc::new(VaultInner {
                filesystem: None,
                source: OnceLock::new(),
                reported_root: PathBuf::from(&location.root),
                remote_status: RwLock::new(Some(SourceStatus {
                    kind: "sftp".into(),
                    location: location.display_uri(),
                    availability: "initializing".into(),
                    catalog_phase: "warming".into(),
                    last_scan: None,
                    last_verification: None,
                    error: None,
                })),
                remote: Some(location),
                source_ready: tokio::sync::Notify::new(),
                source_terminal: AtomicBool::new(false),
                reconcile_needed: Arc::new(AtomicBool::new(false)),
                reconcile_wakeup: Arc::new(tokio::sync::Notify::new()),
                catalog_ready: AtomicBool::new(false),
                generation: AtomicU64::new(0),
                scan_active: AtomicBool::new(false),
                changed: Mutex::new(HashMap::new()),
                source_buffers: Arc::new(tokio::sync::Semaphore::new(64 * 1024)),
                invalid_sources: Mutex::new(HashMap::new()),
                watch_enabled: config.watch,
                mcp_home: PathBuf::new(),
                mcp_data: OnceLock::new(),
                exclude: OnceLock::new(),
                initial_exclude: Arc::clone(&initial),
                index: Arc::new(RwLock::new(VaultIndex::for_source(initial))),
                mutations: tokio::sync::Mutex::new(()),
                tantivy: if config.tantivy {
                    Some(Arc::new(TantivyIndex::empty()?))
                } else {
                    None
                },
                #[cfg(has_embeddings)]
                embedding_runtime: OnceLock::new(),
                #[cfg(has_embeddings)]
                embeddings_requested: config.embeddings,
                _watcher: Mutex::new(None),
            }),
            background: Arc::new(Background::default()),
            read_scope: None,
        };
        *source
            .background
            .owner
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Arc::downgrade(&source.inner);
        let weak = Arc::downgrade(&source.inner);
        let config = config.clone();
        let settings = crate::config::source::SftpSettings::from_env()?;
        let wakeup = Arc::clone(&source.inner.reconcile_wakeup);
        let worker = tokio::spawn(async move {
            let mut initialized = false;
            let mut recovering = false;
            let mut notified = false;
            let mut verified = std::time::Instant::now();
            let mut observations = HashMap::new();
            loop {
                let Some(inner) = weak.upgrade() else { break };
                let service = Self {
                    inner,
                    background: Arc::new(Background::default()),
                    read_scope: None,
                };
                if notified || service.inner.reconcile_needed.load(Ordering::Acquire) {
                    recovering |= service.begin_reconciliation(notified);
                }
                recovering |= notified;
                let full =
                    !initialized || recovering || verified.elapsed() >= settings.verify_interval;
                let result = if !initialized {
                    service.initialize_remote(&config).await
                } else {
                    Ok(())
                };
                let result = match result {
                    Ok(()) => service.reconcile_remote(full, &mut observations).await,
                    Err(e) => Err(e),
                };
                let mut failed_cycle = false;
                match result {
                    Ok(()) => {
                        initialized = true;
                        let invalid = service
                            .inner
                            .invalid_sources
                            .lock()
                            .unwrap_or_else(|e| e.into_inner());
                        let catalog_error = invalid.values().next().cloned();
                        let mut complete = false;
                        service.update_source_status(|s| {
                            if service.inner.reconcile_needed.load(Ordering::Acquire) {
                                return;
                            }
                            complete = true;
                            service.inner.catalog_ready.store(true, Ordering::Release);
                            s.availability = "available".into();
                            s.catalog_phase = if catalog_error.is_some() {
                                "degraded"
                            } else {
                                "ready"
                            }
                            .into();
                            s.error = catalog_error;
                            s.last_scan = Some(chrono::Utc::now());
                            if full {
                                s.last_verification = s.last_scan
                            }
                        });
                        if complete {
                            recovering = false;
                            if full {
                                verified = std::time::Instant::now()
                            }
                            #[cfg(has_embeddings)]
                            if let Some(runtime) = service.inner.embedding_runtime.get() {
                                runtime.activate();
                            }
                        }
                    }
                    Err(error) => {
                        failed_cycle = true;
                        let message = error.to_string();
                        if service.source_status().and_then(|s| s.error).as_deref()
                            != Some(message.as_str())
                        {
                            tracing::warn!(error=%message,"remote source reconciliation failed");
                        }
                        let terminal =
                            terminal_source_error(&error, service.inner.source.get().is_some());
                        service
                            .inner
                            .source_terminal
                            .store(terminal, Ordering::Release);
                        service.update_source_status(|s| {
                            s.availability = "unavailable".into();
                            s.catalog_phase = "degraded".into();
                            s.error = Some(message)
                        });
                        if terminal {
                            #[cfg(has_embeddings)]
                            if let Some(runtime) = service.inner.embedding_runtime.get() {
                                runtime.retire();
                            }
                            break;
                        }
                        recovering = true;
                    }
                }
                if failed_cycle && service.inner.source.get().is_some() {
                    // This failed scan has settled its source jobs. Its own
                    // reconnect notice must not bypass the recovery cadence.
                    let _ = wakeup.notified().now_or_never();
                }
                drop(service);
                notified = if !config.watch && initialized && !recovering {
                    wakeup.notified().await;
                    true
                } else {
                    let retry_interval = if !initialized || recovering {
                        settings
                            .poll_interval
                            .min(std::time::Duration::from_secs(5))
                    } else {
                        settings.poll_interval
                    };
                    tokio::select! {
                        _ = tokio::time::sleep(retry_interval) => false,
                        _ = wakeup.notified() => true,
                    }
                };
            }
        });
        *source
            .background
            .worker
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Some(worker);
        Ok(source)
    }

    async fn initialize_remote(&self, config: &Config) -> VaultResult<()> {
        if self.inner.source.get().is_some() {
            return Ok(());
        }
        let location = self
            .inner
            .remote
            .as_ref()
            .ok_or_else(|| VaultError::Other("missing remote source".into()))?;
        let remote = Arc::new(storage::sftp::SftpStorage::open(location).await?);
        let portable = match remote
            .read_file_bounded(Path::new(".obsidian-mcp/ignore"), 256 * 1024)
            .await
        {
            Ok(raw) => Some(raw),
            Err(e) if e.is_not_found() => None,
            Err(e) => return Err(e),
        };
        remote.pin_policy_revision(portable.as_ref().map(|raw| source::source_revision(raw)));
        remote.set_reconciliation_owner(Arc::clone(&self.inner.reconcile_wakeup))?;
        let base = config
            .mcp_data_dir
            .clone()
            .map(Ok)
            .unwrap_or_else(crate::daemon::home::resolve_semantic_home)?;
        let relative = PathBuf::from("sources-v1").join(remote.source_id());
        let base_clone = base.clone();
        let data_root = tokio::task::spawn_blocking(move || {
            std::fs::create_dir_all(&base_clone)?;
            fs::VaultRoot::open(&base_clone)
        })
        .await
        .map_err(|e| VaultError::Other(format!("machine data initialization failed: {e}")))??;
        data_root.create_dir_all(&relative)?;
        let mut patterns = config.exclude_patterns.clone();
        if let Some(raw) = portable {
            patterns.extend(exclude::parse_ignore_lines(&raw));
        }
        match data_root.read_file(&relative.join("ignore")) {
            Ok(raw) => patterns.extend(exclude::parse_ignore_lines(&raw)),
            Err(e) if e.is_not_found() => (),
            Err(e) => return Err(e),
        }
        let policy = Arc::new(ExcludeSet::build(patterns)?);
        let policy_key = policy.scope_id(remote.path());
        let cache_relative = relative
            .join("embeddings/scopes")
            .join(policy_key)
            .join("embeddings.bin");
        data_root.create_dir_all(
            cache_relative
                .parent()
                .ok_or_else(|| VaultError::InvalidPath("invalid checkpoint path".into()))?,
        )?;
        let storage = Storage::Sftp(remote);
        let mut index = VaultIndex::for_source(Arc::clone(&policy));
        index.begin_source_batch();
        *self.write_index() = index;
        let _ = self.inner.exclude.set(policy);
        let _ = self.inner.mcp_data.set(base.join(&relative));
        #[cfg(has_embeddings)]
        if config.embeddings {
            let model_name = config.embeddings_model.clone();
            let provider = config.embedding_provider;
            let loader = Box::pin(async move {
                let model = embeddings::EmbeddingModel::load(&model_name, provider).await?;
                Ok(Arc::new(model) as Arc<dyn embeddings::Embedder>)
            });
            let runtime = embedding_runtime::EmbeddingRuntime::spawn_with_source(
                storage.clone(),
                Arc::clone(&self.inner.index),
                embeddings::CacheFile::new(data_root, cache_relative),
                crate::config::embedding_batch_size_from_env().map_err(VaultError::Embedding)?,
                false,
                loader,
            );
            let _ = self.inner.embedding_runtime.set(runtime);
        }
        let _ = self.inner.source.set(storage);
        self.update_source_status(|s| {
            s.availability = "available".into();
            s.catalog_phase = "warming".into();
            s.error = None
        });
        Ok(())
    }

    async fn begin_source_scan(&self) -> SourceScan {
        let _mutation = self.inner.mutations.lock().await;
        let mut changed = self.inner.changed.lock().unwrap_or_else(|e| e.into_inner());
        changed.clear();
        self.inner.scan_active.store(true, Ordering::Release);
        SourceScan {
            inner: Arc::clone(&self.inner),
            start: self.inner.generation.load(Ordering::Acquire),
        }
    }

    async fn reconcile_remote(
        &self,
        full: bool,
        observations: &mut HashMap<OsString, FileStat>,
    ) -> VaultResult<()> {
        let source = self.storage().await?;
        let Storage::Sftp(remote) = &source else {
            return Ok(());
        };
        let scan = self.begin_source_scan().await;
        let start = scan.start;
        let entries = remote.entries(Path::new(""), true).await?;
        let mut present = HashSet::new();
        let mut next = HashMap::new();
        let mut changed = Vec::new();
        let invalid_paths = self
            .inner
            .invalid_sources
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .keys()
            .cloned()
            .collect::<HashSet<_>>();
        for entry in entries {
            let path = entry.relative;
            let stat = entry.stat;
            present.insert(path.as_os_str().to_owned());
            next.insert(path.as_os_str().to_owned(), stat.clone());
            let unchanged = observations.get(path.as_os_str()).is_some_and(|old| {
                old.size == stat.size && old.modified.is_some() && old.modified == stat.modified
            });
            if full || !unchanged || invalid_paths.contains(path.as_os_str()) {
                changed.push((path, stat));
            }
        }
        changed.sort_by(|left, right| left.0.as_os_str().cmp(right.0.as_os_str()));
        let note_paths = changed
            .iter()
            .filter(|(path, _)| {
                path::logical_extension(path).is_some_and(|ext| ext.eq_ignore_ascii_case("md"))
                    && !self.exclude().is_excluded(path)
            })
            .map(|(path, _)| path.clone())
            .collect::<Vec<_>>();
        let batch = source.observe_batch(&note_paths).await?;
        let mut changed_any = false;
        let read_budget = Arc::clone(&self.inner.source_buffers);
        for chunk in changed.chunks(16) {
            let failed = Arc::new(AtomicBool::new(false));
            let mut admission_order = None;
            let mut projection_order = None;
            let reads = stream::iter(chunk.iter().cloned().map(|(path, stat)| {
                let batch = &batch;
                let read_budget = Arc::clone(&read_budget);
                let failed = Arc::clone(&failed);
                let mut admission_turn = CatalogTurn::next(&mut admission_order);
                let mut projection_turn = CatalogTurn::next(&mut projection_order);
                async move {
                    let outcome = async {
                        admission_turn.wait().await;
                        if failed.load(Ordering::Acquire) {
                            return Ok(None);
                        }
                        let note = path::logical_extension(&path)
                            .is_some_and(|ext| ext.eq_ignore_ascii_case("md"))
                            && !self.exclude().is_excluded(&path);
                        let max_bytes =
                            stat.size.min(storage::sftp::MAX_READ_BYTES as u64) as usize;
                        let reservation = if note {
                            Some(
                                read_budget
                                    .acquire_many_owned(max_bytes.max(1).div_ceil(1024) as u32)
                                    .await
                                    .map_err(|_| {
                                        self.source_error(
                                            crate::error::StorageErrorKind::Unavailable,
                                            "catalog content admission closed",
                                        )
                                    })?,
                            )
                        } else {
                            None
                        };
                        admission_turn.release();
                        if failed.load(Ordering::Acquire) {
                            return Ok(None);
                        }
                        let read = if note {
                            batch
                                .read_with_stat_bounded(&path, max_bytes)
                                .await
                                .map(|(raw, stat)| Some((Some(raw), stat)))
                        } else {
                            Ok(Some((None, stat)))
                        };
                        if matches!(&read, Err(error)
                            if !error.is_not_found() && !invalid_catalog_document(error))
                        {
                            failed.store(true, Ordering::Release);
                        }
                        projection_turn.wait().await;
                        if failed.load(Ordering::Acquire) && read.is_ok() {
                            return Ok(None);
                        }
                        self.project_catalog_read(path, read, start, reservation)
                            .await
                    }
                    .await;
                    if outcome.is_err() {
                        failed.store(true, Ordering::Release);
                    }
                    outcome
                }
            }))
            .buffered(16);
            // Futures retain their content reservations through projection, then
            // yield only small outcomes. Settle all launched readers before an
            // error leaves the catalog owner; no next chunk starts after failure.
            let outcomes = reads.collect::<Vec<_>>().await;
            let mut first_error = None;
            let mut grown = Vec::new();
            for outcome in outcomes {
                match outcome {
                    Ok(Some(outcome)) => {
                        let (changed, retry) = outcome.record(&mut present, &mut next);
                        changed_any |= changed;
                        grown.extend(retry);
                    }
                    Ok(None) => (),
                    Err(error) => {
                        first_error.get_or_insert(error);
                    }
                }
            }
            if let Some(error) = first_error {
                return Err(error);
            }
            for path in grown {
                let reservation = Arc::clone(&self.inner.source_buffers)
                    .acquire_many_owned((storage::sftp::MAX_READ_BYTES / 1024) as u32)
                    .await
                    .map_err(|_| {
                        self.source_error(
                            crate::error::StorageErrorKind::Unavailable,
                            "source content admission closed",
                        )
                    })?;
                let read = batch
                    .read_with_stat_bounded(&path, storage::sftp::MAX_READ_BYTES)
                    .await
                    .map(|(raw, stat)| Some((Some(raw), stat)));
                if let Some(outcome) = self
                    .project_catalog_read(path, read, start, Some(reservation))
                    .await?
                {
                    let (changed, retry) = outcome.record(&mut present, &mut next);
                    changed_any |= changed;
                    if retry.is_some() {
                        return Err(self.source_error(
                            crate::error::StorageErrorKind::ResourceLimit,
                            "source content exceeded its admitted maximum",
                        ));
                    }
                }
            }
        }
        let _mutation = self.inner.mutations.lock().await;
        let old = self
            .read_index()
            .tracked_paths()
            .map(Path::to_owned)
            .collect::<Vec<_>>();
        for path in old {
            if !present.contains(path.as_os_str())
                && !self
                    .inner
                    .changed
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .get(path.as_os_str())
                    .is_some_and(|g| *g > start)
            {
                self.apply_source_observation(path.clone(), None, false, None)
                    .await?;
                changed_any = true;
            }
        }
        self.inner
            .invalid_sources
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .retain(|path, _| present.contains(path));
        if changed_any || full || !self.catalog_ready() {
            let vault = self.clone();
            tokio::task::spawn_blocking(move || vault.publish_source())
                .await
                .map_err(|e| {
                    VaultError::Other(format!("source index publication task failed: {e}"))
                })??;
        }
        *observations = next;
        Ok(())
    }

    async fn project_catalog_read(
        &self,
        path: PathBuf,
        read: VaultResult<Option<(Option<String>, FileStat)>>,
        start: u64,
        reservation: Option<tokio::sync::OwnedSemaphorePermit>,
    ) -> VaultResult<Option<CatalogOutcome>> {
        if listing_bound_exceeded(&read) {
            return Ok(Some(CatalogOutcome::Retry(path)));
        }
        let missing =
            matches!(&read, Ok(None)) || matches!(&read, Err(error) if error.is_not_found());
        let stat = read
            .as_ref()
            .ok()
            .and_then(|entry| entry.as_ref())
            .map(|(_, stat)| stat.clone());
        Ok(self
            .apply_catalog_read(path.clone(), read, start, reservation)
            .await?
            .map(|changed| CatalogOutcome::Observed {
                path,
                stat,
                missing,
                changed,
            }))
    }

    async fn apply_catalog_read(
        &self,
        path: PathBuf,
        read: VaultResult<Option<(Option<String>, FileStat)>>,
        start: u64,
        reservation: Option<tokio::sync::OwnedSemaphorePermit>,
    ) -> VaultResult<Option<bool>> {
        let _mutation = self.inner.mutations.lock().await;
        if self
            .inner
            .changed
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(path.as_os_str())
            .is_some_and(|g| *g > start)
        {
            return Ok(None);
        }
        let observation = match read {
            Ok(observation) => observation,
            Err(error) if error.is_not_found() => None,
            Err(error) if invalid_catalog_document(&error) => {
                self.apply_source_observation(path.clone(), None, false, None)
                    .await?;
                self.record_invalid_source(&path, &error);
                return Ok(Some(true));
            }
            Err(error) => return Err(error),
        };
        let changed = match self
            .apply_source_observation(path.clone(), observation, false, reservation)
            .await
        {
            Ok(changed) => {
                self.inner
                    .invalid_sources
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .remove(path.as_os_str());
                changed
            }
            Err(error @ VaultError::FrontmatterParse { .. })
            | Err(
                error @ VaultError::Storage {
                    kind: crate::error::StorageErrorKind::InvalidSource,
                    ..
                },
            ) => {
                self.record_invalid_source(&path, &error);
                true
            }
            Err(error) => return Err(error),
        };
        Ok(Some(changed))
    }

    fn record_invalid_source(&self, path: &Path, error: &VaultError) {
        let message = error.to_string();
        let mut invalid = self
            .inner
            .invalid_sources
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if invalid.get(path.as_os_str()) != Some(&message) {
            tracing::warn!(path=%path.display(),error=%message,"invalid remote note removed from indexes");
        }
        invalid.insert(path.as_os_str().to_owned(), message);
    }

    async fn periodic_config(
        &self,
        period: &NotePeriod,
    ) -> VaultResult<crate::models::PeriodicNoteConfig> {
        let source = self.storage().await?;
        if let Some(root) = source.local_root() {
            let root = root.clone();
            let period = period.clone();
            return source_work(move || periodic::read_config_rooted(&root, &period)).await;
        }
        let plugin = match source
            .read_file(Path::new(".obsidian/plugins/periodic-notes/data.json"))
            .await
        {
            Ok(v) => Some(v),
            Err(e) if e.is_not_found() => None,
            Err(e) => return Err(e),
        };
        if let Some(selected) = periodic::plugin_config(plugin.as_deref(), period)? {
            return Ok(selected);
        }
        let daily = if matches!(period, NotePeriod::Daily) {
            match source
                .read_file(Path::new(".obsidian/daily-notes.json"))
                .await
            {
                Ok(v) => Some(v),
                Err(e) if e.is_not_found() => None,
                Err(e) => return Err(e),
            }
        } else {
            None
        };
        periodic::read_config_strings(None, daily.as_deref(), period)
    }
    async fn expand_template(
        &self,
        template_path: &Path,
        date: &NaiveDate,
        title: &str,
    ) -> VaultResult<String> {
        let path = if self.is_remote() && path::logical_extension(template_path).is_none() {
            PathBuf::from(format!(
                "{}.md",
                template_path
                    .to_str()
                    .ok_or_else(|| VaultError::InvalidPath(
                        "remote template paths must be UTF-8".into()
                    ))?
            ))
        } else if !self.is_remote() && template_path.extension().is_none() {
            template_path.with_extension("md")
        } else {
            template_path.to_owned()
        };
        let content = self.storage().await?.read_file(&path).await?;
        let date = *date;
        let title = title.to_owned();
        source_work(move || periodic::expand_template_string(&content, &date, &title)).await
    }
    async fn list_periodic(
        &self,
        config: &crate::models::PeriodicNoteConfig,
        limit: usize,
    ) -> VaultResult<Vec<PathBuf>> {
        let source = self.storage().await?;
        if let Some(root) = source.local_root() {
            let root = root.clone();
            let config = config.clone();
            return source_work(move || periodic::list_rooted(&root, &config, limit)).await;
        }
        let folder = Path::new(config.folder.as_deref().unwrap_or(""));
        let actual = match source.resolve_existing(folder).await {
            Ok(v) => v.relative,
            Err(e) if e.is_not_found() => return Ok(Vec::new()),
            Err(e) => return Err(e),
        };
        let paths = match source
            .list_regular_files(
                Path::new(config.folder.as_deref().unwrap_or("")),
                true,
                None,
            )
            .await
        {
            Ok(v) => v,
            Err(e) if e.is_not_found() => return Ok(Vec::new()),
            Err(e) => return Err(e),
        };
        let mut filtered = Vec::new();
        for path in paths {
            let path_text = path.to_str().ok_or_else(|| {
                VaultError::InvalidPath("remote vault paths must be UTF-8".into())
            })?;
            let actual_text = actual.to_str().ok_or_else(|| {
                VaultError::InvalidPath("remote vault paths must be UTF-8".into())
            })?;
            let suffix = if actual_text.is_empty() {
                path_text
            } else {
                path_text
                    .strip_prefix(actual_text)
                    .and_then(|suffix| suffix.strip_prefix('/'))
                    .ok_or_else(|| VaultError::OutsideVault(path.clone()))?
            };
            filtered.push(path::logical_join(folder, Path::new(suffix))?);
        }
        periodic::filter_recent_paths(config, filtered, limit)
    }

    /// Activate local embeddings once without replacing the watcher submission handle.
    pub fn activate_embeddings(&self) -> bool {
        if self.is_remote() && !self.catalog_ready() {
            return false;
        }
        #[cfg(has_embeddings)]
        {
            self.inner
                .embedding_runtime
                .get()
                .is_some_and(embedding_runtime::EmbeddingRuntime::activate)
        }
        #[cfg(not(has_embeddings))]
        {
            false
        }
    }

    /// Whether the local coordinator has been activated, independently of readiness.
    pub fn embeddings_active(&self) -> bool {
        #[cfg(has_embeddings)]
        {
            self.inner
                .embedding_runtime
                .get()
                .is_some_and(embedding_runtime::EmbeddingRuntime::is_active)
        }
        #[cfg(not(has_embeddings))]
        {
            false
        }
    }

    /// Retain the same source/index with an immutable, read-only audience policy.
    pub fn with_read_scope(&self, scope: ReadScope) -> VaultResult<Self> {
        scope.validate()?;
        if self.is_remote() {
            return Err(VaultError::Other(
                "private memory scopes require a local vault".into(),
            ));
        }
        Ok(Self {
            inner: Arc::clone(&self.inner),
            background: Arc::clone(&self.background),
            read_scope: Some(Arc::new(scope)),
        })
    }

    pub fn is_scoped(&self) -> bool {
        self.read_scope.is_some()
    }

    pub(crate) async fn check_result_revision(
        &self,
        path: &Path,
        revision: &str,
    ) -> VaultResult<()> {
        if self.is_scoped() {
            self.read_source_snapshot(path)
                .await?
                .check_revision(path, Some(revision))?;
        }
        Ok(())
    }

    #[cfg(has_embeddings)]
    fn check_scoped_revision_native(&self, path: &Path, revision: &str) -> VaultResult<()> {
        let raw = self.read_scoped_native(path, Some(source::MAX_SOURCE_BYTES))?;
        source::SourceSnapshot::new(raw).check_revision(path, Some(revision))?;
        Ok(())
    }

    fn read_scoped_native(&self, path: &Path, max_bytes: Option<usize>) -> VaultResult<String> {
        self.check_scope_path(path)?;
        let filesystem = self.inner.filesystem.as_ref().ok_or_else(|| {
            VaultError::Other("private memory scopes require a local vault".into())
        })?;
        let actual = filesystem.resolve_existing(path)?.relative;
        self.check_scope_path(&actual)?;
        let content = match max_bytes {
            Some(max_bytes) => filesystem.read_exact_bounded(&actual, max_bytes)?,
            None => filesystem.read_observed(&actual)?,
        };
        self.check_content_scope(&actual, &content)?;
        Ok(content)
    }

    fn check_scope_path(&self, path: &Path) -> VaultResult<()> {
        if self
            .read_scope
            .as_ref()
            .is_some_and(|scope| !scope.allows_path(path))
        {
            return Err(VaultError::NoteNotFound(path.to_owned()));
        }
        Ok(())
    }

    fn check_content_scope(&self, path: &Path, content: &str) -> VaultResult<()> {
        let Some(scope) = &self.read_scope else {
            return Ok(());
        };
        let metadata = frontmatter::parse_frontmatter(content)
            .map_err(|_| VaultError::NoteNotFound(path.to_owned()))?;
        if scope.expires_at <= chrono::Utc::now().timestamp_millis() as f64 / 1000.0
            || !scope.allows_path(path)
            || !scope.allows_frontmatter(metadata.as_ref())
        {
            return Err(VaultError::NoteNotFound(path.to_owned()));
        }
        Ok(())
    }

    async fn check_read_scope(&self, path: &Path) -> VaultResult<()> {
        if self.read_scope.is_none() {
            return Ok(());
        }
        self.read_note(path).await.map(|_| ())
    }

    /// Vault root path (canonicalized).
    pub fn root(&self) -> &Path {
        &self.inner.reported_root
    }

    pub fn watch_enabled(&self) -> bool {
        self.inner.watch_enabled
    }

    /// Native vault-local configuration directory. Remote sources have no local vault home.
    pub fn mcp_home(&self) -> Option<&Path> {
        self.inner
            .filesystem
            .as_ref()
            .map(|_| self.inner.mcp_home.as_path())
    }

    /// Resolved local machine-data directory, absent until remote initialization completes.
    pub fn mcp_data(&self) -> Option<&Path> {
        self.resolved_mcp_data()
    }

    /// Active path exclusion set.
    pub fn exclude(&self) -> &ExcludeSet {
        self.inner
            .exclude
            .get()
            .unwrap_or(&self.inner.initial_exclude)
    }

    /// Access the Tantivy BM25 index (if enabled via `Config::tantivy`).
    pub fn tantivy(&self) -> Option<&TantivyIndex> {
        self.inner.tantivy.as_deref()
    }

    // ── fs delegation ──────────────────────────────────────────────────

    pub async fn list_files(
        &self,
        dir: &Path,
        recursive: bool,
        glob: Option<&str>,
    ) -> VaultResult<Vec<PathBuf>> {
        let files = self
            .storage()
            .await?
            .list_files(dir, recursive, glob)
            .await?;
        self.filter_scoped_listing(files).await
    }

    async fn filter_scoped_listing(&self, files: Vec<PathBuf>) -> VaultResult<Vec<PathBuf>> {
        if !self.is_scoped() {
            return Ok(files);
        }
        let vault = self.clone();
        source_work(move || {
            let index = vault.read_index();
            Ok(files
                .into_iter()
                .filter(|path| {
                    index.notes().keys().any(|note| {
                        let note = Path::new(note);
                        note == path || note.starts_with(path)
                    })
                })
                .collect())
        })
        .await
    }

    pub(crate) async fn list_regular_files(
        &self,
        dir: &Path,
        recursive: bool,
        glob: Option<&str>,
    ) -> VaultResult<Vec<PathBuf>> {
        let files = self
            .storage()
            .await?
            .list_regular_files(dir, recursive, glob)
            .await?;
        self.filter_scoped_listing(files).await
    }

    pub async fn read_note(&self, path: &Path) -> VaultResult<String> {
        if self.is_scoped() {
            let vault = self.clone();
            let path = path.to_owned();
            return source_work(move || vault.read_scoped_native(&path, None)).await;
        }
        self.storage().await?.read_file(path).await
    }

    pub(crate) async fn read_note_bounded(
        &self,
        path: &Path,
        max_bytes: usize,
    ) -> VaultResult<String> {
        if self.is_scoped() {
            let vault = self.clone();
            let path = path.to_owned();
            return source_work(move || vault.read_scoped_native(&path, Some(max_bytes))).await;
        }
        self.storage()
            .await?
            .read_file_bounded(path, max_bytes)
            .await
    }

    /// Observe one bounded raw note revision for source-bound context reads.
    pub async fn read_source_snapshot(&self, path: &Path) -> VaultResult<source::SourceSnapshot> {
        let raw = self
            .read_note_bounded(path, source::MAX_SOURCE_BYTES)
            .await?;
        tokio::task::spawn_blocking(move || source::SourceSnapshot::new(raw))
            .await
            .map_err(|e| VaultError::Other(format!("source snapshot task failed: {e}")))
    }

    pub async fn file_stat(&self, path: &Path) -> VaultResult<FileStat> {
        self.check_read_scope(path).await?;
        self.storage().await?.file_stat(path).await
    }

    pub async fn write_note(&self, path: &Path, content: &str) -> VaultResult<()> {
        let path = path.to_owned();
        let content = content.to_owned();
        self.run_mutation(move |vault| async move { vault.write_note_inner(&path, &content).await })
            .await
    }

    pub async fn write_note_if_revision(
        &self,
        path: &Path,
        content: &str,
        expected_revision: &str,
    ) -> VaultResult<()> {
        let path = path.to_owned();
        let content = content.to_owned();
        let expected = expected_revision.to_owned();
        self.run_mutation(move |vault| async move {
            let content = validate_source(content).await?;
            let actual = vault
                .storage()
                .await?
                .write_file_if_revision(&path, &content, &expected)
                .await?;
            vault
                .reindex(&actual)
                .await
                .map_err(|error| vault.committed_error(&actual, error))
        })
        .await
    }

    async fn write_note_inner(&self, path: &Path, content: &str) -> VaultResult<()> {
        let content = validate_source(content.to_owned()).await?;
        let actual_path = self.storage().await?.write_file(path, &content).await?;
        self.reindex(&actual_path)
            .await
            .map_err(|error| self.committed_error(&actual_path, error))?;
        Ok(())
    }

    pub async fn append_note(&self, path: &Path, content: &str) -> VaultResult<()> {
        let path = path.to_owned();
        let content = content.to_owned();
        self.run_mutation(
            move |vault| async move { vault.append_note_inner(&path, &content).await },
        )
        .await
    }

    async fn append_note_inner(&self, path: &Path, content: &str) -> VaultResult<()> {
        let storage = self.storage().await?;
        let existing = match storage.read_file(path).await {
            Ok(existing) => Some(existing),
            Err(VaultError::NoteNotFound(_)) => None,
            Err(error) => return Err(error),
        };
        let append = content.to_owned();
        let (combined, expected) = source_work(move || {
            let expected = existing.as_ref().map(|raw| source::source_revision(raw));
            let combined = existing.unwrap_or_default() + &append;
            frontmatter::parse_frontmatter(&combined)?;
            Ok((combined, expected))
        })
        .await?;
        let actual_path = if let Some(root) = storage.local_root() {
            let root = root.clone();
            let path = path.to_owned();
            let content = content.to_owned();
            source_work(move || root.append_file(&path, &content)).await?
        } else if let Some(expected) = expected {
            storage
                .write_file_if_revision(path, &combined, &expected)
                .await?
        } else {
            storage.create_file(path, &combined).await?
        };
        self.reindex(&actual_path)
            .await
            .map_err(|error| self.committed_error(&actual_path, error))?;
        Ok(())
    }

    /// Create a new note. Returns `AlreadyExists` if the path is occupied.
    /// Optionally prepends YAML frontmatter before the body content.
    pub async fn create_note(
        &self,
        path: &Path,
        content: &str,
        frontmatter: Option<&serde_json::Value>,
    ) -> VaultResult<()> {
        let path = path.to_owned();
        let content = content.to_owned();
        let frontmatter = frontmatter.cloned();
        self.run_mutation(move |vault| async move {
            vault
                .create_note_inner(&path, &content, frontmatter.as_ref())
                .await
        })
        .await
    }

    async fn create_note_inner(
        &self,
        path: &Path,
        content: &str,
        frontmatter: Option<&serde_json::Value>,
    ) -> VaultResult<()> {
        if let Some(value) = frontmatter
            && !value.is_object()
        {
            return Err(VaultError::InvalidFrontmatter {
                path: path.to_path_buf(),
                actual: frontmatter::json_type_name(value),
            });
        }
        let body = content.to_owned();
        let frontmatter = frontmatter.cloned();
        let full_content = source_work(move || {
            let content = frontmatter::rebuild_content(frontmatter.as_ref(), &body);
            frontmatter::parse_frontmatter(&content)?;
            Ok(content)
        })
        .await?;
        let actual_path = self
            .storage()
            .await?
            .create_file(path, &full_content)
            .await?;
        self.reindex(&actual_path)
            .await
            .map_err(|error| self.committed_error(&actual_path, error))?;
        Ok(())
    }

    /// Prepend content after frontmatter (or at the start if none exists).
    pub async fn prepend_note(&self, path: &Path, content: &str) -> VaultResult<()> {
        let path = path.to_owned();
        let content = content.to_owned();
        self.run_mutation(
            move |vault| async move { vault.prepend_note_inner(&path, &content).await },
        )
        .await
    }

    async fn prepend_note_inner(&self, path: &Path, content: &str) -> VaultResult<()> {
        let existing = self.storage().await?.read_file(path).await?;
        let content = content.to_owned();
        let (new_content, expected) = source_work(move || {
            let expected = source::source_revision(&existing);
            let new_content = match frontmatter::extract_raw_frontmatter(&existing) {
                Some((_, body_start)) => {
                    let mut result = String::with_capacity(existing.len() + content.len());
                    result.push_str(&existing[..body_start]);
                    result.push_str(&content);
                    result.push_str(&existing[body_start..]);
                    result
                }
                None => format!("{content}{existing}"),
            };
            frontmatter::parse_frontmatter(&new_content)?;
            Ok((new_content, expected))
        })
        .await?;
        let actual_path = self
            .storage()
            .await?
            .write_file_if_revision(path, &new_content, &expected)
            .await?;
        self.reindex(&actual_path)
            .await
            .map_err(|error| self.committed_error(&actual_path, error))?;
        Ok(())
    }

    pub async fn delete_note(&self, path: &Path) -> VaultResult<()> {
        let path = path.to_owned();
        self.run_mutation(move |vault| async move { vault.delete_note_inner(&path).await })
            .await
    }

    async fn delete_note_inner(&self, path: &Path) -> VaultResult<()> {
        let actual_path = self.storage().await?.delete_file(path).await?;
        self.reindex(&actual_path)
            .await
            .map_err(|error| self.committed_error(&actual_path, error))
    }

    pub async fn move_note(&self, from: &Path, to: &Path) -> VaultResult<PathBuf> {
        let from = from.to_owned();
        let to = to.to_owned();
        self.run_mutation(move |vault| async move { vault.move_note_inner(&from, &to).await })
            .await
    }

    async fn move_note_inner(&self, from: &Path, to: &Path) -> VaultResult<PathBuf> {
        let move_result = self.storage().await?.move_file(from, to).await?;
        let old_result = self.reindex(&move_result.from).await;
        let new_result = self.reindex(&move_result.to).await;
        old_result
            .and(new_result)
            .map_err(|error| self.committed_error(&move_result.to, error))?;
        Ok(move_result.to)
    }

    // ── patch delegation ───────────────────────────────────────────────

    pub async fn patch_note(&self, path: &Path, request: &PatchRequest) -> VaultResult<()> {
        let path = path.to_owned();
        let request = request.clone();
        self.run_mutation(move |vault| async move { vault.patch_note_inner(&path, &request).await })
            .await
    }

    async fn patch_note_inner(&self, path: &Path, request: &PatchRequest) -> VaultResult<()> {
        let content = self.storage().await?.read_file(path).await?;
        let request = request.clone();
        let source_path = path.to_owned();
        let (patched, expected) = source_work(move || {
            let expected = source::source_revision(&content);
            let patched = patch::apply_patch(&content, &request, &source_path)?;
            frontmatter::parse_frontmatter(&patched)?;
            Ok((patched, expected))
        })
        .await?;
        let actual_path = self
            .storage()
            .await?
            .write_file_if_revision(path, &patched, &expected)
            .await?;
        self.reindex(&actual_path)
            .await
            .map_err(|error| self.committed_error(&actual_path, error))?;
        Ok(())
    }

    // ── frontmatter delegation ─────────────────────────────────────────

    pub async fn get_frontmatter(&self, path: &Path) -> VaultResult<Option<serde_json::Value>> {
        let content = self.read_note(path).await?;
        source_work(move || frontmatter::parse_frontmatter(&content)).await
    }

    pub async fn set_frontmatter_field(
        &self,
        path: &Path,
        key: &str,
        value: serde_json::Value,
    ) -> VaultResult<()> {
        let path = path.to_owned();
        let key = key.to_owned();
        self.run_mutation(move |vault| async move {
            vault.set_frontmatter_field_inner(&path, &key, value).await
        })
        .await
    }

    async fn set_frontmatter_field_inner(
        &self,
        path: &Path,
        key: &str,
        value: serde_json::Value,
    ) -> VaultResult<()> {
        let content = self.storage().await?.read_file(path).await?;
        let key = key.to_owned();
        let (updated, expected) = source_work(move || {
            let expected = source::source_revision(&content);
            let updated = frontmatter::set_frontmatter_field(&content, &key, value)?;
            frontmatter::parse_frontmatter(&updated)?;
            Ok((updated, expected))
        })
        .await?;
        let actual_path = self
            .storage()
            .await?
            .write_file_if_revision(path, &updated, &expected)
            .await?;
        self.reindex(&actual_path)
            .await
            .map_err(|error| self.committed_error(&actual_path, error))?;
        Ok(())
    }

    pub async fn remove_frontmatter_field(&self, path: &Path, key: &str) -> VaultResult<()> {
        let path = path.to_owned();
        let key = key.to_owned();
        self.run_mutation(move |vault| async move {
            vault.remove_frontmatter_field_inner(&path, &key).await
        })
        .await
    }

    async fn remove_frontmatter_field_inner(&self, path: &Path, key: &str) -> VaultResult<()> {
        let content = self.storage().await?.read_file(path).await?;
        let key = key.to_owned();
        let (updated, expected) = source_work(move || {
            let expected = source::source_revision(&content);
            let updated = frontmatter::remove_frontmatter_field(&content, &key)?;
            frontmatter::parse_frontmatter(&updated)?;
            Ok((updated, expected))
        })
        .await?;
        let actual_path = self
            .storage()
            .await?
            .write_file_if_revision(path, &updated, &expected)
            .await?;
        self.reindex(&actual_path)
            .await
            .map_err(|error| self.committed_error(&actual_path, error))?;
        Ok(())
    }

    // ── index delegation (read-lock) ───────────────────────────────────

    pub(crate) fn indexed_paths(&self) -> Vec<String> {
        self.read_index()
            .notes()
            .keys()
            .map(|path| path.to_string_lossy().into_owned())
            .collect()
    }

    pub async fn get_note_metadata(&self, path: &Path) -> VaultResult<NoteMetadata> {
        self.require_catalog()?;
        let actual_path = self.canonical_existing_relative_path(path).await?;
        self.read_index()
            .get_note(&actual_path)
            .cloned()
            .ok_or(VaultError::NoteNotFound(actual_path))
    }

    /// Look up metadata for exact index paths under a single read lock.
    ///
    /// This intentionally skips filesystem path resolution and is only for paths
    /// already returned by vault APIs such as [`Vault::list_files`]. Raw caller
    /// paths must continue to use [`Vault::get_note_metadata`].
    pub(crate) fn get_indexed_note_metadata_batch<'a>(
        &self,
        paths: impl IntoIterator<Item = &'a Path>,
    ) -> Vec<Option<NoteListMetadata>> {
        let index = self.read_index();
        paths
            .into_iter()
            .map(|path| {
                index.get_note(path).map(|metadata| NoteListMetadata {
                    title: metadata.title.clone(),
                    tags: metadata.tags.clone(),
                    stat: metadata.stat.clone(),
                })
            })
            .collect()
    }

    pub async fn get_document_map(&self, path: &Path) -> VaultResult<DocumentMap> {
        let content = self.read_note(path).await?;
        tokio::task::spawn_blocking(move || parser::build_document_map(&content))
            .await
            .map_err(|e| VaultError::Other(format!("document inspection task failed: {e}")))
    }

    pub async fn search_text(
        &self,
        query: &str,
        context_len: usize,
    ) -> VaultResult<Vec<SearchResult>> {
        let max_results = if self.inner.tantivy.is_some() {
            200
        } else {
            usize::MAX
        };
        self.search_text_with_options(query, context_len, max_results, false, None)
            .await
    }
    pub async fn search_text_with_options(
        &self,
        query: &str,
        context_len: usize,
        max_results: usize,
        fuzzy: bool,
        fields: Option<&[SearchField]>,
    ) -> VaultResult<Vec<SearchResult>> {
        self.require_catalog()?;
        if let Some(tv) = &self.inner.tantivy {
            return self
                .tantivy_search_with_context(tv, query, context_len, max_results, fuzzy, fields)
                .await;
        }
        if query.is_empty() {
            return Ok(Vec::new());
        }
        let regex = regex::RegexBuilder::new(&regex::escape(query))
            .case_insensitive(true)
            .size_limit(1 << 20)
            .build()
            .map_err(|source| VaultError::InvalidRegex {
                pattern: query.into(),
                source,
            })?;
        self.scan_compiled_regex(regex, context_len, Some(max_results))
            .await
    }
    pub(crate) async fn search_text_with_limit(
        &self,
        query: &str,
        context_len: usize,
        max_results: usize,
        fuzzy: bool,
        fields: Option<&[SearchField]>,
    ) -> VaultResult<Vec<SearchResult>> {
        self.search_text_with_options(query, context_len, max_results, fuzzy, fields)
            .await
    }
    pub(crate) async fn search_regex_with_limit(
        &self,
        pattern: &str,
        context_len: usize,
        max_results: usize,
    ) -> VaultResult<Vec<SearchResult>> {
        self.scan_regex(pattern, context_len, Some(max_results))
            .await
    }
    pub async fn search_regex(
        &self,
        pattern: &str,
        context_len: usize,
    ) -> VaultResult<Vec<SearchResult>> {
        self.scan_regex(pattern, context_len, None).await
    }
    async fn scan_regex(
        &self,
        pattern: &str,
        context_len: usize,
        limit: Option<usize>,
    ) -> VaultResult<Vec<SearchResult>> {
        self.require_catalog()?;
        if pattern.len() > 1000 {
            return Err(VaultError::InvalidRegex {
                pattern: pattern.into(),
                source: regex::Error::Syntax(
                    "pattern exceeds maximum length of 1000 characters".into(),
                ),
            });
        }
        let regex = regex::RegexBuilder::new(pattern)
            .size_limit(1 << 20)
            .build()
            .map_err(|source| VaultError::InvalidRegex {
                pattern: pattern.into(),
                source,
            })?;
        self.scan_compiled_regex(regex, context_len, limit).await
    }

    async fn scan_compiled_regex(
        &self,
        regex: regex::Regex,
        context_len: usize,
        limit: Option<usize>,
    ) -> VaultResult<Vec<SearchResult>> {
        if limit == Some(0) {
            return Ok(Vec::new());
        }
        let mut paths = self
            .read_index()
            .notes()
            .keys()
            .map(PathBuf::from)
            .collect::<Vec<_>>();
        paths.sort_by(|left, right| left.as_os_str().cmp(right.as_os_str()));
        let source = self.storage().await?;
        Ok(self
            .search_source_matches(&source, paths, regex, context_len, limit)
            .await?
            .into_iter()
            .filter(|(_, matches)| !matches.is_empty())
            .map(|(path, matches)| SearchResult {
                path,
                matches,
                score: None,
            })
            .collect())
    }

    /// Semantic search via embedding cosine similarity (Layer 2).
    ///
    /// Embeds the query, then performs brute-force cosine similarity against
    /// the embedding store. Returns `(path, score)` pairs sorted by descending
    /// similarity.
    #[cfg(has_embeddings)]
    pub fn search_semantic(&self, query: &str, top_k: usize) -> VaultResult<Vec<(PathBuf, f32)>> {
        Ok(self
            .search_semantic_ranked(query, top_k)?
            .into_iter()
            .map(|hit| (hit.path, hit.score))
            .collect())
    }

    #[cfg(has_embeddings)]
    pub(crate) fn search_semantic_ranked(
        &self,
        query: &str,
        top_k: usize,
    ) -> VaultResult<Vec<semantic_results::RankedNote>> {
        self.require_semantic_catalog()?;
        if query.is_empty() || top_k == 0 {
            return Ok(Vec::new());
        }
        let runtime = self.inner.embedding_runtime.get().ok_or_else(|| {
            VaultError::Embedding("embeddings not enabled (OBSIDIAN_EMBEDDINGS=false)".into())
        })?;
        let snapshot = runtime.query_snapshot()?;
        let current_paths = self
            .read_index()
            .notes()
            .keys()
            .cloned()
            .collect::<std::collections::HashSet<_>>();
        let mut entries = snapshot.capture_entries(&current_paths);
        self.filter_scoped_semantic_entries(&mut entries);
        if entries.is_empty() {
            snapshot.validate_query(query)?;
            self.require_semantic_catalog()?;
            return Ok(Vec::new());
        }
        let query_vector = snapshot.embed_query(query)?;
        let ranked = semantic_results::rank_semantic(entries, &query_vector, top_k)?;
        self.require_semantic_catalog()?;
        Ok(ranked)
    }

    /// Returns `true` if embeddings are available for semantic search.
    #[cfg(has_embeddings)]
    pub fn has_embeddings(&self) -> bool {
        self.inner
            .embedding_runtime
            .get()
            .is_some_and(|runtime| runtime.status().queryable)
    }

    #[cfg(has_embeddings)]
    pub(crate) fn embeddings_configured(&self) -> bool {
        self.inner.embedding_runtime.get().is_some()
    }

    #[cfg(has_embeddings)]
    pub(crate) fn embeddings_requested(&self) -> bool {
        self.inner.embeddings_requested
    }

    /// Returns the error message from a failed embedding model load, if any.
    pub fn embedding_load_error(&self) -> Option<&str> {
        #[cfg(has_embeddings)]
        {
            self.inner
                .embedding_runtime
                .get()
                .and_then(embedding_runtime::EmbeddingRuntime::first_load_error)
        }
        #[cfg(not(has_embeddings))]
        {
            None
        }
    }

    /// Current background embedding lifecycle status, when embeddings are enabled.
    #[cfg(has_embeddings)]
    pub fn embedding_status(&self) -> Option<embedding_runtime::EmbeddingRuntimeStatus> {
        self.inner
            .embedding_runtime
            .get()
            .map(embedding_runtime::EmbeddingRuntime::status)
    }

    /// Hybrid search: BM25 prefetch via Tantivy, then re-rank by combining
    /// normalized BM25 scores with semantic cosine similarity.
    ///
    /// Requires both Tantivy and embeddings to be enabled.
    ///
    /// `alpha` controls the balance: `final = alpha * norm_bm25 + (1-alpha) * cosine_sim`.
    /// Lower alpha = more weight to semantic meaning.
    #[cfg(has_embeddings)]
    pub fn search_hybrid(
        &self,
        query: &str,
        top_k: usize,
        prefetch_count: usize,
        alpha: f32,
    ) -> VaultResult<Vec<(PathBuf, f32)>> {
        Ok(self
            .search_hybrid_ranked(query, top_k, prefetch_count, alpha)?
            .into_iter()
            .map(|hit| (hit.path, hit.score))
            .collect())
    }

    #[cfg(has_embeddings)]
    pub(crate) fn search_hybrid_ranked(
        &self,
        query: &str,
        top_k: usize,
        prefetch_count: usize,
        alpha: f32,
    ) -> VaultResult<Vec<semantic_results::RankedNote>> {
        self.require_semantic_catalog()?;
        if query.is_empty() || top_k == 0 {
            return Ok(Vec::new());
        }

        let tv = self.inner.tantivy.as_ref().ok_or_else(|| {
            VaultError::Other("hybrid search requires Tantivy (set OBSIDIAN_TANTIVY=true)".into())
        })?;
        let runtime = self.inner.embedding_runtime.get().ok_or_else(|| {
            VaultError::Embedding("embeddings not enabled (OBSIDIAN_EMBEDDINGS=false)".into())
        })?;
        let snapshot = runtime.query_snapshot()?;
        snapshot.validate_query(query)?;
        let current_paths = self.read_index().notes().keys().cloned().collect();
        let mut entries = snapshot.capture_entries(&current_paths);
        self.filter_scoped_semantic_entries(&mut entries);
        if entries.is_empty() {
            self.require_semantic_catalog()?;
            return Ok(Vec::new());
        }
        let allowed = semantic_results::paths(&entries);
        let prefetch = prefetch_count.max(top_k);
        let bm25_hits = tv.search_for_paths(query, prefetch, Some(&allowed))?;
        if bm25_hits.is_empty() {
            self.require_semantic_catalog()?;
            return Ok(Vec::new());
        }

        let query_vec = snapshot.embed_query(query)?;
        let ranked = semantic_results::rank_hybrid(
            &entries,
            &bm25_hits,
            &query_vec,
            alpha.clamp(0.0, 1.0),
            top_k,
        )?;
        self.require_semantic_catalog()?;
        Ok(ranked)
    }

    pub fn search_by_tag(&self, tag: &str) -> VaultResult<Vec<NoteMetadata>> {
        self.require_catalog()?;
        Ok(self
            .read_index()
            .notes_with_tag(tag)
            .into_iter()
            .cloned()
            .collect())
    }

    pub fn search_by_tag_prefix(&self, tag: &str) -> VaultResult<Vec<NoteMetadata>> {
        self.require_catalog()?;
        Ok(self
            .read_index()
            .notes_with_tag_prefix(tag)
            .into_iter()
            .cloned()
            .collect())
    }

    pub fn search_frontmatter(
        &self,
        field: &str,
        value: &serde_json::Value,
    ) -> VaultResult<Vec<NoteMetadata>> {
        self.require_catalog()?;
        Ok(self
            .read_index()
            .search_frontmatter(field, value)
            .into_iter()
            .cloned()
            .collect())
    }

    pub fn search_frontmatter_exists(&self, field: &str) -> VaultResult<Vec<NoteMetadata>> {
        self.require_catalog()?;
        Ok(self
            .read_index()
            .search_frontmatter_exists(field)
            .into_iter()
            .cloned()
            .collect())
    }

    pub fn search_frontmatter_contains(
        &self,
        field: &str,
        value: &serde_json::Value,
    ) -> VaultResult<Vec<NoteMetadata>> {
        self.require_catalog()?;
        Ok(self
            .read_index()
            .search_frontmatter_contains(field, value)
            .into_iter()
            .cloned()
            .collect())
    }

    pub async fn backlinks(&self, path: &Path) -> VaultResult<Vec<NoteMetadata>> {
        self.require_catalog()?;
        let actual_path = self.canonical_existing_relative_path(path).await?;
        Ok(self
            .read_index()
            .backlinks_to(&actual_path)
            .into_iter()
            .cloned()
            .collect())
    }

    pub async fn outgoing_links(&self, path: &Path) -> VaultResult<Vec<WikiLink>> {
        self.require_catalog()?;
        let actual_path = self.canonical_existing_relative_path(path).await?;
        Ok(self
            .read_index()
            .outgoing_links(&actual_path)
            .into_iter()
            .cloned()
            .collect())
    }

    pub fn broken_links(&self) -> VaultResult<Vec<(PathBuf, WikiLink)>> {
        self.require_catalog()?;
        Ok(self.read_index().broken_links())
    }

    pub fn resolve_link(&self, target: &str) -> Option<PathBuf> {
        self.read_index().resolve_link(target)
    }

    pub fn orphan_notes(&self) -> VaultResult<Vec<NoteMetadata>> {
        self.require_catalog()?;
        Ok(self
            .read_index()
            .orphan_notes()
            .into_iter()
            .cloned()
            .collect())
    }

    pub fn vault_stats(&self) -> VaultResult<VaultStats> {
        self.require_catalog()?;
        Ok(self.read_index().stats().clone())
    }

    pub(crate) async fn canonical_existing_relative_path(
        &self,
        path: &Path,
    ) -> VaultResult<PathBuf> {
        let actual = self.storage().await?.resolve_existing(path).await?.relative;
        self.check_read_scope(&actual).await?;
        Ok(actual)
    }

    /// Validate that a relative path doesn't escape the vault root.
    pub async fn validate_path(&self, path: &Path) -> VaultResult<()> {
        self.storage().await?.resolve_for_write(path).await?;
        Ok(())
    }

    // ── periodic delegation ────────────────────────────────────────────

    pub async fn get_periodic_note(
        &self,
        period: &NotePeriod,
        date: Option<NaiveDate>,
    ) -> VaultResult<String> {
        let config = self.periodic_config(period).await?;
        let date = date.unwrap_or_else(|| Local::now().date_naive());
        let path = periodic::periodic_note_path(&config, &date)?;
        self.read_note(&path).await
    }

    pub async fn create_periodic_note(
        &self,
        period: &NotePeriod,
        date: Option<NaiveDate>,
        content_override: Option<&str>,
    ) -> VaultResult<PathBuf> {
        let period = period.clone();
        let content = content_override.map(str::to_owned);
        self.run_mutation(move |vault| async move {
            vault
                .create_periodic_note_inner(&period, date, content.as_deref())
                .await
        })
        .await
    }

    async fn create_periodic_note_inner(
        &self,
        period: &NotePeriod,
        date: Option<NaiveDate>,
        content_override: Option<&str>,
    ) -> VaultResult<PathBuf> {
        let config = self.periodic_config(period).await?;
        let date = date.unwrap_or_else(|| Local::now().date_naive());
        let path = periodic::periodic_note_path(&config, &date)?;

        let content = if let Some(custom) = content_override {
            custom.to_owned()
        } else {
            match &config.template {
                Some(tmpl) if !tmpl.is_empty() => {
                    let title = path::logical_file_stem(&path).unwrap_or_default();
                    match self.expand_template(Path::new(tmpl), &date, title).await {
                        Ok(c) => c,
                        Err(VaultError::NoteNotFound(_)) => String::new(),
                        Err(e) => return Err(e),
                    }
                }
                _ => String::new(),
            }
        };

        let content = validate_source(content).await?;
        let actual_path = self.storage().await?.create_file(&path, &content).await?;
        self.reindex(&actual_path)
            .await
            .map_err(|error| self.committed_error(&actual_path, error))?;
        Ok(actual_path)
    }

    pub async fn list_recent_periodic_notes(
        &self,
        period: &NotePeriod,
        limit: usize,
    ) -> VaultResult<Vec<PathBuf>> {
        let config = self.periodic_config(period).await?;
        self.list_periodic(&config, limit).await
    }

    // ── private helpers ────────────────────────────────────────────────

    /// Two-phase Tantivy search: BM25 ranking then context extraction.
    ///
    /// 1. Rank: Tantivy returns top-K `(path, score)` via BM25.
    /// 2. Context: For each hit, read the file and locate query words with a
    ///    case-insensitive regex to produce `SearchMatch` snippets.
    ///
    /// If the query matched only through stemming (no literal occurrence of
    /// any query word), the `matches` vec will be empty but `score` is populated.
    async fn tantivy_search_with_context(
        &self,
        tv: &Arc<TantivyIndex>,
        query: &str,
        context_len: usize,
        max_results: usize,
        fuzzy: bool,
        fields: Option<&[SearchField]>,
    ) -> VaultResult<Vec<SearchResult>> {
        self.require_catalog()?;
        let tv = Arc::clone(tv);
        let search_query = query.to_owned();
        let fields = fields.map(<[SearchField]>::to_vec);
        let vault = self.clone();
        let hits = source_work(move || {
            let allowed = vault.read_scope.as_ref().map(|_| {
                vault
                    .read_index()
                    .notes()
                    .keys()
                    .cloned()
                    .collect::<HashSet<_>>()
            });
            if fuzzy || fields.is_some() {
                tv.search_with_options_for_paths(
                    &search_query,
                    max_results,
                    fuzzy,
                    fields.as_deref(),
                    allowed.as_ref(),
                )
            } else {
                tv.search_for_paths(&search_query, max_results, allowed.as_ref())
            }
        })
        .await?;

        let word_re = (context_len > 0 && !hits.is_empty())
            .then(|| search_utils::compile_query_word_regex(query))
            .flatten();
        if self.is_scoped() {
            let vault = self.clone();
            return source_work(move || {
                hits.into_iter()
                    .map(|(path, score)| {
                        let content = vault.read_scoped_native(&path, None)?;
                        let matches = word_re.as_ref().map_or_else(Vec::new, |regex| {
                            match_source_content(&content, regex, context_len)
                        });
                        Ok(SearchResult {
                            path,
                            matches,
                            score: Some(score as f64),
                        })
                    })
                    .collect()
            })
            .await;
        }
        let matches = if let Some(regex) = word_re {
            let source = self.storage().await?;
            let paths = hits.iter().map(|(path, _)| path.clone()).collect();
            self.search_source_matches(&source, paths, regex, context_len, None)
                .await?
                .into_iter()
                .map(|(_, matches)| matches)
                .collect::<Vec<_>>()
        } else {
            vec![Vec::new(); hits.len()]
        };
        Ok(hits
            .into_iter()
            .zip(matches)
            .map(|((path, score), matches)| SearchResult {
                path,
                matches,
                score: Some(score as f64),
            })
            .collect())
    }

    async fn search_source_matches(
        &self,
        source: &Storage,
        paths: Vec<PathBuf>,
        regex: regex::Regex,
        context_len: usize,
        limit: Option<usize>,
    ) -> VaultResult<Vec<(PathBuf, Vec<SearchMatch>)>> {
        if self.is_scoped() {
            let vault = self.clone();
            return source_work(move || {
                let mut results = Vec::new();
                let mut matched = 0usize;
                for path in paths {
                    let Ok(content) = vault.read_scoped_native(&path, None) else {
                        continue;
                    };
                    let matches = match_source_content(&content, &regex, context_len);
                    matched += usize::from(!matches.is_empty());
                    results.push((path, matches));
                    if limit.is_some_and(|limit| matched >= limit) {
                        break;
                    }
                }
                Ok(results)
            })
            .await;
        }
        let batch = source.observe_batch(&paths).await?;
        let remote = source.is_remote();
        let candidates = {
            let index = self.read_index();
            paths
                .into_iter()
                .map(|path| {
                    let bytes = index
                        .get_note(&path)
                        .map(|note| {
                            note.stat.size.min(storage::sftp::MAX_READ_BYTES as u64) as usize
                        })
                        .unwrap_or(storage::sftp::MAX_READ_BYTES);
                    (path, bytes)
                })
                .collect::<Vec<_>>()
        };
        let width = if remote { 16 } else { 1 };
        let mut results = Vec::new();
        let mut matched = 0usize;
        for chunk in candidates.chunks(width) {
            let reads = stream::iter(chunk.iter().cloned().map(|(path, bytes)| {
                let budget = Arc::clone(&self.inner.source_buffers);
                let regex = regex.clone();
                let batch = &batch;
                async move {
                    let reservation = if remote {
                        Some(
                            budget
                                .acquire_many_owned(bytes.max(1).div_ceil(1024) as u32)
                                .await
                                .map_err(|_| {
                                    self.source_error(
                                        crate::error::StorageErrorKind::Unavailable,
                                        "source content admission closed",
                                    )
                                })?,
                        )
                    } else {
                        None
                    };
                    read_search_matches(
                        batch,
                        &path,
                        remote.then_some(bytes),
                        regex,
                        context_len,
                        reservation,
                    )
                    .await
                }
            }))
            .buffered(width);
            // Finish every launched read before returning for an ordinary limit.
            // Dropping dispatched protocol requests would invalidate the session.
            let mut reads = reads.collect::<Vec<_>>().await;
            for ((path, _), result) in chunk.iter().zip(&mut reads) {
                if listing_bound_exceeded(result) {
                    let reservation = Arc::clone(&self.inner.source_buffers)
                        .acquire_many_owned((storage::sftp::MAX_READ_BYTES / 1024) as u32)
                        .await
                        .map_err(|_| {
                            self.source_error(
                                crate::error::StorageErrorKind::Unavailable,
                                "source content admission closed",
                            )
                        })?;
                    *result = read_search_matches(
                        &batch,
                        path,
                        Some(storage::sftp::MAX_READ_BYTES),
                        regex.clone(),
                        context_len,
                        Some(reservation),
                    )
                    .await;
                }
            }
            for ((path, _), result) in chunk.iter().zip(reads) {
                let matches = result?;
                matched += usize::from(!matches.is_empty());
                results.push((path.clone(), matches));
                if limit.is_some_and(|limit| matched >= limit) {
                    return Ok(results);
                }
            }
        }
        Ok(results)
    }

    fn check_mutation_scope(&self) -> VaultResult<()> {
        if self.is_scoped() {
            return Err(VaultError::Other(
                "private memory scope is read-only".into(),
            ));
        }
        Ok(())
    }

    async fn lock_mutations(&self) -> VaultResult<tokio::sync::MutexGuard<'_, ()>> {
        self.check_mutation_scope()?;
        Ok(self.inner.mutations.lock().await)
    }

    fn read_index(&self) -> IndexRead<'_> {
        let index = self.inner.index.read().unwrap_or_else(|e| e.into_inner());
        if let Some(scope) = &self.read_scope {
            let contents = index
                .notes()
                .iter()
                .filter(|(_, metadata)| scope.allows_frontmatter(metadata.frontmatter.as_ref()))
                .filter_map(|(path, _)| {
                    self.read_scoped_native(Path::new(path), None)
                        .ok()
                        .map(|content| (path.clone(), content))
                })
                .collect();
            IndexRead::Scoped(Box::new(index.read_view(contents)))
        } else {
            IndexRead::Shared(index)
        }
    }

    #[cfg(has_embeddings)]
    fn filter_scoped_semantic_entries(&self, entries: &mut semantic_results::CapturedEntries) {
        if self.is_scoped() {
            entries.retain(|path, entry| {
                self.check_scoped_revision_native(Path::new(path), &entry.source_revision)
                    .is_ok()
            });
        }
    }

    fn write_index(&self) -> std::sync::RwLockWriteGuard<'_, VaultIndex> {
        self.inner.index.write().unwrap_or_else(|e| e.into_inner())
    }

    async fn reindex(&self, path: &Path) -> VaultResult<()> {
        if self.inner.filesystem.is_some() {
            let vault = self.clone();
            let path = path.to_owned();
            return tokio::task::spawn_blocking(move || vault.reindex_local(&path))
                .await
                .map_err(|e| VaultError::Other(format!("index refresh task failed: {e}")))?;
        }

        let generation = self.inner.generation.fetch_add(1, Ordering::AcqRel) + 1;
        {
            let mut changed = self.inner.changed.lock().unwrap_or_else(|e| e.into_inner());
            if self.inner.scan_active.load(Ordering::Acquire) {
                changed.insert(path.as_os_str().to_owned(), generation);
            }
        }
        let storage = self.storage().await?;
        let observation = if path::logical_extension(path)
            .is_some_and(|extension| extension.eq_ignore_ascii_case("md"))
            && !self.exclude().is_excluded(path)
        {
            storage
                .read_observed_with_stat(path)
                .await
                .map(|(content, stat)| (Some(content), stat))
        } else {
            storage.file_stat(path).await.map(|stat| (None, stat))
        };
        let observation = match observation {
            Ok(observation) => Some(observation),
            Err(e) if e.is_not_found() => None,
            Err(e) => return Err(e),
        };
        self.apply_source_observation(path.to_owned(), observation, true, None)
            .await
            .map(|_| ())
    }

    fn reindex_local(&self, path: &Path) -> VaultResult<()> {
        let filesystem = self
            .inner
            .filesystem
            .as_ref()
            .ok_or_else(|| VaultError::Other("missing local source".into()))?;
        let mut index = self.write_index();
        let was_note = index.get_note(path).is_some();
        let result = index.refresh_file(
            filesystem,
            path,
            self.exclude(),
            self.inner.tantivy.as_deref(),
            #[cfg(has_embeddings)]
            self.inner.embedding_runtime.get().is_some(),
        );
        #[cfg(has_embeddings)]
        if !matches!(
            &result,
            Err(VaultError::Storage {
                kind: crate::error::StorageErrorKind::ResourceLimit,
                ..
            })
        ) && let Some(runtime) = self.inner.embedding_runtime.get()
            && let Some(change) = index.embedding_change(path, was_note)
        {
            runtime.submit(path, change);
        }
        drop(index);
        let dirty = result.as_ref().copied().unwrap_or(was_note);
        let flush = self
            .inner
            .tantivy
            .as_ref()
            .filter(|_| dirty)
            .map(|tv| tv.publish())
            .transpose();
        #[cfg(has_embeddings)]
        if let Ok(Some(committed)) = &flush
            && let Some(runtime) = self.inner.embedding_runtime.get()
        {
            self.write_index()
                .publish_embedding_upserts(*committed, |path| runtime.submit_upsert(path));
        }
        match (result, flush) {
            (Err(error), Err(flush)) => Err(VaultError::Other(format!(
                "{error}; search flush failed: {flush}"
            ))),
            (Err(error), _) | (_, Err(error)) => Err(error),
            _ => Ok(()),
        }
    }

    fn publish_source(&self) -> VaultResult<()> {
        self.write_index().finish_source_batch();
        if let Some(tv) = &self.inner.tantivy {
            let _committed = tv.publish()?;
            #[cfg(has_embeddings)]
            if let Some(runtime) = self.inner.embedding_runtime.get() {
                self.write_index()
                    .publish_embedding_upserts(_committed, |path| runtime.submit_upsert(path));
            }
        }
        Ok(())
    }

    async fn apply_source_observation(
        &self,
        path: PathBuf,
        observation: Option<(Option<String>, FileStat)>,
        publish: bool,
        reservation: Option<tokio::sync::OwnedSemaphorePermit>,
    ) -> VaultResult<bool> {
        let vault = self.clone();
        tokio::task::spawn_blocking(move || {
            let _reservation = reservation;
            vault.apply_observation(&path, observation, publish)
        })
        .await
        .map_err(|e| VaultError::Other(format!("source projection task failed: {e}")))?
    }

    fn apply_observation(
        &self,
        path: &Path,
        observation: Option<(Option<String>, FileStat)>,
        publish: bool,
    ) -> VaultResult<bool> {
        let unchanged_revision = if !publish
            && !self
                .inner
                .invalid_sources
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .contains_key(path.as_os_str())
        {
            observation
                .as_ref()
                .and_then(|(raw, _)| raw.as_ref())
                .map(|raw| source::source_revision(raw))
        } else {
            None
        };
        let mut index = self.write_index();
        if let (Some(revision), Some((_, stat))) = (&unchanged_revision, &observation)
            && index.refresh_unchanged_note(path, revision, stat.clone())?
        {
            #[cfg(has_embeddings)]
            if let Some(runtime) = self.inner.embedding_runtime.get()
                && !runtime.has_current_or_pending(path, revision)
                && let Some(change) = index.embedding_change(path, true)
            {
                runtime.submit(path, change);
            }
            return Ok(false);
        }
        let was_note = index.get_note(path).is_some();
        let embeddings_enabled = {
            #[cfg(has_embeddings)]
            {
                self.inner.embedding_runtime.get().is_some()
            }
            #[cfg(not(has_embeddings))]
            {
                false
            }
        };
        let result = match observation {
            Some((content, stat)) => index.apply_observation(
                path,
                content,
                stat,
                self.inner.tantivy.as_deref(),
                embeddings_enabled,
            ),
            None => index.remove_projection(path, self.inner.tantivy.as_deref()),
        };
        #[cfg(has_embeddings)]
        if !matches!(
            &result,
            Err(VaultError::Storage {
                kind: crate::error::StorageErrorKind::ResourceLimit,
                ..
            })
        ) && let Some(runtime) = self.inner.embedding_runtime.get()
            && let Some(change) = index.embedding_change(path, was_note)
        {
            runtime.submit(path, change);
        }
        drop(index);
        if !publish {
            return result.map(|_| true);
        }
        let dirty = result.as_ref().copied().unwrap_or(was_note);
        let flush = self
            .inner
            .tantivy
            .as_ref()
            .filter(|_| dirty)
            .map(|tv| tv.publish())
            .transpose();
        #[cfg(has_embeddings)]
        if let Ok(Some(committed)) = &flush
            && let Some(runtime) = self.inner.embedding_runtime.get()
        {
            self.write_index()
                .publish_embedding_upserts(*committed, |path| runtime.submit_upsert(path));
        }
        match (result, flush) {
            (Err(error), Err(flush)) => Err(VaultError::Other(format!(
                "{error}; search flush failed: {flush}"
            ))),
            (Err(error), _) | (_, Err(error)) => Err(error),
            _ => Ok(true),
        }
    }

    // ── embedding helpers (feature-gated) ─────────────────────────────
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::models::{PatchOperation, PatchTargetType};
    use crate::test_helpers::{create_test_vault, tantivy_config, test_config};
    use unicode_normalization::UnicodeNormalization;

    #[cfg(unix)]
    #[tokio::test]
    async fn scoped_reads_reject_aliases_to_ungranted_native_paths() {
        let directory = tempfile::tempdir().unwrap();
        let scope_id = "slack-t1-dm-u1";
        let granted = format!("Organize/Work/{scope_id}");
        let target = PathBuf::from(format!("Capture/Raw/{scope_id}/private.md"));
        let alias = PathBuf::from(format!("{granted}/alias.md"));
        std::fs::create_dir_all(directory.path().join(target.parent().unwrap())).unwrap();
        std::fs::create_dir_all(directory.path().join(&granted)).unwrap();
        std::fs::write(
            directory.path().join(&target),
            "---\nmemory_schema: slack-code/1\nsensitivity: normal\ntenant_id: tenant\nstar_label: star\nworkspace_id: T1\naudience_kind: dm\naudience_id: U1\n---\n# Private\nSame audience, outside the path grant.\n",
        )
        .unwrap();
        std::os::unix::fs::symlink(
            directory.path().join(&target),
            directory.path().join(&alias),
        )
        .unwrap();
        let vault = Vault::open(&test_config(directory.path())).await.unwrap();
        assert!(vault.read_note(&alias).await.unwrap().contains("# Private"));
        let scoped = vault
            .with_read_scope(ReadScope {
                version: 1,
                tenant_id: "tenant".into(),
                star_label: "star".into(),
                workspace_id: "T1".into(),
                audience_kind: "dm".into(),
                audience_id: "U1".into(),
                purpose: "conversation".into(),
                scope_id: scope_id.into(),
                allowed_prefixes: vec![format!("{granted}/")],
                expires_at: 4_000_000_000.0,
            })
            .unwrap();
        assert!(matches!(
            scoped.read_note(&alias).await,
            Err(VaultError::NoteNotFound(_))
        ));
        assert!(matches!(
            scoped.read_source_snapshot(&alias).await,
            Err(VaultError::NoteNotFound(_))
        ));
        assert!(matches!(
            scoped.read_note_bounded(&alias, 1024).await,
            Err(VaultError::NoteNotFound(_))
        ));
        assert!(scoped.write_note(&alias, "overwrite").await.is_err());
        assert!(
            vault
                .read_note(&target)
                .await
                .unwrap()
                .contains("# Private")
        );
    }

    #[test]
    fn attached_source_protocol_failures_allow_future_reconciliation() {
        use crate::error::StorageErrorKind;

        for kind in [StorageErrorKind::Protocol, StorageErrorKind::Unsupported] {
            let error = VaultError::storage("scan", "note.md", kind, "injected failure");
            assert!(terminal_source_error(&error, false));
            assert!(!terminal_source_error(&error, true));
        }
        let identity = VaultError::storage(
            "reconnect",
            "",
            StorageErrorKind::IdentityChanged,
            "policy changed",
        );
        assert!(terminal_source_error(&identity, false));
        assert!(terminal_source_error(&identity, true));
        let pressure = VaultError::storage(
            "read",
            "note.md",
            StorageErrorKind::ResourceLimit,
            "source memory budget in use",
        );
        assert!(!terminal_source_error(&pressure, false));
    }

    #[tokio::test]
    async fn catalog_projection_turns_preserve_order_after_later_reads_finish_first() {
        let mut previous = None;
        let mut first = CatalogTurn::next(&mut previous);
        let mut second = CatalogTurn::next(&mut previous);
        let intents = Arc::new(Mutex::new(Vec::new()));
        let later_intents = Arc::clone(&intents);
        let (read_complete, later_read) = tokio::sync::oneshot::channel();
        let later = tokio::spawn(async move {
            read_complete.send(()).unwrap();
            second.wait().await;
            later_intents.lock().unwrap().push(("b.md", 2));
        });
        later_read.await.unwrap();
        assert!(intents.lock().unwrap().is_empty());
        first.wait().await;
        intents.lock().unwrap().push(("a.md", 1));
        drop(first);
        later.await.unwrap();
        assert_eq!(*intents.lock().unwrap(), [("a.md", 1), ("b.md", 2)]);
    }

    #[tokio::test]
    async fn reconnect_authentication_errors_preserve_existing_catalog_projections() {
        let directory = tempfile::tempdir().unwrap();
        let path = Path::new("note.md");
        std::fs::write(directory.path().join(path), "retainedneedle").unwrap();
        let vault = Vault::open(&tantivy_config(directory.path()))
            .await
            .unwrap();
        for operation in ["initialize", "connect", "reconnect", "realpath"] {
            let error = VaultError::storage(
                operation,
                "note.md",
                crate::error::StorageErrorKind::PermissionDenied,
                "credentials unavailable",
            );
            assert!(!invalid_catalog_document(&error));
            assert!(matches!(
                vault
                    .apply_catalog_read(path.to_owned(), Err(error), 0, None)
                    .await,
                Err(VaultError::Storage {
                    kind: crate::error::StorageErrorKind::PermissionDenied,
                    ..
                })
            ));
            assert!(vault.read_index().get_note(path).is_some());
            assert_eq!(
                vault.search_text("retainedneedle", 0).await.unwrap().len(),
                1
            );
            assert!(vault.inner.invalid_sources.lock().unwrap().is_empty());
        }
        for operation in ["open", "read", "stat", "lstat", "fstat"] {
            assert!(invalid_catalog_document(&VaultError::storage(
                operation,
                "note.md",
                crate::error::StorageErrorKind::PermissionDenied,
                "note is unreadable",
            )));
        }
    }

    #[tokio::test]
    async fn committed_projection_errors_preserve_the_underlying_failure_kind() {
        use crate::error::{MutationOutcome, StorageErrorKind};

        let directory = tempfile::tempdir().unwrap();
        let mut config = test_config(directory.path());
        config.vault_path = PathBuf::from("sftp://fixture/vault");
        let vault = Vault::open(&config).await.unwrap();
        vault
            .background
            .worker
            .lock()
            .unwrap()
            .take()
            .unwrap()
            .abort();
        for kind in [
            StorageErrorKind::Unavailable,
            StorageErrorKind::PermissionDenied,
            StorageErrorKind::IdentityChanged,
        ] {
            let error = vault.committed_error(
                Path::new("note.md"),
                VaultError::storage("read", "note.md", kind, "original cause"),
            );
            assert!(matches!(error, VaultError::Storage {
                kind: actual,
                outcome: MutationOutcome::Committed,
                ref message,
                ..
            } if actual == kind && message.contains("original cause")));
        }
        for (error, expected) in [
            (
                VaultError::Other("index flush failed".into()),
                StorageErrorKind::ProjectionFailure,
            ),
            (
                VaultError::InvalidFrontmatter {
                    path: "note.md".into(),
                    actual: "array",
                },
                StorageErrorKind::InvalidSource,
            ),
        ] {
            assert!(matches!(
                vault.committed_error(Path::new("note.md"), error),
                VaultError::Storage {
                    kind,
                    outcome: MutationOutcome::Committed,
                    ..
                } if kind == expected
            ));
        }
    }

    #[tokio::test]
    async fn temporary_source_pressure_preserves_complete_projections_without_watching() {
        let directory = tempfile::tempdir().unwrap();
        let path = Path::new("note.md");
        std::fs::write(directory.path().join(path), "#retainedtag\nretainedneedle").unwrap();
        #[cfg(not(has_embeddings))]
        let vault = Vault::open(&tantivy_config(directory.path()))
            .await
            .unwrap();
        #[cfg(has_embeddings)]
        let vault = {
            let model = Arc::new(TestEmbedder::new());
            let loader: EmbeddingLoaderFuture =
                Box::pin(async move { Ok(model as Arc<dyn embeddings::Embedder>) });
            let mut config = embedding_config(directory.path());
            config.tantivy = true;
            let vault = Vault::open_with_embedding_loader(&config, Some(loader))
                .await
                .unwrap();
            tokio::time::timeout(std::time::Duration::from_secs(5), async {
                while !vault
                    .embedding_status()
                    .is_some_and(|status| status.phase == embedding_runtime::EmbeddingPhase::Ready)
                {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();
            vault
        };
        assert!(!vault.watch_enabled());
        let before = vault.read_index().get_note(path).unwrap().clone();
        let mut verified_stat = before.stat.clone();
        verified_stat.modified = Some(chrono::Utc::now() + chrono::Duration::seconds(5));
        let raw = vault.read_note(path).await.unwrap();
        assert_eq!(
            vault
                .apply_catalog_read(
                    path.to_owned(),
                    Ok(Some((Some(raw), verified_stat.clone()))),
                    0,
                    None,
                )
                .await
                .unwrap(),
            Some(false)
        );
        assert_eq!(
            vault.read_index().get_note(path).unwrap().stat.modified,
            verified_stat.modified
        );
        #[cfg(has_embeddings)]
        assert_eq!(
            vault
                .inner
                .embedding_runtime
                .get()
                .unwrap()
                .pending_kind(path),
            None
        );

        #[cfg(has_embeddings)]
        let capture = || {
            vault
                .inner
                .embedding_runtime
                .get()
                .unwrap()
                .query_snapshot()
                .unwrap()
                .capture_entries(&HashSet::from([path.as_os_str().to_owned()]))
                .remove(path.as_os_str())
                .unwrap()
        };
        #[cfg(has_embeddings)]
        let embedding_before = capture();
        let error = VaultError::storage(
            "read",
            "note.md",
            crate::error::StorageErrorKind::ResourceLimit,
            "source memory budget in use",
        );
        assert!(matches!(
            vault
                .apply_catalog_read(path.to_owned(), Err(error), 0, None)
                .await,
            Err(VaultError::Storage {
                kind: crate::error::StorageErrorKind::ResourceLimit,
                ..
            })
        ));
        let after = vault.read_index().get_note(path).unwrap().clone();
        assert_eq!(after.tags, before.tags);
        assert_eq!(after.stat.size, before.stat.size);
        assert_eq!(
            vault.search_text("retainedneedle", 0).await.unwrap().len(),
            1
        );
        assert!(vault.inner.invalid_sources.lock().unwrap().is_empty());
        #[cfg(has_embeddings)]
        assert!(Arc::ptr_eq(&embedding_before, &capture()));
    }

    #[tokio::test]
    async fn mutation_deadline_bounds_admission_and_serialization_before_dispatch() {
        let directory = tempfile::tempdir().unwrap();
        let vault = Vault::open(&test_config(directory.path())).await.unwrap();
        let held = vault.background.admission.acquire_many(16).await.unwrap();
        let dispatched = Arc::new(AtomicBool::new(false));
        let observed = Arc::clone(&dispatched);
        let result = vault
            .run_mutation_before(
                Some(tokio::time::Instant::now() + std::time::Duration::from_millis(20)),
                move |_| async move {
                    observed.store(true, Ordering::Release);
                    Ok(())
                },
            )
            .await;
        assert!(matches!(
            result,
            Err(VaultError::Storage {
                outcome: crate::error::MutationOutcome::NotDispatched,
                ..
            })
        ));
        assert!(!dispatched.load(Ordering::Acquire));
        drop(held);
        let serialized = vault.inner.mutations.lock().await;
        let observed = Arc::clone(&dispatched);
        let result = vault
            .run_mutation_before(
                Some(tokio::time::Instant::now() + std::time::Duration::from_millis(20)),
                move |_| async move {
                    observed.store(true, Ordering::Release);
                    Ok(())
                },
            )
            .await;
        assert!(matches!(
            result,
            Err(VaultError::Storage {
                outcome: crate::error::MutationOutcome::NotDispatched,
                ..
            })
        ));
        assert!(!dispatched.load(Ordering::Acquire));
        drop(serialized);
        assert_eq!(vault.background.admission.available_permits(), 16);
    }

    #[tokio::test]
    async fn cancelled_mutation_caller_retains_owner_through_index_publication() {
        let directory = tempfile::tempdir().unwrap();
        let vault = Vault::open(&tantivy_config(directory.path()))
            .await
            .unwrap();
        let entered = Arc::new(tokio::sync::Notify::new());
        let released = Arc::new(tokio::sync::Notify::new());
        let ready = entered.notified();
        let owner = vault.clone();
        let started = Arc::clone(&entered);
        let gate = Arc::clone(&released);
        let caller = tokio::spawn(async move {
            owner
                .run_mutation(move |vault| async move {
                    started.notify_one();
                    gate.notified().await;
                    vault
                        .write_note_inner(Path::new("retained.md"), "needle retained")
                        .await
                })
                .await
        });
        ready.await;
        caller.abort();
        let next_owner = vault.clone();
        let next = tokio::spawn(async move {
            next_owner
                .append_note(Path::new("retained.md"), " and next")
                .await
        });
        tokio::task::yield_now().await;
        assert!(
            !next.is_finished(),
            "next mutation must wait for the retained owner"
        );
        released.notify_one();
        next.await.unwrap().unwrap();
        assert_eq!(
            vault.read_note(Path::new("retained.md")).await.unwrap(),
            "needle retained and next"
        );
        let hits = vault.search_text("retained", 0).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].path, Path::new("retained.md"));
    }

    #[tokio::test]
    async fn admitted_mutation_keeps_one_deadline_across_source_operations() {
        let directory = tempfile::tempdir().unwrap();
        let vault = Vault::open(&test_config(directory.path())).await.unwrap();
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_millis(100);
        vault
            .run_mutation_before(Some(deadline), move |_| async move {
                let configured = std::time::Duration::from_secs(30);
                assert_eq!(storage::source_deadline(configured), deadline);
                tokio::time::sleep_until(deadline).await;
                assert_eq!(storage::source_deadline(configured), deadline);
                assert!(
                    tokio::time::timeout_at(
                        storage::source_deadline(configured),
                        std::future::pending::<()>(),
                    )
                    .await
                    .is_err()
                );
                Ok(())
            })
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn legal_growth_after_listing_is_admitted_without_another_scan() {
        let directory = tempfile::tempdir().unwrap();
        let path = Path::new("growing.md");
        std::fs::write(directory.path().join(path), "#old").unwrap();
        let vault = Vault::open(&test_config(directory.path())).await.unwrap();
        let storage = vault.storage().await.unwrap();
        let batch = storage.observe_batch(&[path.to_owned()]).await.unwrap();
        let listed = storage.file_stat(path).await.unwrap();
        let grown = "#grown\nA complete note that grew after the directory listing.\n";
        std::fs::write(directory.path().join(path), grown).unwrap();
        let initial = batch
            .read_with_stat_bounded(path, listed.size as usize)
            .await;
        assert!(listing_bound_exceeded(&initial));
        assert_eq!(
            vault.read_index().get_note(path).unwrap().stat.size,
            listed.size
        );

        let (raw, stat) = batch
            .read_with_stat_bounded(path, storage::sftp::MAX_READ_BYTES)
            .await
            .unwrap();
        let mut present = HashSet::from([path.as_os_str().to_owned()]);
        let mut next = HashMap::from([(path.as_os_str().to_owned(), listed)]);
        let outcome = vault
            .project_catalog_read(path.to_owned(), Ok(Some((Some(raw), stat))), 0, None)
            .await
            .unwrap()
            .unwrap();
        let (_, retry) = outcome.record(&mut present, &mut next);
        assert!(retry.is_none());
        assert_eq!(next[path.as_os_str()].size, grown.len() as u64);
        assert_eq!(vault.read_index().get_note(path).unwrap().tags, ["grown"]);
        assert!(vault.inner.invalid_sources.lock().unwrap().is_empty());

        let oversized = std::fs::OpenOptions::new()
            .write(true)
            .open(directory.path().join(path))
            .unwrap();
        oversized
            .set_len(storage::sftp::MAX_READ_BYTES as u64 + 1)
            .unwrap();
        let rejected = batch
            .read_with_stat_bounded(path, storage::sftp::MAX_READ_BYTES)
            .await;
        assert!(rejected.is_err());
        assert!(!listing_bound_exceeded(&rejected));
    }

    #[tokio::test]
    async fn maximum_search_context_retains_native_matches_and_complete_source() {
        let source = "# Context\n\npréface needle tail\n";
        for ranked in [false, true] {
            let directory = tempfile::tempdir().unwrap();
            std::fs::write(directory.path().join("note.md"), source).unwrap();
            let mut config = test_config(directory.path());
            config.tantivy = ranked;
            let vault = Vault::open(&config).await.unwrap();
            for hits in [
                vault.search_regex("needle", usize::MAX).await.unwrap(),
                vault.search_text("needle", usize::MAX).await.unwrap(),
            ] {
                assert_eq!(hits.len(), 1);
                assert_eq!(hits[0].path, Path::new("note.md"));
                assert_eq!(hits[0].matches.len(), 1);
                let matched = &hits[0].matches[0];
                assert_eq!(matched.context, source);
                let start = source[..source.find("needle").unwrap()].chars().count();
                assert_eq!(matched.match_start, start);
                assert_eq!(matched.match_end, start + "needle".len());
            }
        }
    }

    #[tokio::test]
    async fn literal_text_search_does_not_inherit_the_regexp_pattern_length_cap() {
        let directory = tempfile::tempdir().unwrap();
        let literal = ".+[]?*".repeat(180);
        std::fs::write(directory.path().join("literal.md"), &literal).unwrap();
        let vault = Vault::open(&test_config(directory.path())).await.unwrap();
        let hits = vault.search_text(&literal, 0).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].path, Path::new("literal.md"));
        assert_eq!(hits[0].matches[0].context, literal);
        assert!(matches!(
            vault.search_regex(&literal, 0).await,
            Err(VaultError::InvalidRegex { .. })
        ));
        assert!(matches!(
            vault.search_regex(&"a".repeat(1001), 0).await,
            Err(VaultError::InvalidRegex { .. })
        ));
    }

    #[tokio::test]
    async fn remote_data_locations_are_explicitly_unresolved_before_initialization() {
        let directory = tempfile::tempdir().unwrap();
        let mut config = test_config(directory.path());
        config.vault_path = PathBuf::from("sftp://fixture/vault");
        let vault = Vault::open(&config).await.unwrap();
        vault
            .background
            .worker
            .lock()
            .unwrap()
            .take()
            .unwrap()
            .abort();
        assert_eq!(vault.mcp_data(), None);
        assert_eq!(vault.resolved_mcp_data(), None);
        assert_eq!(vault.mcp_home(), None);
    }

    #[tokio::test]
    async fn watch_disabled_scan_cannot_revert_later_mutations_and_releases_tracking() {
        let directory = tempfile::tempdir().unwrap();
        let mut config = test_config(directory.path());
        config.vault_path = PathBuf::from("sftp://fixture/vault");
        config.watch = false;
        config.tantivy = true;
        let vault = Vault::open(&config).await.unwrap();
        // Use the native source adapter to control observations without a network
        // clock; the vault still executes the remote reconciliation path.
        vault
            .background
            .worker
            .lock()
            .unwrap()
            .take()
            .unwrap()
            .abort();
        assert!(
            vault
                .inner
                .source
                .set(Storage::Local(
                    fs::VaultRoot::open(directory.path()).unwrap()
                ))
                .is_ok()
        );
        vault.inner.catalog_ready.store(true, Ordering::Release);
        vault.update_source_status(|status| {
            status.availability = "available".into();
            status.catalog_phase = "ready".into();
        });
        let existing = Path::new("existing.md");
        let created = Path::new("created.md");
        vault.write_note(existing, "old source").await.unwrap();
        assert!(vault.inner.changed.lock().unwrap().is_empty());
        let scan = vault.begin_source_scan().await;
        let old = vault
            .storage()
            .await
            .unwrap()
            .read_observed_with_stat(existing)
            .await
            .unwrap();
        vault.write_note(existing, "updated source").await.unwrap();
        vault
            .create_note(created, "created source", None)
            .await
            .unwrap();
        assert!(
            vault
                .apply_catalog_read(
                    existing.to_owned(),
                    Ok(Some((Some(old.0), old.1))),
                    scan.start,
                    None,
                )
                .await
                .unwrap()
                .is_none()
        );
        assert!(
            vault
                .apply_catalog_read(created.to_owned(), Ok(None), scan.start, None,)
                .await
                .unwrap()
                .is_none()
        );
        assert_eq!(vault.search_text("updated", 0).await.unwrap().len(), 1);
        assert_eq!(vault.search_text("created", 0).await.unwrap().len(), 1);
        drop(scan);
        assert!(!vault.inner.scan_active.load(Ordering::Acquire));
        assert!(vault.inner.changed.lock().unwrap().is_empty());
        vault.write_note(existing, "idle source").await.unwrap();
        assert!(vault.inner.changed.lock().unwrap().is_empty());

        let entered = Arc::new(tokio::sync::Notify::new());
        let ready = entered.notified();
        let owner = vault.clone();
        let started = Arc::clone(&entered);
        let scanner = tokio::spawn(async move {
            let _scan = owner.begin_source_scan().await;
            started.notify_one();
            std::future::pending::<()>().await;
        });
        ready.await;
        vault
            .write_note(existing, "after cancellation")
            .await
            .unwrap();
        assert!(!vault.inner.changed.lock().unwrap().is_empty());
        scanner.abort();
        assert!(scanner.await.unwrap_err().is_cancelled());
        assert!(!vault.inner.scan_active.load(Ordering::Acquire));
        assert!(vault.inner.changed.lock().unwrap().is_empty());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn regular_listing_keeps_file_aliases_and_rejects_selected_escapes() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("note.md"), "inside").unwrap();
        std::fs::create_dir(directory.path().join("folder")).unwrap();
        std::fs::write(directory.path().join("folder/nested.md"), "nested").unwrap();
        symlink("note.md", directory.path().join("alias.md")).unwrap();
        symlink("folder", directory.path().join("directory-alias")).unwrap();
        let vault = Vault::open(&test_config(directory.path())).await.unwrap();
        let files = vault
            .list_regular_files(Path::new(""), true, None)
            .await
            .unwrap();
        assert_eq!(
            files,
            ["alias.md", "folder/nested.md", "note.md"].map(PathBuf::from)
        );
        assert!(
            vault
                .list_files(Path::new(""), false, None)
                .await
                .unwrap()
                .contains(&PathBuf::from("directory-alias"))
        );
        std::fs::write(outside.path().join("secret.md"), "outside").unwrap();
        symlink(
            outside.path().join("secret.md"),
            directory.path().join("escape.md"),
        )
        .unwrap();
        assert!(matches!(
            vault.list_regular_files(Path::new(""), false, None).await,
            Err(VaultError::OutsideVault(_))
        ));
        assert_eq!(
            vault
                .list_regular_files(Path::new(""), false, Some("alias.md"))
                .await
                .unwrap(),
            [PathBuf::from("alias.md")]
        );
    }

    #[tokio::test]
    async fn uncertain_mutation_withdraws_catalog_and_wakes_repair_without_watching() {
        let directory = tempfile::tempdir().unwrap();
        let mut config = test_config(directory.path());
        config.vault_path = PathBuf::from("sftp://fixture/vault");
        config.watch = false;
        let vault = Vault::open(&config).await.unwrap();
        vault.inner.catalog_ready.store(true, Ordering::Release);
        vault.update_source_status(|status| {
            status.availability = "unavailable".into();
            status.catalog_phase = "degraded".into();
        });
        vault
            .write_index()
            .apply_observation(
                Path::new("cached.md"),
                Some("#cached".into()),
                FileStat {
                    size: 7,
                    created: None,
                    modified: None,
                },
                None,
                false,
            )
            .unwrap();
        #[cfg(has_embeddings)]
        assert!(vault.require_semantic_catalog().is_ok());
        assert!(!vault.begin_reconciliation(true));
        assert!(vault.catalog_ready());
        assert_eq!(vault.source_status().unwrap().catalog_phase, "reconciling");
        assert_eq!(vault.search_by_tag("cached").unwrap().len(), 1);
        let error = VaultError::storage(
            "replace",
            "note.md",
            crate::error::StorageErrorKind::Unavailable,
            "publication acknowledgment was lost",
        )
        .with_mutation_outcome(crate::error::MutationOutcome::Unknown);
        vault.reconcile_after_mutation_error(&error);
        assert!(vault.inner.reconcile_needed.load(Ordering::Acquire));
        #[cfg(has_embeddings)]
        assert!(vault.require_semantic_catalog().is_err());
        vault.inner.reconcile_wakeup.notified().await;
        assert!(matches!(
            vault.require_catalog(),
            Err(VaultError::Storage {
                kind: crate::error::StorageErrorKind::Unavailable,
                ..
            })
        ));
        // Consuming a repair request does not make the previous projection valid.
        vault.inner.reconcile_needed.store(false, Ordering::Release);
        assert!(vault.require_catalog().is_err());
        let identity_error = VaultError::storage(
            "replace",
            "note.md",
            crate::error::StorageErrorKind::IdentityChanged,
            "reopen required",
        )
        .with_mutation_outcome(crate::error::MutationOutcome::Unknown);
        vault.reconcile_after_mutation_error(&identity_error);
        assert!(!vault.inner.reconcile_needed.load(Ordering::Acquire));
        // The background owner has not been polled; dropping the vault cancels
        // initialization without launching a connection to the fixture hostname.
        drop(vault);
    }

    const PERIODIC_PERIODS: [NotePeriod; 5] = [
        NotePeriod::Daily,
        NotePeriod::Weekly,
        NotePeriod::Monthly,
        NotePeriod::Quarterly,
        NotePeriod::Yearly,
    ];

    fn configure_periodic_template(root: &Path, period: &NotePeriod, template: &Path) {
        let key = match period {
            NotePeriod::Daily => "daily",
            NotePeriod::Weekly => "weekly",
            NotePeriod::Monthly => "monthly",
            NotePeriod::Quarterly => "quarterly",
            NotePeriod::Yearly => "yearly",
        };
        let entry = serde_json::json!({
            "enabled": true,
            "format": "YYYY-MM-DD",
            "folder": "Periodic",
            "template": template,
        });
        let (config_path, config) = match period {
            NotePeriod::Daily => (root.join(".obsidian/daily-notes.json"), entry),
            _ => (
                root.join(".obsidian/plugins/periodic-notes/data.json"),
                serde_json::json!({(key): entry}),
            ),
        };
        std::fs::create_dir_all(config_path.parent().unwrap()).unwrap();
        std::fs::write(config_path, serde_json::to_vec(&config).unwrap()).unwrap();
    }

    fn periodic_template_fixture() -> (tempfile::TempDir, PathBuf, PathBuf) {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("vault");
        create_test_vault(&root);
        let canary = directory.path().join("canary.md");
        std::fs::write(&canary, "outside template canary").unwrap();
        (directory, root, canary)
    }

    async fn assert_periodic_template_rejected(
        root: &Path,
        canary: &Path,
        period: &NotePeriod,
        template: &Path,
    ) {
        configure_periodic_template(root, period, template);
        let vault = Vault::open(&test_config(root)).await.unwrap();
        let date = NaiveDate::from_ymd_opt(2026, 1, 2).unwrap();

        let result = vault.create_periodic_note(period, Some(date), None).await;

        assert!(
            matches!(
                result,
                Err(VaultError::OutsideVault(_) | VaultError::InvalidPath(_))
            ),
            "{period:?}: {result:?}"
        );
        assert!(!root.join("Periodic").exists());
        assert_eq!(
            std::fs::read_to_string(canary).unwrap(),
            "outside template canary"
        );
    }

    #[tokio::test]
    async fn periodic_templates_reject_parent_directory_escape() {
        for period in PERIODIC_PERIODS {
            let (_directory, root, canary) = periodic_template_fixture();
            assert_periodic_template_rejected(&root, &canary, &period, Path::new("../canary"))
                .await;
        }
    }

    #[tokio::test]
    async fn periodic_templates_reject_absolute_paths() {
        for period in PERIODIC_PERIODS {
            let (_directory, root, canary) = periodic_template_fixture();
            assert_periodic_template_rejected(&root, &canary, &period, &canary).await;
        }
    }

    #[tokio::test]
    async fn periodic_templates_allow_missing_vault_relative_templates() {
        for period in PERIODIC_PERIODS {
            let (_directory, root, _canary) = periodic_template_fixture();
            configure_periodic_template(&root, &period, Path::new("Templates/missing"));
            let vault = Vault::open(&test_config(&root)).await.unwrap();
            let date = NaiveDate::from_ymd_opt(2026, 1, 2).unwrap();

            let note = vault
                .create_periodic_note(&period, Some(date), None)
                .await
                .unwrap();

            assert!(vault.read_note(&note).await.unwrap().is_empty());
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn periodic_templates_reject_symlinks_outside_the_vault() {
        for period in PERIODIC_PERIODS {
            let (_directory, root, canary) = periodic_template_fixture();
            std::fs::create_dir(root.join("Templates")).unwrap();
            std::os::unix::fs::symlink(&canary, root.join("Templates/linked.md")).unwrap();
            assert_periodic_template_rejected(
                &root,
                &canary,
                &period,
                Path::new("Templates/linked"),
            )
            .await;
        }
    }

    #[tokio::test]
    async fn periodic_templates_resolve_vault_relative_unicode_paths_with_optional_extension() {
        for period in PERIODIC_PERIODS {
            for template in [
                "Templates/Élan",
                "Templates/Élan.md",
                "Templates/E\u{301}lan",
            ] {
                let (_directory, root, _canary) = periodic_template_fixture();
                std::fs::create_dir(root.join("Templates")).unwrap();
                std::fs::write(
                    root.join("Templates/E\u{301}lan.md"),
                    "Created {{date:YYYY-MM-DD}} {{title}}",
                )
                .unwrap();
                configure_periodic_template(&root, &period, Path::new(template));
                let vault = Vault::open(&test_config(&root)).await.unwrap();
                let date = NaiveDate::from_ymd_opt(2026, 1, 2).unwrap();

                let note = vault
                    .create_periodic_note(&period, Some(date), None)
                    .await
                    .unwrap();

                assert_eq!(note, Path::new("Periodic/2026-01-02.md"));
                assert_eq!(
                    vault.read_note(&note).await.unwrap(),
                    "Created 2026-01-02 2026-01-02"
                );
            }
        }
    }

    #[tokio::test]
    async fn periodic_templates_follow_native_case_matching() {
        for period in PERIODIC_PERIODS {
            for template in ["templates/daily", "templates/daily.md"] {
                let (_directory, root, _canary) = periodic_template_fixture();
                std::fs::create_dir(root.join("Templates")).unwrap();
                std::fs::write(root.join("Templates/Daily.md"), "Created {{title}}").unwrap();
                let native_match = root.join("templates/daily.md").is_file();
                configure_periodic_template(&root, &period, Path::new(template));
                let vault = Vault::open(&test_config(&root)).await.unwrap();
                let date = NaiveDate::from_ymd_opt(2026, 1, 2).unwrap();

                let note = vault
                    .create_periodic_note(&period, Some(date), None)
                    .await
                    .unwrap();

                assert_eq!(
                    vault.read_note(&note).await.unwrap(),
                    if native_match {
                        "Created 2026-01-02"
                    } else {
                        ""
                    }
                );
            }
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn periodic_templates_reject_native_case_matched_symlinks_outside_the_vault() {
        for period in PERIODIC_PERIODS {
            let (_directory, root, canary) = periodic_template_fixture();
            std::fs::create_dir(root.join("Templates")).unwrap();
            std::os::unix::fs::symlink(&canary, root.join("Templates/Linked.md")).unwrap();
            if !root.join("templates/linked.md").is_file() {
                continue;
            }

            assert_periodic_template_rejected(
                &root,
                &canary,
                &period,
                Path::new("templates/linked"),
            )
            .await;
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn periodic_templates_allow_symlinks_to_templates_inside_the_vault() {
        for period in PERIODIC_PERIODS {
            let (_directory, root, _canary) = periodic_template_fixture();
            std::fs::create_dir(root.join("Templates")).unwrap();
            std::fs::write(root.join("Templates/inside.md"), "Inside {{title}}").unwrap();
            std::os::unix::fs::symlink("inside.md", root.join("Templates/linked.md")).unwrap();
            configure_periodic_template(&root, &period, Path::new("Templates/linked"));
            let vault = Vault::open(&test_config(&root)).await.unwrap();
            let date = NaiveDate::from_ymd_opt(2026, 1, 2).unwrap();

            let note = vault
                .create_periodic_note(&period, Some(date), None)
                .await
                .unwrap();

            assert_eq!(vault.read_note(&note).await.unwrap(), "Inside 2026-01-02");
        }
    }

    #[cfg(has_embeddings)]
    struct TestEmbedder {
        identity: embeddings::EmbeddingSpaceIdentity,
        inputs: Mutex<Vec<String>>,
    }

    #[cfg(has_embeddings)]
    impl TestEmbedder {
        fn new() -> Self {
            Self {
                identity: embeddings::EmbeddingSpaceIdentity {
                    backend: embeddings::EmbeddingBackendKind::Local,
                    model: "vault-test".into(),
                    endpoint_fingerprint: None,
                    dimension: 2,
                    input_version: embeddings::EMBEDDING_INPUT_VERSION,
                    tokenizer_fingerprint: [1; 32],
                    max_input_tokens: 8192,
                    chunking_version: crate::vault::embedding_input::CHUNKING_VERSION,
                },
                inputs: Mutex::new(Vec::new()),
            }
        }
    }

    #[cfg(has_embeddings)]
    impl embeddings::Embedder for TestEmbedder {
        fn input_policy(&self) -> &crate::vault::embedding_input::EmbeddingInputPolicy {
            &crate::vault::embedding_input::EmbeddingInputPolicy {
                max_input_tokens: 8192,
                tokenizer_fingerprint: [1; 32],
            }
        }

        fn count_tokens(&self, text: &str) -> VaultResult<usize> {
            Ok(text.chars().count())
        }

        fn request_limits(&self) -> crate::vault::embedding_input::EmbeddingRequestLimits {
            crate::vault::embedding_input::EmbeddingRequestLimits {
                max_inputs: 64,
                max_total_tokens: 8192 * 64,
                max_request_bytes: 16 * 1024 * 1024,
            }
        }

        fn dimension(&self) -> usize {
            self.identity.dimension
        }

        fn space_identity(&self) -> &embeddings::EmbeddingSpaceIdentity {
            &self.identity
        }

        fn embed_batch(&self, texts: &[&str]) -> VaultResult<Vec<Vec<f32>>> {
            self.inputs
                .lock()
                .unwrap()
                .extend(texts.iter().map(|text| (*text).to_string()));
            Ok(texts.iter().map(|_| vec![1.0, 0.0]).collect())
        }
    }

    #[cfg(has_embeddings)]
    fn blocked_embedding_loader() -> EmbeddingLoaderFuture {
        Box::pin(std::future::pending())
    }

    #[cfg(has_embeddings)]
    fn embedding_config(vault_root: &Path) -> Config {
        let mut config = test_config(vault_root);
        config.embeddings = true;
        config
    }

    #[tokio::test]
    async fn vault_open_succeeds() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let config = test_config(dir.path());

        let vault = Vault::open(&config).await;
        assert!(vault.is_ok());
    }

    #[tokio::test]
    async fn vault_open_without_obsidian_dir() {
        let dir = tempfile::tempdir().unwrap();
        let config = test_config(dir.path());

        let vault = Vault::open(&config).await;
        assert!(vault.is_ok(), "should succeed even without .obsidian/");
    }

    #[tokio::test]
    async fn vault_open_nonexistent_path() {
        let config = Config {
            vault_path: PathBuf::from("/nonexistent/path/that/does/not/exist"),
            watch: false,
            log_level: "error".into(),
            transport: crate::config::Transport::Stdio,
            http_host: crate::config::DEFAULT_HTTP_HOST,
            http_port: crate::config::DEFAULT_HTTP_PORT,
            tantivy: false,
            embeddings: false,
            embeddings_model: String::new(),
            hybrid_alpha: 0.25,
            embedding_provider: None,
            tool_filter: crate::config::ToolFilter::Full,
            mcp_data_dir: None,
            exclude_patterns: vec![],
        };

        let result = Vault::open(&config).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn vault_read_write_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());

        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        vault
            .write_note(Path::new("hello.md"), "# Hello\nWorld")
            .await
            .unwrap();
        let content = vault.read_note(Path::new("hello.md")).await.unwrap();
        assert_eq!(content, "# Hello\nWorld");
    }

    #[tokio::test]
    async fn vault_append_note() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        vault
            .write_note(Path::new("log.md"), "line one\n")
            .await
            .unwrap();
        vault
            .append_note(Path::new("log.md"), "line two\n")
            .await
            .unwrap();

        let content = vault.read_note(Path::new("log.md")).await.unwrap();
        assert_eq!(content, "line one\nline two\n");
    }

    #[tokio::test]
    async fn vault_delete_removes_from_index() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        vault
            .write_note(Path::new("ephemeral.md"), "# Gone soon")
            .await
            .unwrap();
        assert!(
            vault
                .get_note_metadata(Path::new("ephemeral.md"))
                .await
                .is_ok()
        );

        vault.delete_note(Path::new("ephemeral.md")).await.unwrap();
        assert!(
            vault
                .get_note_metadata(Path::new("ephemeral.md"))
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn vault_move_updates_index() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        vault
            .write_note(Path::new("old.md"), "# Moved")
            .await
            .unwrap();
        assert!(vault.get_note_metadata(Path::new("old.md")).await.is_ok());

        let new_path = vault
            .move_note(Path::new("old.md"), Path::new("subdir/new.md"))
            .await
            .unwrap();
        assert_eq!(new_path, PathBuf::from("subdir/new.md"));

        assert!(vault.get_note_metadata(Path::new("old.md")).await.is_err());
        assert!(
            vault
                .get_note_metadata(Path::new("subdir/new.md"))
                .await
                .is_ok()
        );

        let content = vault.read_note(Path::new("subdir/new.md")).await.unwrap();
        assert_eq!(content, "# Moved");
    }

    #[tokio::test]
    async fn vault_metadata_accepts_canonically_equivalent_unicode_path() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let composed = "02_База-знаний/Сущности/lic1c.md";
        let decomposed: String = composed.nfd().collect();
        let disk_path = PathBuf::from(&decomposed);
        std::fs::create_dir_all(dir.path().join(disk_path.parent().unwrap())).unwrap();
        std::fs::write(dir.path().join(&disk_path), "# License\n").unwrap();

        let vault = Vault::open(&test_config(dir.path())).await.unwrap();
        let meta = vault.get_note_metadata(Path::new(composed)).await.unwrap();

        assert_eq!(meta.path, disk_path);
    }

    #[tokio::test]
    async fn vault_batch_metadata_uses_exact_index_paths_without_resolution() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let composed = "Knowledge/é.md";
        let decomposed: String = composed.nfd().collect();
        let disk_path = PathBuf::from(&decomposed);
        std::fs::create_dir_all(dir.path().join(disk_path.parent().unwrap())).unwrap();
        std::fs::write(dir.path().join(&disk_path), "# Unicode\n").unwrap();

        let vault = Vault::open(&test_config(dir.path())).await.unwrap();
        let metadata = vault.get_indexed_note_metadata_batch([
            disk_path.as_path(),
            Path::new("missing.md"),
            Path::new(composed),
        ]);

        assert_eq!(metadata.len(), 3);
        assert_eq!(
            metadata[0].as_ref().map(|meta| meta.title.as_str()),
            disk_path.file_stem().and_then(|stem| stem.to_str())
        );
        assert!(metadata[1].is_none());
        assert!(metadata[2].is_none());
    }

    #[tokio::test]
    async fn vault_write_delete_and_move_use_unicode_canonical_index_key() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let composed = "02_База-знаний/Сущности/lic1c.md";
        let decomposed: String = composed.nfd().collect();
        let disk_path = PathBuf::from(&decomposed);
        std::fs::create_dir_all(dir.path().join(disk_path.parent().unwrap())).unwrap();
        std::fs::write(dir.path().join(&disk_path), "# Old\n").unwrap();

        let vault = Vault::open(&test_config(dir.path())).await.unwrap();
        let before = vault.vault_stats().unwrap().total_notes;

        vault
            .write_note(Path::new(composed), "# New\nunique-unicode-write\n")
            .await
            .unwrap();
        assert_eq!(vault.vault_stats().unwrap().total_notes, before);
        assert_eq!(
            vault
                .search_text("unique-unicode-write", 40)
                .await
                .unwrap()
                .first()
                .map(|result| result.path.clone()),
            Some(disk_path.clone())
        );

        let moved = vault
            .move_note(Path::new(composed), Path::new("Moved/lic1c.md"))
            .await
            .unwrap();
        assert_eq!(moved, PathBuf::from("Moved/lic1c.md"));
        assert!(vault.get_note_metadata(Path::new(composed)).await.is_err());
        assert!(vault.get_note_metadata(&moved).await.is_ok());

        vault.delete_note(&moved).await.unwrap();
        assert!(vault.get_note_metadata(&moved).await.is_err());
    }

    #[tokio::test]
    async fn vault_patch_note() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        vault
            .write_note(
                Path::new("patched.md"),
                "# Section\nOriginal content\n# Other\nKeep this\n",
            )
            .await
            .unwrap();

        let request = PatchRequest {
            operation: PatchOperation::Replace,
            target_type: PatchTargetType::Heading,
            target: "Section".into(),
            content: "Replaced content\n".into(),
        };
        vault
            .patch_note(Path::new("patched.md"), &request)
            .await
            .unwrap();

        let content = vault.read_note(Path::new("patched.md")).await.unwrap();
        assert!(content.contains("Replaced content"));
        assert!(content.contains("# Other"));
        assert!(content.contains("Keep this"));
    }

    #[tokio::test]
    async fn vault_frontmatter_operations() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        vault
            .write_note(Path::new("meta.md"), "# Note\nBody text\n")
            .await
            .unwrap();

        vault
            .set_frontmatter_field(
                Path::new("meta.md"),
                "status",
                serde_json::Value::String("draft".into()),
            )
            .await
            .unwrap();
        vault
            .set_frontmatter_field(
                Path::new("meta.md"),
                "priority",
                serde_json::Value::Number(1.into()),
            )
            .await
            .unwrap();

        let fm = vault.get_frontmatter(Path::new("meta.md")).await.unwrap();
        assert!(fm.is_some());
        let obj = fm.unwrap();
        assert_eq!(obj["status"], "draft");
        assert_eq!(obj["priority"], 1);

        vault
            .remove_frontmatter_field(Path::new("meta.md"), "status")
            .await
            .unwrap();

        let fm = vault.get_frontmatter(Path::new("meta.md")).await.unwrap();
        let obj = fm.unwrap();
        assert!(obj.get("status").is_none());
        assert_eq!(obj["priority"], 1);
    }

    #[tokio::test]
    async fn vault_search_text() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        vault
            .write_note(
                Path::new("searchable.md"),
                "# Rust\nRust is a systems language.\n",
            )
            .await
            .unwrap();
        vault
            .write_note(Path::new("other.md"), "# Python\nPython is dynamic.\n")
            .await
            .unwrap();

        let results = vault.search_text("Rust", 40).await.unwrap();
        assert!(!results.is_empty());
        assert!(results.iter().any(|r| r.path == Path::new("searchable.md")));
        assert!(!results.iter().any(|r| r.path == Path::new("other.md")));
    }

    #[tokio::test]
    async fn vault_document_map() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        vault
            .write_note(
                Path::new("mapped.md"),
                "---\ntags: [rust]\n---\n# Heading\n## Sub\nText ^block1\n",
            )
            .await
            .unwrap();

        let map = vault
            .get_document_map(Path::new("mapped.md"))
            .await
            .unwrap();
        assert!(map.headings.iter().any(|h| h.contains("Heading")));
        assert!(map.headings.iter().any(|h| h.contains("Sub")));
        assert!(map.block_refs.contains(&"block1".to_string()));
        assert!(map.frontmatter_fields.contains(&"tags".to_string()));
    }

    #[tokio::test]
    async fn vault_tags_and_backlinks() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        vault
            .write_note(Path::new("a.md"), "---\ntags: [project]\n---\n# A\n[[b]]\n")
            .await
            .unwrap();
        vault
            .write_note(Path::new("b.md"), "---\ntags: [project]\n---\n# B\n")
            .await
            .unwrap();

        let tagged = vault.search_by_tag("project").unwrap();
        assert_eq!(tagged.len(), 2);

        let links = vault.outgoing_links(Path::new("a.md")).await.unwrap();
        assert!(links.iter().any(|l| l.target == "b"));

        let bl = vault.backlinks(Path::new("b.md")).await.unwrap();
        assert!(bl.iter().any(|m| m.path == Path::new("a.md")));
    }

    #[tokio::test]
    async fn vault_stats() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        vault
            .write_note(Path::new("one.md"), "# One")
            .await
            .unwrap();
        vault
            .write_note(Path::new("two.md"), "# Two")
            .await
            .unwrap();

        let stats = vault.vault_stats().unwrap();
        assert_eq!(stats.total_notes, 2);
    }

    #[tokio::test]
    async fn vault_create_note_basic() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        vault
            .create_note(Path::new("new.md"), "# New\n", None)
            .await
            .unwrap();
        let content = vault.read_note(Path::new("new.md")).await.unwrap();
        assert_eq!(content, "# New\n");
    }

    #[tokio::test]
    async fn vault_create_note_with_frontmatter() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        let fm = serde_json::json!({"tags": ["test"], "draft": true});
        vault
            .create_note(Path::new("fm.md"), "Body\n", Some(&fm))
            .await
            .unwrap();
        let content = vault.read_note(Path::new("fm.md")).await.unwrap();
        assert!(content.starts_with("---\n"));
        assert!(content.contains("Body\n"));
    }

    #[tokio::test]
    async fn vault_create_note_rejects_non_object_frontmatter_without_writing() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        for (path, frontmatter, actual) in [
            ("array.md", serde_json::json!(["rust", "mcp"]), "array"),
            ("string.md", serde_json::json!("{\"draft\":true}"), "string"),
            ("null.md", serde_json::Value::Null, "null"),
        ] {
            let err = vault
                .create_note(Path::new(path), "body", Some(&frontmatter))
                .await
                .unwrap_err();
            assert!(
                matches!(
                    &err,
                    VaultError::InvalidFrontmatter {
                        path: error_path,
                        actual: error_actual
                    } if error_path == Path::new(path) && *error_actual == actual
                ),
                "unexpected error for '{path}': {err:?}"
            );
            assert!(!dir.path().join(path).exists());

            let error_data: rmcp::ErrorData = err.into();
            assert_eq!(error_data.code, rmcp::model::ErrorCode::INVALID_PARAMS);
        }
    }

    #[tokio::test]
    async fn vault_create_note_fails_if_exists() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        vault
            .create_note(Path::new("dup.md"), "", None)
            .await
            .unwrap();
        let err = vault
            .create_note(Path::new("dup.md"), "new", None)
            .await
            .unwrap_err();
        assert!(matches!(err, VaultError::AlreadyExists(_)));
    }

    #[tokio::test]
    async fn vault_prepend_after_frontmatter() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        vault
            .write_note(Path::new("pre.md"), "---\ntags: [a]\n---\nExisting body\n")
            .await
            .unwrap();
        vault
            .prepend_note(Path::new("pre.md"), "Prepended\n")
            .await
            .unwrap();
        let content = vault.read_note(Path::new("pre.md")).await.unwrap();

        assert!(content.starts_with("---\n"));
        let prepended_pos = content.find("Prepended\n").unwrap();
        let existing_pos = content.find("Existing body\n").unwrap();
        assert!(prepended_pos < existing_pos);
    }

    #[tokio::test]
    async fn vault_prepend_no_frontmatter() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        vault
            .write_note(Path::new("nofm.md"), "Existing\n")
            .await
            .unwrap();
        vault
            .prepend_note(Path::new("nofm.md"), "Prepended\n")
            .await
            .unwrap();
        let content = vault.read_note(Path::new("nofm.md")).await.unwrap();
        assert_eq!(content, "Prepended\nExisting\n");
    }

    #[test]
    fn vault_is_send_sync_clone() {
        fn assert_send_sync_clone<T: Send + Sync + Clone>() {}
        assert_send_sync_clone::<Vault>();
    }

    #[tokio::test]
    async fn vault_open_with_tantivy() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());

        std::fs::write(
            dir.path().join("note.md"),
            "# Rust\nRust is a systems language.\n",
        )
        .unwrap();

        let vault = Vault::open(&tantivy_config(dir.path())).await.unwrap();
        assert!(vault.tantivy().is_some());
    }

    #[tokio::test]
    async fn vault_tantivy_syncs_on_write() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&tantivy_config(dir.path())).await.unwrap();

        vault
            .write_note(
                Path::new("alpha.md"),
                "# Alpha\nUnique content about zebras.\n",
            )
            .await
            .unwrap();

        let tv = vault.tantivy().unwrap();
        let results = tv.search("zebras", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, PathBuf::from("alpha.md"));
    }

    #[tokio::test]
    async fn vault_tantivy_syncs_on_delete() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&tantivy_config(dir.path())).await.unwrap();

        vault
            .write_note(Path::new("del.md"), "# Deletable\nEphemeral content.\n")
            .await
            .unwrap();
        assert_eq!(
            vault
                .tantivy()
                .unwrap()
                .search("ephemeral", 10)
                .unwrap()
                .len(),
            1
        );

        vault.delete_note(Path::new("del.md")).await.unwrap();
        assert!(
            vault
                .tantivy()
                .unwrap()
                .search("ephemeral", 10)
                .unwrap()
                .is_empty()
        );
    }

    #[tokio::test]
    async fn vault_tantivy_syncs_on_move() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&tantivy_config(dir.path())).await.unwrap();

        vault
            .write_note(Path::new("src.md"), "# Source\nMovable content.\n")
            .await
            .unwrap();
        vault
            .move_note(Path::new("src.md"), Path::new("dest.md"))
            .await
            .unwrap();

        let tv = vault.tantivy().unwrap();
        let results = tv.search("movable", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, PathBuf::from("dest.md"));
    }

    #[tokio::test]
    async fn vault_tantivy_syncs_on_create() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&tantivy_config(dir.path())).await.unwrap();

        let fm = serde_json::json!({"tags": ["science"]});
        vault
            .create_note(
                Path::new("created.md"),
                "# Created\nBioluminescence in deep sea creatures.\n",
                Some(&fm),
            )
            .await
            .unwrap();

        let tv = vault.tantivy().unwrap();
        let results = tv.search("bioluminescence", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, PathBuf::from("created.md"));
    }

    #[tokio::test]
    async fn vault_search_text_tantivy_returns_scores() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&tantivy_config(dir.path())).await.unwrap();

        vault
            .write_note(
                Path::new("a.md"),
                "# Quantum\nQuantum computing is fascinating.\n",
            )
            .await
            .unwrap();
        vault
            .write_note(Path::new("b.md"), "# Other\nNothing related.\n")
            .await
            .unwrap();

        let results = vault.search_text("quantum", 40).await.unwrap();
        assert!(!results.is_empty());
        assert!(results[0].score.is_some());
        assert!(results[0].score.unwrap() > 0.0);
        assert_eq!(results[0].path, PathBuf::from("a.md"));
    }

    #[tokio::test]
    async fn vault_search_text_with_options_fuzzy() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&tantivy_config(dir.path())).await.unwrap();

        vault
            .write_note(Path::new("target.md"), "# Algorithm\nSorting algorithms.\n")
            .await
            .unwrap();

        // "algorihm" is a typo for "algorithm"
        let results = vault
            .search_text_with_options("algorihm", 40, 10, true, None)
            .await
            .unwrap();
        assert!(
            results.iter().any(|r| r.path == Path::new("target.md")),
            "fuzzy search should find 'algorithm' from typo 'algorihm'"
        );
    }

    #[tokio::test]
    async fn vault_search_text_with_options_field_filter() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&tantivy_config(dir.path())).await.unwrap();

        vault
            .write_note(
                Path::new("headingmatch.md"),
                "# Something Else\n## Cryptography Section\nBasic text.\n",
            )
            .await
            .unwrap();

        let heading_only = vault
            .search_text_with_options(
                "cryptography",
                40,
                10,
                false,
                Some(&[SearchField::Headings]),
            )
            .await
            .unwrap();
        assert!(
            heading_only
                .iter()
                .any(|r| r.path == Path::new("headingmatch.md"))
        );
    }

    #[tokio::test]
    async fn vault_search_text_tantivy_zero_context() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&tantivy_config(dir.path())).await.unwrap();

        vault
            .write_note(
                Path::new("ctx.md"),
                "# Context Test\nSearchable unique phrase here.\n",
            )
            .await
            .unwrap();

        let results = vault.search_text("searchable", 0).await.unwrap();
        assert!(!results.is_empty());
        assert!(results[0].score.is_some());
        assert!(
            results[0].matches.is_empty(),
            "context_length=0 should produce no match snippets"
        );
    }

    #[tokio::test]
    async fn vault_search_text_with_options_fallback_without_tantivy() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        vault
            .write_note(
                Path::new("fallback.md"),
                "# Fallback\nFallback search content.\n",
            )
            .await
            .unwrap();

        let results = vault
            .search_text_with_options("fallback", 40, 10, true, None)
            .await
            .unwrap();
        assert!(
            results.iter().any(|r| r.path == Path::new("fallback.md")),
            "should still find results via regex fallback when tantivy is disabled"
        );
        assert!(
            results[0].score.is_none(),
            "regex fallback should not populate BM25 scores"
        );
    }

    #[cfg(has_embeddings)]
    #[tokio::test]
    async fn deferred_embeddings_reconcile_final_mutations_and_watcher_events() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join(".obsidian")).unwrap();
        std::fs::write(dir.path().join("edit.md"), "OUTDATED_MARKER").unwrap();
        std::fs::write(dir.path().join("delete.md"), "DELETED_MARKER").unwrap();
        let mut config = embedding_config(dir.path());
        config.watch = true;
        config.exclude_patterns = vec!["Hidden/".into()];
        let model = Arc::new(TestEmbedder::new());
        let loader_model = Arc::clone(&model);
        let polls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let loader_polls = Arc::clone(&polls);
        let loader: EmbeddingLoaderFuture = Box::pin(async move {
            loader_polls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(loader_model as Arc<dyn embeddings::Embedder>)
        });
        let vault = Vault::open_inner(&config, Some(loader), false)
            .await
            .unwrap();
        let submitter = vault.inner.embedding_runtime.get().unwrap().downgrade();
        vault
            .write_note(Path::new("edit.md"), "FINAL_EDIT_MARKER")
            .await
            .unwrap();
        vault.delete_note(Path::new("delete.md")).await.unwrap();
        vault
            .create_note(Path::new("created.md"), "FINAL_MOVED_MARKER", None)
            .await
            .unwrap();
        vault
            .move_note(Path::new("created.md"), Path::new("moved.md"))
            .await
            .unwrap();
        vault
            .write_note(Path::new("Hidden/secret.md"), "EXCLUDED_MARKER")
            .await
            .unwrap();
        std::fs::write(dir.path().join("external.md"), "FINAL_EXTERNAL_MARKER").unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while vault
                .get_note_metadata(Path::new("external.md"))
                .await
                .is_err()
            {
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            }
        })
        .await
        .unwrap();
        assert_eq!(polls.load(std::sync::atomic::Ordering::SeqCst), 0);
        assert!(!vault.embeddings_active());
        assert!(model.inputs.lock().unwrap().is_empty());
        assert!(vault.clone().activate_embeddings());
        assert!(!vault.activate_embeddings());
        // A previously captured watcher handle still reaches the activated coordinator.
        submitter.submit_upsert(Path::new("edit.md"));
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while !vault
                .embedding_status()
                .is_some_and(|status| status.phase == embedding_runtime::EmbeddingPhase::Ready)
            {
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        assert!(vault.embeddings_active());
        assert_eq!(polls.load(std::sync::atomic::Ordering::SeqCst), 1);
        let inputs = model.inputs.lock().unwrap();
        assert_eq!(
            inputs.len(),
            3,
            "only final visible notes require embedding: {inputs:?}"
        );
        for marker in [
            "FINAL_EDIT_MARKER",
            "FINAL_MOVED_MARKER",
            "FINAL_EXTERNAL_MARKER",
        ] {
            assert!(
                inputs.iter().any(|text| text.contains(marker)),
                "missing {marker}: {inputs:?}"
            );
        }
        for marker in ["OUTDATED_MARKER", "DELETED_MARKER", "EXCLUDED_MARKER"] {
            assert!(
                !inputs.iter().any(|text| text.contains(marker)),
                "unexpected {marker}: {inputs:?}"
            );
        }
    }

    #[cfg(has_embeddings)]
    #[tokio::test]
    async fn vault_open_and_core_search_return_while_embedding_loader_is_blocked() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let config = embedding_config(dir.path());

        let vault = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            Vault::open_with_embedding_loader(&config, Some(blocked_embedding_loader())),
        )
        .await
        .expect("Vault::open must not wait for the embedding loader")
        .unwrap();

        assert!(vault.search_text("anything", 20).await.unwrap().is_empty());
        let error = vault.search_semantic("anything", 5).unwrap_err();
        assert!(
            matches!(error, VaultError::Embedding(ref message) if message.contains("not ready") && message.contains("warming up")),
            "semantic search should report a warming runtime: {error}"
        );
    }

    #[cfg(has_embeddings)]
    #[tokio::test]
    async fn vault_mutations_submit_only_the_latest_embedding_intent() {
        use embedding_runtime::PendingKind;

        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let mut config = embedding_config(dir.path());
        config.exclude_patterns = vec!["Excluded/".into()];
        let vault = Vault::open_with_embedding_loader(&config, Some(blocked_embedding_loader()))
            .await
            .unwrap();
        let runtime = vault.inner.embedding_runtime.get().unwrap();
        let path = Path::new("intent.md");

        vault
            .create_note(path, "# Intent\nBody\n", None)
            .await
            .unwrap();
        assert_eq!(runtime.pending_kind(path), Some(PendingKind::Upsert));

        vault.append_note(path, "Appended\n").await.unwrap();
        assert_eq!(runtime.pending_kind(path), Some(PendingKind::Upsert));

        vault.prepend_note(path, "Prepended\n").await.unwrap();
        assert_eq!(runtime.pending_kind(path), Some(PendingKind::Upsert));

        vault
            .write_note(path, "# Intent\nReplacement\n")
            .await
            .unwrap();
        assert_eq!(runtime.pending_kind(path), Some(PendingKind::Upsert));

        vault
            .set_frontmatter_field(path, "state", serde_json::json!("active"))
            .await
            .unwrap();
        assert_eq!(runtime.pending_kind(path), Some(PendingKind::Upsert));

        vault.remove_frontmatter_field(path, "state").await.unwrap();
        assert_eq!(runtime.pending_kind(path), Some(PendingKind::Upsert));

        vault
            .patch_note(
                path,
                &PatchRequest {
                    operation: PatchOperation::Replace,
                    target_type: PatchTargetType::Heading,
                    target: "Intent".into(),
                    content: "Patched\n".into(),
                },
            )
            .await
            .unwrap();
        assert_eq!(runtime.pending_kind(path), Some(PendingKind::Upsert));

        vault.delete_note(path).await.unwrap();
        assert_eq!(runtime.pending_kind(path), Some(PendingKind::Remove));

        let excluded_path = Path::new("Excluded/hidden.md");
        vault.write_note(excluded_path, "# Hidden\n").await.unwrap();
        assert_eq!(runtime.pending_kind(excluded_path), None);
        let attachment = Path::new("plain.txt");
        vault.write_note(attachment, "attachment").await.unwrap();
        assert_eq!(runtime.pending_kind(attachment), None);

        let old_path = Path::new("old.md");
        let new_path = Path::new("nested/new.md");
        vault.write_note(old_path, "# Move\n").await.unwrap();
        vault.move_note(old_path, new_path).await.unwrap();
        assert_eq!(runtime.pending_kind(old_path), Some(PendingKind::Remove));
        assert_eq!(runtime.pending_kind(new_path), Some(PendingKind::Upsert));

        let now_excluded = Path::new("Excluded/moved.md");
        vault.move_note(new_path, now_excluded).await.unwrap();
        assert_eq!(runtime.pending_kind(new_path), Some(PendingKind::Remove));
        assert_eq!(runtime.pending_kind(now_excluded), None);
    }

    #[cfg(has_embeddings)]
    #[tokio::test]
    async fn audience_scope_filters_semantic_and_hybrid_candidates_before_top_k() {
        let directory = tempfile::tempdir().unwrap();
        let allowed_scope = "slack-t1-dm-zzz";
        let denied_scope = "slack-t1-dm-aaa";
        for (scope, user) in [(allowed_scope, "UZ"), (denied_scope, "UA")] {
            let folder = directory.path().join(format!("Organize/Work/{scope}"));
            std::fs::create_dir_all(&folder).unwrap();
            std::fs::write(
                folder.join("decision.md"),
                format!("---\nmemory_schema: slack-code/1\nsensitivity: normal\ntenant_id: bloomer\nstar_label: bloom\nworkspace_id: T1\naudience_kind: dm\naudience_id: {user}\n---\n# Launch\nlaunch schedule\n"),
            )
            .unwrap();
        }
        let model = Arc::new(TestEmbedder::new());
        let loader: EmbeddingLoaderFuture =
            Box::pin(async move { Ok(model as Arc<dyn embeddings::Embedder>) });
        let mut config = embedding_config(directory.path());
        config.tantivy = true;
        let vault = Vault::open_with_embedding_loader(&config, Some(loader))
            .await
            .unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(3), async {
            while !vault
                .embedding_status()
                .is_some_and(|status| status.queryable && status.pending_notes == 0)
            {
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap();
        let scope = ReadScope {
            version: 1,
            tenant_id: "bloomer".into(),
            star_label: "bloom".into(),
            workspace_id: "T1".into(),
            audience_kind: "dm".into(),
            audience_id: "UZ".into(),
            purpose: "conversation".into(),
            scope_id: allowed_scope.into(),
            allowed_prefixes: vec![format!("Organize/Work/{allowed_scope}/")],
            expires_at: 4_000_000_000.0,
        };
        let scoped = vault.with_read_scope(scope).unwrap();
        let expected = PathBuf::from(format!("Organize/Work/{allowed_scope}/decision.md"));
        assert_eq!(
            vault.search_semantic("launch", 1).unwrap()[0].0,
            PathBuf::from(format!("Organize/Work/{denied_scope}/decision.md"))
        );
        assert_eq!(scoped.search_semantic("launch", 1).unwrap()[0].0, expected);
        assert_eq!(
            scoped.search_hybrid("launch", 1, 1, 0.25).unwrap()[0].0,
            expected
        );
        assert!(scoped.write_note(&expected, "overwrite").await.is_err());

        // A cached vector must not become evidence for a replacement source revision.
        let replacement = scoped
            .read_note(&expected)
            .await
            .unwrap()
            .replace("schedule", "replacement");
        std::fs::write(directory.path().join(&expected), replacement).unwrap();
        assert!(scoped.search_semantic("launch", 1).unwrap().is_empty());
        assert!(
            scoped
                .search_hybrid("launch", 1, 1, 0.25)
                .unwrap()
                .is_empty()
        );
    }

    #[cfg(has_embeddings)]
    #[tokio::test]
    async fn local_semantic_search_filters_paths_removed_from_primary_index() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let path = PathBuf::from("stale.md");
        std::fs::write(dir.path().join(&path), "# Stale\nBody\n").unwrap();

        let model = Arc::new(TestEmbedder::new());
        let mut store = embeddings::EmbeddingStore::new_with_identity(model.identity.clone());
        store
            .insert_hashed(path.clone(), [7; 32], vec![1.0, 0.0])
            .unwrap();
        store.set_first_pass_complete(true);
        let cache_path = dir
            .path()
            .join(".obsidian-mcp")
            .join("embeddings/scopes")
            .join(ExcludeSet::default().scope_id(&dir.path().canonicalize().unwrap()))
            .join("embeddings.bin");
        std::fs::create_dir_all(cache_path.parent().unwrap()).unwrap();
        store.save(&cache_path).unwrap();

        let loader: EmbeddingLoaderFuture =
            Box::pin(async move { Ok(model as Arc<dyn embeddings::Embedder>) });
        let config = embedding_config(dir.path());
        let vault = Vault::open_with_embedding_loader(&config, Some(loader))
            .await
            .unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            while !vault.has_embeddings() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("compatible cache should become queryable");

        vault.write_index().remove_file(&path);

        assert!(vault.search_semantic("stale", 5).unwrap().is_empty());
    }

    #[cfg(has_embeddings)]
    #[tokio::test]
    async fn semantic_facade_keeps_captured_revision_while_query_inference_blocks() {
        use std::sync::mpsc;
        use std::time::Duration;

        struct BlockingQueryEmbedder {
            inner: TestEmbedder,
            entered: mpsc::SyncSender<()>,
            release: Mutex<mpsc::Receiver<()>>,
        }

        impl embeddings::Embedder for BlockingQueryEmbedder {
            fn input_policy(&self) -> &embedding_input::EmbeddingInputPolicy {
                embeddings::Embedder::input_policy(&self.inner)
            }

            fn count_tokens(&self, text: &str) -> VaultResult<usize> {
                embeddings::Embedder::count_tokens(&self.inner, text)
            }

            fn request_limits(&self) -> embedding_input::EmbeddingRequestLimits {
                embeddings::Embedder::request_limits(&self.inner)
            }

            fn dimension(&self) -> usize {
                embeddings::Embedder::dimension(&self.inner)
            }

            fn space_identity(&self) -> &embeddings::EmbeddingSpaceIdentity {
                embeddings::Embedder::space_identity(&self.inner)
            }

            fn embed_batch(&self, texts: &[&str]) -> VaultResult<Vec<Vec<f32>>> {
                if texts.len() == 1 && texts[0] == "REVISION_BOUND_QUERY" {
                    self.entered.send(()).unwrap();
                    self.release
                        .lock()
                        .unwrap()
                        .recv_timeout(Duration::from_secs(5))
                        .unwrap();
                }
                embeddings::Embedder::embed_batch(&self.inner, texts)
            }
        }

        struct ReleaseOnDrop(Option<mpsc::Sender<()>>);

        impl Drop for ReleaseOnDrop {
            fn drop(&mut self) {
                if let Some(sender) = self.0.take() {
                    let _ = sender.send(());
                }
            }
        }

        async fn wait_for_revision(vault: &Vault, expected: &str) {
            let runtime = vault.inner.embedding_runtime.get().unwrap();
            let paths = std::collections::HashSet::from([OsString::from("policy.md")]);
            tokio::time::timeout(Duration::from_secs(3), async {
                loop {
                    if let Ok(snapshot) = runtime.query_snapshot()
                        && snapshot
                            .capture_entries(&paths)
                            .get(Path::new("policy.md").as_os_str())
                            .is_some_and(|entry| entry.source_revision == expected)
                    {
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
            })
            .await
            .expect("complete note generation should commit while query inference is held");
        }

        let directory = tempfile::tempdir().unwrap();
        create_test_vault(directory.path());
        let old_source = "# Policy\nRetain the original source.\n";
        let new_source =
            "---\ntags: [updated]\n---\n# Replacement\nDifferent source and offsets.\n";
        let old_revision = source::source_revision(old_source);
        let new_revision = source::source_revision(new_source);
        std::fs::write(directory.path().join("policy.md"), old_source).unwrap();
        let (entered_tx, entered_rx) = mpsc::sync_channel(1);
        let (release_tx, release_rx) = mpsc::channel();
        let release_guard = ReleaseOnDrop(Some(release_tx));
        let model = Arc::new(BlockingQueryEmbedder {
            inner: TestEmbedder::new(),
            entered: entered_tx,
            release: Mutex::new(release_rx),
        });
        let loader: EmbeddingLoaderFuture =
            Box::pin(async move { Ok(model as Arc<dyn embeddings::Embedder>) });
        let vault =
            Vault::open_with_embedding_loader(&embedding_config(directory.path()), Some(loader))
                .await
                .unwrap();
        wait_for_revision(&vault, &old_revision).await;

        let searching_vault = vault.clone();
        let query = tokio::task::spawn_blocking(move || {
            searching_vault.search_semantic_ranked("REVISION_BOUND_QUERY", 1)
        });
        tokio::task::spawn_blocking(move || entered_rx.recv_timeout(Duration::from_secs(3)))
            .await
            .unwrap()
            .expect("query should reach the held provider after capturing source entries");

        vault
            .write_note(Path::new("policy.md"), new_source)
            .await
            .unwrap();
        wait_for_revision(&vault, &new_revision).await;
        drop(release_guard);
        let mut hits = query.await.unwrap().unwrap();
        assert_eq!(hits.len(), 1);
        let hit = hits.remove(0);
        assert_eq!(hit.entry.source_revision, old_revision);
        assert_eq!(hit.evidence.scope, crate::models::EvidenceScope::Note);
        assert_eq!(
            hit.evidence.range,
            crate::models::SourceRange {
                start_byte: 0,
                end_byte: old_source.len(),
            }
        );

        let rendered = semantic_results::render(vault.root(), hit, "REVISION_BOUND_QUERY", false);
        assert_eq!(rendered.source_revision, old_revision);
        assert_eq!(rendered.source_status, crate::models::SourceStatus::Changed);
        let error = rendered.context_error.unwrap();
        assert_eq!(error.code, "source_changed");
        assert_eq!(
            error.current_revision.as_deref(),
            Some(new_revision.as_str())
        );
        assert!(rendered.snippet.is_none());
        assert!(rendered.content.is_none());
        assert!(rendered.preview.is_none());
    }

    #[cfg(has_embeddings)]
    #[tokio::test]
    async fn graceful_degradation_captures_embedding_load_error() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());

        let config = Config {
            vault_path: dir.path().to_path_buf(),
            watch: false,
            log_level: "error".into(),
            transport: crate::config::Transport::Stdio,
            http_host: crate::config::DEFAULT_HTTP_HOST,
            http_port: crate::config::DEFAULT_HTTP_PORT,
            tantivy: false,
            embeddings: true,
            embeddings_model: "nonexistent-model-that-will-fail".into(),
            hybrid_alpha: 0.25,
            embedding_provider: Some(crate::config::EmbeddingProvider::Local),
            tool_filter: crate::config::ToolFilter::Full,
            mcp_data_dir: None,
            exclude_patterns: vec![],
        };

        let vault = Vault::open(&config)
            .await
            .expect("vault should open despite embedding failure");
        assert!(
            !vault.has_embeddings(),
            "embeddings should not be available after load failure"
        );
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            while vault.embedding_load_error().is_none() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("background model load should report its failure");
        let err = vault.embedding_load_error().unwrap();
        assert!(!err.is_empty(), "error message should be descriptive");
    }

    #[cfg(has_embeddings)]
    #[tokio::test]
    async fn disabled_embeddings_do_not_relocate_legacy_cache() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());

        let legacy_dir = dir.path().join(".obsidian").join("obsidian-mcp");
        std::fs::create_dir_all(&legacy_dir).unwrap();
        let legacy_path = legacy_dir.join("embeddings.bin");
        let test_bytes: Vec<u8> = vec![0xDE, 0xAD, 0xBE, 0xEF, 1, 2, 3, 4];
        std::fs::write(&legacy_path, &test_bytes).unwrap();

        let _vault = Vault::open(&test_config(dir.path())).await.unwrap();

        let new_path = dir
            .path()
            .join(".obsidian-mcp")
            .join("embeddings")
            .join("embeddings.bin");
        assert!(
            !new_path.exists(),
            "disabled embeddings must not move caches"
        );
        assert!(
            legacy_path.exists(),
            "disabled embeddings must leave the legacy file untouched"
        );
        assert_eq!(std::fs::read(legacy_path).unwrap(), test_bytes);
    }

    #[cfg(has_embeddings)]
    #[tokio::test]
    async fn legacy_embedding_cache_relocation_waits_for_background_loader() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());

        let legacy_dir = dir.path().join(".obsidian").join("obsidian-mcp");
        std::fs::create_dir_all(&legacy_dir).unwrap();
        std::fs::write(legacy_dir.join("embeddings.bin"), b"old-legacy-data").unwrap();

        let config = embedding_config(dir.path());
        let vault = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            Vault::open_with_embedding_loader(&config, Some(blocked_embedding_loader())),
        )
        .await
        .expect("vault open must not wait for cache relocation")
        .unwrap();

        assert!(
            !dir.path()
                .join(".obsidian-mcp")
                .join("embeddings")
                .join("embeddings.bin")
                .exists(),
            "cache relocation must stay behind the blocked background loader"
        );
        drop(vault);
    }

    #[cfg(has_embeddings)]
    #[tokio::test]
    async fn no_legacy_cache_no_migration_error() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());

        let vault = Vault::open(&test_config(dir.path())).await;
        assert!(vault.is_ok(), "vault should open without legacy cache");

        let new_cache = dir
            .path()
            .join(".obsidian-mcp")
            .join("embeddings")
            .join("embeddings.bin");
        assert!(
            !new_cache.exists(),
            "no cache file should be created when no legacy exists"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn replacement_at_an_unlinked_source_survives_direct_reconciliation() {
        let directory = tempfile::tempdir().unwrap();
        let vault = Vault::open(&crate::test_helpers::tantivy_config(directory.path()))
            .await
            .unwrap();
        let source = Path::new("source.md");
        let on_disk = directory.path().join(source);
        for moving in [false, true] {
            vault.write_note(source, "departed").await.unwrap();
            // Hold the metadata lock until the filesystem operation has finished.
            // A watcher can then observe the replacement before the tool gets the lock.
            let operation = {
                let mut index = vault.write_index();
                let writer = vault.clone();
                let handle = tokio::runtime::Handle::current();
                let operation = std::thread::spawn(move || {
                    handle.block_on(async move {
                        if moving {
                            writer
                                .move_note(source, Path::new("moved.md"))
                                .await
                                .map(|_| ())
                        } else {
                            writer.delete_note(source).await
                        }
                    })
                });
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
                while on_disk.exists() {
                    assert!(
                        std::time::Instant::now() < deadline,
                        "source was never unlinked"
                    );
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                std::fs::write(&on_disk, "#replacement presentagain").unwrap();
                assert!(
                    index
                        .refresh_file(
                            vault.inner.filesystem.as_ref().unwrap(),
                            source,
                            vault.exclude(),
                            vault.inner.tantivy.as_deref(),
                            #[cfg(has_embeddings)]
                            false,
                        )
                        .unwrap()
                );
                operation
            };
            operation.join().unwrap().unwrap();

            assert_eq!(
                vault.get_note_metadata(source).await.unwrap().tags,
                ["replacement"]
            );
            let hits = vault.search_text("presentagain", 0).await.unwrap();
            assert_eq!(hits.len(), 1);
            assert_eq!(hits[0].path, source);
            if moving {
                assert_eq!(
                    std::fs::read_to_string(directory.path().join("moved.md")).unwrap(),
                    "departed"
                );
            }
        }
    }

    #[tokio::test]
    async fn note_and_attachment_mutations_match_a_cold_vault() {
        async fn assert_matches_disk(live: &Vault, config: &Config) {
            let cold = Vault::open(config).await.unwrap();
            assert_eq!(
                serde_json::to_value(live.vault_stats().unwrap()).unwrap(),
                serde_json::to_value(cold.vault_stats().unwrap()).unwrap()
            );
            async fn projection(
                vault: &Vault,
            ) -> (
                Vec<(PathBuf, Vec<String>, Vec<PathBuf>)>,
                Vec<PathBuf>,
                Vec<PathBuf>,
            ) {
                let notes = {
                    let index = vault.read_index();
                    let mut notes: Vec<_> = index
                        .notes()
                        .iter()
                        .map(|(path, note)| {
                            let mut backlinks: Vec<_> = index
                                .backlinks_to(Path::new(path))
                                .iter()
                                .map(|n| n.path.clone())
                                .collect();
                            backlinks.sort();
                            (PathBuf::from(path), note.tags.clone(), backlinks)
                        })
                        .collect();
                    notes.sort();
                    notes
                };
                let mut lexical: Vec<_> = vault
                    .search_text("needle", 20)
                    .await
                    .unwrap()
                    .into_iter()
                    .map(|hit| hit.path)
                    .collect();
                lexical.sort();
                let mut literal: Vec<_> = vault
                    .search_regex("need.e", 20)
                    .await
                    .unwrap()
                    .into_iter()
                    .map(|hit| hit.path)
                    .collect();
                literal.sort();
                (notes, lexical, literal)
            }
            assert_eq!(projection(live).await, projection(&cold).await);
        }
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        let mut config = crate::test_helpers::tantivy_config(root);
        config.exclude_patterns = vec!["Archive/**".into()];
        std::fs::write(root.join("existing.bin"), [0xff, 0, 0x80]).unwrap();
        let live = Vault::open(&config).await.unwrap();
        live.create_note(Path::new("folder/anchor.md"), "#anchor needle", None)
            .await
            .unwrap();
        assert!(
            live.move_note(Path::new("folder"), Path::new("unused/subdir"))
                .await
                .is_err()
        );
        assert!(root.join("folder/anchor.md").is_file());
        assert!(!root.join("unused").exists());
        live.write_note(Path::new("item.txt"), "needle #tag [[anchor]]")
            .await
            .unwrap();
        assert!(live.get_note_metadata(Path::new("item.txt")).await.is_err());
        assert_matches_disk(&live, &config).await;
        live.append_note(Path::new("item.txt"), " more")
            .await
            .unwrap();
        assert_matches_disk(&live, &config).await;
        for (from, to) in [
            ("item.txt", "item.MD"),
            ("item.MD", "Archive/hidden.md"),
            ("Archive/hidden.md", "visible.md"),
            ("visible.md", "last.txt"),
            ("last.txt", ".private/last.txt"),
            (".private/last.txt", "last.txt"),
        ] {
            live.move_note(Path::new(from), Path::new(to))
                .await
                .unwrap();
            assert_matches_disk(&live, &config).await;
        }
        live.delete_note(Path::new("last.txt")).await.unwrap();
        live.delete_note(Path::new("existing.bin")).await.unwrap();
        assert_matches_disk(&live, &config).await;
    }

    fn test_config_with_exclusions(vault_root: &Path, patterns: Vec<String>) -> Config {
        Config {
            exclude_patterns: patterns,
            ..test_config(vault_root)
        }
    }

    #[tokio::test]
    async fn move_into_excluded_dir() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let config = test_config_with_exclusions(dir.path(), vec!["Archive/**".into()]);
        let vault = Vault::open(&config).await.unwrap();

        vault
            .write_note(Path::new("active.md"), "# Active")
            .await
            .unwrap();
        assert!(
            vault
                .get_note_metadata(Path::new("active.md"))
                .await
                .is_ok()
        );

        let new_path = vault
            .move_note(Path::new("active.md"), Path::new("Archive/archived.md"))
            .await
            .unwrap();
        assert_eq!(new_path, PathBuf::from("Archive/archived.md"));

        assert!(
            vault
                .get_note_metadata(Path::new("active.md"))
                .await
                .is_err()
        );
        assert!(
            vault
                .get_note_metadata(Path::new("Archive/archived.md"))
                .await
                .is_err()
        );
        assert_eq!(vault.vault_stats().unwrap().excluded_notes, 1);

        let content = vault
            .read_note(Path::new("Archive/archived.md"))
            .await
            .unwrap();
        assert_eq!(content, "# Active");
    }

    #[tokio::test]
    async fn move_out_of_excluded_dir() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        std::fs::create_dir_all(dir.path().join("Archive")).unwrap();
        std::fs::write(dir.path().join("Archive/hidden.md"), "# Hidden").unwrap();

        let config = test_config_with_exclusions(dir.path(), vec!["Archive/**".into()]);
        let vault = Vault::open(&config).await.unwrap();

        assert!(
            vault
                .get_note_metadata(Path::new("Archive/hidden.md"))
                .await
                .is_err()
        );

        let new_path = vault
            .move_note(Path::new("Archive/hidden.md"), Path::new("visible.md"))
            .await
            .unwrap();
        assert_eq!(new_path, PathBuf::from("visible.md"));

        assert!(
            vault
                .get_note_metadata(Path::new("visible.md"))
                .await
                .is_ok()
        );
        assert!(
            vault
                .get_note_metadata(Path::new("Archive/hidden.md"))
                .await
                .is_err()
        );
        assert_eq!(vault.vault_stats().unwrap().excluded_notes, 0);
    }

    #[tokio::test]
    async fn move_between_excluded_dirs() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        std::fs::create_dir_all(dir.path().join("Archive")).unwrap();
        std::fs::write(dir.path().join("Archive/old.md"), "# Old").unwrap();

        let config =
            test_config_with_exclusions(dir.path(), vec!["Archive/**".into(), "Trash/**".into()]);
        let vault = Vault::open(&config).await.unwrap();

        assert!(
            vault
                .get_note_metadata(Path::new("Archive/old.md"))
                .await
                .is_err()
        );

        let new_path = vault
            .move_note(Path::new("Archive/old.md"), Path::new("Trash/old.md"))
            .await
            .unwrap();
        assert_eq!(new_path, PathBuf::from("Trash/old.md"));

        assert!(
            vault
                .get_note_metadata(Path::new("Archive/old.md"))
                .await
                .is_err()
        );
        assert!(
            vault
                .get_note_metadata(Path::new("Trash/old.md"))
                .await
                .is_err()
        );

        let content = vault.read_note(Path::new("Trash/old.md")).await.unwrap();
        assert_eq!(content, "# Old");
    }

    #[tokio::test]
    async fn move_into_excluded_dir_with_tantivy() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let config = Config {
            tantivy: true,
            exclude_patterns: vec!["Archive/**".into()],
            ..test_config(dir.path())
        };
        let vault = Vault::open(&config).await.unwrap();

        vault
            .write_note(Path::new("indexed.md"), "# Indexed note with content")
            .await
            .unwrap();
        assert!(
            vault
                .get_note_metadata(Path::new("indexed.md"))
                .await
                .is_ok()
        );

        vault
            .move_note(Path::new("indexed.md"), Path::new("Archive/gone.md"))
            .await
            .unwrap();

        assert!(
            vault
                .get_note_metadata(Path::new("indexed.md"))
                .await
                .is_err()
        );
        assert!(
            vault
                .get_note_metadata(Path::new("Archive/gone.md"))
                .await
                .is_err()
        );

        let results = vault.search_text("Indexed note", 100).await.unwrap();
        assert!(
            results.is_empty(),
            "moved-to-excluded note should not appear in search"
        );
    }

    #[tokio::test]
    async fn write_note_to_excluded_path_does_not_index() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let config = test_config_with_exclusions(dir.path(), vec!["Archive/**".into()]);
        let vault = Vault::open(&config).await.unwrap();

        vault
            .write_note(Path::new("Archive/direct.md"), "# Direct excluded")
            .await
            .unwrap();

        assert!(
            vault
                .get_note_metadata(Path::new("Archive/direct.md"))
                .await
                .is_err()
        );
        assert_eq!(vault.vault_stats().unwrap().excluded_notes, 1);
        assert_eq!(
            vault
                .read_note(Path::new("Archive/direct.md"))
                .await
                .unwrap(),
            "# Direct excluded"
        );

        vault
            .delete_note(Path::new("Archive/direct.md"))
            .await
            .unwrap();
        assert_eq!(vault.vault_stats().unwrap().excluded_notes, 0);
    }

    #[tokio::test]
    async fn mcp_data_equal_to_mcp_home_uses_default_location() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let mcp_home = dir.path().join(".obsidian-mcp");
        let config = Config {
            mcp_data_dir: Some(mcp_home.clone()),
            ..test_config(dir.path())
        };

        let vault = Vault::open(&config).await.unwrap();

        assert_eq!(
            vault.mcp_home(),
            Some(vault.root().join(".obsidian-mcp").as_path())
        );
        assert_eq!(vault.mcp_data(), vault.mcp_home());
        assert!(
            !vault.mcp_home().unwrap().join("vaults").exists(),
            "default mcp home must not be namespaced under itself"
        );
    }

    #[tokio::test]
    async fn concurrent_transformations_preserve_every_successful_edit() {
        let dir = tempfile::tempdir().unwrap();
        let vault = Vault::open(&tantivy_config(dir.path())).await.unwrap();
        vault
            .write_note(Path::new("shared.md"), &"body\n".repeat(20000))
            .await
            .unwrap();
        let barrier = Arc::new(tokio::sync::Barrier::new(8));
        let workers: Vec<_> = (0..8)
            .map(|id| {
                let vault = vault.clone();
                let barrier = barrier.clone();
                tokio::spawn(async move {
                    barrier.wait().await;
                    vault
                        .prepend_note(Path::new("shared.md"), &format!("marker-{id}\n"))
                        .await
                        .unwrap();
                    vault
                        .set_frontmatter_field(
                            Path::new("shared.md"),
                            &format!("key-{id}"),
                            id.into(),
                        )
                        .await
                        .unwrap();
                })
            })
            .collect();
        for worker in workers {
            worker.await.unwrap();
        }
        let content = vault.read_note(Path::new("shared.md")).await.unwrap();
        let fm = vault
            .get_frontmatter(Path::new("shared.md"))
            .await
            .unwrap()
            .unwrap();
        for id in 0..8 {
            assert!(content.contains(&format!("marker-{id}\n")));
            assert_eq!(fm[format!("key-{id}")], id);
        }
    }

    #[tokio::test]
    async fn invalid_yaml_write_preserves_file_and_all_indexes() {
        let dir = tempfile::tempdir().unwrap();
        let vault = Vault::open(&tantivy_config(dir.path())).await.unwrap();
        let content = "---\ntags: [Work]\n---\nquokka\n";
        vault
            .write_note(Path::new("note.md"), content)
            .await
            .unwrap();
        let stats = vault.vault_stats().unwrap();
        assert!(
            vault
                .write_note(Path::new("note.md"), "---\ntags: [unclosed\n---\nchanged")
                .await
                .is_err()
        );
        assert_eq!(
            vault.read_note(Path::new("note.md")).await.unwrap(),
            content
        );
        assert_eq!(
            vault
                .get_note_metadata(Path::new("note.md"))
                .await
                .unwrap()
                .tags,
            vec!["Work"]
        );
        assert_eq!(vault.search_text("quokka", 0).await.unwrap().len(), 1);
        assert_eq!(vault.vault_stats().unwrap().total_notes, stats.total_notes);
    }

    #[tokio::test]
    async fn hidden_writes_and_moves_remain_outside_indexes() {
        let dir = tempfile::tempdir().unwrap();
        let vault = Vault::open(&tantivy_config(dir.path())).await.unwrap();
        vault
            .write_note(Path::new(".secret/hidden.md"), "quokka")
            .await
            .unwrap();
        assert!(
            vault
                .get_note_metadata(Path::new(".secret/hidden.md"))
                .await
                .is_err()
        );
        vault
            .write_note(Path::new("visible.md"), "quokka")
            .await
            .unwrap();
        vault
            .move_note(Path::new("visible.md"), Path::new("notes/.hidden.md"))
            .await
            .unwrap();
        assert!(vault.search_text("quokka", 0).await.unwrap().is_empty());
        assert_eq!(vault.vault_stats().unwrap().total_notes, 0);
    }

    #[tokio::test]
    async fn tag_identity_is_case_insensitive_across_updates_and_search_fields() {
        let dir = tempfile::tempdir().unwrap();
        let vault = Vault::open(&tantivy_config(dir.path())).await.unwrap();
        vault
            .write_note(Path::new("a.md"), "---\ntags: [Work]\n---\n#WORK/Inbox")
            .await
            .unwrap();
        assert_eq!(vault.search_by_tag("work").unwrap().len(), 1);
        assert_eq!(vault.search_by_tag_prefix("wOrK").unwrap().len(), 1);
        assert_eq!(
            vault
                .search_text_with_options("WORK", 0, 10, false, Some(&[SearchField::Tags]))
                .await
                .unwrap()
                .len(),
            1
        );
        vault.write_note(Path::new("a.md"), "#work").await.unwrap();
        assert_eq!(vault.search_by_tag("WORK").unwrap().len(), 1);
        assert!(vault.search_by_tag("work/inbox").unwrap().is_empty());
        vault.delete_note(Path::new("a.md")).await.unwrap();
        assert!(vault.search_by_tag("work").unwrap().is_empty());
    }
}
