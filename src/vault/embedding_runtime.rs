//! Managed background lifecycle for optional semantic embeddings.

use std::collections::{HashMap, HashSet};
use std::ffi::OsString;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock, Weak};
use std::time::{Duration, Instant};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio::sync::Notify;
use tokio::task::JoinHandle;
use tracing::{Instrument, instrument::WithSubscriber};

use crate::error::{VaultError, VaultResult};

#[cfg(test)]
use super::embedding_chunks::prepare_note;
use super::embedding_chunks::{PreparedNote, prepare_note_with_budget};
#[cfg(test)]
use super::embeddings::prepared_text_hash;
use super::embeddings::{
    CacheFile, ChunkEmbedding, Embedder, EmbeddingStore, NoteEmbeddingEntry,
    validate_embedding_batch,
};
use super::index::VaultIndex;
use super::storage::Storage;

#[cfg(not(test))]
const MAX_DIRTY_INTERVAL: Duration = Duration::from_secs(2);
#[cfg(test)]
const MAX_DIRTY_INTERVAL: Duration = Duration::from_millis(50);
const MAX_RETRY_DELAY: Duration = Duration::from_secs(30);
const MAX_STAGING_BYTES: usize = 64 * 1024 * 1024;
#[cfg(test)]
const TEST_BATCH_SIZE: usize = 32;
#[cfg(not(test))]
const PROGRESS_INTERVAL: Duration = Duration::from_secs(30);
#[cfg(test)]
const PROGRESS_INTERVAL: Duration = Duration::from_millis(100);

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingPhase {
    #[default]
    Warming,
    Ready,
    Degraded,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct EmbeddingRuntimeStatus {
    pub phase: EmbeddingPhase,
    pub queryable: bool,
    pub indexed_notes: usize,
    pub total_notes: usize,
    pub pending_notes: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_error: Option<String>,
}

#[derive(Clone)]
pub struct EmbeddingRuntime {
    control: Arc<RuntimeControl>,
}

/// Non-owning submit handle for background producers such as filesystem watchers.
///
/// A watcher may outlive its owning `Vault` or daemon context by one scheduler
/// turn while its event channel closes. Keeping only a weak handle ensures that
/// this tail cannot keep the embedding coordinator alive or permit a late cache
/// write after the owner has been dropped.
#[derive(Clone)]
pub(crate) struct EmbeddingRuntimeWeak {
    control: Weak<RuntimeControl>,
}

struct RuntimeControl {
    shared: Arc<RuntimeShared>,
    worker: Mutex<Option<JoinHandle<()>>>,
}

impl RuntimeControl {
    fn retire(&self) {
        self.shared.live.store(false, Ordering::Release);
        // Synchronize with the final store commit or atomic cache publication.
        // An operation that already passed its liveness check may finish before
        // retirement returns, but it cannot mutate observable state afterward.
        drop(
            self.shared
                .lifecycle_gate
                .lock()
                .unwrap_or_else(|error| error.into_inner()),
        );
        self.shared.notify.notify_waiters();
        if let Some(worker) = self
            .worker
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take()
        {
            worker.abort();
        }
    }
}

impl Drop for RuntimeControl {
    fn drop(&mut self) {
        self.retire();
    }
}

struct RuntimeShared {
    source: Storage,
    index: Arc<RwLock<VaultIndex>>,
    cache: CacheFile,
    batch_size: usize,
    active: AtomicBool,
    state: Mutex<RuntimeState>,
    notify: Notify,
    live: Arc<AtomicBool>,
    lifecycle_gate: Arc<Mutex<()>>,
    first_load_error: OnceLock<String>,
}

impl RuntimeShared {
    /// Withdraw semantic publication after a source identity has been rejected.
    fn source_is_live(&self) -> bool {
        let Storage::Sftp(source) = &self.source else {
            return self.live.load(Ordering::Acquire);
        };
        let Some((crate::error::StorageErrorKind::IdentityChanged, message)) =
            source.last_failure()
        else {
            return self.live.load(Ordering::Acquire);
        };
        self.live.store(false, Ordering::Release);
        let total_notes = self
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .status
            .total_notes;
        record_terminal_failure_message(self, message, total_notes);
        self.notify.notify_waiters();
        false
    }
}

#[derive(Default)]
struct RuntimeState {
    next_generation: u64,
    pending: HashMap<OsString, PendingWork>,
    inflight: HashMap<OsString, (u64, PendingKind)>,
    batch_started_at: Option<Instant>,
    initial_remaining: HashSet<OsString>,
    initial_started: bool,
    reconciliation_complete: bool,
    failures: HashMap<OsString, String>,
    persistence_error: Option<String>,
    cache_error: Option<String>,
    staged: HashMap<OsString, StagedNote>,
    model: Option<Arc<dyn Embedder>>,
    store: Option<Arc<RwLock<EmbeddingStore>>>,
    store_queryable: bool,
    status: EmbeddingRuntimeStatus,
}

/// Directory authority for the finite set of work due at one observation.
/// Provider batches and preparation admission remain independently bounded.
struct ObservationPass<'a> {
    generations: HashMap<OsString, u64>,
    observed: VaultResult<super::storage::ObservedBatch<'a>>,
}

impl ObservationPass<'_> {
    fn contains(&self, batch: &[PendingWork]) -> bool {
        batch.iter().all(|work| {
            work.kind == PendingKind::Remove
                || self.generations.get(work.path.as_os_str()) == Some(&work.generation)
        })
    }

    fn consume(&mut self, batch: &[PendingWork]) {
        for work in batch {
            if self.generations.get(work.path.as_os_str()) == Some(&work.generation) {
                self.generations.remove(work.path.as_os_str());
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PendingKind {
    /// Invalidates older work and keeps the last published evidence queryable.
    /// It becomes runnable only after the corresponding lexical commit.
    AwaitingLexical,
    Upsert,
    Remove,
}

#[derive(Debug, Clone)]
struct PendingWork {
    path: PathBuf,
    generation: u64,
    kind: PendingKind,
    attempt: u8,
    not_before: Instant,
}

struct StagedNote {
    work: PendingWork,
    prepared: Arc<PreparedNote>,
    vectors: Vec<Option<Vec<f32>>>,
    hash_order: Vec<usize>,
    memory_bytes: usize,
    last_used: Instant,
}

impl StagedNote {
    fn complete(&self) -> bool {
        self.vectors.len() == self.prepared.chunks.len() && self.vectors.iter().all(Option::is_some)
    }

    fn into_entry(self) -> VaultResult<NoteEmbeddingEntry> {
        if !self.complete() {
            return Err(VaultError::Embedding(
                "attempted to publish an incomplete note embedding".into(),
            ));
        }
        let prepared = Arc::try_unwrap(self.prepared).map_err(|_| {
            VaultError::Embedding(
                "embedding preparation is still borrowed during publication".into(),
            )
        })?;
        let mut chunks = Vec::with_capacity(prepared.chunks.len());
        for (chunk, vector) in prepared.chunks.into_iter().zip(self.vectors) {
            let vector = vector.ok_or_else(|| {
                VaultError::Embedding("attempted to publish an incomplete note embedding".into())
            })?;
            chunks.push(ChunkEmbedding {
                input_hash: chunk.input_hash,
                vector,
                scope: chunk.scope,
                range: chunk.range,
                overlap_range: chunk.overlap_range,
                heading_path: chunk.heading_path,
                section_range: chunk.section_range,
                block_kind: chunk.block_kind,
            });
        }
        Ok(NoteEmbeddingEntry {
            source_revision: prepared.source_revision,
            source_bytes: prepared.source_bytes,
            body_range: prepared.body_range,
            title: prepared.title,
            tags: prepared.tags,
            chunks,
        })
    }
}

#[derive(Clone)]
pub(crate) struct EmbeddingQuerySnapshot {
    model: Arc<dyn Embedder>,
    store: Arc<RwLock<EmbeddingStore>>,
}

impl EmbeddingQuerySnapshot {
    pub(crate) fn capture_entries(
        &self,
        allowed_paths: &HashSet<OsString>,
    ) -> HashMap<OsString, Arc<NoteEmbeddingEntry>> {
        self.store
            .read()
            .unwrap_or_else(|error| error.into_inner())
            .capture_entries(allowed_paths)
    }

    pub(crate) fn validate_query(&self, query: &str) -> VaultResult<()> {
        let tokens = self.model.count_tokens(query)?;
        let max_tokens = self.model.input_policy().max_input_tokens;
        if tokens > max_tokens {
            return Err(VaultError::EmbeddingInputTooLong { tokens, max_tokens });
        }
        Ok(())
    }

    pub(crate) fn embed_query(&self, query: &str) -> VaultResult<Vec<f32>> {
        self.model.embed_query(query)
    }

    #[cfg(test)]
    pub(crate) fn semantic_scores(
        &self,
        query: &str,
        top_k: usize,
    ) -> VaultResult<Vec<(PathBuf, f32)>> {
        let query_vector = self.embed_query(query)?;
        let store = self.store.read().unwrap_or_else(|error| error.into_inner());
        Ok(store.query(&query_vector, top_k))
    }

    #[cfg(test)]
    pub(crate) fn score_for(&self, path: &Path, query_vector: &[f32]) -> f32 {
        let store = self.store.read().unwrap_or_else(|error| error.into_inner());
        store
            .get(path)
            .map(|vector| super::embeddings::cosine_similarity(query_vector, vector))
            .unwrap_or(0.0)
    }
}

impl EmbeddingRuntime {
    #[cfg(test)]
    pub(crate) fn spawn<F>(
        vault_root: PathBuf,
        index: Arc<RwLock<VaultIndex>>,
        cache_path: PathBuf,
        loader: F,
    ) -> Self
    where
        F: Future<Output = VaultResult<Arc<dyn Embedder>>> + Send + 'static,
    {
        Self::spawn_for_test(vault_root, index, cache_path, TEST_BATCH_SIZE, true, loader)
    }

    #[cfg(test)]
    fn spawn_for_test<F>(
        vault_root: PathBuf,
        index: Arc<RwLock<VaultIndex>>,
        cache_path: PathBuf,
        batch_size: usize,
        initially_active: bool,
        loader: F,
    ) -> Self
    where
        F: Future<Output = VaultResult<Arc<dyn Embedder>>> + Send + 'static,
    {
        let root = super::fs::VaultRoot::open(&vault_root).unwrap();
        let cache = CacheFile::new(
            root.clone(),
            cache_path.strip_prefix(vault_root).unwrap().to_owned(),
        );
        Self::spawn_with_scoped_cache(root, index, cache, batch_size, initially_active, loader)
    }

    pub(crate) fn spawn_with_scoped_cache<F>(
        vault_root: super::fs::VaultRoot,
        index: Arc<RwLock<VaultIndex>>,
        cache: CacheFile,
        batch_size: usize,
        initially_active: bool,
        loader: F,
    ) -> Self
    where
        F: Future<Output = VaultResult<Arc<dyn Embedder>>> + Send + 'static,
    {
        Self::spawn_with_source(
            Storage::Local(vault_root),
            index,
            cache,
            batch_size,
            initially_active,
            loader,
        )
    }

    pub(crate) fn spawn_with_source<F>(
        source: Storage,
        index: Arc<RwLock<VaultIndex>>,
        cache: CacheFile,
        batch_size: usize,
        initially_active: bool,
        loader: F,
    ) -> Self
    where
        F: Future<Output = VaultResult<Arc<dyn Embedder>>> + Send + 'static,
    {
        #[cfg(test)]
        tests::initialize_test_logging();
        debug_assert!(batch_size > 0);
        let initial_paths = initial_embedding_work(&index);
        let total_notes = initial_paths.len();
        let mut state = RuntimeState {
            status: EmbeddingRuntimeStatus {
                total_notes,
                ..EmbeddingRuntimeStatus::default()
            },
            ..RuntimeState::default()
        };
        for (path, kind) in initial_paths {
            let generation = next_generation(&mut state);
            state.pending.insert(
                path.as_os_str().to_owned(),
                PendingWork {
                    path,
                    generation,
                    kind,
                    attempt: 0,
                    not_before: Instant::now(),
                },
            );
        }
        update_pending_status(&mut state);
        let shared = Arc::new(RuntimeShared {
            source,
            index,
            cache,
            batch_size,
            active: AtomicBool::new(initially_active),
            state: Mutex::new(state),
            notify: Notify::new(),
            live: Arc::new(AtomicBool::new(true)),
            lifecycle_gate: Arc::new(Mutex::new(())),
            first_load_error: OnceLock::new(),
        });
        let worker_shared = Arc::clone(&shared);
        let span = tracing::info_span!(
            "embedding_runtime",
            source = if shared.source.is_remote() { "sftp" } else { "local" },
            vault = %shared.source.path().display(),
        );
        let worker = tokio::spawn(
            async move {
                let coordinator = run_coordinator(Arc::clone(&worker_shared), loader);
                tokio::pin!(coordinator);
                let mut progress = tokio::time::interval(PROGRESS_INTERVAL);
                progress.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                loop {
                    tokio::select! {
                        _ = &mut coordinator => break,
                        _ = progress.tick() => report_progress(&worker_shared),
                    }
                }
            }
            .instrument(span)
            .with_current_subscriber(),
        );
        Self {
            control: Arc::new(RuntimeControl {
                shared,
                worker: Mutex::new(Some(worker)),
            }),
        }
    }

    /// Withdraw document dispatch and cache publication even if a query still owns this runtime.
    pub(crate) fn retire(&self) {
        self.control.retire();
    }

    /// Activate the existing coordinator once, preserving queued mutation intents.
    pub(crate) fn activate(&self) -> bool {
        if !self.control.shared.source_is_live() {
            return false;
        }
        let activated = !self.control.shared.active.swap(true, Ordering::AcqRel);
        if activated {
            self.control.shared.notify.notify_one();
        }
        activated
    }

    pub(crate) fn is_active(&self) -> bool {
        self.control.shared.active.load(Ordering::Acquire)
    }

    pub(crate) fn submit_upsert(&self, path: &Path) {
        self.submit(path, PendingKind::Upsert);
    }

    /// Report existing evidence or retained note work without creating a generation.
    /// Called while the source index is locked; use only the state-to-store lock order.
    pub(crate) fn has_current_or_pending(&self, path: &Path, source_revision: &str) -> bool {
        if !self.control.shared.source_is_live() {
            return false;
        }
        let Ok(path) = normalize_intent_path(path, self.control.shared.source.is_remote()) else {
            return false;
        };
        let state = self
            .control
            .shared
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let current = state_generation(&state, &path);
        let removing = state
            .pending
            .get(path.as_os_str())
            .is_some_and(|work| work.kind == PendingKind::Remove)
            || state
                .inflight
                .get(path.as_os_str())
                .is_some_and(|(generation, kind)| {
                    current == Some(*generation) && *kind == PendingKind::Remove
                });
        if removing {
            return false;
        }
        let note_work = |generation, kind| {
            current == Some(generation)
                && matches!(kind, PendingKind::Upsert | PendingKind::AwaitingLexical)
        };
        if state
            .pending
            .get(path.as_os_str())
            .is_some_and(|work| note_work(work.generation, work.kind))
            || state
                .inflight
                .get(path.as_os_str())
                .is_some_and(|(generation, kind)| note_work(*generation, *kind))
            || state
                .staged
                .get(path.as_os_str())
                .is_some_and(|draft| note_work(draft.work.generation, draft.work.kind))
        {
            return true;
        }
        state.store.as_ref().is_some_and(|store| {
            store
                .read()
                .unwrap_or_else(|error| error.into_inner())
                .note(&path)
                .is_some_and(|entry| entry.source_revision == source_revision)
        })
    }

    #[cfg(test)]
    pub(crate) fn submit_remove(&self, path: &Path) {
        self.submit(path, PendingKind::Remove);
    }

    pub(crate) fn downgrade(&self) -> EmbeddingRuntimeWeak {
        EmbeddingRuntimeWeak {
            control: Arc::downgrade(&self.control),
        }
    }

    pub(super) fn submit(&self, path: &Path, kind: PendingKind) {
        if !self.control.shared.source_is_live() {
            return;
        }
        let normalized = match normalize_intent_path(path, self.control.shared.source.is_remote()) {
            Ok(path) if !path.as_os_str().is_empty() => path,
            Ok(_) => return,
            Err(error) => {
                tracing::warn!(path = %path.display(), error = %error, "ignoring invalid embedding intent path");
                return;
            }
        };
        let mut state = self
            .control
            .shared
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let generation = next_generation(&mut state);
        state.staged.remove(normalized.as_os_str());
        state.failures.remove(normalized.as_os_str());
        state.pending.insert(
            normalized.as_os_str().to_owned(),
            PendingWork {
                path: normalized,
                generation,
                kind,
                attempt: 0,
                not_before: Instant::now(),
            },
        );
        update_pending_status(&mut state);
        drop(state);
        if kind != PendingKind::AwaitingLexical {
            self.control.shared.notify.notify_one();
        }
    }

    pub(crate) fn status(&self) -> EmbeddingRuntimeStatus {
        let live = self.control.shared.source_is_live();
        let paths = current_paths(&self.control.shared.index);
        let state = self
            .control
            .shared
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let mut status = status_for_current_paths(&state, &paths);
        if !live {
            status.queryable = false;
        }
        status
    }

    pub(crate) fn query_snapshot(&self) -> VaultResult<EmbeddingQuerySnapshot> {
        if !self.control.shared.source_is_live() {
            let state = self
                .control
                .shared
                .state
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            return Err(VaultError::Embedding(
                state
                    .status
                    .last_error
                    .clone()
                    .unwrap_or_else(|| "embedding runtime has stopped".into()),
            ));
        }
        {
            let state = self
                .control
                .shared
                .state
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            if let (Some(model), Some(store)) = (&state.model, &state.store)
                && state.store_queryable
            {
                return Ok(EmbeddingQuerySnapshot {
                    model: Arc::clone(model),
                    store: Arc::clone(store),
                });
            }
        }

        let paths = current_paths(&self.control.shared.index);
        let state = self
            .control
            .shared
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let (Some(model), Some(store)) = (&state.model, &state.store)
            && state.store_queryable
        {
            return Ok(EmbeddingQuerySnapshot {
                model: Arc::clone(model),
                store: Arc::clone(store),
            });
        }
        let status = status_for_current_paths(&state, &paths);
        Err(VaultError::Embedding(not_ready_message(&status)))
    }

    pub(crate) fn first_load_error(&self) -> Option<&str> {
        self.control
            .shared
            .first_load_error
            .get()
            .map(String::as_str)
    }

    #[cfg(test)]
    pub(crate) fn pending_kind(&self, path: &Path) -> Option<PendingKind> {
        let normalized =
            normalize_intent_path(path, self.control.shared.source.is_remote()).ok()?;
        self.control
            .shared
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .pending
            .get(normalized.as_os_str())
            .map(|work| work.kind)
    }
}

impl EmbeddingRuntimeWeak {
    pub(crate) fn submit_upsert(&self, path: &Path) {
        self.submit(path, PendingKind::Upsert);
    }

    pub(super) fn submit(&self, path: &Path, kind: PendingKind) {
        let Some(control) = self.control.upgrade() else {
            return;
        };
        EmbeddingRuntime { control }.submit(path, kind);
    }

    #[cfg(test)]
    fn is_alive(&self) -> bool {
        self.control.upgrade().is_some()
    }
}

async fn run_coordinator<F>(shared: Arc<RuntimeShared>, loader: F)
where
    F: Future<Output = VaultResult<Arc<dyn Embedder>>> + Send,
{
    while !shared.active.load(Ordering::Acquire) {
        let notified = shared.notify.notified();
        if !shared.source_is_live() {
            return;
        }
        if shared.active.load(Ordering::Acquire) {
            break;
        }
        notified.await;
    }
    if !shared.source_is_live() {
        return;
    }

    let model = match loader.await {
        Ok(model) => model,
        Err(error) => {
            record_terminal_failure(&shared, error);
            return;
        }
    };
    if !shared.source_is_live() {
        return;
    }

    let current_paths = current_paths(&shared.index);
    let expected_identity = model.space_identity().clone();
    let cache = shared.cache.clone();
    let current_count = current_paths.len();
    let load_cache = move || cache.load(&expected_identity, current_count);
    let loaded = tokio::task::spawn_blocking(load_cache).await;
    let (loaded, cache_error) = match loaded {
        Ok(Ok(store)) => (store, None),
        Ok(Err(error @ (VaultError::OutsideVault(_) | VaultError::InvalidPath(_)))) => {
            record_terminal_failure(&shared, error);
            return;
        }
        Ok(Err(VaultError::EmbeddingCacheIncompatible(reason))) => {
            tracing::info!(reason = %reason, "rebuilding incompatible embedding cache");
            (None, None)
        }
        Ok(Err(error)) => {
            let message = format!("embedding cache could not be loaded; rebuilding: {error}");
            tracing::warn!(error = %message, "embedding cache load failed");
            (None, Some(message))
        }
        Err(error) => {
            let message = format!("embedding cache task failed; rebuilding: {error}");
            tracing::warn!(error = %message, "embedding cache load failed");
            (None, Some(message))
        }
    };
    if !shared.source_is_live() {
        return;
    }

    let mut store =
        loaded.unwrap_or_else(|| EmbeddingStore::new_with_identity(model.space_identity().clone()));
    let previously_publishable = store.first_pass_complete();
    let mut dirty = match store.retain_paths(&current_paths) {
        Ok(dirty) => dirty,
        Err(error) => {
            record_terminal_failure(&shared, error);
            return;
        }
    };
    let has_cached_vectors = !store.is_empty();
    let store = Arc::new(RwLock::new(store));

    {
        let initial_paths = initial_embedding_work(&shared.index);
        let mut state = shared
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        state.cache_error = cache_error;
        state.model = Some(Arc::clone(&model));
        state.store = Some(Arc::clone(&store));
        state.store_queryable =
            previously_publishable && (current_paths.is_empty() || has_cached_vectors);
        state.initial_started = true;
        state.initial_remaining = initial_paths
            .iter()
            .map(|(path, _)| path.as_os_str().to_owned())
            .collect();
        for (path, kind) in initial_paths {
            if !state.pending.contains_key(path.as_os_str()) {
                let generation = next_generation(&mut state);
                state.pending.insert(
                    path.as_os_str().to_owned(),
                    PendingWork {
                        path,
                        generation,
                        kind,
                        attempt: 0,
                        not_before: Instant::now(),
                    },
                );
            }
        }
        update_pending_status(&mut state);
    }

    dirty |= finalize_reconciliation_if_complete(&shared, &store);
    if !shared.source_is_live() {
        return;
    }
    refresh_status(&shared, &store);

    let mut last_persist = Instant::now();
    let mut persist_retry_at = Instant::now();
    let mut persist_attempt = 0u8;
    let mut observation: Option<ObservationPass<'_>> = None;

    loop {
        if !shared.source_is_live() {
            return;
        }
        let notified = shared.notify.notified();
        let (batch, next_pending, pending_empty) = take_due_batch(&shared);
        if !batch.is_empty() {
            if batch.iter().all(|work| work.kind == PendingKind::Remove) {
                for work in &batch {
                    dirty |= commit_remove(&shared, &store, work);
                }
                if let Some(pass) = observation.as_mut() {
                    pass.consume(&batch);
                }
            } else {
                if observation
                    .as_ref()
                    .is_none_or(|pass| !pass.contains(&batch))
                {
                    let generations = due_observation_paths(&shared, &batch);
                    let paths = generations.keys().map(PathBuf::from).collect::<Vec<_>>();
                    observation = Some(ObservationPass {
                        generations,
                        observed: shared.source.observe_batch(&paths).await,
                    });
                }
                if let Some(pass) = observation.as_mut() {
                    let (changed, source_failed) =
                        process_batch(&shared, &store, &model, &batch, pass.observed.as_ref())
                            .await;
                    dirty |= changed;
                    pass.consume(&batch);
                    if source_failed || pass.generations.is_empty() {
                        observation = None;
                    }
                }
            }
            if !shared.source_is_live() {
                return;
            }
            shared
                .state
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .batch_started_at = None;
            dirty |= finalize_reconciliation_if_complete(&shared, &store);
            refresh_status(&shared, &store);
            let now = Instant::now();
            if dirty
                && now >= persist_retry_at
                && now.duration_since(last_persist) >= MAX_DIRTY_INTERVAL
                && !persist_dirty(
                    &shared,
                    &store,
                    &mut dirty,
                    &mut last_persist,
                    &mut persist_retry_at,
                    &mut persist_attempt,
                )
                .await
            {
                return;
            }
            continue;
        }
        observation = None;

        let now = Instant::now();
        let persistence_due = dirty
            && now >= persist_retry_at
            && (pending_empty || now.duration_since(last_persist) >= MAX_DIRTY_INTERVAL);
        if persistence_due {
            if !persist_dirty(
                &shared,
                &store,
                &mut dirty,
                &mut last_persist,
                &mut persist_retry_at,
                &mut persist_attempt,
            )
            .await
            {
                return;
            }
            continue;
        }

        let persistence_wake = dirty.then(|| {
            if pending_empty {
                persist_retry_at
            } else {
                std::cmp::max(persist_retry_at, last_persist + MAX_DIRTY_INTERVAL)
            }
        });
        let wake_at = match (next_pending, persistence_wake) {
            (Some(left), Some(right)) => Some(std::cmp::min(left, right)),
            (Some(value), None) | (None, Some(value)) => Some(value),
            (None, None) => None,
        };
        if let Some(wake_at) = wake_at {
            tokio::select! {
                _ = notified => {}
                _ = tokio::time::sleep_until(tokio::time::Instant::from_std(wake_at)) => {}
            }
        } else {
            notified.await;
        }
    }
}

fn record_terminal_failure(shared: &RuntimeShared, error: VaultError) {
    let total_notes = current_paths(&shared.index).len();
    record_terminal_failure_message(shared, error.to_string(), total_notes);
}

fn record_terminal_failure_message(shared: &RuntimeShared, message: String, total_notes: usize) {
    let _ = shared.first_load_error.set(message.clone());
    let mut state = shared
        .state
        .lock()
        .unwrap_or_else(|lock_error| lock_error.into_inner());
    // A confinement failure can also occur after a snapshot became queryable.
    // Drop the coordinator's published model and store before reporting it.
    state.model = None;
    state.store = None;
    state.store_queryable = false;
    let pending_notes = distinct_pending_count(&state);
    replace_status(
        &mut state,
        EmbeddingRuntimeStatus {
            phase: EmbeddingPhase::Degraded,
            queryable: false,
            indexed_notes: 0,
            total_notes,
            pending_notes,
            last_error: Some(message),
        },
    );
}

async fn persist_dirty(
    shared: &Arc<RuntimeShared>,
    store: &Arc<RwLock<EmbeddingStore>>,
    dirty: &mut bool,
    last_persist: &mut Instant,
    persist_retry_at: &mut Instant,
    persist_attempt: &mut u8,
) -> bool {
    match persist_store(shared, store).await {
        Ok(true) => {
            *dirty = false;
            *persist_attempt = 0;
            *persist_retry_at = Instant::now();
            *last_persist = Instant::now();
            let mut state = shared
                .state
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            state.persistence_error = None;
        }
        Ok(false) => return false,
        Err(error @ (VaultError::OutsideVault(_) | VaultError::InvalidPath(_))) => {
            record_terminal_failure(shared, error);
            return false;
        }
        Err(error) => {
            *persist_attempt = persist_attempt.saturating_add(1);
            *persist_retry_at = Instant::now() + retry_delay(*persist_attempt);
            let mut state = shared
                .state
                .lock()
                .unwrap_or_else(|lock_error| lock_error.into_inner());
            state.persistence_error = Some(error.to_string());
        }
    }
    refresh_status(shared, store);
    true
}

fn take_due_batch(shared: &RuntimeShared) -> (Vec<PendingWork>, Option<Instant>, bool) {
    let now = Instant::now();
    let mut state = shared
        .state
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let mut due = state
        .pending
        .iter()
        .filter(|(_, work)| work.kind != PendingKind::AwaitingLexical && work.not_before <= now)
        .map(|(path, work)| (path.clone(), work.kind, work.generation))
        .collect::<Vec<_>>();
    due.sort_unstable_by(
        |(left_path, left_kind, left_generation), (right_path, right_kind, right_generation)| {
            let left_priority = usize::from(*left_kind == PendingKind::Upsert);
            let right_priority = usize::from(*right_kind == PendingKind::Upsert);
            left_priority
                .cmp(&right_priority)
                .then_with(|| left_generation.cmp(right_generation))
                .then_with(|| left_path.cmp(right_path))
        },
    );
    due.truncate(shared.batch_size);

    let mut batch = Vec::with_capacity(due.len());
    for (path, _, _) in due {
        if let Some(work) = state.pending.remove(path.as_os_str()) {
            state.inflight.insert(path, (work.generation, work.kind));
            batch.push(work);
        }
    }
    let next_pending = state
        .pending
        .values()
        .filter(|work| work.kind != PendingKind::AwaitingLexical)
        .map(|work| work.not_before)
        .min();
    let pending_empty = state.pending.is_empty() && state.inflight.is_empty();
    state.batch_started_at = (!batch.is_empty()).then_some(now);
    update_pending_status(&mut state);
    (batch, next_pending, pending_empty)
}

fn due_observation_paths(shared: &RuntimeShared, batch: &[PendingWork]) -> HashMap<OsString, u64> {
    let now = Instant::now();
    let state = shared
        .state
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    state
        .pending
        .values()
        .chain(batch)
        .filter(|work| work.kind == PendingKind::Upsert && work.not_before <= now)
        .map(|work| (work.path.as_os_str().to_owned(), work.generation))
        .collect()
}

async fn process_batch(
    shared: &Arc<RuntimeShared>,
    store: &Arc<RwLock<EmbeddingStore>>,
    model: &Arc<dyn Embedder>,
    batch: &[PendingWork],
    observed: Result<&super::storage::ObservedBatch<'_>, &VaultError>,
) -> (bool, bool) {
    let mut dirty = false;
    let mut source_failed = observed.is_err();
    let mut active: Vec<Option<StagedNote>> = Vec::new();
    let mut active_bytes = 0usize;
    // Source bodies enter preparation individually under its staging budget.
    for group in batch.chunks(32) {
        for work in group.iter().filter(|work| work.kind == PendingKind::Remove) {
            dirty |= commit_remove(shared, store, work);
        }
        for work in group
            .iter()
            .filter(|work| work.kind != PendingKind::Remove)
            .cloned()
        {
            if !work_is_current(shared, &work) {
                discard_inflight(shared, &work);
                continue;
            }
            let existing_draft = shared
                .state
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .staged
                .remove(work.path.as_os_str())
                .filter(|draft| draft.work.generation == work.generation);
            let mut draft = if let Some(mut draft) = existing_draft {
                draft.work = work.clone();
                draft
            } else {
                let observed = match observed {
                    Ok(observed) => observed,
                    Err(error) => {
                        record_failure_message(
                            shared,
                            work,
                            error.to_string(),
                            retry_hint(error),
                            None,
                        );
                        continue;
                    }
                };
                let previous = store
                    .read()
                    .unwrap_or_else(|error| error.into_inner())
                    .note(&work.path);
                let reuse_index_bytes = previous.as_ref().map_or(0, |note| {
                    std::mem::size_of::<Vec<&ChunkEmbedding>>().saturating_add(
                        note.chunks
                            .len()
                            .saturating_mul(std::mem::size_of::<&ChunkEmbedding>()),
                    )
                });
                let prepared = match prepare_work_observed(
                    shared,
                    model,
                    &work,
                    active_bytes,
                    reuse_index_bytes,
                    previous.clone(),
                    observed,
                )
                .await
                {
                    Ok(PreparationOutcome::Prepared(prepared)) => prepared,
                    Ok(PreparationOutcome::Missing) => {
                        dirty |= commit_remove(shared, store, &work);
                        continue;
                    }
                    Ok(PreparationOutcome::Unchanged) => {
                        mark_success(shared, &work);
                        continue;
                    }
                    Ok(PreparationOutcome::Deferred) => {
                        defer_pending(shared, &work);
                        continue;
                    }
                    Err(error) => {
                        source_failed |= matches!(
                            error,
                            VaultError::Storage {
                                kind: crate::error::StorageErrorKind::Unavailable
                                    | crate::error::StorageErrorKind::Protocol
                                    | crate::error::StorageErrorKind::IdentityChanged,
                                ..
                            }
                        );
                        record_failure(shared, work, error, None);
                        continue;
                    }
                };
                let memory_bytes = match staging_size(&prepared, model.dimension(), &work) {
                    Ok(bytes) => bytes,
                    Err(error) => {
                        record_failure(shared, work, error, None);
                        continue;
                    }
                };
                // The preparation reservation includes every eventual vector, so
                // reusable vectors are copied only after their complete note is admitted.
                if !admit_staging(
                    shared,
                    active_bytes,
                    memory_bytes.saturating_add(reuse_index_bytes),
                ) {
                    defer_pending(shared, &work);
                    continue;
                }
                let vectors = {
                    let mut reusable = previous.as_ref().map(|note| {
                        let mut index = Vec::with_capacity(note.chunks.len());
                        index.extend(note.chunks.iter());
                        index.sort_unstable_by_key(|chunk| chunk.input_hash);
                        index
                    });
                    let mut vectors = Vec::with_capacity(prepared.chunks.len());
                    for chunk in &prepared.chunks {
                        let cached = reusable.as_mut().and_then(|index| {
                            index
                                .binary_search_by_key(&chunk.input_hash, |cached| cached.input_hash)
                                .ok()
                                .map(|offset| index[offset])
                        });
                        vectors.push(cached.map(|cached| cached.vector.clone()));
                    }
                    vectors
                };
                let mut hash_order = (0..prepared.chunks.len()).collect::<Vec<_>>();
                hash_order
                    .sort_unstable_by_key(|&index| (prepared.chunks[index].input_hash, index));
                StagedNote {
                    work: work.clone(),
                    prepared: Arc::new(prepared),
                    vectors,
                    hash_order,
                    memory_bytes,
                    last_used: Instant::now(),
                }
            };
            if !work_is_current(shared, &work) {
                discard_inflight(shared, &work);
                continue;
            }
            if !admit_staging(shared, active_bytes, draft.memory_bytes) {
                draft.last_used = Instant::now();
                defer_work(shared, draft, false);
                continue;
            }
            if draft.complete() {
                let work = draft.work.clone();
                match draft.into_entry() {
                    Ok(entry) => dirty |= commit_note(shared, store, &work, entry),
                    Err(error) => record_failure(shared, work, error, None),
                }
            } else {
                active_bytes += draft.memory_bytes;
                active.push(Some(draft));
            }
        }
    }

    loop {
        let capacity = active
            .iter()
            .filter_map(Option::as_ref)
            .map(|draft| draft.prepared.chunks.len())
            .sum();
        let mut jobs = Vec::with_capacity(capacity);
        let mut inputs = Vec::with_capacity(capacity);
        for (note_index, slot) in active.iter_mut().enumerate() {
            let Some(draft) = slot.as_ref() else {
                continue;
            };
            if !work_is_current(shared, &draft.work) {
                discard_inflight(shared, &draft.work);
                active_bytes = active_bytes.saturating_sub(draft.memory_bytes);
                *slot = None;
                continue;
            }
            for (chunk_index, (chunk, vector)) in
                draft.prepared.chunks.iter().zip(&draft.vectors).enumerate()
            {
                if vector.is_some() {
                    continue;
                }
                let first = draft.hash_order.partition_point(|&index| {
                    draft.prepared.chunks[index].input_hash < chunk.input_hash
                });
                if draft.hash_order[first] != chunk_index {
                    continue;
                }
                jobs.push((note_index, chunk_index));
                inputs.push((Arc::clone(&draft.prepared), chunk_index));
            }
        }
        if jobs.is_empty() {
            break;
        }
        let embedder = Arc::clone(model);
        let source_owner = Arc::clone(shared);
        let owner_live = Arc::clone(&shared.live);
        let started = Instant::now();
        let dispatcher = tracing::dispatcher::get_default(Clone::clone);
        let span = tracing::Span::current();
        let inference = tokio::task::spawn_blocking(move || {
            tracing::dispatcher::with_default(&dispatcher, || {
                span.in_scope(|| {
                    if !source_owner.source_is_live() {
                        return (
                            0,
                            Err(VaultError::Embedding(
                                "embedding source authority was withdrawn".into(),
                            )),
                        );
                    }
                    let texts = inputs
                        .iter()
                        .map(|(prepared, chunk)| prepared.chunks[*chunk].text.as_str())
                        .collect::<Vec<_>>();
                    tracing::debug!(pending_inputs = texts.len(), "embedding batch started");
                    let (count, result) = embedder.embed_next_batch(&texts, &owner_live);
                    let result = result.and_then(|vectors| {
                        validate_embedding_batch(vectors, count, embedder.dimension())
                    });
                    (count, result)
                })
            })
        })
        .await;
        let (count, result) = match inference {
            Ok(result) => result,
            Err(error) => (
                jobs.len(),
                Err(VaultError::EmbeddingRequest {
                    message: format!("embedding inference task failed: {error}"),
                    retryable: true,
                    retry_after: None,
                }),
            ),
        };
        match result {
            Ok(vectors) => {
                tracing::debug!(
                    batch_size = count,
                    duration_ms = started.elapsed().as_millis() as u64,
                    outcome = "completed",
                    "embedding batch finished"
                );
                for ((note_index, chunk_index), vector) in jobs.iter().take(count).zip(vectors) {
                    if let Some(draft) = active[*note_index].as_mut() {
                        let input_hash = draft.prepared.chunks[*chunk_index].input_hash;
                        let first = draft.hash_order.partition_point(|&index| {
                            draft.prepared.chunks[index].input_hash < input_hash
                        });
                        let end = draft.hash_order.partition_point(|&index| {
                            draft.prepared.chunks[index].input_hash <= input_hash
                        });
                        for &index in &draft.hash_order[first..end] {
                            if index != *chunk_index && draft.vectors[index].is_none() {
                                draft.vectors[index] = Some(vector.clone());
                            }
                        }
                        draft.vectors[*chunk_index] = Some(vector);
                    }
                }
                for slot in &mut active {
                    if slot.as_ref().is_some_and(StagedNote::complete)
                        && let Some(draft) = slot.take()
                    {
                        active_bytes = active_bytes.saturating_sub(draft.memory_bytes);
                        let work = draft.work.clone();
                        match draft.into_entry() {
                            Ok(entry) => dirty |= commit_note(shared, store, &work, entry),
                            Err(error) => record_failure(shared, work, error, None),
                        }
                    }
                }
            }
            Err(error) => {
                tracing::debug!(batch_size = count, duration_ms = started.elapsed().as_millis() as u64, outcome = "failed", error = %error, "embedding batch finished");
                let affected = jobs
                    .iter()
                    .take(count)
                    .map(|(note, _)| *note)
                    .collect::<HashSet<_>>();
                let retry_after = retry_hint(&error);
                let message = error.to_string();
                for (index, slot) in active.iter_mut().enumerate() {
                    if let Some(draft) = slot.take() {
                        if affected.contains(&index) {
                            record_failure_message(
                                shared,
                                draft.work.clone(),
                                message.clone(),
                                retry_after,
                                Some(draft),
                            );
                        } else {
                            defer_work(shared, draft, true);
                        }
                    }
                }
                break;
            }
        }
    }
    (dirty, source_failed)
}

enum PreparationOutcome {
    Prepared(PreparedNote),
    Unchanged,
    Missing,
    Deferred,
}

fn draft_overhead(work: &PendingWork) -> usize {
    std::mem::size_of::<StagedNote>()
        .saturating_add(2 * std::mem::size_of::<usize>())
        .saturating_add(work.path.capacity().saturating_mul(2))
}

fn available_staging(shared: &RuntimeShared, active_bytes: usize) -> usize {
    let state = shared
        .state
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let paused = state
        .staged
        .values()
        .map(|draft| draft.memory_bytes)
        .sum::<usize>();
    MAX_STAGING_BYTES.saturating_sub(active_bytes.saturating_add(paused))
}

#[cfg(test)]
async fn prepare_work(
    shared: &Arc<RuntimeShared>,
    model: &Arc<dyn Embedder>,
    work: &PendingWork,
    active_bytes: usize,
    reuse_index_bytes: usize,
    previous: Option<Arc<NoteEmbeddingEntry>>,
) -> VaultResult<PreparationOutcome> {
    let observed = shared
        .source
        .observe_batch(std::slice::from_ref(&work.path))
        .await?;
    prepare_work_observed(
        shared,
        model,
        work,
        active_bytes,
        reuse_index_bytes,
        previous,
        &observed,
    )
    .await
}

async fn prepare_work_observed(
    shared: &Arc<RuntimeShared>,
    model: &Arc<dyn Embedder>,
    work: &PendingWork,
    active_bytes: usize,
    reuse_index_bytes: usize,
    previous: Option<Arc<NoteEmbeddingEntry>>,
    observed: &super::storage::ObservedBatch<'_>,
) -> VaultResult<PreparationOutcome> {
    loop {
        let available = available_staging(shared, active_bytes);
        let overhead = draft_overhead(work).saturating_add(reuse_index_bytes);
        let read_limit = super::source::MAX_SOURCE_BYTES
            .min(available.saturating_sub(overhead.saturating_add(1)));
        let prepared = if available <= overhead {
            Err(VaultError::SemanticResourceLimit {
                resource: "embedding preparation budget".into(),
                actual: overhead.saturating_add(1),
                limit: available,
            })
        } else {
            let path = work.path.clone();
            if shared
                .index
                .read()
                .unwrap_or_else(|error| error.into_inner())
                .get_note(&path)
                .is_none()
            {
                return Ok(PreparationOutcome::Missing);
            }
            let embedder = Arc::clone(model);
            let previous = previous.clone();
            let raw = observed.read_exact_bounded(&path, read_limit).await;
            match raw {
                Err(error) => Err(error),
                Ok(raw) => tokio::task::spawn_blocking(move || {
                    if previous.as_ref().is_some_and(|previous| {
                        super::source::source_revision_matches(&raw, &previous.source_revision)
                    }) {
                        return Ok(PreparationOutcome::Unchanged);
                    }
                    let reserved = overhead.saturating_add(raw.capacity());
                    let budget = available.saturating_sub(reserved);
                    prepare_note_with_budget(&path, &raw, embedder.as_ref(), budget)
                        .map(PreparationOutcome::Prepared)
                        .map_err(|error| match error {
                            VaultError::SemanticResourceLimit {
                                resource, actual, ..
                            } if resource == "embedding preparation budget" => {
                                VaultError::SemanticResourceLimit {
                                    resource,
                                    actual: reserved.saturating_add(actual),
                                    limit: available,
                                }
                            }
                            other => other,
                        })
                })
                .await
                .map_err(|error| VaultError::EmbeddingRequest {
                    message: format!("embedding preparation task failed: {error}"),
                    retryable: true,
                    retry_after: None,
                })?,
            }
        };
        match prepared {
            Err(VaultError::SemanticResourceLimit {
                resource,
                actual,
                limit,
            }) if resource == "embedding preparation budget"
                || (resource.starts_with("source bytes for ")
                    && limit < super::source::MAX_SOURCE_BYTES) =>
            {
                // A failed allocation measures only the work observed so far.
                // Convert it to the whole reservation before freeing retry progress.
                let required = if resource == "embedding preparation budget" {
                    actual
                } else {
                    overhead.saturating_add(actual).saturating_add(1)
                };
                if admit_staging(shared, active_bytes, required) {
                    continue;
                }
                if active_bytes > 0 {
                    return Ok(PreparationOutcome::Deferred);
                }
                return Err(VaultError::SemanticResourceLimit {
                    resource: "embedding preparation budget".into(),
                    actual: required,
                    limit: MAX_STAGING_BYTES,
                });
            }
            result => return result,
        }
    }
}

fn staging_size(
    prepared: &PreparedNote,
    dimension: usize,
    work: &PendingWork,
) -> VaultResult<usize> {
    let bytes = draft_overhead(work).saturating_add(prepared.staging_bytes(dimension));
    if bytes > MAX_STAGING_BYTES {
        return Err(VaultError::SemanticResourceLimit {
            resource: "unpublished embedding staging bytes".into(),
            actual: bytes,
            limit: MAX_STAGING_BYTES,
        });
    }
    Ok(bytes)
}

fn defer_pending(shared: &RuntimeShared, work: &PendingWork) {
    let mut state = shared
        .state
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    if state_generation(&state, &work.path) == Some(work.generation) {
        let mut deferred = work.clone();
        deferred.not_before = Instant::now();
        state
            .pending
            .insert(work.path.as_os_str().to_owned(), deferred);
    }
    if state
        .inflight
        .get(work.path.as_os_str())
        .map(|(generation, _)| generation)
        == Some(&work.generation)
    {
        state.inflight.remove(work.path.as_os_str());
    }
    update_pending_status(&mut state);
}

fn admit_staging(shared: &RuntimeShared, active_bytes: usize, required: usize) -> bool {
    // Paused progress cannot help when active work alone prevents admission.
    if active_bytes.saturating_add(required) > MAX_STAGING_BYTES {
        return false;
    }
    let mut state = shared
        .state
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    loop {
        let paused = state
            .staged
            .values()
            .map(|draft| draft.memory_bytes)
            .sum::<usize>();
        if active_bytes.saturating_add(paused).saturating_add(required) <= MAX_STAGING_BYTES {
            return true;
        }
        let oldest = state
            .staged
            .iter()
            .min_by_key(|(_, draft)| draft.last_used)
            .map(|(path, _)| path.clone());
        match oldest {
            Some(path) => {
                state.staged.remove(path.as_os_str());
            }
            None => return false,
        }
    }
}

fn work_is_current(shared: &RuntimeShared, work: &PendingWork) -> bool {
    shared.source_is_live()
        && state_generation(
            &shared
                .state
                .lock()
                .unwrap_or_else(|error| error.into_inner()),
            &work.path,
        ) == Some(work.generation)
}

fn discard_inflight(shared: &RuntimeShared, work: &PendingWork) {
    let mut state = shared
        .state
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    if state
        .inflight
        .get(work.path.as_os_str())
        .map(|(generation, _)| generation)
        == Some(&work.generation)
    {
        state.inflight.remove(work.path.as_os_str());
    }
    update_pending_status(&mut state);
}

fn defer_work(shared: &RuntimeShared, mut draft: StagedNote, retain: bool) {
    let mut state = shared
        .state
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let path = draft.work.path.clone();
    let generation = draft.work.generation;
    if state_generation(&state, &path) == Some(generation) {
        draft.work.not_before = Instant::now();
        state
            .pending
            .insert(draft.work.path.as_os_str().to_owned(), draft.work.clone());
        if retain {
            state
                .staged
                .insert(draft.work.path.as_os_str().to_owned(), draft);
        }
    }
    if state
        .inflight
        .get(path.as_os_str())
        .map(|(generation, _)| generation)
        == Some(&generation)
    {
        state.inflight.remove(path.as_os_str());
    }
    update_pending_status(&mut state);
}

fn commit_remove(
    shared: &RuntimeShared,
    store: &Arc<RwLock<EmbeddingStore>>,
    work: &PendingWork,
) -> bool {
    let _lifecycle_guard = shared
        .lifecycle_gate
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    if !shared.source_is_live() {
        return false;
    }
    let mut state = shared
        .state
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    if state_generation(&state, &work.path) != Some(work.generation) {
        state.inflight.remove(work.path.as_os_str());
        update_pending_status(&mut state);
        return false;
    }
    let removed = store
        .write()
        .unwrap_or_else(|error| error.into_inner())
        .remove(&work.path);
    match removed {
        Ok(removed) => {
            finish_success(&mut state, work);
            removed
        }
        Err(error) => {
            drop(state);
            record_failure(shared, work.clone(), error, None);
            false
        }
    }
}

fn commit_note(
    shared: &RuntimeShared,
    store: &Arc<RwLock<EmbeddingStore>>,
    work: &PendingWork,
    entry: NoteEmbeddingEntry,
) -> bool {
    let _lifecycle_guard = shared
        .lifecycle_gate
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    if !shared.source_is_live() {
        return false;
    }
    let mut state = shared
        .state
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    if state_generation(&state, &work.path) != Some(work.generation) {
        state.inflight.remove(work.path.as_os_str());
        update_pending_status(&mut state);
        return false;
    }
    let result = store
        .write()
        .unwrap_or_else(|error| error.into_inner())
        .insert_note(work.path.clone(), entry);
    match result {
        Ok(()) => {
            finish_success(&mut state, work);
            if state.reconciliation_complete && !state.store_queryable {
                state.store_queryable = true;
            }
            true
        }
        Err(error) => {
            drop(state);
            record_failure(shared, work.clone(), error, None);
            false
        }
    }
}

fn mark_success(shared: &RuntimeShared, work: &PendingWork) {
    let _lifecycle_guard = shared
        .lifecycle_gate
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    if !shared.source_is_live() {
        return;
    }
    let mut state = shared
        .state
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    if state_generation(&state, &work.path) == Some(work.generation) {
        finish_success(&mut state, work);
    } else {
        state.inflight.remove(work.path.as_os_str());
        update_pending_status(&mut state);
    }
}

fn finish_success(state: &mut RuntimeState, work: &PendingWork) {
    state.inflight.remove(work.path.as_os_str());
    state.initial_remaining.remove(work.path.as_os_str());
    state.failures.remove(work.path.as_os_str());
    update_pending_status(state);
}

/// A minimum retry delay; zero keeps ordinary backoff, and None means terminal.
fn retry_hint(error: &VaultError) -> Option<Duration> {
    match error {
        VaultError::EmbeddingRequest {
            retryable: true,
            retry_after,
            ..
        } => Some(retry_after.unwrap_or(Duration::ZERO)),
        VaultError::Io(error) if transient_source_io(error) => Some(Duration::ZERO),
        VaultError::Storage {
            kind:
                crate::error::StorageErrorKind::Unavailable
                | crate::error::StorageErrorKind::RemoteFailure,
            ..
        } => Some(Duration::ZERO),
        _ => None,
    }
}

fn transient_source_io(error: &std::io::Error) -> bool {
    #[cfg(windows)]
    if matches!(error.raw_os_error(), Some(32 | 33)) {
        return true;
    }
    #[cfg(unix)]
    if error.raw_os_error() == Some(5) {
        return true;
    }
    matches!(
        error.kind(),
        std::io::ErrorKind::Interrupted
            | std::io::ErrorKind::WouldBlock
            | std::io::ErrorKind::TimedOut
            | std::io::ErrorKind::ResourceBusy
            | std::io::ErrorKind::ConnectionAborted
            | std::io::ErrorKind::ConnectionReset
            | std::io::ErrorKind::NotConnected
            | std::io::ErrorKind::BrokenPipe
    )
}

fn record_failure(
    shared: &RuntimeShared,
    work: PendingWork,
    error: VaultError,
    staged: Option<StagedNote>,
) {
    let retry_after = retry_hint(&error);
    record_failure_message(shared, work, error.to_string(), retry_after, staged);
}

fn record_failure_message(
    shared: &RuntimeShared,
    mut work: PendingWork,
    mut error: String,
    retry_after: Option<Duration>,
    staged: Option<StagedNote>,
) {
    let mut state = shared
        .state
        .lock()
        .unwrap_or_else(|lock_error| lock_error.into_inner());
    let is_current = state_generation(&state, &work.path) == Some(work.generation);
    if state
        .inflight
        .get(work.path.as_os_str())
        .map(|(generation, _)| generation)
        == Some(&work.generation)
    {
        state.inflight.remove(work.path.as_os_str());
    }
    if is_current {
        let retry_at = retry_after.and_then(|minimum| {
            work.attempt = work.attempt.saturating_add(1);
            let delay = retry_delay(work.attempt).max(minimum);
            let deadline = Instant::now().checked_add(delay);
            if deadline.is_none() {
                error.push_str("; retry delay cannot be represented by the runtime clock");
            }
            deadline
        });
        state.initial_remaining.remove(work.path.as_os_str());
        state.failures.insert(
            work.path.as_os_str().to_owned(),
            format!(
                "embedding update for {} failed: {error}",
                work.path.display()
            ),
        );
        if let Some(deadline) = retry_at {
            work.not_before = deadline;
            state
                .pending
                .insert(work.path.as_os_str().to_owned(), work.clone());
            if let Some(mut draft) = staged {
                draft.work = work;
                draft.last_used = Instant::now();
                state
                    .staged
                    .insert(draft.work.path.as_os_str().to_owned(), draft);
            }
        } else {
            state.staged.remove(work.path.as_os_str());
        }
    }
    update_pending_status(&mut state);
    drop(state);
    shared.notify.notify_one();
}

fn finalize_reconciliation_if_complete(
    shared: &RuntimeShared,
    store: &Arc<RwLock<EmbeddingStore>>,
) -> bool {
    // A last watcher submission may retire the runtime while holding metadata.
    // Release the index before acquiring either runtime lock.
    let empty_vault = shared
        .index
        .read()
        .unwrap_or_else(|error| error.into_inner())
        .notes()
        .is_empty();
    let _lifecycle_guard = shared
        .lifecycle_gate
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    if !shared.source_is_live() {
        return false;
    }
    let mut state = shared
        .state
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    if !state.initial_started
        || state.reconciliation_complete
        || !state.initial_remaining.is_empty()
    {
        return false;
    }
    state.reconciliation_complete = true;
    state.cache_error = None;
    let mut store_guard = store.write().unwrap_or_else(|error| error.into_inner());
    let changed = !store_guard.first_pass_complete();
    store_guard.set_first_pass_complete(true);
    let has_vectors = !store_guard.is_empty();
    drop(store_guard);
    if !state.store_queryable && (empty_vault || has_vectors) {
        state.store_queryable = true;
    }
    changed
}

async fn persist_store(
    shared: &Arc<RuntimeShared>,
    store: &Arc<RwLock<EmbeddingStore>>,
) -> VaultResult<bool> {
    let store = Arc::clone(store);
    let cache = shared.cache.clone();
    let live = Arc::clone(&shared.live);
    let lifecycle_gate = Arc::clone(&shared.lifecycle_gate);
    let source_owner = Arc::clone(shared);
    tokio::task::spawn_blocking(move || {
        let bytes = {
            let store = store.read().unwrap_or_else(|error| error.into_inner());
            store.encode_cache()?
        };
        if !source_owner.source_is_live() {
            return Ok(false);
        }
        cache.publish(&bytes, Some((&live, &lifecycle_gate)))
    })
    .await
    .map_err(|error| VaultError::Embedding(format!("cache write task failed: {error}")))?
}

fn refresh_status(shared: &RuntimeShared, store: &Arc<RwLock<EmbeddingStore>>) {
    let paths = current_paths(&shared.index);
    let indexed_notes = {
        let store = store.read().unwrap_or_else(|error| error.into_inner());
        paths
            .iter()
            .filter(|path| store.get(Path::new(path)).is_some())
            .count()
    };
    let mut state = shared
        .state
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let next = status_for_loaded_state(&state, paths.len(), indexed_notes);
    replace_status(&mut state, next);
}

fn status_for_loaded_state(
    state: &RuntimeState,
    total_notes: usize,
    indexed_notes: usize,
) -> EmbeddingRuntimeStatus {
    let pending_notes = distinct_pending_count(state);
    let last_error = state
        .persistence_error
        .clone()
        .or_else(|| state.cache_error.clone())
        .or_else(|| state.failures.values().next().cloned());
    let phase = if last_error.is_some() {
        EmbeddingPhase::Degraded
    } else if !state.reconciliation_complete || pending_notes > 0 {
        EmbeddingPhase::Warming
    } else {
        EmbeddingPhase::Ready
    };
    EmbeddingRuntimeStatus {
        phase,
        queryable: state.store_queryable && state.model.is_some() && state.store.is_some(),
        indexed_notes,
        total_notes,
        pending_notes,
        last_error,
    }
}

fn status_for_current_paths(
    state: &RuntimeState,
    paths: &HashSet<OsString>,
) -> EmbeddingRuntimeStatus {
    if state.model.is_none() {
        let mut status = state.status.clone();
        status.indexed_notes = 0;
        status.total_notes = paths.len();
        status.pending_notes = distinct_pending_count(state);
        return status;
    }
    let indexed_notes = state.store.as_ref().map_or(0, |store| {
        let store = store.read().unwrap_or_else(|error| error.into_inner());
        paths
            .iter()
            .filter(|path| store.get(Path::new(path)).is_some())
            .count()
    });
    status_for_loaded_state(state, paths.len(), indexed_notes)
}

fn report_progress(shared: &RuntimeShared) {
    if !shared.active.load(Ordering::Acquire) || !shared.source_is_live() {
        return;
    }
    let paths = current_paths(&shared.index);
    let (status, inflight_notes, batch_elapsed_secs) = {
        let state = shared
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        (
            status_for_current_paths(&state, &paths),
            state.inflight.len(),
            state
                .batch_started_at
                .map(|started| started.elapsed().as_secs_f64()),
        )
    };
    if status.phase == EmbeddingPhase::Ready && status.pending_notes == 0 {
        return;
    }
    tracing::info!(
        phase = ?status.phase,
        queryable = status.queryable,
        indexed_notes = status.indexed_notes,
        total_notes = status.total_notes,
        pending_notes = status.pending_notes,
        inflight_notes,
        batch_elapsed_secs,
        "embedding progress"
    );
}

fn replace_status(state: &mut RuntimeState, next: EmbeddingRuntimeStatus) {
    let previous = &state.status;
    let meaningful_transition = previous.phase != next.phase
        || previous.queryable != next.queryable
        || previous.last_error != next.last_error
        || (previous.total_notes == 0 && next.total_notes > 0);

    if meaningful_transition {
        match next.phase {
            EmbeddingPhase::Degraded => tracing::warn!(
                phase = ?next.phase,
                queryable = next.queryable,
                indexed_notes = next.indexed_notes,
                total_notes = next.total_notes,
                pending_notes = next.pending_notes,
                error = next.last_error.as_deref(),
                "embedding runtime status changed"
            ),
            EmbeddingPhase::Warming | EmbeddingPhase::Ready => tracing::info!(
                phase = ?next.phase,
                queryable = next.queryable,
                indexed_notes = next.indexed_notes,
                total_notes = next.total_notes,
                pending_notes = next.pending_notes,
                "embedding runtime status changed"
            ),
        }
    }

    state.status = next;
}

fn normalize_intent_path(path: &Path, remote: bool) -> VaultResult<PathBuf> {
    if !remote {
        return super::path::normalize_relative(path);
    }
    let text = path
        .to_str()
        .ok_or_else(|| VaultError::InvalidPath("vault-relative paths must be UTF-8".into()))?;
    super::path::VaultPath::parse(text).map(|path| PathBuf::from(path.as_str()))
}

fn initial_embedding_work(index: &RwLock<VaultIndex>) -> Vec<(PathBuf, PendingKind)> {
    let index = index.read().unwrap_or_else(|error| error.into_inner());
    let mut paths: Vec<_> = index
        .notes()
        .keys()
        .filter_map(|path| {
            index
                .embedding_change(Path::new(path), false)
                .map(|kind| (PathBuf::from(path), kind))
        })
        .collect();
    paths.sort_unstable_by(|(left, _), (right, _)| left.as_os_str().cmp(right.as_os_str()));
    paths
}

fn current_paths(index: &Arc<RwLock<VaultIndex>>) -> HashSet<OsString> {
    index
        .read()
        .unwrap_or_else(|error| error.into_inner())
        .notes()
        .keys()
        .cloned()
        .collect()
}

fn update_pending_status(state: &mut RuntimeState) {
    state.status.pending_notes = distinct_pending_count(state);
    if state.status.pending_notes > 0
        && state.status.last_error.is_none()
        && state.status.phase == EmbeddingPhase::Ready
    {
        state.status.phase = EmbeddingPhase::Warming;
    }
}

fn distinct_pending_count(state: &RuntimeState) -> usize {
    let mut paths = state.pending.keys().collect::<HashSet<_>>();
    paths.extend(state.inflight.keys());
    paths.len()
}

fn next_generation(state: &mut RuntimeState) -> u64 {
    state.next_generation = state.next_generation.wrapping_add(1);
    if state.next_generation == 0 {
        state.next_generation = 1;
    }
    state.next_generation
}

fn state_generation(state: &RuntimeState, path: &Path) -> Option<u64> {
    state
        .pending
        .get(path.as_os_str())
        .map(|work| work.generation)
        .or_else(|| {
            state
                .inflight
                .get(path.as_os_str())
                .map(|(generation, _)| *generation)
        })
}

fn retry_delay(attempt: u8) -> Duration {
    let seconds = 1u64 << attempt.saturating_sub(1).min(5);
    Duration::from_secs(seconds).min(MAX_RETRY_DELAY)
}

fn not_ready_message(status: &EmbeddingRuntimeStatus) -> String {
    let detail = status
        .last_error
        .as_deref()
        .unwrap_or("embedding runtime is warming up");
    format!(
        "semantic embeddings are not ready ({}/{}, {} pending): {detail}",
        status.indexed_notes, status.total_notes, status.pending_notes
    )
}

#[cfg(test)]
mod tests {
    use std::ffi::OsStr;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

    use super::*;
    use crate::vault::embeddings::{
        EMBEDDING_INPUT_VERSION, EmbeddingBackendKind, EmbeddingSpaceIdentity,
    };
    use crate::vault::exclude::ExcludeSet;

    #[derive(Clone, Default)]
    struct CapturedLogs(Arc<Mutex<Vec<u8>>>);

    impl std::io::Write for CapturedLogs {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl CapturedLogs {
        fn text_for(&self, marker: &str) -> String {
            String::from_utf8(self.0.lock().unwrap().clone())
                .unwrap()
                .lines()
                .filter(|line| line.contains(marker))
                .collect::<Vec<_>>()
                .join("\n")
        }
    }

    fn log_subscriber(logs: CapturedLogs) -> impl tracing::Subscriber + Send + Sync {
        tracing_subscriber::fmt()
            .without_time()
            .with_ansi(false)
            .with_max_level(tracing::Level::DEBUG)
            .with_writer(move || logs.clone())
            .finish()
    }

    fn captured_logs() -> CapturedLogs {
        static LOGS: OnceLock<CapturedLogs> = OnceLock::new();
        LOGS.get_or_init(|| {
            let logs = CapturedLogs::default();
            // A permanent dispatcher keeps callsite interest stable while unrelated
            // runtime tests execute concurrently. Each assertion filters its own span.
            tracing::subscriber::set_global_default(log_subscriber(logs.clone())).unwrap();
            logs
        })
        .clone()
    }

    pub(super) fn initialize_test_logging() {
        captured_logs();
    }

    #[test]
    fn status_logs_changed_error_causes_without_repeating_unchanged_failures() {
        let logs = captured_logs();
        tracing::info_span!("changed_error_cause_test").in_scope(|| {
            let mut state = RuntimeState::default();
            let mut status = EmbeddingRuntimeStatus {
                phase: EmbeddingPhase::Degraded,
                last_error: Some("embedding API request timed out".into()),
                ..Default::default()
            };
            replace_status(&mut state, status.clone());
            status.pending_notes = 4;
            replace_status(&mut state, status.clone());
            status.last_error =
                Some("embedding API returned HTTP status 503 Service Unavailable".into());
            replace_status(&mut state, status);
        });
        let output = logs.text_for("changed_error_cause_test");
        assert_eq!(
            output.matches("embedding runtime status changed").count(),
            2,
            "{output}"
        );
        assert!(output.contains("embedding API request timed out"));
        assert!(output.contains("503 Service Unavailable"));
    }

    #[tokio::test]
    async fn progress_remains_visible_during_a_blocked_batch_and_reports_changed_counts() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(directory.path().join(".obsidian")).unwrap();
        for name in ["one", "two"] {
            std::fs::write(
                directory.path().join(format!("{name}.md")),
                format!("# {name}\nbody"),
            )
            .unwrap();
        }
        let index = test_index(directory.path()).await;
        let identity = FakeEmbedder::new().identity;
        let (started_tx, mut started_rx) = tokio::sync::mpsc::unbounded_channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let logs = captured_logs();
        // The handle's canonical path can expand a Windows short-name ancestor.
        let marker = directory.path().file_name().unwrap().to_string_lossy();
        let started = Instant::now();
        let runtime = EmbeddingRuntime::spawn_for_test(
            directory.path().to_path_buf(),
            index,
            directory.path().join("cache.bin"),
            1,
            true,
            async move {
                Ok(Arc::new(SequencedBlockingEmbedder {
                    identity,
                    calls: AtomicUsize::new(0),
                    started: started_tx,
                    releases: Mutex::new(release_rx),
                }) as Arc<dyn Embedder>)
            },
        );
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(5), started_rx.recv())
                .await
                .unwrap(),
            Some(0)
        );
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let output = logs.text_for(&marker);
                let inflight_reports = output
                    .lines()
                    .filter(|line| {
                        line.contains("embedding progress") && line.contains("inflight_notes=1")
                    })
                    .count();
                if inflight_reports >= 2 {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap_or_else(|error| {
            panic!(
                "progress timer must remain pollable during inference: {error}; {}",
                logs.text_for(&marker)
            )
        });
        assert!(!runtime.status().queryable);
        release_tx.send(()).unwrap();
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(5), started_rx.recv())
                .await
                .unwrap(),
            Some(1)
        );
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if logs.text_for(&marker).lines().any(|line| {
                    line.contains("embedding progress")
                        && line.contains("indexed_notes=1")
                        && line.contains("pending_notes=1")
                }) {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("next heartbeat must contain current counts");
        release_tx.send(()).unwrap();
        wait_for_status(&runtime, |status| status.phase == EmbeddingPhase::Ready).await;
        let output = logs.text_for(&marker);
        assert!(output.contains("batch_elapsed_secs="));
        assert_eq!(
            output.matches("embedding batch started").count(),
            2,
            "{output}"
        );
        assert_eq!(
            output.matches("embedding batch finished").count(),
            2,
            "{output}"
        );
        assert!(output.contains("outcome=\"completed\""));
        assert!(output.contains("duration_ms="));
        let maximum_reports = started.elapsed().as_millis() / PROGRESS_INTERVAL.as_millis() + 2;
        assert!(output.matches("embedding progress").count() as u128 <= maximum_reports);
        assert!(!output.contains("eta_secs="));
        assert!(!output.contains("rate_per_min="));
    }

    fn fake_input_policy() -> &'static super::super::embedding_input::EmbeddingInputPolicy {
        static POLICY: OnceLock<super::super::embedding_input::EmbeddingInputPolicy> =
            OnceLock::new();
        POLICY.get_or_init(|| super::super::embedding_input::EmbeddingInputPolicy {
            max_input_tokens: 512,
            tokenizer_fingerprint: [0; 32],
        })
    }

    struct FakeEmbedder {
        identity: EmbeddingSpaceIdentity,
        calls: AtomicUsize,
        tokenizations: AtomicUsize,
        fail: AtomicBool,
        inputs: Mutex<Vec<String>>,
        batches: Mutex<Vec<usize>>,
    }

    impl FakeEmbedder {
        fn new() -> Self {
            Self {
                identity: EmbeddingSpaceIdentity {
                    backend: EmbeddingBackendKind::Local,
                    model: "fake".to_string(),
                    endpoint_fingerprint: None,
                    dimension: 3,
                    input_version: EMBEDDING_INPUT_VERSION,
                    tokenizer_fingerprint: [0; 32],
                    max_input_tokens: 512,
                    chunking_version: super::super::embedding_input::CHUNKING_VERSION,
                },
                calls: AtomicUsize::new(0),
                tokenizations: AtomicUsize::new(0),
                fail: AtomicBool::new(false),
                inputs: Mutex::new(Vec::new()),
                batches: Mutex::new(Vec::new()),
            }
        }
    }

    impl Embedder for FakeEmbedder {
        fn input_policy(&self) -> &super::super::embedding_input::EmbeddingInputPolicy {
            fake_input_policy()
        }
        fn count_tokens(&self, text: &str) -> VaultResult<usize> {
            self.tokenizations.fetch_add(1, AtomicOrdering::SeqCst);
            Ok(text.chars().count())
        }
        fn request_limits(&self) -> super::super::embedding_input::EmbeddingRequestLimits {
            super::super::embedding_input::EmbeddingRequestLimits {
                max_inputs: 32,
                max_total_tokens: 16_384,
                max_request_bytes: 16 * 1024 * 1024,
            }
        }

        fn dimension(&self) -> usize {
            self.identity.dimension
        }

        fn space_identity(&self) -> &EmbeddingSpaceIdentity {
            &self.identity
        }

        fn embed_batch(&self, texts: &[&str]) -> VaultResult<Vec<Vec<f32>>> {
            self.calls.fetch_add(texts.len(), AtomicOrdering::SeqCst);
            self.batches.lock().unwrap().push(texts.len());
            self.inputs
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .extend(texts.iter().map(|text| (*text).to_string()));
            if self.fail.load(AtomicOrdering::SeqCst) {
                return Err(VaultError::EmbeddingRequest {
                    message: "injected inference failure".into(),
                    retryable: true,
                    retry_after: None,
                });
            }
            Ok(texts
                .iter()
                .map(|text| vec![text.len() as f32, 1.0, 0.0])
                .collect())
        }
    }

    struct BlockingEmbedder {
        identity: EmbeddingSpaceIdentity,
        started: std::sync::mpsc::Sender<()>,
        release: Mutex<std::sync::mpsc::Receiver<()>>,
        finished: Option<std::sync::mpsc::Sender<()>>,
    }

    struct SequencedBlockingEmbedder {
        identity: EmbeddingSpaceIdentity,
        calls: AtomicUsize,
        started: tokio::sync::mpsc::UnboundedSender<usize>,
        releases: Mutex<std::sync::mpsc::Receiver<()>>,
    }

    impl Embedder for SequencedBlockingEmbedder {
        fn input_policy(&self) -> &super::super::embedding_input::EmbeddingInputPolicy {
            fake_input_policy()
        }
        fn count_tokens(&self, text: &str) -> VaultResult<usize> {
            Ok(text.chars().count())
        }
        fn request_limits(&self) -> super::super::embedding_input::EmbeddingRequestLimits {
            super::super::embedding_input::EmbeddingRequestLimits {
                max_inputs: 32,
                max_total_tokens: 16_384,
                max_request_bytes: 16 * 1024 * 1024,
            }
        }

        fn dimension(&self) -> usize {
            self.identity.dimension
        }

        fn space_identity(&self) -> &EmbeddingSpaceIdentity {
            &self.identity
        }

        fn embed_batch(&self, texts: &[&str]) -> VaultResult<Vec<Vec<f32>>> {
            let call = self.calls.fetch_add(1, AtomicOrdering::SeqCst);
            let _ = self.started.send(call);
            self.releases
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .recv()
                .map_err(|error| VaultError::Embedding(error.to_string()))?;
            Ok(texts
                .iter()
                .map(|text| vec![text.len() as f32, 1.0, 0.0])
                .collect())
        }
    }

    struct PartialBatchEmbedder {
        identity: EmbeddingSpaceIdentity,
    }

    impl Embedder for PartialBatchEmbedder {
        fn input_policy(&self) -> &super::super::embedding_input::EmbeddingInputPolicy {
            fake_input_policy()
        }
        fn count_tokens(&self, text: &str) -> VaultResult<usize> {
            Ok(text.chars().count())
        }
        fn request_limits(&self) -> super::super::embedding_input::EmbeddingRequestLimits {
            super::super::embedding_input::EmbeddingRequestLimits {
                max_inputs: 32,
                max_total_tokens: 16_384,
                max_request_bytes: 16 * 1024 * 1024,
            }
        }

        fn dimension(&self) -> usize {
            self.identity.dimension
        }

        fn space_identity(&self) -> &EmbeddingSpaceIdentity {
            &self.identity
        }

        fn embed_batch(&self, _texts: &[&str]) -> VaultResult<Vec<Vec<f32>>> {
            Ok(vec![vec![0.0, 0.0, 1.0]])
        }
    }

    impl Embedder for BlockingEmbedder {
        fn input_policy(&self) -> &super::super::embedding_input::EmbeddingInputPolicy {
            fake_input_policy()
        }
        fn count_tokens(&self, text: &str) -> VaultResult<usize> {
            Ok(text.chars().count())
        }
        fn request_limits(&self) -> super::super::embedding_input::EmbeddingRequestLimits {
            super::super::embedding_input::EmbeddingRequestLimits {
                max_inputs: 32,
                max_total_tokens: 16_384,
                max_request_bytes: 16 * 1024 * 1024,
            }
        }

        fn dimension(&self) -> usize {
            self.identity.dimension
        }

        fn space_identity(&self) -> &EmbeddingSpaceIdentity {
            &self.identity
        }

        fn embed_batch(&self, texts: &[&str]) -> VaultResult<Vec<Vec<f32>>> {
            let _ = self.started.send(());
            self.release
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .recv()
                .map_err(|error| VaultError::Embedding(error.to_string()))?;
            if let Some(finished) = &self.finished {
                let _ = finished.send(());
            }
            Ok(texts.iter().map(|_| vec![1.0, 0.0, 0.0]).collect())
        }
    }

    async fn test_index(root: &Path) -> Arc<RwLock<VaultIndex>> {
        Arc::new(RwLock::new(
            VaultIndex::build(root, Arc::new(ExcludeSet::build(vec![]).unwrap()))
                .await
                .unwrap(),
        ))
    }

    async fn wait_for_status<F>(runtime: &EmbeddingRuntime, predicate: F)
    where
        F: Fn(&EmbeddingRuntimeStatus) -> bool,
    {
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let status = runtime.status();
                if predicate(&status) {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("runtime status timed out: {:?}", runtime.status()));
    }

    fn prepared_note_text(root: &Path, _index: &Arc<RwLock<VaultIndex>>, path: &Path) -> String {
        let content = super::super::fs::read_file(root, path).unwrap();
        let prepared = prepare_note(path, &content, &FakeEmbedder::new()).unwrap();
        assert_eq!(prepared.chunks.len(), 1);
        prepared.chunks[0].text.clone()
    }

    #[tokio::test]
    async fn dormant_runtime_activates_once_across_clones() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("one.md"), "# One\nbody").unwrap();
        let index = test_index(directory.path()).await;
        let polls = Arc::new(AtomicUsize::new(0));
        let loader_polls = Arc::clone(&polls);
        let runtime = EmbeddingRuntime::spawn_for_test(
            directory.path().to_path_buf(),
            index,
            directory.path().join("cache.bin"),
            32,
            false,
            async move {
                loader_polls.fetch_add(1, AtomicOrdering::SeqCst);
                Ok(Arc::new(FakeEmbedder::new()) as Arc<dyn Embedder>)
            },
        );
        for _ in 0..10 {
            tokio::task::yield_now().await;
        }
        assert!(!runtime.is_active());
        assert_eq!(polls.load(AtomicOrdering::SeqCst), 0);
        assert!(!directory.path().join("cache.bin").exists());

        let activated = std::thread::scope(|scope| {
            let threads = (0..8)
                .map(|_| {
                    let runtime = runtime.clone();
                    scope.spawn(move || runtime.activate())
                })
                .collect::<Vec<_>>();
            threads
                .into_iter()
                .map(|thread| thread.join().unwrap())
                .filter(|activated| *activated)
                .count()
        });
        assert_eq!(activated, 1);
        wait_for_status(&runtime, |status| status.phase == EmbeddingPhase::Ready).await;
        assert!(runtime.is_active());
        assert_eq!(polls.load(AtomicOrdering::SeqCst), 1);
    }

    #[tokio::test]
    async fn dropping_dormant_runtime_drops_unpolled_loader() {
        let directory = tempfile::tempdir().unwrap();
        let index = test_index(directory.path()).await;
        let fake = Arc::new(FakeEmbedder::new());
        let loader_fake = Arc::clone(&fake);
        let runtime = EmbeddingRuntime::spawn_for_test(
            directory.path().to_path_buf(),
            index,
            directory.path().join("cache.bin"),
            32,
            false,
            async move { Ok(loader_fake as Arc<dyn Embedder>) },
        );
        let weak = runtime.downgrade();
        tokio::task::yield_now().await;
        drop(runtime);
        assert!(!weak.is_alive());
        tokio::time::timeout(Duration::from_secs(2), async {
            while Arc::strong_count(&fake) != 1 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_eq!(fake.calls.load(AtomicOrdering::SeqCst), 0);
        assert!(!directory.path().join("cache.bin").exists());
    }

    #[tokio::test]
    async fn initial_reconciliation_assigns_generations_in_path_order() {
        let directory = tempfile::tempdir().unwrap();
        let paths = (0..17)
            .map(|id| PathBuf::from(format!("note-{id:02}.md")))
            .collect::<Vec<_>>();
        for path in paths.iter().rev() {
            std::fs::write(directory.path().join(path), "# Note\nbody").unwrap();
        }
        let index = test_index(directory.path()).await;
        let expected = paths
            .iter()
            .map(|path| prepared_note_text(directory.path(), &index, path))
            .collect::<Vec<_>>();
        let fake = Arc::new(FakeEmbedder::new());
        let loader_fake = Arc::clone(&fake);
        let runtime = EmbeddingRuntime::spawn_for_test(
            directory.path().to_path_buf(),
            index,
            directory.path().join("cache.bin"),
            3,
            true,
            async move { Ok(loader_fake as Arc<dyn Embedder>) },
        );
        wait_for_status(&runtime, |status| status.phase == EmbeddingPhase::Ready).await;
        assert_eq!(*fake.inputs.lock().unwrap(), expected);
        assert_eq!(*fake.batches.lock().unwrap(), [3, 3, 3, 3, 3, 2]);
    }

    #[tokio::test]
    async fn configured_batch_size_bounds_inference_without_preallocating_the_limit() {
        let directory = tempfile::tempdir().unwrap();
        for id in 0..5 {
            std::fs::write(directory.path().join(format!("{id}.md")), "# Note\nbody").unwrap();
        }
        let index = test_index(directory.path()).await;
        for (batch_size, expected) in [(2, vec![2, 2, 1]), (usize::MAX, vec![5])] {
            let fake = Arc::new(FakeEmbedder::new());
            let loader_fake = Arc::clone(&fake);
            let runtime = EmbeddingRuntime::spawn_for_test(
                directory.path().to_path_buf(),
                Arc::clone(&index),
                directory.path().join(format!("cache-{batch_size}.bin")),
                batch_size,
                true,
                async move { Ok(loader_fake as Arc<dyn Embedder>) },
            );
            wait_for_status(&runtime, |status| status.phase == EmbeddingPhase::Ready).await;
            assert_eq!(*fake.batches.lock().unwrap(), expected);
            assert_eq!(runtime.status().indexed_notes, 5);
        }
    }

    #[tokio::test]
    async fn startup_discovers_only_runnable_lexical_revisions() {
        let directory = tempfile::tempdir().unwrap();
        let root_path = directory.path().canonicalize().unwrap();
        let root = super::super::fs::VaultRoot::open(&root_path).unwrap();
        let exclude = Arc::new(ExcludeSet::default());
        std::fs::write(root_path.join("ready.md"), "available marker").unwrap();
        let (index, lexical) = VaultIndex::build_ranked(root.clone(), Arc::clone(&exclude))
            .await
            .unwrap();
        let index = Arc::new(RwLock::new(index));
        let stage = |path: &Path| {
            std::fs::write(root_path.join(path), "withheld marker").unwrap();
            assert!(
                index
                    .write()
                    .unwrap()
                    .refresh_file(&root, path, &exclude, Some(&lexical), true)
                    .unwrap()
            );
        };
        let before = Path::new("before-spawn.md");
        let during = Path::new("during-load.md");
        stage(before);
        let (release, wait) = tokio::sync::oneshot::channel();
        let fake = Arc::new(FakeEmbedder::new());
        let provider = Arc::clone(&fake);
        let runtime = EmbeddingRuntime::spawn(
            root_path.clone(),
            Arc::clone(&index),
            root_path.join("cache.bin"),
            async move {
                wait.await.unwrap();
                Ok(provider as Arc<dyn Embedder>)
            },
        );
        assert_eq!(
            runtime.pending_kind(before),
            Some(PendingKind::AwaitingLexical)
        );
        // The coordinator must also honor stamps discovered after its first seed.
        stage(during);
        release.send(()).unwrap();
        wait_for_status(&runtime, |s| {
            s.indexed_notes == 1 && s.pending_notes == 2 && s.phase == EmbeddingPhase::Warming
        })
        .await;
        assert_eq!(
            runtime.pending_kind(during),
            Some(PendingKind::AwaitingLexical)
        );
        {
            let inputs = fake.inputs.lock().unwrap();
            assert_eq!(inputs.len(), 1);
            assert!(inputs[0].contains("available marker"));
        }

        let committed = lexical.publish().unwrap();
        index
            .write()
            .unwrap()
            .publish_embedding_upserts(committed, |path| runtime.submit_upsert(path));
        wait_for_status(&runtime, |s| {
            s.phase == EmbeddingPhase::Ready && s.indexed_notes == 3
        })
        .await;
        assert_eq!(fake.inputs.lock().unwrap().len(), 3);
    }

    #[tokio::test]
    async fn spawn_returns_while_loader_is_blocked() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(directory.path().join(".obsidian")).unwrap();
        std::fs::write(directory.path().join("one.md"), "# One\nbody").unwrap();
        let index = test_index(directory.path()).await;
        let (release_tx, release_rx) = tokio::sync::oneshot::channel();
        let runtime = EmbeddingRuntime::spawn(
            directory.path().to_path_buf(),
            Arc::clone(&index),
            directory.path().join("cache.bin"),
            async move {
                release_rx.await.unwrap();
                Ok(Arc::new(FakeEmbedder::new()) as Arc<dyn Embedder>)
            },
        );

        assert_eq!(runtime.status().phase, EmbeddingPhase::Warming);
        assert!(!runtime.status().queryable);
        assert_eq!(runtime.status().total_notes, 1);
        assert_eq!(runtime.status().pending_notes, 1);

        std::fs::write(directory.path().join("two.md"), "# Two\nbody").unwrap();
        index
            .write()
            .unwrap_or_else(|error| error.into_inner())
            .reindex_file(directory.path(), Path::new("two.md"))
            .unwrap();
        runtime.submit_upsert(Path::new("two.md"));
        assert_eq!(runtime.status().total_notes, 2);
        assert_eq!(runtime.status().pending_notes, 2);
        let error = runtime
            .query_snapshot()
            .err()
            .expect("blocked loader should not expose a query snapshot");
        assert!(error.to_string().contains("0/2, 2 pending"));

        release_tx.send(()).unwrap();
        wait_for_status(&runtime, |status| status.phase == EmbeddingPhase::Ready).await;
        assert!(runtime.status().queryable);
        assert_eq!(runtime.status().total_notes, 2);
        assert_eq!(runtime.status().pending_notes, 0);
    }

    #[tokio::test]
    async fn source_reader_and_checkpoint_root_are_independent() {
        let source = tempfile::tempdir().unwrap();
        let data = tempfile::tempdir().unwrap();
        std::fs::write(source.path().join("one.md"), "# One\nsource body").unwrap();
        let index = test_index(source.path()).await;
        let runtime = EmbeddingRuntime::spawn_with_source(
            Storage::Local(super::super::fs::VaultRoot::open(source.path()).unwrap()),
            index,
            CacheFile::new(
                super::super::fs::VaultRoot::open(data.path()).unwrap(),
                PathBuf::from("embeddings.bin"),
            ),
            TEST_BATCH_SIZE,
            true,
            async { Ok(Arc::new(FakeEmbedder::new()) as Arc<dyn Embedder>) },
        );
        wait_for_status(&runtime, |status| {
            status.phase == EmbeddingPhase::Ready && status.indexed_notes == 1
        })
        .await;
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        while !data.path().join("embeddings.bin").is_file() {
            assert!(tokio::time::Instant::now() < deadline);
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(!source.path().join("embeddings.bin").exists());
        assert_eq!(
            std::fs::read_to_string(source.path().join("one.md")).unwrap(),
            "# One\nsource body"
        );
    }

    #[tokio::test]
    async fn weak_submit_handle_does_not_extend_runtime_lifetime() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(directory.path().join(".obsidian")).unwrap();
        let index = test_index(directory.path()).await;
        let runtime = EmbeddingRuntime::spawn(
            directory.path().to_path_buf(),
            index,
            directory.path().join("cache.bin"),
            async { std::future::pending::<VaultResult<Arc<dyn Embedder>>>().await },
        );
        let submitter = runtime.downgrade();

        assert!(submitter.is_alive());
        drop(runtime);
        assert!(!submitter.is_alive());
        submitter.submit_upsert(Path::new("ignored.md"));
    }

    #[tokio::test]
    async fn reconciliation_indexes_notes_and_persists_complete_cache() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(directory.path().join(".obsidian")).unwrap();
        std::fs::write(directory.path().join("one.md"), "# One\nsemantic body").unwrap();
        let index = test_index(directory.path()).await;
        let cache_path = directory.path().join("cache.bin");
        let runtime = EmbeddingRuntime::spawn(
            directory.path().to_path_buf(),
            index,
            cache_path.clone(),
            async { Ok(Arc::new(FakeEmbedder::new()) as Arc<dyn Embedder>) },
        );

        wait_for_status(&runtime, |status| status.phase == EmbeddingPhase::Ready).await;
        let status = runtime.status();
        assert_eq!(status.indexed_notes, 1);
        assert_eq!(status.total_notes, 1);
        assert!(
            runtime
                .query_snapshot()
                .unwrap()
                .semantic_scores("semantic", 1)
                .unwrap()
                .iter()
                .any(|(path, _)| path == Path::new("one.md"))
        );

        tokio::time::timeout(Duration::from_secs(5), async {
            while !cache_path.is_file() {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        let cache = EmbeddingStore::load(&cache_path).unwrap();
        assert!(cache.first_pass_complete());
        assert_eq!(cache.len(), 1);
    }

    #[tokio::test]
    async fn newer_remove_wins_over_queued_upsert() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(directory.path().join(".obsidian")).unwrap();
        std::fs::write(directory.path().join("one.md"), "# One\nbody").unwrap();
        let index = test_index(directory.path()).await;
        let runtime = EmbeddingRuntime::spawn(
            directory.path().to_path_buf(),
            Arc::clone(&index),
            directory.path().join("cache.bin"),
            async { Ok(Arc::new(FakeEmbedder::new()) as Arc<dyn Embedder>) },
        );
        wait_for_status(&runtime, |status| status.phase == EmbeddingPhase::Ready).await;

        index
            .write()
            .unwrap_or_else(|error| error.into_inner())
            .remove_file(Path::new("one.md"));
        runtime.submit_upsert(Path::new("one.md"));
        runtime.submit_remove(Path::new("one.md"));
        wait_for_status(&runtime, |status| {
            status.pending_notes == 0 && status.total_notes == 0
        })
        .await;
        let snapshot = runtime.query_snapshot().unwrap();
        assert!(snapshot.semantic_scores("body", 10).unwrap().is_empty());
    }

    #[tokio::test]
    async fn compatible_unchanged_cache_skips_inference() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(directory.path().join(".obsidian")).unwrap();
        std::fs::write(directory.path().join("one.md"), "# One\nsemantic body").unwrap();
        let index = test_index(directory.path()).await;
        let cache_path = directory.path().join("cache.bin");
        let fake = Arc::new(FakeEmbedder::new());
        let text = prepared_note_text(directory.path(), &index, Path::new("one.md"));
        let mut cache = EmbeddingStore::new_with_identity(fake.identity.clone());
        cache
            .insert_hashed(
                PathBuf::from("one.md"),
                prepared_text_hash(&text),
                vec![1.0, 0.0, 0.0],
            )
            .unwrap();
        cache.set_first_pass_complete(true);
        cache.save(&cache_path).unwrap();

        let loader_fake = Arc::clone(&fake);
        let runtime = EmbeddingRuntime::spawn(
            directory.path().to_path_buf(),
            Arc::clone(&index),
            cache_path,
            async move { Ok(loader_fake as Arc<dyn Embedder>) },
        );
        wait_for_status(&runtime, |status| status.phase == EmbeddingPhase::Ready).await;

        assert_eq!(fake.calls.load(AtomicOrdering::SeqCst), 0);
        assert!(runtime.status().queryable);

        index
            .write()
            .unwrap_or_else(|error| error.into_inner())
            .remove_file(Path::new("one.md"));
        let status = runtime.status();
        assert_eq!(status.total_notes, 0);
        assert_eq!(status.indexed_notes, 0);
    }

    #[tokio::test]
    async fn scoped_checkpoints_rebuild_without_importing_unscoped_vectors() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(directory.path().join(".obsidian")).unwrap();
        std::fs::write(directory.path().join("one.md"), "# One\nsemantic body").unwrap();
        let index = test_index(directory.path()).await;
        let source_path = directory.path().join("legacy").join("embeddings.bin");
        let target_path = directory.path().join("active").join("embeddings.bin");
        let fake = Arc::new(FakeEmbedder::new());
        let text = prepared_note_text(directory.path(), &index, Path::new("one.md"));
        let mut cache = EmbeddingStore::new_with_identity(fake.identity.clone());
        cache
            .insert_hashed(
                PathBuf::from("one.md"),
                prepared_text_hash(&text),
                vec![1.0, 0.0, 0.0],
            )
            .unwrap();
        cache.set_first_pass_complete(true);
        cache.save(&source_path).unwrap();

        let loader_fake = Arc::clone(&fake);
        let runtime = EmbeddingRuntime::spawn_for_test(
            directory.path().to_path_buf(),
            index,
            target_path.clone(),
            32,
            true,
            async move { Ok(loader_fake as Arc<dyn Embedder>) },
        );
        wait_for_status(&runtime, |status| {
            status.phase == EmbeddingPhase::Ready && target_path.is_file()
        })
        .await;

        assert!(fake.calls.load(AtomicOrdering::SeqCst) > 0);
        assert!(runtime.status().queryable);
        assert!(
            source_path.is_file(),
            "unscoped checkpoint must remain available"
        );
        let loaded = EmbeddingStore::load(&target_path).unwrap();
        assert!(loaded.first_pass_complete());
        assert_eq!(loaded.len(), 1);
    }

    #[tokio::test]
    async fn reconciliation_embeds_only_changed_and_new_notes() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(directory.path().join(".obsidian")).unwrap();
        std::fs::write(directory.path().join("one.md"), "# One\nunchanged body").unwrap();
        std::fs::write(directory.path().join("two.md"), "# Two\nold body").unwrap();
        let old_index = test_index(directory.path()).await;
        let cache_path = directory.path().join("cache.bin");
        let fake = Arc::new(FakeEmbedder::new());
        let mut cache = EmbeddingStore::new_with_identity(fake.identity.clone());
        for path in [Path::new("one.md"), Path::new("two.md")] {
            let text = prepared_note_text(directory.path(), &old_index, path);
            cache
                .insert_hashed(
                    path.to_path_buf(),
                    prepared_text_hash(&text),
                    vec![1.0, 0.0, 0.0],
                )
                .unwrap();
        }
        cache.set_first_pass_complete(true);
        cache.save(&cache_path).unwrap();

        std::fs::write(directory.path().join("two.md"), "# Two\nchanged body").unwrap();
        std::fs::write(directory.path().join("three.md"), "# Three\nnew body").unwrap();
        let index = test_index(directory.path()).await;
        let expected = [Path::new("two.md"), Path::new("three.md")]
            .map(|path| prepared_note_text(directory.path(), &index, path));
        let loader_fake = Arc::clone(&fake);
        let runtime = EmbeddingRuntime::spawn(
            directory.path().to_path_buf(),
            index,
            cache_path,
            async move { Ok(loader_fake as Arc<dyn Embedder>) },
        );

        wait_for_status(&runtime, |status| status.phase == EmbeddingPhase::Ready).await;
        let mut actual = fake
            .inputs
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone();
        actual.sort_unstable();
        let mut expected = expected.to_vec();
        expected.sort_unstable();
        assert_eq!(actual, expected);
    }

    #[tokio::test]
    async fn invalid_batch_preserves_every_last_known_good_entry() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(directory.path().join(".obsidian")).unwrap();
        std::fs::write(directory.path().join("one.md"), "# One\nold one").unwrap();
        std::fs::write(directory.path().join("two.md"), "# Two\nold two").unwrap();
        let old_index = test_index(directory.path()).await;
        let cache_path = directory.path().join("cache.bin");
        let identity = FakeEmbedder::new().identity;
        let mut cache = EmbeddingStore::new_with_identity(identity.clone());
        for (path, vector) in [
            (Path::new("one.md"), vec![1.0, 0.0, 0.0]),
            (Path::new("two.md"), vec![0.0, 1.0, 0.0]),
        ] {
            let text = prepared_note_text(directory.path(), &old_index, path);
            cache
                .insert_hashed(path.to_path_buf(), prepared_text_hash(&text), vector)
                .unwrap();
        }
        cache.set_first_pass_complete(true);
        cache.save(&cache_path).unwrap();

        std::fs::write(directory.path().join("one.md"), "# One\nnew one").unwrap();
        std::fs::write(directory.path().join("two.md"), "# Two\nnew two").unwrap();
        let index = test_index(directory.path()).await;
        let runtime = EmbeddingRuntime::spawn(
            directory.path().to_path_buf(),
            index,
            cache_path,
            async move { Ok(Arc::new(PartialBatchEmbedder { identity }) as Arc<dyn Embedder>) },
        );

        wait_for_status(&runtime, |status| status.phase == EmbeddingPhase::Degraded).await;
        let status = runtime.status();
        assert!(status.queryable);
        assert_eq!(status.indexed_notes, 2);
        assert_eq!(status.pending_notes, 0);
        assert!(
            status
                .last_error
                .as_deref()
                .is_some_and(|error| error.contains("1 vectors for 2 inputs"))
        );
        let snapshot = runtime.query_snapshot().unwrap();
        assert_eq!(
            snapshot.score_for(Path::new("one.md"), &[1.0, 0.0, 0.0]),
            1.0
        );
        assert_eq!(
            snapshot.score_for(Path::new("two.md"), &[0.0, 1.0, 0.0]),
            1.0
        );
        let marker = directory.path().file_name().unwrap().to_string_lossy();
        let output = captured_logs().text_for(&marker);
        assert!(output.contains("outcome=\"failed\""));
        assert!(output.contains("1 vectors for 2 inputs"));
    }

    #[tokio::test]
    async fn busy_initial_reconciliation_persists_an_incomplete_checkpoint() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(directory.path().join(".obsidian")).unwrap();
        for index in 0..(TEST_BATCH_SIZE + 1) {
            std::fs::write(
                directory.path().join(format!("note-{index:02}.md")),
                format!("# Note {index}\nbody {index}"),
            )
            .unwrap();
        }
        let index = test_index(directory.path()).await;
        let cache_path = directory.path().join("cache.bin");
        let identity = FakeEmbedder::new().identity;
        let (started_tx, mut started_rx) = tokio::sync::mpsc::unbounded_channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let runtime = EmbeddingRuntime::spawn(
            directory.path().to_path_buf(),
            index,
            cache_path.clone(),
            async move {
                Ok(Arc::new(SequencedBlockingEmbedder {
                    identity,
                    calls: AtomicUsize::new(0),
                    started: started_tx,
                    releases: Mutex::new(release_rx),
                }) as Arc<dyn Embedder>)
            },
        );

        assert_eq!(
            tokio::time::timeout(Duration::from_secs(5), started_rx.recv())
                .await
                .unwrap(),
            Some(0)
        );
        tokio::time::sleep(MAX_DIRTY_INTERVAL + Duration::from_millis(20)).await;
        release_tx.send(()).unwrap();
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(5), started_rx.recv())
                .await
                .unwrap(),
            Some(1)
        );

        let checkpoint = EmbeddingStore::load(&cache_path).unwrap();
        assert_eq!(checkpoint.len(), TEST_BATCH_SIZE);
        assert!(!checkpoint.first_pass_complete());
        assert!(!runtime.status().queryable);

        release_tx.send(()).unwrap();
        wait_for_status(&runtime, |status| status.phase == EmbeddingPhase::Ready).await;
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let persisted = EmbeddingStore::load(&cache_path).ok();
                if persisted
                    .as_ref()
                    .is_some_and(|store| store.first_pass_complete() && store.len() == 33)
                {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn retiring_a_scope_cancels_document_retries_with_outstanding_runtime_handles() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("one.md"), "# One\nprivate").unwrap();
        let model = Arc::new(FakeEmbedder::new());
        model.fail.store(true, AtomicOrdering::SeqCst);
        let loader = Arc::clone(&model);
        let runtime = EmbeddingRuntime::spawn(
            directory.path().to_path_buf(),
            test_index(directory.path()).await,
            directory.path().join("cache.bin"),
            async move { Ok(loader as Arc<dyn Embedder>) },
        );
        let outstanding = runtime.clone();
        wait_for_status(&runtime, |status| status.phase == EmbeddingPhase::Degraded).await;
        let retry =
            runtime.control.shared.state.lock().unwrap().pending[OsStr::new("one.md")].not_before;
        runtime.retire();
        drop(runtime);
        model.fail.store(false, AtomicOrdering::SeqCst);
        let calls = model.calls.load(AtomicOrdering::SeqCst);
        assert!(!outstanding.activate());
        outstanding.submit_upsert(Path::new("one.md"));
        tokio::time::sleep_until(
            tokio::time::Instant::from_std(retry) + Duration::from_millis(100),
        )
        .await;
        assert_eq!(model.calls.load(AtomicOrdering::SeqCst), calls);
    }

    #[tokio::test]
    async fn retiring_a_ready_runtime_withdraws_retained_queries_and_status() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("one.md"), "# One\nprivate").unwrap();
        let model = Arc::new(FakeEmbedder::new());
        let runtime = EmbeddingRuntime::spawn(
            directory.path().to_path_buf(),
            test_index(directory.path()).await,
            directory.path().join("cache.bin"),
            async move { Ok(model as Arc<dyn Embedder>) },
        );
        wait_for_status(&runtime, |status| status.queryable).await;
        assert!(runtime.query_snapshot().is_ok());
        runtime.retire();
        assert!(!runtime.status().queryable);
        assert!(runtime.query_snapshot().is_err());
    }

    #[tokio::test]
    async fn failed_refresh_keeps_last_known_good_then_recovers() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(directory.path().join(".obsidian")).unwrap();
        std::fs::write(directory.path().join("one.md"), "# One\nold body").unwrap();
        let old_index = test_index(directory.path()).await;
        let cache_path = directory.path().join("cache.bin");
        let fake = Arc::new(FakeEmbedder::new());
        let old_text = prepared_note_text(directory.path(), &old_index, Path::new("one.md"));
        let mut cache = EmbeddingStore::new_with_identity(fake.identity.clone());
        cache
            .insert_hashed(
                PathBuf::from("one.md"),
                prepared_text_hash(&old_text),
                vec![1.0, 0.0, 0.0],
            )
            .unwrap();
        cache.set_first_pass_complete(true);
        cache.save(&cache_path).unwrap();

        std::fs::write(
            directory.path().join("one.md"),
            "# One\nnew body with more words",
        )
        .unwrap();
        let index = test_index(directory.path()).await;
        fake.fail.store(true, AtomicOrdering::SeqCst);
        let loader_fake = Arc::clone(&fake);
        let runtime = EmbeddingRuntime::spawn(
            directory.path().to_path_buf(),
            index,
            cache_path,
            async move { Ok(loader_fake as Arc<dyn Embedder>) },
        );
        wait_for_status(&runtime, |status| status.phase == EmbeddingPhase::Degraded).await;
        let snapshot = runtime.query_snapshot().unwrap();
        assert!((snapshot.score_for(Path::new("one.md"), &[1.0, 0.0, 0.0]) - 1.0).abs() < 1e-6);

        fake.fail.store(false, AtomicOrdering::SeqCst);
        wait_for_status(&runtime, |status| status.phase == EmbeddingPhase::Ready).await;
        assert!(fake.calls.load(AtomicOrdering::SeqCst) >= 2);
    }

    #[tokio::test]
    async fn persistence_failure_degrades_then_recovers_without_losing_store() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(directory.path().join(".obsidian")).unwrap();
        std::fs::write(directory.path().join("one.md"), "# One\nbody").unwrap();
        let index = test_index(directory.path()).await;
        let blocked_parent = directory.path().join("blocked");
        std::fs::write(&blocked_parent, "not a directory").unwrap();
        let cache_path = blocked_parent.join("cache.bin");
        let runtime = EmbeddingRuntime::spawn(
            directory.path().to_path_buf(),
            index,
            cache_path.clone(),
            async { Ok(Arc::new(FakeEmbedder::new()) as Arc<dyn Embedder>) },
        );

        wait_for_status(&runtime, |status| {
            status.phase == EmbeddingPhase::Degraded && status.queryable
        })
        .await;
        assert_eq!(runtime.status().indexed_notes, 1);
        std::fs::remove_file(&blocked_parent).unwrap();
        std::fs::create_dir(&blocked_parent).unwrap();
        wait_for_status(&runtime, |status| status.phase == EmbeddingPhase::Ready).await;
        assert!(cache_path.is_file());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn replacing_a_ready_cache_directory_with_a_symlink_stops_publication_and_queries() {
        let vault = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let cache_dir = vault.path().join(".cache");
        let cache = cache_dir.join("cache.bin");
        let note = vault.path().join("one.md");
        std::fs::write(&note, "# One\nfirst revision").unwrap();
        let runtime = EmbeddingRuntime::spawn(
            vault.path().to_owned(),
            test_index(vault.path()).await,
            cache.clone(),
            async { Ok(Arc::new(FakeEmbedder::new()) as Arc<dyn Embedder>) },
        );
        wait_for_status(&runtime, |status| status.queryable && cache.is_file()).await;
        assert!(runtime.query_snapshot().is_ok());
        let protected = outside.path().join("cache.bin");
        std::fs::write(&protected, b"outside checkpoint").unwrap();
        std::fs::rename(&cache_dir, vault.path().join(".saved")).unwrap();
        std::os::unix::fs::symlink(outside.path(), &cache_dir).unwrap();
        std::fs::write(&note, "# One\nsecond revision").unwrap();
        runtime.submit_upsert(Path::new("one.md"));
        wait_for_status(&runtime, |status| {
            status.phase == EmbeddingPhase::Degraded && !status.queryable
        })
        .await;
        assert!(runtime.query_snapshot().is_err());
        assert_eq!(std::fs::read(&protected).unwrap(), b"outside checkpoint");
    }

    #[tokio::test]
    async fn drop_during_inference_prevents_late_commit_and_persistence() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(directory.path().join(".obsidian")).unwrap();
        std::fs::write(directory.path().join("one.md"), "# One\nold body").unwrap();
        let old_index = test_index(directory.path()).await;
        let cache_path = directory.path().join("cache.bin");
        let identity = FakeEmbedder::new().identity;
        let old_text = prepared_note_text(directory.path(), &old_index, Path::new("one.md"));
        let mut cache = EmbeddingStore::new_with_identity(identity.clone());
        cache
            .insert_hashed(
                PathBuf::from("one.md"),
                prepared_text_hash(&old_text),
                vec![0.0, 1.0, 0.0],
            )
            .unwrap();
        cache.set_first_pass_complete(true);
        cache.save(&cache_path).unwrap();
        let original_cache = std::fs::read(&cache_path).unwrap();

        std::fs::write(directory.path().join("one.md"), "# One\nnew body").unwrap();
        let index = test_index(directory.path()).await;
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let (finished_tx, finished_rx) = std::sync::mpsc::channel();
        let model = Arc::new(BlockingEmbedder {
            identity,
            started: started_tx,
            release: Mutex::new(release_rx),
            finished: Some(finished_tx),
        });
        let loader_model = Arc::clone(&model);
        let runtime = EmbeddingRuntime::spawn(
            directory.path().to_path_buf(),
            index,
            cache_path.clone(),
            async move { Ok(loader_model as Arc<dyn Embedder>) },
        );

        tokio::task::spawn_blocking(move || started_rx.recv_timeout(Duration::from_secs(5)))
            .await
            .unwrap()
            .unwrap();
        let snapshot = runtime.query_snapshot().unwrap();
        assert_eq!(
            snapshot.score_for(Path::new("one.md"), &[0.0, 1.0, 0.0]),
            1.0
        );

        let shared = Arc::clone(&runtime.control.shared);
        let (store, work) = {
            let state = shared
                .state
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            let (generation, _) = *state
                .inflight
                .get(OsStr::new("one.md"))
                .expect("changed note should be in flight");
            (
                Arc::clone(state.store.as_ref().expect("cache should be loaded")),
                PendingWork {
                    path: PathBuf::from("one.md"),
                    generation,
                    kind: PendingKind::Upsert,
                    attempt: 0,
                    not_before: Instant::now(),
                },
            )
        };
        let lifecycle_gate = Arc::clone(&shared.lifecycle_gate);
        let lifecycle_guard = lifecycle_gate
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let (drop_done_tx, drop_done_rx) = std::sync::mpsc::channel();
        let dropper = std::thread::spawn(move || {
            drop(runtime);
            let _ = drop_done_tx.send(());
        });
        let deadline = Instant::now() + Duration::from_secs(5);
        while shared.live.load(AtomicOrdering::Acquire) && Instant::now() < deadline {
            std::thread::yield_now();
        }
        assert!(
            !shared.live.load(AtomicOrdering::Acquire),
            "runtime drop should mark the coordinator dead before waiting on lifecycle work"
        );

        let commit_shared = Arc::clone(&shared);
        let commit_store = Arc::clone(&store);
        let entry = store
            .read()
            .unwrap()
            .note(Path::new("one.md"))
            .unwrap()
            .as_ref()
            .clone();
        let committer =
            std::thread::spawn(move || commit_note(&commit_shared, &commit_store, &work, entry));
        drop(lifecycle_guard);
        drop_done_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("runtime drop should finish after lifecycle work drains");
        dropper.join().expect("runtime drop thread should join");
        assert!(
            !committer.join().expect("late commit thread should join"),
            "a commit whose result arrives during shutdown must be discarded"
        );

        release_tx.send(()).unwrap();
        tokio::task::spawn_blocking(move || finished_rx.recv_timeout(Duration::from_secs(5)))
            .await
            .unwrap()
            .expect("in-flight inference should finish after release");
        drop(store);
        drop(shared);
        tokio::time::timeout(Duration::from_secs(5), async {
            while Arc::strong_count(&model) > 2 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("aborted coordinator should release its model handles");

        assert_eq!(
            snapshot.score_for(Path::new("one.md"), &[0.0, 1.0, 0.0]),
            1.0,
            "a held last-known-good snapshot must not change after runtime drop"
        );
        assert_eq!(
            snapshot.score_for(Path::new("one.md"), &[1.0, 0.0, 0.0]),
            0.0
        );
        assert_eq!(std::fs::read(&cache_path).unwrap(), original_cache);
    }

    #[tokio::test]
    async fn incomplete_checkpoint_is_reused_but_not_published_early() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(directory.path().join(".obsidian")).unwrap();
        std::fs::write(directory.path().join("one.md"), "# One\ncached").unwrap();
        std::fs::write(directory.path().join("two.md"), "# Two\nnew").unwrap();
        let index = test_index(directory.path()).await;
        let cache_path = directory.path().join("cache.bin");
        let identity = FakeEmbedder::new().identity;
        let cached_text = prepared_note_text(directory.path(), &index, Path::new("one.md"));
        let mut cache = EmbeddingStore::new_with_identity(identity.clone());
        cache
            .insert_hashed(
                PathBuf::from("one.md"),
                prepared_text_hash(&cached_text),
                vec![1.0, 0.0, 0.0],
            )
            .unwrap();
        cache.save(&cache_path).unwrap();

        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let runtime = EmbeddingRuntime::spawn(
            directory.path().to_path_buf(),
            index,
            cache_path,
            async move {
                Ok(Arc::new(BlockingEmbedder {
                    identity,
                    started: started_tx,
                    release: Mutex::new(release_rx),
                    finished: None,
                }) as Arc<dyn Embedder>)
            },
        );
        tokio::task::spawn_blocking(move || started_rx.recv_timeout(Duration::from_secs(5)))
            .await
            .unwrap()
            .unwrap();

        assert!(!runtime.status().queryable);
        assert!(runtime.query_snapshot().is_err());
        release_tx.send(()).unwrap();
        wait_for_status(&runtime, |status| status.phase == EmbeddingPhase::Ready).await;
        assert_eq!(runtime.status().indexed_notes, 2);
    }

    struct RequestControlledEmbedder {
        identity: EmbeddingSpaceIdentity,
        calls: AtomicUsize,
        fail_on_call: AtomicUsize,
        panic_on_call: AtomicUsize,
        transient: AtomicBool,
        inputs: Mutex<Vec<String>>,
    }

    impl RequestControlledEmbedder {
        fn new() -> Self {
            Self {
                identity: FakeEmbedder::new().identity,
                calls: AtomicUsize::new(0),
                fail_on_call: AtomicUsize::new(usize::MAX),
                panic_on_call: AtomicUsize::new(usize::MAX),
                transient: AtomicBool::new(true),
                inputs: Mutex::new(Vec::new()),
            }
        }
    }

    impl Embedder for RequestControlledEmbedder {
        fn dimension(&self) -> usize {
            self.identity.dimension
        }
        fn space_identity(&self) -> &EmbeddingSpaceIdentity {
            &self.identity
        }
        fn input_policy(&self) -> &super::super::embedding_input::EmbeddingInputPolicy {
            fake_input_policy()
        }
        fn count_tokens(&self, text: &str) -> VaultResult<usize> {
            Ok(text.chars().count())
        }
        fn request_limits(&self) -> super::super::embedding_input::EmbeddingRequestLimits {
            super::super::embedding_input::EmbeddingRequestLimits {
                max_inputs: 1,
                max_total_tokens: 512,
                max_request_bytes: 16 * 1024 * 1024,
            }
        }
        fn embed_batch(&self, texts: &[&str]) -> VaultResult<Vec<Vec<f32>>> {
            assert_eq!(texts.len(), 1);
            let call = self.calls.fetch_add(1, AtomicOrdering::SeqCst);
            self.inputs.lock().unwrap().push(texts[0].to_string());
            assert_ne!(
                self.panic_on_call.load(AtomicOrdering::SeqCst),
                call,
                "injected inference task failure"
            );
            if self.fail_on_call.load(AtomicOrdering::SeqCst) == call {
                return Err(VaultError::EmbeddingRequest {
                    message: "provider rejected this request".into(),
                    retryable: self.transient.load(AtomicOrdering::SeqCst),
                    retry_after: None,
                });
            }
            Ok(vec![vec![texts[0].len() as f32, 1.0, 0.0]])
        }
    }

    fn long_section_note() -> String {
        format!(
            "# Policy\n\n## First\n{}\n\n## Second\n{}\n\n## Third\n{}\n",
            "alpha ".repeat(65),
            "bravo ".repeat(65),
            "delta ".repeat(65)
        )
    }

    #[tokio::test]
    async fn later_request_failure_keeps_complete_note_and_reuses_successful_progress() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("policy.md"), "# Policy\nold body").unwrap();
        let index = test_index(directory.path()).await;
        let model = Arc::new(RequestControlledEmbedder::new());
        let loaded = Arc::clone(&model);
        let runtime = EmbeddingRuntime::spawn(
            directory.path().to_path_buf(),
            Arc::clone(&index),
            directory.path().join("cache.bin"),
            async move { Ok(loaded as Arc<dyn Embedder>) },
        );
        wait_for_status(&runtime, |status| status.phase == EmbeddingPhase::Ready).await;
        let before = runtime
            .query_snapshot()
            .unwrap()
            .capture_entries(&HashSet::from([OsString::from("policy.md")]))
            [OsStr::new("policy.md")]
        .clone();
        let changed = long_section_note();
        let prepared = prepare_note(Path::new("policy.md"), &changed, model.as_ref()).unwrap();
        assert!(prepared.chunks.len() >= 3);
        let first_new_input = prepared.chunks[0].text.clone();
        let first_new_call = model.calls.load(AtomicOrdering::SeqCst);
        model
            .fail_on_call
            .store(first_new_call + 1, AtomicOrdering::SeqCst);
        std::fs::write(directory.path().join("policy.md"), &changed).unwrap();
        runtime.submit_upsert(Path::new("policy.md"));
        wait_for_status(&runtime, |status| status.phase == EmbeddingPhase::Degraded).await;
        let during = runtime
            .query_snapshot()
            .unwrap()
            .capture_entries(&HashSet::from([OsString::from("policy.md")]))
            [OsStr::new("policy.md")]
        .clone();
        assert_eq!(during.source_revision, before.source_revision);
        assert_eq!(during.chunks.len(), 1);
        assert_eq!(runtime.status().pending_notes, 1);
        wait_for_status(&runtime, |status| status.phase == EmbeddingPhase::Ready).await;
        let after = runtime
            .query_snapshot()
            .unwrap()
            .capture_entries(&HashSet::from([OsString::from("policy.md")]))
            [OsStr::new("policy.md")]
        .clone();
        assert_eq!(
            after.source_revision,
            super::super::source::source_revision(&changed)
        );
        assert_eq!(after.chunks.len(), prepared.chunks.len());
        assert_eq!(
            model
                .inputs
                .lock()
                .unwrap()
                .iter()
                .filter(|input| **input == first_new_input)
                .count(),
            1,
            "successful request must survive a later transient failure"
        );
    }

    #[tokio::test]
    async fn retained_note_work_needs_no_replacement_generation() {
        let directory = tempfile::tempdir().unwrap();
        let path = Path::new("one.md");
        std::fs::write(directory.path().join(path), "# One\nbody").unwrap();
        let index = test_index(directory.path()).await;
        let runtime = EmbeddingRuntime::spawn_for_test(
            directory.path().to_path_buf(),
            index,
            directory.path().join("cache.bin"),
            TEST_BATCH_SIZE,
            false,
            async { Ok(Arc::new(FakeEmbedder::new()) as Arc<dyn Embedder>) },
        );
        let shared = &runtime.control.shared;
        let generation = shared.state.lock().unwrap().next_generation;
        assert!(runtime.has_current_or_pending(path, "unpublished"));
        assert_eq!(shared.state.lock().unwrap().next_generation, generation);
        let (batch, _, _) = take_due_batch(shared);
        assert_eq!(batch.len(), 1);
        assert!(runtime.has_current_or_pending(path, "unpublished"));
        assert_eq!(shared.state.lock().unwrap().next_generation, generation);

        runtime.submit_remove(path);
        assert!(
            !runtime.has_current_or_pending(path, "unpublished"),
            "a superseded in-flight upsert must not hide the current removal"
        );
        let (batch, _, _) = take_due_batch(shared);
        assert_eq!(batch[0].kind, PendingKind::Remove);
        assert!(!runtime.has_current_or_pending(path, "unpublished"));

        runtime.submit(path, PendingKind::AwaitingLexical);
        let generation = shared.state.lock().unwrap().next_generation;
        assert!(runtime.has_current_or_pending(path, "unpublished"));
        assert_eq!(shared.state.lock().unwrap().next_generation, generation);
        runtime.retire();
        assert!(!runtime.has_current_or_pending(path, "unpublished"));
    }

    #[tokio::test]
    async fn terminal_request_failure_waits_for_a_new_generation() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("one.md"), "# One\nold body").unwrap();
        let index = test_index(directory.path()).await;
        let model = Arc::new(RequestControlledEmbedder::new());
        let loaded = Arc::clone(&model);
        let runtime = EmbeddingRuntime::spawn(
            directory.path().to_path_buf(),
            index,
            directory.path().join("cache.bin"),
            async move { Ok(loaded as Arc<dyn Embedder>) },
        );
        wait_for_status(&runtime, |status| status.phase == EmbeddingPhase::Ready).await;
        let original_revision = super::super::source::source_revision("# One\nold body");
        let changed_revision = super::super::source::source_revision("# One\nnew body");
        assert!(runtime.has_current_or_pending(Path::new("one.md"), &original_revision));
        assert!(!runtime.has_current_or_pending(Path::new("one.md"), &changed_revision));
        model.transient.store(false, AtomicOrdering::SeqCst);
        model.fail_on_call.store(
            model.calls.load(AtomicOrdering::SeqCst),
            AtomicOrdering::SeqCst,
        );
        std::fs::write(directory.path().join("one.md"), "# One\nnew body").unwrap();
        runtime.submit_upsert(Path::new("one.md"));
        wait_for_status(&runtime, |status| {
            status.phase == EmbeddingPhase::Degraded && status.pending_notes == 0
        })
        .await;
        assert!(
            !runtime.has_current_or_pending(Path::new("one.md"), &changed_revision),
            "a terminal failure without retained work needs a repair generation"
        );
        let failed_calls = model.calls.load(AtomicOrdering::SeqCst);
        tokio::time::sleep(Duration::from_millis(1200)).await;
        assert_eq!(model.calls.load(AtomicOrdering::SeqCst), failed_calls);
        if !runtime.has_current_or_pending(Path::new("one.md"), &changed_revision) {
            runtime.submit_upsert(Path::new("one.md"));
        }
        wait_for_status(&runtime, |status| status.phase == EmbeddingPhase::Ready).await;
        assert_eq!(model.calls.load(AtomicOrdering::SeqCst), failed_calls + 1);
        assert!(runtime.has_current_or_pending(Path::new("one.md"), &changed_revision));
        let generation = runtime.control.shared.state.lock().unwrap().next_generation;
        if !runtime.has_current_or_pending(Path::new("one.md"), &changed_revision) {
            runtime.submit_upsert(Path::new("one.md"));
        }
        assert_eq!(
            runtime.control.shared.state.lock().unwrap().next_generation,
            generation
        );
        runtime.submit_remove(Path::new("one.md"));
        assert!(
            !runtime.has_current_or_pending(Path::new("one.md"), &changed_revision),
            "evidence scheduled for removal still needs a note repair intent"
        );
        runtime.submit_upsert(Path::new("one.md"));
    }

    #[tokio::test]
    async fn transient_source_read_failure_recovers_the_same_generation() {
        let directory = tempfile::tempdir().unwrap();
        let path = Path::new("one.md");
        std::fs::write(directory.path().join(path), "# One\nold body").unwrap();
        let index = test_index(directory.path()).await;
        let model = Arc::new(FakeEmbedder::new());
        let loaded = Arc::clone(&model);
        let runtime = EmbeddingRuntime::spawn(
            directory.path().to_path_buf(),
            index,
            directory.path().join("cache.bin"),
            async move { Ok(loaded as Arc<dyn Embedder>) },
        );
        wait_for_status(&runtime, |status| status.phase == EmbeddingPhase::Ready).await;
        let before = runtime
            .query_snapshot()
            .unwrap()
            .capture_entries(&HashSet::from([path.as_os_str().to_owned()]))[path.as_os_str()]
        .clone();
        let changed = "# One\nrecovered without another event";
        std::fs::write(directory.path().join(path), changed).unwrap();
        let work = {
            let mut state = runtime.control.shared.state.lock().unwrap();
            let generation = next_generation(&mut state);
            state.inflight.insert(
                path.as_os_str().to_owned(),
                (generation, PendingKind::Upsert),
            );
            PendingWork {
                path: path.to_path_buf(),
                generation,
                kind: PendingKind::Upsert,
                attempt: 0,
                not_before: Instant::now(),
            }
        };
        record_failure(
            &runtime.control.shared,
            work.clone(),
            VaultError::Io(std::io::Error::from(std::io::ErrorKind::TimedOut)),
            None,
        );
        {
            let state = runtime.control.shared.state.lock().unwrap();
            assert_eq!(state.pending[path.as_os_str()].generation, work.generation);
            assert_eq!(state.next_generation, work.generation);
        }
        assert_eq!(runtime.status().phase, EmbeddingPhase::Degraded);
        let during = runtime
            .query_snapshot()
            .unwrap()
            .capture_entries(&HashSet::from([path.as_os_str().to_owned()]))[path.as_os_str()]
        .clone();
        assert_eq!(during.source_revision, before.source_revision);

        wait_for_status(&runtime, |status| status.phase == EmbeddingPhase::Ready).await;
        let after = runtime
            .query_snapshot()
            .unwrap()
            .capture_entries(&HashSet::from([path.as_os_str().to_owned()]))[path.as_os_str()]
        .clone();
        assert_eq!(
            after.source_revision,
            super::super::source::source_revision(changed)
        );
        assert_eq!(
            runtime.control.shared.state.lock().unwrap().next_generation,
            work.generation
        );
        assert_eq!(model.calls.load(AtomicOrdering::SeqCst), 2);
    }

    #[test]
    fn remote_intents_preserve_server_names() {
        for path in ["a:b.md", "CON.md", "folder/a:b.md"] {
            assert_eq!(
                normalize_intent_path(Path::new(path), true).unwrap(),
                PathBuf::from(path)
            );
        }
        assert!(normalize_intent_path(Path::new("../outside.md"), true).is_err());
        assert!(normalize_intent_path(Path::new("/absolute.md"), true).is_err());
        #[cfg(windows)]
        for path in ["a:b.md", "CON.md"] {
            assert!(normalize_intent_path(Path::new(path), false).is_err());
        }
    }

    #[test]
    fn intent_generations_distinguish_exact_server_spellings() {
        let mut state = RuntimeState::default();
        for (generation, name) in [(1, "a:b.md"), (2, "A:b.md")] {
            let path = normalize_intent_path(Path::new(name), true).unwrap();
            state.pending.insert(
                path.as_os_str().to_owned(),
                PendingWork {
                    path,
                    generation,
                    kind: PendingKind::Upsert,
                    attempt: 0,
                    not_before: Instant::now(),
                },
            );
        }
        assert_eq!(distinct_pending_count(&state), 2);
        assert_eq!(state_generation(&state, Path::new("a:b.md")), Some(1));
        assert_eq!(state_generation(&state, Path::new("A:b.md")), Some(2));
    }

    #[test]
    fn source_observation_pass_reuses_only_its_captured_generations() {
        let work = |path: &str, generation| PendingWork {
            path: PathBuf::from(path),
            generation,
            kind: PendingKind::Upsert,
            attempt: 0,
            not_before: Instant::now(),
        };
        let first = work("a:b.md", 1);
        let second = work("A:b.md", 2);
        let mut pass = ObservationPass {
            generations: HashMap::from([
                (first.path.as_os_str().to_owned(), first.generation),
                (second.path.as_os_str().to_owned(), second.generation),
            ]),
            observed: Err(VaultError::Other("observation fixture".into())),
        };
        assert!(pass.contains(std::slice::from_ref(&first)));
        pass.consume(std::slice::from_ref(&first));
        assert!(pass.contains(std::slice::from_ref(&second)));
        assert!(!pass.contains(std::slice::from_ref(&work("A:b.md", 3))));
        assert!(!pass.contains(std::slice::from_ref(&work("new.md", 4))));
        pass.consume(std::slice::from_ref(&second));
        assert!(pass.generations.is_empty());
    }

    #[test]
    fn source_retry_classification_preserves_terminal_input_failures() {
        use crate::error::StorageErrorKind;

        for (kind, expected) in [
            (StorageErrorKind::Unavailable, Some(Duration::ZERO)),
            (StorageErrorKind::RemoteFailure, Some(Duration::ZERO)),
            (StorageErrorKind::PermissionDenied, None),
            (StorageErrorKind::Unsupported, None),
            (StorageErrorKind::Protocol, None),
            (StorageErrorKind::Conflict, None),
            (StorageErrorKind::IdentityChanged, None),
            (StorageErrorKind::Warming, None),
            (StorageErrorKind::ResourceLimit, None),
        ] {
            assert_eq!(
                retry_hint(&VaultError::storage(
                    "read",
                    "note.md",
                    kind,
                    "source failure"
                )),
                expected,
                "{kind:?}"
            );
        }
        for kind in [
            std::io::ErrorKind::WouldBlock,
            std::io::ErrorKind::TimedOut,
            std::io::ErrorKind::Interrupted,
            std::io::ErrorKind::ResourceBusy,
            std::io::ErrorKind::ConnectionReset,
        ] {
            assert_eq!(
                retry_hint(&VaultError::Io(std::io::Error::from(kind))),
                Some(Duration::ZERO),
                "{kind:?}"
            );
        }
        #[cfg(unix)]
        assert_eq!(
            retry_hint(&VaultError::Io(std::io::Error::from_raw_os_error(5))),
            Some(Duration::ZERO)
        );
        #[cfg(windows)]
        for code in [32, 33] {
            assert_eq!(
                retry_hint(&VaultError::Io(std::io::Error::from_raw_os_error(code))),
                Some(Duration::ZERO)
            );
        }
        for kind in [
            std::io::ErrorKind::InvalidData,
            std::io::ErrorKind::PermissionDenied,
            std::io::ErrorKind::NotFound,
        ] {
            assert_eq!(
                retry_hint(&VaultError::Io(std::io::Error::from(kind))),
                None
            );
        }
        assert_eq!(
            retry_hint(&VaultError::EmbeddingConfiguration(
                "invalid profile".into()
            )),
            None
        );
        assert_eq!(
            retry_hint(&VaultError::Embedding("invalid response".into())),
            None
        );
    }

    #[tokio::test]
    async fn inference_task_failure_recovers_without_a_new_generation() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("one.md"), "# One\nbody").unwrap();
        let index = test_index(directory.path()).await;
        let model = Arc::new(RequestControlledEmbedder::new());
        model.panic_on_call.store(0, AtomicOrdering::SeqCst);
        let loaded = Arc::clone(&model);
        let runtime = EmbeddingRuntime::spawn(
            directory.path().to_path_buf(),
            index,
            directory.path().join("cache.bin"),
            async move { Ok(loaded as Arc<dyn Embedder>) },
        );
        wait_for_status(&runtime, |status| {
            status.phase == EmbeddingPhase::Degraded && status.pending_notes == 1
        })
        .await;
        let generation = runtime.control.shared.state.lock().unwrap().next_generation;
        wait_for_status(&runtime, |status| status.phase == EmbeddingPhase::Ready).await;
        assert_eq!(
            runtime.control.shared.state.lock().unwrap().next_generation,
            generation
        );
        assert_eq!(model.calls.load(AtomicOrdering::SeqCst), 2);
    }

    #[tokio::test]
    async fn retry_schedule_honors_provider_minimum_and_rejects_unrepresentable_deadlines() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("one.md"), "# One\nbody").unwrap();
        let index = test_index(directory.path()).await;
        for (attempt, hint, expected) in [
            (0, Duration::from_secs(45), Some(Duration::from_secs(45))),
            (5, Duration::from_secs(3), Some(Duration::from_secs(30))),
            (0, Duration::MAX, None),
        ] {
            let runtime = EmbeddingRuntime::spawn_for_test(
                directory.path().to_path_buf(),
                Arc::clone(&index),
                directory.path().join("cache.bin"),
                32,
                false,
                async { Ok(Arc::new(FakeEmbedder::new()) as Arc<dyn Embedder>) },
            );
            let (batch, _, _) = take_due_batch(&runtime.control.shared);
            let mut work = batch.into_iter().next().unwrap();
            work.attempt = attempt;
            let before = Instant::now();
            record_failure(
                &runtime.control.shared,
                work,
                VaultError::EmbeddingRequest {
                    message: "provider rate limit".into(),
                    retryable: true,
                    retry_after: Some(hint),
                },
                None,
            );
            let state = runtime.control.shared.state.lock().unwrap();
            if let Some(expected) = expected {
                assert!(state.pending[OsStr::new("one.md")].not_before >= before + expected);
            } else {
                assert!(state.pending.is_empty());
                assert!(state.failures[OsStr::new("one.md")].contains("runtime clock"));
            }
        }
    }

    #[tokio::test]
    async fn source_revision_metadata_and_ranges_come_from_one_observation() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("one.md"), "# Old\n#oldtag").unwrap();
        let index = test_index(directory.path()).await;
        let changed = "---\ntags: [newtag]\n---\n# New heading\nnew body\n";
        std::fs::write(directory.path().join("one.md"), changed).unwrap();
        let runtime = EmbeddingRuntime::spawn(
            directory.path().to_path_buf(),
            index,
            directory.path().join("cache.bin"),
            async { Ok(Arc::new(FakeEmbedder::new()) as Arc<dyn Embedder>) },
        );
        wait_for_status(&runtime, |status| status.phase == EmbeddingPhase::Ready).await;
        let entries = runtime
            .query_snapshot()
            .unwrap()
            .capture_entries(&HashSet::from([OsString::from("one.md")]));
        let note = &entries[OsStr::new("one.md")];
        assert_eq!(
            note.source_revision,
            super::super::source::source_revision(changed)
        );
        assert!(note.tags.contains(&"newtag".to_string()));
        assert!(!note.tags.contains(&"oldtag".to_string()));
        assert_eq!(
            &changed[note.body_range.start_byte..note.body_range.end_byte],
            "# New heading\nnew body\n"
        );
    }

    #[tokio::test]
    async fn paused_staging_is_evicted_for_ready_work_and_removals_have_priority() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("a.md"), "# A\nbody").unwrap();
        std::fs::write(directory.path().join("z.md"), "# Z\nbody").unwrap();
        let index = test_index(directory.path()).await;
        let runtime = EmbeddingRuntime::spawn_for_test(
            directory.path().to_path_buf(),
            index,
            directory.path().join("cache.bin"),
            1,
            false,
            async { Ok(Arc::new(FakeEmbedder::new()) as Arc<dyn Embedder>) },
        );
        let shared = &runtime.control.shared;
        let prepared = prepare_note(Path::new("a.md"), "# A\nbody", &FakeEmbedder::new()).unwrap();
        {
            let mut state = shared.state.lock().unwrap();
            let work = state.pending[OsStr::new("a.md")].clone();
            state.staged.insert(
                work.path.as_os_str().to_owned(),
                StagedNote {
                    work,
                    vectors: vec![None; prepared.chunks.len()],
                    hash_order: (0..prepared.chunks.len()).collect(),
                    prepared: Arc::new(prepared),
                    memory_bytes: MAX_STAGING_BYTES,
                    last_used: Instant::now(),
                },
            );
        }
        assert!(admit_staging(shared, 0, 1));
        assert!(shared.state.lock().unwrap().staged.is_empty());
        runtime.submit_remove(Path::new("z.md"));
        let (batch, _, _) = take_due_batch(shared);
        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0].path, Path::new("z.md"));
        assert_eq!(batch[0].kind, PendingKind::Remove);
    }

    #[tokio::test]
    async fn impossible_admission_preserves_retry_progress_and_evicts_only_the_known_shortfall() {
        let directory = tempfile::tempdir().unwrap();
        let raw = format!("# Section\n{}", "word ".repeat(120));
        for name in ["a.md", "b.md", "c.md"] {
            std::fs::write(directory.path().join(name), &raw).unwrap();
        }
        let index = test_index(directory.path()).await;
        let runtime = EmbeddingRuntime::spawn_for_test(
            directory.path().to_path_buf(),
            index,
            directory.path().join("cache.bin"),
            32,
            false,
            async { Ok(Arc::new(FakeEmbedder::new()) as Arc<dyn Embedder>) },
        );
        let shared = &runtime.control.shared;
        let now = Instant::now();
        {
            let mut state = shared.state.lock().unwrap();
            for (ordinal, name) in ["a.md", "b.md", "c.md"].into_iter().enumerate() {
                let work = state.pending[OsStr::new(name)].clone();
                let prepared = prepare_note(Path::new(name), &raw, &FakeEmbedder::new()).unwrap();
                assert!(prepared.chunks.len() > 1);
                assert!(prepared.staging_bytes(3) <= 4096);
                let mut vectors = vec![None; prepared.chunks.len()];
                vectors[0] = Some(vec![1.0, 0.0, 0.0]);
                state.staged.insert(
                    work.path.as_os_str().to_owned(),
                    StagedNote {
                        work,
                        vectors,
                        hash_order: (0..prepared.chunks.len()).collect(),
                        prepared: Arc::new(prepared),
                        memory_bytes: 4096,
                        last_used: now + Duration::from_millis(ordinal as u64),
                    },
                );
            }
        }
        let active = MAX_STAGING_BYTES - 16_384;
        assert!(!admit_staging(shared, active, 16_385));
        assert_eq!(shared.state.lock().unwrap().staged.len(), 3);
        assert!(admit_staging(shared, active, 11_200));
        let state = shared.state.lock().unwrap();
        assert_eq!(state.staged.len(), 1);
        assert!(state.staged[OsStr::new("c.md")].vectors[0].is_some());
    }

    #[tokio::test]
    async fn preparation_that_cannot_fit_active_work_preserves_paused_progress() {
        let directory = tempfile::tempdir().unwrap();
        let path = Path::new("one.md");
        std::fs::write(directory.path().join(path), "# One\nbody").unwrap();
        let paused_raw = format!("# Paused\n{}", "word ".repeat(120));
        std::fs::write(directory.path().join("paused.md"), &paused_raw).unwrap();
        let index = test_index(directory.path()).await;
        let model: Arc<dyn Embedder> = Arc::new(FakeEmbedder::new());
        let loaded = Arc::clone(&model);
        let runtime = EmbeddingRuntime::spawn_for_test(
            directory.path().to_path_buf(),
            index,
            directory.path().join("cache.bin"),
            1,
            false,
            async move { Ok(loaded) },
        );
        let shared = &runtime.control.shared;
        {
            let prepared =
                prepare_note(Path::new("paused.md"), &paused_raw, model.as_ref()).unwrap();
            assert!(prepared.chunks.len() > 1);
            assert!(prepared.staging_bytes(3) <= 4096);
            let mut vectors = vec![None; prepared.chunks.len()];
            vectors[0] = Some(vec![1.0, 0.0, 0.0]);
            let mut state = shared.state.lock().unwrap();
            let work = state.pending[OsStr::new("paused.md")].clone();
            state.staged.insert(
                work.path.as_os_str().to_owned(),
                StagedNote {
                    work,
                    vectors,
                    hash_order: (0..prepared.chunks.len()).collect(),
                    prepared: Arc::new(prepared),
                    memory_bytes: 4096,
                    last_used: Instant::now(),
                },
            );
        }
        let (batch, _, _) = take_due_batch(shared);
        assert_eq!(batch[0].path, path);
        std::fs::remove_file(directory.path().join(path)).unwrap();
        assert!(matches!(
            prepare_work(
                shared,
                &model,
                &batch[0],
                MAX_STAGING_BYTES - 8192,
                8192,
                None
            )
            .await
            .unwrap(),
            PreparationOutcome::Deferred
        ));
        assert!(shared.state.lock().unwrap().staged[OsStr::new("paused.md")].vectors[0].is_some());
    }

    #[tokio::test]
    async fn preparation_pressure_defers_before_reading_or_copying_vectors() {
        let directory = tempfile::tempdir().unwrap();
        let path = Path::new("one.md");
        std::fs::write(directory.path().join(path), "# One\nbody").unwrap();
        let index = test_index(directory.path()).await;
        let model: Arc<dyn Embedder> = Arc::new(FakeEmbedder::new());
        let loader_model = Arc::clone(&model);
        let runtime = EmbeddingRuntime::spawn_for_test(
            directory.path().to_path_buf(),
            index,
            directory.path().join("cache.bin"),
            32,
            false,
            async move { Ok(loader_model) },
        );
        let shared = &runtime.control.shared;
        let (batch, _, _) = take_due_batch(shared);
        let work = &batch[0];
        // If preparation reads the file before checking the occupied budget,
        // this deletion would instead produce a source error.
        std::fs::remove_file(directory.path().join(path)).unwrap();
        assert!(matches!(
            prepare_work(shared, &model, work, MAX_STAGING_BYTES, 0, None)
                .await
                .unwrap(),
            PreparationOutcome::Deferred
        ));
        assert!(shared.state.lock().unwrap().failures.is_empty());
        std::fs::write(directory.path().join(path), "# One\nbody").unwrap();
        let prepared = prepare_work(shared, &model, work, 0, 0, None)
            .await
            .unwrap();
        assert!(matches!(prepared, PreparationOutcome::Prepared(_)));
    }

    #[tokio::test]
    async fn temporary_read_budget_exhaustion_does_not_fail_a_note() {
        let directory = tempfile::tempdir().unwrap();
        let path = Path::new("one.md");
        std::fs::write(
            directory.path().join(path),
            format!("# One\n{}", "body ".repeat(2_000)),
        )
        .unwrap();
        let index = test_index(directory.path()).await;
        let model: Arc<dyn Embedder> = Arc::new(FakeEmbedder::new());
        let loader_model = Arc::clone(&model);
        let runtime = EmbeddingRuntime::spawn_for_test(
            directory.path().to_path_buf(),
            index,
            directory.path().join("cache.bin"),
            32,
            false,
            async move { Ok(loader_model) },
        );
        let shared = &runtime.control.shared;
        let (batch, _, _) = take_due_batch(shared);
        assert!(matches!(
            prepare_work(
                shared,
                &model,
                &batch[0],
                MAX_STAGING_BYTES - 4_096,
                0,
                None
            )
            .await
            .unwrap(),
            PreparationOutcome::Deferred
        ));
        assert!(shared.state.lock().unwrap().failures.is_empty());
        let prepared = prepare_work(shared, &model, &batch[0], 0, 0, None)
            .await
            .unwrap();
        let PreparationOutcome::Prepared(prepared) = prepared else {
            panic!("released capacity must admit preparation");
        };
        assert!(
            staging_size(&prepared, model.dimension(), &batch[0]).unwrap() <= MAX_STAGING_BYTES
        );
    }

    #[tokio::test]
    async fn matching_raw_revision_skips_tokenization_and_frontmatter_edits_refresh_metadata() {
        let directory = tempfile::tempdir().unwrap();
        let path = Path::new("one.md");
        let original = "# One\nunchanged body\n";
        std::fs::write(directory.path().join(path), original).unwrap();
        let index = test_index(directory.path()).await;
        let prepared = prepare_note(path, original, &FakeEmbedder::new()).unwrap();
        let entry = NoteEmbeddingEntry {
            source_revision: prepared.source_revision,
            source_bytes: prepared.source_bytes,
            body_range: prepared.body_range,
            title: prepared.title,
            tags: prepared.tags,
            chunks: prepared
                .chunks
                .into_iter()
                .map(|chunk| ChunkEmbedding {
                    input_hash: chunk.input_hash,
                    vector: vec![1.0, 0.0, 0.0],
                    scope: chunk.scope,
                    range: chunk.range,
                    overlap_range: chunk.overlap_range,
                    heading_path: chunk.heading_path,
                    section_range: chunk.section_range,
                    block_kind: chunk.block_kind,
                })
                .collect(),
        };
        let model = Arc::new(FakeEmbedder::new());
        let mut cache = EmbeddingStore::new_with_identity(model.identity.clone());
        cache.insert_note(path.to_path_buf(), entry).unwrap();
        cache.set_first_pass_complete(true);
        let cache_path = directory.path().join("cache.bin");
        cache.save(&cache_path).unwrap();
        let loaded = Arc::clone(&model);
        let runtime = EmbeddingRuntime::spawn(
            directory.path().to_path_buf(),
            index,
            cache_path,
            async move { Ok(loaded as Arc<dyn Embedder>) },
        );
        wait_for_status(&runtime, |status| status.phase == EmbeddingPhase::Ready).await;
        assert_eq!(model.calls.load(AtomicOrdering::SeqCst), 0);
        assert_eq!(
            model.tokenizations.load(AtomicOrdering::SeqCst),
            0,
            "an exact current raw revision must bypass preparation and tokenization"
        );

        let updated = format!("---\ntags: [fresh]\n---\n{original}");
        std::fs::write(directory.path().join(path), &updated).unwrap();
        runtime.submit_upsert(path);
        wait_for_status(&runtime, |status| status.phase == EmbeddingPhase::Ready).await;
        assert_eq!(
            model.calls.load(AtomicOrdering::SeqCst),
            0,
            "a frontmatter-only edit must reuse the unchanged prepared input vector"
        );
        assert!(model.tokenizations.load(AtomicOrdering::SeqCst) > 0);
        let entries = runtime
            .query_snapshot()
            .unwrap()
            .capture_entries(&HashSet::from([path.as_os_str().to_owned()]));
        let current = &entries[path.as_os_str()];
        assert_eq!(
            current.source_revision,
            super::super::source::source_revision(&updated)
        );
        assert_eq!(current.tags, ["fresh"]);
        assert_eq!(
            &updated[current.body_range.start_byte..current.body_range.end_byte],
            original
        );
        assert!(current.body_range.start_byte > 0);
    }
}
