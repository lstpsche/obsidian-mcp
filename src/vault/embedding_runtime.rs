//! Managed background lifecycle for optional semantic embeddings.

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock, Weak};
use std::time::{Duration, Instant};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio::sync::Notify;
use tokio::task::JoinHandle;

use crate::error::{VaultError, VaultResult};

use super::embeddings::{
    Embedder, EmbeddingStore, LegacyCacheMigration, migrate_cache_candidates_to_path,
    prepare_embed_text, prepared_text_hash, validate_embedding_batch,
};
use super::index::VaultIndex;

const RECONCILE_BATCH_SIZE: usize = 32;
#[cfg(not(test))]
const MAX_DIRTY_INTERVAL: Duration = Duration::from_secs(2);
#[cfg(test)]
const MAX_DIRTY_INTERVAL: Duration = Duration::from_millis(50);
const MAX_RETRY_DELAY: Duration = Duration::from_secs(30);

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

impl Drop for RuntimeControl {
    fn drop(&mut self) {
        self.shared.live.store(false, Ordering::Release);
        // Synchronize with the final store commit or atomic cache publication.
        // An operation that already passed its liveness check may finish before
        // Drop returns, but it cannot mutate observable state afterward.
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

struct RuntimeShared {
    vault_root: PathBuf,
    index: Arc<RwLock<VaultIndex>>,
    cache_path: PathBuf,
    cache_migration_sources: Vec<PathBuf>,
    state: Mutex<RuntimeState>,
    notify: Notify,
    live: Arc<AtomicBool>,
    lifecycle_gate: Arc<Mutex<()>>,
    first_load_error: OnceLock<String>,
}

#[derive(Default)]
struct RuntimeState {
    next_generation: u64,
    pending: HashMap<PathBuf, PendingWork>,
    inflight: HashMap<PathBuf, u64>,
    initial_remaining: HashSet<PathBuf>,
    initial_started: bool,
    reconciliation_complete: bool,
    failures: HashMap<PathBuf, String>,
    persistence_error: Option<String>,
    model: Option<Arc<dyn Embedder>>,
    store: Option<Arc<RwLock<EmbeddingStore>>>,
    store_queryable: bool,
    status: EmbeddingRuntimeStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PendingKind {
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

#[derive(Clone)]
pub(crate) struct EmbeddingQuerySnapshot {
    model: Arc<dyn Embedder>,
    store: Arc<RwLock<EmbeddingStore>>,
}

impl EmbeddingQuerySnapshot {
    pub(crate) fn embed_query(&self, query: &str) -> VaultResult<Vec<f32>> {
        let mut vectors = self.model.embed_batch(&[query])?;
        vectors
            .pop()
            .ok_or_else(|| VaultError::Embedding("embedding query returned no vector".into()))
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

    pub(crate) fn semantic_scores_for_paths(
        &self,
        query: &str,
        allowed_paths: &HashSet<PathBuf>,
        top_k: usize,
    ) -> VaultResult<Vec<(PathBuf, f32)>> {
        let query_vector = self.embed_query(query)?;
        let store = self.store.read().unwrap_or_else(|error| error.into_inner());
        Ok(store.query_paths(&query_vector, allowed_paths, top_k))
    }

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
        Self::spawn_with_cache_sources(vault_root, index, cache_path, Vec::new(), loader)
    }

    pub(crate) fn spawn_with_cache_sources<F>(
        vault_root: PathBuf,
        index: Arc<RwLock<VaultIndex>>,
        cache_path: PathBuf,
        cache_migration_sources: Vec<PathBuf>,
        loader: F,
    ) -> Self
    where
        F: Future<Output = VaultResult<Arc<dyn Embedder>>> + Send + 'static,
    {
        let initial_paths = current_paths(&index);
        let total_notes = initial_paths.len();
        let mut state = RuntimeState {
            status: EmbeddingRuntimeStatus {
                total_notes,
                ..EmbeddingRuntimeStatus::default()
            },
            ..RuntimeState::default()
        };
        for path in initial_paths {
            let generation = next_generation(&mut state);
            state.pending.insert(
                path.clone(),
                PendingWork {
                    path,
                    generation,
                    kind: PendingKind::Upsert,
                    attempt: 0,
                    not_before: Instant::now(),
                },
            );
        }
        update_pending_status(&mut state);
        let shared = Arc::new(RuntimeShared {
            vault_root,
            index,
            cache_path,
            cache_migration_sources,
            state: Mutex::new(state),
            notify: Notify::new(),
            live: Arc::new(AtomicBool::new(true)),
            lifecycle_gate: Arc::new(Mutex::new(())),
            first_load_error: OnceLock::new(),
        });
        let worker_shared = Arc::clone(&shared);
        let worker = tokio::spawn(async move {
            run_coordinator(worker_shared, loader).await;
        });
        Self {
            control: Arc::new(RuntimeControl {
                shared,
                worker: Mutex::new(Some(worker)),
            }),
        }
    }

    pub(crate) fn submit_upsert(&self, path: &Path) {
        self.submit(path, PendingKind::Upsert);
    }

    pub(crate) fn submit_remove(&self, path: &Path) {
        self.submit(path, PendingKind::Remove);
    }

    pub(crate) fn downgrade(&self) -> EmbeddingRuntimeWeak {
        EmbeddingRuntimeWeak {
            control: Arc::downgrade(&self.control),
        }
    }

    fn submit(&self, path: &Path, kind: PendingKind) {
        let normalized = match super::path::normalize_relative(path) {
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
        state.pending.insert(
            normalized.clone(),
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
        self.control.shared.notify.notify_one();
    }

    pub(crate) fn status(&self) -> EmbeddingRuntimeStatus {
        let paths = current_paths(&self.control.shared.index);
        let state = self
            .control
            .shared
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        status_for_current_paths(&state, &paths)
    }

    pub(crate) fn query_snapshot(&self) -> VaultResult<EmbeddingQuerySnapshot> {
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
        let normalized = super::path::normalize_relative(path).ok()?;
        self.control
            .shared
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .pending
            .get(&normalized)
            .map(|work| work.kind)
    }
}

impl EmbeddingRuntimeWeak {
    pub(crate) fn submit_upsert(&self, path: &Path) {
        self.submit(path, PendingKind::Upsert);
    }

    pub(crate) fn submit_remove(&self, path: &Path) {
        self.submit(path, PendingKind::Remove);
    }

    fn submit(&self, path: &Path, kind: PendingKind) {
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
    let model = match loader.await {
        Ok(model) => model,
        Err(error) => {
            let message = error.to_string();
            let _ = shared.first_load_error.set(message.clone());
            let total_notes = current_paths(&shared.index).len();
            let mut state = shared
                .state
                .lock()
                .unwrap_or_else(|lock_error| lock_error.into_inner());
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
            return;
        }
    };
    if !shared.live.load(Ordering::Acquire) {
        return;
    }

    let current_paths = current_paths(&shared.index);
    let expected_identity = model.space_identity().clone();
    let cache_path = shared.cache_path.clone();
    let cache_migration_sources = shared.cache_migration_sources.clone();
    let live = Arc::clone(&shared.live);
    let lifecycle_gate = Arc::clone(&shared.lifecycle_gate);
    let current_count = current_paths.len();
    let loaded = tokio::task::spawn_blocking(move || {
        if !cache_migration_sources.is_empty() {
            let _guard = lifecycle_gate
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            if !live.load(Ordering::Acquire) {
                return None;
            }
            match migrate_cache_candidates_to_path(&cache_migration_sources, &cache_path) {
                Ok(LegacyCacheMigration::Migrated(path)) => {
                    tracing::info!(path = %path.display(), "relocated embedding cache in background");
                }
                Ok(LegacyCacheMigration::AlreadyPresent(_) | LegacyCacheMigration::NotFound) => {}
                Err(error) => {
                    tracing::warn!(
                        path = %cache_path.display(),
                        error = %error,
                        "failed to relocate embedding cache; rebuilding if necessary"
                    );
                }
            }
        }
        if !live.load(Ordering::Acquire) {
            return None;
        }
        if cache_path.is_file() {
            EmbeddingStore::load_for_space(&cache_path, &expected_identity, current_count).ok()
        } else {
            None
        }
    })
    .await
    .ok()
    .flatten();
    if !shared.live.load(Ordering::Acquire) {
        return;
    }

    let mut store =
        loaded.unwrap_or_else(|| EmbeddingStore::new_with_identity(model.space_identity().clone()));
    let previously_publishable = store.first_pass_complete();
    let mut dirty = store.retain_paths(&current_paths);
    let has_cached_vectors = !store.is_empty();
    let store = Arc::new(RwLock::new(store));

    {
        let mut state = shared
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        state.model = Some(Arc::clone(&model));
        state.store = Some(Arc::clone(&store));
        state.store_queryable =
            previously_publishable && (current_paths.is_empty() || has_cached_vectors);
        state.initial_started = true;
        state.initial_remaining = current_paths.clone();
        for path in current_paths {
            if !state.pending.contains_key(&path) {
                let generation = next_generation(&mut state);
                state.pending.insert(
                    path.clone(),
                    PendingWork {
                        path,
                        generation,
                        kind: PendingKind::Upsert,
                        attempt: 0,
                        not_before: Instant::now(),
                    },
                );
            }
        }
        update_pending_status(&mut state);
    }

    dirty |= finalize_reconciliation_if_complete(&shared, &store);
    refresh_status(&shared, &store);

    let mut last_persist = Instant::now();
    let mut persist_retry_at = Instant::now();
    let mut persist_attempt = 0u8;

    loop {
        if !shared.live.load(Ordering::Acquire) {
            return;
        }
        let notified = shared.notify.notified();
        let (batch, next_pending, pending_empty) = take_due_batch(&shared);
        if !batch.is_empty() {
            dirty |= process_batch(&shared, &store, &model, batch).await;
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
        .filter(|(_, work)| work.not_before <= now)
        .map(|(path, work)| (path.clone(), work.kind))
        .collect::<Vec<_>>();
    due.sort_unstable_by(|(left_path, left_kind), (right_path, right_kind)| {
        let left_priority = usize::from(*left_kind == PendingKind::Upsert);
        let right_priority = usize::from(*right_kind == PendingKind::Upsert);
        left_priority
            .cmp(&right_priority)
            .then_with(|| left_path.cmp(right_path))
    });
    due.truncate(RECONCILE_BATCH_SIZE);

    let mut batch = Vec::with_capacity(due.len());
    for (path, _) in due {
        if let Some(work) = state.pending.remove(&path) {
            state.inflight.insert(path, work.generation);
            batch.push(work);
        }
    }
    let next_pending = state.pending.values().map(|work| work.not_before).min();
    let pending_empty = state.pending.is_empty() && state.inflight.is_empty();
    update_pending_status(&mut state);
    (batch, next_pending, pending_empty)
}

async fn process_batch(
    shared: &Arc<RuntimeShared>,
    store: &Arc<RwLock<EmbeddingStore>>,
    model: &Arc<dyn Embedder>,
    batch: Vec<PendingWork>,
) -> bool {
    let vault_root = shared.vault_root.clone();
    let index = Arc::clone(&shared.index);
    let preparation_fallback = batch.clone();
    let prepared = match tokio::task::spawn_blocking(move || {
        prepare_batch(&vault_root, &index, batch)
    })
    .await
    {
        Ok(prepared) => prepared,
        Err(error) => {
            let message = format!("embedding preparation task failed: {error}");
            for work in preparation_fallback {
                requeue_failure(shared, work, message.clone());
            }
            return false;
        }
    };

    let mut dirty = false;
    let mut changed = Vec::new();
    for item in prepared {
        match item {
            PreparedWork::Remove(work) => {
                dirty |= commit_remove(shared, store, &work);
            }
            PreparedWork::Failed(work, error) => {
                requeue_failure(shared, work, error);
            }
            PreparedWork::Upsert {
                work,
                text,
                content_hash,
            } => {
                let unchanged = store
                    .read()
                    .unwrap_or_else(|error| error.into_inner())
                    .content_hash(&work.path)
                    .is_some_and(|cached| cached == &content_hash);
                if unchanged {
                    mark_success(shared, &work);
                } else {
                    changed.push((work, text, content_hash));
                }
            }
        }
    }

    if changed.is_empty() {
        return dirty;
    }

    let embedder = Arc::clone(model);
    let inference_fallback = changed
        .iter()
        .map(|(work, _, _)| work.clone())
        .collect::<Vec<_>>();
    let inference = tokio::task::spawn_blocking(move || {
        let texts = changed
            .iter()
            .map(|(_, text, _)| text.as_str())
            .collect::<Vec<_>>();
        let result = embedder.embed_batch(&texts).and_then(|vectors| {
            validate_embedding_batch(vectors, texts.len(), embedder.dimension())
        });
        (changed, result)
    })
    .await;

    let (changed, vectors) = match inference {
        Ok((changed, Ok(vectors))) => (changed, vectors),
        Ok((changed, Err(error))) => {
            let message = error.to_string();
            for (work, _, _) in changed {
                requeue_failure(shared, work, message.clone());
            }
            return dirty;
        }
        Err(error) => {
            let message = format!("embedding inference task failed: {error}");
            for work in inference_fallback {
                requeue_failure(shared, work, message.clone());
            }
            return dirty;
        }
    };

    for ((work, _, content_hash), vector) in changed.into_iter().zip(vectors) {
        dirty |= commit_upsert(shared, store, &work, content_hash, vector);
    }
    dirty
}

enum PreparedWork {
    Remove(PendingWork),
    Failed(PendingWork, String),
    Upsert {
        work: PendingWork,
        text: String,
        content_hash: [u8; 32],
    },
}

fn prepare_batch(
    vault_root: &Path,
    index: &Arc<RwLock<VaultIndex>>,
    batch: Vec<PendingWork>,
) -> Vec<PreparedWork> {
    let metadata = {
        let index = index.read().unwrap_or_else(|error| error.into_inner());
        batch
            .iter()
            .filter_map(|work| {
                index
                    .get_note(&work.path)
                    .cloned()
                    .map(|metadata| (work.path.clone(), metadata))
            })
            .collect::<HashMap<_, _>>()
    };

    batch
        .into_iter()
        .map(|work| {
            if work.kind == PendingKind::Remove {
                return PreparedWork::Remove(work);
            }
            let Some(metadata) = metadata.get(&work.path) else {
                return PreparedWork::Remove(work);
            };
            match super::fs::read_file(vault_root, &work.path) {
                Ok(content) => {
                    let headings = metadata
                        .headings
                        .iter()
                        .map(|heading| heading.text.clone())
                        .collect::<Vec<_>>();
                    let text = prepare_embed_text(
                        &metadata.title,
                        &headings,
                        super::frontmatter::get_body(&content),
                    );
                    let content_hash = prepared_text_hash(&text);
                    PreparedWork::Upsert {
                        work,
                        text,
                        content_hash,
                    }
                }
                Err(error) => PreparedWork::Failed(work, error.to_string()),
            }
        })
        .collect()
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
    if !shared.live.load(Ordering::Acquire) {
        return false;
    }
    let mut state = shared
        .state
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    if state_generation(&state, &work.path) != Some(work.generation) {
        state.inflight.remove(&work.path);
        update_pending_status(&mut state);
        return false;
    }
    let removed = store
        .write()
        .unwrap_or_else(|error| error.into_inner())
        .remove(&work.path);
    finish_success(&mut state, work);
    removed
}

fn commit_upsert(
    shared: &RuntimeShared,
    store: &Arc<RwLock<EmbeddingStore>>,
    work: &PendingWork,
    content_hash: [u8; 32],
    vector: Vec<f32>,
) -> bool {
    let _lifecycle_guard = shared
        .lifecycle_gate
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    if !shared.live.load(Ordering::Acquire) {
        return false;
    }
    let mut state = shared
        .state
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    if state_generation(&state, &work.path) != Some(work.generation) {
        state.inflight.remove(&work.path);
        update_pending_status(&mut state);
        return false;
    }
    let result = store
        .write()
        .unwrap_or_else(|error| error.into_inner())
        .insert_hashed(work.path.clone(), content_hash, vector);
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
            requeue_failure(shared, work.clone(), error.to_string());
            false
        }
    }
}

fn mark_success(shared: &RuntimeShared, work: &PendingWork) {
    let mut state = shared
        .state
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    if state_generation(&state, &work.path) == Some(work.generation) {
        finish_success(&mut state, work);
    } else {
        state.inflight.remove(&work.path);
        update_pending_status(&mut state);
    }
}

fn finish_success(state: &mut RuntimeState, work: &PendingWork) {
    state.inflight.remove(&work.path);
    state.initial_remaining.remove(&work.path);
    state.failures.remove(&work.path);
    update_pending_status(state);
}

fn requeue_failure(shared: &RuntimeShared, mut work: PendingWork, error: String) {
    let mut state = shared
        .state
        .lock()
        .unwrap_or_else(|lock_error| lock_error.into_inner());
    let is_current = state_generation(&state, &work.path) == Some(work.generation);
    state.inflight.remove(&work.path);
    if is_current {
        state.initial_remaining.remove(&work.path);
        state.failures.insert(work.path.clone(), error);
        work.attempt = work.attempt.saturating_add(1);
        work.not_before = Instant::now() + retry_delay(work.attempt);
        state.pending.insert(work.path.clone(), work);
    }
    update_pending_status(&mut state);
    drop(state);
    shared.notify.notify_one();
}

fn finalize_reconciliation_if_complete(
    shared: &RuntimeShared,
    store: &Arc<RwLock<EmbeddingStore>>,
) -> bool {
    let _lifecycle_guard = shared
        .lifecycle_gate
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    if !shared.live.load(Ordering::Acquire) {
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
    let mut store_guard = store.write().unwrap_or_else(|error| error.into_inner());
    let changed = !store_guard.first_pass_complete();
    store_guard.set_first_pass_complete(true);
    let has_vectors = !store_guard.is_empty();
    drop(store_guard);
    let total_notes = shared
        .index
        .read()
        .unwrap_or_else(|error| error.into_inner())
        .notes()
        .len();
    if !state.store_queryable && (total_notes == 0 || has_vectors) {
        state.store_queryable = true;
    }
    changed
}

async fn persist_store(
    shared: &Arc<RuntimeShared>,
    store: &Arc<RwLock<EmbeddingStore>>,
) -> VaultResult<bool> {
    let store = Arc::clone(store);
    let cache_path = shared.cache_path.clone();
    let live = Arc::clone(&shared.live);
    let lifecycle_gate = Arc::clone(&shared.lifecycle_gate);
    tokio::task::spawn_blocking(move || {
        let bytes = {
            let store = store.read().unwrap_or_else(|error| error.into_inner());
            store.encode_cache()?
        };
        let _guard = lifecycle_gate
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        EmbeddingStore::persist_cache_bytes_if_live(&cache_path, &bytes, &live)
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
            .filter(|path| store.get(path).is_some())
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
    paths: &HashSet<PathBuf>,
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
            .filter(|path| store.get(path).is_some())
            .count()
    });
    status_for_loaded_state(state, paths.len(), indexed_notes)
}

fn replace_status(state: &mut RuntimeState, next: EmbeddingRuntimeStatus) {
    let previous = &state.status;
    let meaningful_transition = previous.phase != next.phase
        || previous.queryable != next.queryable
        || previous.last_error.is_some() != next.last_error.is_some()
        || (previous.total_notes == 0 && next.total_notes > 0);

    if meaningful_transition {
        match next.phase {
            EmbeddingPhase::Degraded => tracing::warn!(
                phase = ?next.phase,
                queryable = next.queryable,
                indexed_notes = next.indexed_notes,
                total_notes = next.total_notes,
                pending_notes = next.pending_notes,
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

fn current_paths(index: &Arc<RwLock<VaultIndex>>) -> HashSet<PathBuf> {
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
        .get(path)
        .map(|work| work.generation)
        .or_else(|| state.inflight.get(path).copied())
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
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

    use super::*;
    use crate::vault::embeddings::{
        EMBEDDING_INPUT_VERSION, EmbeddingBackendKind, EmbeddingSpaceIdentity,
    };
    use crate::vault::exclude::ExcludeSet;

    struct FakeEmbedder {
        identity: EmbeddingSpaceIdentity,
        calls: AtomicUsize,
        fail: AtomicBool,
        inputs: Mutex<Vec<String>>,
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
                },
                calls: AtomicUsize::new(0),
                fail: AtomicBool::new(false),
                inputs: Mutex::new(Vec::new()),
            }
        }
    }

    impl Embedder for FakeEmbedder {
        fn dimension(&self) -> usize {
            self.identity.dimension
        }

        fn space_identity(&self) -> &EmbeddingSpaceIdentity {
            &self.identity
        }

        fn embed_batch(&self, texts: &[&str]) -> VaultResult<Vec<Vec<f32>>> {
            self.calls.fetch_add(texts.len(), AtomicOrdering::SeqCst);
            self.inputs
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .extend(texts.iter().map(|text| (*text).to_string()));
            if self.fail.load(AtomicOrdering::SeqCst) {
                return Err(VaultError::Embedding("injected inference failure".into()));
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

    fn prepared_note_text(root: &Path, index: &Arc<RwLock<VaultIndex>>, path: &Path) -> String {
        let metadata = index
            .read()
            .unwrap_or_else(|error| error.into_inner())
            .get_note(path)
            .cloned()
            .unwrap();
        let content = super::super::fs::read_file(root, path).unwrap();
        let headings = metadata
            .headings
            .iter()
            .map(|heading| heading.text.clone())
            .collect::<Vec<_>>();
        prepare_embed_text(
            &metadata.title,
            &headings,
            super::super::frontmatter::get_body(&content),
        )
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
    async fn compatible_cache_relocation_runs_in_background_and_reuses_vectors() {
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
        let runtime = EmbeddingRuntime::spawn_with_cache_sources(
            directory.path().to_path_buf(),
            index,
            target_path.clone(),
            vec![source_path.clone()],
            async move { Ok(loader_fake as Arc<dyn Embedder>) },
        );
        wait_for_status(&runtime, |status| status.phase == EmbeddingPhase::Ready).await;

        assert_eq!(fake.calls.load(AtomicOrdering::SeqCst), 0);
        assert!(runtime.status().queryable);
        assert!(
            source_path.is_file(),
            "relocation must not delete its source"
        );
        assert!(target_path.is_file(), "relocation must publish the target");
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
        assert_eq!(status.pending_notes, 2);
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
    }

    #[tokio::test]
    async fn busy_initial_reconciliation_persists_an_incomplete_checkpoint() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(directory.path().join(".obsidian")).unwrap();
        for index in 0..(RECONCILE_BATCH_SIZE + 1) {
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
        assert_eq!(checkpoint.len(), RECONCILE_BATCH_SIZE);
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
            let generation = *state
                .inflight
                .get(Path::new("one.md"))
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
        let committer = std::thread::spawn(move || {
            commit_upsert(
                &commit_shared,
                &commit_store,
                &work,
                prepared_text_hash("new body"),
                vec![1.0, 0.0, 0.0],
            )
        });
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
}
