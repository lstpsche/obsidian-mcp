//! Managed background lifecycle for optional semantic embeddings.

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::{Duration, Instant};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio::sync::Notify;
use tokio::task::JoinHandle;

use crate::error::{VaultError, VaultResult};

use super::embeddings::{
    Embedder, EmbeddingStore, prepare_embed_text, prepared_text_hash, validate_embedding_batch,
};
use super::index::VaultIndex;

const RECONCILE_BATCH_SIZE: usize = 32;
const MAX_DIRTY_INTERVAL: Duration = Duration::from_secs(2);
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

struct RuntimeControl {
    shared: Arc<RuntimeShared>,
    worker: Mutex<Option<JoinHandle<()>>>,
}

impl Drop for RuntimeControl {
    fn drop(&mut self) {
        self.shared.live.store(false, Ordering::Release);
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
    state: Mutex<RuntimeState>,
    notify: Notify,
    live: Arc<AtomicBool>,
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
    published_store: Option<Arc<RwLock<EmbeddingStore>>>,
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

    pub(crate) fn semantic_scores(
        &self,
        query: &str,
        top_k: usize,
    ) -> VaultResult<Vec<(PathBuf, f32)>> {
        let query_vector = self.embed_query(query)?;
        let store = self.store.read().unwrap_or_else(|error| error.into_inner());
        Ok(store.query(&query_vector, top_k))
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
    pub(crate) fn spawn<F>(
        vault_root: PathBuf,
        index: Arc<RwLock<VaultIndex>>,
        cache_path: PathBuf,
        loader: F,
    ) -> Self
    where
        F: Future<Output = VaultResult<Arc<dyn Embedder>>> + Send + 'static,
    {
        let shared = Arc::new(RuntimeShared {
            vault_root,
            index,
            cache_path,
            state: Mutex::new(RuntimeState::default()),
            notify: Notify::new(),
            live: Arc::new(AtomicBool::new(true)),
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
        self.control
            .shared
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .status
            .clone()
    }

    pub(crate) fn query_snapshot(&self) -> VaultResult<EmbeddingQuerySnapshot> {
        let state = self
            .control
            .shared
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        match (&state.model, &state.published_store) {
            (Some(model), Some(store)) if state.status.queryable => Ok(EmbeddingQuerySnapshot {
                model: Arc::clone(model),
                store: Arc::clone(store),
            }),
            _ => Err(VaultError::Embedding(not_ready_message(&state.status))),
        }
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
    let current_count = current_paths.len();
    let loaded = tokio::task::spawn_blocking(move || {
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
        if previously_publishable && (current_paths.is_empty() || has_cached_vectors) {
            state.published_store = Some(Arc::clone(&store));
        }
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
            continue;
        }

        let now = Instant::now();
        let persistence_due = dirty
            && now >= persist_retry_at
            && (pending_empty || now.duration_since(last_persist) >= MAX_DIRTY_INTERVAL);
        if persistence_due {
            match persist_store(&shared, &store).await {
                Ok(true) => {
                    dirty = false;
                    persist_attempt = 0;
                    persist_retry_at = Instant::now();
                    last_persist = Instant::now();
                    let mut state = shared
                        .state
                        .lock()
                        .unwrap_or_else(|error| error.into_inner());
                    state.persistence_error = None;
                }
                Ok(false) => return,
                Err(error) => {
                    persist_attempt = persist_attempt.saturating_add(1);
                    persist_retry_at = Instant::now() + retry_delay(persist_attempt);
                    let mut state = shared
                        .state
                        .lock()
                        .unwrap_or_else(|lock_error| lock_error.into_inner());
                    state.persistence_error = Some(error.to_string());
                }
            }
            refresh_status(&shared, &store);
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
            if state.reconciliation_complete && state.published_store.is_none() {
                state.published_store = Some(Arc::clone(store));
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
    if !state.pending.contains_key(&work.path) {
        state.inflight.remove(&work.path);
    }
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
    if state.published_store.is_none() && (total_notes == 0 || has_vectors) {
        state.published_store = Some(Arc::clone(store));
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
    tokio::task::spawn_blocking(move || {
        let bytes = {
            let store = store.read().unwrap_or_else(|error| error.into_inner());
            store.encode_cache()?
        };
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
    let queryable = state.model.is_some() && state.published_store.is_some();
    let pending_notes = distinct_pending_count(&state);
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
    replace_status(
        &mut state,
        EmbeddingRuntimeStatus {
            phase,
            queryable,
            indexed_notes,
            total_notes: paths.len(),
            pending_notes,
            last_error,
        },
    );
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
        let index = test_index(directory.path()).await;
        let (release_tx, release_rx) = tokio::sync::oneshot::channel();
        let runtime = EmbeddingRuntime::spawn(
            directory.path().to_path_buf(),
            index,
            directory.path().join("cache.bin"),
            async move {
                release_rx.await.unwrap();
                Ok(Arc::new(FakeEmbedder::new()) as Arc<dyn Embedder>)
            },
        );

        assert_eq!(runtime.status().phase, EmbeddingPhase::Warming);
        assert!(!runtime.status().queryable);
        release_tx.send(()).unwrap();
        wait_for_status(&runtime, |status| status.phase == EmbeddingPhase::Ready).await;
        assert!(runtime.status().queryable);
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
            index,
            cache_path,
            async move { Ok(loader_fake as Arc<dyn Embedder>) },
        );
        wait_for_status(&runtime, |status| status.phase == EmbeddingPhase::Ready).await;

        assert_eq!(fake.calls.load(AtomicOrdering::SeqCst), 0);
        assert!(runtime.status().queryable);
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
        std::fs::write(directory.path().join("one.md"), "# One\nbody").unwrap();
        let index = test_index(directory.path()).await;
        let cache_path = directory.path().join("cache.bin");
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let identity = FakeEmbedder::new().identity;
        let runtime = EmbeddingRuntime::spawn(
            directory.path().to_path_buf(),
            index,
            cache_path.clone(),
            async move {
                Ok(Arc::new(BlockingEmbedder {
                    identity,
                    started: started_tx,
                    release: Mutex::new(release_rx),
                }) as Arc<dyn Embedder>)
            },
        );

        tokio::task::spawn_blocking(move || started_rx.recv_timeout(Duration::from_secs(5)))
            .await
            .unwrap()
            .unwrap();
        drop(runtime);
        release_tx.send(()).unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;

        assert!(!cache_path.exists());
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
