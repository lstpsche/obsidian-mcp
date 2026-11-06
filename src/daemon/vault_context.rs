//! Per-vault daemon runtime context (index, semantic state, watcher).

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};

use notify_debouncer_mini::Debouncer;

use crate::error::{VaultError, VaultResult};
use crate::vault::exclude::ExcludeSet;
use crate::vault::index::VaultIndex;
use crate::vault::tantivy_index::TantivyIndex;
use crate::vault::watcher::ChangeWatcher;

#[cfg(has_embeddings)]
use crate::vault::embedding_runtime::{EmbeddingRuntime, EmbeddingRuntimeStatus};
#[cfg(has_embeddings)]
use crate::vault::embeddings::Embedder;

use crate::vault::{fs::VaultRoot, watcher::PendingWatcher};

#[cfg(has_embeddings)]
pub(crate) type EmbeddingLoaderFuture = std::pin::Pin<
    Box<dyn std::future::Future<Output = VaultResult<Arc<dyn Embedder>>> + Send + 'static>,
>;

pub struct VaultContext {
    vault_id: String,
    root: VaultRoot,
    exclude: Arc<ExcludeSet>,
    model_name: String,
    index: Arc<RwLock<VaultIndex>>,
    tantivy: Arc<TantivyIndex>,
    #[cfg(has_embeddings)]
    embedding_runtime: EmbeddingRuntime,
    watcher: Mutex<Option<Debouncer<ChangeWatcher>>>,
}

impl VaultContext {
    pub(crate) async fn open(
        vault_id: String,
        root: VaultRoot,
        model_name: String,
        exclude: Arc<ExcludeSet>,
        watch_enabled: bool,
        #[cfg(has_embeddings)] cache: crate::vault::embeddings::CacheFile,
        #[cfg(has_embeddings)] embedding_loader: EmbeddingLoaderFuture,
    ) -> VaultResult<Self> {
        let pending_watcher = if watch_enabled {
            let watch_root = root.clone();
            Some(
                tokio::task::spawn_blocking(move || PendingWatcher::register(watch_root.path()))
                    .await
                    .map_err(|error| {
                        VaultError::Watcher(format!("daemon watcher startup failed: {error}"))
                    })??,
            )
        } else {
            None
        };
        let (index, tantivy) = VaultIndex::build_ranked(root.clone(), Arc::clone(&exclude)).await?;
        let index = Arc::new(RwLock::new(index));
        let tantivy = Arc::new(tantivy);

        #[cfg(has_embeddings)]
        let embedding_runtime = {
            let batch_size =
                crate::config::embedding_batch_size_from_env().map_err(VaultError::Embedding)?;
            EmbeddingRuntime::spawn_with_scoped_cache(
                root.clone(),
                Arc::clone(&index),
                cache,
                batch_size,
                false,
                embedding_loader,
            )
        };

        let context = Self {
            vault_id,
            root,
            exclude,
            model_name,
            index,
            tantivy,
            #[cfg(has_embeddings)]
            embedding_runtime,
            watcher: Mutex::new(None),
        };
        match pending_watcher {
            Some(pending) => tokio::task::spawn_blocking(move || {
                context.install_watcher(context.start_watcher(pending));
                context
            })
            .await
            .map_err(|error| {
                VaultError::Watcher(format!("daemon watcher catch-up failed: {error}"))
            }),
            None => Ok(context),
        }
    }

    pub fn vault_id(&self) -> &str {
        &self.vault_id
    }

    pub fn vault_root(&self) -> &Path {
        self.root.path()
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    pub fn watch_enabled(&self) -> bool {
        self.watcher
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .is_some()
    }

    fn start_watcher(&self, pending: PendingWatcher) -> Debouncer<ChangeWatcher> {
        pending.start(
            Arc::clone(&self.index),
            Some(Arc::clone(&self.tantivy)),
            #[cfg(has_embeddings)]
            Some(self.embedding_runtime.clone()),
            Arc::clone(&self.exclude),
        )
    }

    /// A view can acquire its first watching holder after an unwatched interval.
    /// Register before scanning so edits during catch-up are also reconciled.
    pub(crate) async fn resume_watcher(self: &Arc<Self>) -> VaultResult<Debouncer<ChangeWatcher>> {
        let context = Arc::clone(self);
        tokio::task::spawn_blocking(move || {
            let mut pending = PendingWatcher::register(context.root.path())?;
            pending.request_rescan();
            Ok(context.start_watcher(pending))
        })
        .await
        .map_err(|error| VaultError::Watcher(format!("daemon watcher startup failed: {error}")))?
    }

    pub(crate) fn install_watcher(&self, watcher: Debouncer<ChangeWatcher>) {
        *self.watcher.lock().unwrap_or_else(|err| err.into_inner()) = Some(watcher);
    }

    pub(crate) fn stop_watcher(&self) {
        self.watcher
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .take();
    }

    pub(crate) fn activate(&self) {
        #[cfg(has_embeddings)]
        self.embedding_runtime.activate();
    }

    pub(crate) fn retire(&self) {
        #[cfg(has_embeddings)]
        self.embedding_runtime.retire();
        self.stop_watcher();
    }

    pub fn read_note(&self, path: &Path) -> VaultResult<String> {
        self.root.read_file(path)
    }

    pub fn canonical_existing_relative_path(&self, path: &Path) -> VaultResult<PathBuf> {
        Ok(self.root.resolve_existing(path)?.relative)
    }

    #[cfg(has_embeddings)]
    fn captured_semantic_entries(
        &self,
        allowed_paths: Option<&[String]>,
    ) -> VaultResult<(
        crate::vault::embedding_runtime::EmbeddingQuerySnapshot,
        crate::vault::semantic_results::CapturedEntries,
    )> {
        let allowed = allowed_paths.map(|paths| {
            paths
                .iter()
                .map(PathBuf::from)
                .collect::<std::collections::HashSet<_>>()
        });
        let current_paths = self
            .index
            .read()
            .map_err(|error| VaultError::Other(format!("daemon index lock poisoned: {error}")))?
            .notes()
            .keys()
            .filter(|path| {
                allowed
                    .as_ref()
                    .is_none_or(|allowed| allowed.contains(*path))
            })
            .cloned()
            .collect();
        let snapshot = self.embedding_runtime.query_snapshot()?;
        let entries = snapshot.capture_entries(&current_paths);
        Ok((snapshot, entries))
    }

    #[cfg(has_embeddings)]
    pub(crate) fn search_semantic_results(
        &self,
        query: &str,
        top_k: usize,
        allowed_paths: Option<&[String]>,
        include_content: bool,
    ) -> VaultResult<Vec<crate::models::SemanticNoteResult>> {
        if query.is_empty() || top_k == 0 {
            return Ok(Vec::new());
        }
        let (snapshot, entries) = self.captured_semantic_entries(allowed_paths)?;
        if entries.is_empty() {
            snapshot.validate_query(query)?;
            return Ok(Vec::new());
        }
        let query_vector = snapshot.embed_query(query)?;
        let hits = crate::vault::semantic_results::rank_semantic(entries, &query_vector, top_k)?;
        Ok(hits
            .into_iter()
            .map(|hit| {
                crate::vault::semantic_results::render_rooted(
                    &self.root,
                    hit,
                    query,
                    include_content,
                )
            })
            .collect())
    }

    #[cfg(has_embeddings)]
    pub(crate) fn search_hybrid_results(
        &self,
        query: &str,
        top_k: usize,
        prefetch: usize,
        alpha: f32,
        allowed_paths: Option<&[String]>,
        include_content: bool,
    ) -> VaultResult<Vec<crate::models::SemanticNoteResult>> {
        if query.is_empty() || top_k == 0 {
            return Ok(Vec::new());
        }
        let (snapshot, entries) = self.captured_semantic_entries(allowed_paths)?;
        snapshot.validate_query(query)?;
        if entries.is_empty() {
            return Ok(Vec::new());
        }
        let allowed = crate::vault::semantic_results::paths(&entries);
        let lexical = self
            .tantivy
            .search_for_paths(query, prefetch.max(top_k), Some(&allowed))?;
        if lexical.is_empty() {
            return Ok(Vec::new());
        }
        let query_vector = snapshot.embed_query(query)?;
        let hits = crate::vault::semantic_results::rank_hybrid(
            &entries,
            &lexical,
            &query_vector,
            alpha.clamp(0.0, 1.0),
            top_k,
        )?;
        Ok(hits
            .into_iter()
            .map(|hit| {
                crate::vault::semantic_results::render_rooted(
                    &self.root,
                    hit,
                    query,
                    include_content,
                )
            })
            .collect())
    }

    #[cfg(not(has_embeddings))]
    pub(crate) fn search_semantic_results(
        &self,
        _query: &str,
        _top_k: usize,
        _allowed_paths: Option<&[String]>,
        _include_content: bool,
    ) -> VaultResult<Vec<crate::models::SemanticNoteResult>> {
        Err(VaultError::Embedding(
            "daemon binary compiled without embeddings feature".into(),
        ))
    }

    #[cfg(not(has_embeddings))]
    pub(crate) fn search_hybrid_results(
        &self,
        _query: &str,
        _top_k: usize,
        _prefetch: usize,
        _alpha: f32,
        _allowed_paths: Option<&[String]>,
        _include_content: bool,
    ) -> VaultResult<Vec<crate::models::SemanticNoteResult>> {
        Err(VaultError::Embedding(
            "daemon binary compiled without embeddings feature".into(),
        ))
    }

    #[cfg(has_embeddings)]
    pub fn embedding_status(&self) -> EmbeddingRuntimeStatus {
        self.embedding_runtime.status()
    }
}
