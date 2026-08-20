//! Per-vault daemon runtime context (index, semantic state, watcher).

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};

use notify_debouncer_mini::Debouncer;

use crate::error::{VaultError, VaultResult};
use crate::models::NoteMetadata;
use crate::vault::exclude::ExcludeSet;
use crate::vault::index::VaultIndex;
use crate::vault::tantivy_index::TantivyIndex;

#[cfg(has_embeddings)]
use crate::vault::embedding_runtime::{EmbeddingRuntime, EmbeddingRuntimeStatus};
#[cfg(has_embeddings)]
use crate::vault::embeddings::Embedder;

use super::watcher;

#[cfg(has_embeddings)]
pub(crate) type EmbeddingLoaderFuture = std::pin::Pin<
    Box<dyn std::future::Future<Output = VaultResult<Arc<dyn Embedder>>> + Send + 'static>,
>;

pub struct VaultContext {
    vault_id: String,
    vault_root: PathBuf,
    model_name: String,
    index: Arc<RwLock<VaultIndex>>,
    tantivy: Arc<TantivyIndex>,
    #[cfg(has_embeddings)]
    embedding_runtime: EmbeddingRuntime,
    watcher: Mutex<Option<Debouncer<notify::RecommendedWatcher>>>,
}

impl VaultContext {
    pub(crate) async fn open(
        vault_id: String,
        vault_root: PathBuf,
        model_name: String,
        state_dir: PathBuf,
        watch_enabled: bool,
        #[cfg(has_embeddings)] embedding_loader: EmbeddingLoaderFuture,
    ) -> VaultResult<Self> {
        std::fs::create_dir_all(&state_dir)?;

        let index = Arc::new(RwLock::new(
            VaultIndex::build(&vault_root, Arc::new(ExcludeSet::build(vec![])?)).await?,
        ));
        let tantivy = {
            let index_guard = index
                .read()
                .map_err(|err| VaultError::Other(format!("daemon index lock poisoned: {err}")))?;
            TantivyIndex::build(&vault_root, index_guard.notes())?
        };
        let tantivy = Arc::new(tantivy);

        #[cfg(has_embeddings)]
        let embedding_runtime = EmbeddingRuntime::spawn(
            vault_root.clone(),
            Arc::clone(&index),
            state_dir.join("embeddings.bin"),
            embedding_loader,
        );

        let context = Self {
            vault_id,
            vault_root,
            model_name,
            index,
            tantivy,
            #[cfg(has_embeddings)]
            embedding_runtime,
            watcher: Mutex::new(None),
        };

        if watch_enabled {
            context.ensure_watcher()?;
        }

        Ok(context)
    }

    pub fn vault_id(&self) -> &str {
        &self.vault_id
    }

    pub fn vault_root(&self) -> &Path {
        &self.vault_root
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    pub fn watch_enabled(&self) -> VaultResult<bool> {
        let guard = self
            .watcher
            .lock()
            .map_err(|err| VaultError::Other(format!("daemon watcher lock poisoned: {err}")))?;
        Ok(guard.is_some())
    }

    pub fn ensure_watcher(&self) -> VaultResult<bool> {
        let mut guard = self
            .watcher
            .lock()
            .map_err(|err| VaultError::Other(format!("daemon watcher lock poisoned: {err}")))?;

        if guard.is_some() {
            return Ok(true);
        }

        #[cfg(has_embeddings)]
        let debouncer = watcher::start_watcher(
            self.vault_root.clone(),
            Arc::clone(&self.index),
            Some(Arc::clone(&self.tantivy)),
            self.embedding_runtime.clone(),
            Arc::new(ExcludeSet::build(vec![])?),
        )?;

        #[cfg(not(has_embeddings))]
        let debouncer = watcher::start_watcher(
            self.vault_root.clone(),
            Arc::clone(&self.index),
            Some(Arc::clone(&self.tantivy)),
            Arc::new(ExcludeSet::build(vec![])?),
        )?;

        *guard = Some(debouncer);
        Ok(true)
    }

    pub fn note_metadata(&self, path: &Path) -> VaultResult<Option<NoteMetadata>> {
        let actual_path = match self.canonical_existing_relative_path(path) {
            Ok(path) => path,
            Err(VaultError::NoteNotFound(_)) => return Ok(None),
            Err(err) => return Err(err),
        };
        let guard = self
            .index
            .read()
            .map_err(|err| VaultError::Other(format!("daemon index lock poisoned: {err}")))?;
        Ok(guard.get_note(&actual_path).cloned())
    }

    pub fn read_note(&self, path: &Path) -> VaultResult<String> {
        crate::vault::fs::read_file(&self.vault_root, path)
    }

    pub fn canonical_existing_relative_path(&self, path: &Path) -> VaultResult<PathBuf> {
        Ok(crate::vault::path::resolve_existing(&self.vault_root, path)?.relative)
    }

    pub fn search_bm25(&self, query: &str, top_k: usize) -> VaultResult<Vec<(PathBuf, f32)>> {
        self.tantivy.search(query, top_k)
    }

    #[cfg(has_embeddings)]
    pub fn search_semantic_scores(
        &self,
        query: &str,
        top_k: usize,
    ) -> VaultResult<Vec<(PathBuf, f32)>> {
        self.embedding_runtime
            .query_snapshot()?
            .semantic_scores(query, top_k)
    }

    #[cfg(has_embeddings)]
    pub fn query_embedding(&self, query: &str) -> VaultResult<Vec<f32>> {
        self.embedding_runtime.query_snapshot()?.embed_query(query)
    }

    #[cfg(has_embeddings)]
    pub fn semantic_score_for(&self, path: &Path, query_embedding: &[f32]) -> VaultResult<f32> {
        Ok(self
            .embedding_runtime
            .query_snapshot()?
            .score_for(path, query_embedding))
    }

    #[cfg(has_embeddings)]
    pub fn embedding_status(&self) -> EmbeddingRuntimeStatus {
        self.embedding_runtime.status()
    }

    #[cfg(not(has_embeddings))]
    pub fn search_semantic_scores(
        &self,
        _query: &str,
        _top_k: usize,
    ) -> VaultResult<Vec<(PathBuf, f32)>> {
        Err(VaultError::Embedding(
            "daemon binary compiled without embeddings feature".to_string(),
        ))
    }

    #[cfg(not(has_embeddings))]
    pub fn query_embedding(&self, _query: &str) -> VaultResult<Vec<f32>> {
        Err(VaultError::Embedding(
            "daemon binary compiled without embeddings feature".to_string(),
        ))
    }

    #[cfg(not(has_embeddings))]
    pub fn semantic_score_for(&self, _path: &Path, _query_embedding: &[f32]) -> VaultResult<f32> {
        Err(VaultError::Embedding(
            "daemon binary compiled without embeddings feature".to_string(),
        ))
    }
}
