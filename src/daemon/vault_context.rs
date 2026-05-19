//! Per-vault daemon runtime context (index, semantic state, watcher).

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};

use notify_debouncer_mini::Debouncer;

use crate::error::{VaultError, VaultResult};
use crate::models::NoteMetadata;
use crate::vault::index::VaultIndex;
use crate::vault::tantivy_index::TantivyIndex;

#[cfg(has_embeddings)]
use crate::vault::embeddings::{EmbeddingModel, EmbeddingStore};

#[cfg(has_embeddings)]
use super::indexer;
use super::watcher;

pub struct VaultContext {
    vault_id: String,
    vault_root: PathBuf,
    model_name: String,
    index: Arc<RwLock<VaultIndex>>,
    tantivy: Arc<TantivyIndex>,
    #[cfg(has_embeddings)]
    embedding_model: Arc<EmbeddingModel>,
    #[cfg(has_embeddings)]
    embedding_store: Arc<RwLock<EmbeddingStore>>,
    #[cfg(has_embeddings)]
    embedding_cache_path: PathBuf,
    watcher: Mutex<Option<Debouncer<notify::RecommendedWatcher>>>,
}

impl VaultContext {
    pub async fn open(
        vault_id: String,
        vault_root: PathBuf,
        model_name: String,
        state_dir: PathBuf,
        watch_enabled: bool,
        #[cfg(has_embeddings)] embedding_model: Arc<EmbeddingModel>,
    ) -> VaultResult<Self> {
        std::fs::create_dir_all(&state_dir)?;

        let index = Arc::new(RwLock::new(VaultIndex::build(&vault_root).await?));
        let tantivy = {
            let index_guard = index
                .read()
                .map_err(|err| VaultError::Other(format!("daemon index lock poisoned: {err}")))?;
            TantivyIndex::build(&vault_root, index_guard.notes())?
        };
        let tantivy = Arc::new(tantivy);

        #[cfg(has_embeddings)]
        let (embedding_store, embedding_cache_path) = {
            let embedding_cache_path = state_dir.join("embeddings.bin");
            let store = indexer::build_or_load_embeddings(
                &vault_root,
                &index,
                &embedding_model,
                &embedding_cache_path,
            )?;
            let store = Arc::new(RwLock::new(store));
            (store, embedding_cache_path)
        };

        let context = Self {
            vault_id,
            vault_root,
            model_name,
            index,
            tantivy,
            #[cfg(has_embeddings)]
            embedding_model,
            #[cfg(has_embeddings)]
            embedding_store,
            #[cfg(has_embeddings)]
            embedding_cache_path,
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
            Arc::clone(&self.embedding_model),
            Arc::clone(&self.embedding_store),
            self.embedding_cache_path.clone(),
        )?;

        #[cfg(not(has_embeddings))]
        let debouncer = watcher::start_watcher(
            self.vault_root.clone(),
            Arc::clone(&self.index),
            Some(Arc::clone(&self.tantivy)),
        )?;

        *guard = Some(debouncer);
        Ok(true)
    }

    pub fn note_metadata(&self, path: &Path) -> VaultResult<Option<NoteMetadata>> {
        let guard = self
            .index
            .read()
            .map_err(|err| VaultError::Other(format!("daemon index lock poisoned: {err}")))?;
        Ok(guard.get_note(path).cloned())
    }

    pub fn read_note(&self, path: &Path) -> VaultResult<String> {
        crate::vault::fs::read_file(&self.vault_root, path)
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
        let query_vec = self.embedding_model.embed_one(query)?;
        let guard = self.embedding_store.read().map_err(|err| {
            VaultError::Other(format!("daemon embedding store lock poisoned: {err}"))
        })?;
        Ok(guard.query(&query_vec, top_k))
    }

    #[cfg(has_embeddings)]
    pub fn query_embedding(&self, query: &str) -> VaultResult<Vec<f32>> {
        self.embedding_model.embed_one(query)
    }

    #[cfg(has_embeddings)]
    pub fn semantic_score_for(&self, path: &Path, query_embedding: &[f32]) -> VaultResult<f32> {
        let guard = self.embedding_store.read().map_err(|err| {
            VaultError::Other(format!("daemon embedding store lock poisoned: {err}"))
        })?;
        Ok(guard
            .get(path)
            .map(|embedding| {
                crate::vault::embeddings::cosine_similarity(query_embedding, embedding)
            })
            .unwrap_or(0.0))
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
