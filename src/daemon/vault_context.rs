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
    watcher: Mutex<Option<Debouncer<ChangeWatcher>>>,
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
        let embedding_runtime = {
            let batch_size =
                crate::config::embedding_batch_size_from_env().map_err(VaultError::Embedding)?;
            let cache_migration_sources = vec![
                vault_root
                    .join(".obsidian-mcp")
                    .join("embeddings")
                    .join("embeddings.bin"),
                vault_root
                    .join(".obsidian")
                    .join("obsidian-mcp")
                    .join("embeddings.bin"),
            ];
            EmbeddingRuntime::spawn_with_cache_sources(
                vault_root.clone(),
                Arc::clone(&index),
                state_dir.join("embeddings.bin"),
                cache_migration_sources,
                batch_size,
                true,
                embedding_loader,
            )
        };

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

    pub fn read_note(&self, path: &Path) -> VaultResult<String> {
        crate::vault::fs::read_file(&self.vault_root, path)
    }

    pub fn canonical_existing_relative_path(&self, path: &Path) -> VaultResult<PathBuf> {
        Ok(crate::vault::path::resolve_existing(&self.vault_root, path)?.relative)
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
                crate::vault::semantic_results::render(
                    &self.vault_root,
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
                crate::vault::semantic_results::render(
                    &self.vault_root,
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
