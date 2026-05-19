//! Registry of active daemon vault contexts keyed by stable `vault_id`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use tokio::sync::RwLock;

#[cfg(has_embeddings)]
use tokio::sync::OnceCell;

use crate::error::{VaultError, VaultResult};

#[cfg(has_embeddings)]
use crate::vault::embeddings::EmbeddingModel;

use super::home::{self, SemanticHomePaths};
use super::vault_context::VaultContext;

pub struct VaultRegistry {
    paths: SemanticHomePaths,
    model_name: String,
    contexts: RwLock<HashMap<String, Arc<VaultContext>>>,
    init_locks: tokio::sync::Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
    #[cfg(has_embeddings)]
    embedding_model: OnceCell<Arc<EmbeddingModel>>,
}

impl VaultRegistry {
    pub fn new(semantic_home: PathBuf, model_name: String) -> VaultResult<Self> {
        let paths = home::semantic_home_paths(&semantic_home);
        home::ensure_home_layout(&paths)?;

        Ok(Self {
            paths,
            model_name,
            contexts: RwLock::new(HashMap::new()),
            init_locks: tokio::sync::Mutex::new(HashMap::new()),
            #[cfg(has_embeddings)]
            embedding_model: OnceCell::new(),
        })
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    pub async fn ensure_vault(
        &self,
        vault_root: &Path,
        watch_enabled: bool,
        requested_model: &str,
    ) -> VaultResult<Arc<VaultContext>> {
        if requested_model != self.model_name {
            return Err(VaultError::InvalidPath(format!(
                "requested model '{requested_model}' does not match daemon model '{}'",
                self.model_name
            )));
        }

        let canonical_root = canonicalize_vault_root(vault_root)?;
        let vault_id = home::compute_vault_id(&canonical_root)?;

        if let Some(existing) = self.get_by_id(&vault_id).await {
            if watch_enabled {
                existing.ensure_watcher()?;
            }
            return Ok(existing);
        }

        let init_lock = {
            let mut locks = self.init_locks.lock().await;
            Arc::clone(locks.entry(vault_id.clone()).or_default())
        };
        let _init_guard = init_lock.lock().await;

        if let Some(existing) = self.get_by_id(&vault_id).await {
            if watch_enabled {
                existing.ensure_watcher()?;
            }
            return Ok(existing);
        }

        #[cfg(has_embeddings)]
        let embedding_model = self.embedding_model().await?;

        let state_dir = self.paths.vaults_dir.join(&vault_id);
        let context = VaultContext::open(
            vault_id.clone(),
            canonical_root,
            self.model_name.clone(),
            state_dir,
            watch_enabled,
            #[cfg(has_embeddings)]
            embedding_model,
        )
        .await?;
        let context = Arc::new(context);

        let mut guard = self.contexts.write().await;
        guard.insert(vault_id, Arc::clone(&context));
        drop(guard);

        if watch_enabled {
            context.ensure_watcher()?;
        }
        Ok(context)
    }

    pub async fn get_context_by_root(
        &self,
        vault_root: &Path,
    ) -> VaultResult<Option<Arc<VaultContext>>> {
        let canonical_root = canonicalize_vault_root(vault_root)?;
        let vault_id = home::compute_vault_id(&canonical_root)?;
        Ok(self.get_by_id(&vault_id).await)
    }

    async fn get_by_id(&self, vault_id: &str) -> Option<Arc<VaultContext>> {
        let guard = self.contexts.read().await;
        guard.get(vault_id).cloned()
    }

    #[cfg(has_embeddings)]
    async fn embedding_model(&self) -> VaultResult<Arc<EmbeddingModel>> {
        let model = self
            .embedding_model
            .get_or_try_init(|| async {
                let loaded = EmbeddingModel::load(&self.model_name, None).await?;
                Ok::<Arc<EmbeddingModel>, VaultError>(Arc::new(loaded))
            })
            .await?;
        Ok(Arc::clone(model))
    }
}

fn canonicalize_vault_root(vault_root: &Path) -> VaultResult<PathBuf> {
    if !vault_root.is_absolute() {
        return Err(VaultError::InvalidPath(format!(
            "vault_root must be absolute: {}",
            vault_root.display()
        )));
    }

    let canonical = vault_root.canonicalize().map_err(|err| {
        VaultError::InvalidPath(format!(
            "failed to canonicalize vault root '{}': {err}",
            vault_root.display()
        ))
    })?;

    if !canonical.is_dir() {
        return Err(VaultError::InvalidPath(format!(
            "vault_root is not a directory: {}",
            canonical.display()
        )));
    }

    Ok(canonical)
}
