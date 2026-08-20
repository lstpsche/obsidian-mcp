//! Registry of active daemon vault contexts keyed by stable `vault_id`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use tokio::sync::RwLock;

#[cfg(has_embeddings)]
use tokio::sync::OnceCell;

use crate::error::{VaultError, VaultResult};

#[cfg(has_embeddings)]
use crate::vault::embeddings::{Embedder, EmbeddingModel};

use super::home::{self, SemanticHomePaths};
#[cfg(has_embeddings)]
use super::vault_context::EmbeddingLoaderFuture;
use super::vault_context::VaultContext;

#[cfg(has_embeddings)]
type EmbeddingLoaderFactory = Arc<dyn Fn() -> EmbeddingLoaderFuture + Send + Sync>;

pub struct VaultRegistry {
    paths: SemanticHomePaths,
    model_name: String,
    contexts: RwLock<HashMap<String, Arc<VaultContext>>>,
    init_locks: tokio::sync::Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
    #[cfg(has_embeddings)]
    embedding_model: Arc<OnceCell<Arc<dyn Embedder>>>,
    #[cfg(has_embeddings)]
    embedding_loader: EmbeddingLoaderFactory,
}

impl VaultRegistry {
    pub fn new(semantic_home: PathBuf, model_name: String) -> VaultResult<Self> {
        #[cfg(has_embeddings)]
        let embedding_loader: EmbeddingLoaderFactory = {
            let model_name = model_name.clone();
            Arc::new(move || {
                let model_name = model_name.clone();
                Box::pin(async move {
                    let loaded = EmbeddingModel::load(&model_name, None).await?;
                    Ok(Arc::new(loaded) as Arc<dyn Embedder>)
                })
            })
        };

        Self::new_with_loader(
            semantic_home,
            model_name,
            #[cfg(has_embeddings)]
            embedding_loader,
        )
    }

    fn new_with_loader(
        semantic_home: PathBuf,
        model_name: String,
        #[cfg(has_embeddings)] embedding_loader: EmbeddingLoaderFactory,
    ) -> VaultResult<Self> {
        let paths = home::semantic_home_paths(&semantic_home);
        home::ensure_home_layout(&paths)?;

        Ok(Self {
            paths,
            model_name,
            contexts: RwLock::new(HashMap::new()),
            init_locks: tokio::sync::Mutex::new(HashMap::new()),
            #[cfg(has_embeddings)]
            embedding_model: Arc::new(OnceCell::new()),
            #[cfg(has_embeddings)]
            embedding_loader,
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

        let state_dir = self.paths.vaults_dir.join(&vault_id);
        let context = VaultContext::open(
            vault_id.clone(),
            canonical_root,
            self.model_name.clone(),
            state_dir,
            watch_enabled,
            #[cfg(has_embeddings)]
            self.embedding_loader(),
        )
        .await?;
        let context = Arc::new(context);

        let mut guard = self.contexts.write().await;
        guard.insert(vault_id, Arc::clone(&context));
        drop(guard);
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
    fn embedding_loader(&self) -> EmbeddingLoaderFuture {
        let shared_model = Arc::clone(&self.embedding_model);
        let loader = Arc::clone(&self.embedding_loader);
        Box::pin(async move {
            let model = shared_model.get_or_try_init(|| loader()).await?;
            Ok(Arc::clone(model))
        })
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

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use super::*;

    #[cfg(has_embeddings)]
    #[tokio::test]
    async fn concurrent_ensure_returns_one_context_while_shared_loader_is_blocked() {
        let dir = tempfile::tempdir().unwrap();
        let vault_root = dir.path().join("vault");
        std::fs::create_dir_all(vault_root.join(".obsidian")).unwrap();
        std::fs::write(vault_root.join("note.md"), "# Note\n").unwrap();

        let calls = Arc::new(AtomicUsize::new(0));
        let loader_calls = Arc::clone(&calls);
        let loader: EmbeddingLoaderFactory = Arc::new(move || {
            loader_calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(std::future::pending())
        });
        let registry = Arc::new(
            VaultRegistry::new_with_loader(
                dir.path().join("semantic-home"),
                "blocked-test-model".into(),
                loader,
            )
            .unwrap(),
        );

        let first_registry = Arc::clone(&registry);
        let first_root = vault_root.clone();
        let first = tokio::spawn(async move {
            first_registry
                .ensure_vault(&first_root, false, "blocked-test-model")
                .await
        });
        let second_registry = Arc::clone(&registry);
        let second_root = vault_root.clone();
        let second = tokio::spawn(async move {
            second_registry
                .ensure_vault(&second_root, false, "blocked-test-model")
                .await
        });

        let (first, second) = tokio::time::timeout(Duration::from_secs(2), async {
            (
                first.await.unwrap().unwrap(),
                second.await.unwrap().unwrap(),
            )
        })
        .await
        .expect("ensure_vault must not wait for model loading");

        assert!(Arc::ptr_eq(&first, &second));
        assert!(!first.embedding_status().queryable);
        tokio::time::timeout(Duration::from_secs(1), async {
            while calls.load(Ordering::SeqCst) == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("shared loader should start in the background");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}
