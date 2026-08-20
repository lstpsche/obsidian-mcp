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

#[cfg(all(test, has_embeddings))]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use super::*;
    use crate::daemon::protocol::{EnsureVaultParams, SearchSemanticParams, SemanticPhase};
    use crate::daemon::query;
    use crate::vault::embeddings::{
        EMBEDDING_INPUT_VERSION, Embedder, EmbeddingBackendKind, EmbeddingSpaceIdentity,
    };

    struct FakeEmbedder {
        identity: EmbeddingSpaceIdentity,
    }

    impl FakeEmbedder {
        fn new() -> Self {
            Self {
                identity: EmbeddingSpaceIdentity {
                    backend: EmbeddingBackendKind::Local,
                    model: "daemon-test-model".into(),
                    endpoint_fingerprint: None,
                    dimension: 3,
                    input_version: EMBEDDING_INPUT_VERSION,
                },
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
            Ok(texts
                .iter()
                .map(|text| vec![text.len() as f32, 1.0, 0.0])
                .collect())
        }
    }

    fn create_vault(root: &Path) {
        std::fs::create_dir_all(root.join(".obsidian")).unwrap();
        std::fs::write(root.join("note.md"), "# Note\nsemantic content\n").unwrap();
    }

    fn ensure_params(vault_root: &Path) -> EnsureVaultParams {
        EnsureVaultParams {
            vault_root: vault_root.display().to_string(),
            watch: Some(false),
            model_name: Some("blocked-test-model".into()),
        }
    }

    #[tokio::test]
    async fn concurrent_ensure_returns_one_context_while_shared_loader_is_blocked() {
        let dir = tempfile::tempdir().unwrap();
        let vault_root = dir.path().join("vault");
        create_vault(&vault_root);

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

    #[tokio::test]
    async fn ensure_reports_warming_and_search_returns_structured_not_ready() {
        let dir = tempfile::tempdir().unwrap();
        let vault_root = dir.path().join("vault");
        create_vault(&vault_root);
        let loader: EmbeddingLoaderFactory = Arc::new(|| Box::pin(std::future::pending()));
        let registry = VaultRegistry::new_with_loader(
            dir.path().join("semantic-home"),
            "blocked-test-model".into(),
            loader,
        )
        .unwrap();

        let ensured = tokio::time::timeout(
            Duration::from_secs(2),
            query::ensure_vault(&registry, ensure_params(&vault_root)),
        )
        .await
        .expect("ensure must not wait for the blocked model loader")
        .unwrap();
        assert!(!ensured.ready);
        assert_eq!(ensured.phase, Some(SemanticPhase::Warming));
        assert_eq!(ensured.total_notes, Some(1));

        let error = query::search_semantic(
            &registry,
            SearchSemanticParams {
                vault_root: vault_root.display().to_string(),
                query: "semantic".into(),
                top_k: Some(10),
                include_content: Some(false),
            },
        )
        .await
        .expect_err("warming semantic search must not masquerade as empty success");
        assert_eq!(error.code, crate::daemon::protocol::ERR_VAULT_NOT_READY);
        assert!(error.message.contains("warming"));
        let data = error.data.expect("status data should be attached");
        assert_eq!(data["phase"], "warming");
        assert_eq!(data["ready"], false);
        assert_eq!(data["total_notes"], 1);
    }

    #[tokio::test]
    async fn ensure_tracks_ready_and_direct_queries_omit_missing_files() {
        let dir = tempfile::tempdir().unwrap();
        let vault_root = dir.path().join("vault");
        create_vault(&vault_root);
        let loader: EmbeddingLoaderFactory =
            Arc::new(|| Box::pin(async { Ok(Arc::new(FakeEmbedder::new()) as Arc<dyn Embedder>) }));
        let registry = VaultRegistry::new_with_loader(
            dir.path().join("semantic-home"),
            "blocked-test-model".into(),
            loader,
        )
        .unwrap();

        let ready = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let result = query::ensure_vault(&registry, ensure_params(&vault_root))
                    .await
                    .unwrap();
                if result.ready {
                    break result;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("semantic runtime should become ready");
        assert_eq!(ready.phase, Some(SemanticPhase::Ready));
        assert_eq!(ready.indexed_notes, Some(1));
        assert_eq!(ready.total_notes, Some(1));
        assert_eq!(ready.pending_notes, Some(0));

        std::fs::remove_file(vault_root.join("note.md")).unwrap();
        let result = query::search_semantic(
            &registry,
            SearchSemanticParams {
                vault_root: vault_root.display().to_string(),
                query: "semantic".into(),
                top_k: Some(10),
                include_content: Some(false),
            },
        )
        .await
        .unwrap();
        assert!(
            result.results.is_empty(),
            "a stale vector must not produce a ghost hit for a missing note"
        );
    }

    #[tokio::test]
    async fn ensure_reports_degraded_loader_failure_and_search_status() {
        let dir = tempfile::tempdir().unwrap();
        let vault_root = dir.path().join("vault");
        create_vault(&vault_root);
        let loader: EmbeddingLoaderFactory = Arc::new(|| {
            Box::pin(async { Err(VaultError::Embedding("injected safe loader failure".into())) })
        });
        let registry = VaultRegistry::new_with_loader(
            dir.path().join("semantic-home"),
            "blocked-test-model".into(),
            loader,
        )
        .unwrap();

        let degraded = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let result = query::ensure_vault(&registry, ensure_params(&vault_root))
                    .await
                    .unwrap();
                if result.phase == Some(SemanticPhase::Degraded) {
                    break result;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("loader failure should become visible as degraded status");
        assert!(!degraded.ready);
        assert_eq!(
            degraded.last_error.as_deref(),
            Some("Embedding error: injected safe loader failure")
        );

        let error = query::search_semantic(
            &registry,
            SearchSemanticParams {
                vault_root: vault_root.display().to_string(),
                query: "semantic".into(),
                top_k: Some(10),
                include_content: Some(false),
            },
        )
        .await
        .expect_err("an unqueryable degraded runtime must reject semantic search");
        assert_eq!(error.code, crate::daemon::protocol::ERR_VAULT_NOT_READY);
        assert!(error.message.contains("degraded"));
        let data = error.data.expect("degraded status should be attached");
        assert_eq!(data["phase"], "degraded");
        assert_eq!(data["ready"], false);
        assert_eq!(
            data["last_error"],
            "Embedding error: injected safe loader failure"
        );
    }
}
