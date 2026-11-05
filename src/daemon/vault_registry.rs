//! A live IPC attachment owns one authorized view of a vault.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};

#[cfg(has_embeddings)]
use tokio::sync::OnceCell;

use crate::error::{VaultError, VaultResult};
#[cfg(has_embeddings)]
use crate::vault::embeddings::{CacheFile, Embedder, EmbeddingModel};
use crate::vault::exclude::{ExcludeSet, daemon_exclusions};
use crate::vault::fs::VaultRoot;

use super::home;
use super::protocol::{self, IndexScope};
#[cfg(has_embeddings)]
use super::vault_context::EmbeddingLoaderFuture;
use super::vault_context::VaultContext;

#[cfg(has_embeddings)]
type EmbeddingLoaderFactory = Arc<dyn Fn() -> EmbeddingLoaderFuture + Send + Sync>;

type Roots = Mutex<HashMap<PathBuf, Weak<AttachedVault>>>;

pub struct VaultRegistry {
    roots: Arc<Roots>,
    closed: AtomicBool,
    model_name: String,
    #[cfg(has_embeddings)]
    home: VaultRoot,
    #[cfg(has_embeddings)]
    embedding_model: Arc<OnceCell<Arc<dyn Embedder>>>,
    #[cfg(has_embeddings)]
    embedding_loader: EmbeddingLoaderFactory,
}

struct AttachedVault {
    root: VaultRoot,
    registry: Weak<Roots>,
    // Serialize observations of the floor so an earlier slow read cannot replace a newer policy.
    refresh_floor: tokio::sync::Mutex<()>,
    views: Mutex<VaultViews>,
}

#[derive(Default)]
struct VaultViews {
    replaced: bool,
    floor: Option<Arc<ExcludeSet>>,
    scopes: HashMap<String, LiveScope>,
}

#[derive(Default)]
struct LiveScope {
    initialization: Arc<tokio::sync::Mutex<()>>,
    context: Option<Arc<VaultContext>>,
    // Each connection owns its own watch preference.
    holders: HashMap<String, bool>,
}

struct ScopeReservation {
    exclude: Arc<ExcludeSet>,
    grant: Option<VaultAttachment>,
    initialization: Arc<tokio::sync::Mutex<()>>,
}

impl LiveScope {
    fn watching(&self) -> bool {
        self.holders.values().any(|watch| *watch)
    }
}

impl VaultViews {
    fn retire_all(&mut self) {
        for scope in self.scopes.drain().map(|(_, scope)| scope) {
            if let Some(context) = scope.context {
                context.retire();
            }
        }
        self.floor = None;
    }

    fn by_holder(&self, id: &str) -> Option<&LiveScope> {
        self.scopes
            .values()
            .find(|scope| scope.holders.contains_key(id))
    }
}

impl AttachedVault {
    /// Load and apply a floor in observation order; only the short registry update holds the view lock.
    async fn reserve(
        self: &Arc<Self>,
        descriptor: &IndexScope,
        watch: bool,
        current: Option<&VaultAttachment>,
        closed: &AtomicBool,
    ) -> VaultResult<ScopeReservation> {
        let _refresh = self.refresh_floor.lock().await;
        let root = self.root.clone();
        let rules = descriptor.exclude_patterns.clone();
        let policy = tokio::task::spawn_blocking(move || {
            let floor = daemon_exclusions(&root)?;
            let effective = floor.merged(&rules).map(|effective| {
                let id = effective.scope_id(root.path());
                (Arc::new(effective), id)
            });
            Ok((floor, effective))
        })
        .await
        .map_err(|err| VaultError::Other(format!("daemon exclusion loader failed: {err}")))
        .and_then(|result| result);

        let mut views = self.views.lock().unwrap_or_else(|e| e.into_inner());
        if views.replaced || closed.load(Ordering::Acquire) {
            return Err(unattached());
        }
        let (floor, effective) = match policy {
            Ok(policy) => policy,
            Err(error) => {
                views.retire_all();
                return Err(error);
            }
        };
        if views
            .floor
            .as_ref()
            .is_none_or(|previous| previous.patterns() != floor.patterns())
        {
            views.retire_all();
            views.floor = Some(Arc::new(floor));
        }
        let (exclude, scope_id) = effective?;
        if let Some(holder) = current
            && let Some(scope) = views.by_holder(&holder.id)
        {
            if holder.scope_id != scope_id || scope.holders.get(&holder.id) != Some(&watch) {
                return Err(VaultError::InvalidPath(
                    "an attachment connection retains its indexing scope and watch preference"
                        .into(),
                ));
            }
            return Ok(ScopeReservation {
                exclude,
                grant: None,
                initialization: Arc::clone(&scope.initialization),
            });
        }

        let mut random = [0u8; 16];
        getrandom::fill(&mut random).map_err(|error| {
            VaultError::Other(format!("generate attachment identifier: {error}"))
        })?;
        let id = format!("{:032x}", u128::from_ne_bytes(random));
        let scope = views.scopes.entry(scope_id.clone()).or_default();
        scope.holders.insert(id.clone(), watch);
        let attachment = VaultAttachment {
            vault: Arc::clone(self),
            scope_id,
            id,
        };
        Ok(ScopeReservation {
            exclude,
            grant: Some(attachment),
            initialization: Arc::clone(&scope.initialization),
        })
    }
}

impl Drop for AttachedVault {
    fn drop(&mut self) {
        if let Some(registry) = self.registry.upgrade() {
            let mut roots = registry.lock().unwrap_or_else(|e| e.into_inner());
            if roots
                .get(self.root.path())
                .is_some_and(|entry| std::ptr::eq(entry.as_ptr(), self))
            {
                roots.remove(self.root.path());
            }
        }
    }
}

/// Connection-owned authority. Queries borrow the context and cannot prolong this grant.
pub(crate) struct VaultAttachment {
    vault: Arc<AttachedVault>,
    pub(crate) scope_id: String,
    pub(crate) id: String,
}

impl Drop for VaultAttachment {
    fn drop(&mut self) {
        let mut views = self.vault.views.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(scope) = views.scopes.get_mut(&self.scope_id) {
            if scope.holders.remove(&self.id).is_none() {
                return;
            }
            if scope.holders.is_empty() {
                if let Some(context) = &scope.context {
                    context.retire();
                }
                views.scopes.remove(&self.scope_id);
            } else if !scope.watching()
                && let Some(context) = &scope.context
            {
                context.stop_watcher();
            }
        }
    }
}

impl VaultRegistry {
    pub fn new(semantic_home: PathBuf, model_name: String) -> VaultResult<Self> {
        #[cfg(has_embeddings)]
        let embedding_loader: EmbeddingLoaderFactory = {
            let provider =
                crate::config::EmbeddingProvider::from_env().map_err(VaultError::Embedding)?;
            let model_name = model_name.clone();
            Arc::new(move || {
                let model_name = model_name.clone();
                Box::pin(async move {
                    let loaded = EmbeddingModel::load(&model_name, provider).await?;
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
        let home = home::open_home_layout(&home::semantic_home_paths(&semantic_home))?;
        #[cfg(not(has_embeddings))]
        drop(home);
        Ok(Self {
            roots: Arc::new(Mutex::new(HashMap::new())),
            closed: AtomicBool::new(false),
            model_name,
            #[cfg(has_embeddings)]
            home,
            #[cfg(has_embeddings)]
            embedding_model: Arc::new(OnceCell::new()),
            #[cfg(has_embeddings)]
            embedding_loader,
        })
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    pub(crate) fn close(&self) {
        self.closed.store(true, Ordering::Release);
        let vaults: Vec<_> = self
            .roots
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .filter_map(Weak::upgrade)
            .collect();
        for vault in vaults {
            vault
                .views
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .retire_all();
        }
    }

    /// Returns the context plus a new grant only when the connection needs to retain one.
    pub(crate) async fn attach(
        &self,
        vault_root: &Path,
        watch: bool,
        requested_model: &str,
        descriptor: &IndexScope,
        current: Option<&VaultAttachment>,
    ) -> VaultResult<(Arc<VaultContext>, Option<VaultAttachment>)> {
        descriptor.validate().map_err(VaultError::InvalidPath)?;
        if requested_model != self.model_name {
            return Err(VaultError::InvalidPath(format!(
                "requested model '{requested_model}' does not match daemon model '{}'",
                self.model_name
            )));
        }
        let vault = self.open_root(vault_root)?;
        if current.is_some_and(|active| active.vault.root.path() != vault.root.path()) {
            return Err(VaultError::InvalidPath(
                "an attachment connection belongs to one vault".into(),
            ));
        }
        let reservation = vault
            .reserve(descriptor, watch, current, &self.closed)
            .await?;

        let holder = reservation
            .grant
            .as_ref()
            .or(current)
            .ok_or_else(unattached)?;
        let _init = reservation.initialization.lock().await;
        let (context, wants_watcher) = {
            let views = vault.views.lock().unwrap_or_else(|e| e.into_inner());
            let scope = views.by_holder(&holder.id).ok_or_else(unattached)?;
            (scope.context.clone(), scope.watching())
        };
        let mut pending = None;
        let context = if let Some(context) = context {
            if wants_watcher && !context.watch_enabled() {
                pending = Some(context.resume_watcher().await?);
            }
            context
        } else {
            let vault_id = home::vault_id_from_canonical(vault.root.path());
            #[cfg(has_embeddings)]
            let cache = CacheFile::new(
                self.home.clone(),
                PathBuf::from("vaults")
                    .join(&vault_id)
                    .join("scopes")
                    .join(&holder.scope_id)
                    .join("embeddings.bin"),
            );
            Arc::new(
                VaultContext::open(
                    vault_id,
                    vault.root.clone(),
                    self.model_name.clone(),
                    reservation.exclude,
                    wants_watcher,
                    #[cfg(has_embeddings)]
                    cache,
                    #[cfg(has_embeddings)]
                    self.embedding_loader(),
                )
                .await?,
            )
        };
        {
            let mut views = vault.views.lock().unwrap_or_else(|e| e.into_inner());
            let scope = views
                .scopes
                .get_mut(&holder.scope_id)
                .filter(|scope| scope.holders.contains_key(&holder.id))
                .ok_or_else(unattached)?;
            if scope.watching() {
                if let Some(pending) = pending {
                    context.install_watcher(pending);
                }
            } else {
                context.stop_watcher();
            }
            scope.context = Some(Arc::clone(&context));
            context.activate();
        }
        Ok((context, reservation.grant))
    }

    fn open_root(&self, path: &Path) -> VaultResult<Arc<AttachedVault>> {
        self.lookup_root(path, true)?.ok_or_else(unattached)
    }

    fn lookup_root(&self, path: &Path, create: bool) -> VaultResult<Option<Arc<AttachedVault>>> {
        let root = open_requested_root(path)?;
        // Dropping the last reference also cleans this table, after its lock is released.
        let previous;
        let mut roots = self.roots.lock().unwrap_or_else(|e| e.into_inner());
        if self.closed.load(Ordering::Acquire) {
            return Ok(None);
        }
        previous = roots.get(root.path()).and_then(Weak::upgrade);
        if let Some(vault) = &previous
            && root.same_root(&vault.root)?
        {
            return Ok(Some(Arc::clone(vault)));
        }
        if previous.is_none() && !create {
            return Ok(None);
        }
        // Another lookup may have enrolled a replacement after our first open.
        // A delayed request cannot retire the now-current directory or revive its predecessor.
        let current = open_requested_root(path)?;
        if current.path() != root.path() || !current.same_root(&root)? {
            return Ok(None);
        }
        if let Some(vault) = &previous {
            let mut views = vault.views.lock().unwrap_or_else(|e| e.into_inner());
            views.replaced = true;
            views.retire_all();
            roots.remove(root.path());
        }
        if !create {
            return Ok(None);
        }
        let vault = Arc::new(AttachedVault {
            root,
            registry: Arc::downgrade(&self.roots),
            refresh_floor: tokio::sync::Mutex::new(()),
            views: Mutex::new(VaultViews::default()),
        });
        roots.insert(vault.root.path().to_owned(), Arc::downgrade(&vault));
        Ok(Some(vault))
    }

    pub(crate) fn context_for_holder(
        &self,
        path: &Path,
        holder: &str,
    ) -> VaultResult<Option<Arc<VaultContext>>> {
        let Some(vault) = self.find_root(path)? else {
            return Ok(None);
        };
        let views = vault.views.lock().unwrap_or_else(|e| e.into_inner());
        Ok(views
            .by_holder(holder)
            .and_then(|scope| scope.context.clone()))
    }

    pub(crate) async fn context_for_policy(
        &self,
        path: &Path,
        descriptor: IndexScope,
    ) -> VaultResult<Option<Arc<VaultContext>>> {
        descriptor.validate().map_err(VaultError::InvalidPath)?;
        let vault = self.find_root(path)?;
        tokio::task::spawn_blocking(move || {
            let floor = vault.as_ref().and_then(|v| {
                v.views
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .floor
                    .clone()
            });
            let effective = match &floor {
                Some(floor) => floor.merged(&descriptor.exclude_patterns),
                None => ExcludeSet::build(descriptor.exclude_patterns),
            }?;
            let Some((vault, floor)) = vault.zip(floor) else {
                return Ok(None);
            };
            let id = effective.scope_id(vault.root.path());
            let views = vault.views.lock().unwrap_or_else(|e| e.into_inner());
            if !views
                .floor
                .as_ref()
                .is_some_and(|current| Arc::ptr_eq(current, &floor))
            {
                return Ok(None);
            }
            Ok(views
                .scopes
                .get(&id)
                .and_then(|scope| scope.context.clone()))
        })
        .await
        .map_err(|err| VaultError::Other(format!("daemon status policy worker failed: {err}")))?
    }

    fn find_root(&self, path: &Path) -> VaultResult<Option<Arc<AttachedVault>>> {
        self.lookup_root(path, false)
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

fn unattached() -> VaultError {
    VaultError::DaemonRpc {
        code: protocol::ERR_VAULT_NOT_ATTACHED,
        message: "vault attachment expired; attach again".into(),
        data: None,
    }
}

fn open_requested_root(vault_root: &Path) -> VaultResult<VaultRoot> {
    if !vault_root.is_absolute() {
        return Err(VaultError::InvalidPath(format!(
            "vault_root must be absolute: {}",
            vault_root.display()
        )));
    }

    VaultRoot::open(vault_root).map_err(|err| {
        VaultError::InvalidPath(format!(
            "failed to open vault root '{}': {err}",
            vault_root.display()
        ))
    })
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

    const TEST_DEADLINE: Duration = Duration::from_secs(20);

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
                    tokenizer_fingerprint: [1; 32],
                    max_input_tokens: 8192,
                    chunking_version: crate::vault::embedding_input::CHUNKING_VERSION,
                },
            }
        }
    }

    impl Embedder for FakeEmbedder {
        fn input_policy(&self) -> &crate::vault::embedding_input::EmbeddingInputPolicy {
            &crate::vault::embedding_input::EmbeddingInputPolicy {
                max_input_tokens: 8192,
                tokenizer_fingerprint: [1; 32],
            }
        }

        fn count_tokens(&self, text: &str) -> VaultResult<usize> {
            Ok(text.chars().count())
        }

        fn request_limits(&self) -> crate::vault::embedding_input::EmbeddingRequestLimits {
            crate::vault::embedding_input::EmbeddingRequestLimits {
                max_inputs: 64,
                max_total_tokens: 8192 * 64,
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
            index_scope: IndexScope::default(),
        }
    }

    #[tokio::test]
    async fn passive_status_does_not_start_a_loader_or_attach_a_vault() {
        let dir = tempfile::tempdir().unwrap();
        let vault_root = dir.path().join("vault");
        create_vault(&vault_root);
        let calls = Arc::new(AtomicUsize::new(0));
        let loader_calls = Arc::clone(&calls);
        let loader: EmbeddingLoaderFactory = Arc::new(move || {
            loader_calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(std::future::pending())
        });
        let registry = VaultRegistry::new_with_loader(
            dir.path().join("semantic-home"),
            "blocked-test-model".into(),
            loader,
        )
        .unwrap();
        let params = || super::super::protocol::VaultStatusParams {
            vault_root: vault_root.display().to_string(),
            index_scope: Some(IndexScope::default()),
            attachment_id: None,
        };
        for _ in 0..2 {
            let error = query::vault_status(&registry, params()).await.unwrap_err();
            assert_eq!(error.code, super::super::protocol::ERR_VAULT_NOT_ATTACHED);
        }
        assert!(registry.roots.lock().unwrap().is_empty());
        assert_eq!(calls.load(Ordering::SeqCst), 0);

        let mut holder = None;
        query::ensure_vault(&registry, ensure_params(&vault_root), &mut holder)
            .await
            .unwrap();
        let status = query::vault_status(&registry, params()).await.unwrap();
        assert_eq!(status.phase, SemanticPhase::Warming);
        assert!(!status.ready);
        assert_eq!(status.total_notes, 1);
        assert_eq!(status.pending_notes, 1);
        assert_eq!(registry.roots.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn concurrent_ensure_returns_one_context_while_shared_loader_is_blocked() {
        let dir = tempfile::tempdir().unwrap();
        let vault_root = dir.path().join("vault");
        create_vault(&vault_root);

        let calls = Arc::new(AtomicUsize::new(0));
        let loader_calls = Arc::clone(&calls);
        let started = Arc::new(tokio::sync::Notify::new());
        let signal = Arc::clone(&started);
        let loader: EmbeddingLoaderFactory = Arc::new(move || {
            loader_calls.fetch_add(1, Ordering::SeqCst);
            signal.notify_one();
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
                .attach(
                    &first_root,
                    false,
                    "blocked-test-model",
                    &IndexScope::default(),
                    None,
                )
                .await
        });
        let second_registry = Arc::clone(&registry);
        let second_root = vault_root.clone();
        let second = tokio::spawn(async move {
            second_registry
                .attach(
                    &second_root,
                    false,
                    "blocked-test-model",
                    &IndexScope::default(),
                    None,
                )
                .await
        });

        let (first, second) = tokio::time::timeout(TEST_DEADLINE, async {
            (
                first.await.unwrap().unwrap(),
                second.await.unwrap().unwrap(),
            )
        })
        .await
        .expect("ensure_vault must not wait for model loading");

        assert!(Arc::ptr_eq(&first.0, &second.0));
        assert_ne!(first.1.as_ref().unwrap().id, second.1.as_ref().unwrap().id);
        assert!(!first.0.embedding_status().queryable);
        tokio::time::timeout(TEST_DEADLINE, started.notified())
            .await
            .expect("shared loader should start in the background");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn watch_and_provider_ownership_follow_live_connections() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("vault");
        create_vault(&root);
        let loader: EmbeddingLoaderFactory = Arc::new(|| Box::pin(std::future::pending()));
        let registry =
            VaultRegistry::new_with_loader(dir.path().join("state"), "test".into(), loader)
                .unwrap();
        let scope = IndexScope::new(&["Private/".to_owned()]);
        let equivalent = IndexScope::new(&["Private/**".to_owned(), " Private/ ".to_owned()]);

        let (unwatched, first) = registry
            .attach(&root, false, "test", &scope, None)
            .await
            .unwrap();
        let first = first.unwrap();
        assert!(!unwatched.watch_enabled());
        let (watched, second) = registry
            .attach(&root, true, "test", &equivalent, None)
            .await
            .unwrap();
        let second = second.unwrap();
        assert!(Arc::ptr_eq(&unwatched, &watched));
        assert_eq!(first.scope_id, second.scope_id);
        assert_ne!(first.id, second.id);
        assert!(watched.watch_enabled());
        let (_, repeated) = registry
            .attach(&root, true, "test", &scope, Some(&second))
            .await
            .unwrap();
        assert!(repeated.is_none());

        let second_id = second.id.clone();
        drop(second);
        assert!(!watched.watch_enabled());
        assert!(
            registry
                .context_for_holder(&root, &second_id)
                .unwrap()
                .is_none()
        );
        assert!(
            registry
                .context_for_holder(&root, &first.id)
                .unwrap()
                .is_some()
        );
        let first_id = first.id.clone();
        drop(first);
        assert!(
            registry
                .context_for_holder(&root, &first_id)
                .unwrap()
                .is_none()
        );
        assert!(registry.roots.lock().unwrap().is_empty());

        let (fresh, grant) = registry
            .attach(&root, false, "test", &scope, None)
            .await
            .unwrap();
        assert!(!Arc::ptr_eq(&fresh, &unwatched));
        assert_ne!(grant.as_ref().unwrap().id, first_id);
        assert!(
            registry
                .context_for_holder(&root, &first_id)
                .unwrap()
                .is_none()
        );
        registry.close();
        assert!(
            registry
                .context_for_policy(&root, scope.clone())
                .await
                .unwrap()
                .is_none()
        );
        assert!(
            registry
                .attach(&root, false, "test", &scope, None)
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn portable_policy_changes_revoke_every_old_holder_including_equal_effective_views() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("vault");
        create_vault(&root);
        std::fs::create_dir(root.join("Private")).unwrap();
        std::fs::write(root.join("Private/secret.md"), "private").unwrap();
        std::fs::create_dir(root.join(".obsidian-mcp")).unwrap();
        let ignore = root.join(".obsidian-mcp/ignore");
        let loader: EmbeddingLoaderFactory = Arc::new(|| Box::pin(std::future::pending()));
        let registry =
            VaultRegistry::new_with_loader(dir.path().join("state"), "test".into(), loader)
                .unwrap();
        let empty = IndexScope::default();
        let restricted = IndexScope::new(&["Private/".into()]);
        let (broad, first) = registry
            .attach(&root, false, "test", &empty, None)
            .await
            .unwrap();
        let (narrow, second) = registry
            .attach(&root, false, "test", &restricted, None)
            .await
            .unwrap();
        let first = first.unwrap();
        let second = second.unwrap();
        assert_eq!(broad.embedding_status().total_notes, 2);
        assert_eq!(narrow.embedding_status().total_notes, 1);

        std::fs::write(&ignore, "Private/\n").unwrap();
        let (updated, third) = registry
            .attach(&root, false, "test", &empty, Some(&first))
            .await
            .unwrap();
        let third = third.unwrap();
        assert_eq!(updated.embedding_status().total_notes, 1);
        assert_eq!(third.scope_id, second.scope_id);
        assert!(!Arc::ptr_eq(&updated, &narrow));
        assert!(
            registry
                .context_for_holder(&root, &first.id)
                .unwrap()
                .is_none()
        );
        assert!(
            registry
                .context_for_holder(&root, &second.id)
                .unwrap()
                .is_none()
        );
        // Dropping an old equal-policy grant must never withdraw its successor.
        drop(second);
        assert!(
            registry
                .context_for_holder(&root, &third.id)
                .unwrap()
                .is_some()
        );

        std::fs::write(&ignore, "[unclosed\n").unwrap();
        assert!(
            registry
                .attach(&root, false, "test", &empty, None)
                .await
                .is_err()
        );
        assert!(
            registry
                .context_for_holder(&root, &third.id)
                .unwrap()
                .is_none()
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn replacing_the_directory_at_a_vault_path_retires_the_displaced_view() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("vault");
        create_vault(&root);
        let registry = VaultRegistry::new_with_loader(
            directory.path().join("state"),
            "test".into(),
            Arc::new(|| Box::pin(std::future::pending())),
        )
        .unwrap();
        let descriptor = IndexScope::default();
        let (original, old) = registry
            .attach(&root, false, "test", &descriptor, None)
            .await
            .unwrap();
        let old = old.unwrap();
        std::fs::rename(&root, directory.path().join("previous")).unwrap();
        std::fs::create_dir_all(root.join(".obsidian-mcp")).unwrap();
        std::fs::write(root.join(".obsidian-mcp/ignore"), "private.md\n").unwrap();
        std::fs::write(root.join("private.md"), "hidden in replacement").unwrap();
        std::fs::write(root.join("current.md"), "new source").unwrap();

        let (current, new) = registry
            .attach(&root, false, "test", &descriptor, None)
            .await
            .unwrap();
        let new = new.unwrap();
        assert!(!Arc::ptr_eq(&original, &current));
        assert_ne!(old.scope_id, new.scope_id);
        assert!(
            registry
                .context_for_holder(&root, &old.id)
                .unwrap()
                .is_none()
        );
        assert_eq!(current.embedding_status().total_notes, 1);
        assert_eq!(
            current.read_note(Path::new("current.md")).unwrap(),
            "new source"
        );
        assert!(current.read_note(Path::new("note.md")).is_err());
        assert!(
            old.vault
                .reserve(&descriptor, false, None, &registry.closed)
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn connection_eof_during_initialization_releases_its_provisional_holder() {
        exercise_initializing_disconnect(false).await;
    }

    #[tokio::test]
    async fn equivalent_waiting_holder_survives_an_initializing_peer_disconnect() {
        exercise_initializing_disconnect(true).await;
    }

    async fn exercise_initializing_disconnect(keep_another_holder: bool) {
        use tokio::io::AsyncWriteExt;

        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("vault");
        create_vault(&root);
        let calls = Arc::new(AtomicUsize::new(0));
        let attempts = Arc::clone(&calls);
        let started = Arc::new(tokio::sync::Notify::new());
        let signal = Arc::clone(&started);
        let loader: EmbeddingLoaderFactory = Arc::new(move || {
            attempts.fetch_add(1, Ordering::SeqCst);
            signal.notify_one();
            Box::pin(std::future::pending())
        });
        let registry = Arc::new(
            VaultRegistry::new_with_loader(dir.path().join("state"), "test".into(), loader)
                .unwrap(),
        );
        let vault = registry.open_root(&root).unwrap();
        let floor = Arc::new(daemon_exclusions(&vault.root).unwrap());
        let id = floor.scope_id(vault.root.path());
        let gate = {
            let mut views = vault.views.lock().unwrap();
            views.floor = Some(floor);
            Arc::clone(&views.scopes.entry(id.clone()).or_default().initialization)
        };
        let blocked = gate.lock().await;
        let (mut peer, server) = tokio::io::duplex(4096);
        let task = tokio::spawn(crate::daemon::server::serve_test_connection(
            server,
            dir.path().join("state"),
            Arc::clone(&registry),
        ));
        let request = serde_json::json!({
            "jsonrpc": "2.0", "id": 1, "method": "ensure_vault",
            "params": {"vault_root":root, "watch":true, "index_scope": IndexScope::default()}
        });
        peer.write_all(format!("{request}\n").as_bytes())
            .await
            .unwrap();
        let remaining = keep_another_holder.then(|| {
            let registry = Arc::clone(&registry);
            let root = root.clone();
            tokio::spawn(async move {
                registry
                    .attach(&root, false, "test", &IndexScope::default(), None)
                    .await
            })
        });
        let expected = if remaining.is_some() { 2 } else { 1 };
        tokio::time::timeout(TEST_DEADLINE, async {
            while vault.views.lock().unwrap().scopes[&id].holders.len() != expected {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        drop(peer);
        tokio::time::timeout(TEST_DEADLINE, task)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        if let Some(remaining) = remaining {
            assert_eq!(vault.views.lock().unwrap().scopes[&id].holders.len(), 1);
            drop(blocked);
            let (context, grant) = tokio::time::timeout(TEST_DEADLINE, remaining)
                .await
                .unwrap()
                .unwrap()
                .unwrap();
            assert!(grant.is_some());
            assert!(!context.watch_enabled());
            tokio::time::timeout(TEST_DEADLINE, started.notified())
                .await
                .unwrap();
            assert_eq!(calls.load(Ordering::SeqCst), 1);
        } else {
            assert!(vault.views.lock().unwrap().scopes.is_empty());
        }
    }

    #[tokio::test]
    async fn ensure_reports_warming_and_search_returns_structured_not_ready() {
        let dir = tempfile::tempdir().unwrap();
        let vault_root = dir.path().join("vault");
        create_vault(&vault_root);
        let started = Arc::new(tokio::sync::Notify::new());
        let signal = Arc::clone(&started);
        let loader: EmbeddingLoaderFactory = Arc::new(move || {
            signal.notify_one();
            Box::pin(std::future::pending())
        });
        let registry = VaultRegistry::new_with_loader(
            dir.path().join("semantic-home"),
            "blocked-test-model".into(),
            loader,
        )
        .unwrap();

        let mut holder = None;
        let ensured = tokio::time::timeout(
            TEST_DEADLINE,
            query::ensure_vault(&registry, ensure_params(&vault_root), &mut holder),
        )
        .await
        .expect("ensure must not wait for the blocked model loader")
        .unwrap();
        tokio::time::timeout(TEST_DEADLINE, started.notified())
            .await
            .expect("the holder should start the blocked loader");
        assert!(!ensured.ready);
        assert_eq!(ensured.phase, Some(SemanticPhase::Warming));
        assert_eq!(ensured.total_notes, Some(1));
        assert_eq!(ensured.pending_notes, Some(1));

        let error = query::search_semantic(
            &registry,
            SearchSemanticParams {
                vault_root: vault_root.display().to_string(),
                attachment_id: holder.as_ref().unwrap().id.clone(),
                query: "semantic".into(),
                top_k: Some(10),
                include_content: Some(false),
                allowed_paths: None,
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
    async fn ensure_tracks_ready_and_reports_unavailable_deleted_source() {
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

        let mut holder = None;
        let ready = tokio::time::timeout(TEST_DEADLINE, async {
            loop {
                let result =
                    query::ensure_vault(&registry, ensure_params(&vault_root), &mut holder)
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
                attachment_id: holder.as_ref().unwrap().id.clone(),
                query: "semantic".into(),
                top_k: Some(10),
                include_content: Some(false),
                allowed_paths: None,
            },
        )
        .await
        .unwrap();
        assert_eq!(result.results.len(), 1);
        let hit = &result.results[0];
        assert_eq!(hit.path, "note.md");
        assert_eq!(hit.source_status, crate::models::SourceStatus::Unavailable);
        assert_eq!(
            hit.context_error.as_ref().unwrap().code,
            "source_unavailable"
        );
        assert!(hit.snippet.is_none());
        assert!(hit.content.is_none());
        assert!(hit.preview.is_none());
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

        let mut holder = None;
        let degraded = tokio::time::timeout(TEST_DEADLINE, async {
            loop {
                let result =
                    query::ensure_vault(&registry, ensure_params(&vault_root), &mut holder)
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
                attachment_id: holder.as_ref().unwrap().id.clone(),
                query: "semantic".into(),
                top_k: Some(10),
                include_content: Some(false),
                allowed_paths: None,
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
