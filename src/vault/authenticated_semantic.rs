//! Service-owned semantic corpora for explicitly activated readable policies.

use std::collections::{HashMap, HashSet};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::Duration;

use sha2::{Digest, Sha256};
use tokio::sync::{Notify, OnceCell};
use tokio::task::JoinHandle;

use crate::access::clients::{Action, SourceIdentity};
use crate::access::store::{AccessPermit, AccessStore, AuthorizedClient};
use crate::error::{VaultError, VaultResult};

use super::authorized_index::AuthorizedProjection;
use super::embedding_runtime::{EmbeddingAuthorization, EmbeddingRuntime, PendingKind};
use super::embeddings::{CacheFile, Embedder, EmbeddingModel};
use super::fs::VaultRoot;
use super::{Vault, VaultInner, source, storage::Storage};

const RECONCILIATION_INTERVAL: Duration = Duration::from_millis(250);
const CACHE_ENTRIES_PER_PASS: usize = 64;

type CacheCursor = Box<dyn Iterator<Item = std::io::Result<(OsString, bool)>> + Send>;

struct CacheSweep {
    cursor: Option<CacheCursor>,
    requested: bool,
}

impl Default for CacheSweep {
    fn default() -> Self {
        Self {
            cursor: None,
            requested: true,
        }
    }
}

#[derive(Clone)]
struct ActivatedClient {
    store: AccessStore,
    client_id: String,
    read_policy: String,
}

impl ActivatedClient {
    fn from_client(client: &AuthorizedClient) -> Self {
        Self {
            store: client.store().clone(),
            client_id: client.client_id().to_owned(),
            read_policy: client.read_policy_key().to_owned(),
        }
    }

    fn permit(&self) -> VaultResult<AccessPermit> {
        self.store
            .ingestion_permit(&self.client_id, &self.read_policy)
            .map_err(Into::into)
    }
}

struct SemanticEntry {
    source: Weak<VaultInner>,
    root: VaultRoot,
    projection: Arc<AuthorizedProjection>,
    holders: Mutex<HashMap<String, ActivatedClient>>,
    runtime: OnceLock<EmbeddingRuntime>,
    revisions: Mutex<HashMap<OsString, String>>,
    last_error: Mutex<Option<String>>,
    diagnostic_id: String,
}

impl SemanticEntry {
    fn holder(&self) -> VaultResult<(ActivatedClient, AccessPermit)> {
        let holders = self
            .holders
            .lock()
            .map_err(|_| authorization_error("semantic holder state is unavailable"))?
            .values()
            .cloned()
            .collect::<Vec<_>>();
        let mut failure = None;
        for holder in holders {
            match holder.permit() {
                Ok(permit) => return Ok((holder, permit)),
                Err(error) => failure = Some(error),
            }
        }
        Err(failure.unwrap_or_else(|| authorization_error("semantic policy was withdrawn")))
    }

    fn admit(&self, path: Option<&Path>) -> VaultResult<Box<dyn Send>> {
        let (_, permit) = self.holder()?;
        if !self.projection.is_ready() {
            return Err(VaultError::EmbeddingRequest {
                message: "authorized source projection is rebuilding".into(),
                retryable: true,
                retry_after: None,
            });
        }
        if let Some(path) = path {
            permit.check(Action::Read, path, path)?;
            drop(permit);
            let actual = self.root.resolve_existing(path)?.relative;
            self.root.validate_access_file(&actual)?;
            let (_, permit) = self.holder()?;
            permit.check(Action::Read, path, &actual)?;
            return Ok(Box::new(permit));
        }
        Ok(Box::new(permit))
    }

    fn reconcile(&self, retry_unpublished: bool) -> VaultResult<()> {
        let (holder, permit) = self.holder()?;
        let policy = permit.policy().clone();
        drop(permit);
        let source = self
            .source
            .upgrade()
            .ok_or_else(|| authorization_error("semantic source owner has stopped"))?;
        let config = source
            .authenticated_config
            .as_ref()
            .ok_or_else(|| authorization_error("authenticated source configuration is missing"))?;
        // Refresh source authority independently of the credential that first
        // activated this corpus. A later overlapping token remains eligible.
        let projection =
            source
                .authorized_indices
                .project(&source.index, &policy, config.tantivy, |path| {
                    let permit = holder.permit()?;
                    permit.check(Action::Read, path, path)?;
                    drop(permit);
                    let actual = self.root.resolve_existing(path)?.relative;
                    self.root.validate_access_file(&actual)?;
                    holder.permit()?.check(Action::Read, path, &actual)?;
                    let raw = self
                        .root
                        .read_exact_bounded(&actual, source::MAX_SOURCE_BYTES)?;
                    let permit = holder.permit()?;
                    permit.check(Action::Read, path, &actual)?;
                    Ok(raw)
                })?;
        let current = projection
            .index
            .read()
            .map_err(|_| authorization_error("semantic projection lock is unavailable"))?
            .authorized_snapshot(|_| true)?
            .revisions;
        let mut previous = self
            .revisions
            .lock()
            .map_err(|_| authorization_error("semantic revision state is unavailable"))?;
        if let Some(runtime) = self.runtime.get() {
            for (path, revision) in &current {
                if previous.get(path) != Some(revision)
                    || (retry_unpublished
                        && !runtime.has_current_or_pending(Path::new(path), revision))
                {
                    runtime.submit_upsert(Path::new(path));
                }
            }
            for path in previous.keys() {
                if !current.contains_key(path) {
                    runtime.submit(Path::new(path), PendingKind::Remove);
                }
            }
        }
        *previous = current;
        Ok(())
    }

    fn record_reconciliation(&self, result: VaultResult<()>) {
        let category = result.as_ref().err().map(VaultError::diagnostic_category);
        let next = result.err().map(|error| error.to_string());
        let mut previous = self
            .last_error
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if *previous != next {
            if let Some(category) = category {
                tracing::warn!(corpus = %self.diagnostic_id, error_category = category, "authenticated semantic source reconciliation failed");
            }
            *previous = next;
        }
    }

    fn retire(&self) {
        if let Some(runtime) = self.runtime.get() {
            runtime.retire();
        }
    }
}

struct CacheCleanup {
    root: VaultRoot,
    directory: PathBuf,
    store: AccessStore,
    source: SourceIdentity,
    exclusions: String,
    diagnostic_id: String,
    sweep: Mutex<CacheSweep>,
}

impl CacheCleanup {
    fn request_sweep(&self) {
        self.sweep
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .requested = true;
    }

    /// Return whether another bounded pass is required. Keep the native cursor
    /// between passes so repeated limits cannot starve later directory entries.
    fn sweep(&self) -> VaultResult<bool> {
        let mut sweep = self
            .sweep
            .lock()
            .map_err(|_| authorization_error("cache cleanup state is unavailable"))?;
        let result = self.sweep_pass(&mut sweep);
        if result.is_err() {
            // Fail visibly once. A new activation/retirement can request another
            // pass after the owner repairs the store or filesystem failure.
            sweep.cursor = None;
            sweep.requested = false;
        }
        result
    }

    fn sweep_pass(&self, sweep: &mut CacheSweep) -> VaultResult<bool> {
        if sweep.cursor.is_none() {
            if !std::mem::take(&mut sweep.requested) {
                return Ok(false);
            }
            let directory = match self.root.open_dir_exact(&self.directory) {
                Ok(directory) => directory,
                Err(error) if error.is_not_found() => return Ok(false),
                Err(error) => return Err(error),
            };
            sweep.cursor = Some(Box::new(super::fs::dir_entries(&directory)?.map(|entry| {
                let entry = entry?;
                Ok((entry.file_name(), entry.file_type()?.is_dir()))
            })));
        }
        let mut observed_policy = String::new();
        let mut eligible = HashSet::new();
        for _ in 0..CACHE_ENTRIES_PER_PASS {
            // Pull and classify a name outside the policy lock, counting every
            // entry against this pass even when it is not one of our cache keys.
            let Some(entry) = sweep.cursor.as_mut().and_then(Iterator::next) else {
                sweep.cursor = None;
                return Ok(sweep.requested);
            };
            let (name, is_directory) = entry?;
            let Some(name) = name.to_str().filter(|name| {
                name.len() == 64
                    && name
                        .bytes()
                        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            }) else {
                continue;
            };
            if !is_directory {
                continue;
            }
            let relative = self.directory.join(name).join("embeddings.bin");
            // One fresh shared lease covers only this candidate's eligibility
            // and fixed-leaf unlink, never directory enumeration or later names.
            self.store.with_snapshot(|snapshot| {
                if observed_policy != snapshot.content_hash {
                    let now = chrono::Utc::now().timestamp();
                    eligible = snapshot
                        .policy
                        .clients
                        .values()
                        .filter(|client| {
                            client.enabled
                                && client.credentials.iter().any(|credential| {
                                    !credential.revoked
                                        && credential.expires_at.is_none_or(|expiry| expiry > now)
                                })
                        })
                        .map(|client| {
                            corpus_cache_key(
                                &self.source,
                                &client.read_policy_hash(),
                                &self.exclusions,
                            )
                        })
                        .collect();
                    observed_policy.clone_from(&snapshot.content_hash);
                }
                if !eligible.contains(name) {
                    CacheFile::new(self.root.clone(), relative)
                        .remove()
                        .map_err(|error| {
                            crate::access::store::AccessError::Configuration(format!(
                                "obsolete semantic checkpoint cleanup failed: {error}"
                            ))
                        })?;
                }
                Ok(())
            })?;
        }
        Ok(true)
    }
}

/// One registry and reconciliation task belong to the source service, rather
/// than a request or modern HTTP connection. Entries activate only on search.
pub(super) struct AuthenticatedSemanticRegistry {
    entries: Mutex<HashMap<String, Arc<SemanticEntry>>>,
    model: Arc<OnceCell<Arc<dyn Embedder>>>,
    notify: Arc<Notify>,
    worker: Mutex<Option<JoinHandle<()>>>,
    cleanup: OnceLock<CacheCleanup>,
}

impl AuthenticatedSemanticRegistry {
    fn new() -> Arc<Self> {
        let registry = Arc::new(Self {
            entries: Mutex::new(HashMap::new()),
            model: Arc::new(OnceCell::new()),
            notify: Arc::new(Notify::new()),
            worker: Mutex::new(None),
            cleanup: OnceLock::new(),
        });
        let weak = Arc::downgrade(&registry);
        let notify = Arc::clone(&registry.notify);
        let worker = tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = notify.notified() => (),
                    _ = tokio::time::sleep(RECONCILIATION_INTERVAL) => (),
                }
                let Some(registry) = weak.upgrade() else {
                    return;
                };
                let result = tokio::task::spawn_blocking(move || registry.reconcile()).await;
                if result.is_err() {
                    tracing::warn!(
                        error_category = "worker_failure",
                        "authenticated semantic reconciliation task failed"
                    );
                }
            }
        });
        *registry
            .worker
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = Some(worker);
        registry
    }

    fn reconcile(&self) {
        let entries = self
            .entries
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .iter()
            .map(|(key, entry)| (key.clone(), Arc::clone(entry)))
            .collect::<Vec<_>>();
        let mut retired = false;
        for (key, entry) in entries {
            // A failed authority read withdraws work. Restoration is a new lazy
            // activation; no stale grant or broader checkpoint is used meanwhile.
            if entry.holder().is_err() {
                retired |= self.retire_withdrawn(&key, &entry);
                continue;
            }
            entry.record_reconciliation(entry.reconcile(false));
        }
        if retired && let Some(cleanup) = self.cleanup.get() {
            cleanup.request_sweep();
        }
        self.cleanup_obsolete_caches();
    }

    fn retire_withdrawn(&self, key: &str, observed: &Arc<SemanticEntry>) -> bool {
        let removed = {
            let mut entries = self
                .entries
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            let Some(current) = entries.get(key) else {
                return false;
            };
            // A stale observation cannot retire a replacement, or an entry
            // which gained a newly valid holder while reconciliation was queued.
            if !Arc::ptr_eq(current, observed) || current.holder().is_ok() {
                return false;
            }
            entries.remove(key)
        };
        if let Some(entry) = removed {
            entry.retire();
            true
        } else {
            false
        }
    }

    fn cleanup_obsolete_caches(&self) {
        if let Some(cleanup) = self.cleanup.get()
            && let Err(error) = cleanup.sweep()
        {
            tracing::warn!(cache_scope = %cleanup.diagnostic_id, error_category = error.diagnostic_category(), "obsolete semantic checkpoint cleanup failed");
        }
    }

    fn runtime(&self, vault: &Vault, activate: bool) -> VaultResult<Option<EmbeddingRuntime>> {
        let client = vault
            .access_client()
            .ok_or_else(|| authorization_error("authenticated invocation context is missing"))?;
        client.current()?;
        let projection = vault
            .projection
            .as_ref()
            .cloned()
            .ok_or_else(|| authorization_error("authorized projection is missing"))?;
        let key = client.read_policy_key().to_owned();
        let mut entries = self
            .entries
            .lock()
            .map_err(|_| authorization_error("semantic registry lock is unavailable"))?;
        if let Some(entry) = entries.get(&key)
            && (!Arc::ptr_eq(&entry.projection, &projection)
                || (activate
                    && entry
                        .runtime
                        .get()
                        .is_some_and(|runtime| runtime.first_load_error().is_some())))
        {
            // A restored profile or repaired loader gets a new service owner on
            // explicit activation, never a stale grant or an automatic retry loop.
            entry.retire();
            entries.remove(&key);
        }
        if let Some(entry) = entries.get(&key).cloned() {
            if activate {
                // Registration and retirement share the map lock. Otherwise a
                // reaper could remove this entry between lookup and holder insert.
                entry
                    .holders
                    .lock()
                    .map_err(|_| authorization_error("semantic holder state is unavailable"))?
                    .insert(
                        client.client_id().to_owned(),
                        ActivatedClient::from_client(client),
                    );
            }
            drop(entries);
            if activate {
                entry.reconcile(true)?;
            }
            return Ok(entry.runtime.get().cloned());
        }
        if !activate {
            return Ok(None);
        }
        let config =
            vault.inner.authenticated_config.as_ref().ok_or_else(|| {
                authorization_error("authenticated source configuration is missing")
            })?;
        if !config.embeddings {
            return Ok(None);
        }
        let root = vault.authenticated_root()?;
        let cache_root =
            VaultRoot::open(vault.resolved_mcp_data().ok_or_else(|| {
                authorization_error("external semantic data directory is missing")
            })?)?;
        let profile_key =
            profile_cache_key(client.store(), &vault.exclude().scope_id(vault.root()));
        let profile_directory = PathBuf::from("embeddings/access-v1").join(&profile_key);
        if self.cleanup.get().is_none() {
            let _ = self.cleanup.set(CacheCleanup {
                root: cache_root.clone(),
                directory: profile_directory.clone(),
                store: client.store().clone(),
                source: client.source().clone(),
                exclusions: vault.exclude().scope_id(vault.root()),
                diagnostic_id: profile_key,
                sweep: Mutex::new(CacheSweep::default()),
            });
        }
        if let Some(cleanup) = self.cleanup.get() {
            cleanup.request_sweep();
        }
        self.cleanup_obsolete_caches();
        let relative = profile_directory
            .join(cache_key(vault, client))
            .join("embeddings.bin");
        let cache = CacheFile::new(cache_root, relative);
        let initial_revisions = projection
            .index
            .read()
            .map_err(|_| authorization_error("semantic projection lock is unavailable"))?
            .authorized_snapshot(|_| true)?
            .revisions;
        let entry = Arc::new(SemanticEntry {
            source: Arc::downgrade(&vault.inner),
            root: root.clone(),
            projection: Arc::clone(&projection),
            holders: Mutex::new(HashMap::from([(
                client.client_id().to_owned(),
                ActivatedClient::from_client(client),
            )])),
            runtime: OnceLock::new(),
            revisions: Mutex::new(initial_revisions),
            last_error: Mutex::new(None),
            diagnostic_id: cache.diagnostic_id(),
        });
        let weak = Arc::downgrade(&entry);
        let authorization = EmbeddingAuthorization::new(move |path| {
            weak.upgrade()
                .ok_or_else(|| authorization_error("semantic corpus has stopped"))?
                .admit(path)
        });
        let shared_model = Arc::clone(&self.model);
        let model_name = config.embeddings_model.clone();
        let provider = config.embedding_provider;
        let loader_authorization = authorization.clone();
        let runtime = EmbeddingRuntime::spawn_with_authorized_source(
            Storage::Local(root),
            Arc::clone(&projection.index),
            cache,
            crate::config::embedding_batch_size_from_env().map_err(VaultError::Embedding)?,
            true,
            authorization,
            async move {
                shared_model
                    .get_or_try_init(|| async move {
                        let model = EmbeddingModel::load_authorized(
                            &model_name,
                            provider,
                            Some(loader_authorization),
                        )
                        .await?;
                        Ok::<_, VaultError>(Arc::new(model) as Arc<dyn Embedder>)
                    })
                    .await
                    .cloned()
            },
        );
        entry
            .runtime
            .set(runtime.clone())
            .map_err(|_| authorization_error("semantic runtime was initialized twice"))?;
        entries.insert(key, Arc::clone(&entry));
        drop(entries);
        entry.reconcile(true)?;
        self.notify.notify_one();
        Ok(Some(runtime))
    }
}

impl Drop for AuthenticatedSemanticRegistry {
    fn drop(&mut self) {
        if let Some(worker) = self
            .worker
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take()
        {
            worker.abort();
        }
        for entry in self
            .entries
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .values()
        {
            if let Some(runtime) = entry.runtime.get() {
                runtime.retire();
            }
        }
    }
}

impl Vault {
    pub(crate) fn authenticated_semantic_runtime(
        &self,
        activate: bool,
    ) -> VaultResult<Option<EmbeddingRuntime>> {
        if !self.is_authenticated() {
            return Ok(None);
        }
        self.check_access_current()?;
        if !self
            .inner
            .authenticated_config
            .as_ref()
            .is_some_and(|config| config.embeddings)
        {
            return Ok(None);
        }
        if activate {
            self.refresh_authorized_projection()?;
        }
        let registry = if activate {
            self.inner
                .authenticated_semantics
                .get_or_init(AuthenticatedSemanticRegistry::new)
        } else {
            let Some(registry) = self.inner.authenticated_semantics.get() else {
                return Ok(None);
            };
            registry
        };
        registry.runtime(self, activate)
    }

    pub(crate) fn authenticated_query_authorization(&self) -> VaultResult<EmbeddingAuthorization> {
        let client = self
            .access_client()
            .cloned()
            .ok_or_else(|| authorization_error("authenticated invocation context is missing"))?;
        let root = self.authenticated_root()?;
        Ok(EmbeddingAuthorization::new(move |path| {
            let permit = client.permit()?;
            if let Some(path) = path {
                permit.check(Action::Read, path, path)?;
                drop(permit);
                let actual = root.resolve_existing(path)?.relative;
                root.validate_access_file(&actual)?;
                let permit = client.permit()?;
                permit.check(Action::Read, path, &actual)?;
                return Ok(Box::new(permit));
            }
            Ok(Box::new(permit))
        }))
    }

    pub(crate) fn notify_authenticated_semantics(&self) {
        if let Some(registry) = self.inner.authenticated_semantics.get() {
            registry.notify.notify_one();
        }
    }
}

fn profile_cache_key(store: &AccessStore, exclusions: &str) -> String {
    let mut hash = Sha256::new();
    hash.update(b"obsidian-mcp/access-profile-cache/v1\0");
    hash.update(store.path().as_os_str().as_encoded_bytes());
    hash.update([0]);
    hash.update(exclusions.as_bytes());
    crate::config::digest_hex(hash)
}

fn cache_key(vault: &Vault, client: &AuthorizedClient) -> String {
    corpus_cache_key(
        client.source(),
        client.read_policy_key(),
        &vault.exclude().scope_id(vault.root()),
    )
}

fn corpus_cache_key(source: &SourceIdentity, read_policy: &str, exclusions: &str) -> String {
    let mut hash = Sha256::new();
    for field in [
        "obsidian-mcp/authorized-semantic/v1",
        &source.storage_key,
        &source.object_id,
        read_policy,
        exclusions,
    ] {
        hash.update((field.len() as u64).to_be_bytes());
        hash.update(field.as_bytes());
    }
    crate::config::digest_hex(hash)
}

fn authorization_error(message: &str) -> VaultError {
    VaultError::Authorization(message.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::access::clients::{ClientPolicy, Grant, Role};

    #[tokio::test]
    async fn stale_retirement_cannot_remove_a_replacement_or_revalidated_owner() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("vault");
        std::fs::create_dir(&root).unwrap();
        let data = directory.path().join("private/data");
        let store =
            AccessStore::initialize(&directory.path().join("private/access.json"), &root, &data)
                .unwrap();
        store
            .update(None, |policy| {
                policy.clients.insert(
                    "reader".into(),
                    ClientPolicy {
                        grants: vec![Grant {
                            id: "root-read".into(),
                            directory: Some(String::new()),
                            file: None,
                            role: Role::Reader,
                        }],
                        ..Default::default()
                    },
                );
                Ok(())
            })
            .unwrap();
        let token_path = directory.path().join("private/reader.token");
        store.issue_token("reader", &token_path, None).unwrap();
        let token = crate::access::store::read_token_file(&token_path).unwrap();
        let client = store.authenticate(token.as_str()).unwrap();
        let mut config = crate::test_helpers::test_config(&root);
        config.mcp_data_dir = Some(data);
        let vault = Vault::open_authenticated(&config)
            .await
            .unwrap()
            .with_access(client.clone())
            .unwrap();
        let make_entry = || {
            Arc::new(SemanticEntry {
                source: Arc::downgrade(&vault.inner),
                root: vault.authenticated_root().unwrap(),
                projection: Arc::clone(vault.projection.as_ref().unwrap()),
                holders: Mutex::new(HashMap::new()),
                runtime: OnceLock::new(),
                revisions: Mutex::new(HashMap::new()),
                last_error: Mutex::new(None),
                diagnostic_id: "retirement-fixture".into(),
            })
        };
        let registry = AuthenticatedSemanticRegistry {
            entries: Mutex::new(HashMap::new()),
            model: Arc::new(OnceCell::new()),
            notify: Arc::new(Notify::new()),
            worker: Mutex::new(None),
            cleanup: OnceLock::new(),
        };
        let key = client.read_policy_key();
        let old = make_entry();
        registry
            .entries
            .lock()
            .unwrap()
            .insert(key.into(), Arc::clone(&old));
        // Reconciliation retains this failed observation while a later explicit
        // activation replaces the same key. The old reaper must leave it alone.
        let observed = registry.entries.lock().unwrap()[key].clone();
        assert!(observed.holder().is_err());
        let replacement = make_entry();
        registry
            .entries
            .lock()
            .unwrap()
            .insert(key.into(), Arc::clone(&replacement));
        assert!(!registry.retire_withdrawn(key, &observed));
        assert!(Arc::ptr_eq(
            &registry.entries.lock().unwrap()[key],
            &replacement
        ));
        // A new valid holder can also reuse the same entry after a failed read.
        assert!(registry.runtime(&vault, true).unwrap().is_none());
        assert!(!registry.retire_withdrawn(key, &replacement));
        assert!(Arc::ptr_eq(
            &registry.entries.lock().unwrap()[key],
            &replacement
        ));
        store
            .update(None, |policy| {
                policy.clients.get_mut("reader").unwrap().enabled = false;
                Ok(())
            })
            .unwrap();
        assert!(registry.retire_withdrawn(key, &replacement));
        assert!(!registry.entries.lock().unwrap().contains_key(key));
    }

    #[test]
    fn cache_sweep_reclaims_only_ineligible_corpora_in_its_profile_and_exclusion_namespace() {
        let directory = tempfile::tempdir().unwrap();
        let vault = directory.path().join("vault");
        std::fs::create_dir(&vault).unwrap();
        let profile = directory.path().join("private/access.json");
        let data = directory.path().join("private/data");
        let store = AccessStore::initialize(&profile, &vault, &data).unwrap();
        store
            .update(None, |policy| {
                let client = ClientPolicy {
                    grants: vec![Grant {
                        id: "shared-read".into(),
                        directory: Some("Shared/".into()),
                        file: None,
                        role: Role::Reader,
                    }],
                    ..Default::default()
                };
                policy.clients.insert("first".into(), client.clone());
                policy.clients.insert("second".into(), client);
                Ok(())
            })
            .unwrap();
        store
            .issue_token("first", &directory.path().join("private/first.token"), None)
            .unwrap();
        store
            .issue_token(
                "second",
                &directory.path().join("private/second.token"),
                None,
            )
            .unwrap();
        let snapshot = store.snapshot().unwrap();
        let source = snapshot.policy.source.clone();
        let exclusions = "effective-exclusions";
        let namespace =
            PathBuf::from("embeddings/access-v1").join(profile_cache_key(&store, exclusions));
        let current = corpus_cache_key(
            &source,
            &snapshot.policy.clients["first"].read_policy_hash(),
            exclusions,
        );
        let obsolete = "a".repeat(64);
        assert_ne!(current, obsolete);
        let root = VaultRoot::open(&data).unwrap();
        let current_file = namespace.join(&current).join("embeddings.bin");
        let obsolete_file = namespace.join(&obsolete).join("embeddings.bin");
        let unrelated_leaf = namespace.join(&obsolete).join("owner-file");
        let other_exclusions = PathBuf::from("embeddings/access-v1")
            .join(profile_cache_key(&store, "other-exclusions"))
            .join(&obsolete)
            .join("embeddings.bin");
        let unknown_directory = namespace.join("owner-directory/embeddings.bin");
        for path in [
            &current_file,
            &obsolete_file,
            &unrelated_leaf,
            &other_exclusions,
            &unknown_directory,
        ] {
            root.create_file(path, "checkpoint marker").unwrap();
        }
        let extra = (0..CACHE_ENTRIES_PER_PASS + 5)
            .map(|index| {
                namespace
                    .join(format!("{index:064x}"))
                    .join("embeddings.bin")
            })
            .collect::<Vec<_>>();
        for path in &extra {
            root.create_file(path, "obsolete bounded-pass marker")
                .unwrap();
        }
        let cleanup = CacheCleanup {
            root: root.clone(),
            directory: namespace,
            store: store.clone(),
            source,
            exclusions: exclusions.into(),
            diagnostic_id: "synthetic-cache-scope".into(),
            sweep: Mutex::new(CacheSweep::default()),
        };
        cleanup.request_sweep();
        assert!(cleanup.sweep().unwrap());
        let remaining = extra
            .iter()
            .filter(|path| root.read_file(path).is_ok())
            .count();
        assert!(
            remaining > 0,
            "one pass must leave work beyond its entry limit"
        );
        assert!(extra.len() - remaining <= CACHE_ENTRIES_PER_PASS);
        let mut passes = 1;
        while cleanup.sweep().unwrap() {
            passes += 1;
            assert!(
                passes < 4,
                "the retained cursor must advance beyond earlier names"
            );
        }
        assert!(
            extra
                .iter()
                .all(|path| root.read_file(path).unwrap_err().is_not_found())
        );
        assert!(root.read_file(&current_file).is_ok());
        assert!(root.read_file(&obsolete_file).unwrap_err().is_not_found());
        assert!(root.read_file(&unrelated_leaf).is_ok());
        assert!(root.read_file(&other_exclusions).is_ok());
        assert!(root.read_file(&unknown_directory).is_ok());
        // A different still-eligible client or process can own this same corpus.
        store
            .update(None, |policy| {
                policy.clients.get_mut("first").unwrap().enabled = false;
                Ok(())
            })
            .unwrap();
        cleanup.request_sweep();
        while cleanup.sweep().unwrap() {}
        assert!(root.read_file(&current_file).is_ok());
        store
            .update(None, |policy| {
                policy.clients.get_mut("second").unwrap().enabled = false;
                Ok(())
            })
            .unwrap();
        cleanup.request_sweep();
        while cleanup.sweep().unwrap() {}
        assert!(root.read_file(&current_file).unwrap_err().is_not_found());
        assert!(root.read_file(&other_exclusions).is_ok());
    }
}
