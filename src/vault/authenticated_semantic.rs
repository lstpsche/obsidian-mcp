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

#[derive(Default)]
struct SemanticRevisions {
    generation: Option<u64>,
    notes: HashMap<OsString, String>,
}

struct SemanticEntry {
    source: Weak<VaultInner>,
    root: VaultRoot,
    projection: Arc<AuthorizedProjection>,
    holders: Mutex<HashMap<String, ActivatedClient>>,
    runtime: OnceLock<EmbeddingRuntime>,
    revisions: Mutex<SemanticRevisions>,
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
            self.root.validate_access_file_exact(&actual)?;
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
        let mut previous = self
            .revisions
            .lock()
            .map_err(|_| authorization_error("semantic revision state is unavailable"))?;
        let generation = source
            .index
            .read()
            .map_err(|_| authorization_error("semantic source lock is unavailable"))?
            .generation();
        if !retry_unpublished
            && previous.generation == Some(generation)
            && self.projection.is_ready()
        {
            return Ok(());
        }
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
                    self.root.validate_access_file_exact(&actual)?;
                    holder.permit()?.check(Action::Read, path, &actual)?;
                    let raw = self
                        .root
                        .read_exact_bounded(&actual, source::MAX_SOURCE_BYTES)?;
                    let permit = holder.permit()?;
                    permit.check(Action::Read, path, &actual)?;
                    Ok(raw)
                })?;
        let index = projection
            .index
            .read()
            .map_err(|_| authorization_error("semantic projection lock is unavailable"))?;
        let current = index.source_revisions_snapshot()?;
        drop(index);
        if let Some(runtime) = self.runtime.get() {
            for (path, revision) in &current {
                if previous.notes.get(path) != Some(revision)
                    || (retry_unpublished
                        && !runtime.has_current_or_pending(Path::new(path), revision))
                {
                    runtime.submit_upsert(Path::new(path));
                }
            }
            for path in previous.notes.keys() {
                if !current.contains_key(path) {
                    runtime.submit(Path::new(path), PendingKind::Remove);
                }
            }
        }
        *previous = SemanticRevisions {
            // Source changes outside this corpus can leave its projection
            // generation unchanged. Remember the source observation instead.
            generation: Some(generation),
            notes: current,
        };
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
            && !activate
            && !Arc::ptr_eq(&entry.projection, &projection)
        {
            // An immutable caller view can outlive projection replacement.
            // Passive inspection cannot change the newer owner's lifecycle.
            return Ok(None);
        }
        if let Some(entry) = entries.get(&key)
            && activate
            && (!Arc::ptr_eq(&entry.projection, &projection)
                || entry
                    .runtime
                    .get()
                    .is_some_and(|runtime| runtime.first_load_error().is_some()))
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
        let relative = profile_directory
            .join(cache_key(vault, client))
            .join("embeddings.bin");
        let cache = CacheFile::new(cache_root, relative);
        let index = projection
            .index
            .read()
            .map_err(|_| authorization_error("semantic projection lock is unavailable"))?;
        let initial_revisions = SemanticRevisions {
            generation: Some(index.generation()),
            notes: index.source_revisions_snapshot()?,
        };
        drop(index);
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
        // Directory and policy I/O can wait independently of registration,
        // retirement and passive status lookups for every readable policy.
        if let Some(cleanup) = self.cleanup.get() {
            cleanup.request_sweep();
        }
        self.cleanup_obsolete_caches();
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
                root.validate_access_file_exact(&actual)?;
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
    use super::super::embedding_input::{EmbeddingInputPolicy, EmbeddingRequestLimits};
    use super::super::embedding_runtime::{EmbeddingPhase, EmbeddingRuntimeStatus};
    use super::super::embeddings::{EmbeddingBackendKind, EmbeddingSpaceIdentity};
    use super::*;
    use crate::access::clients::{ClientPolicy, Grant, Role};
    use std::sync::atomic::{AtomicBool, Ordering};

    type DispatchGate = (
        tokio::sync::oneshot::Sender<()>,
        std::sync::mpsc::Receiver<()>,
    );

    struct CapturingEmbedder {
        identity: EmbeddingSpaceIdentity,
        policy: EmbeddingInputPolicy,
        inputs: Mutex<Vec<String>>,
        gate: Mutex<Option<DispatchGate>>,
        fail: AtomicBool,
    }

    impl CapturingEmbedder {
        fn new() -> Self {
            Self {
                identity: EmbeddingSpaceIdentity {
                    backend: EmbeddingBackendKind::Local,
                    model: "authorized-fixture".into(),
                    endpoint_fingerprint: None,
                    dimension: 3,
                    input_version: super::super::embeddings::EMBEDDING_INPUT_VERSION,
                    tokenizer_fingerprint: [0; 32],
                    max_input_tokens: 512,
                    chunking_version: super::super::embedding_input::CHUNKING_VERSION,
                },
                policy: EmbeddingInputPolicy {
                    max_input_tokens: 512,
                    tokenizer_fingerprint: [0; 32],
                },
                inputs: Mutex::new(Vec::new()),
                gate: Mutex::new(None),
                fail: AtomicBool::new(false),
            }
        }
    }

    impl Embedder for CapturingEmbedder {
        fn dimension(&self) -> usize {
            self.identity.dimension
        }
        fn space_identity(&self) -> &EmbeddingSpaceIdentity {
            &self.identity
        }
        fn input_policy(&self) -> &EmbeddingInputPolicy {
            &self.policy
        }
        fn count_tokens(&self, text: &str) -> VaultResult<usize> {
            Ok(text.chars().count())
        }
        fn request_limits(&self) -> EmbeddingRequestLimits {
            EmbeddingRequestLimits {
                max_inputs: 32,
                max_total_tokens: 16_384,
                max_request_bytes: 16 * 1024 * 1024,
            }
        }
        fn embed_batch(&self, texts: &[&str]) -> VaultResult<Vec<Vec<f32>>> {
            self.inputs
                .lock()
                .unwrap()
                .extend(texts.iter().map(|text| (*text).to_owned()));
            if let Some((started, release)) = self.gate.lock().unwrap().take() {
                started.send(()).unwrap();
                release.recv_timeout(Duration::from_secs(5)).unwrap();
            }
            if self.fail.load(Ordering::Acquire) {
                return Err(VaultError::EmbeddingRequest {
                    message: "synthetic document provider failure".into(),
                    retryable: true,
                    retry_after: None,
                });
            }
            Ok(texts.iter().map(|_| vec![1.0, 0.0, 0.0]).collect())
        }
    }

    fn install_registry(
        vault: &Vault,
        model: Arc<CapturingEmbedder>,
    ) -> Arc<AuthenticatedSemanticRegistry> {
        let registry = Arc::new(AuthenticatedSemanticRegistry {
            entries: Mutex::new(HashMap::new()),
            model: Arc::new(OnceCell::new()),
            notify: Arc::new(Notify::new()),
            worker: Mutex::new(None),
            cleanup: OnceLock::new(),
        });
        assert!(registry.model.set(model as Arc<dyn Embedder>).is_ok());
        assert!(
            vault
                .inner
                .authenticated_semantics
                .set(Arc::clone(&registry))
                .is_ok()
        );
        registry
    }

    async fn wait_for_runtime(
        runtime: &EmbeddingRuntime,
        predicate: impl Fn(&EmbeddingRuntimeStatus) -> bool,
    ) {
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if predicate(&runtime.status()) {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("runtime status timed out: {:?}", runtime.status()));
    }

    async fn reader_fixture(
        embeddings: bool,
    ) -> (tempfile::TempDir, AccessStore, Vault, AuthorizedClient) {
        reader_fixture_with_lexical(embeddings, false).await
    }

    async fn reader_fixture_with_lexical(
        embeddings: bool,
        tantivy: bool,
    ) -> (tempfile::TempDir, AccessStore, Vault, AuthorizedClient) {
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
        config.embeddings = embeddings;
        config.tantivy = tantivy;
        let vault = Vault::open_authenticated(&config)
            .await
            .unwrap()
            .with_access(client.clone())
            .unwrap();
        (directory, store, vault, client)
    }

    #[tokio::test]
    async fn authenticated_vault_info_propagates_semantic_registry_failure() {
        let (_directory, _store, vault, _client) = reader_fixture(true).await;
        let registry = AuthenticatedSemanticRegistry::new();
        assert!(
            vault
                .inner
                .authenticated_semantics
                .set(Arc::clone(&registry))
                .is_ok()
        );
        let poison = std::thread::spawn(move || {
            let _entries = registry.entries.lock().unwrap();
            panic!("synthetic semantic registry failure");
        });
        assert!(poison.join().is_err());
        let runtime = crate::tools::SemanticRuntime {
            mode: crate::config::SemanticMode::Local,
            daemon_client: None,
            pending_daemon_bootstrap: None,
            daemon_unavailable_reason: None,
            prefetch_count: 50,
            attachments: Arc::default(),
            local_fallback_active: Arc::default(),
            daemon_health_pending: Arc::default(),
        };
        let result = crate::tools::utility::vault_info(
            &vault,
            &runtime,
            crate::tools::utility::VaultInfoParams {},
        )
        .await;
        assert!(
            result.is_err(),
            "registry failure must not become absent semantic status"
        );
    }

    #[tokio::test]
    async fn activation_cache_sweep_does_not_hold_the_semantic_registry_lock() {
        let (_directory, store, vault, client) = reader_fixture(true).await;
        let registry = install_registry(&vault, Arc::new(CapturingEmbedder::new()));
        let exclusions = vault.exclude().scope_id(vault.root());
        let directory =
            PathBuf::from("embeddings/access-v1").join(profile_cache_key(&store, &exclusions));
        let obsolete = "a".repeat(64);
        assert_ne!(obsolete, cache_key(&vault, &client));
        let root = VaultRoot::open(vault.resolved_mcp_data().unwrap()).unwrap();
        let checkpoint = directory.join(&obsolete).join("embeddings.bin");
        root.create_file(&checkpoint, "obsolete checkpoint marker")
            .unwrap();
        let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let cursor = std::iter::once_with(move || {
            entered_tx.send(()).unwrap();
            release_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            Ok((OsString::from(obsolete), true))
        });
        assert!(
            registry
                .cleanup
                .set(CacheCleanup {
                    root: root.clone(),
                    directory,
                    store,
                    source: client.source().clone(),
                    exclusions,
                    diagnostic_id: "held-cache-sweep".into(),
                    sweep: Mutex::new(CacheSweep {
                        cursor: Some(Box::new(cursor)),
                        requested: false,
                    }),
                })
                .is_ok()
        );
        let activation_registry = Arc::clone(&registry);
        let activation_vault = vault.clone();
        let activation = tokio::task::spawn_blocking(move || {
            activation_registry.runtime(&activation_vault, true)
        });
        tokio::time::timeout(Duration::from_secs(5), entered_rx)
            .await
            .unwrap()
            .unwrap();
        let registry_available = registry.entries.try_lock().is_ok();
        release_tx.send(()).unwrap();
        let runtime = activation.await.unwrap().unwrap();
        assert!(runtime.is_some());
        assert!(
            registry_available,
            "cache I/O must not hold semantic registration or passive lookup behind the map lock"
        );
        assert!(
            root.read_file(&checkpoint).is_err(),
            "first activation must finish its bounded obsolete-checkpoint sweep"
        );
    }

    #[tokio::test]
    async fn incremental_projection_publication_keeps_semantic_admission_available() {
        let (_directory, _store, vault, client) = reader_fixture_with_lexical(true, true).await;
        let path = Path::new("one.md");
        std::fs::write(vault.root().join(path), "# One\noriginalmarker").unwrap();
        vault.reindex(path).await.unwrap();
        let registry = install_registry(&vault, Arc::new(CapturingEmbedder::new()));
        let runtime = registry.runtime(&vault, true).unwrap().unwrap();
        wait_for_runtime(&runtime, |status| status.phase == EmbeddingPhase::Ready).await;
        let entry = registry.entries.lock().unwrap()[client.read_policy_key()].clone();
        let projection = Arc::clone(&entry.projection);
        let index_guard = projection.index.read().unwrap();
        std::fs::write(vault.root().join(path), "# One\nincrementalmarker").unwrap();
        vault.reindex_source_local(path).unwrap();
        let source = Arc::clone(&vault.inner);
        let policy = client.policy().clone();
        let root = vault.authenticated_root().unwrap();
        let worker = std::thread::spawn(move || {
            source
                .authorized_indices
                .project(&source.index, &policy, true, |path| {
                    root.read_exact_bounded(path, source::MAX_SOURCE_BYTES)
                })
        });
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        let published_lexical = loop {
            if projection
                .tantivy
                .as_ref()
                .unwrap()
                .search("incrementalmarker", 5)
                .unwrap()
                .iter()
                .any(|(hit, _)| hit == path)
            {
                break true;
            }
            if std::time::Instant::now() >= deadline {
                break false;
            }
            std::thread::sleep(Duration::from_millis(5));
        };
        // Tantivy has committed the update, but the metadata write remains
        // blocked. Existing readable-corpus authority must still admit work.
        let admitted = entry.admit(None).is_ok();
        drop(index_guard);
        worker.join().unwrap().unwrap();
        assert!(
            published_lexical,
            "fixture must reach incremental publication"
        );
        assert!(
            admitted,
            "successful incremental publication must not withdraw semantic admission"
        );
    }

    #[tokio::test]
    async fn stale_passive_projection_cannot_retire_a_new_owner_or_stop_ingestion() {
        let (_directory, _store, stale, client) = reader_fixture(true).await;
        let path = Path::new("one.md");
        std::fs::write(stale.root().join(path), "# One\nbefore replacement").unwrap();
        stale.reindex(path).await.unwrap();
        let model = Arc::new(CapturingEmbedder::new());
        let registry = install_registry(&stale, Arc::clone(&model));
        assert!(stale.authenticated_embedding_status().unwrap().is_none());
        assert!(registry.entries.lock().unwrap().is_empty());
        let old_runtime = registry.runtime(&stale, true).unwrap().unwrap();
        wait_for_runtime(&old_runtime, |status| status.phase == EmbeddingPhase::Ready).await;

        // Projection withdrawal/recreation retains the old immutable view while
        // another valid view explicitly activates the new service-owned corpus.
        stale
            .inner
            .authorized_indices
            .retire(client.read_policy_key())
            .unwrap();
        let fresh = stale.with_access(client.clone()).unwrap();
        assert!(!Arc::ptr_eq(
            stale.projection.as_ref().unwrap(),
            fresh.projection.as_ref().unwrap()
        ));
        let current = registry.runtime(&fresh, true).unwrap().unwrap();
        wait_for_runtime(&current, |status| status.phase == EmbeddingPhase::Ready).await;
        assert!(stale.authenticated_embedding_status().unwrap().is_none());

        std::fs::write(
            fresh.root().join(path),
            "# One\nafter-passive-status-marker",
        )
        .unwrap();
        fresh.reindex(path).await.unwrap();
        registry.reconcile();
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if model
                    .inputs
                    .lock()
                    .unwrap()
                    .iter()
                    .any(|input| input.contains("after-passive-status-marker"))
                {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("passive stale status must preserve the current owner's later document dispatch");
        assert!(current.is_active());
        assert!(
            registry
                .entries
                .lock()
                .unwrap()
                .contains_key(client.read_policy_key())
        );
    }

    #[tokio::test]
    async fn empty_current_corpus_distinguishes_pending_work_from_provider_failure() {
        let (_directory, _store, vault, _client) = reader_fixture(true).await;
        let old = Path::new("old.md");
        let new = Path::new("new.md");
        std::fs::write(vault.root().join(old), "# Old\nprevious source").unwrap();
        vault.reindex(old).await.unwrap();
        let model = Arc::new(CapturingEmbedder::new());
        let registry = install_registry(&vault, Arc::clone(&model));
        let runtime = registry.runtime(&vault, true).unwrap().unwrap();
        wait_for_runtime(&runtime, |status| status.phase == EmbeddingPhase::Ready).await;
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        *model.gate.lock().unwrap() = Some((started_tx, release_rx));
        std::fs::remove_file(vault.root().join(old)).unwrap();
        std::fs::write(vault.root().join(new), "# New\nreplacement source").unwrap();
        vault.reindex(old).await.unwrap();
        vault.reindex(new).await.unwrap();
        registry.reconcile();
        tokio::time::timeout(Duration::from_secs(5), started_rx)
            .await
            .unwrap()
            .unwrap();
        assert!(
            runtime.query_snapshot().is_ok(),
            "retained snapshot stays queryable"
        );
        let warming = vault.search_semantic_ranked("replacement source", 5);
        model.fail.store(true, Ordering::Release);
        release_tx.send(()).unwrap();
        assert!(matches!(
            warming,
            Err(VaultError::Storage {
                kind: crate::error::StorageErrorKind::Warming,
                ..
            })
        ));
        wait_for_runtime(&runtime, |status| {
            status.phase == EmbeddingPhase::Degraded && status.pending_notes > 0
        })
        .await;
        let unavailable = vault.search_semantic_ranked("replacement source", 5);
        assert!(matches!(
            unavailable,
            Err(VaultError::Storage {
                kind: crate::error::StorageErrorKind::Unavailable,
                ..
            })
        ));
    }

    #[tokio::test]
    async fn warm_semantic_queries_reconcile_a_newly_ineligible_exact_source() {
        let variants: &[bool] = if cfg!(unix) { &[false, true] } else { &[false] };
        for denied_alias in variants {
            let (directory, store, vault, _) = reader_fixture(true).await;
            store
                .update(None, |policy| {
                    policy.clients.get_mut("reader").unwrap().denies =
                        vec![crate::access::clients::Deny::directory(
                            "private", "Private/",
                        )];
                    Ok(())
                })
                .unwrap();
            let token = crate::access::store::read_token_file(
                &directory.path().join("private/reader.token"),
            )
            .unwrap();
            let vault = vault
                .with_access(store.authenticate(token.as_str()).unwrap())
                .unwrap();
            let unsafe_path = Path::new("a-unsafe.md");
            let safe_path = Path::new("b-allowed.md");
            std::fs::write(
                vault.root().join(unsafe_path),
                "# Before\nformerly readable source",
            )
            .unwrap();
            std::fs::write(
                vault.root().join(safe_path),
                "# Allowed\nremaining readable source",
            )
            .unwrap();
            vault.reindex(unsafe_path).await.unwrap();
            vault.reindex(safe_path).await.unwrap();
            let model = Arc::new(CapturingEmbedder::new());
            let registry = install_registry(&vault, Arc::clone(&model));
            let owner = registry.runtime(&vault, true).unwrap().unwrap();
            wait_for_runtime(&owner, |status| status.phase == EmbeddingPhase::Ready).await;
            let runtime = crate::tools::SemanticRuntime {
                mode: crate::config::SemanticMode::Local,
                daemon_client: None,
                pending_daemon_bootstrap: None,
                daemon_unavailable_reason: None,
                prefetch_count: 50,
                attachments: Arc::default(),
                local_fallback_active: Arc::default(),
                daemon_health_pending: Arc::default(),
            };
            let query = "semantic eligibility probe";
            let search = || {
                crate::tools::search::search_semantic(
                    &vault,
                    crate::tools::search::SearchSemanticParams {
                        query: query.into(),
                        top_k: Some(10),
                        include_content: Some(true),
                        ..Default::default()
                    },
                    0.25,
                    &runtime,
                )
            };
            let initial = search().await.unwrap();
            assert_eq!(
                initial.structured_content.unwrap()["results"]
                    .as_array()
                    .unwrap()
                    .len(),
                2
            );
            let before = vault.inner.index.read().unwrap().generation();
            let calls = || {
                model
                    .inputs
                    .lock()
                    .unwrap()
                    .iter()
                    .filter(|input| input.as_str() == query)
                    .count()
            };
            let before_calls = calls();
            if *denied_alias {
                #[cfg(unix)]
                {
                    std::fs::create_dir(vault.root().join("Private")).unwrap();
                    std::fs::write(
                        vault.root().join("Private/hidden.md"),
                        "private replacement marker",
                    )
                    .unwrap();
                    std::fs::remove_file(vault.root().join(unsafe_path)).unwrap();
                    std::os::unix::fs::symlink("Private/hidden.md", vault.root().join(unsafe_path))
                        .unwrap();
                }
            } else {
                std::fs::hard_link(
                    vault.root().join(unsafe_path),
                    directory.path().join("shadow.md"),
                )
                .unwrap();
            }
            let revision = source::source_revision("# Before\nformerly readable source");
            let page = crate::tools::notes::note_read(
                &vault,
                crate::tools::notes::NoteReadParams {
                    path: "a-unsafe.md".into(),
                    expected_revision: Some(revision.clone()),
                    ..Default::default()
                },
            )
            .await
            .unwrap_err();
            let headings = crate::tools::metadata::note_inspect(
                &vault,
                crate::tools::metadata::NoteInspectParams {
                    path: "a-unsafe.md".into(),
                    fields: Some(vec!["headings".into()]),
                    expected_revision: Some(revision),
                    ..Default::default()
                },
            )
            .await
            .unwrap_err();
            for denied in [page, headings] {
                assert_eq!(denied.code, rmcp::model::ErrorCode::RESOURCE_NOT_FOUND);
                assert!(denied.data.is_none());
            }
            assert_eq!(vault.inner.index.read().unwrap().generation(), before);
            assert!(vault.read_index().get_note(unsafe_path).is_some());
            let first = search().await;
            let second = search().await;
            assert_eq!(
                calls() - before_calls,
                2,
                "each explicit query performs one inference without an automatic replay"
            );
            let code = first
                .as_ref()
                .err()
                .and_then(|error| error.data.as_ref())
                .and_then(|data| data["code"].as_str());
            assert_eq!(
                code,
                Some("semantic_warming"),
                "first={first:?}; second={second:?}; generation before={before}, after={}",
                vault.inner.index.read().unwrap().generation()
            );
            let second = second.unwrap();
            let results = &second.structured_content.as_ref().unwrap()["results"];
            assert_eq!(results.as_array().unwrap().len(), 1);
            assert_eq!(results[0]["path"], "b-allowed.md");
            assert!(
                !serde_json::to_string(&second)
                    .unwrap()
                    .contains("private replacement marker")
            );
            assert!(vault.inner.index.read().unwrap().generation() > before);
        }
    }

    #[tokio::test]
    async fn idle_reconciliation_skips_projection_snapshots_but_checks_live_authority() {
        let (_directory, store, vault, client) = reader_fixture(false).await;
        let projection = Arc::clone(vault.projection.as_ref().unwrap());
        let generation = vault.inner.index.read().unwrap().generation();
        let entry = Arc::new(SemanticEntry {
            source: Arc::downgrade(&vault.inner),
            root: vault.authenticated_root().unwrap(),
            projection: Arc::clone(&projection),
            holders: Mutex::new(HashMap::from([(
                client.client_id().into(),
                ActivatedClient::from_client(&client),
            )])),
            runtime: OnceLock::new(),
            revisions: Mutex::new(SemanticRevisions {
                generation: Some(generation),
                notes: HashMap::new(),
            }),
            last_error: Mutex::new(None),
            diagnostic_id: "idle-fixture".into(),
        });
        // The unchanged pass must not need even a projection read lock. Holding
        // it distinguishes the generation fast path from snapshot-and-compare.
        let projection_guard = projection.index.write().unwrap();
        let (finished_tx, finished_rx) = std::sync::mpsc::channel();
        let worker_entry = Arc::clone(&entry);
        let worker = std::thread::spawn(move || {
            finished_tx.send(worker_entry.reconcile(false)).unwrap();
        });
        let result = finished_rx.recv_timeout(Duration::from_secs(1));
        drop(projection_guard);
        worker.join().unwrap();
        result
            .expect("idle pass must skip the projection snapshot")
            .unwrap();
        // Explicit activation keeps its repair pass even without source edits.
        entry.reconcile(true).unwrap();
        store
            .update(None, |policy| {
                policy.clients.get_mut("reader").unwrap().enabled = false;
                Ok(())
            })
            .unwrap();
        assert!(entry.reconcile(false).is_err());
    }

    #[tokio::test]
    async fn stale_retirement_cannot_remove_a_replacement_or_revalidated_owner() {
        let (_directory, store, vault, client) = reader_fixture(false).await;
        let make_entry = || {
            Arc::new(SemanticEntry {
                source: Arc::downgrade(&vault.inner),
                root: vault.authenticated_root().unwrap(),
                projection: Arc::clone(vault.projection.as_ref().unwrap()),
                holders: Mutex::new(HashMap::new()),
                runtime: OnceLock::new(),
                revisions: Mutex::new(SemanticRevisions::default()),
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
