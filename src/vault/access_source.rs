//! Delegated source admission over the retained vault owner.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::access::clients::{Action, DecisionReason};
use crate::access::store::AuthorizedClient;
use crate::config::Config;
use crate::error::{VaultError, VaultResult};
use crate::models::FileStat;

use super::fs::{MoveResult, VaultRoot};
use super::{Vault, path, source, source_work};

impl Vault {
    /// Open one local source owner without activating an unrestricted search provider.
    pub async fn open_authenticated(config: &Config) -> VaultResult<Self> {
        if config.sftp_location()?.is_some() {
            return Err(VaultError::Authorization(
                "authenticated clients require a local vault source".into(),
            ));
        }
        if config.mcp_data_dir.is_none() {
            return Err(VaultError::Authorization(
                "authenticated clients require external machine data".into(),
            ));
        }
        let root = VaultRoot::open(&config.vault_path)?;
        let machine = config
            .mcp_data_dir
            .as_ref()
            .ok_or_else(|| VaultError::Authorization("external machine data is missing".into()))?
            .canonicalize()?;
        if machine.starts_with(root.path()) {
            return Err(VaultError::Authorization(
                "authenticated machine data must be outside the vault".into(),
            ));
        }
        let mut source_config = config.clone();
        source_config.tantivy = false;
        source_config.embeddings = false;
        Self::open_inner(&source_config, None, false, Some(config.clone())).await
    }

    /// Select an immutable client policy while retaining shared source mutation ownership.
    pub fn with_access(&self, client: AuthorizedClient) -> VaultResult<Self> {
        if !self.is_authenticated() || self.is_remote() {
            return Err(VaultError::Authorization(
                "client authorization requires an authenticated local source owner".into(),
            ));
        }
        client.current()?;
        let source = self.inner.filesystem.as_ref().ok_or_else(|| {
            VaultError::Authorization("authenticated local source is missing".into())
        })?;
        let named = VaultRoot::open(&client.source().canonical_path)?;
        if source.path().as_os_str() != client.source().canonical_path.as_os_str()
            || !source.same_root(&named)?
        {
            return Err(VaultError::Authorization(
                "client policy belongs to a different source; reopen the authenticated vault"
                    .into(),
            ));
        }
        client.current()?;
        let mut view = self.clone();
        view.access = Some(client);
        let projection = view.project_access()?;
        tracing::debug!(policy = %view.require_access()?.read_policy_key(), "selected authorized corpus");
        view.projection = Some(projection);
        Ok(view)
    }

    pub fn is_authenticated(&self) -> bool {
        self.inner.authenticated_config.is_some()
    }

    pub fn access_client(&self) -> Option<&AuthorizedClient> {
        self.access.as_ref()
    }

    pub(crate) fn check_access_current(&self) -> VaultResult<()> {
        if self.is_authenticated() {
            self.require_access()?.current()?;
        }
        Ok(())
    }

    pub(crate) async fn check_access_current_async(&self) -> VaultResult<()> {
        if self.is_authenticated() {
            self.require_access()?.current_async().await?;
        }
        Ok(())
    }

    fn require_access(&self) -> VaultResult<&AuthorizedClient> {
        self.access.as_ref().ok_or_else(|| {
            VaultError::Authorization("authenticated invocation context is missing".into())
        })
    }

    pub(crate) fn authenticated_root(&self) -> VaultResult<VaultRoot> {
        self.inner
            .filesystem
            .as_ref()
            .map(VaultRoot::restricted_content)
            .ok_or_else(|| {
                VaultError::Authorization("authenticated local source is missing".into())
            })
    }

    fn project_access(&self) -> VaultResult<Arc<super::authorized_index::AuthorizedProjection>> {
        let client = self.require_access()?;
        let permit = client.permit()?;
        let now = chrono::Utc::now().timestamp();
        let readable_policies = permit
            .snapshot()
            .policy
            .clients
            .values()
            .filter(|policy| {
                policy.enabled
                    && policy.credentials.iter().any(|credential| {
                        !credential.revoked
                            && credential.expires_at.is_none_or(|expiry| expiry > now)
                    })
            })
            .map(|policy| policy.read_policy_hash())
            .collect();
        self.inner
            .authorized_indices
            .retain_read_policies(&readable_policies)?;
        drop(permit);
        let config = self.inner.authenticated_config.as_ref().ok_or_else(|| {
            VaultError::Authorization("authenticated source configuration is missing".into())
        })?;
        let reconcile_path = std::cell::RefCell::new(None);
        let projected = self.inner.authorized_indices.project(
            &self.inner.index,
            client.policy(),
            config.tantivy,
            |path| {
                let content = self.read_access_native(path, None);
                // The snapshot already admits this exact key under the current
                // immutable read policy. A hidden denial here means its source
                // eligibility changed, such as a new hardlink or denied alias.
                // Fresh raw-key authority is checked again before reconciliation.
                if content.as_ref().is_err_and(|error| {
                    error.is_not_found()
                        || matches!(error, VaultError::AccessDenied { visible: false, .. })
                }) {
                    *reconcile_path.borrow_mut() = Some(path.to_owned());
                }
                content
            },
        );
        let repair = match &projected {
            Err(VaultError::SourceChanged { path, .. }) => Some(path.clone()),
            Err(error)
                if error.is_not_found()
                    || matches!(error, VaultError::AccessDenied { visible: false, .. }) =>
            {
                reconcile_path.into_inner()
            }
            _ => None,
        };
        let Some(path) = repair else {
            return projected;
        };

        // A source observation can be superseded before projection reads it.
        // Reconcile that exact admitted key in the shared owner, then require
        // one complete current projection. Further races and other errors remain
        // visible; no previous or partial catalog is returned as a substitute.
        self.reconcile_access_source(&path)?;
        self.inner.authorized_indices.project(
            &self.inner.index,
            client.policy(),
            config.tantivy,
            |path| self.read_access_native(path, None),
        )
    }

    fn reconcile_access_source(&self, path: &Path) -> VaultResult<()> {
        let client = self.require_access()?;
        client.check(Action::Read, path, path)?;
        self.reindex_source_local(path)?;
        client.current()?;
        #[cfg(has_embeddings)]
        self.notify_authenticated_semantics();
        Ok(())
    }

    pub(crate) async fn search_with_reconciliation<T, F, Fut>(
        &self,
        mut search: F,
    ) -> VaultResult<T>
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = VaultResult<T>>,
    {
        let result = search().await;
        if !self.is_authenticated() {
            return result;
        }
        let path = match &result {
            Err(VaultError::SourceChanged { path, .. })
            | Err(VaultError::NoteNotFound(path))
            | Err(VaultError::DirectoryNotFound(path))
            | Err(VaultError::AccessDenied {
                path,
                visible: false,
            }) => path.clone(),
            _ => return result,
        };
        let vault = self.clone();
        source_work(move || {
            vault.reconcile_access_source(&path)?;
            vault.refresh_authorized_projection()
        })
        .await?;
        // Ranking and limits must use the complete repaired corpus. In
        // particular, dropping a failed BM25 hit would retain its statistics.
        search().await
    }

    pub(crate) fn read_access_indexed_native(&self, path: &Path) -> VaultResult<String> {
        let expected = {
            let index = self.read_index();
            match index.source_revision(path) {
                Some(revision) => revision.to_owned(),
                None if index.get_note(path).is_none() => {
                    return Err(VaultError::NoteNotFound(path.to_owned()));
                }
                None => {
                    return Err(VaultError::Other(
                        "authorized source revision requires reconciliation".into(),
                    ));
                }
            }
        };
        self.read_access_revision_native(path, &expected)
    }

    pub(crate) fn read_access_revision_native(
        &self,
        path: &Path,
        expected: &str,
    ) -> VaultResult<String> {
        let client = self.require_access()?;
        client.check(Action::Read, path, path)?;
        let content = self
            .authenticated_root()?
            .read_observed(path)
            .map_err(|error| {
                if super::fs::is_missing_entry(&error) {
                    VaultError::NoteNotFound(path.to_owned())
                } else {
                    error
                }
            })?;
        client.check(Action::Read, path, path)?;
        let current = source::source_revision(&content);
        if current != expected {
            return Err(VaultError::SourceChanged {
                path: path.to_owned(),
                expected_revision: expected.to_owned(),
                current_revision: current,
            });
        }
        Ok(content)
    }

    pub(crate) fn refresh_authorized_projection(&self) -> VaultResult<()> {
        if self.is_authenticated() {
            let current = self.project_access()?;
            if let Some(selected) = &self.projection
                && (!selected.is_ready() || !Arc::ptr_eq(selected, &current))
            {
                return Err(VaultError::Authorization(
                    "authorized corpus changed; reopen the authorized view".into(),
                ));
            }
        }
        Ok(())
    }

    fn resolve_access_path(
        &self,
        root: &VaultRoot,
        requested: &Path,
        allow_missing: bool,
    ) -> VaultResult<(PathBuf, PathBuf)> {
        let client = self.require_access()?;
        self.check_access_current()?;
        let requested = path::normalize_relative(requested)?;
        let decision = client.policy().decision(Action::Read, &requested);
        if decision.reason == DecisionReason::InvalidPath {
            return Err(VaultError::InvalidPath(
                "expected a normalized vault-relative content path".into(),
            ));
        }
        if matches!(
            decision.reason,
            DecisionReason::Denied | DecisionReason::Disabled | DecisionReason::ReservedPath
        ) {
            return Err(VaultError::AccessDenied {
                path: requested,
                visible: false,
            });
        }
        let resolved = if allow_missing {
            root.resolve_access_for_write(&requested)
        } else {
            root.resolve_access_existing(&requested)
        };
        let (logical, target) = resolved.map_err(|error| {
            if !decision.allowed && !client.policy().allows_container(&requested) {
                VaultError::AccessDenied {
                    path: requested.clone(),
                    visible: false,
                }
            } else {
                error
            }
        })?;
        Ok((logical, target.relative))
    }

    pub(crate) fn read_access_native(
        &self,
        requested: &Path,
        max_bytes: Option<usize>,
    ) -> VaultResult<String> {
        let client = self.require_access()?;
        let root = self.authenticated_root()?;
        let (requested, actual) = self.resolve_access_path(&root, requested, false)?;
        self.check_scope_path(&requested)?;
        {
            let permit = client.permit()?;
            permit.check(Action::Read, &requested, &actual)?;
        }
        let content = match max_bytes {
            Some(max_bytes) => root.read_exact_bounded(&actual, max_bytes)?,
            None => root.read_observed(&actual)?,
        };
        self.check_content_scope(&actual, &content)?;
        client.check(Action::Read, &requested, &actual)?;
        Ok(content)
    }

    /// Admit only paths from server-ranked semantic evidence. Raw caller paths
    /// use normal source reads, whose hidden-denial behavior is independent of
    /// index history. Scored evidence may retain an unavailable source, but never
    /// a resource outside the current client's authority.
    pub(crate) async fn check_access_result(&self, path: &Path) -> VaultResult<()> {
        let vault = self.clone();
        let path = path.to_owned();
        source_work(move || {
            let client = vault.require_access()?;
            client.check(Action::Read, &path, &path)?;
            let root = vault.authenticated_root()?;
            let physical = (|| -> VaultResult<()> {
                match root.resolve_existing(&path) {
                    Ok(actual) => {
                        client.check(Action::Read, &path, &actual.relative)?;
                        if let Err(error @ VaultError::AccessDenied { .. }) =
                            root.validate_access_file_exact(&actual.relative)
                        {
                            return Err(error);
                        }
                    }
                    Err(VaultError::Io(_)) => (),
                    Err(error) if error.is_not_found() => (),
                    Err(error) => return Err(error),
                }
                Ok(())
            })();
            if matches!(
                physical,
                Err(VaultError::AccessDenied { visible: false, .. })
            ) && vault.read_index().get_note(&path).is_some()
            {
                // The logical grant is current, but the exact ranked source is
                // no longer eligible. Repair that observation and require fresh
                // ranking; never replay provider inference inside this query.
                vault.reconcile_access_source(&path)?;
                vault.refresh_authorized_projection()?;
                return Err(VaultError::storage(
                    "search",
                    "",
                    crate::error::StorageErrorKind::Warming,
                    "authorized semantic source was reconciled; submit a fresh query",
                ));
            }
            physical?;
            // Missing/unreadable/changed source is represented by the semantic
            // renderer's revision-bound context_error rather than mixed bytes.
            client.current()?;
            Ok(())
        })
        .await
    }

    pub(crate) async fn access_is_file(&self, requested: &Path) -> VaultResult<bool> {
        let vault = self.clone();
        let requested = requested.to_owned();
        source_work(move || {
            let client = vault.require_access()?;
            let root = vault.authenticated_root()?;
            let (requested, actual) = match vault.resolve_access_path(&root, &requested, false) {
                Ok(resolved) => resolved,
                Err(error) if error.is_not_found() => return Ok(false),
                Err(error) => return Err(error),
            };
            client.check(Action::Read, &requested, &actual)?;
            match root.validate_access_file_exact(&actual) {
                Ok(()) => Ok(true),
                Err(error) if error.is_not_found() => Ok(false),
                Err(error) => Err(error),
            }
        })
        .await
    }

    pub(crate) async fn access_file_stat(&self, requested: &Path) -> VaultResult<FileStat> {
        let vault = self.clone();
        let requested = requested.to_owned();
        source_work(move || {
            let client = vault.require_access()?;
            let root = vault.authenticated_root()?;
            let (requested, actual) = vault.resolve_access_path(&root, &requested, false)?;
            client.check(Action::Read, &requested, &actual)?;
            let stat = root.file_stat_exact(&actual)?;
            client.check(Action::Read, &requested, &actual)?;
            Ok(stat)
        })
        .await
    }

    pub(crate) async fn access_canonical_existing(&self, requested: &Path) -> VaultResult<PathBuf> {
        let vault = self.clone();
        let requested = requested.to_owned();
        source_work(move || {
            let client = vault.require_access()?;
            client.current()?;
            let root = vault.authenticated_root()?;
            let (requested, actual) = vault.resolve_access_path(&root, &requested, false)?;
            if !client.policy().allows_container(&requested)
                && !client.policy().allows(Action::Read, &requested)
            {
                return Err(VaultError::AccessDenied {
                    path: requested,
                    visible: false,
                });
            }
            if root.entry_metadata_exact(&actual)?.is_dir() {
                if !client.policy().allows_container(&actual) {
                    return Err(VaultError::AccessDenied {
                        path: requested,
                        visible: false,
                    });
                }
                client.current()?;
            } else {
                client.check(Action::Read, &requested, &actual)?;
                root.validate_access_file_exact(&actual)?;
            }
            Ok(actual)
        })
        .await
    }

    pub(crate) async fn access_list_files(
        &self,
        requested: &Path,
        recursive: bool,
        glob: Option<&str>,
        regular_only: bool,
    ) -> VaultResult<Vec<PathBuf>> {
        let vault = self.clone();
        let requested = requested.to_owned();
        let glob = glob.map(str::to_owned);
        source_work(move || {
            let client = vault.require_access()?;
            client.current()?;
            let root = vault.authenticated_root()?;
            let (requested, actual) = vault
                .resolve_access_path(&root, &requested, false)
                .map_err(|error| {
                    if super::fs::is_missing_entry(&error) {
                        VaultError::DirectoryNotFound(requested.clone())
                    } else {
                        error
                    }
                })?;
            if !client.policy().allows_container(&requested) {
                return Err(VaultError::AccessDenied {
                    path: requested,
                    visible: false,
                });
            }
            if !client.policy().allows_container(&actual) {
                return Err(VaultError::AccessDenied {
                    path: requested,
                    visible: false,
                });
            }
            let matcher = glob
                .as_deref()
                .map(|pattern| {
                    globset::Glob::new(pattern)
                        .map(|glob| glob.compile_matcher())
                        .map_err(|error| {
                            VaultError::InvalidPath(format!("invalid glob pattern: {error}"))
                        })
                })
                .transpose()?;
            let source_base = actual.clone();
            let mut pending = vec![actual];
            let mut paths = Vec::new();
            while let Some(directory) = pending.pop() {
                client.current()?;
                let directory_handle = match root.open_dir_exact(&directory) {
                    Ok(directory) => directory,
                    Err(error)
                        if directory != source_base && super::fs::skip_changed_entry(&error) =>
                    {
                        continue;
                    }
                    Err(error) if super::fs::is_missing_entry(&error) => {
                        return Err(VaultError::DirectoryNotFound(requested.clone()));
                    }
                    Err(error) => return Err(error),
                };
                let entries = super::fs::dir_entries(&directory_handle)?;
                for entry in entries {
                    let entry = entry?;
                    let raw = directory.join(entry.file_name());
                    let Ok(path) = path::portable_native(&raw) else {
                        continue;
                    };
                    if !super::exclude::is_visible_path(&path) {
                        continue;
                    }
                    let suffix = path
                        .strip_prefix(&source_base)
                        .map_err(|_| VaultError::OutsideVault(path.clone()))?;
                    let logical = path::logical_join(&requested, suffix)?;
                    if !client.policy().allows_container(&logical)
                        && !client.policy().allows(Action::Read, &logical)
                    {
                        continue;
                    }
                    if !client.policy().allows_container(&path)
                        && !client.policy().allows(Action::Read, &path)
                    {
                        continue;
                    }
                    let kind = entry.file_type()?;
                    let visible = if kind.is_dir() {
                        let visible = client.policy().allows_container(&path)
                            && client.policy().allows_container(&logical);
                        if visible && recursive {
                            pending.push(path.clone());
                        }
                        visible && !regular_only
                    } else {
                        match root.resolve_existing(&path) {
                            Ok(target) => {
                                let metadata = match root.entry_metadata_exact(&target.relative) {
                                    Ok(metadata) => metadata,
                                    Err(VaultError::AccessDenied { .. }) => continue,
                                    Err(error) if error.is_not_found() => continue,
                                    Err(error) => return Err(error),
                                };
                                if metadata.is_dir() {
                                    !regular_only
                                        && client.policy().allows_container(&logical)
                                        && client.policy().allows_container(&target.relative)
                                } else if metadata.is_file()
                                    && client.policy().allows(Action::Read, &logical)
                                    && client.policy().allows(Action::Read, &target.relative)
                                {
                                    match root.validate_access_file_exact(&target.relative) {
                                        Ok(()) => true,
                                        Err(VaultError::AccessDenied { .. }) => false,
                                        Err(error) if error.is_not_found() => false,
                                        Err(error) => return Err(error),
                                    }
                                } else {
                                    false
                                }
                            }
                            Err(
                                VaultError::AccessDenied { .. }
                                | VaultError::OutsideVault(_)
                                | VaultError::InvalidPath(_),
                            ) => false,
                            Err(error) if error.is_not_found() => false,
                            Err(error) => return Err(error),
                        }
                    };
                    if visible
                        && matcher
                            .as_ref()
                            .is_none_or(|matcher| matcher.is_match(&path))
                    {
                        paths.push(path);
                    }
                }
            }
            client.current()?;
            paths.sort();
            Ok(paths)
        })
        .await
    }

    async fn access_write_source(
        &self,
        requested: &Path,
        content: &str,
        expected: Option<&str>,
        create_only: bool,
        append: bool,
    ) -> VaultResult<PathBuf> {
        let vault = self.clone();
        let requested = requested.to_owned();
        let content = content.to_owned();
        let expected = expected.map(str::to_owned);
        source_work(move || {
            let root = vault.authenticated_root()?;
            let client = vault.require_access()?;
            let (requested, actual) = vault.resolve_access_path(&root, &requested, true)?;
            client.check(Action::Read, &requested, &actual)?;
            let exists = match root.entry_metadata_exact(&actual) {
                Ok(metadata) => {
                    if !metadata.is_file() {
                        return Err(VaultError::AccessDenied {
                            path: requested,
                            visible: false,
                        });
                    }
                    true
                }
                Err(error) if error.is_not_found() => false,
                Err(error) => return Err(error),
            };
            let action = if create_only || !exists {
                Action::Create
            } else {
                Action::Update
            };
            {
                let permit = client.permit()?;
                permit.check(action, &requested, &actual)?;
                if expected.is_some() || (append && exists) {
                    permit.check(Action::Read, &requested, &actual)?;
                }
            }
            if let Some(expected) = expected {
                let current = root.read_observed(&actual)?;
                source::SourceSnapshot::new(current).check_revision(&actual, Some(&expected))?;
                client.check(Action::Update, &requested, &actual)?;
            }
            if create_only {
                root.create_file_exact(&actual, &content)
            } else if append {
                root.append_file_exact(&actual, &content)
            } else {
                root.write_file_exact(&actual, &content)
            }
        })
        .await
    }

    pub(crate) async fn write_source(&self, path: &Path, content: &str) -> VaultResult<PathBuf> {
        if self.is_authenticated() {
            self.access_write_source(path, content, None, false, false)
                .await
        } else {
            self.storage().await?.write_file(path, content).await
        }
    }

    pub(crate) async fn create_source(&self, path: &Path, content: &str) -> VaultResult<PathBuf> {
        if self.is_authenticated() {
            self.access_write_source(path, content, None, true, false)
                .await
        } else {
            self.storage().await?.create_file(path, content).await
        }
    }

    pub(crate) async fn replace_source(
        &self,
        path: &Path,
        content: &str,
        expected: &str,
    ) -> VaultResult<PathBuf> {
        if self.is_authenticated() {
            self.access_write_source(path, content, Some(expected), false, false)
                .await
        } else {
            self.storage()
                .await?
                .write_file_if_revision(path, content, expected)
                .await
        }
    }

    pub(crate) async fn append_source(&self, path: &Path, content: &str) -> VaultResult<PathBuf> {
        if self.is_authenticated() {
            self.access_write_source(path, content, None, false, true)
                .await
        } else {
            let root =
                self.storage().await?.local_root().cloned().ok_or_else(|| {
                    VaultError::Other("native append requires local source".into())
                })?;
            let path = path.to_owned();
            let content = content.to_owned();
            source_work(move || root.append_file(&path, &content)).await
        }
    }

    pub(crate) async fn delete_source(&self, requested: &Path) -> VaultResult<PathBuf> {
        if !self.is_authenticated() {
            return self.storage().await?.delete_file(requested).await;
        }
        let vault = self.clone();
        let requested = requested.to_owned();
        source_work(move || {
            let client = vault.require_access()?;
            let root = vault.authenticated_root()?;
            let (logical, target) = vault.resolve_access_path(&root, &requested, false)?;
            let entry = root.resolve_entry(&requested, false)?;
            root.validate_access_file_exact(&target)?;
            {
                let permit = client.permit()?;
                permit.check(Action::Delete, &logical, &entry)?;
                permit.check(Action::Delete, &logical, &target)?;
            }
            root.delete_entry_exact(&entry)
        })
        .await
    }

    pub(crate) async fn move_source(
        &self,
        requested: &Path,
        destination: &Path,
    ) -> VaultResult<MoveResult> {
        if !self.is_authenticated() {
            return self
                .storage()
                .await?
                .move_file(requested, destination)
                .await;
        }
        let vault = self.clone();
        let requested = requested.to_owned();
        let destination = destination.to_owned();
        source_work(move || {
            let client = vault.require_access()?;
            let root = vault.authenticated_root()?;
            let (logical, target) = vault.resolve_access_path(&root, &requested, false)?;
            let (logical_to, to_target) = vault.resolve_access_path(&root, &destination, true)?;
            let entry = root.resolve_entry(&requested, false)?;
            let to = root.resolve_entry(&destination, true)?;
            root.validate_access_file_exact(&target)?;
            {
                let permit = client.permit()?;
                permit.check(Action::Delete, &logical, &entry)?;
                permit.check(Action::Delete, &logical, &target)?;
                permit.check(Action::Read, &logical, &target)?;
                permit.check(Action::Create, &logical_to, &to)?;
                permit.check(Action::Create, &logical_to, &to_target)?;
            }
            root.move_entry_exact(&entry, &to, &target)
        })
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::access::clients::{ClientPolicy, Deny, Grant, Role};
    use crate::access::store::AccessStore;
    use crate::models::NotePeriod;

    struct Fixture {
        _temporary: tempfile::TempDir,
        root: PathBuf,
        store: AccessStore,
        vault: Vault,
    }

    async fn fixture(grants: Vec<Grant>, denies: Vec<Deny>) -> Fixture {
        fixture_with_tantivy(grants, denies, false).await
    }

    async fn fixture_with_tantivy(grants: Vec<Grant>, denies: Vec<Deny>, tantivy: bool) -> Fixture {
        let temporary = tempfile::tempdir().unwrap();
        let parent = temporary.path().canonicalize().unwrap();
        let root = parent.join("vault");
        std::fs::create_dir_all(root.join("Shared")).unwrap();
        std::fs::create_dir_all(root.join("Private")).unwrap();
        std::fs::create_dir_all(root.join(".obsidian")).unwrap();
        std::fs::write(root.join("Shared/note.md"), "original").unwrap();
        std::fs::write(root.join("Private/note.md"), "private source").unwrap();
        std::fs::write(root.join("Private/template.md"), "private template").unwrap();
        std::fs::write(
            root.join(".obsidian/daily-notes.json"),
            r#"{"folder":"Shared/Daily","template":"Private/template"}"#,
        )
        .unwrap();
        let data = parent.join("state/data");
        let store =
            AccessStore::initialize(&parent.join("state/access.json"), &root, &data).unwrap();
        store
            .update(None, |policy| {
                policy.clients.insert(
                    "client".into(),
                    ClientPolicy {
                        grants,
                        denies,
                        ..Default::default()
                    },
                );
                Ok(())
            })
            .unwrap();
        let output = parent.join("state/client.token");
        store.issue_token("client", &output, None).unwrap();
        let token = std::fs::read_to_string(output).unwrap();
        let client = store.authenticate(token.trim()).unwrap();
        let mut config = crate::test_helpers::test_config(&root);
        config.mcp_data_dir = Some(data);
        config.tantivy = tantivy;
        let vault = Vault::open_authenticated(&config)
            .await
            .unwrap()
            .with_access(client)
            .unwrap();
        Fixture {
            _temporary: temporary,
            root,
            store,
            vault,
        }
    }

    fn directory(path: &str, role: Role) -> Grant {
        Grant {
            id: "grant".into(),
            directory: Some(path.into()),
            file: None,
            role,
        }
    }

    fn deny_private() -> Deny {
        Deny {
            id: "private".into(),
            directory: Some("Private".into()),
            file: None,
        }
    }

    #[tokio::test]
    async fn bulk_path_validation_does_not_observe_hidden_source_shapes() {
        use crate::tools::notes::{NoteReadManyParams, NoteReadManySkipReason, note_read_many};

        let f = fixture(vec![directory("Shared", Role::Reader)], vec![]).await;
        for hidden in ["Private/note.md/child", "Private/missing.md/child"] {
            f.vault.validate_path(Path::new(hidden)).await.unwrap();
            let output = note_read_many(
                &f.vault,
                NoteReadManyParams {
                    paths: Some(vec![hidden.into(), "Shared/note.md".into()]),
                    ..Default::default()
                },
            )
            .await
            .unwrap()
            .0;
            assert_eq!(output.notes.len(), 1);
            assert_eq!(output.notes[0].content, "original");
            assert_eq!(output.skipped.len(), 1);
            assert_eq!(output.skipped[0].reason, NoteReadManySkipReason::NotFound);
        }
        for invalid in ["../outside.md", "/outside.md", "Shared/control\n.md"] {
            let error = note_read_many(
                &f.vault,
                NoteReadManyParams {
                    paths: Some(vec!["Shared/note.md".into(), invalid.into()]),
                    max_files: Some(1),
                    ..Default::default()
                },
            )
            .await
            .err()
            .expect("all explicit paths must pass syntax validation");
            assert_eq!(error.code, rmcp::model::ErrorCode::INVALID_PARAMS);
        }
    }

    #[tokio::test]
    async fn a_readable_file_is_not_a_directory_for_listing_or_bulk_reads() {
        use crate::tools::notes::{NoteReadManyParams, note_read_many};

        let f = fixture(vec![directory("Shared", Role::Reader)], vec![]).await;
        for path in ["Shared/note.md", "Shared/note.md/child"] {
            assert!(matches!(
                f.vault.list_files(Path::new(path), true, None).await,
                Err(VaultError::DirectoryNotFound(requested)) if requested == Path::new(path)
            ));
            let error = note_read_many(
                &f.vault,
                NoteReadManyParams {
                    dir: Some(path.into()),
                    ..Default::default()
                },
            )
            .await
            .err()
            .expect("a file cannot be read as a directory");
            assert_eq!(error.code, rmcp::model::ErrorCode::RESOURCE_NOT_FOUND);
        }
        assert!(matches!(
            f.vault
                .list_files(Path::new("Private/note.md"), true, None)
                .await,
            Err(VaultError::AccessDenied { visible: false, .. })
        ));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn directory_aliases_return_canonical_listing_keys_and_metadata() {
        use crate::tools::navigation::{VaultListParams, vault_list};

        let f = fixture(vec![directory("", Role::Reader)], vec![]).await;
        std::os::unix::fs::symlink("Shared", f.root.join("Alias")).unwrap();
        assert_eq!(
            f.vault
                .list_files(Path::new("Alias"), true, None)
                .await
                .unwrap(),
            vec![PathBuf::from("Shared/note.md")]
        );
        let listing = vault_list(
            &f.vault,
            VaultListParams {
                path: Some("Alias".into()),
                include_metadata: Some(true),
                glob: Some("Shared/*.md".into()),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let entries: Vec<serde_json::Value> =
            serde_json::from_str(crate::test_helpers::extract_text(&listing)).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0]["path"], "Shared/note.md");
        assert_eq!(entries[0]["title"], "note");

        let tree = vault_list(
            &f.vault,
            VaultListParams {
                path: Some("Alias".into()),
                format: Some("tree".into()),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(
            crate::test_helpers::extract_text(&tree),
            "Shared\n└── note.md"
        );

        std::fs::write(f.root.join("Shared/2026-11-13.md"), "daily").unwrap();
        let config = crate::models::PeriodicNoteConfig {
            folder: Some("Alias".into()),
            format: "YYYY-MM-DD".into(),
            template: None,
        };
        assert_eq!(
            f.vault.list_periodic(&config, 10).await.unwrap(),
            vec![PathBuf::from("Alias/2026-11-13.md")]
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn a_held_projection_refresh_does_not_block_async_catalog_consumers() {
        let f = fixture(vec![directory("Shared", Role::Reader)], vec![]).await;
        std::fs::write(f.root.join("Shared/note.md"), "updated source").unwrap();
        f.vault
            .write_index()
            .refresh_file(
                f.vault.inner.filesystem.as_ref().unwrap(),
                Path::new("Shared/note.md"),
                f.vault.exclude(),
                None,
                #[cfg(has_embeddings)]
                false,
            )
            .unwrap();

        let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
        let entered_tx = std::sync::Mutex::new(Some(entered_tx));
        let release = Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
        let held_release = Arc::clone(&release);
        let owner = f.vault.clone();
        let refresh = tokio::task::spawn_blocking(move || {
            owner.inner.authorized_indices.project(
                &owner.inner.index,
                owner.access_client().unwrap().policy(),
                false,
                |path| {
                    if let Some(sender) = entered_tx.lock().unwrap().take() {
                        sender.send(()).unwrap();
                    }
                    let (released, ready) = &*held_release;
                    let mut released = released.lock().unwrap();
                    while !*released {
                        released = ready.wait(released).unwrap();
                    }
                    owner.read_access_native(path, None)
                },
            )
        });
        entered_rx.await.unwrap();
        // A native watchdog bounds the regression even if an async worker is blocked.
        let watchdog_release = Arc::clone(&release);
        let watchdog = std::thread::spawn(move || {
            let (released, ready) = &*watchdog_release;
            let (mut released, _) = ready
                .wait_timeout_while(
                    released.lock().unwrap(),
                    std::time::Duration::from_secs(2),
                    |released| !*released,
                )
                .unwrap();
            *released = true;
            ready.notify_all();
        });
        let reader = f.vault.clone();
        let (polling_tx, polling_rx) = tokio::sync::oneshot::channel();
        let started = std::time::Instant::now();
        let operation = tokio::spawn(async move {
            polling_tx.send(()).unwrap();
            reader.get_note_metadata(Path::new("Shared/note.md")).await
        });
        polling_rx.await.unwrap();
        tokio::task::yield_now().await;
        let elapsed = started.elapsed();
        let (released, ready) = &*release;
        *released.lock().unwrap() = true;
        ready.notify_all();
        watchdog.join().unwrap();
        refresh.await.unwrap().unwrap();
        operation.await.unwrap().unwrap();
        assert!(
            elapsed < std::time::Duration::from_secs(1),
            "catalog refresh occupied the async worker for {elapsed:?}"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn a_policy_writer_does_not_block_async_source_or_mutation_admission() {
        let f = fixture(vec![directory("Shared", Role::Editor)], vec![]).await;
        for mutation in [false, true] {
            let store = f.store.clone();
            let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
            let (release, released) = std::sync::mpsc::channel();
            let writer = std::thread::spawn(move || {
                store.update(None, |_| {
                    entered_tx.send(()).unwrap();
                    // Bound a synchronous-admission regression without a hung test.
                    let _ = released.recv_timeout(std::time::Duration::from_secs(2));
                    Ok(())
                })
            });
            entered_rx.await.unwrap();
            let operation = async {
                if mutation {
                    f.vault
                        .write_note(Path::new("Shared/note.md"), "updated")
                        .await
                } else {
                    f.vault.storage().await.map(drop)
                }
            };
            tokio::pin!(operation);
            let started = std::time::Instant::now();
            tokio::select! {
                result = &mut operation => panic!("source admission bypassed the held policy writer: {result:?}"),
                _ = tokio::time::sleep(std::time::Duration::from_millis(20)) => release.send(()).unwrap(),
            }
            assert!(started.elapsed() < std::time::Duration::from_secs(1));
            tokio::time::timeout(std::time::Duration::from_secs(2), operation)
                .await
                .unwrap()
                .unwrap();
            writer.join().unwrap().unwrap();
        }
    }

    async fn source_view_reconciles_intervening_change(deleted: bool) {
        let f = source_with_pending_readable_revision().await;
        if deleted {
            std::fs::remove_file(f.root.join("Shared/note.md")).unwrap();
        } else {
            std::fs::write(
                f.root.join("Shared/note.md"),
                "---\ntags: [current]\n---\ncurrent source",
            )
            .unwrap();
        }
        let selected = f
            .vault
            .with_access(f.vault.access_client().unwrap().clone())
            .expect("a stable intervening source change must reconcile without a watcher");
        assert_eq!(
            selected
                .read_note(Path::new("Shared/other.md"))
                .await
                .unwrap(),
            "unrelated source"
        );
        if deleted {
            assert_eq!(selected.vault_stats().unwrap().total_notes, 1);
            assert!(matches!(
                selected
                    .get_note_metadata(Path::new("Shared/note.md"))
                    .await,
                Err(VaultError::NoteNotFound(_))
            ));
        } else {
            assert_eq!(selected.vault_stats().unwrap().total_notes, 2);
            let notes = selected.search_by_tag("current").unwrap();
            assert_eq!(notes.len(), 1);
            assert_eq!(notes[0].path, Path::new("Shared/note.md"));
            assert_eq!(
                f.vault
                    .inner
                    .index
                    .read()
                    .unwrap()
                    .get_note(Path::new("Shared/note.md"))
                    .unwrap()
                    .tags,
                ["current"]
            );
        }
    }

    #[tokio::test]
    async fn source_view_reconciles_a_note_edited_after_its_index_observation() {
        source_view_reconciles_intervening_change(false).await;
    }

    #[tokio::test]
    async fn source_view_reconciles_a_note_deleted_after_its_index_observation() {
        source_view_reconciles_intervening_change(true).await;
    }

    async fn source_with_pending_readable_revision() -> Fixture {
        let f = fixture(
            vec![directory("Shared", Role::Editor)],
            vec![deny_private()],
        )
        .await;
        assert!(!f.vault.watch_enabled());
        f.vault
            .create_note(Path::new("Shared/other.md"), "unrelated source", None)
            .await
            .unwrap();
        std::fs::write(
            f.root.join("Shared/note.md"),
            "---\ntags: [indexed]\n---\nupdated source",
        )
        .unwrap();
        let generation = f.vault.inner.index.read().unwrap().generation();
        f.vault
            .write_index()
            .refresh_file(
                f.vault.inner.filesystem.as_ref().unwrap(),
                Path::new("Shared/note.md"),
                f.vault.exclude(),
                None,
                #[cfg(has_embeddings)]
                false,
            )
            .unwrap();
        assert!(f.vault.inner.index.read().unwrap().generation() > generation);
        assert!(
            f.vault
                .projection
                .as_ref()
                .unwrap()
                .index
                .read()
                .unwrap()
                .get_note(Path::new("Shared/note.md"))
                .unwrap()
                .tags
                .is_empty()
        );
        f
    }

    async fn assert_unsafe_source_reconciles(f: &Fixture) {
        let selected = f
            .vault
            .with_access(f.vault.access_client().unwrap().clone())
            .expect("an ineligible changed source must not disable unrelated readable notes");
        assert_eq!(
            selected
                .read_note(Path::new("Shared/other.md"))
                .await
                .unwrap(),
            "unrelated source"
        );
        assert_eq!(selected.vault_stats().unwrap().total_notes, 1);
        assert!(selected.search_by_tag("indexed").unwrap().is_empty());
        assert!(
            f.vault
                .inner
                .index
                .read()
                .unwrap()
                .get_note(Path::new("Shared/note.md"))
                .is_none()
        );
        assert!(matches!(
            selected.read_note(Path::new("Shared/note.md")).await,
            Err(VaultError::AccessDenied { visible: false, .. })
        ));
    }

    #[tokio::test]
    async fn source_view_reconciles_unsafe_multilink_after_its_index_observation() {
        let f = source_with_pending_readable_revision().await;
        std::fs::hard_link(
            f.root.join("Shared/note.md"),
            f.root.join("Private/linked.md"),
        )
        .unwrap();
        assert_unsafe_source_reconciles(&f).await;
        assert_eq!(
            std::fs::read_to_string(f.root.join("Private/linked.md")).unwrap(),
            "---\ntags: [indexed]\n---\nupdated source"
        );
    }

    #[tokio::test]
    async fn source_reconciliation_preserves_nonconcealed_read_failures() {
        let f = source_with_pending_readable_revision().await;
        std::fs::write(f.root.join("Shared/note.md"), [0xff]).unwrap();
        assert!(matches!(
            f.vault.with_access(f.vault.access_client().unwrap().clone()),
            Err(VaultError::Io(error)) if error.kind() == std::io::ErrorKind::InvalidData
        ));
        assert_eq!(
            f.vault
                .inner
                .index
                .read()
                .unwrap()
                .get_note(Path::new("Shared/note.md"))
                .unwrap()
                .tags,
            ["indexed"]
        );
        assert!(!f.vault.projection.as_ref().unwrap().is_ready());
    }

    async fn lexical_tool_query(
        vault: &Vault,
        tantivy: bool,
    ) -> Result<rmcp::model::CallToolResult, rmcp::ErrorData> {
        use crate::tools::search::{
            SearchRegexParams, SearchTextParams, search_regex, search_text,
        };
        if tantivy {
            search_text(
                vault,
                SearchTextParams {
                    query: "quokka".into(),
                    context_length: Some(0),
                    max_results: Some(1),
                    ..Default::default()
                },
            )
            .await
        } else {
            search_regex(
                vault,
                SearchRegexParams {
                    pattern: "quokka".into(),
                    context_length: Some(20),
                    max_results: Some(1),
                },
            )
            .await
        }
    }

    async fn lexical_query_reconciles_external_change(tantivy: bool, change: &str) {
        let f = fixture_with_tantivy(
            vec![directory("Shared", Role::Editor)],
            vec![deny_private()],
            tantivy,
        )
        .await;
        f.vault
            .write_note(Path::new("Shared/note.md"), "quokka quokka quokka")
            .await
            .unwrap();
        f.vault
            .create_note(
                Path::new("Shared/z.md"),
                "quokka retained source with enough additional words to keep its term frequency lower than the other matching note",
                None,
            )
            .await
            .unwrap();
        let generation = f.vault.inner.index.read().unwrap().generation();
        match change {
            "deleted" => std::fs::remove_file(f.root.join("Shared/note.md")).unwrap(),
            "hardlink" => {
                std::fs::hard_link(
                    f.root.join("Shared/note.md"),
                    f.root.join("Private/linked.md"),
                )
                .unwrap();
            }
            "alias" => {
                #[cfg(unix)]
                {
                    std::fs::remove_file(f.root.join("Shared/note.md")).unwrap();
                    std::os::unix::fs::symlink("../Private/note.md", f.root.join("Shared/note.md"))
                        .unwrap();
                }
                #[cfg(not(unix))]
                unreachable!("native alias fixture requires Unix");
            }
            "edited" => {
                std::fs::write(f.root.join("Shared/note.md"), "replacement source").unwrap()
            }
            _ => unreachable!("unknown source change"),
        }
        let selected = f
            .vault
            .with_access(f.vault.access_client().unwrap().clone())
            .unwrap();
        assert_eq!(f.vault.inner.index.read().unwrap().generation(), generation);
        let result = lexical_tool_query(&selected, tantivy)
            .await
            .expect("the complete query must reconcile its stale candidate");
        let results = &result.structured_content.as_ref().unwrap()["results"];
        assert_eq!(results.as_array().unwrap().len(), 1);
        assert_eq!(results[0]["path"], "Shared/z.md");
        assert!(f.vault.inner.index.read().unwrap().generation() > generation);
        assert_eq!(
            selected.vault_stats().unwrap().total_notes,
            if change == "edited" { 2 } else { 1 }
        );

        let fresh = Vault::open_authenticated(f.vault.inner.authenticated_config.as_ref().unwrap())
            .await
            .unwrap()
            .with_access(f.vault.access_client().unwrap().clone())
            .unwrap();
        let fresh_result = lexical_tool_query(&fresh, tantivy).await.unwrap();
        assert_eq!(result.structured_content, fresh_result.structured_content);
        assert_eq!(
            std::fs::read_to_string(f.root.join("Private/note.md")).unwrap(),
            "private source"
        );
    }

    #[tokio::test]
    async fn bm25_query_repairs_a_deleted_candidate_before_ranking_again() {
        lexical_query_reconciles_external_change(true, "deleted").await;
    }

    #[tokio::test]
    async fn bm25_query_repairs_a_multilink_candidate_before_ranking_again() {
        lexical_query_reconciles_external_change(true, "hardlink").await;
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn bm25_query_repairs_a_denied_alias_before_ranking_again() {
        lexical_query_reconciles_external_change(true, "alias").await;
    }

    #[tokio::test]
    async fn bm25_query_repairs_changed_source_before_ranking_again() {
        lexical_query_reconciles_external_change(true, "edited").await;
    }

    #[tokio::test]
    async fn regex_query_repairs_a_deleted_candidate_before_searching_again() {
        lexical_query_reconciles_external_change(false, "deleted").await;
    }

    #[tokio::test]
    async fn regex_query_repairs_a_multilink_candidate_before_searching_again() {
        lexical_query_reconciles_external_change(false, "hardlink").await;
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn regex_query_repairs_a_denied_alias_before_searching_again() {
        lexical_query_reconciles_external_change(false, "alias").await;
    }

    #[tokio::test]
    async fn regex_query_repairs_changed_source_before_searching_again() {
        lexical_query_reconciles_external_change(false, "edited").await;
    }

    #[tokio::test]
    async fn lexical_queries_preserve_source_io_and_current_authorization_failures() {
        for tantivy in [false, true] {
            let f = fixture_with_tantivy(vec![directory("Shared", Role::Editor)], vec![], tantivy)
                .await;
            f.vault
                .write_note(Path::new("Shared/note.md"), "quokka")
                .await
                .unwrap();
            let generation = f.vault.inner.index.read().unwrap().generation();
            std::fs::write(f.root.join("Shared/note.md"), [0xff]).unwrap();
            let error = lexical_tool_query(&f.vault, tantivy).await.unwrap_err();
            assert_eq!(error.code, rmcp::model::ErrorCode::INTERNAL_ERROR);
            assert_eq!(f.vault.inner.index.read().unwrap().generation(), generation);
            assert!(
                f.vault
                    .inner
                    .index
                    .read()
                    .unwrap()
                    .get_note(Path::new("Shared/note.md"))
                    .is_some()
            );
            f.store
                .update(None, |policy| {
                    policy.clients.get_mut("client").unwrap().enabled = false;
                    Ok(())
                })
                .unwrap();
            let error = lexical_tool_query(&f.vault, tantivy).await.unwrap_err();
            assert_eq!(
                error.data.as_ref().unwrap()["code"],
                "authorization_unavailable"
            );
            assert_eq!(f.vault.inner.index.read().unwrap().generation(), generation);
        }
    }

    #[tokio::test]
    async fn lexical_source_reads_keep_the_captured_ranking_revision() {
        let f = fixture(vec![directory("Shared", Role::Editor)], vec![]).await;
        let path = Path::new("Shared/note.md");
        let scored_revision = f
            .vault
            .read_index()
            .source_revision(path)
            .unwrap()
            .to_owned();
        f.vault
            .write_note(path, "replacement source")
            .await
            .unwrap();
        assert!(matches!(
            f.vault.read_access_revision_native(path, &scored_revision),
            Err(VaultError::SourceChanged { expected_revision, .. }) if expected_revision == scored_revision
        ));
        assert_eq!(
            f.vault.read_access_indexed_native(path).unwrap(),
            "replacement source"
        );
    }

    #[tokio::test]
    async fn lexical_query_repair_is_bounded_and_does_not_return_partial_hits() {
        for tantivy in [false, true] {
            let f = fixture_with_tantivy(vec![directory("Shared", Role::Editor)], vec![], tantivy)
                .await;
            f.vault
                .write_note(Path::new("Shared/note.md"), "quokka")
                .await
                .unwrap();
            f.vault
                .create_note(Path::new("Shared/other.md"), "quokka quokka", None)
                .await
                .unwrap();
            f.vault
                .create_note(
                    Path::new("Shared/z.md"),
                    "quokka with several other words",
                    None,
                )
                .await
                .unwrap();
            for name in ["note.md", "other.md"] {
                std::fs::hard_link(
                    f.root.join("Shared").join(name),
                    f.root.join("Private").join(format!("linked-{name}")),
                )
                .unwrap();
            }
            let error = lexical_tool_query(&f.vault, tantivy).await.unwrap_err();
            assert_eq!(error.code, rmcp::model::ErrorCode::RESOURCE_NOT_FOUND);
            let still_indexed = ["Shared/note.md", "Shared/other.md"]
                .iter()
                .filter(|path| {
                    f.vault
                        .inner
                        .index
                        .read()
                        .unwrap()
                        .get_note(Path::new(path))
                        .is_some()
                })
                .count();
            assert_eq!(still_indexed, 1);
            let result = lexical_tool_query(&f.vault, tantivy).await.unwrap();
            let results = &result.structured_content.as_ref().unwrap()["results"];
            assert_eq!(results.as_array().unwrap().len(), 1);
            assert_eq!(results[0]["path"], "Shared/z.md");
            assert_eq!(f.vault.vault_stats().unwrap().total_notes, 1);
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn source_view_reconciles_unsafe_alias_after_its_index_observation() {
        let f = source_with_pending_readable_revision().await;
        std::fs::remove_file(f.root.join("Shared/note.md")).unwrap();
        std::os::unix::fs::symlink("../Private/note.md", f.root.join("Shared/note.md")).unwrap();
        assert_unsafe_source_reconciles(&f).await;
        assert_eq!(
            std::fs::read_to_string(f.root.join("Private/note.md")).unwrap(),
            "private source"
        );
    }

    #[tokio::test]
    async fn source_reconciliation_is_bounded_and_never_publishes_a_partial_catalog() {
        let f = fixture(vec![directory("Shared", Role::Editor)], vec![]).await;
        f.vault
            .create_note(Path::new("Shared/other.md"), "other original", None)
            .await
            .unwrap();
        for path in ["Shared/note.md", "Shared/other.md"] {
            std::fs::write(f.root.join(path), "indexed precursor").unwrap();
            f.vault
                .write_index()
                .refresh_file(
                    f.vault.inner.filesystem.as_ref().unwrap(),
                    Path::new(path),
                    f.vault.exclude(),
                    None,
                    #[cfg(has_embeddings)]
                    false,
                )
                .unwrap();
            std::fs::write(
                f.root.join(path),
                "---\ntags: [current]\n---\ncurrent source",
            )
            .unwrap();
        }
        assert!(matches!(
            f.vault.with_access(f.vault.access_client().unwrap().clone()),
            Err(VaultError::SourceChanged { path, .. }) if path == Path::new("Shared/other.md")
        ));
        assert!(!f.vault.projection.as_ref().unwrap().is_ready());
        let recovered = f
            .vault
            .with_access(f.vault.access_client().unwrap().clone())
            .unwrap();
        assert_eq!(recovered.search_by_tag("current").unwrap().len(), 2);
        assert_eq!(recovered.vault_stats().unwrap().total_notes, 2);
    }

    #[tokio::test]
    async fn content_authority_preserves_shared_edits_and_reserves_owner_paths() {
        let f = fixture(vec![directory("", Role::Editor)], vec![deny_private()]).await;
        assert_eq!(
            f.vault
                .read_note(Path::new("Shared/note.md"))
                .await
                .unwrap(),
            "original"
        );
        for path in [
            "Private/note.md",
            ".obsidian/daily-notes.json",
            ".obsidian-mcp/ignore",
            ".git/config",
        ] {
            assert!(
                matches!(
                    f.vault.read_note(Path::new(path)).await,
                    Err(VaultError::AccessDenied { visible: false, .. })
                ),
                "{path}"
            );
            assert!(
                matches!(
                    f.vault.write_note(Path::new(path), "blocked").await,
                    Err(VaultError::AccessDenied { visible: false, .. })
                ),
                "{path}"
            );
        }
        f.vault
            .write_note(Path::new("Shared/note.md"), "updated")
            .await
            .unwrap();
        assert_eq!(
            std::fs::read_to_string(f.root.join("Shared/note.md")).unwrap(),
            "updated"
        );
        assert!(f.vault.delete_note(Path::new("Shared")).await.is_err());
        assert!(
            f.vault
                .move_note(Path::new("Shared"), Path::new("moved"))
                .await
                .is_err()
        );
        assert!(f.root.join("Shared/note.md").exists());
    }

    #[tokio::test]
    async fn exact_file_grant_creates_only_necessary_ancestors() {
        let grant = Grant {
            id: "exact".into(),
            directory: None,
            file: Some("Shared/New/deep/note.md".into()),
            role: Role::Editor,
        };
        let f = fixture(vec![grant], vec![]).await;
        f.vault
            .create_note(Path::new("Shared/New/deep/note.md"), "created", None)
            .await
            .unwrap();
        assert_eq!(
            f.vault
                .list_files(Path::new("Shared"), true, None)
                .await
                .unwrap(),
            vec![
                PathBuf::from("Shared/New"),
                PathBuf::from("Shared/New/deep"),
                PathBuf::from("Shared/New/deep/note.md")
            ]
        );
        assert!(
            f.vault
                .create_note(Path::new("Shared/New/deep/sibling.md"), "denied", None)
                .await
                .is_err()
        );
        assert!(!f.root.join("Shared/New/deep/sibling.md").exists());
    }

    #[tokio::test]
    async fn denied_template_does_not_become_missing_template_content() {
        let f = fixture(vec![directory("Shared", Role::Editor)], vec![]).await;
        let error = f
            .vault
            .create_periodic_note(&NotePeriod::Daily, None, None)
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            VaultError::AccessDenied { visible: false, .. }
        ));
        assert!(!f.root.join("Shared/Daily").exists());
    }

    #[tokio::test]
    async fn read_only_client_cannot_write_through_mixed_operations() {
        let f = fixture(vec![directory("Shared", Role::Reader)], vec![]).await;
        assert!(matches!(
            f.vault
                .write_note(Path::new("Shared/note.md"), "denied")
                .await,
            Err(VaultError::AccessDenied { visible: true, .. })
        ));
        assert!(
            f.vault
                .append_note(Path::new("Shared/new.md"), "denied")
                .await
                .is_err()
        );
        assert!(
            f.vault
                .set_frontmatter_field(
                    Path::new("Shared/note.md"),
                    "title",
                    serde_json::json!("denied")
                )
                .await
                .is_err()
        );
        assert_eq!(
            std::fs::read_to_string(f.root.join("Shared/note.md")).unwrap(),
            "original"
        );
    }

    #[tokio::test]
    async fn native_alias_spelling_does_not_expand_symlink_authority() {
        let f = fixture(vec![directory("Shared", Role::Editor)], vec![]).await;
        std::fs::write(f.root.join("Shared/café.md"), "unicode").unwrap();
        assert_eq!(
            f.vault
                .read_note(Path::new("Shared/cafe\u{301}.md"))
                .await
                .unwrap(),
            "unicode"
        );
        if f.root.join("shared/note.md").exists() {
            assert_eq!(
                f.vault
                    .read_note(Path::new("shared/note.md"))
                    .await
                    .unwrap(),
                "original"
            );
            f.vault
                .write_note(Path::new("shared/note.md"), "native alias")
                .await
                .unwrap();
        }
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink("Shared", f.root.join("Alias")).unwrap();
            assert!(matches!(
                f.vault.read_note(Path::new("Alias/note.md")).await,
                Err(VaultError::AccessDenied { .. })
            ));
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn delegated_operations_reject_hardlinks_without_changing_the_other_name() {
        let f = fixture(vec![directory("Shared", Role::Editor)], vec![]).await;
        let outside = f.root.parent().unwrap().join("outside.md");
        std::fs::write(&outside, "external original").unwrap();
        std::fs::hard_link(&outside, f.root.join("Shared/linked.md")).unwrap();
        assert!(matches!(
            f.vault.read_note(Path::new("Shared/linked.md")).await,
            Err(VaultError::AccessDenied { .. })
        ));
        assert!(matches!(
            f.vault
                .write_note(Path::new("Shared/linked.md"), "denied")
                .await,
            Err(VaultError::AccessDenied { .. })
        ));
        assert!(matches!(
            f.vault
                .create_note(Path::new("Shared/linked.md"), "denied", None)
                .await,
            Err(VaultError::AccessDenied { .. })
        ));
        assert!(matches!(
            f.vault
                .move_note(Path::new("Shared/note.md"), Path::new("Shared/linked.md"))
                .await,
            Err(VaultError::AccessDenied { .. })
        ));
        assert!(f.root.join("Shared/note.md").exists());
        assert_eq!(
            std::fs::read_to_string(outside).unwrap(),
            "external original"
        );
    }

    #[tokio::test]
    async fn mutation_waiting_for_shared_owner_cannot_keep_revoked_authority() {
        let f = fixture(vec![directory("Shared", Role::Editor)], vec![]).await;
        let owner = f.vault.clone();
        let lock = owner.inner.mutations.lock().await;
        let writer = f.vault.clone();
        let task = tokio::spawn(async move {
            writer
                .write_note(Path::new("Shared/note.md"), "late write")
                .await
        });
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            while owner.background.admission.available_permits() == 16 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        f.store
            .update(None, |policy| {
                policy.clients.get_mut("client").unwrap().enabled = false;
                Ok(())
            })
            .unwrap();
        drop(lock);
        assert!(task.await.unwrap().is_err());
        assert_eq!(
            std::fs::read_to_string(f.root.join("Shared/note.md")).unwrap(),
            "original"
        );
    }

    #[tokio::test]
    async fn a_client_for_another_vault_cannot_select_this_source() {
        let first = fixture(vec![directory("", Role::Editor)], vec![]).await;
        let second = fixture(vec![directory("", Role::Editor)], vec![]).await;
        let other = second.vault.access_client().unwrap().clone();
        assert!(matches!(
            first.vault.with_access(other),
            Err(VaultError::Authorization(_))
        ));
        assert_eq!(
            std::fs::read_to_string(first.root.join("Shared/note.md")).unwrap(),
            "original"
        );
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn an_open_source_root_blocks_native_replacement() {
        let fixture = fixture(vec![directory("", Role::Editor)], vec![]).await;
        let original = fixture.root.with_file_name("retained-original");
        let error = std::fs::rename(&fixture.root, &original).unwrap_err();
        assert_eq!(
            error.raw_os_error(),
            Some(windows_sys::Win32::Foundation::ERROR_SHARING_VIOLATION as i32)
        );
        assert!(!original.exists());
        let named = VaultRoot::open(&fixture.root).unwrap();
        assert!(
            fixture
                .vault
                .inner
                .filesystem
                .as_ref()
                .unwrap()
                .same_root(&named)
                .unwrap()
        );
        let client = fixture.vault.access_client().unwrap().clone();
        client.current().unwrap();
        let selected = fixture.vault.with_access(client).unwrap();
        assert_eq!(
            selected
                .read_note(Path::new("Shared/note.md"))
                .await
                .unwrap(),
            "original"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_rebound_client_cannot_reuse_an_old_root_at_the_same_path() {
        let fixture = fixture(vec![directory("", Role::Editor)], vec![]).await;
        let original = fixture.root.with_file_name("retained-original");
        std::fs::rename(&fixture.root, &original).unwrap();
        std::fs::create_dir_all(fixture.root.join("Shared")).unwrap();
        std::fs::write(fixture.root.join("Shared/note.md"), "replacement source").unwrap();
        fixture
            .store
            .update(None, |policy| {
                policy.source = crate::access::clients::SourceIdentity::capture(&fixture.root)?;
                Ok(())
            })
            .unwrap();
        let reopened_store = AccessStore::open(fixture.store.path()).unwrap();
        let token = crate::access::store::read_token_file(
            &fixture.root.parent().unwrap().join("state/client.token"),
        )
        .unwrap();
        let rebound = reopened_store.authenticate(token.expose()).unwrap();
        assert!(matches!(
            fixture.vault.with_access(rebound.clone()),
            Err(VaultError::Authorization(_))
        ));
        let mut config = crate::test_helpers::test_config(&fixture.root);
        config.mcp_data_dir = Some(reopened_store.snapshot().unwrap().policy.data_dir);
        let reopened = Vault::open_authenticated(&config)
            .await
            .unwrap()
            .with_access(rebound)
            .unwrap();
        assert_eq!(
            reopened
                .read_note(Path::new("Shared/note.md"))
                .await
                .unwrap(),
            "replacement source"
        );
        assert_eq!(
            std::fs::read_to_string(original.join("Shared/note.md")).unwrap(),
            "original"
        );
    }

    #[tokio::test]
    async fn semantic_result_admission_retains_changed_and_missing_authorized_source() {
        let fixture = fixture(vec![directory("Shared", Role::Reader)], vec![]).await;
        let path = Path::new("Shared/note.md");
        let revision = super::source::source_revision("original");
        std::fs::write(fixture.root.join(path), "changed source").unwrap();
        fixture
            .vault
            .check_result_revision(path, &revision)
            .await
            .unwrap();
        std::fs::remove_file(fixture.root.join(path)).unwrap();
        fixture
            .vault
            .check_result_revision(path, &revision)
            .await
            .unwrap();
        assert!(matches!(
            fixture
                .vault
                .check_result_revision(Path::new("Private/note.md"), &revision)
                .await,
            Err(VaultError::AccessDenied { .. })
        ));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn external_and_dangling_aliases_are_hidden_at_every_delegated_boundary() {
        let fixture = fixture(vec![directory("Shared", Role::Editor)], vec![]).await;
        let external = fixture
            .root
            .parent()
            .unwrap()
            .join("undisclosed-external.md");
        std::fs::write(&external, "external marker").unwrap();
        let missing = fixture
            .root
            .parent()
            .unwrap()
            .join("undisclosed-missing.md");
        for (name, target) in [("outside.md", &external), ("dangling.md", &missing)] {
            let relative = PathBuf::from(format!("Shared/{name}"));
            std::os::unix::fs::symlink(target, fixture.root.join(&relative)).unwrap();
            let errors = [
                fixture.vault.read_note(&relative).await.unwrap_err(),
                fixture
                    .vault
                    .write_note(&relative, "blocked")
                    .await
                    .unwrap_err(),
                fixture
                    .vault
                    .create_note(&relative, "blocked", None)
                    .await
                    .unwrap_err(),
                fixture.vault.file_stat(&relative).await.unwrap_err(),
                fixture.vault.delete_note(&relative).await.unwrap_err(),
                fixture
                    .vault
                    .move_note(&relative, Path::new("Shared/moved.md"))
                    .await
                    .unwrap_err(),
                fixture
                    .vault
                    .move_note(Path::new("Shared/note.md"), &relative)
                    .await
                    .unwrap_err(),
                fixture
                    .vault
                    .check_result_revision(&relative, "old evidence")
                    .await
                    .unwrap_err(),
            ];
            for error in errors {
                assert!(matches!(
                    error,
                    VaultError::AccessDenied { visible: false, .. }
                ));
                let outward: rmcp::ErrorData = error.into();
                assert_eq!(outward.code, rmcp::model::ErrorCode::RESOURCE_NOT_FOUND);
                assert_eq!(outward.message, "Resource not found");
                assert!(outward.data.is_none());
            }
            assert!(
                fixture
                    .root
                    .join(&relative)
                    .symlink_metadata()
                    .unwrap()
                    .is_symlink()
            );
        }
        let outside_directory = fixture.root.parent().unwrap().join("external-directory");
        std::fs::create_dir(&outside_directory).unwrap();
        std::os::unix::fs::symlink(
            &outside_directory,
            fixture.root.join("Shared/outside-directory"),
        )
        .unwrap();
        assert!(matches!(
            fixture
                .vault
                .list_files(Path::new("Shared/outside-directory"), true, None)
                .await,
            Err(VaultError::AccessDenied { visible: false, .. })
        ));
        assert_eq!(
            fixture
                .vault
                .list_files(Path::new("Shared"), true, None)
                .await
                .unwrap(),
            vec![PathBuf::from("Shared/note.md")]
        );
        assert_eq!(
            std::fs::read_to_string(&external).unwrap(),
            "external marker"
        );
        assert_eq!(
            std::fs::read_to_string(fixture.root.join("Shared/note.md")).unwrap(),
            "original"
        );

        // These are also the physical reopen primitives used after source
        // admission, so late link replacement cannot change the public error.
        let delegated_root = fixture.vault.authenticated_root().unwrap();
        assert!(matches!(
            delegated_root.open_exact(Path::new("Shared/outside.md")),
            Err(VaultError::AccessDenied { .. })
        ));
        assert!(matches!(
            delegated_root.write_file(Path::new("Shared/outside.md"), "blocked"),
            Err(VaultError::AccessDenied { .. })
        ));
        let owner_root = VaultRoot::open(&fixture.root).unwrap();
        assert!(matches!(
            owner_root.read_file(Path::new("Shared/outside.md")),
            Err(VaultError::OutsideVault(_))
        ));
        assert!(matches!(
            owner_root.read_file(Path::new("Shared/dangling.md")),
            Err(VaultError::InvalidPath(_))
        ));
    }

    #[tokio::test]
    async fn malformed_caller_paths_remain_parameter_errors() {
        let fixture = fixture(vec![directory("Shared", Role::Editor)], vec![]).await;
        for path in [
            PathBuf::from("../outside.md"),
            fixture.root.join("Shared/note.md"),
            PathBuf::from("Shared/bad\u{0}.md"),
        ] {
            let error = fixture.vault.read_note(&path).await.unwrap_err();
            let outward: rmcp::ErrorData = error.into();
            assert_eq!(outward.code, rmcp::model::ErrorCode::INVALID_PARAMS);
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn admitted_physical_keys_never_follow_replacement_links() {
        let fixture = fixture(vec![directory("Shared", Role::Editor)], vec![]).await;
        let root = fixture.vault.authenticated_root().unwrap();
        let (logical, admitted) = root
            .resolve_access_existing(Path::new("Shared/note.md"))
            .unwrap();
        fixture
            .vault
            .access_client()
            .unwrap()
            .check(Action::Update, &logical, &admitted.relative)
            .unwrap();
        std::fs::remove_file(fixture.root.join(&admitted.relative)).unwrap();
        std::os::unix::fs::symlink("../Private/note.md", fixture.root.join(&admitted.relative))
            .unwrap();
        for error in [
            root.write_file_exact(&admitted.relative, "blocked")
                .unwrap_err(),
            root.append_file_exact(&admitted.relative, "blocked")
                .unwrap_err(),
            root.create_file_exact(&admitted.relative, "blocked")
                .unwrap_err(),
            root.file_stat_exact(&admitted.relative).unwrap_err(),
            root.validate_access_file_exact(&admitted.relative)
                .unwrap_err(),
            root.move_entry_exact(
                &admitted.relative,
                Path::new("Shared/moved.md"),
                &admitted.relative,
            )
            .unwrap_err(),
        ] {
            assert!(matches!(
                error,
                VaultError::AccessDenied { visible: false, .. }
            ));
        }
        assert_eq!(
            std::fs::read_to_string(fixture.root.join("Private/note.md")).unwrap(),
            "private source"
        );
        assert!(!fixture.root.join("Shared/moved.md").exists());
        // Deletion acts on the admitted namespace entry, never its new target.
        root.delete_entry_exact(&admitted.relative).unwrap();
        assert!(fixture.root.join("Private/note.md").exists());

        std::fs::create_dir(fixture.root.join("Shared/New")).unwrap();
        let (logical, future) = root
            .resolve_access_for_write(Path::new("Shared/New/new.md"))
            .unwrap();
        fixture
            .vault
            .access_client()
            .unwrap()
            .check(Action::Create, &logical, &future.relative)
            .unwrap();
        std::fs::rename(
            fixture.root.join("Shared/New"),
            fixture.root.join("Shared/retained-parent"),
        )
        .unwrap();
        std::os::unix::fs::symlink("../Private", fixture.root.join("Shared/New")).unwrap();
        assert!(matches!(
            root.create_file_exact(&future.relative, "blocked"),
            Err(VaultError::AccessDenied { visible: false, .. })
        ));
        assert!(!fixture.root.join("Private/new.md").exists());
    }
}
