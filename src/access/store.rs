//! Owner-managed access policies and fresh, credential-bound authorization.

use std::ffi::OsString;
use std::fmt;
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;
use zeroize::Zeroizing;

use super::clients::{
    AccessAction, AccessPolicy, ClientPolicy, Credential, MAX_CREDENTIALS_PER_CLIENT, PolicyError,
    SourceIdentity,
};

#[path = "secure.rs"]
mod secure;

const MAX_TOKEN_BYTES: usize = 256;
const MAX_POLICY_BYTES: usize = 1_048_576;
const TOKEN_PREFIX: &str = "omcp1_";

pub type AccessResult<T> = Result<T, AccessError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PublicationOutcome {
    NotPublished,
    Published,
    Unknown,
}

#[derive(Debug, thiserror::Error)]
pub enum AccessError {
    #[error("resource is not available")]
    Denied { path: PathBuf, visible: bool },
    #[error("invalid or expired access credential")]
    InvalidCredential,
    #[error("access configuration: {0}")]
    Configuration(String),
    #[error("access file I/O: {0}")]
    Io(#[from] io::Error),
    #[error("access policy revision changed (expected {expected}, current {actual})")]
    Conflict { expected: u64, actual: u64 },
    #[error("access policy publication {outcome:?}: {message}")]
    Publication {
        outcome: PublicationOutcome,
        message: String,
    },
    #[error("access policy publication is in progress")]
    Busy,
}

impl AccessError {
    fn publication(published: bool, error: impl fmt::Display) -> Self {
        Self::Publication {
            outcome: if published {
                PublicationOutcome::Published
            } else {
                PublicationOutcome::NotPublished
            },
            message: error.to_string(),
        }
    }
}

impl From<PolicyError> for AccessError {
    fn from(error: PolicyError) -> Self {
        Self::Configuration(error.to_string())
    }
}

/// A plaintext credential with redacted diagnostics and zeroized storage.
pub struct SecretToken(Zeroizing<String>);

impl SecretToken {
    pub fn expose(&self) -> &str {
        &self.0
    }

    pub fn as_str(&self) -> &str {
        self.expose()
    }
}

impl fmt::Debug for SecretToken {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("SecretToken([REDACTED])")
    }
}

#[derive(Clone)]
pub struct AccessStore(Arc<StoreInner>);

struct StoreInner {
    directory: secure::ProtectedDirectory,
    name: OsString,
    lock_name: OsString,
    lock_identity: String,
    path: PathBuf,
    source: SourceIdentity,
    data_directory: secure::ProtectedDirectory,
}

impl fmt::Debug for AccessStore {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AccessStore")
            .field("path", &self.0.path)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone)]
pub struct PolicySnapshot {
    pub policy: AccessPolicy,
    pub content_hash: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct CredentialReceipt {
    pub client_id: String,
    pub credential_id: String,
    pub path: PathBuf,
    pub revision: u64,
}

impl AccessStore {
    /// Create a new protected store. Existing policies are never replaced.
    pub fn initialize(path: &Path, vault: &Path, data_dir: &Path) -> AccessResult<Self> {
        validate_initialization(path, vault, data_dir)?;
        let source = SourceIdentity::capture(vault)?;
        ensure_external(
            path.parent().ok_or_else(|| {
                AccessError::Configuration("access file requires a parent directory".into())
            })?,
            &source,
        )?;
        ensure_external(data_dir, &source)?;
        let (directory, name) = secure::split_file(path, true)?;
        create_protected_directory(data_dir)?;
        let data_dir = data_dir.canonicalize()?;
        let policy = AccessPolicy::new(source.clone(), data_dir);
        policy.validate()?;
        let lock_name = lock_name(&name);
        match directory.create(&lock_name, b"") {
            Ok(()) => (),
            Err(AccessError::Io(error)) if error.kind() == io::ErrorKind::AlreadyExists => (),
            Err(error) => return Err(error),
        }
        let lock = directory.open_file(&lock_name, true, false)?;
        fs2::FileExt::lock_exclusive(&lock)?;
        let bytes = encode_policy(&policy)?;
        directory.create(&name, &bytes)?;
        drop(lock);
        Self::open(&directory.path().join(&name))
    }

    /// Open the protected store without initializing source indexes or inference.
    pub fn open(path: &Path) -> AccessResult<Self> {
        let (directory, name) = secure::split_file(path, false)?;
        let lock_name = lock_name(&name);
        let lock = directory.open_file(&lock_name, true, false)?;
        fs2::FileExt::lock_shared(&lock)?;
        directory.verify_named()?;
        let policy = AccessPolicy::parse(&directory.read(&name, MAX_POLICY_BYTES)?)?;
        ensure_external(directory.path(), &policy.source)?;
        ensure_external(&policy.data_dir, &policy.source)?;
        let data_directory = secure::ProtectedDirectory::open(&policy.data_dir, false)?;
        let source = policy.source;
        let lock_identity = secure::object_identity(&lock)?;
        let path = directory.path().join(&name);
        drop(lock);
        Ok(Self(Arc::new(StoreInner {
            directory,
            name,
            lock_name,
            lock_identity,
            path,
            source,
            data_directory,
        })))
    }

    pub fn path(&self) -> &Path {
        &self.0.path
    }

    pub fn snapshot(&self) -> AccessResult<PolicySnapshot> {
        let _lock = self.lock(false, false)?;
        self.read_snapshot()
    }

    /// Run a short synchronous service publication/cleanup operation against a
    /// current snapshot. Compute candidates first; do not perform source I/O,
    /// network work, nested store acquisition or async waits in the callback.
    pub fn with_snapshot<T>(
        &self,
        operation: impl FnOnce(&PolicySnapshot) -> AccessResult<T>,
    ) -> AccessResult<T> {
        let _lock = self.lock(false, false)?;
        let snapshot = self.read_snapshot()?;
        self.validate_binding(&snapshot.policy)?;
        operation(&snapshot)
    }

    pub fn validate_source(&self, vault: &Path) -> AccessResult<()> {
        let _lock = self.lock(false, false)?;
        let snapshot = self.read_snapshot()?;
        self.validate_binding(&snapshot.policy)?;
        let actual = SourceIdentity::capture(vault)?;
        if snapshot.policy.source != actual || self.0.source != actual {
            return Err(AccessError::Configuration(
                "policy belongs to a different vault; use owner rebind and reopen the service"
                    .into(),
            ));
        }
        snapshot.policy.source.verify()?;
        ensure_external(self.0.directory.path(), &actual)?;
        ensure_external(&snapshot.policy.data_dir, &actual)?;
        Ok(())
    }

    /// Publish one complete policy under the stable lock. A changed revision is
    /// detected before invoking the caller's transformation.
    pub fn update(
        &self,
        expected_revision: Option<u64>,
        change: impl FnOnce(&mut AccessPolicy) -> AccessResult<()>,
    ) -> AccessResult<PolicySnapshot> {
        let _lock = self.lock(true, false)?;
        let previous = self.read_snapshot()?;
        check_revision(&previous.policy, expected_revision)?;
        let mut policy = previous.policy;
        let revision = policy
            .revision
            .checked_add(1)
            .ok_or_else(|| AccessError::Configuration("access policy revision exhausted".into()))?;
        change(&mut policy)?;
        policy.revision = revision;
        policy.validate()?;
        ensure_external(self.0.directory.path(), &policy.source)?;
        ensure_external(&policy.data_dir, &policy.source)?;
        let bytes = encode_policy(&policy)?;
        self.0.directory.publish(&self.0.name, &bytes)?;
        Ok(PolicySnapshot {
            policy,
            content_hash: hex_digest(&bytes),
        })
    }

    pub fn validate_token_output(&self, output: &Path) -> AccessResult<()> {
        let snapshot = self.snapshot()?;
        validate_output_path(output)?;
        ensure_external(
            output.parent().ok_or_else(|| {
                AccessError::Configuration("token output requires a directory".into())
            })?,
            &snapshot.policy.source,
        )
    }

    pub fn issue_token(
        &self,
        client_id: &str,
        output: &Path,
        expires_at: Option<i64>,
    ) -> AccessResult<CredentialReceipt> {
        self.issue_token_at_revision(client_id, output, expires_at, None)
    }

    pub fn issue_token_at_revision(
        &self,
        client_id: &str,
        output: &Path,
        expires_at: Option<i64>,
        expected_revision: Option<u64>,
    ) -> AccessResult<CredentialReceipt> {
        self.deliver_credential(client_id, None, output, expires_at, expected_revision)
    }

    pub fn create_client_with_token(
        &self,
        client_id: &str,
        client: &ClientPolicy,
        output: &Path,
        expires_at: Option<i64>,
        expected_revision: Option<u64>,
    ) -> AccessResult<CredentialReceipt> {
        self.deliver_credential(
            client_id,
            Some(client),
            output,
            expires_at,
            expected_revision,
        )
    }

    fn deliver_credential(
        &self,
        client_id: &str,
        initial_client: Option<&ClientPolicy>,
        output: &Path,
        expires_at: Option<i64>,
        expected_revision: Option<u64>,
    ) -> AccessResult<CredentialReceipt> {
        let _lock = self.lock(true, false)?;
        let mut snapshot = self.read_snapshot()?;
        check_revision(&snapshot.policy, expected_revision)?;
        if let Some(client) = initial_client {
            if snapshot.policy.clients.contains_key(client_id) {
                return Err(AccessError::Configuration("client already exists".into()));
            }
            if !client.credentials.is_empty() {
                return Err(AccessError::Configuration(
                    "new clients cannot import credential digests".into(),
                ));
            }
            snapshot
                .policy
                .clients
                .insert(client_id.into(), client.clone());
            snapshot.policy.validate()?;
        }
        if expires_at.is_some_and(|expiry| expiry <= now()) {
            return Err(AccessError::Configuration(
                "credential expiry must be in the future".into(),
            ));
        }
        let client = snapshot
            .policy
            .clients
            .get_mut(client_id)
            .ok_or_else(|| AccessError::Configuration("client does not exist".into()))?;
        // Retain intentional overlap, while discarding inactive verifier metadata
        // only as part of the successful replacement that adds the new token.
        client.credentials.retain(credential_live);
        if client.credentials.len() >= MAX_CREDENTIALS_PER_CLIENT {
            return Err(AccessError::Configuration("client has too many active credentials; revoke an unused credential before creating another".into()));
        }
        ensure_external(
            output.parent().ok_or_else(|| {
                AccessError::Configuration("token output requires a directory".into())
            })?,
            &snapshot.policy.source,
        )?;
        let id = random_id()?;
        let token = SecretToken(Zeroizing::new(format!(
            "{TOKEN_PREFIX}{id}_{}",
            random_secret()?
        )));
        let mut delivered = Zeroizing::new(token.expose().as_bytes().to_vec());
        delivered.push(b'\n');
        // A durable delivery precedes activation. Never erase this output after
        // a later publication failure, which can have a committed outcome.
        write_protected_file(output, &delivered)?;
        let publication = (|| -> AccessResult<CredentialReceipt> {
            let mut policy = snapshot.policy;
            let client = policy
                .clients
                .get_mut(client_id)
                .ok_or_else(|| AccessError::Configuration("client does not exist".into()))?;
            client.credentials.push(Credential {
                id: id.clone(),
                digest: token_digest(token.expose()),
                created_at: now(),
                expires_at,
                revoked: false,
            });
            policy.revision = policy.revision.checked_add(1).ok_or_else(|| {
                AccessError::Configuration("access policy revision exhausted".into())
            })?;
            policy.validate()?;
            self.0
                .directory
                .publish(&self.0.name, &encode_policy(&policy)?)?;
            Ok(CredentialReceipt {
                client_id: client_id.into(),
                credential_id: id,
                path: output.to_owned(),
                revision: policy.revision,
            })
        })();
        publication.map_err(|error| {
            let (outcome, message) = match error {
                AccessError::Publication { outcome, message } => (outcome, message),
                error => (PublicationOutcome::NotPublished, error.to_string()),
            };
            AccessError::Publication {
                outcome,
                message: format!("{message}; delivered token remains at {}; inspect the policy by credential ID before retrying", output.display()),
            }
        })
    }

    pub fn authenticate(&self, token: &str) -> AccessResult<AuthorizedClient> {
        let credential_id = token_id(token).ok_or(AccessError::InvalidCredential)?;
        let _lock = self.lock(false, false)?;
        let snapshot = self.read_snapshot()?;
        self.validate_binding(&snapshot.policy)?;
        let digest = token_digest(token);
        let (client_id, policy) = snapshot
            .policy
            .clients
            .iter()
            .find(|(_, client)| {
                client.enabled
                    && client.credentials.iter().any(|credential| {
                        credential.id == credential_id
                            && credential_live(credential)
                            && bool::from(credential.digest.as_bytes().ct_eq(digest.as_bytes()))
                    })
            })
            .ok_or(AccessError::InvalidCredential)?;
        Ok(AuthorizedClient {
            store: self.clone(),
            client_id: client_id.clone(),
            credential_id: credential_id.into(),
            credential_digest: digest,
            policy: policy.clone(),
            policy_hash: policy.authorization_hash(),
            read_policy_key: policy.read_policy_hash(),
            source: snapshot.policy.source,
        })
    }

    /// Validate an already activated service corpus independently of any one
    /// request credential. Callers must establish activation through a valid
    /// semantic request before retaining this authority.
    pub fn ingestion_permit(
        &self,
        client_id: &str,
        read_policy_key: &str,
    ) -> AccessResult<AccessPermit> {
        let lock = self.lock(false, false)?;
        let snapshot = self.read_snapshot()?;
        self.validate_binding(&snapshot.policy)?;
        let client = snapshot
            .policy
            .clients
            .get(client_id)
            .ok_or(AccessError::InvalidCredential)?;
        if !client.enabled
            || client.read_policy_hash() != read_policy_key
            || !client.credentials.iter().any(credential_live)
        {
            return Err(AccessError::InvalidCredential);
        }
        Ok(AccessPermit {
            _lock: lock,
            snapshot,
            client_id: client_id.into(),
        })
    }

    fn lock(&self, exclusive: bool, nonblocking: bool) -> AccessResult<File> {
        self.0.directory.verify_named()?;
        let file = self.0.directory.open_file(&self.0.lock_name, true, false)?;
        if secure::object_identity(&file)? != self.0.lock_identity {
            return Err(AccessError::Configuration(
                "access lock identity changed; reopen the service".into(),
            ));
        }
        let result = match (exclusive, nonblocking) {
            (true, true) => fs2::FileExt::try_lock_exclusive(&file),
            (false, true) => fs2::FileExt::try_lock_shared(&file),
            (true, false) => fs2::FileExt::lock_exclusive(&file),
            (false, false) => fs2::FileExt::lock_shared(&file),
        };
        result.map_err(|error| {
            if error.kind() == io::ErrorKind::WouldBlock {
                AccessError::Busy
            } else {
                error.into()
            }
        })?;
        Ok(file)
    }

    fn read_snapshot(&self) -> AccessResult<PolicySnapshot> {
        self.0.directory.verify_named()?;
        let bytes = self.0.directory.read(&self.0.name, MAX_POLICY_BYTES)?;
        let policy = AccessPolicy::parse(&bytes)?;
        Ok(PolicySnapshot {
            policy,
            content_hash: hex_digest(&bytes),
        })
    }

    fn validate_binding(&self, policy: &AccessPolicy) -> AccessResult<()> {
        if policy.source != self.0.source {
            return Err(AccessError::Configuration(
                "vault binding changed; reopen the service".into(),
            ));
        }
        policy.source.verify()?;
        if policy.data_dir.canonicalize()? != self.0.data_directory.path() {
            return Err(AccessError::Configuration(
                "access machine-data binding changed; reopen the service".into(),
            ));
        }
        self.0.data_directory.verify_named()?;
        ensure_external(self.0.directory.path(), &policy.source)?;
        ensure_external(&policy.data_dir, &policy.source)?;
        Ok(())
    }
}

#[derive(Clone)]
pub struct AuthorizedClient {
    store: AccessStore,
    client_id: String,
    credential_id: String,
    credential_digest: String,
    policy: ClientPolicy,
    policy_hash: String,
    read_policy_key: String,
    source: SourceIdentity,
}

impl fmt::Debug for AuthorizedClient {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AuthorizedClient")
            .field("client_id", &self.client_id)
            .field("credential_id", &self.credential_id)
            .finish_non_exhaustive()
    }
}

impl AuthorizedClient {
    pub fn client_id(&self) -> &str {
        &self.client_id
    }
    pub fn credential_id(&self) -> &str {
        &self.credential_id
    }
    pub fn policy_hash(&self) -> &str {
        &self.policy_hash
    }
    pub fn read_policy_key(&self) -> &str {
        &self.read_policy_key
    }
    pub fn policy(&self) -> &ClientPolicy {
        &self.policy
    }
    pub fn store(&self) -> &AccessStore {
        &self.store
    }
    pub fn source(&self) -> &SourceIdentity {
        &self.source
    }

    pub fn current(&self) -> AccessResult<()> {
        self.permit().map(drop)
    }

    pub async fn current_async(&self) -> AccessResult<()> {
        self.permit_async().await.map(drop)
    }

    pub fn check(&self, action: AccessAction, requested: &Path, actual: &Path) -> AccessResult<()> {
        self.permit()?.check(action, requested, actual)
    }

    pub fn permit(&self) -> AccessResult<AccessPermit> {
        self.acquire(false)
    }
    pub fn try_permit(&self) -> AccessResult<AccessPermit> {
        self.acquire(true)
    }

    /// Wait for policy publication without blocking an async runtime worker.
    /// The returned permit still belongs only around synchronous work.
    pub async fn permit_async(&self) -> AccessResult<AccessPermit> {
        loop {
            match self.try_permit() {
                Err(AccessError::Busy) => {
                    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                }
                result => return result,
            }
        }
    }

    fn acquire(&self, nonblocking: bool) -> AccessResult<AccessPermit> {
        let lock = self.store.lock(false, nonblocking)?;
        let snapshot = self.store.read_snapshot()?;
        self.store.validate_binding(&snapshot.policy)?;
        let client = snapshot
            .policy
            .clients
            .get(&self.client_id)
            .ok_or(AccessError::InvalidCredential)?;
        let credential = client
            .credentials
            .iter()
            .find(|credential| credential.id == self.credential_id)
            .ok_or(AccessError::InvalidCredential)?;
        if !client.enabled
            || !credential_live(credential)
            || !bool::from(
                credential
                    .digest
                    .as_bytes()
                    .ct_eq(self.credential_digest.as_bytes()),
            )
            || client.authorization_hash() != self.policy_hash
        {
            return Err(AccessError::InvalidCredential);
        }
        Ok(AccessPermit {
            _lock: lock,
            snapshot,
            client_id: self.client_id.clone(),
        })
    }
}

/// A short shared-lock lease for synchronous admission or result publication.
/// Never retain it across an await, body Pending or network backpressure.
pub struct AccessPermit {
    _lock: File,
    snapshot: PolicySnapshot,
    client_id: String,
}

impl AccessPermit {
    pub fn snapshot(&self) -> &PolicySnapshot {
        &self.snapshot
    }
    pub fn policy(&self) -> &ClientPolicy {
        // The client was validated when this immutable permit was acquired.
        &self.snapshot.policy.clients[&self.client_id]
    }

    pub fn check(&self, action: AccessAction, requested: &Path, actual: &Path) -> AccessResult<()> {
        let policy = self.policy();
        if policy.allows(action, requested) && policy.allows(action, actual) {
            return Ok(());
        }
        Err(AccessError::Denied {
            path: requested.to_owned(),
            visible: policy.allows(AccessAction::Read, requested)
                && policy.allows(AccessAction::Read, actual),
        })
    }
}

impl SourceIdentity {
    pub fn capture(vault: &Path) -> AccessResult<Self> {
        let canonical_path = vault.canonicalize()?;
        let directory =
            cap_std::fs::Dir::open_ambient_dir(&canonical_path, cap_std::ambient_authority())?;
        let file = directory.into_std_file();
        if !file.metadata()?.is_dir() {
            return Err(AccessError::Configuration(
                "vault must be a directory".into(),
            ));
        }
        let object_id = secure::object_identity(&file)?;
        let storage_key = crate::config::vault_storage_key(&canonical_path);
        Ok(Self {
            canonical_path,
            storage_key,
            object_id,
        })
    }

    pub fn verify(&self) -> AccessResult<()> {
        let actual = Self::capture(&self.canonical_path)?;
        if &actual != self {
            return Err(AccessError::Configuration(
                "vault identity changed; use owner rebind and reopen the service".into(),
            ));
        }
        Ok(())
    }
}

pub fn validate_initialization(path: &Path, vault: &Path, data_dir: &Path) -> AccessResult<()> {
    let source = SourceIdentity::capture(vault)?;
    validate_output_path(path)?;
    ensure_external(
        path.parent().ok_or_else(|| {
            AccessError::Configuration("access file requires a parent directory".into())
        })?,
        &source,
    )?;
    secure::validate_directory_path(data_dir)?;
    ensure_external(data_dir, &source)?;
    let profile = prospective_canonical(path)?;
    let data = prospective_canonical(data_dir)?;
    let lock =
        profile.with_file_name(lock_name(profile.file_name().ok_or_else(|| {
            AccessError::Configuration("access file requires a filename".into())
        })?));
    if data.starts_with(&profile) || data.starts_with(&lock) {
        return Err(AccessError::Configuration(
            "machine data directory conflicts with the access policy or lock file".into(),
        ));
    }
    Ok(())
}

fn validate_output_path(path: &Path) -> AccessResult<()> {
    if !path.is_absolute() || path.file_name().is_none() {
        return Err(AccessError::Configuration(
            "access output requires an absolute file path".into(),
        ));
    }
    match std::fs::symlink_metadata(path) {
        Ok(_) => {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "access output already exists",
            )
            .into());
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => (),
        Err(error) => return Err(error.into()),
    }
    secure::validate_directory_path(path.parent().ok_or_else(|| {
        AccessError::Configuration("access output requires a parent directory".into())
    })?)
}

pub fn create_protected_directory(path: &Path) -> AccessResult<()> {
    secure::ProtectedDirectory::open(path, true).map(drop)
}

pub fn write_protected_file(path: &Path, bytes: &[u8]) -> AccessResult<()> {
    validate_output_path(path)?;
    let (parent, name) = secure::split_file(path, true)?;
    parent.create(&name, bytes)
}

pub fn read_token_file(path: &Path) -> AccessResult<SecretToken> {
    let (parent, name) = secure::split_file(path, false)?;
    parent.verify_named()?;
    let bytes = Zeroizing::new(parent.read(&name, MAX_TOKEN_BYTES)?);
    let value = std::str::from_utf8(&bytes).map_err(|_| AccessError::InvalidCredential)?;
    let value = value.strip_suffix('\n').unwrap_or(value);
    let value = value.strip_suffix('\r').unwrap_or(value);
    if token_id(value).is_none() {
        return Err(AccessError::InvalidCredential);
    }
    Ok(SecretToken(Zeroizing::new(value.to_owned())))
}

pub fn default_profile_path(vault: &Path) -> AccessResult<PathBuf> {
    let source = SourceIdentity::capture(vault)?;
    #[cfg(target_os = "macos")]
    let base = home_directory()?.join("Library/Application Support");
    #[cfg(windows)]
    let base = std::env::var_os("LOCALAPPDATA")
        .map(PathBuf::from)
        .ok_or_else(|| {
            AccessError::Configuration(
                "LOCALAPPDATA is required to choose the access directory".into(),
            )
        })?;
    #[cfg(not(any(target_os = "macos", windows)))]
    let base = match std::env::var_os("XDG_CONFIG_HOME") {
        Some(path) if Path::new(&path).is_absolute() => PathBuf::from(path),
        Some(_) => {
            return Err(AccessError::Configuration(
                "XDG_CONFIG_HOME must be absolute".into(),
            ));
        }
        None => home_directory()?.join(".config"),
    };
    Ok(base
        .join("obsidian-mcp/access")
        .join(source.storage_key)
        .join("access.json"))
}

pub fn default_data_dir(profile: &Path) -> PathBuf {
    profile.with_file_name("data")
}

#[cfg(not(windows))]
fn home_directory() -> AccessResult<PathBuf> {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .filter(|path| path.is_absolute())
        .ok_or_else(|| {
            AccessError::Configuration(
                "an absolute HOME is required to choose the access directory".into(),
            )
        })
}

fn lock_name(name: &std::ffi::OsStr) -> OsString {
    let mut value = name.to_owned();
    value.push(".lock");
    value
}

fn now() -> i64 {
    chrono::Utc::now().timestamp()
}
fn credential_live(credential: &Credential) -> bool {
    !credential.revoked && credential.expires_at.is_none_or(|expiry| expiry > now())
}
fn check_revision(policy: &AccessPolicy, expected: Option<u64>) -> AccessResult<()> {
    if let Some(expected) = expected
        && expected != policy.revision
    {
        return Err(AccessError::Conflict {
            expected,
            actual: policy.revision,
        });
    }
    Ok(())
}
fn encode_policy(policy: &AccessPolicy) -> AccessResult<Vec<u8>> {
    let bytes = serde_json::to_vec_pretty(policy)
        .map_err(|error| AccessError::Configuration(error.to_string()))?;
    if bytes.len() > MAX_POLICY_BYTES {
        return Err(AccessError::Configuration(
            "access policy exceeds its size limit".into(),
        ));
    }
    Ok(bytes)
}
fn token_id(token: &str) -> Option<&str> {
    let token = token.strip_prefix(TOKEN_PREFIX)?;
    let (id, secret) = token.split_once('_')?;
    (id.len() == 32
        && secret.len() == 64
        && id
            .bytes()
            .chain(secret.bytes())
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()))
    .then_some(id)
}
fn token_digest(token: &str) -> String {
    let mut hash = Sha256::new();
    hash.update(b"obsidian-mcp/access-token/v1\0");
    hash.update(token.as_bytes());
    hex_bytes(&hash.finalize())
}
fn hex_digest(bytes: &[u8]) -> String {
    hex_bytes(&Sha256::digest(bytes))
}
fn hex_bytes(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut result = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(result, "{byte:02x}");
    }
    result
}
fn random_secret() -> AccessResult<String> {
    let mut bytes = Zeroizing::new([0u8; 32]);
    getrandom::fill(bytes.as_mut())
        .map_err(|_| AccessError::Configuration("secure randomness is unavailable".into()))?;
    Ok(hex_bytes(bytes.as_ref()))
}
fn random_id() -> AccessResult<String> {
    let mut bytes = [0u8; 16];
    getrandom::fill(&mut bytes)
        .map_err(|_| AccessError::Configuration("secure randomness is unavailable".into()))?;
    Ok(hex_bytes(&bytes))
}
fn ensure_external(path: &Path, source: &SourceIdentity) -> AccessResult<()> {
    let prospective = prospective_canonical(path)?;
    if prospective.starts_with(&source.canonical_path)
        || source.canonical_path.starts_with(&prospective)
    {
        return Err(AccessError::Configuration("access policy, credentials and machine data require a private directory outside the vault".into()));
    }
    Ok(())
}
fn prospective_canonical(path: &Path) -> AccessResult<PathBuf> {
    if !path.is_absolute()
        || path.components().any(|part| {
            matches!(
                part,
                std::path::Component::CurDir | std::path::Component::ParentDir
            )
        })
    {
        return Err(AccessError::Configuration(
            "access paths must be absolute and normalized".into(),
        ));
    }
    let mut ancestor = path;
    let mut missing = Vec::new();
    while !ancestor.exists() {
        missing.push(
            ancestor
                .file_name()
                .ok_or_else(|| AccessError::Configuration("invalid access path".into()))?
                .to_owned(),
        );
        ancestor = ancestor
            .parent()
            .ok_or_else(|| AccessError::Configuration("invalid access path".into()))?;
    }
    let mut result = ancestor.canonicalize()?;
    for name in missing.into_iter().rev() {
        result.push(name);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::access::clients::{Grant, Role};
    use std::sync::{Barrier, mpsc};
    use std::time::Duration;

    struct Fixture {
        _temporary: tempfile::TempDir,
        vault: PathBuf,
        store: AccessStore,
        token_path: PathBuf,
    }

    impl Fixture {
        fn new() -> Self {
            let temporary = tempfile::tempdir().unwrap();
            let vault = temporary.path().join("vault");
            std::fs::create_dir(&vault).unwrap();
            let profile = temporary.path().join("private/access.json");
            let data = temporary.path().join("private/data");
            let store = AccessStore::initialize(&profile, &vault, &data).unwrap();
            store
                .update(None, |policy| {
                    let mut client = ClientPolicy::default();
                    client
                        .grants
                        .push(Grant::directory("shared", "Shared/", Role::Editor));
                    policy.clients.insert("writer".into(), client);
                    Ok(())
                })
                .unwrap();
            let token_path = temporary.path().join("private/tokens/writer.token");
            Self {
                _temporary: temporary,
                vault,
                store,
                token_path,
            }
        }

        fn issue(&self) -> AuthorizedClient {
            self.store
                .issue_token("writer", &self.token_path, None)
                .unwrap();
            let token = read_token_file(&self.token_path).unwrap();
            self.store.authenticate(token.expose()).unwrap()
        }
    }

    #[test]
    fn credential_is_delivered_before_activation_and_diagnostics_redact_it() {
        let fixture = Fixture::new();
        let receipt = fixture
            .store
            .issue_token("writer", &fixture.token_path, None)
            .unwrap();
        let token = read_token_file(&fixture.token_path).unwrap();
        let text = format!("{token:?}");
        assert_eq!(text, "SecretToken([REDACTED])");
        assert!(!text.contains(token.expose()));
        let bytes = std::fs::read_to_string(fixture.store.path()).unwrap();
        assert!(!bytes.contains(token.expose()));
        let client = fixture.store.authenticate(token.expose()).unwrap();
        assert_eq!(client.credential_id(), receipt.credential_id);
        assert!(
            client
                .check(
                    AccessAction::Update,
                    Path::new("Shared/a.md"),
                    Path::new("Shared/a.md")
                )
                .is_ok()
        );
        assert!(matches!(
            client.check(
                AccessAction::Read,
                Path::new("Private/a.md"),
                Path::new("Private/a.md")
            ),
            Err(AccessError::Denied { visible: false, .. })
        ));
    }

    #[test]
    fn rotation_does_not_invalidate_the_old_credential_and_revoke_is_individual() {
        let fixture = Fixture::new();
        let old = fixture.issue();
        let next = fixture.token_path.with_file_name("second.token");
        fixture.store.issue_token("writer", &next, None).unwrap();
        let token = read_token_file(&next).unwrap();
        let new = fixture.store.authenticate(token.expose()).unwrap();
        assert!(old.current().is_ok());
        fixture
            .store
            .update(None, |policy| {
                policy
                    .clients
                    .get_mut("writer")
                    .unwrap()
                    .credentials
                    .iter_mut()
                    .find(|credential| credential.id == old.credential_id())
                    .unwrap()
                    .revoked = true;
                Ok(())
            })
            .unwrap();
        assert!(matches!(old.current(), Err(AccessError::InvalidCredential)));
        assert!(new.current().is_ok());
        assert!(
            fixture
                .store
                .ingestion_permit("writer", old.read_policy_key())
                .is_ok()
        );
    }

    #[test]
    fn changed_grant_or_disabled_client_invalidates_retained_authority() {
        let fixture = Fixture::new();
        let client = fixture.issue();
        fixture
            .store
            .update(None, |policy| {
                policy.clients.get_mut("writer").unwrap().grants.clear();
                Ok(())
            })
            .unwrap();
        assert!(matches!(
            client.current(),
            Err(AccessError::InvalidCredential)
        ));
        assert!(
            fixture
                .store
                .ingestion_permit("writer", client.read_policy_key())
                .is_err()
        );
        let token = read_token_file(&fixture.token_path).unwrap();
        let narrow = fixture.store.authenticate(token.expose()).unwrap();
        fixture
            .store
            .update(None, |policy| {
                policy.clients.get_mut("writer").unwrap().enabled = false;
                Ok(())
            })
            .unwrap();
        assert!(narrow.current().is_err());
    }

    #[test]
    fn repeated_rotation_prunes_only_inactive_credentials() {
        let fixture = Fixture::new();
        let mut previous: Option<AuthorizedClient> = None;
        let mut old_tokens = Vec::new();
        for index in 0..(MAX_CREDENTIALS_PER_CLIENT + 4) {
            let output = fixture
                .token_path
                .with_file_name(format!("rotation-{index}.token"));
            fixture.store.issue_token("writer", &output, None).unwrap();
            let token = read_token_file(&output).unwrap();
            let client = fixture.store.authenticate(token.expose()).unwrap();
            if let Some(previous) = previous {
                assert!(previous.current().is_ok());
                fixture
                    .store
                    .update(None, |policy| {
                        policy
                            .clients
                            .get_mut("writer")
                            .unwrap()
                            .credentials
                            .iter_mut()
                            .find(|credential| credential.id == previous.credential_id())
                            .unwrap()
                            .revoked = true;
                        Ok(())
                    })
                    .unwrap();
            }
            assert!(client.current().is_ok());
            assert!(
                fixture.store.snapshot().unwrap().policy.clients["writer"]
                    .credentials
                    .len()
                    <= 2
            );
            old_tokens.push(token);
            previous = Some(client);
        }
        let live = old_tokens.pop().unwrap();
        assert!(fixture.store.authenticate(live.expose()).is_ok());
        for token in old_tokens {
            assert!(fixture.store.authenticate(token.expose()).is_err());
        }
    }

    #[test]
    fn revoked_last_credential_withdraws_service_ingestion() {
        let fixture = Fixture::new();
        let client = fixture.issue();
        fixture
            .store
            .update(None, |policy| {
                policy.clients.get_mut("writer").unwrap().credentials[0].revoked = true;
                Ok(())
            })
            .unwrap();
        assert!(
            fixture
                .store
                .ingestion_permit("writer", client.read_policy_key())
                .is_err()
        );
    }

    #[test]
    fn corruption_fails_closed_and_valid_restoration_recovers() {
        let fixture = Fixture::new();
        let client = fixture.issue();
        let previous = std::fs::read(fixture.store.path()).unwrap();
        std::fs::write(fixture.store.path(), b"{ invalid").unwrap();
        assert!(matches!(
            client.current(),
            Err(AccessError::Configuration(_))
        ));
        std::fs::write(fixture.store.path(), previous).unwrap();
        assert!(client.current().is_ok());
    }

    #[test]
    fn revision_conflict_precedes_token_delivery() {
        let fixture = Fixture::new();
        let revision = fixture.store.snapshot().unwrap().policy.revision;
        fixture
            .store
            .update(Some(revision), |policy| {
                policy.clients.get_mut("writer").unwrap().enabled = false;
                Ok(())
            })
            .unwrap();
        let error = fixture
            .store
            .issue_token_at_revision("writer", &fixture.token_path, None, Some(revision))
            .unwrap_err();
        assert!(matches!(error, AccessError::Conflict { .. }));
        assert!(!fixture.token_path.exists());
    }

    #[test]
    fn client_and_initial_token_are_published_in_one_revision() {
        let fixture = Fixture::new();
        let before = fixture.store.snapshot().unwrap().policy.revision;
        let mut client = ClientPolicy::default();
        client
            .grants
            .push(Grant::directory("grant", "", Role::Reader));
        let receipt = fixture
            .store
            .create_client_with_token("reader", &client, &fixture.token_path, None, Some(before))
            .unwrap();
        assert_eq!(receipt.revision, before + 1);
        let token = read_token_file(&fixture.token_path).unwrap();
        assert_eq!(
            fixture
                .store
                .authenticate(token.expose())
                .unwrap()
                .client_id(),
            "reader"
        );
    }

    #[test]
    fn failed_initial_token_delivery_does_not_publish_the_client() {
        let fixture = Fixture::new();
        secure::fail_next(secure::FailurePoint::DeliveryFileSync);
        assert!(
            fixture
                .store
                .create_client_with_token(
                    "reader",
                    &ClientPolicy::default(),
                    &fixture.token_path,
                    None,
                    None
                )
                .is_err()
        );
        assert!(
            !fixture
                .store
                .snapshot()
                .unwrap()
                .policy
                .clients
                .contains_key("reader")
        );
    }

    #[test]
    fn token_output_collision_does_not_activate_another_credential() {
        let fixture = Fixture::new();
        fixture.issue();
        let previous = fixture.store.snapshot().unwrap();
        let bytes = std::fs::read(&fixture.token_path).unwrap();
        assert!(
            fixture
                .store
                .issue_token("writer", &fixture.token_path, None)
                .is_err()
        );
        assert_eq!(
            fixture.store.snapshot().unwrap().content_hash,
            previous.content_hash
        );
        assert_eq!(std::fs::read(&fixture.token_path).unwrap(), bytes);
    }

    #[test]
    fn independent_shared_permits_block_publication_until_all_are_released() {
        let fixture = Fixture::new();
        let client = fixture.issue();
        let first = client.permit().unwrap();
        let second = client.permit().unwrap();
        drop(first);
        assert!(matches!(
            fixture.store.lock(true, true),
            Err(AccessError::Busy)
        ));
        let store = fixture.store.clone();
        let barrier = Arc::new(Barrier::new(2));
        let other_barrier = Arc::clone(&barrier);
        let (sender, receiver) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            other_barrier.wait();
            store
                .update(None, |policy| {
                    policy.clients.get_mut("writer").unwrap().enabled = false;
                    Ok(())
                })
                .unwrap();
            sender.send(()).unwrap();
        });
        barrier.wait();
        assert!(receiver.recv_timeout(Duration::from_millis(100)).is_err());
        drop(second);
        receiver.recv_timeout(Duration::from_secs(3)).unwrap();
        worker.join().unwrap();
        assert!(client.current().is_err());
    }

    #[test]
    fn token_delivery_failures_never_activate_credentials() {
        use secure::FailurePoint::*;
        for point in [
            DeliveryCreate,
            DeliveryWrite,
            DeliveryFileSync,
            DeliveryDirectorySync,
        ] {
            let fixture = Fixture::new();
            let before = fixture.store.snapshot().unwrap().content_hash;
            secure::fail_next(point);
            assert!(
                fixture
                    .store
                    .issue_token("writer", &fixture.token_path, None)
                    .is_err()
            );
            assert_eq!(fixture.store.snapshot().unwrap().content_hash, before);
            if let Ok(token) = read_token_file(&fixture.token_path) {
                assert!(fixture.store.authenticate(token.expose()).is_err());
            }
        }
    }

    #[test]
    fn failed_policy_publication_preserves_delivered_token_and_truthful_outcome() {
        use secure::FailurePoint::*;
        for point in [
            PolicyCreate,
            PolicyWrite,
            PolicyFileSync,
            PolicyRename,
            PolicyDirectorySync,
        ] {
            let fixture = Fixture::new();
            let before = fixture.store.snapshot().unwrap().content_hash;
            secure::fail_next(point);
            let error = fixture
                .store
                .issue_token("writer", &fixture.token_path, None)
                .unwrap_err();
            let token = read_token_file(&fixture.token_path).unwrap();
            if point == PolicyDirectorySync {
                assert!(matches!(
                    error,
                    AccessError::Publication {
                        outcome: PublicationOutcome::Published,
                        ..
                    }
                ));
                assert!(fixture.store.authenticate(token.expose()).is_ok());
                assert_ne!(fixture.store.snapshot().unwrap().content_hash, before);
            } else {
                assert!(matches!(
                    error,
                    AccessError::Publication {
                        outcome: PublicationOutcome::NotPublished,
                        ..
                    }
                ));
                assert!(fixture.store.authenticate(token.expose()).is_err());
                assert_eq!(fixture.store.snapshot().unwrap().content_hash, before);
            }
        }
    }

    #[test]
    fn try_permit_does_not_wait_for_a_policy_writer() {
        let fixture = Fixture::new();
        let client = fixture.issue();
        let writer = fixture.store.lock(true, false).unwrap();
        assert!(matches!(client.try_permit(), Err(AccessError::Busy)));
        drop(writer);
        assert!(client.try_permit().is_ok());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn async_admission_allows_the_runtime_to_release_a_policy_writer() {
        let fixture = Fixture::new();
        let client = fixture.issue();
        let writer = fixture.store.lock(true, false).unwrap();
        let (release, released) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            // A blocking regression must fail without hanging the test runner.
            let _ = released.recv_timeout(Duration::from_secs(1));
            drop(writer);
        });
        let admission = client.current_async();
        tokio::pin!(admission);
        tokio::select! {
            result = &mut admission => panic!("admission bypassed the policy writer: {result:?}"),
            _ = tokio::time::sleep(Duration::from_millis(20)) => release.send(()).unwrap(),
        }
        tokio::time::timeout(Duration::from_secs(1), admission)
            .await
            .unwrap()
            .unwrap();
        worker.join().unwrap();
        // current_async must drop its shared permit before returning.
        assert!(fixture.store.lock(true, true).is_ok());
    }

    #[test]
    fn replaced_lock_file_is_not_a_new_authorization_domain() {
        let fixture = Fixture::new();
        let client = fixture.issue();
        let lock_path = fixture
            .store
            .0
            .directory
            .path()
            .join(&fixture.store.0.lock_name);
        std::fs::rename(&lock_path, lock_path.with_extension("old-lock")).unwrap();
        write_protected_file(&lock_path, b"").unwrap();
        assert!(matches!(
            client.current(),
            Err(AccessError::Configuration(_))
        ));
    }

    #[test]
    fn initialization_preflight_has_no_filesystem_side_effects() {
        let temporary = tempfile::tempdir().unwrap();
        let vault = temporary.path().join("vault");
        std::fs::create_dir(&vault).unwrap();
        let profile = temporary.path().join("new-profile/access.json");
        let data = temporary.path().join("new-profile/data");
        validate_initialization(&profile, &vault, &data).unwrap();
        assert!(!profile.parent().unwrap().exists());
        assert!(!vault.join(".obsidian-mcp").exists());
        assert!(
            validate_initialization(&vault.join("private/access.json"), &vault, &data).is_err()
        );
        assert!(!vault.join("private").exists());
        assert!(validate_initialization(&profile, &vault, &profile).is_err());
        assert!(validate_initialization(&profile, &vault, &profile.join("data")).is_err());
        assert!(!profile.parent().unwrap().exists());
    }

    #[test]
    fn concurrent_updates_preserve_both_changes() {
        let fixture = Fixture::new();
        let barrier = Arc::new(Barrier::new(3));
        let mut workers = Vec::new();
        for name in ["first", "second"] {
            let store = fixture.store.clone();
            let barrier = Arc::clone(&barrier);
            workers.push(std::thread::spawn(move || {
                barrier.wait();
                store
                    .update(None, |policy| {
                        policy.clients.insert(name.into(), ClientPolicy::default());
                        Ok(())
                    })
                    .unwrap();
            }));
        }
        barrier.wait();
        for worker in workers {
            worker.join().unwrap();
        }
        let snapshot = fixture.store.snapshot().unwrap();
        assert!(snapshot.policy.clients.contains_key("first"));
        assert!(snapshot.policy.clients.contains_key("second"));
    }

    #[test]
    fn same_path_root_replacement_requires_owner_rebind_and_service_reopen() {
        let fixture = Fixture::new();
        let client = fixture.issue();
        let old = fixture.vault.with_file_name("old-vault");
        std::fs::rename(&fixture.vault, &old).unwrap();
        std::fs::create_dir(&fixture.vault).unwrap();
        assert!(client.current().is_err());
        let source = SourceIdentity::capture(&fixture.vault).unwrap();
        fixture
            .store
            .update(None, |policy| {
                policy.source = source;
                Ok(())
            })
            .unwrap();
        assert!(client.current().is_err());
        let reopened = AccessStore::open(fixture.store.path()).unwrap();
        reopened.validate_source(&fixture.vault).unwrap();
        let token = read_token_file(&fixture.token_path).unwrap();
        assert!(reopened.authenticate(token.expose()).is_ok());
    }

    #[test]
    fn profiles_and_token_outputs_cannot_be_placed_inside_the_source() {
        let temporary = tempfile::tempdir().unwrap();
        let vault = temporary.path().join("vault");
        std::fs::create_dir(&vault).unwrap();
        let profile = vault.join("private/access.json");
        let data = temporary.path().join("data");
        assert!(AccessStore::initialize(&profile, &vault, &data).is_err());
        assert!(!vault.join("private").exists());
        let fixture = Fixture::new();
        let output = fixture.vault.join("tokens/client.token");
        assert!(fixture.store.issue_token("writer", &output, None).is_err());
        assert!(!fixture.vault.join("tokens").exists());
    }

    #[cfg(unix)]
    #[test]
    fn unsafe_modes_symlinks_hardlinks_and_special_files_are_rejected() {
        use std::os::unix::fs::{PermissionsExt, symlink};
        let fixture = Fixture::new();
        let client = fixture.issue();
        std::fs::set_permissions(fixture.store.path(), std::fs::Permissions::from_mode(0o644))
            .unwrap();
        assert!(client.current().is_err());
        std::fs::set_permissions(fixture.store.path(), std::fs::Permissions::from_mode(0o600))
            .unwrap();
        assert!(client.current().is_ok());
        let alias = fixture.token_path.with_file_name("alias.token");
        symlink(&fixture.token_path, &alias).unwrap();
        assert!(read_token_file(&alias).is_err());
        std::fs::remove_file(&alias).unwrap();
        std::fs::hard_link(&fixture.token_path, &alias).unwrap();
        assert!(read_token_file(&alias).is_err());
        assert!(read_token_file(&fixture.token_path).is_err());
    }
}
