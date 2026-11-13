//! Owner-provisioned client policies, independent of transport and source I/O.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path, PathBuf};

use icu_casemap::CaseMapper;
use icu_normalizer::ComposingNormalizer;
use icu_properties::{CodePointSetData, props::DefaultIgnorableCodePoint};
use serde::de::{self, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use sha2::{Digest, Sha256};

pub const ACCESS_VERSION: u32 = 1;
pub const DENY_GUARD_VERSION: &str = "nfkc-casefold-icu4x-2.3";
pub const MAX_POLICY_BYTES: usize = 1024 * 1024;
pub const MAX_CLIENTS: usize = 256;
pub const MAX_RULES_PER_CLIENT: usize = 256;
pub const MAX_CREDENTIALS_PER_CLIENT: usize = 32;
const MAX_POLICY_RULES: usize = 4096;
const MAX_PATH_BYTES: usize = 4096;
const MAX_PATH_COMPONENTS: usize = 128;

#[derive(thiserror::Error)]
pub enum PolicyError {
    #[error("invalid access policy: {0}")]
    Invalid(&'static str),
    #[error(
        "invalid access policy JSON ({:?}) at line {} column {}",
        .0.classify(), .0.line(), .0.column()
    )]
    Json(#[from] serde_json::Error),
}

impl std::fmt::Debug for PolicyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(message) => f.debug_tuple("Invalid").field(message).finish(),
            Self::Json(error) => f
                .debug_struct("InvalidJson")
                .field("category", &error.classify())
                .field("line", &error.line())
                .field("column", &error.column())
                .finish(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SourceIdentity {
    pub canonical_path: PathBuf,
    pub storage_key: String,
    pub object_id: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AccessPolicy {
    pub version: u32,
    pub deny_guard: String,
    pub revision: u64,
    pub source: SourceIdentity,
    pub data_dir: PathBuf,
    #[serde(deserialize_with = "unique_clients")]
    pub clients: BTreeMap<String, ClientPolicy>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ClientPolicy {
    pub enabled: bool,
    #[serde(default)]
    pub grants: Vec<Grant>,
    #[serde(default)]
    pub denies: Vec<Deny>,
    #[serde(default)]
    pub credentials: Vec<Credential>,
}

impl Default for ClientPolicy {
    fn default() -> Self {
        Self {
            enabled: true,
            grants: Vec::new(),
            denies: Vec::new(),
            credentials: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    Reader,
    Editor,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum AccessAction {
    Read,
    Create,
    Update,
    Delete,
}

/// Operation authority used by source and credential admission.
pub type Action = AccessAction;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Grant {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub directory: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    pub role: Role,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Deny {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub directory: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
}

#[derive(Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Credential {
    pub id: String,
    pub digest: String,
    pub created_at: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<i64>,
    #[serde(default)]
    pub revoked: bool,
}

impl std::fmt::Debug for Credential {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Credential")
            .field("id", &self.id)
            .field("created_at", &self.created_at)
            .field("expires_at", &self.expires_at)
            .field("revoked", &self.revoked)
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct AccessDecision {
    pub allowed: bool,
    pub reason: DecisionReason,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rule_id: Option<String>,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DecisionReason {
    Allowed,
    Disabled,
    ReservedPath,
    Denied,
    NoGrant,
    InvalidPath,
}

impl AccessPolicy {
    pub fn new(source: SourceIdentity, data_dir: PathBuf) -> Self {
        Self {
            version: ACCESS_VERSION,
            deny_guard: DENY_GUARD_VERSION.into(),
            revision: 1,
            source,
            data_dir,
            clients: BTreeMap::new(),
        }
    }

    pub fn parse(bytes: &[u8]) -> Result<Self, PolicyError> {
        if bytes.len() > MAX_POLICY_BYTES {
            return invalid("store exceeds the maximum policy size");
        }
        let policy: Self = serde_json::from_slice(bytes)?;
        policy.validate()?;
        Ok(policy)
    }

    pub fn validate(&self) -> Result<(), PolicyError> {
        if self.version != ACCESS_VERSION {
            return invalid("unsupported schema version");
        }
        if self.deny_guard != DENY_GUARD_VERSION {
            return invalid("unsupported deny matching version");
        }
        if self.revision == 0 {
            return invalid("revision must be positive");
        }
        if !self.source.canonical_path.is_absolute()
            || !self.data_dir.is_absolute()
            || self.data_dir == self.source.canonical_path
            || !valid_digest(&self.source.storage_key)
            || self.source.object_id.is_empty()
            || self.source.object_id.len() > 256
            || self
                .source
                .object_id
                .chars()
                .any(|character| character.is_control())
        {
            return invalid("invalid source binding or machine-data directory");
        }
        if self.clients.len() > MAX_CLIENTS {
            return invalid("too many clients");
        }
        let mut rule_count = 0usize;
        let mut credential_ids = BTreeSet::new();
        for (id, client) in &self.clients {
            validate_client_id(id)?;
            client.validate()?;
            rule_count += client.grants.len() + client.denies.len();
            for credential in &client.credentials {
                if !credential_ids.insert(&credential.id) {
                    return invalid("credential IDs must be unique across clients");
                }
            }
        }
        if rule_count > MAX_POLICY_RULES {
            return invalid("too many policy rules");
        }
        Ok(())
    }

    pub fn client(&self, id: &str) -> Option<&ClientPolicy> {
        self.clients.get(id)
    }

    /// Source-bound identity of the readable policy, independent of credentials.
    pub fn read_policy_hash(&self, client_id: &str) -> Option<String> {
        let client = self.clients.get(client_id)?;
        let mut hash = Sha256::new();
        hash_field(&mut hash, b"obsidian-mcp/access-source-read/v1");
        hash_field(&mut hash, self.source.storage_key.as_bytes());
        hash_field(&mut hash, self.source.object_id.as_bytes());
        hash_field(&mut hash, client.read_policy_hash().as_bytes());
        Some(hex_digest(hash))
    }
}

impl ClientPolicy {
    pub fn validate(&self) -> Result<(), PolicyError> {
        if self.grants.len() + self.denies.len() > MAX_RULES_PER_CLIENT {
            return invalid("too many rules for one client");
        }
        if self.credentials.len() > MAX_CREDENTIALS_PER_CLIENT {
            return invalid("too many credentials for one client");
        }
        let mut ids = BTreeSet::new();
        for grant in &self.grants {
            validate_id(&grant.id)?;
            if !ids.insert(&grant.id) {
                return invalid("duplicate rule ID");
            }
            resource(&grant.directory, &grant.file)?;
        }
        for deny in &self.denies {
            validate_id(&deny.id)?;
            if !ids.insert(&deny.id) {
                return invalid("duplicate rule ID");
            }
            resource(&deny.directory, &deny.file)?;
        }
        let mut credentials = BTreeSet::new();
        for credential in &self.credentials {
            validate_id(&credential.id)?;
            if !credentials.insert(&credential.id) {
                return invalid("duplicate credential ID");
            }
            if !valid_digest(&credential.digest)
                || credential.created_at < 0
                || credential
                    .expires_at
                    .is_some_and(|expiry| expiry <= credential.created_at)
            {
                return invalid("invalid credential digest or lifetime");
            }
        }
        Ok(())
    }

    pub fn allows(&self, action: AccessAction, path: &Path) -> bool {
        self.decision(action, path).allowed
    }

    pub fn decision(&self, action: AccessAction, path: &Path) -> AccessDecision {
        let decision = |allowed, reason, rule_id| AccessDecision {
            allowed,
            reason,
            rule_id,
        };
        if !self.enabled {
            return decision(false, DecisionReason::Disabled, None);
        }
        let Some(parts) = runtime_components(path) else {
            return decision(false, DecisionReason::InvalidPath, None);
        };
        if parts.iter().any(|part| reserved_component(part)) {
            return decision(false, DecisionReason::ReservedPath, None);
        }
        let guarded = guard_components(&parts);
        for deny in &self.denies {
            let Ok(selector) = resource(&deny.directory, &deny.file) else {
                return decision(false, DecisionReason::Denied, Some(deny.id.clone()));
            };
            if selector.matches_guarded(&guarded) {
                return decision(false, DecisionReason::Denied, Some(deny.id.clone()));
            }
        }
        for grant in &self.grants {
            if grant.role == Role::Reader && action != AccessAction::Read {
                continue;
            }
            if resource(&grant.directory, &grant.file)
                .is_ok_and(|selector| selector.matches(&parts))
            {
                return decision(true, DecisionReason::Allowed, Some(grant.id.clone()));
            }
        }
        decision(false, DecisionReason::NoGrant, None)
    }

    /// A directory grant or a granted descendant permits namespace traversal only.
    pub fn allows_container(&self, path: &Path) -> bool {
        if !self.enabled {
            return false;
        }
        let Some(parts) = runtime_components(path) else {
            return false;
        };
        if parts.iter().any(|part| reserved_component(part)) {
            return false;
        }
        let guarded = guard_components(&parts);
        if self.denies.iter().any(|deny| {
            resource(&deny.directory, &deny.file)
                .map_or(true, |selector| selector.matches_guarded(&guarded))
        }) {
            return false;
        }
        self.grants.iter().any(|grant| {
            resource(&grant.directory, &grant.file).is_ok_and(|selector| {
                selector.matches(&parts) || selector.components.starts_with(&parts)
            })
        })
    }

    pub fn has_write_access(&self) -> bool {
        self.enabled && self.grants.iter().any(|grant| grant.role == Role::Editor)
    }

    /// Equal keys mean equal future read admission, not merely equal current files.
    pub fn read_policy_hash(&self) -> String {
        let mut hash = Sha256::new();
        hash_field(&mut hash, b"obsidian-mcp/access-readable/v1");
        hash_field(&mut hash, DENY_GUARD_VERSION.as_bytes());
        let resources = self
            .grants
            .iter()
            .map(|grant| selector_key(&grant.directory, &grant.file, false))
            .collect::<BTreeSet<_>>();
        for selector in resources {
            hash_field(&mut hash, b"grant");
            hash_field(&mut hash, selector.as_bytes());
        }
        self.hash_denials(&mut hash);
        hex_digest(hash)
    }

    /// Rotation metadata does not invalidate sessions or change client permissions.
    pub fn authorization_hash(&self) -> String {
        let mut hash = Sha256::new();
        hash_field(&mut hash, b"obsidian-mcp/access-authorization/v1");
        hash_field(&mut hash, DENY_GUARD_VERSION.as_bytes());
        hash_field(&mut hash, &[u8::from(self.enabled)]);
        let resources = self
            .grants
            .iter()
            .map(|grant| {
                (
                    selector_key(&grant.directory, &grant.file, false),
                    grant.role,
                )
            })
            .collect::<BTreeSet<_>>();
        for (selector, role) in resources {
            hash_field(&mut hash, b"grant");
            hash_field(&mut hash, selector.as_bytes());
            hash_field(
                &mut hash,
                match role {
                    Role::Reader => b"reader",
                    Role::Editor => b"editor",
                },
            );
        }
        self.hash_denials(&mut hash);
        hex_digest(hash)
    }

    fn hash_denials(&self, hash: &mut Sha256) {
        for selector in self
            .denies
            .iter()
            .map(|deny| selector_key(&deny.directory, &deny.file, true))
            .collect::<BTreeSet<_>>()
        {
            hash_field(hash, b"deny");
            hash_field(hash, selector.as_bytes());
        }
    }
}

impl Grant {
    pub fn directory(id: impl Into<String>, path: impl Into<String>, role: Role) -> Self {
        Self {
            id: id.into(),
            directory: Some(path.into()),
            file: None,
            role,
        }
    }

    pub fn file(id: impl Into<String>, path: impl Into<String>, role: Role) -> Self {
        Self {
            id: id.into(),
            directory: None,
            file: Some(path.into()),
            role,
        }
    }
}

impl Deny {
    pub fn directory(id: impl Into<String>, path: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            directory: Some(path.into()),
            file: None,
        }
    }

    pub fn file(id: impl Into<String>, path: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            directory: None,
            file: Some(path.into()),
        }
    }
}

/// Policy paths use portable separators and retain native case and Unicode spelling.
pub fn normalize_resource_path(path: &str, is_directory: bool) -> Result<String, PolicyError> {
    if path.len() > MAX_PATH_BYTES || path.starts_with('/') || path.contains('\\') {
        return invalid(
            "resource paths must be relative and use / separators; use an empty directory for the vault root",
        );
    }
    let path = if is_directory {
        path.strip_suffix('/').unwrap_or(path)
    } else {
        path
    };
    if path.is_empty() && is_directory {
        return Ok(String::new());
    }
    let parts = path.split('/').collect::<Vec<_>>();
    if parts.len() > MAX_PATH_COMPONENTS
        || parts.iter().any(|part| !valid_component(part))
        || Path::new(path)
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
    {
        return invalid("resource path must be a relative visible file or directory");
    }
    Ok(path.to_owned())
}

pub fn validate_client_id(id: &str) -> Result<(), PolicyError> {
    validate_id(id)
}

pub fn validate_id(id: &str) -> Result<(), PolicyError> {
    if id.is_empty()
        || id.len() > 128
        || !id.as_bytes()[0].is_ascii_alphanumeric()
        || !id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return invalid(
            "IDs must start with a letter or digit and use only letters, digits, - or _",
        );
    }
    Ok(())
}

struct Resource {
    directory: bool,
    components: Vec<String>,
}

impl Resource {
    fn matches(&self, path: &[String]) -> bool {
        if self.directory {
            path.starts_with(&self.components)
        } else {
            path == self.components
        }
    }

    fn matches_guarded(&self, path: &[String]) -> bool {
        let components = guard_components(&self.components);
        if self.directory {
            path.starts_with(&components)
        } else {
            path == components
        }
    }
}

fn resource(directory: &Option<String>, file: &Option<String>) -> Result<Resource, PolicyError> {
    let (path, is_directory) = match (directory, file) {
        (Some(path), None) => (path, true),
        (None, Some(path)) => (path, false),
        _ => return invalid("each rule requires exactly one directory or file selector"),
    };
    let normalized = normalize_resource_path(path, is_directory)?;
    Ok(Resource {
        directory: is_directory,
        components: if normalized.is_empty() {
            Vec::new()
        } else {
            normalized.split('/').map(str::to_owned).collect()
        },
    })
}

fn runtime_components(path: &Path) -> Option<Vec<String>> {
    if path.as_os_str().len() > MAX_PATH_BYTES {
        return None;
    }
    let parts = path
        .components()
        .map(|part| match part {
            Component::Normal(part) => part.to_str().map(str::to_owned),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;
    if parts.len() > MAX_PATH_COMPONENTS
        || parts.iter().any(|part| {
            part.is_empty()
                || part.contains(['/', '\\', ':'])
                || part.chars().any(char::is_control)
                || part.ends_with(['.', ' '])
                || windows_device_name(part)
        })
    {
        return None;
    }
    Some(parts)
}

fn valid_component(part: &str) -> bool {
    !part.is_empty()
        && !reserved_component(part)
        && !part.contains(['\\', ':'])
        && !part.chars().any(char::is_control)
        && !part.ends_with(['.', ' '])
        && !windows_device_name(part)
}

fn reserved_component(part: &str) -> bool {
    part.starts_with('.') || deny_guard_component(part).starts_with('.')
}

fn windows_device_name(part: &str) -> bool {
    let stem = part.split('.').next().unwrap_or(part).to_ascii_uppercase();
    matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        || stem.strip_prefix("COM").is_some_and(device_number)
        || stem.strip_prefix("LPT").is_some_and(device_number)
}

fn device_number(value: &str) -> bool {
    matches!(
        value,
        "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | "¹" | "²" | "³"
    )
}

fn guard_components(parts: &[String]) -> Vec<String> {
    parts
        .iter()
        .map(|part| deny_guard_component(part))
        .collect()
}

/// Never split this output: compatibility normalization can produce separator characters.
fn deny_guard_component(part: &str) -> String {
    let nfkc = ComposingNormalizer::new_nfkc();
    let normalized = nfkc.normalize(part);
    let folded = CaseMapper::new().fold_string(&normalized);
    let ignorables = CodePointSetData::new::<DefaultIgnorableCodePoint>();
    let visible = folded
        .chars()
        .filter(|character| !ignorables.contains(*character))
        .collect::<String>();
    nfkc.normalize(&visible).into_owned()
}

fn selector_key(directory: &Option<String>, file: &Option<String>, guarded: bool) -> String {
    match resource(directory, file) {
        Ok(selector) => {
            let mut key = if selector.directory { "d" } else { "f" }.to_owned();
            let parts = if guarded {
                guard_components(&selector.components)
            } else {
                selector.components
            };
            for part in parts {
                key.push_str(&format!("{}:{part}", part.len()));
            }
            key
        }
        // Invalid policy instances never alias a valid selector identity.
        Err(_) => format!("invalid:{directory:?}:{file:?}"),
    }
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
}

fn hash_field(hash: &mut Sha256, value: &[u8]) {
    hash.update((value.len() as u64).to_le_bytes());
    hash.update(value);
}

fn hex_digest(hash: Sha256) -> String {
    hash.finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn invalid<T>(message: &'static str) -> Result<T, PolicyError> {
    Err(PolicyError::Invalid(message))
}

fn unique_clients<'de, D>(deserializer: D) -> Result<BTreeMap<String, ClientPolicy>, D::Error>
where
    D: Deserializer<'de>,
{
    struct UniqueClients;
    impl<'de> Visitor<'de> for UniqueClients {
        type Value = BTreeMap<String, ClientPolicy>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("a client object with unique IDs")
        }

        fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
        where
            A: MapAccess<'de>,
        {
            let mut clients = BTreeMap::new();
            while let Some(id) = map.next_key::<String>()? {
                if clients.len() >= MAX_CLIENTS {
                    return Err(de::Error::custom("too many clients"));
                }
                if clients.contains_key(&id) {
                    return Err(de::Error::custom("duplicate client ID"));
                }
                clients.insert(id, map.next_value()?);
            }
            Ok(clients)
        }
    }
    deserializer.deserialize_map(UniqueClients)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source() -> SourceIdentity {
        SourceIdentity {
            canonical_path: std::env::temp_dir().join("vault"),
            storage_key: "a".repeat(64),
            object_id: "native-root-1".into(),
        }
    }

    fn client() -> ClientPolicy {
        ClientPolicy {
            grants: vec![Grant::directory("root", "", Role::Editor)],
            ..Default::default()
        }
    }

    fn policy() -> AccessPolicy {
        let mut policy = AccessPolicy::new(source(), std::env::temp_dir().join("data"));
        policy.clients.insert("assistant".into(), client());
        policy
    }

    #[test]
    fn denials_override_grants_regardless_of_order_and_specificity() {
        let mut client = client();
        client.denies = vec![Deny::directory("private", "Private/")];
        client
            .grants
            .push(Grant::file("exception", "Private/allowed.md", Role::Editor));
        for action in [
            AccessAction::Read,
            AccessAction::Create,
            AccessAction::Update,
            AccessAction::Delete,
        ] {
            assert!(!client.allows(action, Path::new("Private/allowed.md")));
            assert!(!client.allows(action, Path::new("Private/new/descendant.md")));
            assert!(client.allows(action, Path::new("Private-old/note.md")));
        }
        client.grants.reverse();
        assert!(!client.allows(AccessAction::Read, Path::new("Private/allowed.md")));
        client.denies.push(Deny::directory("all", ""));
        assert!(!client.allows(AccessAction::Read, Path::new("anything.md")));
    }

    #[test]
    fn reader_operations_and_exact_file_traversal_do_not_authorize_siblings() {
        let client = ClientPolicy {
            grants: vec![Grant::file("one", "Shared/Deep/note.md", Role::Reader)],
            ..Default::default()
        };
        assert!(client.allows(AccessAction::Read, Path::new("Shared/Deep/note.md")));
        assert!(!client.allows(AccessAction::Create, Path::new("Shared/Deep/note.md")));
        assert!(!client.allows(AccessAction::Read, Path::new("Shared/Deep/note.md-more")));
        assert!(!client.allows(AccessAction::Read, Path::new("Shared/Deep/other.md")));
        assert!(client.allows_container(Path::new("")));
        assert!(client.allows_container(Path::new("Shared")));
        assert!(client.allows_container(Path::new("Shared/Deep")));
        assert!(!client.allows_container(Path::new("Shared/Other")));
    }

    #[test]
    fn deny_guard_covers_missing_case_compatibility_and_ignorable_variants() {
        let mut client = client();
        client.denies = vec![
            Deny::directory("private", "Private"),
            Deny::file("german", "Straße.md"),
            Deny::directory("composed", "Café"),
        ];
        for path in [
            "PRIVATE/new.md",
            "Ｐｒｉｖａｔｅ/new.md",
            "Pri\u{200d}vate/new.md",
            "STRASSE.md",
            "Cafe\u{301}/new.md",
        ] {
            assert!(
                !client.allows(AccessAction::Create, Path::new(path)),
                "{path}"
            );
        }
        assert!(client.allows(AccessAction::Read, Path::new("Private-old/note.md")));
        let narrow = ClientPolicy {
            grants: vec![Grant::directory("only", "Shared", Role::Reader)],
            ..Default::default()
        };
        assert!(!narrow.allows(AccessAction::Read, Path::new("Ｓｈａｒｅｄ/note.md")));
    }

    #[test]
    fn normalization_does_not_turn_a_component_into_multiple_components() {
        let mut client = client();
        client.denies = vec![Deny::directory("private", "A/B")];
        assert!(client.allows(AccessAction::Read, Path::new("A／B/note.md")));
        assert_eq!(deny_guard_component("A／B"), "a/b");
    }

    #[test]
    fn root_editor_cannot_access_owner_metadata_or_ambiguous_names() {
        let client = client();
        for path in [
            ".obsidian/plugins/plugin/main.js",
            "Shared/.git/hooks/run",
            ".obsidian-mcp/embeddings/cache",
            "．obsidian/config",
            "../outside.md",
            "/outside.md",
            "Shared/note.md:stream",
            "Shared/note.md.",
            "Shared/NUL.txt",
            "Shared/COM¹.txt",
            "Shared/a\u{0}b.md",
        ] {
            assert!(
                !client.allows(AccessAction::Read, Path::new(path)),
                "{path}"
            );
            assert!(!client.allows_container(Path::new(path)), "{path}");
        }
    }

    #[test]
    fn readable_keys_ignore_roles_credentials_order_and_rule_ids_but_not_future_scope() {
        let mut first = ClientPolicy {
            grants: vec![
                Grant::directory("one", "Shared/", Role::Reader),
                Grant::file("two", "Exact.md", Role::Reader),
            ],
            denies: vec![Deny::directory("deny", "PRIVATE")],
            ..Default::default()
        };
        let mut second = first.clone();
        second.grants.reverse();
        second.grants[0].id = "changed".into();
        second.grants[0].role = Role::Editor;
        second.grants[1].directory = Some("Shared".into());
        second.denies[0].directory = Some("Private/".into());
        second.credentials.push(Credential {
            id: "token".into(),
            digest: "a".repeat(64),
            created_at: 1,
            expires_at: None,
            revoked: false,
        });
        assert_eq!(first.read_policy_hash(), second.read_policy_hash());
        assert_ne!(first.authorization_hash(), second.authorization_hash());
        first.grants = vec![Grant::file("one", "Shared/existing.md", Role::Reader)];
        assert_ne!(first.read_policy_hash(), second.read_policy_hash());
        let before = second.authorization_hash();
        second.credentials[0].revoked = true;
        assert_eq!(before, second.authorization_hash());
    }

    #[test]
    fn strict_json_rejects_duplicate_clients_unknown_fields_and_unknown_versions() {
        let bytes = serde_json::to_vec(&policy()).unwrap();
        assert!(AccessPolicy::parse(&bytes).is_ok());
        let client = serde_json::to_string(&client()).unwrap();
        let duplicate = String::from_utf8(bytes.clone()).unwrap().replace(
            &format!("\"assistant\":{client}"),
            &format!("\"assistant\":{client},\"assistant\":{client}"),
        );
        assert!(AccessPolicy::parse(duplicate.as_bytes()).is_err());
        for (field, value) in [
            ("version", serde_json::json!(2)),
            ("deny_guard", serde_json::json!("other")),
        ] {
            let mut value_object = serde_json::to_value(policy()).unwrap();
            value_object[field] = value;
            assert!(AccessPolicy::parse(&serde_json::to_vec(&value_object).unwrap()).is_err());
        }
        let mut value = serde_json::to_value(policy()).unwrap();
        value["unexpected"] = serde_json::json!(true);
        assert!(AccessPolicy::parse(&serde_json::to_vec(&value).unwrap()).is_err());
    }

    #[test]
    fn malformed_policy_diagnostics_redact_values_and_unknown_field_names() {
        let canary = "omcp1_synthetic_secret_diagnostic_canary";
        let mut invalid_role = serde_json::to_value(policy()).unwrap();
        invalid_role["clients"]["assistant"]["grants"][0]["role"] = serde_json::json!(canary);
        let mut invalid_enabled = serde_json::to_value(policy()).unwrap();
        invalid_enabled["clients"]["assistant"]["enabled"] = serde_json::json!(canary);
        let mut unknown_field = serde_json::to_value(policy()).unwrap();
        unknown_field[canary] = serde_json::json!(true);
        for value in [invalid_role, invalid_enabled, unknown_field] {
            let error =
                AccessPolicy::parse(&serde_json::to_vec_pretty(&value).unwrap()).unwrap_err();
            let display = error.to_string();
            let debug = format!("{error:?}");
            assert!(display.starts_with("invalid access policy JSON (Data) at line "));
            assert!(!display.contains(canary));
            assert!(!debug.contains(canary));
            assert!(debug.contains("category: Data"));
            let source = std::error::Error::source(&error).unwrap();
            assert!(source.to_string().contains(canary));
        }
    }

    #[test]
    fn validation_rejects_conflicting_selectors_duplicate_ids_and_bad_paths() {
        let mut policy = policy();
        for bad in [
            "/",
            "../a",
            "Shared//",
            "Shared//note.md",
            ".obsidian",
            "C:/secret",
            "Shared\\note.md",
        ] {
            policy.clients.get_mut("assistant").unwrap().grants =
                vec![Grant::directory("one", bad, Role::Reader)];
            assert!(policy.validate().is_err(), "{bad}");
        }
        policy = self::policy();
        let client = policy.clients.get_mut("assistant").unwrap();
        client.grants[0].file = Some("note.md".into());
        assert!(client.validate().is_err());
        client.grants[0].file = None;
        client.denies.push(Deny::directory("root", "Private"));
        assert!(client.validate().is_err());
        assert!(AccessPolicy::parse(&vec![b' '; MAX_POLICY_BYTES + 1]).is_err());
    }
}
