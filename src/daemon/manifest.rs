//! Typed semantic runtime manifest schema and IO helpers.

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use chrono::Utc;
use serde::{Deserialize, Serialize};

use crate::error::{VaultError, VaultResult};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeManifest {
    pub schema_version: u32,
    pub daemon_api_version: u32,
    pub daemon_version: String,
    pub binary_path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub binary_sha256: Option<String>,
    pub ipc: ManifestIpc,
    pub pid: u32,
    pub semantic_home: String,
    pub fastembed_cache_dir: String,
    pub model_name: String,
    pub created_at: String,
    pub updated_at: String,
    pub last_health_check_at: String,
    pub last_bootstrap_client: String,
    pub last_bootstrap_client_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestIpc {
    pub transport: String,
    pub endpoint: String,
}

#[derive(Debug, Clone)]
pub struct RuntimeManifestInput {
    pub daemon_api_version: u32,
    pub daemon_version: String,
    pub binary_path: String,
    pub binary_sha256: Option<String>,
    pub ipc: ManifestIpc,
    pub pid: u32,
    pub semantic_home: String,
    pub fastembed_cache_dir: String,
    pub model_name: String,
    pub bootstrap_client_name: String,
    pub bootstrap_client_version: String,
}

impl RuntimeManifest {
    pub fn from_input(input: RuntimeManifestInput) -> Self {
        let now = now_rfc3339();
        Self {
            schema_version: 1,
            daemon_api_version: input.daemon_api_version,
            daemon_version: input.daemon_version,
            binary_path: input.binary_path,
            binary_sha256: input.binary_sha256,
            ipc: input.ipc,
            pid: input.pid,
            semantic_home: input.semantic_home,
            fastembed_cache_dir: input.fastembed_cache_dir,
            model_name: input.model_name,
            created_at: now.clone(),
            updated_at: now.clone(),
            last_health_check_at: now,
            last_bootstrap_client: input.bootstrap_client_name,
            last_bootstrap_client_version: input.bootstrap_client_version,
        }
    }

    pub fn touch_health(&mut self) {
        let now = now_rfc3339();
        self.updated_at = now.clone();
        self.last_health_check_at = now;
    }
}

pub fn load(path: &Path) -> VaultResult<Option<RuntimeManifest>> {
    let content = match std::fs::read_to_string(path) {
        Ok(content) => content,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(VaultError::Io(err)),
    };

    let manifest = serde_json::from_str::<RuntimeManifest>(&content).map_err(|err| {
        VaultError::DaemonBootstrap(format!(
            "failed to parse semantic manifest '{}': {err}",
            path.display()
        ))
    })?;
    Ok(Some(manifest))
}

pub fn save_atomic(path: &Path, manifest: &RuntimeManifest) -> VaultResult<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let encoded = serde_json::to_vec_pretty(manifest).map_err(|err| {
        VaultError::DaemonBootstrap(format!(
            "failed to serialize semantic manifest '{}': {err}",
            path.display()
        ))
    })?;

    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    let tmp_path = path.with_extension(format!("tmp-{}-{nonce}", std::process::id()));

    std::fs::write(&tmp_path, encoded)?;
    std::fs::rename(&tmp_path, path)?;
    Ok(())
}

pub fn now_rfc3339() -> String {
    Utc::now().to_rfc3339()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_roundtrip() {
        let dir = tempfile::tempdir().expect("tempdir should be created");
        let path = dir.path().join("manifest.json");
        let manifest = RuntimeManifest::from_input(RuntimeManifestInput {
            daemon_api_version: 1,
            daemon_version: "1.0.1".to_string(),
            binary_path: "/tmp/obsidian-semanticd".to_string(),
            binary_sha256: Some("abc123".to_string()),
            ipc: ManifestIpc {
                transport: "unix_socket".to_string(),
                endpoint: "/tmp/semanticd.sock".to_string(),
            },
            pid: 1234,
            semantic_home: "/tmp/semantic-home".to_string(),
            fastembed_cache_dir: "/tmp/semantic-home/model/fastembed-cache".to_string(),
            model_name: "BAAI/bge-small-en-v1.5".to_string(),
            bootstrap_client_name: "obsidian-mcp".to_string(),
            bootstrap_client_version: "1.0.1".to_string(),
        });

        save_atomic(&path, &manifest).expect("manifest write should succeed");
        let loaded = load(&path)
            .expect("manifest load should succeed")
            .expect("manifest should exist");
        assert_eq!(loaded.schema_version, 1);
        assert_eq!(loaded.daemon_api_version, 1);
        assert_eq!(loaded.ipc.transport, "unix_socket");
    }
}
