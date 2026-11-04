//! JSON-RPC protocol DTOs for the semantic daemon (v2).

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

pub const JSONRPC_VERSION: &str = "2.0";
pub const DAEMON_API_VERSION: u32 = 2;
pub const SEMANTIC_SOURCE_EVIDENCE_CAPABILITY: &str = "semantic_source_evidence_v1";

pub const ERR_PARSE: i64 = -32700;
pub const ERR_INVALID_REQUEST: i64 = -32600;
pub const ERR_METHOD_NOT_FOUND: i64 = -32601;
pub const ERR_INVALID_PARAMS: i64 = -32602;
pub const ERR_INTERNAL: i64 = -32603;
pub const ERR_INCOMPATIBLE_API_VERSION: i64 = -32010;
pub const ERR_UNSUPPORTED_CAPABILITY: i64 = -32011;
pub const ERR_DAEMON_UNAVAILABLE: i64 = -32020;
pub const ERR_VAULT_NOT_READY: i64 = -32030;
pub const ERR_VAULT_NOT_ATTACHED: i64 = -32031;
pub const ERR_BOOTSTRAP_REQUIRED: i64 = -32040;

fn default_params() -> Value {
    Value::Object(Map::new())
}

#[derive(Debug, Deserialize)]
pub struct RpcRequest {
    pub jsonrpc: String,
    #[serde(default)]
    pub id: Option<Value>,
    pub method: String,
    #[serde(default = "default_params")]
    pub params: Value,
}

#[derive(Debug, Serialize)]
pub struct RpcResponse {
    pub jsonrpc: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<RpcError>,
}

impl RpcResponse {
    pub fn success(id: Option<Value>, result: Value) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION,
            id,
            result: Some(result),
            error: None,
        }
    }

    pub fn error(id: Option<Value>, code: i64, message: impl Into<String>) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION,
            id,
            result: None,
            error: Some(RpcError {
                code,
                message: message.into(),
                data: None,
            }),
        }
    }

    pub fn error_with_data(
        id: Option<Value>,
        code: i64,
        message: impl Into<String>,
        data: Value,
    ) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION,
            id,
            result: None,
            error: Some(RpcError {
                code,
                message: message.into(),
                data: Some(data),
            }),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct RpcError {
    pub code: i64,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Default)]
pub struct HealthParams {
    #[serde(default)]
    pub client_name: Option<String>,
    #[serde(default)]
    pub client_version: Option<String>,
    #[serde(default)]
    pub min_api_version: Option<u32>,
    #[serde(default)]
    pub max_api_version: Option<u32>,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
pub struct HealthResult {
    pub daemon_version: String,
    pub daemon_api_version: u32,
    #[serde(default)]
    pub pid: u32,
    pub status: String,
    pub uptime_ms: u64,
    pub model_name: String,
    pub semantic_home: String,
    /// Additive features supported independently of the transport API version.
    #[serde(default)]
    pub capabilities: Vec<String>,
}

impl HealthResult {
    pub fn supports_source_evidence(&self) -> bool {
        self.capabilities
            .iter()
            .any(|capability| capability == SEMANTIC_SOURCE_EVIDENCE_CAPABILITY)
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Default)]
#[serde(deny_unknown_fields)]
pub struct ShutdownParams {}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
pub struct ShutdownResult {
    pub accepted: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
pub struct EnsureVaultParams {
    pub vault_root: String,
    #[serde(default)]
    pub watch: Option<bool>,
    #[serde(default)]
    pub model_name: Option<String>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SemanticPhase {
    Warming,
    Ready,
    Degraded,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
pub struct SemanticStatus {
    pub phase: SemanticPhase,
    pub ready: bool,
    pub indexed_notes: usize,
    pub total_notes: usize,
    pub pending_notes: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct VaultStatusParams {
    pub vault_root: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
pub struct EnsureVaultResult {
    pub vault_id: String,
    pub ready: bool,
    pub watch_enabled: bool,
    pub model_name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase: Option<SemanticPhase>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub indexed_notes: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_notes: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pending_notes: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
pub struct SearchSemanticParams {
    pub vault_root: String,
    pub query: String,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub include_content: Option<bool>,
    /// Restrict ranking to these vault-relative paths; absent means the entire index.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allowed_paths: Option<Vec<String>>,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
pub struct SearchHybridParams {
    pub vault_root: String,
    pub query: String,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub prefetch: Option<usize>,
    #[serde(default)]
    pub alpha: Option<f32>,
    #[serde(default)]
    pub include_content: Option<bool>,
    /// Restrict ranking to these vault-relative paths; absent means the entire index.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allowed_paths: Option<Vec<String>>,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
pub struct OpenHintParams {
    pub vault_root: String,
    pub path: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
pub struct OpenHintResult {
    pub path: String,
    pub exists: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subpath: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
pub struct SearchResult {
    pub results: Vec<SemanticHit>,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
pub struct SemanticHit {
    pub path: String,
    pub title: String,
    pub score: f32,
    pub tags: Vec<String>,
    pub source_revision: String,
    pub source_bytes: usize,
    pub source_status: crate::models::SourceStatus,
    pub evidence: crate::models::SemanticEvidence,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snippet: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subpath: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preview: Option<crate::models::SemanticPreview>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_error: Option<crate::models::SemanticContextError>,
}

impl From<crate::models::SemanticNoteResult> for SemanticHit {
    fn from(hit: crate::models::SemanticNoteResult) -> Self {
        Self {
            path: hit.path.to_string_lossy().into_owned(),
            title: hit.title,
            score: hit.score,
            tags: hit.tags,
            source_revision: hit.source_revision,
            source_bytes: hit.source_bytes,
            source_status: hit.source_status,
            evidence: hit.evidence,
            snippet: hit.snippet,
            content: hit.content,
            subpath: None,
            preview: hit.preview,
            context_error: hit.context_error,
        }
    }
}

impl From<SemanticHit> for crate::models::SemanticNoteResult {
    fn from(hit: SemanticHit) -> Self {
        Self {
            path: hit.path.into(),
            title: hit.title,
            score: hit.score,
            tags: hit.tags,
            source_revision: hit.source_revision,
            source_bytes: hit.source_bytes,
            source_status: hit.source_status,
            evidence: hit.evidence,
            snippet: hit.snippet,
            content: hit.content,
            preview: hit.preview,
            context_error: hit.context_error,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_defaults_params_to_object() {
        let req: RpcRequest = serde_json::from_str(r#"{"jsonrpc":"2.0","id":1,"method":"health"}"#)
            .expect("request should deserialize");
        assert!(req.params.is_object());
    }

    #[test]
    fn response_error_has_expected_shape() {
        let response = RpcResponse::error(Some(Value::from(1)), ERR_INVALID_PARAMS, "bad params");
        let value = serde_json::to_value(response).expect("response should serialize");
        assert_eq!(value["error"]["code"], Value::from(ERR_INVALID_PARAMS));
    }

    #[test]
    fn old_v2_health_remains_decodable_without_advertising_new_semantics() {
        let health: HealthResult = serde_json::from_value(serde_json::json!({
            "daemon_version": "3.2.1",
            "daemon_api_version": 2,
            "status": "ok",
            "uptime_ms": 1,
            "model_name": "old-model",
            "semantic_home": "/tmp/semantic"
        }))
        .unwrap();

        assert_eq!(health.daemon_api_version, DAEMON_API_VERSION);
        assert!(health.capabilities.is_empty());
        assert!(!health.supports_source_evidence());
    }

    #[test]
    fn semantic_hit_without_source_evidence_is_not_promoted() {
        let legacy = serde_json::json!({
            "path": "note.md",
            "title": "note",
            "score": 0.8,
            "tags": [],
            "snippet": "old preview"
        });
        assert!(serde_json::from_value::<SemanticHit>(legacy).is_err());
    }

    #[test]
    fn ensure_vault_decodes_legacy_v1_without_progress_fields() {
        let result: EnsureVaultResult = serde_json::from_value(serde_json::json!({
            "vault_id": "legacy",
            "ready": true,
            "watch_enabled": true,
            "model_name": "legacy-model"
        }))
        .unwrap();

        assert!(result.ready);
        assert_eq!(result.phase, None);
        assert_eq!(result.indexed_notes, None);
        assert_eq!(result.total_notes, None);
        assert_eq!(result.pending_notes, None);
        assert_eq!(result.last_error, None);
    }

    #[test]
    fn ensure_vault_round_trips_additive_progress_fields() {
        let expected = EnsureVaultResult {
            vault_id: "current".into(),
            ready: false,
            watch_enabled: true,
            model_name: "current-model".into(),
            phase: Some(SemanticPhase::Warming),
            indexed_notes: Some(12),
            total_notes: Some(20),
            pending_notes: Some(8),
            last_error: None,
        };

        let encoded = serde_json::to_value(&expected).unwrap();
        let decoded: EnsureVaultResult = serde_json::from_value(encoded).unwrap();
        assert!(!decoded.ready);
        assert_eq!(decoded.phase, Some(SemanticPhase::Warming));
        assert_eq!(decoded.indexed_notes, Some(12));
        assert_eq!(decoded.total_notes, Some(20));
        assert_eq!(decoded.pending_notes, Some(8));
    }
}
