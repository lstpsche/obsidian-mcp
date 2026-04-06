//! JSON-RPC protocol DTOs for the semantic daemon (v1).

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

pub const JSONRPC_VERSION: &str = "2.0";
pub const DAEMON_API_VERSION: u32 = 1;

pub const ERR_PARSE: i64 = -32700;
pub const ERR_INVALID_REQUEST: i64 = -32600;
pub const ERR_METHOD_NOT_FOUND: i64 = -32601;
pub const ERR_INVALID_PARAMS: i64 = -32602;
pub const ERR_INTERNAL: i64 = -32603;
pub const ERR_INCOMPATIBLE_API_VERSION: i64 = -32010;
pub const ERR_DAEMON_UNAVAILABLE: i64 = -32020;
pub const ERR_VAULT_NOT_READY: i64 = -32030;
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
    pub status: String,
    pub uptime_ms: u64,
    pub model_name: String,
    pub semantic_home: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
pub struct EnsureVaultParams {
    pub vault_root: String,
    #[serde(default)]
    pub watch: Option<bool>,
    #[serde(default)]
    pub model_name: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
pub struct EnsureVaultResult {
    pub vault_id: String,
    pub ready: bool,
    pub watch_enabled: bool,
    pub model_name: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
pub struct SearchSemanticParams {
    pub vault_root: String,
    pub query: String,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub include_content: Option<bool>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snippet: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subpath: Option<String>,
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
}
