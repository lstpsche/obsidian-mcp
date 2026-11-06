//! Optional private-memory authentication for the fixed HTTP reader.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use obsidian_mcp::access::ReadScope;
use sha2::{Digest, Sha256};

const URL_ENV: &str = "OBSIDIAN_MEMORY_AUTHORIZER_URL";
const KEY_ENV: &str = "OBSIDIAN_MEMORY_KEY_FILE";

tokio::task_local! {
    pub static REQUEST_READ_SCOPE: ReadScope;
}

#[derive(Clone)]
pub struct Authorizer {
    url: reqwest::Url,
    key: Arc<str>,
    client: reqwest::Client,
    sessions: Arc<Mutex<HashMap<String, SessionBinding>>>,
}

struct SessionBinding {
    token: [u8; 32],
    policy: [u8; 32],
    expires_at: f64,
}

impl Authorizer {
    pub fn from_env() -> Result<Option<Self>, Box<dyn std::error::Error>> {
        let url = std::env::var_os(URL_ENV)
            .map(std::ffi::OsString::into_string)
            .transpose()
            .map_err(|_| "private memory authorizer URL must be UTF-8")?;
        let key_file = std::env::var_os(KEY_ENV)
            .map(std::ffi::OsString::into_string)
            .transpose()
            .map_err(|_| "private memory key path must be UTF-8")?;
        let (url, key_file) = match (url, key_file) {
            (None, None) => return Ok(None),
            (Some(url), Some(key)) => (url, key),
            _ => return Err("private memory authorization needs both URL and key file".into()),
        };
        let url = reqwest::Url::parse(&url)?;
        if url.scheme() != "http"
            || !matches!(url.host_str(), Some("127.0.0.1" | "[::1]"))
            || !url.username().is_empty()
            || url.password().is_some()
            || url.path() != "/vault/memory/authorize"
            || url.query().is_some()
            || url.fragment().is_some()
        {
            return Err("private memory authorizer must be the fixed loopback endpoint".into());
        }
        let path = Path::new(&key_file);
        let info = std::fs::symlink_metadata(path)?;
        if !path.is_absolute() || !info.is_file() || info.len() > 4096 {
            return Err("private memory key must be a regular absolute file".into());
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if info.permissions().mode() & 0o037 != 0 {
                return Err("private memory key permissions are unsafe".into());
            }
        }
        let key = std::fs::read_to_string(path)?;
        let key = key.trim();
        if key.is_empty() || key.len() > 4096 || key.bytes().any(|b| b.is_ascii_control()) {
            return Err("private memory key is invalid".into());
        }
        let client = reqwest::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(Duration::from_millis(300))
            .timeout(Duration::from_secs(2))
            .build()?;
        Ok(Some(Self {
            url,
            key: Arc::from(key),
            client,
            sessions: Default::default(),
        }))
    }

    async fn authorize(&self, token: &str) -> Result<ReadScope, StatusCode> {
        let body = serde_json::to_vec(&serde_json::json!({"version":1,"token":token}))
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        let mut response = self
            .client
            .post(self.url.clone())
            .header("X-Vault-Key", self.key.as_ref())
            .header("Content-Type", "application/json")
            .body(body)
            .send()
            .await
            .map_err(|_| StatusCode::SERVICE_UNAVAILABLE)?;
        if matches!(response.status().as_u16(), 401 | 403 | 404) {
            return Err(StatusCode::UNAUTHORIZED);
        }
        if !response.status().is_success() {
            return Err(StatusCode::SERVICE_UNAVAILABLE);
        }
        if response.content_length().is_some_and(|size| size > 65536) {
            return Err(StatusCode::BAD_GATEWAY);
        }
        let mut bytes = Vec::new();
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|_| StatusCode::BAD_GATEWAY)?
        {
            if bytes.len() + chunk.len() > 65536 {
                return Err(StatusCode::BAD_GATEWAY);
            }
            bytes.extend_from_slice(&chunk);
        }
        let scope: ReadScope =
            serde_json::from_slice(&bytes).map_err(|_| StatusCode::BAD_GATEWAY)?;
        scope.validate().map_err(|_| StatusCode::UNAUTHORIZED)?;
        Ok(scope)
    }
}

fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |value| value.as_secs_f64())
}
fn digest(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

pub async fn middleware(
    State(authorizer): State<Authorizer>,
    request: Request,
    next: Next,
) -> Response {
    let token = request
        .headers()
        .get("Authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .filter(|token| {
            !token.is_empty()
                && token.len() <= 8192
                && !token
                    .bytes()
                    .any(|b| b.is_ascii_whitespace() || b.is_ascii_control())
        });
    let Some(token) = token else {
        return (
            StatusCode::UNAUTHORIZED,
            "private memory requires a read grant",
        )
            .into_response();
    };
    let scope = match authorizer.authorize(token).await {
        Ok(scope) => scope,
        Err(status) => return (status, "private memory authorization failed").into_response(),
    };
    let token = digest(token.as_bytes());
    let policy = match serde_json::to_vec(&scope) {
        Ok(bytes) => digest(&bytes),
        Err(_) => return StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    };
    let incoming = request
        .headers()
        .get("Mcp-Session-Id")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    {
        let Ok(mut sessions) = authorizer.sessions.lock() else {
            return StatusCode::SERVICE_UNAVAILABLE.into_response();
        };
        sessions.retain(|_, binding| binding.expires_at > now());
        if let Some(id) = &incoming {
            if !sessions
                .get(id)
                .is_some_and(|binding| binding.token == token && binding.policy == policy)
            {
                return (StatusCode::NOT_FOUND, "unknown private memory session").into_response();
            }
        } else if sessions.len() >= 1024 {
            return StatusCode::SERVICE_UNAVAILABLE.into_response();
        }
    }
    let deleting = request.method() == axum::http::Method::DELETE;
    let expires_at = scope.expires_at;
    let response = REQUEST_READ_SCOPE.scope(scope, next.run(request)).await;
    if response.status().is_success() {
        let id = response
            .headers()
            .get("Mcp-Session-Id")
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned)
            .or(incoming);
        if let Some(id) = id {
            let Ok(mut sessions) = authorizer.sessions.lock() else {
                return StatusCode::SERVICE_UNAVAILABLE.into_response();
            };
            if deleting {
                sessions.remove(&id);
            } else {
                sessions.insert(
                    id,
                    SessionBinding {
                        token,
                        policy,
                        expires_at,
                    },
                );
            }
        }
    }
    response
}
