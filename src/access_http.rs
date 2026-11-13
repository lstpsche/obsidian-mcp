//! Per-message authentication and revocable outbound HTTP publication.

use std::collections::HashMap;
use std::io;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Duration;

use axum::body::{Body, Bytes, HttpBody};
use axum::extract::{Request, State};
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use http_body::Frame;
use obsidian_mcp::access::store::{AccessError, AccessStore, AuthorizedClient};
use rmcp::transport::streamable_http_server::session::{
    SessionManager, local::LocalSessionManager,
};

const MAX_SESSIONS: usize = 1024;
const RELEASE_RECHECK: Duration = Duration::from_millis(250);
const PUBLICATION_WAIT: Duration = Duration::from_millis(10);

#[derive(Clone)]
pub struct Authorizer {
    store: AccessStore,
    manager: Arc<LocalSessionManager>,
    sessions: Arc<Mutex<HashMap<String, AuthorizedClient>>>,
}

pub struct Reaper(tokio::task::JoinHandle<()>);

impl Drop for Reaper {
    fn drop(&mut self) {
        self.0.abort();
    }
}

impl Authorizer {
    pub fn new(store: AccessStore, manager: Arc<LocalSessionManager>) -> Self {
        Self {
            store,
            manager,
            sessions: Default::default(),
        }
    }

    pub fn start_reaper(&self) -> Reaper {
        let authorizer = self.clone();
        Reaper(tokio::spawn(async move {
            let mut interval = tokio::time::interval(RELEASE_RECHECK);
            loop {
                interval.tick().await;
                authorizer.retire_invalid_sessions().await;
            }
        }))
    }

    async fn retire_invalid_sessions(&self) {
        let bindings = match self.sessions.lock() {
            Ok(sessions) => sessions
                .iter()
                .map(|(id, client)| (id.clone(), client.clone()))
                .collect::<Vec<_>>(),
            Err(_) => return,
        };
        if bindings.is_empty() {
            return;
        }
        let mut live_bindings = Vec::with_capacity(bindings.len());
        for (id, client) in bindings {
            if matches!(
                self.manager.has_session(&Arc::from(id.clone())).await,
                Ok(false)
            ) {
                if let Ok(mut sessions) = self.sessions.lock() {
                    sessions.remove(&id);
                }
            } else {
                live_bindings.push((id, client));
            }
        }
        if live_bindings.is_empty() {
            return;
        }
        let invalid = tokio::task::spawn_blocking(move || {
            live_bindings
                .into_iter()
                .filter_map(|(id, client)| match client.try_permit() {
                    Ok(_) | Err(AccessError::Busy) => None,
                    Err(_) => Some(id),
                })
                .collect::<Vec<_>>()
        })
        .await;
        let Ok(invalid) = invalid else { return };
        for id in invalid {
            if let Ok(mut sessions) = self.sessions.lock() {
                sessions.remove(&id);
            }
            let _ = self.manager.close_session(&Arc::from(id)).await;
        }
    }

    async fn session_matches(&self, id: &str, client: &AuthorizedClient) -> bool {
        let bound = self
            .sessions
            .lock()
            .ok()
            .and_then(|sessions| sessions.get(id).cloned());
        let Some(bound) = bound else { return false };
        // An owner may replace a digest without changing its safe ID. The
        // retained credential must still be valid; the ID alone is no grant.
        bound.current_async().await.is_ok()
            && bound.client_id() == client.client_id()
            && bound.credential_id() == client.credential_id()
            && bound.policy_hash() == client.policy_hash()
    }

    async fn bind_session(&self, id: &str, client: &AuthorizedClient, deleting: bool) -> bool {
        let Ok(_permit) = client.permit_async().await else {
            return false;
        };
        let Ok(mut sessions) = self.sessions.lock() else {
            return false;
        };
        if deleting {
            sessions.remove(id);
            true
        } else if sessions.contains_key(id) || sessions.len() < MAX_SESSIONS {
            sessions.insert(id.to_owned(), client.clone());
            true
        } else {
            false
        }
    }
}

pub async fn middleware(
    State(authorizer): State<Authorizer>,
    mut request: Request,
    next: Next,
) -> Response {
    let Some(token) = bearer(request.headers()) else {
        return rejection(StatusCode::UNAUTHORIZED);
    };
    let store = authorizer.store.clone();
    // The credential stays out of tracing/debug/argv, including the blocking
    // work used to keep protected-file I/O off the request executor.
    let token = zeroize::Zeroizing::new(token.to_owned());
    let client = match tokio::task::spawn_blocking(move || store.authenticate(&token)).await {
        Ok(Ok(client)) => client,
        Ok(Err(AccessError::InvalidCredential)) => return rejection(StatusCode::UNAUTHORIZED),
        _ => return rejection(StatusCode::SERVICE_UNAVAILABLE),
    };
    let incoming = match single_session_id(request.headers()) {
        Ok(id) => id,
        Err(()) => return rejection(StatusCode::BAD_REQUEST),
    };
    if let Some(id) = &incoming
        && !authorizer.session_matches(id, &client).await
    {
        return rejection(StatusCode::NOT_FOUND);
    }
    let deleting = request.method() == Method::DELETE;
    request.extensions_mut().insert(client.clone());
    let mut response = next.run(request).await;
    if response.status().is_success() {
        let returned = response
            .headers()
            .get("Mcp-Session-Id")
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned);
        let id = returned.or(incoming);
        if let Some(id) = id {
            let accepted = authorizer.bind_session(&id, &client, deleting).await;
            if !accepted {
                let _ = authorizer.manager.close_session(&Arc::from(id)).await;
                return rejection(StatusCode::SERVICE_UNAVAILABLE);
            }
        }
    }
    response
        .headers_mut()
        .insert("Cache-Control", HeaderValue::from_static("no-store"));
    let (parts, body) = response.into_parts();
    Response::from_parts(parts, Body::new(AuthorizedBody::new(body, client)))
}

fn bearer(headers: &HeaderMap) -> Option<&str> {
    let mut values = headers.get_all("Authorization").iter();
    let value = values.next()?.to_str().ok()?;
    if values.next().is_some() {
        return None;
    }
    let (scheme, token) = value.split_once(' ')?;
    if !scheme.eq_ignore_ascii_case("bearer")
        || token.is_empty()
        || token.len() > 256
        || token
            .bytes()
            .any(|byte| byte.is_ascii_whitespace() || byte.is_ascii_control() || byte == b',')
    {
        return None;
    }
    Some(token)
}

fn single_session_id(headers: &HeaderMap) -> Result<Option<String>, ()> {
    let mut values = headers.get_all("Mcp-Session-Id").iter();
    let Some(value) = values.next() else {
        return Ok(None);
    };
    if values.next().is_some() {
        return Err(());
    }
    let value = value.to_str().map_err(|_| ())?;
    if value.is_empty()
        || value.len() > 256
        || value
            .bytes()
            .any(|byte| !byte.is_ascii_alphanumeric() && !matches!(byte, b'-' | b'_'))
    {
        return Err(());
    }
    Ok(Some(value.to_owned()))
}

fn rejection(status: StatusCode) -> Response {
    let message = match status {
        StatusCode::UNAUTHORIZED => "A valid client bearer credential is required",
        StatusCode::NOT_FOUND => "Unknown client session; reconnect the MCP client",
        StatusCode::BAD_REQUEST => "Invalid authentication request",
        _ => "Authorization is unavailable; contact the vault owner",
    };
    let mut response = (status, message).into_response();
    response
        .headers_mut()
        .insert("Cache-Control", HeaderValue::from_static("no-store"));
    if status == StatusCode::UNAUTHORIZED {
        response.headers_mut().insert(
            "WWW-Authenticate",
            HeaderValue::from_static("Bearer realm=\"obsidian-mcp\""),
        );
    }
    response
}

struct AuthorizedBody {
    inner: Body,
    client: AuthorizedClient,
    pending: Option<Frame<Bytes>>,
    wake: Option<Pin<Box<tokio::time::Sleep>>>,
    finished: bool,
}

impl AuthorizedBody {
    fn new(inner: Body, client: AuthorizedClient) -> Self {
        Self {
            inner,
            client,
            pending: None,
            wake: None,
            finished: false,
        }
    }

    fn wake_after(&mut self, cx: &mut Context<'_>, duration: Duration) {
        if self.wake.is_none() {
            self.wake = Some(Box::pin(tokio::time::sleep(duration)));
        }
        if let Some(wake) = &mut self.wake
            && std::future::Future::poll(wake.as_mut(), cx).is_ready()
        {
            self.wake = None;
            cx.waker().wake_by_ref();
        }
    }
}

impl HttpBody for AuthorizedBody {
    type Data = Bytes;
    type Error = io::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, Self::Error>>> {
        if self.finished {
            return Poll::Ready(None);
        }
        if self.pending.is_none() {
            match Pin::new(&mut self.inner).poll_frame(cx) {
                Poll::Ready(Some(Ok(frame))) => self.pending = Some(frame),
                Poll::Ready(Some(Err(_))) => {
                    self.finished = true;
                    return Poll::Ready(Some(Err(io::Error::other(
                        "MCP response transport failed",
                    ))));
                }
                Poll::Ready(None) => {
                    self.finished = true;
                    return Poll::Ready(None);
                }
                Poll::Pending => (),
            }
        }
        // Independently open/read the protected store under its shared lock at
        // the actual release point. A writer cannot publish revocation between
        // this observation and this synchronous frame handoff. No permit lives
        // across Pending, a provider wait, or network backpressure.
        let permit = match self.client.try_permit() {
            Ok(permit) => permit,
            Err(AccessError::Busy) => {
                self.wake_after(cx, PUBLICATION_WAIT);
                return Poll::Pending;
            }
            Err(_) => {
                self.finished = true;
                self.pending = None;
                return Poll::Ready(Some(Err(io::Error::other(
                    "MCP authorization expired or changed",
                ))));
            }
        };
        if let Some(frame) = self.pending.take() {
            drop(permit);
            return Poll::Ready(Some(Ok(frame)));
        }
        drop(permit);
        self.wake_after(cx, RELEASE_RECHECK);
        Poll::Pending
    }

    fn is_end_stream(&self) -> bool {
        self.finished
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use obsidian_mcp::access::clients::{ClientPolicy, Grant, Role};

    fn credential() -> (tempfile::TempDir, AccessStore, AuthorizedClient) {
        let temp = tempfile::tempdir().unwrap();
        let vault = temp.path().join("vault");
        std::fs::create_dir(&vault).unwrap();
        let store = AccessStore::initialize(
            &temp.path().join("private/access.json"),
            &vault,
            &temp.path().join("data"),
        )
        .unwrap();
        store
            .update(None, |policy| {
                policy.clients.insert(
                    "reader".into(),
                    ClientPolicy {
                        grants: vec![Grant {
                            id: "shared".into(),
                            directory: Some("".into()),
                            file: None,
                            role: Role::Reader,
                        }],
                        ..ClientPolicy::default()
                    },
                );
                Ok(())
            })
            .unwrap();
        let token_file = temp.path().join("tokens/reader.token");
        store.issue_token("reader", &token_file, None).unwrap();
        let token = obsidian_mcp::access::store::read_token_file(&token_file).unwrap();
        let client = store.authenticate(token.expose()).unwrap();
        (temp, store, client)
    }

    async fn frame(body: &mut AuthorizedBody) -> Option<Result<Frame<Bytes>, io::Error>> {
        std::future::poll_fn(|cx| Pin::new(&mut *body).poll_frame(cx)).await
    }

    #[tokio::test(flavor = "current_thread")]
    async fn session_binding_waits_for_publication_without_losing_a_valid_session() {
        let (_temp, store, client) = credential();
        let authorizer = Authorizer::new(store.clone(), Arc::new(LocalSessionManager::default()));
        let lock = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(store.path().with_file_name("access.json.lock"))
            .unwrap();
        fs2::FileExt::lock_exclusive(&lock).unwrap();
        let (release, wait) = std::sync::mpsc::channel();
        let owner = std::thread::spawn(move || {
            let _ = wait.recv_timeout(Duration::from_secs(2));
            drop(lock);
        });
        let binding = tokio::spawn({
            let authorizer = authorizer.clone();
            async move { authorizer.bind_session("retained", &client, false).await }
        });
        tokio::time::sleep(Duration::from_millis(25)).await;
        assert!(!binding.is_finished());
        release.send(()).unwrap();
        assert!(
            tokio::time::timeout(Duration::from_secs(1), binding)
                .await
                .unwrap()
                .unwrap()
        );
        owner.join().unwrap();
        assert!(authorizer.sessions.lock().unwrap().contains_key("retained"));
    }

    #[tokio::test]
    async fn expired_sdk_sessions_release_authorization_binding_capacity() {
        let (_temp, store, client) = credential();
        let manager = Arc::new(LocalSessionManager::default());
        let authorizer = Authorizer::new(store, manager.clone());
        let (id, transport) = manager.create_session().await.unwrap();
        assert!(authorizer.bind_session(&id, &client, false).await);
        manager.close_session(&id).await.unwrap();
        drop(transport);
        authorizer.retire_invalid_sessions().await;
        assert!(authorizer.sessions.lock().unwrap().is_empty());

        // An absent SDK session must not consume the binding limit, even when
        // the credential remains valid for the lifetime of the HTTP service.
        for index in 0..MAX_SESSIONS {
            authorizer
                .sessions
                .lock()
                .unwrap()
                .insert(format!("expired-{index}"), client.clone());
        }
        authorizer.retire_invalid_sessions().await;
        assert!(authorizer.sessions.lock().unwrap().is_empty());
        assert!(authorizer.bind_session("fresh", &client, false).await);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn revocation_blocks_first_and_subsequent_buffered_frames() {
        for release_first in [false, true] {
            let (_temp, store, client) = credential();
            let chunks = futures_util::stream::iter([
                Ok::<_, io::Error>(Bytes::from_static(b"first")),
                Ok(Bytes::from_static(b"second")),
            ]);
            let mut body = AuthorizedBody::new(Body::from_stream(chunks), client);
            if release_first {
                assert_eq!(
                    frame(&mut body)
                        .await
                        .unwrap()
                        .unwrap()
                        .into_data()
                        .unwrap(),
                    "first"
                );
            }
            store
                .update(None, |policy| {
                    policy.clients.get_mut("reader").unwrap().enabled = false;
                    Ok(())
                })
                .unwrap();
            assert!(frame(&mut body).await.unwrap().is_err());
            assert!(frame(&mut body).await.is_none());
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn pending_response_does_not_hold_publication_lock_and_notices_revocation() {
        let (_temp, store, client) = credential();
        let mut body = AuthorizedBody::new(
            Body::from_stream(futures_util::stream::pending::<Result<Bytes, io::Error>>()),
            client,
        );
        assert!(
            tokio::time::timeout(Duration::from_millis(20), frame(&mut body))
                .await
                .is_err()
        );
        store
            .update(None, |policy| {
                policy.clients.get_mut("reader").unwrap().enabled = false;
                Ok(())
            })
            .unwrap();
        assert!(
            tokio::time::timeout(Duration::from_secs(1), frame(&mut body))
                .await
                .unwrap()
                .unwrap()
                .is_err()
        );
    }

    #[test]
    fn bearer_accepts_scheme_case_and_rejects_ambiguous_or_combined_credentials() {
        let mut headers = HeaderMap::new();
        headers.insert("Authorization", HeaderValue::from_static("bEaReR secret"));
        assert_eq!(bearer(&headers), Some("secret"));
        headers.append("Authorization", HeaderValue::from_static("Bearer second"));
        assert!(bearer(&headers).is_none());
        for value in [
            "Bearer",
            "Bearer  two",
            "Bearer first,Bearer second",
            "Basic secret",
        ] {
            headers.clear();
            headers.insert("Authorization", HeaderValue::from_str(value).unwrap());
            assert!(bearer(&headers).is_none(), "{value}");
        }
    }
}
