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
                if let Err(category) = authorizer.retire_invalid_sessions().await {
                    tracing::error!(category, "client session retirement stopped");
                    break;
                }
            }
        }))
    }

    async fn retire_invalid_sessions(&self) -> Result<(), &'static str> {
        let bindings = self
            .sessions
            .lock()
            .map_err(|_| "session_registry_poisoned")?
            .iter()
            .map(|(id, client)| (id.clone(), client.clone()))
            .collect::<Vec<_>>();
        if bindings.is_empty() {
            return Ok(());
        }
        let mut live_bindings = Vec::with_capacity(bindings.len());
        for (id, client) in bindings {
            if !self
                .manager
                .has_session(&Arc::from(id.clone()))
                .await
                .map_err(|_| "session_registry_inspection_failed")?
            {
                self.sessions
                    .lock()
                    .map_err(|_| "session_registry_poisoned")?
                    .remove(&id);
            } else {
                live_bindings.push((id, client));
            }
        }
        if live_bindings.is_empty() {
            return Ok(());
        }
        let invalid = tokio::task::spawn_blocking(move || {
            let mut invalid = Vec::new();
            for (client, ids) in group_bindings(live_bindings) {
                match client.try_permit() {
                    Ok(_) | Err(AccessError::Busy) => (),
                    Err(_) => invalid.extend(ids),
                }
            }
            invalid
        })
        .await
        .map_err(|error| {
            if error.is_panic() {
                "session_retirement_worker_panicked"
            } else {
                "session_retirement_worker_cancelled"
            }
        })?;
        for id in invalid {
            self.sessions
                .lock()
                .map_err(|_| "session_registry_poisoned")?
                .remove(&id);
            self.close_session(&id).await;
        }
        Ok(())
    }

    async fn close_session(&self, id: &str) {
        if let Err(error) = self.manager.close_session(&Arc::from(id)).await {
            use rmcp::transport::streamable_http_server::session::local::LocalSessionManagerError;
            let category = match error {
                LocalSessionManagerError::SessionNotFound(_) => "session_not_found",
                LocalSessionManagerError::SessionError(_) => "session_worker_failed",
                LocalSessionManagerError::InvalidEventId(_) => "session_event_invalid",
                _ => "session_close_failed",
            };
            tracing::warn!(category, "MCP session cleanup failed");
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
        bound.current_async().await.is_ok() && bound.same_authorization(client)
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

fn group_bindings(
    bindings: Vec<(String, AuthorizedClient)>,
) -> Vec<(AuthorizedClient, Vec<String>)> {
    let mut groups: Vec<(AuthorizedClient, Vec<String>)> = Vec::new();
    for (id, client) in bindings {
        if let Some((_, ids)) = groups
            .iter_mut()
            .find(|(bound, _)| bound.same_authorization(&client))
        {
            ids.push(id);
        } else {
            groups.push((client, vec![id]));
        }
    }
    groups
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
                authorizer.close_session(&id).await;
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
        let deadline = tokio::time::Instant::now() + duration;
        let wake = self
            .wake
            .get_or_insert_with(|| Box::pin(tokio::time::sleep_until(deadline)));
        wake.as_mut().reset(deadline);
        if std::future::Future::poll(wake.as_mut(), cx).is_ready() {
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
        authorizer.retire_invalid_sessions().await.unwrap();
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
        authorizer.retire_invalid_sessions().await.unwrap();
        assert!(authorizer.sessions.lock().unwrap().is_empty());
        assert!(authorizer.bind_session("fresh", &client, false).await);
    }

    #[tokio::test]
    async fn session_retirement_keeps_replacement_digests_separate() {
        use sha2::{Digest, Sha256};

        let (_temp, store, original) = credential();
        let manager = Arc::new(LocalSessionManager::default());
        let authorizer = Authorizer::new(store.clone(), manager.clone());
        let (old_id, _old_transport) = manager.create_session().await.unwrap();
        assert!(authorizer.bind_session(&old_id, &original, false).await);
        let token = format!("omcp1_{}_{}", original.credential_id(), "a".repeat(64));
        let mut hash = Sha256::new();
        hash.update(b"obsidian-mcp/access-token/v1\0");
        hash.update(token.as_bytes());
        let digest = hash
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        store
            .update(None, |policy| {
                let credentials = &mut policy.clients.get_mut("reader").unwrap().credentials;
                credentials[0].digest = digest;
                Ok(())
            })
            .unwrap();
        let current = store.authenticate(&token).unwrap();
        assert_eq!(original.credential_id(), current.credential_id());
        let groups = group_bindings(vec![
            ("old-a".into(), original.clone()),
            ("new-a".into(), current.clone()),
            ("old-b".into(), original.clone()),
            ("new-b".into(), current.clone()),
        ]);
        assert_eq!(groups.len(), 2);
        assert!(groups.iter().all(|(_, ids)| ids.len() == 2));
        let (new_id, _new_transport) = manager.create_session().await.unwrap();
        assert!(authorizer.bind_session(&new_id, &current, false).await);
        authorizer.retire_invalid_sessions().await.unwrap();
        assert!(!manager.has_session(&old_id).await.unwrap());
        assert!(manager.has_session(&new_id).await.unwrap());
        let sessions = authorizer.sessions.lock().unwrap();
        assert_eq!(sessions.len(), 1);
        assert!(sessions.contains_key(new_id.as_ref()));
    }

    #[derive(Default)]
    struct CountWake(std::sync::atomic::AtomicUsize);

    impl std::task::Wake for CountWake {
        fn wake(self: Arc<Self>) {
            self.wake_by_ref();
        }

        fn wake_by_ref(self: &Arc<Self>) {
            self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn idle_recheck_does_not_immediately_schedule_a_second_validation() {
        let (_temp, _store, client) = credential();
        let mut body = AuthorizedBody::new(
            Body::from_stream(futures_util::stream::pending::<Result<Bytes, io::Error>>()),
            client,
        );
        let counter = Arc::new(CountWake::default());
        let waker = std::task::Waker::from(Arc::clone(&counter));
        let mut context = Context::from_waker(&waker);
        assert!(Pin::new(&mut body).poll_frame(&mut context).is_pending());
        let deadline = body.wake.as_ref().unwrap().deadline();
        tokio::time::sleep_until(deadline + Duration::from_millis(20)).await;
        counter.0.store(0, std::sync::atomic::Ordering::SeqCst);
        assert!(Pin::new(&mut body).poll_frame(&mut context).is_pending());
        assert_eq!(counter.0.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn ready_frame_does_not_wait_for_an_old_idle_timer_after_publication() {
        let (_temp, store, client) = credential();
        let (send, receive) = tokio::sync::oneshot::channel();
        let stream =
            futures_util::stream::once(async move { Ok::<_, io::Error>(receive.await.unwrap()) });
        let mut body = AuthorizedBody::new(Body::from_stream(stream), client);
        let mut context = Context::from_waker(std::task::Waker::noop());
        assert!(Pin::new(&mut body).poll_frame(&mut context).is_pending());
        let writer = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(store.path().with_file_name("access.json.lock"))
            .unwrap();
        fs2::FileExt::lock_exclusive(&writer).unwrap();
        send.send(Bytes::from_static(b"current frame")).unwrap();
        assert!(Pin::new(&mut body).poll_frame(&mut context).is_pending());
        // Observe a separate task so the watchdog cannot poll the body into
        // making progress when its own timer has not woken it.
        let mut publication = tokio::spawn(async move { frame(&mut body).await });
        let (published, ()) = tokio::join!(
            tokio::time::timeout(Duration::from_millis(180), &mut publication),
            async move {
                tokio::time::sleep(Duration::from_millis(20)).await;
                drop(writer);
            }
        );
        if published.is_err() {
            publication.abort();
            let _ = publication.await;
        }
        assert_eq!(
            published
                .expect("publication retry was delayed by the idle recheck")
                .unwrap()
                .unwrap()
                .unwrap()
                .into_data()
                .unwrap(),
            "current frame"
        );
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
