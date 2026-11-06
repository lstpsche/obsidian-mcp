//! Binary-safe SFTP requests over a retained, bounded SSH subsystem connection.

use std::borrow::Cow;
use std::collections::HashMap;
use std::future::Future;
use std::io;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use bytes::Bytes;
use openssh_sftp_client_lowlevel as wire;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::process::{Child, Command};
use tokio::sync::{Semaphore, mpsc, watch};
use tokio::task::JoinHandle;
use tokio::time::timeout_at;
pub(super) use wire::{FileAttrs, HandleOwned, NameEntry};

use crate::config::source::SftpLocation;
use crate::error::{MutationOutcome, StorageErrorKind, VaultError, VaultResult};

const MAX_PACKET: usize = 256 * 1024;
const MAX_REQUESTS: usize = 16;
pub(super) const CHUNK_SIZE: u32 = 32 * 1024;
const STDERR_LIMIT: usize = 16 * 1024;

type Id = wire::Id<Vec<u8>>;
type Writer = wire::WriteEnd<Vec<u8>, PacketQueue>;
type SharedResponse = Arc<Mutex<ResponseMetadata>>;

#[derive(Default)]
struct ResponseMetadata {
    directory: bool,
    status: Option<u32>,
    skipped_count: usize,
    skipped_bytes: usize,
}

#[derive(Default)]
struct Requests {
    building: Option<SharedResponse>,
    pending: HashMap<u32, SharedResponse>,
}

pub(super) struct DirectoryPage {
    pub entries: Box<[NameEntry]>,
    pub skipped_count: usize,
    pub skipped_bytes: usize,
}

/// Admission limits the number and size of packets before this infallible
/// protocol queue is reached. All file writes use bounded buffered chunks.
#[derive(Clone)]
struct PacketQueue {
    sender: mpsc::UnboundedSender<Bytes>,
    state: Arc<State>,
}

impl std::fmt::Debug for PacketQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PacketQueue").finish_non_exhaustive()
    }
}

impl wire::Queue for PacketQueue {
    fn push(&self, bytes: Bytes) {
        if bytes.len() >= 9 && bytes[4] != 1 {
            let id = u32::from_be_bytes([bytes[5], bytes[6], bytes[7], bytes[8]]);
            let mut requests = self
                .state
                .requests
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            if let Some(response) = requests.building.clone() {
                requests.pending.insert(id, response);
            }
        }
        let _ = self.sender.send(bytes);
    }

    fn extend(&self, header: Bytes, body: &[&[Bytes]]) {
        let size = header.len()
            + body
                .iter()
                .flat_map(|parts| parts.iter())
                .map(Bytes::len)
                .sum::<usize>();
        let mut packet = Vec::with_capacity(size);
        packet.extend_from_slice(&header);
        for part in body.iter().flat_map(|parts| parts.iter()) {
            packet.extend_from_slice(part);
        }
        self.push(packet.into());
    }
}

struct State {
    failure: watch::Sender<Option<(StorageErrorKind, String)>>,
    stderr: Mutex<Vec<u8>>,
    pumps: Mutex<Vec<tokio::task::AbortHandle>>,
    requests: Mutex<Requests>,
}

impl State {
    fn fail(&self, cause: impl Into<String>) {
        self.fail_with_kind(StorageErrorKind::Unavailable, cause);
    }

    fn fail_with_kind(&self, kind: StorageErrorKind, cause: impl Into<String>) {
        let cause = cause.into();
        let changed = self.failure.send_if_modified(|current| {
            if current.is_none() {
                *current = Some((kind, cause));
                true
            } else {
                false
            }
        });
        if changed {
            let mut requests = self
                .requests
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            requests.building = None;
            requests.pending.clear();
            drop(requests);
            for pump in self
                .pumps
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .iter()
            {
                pump.abort();
            }
        }
    }

    fn register_pump(&self, pump: tokio::task::AbortHandle) {
        let mut pumps = self.pumps.lock().unwrap_or_else(|error| error.into_inner());
        if self.failure.borrow().is_some() {
            pump.abort();
        } else {
            pumps.push(pump);
        }
    }

    fn message(&self) -> String {
        let cause = self
            .failure
            .borrow()
            .as_ref()
            .map(|(_, message)| message.clone())
            .unwrap_or_else(|| "SFTP connection is unavailable".into());
        let stderr = self
            .stderr
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if stderr.is_empty() {
            cause
        } else {
            let detail: String = String::from_utf8_lossy(&stderr)
                .chars()
                .filter(|c| !c.is_control() || *c == '\n')
                .collect();
            format!("{cause}: {}", detail.trim())
        }
    }
}

struct Inner {
    writer: tokio::sync::Mutex<Writer>,
    admission: Semaphore,
    timeout: Duration,
    extensions: wire::Extensions,
    state: Arc<State>,
    tasks: Tasks,
}

impl Drop for Inner {
    fn drop(&mut self) {
        self.state.fail("SFTP session closed");
        self.tasks.abort();
    }
}

struct ChildOwner(Option<Child>);

impl Drop for ChildOwner {
    fn drop(&mut self) {
        if let Some(mut child) = self.0.take() {
            let _ = child.start_kill();
            if let Ok(runtime) = tokio::runtime::Handle::try_current() {
                runtime.spawn(async move {
                    let _ = child.wait().await;
                });
            }
        }
    }
}

#[derive(Default)]
struct Tasks(Vec<JoinHandle<()>>);

impl Tasks {
    fn push(&mut self, task: JoinHandle<()>) {
        self.0.push(task);
    }
    fn push_pump(&mut self, task: JoinHandle<()>, state: &State) {
        state.register_pump(task.abort_handle());
        self.push(task);
    }
    fn abort(&self) {
        for task in &self.0 {
            task.abort();
        }
    }
}
impl Drop for Tasks {
    fn drop(&mut self) {
        self.abort();
    }
}

#[derive(Clone)]
pub(super) struct Session(Arc<Inner>);

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session")
            .field("healthy", &self.is_healthy())
            .finish()
    }
}

/// Retire the connection if an acknowledged remote handle is abandoned.
pub(super) struct HandleLifetime<'a> {
    session: &'a Session,
    handle: HandleOwned,
    closed: bool,
}

impl HandleLifetime<'_> {
    pub(super) fn handle(&self) -> &HandleOwned {
        &self.handle
    }

    pub(super) async fn close(mut self) -> VaultResult<()> {
        self.session.close(&self.handle).await?;
        self.closed = true;
        Ok(())
    }
}

impl Drop for HandleLifetime<'_> {
    fn drop(&mut self) {
        if !self.closed {
            self.session
                .0
                .state
                .fail("SFTP handle was abandoned before confirmed close");
        }
    }
}

/// A dropped caller must not leave a request ID or partial frame reusable.
struct RequestGuard<'a> {
    state: &'a State,
    response: SharedResponse,
    completed: bool,
}

impl Drop for RequestGuard<'_> {
    fn drop(&mut self) {
        self.state
            .requests
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .pending
            .retain(|_, response| !Arc::ptr_eq(response, &self.response));
        if !self.completed {
            self.state
                .fail("SFTP request was cancelled or exceeded its deadline");
        }
    }
}

impl Session {
    pub(super) async fn connect(location: &SftpLocation, timeout: Duration) -> VaultResult<Self> {
        let mut command = Command::new("ssh");
        command
            .env("LC_ALL", "C")
            .args(location.ssh_args())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        let mut child = command.spawn().map_err(|error| {
            VaultError::storage(
                "connect",
                &location.root,
                if error.kind() == io::ErrorKind::NotFound {
                    StorageErrorKind::Unsupported
                } else {
                    StorageErrorKind::Unavailable
                },
                format!("cannot launch OpenSSH client: {error}"),
            )
        })?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| VaultError::Other("SSH stdin was not piped".into()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| VaultError::Other("SSH stdout was not piped".into()))?;
        let stderr = child.stderr.take();
        Self::start(stdout, stdin, timeout, Some(child), stderr).await
    }

    #[cfg(test)]
    pub(super) async fn from_io<R, W>(reader: R, writer: W, timeout: Duration) -> VaultResult<Self>
    where
        R: AsyncRead + Unpin + Send + 'static,
        W: AsyncWrite + Unpin + Send + 'static,
    {
        Self::start(reader, writer, timeout, None, None::<tokio::io::Empty>).await
    }

    async fn start<R, W, E>(
        mut reader: R,
        mut output: W,
        timeout: Duration,
        child: Option<Child>,
        stderr: Option<E>,
    ) -> VaultResult<Self>
    where
        R: AsyncRead + Unpin + Send + 'static,
        W: AsyncWrite + Unpin + Send + 'static,
        E: AsyncRead + Unpin + Send + 'static,
    {
        let mut child = ChildOwner(child);
        let (failure, _) = watch::channel(None);
        let state = Arc::new(State {
            failure,
            stderr: Mutex::new(Vec::new()),
            pumps: Mutex::new(Vec::new()),
            requests: Mutex::new(Requests::default()),
        });
        let (sender, mut receiver) = mpsc::unbounded_channel::<Bytes>();
        let writer: Writer = wire::connect(
            PacketQueue {
                sender,
                state: state.clone(),
            },
            (),
        )
        .map_err(|error| map_error("initialize", "", error, false))?;
        let mut tasks = Tasks::default();
        let task_state = state.clone();
        tasks.push_pump(
            tokio::spawn(async move {
                let mut failed = task_state.failure.subscribe();
                loop {
                    tokio::select! {
                        _ = failed.changed() => break,
                        packet = receiver.recv() => match packet {
                            Some(packet) => {
                                if let Err(error) = output.write_all(&packet).await {
                                    task_state.fail(format!("SSH subsystem write failed: {error}"));
                                    break;
                                }
                                if let Err(error) = output.flush().await {
                                    task_state.fail(format!("SSH subsystem flush failed: {error}"));
                                    break;
                                }
                            }
                            None => break,
                        }
                    }
                }
            }),
            &state,
        );
        // Whole bounded frames are admitted before the library sees lengths.
        // This protects nested vector allocations as well as packet arithmetic.
        let (guarded, mut admitted) = tokio::io::duplex(MAX_PACKET + 4);
        let task_state = state.clone();
        tasks.push_pump(
            tokio::spawn(async move {
                let mut failed = task_state.failure.subscribe();
                loop {
                    let packet = tokio::select! {
                        _ = failed.changed() => break,
                        packet = read_frame(&mut reader) => packet,
                    }
                    .and_then(|packet| admit_response(packet, &task_state));
                    match packet {
                        Ok(packet) => {
                            if let Err(error) = admitted.write_all(&packet).await {
                                task_state.fail(format!("SFTP frame forwarding failed: {error}"));
                                break;
                            }
                        }
                        Err(error) => {
                            let kind = if error.kind() == io::ErrorKind::InvalidData {
                                StorageErrorKind::Protocol
                            } else {
                                StorageErrorKind::Unavailable
                            };
                            task_state
                                .fail_with_kind(kind, format!("SFTP frame read failed: {error}"));
                            break;
                        }
                    }
                }
            }),
            &state,
        );
        if let Some(mut stderr) = stderr {
            let task_state = state.clone();
            tasks.push(tokio::spawn(async move {
                let mut buffer = [0_u8; 1024];
                while let Ok(count) = stderr.read(&mut buffer).await {
                    if count == 0 {
                        break;
                    }
                    let mut captured = task_state
                        .stderr
                        .lock()
                        .unwrap_or_else(|error| error.into_inner());
                    let excess = captured
                        .len()
                        .saturating_add(count)
                        .saturating_sub(STDERR_LIMIT);
                    if excess > 0 {
                        captured.drain(..excess);
                    }
                    captured.extend_from_slice(&buffer[..count]);
                }
            }));
        }
        let size = NonZeroUsize::new(8192)
            .ok_or_else(|| VaultError::Other("invalid SFTP buffer size".into()))?;
        let mut read_end = wire::ReadEnd::new(guarded, size, (*writer).clone());
        let handshake = timeout_at(
            super::super::source_deadline(timeout),
            read_end.receive_server_hello(),
        )
        .await;
        let extensions = match handshake {
            Ok(Ok(extensions)) => extensions,
            result => {
                match result {
                    Ok(Err(error)) => {
                        let kind = match &error {
                            wire::Error::UnsupportedSftpProtocol { .. } => {
                                StorageErrorKind::Unsupported
                            }
                            wire::Error::IOError(_) => StorageErrorKind::Unavailable,
                            _ => StorageErrorKind::Protocol,
                        };
                        state.fail_with_kind(kind, format!("SFTP negotiation failed: {error}"));
                    }
                    _ => state.fail("SFTP negotiation timed out"),
                }
                if let Some(child) = child.0.as_mut() {
                    let _ = child.start_kill();
                    let _ = child.wait().await;
                }
                tokio::task::yield_now().await;
                let message = state.message();
                let kind = if message.contains("Permission denied")
                    || message.contains("Host key verification failed")
                    || message.contains("HOST IDENTIFICATION HAS CHANGED")
                    || message.contains("host key is known")
                {
                    StorageErrorKind::PermissionDenied
                } else {
                    state
                        .failure
                        .borrow()
                        .as_ref()
                        .map_or(StorageErrorKind::Unavailable, |(kind, _)| *kind)
                };
                tasks.abort();
                return Err(VaultError::storage("initialize", "", kind, message));
            }
        };
        let task_state = state.clone();
        tasks.push_pump(tokio::spawn(async move {
            let mut failed = task_state.failure.subscribe();
            loop {
                // Cancellation only occurs when the entire session is poisoned;
                // a partly consumed packet is never resumed on this connection.
                tokio::select! {
                    _ = failed.changed() => break,
                    result = read_end.read_in_one_packet() => if let Err(error) = result {
                        let kind = if matches!(&error, wire::Error::IOError(_)) {
                            StorageErrorKind::Unavailable
                        } else {
                            StorageErrorKind::Protocol
                        };
                        task_state.fail_with_kind(kind, format!("SFTP response failed: {error}"));
                        break;
                    }
                }
            }
        }), &state);
        if let Some(process) = child.0.take() {
            let task_state = state.clone();
            let mut child = ChildOwner(Some(process));
            tasks.push(tokio::spawn(async move {
                let mut failed = task_state.failure.subscribe();
                if failed.borrow().is_none()
                    && let Some(process) = child.0.as_mut()
                {
                    tokio::select! {
                        _ = failed.changed() => {},
                        status = process.wait() => {
                            task_state.fail(format!("SSH subsystem exited: {status:?}"));
                            return;
                        }
                    }
                }
                if let Some(process) = child.0.as_mut() {
                    let _ = process.start_kill();
                    let _ = process.wait().await;
                }
            }));
        }
        Ok(Self(Arc::new(Inner {
            writer: tokio::sync::Mutex::new(writer),
            admission: Semaphore::new(MAX_REQUESTS),
            timeout,
            extensions,
            state,
            tasks,
        })))
    }

    pub(super) fn is_healthy(&self) -> bool {
        self.0.state.failure.borrow().is_none()
    }

    pub(super) fn failure_snapshot(&self) -> Option<(StorageErrorKind, String)> {
        let kind = self
            .0
            .state
            .failure
            .borrow()
            .as_ref()
            .map(|(kind, _)| *kind)?;
        let message = self
            .0
            .state
            .message()
            .chars()
            .filter(|c| !c.is_control())
            .take(1024)
            .collect();
        Some((kind, message))
    }
    pub(super) fn has_posix_rename(&self) -> bool {
        self.0.extensions.contains(wire::Extensions::POSIX_RENAME)
    }
    pub(super) fn has_fsync(&self) -> bool {
        self.0.extensions.contains(wire::Extensions::FSYNC)
    }

    async fn request<T, F, Fut>(
        &self,
        operation: &str,
        path: &str,
        mutating: bool,
        build: F,
    ) -> VaultResult<T>
    where
        F: FnOnce(&mut Writer, Id) -> Result<Fut, wire::Error>,
        Fut: Future<Output = Result<(Id, T), wire::Error>> + Send,
    {
        self.request_with_metadata(operation, path, mutating, build)
            .await
            .map(|(value, _)| value)
    }

    async fn request_with_metadata<T, F, Fut>(
        &self,
        operation: &str,
        path: &str,
        mutating: bool,
        build: F,
    ) -> VaultResult<(T, ResponseMetadata)>
    where
        F: FnOnce(&mut Writer, Id) -> Result<Fut, wire::Error>,
        Fut: Future<Output = Result<(Id, T), wire::Error>> + Send,
    {
        if path.len() > MAX_PACKET / 4 {
            return Err(VaultError::storage(
                operation,
                "",
                StorageErrorKind::ResourceLimit,
                "SFTP path exceeds the supported request size",
            ));
        }
        let deadline = super::super::source_deadline(self.0.timeout);
        let mut failed = self.0.state.failure.subscribe();
        if failed.borrow().is_some() {
            return Err(self.unavailable(operation, path, false));
        }
        let dispatched = AtomicBool::new(false);
        let work = async {
            let _permit = tokio::select! {
                _ = failed.changed() => return Err(self.unavailable(operation, path, false)),
                permit = self.0.admission.acquire() => permit.map_err(|_| self.unavailable(operation, path, false))?,
            };
            let response = Arc::new(Mutex::new(ResponseMetadata {
                directory: operation == "readdir",
                ..ResponseMetadata::default()
            }));
            let mut guard = RequestGuard {
                state: &self.0.state,
                response: response.clone(),
                completed: true,
            };
            let future = {
                let mut writer = self.0.writer.lock().await;
                if !self.is_healthy() {
                    return Err(self.unavailable(operation, path, false));
                }
                self.0
                    .state
                    .requests
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .building = Some(response.clone());
                let id = writer.create_response_id();
                let result = build(&mut writer, id);
                self.0
                    .state
                    .requests
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .building = None;
                result.map_err(|error| map_error(operation, path, error, false))?
            };
            dispatched.store(true, Ordering::Release);
            guard.completed = false;
            let mut lost_connection = false;
            let result = tokio::select! {
                biased;
                result = future => {
                    let mut metadata = response.lock().unwrap_or_else(|error| error.into_inner());
                    let metadata = std::mem::take(&mut *metadata);
                    let status = metadata.status;
                    lost_connection = matches!(status, Some(6 | 7));
                    match result {
                        Ok((_id, value)) => Ok((value, metadata)),
                        Err(error) => {
                            let confirmed_status = matches!(&error, wire::Error::SftpError(_, _))
                                || (status == Some(1) && matches!(&error, wire::Error::InvalidResponse(_)));
                            if !confirmed_status && !self.is_healthy() {
                                Err(self.unavailable(operation, path, mutating))
                            } else {
                                Err(map_response_error(operation, path, error, mutating, status))
                            }
                        }
                    }
                },
                _ = failed.changed() => Err(self.unavailable(operation, path, mutating)),
            };
            guard.completed = true;
            if lost_connection {
                self.0.state.fail("SFTP server reported a lost connection");
            }
            result
        };
        match timeout_at(deadline, work).await {
            Ok(result) => result,
            Err(_) => {
                let dispatched = dispatched.load(Ordering::Acquire);
                if dispatched {
                    self.0.state.fail("SFTP operation deadline exceeded");
                }
                Err(VaultError::storage(
                    operation,
                    path,
                    StorageErrorKind::Unavailable,
                    "SFTP operation deadline exceeded",
                )
                .with_mutation_outcome(if mutating && dispatched {
                    MutationOutcome::Unknown
                } else {
                    MutationOutcome::NotDispatched
                }))
            }
        }
    }

    fn unavailable(&self, operation: &str, path: &str, mutating: bool) -> VaultError {
        VaultError::storage(
            operation,
            path,
            self.0
                .state
                .failure
                .borrow()
                .as_ref()
                .map_or(StorageErrorKind::Unavailable, |(kind, _)| *kind),
            self.0.state.message(),
        )
        .with_mutation_outcome(if mutating {
            MutationOutcome::Unknown
        } else {
            MutationOutcome::NotDispatched
        })
    }

    pub(super) fn track_handle(&self, handle: HandleOwned) -> HandleLifetime<'_> {
        HandleLifetime {
            session: self,
            handle,
            closed: false,
        }
    }

    pub(super) async fn open_read(&self, path: &str) -> VaultResult<HandleOwned> {
        self.request("open", path, false, |writer, id| {
            writer
                .send_open_file_request(
                    id,
                    wire::OpenOptions::new()
                        .read(true)
                        .open(Cow::Borrowed(Path::new(path))),
                )
                .map(|request| request.wait())
        })
        .await
    }

    pub(super) async fn open_create(
        &self,
        path: &str,
        attrs: FileAttrs,
    ) -> VaultResult<HandleOwned> {
        self.request("create", path, true, |writer, id| {
            writer
                .send_open_file_request(
                    id,
                    wire::OpenOptions::new().write(true).create(
                        Cow::Borrowed(Path::new(path)),
                        wire::CreateFlags::Excl,
                        attrs,
                    ),
                )
                .map(|request| request.wait())
        })
        .await
    }

    pub(super) async fn read(
        &self,
        handle: &HandleOwned,
        offset: u64,
        length: u32,
    ) -> VaultResult<Option<Vec<u8>>> {
        if length > CHUNK_SIZE {
            return Err(VaultError::storage(
                "read",
                "",
                StorageErrorKind::ResourceLimit,
                "SFTP read chunk exceeds 32 KiB",
            ));
        }
        let data = self
            .request("read", "", false, |writer, id| {
                writer
                    .send_read_request(id, Cow::Borrowed(handle), offset, length, None)
                    .map(|request| request.wait())
            })
            .await?;
        match data {
            wire::Data::Eof => Ok(None),
            wire::Data::Buffer(data) => self.checked_data(data, length),
            wire::Data::AllocatedBox(data) => self.checked_data(data.into_vec(), length),
        }
    }

    fn checked_data(&self, data: Vec<u8>, requested: u32) -> VaultResult<Option<Vec<u8>>> {
        if data.is_empty() || data.len() > requested as usize {
            self.0
                .state
                .fail("SFTP server returned invalid read progress");
            return Err(VaultError::storage(
                "read",
                "",
                StorageErrorKind::Protocol,
                "SFTP data length does not satisfy the requested read",
            ));
        }
        Ok(Some(data))
    }

    pub(super) async fn write(
        &self,
        handle: &HandleOwned,
        offset: u64,
        data: &[u8],
    ) -> VaultResult<()> {
        if data.len() > CHUNK_SIZE as usize {
            return Err(VaultError::storage(
                "write",
                "",
                StorageErrorKind::ResourceLimit,
                "SFTP write chunk exceeds 32 KiB",
            ));
        }
        self.request("write", "", true, |writer, id| {
            writer
                .send_write_request_buffered(id, Cow::Borrowed(handle), offset, Cow::Borrowed(data))
                .map(|request| request.wait())
        })
        .await
    }

    pub(super) async fn close(&self, handle: &HandleOwned) -> VaultResult<()> {
        self.request("close", "", false, |writer, id| {
            writer
                .send_close_request(id, Cow::Borrowed(handle))
                .map(|request| request.wait())
        })
        .await
    }
    pub(super) async fn fstat(&self, handle: &HandleOwned) -> VaultResult<FileAttrs> {
        self.request("fstat", "", false, |writer, id| {
            writer
                .send_fstat_request(id, Cow::Borrowed(handle))
                .map(|request| request.wait())
        })
        .await
    }
    pub(super) async fn fsetstat(&self, handle: &HandleOwned, attrs: FileAttrs) -> VaultResult<()> {
        self.request("fsetstat", "", true, |writer, id| {
            writer
                .send_fsetstat_request(id, Cow::Borrowed(handle), attrs)
                .map(|request| request.wait())
        })
        .await
    }
    pub(super) async fn fsync(&self, handle: &HandleOwned) -> VaultResult<()> {
        if !self.has_fsync() {
            return Err(VaultError::storage(
                "fsync",
                "",
                StorageErrorKind::Unsupported,
                "server does not support fsync@openssh.com",
            ));
        }
        self.request("fsync", "", true, |writer, id| {
            writer
                .send_fsync_request(id, Cow::Borrowed(handle))
                .map(|request| request.wait())
        })
        .await
    }
    pub(super) async fn stat(&self, path: &str) -> VaultResult<FileAttrs> {
        self.request("stat", path, false, |writer, id| {
            writer
                .send_stat_request(id, Cow::Borrowed(Path::new(path)))
                .map(|request| request.wait())
        })
        .await
    }
    pub(super) async fn lstat(&self, path: &str) -> VaultResult<FileAttrs> {
        self.request("lstat", path, false, |writer, id| {
            writer
                .send_lstat_request(id, Cow::Borrowed(Path::new(path)))
                .map(|request| request.wait())
        })
        .await
    }
    pub(super) async fn realpath(&self, path: &str) -> VaultResult<String> {
        let result = self
            .request("realpath", path, false, |writer, id| {
                writer
                    .send_realpath_request(id, Cow::Borrowed(Path::new(path)))
                    .map(|request| request.wait())
            })
            .await?;
        remote_name(&result)
    }
    pub(super) async fn readlink(&self, path: &str) -> VaultResult<String> {
        let result = self
            .request("readlink", path, false, |writer, id| {
                writer
                    .send_readlink_request(id, Cow::Borrowed(Path::new(path)))
                    .map(|request| request.wait())
            })
            .await?;
        remote_name(&result)
    }
    pub(super) async fn opendir(&self, path: &str) -> VaultResult<HandleOwned> {
        self.request("opendir", path, false, |writer, id| {
            writer
                .send_opendir_request(id, Cow::Borrowed(Path::new(path)))
                .map(|request| request.wait())
        })
        .await
    }
    pub(super) async fn readdir(&self, handle: &HandleOwned) -> VaultResult<DirectoryPage> {
        let (entries, metadata) = self
            .request_with_metadata("readdir", "", false, |writer, id| {
                writer
                    .send_readdir_request(id, Cow::Borrowed(handle))
                    .map(|request| request.wait())
            })
            .await?;
        Ok(DirectoryPage {
            entries,
            skipped_count: metadata.skipped_count,
            skipped_bytes: metadata.skipped_bytes,
        })
    }
    pub(super) async fn mkdir(&self, path: &str) -> VaultResult<()> {
        self.request("mkdir", path, true, |writer, id| {
            writer
                .send_mkdir_request(id, Cow::Borrowed(Path::new(path)), FileAttrs::new())
                .map(|request| request.wait())
        })
        .await
    }
    pub(super) async fn rename(&self, from: &str, to: &str, replace: bool) -> VaultResult<()> {
        if to.len() > MAX_PACKET / 4 {
            return Err(VaultError::storage(
                "rename",
                "",
                StorageErrorKind::ResourceLimit,
                "SFTP destination exceeds the supported request size",
            ));
        }
        if replace && !self.has_posix_rename() {
            return Err(VaultError::storage(
                "replace",
                to,
                StorageErrorKind::Unsupported,
                "server does not support atomic posix-rename@openssh.com replacement",
            ));
        }
        self.request(
            if replace { "replace" } else { "rename" },
            from,
            true,
            |writer, id| {
                if replace {
                    writer
                        .send_posix_rename_request(
                            id,
                            Cow::Borrowed(Path::new(from)),
                            Cow::Borrowed(Path::new(to)),
                        )
                        .map(|request| request.wait())
                } else {
                    writer
                        .send_rename_request(
                            id,
                            Cow::Borrowed(Path::new(from)),
                            Cow::Borrowed(Path::new(to)),
                        )
                        .map(|request| request.wait())
                }
            },
        )
        .await
    }
    pub(super) async fn remove(&self, path: &str) -> VaultResult<()> {
        self.request("remove", path, true, |writer, id| {
            writer
                .send_remove_request(id, Cow::Borrowed(Path::new(path)))
                .map(|request| request.wait())
        })
        .await
    }
    pub(super) async fn rmdir(&self, path: &str) -> VaultResult<()> {
        self.request("rmdir", path, true, |writer, id| {
            writer
                .send_rmdir_request(id, Cow::Borrowed(Path::new(path)))
                .map(|request| request.wait())
        })
        .await
    }
}

fn remote_name(path: &Path) -> VaultResult<String> {
    path.to_str().map(str::to_owned).ok_or_else(|| {
        VaultError::storage(
            "decode_path",
            "",
            StorageErrorKind::Protocol,
            "remote path is not UTF-8",
        )
    })
}

fn map_error(operation: &str, path: &str, error: wire::Error, mutating: bool) -> VaultError {
    use wire::SftpErrorKind;
    if matches!(&error, wire::Error::SftpError(SftpErrorKind::NoSuchFile, _)) {
        return VaultError::NoteNotFound(PathBuf::from(path));
    }
    let (kind, confirmed) = match &error {
        wire::Error::SftpError(SftpErrorKind::PermDenied, _) => {
            (StorageErrorKind::PermissionDenied, true)
        }
        wire::Error::SftpError(SftpErrorKind::OpUnsupported, _)
        | wire::Error::UnsupportedSftpProtocol { .. }
        | wire::Error::UnsupportedExtension(_) => (StorageErrorKind::Unsupported, true),
        wire::Error::SftpError(_, _) => (StorageErrorKind::RemoteFailure, true),
        wire::Error::IOError(_) | wire::Error::AwaitableError(_) => {
            (StorageErrorKind::Unavailable, false)
        }
        _ => (StorageErrorKind::Protocol, false),
    };
    let message: String = error
        .to_string()
        .chars()
        .filter(|c| !c.is_control())
        .take(1024)
        .collect();
    VaultError::storage(operation, path, kind, message).with_mutation_outcome(if !mutating {
        MutationOutcome::NotDispatched
    } else if confirmed {
        MutationOutcome::NotCommitted
    } else {
        MutationOutcome::Unknown
    })
}

fn map_response_error(
    operation: &str,
    path: &str,
    error: wire::Error,
    mutating: bool,
    status: Option<u32>,
) -> VaultError {
    let Some(status @ (1 | 6 | 7)) = status else {
        return map_error(operation, path, error, mutating);
    };
    let kind = if status == 1 {
        StorageErrorKind::RemoteFailure
    } else {
        StorageErrorKind::Unavailable
    };
    let message: String = format!("server returned SFTP status {status}: {error}")
        .chars()
        .filter(|c| !c.is_control())
        .take(1024)
        .collect();
    VaultError::storage(operation, path, kind, message).with_mutation_outcome(if !mutating {
        MutationOutcome::NotDispatched
    } else if status == 1 {
        MutationOutcome::NotCommitted
    } else {
        MutationOutcome::Unknown
    })
}

async fn read_frame(reader: &mut (impl AsyncRead + Unpin)) -> io::Result<Vec<u8>> {
    let size = reader.read_u32().await? as usize;
    if !(5..=MAX_PACKET).contains(&size) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "SFTP frame length is outside supported bounds",
        ));
    }
    let mut packet = vec![0_u8; size + 4];
    packet[..4].copy_from_slice(&(size as u32).to_be_bytes());
    reader.read_exact(&mut packet[4..]).await?;
    validate_packet(&packet[4..])?;
    Ok(packet)
}

fn admit_response(mut packet: Vec<u8>, state: &State) -> io::Result<Vec<u8>> {
    if packet[4] == 2 {
        return Ok(packet);
    }
    let id = u32::from_be_bytes([packet[5], packet[6], packet[7], packet[8]]);
    let response = state
        .requests
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .pending
        .remove(&id)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "unknown or duplicate SFTP response ID",
            )
        })?;
    let mut metadata = response.lock().unwrap_or_else(|error| error.into_inner());
    if packet[4] == 101 {
        let status = u32::from_be_bytes([packet[9], packet[10], packet[11], packet[12]]);
        metadata.status = Some(status);
        if matches!(status, 6 | 7) {
            // The decoder rejects these connection-status pseudo-errors. Keep
            // their authoritative meaning beside the decoder-facing failure.
            packet[9..13].copy_from_slice(&4_u32.to_be_bytes());
        }
    } else if packet[4] == 104 && metadata.directory {
        let body = &packet[4..];
        let mut offset = 5;
        let count = u32_at(body, &mut offset)?;
        let mut filtered = packet[..13].to_vec();
        let mut accepted = 0_u32;
        for _ in 0..count {
            let start = offset;
            string(body, &mut offset)?;
            let filename = &body[start + 4..offset];
            if filename.is_empty() || filename.contains(&0) || filename.contains(&b'/') {
                return Err(invalid());
            }
            string(body, &mut offset)?;
            attrs(body, &mut offset)?;
            if std::str::from_utf8(filename).is_ok() {
                filtered.extend_from_slice(&body[start..offset]);
                accepted += 1;
            } else {
                metadata.skipped_count += 1;
                metadata.skipped_bytes += filename.len();
            }
        }
        if metadata.skipped_count != 0 {
            filtered[9..13].copy_from_slice(&accepted.to_be_bytes());
            let length = filtered.len() as u32 - 4;
            filtered[..4].copy_from_slice(&length.to_be_bytes());
            packet = filtered;
        }
    }
    Ok(packet)
}

fn invalid() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        "invalid SFTP response container",
    )
}
fn u32_at(bytes: &[u8], offset: &mut usize) -> io::Result<usize> {
    let end = offset.checked_add(4).ok_or_else(invalid)?;
    let data: [u8; 4] = bytes
        .get(*offset..end)
        .ok_or_else(invalid)?
        .try_into()
        .map_err(|_| invalid())?;
    *offset = end;
    Ok(u32::from_be_bytes(data) as usize)
}
fn string(bytes: &[u8], offset: &mut usize) -> io::Result<()> {
    let length = u32_at(bytes, offset)?;
    *offset = offset
        .checked_add(length)
        .filter(|end| *end <= bytes.len())
        .ok_or_else(invalid)?;
    Ok(())
}
fn attrs(bytes: &[u8], offset: &mut usize) -> io::Result<()> {
    let flags = u32_at(bytes, offset)?;
    if flags & !0x8000000f != 0 {
        return Err(invalid());
    }
    for (flag, size) in [(1, 8), (2, 8), (4, 4), (8, 8)] {
        if flags & flag != 0 {
            if flag == 4 {
                let mut cursor = *offset;
                let mode = u32_at(bytes, &mut cursor)? as u32 & 0o170000;
                if !matches!(
                    mode,
                    0 | 0o010000 | 0o020000 | 0o040000 | 0o060000 | 0o100000 | 0o120000 | 0o140000
                ) {
                    return Err(invalid());
                }
            }
            *offset = offset
                .checked_add(size)
                .filter(|end| *end <= bytes.len())
                .ok_or_else(invalid)?;
        }
    }
    if flags & 0x80000000 != 0 {
        let count = u32_at(bytes, offset)?;
        if count > (bytes.len() - *offset) / 8 {
            return Err(invalid());
        }
        for _ in 0..count {
            string(bytes, offset)?;
            string(bytes, offset)?;
        }
    }
    Ok(())
}

/// Validate the few variable-length response containers before Serde can use
/// their counts as allocation hints. Encoding/decoding remains in the library.
fn validate_packet(packet: &[u8]) -> io::Result<()> {
    let mut offset = 5;
    match packet[0] {
        2 => {
            while offset < packet.len() {
                string(packet, &mut offset)?;
                string(packet, &mut offset)?;
            }
        }
        101 => {
            u32_at(packet, &mut offset)?;
            string(packet, &mut offset)?;
            string(packet, &mut offset)?;
        }
        102 => {
            let mut cursor = offset;
            if u32_at(packet, &mut cursor)? > 256 {
                return Err(invalid());
            }
            string(packet, &mut offset)?;
        }
        103 => string(packet, &mut offset)?,
        104 => {
            let count = u32_at(packet, &mut offset)?;
            if count > (packet.len() - offset) / 12 {
                return Err(invalid());
            }
            for _ in 0..count {
                string(packet, &mut offset)?;
                string(packet, &mut offset)?;
                attrs(packet, &mut offset)?;
            }
        }
        105 => attrs(packet, &mut offset)?,
        201 => return Ok(()),
        _ => return Err(invalid()),
    }
    if offset != packet.len() {
        return Err(invalid());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn frame_guard_rejects_lengths_before_allocation() {
        for length in [0_u32, 4, MAX_PACKET as u32 + 1, u32::MAX] {
            let bytes = length.to_be_bytes();
            assert_eq!(
                read_frame(&mut bytes.as_slice()).await.unwrap_err().kind(),
                io::ErrorKind::InvalidData
            );
        }
    }

    #[test]
    fn frame_guard_rejects_directory_allocation_hint() {
        let mut body = vec![104, 0, 0, 0, 1];
        body.extend_from_slice(&u32::MAX.to_be_bytes());
        assert!(validate_packet(&body).is_err());
    }

    #[test]
    fn frame_guard_accepts_binary_handles() {
        let body = [102, 0, 0, 0, 1, 0, 0, 0, 3, 0xff, 0, 0xfe];
        assert!(validate_packet(&body).is_ok());
    }

    #[test]
    fn frame_guard_rejects_oversized_handles() {
        let mut body = vec![102, 0, 0, 0, 1];
        body.extend_from_slice(&257_u32.to_be_bytes());
        body.extend_from_slice(&[0; 257]);
        assert!(validate_packet(&body).is_err());
        body[5..9].copy_from_slice(&256_u32.to_be_bytes());
        body.pop();
        assert!(validate_packet(&body).is_ok());
    }

    #[test]
    fn frame_guard_rejects_unknown_file_types() {
        let mut body = vec![105, 0, 0, 0, 1];
        body.extend_from_slice(&4_u32.to_be_bytes());
        body.extend_from_slice(&0o030600_u32.to_be_bytes());
        assert!(validate_packet(&body).is_err());
        body[9..13].copy_from_slice(&0o100600_u32.to_be_bytes());
        assert!(validate_packet(&body).is_ok());
        body[9..13].copy_from_slice(&0o600_u32.to_be_bytes());
        assert!(validate_packet(&body).is_ok());
    }

    #[tokio::test]
    async fn request_ids_correlate_reordered_responses() {
        let (client, mut server) = tokio::io::duplex(4096);
        let fake = tokio::spawn(async move {
            server_hello(&mut server, false).await;
            let first = read_frame_request(&mut server).await;
            let second = read_frame_request(&mut server).await;
            for (request, size) in [(&second, 8_u64), (&first, 7)] {
                let mut reply = vec![105];
                reply.extend_from_slice(&request[1..5]);
                reply.extend_from_slice(&1_u32.to_be_bytes());
                reply.extend_from_slice(&size.to_be_bytes());
                write_reply(&mut server, &reply).await;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        });
        let (reader, writer) = tokio::io::split(client);
        let session = Session::from_io(reader, writer, Duration::from_secs(2))
            .await
            .unwrap();
        let (first, second) = tokio::join!(session.stat("/first"), session.stat("/second"));
        assert_eq!(first.unwrap().get_size(), Some(7));
        assert_eq!(second.unwrap().get_size(), Some(8));
        fake.await.unwrap();
    }

    #[tokio::test]
    async fn rename_uses_the_requested_publication_semantics() {
        let (client, mut server) = tokio::io::duplex(4096);
        let fake = tokio::spawn(async move {
            server_hello(&mut server, true).await;
            let standard = read_frame_request(&mut server).await;
            assert_eq!(standard[0], 18);
            status_ok(&mut server, &standard).await;
            let replacement = read_frame_request(&mut server).await;
            assert_eq!(replacement[0], 200);
            assert_eq!(&replacement[9..33], b"posix-rename@openssh.com");
            status_ok(&mut server, &replacement).await;
            tokio::time::sleep(Duration::from_millis(50)).await;
        });
        let (reader, writer) = tokio::io::split(client);
        let session = Session::from_io(reader, writer, Duration::from_secs(2))
            .await
            .unwrap();
        session.rename("/source", "/target", false).await.unwrap();
        session.rename("/upload", "/target", true).await.unwrap();
        fake.await.unwrap();
    }

    #[tokio::test]
    async fn only_a_confirmed_handle_close_keeps_the_session_reusable() {
        for close_status in [Some(0), Some(4), None] {
            let (client, mut server) = tokio::io::duplex(4096);
            let (finish, finished) = tokio::sync::oneshot::channel();
            let fake = tokio::spawn(async move {
                server_hello(&mut server, false).await;
                let request = read_frame_request(&mut server).await;
                assert_eq!(request[0], 3);
                let mut reply = vec![102];
                reply.extend_from_slice(&request[1..5]);
                reply.extend_from_slice(&[0, 0, 0, 1, 0xff]);
                write_reply(&mut server, &reply).await;
                if let Some(status) = close_status {
                    let request = read_frame_request(&mut server).await;
                    assert_eq!(request[0], 4);
                    status_reply(&mut server, &request, status).await;
                }
                let _ = finished.await;
            });
            let (reader, writer) = tokio::io::split(client);
            let session = Session::from_io(reader, writer, Duration::from_secs(2))
                .await
                .unwrap();
            let handle = session.track_handle(session.open_read("/vault/note.md").await.unwrap());
            assert_eq!(handle.handle().into_inner(), &[0xff]);
            match close_status {
                Some(0) => handle.close().await.unwrap(),
                Some(_) => assert!(handle.close().await.is_err()),
                None => drop(handle),
            }
            assert_eq!(session.is_healthy(), close_status == Some(0));
            finish.send(()).unwrap();
            fake.await.unwrap();
        }
    }

    #[tokio::test]
    async fn rejected_mutations_preserve_the_session_and_confirm_no_commit() {
        let (client, mut server) = tokio::io::duplex(4096);
        let (finish, finished) = tokio::sync::oneshot::channel();
        let fake = tokio::spawn(async move {
            server_hello(&mut server, false).await;
            for code in [1, 4, 5, 9] {
                let request = read_frame_request(&mut server).await;
                assert_eq!(request[0], 13);
                status_reply(&mut server, &request, code).await;
            }
            let request = read_frame_request(&mut server).await;
            assert_eq!(request[0], 3);
            let mut reply = vec![102];
            reply.extend_from_slice(&request[1..5]);
            reply.extend_from_slice(&[0, 0, 0, 1, 1]);
            write_reply(&mut server, &reply).await;
            let request = read_frame_request(&mut server).await;
            assert_eq!(request[0], 5);
            let mut reply = vec![103];
            reply.extend_from_slice(&request[1..5]);
            reply.extend_from_slice(&[0, 0, 0, 2, b'o', b'k']);
            write_reply(&mut server, &reply).await;
            let _ = finished.await;
        });
        let (reader, writer) = tokio::io::split(client);
        let session = Session::from_io(reader, writer, Duration::from_secs(2))
            .await
            .unwrap();
        for _ in [1, 4, 5, 9] {
            assert!(matches!(
                session.remove("/vault/note.md").await,
                Err(VaultError::Storage {
                    kind: StorageErrorKind::RemoteFailure,
                    outcome: MutationOutcome::NotCommitted,
                    ..
                })
            ));
            assert!(session.is_healthy());
        }
        let handle = session.open_read("/vault/note.md").await.unwrap();
        assert_eq!(session.read(&handle, 0, 10).await.unwrap().unwrap(), b"ok");
        assert!(session.is_healthy());
        assert!(session.0.state.requests.lock().unwrap().pending.is_empty());
        finish.send(()).unwrap();
        fake.await.unwrap();
    }

    #[tokio::test]
    async fn connection_loss_status_after_remove_has_unknown_outcome_and_retires_session() {
        for code in [6, 7] {
            let (client, mut server) = tokio::io::duplex(4096);
            let (finish, finished) = tokio::sync::oneshot::channel();
            let source_exists = Arc::new(AtomicBool::new(true));
            let remote_exists = source_exists.clone();
            let fake = tokio::spawn(async move {
                server_hello(&mut server, false).await;
                let request = read_frame_request(&mut server).await;
                assert_eq!(request[0], 13);
                assert_eq!(&request[9..], b"/vault/note.md");
                remote_exists.store(false, Ordering::Release);
                status_reply(&mut server, &request, code).await;
                let _ = finished.await;
            });
            let (reader, writer) = tokio::io::split(client);
            let session = Session::from_io(reader, writer, Duration::from_secs(2))
                .await
                .unwrap();
            assert!(matches!(
                session.remove("/vault/note.md").await,
                Err(VaultError::Storage {
                    kind: StorageErrorKind::Unavailable,
                    outcome: MutationOutcome::Unknown,
                    ..
                })
            ));
            assert!(!source_exists.load(Ordering::Acquire));
            assert!(!session.is_healthy());
            assert!(session.0.state.requests.lock().unwrap().pending.is_empty());
            finish.send(()).unwrap();
            fake.await.unwrap();
        }
    }

    #[tokio::test]
    async fn unknown_response_id_retires_the_session_with_uncertain_mutation() {
        let (client, mut server) = tokio::io::duplex(4096);
        let (finish, finished) = tokio::sync::oneshot::channel();
        let fake = tokio::spawn(async move {
            server_hello(&mut server, false).await;
            let mut request = read_frame_request(&mut server).await;
            request[4] ^= 1;
            status_ok(&mut server, &request).await;
            let _ = finished.await;
        });
        let (reader, writer) = tokio::io::split(client);
        let session = Session::from_io(reader, writer, Duration::from_secs(2))
            .await
            .unwrap();
        assert!(matches!(
            session.remove("/vault/note.md").await,
            Err(VaultError::Storage {
                kind: StorageErrorKind::Protocol,
                outcome: MutationOutcome::Unknown,
                ..
            })
        ));
        assert!(!session.is_healthy());
        finish.send(()).unwrap();
        fake.await.unwrap();
    }

    #[tokio::test]
    async fn duplicate_response_id_retires_the_session() {
        let (client, mut server) = tokio::io::duplex(4096);
        let (duplicate, duplicate_now) = tokio::sync::oneshot::channel();
        let (finish, finished) = tokio::sync::oneshot::channel();
        let fake = tokio::spawn(async move {
            server_hello(&mut server, false).await;
            let request = read_frame_request(&mut server).await;
            status_ok(&mut server, &request).await;
            duplicate_now.await.unwrap();
            status_ok(&mut server, &request).await;
            let _ = finished.await;
        });
        let (reader, writer) = tokio::io::split(client);
        let session = Session::from_io(reader, writer, Duration::from_secs(2))
            .await
            .unwrap();
        session.remove("/vault/note.md").await.unwrap();
        let mut failed = session.0.state.failure.subscribe();
        duplicate.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(1), failed.changed())
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(
            session.stat("/vault/note.md").await,
            Err(VaultError::Storage {
                kind: StorageErrorKind::Protocol,
                outcome: MutationOutcome::NotDispatched,
                ..
            })
        ));
        assert!(!session.is_healthy());
        finish.send(()).unwrap();
        fake.await.unwrap();
    }

    #[tokio::test]
    async fn non_utf8_resolved_paths_remain_protocol_failures() {
        for readlink in [false, true] {
            let (client, mut server) = tokio::io::duplex(4096);
            let (finish, finished) = tokio::sync::oneshot::channel();
            let fake = tokio::spawn(async move {
                server_hello(&mut server, false).await;
                let request = read_frame_request(&mut server).await;
                assert_eq!(request[0], if readlink { 19 } else { 16 });
                let mut reply = vec![104];
                reply.extend_from_slice(&request[1..5]);
                reply.extend_from_slice(&1_u32.to_be_bytes());
                reply.extend_from_slice(&[0, 0, 0, 1, 0xff]);
                reply.extend_from_slice(&[0; 8]);
                write_reply(&mut server, &reply).await;
                let _ = finished.await;
            });
            let (reader, writer) = tokio::io::split(client);
            let session = Session::from_io(reader, writer, Duration::from_secs(2))
                .await
                .unwrap();
            let result = if readlink {
                session.readlink("/vault/link").await
            } else {
                session.realpath("/vault").await
            };
            assert!(matches!(
                result,
                Err(VaultError::Storage {
                    kind: StorageErrorKind::Protocol,
                    outcome: MutationOutcome::NotDispatched,
                    ..
                })
            ));
            assert!(!session.is_healthy());
            assert!(session.0.state.requests.lock().unwrap().pending.is_empty());
            finish.send(()).unwrap();
            fake.await.unwrap();
        }
    }

    #[tokio::test]
    async fn replacement_without_extension_is_rejected_before_dispatch() {
        let (client, mut server) = tokio::io::duplex(4096);
        let fake = tokio::spawn(async move {
            server_hello(&mut server, false).await;
            let mut byte = [0];
            assert!(
                tokio::time::timeout(Duration::from_millis(50), server.read(&mut byte))
                    .await
                    .is_err()
            );
        });
        let (reader, writer) = tokio::io::split(client);
        let session = Session::from_io(reader, writer, Duration::from_secs(2))
            .await
            .unwrap();
        assert!(matches!(
            session.rename("/upload", "/target", true).await,
            Err(VaultError::Storage {
                kind: StorageErrorKind::Unsupported,
                outcome: MutationOutcome::NotDispatched,
                ..
            })
        ));
        fake.await.unwrap();
    }

    #[tokio::test]
    async fn dispatched_mutation_timeout_poisoned_connection_has_unknown_outcome() {
        let (client, mut server) = tokio::io::duplex(4096);
        let fake = tokio::spawn(async move {
            server_hello(&mut server, false).await;
            let request = read_frame_request(&mut server).await;
            assert_eq!(request[0], 18);
            tokio::time::sleep(Duration::from_millis(100)).await;
        });
        let (reader, writer) = tokio::io::split(client);
        let session = Session::from_io(reader, writer, Duration::from_millis(30))
            .await
            .unwrap();
        assert!(matches!(
            session.rename("/source", "/target", false).await,
            Err(VaultError::Storage {
                kind: StorageErrorKind::Unavailable,
                outcome: MutationOutcome::Unknown,
                ..
            })
        ));
        assert!(!session.is_healthy());
        assert!(matches!(
            session.stat("/source").await,
            Err(VaultError::Storage {
                outcome: MutationOutcome::NotDispatched,
                ..
            })
        ));
        fake.await.unwrap();
    }

    #[tokio::test]
    async fn cancelled_request_does_not_leave_a_reusable_session() {
        let (client, mut server) = tokio::io::duplex(4096);
        let (seen, received) = tokio::sync::oneshot::channel();
        let (finish, finished) = tokio::sync::oneshot::channel();
        let fake = tokio::spawn(async move {
            server_hello(&mut server, false).await;
            read_frame_request(&mut server).await;
            seen.send(()).unwrap();
            let _ = finished.await;
        });
        let (reader, writer) = tokio::io::split(client);
        let session = Session::from_io(reader, writer, Duration::from_secs(2))
            .await
            .unwrap();
        let worker_session = session.clone();
        let request = tokio::spawn(async move { worker_session.stat("/note").await });
        received.await.unwrap();
        request.abort();
        assert!(request.await.unwrap_err().is_cancelled());
        assert!(!session.is_healthy());
        assert!(session.0.state.requests.lock().unwrap().pending.is_empty());
        finish.send(()).unwrap();
        fake.await.unwrap();
    }

    #[tokio::test]
    async fn deadline_drops_a_blocked_writer_while_session_is_retained() {
        use std::pin::Pin;
        use std::task::{Context, Poll};

        struct BlockAfterHello {
            output: tokio::io::WriteHalf<tokio::io::DuplexStream>,
            written: usize,
            dropped: Option<tokio::sync::oneshot::Sender<()>>,
        }
        impl AsyncWrite for BlockAfterHello {
            fn poll_write(
                mut self: Pin<&mut Self>,
                cx: &mut Context<'_>,
                bytes: &[u8],
            ) -> Poll<io::Result<usize>> {
                if self.written >= 9 {
                    return Poll::Pending;
                }
                let allowed = bytes.len().min(9 - self.written);
                match Pin::new(&mut self.output).poll_write(cx, &bytes[..allowed]) {
                    Poll::Ready(Ok(count)) => {
                        self.written += count;
                        Poll::Ready(Ok(count))
                    }
                    other => other,
                }
            }
            fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
                Pin::new(&mut self.output).poll_flush(cx)
            }
            fn poll_shutdown(
                mut self: Pin<&mut Self>,
                cx: &mut Context<'_>,
            ) -> Poll<io::Result<()>> {
                Pin::new(&mut self.output).poll_shutdown(cx)
            }
        }
        impl Drop for BlockAfterHello {
            fn drop(&mut self) {
                if let Some(dropped) = self.dropped.take() {
                    let _ = dropped.send(());
                }
            }
        }

        let (client, mut server) = tokio::io::duplex(4096);
        let (finish, finished) = tokio::sync::oneshot::channel();
        let fake = tokio::spawn(async move {
            server_hello(&mut server, false).await;
            let _ = finished.await;
        });
        let (reader, writer) = tokio::io::split(client);
        let (dropped, observed_drop) = tokio::sync::oneshot::channel();
        let session = Session::from_io(
            reader,
            BlockAfterHello {
                output: writer,
                written: 0,
                dropped: Some(dropped),
            },
            Duration::from_millis(30),
        )
        .await
        .unwrap();
        assert!(matches!(
            session.rename("/source", "/target", false).await,
            Err(VaultError::Storage {
                outcome: MutationOutcome::Unknown,
                ..
            })
        ));
        tokio::time::timeout(Duration::from_secs(1), observed_drop)
            .await
            .unwrap()
            .unwrap();
        assert!(!session.is_healthy());
        finish.send(()).unwrap();
        fake.await.unwrap();
    }

    async fn server_hello(server: &mut (impl AsyncRead + AsyncWrite + Unpin), posix: bool) {
        let mut hello = [0; 9];
        server.read_exact(&mut hello).await.unwrap();
        let mut reply = vec![2, 0, 0, 0, 3];
        if posix {
            let name = b"posix-rename@openssh.com";
            reply.extend_from_slice(&(name.len() as u32).to_be_bytes());
            reply.extend_from_slice(name);
            reply.extend_from_slice(&[0, 0, 0, 1, b'1']);
        }
        write_reply(server, &reply).await;
    }

    async fn status_ok(server: &mut (impl AsyncWrite + Unpin), request: &[u8]) {
        status_reply(server, request, 0).await;
    }

    async fn status_reply(server: &mut (impl AsyncWrite + Unpin), request: &[u8], code: u32) {
        let mut reply = vec![101];
        reply.extend_from_slice(&request[1..5]);
        reply.extend_from_slice(&code.to_be_bytes());
        reply.extend_from_slice(&[0; 8]);
        write_reply(server, &reply).await;
    }

    #[tokio::test]
    async fn binary_handle_and_short_read_survive_protocol_round_trip() {
        let (client, mut server) = tokio::io::duplex(4096);
        let fake = tokio::spawn(async move {
            let mut hello = [0; 9];
            server.read_exact(&mut hello).await.unwrap();
            server
                .write_all(&[0, 0, 0, 5, 2, 0, 0, 0, 3])
                .await
                .unwrap();
            let request = read_frame_request(&mut server).await;
            let id = &request[1..5];
            let mut reply = vec![102];
            reply.extend_from_slice(id);
            reply.extend_from_slice(&[0, 0, 0, 3, 0xff, 0, 0xfe]);
            write_reply(&mut server, &reply).await;
            let request = read_frame_request(&mut server).await;
            assert_eq!(request[0], 5);
            assert_eq!(&request[9..12], &[0xff, 0, 0xfe]);
            let mut reply = vec![103];
            reply.extend_from_slice(&request[1..5]);
            reply.extend_from_slice(&[0, 0, 0, 2, b'o', b'k']);
            write_reply(&mut server, &reply).await;
            tokio::time::sleep(Duration::from_millis(100)).await;
        });
        let (reader, writer) = tokio::io::split(client);
        let session = Session::from_io(reader, writer, Duration::from_secs(2))
            .await
            .unwrap();
        let handle = session.open_read("/vault/a.md").await.unwrap();
        assert_eq!(handle.into_inner(), &[0xff, 0, 0xfe]);
        assert_eq!(session.read(&handle, 0, 10).await.unwrap().unwrap(), b"ok");
        fake.await.unwrap();
    }

    async fn read_frame_request(reader: &mut (impl AsyncRead + Unpin)) -> Vec<u8> {
        let length = reader.read_u32().await.unwrap();
        let mut result = vec![0; length as usize];
        reader.read_exact(&mut result).await.unwrap();
        result
    }
    async fn write_reply(writer: &mut (impl AsyncWrite + Unpin), body: &[u8]) {
        writer
            .write_all(&(body.len() as u32).to_be_bytes())
            .await
            .unwrap();
        writer.write_all(body).await.unwrap();
    }
}
