//! Ownership checks for the HTTP listener controlled by the local CLI.

use std::io;
use std::net::SocketAddr;
#[cfg(not(windows))]
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, TcpStream};
use std::path::Path;
use std::time::Duration;

pub(crate) fn bind_available_tcp(endpoint: SocketAddr) -> io::Result<tokio::net::TcpSocket> {
    #[cfg(windows)]
    let occupied = endpoint.port() != 0 && platform::has_overlapping_listener(endpoint)?;
    #[cfg(not(windows))]
    let occupied = endpoint.port() != 0 && endpoint_is_reachable(endpoint)?;
    if occupied {
        return Err(io::Error::new(
            io::ErrorKind::AddrInUse,
            format!("{endpoint} already has a listener"),
        ));
    }
    let socket = if endpoint.is_ipv4() {
        tokio::net::TcpSocket::new_v4()?
    } else {
        tokio::net::TcpSocket::new_v6()?
    };
    // Retain Mio's platform policy so closed connections do not block a restart.
    socket.set_reuseaddr(cfg!(unix))?;
    socket.bind(endpoint)?;
    Ok(socket)
}

#[cfg(not(windows))]
fn endpoint_is_reachable(endpoint: SocketAddr) -> io::Result<bool> {
    let connect_ip = match endpoint.ip() {
        IpAddr::V4(ip) if ip.is_unspecified() => IpAddr::V4(Ipv4Addr::LOCALHOST),
        IpAddr::V6(ip) if ip.is_unspecified() => IpAddr::V6(Ipv6Addr::LOCALHOST),
        ip => ip,
    };
    match TcpStream::connect_timeout(
        &SocketAddr::new(connect_ip, endpoint.port()),
        Duration::from_millis(200),
    ) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == io::ErrorKind::ConnectionRefused => Ok(false),
        Err(error) => Err(error),
    }
}

pub(crate) fn listener_pid(endpoint: SocketAddr) -> io::Result<Option<u32>> {
    if endpoint.port() == 0 {
        return Ok(None);
    }
    // Unix lsof can hide another user's listener. Windows supplies the
    // system-wide owner table directly, including an empty endpoint.
    #[cfg(unix)]
    let reachable = endpoint_is_reachable(endpoint)?;
    #[cfg(unix)]
    if !reachable && !endpoint.ip().is_unspecified() {
        return Ok(None);
    }
    let mut pids = platform_listener_pids(endpoint)?;
    // A wildcard also overlaps listeners bound exclusively to another interface.
    // lsof can hide another user's sockets even when its command succeeds.
    #[cfg(target_os = "linux")]
    if !reachable && pids.is_empty() && platform::wildcard_port_in_use(endpoint)? {
        return Err(io::Error::other(format!(
            "{endpoint} overlaps an unidentified listener; stop it through its owner"
        )));
    }
    pids.sort_unstable();
    pids.dedup();
    match pids.as_slice() {
        [pid] => Ok(Some(*pid)),
        #[cfg(unix)]
        [] if reachable => Err(io::Error::other(format!(
            "{endpoint} is occupied but its exact listener could not be identified; use the owning service to stop it"
        ))),
        [] => Ok(None),
        _ => Err(io::Error::other(format!(
            "multiple processes own the listener on {endpoint}; use the owning service to stop it"
        ))),
    }
}

fn endpoints_overlap(requested: SocketAddr, bound: SocketAddr) -> bool {
    requested.is_ipv4() == bound.is_ipv4()
        && requested.port() == bound.port()
        && (requested == bound || requested.ip().is_unspecified() || bound.ip().is_unspecified())
}

fn collect_owner(
    pids: &mut Vec<u32>,
    requested: SocketAddr,
    bound: SocketAddr,
    pid: u32,
) -> io::Result<()> {
    if bound == requested {
        pids.push(pid);
    } else if endpoints_overlap(requested, bound) {
        return Err(io::Error::other(format!(
            "{requested} overlaps the listener on {bound}; specify its exact bind address"
        )));
    }
    Ok(())
}

#[cfg(unix)]
fn platform_listener_pids(endpoint: SocketAddr) -> io::Result<Vec<u32>> {
    let family = if endpoint.is_ipv4() { 4 } else { 6 };
    let selector = format!("-i{family}TCP:{}", endpoint.port());
    #[cfg(target_os = "macos")]
    let lsof = "/usr/sbin/lsof";
    #[cfg(not(target_os = "macos"))]
    let lsof = "lsof";
    let output = std::process::Command::new(lsof)
        .args(["-nP", &selector, "-sTCP:LISTEN", "-Fpn"])
        .output()?;
    if !output.status.success() && !(output.status.code() == Some(1) && output.stderr.is_empty()) {
        return Err(io::Error::other(format!(
            "cannot inspect listeners with lsof ({}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    let mut pids = Vec::new();
    let mut pid = None;
    for field in std::str::from_utf8(&output.stdout)
        .map_err(io::Error::other)?
        .lines()
    {
        if let Some(value) = field.strip_prefix('p') {
            pid = Some(value.parse::<u32>().map_err(io::Error::other)?);
        } else if let Some(name) = field.strip_prefix('n') {
            let addr = if let Some(port) = name.strip_prefix("*:") {
                let wildcard = if endpoint.is_ipv4() {
                    IpAddr::V4(Ipv4Addr::UNSPECIFIED)
                } else {
                    IpAddr::V6(Ipv6Addr::UNSPECIFIED)
                };
                SocketAddr::new(wildcard, port.parse().map_err(io::Error::other)?)
            } else {
                name.parse().map_err(io::Error::other)?
            };
            collect_owner(
                &mut pids,
                endpoint,
                addr,
                pid.ok_or_else(|| io::Error::other("lsof listener has no process ID"))?,
            )?;
        }
    }
    Ok(pids)
}

#[cfg(windows)]
use platform::platform_listener_pids;

#[cfg(not(any(target_os = "macos", target_os = "linux", windows)))]
pub(crate) fn stop(endpoint: SocketAddr, _expected: &Path) -> io::Result<bool> {
    if listener_pid(endpoint)?.is_some() {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "this platform cannot safely signal a specific process instance; stop it through its owner",
        ));
    }
    Ok(false)
}

#[cfg(any(target_os = "macos", target_os = "linux", windows))]
#[derive(Debug, thiserror::Error)]
#[error("listener inspection {operation}: {source}")]
struct ListenerInspectionError {
    operation: &'static str,
    source: io::Error,
}

#[cfg(any(target_os = "macos", target_os = "linux", windows))]
fn inspect_listener(endpoint: SocketAddr, operation: &'static str) -> io::Result<Option<u32>> {
    listener_pid(endpoint).map_err(|source| {
        io::Error::new(source.kind(), ListenerInspectionError { operation, source })
    })
}

#[cfg(any(target_os = "macos", target_os = "linux", windows))]
pub(crate) fn stop(endpoint: SocketAddr, expected_executable: &Path) -> io::Result<bool> {
    let Some(pid) = inspect_listener(endpoint, "before stopping")? else {
        // Darwin may hide other users' kernel socket records. An exclusive bind
        // can confirm this empty wildcard before any child has been spawned.
        // Never take this port while waiting for the actual child to bind.
        #[cfg(target_os = "macos")]
        if endpoint.ip().is_unspecified()
            && endpoint.port() != 0
            && platform::wildcard_port_in_use(endpoint)?
        {
            return Err(io::Error::other(format!(
                "cannot verify {endpoint} is free: the address is still in use; check its owning service"
            )));
        }
        return Ok(false);
    };
    if pid == std::process::id() {
        return Err(io::Error::other("refusing to stop the calling process"));
    }
    let process = platform::Process::open(pid).map_err(|error| {
        io::Error::new(error.kind(), format!(
            "cannot pin the listener on {endpoint} (PID {pid}): {error}; stop it through its owner"
        ))
    })?;
    stop_instance(endpoint, pid, &process, expected_executable)?;
    Ok(true)
}

#[cfg(any(target_os = "macos", target_os = "linux", windows))]
fn stop_instance(
    endpoint: SocketAddr,
    pid: u32,
    process: &platform::Process,
    expected_executable: &Path,
) -> io::Result<()> {
    process.verify_executable(expected_executable)?;
    if inspect_listener(endpoint, "before termination")? != Some(pid) {
        return Err(io::Error::other(format!(
            "listener on {endpoint} changed during inspection"
        )));
    }
    process.terminate(false)?;
    if !process.wait(Duration::from_secs(5))? {
        // The same instance may still be draining requests after closing its listener.
        // If a supervisor replaced it, leave the replacement to its owner.
        if inspect_listener(endpoint, "before forced termination")?
            .is_some_and(|current| current != pid)
        {
            return Err(io::Error::other(format!(
                "another process now owns {endpoint}"
            )));
        }
        process.verify_executable(expected_executable)?;
        process.terminate(true)?;
        if !process.wait(Duration::from_secs(2))? {
            return Err(io::Error::other(format!(
                "HTTP server process {pid} did not exit"
            )));
        }
    }
    if let Some(owner) = inspect_listener(endpoint, "after process exit")? {
        return Err(io::Error::other(format!(
            "HTTP server process {pid} exited but process {owner} now owns {endpoint}"
        )));
    }
    Ok(())
}

#[cfg(any(target_os = "macos", target_os = "linux", windows))]
fn foreign_executable(expected: &Path) -> io::Error {
    io::Error::new(
        io::ErrorKind::PermissionDenied,
        format!(
            "refusing to stop the listener: its executable does not match '{}'; use its owning service to stop it",
            expected.display()
        ),
    )
}

#[cfg(target_os = "linux")]
mod platform {
    use super::*;
    use rustix::event::{PollFd, PollFlags, Timespec, poll};
    use rustix::process::{Pid, PidfdFlags, Signal, pidfd_open, pidfd_send_signal};
    use std::os::fd::OwnedFd;
    use std::os::unix::fs::MetadataExt;

    pub(super) fn wildcard_port_in_use(endpoint: SocketAddr) -> io::Result<bool> {
        let table = if endpoint.is_ipv4() {
            "/proc/net/tcp"
        } else {
            "/proc/net/tcp6"
        };
        let suffix = format!(":{:04X}", endpoint.port());
        for row in std::fs::read_to_string(table)?.lines().skip(1) {
            let mut fields = row.split_ascii_whitespace();
            let malformed = || io::Error::other(format!("invalid socket row in {table}"));
            let address = fields.nth(1).ok_or_else(malformed)?;
            let state = fields.nth(1).ok_or_else(malformed)?;
            if state == "0A" && address.ends_with(&suffix) {
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub(super) struct Process {
        pid: u32,
        handle: OwnedFd,
    }

    impl Process {
        pub(super) fn open(pid: u32) -> io::Result<Self> {
            let id = Pid::from_raw(pid.try_into().map_err(io::Error::other)?)
                .ok_or_else(|| io::Error::other("invalid listener process ID"))?;
            Ok(Self {
                pid,
                handle: pidfd_open(id, PidfdFlags::empty())?,
            })
        }

        pub(super) fn verify_executable(&self, expected: &Path) -> io::Result<()> {
            use std::os::unix::ffi::OsStrExt;

            let proc_exe = format!("/proc/{}/exe", self.pid);
            let mapped = std::fs::read_link(&proc_exe)?;
            let image = std::fs::metadata(proc_exe)?;
            let installed = expected.canonicalize()?;
            let installed_image = std::fs::metadata(&installed)?;
            if mapped == installed
                && (image.dev(), image.ino()) == (installed_image.dev(), installed_image.ino())
            {
                return Ok(());
            }
            // Cargo can replace the installed binary while its previous image still
            // serves HTTP. The kernel retains the mapped executable's original path.
            if image.nlink() == 0
                && mapped.as_os_str().as_bytes().strip_suffix(b" (deleted)")
                    == Some(installed.as_os_str().as_bytes())
            {
                return Ok(());
            }
            Err(foreign_executable(expected))
        }

        pub(super) fn terminate(&self, force: bool) -> io::Result<()> {
            let signal = if force { Signal::KILL } else { Signal::TERM };
            Ok(pidfd_send_signal(&self.handle, signal)?)
        }

        pub(super) fn wait(&self, timeout: Duration) -> io::Result<bool> {
            let mut entry = [PollFd::new(&self.handle, PollFlags::IN)];
            let duration = Timespec::try_from(timeout).map_err(io::Error::other)?;
            Ok(poll(&mut entry, Some(&duration))? > 0)
        }
    }
}

#[cfg(target_os = "macos")]
mod platform {
    use super::*;
    use std::ffi::{CStr, OsStr, c_void};
    use std::os::unix::ffi::OsStrExt;
    use std::path::PathBuf;
    use std::time::Instant;

    pub(super) fn wildcard_port_in_use(endpoint: SocketAddr) -> io::Result<bool> {
        use rustix::net::{AddressFamily, SocketType, bind, socket};

        let family = if endpoint.is_ipv4() {
            AddressFamily::INET
        } else {
            AddressFamily::INET6
        };
        let probe = socket(family, SocketType::STREAM, None)?;
        match bind(&probe, &endpoint) {
            Ok(()) => Ok(false),
            Err(rustix::io::Errno::ADDRINUSE) => Ok(true),
            Err(error) => Err(error.into()),
        }
    }

    // Darwin's audit_token_t and TASK_AUDIT_TOKEN as specified in mach/message.h
    // and mach/task_info.h. libproc checks the token's PID generation on each call.
    #[repr(C)]
    #[derive(Default)]
    struct AuditToken([u32; 8]);

    unsafe extern "C" {
        fn task_name_for_pid(caller: u32, pid: i32, task_name: *mut u32) -> i32;
        fn mach_port_deallocate(caller: u32, name: u32) -> i32;
    }

    type PidPath = unsafe extern "C" fn(*const AuditToken, *mut u8, u32) -> i32;
    type SendSignal = unsafe extern "C" fn(*const AuditToken, i32) -> i32;

    // Intel deployment targets may predate token-aware libproc. Resolve these
    // only for CLI process control so ordinary server startup still works there.
    fn libproc_symbol(name: &CStr) -> io::Result<*mut c_void> {
        let symbol = unsafe { libc::dlsym(libc::RTLD_DEFAULT, name.as_ptr()) };
        if symbol.is_null() {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!(
                    "this macOS lacks {}; stop the HTTP process through its owner",
                    name.to_string_lossy()
                ),
            ));
        }
        Ok(symbol)
    }

    pub(super) struct Process {
        token: AuditToken,
        pid_path: PidPath,
        send_signal: SendSignal,
    }

    impl Process {
        pub(super) fn open(pid: u32) -> io::Result<Self> {
            let pid = i32::try_from(pid).map_err(io::Error::other)?;
            #[allow(deprecated)]
            let caller = unsafe { libc::mach_task_self() };
            let mut name = 0;
            // SAFETY: name is initialized by the kernel on success. Its send right
            // is released as soon as the kernel-supplied token has been copied.
            let status = unsafe { task_name_for_pid(caller, pid, &mut name) };
            if status != 0 {
                return Err(io::Error::other(format!(
                    "cannot inspect process {pid} (Mach error {status})"
                )));
            }
            let mut token = AuditToken::default();
            let mut count = token.0.len() as u32;
            let status =
                unsafe { libc::task_info(name, 15, token.0.as_mut_ptr().cast(), &mut count) };
            unsafe { mach_port_deallocate(caller, name) };
            if status != 0 || count as usize != token.0.len() {
                return Err(io::Error::other(format!(
                    "cannot obtain process {pid} audit token (Mach error {status})"
                )));
            }
            // SAFETY: both symbols' signatures match libproc.h, and the system
            // library remains loaded for the lifetime of the process.
            let pid_path = unsafe {
                std::mem::transmute::<*mut c_void, PidPath>(libproc_symbol(
                    c"proc_pidpath_audittoken",
                )?)
            };
            let send_signal = unsafe {
                std::mem::transmute::<*mut c_void, SendSignal>(libproc_symbol(
                    c"proc_signal_with_audittoken",
                )?)
            };
            Ok(Self {
                token,
                pid_path,
                send_signal,
            })
        }

        fn executable(&self) -> io::Result<PathBuf> {
            let mut path = [0u8; libc::PROC_PIDPATHINFO_MAXSIZE as usize];
            // SAFETY: libproc writes at most the provided buffer length.
            let len = unsafe { (self.pid_path)(&self.token, path.as_mut_ptr(), path.len() as u32) };
            if len <= 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(PathBuf::from(OsStr::from_bytes(&path[..len as usize])))
        }

        pub(super) fn verify_executable(&self, expected: &Path) -> io::Result<()> {
            if self.executable()? != expected.canonicalize()? {
                return Err(foreign_executable(expected));
            }
            Ok(())
        }

        pub(super) fn terminate(&self, force: bool) -> io::Result<()> {
            let signal = if force { libc::SIGKILL } else { libc::SIGTERM };
            let error = unsafe { (self.send_signal)(&self.token, signal) };
            if error == 0 {
                Ok(())
            } else {
                Err(io::Error::from_raw_os_error(error))
            }
        }

        pub(super) fn wait(&self, timeout: Duration) -> io::Result<bool> {
            let deadline = Instant::now() + timeout;
            loop {
                match self.executable() {
                    Err(err) if err.raw_os_error() == Some(libc::ESRCH) => return Ok(true),
                    Err(err) => return Err(err),
                    Ok(_) if Instant::now() >= deadline => return Ok(false),
                    Ok(_) => std::thread::sleep(Duration::from_millis(50)),
                }
            }
        }
    }
}

#[cfg(windows)]
mod platform {
    use super::*;
    use std::net::{Ipv4Addr, Ipv6Addr, SocketAddrV6};
    use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
    use windows_sys::Win32::Foundation::{
        ERROR_INSUFFICIENT_BUFFER, HANDLE, NO_ERROR, WAIT_FAILED, WAIT_OBJECT_0, WAIT_TIMEOUT,
    };
    use windows_sys::Win32::NetworkManagement::IpHelper::{
        GetExtendedTcpTable, MIB_TCP6ROW_OWNER_PID, MIB_TCPROW_OWNER_PID,
        TCP_TABLE_OWNER_PID_LISTENER,
    };
    use windows_sys::Win32::Networking::WinSock::{AF_INET, AF_INET6};
    use windows_sys::Win32::System::Threading::{
        GetCurrentProcess, OpenProcess, PROCESS_NAME_NATIVE, PROCESS_QUERY_LIMITED_INFORMATION,
        PROCESS_SYNCHRONIZE, PROCESS_TERMINATE, QueryFullProcessImageNameW, TerminateProcess,
        WaitForSingleObject,
    };

    pub(super) fn platform_listener_pids(endpoint: SocketAddr) -> io::Result<Vec<u32>> {
        let mut pids = Vec::new();
        for (address, pid) in tcp_listeners(endpoint.is_ipv4())? {
            collect_owner(&mut pids, endpoint, address, pid)?;
        }
        Ok(pids)
    }

    pub(super) fn has_overlapping_listener(endpoint: SocketAddr) -> io::Result<bool> {
        Ok(tcp_listeners(endpoint.is_ipv4())?
            .into_iter()
            .any(|(bound, _)| endpoints_overlap(endpoint, bound)))
    }

    fn tcp_listeners(ipv4: bool) -> io::Result<Vec<(SocketAddr, u32)>> {
        let family = if ipv4 { AF_INET } else { AF_INET6 };
        let mut buffer: Vec<u32> = Vec::new();
        let mut size = 0;
        loop {
            let data = if buffer.is_empty() {
                std::ptr::null_mut()
            } else {
                buffer.as_mut_ptr().cast()
            };
            // Both MIB table layouts contain DWORD-aligned rows after a DWORD row count.
            let status = unsafe {
                GetExtendedTcpTable(
                    data,
                    &mut size,
                    0,
                    family.into(),
                    TCP_TABLE_OWNER_PID_LISTENER,
                    0,
                )
            };
            match status {
                ERROR_INSUFFICIENT_BUFFER => buffer.resize((size as usize).div_ceil(4), 0),
                NO_ERROR => break,
                code => return Err(io::Error::from_raw_os_error(code as i32)),
            }
        }
        let (&count, entries) = buffer
            .split_first()
            .ok_or_else(|| io::Error::other("empty TCP listener table"))?;
        let entry_size = if ipv4 {
            size_of::<MIB_TCPROW_OWNER_PID>()
        } else {
            size_of::<MIB_TCP6ROW_OWNER_PID>()
        };
        if count as usize > size_of_val(entries) / entry_size {
            return Err(io::Error::other("truncated TCP listener table"));
        }
        let mut listeners = Vec::new();
        for raw in entries.chunks_exact(entry_size / 4).take(count as usize) {
            // SAFETY: rows consist of DWORDs and byte arrays with alignment four,
            // and each chunk has exactly the size of the selected row layout.
            let (addr, pid) = if ipv4 {
                let row = unsafe { &*raw.as_ptr().cast::<MIB_TCPROW_OWNER_PID>() };
                let ip = Ipv4Addr::from(row.dwLocalAddr.to_ne_bytes());
                (
                    SocketAddr::new(ip.into(), u16::from_be(row.dwLocalPort as u16)),
                    row.dwOwningPid,
                )
            } else {
                let row = unsafe { &*raw.as_ptr().cast::<MIB_TCP6ROW_OWNER_PID>() };
                let ip = Ipv6Addr::from(row.ucLocalAddr);
                let addr = SocketAddrV6::new(
                    ip,
                    u16::from_be(row.dwLocalPort as u16),
                    0,
                    u32::from_be(row.dwLocalScopeId),
                );
                (addr.into(), row.dwOwningPid)
            };
            listeners.push((addr, pid));
        }
        Ok(listeners)
    }

    fn image_name(handle: HANDLE) -> io::Result<Vec<u16>> {
        let mut name = vec![0; 32_768];
        let mut length = name.len() as u32;
        if unsafe {
            QueryFullProcessImageNameW(handle, PROCESS_NAME_NATIVE, name.as_mut_ptr(), &mut length)
        } == 0
        {
            return Err(io::Error::last_os_error());
        }
        name.truncate(length as usize);
        Ok(name)
    }

    pub(super) struct Process(OwnedHandle);

    impl Process {
        pub(super) fn open(pid: u32) -> io::Result<Self> {
            let handle = unsafe {
                OpenProcess(
                    PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_TERMINATE | PROCESS_SYNCHRONIZE,
                    0,
                    pid,
                )
            };
            if handle.is_null() {
                return Err(io::Error::last_os_error());
            }
            // SAFETY: OpenProcess returned an owned handle that must be closed once.
            Ok(Self(unsafe { OwnedHandle::from_raw_handle(handle) }))
        }

        pub(super) fn verify_executable(&self, expected: &Path) -> io::Result<()> {
            // Compare image names in the same native namespace without resolving
            // the foreign process's path through a potentially replaced symlink.
            if image_name(self.0.as_raw_handle())? != image_name(unsafe { GetCurrentProcess() })? {
                return Err(foreign_executable(expected));
            }
            Ok(())
        }

        pub(super) fn terminate(&self, _force: bool) -> io::Result<()> {
            if unsafe { TerminateProcess(self.0.as_raw_handle(), 0) } == 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        }

        pub(super) fn wait(&self, timeout: Duration) -> io::Result<bool> {
            let millis = u32::try_from(timeout.as_millis()).map_err(io::Error::other)?;
            match unsafe { WaitForSingleObject(self.0.as_raw_handle(), millis) } {
                WAIT_OBJECT_0 => Ok(true),
                WAIT_TIMEOUT => Ok(false),
                WAIT_FAILED => Err(io::Error::last_os_error()),
                other => Err(io::Error::other(format!(
                    "unexpected process wait result {other}"
                ))),
            }
        }
    }
}

#[cfg(all(test, any(target_os = "linux", target_os = "macos", windows)))]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader};
    use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, TcpListener, TcpStream};
    use std::process::{Child, Command, Stdio};

    const LISTENER_READY: &str = "test listener address: ";

    #[test]
    fn endpoint_overlap_keeps_exact_owner_selection_separate_from_wildcards() {
        for (requested, bound, overlap) in [
            ("127.0.0.1:12345", "127.0.0.1:12345", true),
            ("127.0.0.1:12345", "127.0.0.2:12345", false),
            ("127.0.0.1:12345", "127.0.0.1:12346", false),
            ("127.0.0.1:12345", "0.0.0.0:12345", true),
            ("0.0.0.0:12345", "127.0.0.1:12345", true),
            ("[::1]:12345", "[::]:12345", true),
            ("[::]:12345", "[::1]:12345", true),
            ("127.0.0.1:12345", "[::]:12345", false),
        ] {
            let requested = requested.parse().unwrap();
            let bound = bound.parse().unwrap();
            assert_eq!(endpoints_overlap(requested, bound), overlap);
            let mut pids = Vec::new();
            let selected = collect_owner(&mut pids, requested, bound, 42);
            if requested == bound {
                assert!(selected.is_ok());
                assert_eq!(pids, [42]);
            } else {
                assert_eq!(selected.is_err(), overlap);
                assert!(pids.is_empty());
            }
        }
    }

    #[test]
    fn checked_tcp_bind_accepts_a_free_endpoint_without_a_runtime() {
        for endpoint in ["127.0.0.1:0", "[::1]:0"] {
            let socket = bind_available_tcp(endpoint.parse().unwrap()).unwrap();
            let bound = socket.local_addr().unwrap();
            assert_ne!(bound.port(), 0);
        }
    }

    #[test]
    fn checked_tcp_bind_rejects_a_live_wildcard_listener_without_sending_bytes() {
        for endpoint in ["0.0.0.0:0", "[::]:0"] {
            let listener = TcpListener::bind(endpoint).unwrap();
            let bound = listener.local_addr().unwrap();
            let loopback = if bound.is_ipv4() {
                IpAddr::V4(Ipv4Addr::LOCALHOST)
            } else {
                IpAddr::V6(Ipv6Addr::LOCALHOST)
            };
            assert_eq!(
                bind_available_tcp(SocketAddr::new(loopback, bound.port()))
                    .unwrap_err()
                    .kind(),
                io::ErrorKind::AddrInUse
            );
            listener.set_nonblocking(true).unwrap();
            #[cfg(unix)]
            {
                let (mut probe, _) = listener.accept().unwrap();
                probe.set_nonblocking(false).unwrap();
                probe
                    .set_read_timeout(Some(Duration::from_secs(2)))
                    .unwrap();
                assert_eq!(std::io::Read::read(&mut probe, &mut [0]).unwrap(), 0);
            }
            #[cfg(windows)]
            assert_eq!(
                listener.accept().unwrap_err().kind(),
                io::ErrorKind::WouldBlock
            );
        }
    }

    #[tokio::test]
    async fn checked_tcp_bind_allows_immediate_restart_after_server_active_close() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        for endpoint in ["127.0.0.1:0", "[::1]:0"] {
            tokio::time::timeout(Duration::from_secs(2), async {
                let listener = bind_available_tcp(endpoint.parse().unwrap())
                    .unwrap()
                    .listen(128)
                    .unwrap();
                let bound = listener.local_addr().unwrap();
                let mut client = tokio::net::TcpStream::connect(bound).await.unwrap();
                let (mut server, _) = listener.accept().await.unwrap();
                server.shutdown().await.unwrap();
                assert_eq!(client.read_to_end(&mut Vec::new()).await.unwrap(), 0);
                drop(client);
                assert_eq!(server.read_to_end(&mut Vec::new()).await.unwrap(), 0);
                drop(server);
                drop(listener);
                let restarted = bind_available_tcp(bound).unwrap().listen(128).unwrap();
                assert_eq!(restarted.local_addr().unwrap(), bound);
            })
            .await
            .expect("controlled active-close restart timed out");
        }
    }

    #[test]
    fn process_listener_fixture() {
        let Some(addr) = std::env::var_os("OBSIDIAN_TEST_PROCESS_LISTENER") else {
            return;
        };
        #[cfg(unix)]
        {
            // SIGTERM wakes a controlled listener handoff, or is ignored until forced.
            unsafe { libc::signal(libc::SIGTERM, on_term as *const () as libc::sighandler_t) };
        }
        let listener = TcpListener::bind(addr.to_str().unwrap()).unwrap();
        eprintln!("{LISTENER_READY}{}", listener.local_addr().unwrap());
        #[cfg(unix)]
        if let Some(path) = std::env::var_os("OBSIDIAN_TEST_LISTENER_HANDOFF") {
            let directory = Path::new(&path);
            wait_until(|| TERM_RECEIVED.load(std::sync::atomic::Ordering::Relaxed));
            drop(listener);
            std::fs::write(directory.join("released"), []).unwrap();
            wait_until(|| directory.join("replacement-ready").is_file());
            return;
        }
        std::thread::sleep(Duration::from_secs(60));
    }

    #[cfg(unix)]
    static TERM_RECEIVED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

    #[cfg(unix)]
    extern "C" fn on_term(_: libc::c_int) {
        TERM_RECEIVED.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    #[cfg(unix)]
    fn wait_until(ready: impl Fn() -> bool) {
        let deadline = std::time::Instant::now() + Duration::from_secs(15);
        while !ready() {
            assert!(
                std::time::Instant::now() < deadline,
                "listener handoff timed out"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    struct ChildListener(Child, SocketAddr);

    fn listener_address(stderr: impl BufRead) -> io::Result<SocketAddr> {
        for line in stderr.lines() {
            let line = line?;
            if let Some(address) = line.strip_prefix(LISTENER_READY) {
                return address.parse().map_err(io::Error::other);
            }
            eprintln!("{line}");
        }
        Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "listener exited before reporting its address",
        ))
    }

    impl ChildListener {
        fn start(bind: &str, handoff: Option<&Path>) -> Self {
            let mut command = Command::new(std::env::current_exe().unwrap());
            if let Some(directory) = handoff {
                command.env("OBSIDIAN_TEST_LISTENER_HANDOFF", directory);
            }
            let mut child = command
                .args([
                    "--exact",
                    "http_process::tests::process_listener_fixture",
                    "--nocapture",
                ])
                .env("OBSIDIAN_TEST_PROCESS_LISTENER", bind)
                .stdout(Stdio::null())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap();
            let stderr = BufReader::new(child.stderr.take().unwrap());
            let (send, receive) = std::sync::mpsc::channel();
            std::thread::spawn(move || {
                let _ = send.send(listener_address(stderr));
            });
            match receive
                .recv_timeout(Duration::from_secs(10))
                .map_err(io::Error::other)
                .and_then(|address| address)
            {
                Ok(address) => Self(child, address),
                Err(error) => {
                    let _ = child.kill();
                    let exit = child.wait();
                    panic!("listener fixture failed to start: {error}, child status {exit:?}");
                }
            }
        }
    }

    #[test]
    fn listener_readiness_ignores_library_startup_diagnostics() {
        let stderr = format!("runtime warning\n127.0.0.1:1\n{LISTENER_READY}127.0.0.1:37901\n");
        assert_eq!(
            listener_address(stderr.as_bytes()).unwrap(),
            SocketAddr::from(([127, 0, 0, 1], 37901))
        );
        assert!(listener_address(b"runtime warning\n".as_slice()).is_err());
    }

    impl Drop for ChildListener {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }

    #[cfg(unix)]
    #[test]
    fn wildcard_inspection_sees_specific_ipv4_and_ipv6_listeners() {
        for address in ["127.0.0.1:0", "[::1]:0"] {
            let mut observed_release = false;
            for _ in 0..8 {
                let listener = TcpListener::bind(address).unwrap();
                let bound = listener.local_addr().unwrap();
                let wildcard = if bound.is_ipv4() {
                    IpAddr::V4(Ipv4Addr::UNSPECIFIED)
                } else {
                    IpAddr::V6(Ipv6Addr::UNSPECIFIED)
                };
                let endpoint = SocketAddr::new(wildcard, bound.port());
                assert!(platform::wildcard_port_in_use(endpoint).unwrap());
                drop(listener);
                if !platform::wildcard_port_in_use(endpoint).unwrap() {
                    observed_release = true;
                    break;
                }
                // Another test may reuse a released ephemeral port. Retry with
                // another owned endpoint, still requiring a real nonzero release.
            }
            assert!(
                observed_release,
                "no released endpoint observed for {address}"
            );
        }
    }

    #[test]
    fn detached_startup_requires_the_spawned_child_to_own_the_endpoint() {
        let mut child = ChildListener::start("127.0.0.1:0", None);
        let mut other = ChildListener::start("127.0.0.1:0", None);
        let error = crate::wait_for_http_child(&mut child.0, other.1).unwrap_err();
        assert!(
            error.to_string().contains(&format!("PID {}", other.0.id())),
            "{error}"
        );
        assert!(child.0.try_wait().unwrap().is_none());
        assert!(other.0.try_wait().unwrap().is_none());
    }

    #[cfg(unix)]
    #[test]
    fn a_listener_that_ignores_termination_is_stopped_through_its_pinned_instance() {
        use std::os::unix::process::ExitStatusExt;

        let mut listener = ChildListener::start("127.0.0.1:0", None);
        assert!(stop(listener.1, &std::env::current_exe().unwrap()).unwrap());
        assert_eq!(listener.0.wait().unwrap().signal(), Some(libc::SIGKILL));
    }

    #[cfg(unix)]
    #[test]
    fn a_replacement_that_binds_while_the_old_instance_is_draining_survives() {
        let handoff = tempfile::tempdir().unwrap();
        let old = ChildListener::start("127.0.0.1:0", Some(handoff.path()));
        let endpoint = old.1;
        let stopping =
            std::thread::spawn(move || stop(endpoint, &std::env::current_exe().unwrap()));
        wait_until(|| handoff.path().join("released").is_file());
        let mut replacement = ChildListener::start(&endpoint.to_string(), None);
        std::fs::write(handoff.path().join("replacement-ready"), []).unwrap();

        let error = stopping.join().unwrap().unwrap_err();
        assert!(
            error
                .to_string()
                .contains(&format!("process {} now owns", replacement.0.id())),
            "{error}"
        );
        assert!(replacement.0.try_wait().unwrap().is_none());
        TcpStream::connect(endpoint).unwrap();
    }

    #[test]
    fn replacing_a_captured_listener_cannot_stop_its_successor() {
        let old = ChildListener::start("127.0.0.1:0", None);
        let endpoint = old.1;
        let pid = listener_pid(endpoint).unwrap().unwrap();
        let process = platform::Process::open(pid).unwrap();
        let exe = std::env::current_exe().unwrap();
        process.verify_executable(&exe).unwrap();
        drop(old);
        let mut replacement = ChildListener::start(&endpoint.to_string(), None);
        assert!(stop_instance(endpoint, pid, &process, &exe).is_err());
        assert!(replacement.0.try_wait().unwrap().is_none());
        TcpStream::connect(endpoint).unwrap();
    }
}
