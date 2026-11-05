//! Exclusive ownership of a filesystem-backed Unix IPC endpoint.

use std::collections::VecDeque;
use std::ffi::{OsStr, OsString};
use std::fs::File;
use std::io;
#[cfg(target_os = "macos")]
use std::os::fd::AsRawFd;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::MetadataExt;
use std::os::unix::net::UnixDatagram;
use std::path::{Path, PathBuf};

use rustix::fs::{
    AtFlags, FileType, Mode, OFlags, Stat, mkdirat, open, openat, readlinkat, statat, unlinkat,
};
use rustix::process::geteuid;
use tokio::net::{UnixListener, UnixSocket};

pub(super) struct UnixEndpoint {
    directory: File,
    name: OsString,
    path: PathBuf,
    _lock: File,
    bound: Option<Stat>,
}

impl UnixEndpoint {
    pub fn bind(requested: &Path) -> io::Result<(Self, UnixListener)> {
        let absolute = std::path::absolute(requested)?;
        let name = absolute.file_name().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "socket path needs a filename")
        })?;
        let parent = absolute
            .parent()
            .ok_or_else(|| io::Error::other("socket path has no parent"))?;
        let directory = private_parent(parent)?;
        let mut lock_name = name.to_owned();
        lock_name.push(".lock");
        let lock = File::from(openat(
            &directory,
            &lock_name,
            OFlags::CREATE
                | OFlags::RDWR
                | OFlags::CLOEXEC
                | OFlags::NOFOLLOW
                | OFlags::NONBLOCK
                | OFlags::NOCTTY,
            Mode::RUSR | Mode::WUSR,
        )?);
        let metadata = lock.metadata()?;
        if !metadata.is_file()
            || metadata.uid() != geteuid().as_raw()
            || metadata.mode() & 0o077 != 0
            || metadata.nlink() != 1
        {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "IPC lock must be a private, owner-controlled regular file",
            ));
        }
        #[cfg(target_os = "macos")]
        check_acl(&lock, &absolute)?;
        fs2::FileExt::try_lock_exclusive(&lock)?;
        // Every new owner locks the same inode. Removing the lock file would let
        // another opener race this process on a different inode.
        let mut owner = Self {
            directory,
            name: name.to_owned(),
            path: requested.to_owned(),
            _lock: lock,
            bound: None,
        };
        let socket = UnixSocket::new_stream()?;
        // On Linux bind copies the socket descriptor's mode to the filesystem
        // inode (subject to umask). The listener never needs an ambient chmod.
        #[cfg(target_os = "linux")]
        rustix::fs::fchmod(&socket, Mode::RUSR | Mode::WUSR)?;
        owner.bind_path(&socket)?;
        let bound = owner.entry()?;
        if FileType::from_raw_mode(bound.st_mode) != FileType::Socket
            || bound.st_uid != geteuid().as_raw()
        {
            return Err(io::Error::new(
                io::ErrorKind::AddrInUse,
                "socket entry changed during bind",
            ));
        }
        owner.bound = Some(bound);
        #[cfg(not(target_os = "linux"))]
        rustix::fs::chmodat(
            &owner.directory,
            &owner.name,
            Mode::RUSR | Mode::WUSR,
            AtFlags::SYMLINK_NOFOLLOW,
        )
        .map_err(io::Error::from)?;
        if owner.entry()?.st_mode & 0o777 != 0o600 {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "IPC socket requires owner read/write permissions",
            ));
        }
        let listener = socket.listen(128)?;
        Ok((owner, listener))
    }

    fn bind_path(&self, socket: &UnixSocket) -> io::Result<()> {
        let path = &self.path;
        let conflict = match socket.bind(path) {
            Ok(()) => return Ok(()),
            Err(err) if err.kind() == io::ErrorKind::AddrInUse => err,
            Err(err) => return Err(err),
        };
        let old = self.entry()?;
        if FileType::from_raw_mode(old.st_mode) != FileType::Socket
            || old.st_uid != geteuid().as_raw()
        {
            return Err(conflict);
        }
        // A stream connect can report ECONNREFUSED on a live Darwin socket
        // before listen or when its backlog is full. A datagram probe checks
        // the live peer's type without consuming a connection or backlog slot.
        let probe = UnixDatagram::unbound()?;
        probe.set_nonblocking(true)?;
        match probe.connect(path) {
            Err(error) if error.kind() == io::ErrorKind::ConnectionRefused => {}
            Err(error) if error.raw_os_error() != Some(libc::EPROTOTYPE) => return Err(error),
            _ => return Err(conflict),
        }
        if !self.unlink_if_same(&old)? {
            return Err(conflict);
        }
        socket.bind(path)
    }

    fn entry(&self) -> io::Result<Stat> {
        Ok(statat(
            &self.directory,
            &self.name,
            AtFlags::SYMLINK_NOFOLLOW,
        )?)
    }

    fn unlink_if_same(&self, expected: &Stat) -> io::Result<bool> {
        match self.entry() {
            Ok(entry)
                if entry.st_dev == expected.st_dev
                    && entry.st_ino == expected.st_ino
                    && FileType::from_raw_mode(entry.st_mode) == FileType::Socket =>
            {
                unlinkat(&self.directory, &self.name, AtFlags::empty())?;
                Ok(true)
            }
            Ok(_) => Ok(false),
            Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(false),
            Err(err) => Err(err),
        }
    }

    pub fn close(mut self) -> io::Result<()> {
        self.remove_bound()
    }

    fn remove_bound(&mut self) -> io::Result<()> {
        if let Some(bound) = self.bound.take() {
            self.unlink_if_same(&bound)?;
        }
        Ok(())
    }
}

impl Drop for UnixEndpoint {
    fn drop(&mut self) {
        if let Err(err) = self.remove_bound() {
            tracing::warn!(endpoint = %self.path.display(), error = %err, "failed to remove daemon socket");
        }
    }
}

// bind/connect resolve the configured name again. Check every component
// (including the spelling of symlink targets) against replacement by another
// user, and keep the directory fd for all destructive operations. Using the
// original address preserves short symlinks to directories beyond SUN_LEN.
fn private_parent(path: &Path) -> io::Result<File> {
    let uid = geteuid().as_raw();
    // Resolving or binding an address needs search access, even when the
    // directory owner has deliberately disabled directory listings.
    #[cfg(target_os = "linux")]
    let access = OFlags::PATH;
    #[cfg(target_vendor = "apple")]
    let access = OFlags::from_bits_retain(libc::O_SEARCH as u32);
    #[cfg(not(any(target_os = "linux", target_vendor = "apple")))]
    let access = OFlags::RDONLY;
    let flags = access | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC;
    let mut components: VecDeque<_> = path
        .components()
        .skip(1) // The requested path is absolute.
        .map(|c| c.as_os_str().to_owned())
        .collect();
    let root = File::from(open("/", flags, Mode::empty())?);
    check_ancestor(&root, path)?;
    let mut directory = root.try_clone()?;
    let mut links = 0;
    while let Some(name) = components.pop_front() {
        if name == "/" {
            directory = root.try_clone()?;
            continue;
        }
        let stat = match statat(&directory, &name, AtFlags::SYMLINK_NOFOLLOW) {
            Ok(entry) => entry,
            Err(rustix::io::Errno::NOENT) => {
                match mkdirat(&directory, &name, Mode::RWXU) {
                    Ok(()) | Err(rustix::io::Errno::EXIST) => {}
                    Err(error) => return Err(error.into()),
                }
                statat(&directory, &name, AtFlags::SYMLINK_NOFOLLOW)?
            }
            Err(error) => return Err(error.into()),
        };
        if stat.st_uid != uid && stat.st_uid != 0 {
            return Err(untrusted_parent(path));
        }
        if FileType::from_raw_mode(stat.st_mode) == FileType::Symlink {
            links += 1;
            if links > 40 {
                return Err(rustix::io::Errno::LOOP.into());
            }
            // A dangling link is not permission to create its missing target.
            statat(&directory, &name, AtFlags::empty())?;
            #[cfg(target_os = "macos")]
            {
                let link = File::from(openat(
                    &directory,
                    &name,
                    OFlags::from_bits_retain(libc::O_SYMLINK as u32) | OFlags::CLOEXEC,
                    Mode::empty(),
                )?);
                check_acl(&link, path)?;
            }
            let target = readlinkat(&directory, &name, Vec::new())?;
            let target = Path::new(OsStr::from_bytes(target.to_bytes()));
            for component in target.components().rev() {
                components.push_front(component.as_os_str().to_owned());
            }
            continue;
        }
        directory = File::from(openat(&directory, &name, flags, Mode::empty())?);
        check_ancestor(&directory, path)?;
    }
    let meta = directory.metadata()?;
    if meta.uid() != uid || meta.mode() & 0o022 != 0 {
        return Err(untrusted_parent(path));
    }
    Ok(directory)
}

fn check_ancestor(directory: &File, path: &Path) -> io::Result<()> {
    let meta = directory.metadata()?;
    // A sticky ancestor (e.g. /tmp) protects its owner-controlled child.
    if (meta.uid() != geteuid().as_raw() && meta.uid() != 0)
        || (meta.mode() & 0o022 != 0 && meta.mode() & 0o1000 == 0)
    {
        return Err(untrusted_parent(path));
    }
    #[cfg(target_os = "macos")]
    check_acl(directory, path)?;
    Ok(())
}

#[cfg(target_os = "macos")]
fn check_acl(file: &File, path: &Path) -> io::Result<()> {
    unsafe extern "C" {
        fn obsidian_ipc_foreign_write_acl(fd: libc::c_int, owner: libc::uid_t) -> libc::c_int;
    }
    // The descriptor stays open for the entire native inspection.
    match unsafe { obsidian_ipc_foreign_write_acl(file.as_raw_fd(), geteuid().as_raw()) } {
        0 => Ok(()),
        1 => Err(untrusted_parent(path)),
        _ => Err(io::Error::last_os_error()),
    }
}

fn untrusted_parent(path: &Path) -> io::Error {
    io::Error::new(
        io::ErrorKind::PermissionDenied,
        format!(
            "IPC path '{}' must be owner-controlled and protected from other writers",
            path.display()
        ),
    )
}
