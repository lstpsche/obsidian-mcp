//! A directory capability retained for the lifetime of a vault or data root.

use std::ffi::{OsStr, OsString};
use std::fs::File;
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use cap_fs_ext::{DirExt, FollowSymlinks, MetadataExt, OpenOptionsFollowExt};
use cap_std::fs::{Dir, Metadata, OpenOptions};

use super::super::path::{self, ResolvedPath};
use super::dir_entries;
use crate::error::{VaultError, VaultResult};

#[derive(Clone, Debug)]
pub struct VaultRoot(Arc<RootDirectory>, bool);

#[derive(Debug)]
struct RootDirectory {
    dir: Dir,
    path: PathBuf,
}

impl VaultRoot {
    /// Open the directory designated by the owner. All descendant I/O is
    /// relative to this handle, including after the caller's pathname changes.
    pub fn open(root: &Path) -> VaultResult<Self> {
        Self::from_directory(
            Dir::open_ambient_dir(root, cap_std::ambient_authority())?,
            root,
        )
    }

    pub(super) fn from_directory(dir: Dir, root: &Path) -> VaultResult<Self> {
        // Derive the namespace identity from the same handle that will perform
        // I/O, even if the operator's root alias changes while opening it.
        #[cfg(unix)]
        if dir.dir_metadata()?.nlink() == 0 {
            return Err(VaultError::InvalidPath("unlinked vault root".into()));
        }
        let path = match handle_path(&dir.try_clone()?.into_std_file())? {
            Some(path) if path.is_absolute() => path,
            Some(_) => return Err(VaultError::InvalidPath("invalid native root path".into())),
            None => {
                let path = root.canonicalize()?;
                let named = Dir::open_ambient_dir(&path, cap_std::ambient_authority())?;
                if !same_object(&dir.dir_metadata()?, &named.dir_metadata()?) {
                    return Err(VaultError::InvalidPath(format!(
                        "root changed while opening {}",
                        root.display()
                    )));
                }
                path
            }
        };
        Ok(Self(Arc::new(RootDirectory { dir, path }), false))
    }

    /// Restrict content operations without changing the retained directory owner.
    pub(crate) fn restricted_content(&self) -> Self {
        Self(Arc::clone(&self.0), true)
    }

    /// Validate a delegated content file without reading its bytes.
    pub fn validate_access_file(&self, path: &Path) -> VaultResult<()> {
        self.restricted_content().open_file(path).map(|_| ())
    }

    /// Validate a physical source key already selected by authorization.
    pub(crate) fn validate_access_file_exact(&self, path: &Path) -> VaultResult<()> {
        self.restricted_content().open_exact(path).map(|_| ())
    }

    /// Physical aliases outside a delegated namespace are hidden resources.
    /// Preserve the original native diagnostics for unrestricted owners.
    pub(super) fn content_result<T>(
        &self,
        requested: &Path,
        result: VaultResult<T>,
    ) -> VaultResult<T> {
        result.map_err(|error| {
            if self.content_is_restricted()
                && matches!(
                    error,
                    VaultError::OutsideVault(_) | VaultError::InvalidPath(_)
                )
            {
                VaultError::AccessDenied {
                    path: requested.to_owned(),
                    visible: false,
                }
            } else {
                error
            }
        })
    }

    pub(crate) fn content_is_restricted(&self) -> bool {
        self.1
    }

    pub(crate) fn check_content_file(&self, file: &File, path: &Path) -> VaultResult<()> {
        if self.1 {
            require_single_link(file, path)?;
        }
        Ok(())
    }

    pub fn path(&self) -> &Path {
        &self.0.path
    }

    pub(crate) fn same_root(&self, other: &Self) -> io::Result<bool> {
        self.same_directory(&other.0.dir)
    }

    pub(crate) fn same_directory(&self, directory: &Dir) -> io::Result<bool> {
        Ok(same_object(
            &self.0.dir.dir_metadata()?,
            &directory.dir_metadata()?,
        ))
    }

    pub fn resolve_existing(&self, relative: &Path) -> VaultResult<ResolvedPath> {
        if self.content_is_restricted() {
            path::normalize_relative(relative)?;
        }
        self.content_result(relative, self.resolve(relative, false, true))
    }

    pub fn resolve_for_write(&self, relative: &Path) -> VaultResult<ResolvedPath> {
        if self.content_is_restricted() {
            path::normalize_relative(relative)?;
        }
        self.content_result(relative, self.resolve(relative, true, true))
    }

    /// Recover native entry spellings without substituting symlink targets into
    /// the logical key used for path grants.
    pub(crate) fn resolve_access_existing(
        &self,
        relative: &Path,
    ) -> VaultResult<(PathBuf, ResolvedPath)> {
        if self.content_is_restricted() {
            path::normalize_relative(relative)?;
        }
        self.content_result(relative, self.resolve_keys(relative, false, true))
    }

    pub(crate) fn resolve_access_for_write(
        &self,
        relative: &Path,
    ) -> VaultResult<(PathBuf, ResolvedPath)> {
        if self.content_is_restricted() {
            path::normalize_relative(relative)?;
        }
        self.content_result(relative, self.resolve_keys(relative, true, true))
    }

    fn resolve(
        &self,
        relative: &Path,
        allow_missing: bool,
        follow_leaf: bool,
    ) -> VaultResult<ResolvedPath> {
        self.resolve_keys(relative, allow_missing, follow_leaf)
            .map(|(_, resolved)| resolved)
    }

    /// Resolve user aliases to physical vault keys. This translates names only.
    /// Every consuming operation opens each physical component without following
    /// links, so replacing a discovered name cannot redirect the operation.
    fn resolve_keys(
        &self,
        relative: &Path,
        allow_missing: bool,
        follow_leaf: bool,
    ) -> VaultResult<(PathBuf, ResolvedPath)> {
        let normalized = path::normalize_relative(relative)?;
        let mut parts = normalized.iter().peekable();
        let mut physical = PathBuf::new();
        let mut logical = PathBuf::new();
        let mut directory = self.0.dir.try_clone()?;
        while let Some(requested) = parts.next() {
            let Some((name, metadata)) = lookup_component(&directory, requested)? else {
                if !allow_missing {
                    return Err(VaultError::NoteNotFound(relative.to_owned()));
                }
                logical.push(requested);
                logical.extend(parts.clone());
                physical.push(requested);
                physical.extend(parts);
                break;
            };
            logical.push(&name);
            let last = parts.peek().is_none();
            if is_link(&metadata) && (!last || follow_leaf) {
                // Absolute in-vault symlinks are supported. Native canonicalization
                // translates their target (including intervening links) into a
                // relative spelling. Its ambient result is never opened directly.
                physical = self
                    .link_target(&physical.join(&name), relative)?
                    .ok_or_else(|| {
                        VaultError::InvalidPath(format!(
                            "dangling vault link: {}",
                            relative.display()
                        ))
                    })?;
                if !last {
                    directory = self.open_dir_exact(&physical)?;
                }
            } else {
                physical.push(&name);
                if !last {
                    directory = open_child_dir(&directory, &name, &physical)?;
                }
            }
        }
        let relative = path::portable_native(&physical)?;
        Ok((
            path::portable_native(&logical)?,
            ResolvedPath {
                absolute: self.path().join(&relative),
                relative,
            },
        ))
    }

    /// Validate an existing link target. Unlink can also operate on a broken leaf.
    pub(super) fn link_target(
        &self,
        entry: &Path,
        requested: &Path,
    ) -> VaultResult<Option<PathBuf>> {
        let target = match self.path().join(entry).canonicalize() {
            Ok(target) => target,
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        path::relative_from_absolute(self.path(), &target)
            .map(Some)
            .map_err(|error| match error {
                VaultError::OutsideVault(_) => VaultError::OutsideVault(requested.to_owned()),
                other => other,
            })
    }

    /// Open a physical directory obtained from a confined walk. Intermediate
    /// directories and the leaf must each be a directory entry, never a symlink.
    pub fn open_dir_exact(&self, relative: &Path) -> VaultResult<Dir> {
        if self.content_is_restricted() {
            path::portable_native(relative)?;
        }
        self.content_result(relative, self.physical_dir(relative, false))
    }

    fn physical_dir(&self, relative: &Path, create: bool) -> VaultResult<Dir> {
        let exact = path::portable_native(relative)?;
        let mut dir = self.0.dir.try_clone()?;
        for name in exact.iter() {
            match open_child_dir(&dir, name, relative) {
                Err(VaultError::NoteNotFound(_)) if create => {
                    match super::create_child_dir(&dir, name) {
                        Ok(()) => (),
                        Err(e) if e.kind() == ErrorKind::AlreadyExists => (),
                        Err(e) => return Err(e.into()),
                    }
                    dir = open_child_dir(&dir, name, relative)?;
                }
                result => dir = result?,
            }
        }
        Ok(dir)
    }

    pub fn open_dir(&self, relative: &Path) -> VaultResult<Dir> {
        self.open_dir_exact(&self.resolve_existing(relative)?.relative)
    }

    pub fn create_dir_all(&self, relative: &Path) -> VaultResult<()> {
        self.physical_dir(&self.resolve_for_write(relative)?.relative, true)?;
        Ok(())
    }

    /// No-follow metadata of an exact entry. Symlink leaves are identifiable
    /// for watcher/index classification. An intermediate link is an error.
    pub fn entry_metadata_exact(&self, relative: &Path) -> VaultResult<Metadata> {
        if self.content_is_restricted() {
            path::portable_native(relative)?;
        }
        let result = (|| {
            if relative.as_os_str().is_empty() {
                return Ok(self.0.dir.dir_metadata()?);
            }
            let (parent, name) = self.parent(relative, false)?;
            parent
                .symlink_metadata(name)
                .map_err(super::map_not_found(relative))
        })();
        self.content_result(relative, result)
    }

    pub fn open_exact(&self, relative: &Path) -> VaultResult<File> {
        if self.content_is_restricted() {
            path::portable_native(relative)?;
        }
        let result = (|| {
            let (parent, name) = self.parent(relative, false)?;
            let mut options = OpenOptions::new();
            options.read(true);
            let file = open_regular(&parent, name, relative, options)?;
            self.check_content_file(&file, relative)?;
            #[cfg(test)]
            crate::test_helpers::count_io(&crate::test_helpers::EXACT_FILE_OPENS);
            Ok(file)
        })();
        self.content_result(relative, result)
    }

    /// Verify that a cached key still names this opened file with the same
    /// on-disk spelling. Native case/normalization aliases must never inherit
    /// a stale key's exclusion decision.
    pub(super) fn verify_observed(&self, path: &Path, file: &File) -> VaultResult<()> {
        let expected = path::portable_native(path)?;
        if let Some(actual) = handle_path(file)?
            && actual == self.path().join(&expected)
        {
            return Ok(());
        }

        // Some systems have no handle-to-path API. A hard-linked file can
        // also report a different link on a system that implements one.
        // Check each actual directory entry and the identity of the open file.
        let mut dir = self.0.dir.try_clone()?;
        let mut parts = expected.iter().peekable();
        while let Some(name) = parts.next() {
            let mut found = false;
            #[cfg(test)]
            crate::test_helpers::count_io(&crate::test_helpers::NAME_LOOKUP_SCANS);
            for entry in dir_entries(&dir)? {
                if entry?.file_name() == name {
                    found = true;
                    break;
                }
            }
            if !found {
                break;
            }
            if parts.peek().is_none() {
                let named = dir
                    .symlink_metadata(name)
                    .map_err(super::map_not_found(path))?;
                if !is_link(&named) && same_object(&named, &Metadata::from_file(file)?) {
                    return Ok(());
                }
            } else {
                dir = open_child_dir(&dir, name, path)?;
            }
        }
        Err(VaultError::NoteNotFound(path.to_owned()))
    }

    pub fn open_file(&self, relative: &Path) -> VaultResult<File> {
        self.open_exact(&self.resolve_existing(relative)?.relative)
    }

    pub(crate) fn parent<'p>(
        &self,
        relative: &'p Path,
        create: bool,
    ) -> VaultResult<(Dir, &'p OsStr)> {
        path::portable_native(relative)?;
        let parent = relative.parent().ok_or_else(|| {
            VaultError::InvalidPath("expected a file below the vault root".into())
        })?;
        let name = relative
            .file_name()
            .ok_or_else(|| VaultError::InvalidPath("expected a filename".into()))?;
        Ok((self.physical_dir(parent, create)?, name))
    }

    pub(crate) fn resolve_entry(
        &self,
        relative: &Path,
        allow_missing: bool,
    ) -> VaultResult<PathBuf> {
        if self.content_is_restricted() {
            path::normalize_relative(relative)?;
        }
        self.content_result(
            relative,
            self.resolve(relative, allow_missing, false)
                .map(|resolved| resolved.relative),
        )
    }
}

fn same_object(left: &Metadata, right: &Metadata) -> bool {
    left.dev() == right.dev() && left.ino() == right.ino()
}

#[cfg(target_vendor = "apple")]
fn handle_path(file: &File) -> io::Result<Option<PathBuf>> {
    use std::os::unix::ffi::OsStringExt;
    let native = rustix::fs::getpath(file)?;
    Ok(Some(OsString::from_vec(native.into_bytes()).into()))
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn handle_path(file: &File) -> io::Result<Option<PathBuf>> {
    use std::os::fd::AsRawFd;
    let link = format!("/proc/self/fd/{}", file.as_raw_fd());
    match std::fs::read_link(link) {
        Ok(path) => Ok(Some(path)),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

#[cfg(windows)]
fn handle_path(file: &File) -> io::Result<Option<PathBuf>> {
    use std::os::windows::ffi::OsStringExt;
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Storage::FileSystem::GetFinalPathNameByHandleW;
    let mut buffer = vec![0u16; 256];
    loop {
        // FILE_NAME_NORMALIZED and VOLUME_NAME_DOS are both zero. The call
        // returns the required capacity if the UTF-16 buffer is too short.
        let size = unsafe {
            GetFinalPathNameByHandleW(
                file.as_raw_handle().cast(),
                buffer.as_mut_ptr(),
                buffer.len() as u32,
                0,
            )
        };
        if size == 0 {
            return Err(io::Error::last_os_error());
        }
        let size = size as usize;
        if size < buffer.len() {
            return Ok(Some(OsString::from_wide(&buffer[..size]).into()));
        }
        buffer.resize(size + 1, 0);
    }
}

#[cfg(not(any(
    target_vendor = "apple",
    target_os = "linux",
    target_os = "android",
    windows
)))]
fn handle_path(_file: &File) -> io::Result<Option<PathBuf>> {
    Ok(None)
}

pub(super) fn open_regular(
    parent: &Dir,
    name: &OsStr,
    relative: &Path,
    mut options: OpenOptions,
) -> VaultResult<File> {
    options.follow(FollowSymlinks::No);
    // A file can become a FIFO after discovery. Opening it must not wait for a
    // peer while a mutation lock or a worker is held.
    #[cfg(unix)]
    {
        use cap_fs_ext::OpenOptionsExt;
        options.custom_flags(libc::O_NONBLOCK | libc::O_NOCTTY);
    }
    let file = parent
        .open_with(name, &options)
        .map_err(|e| open_error(parent, name, relative, e))?
        .into_std();
    let metadata = file.metadata()?;
    if !metadata.is_file() || std_is_reparse(&metadata) {
        return Err(VaultError::InvalidPath(format!(
            "expected a regular vault file: {}",
            relative.display()
        )));
    }
    Ok(file)
}

pub(crate) fn is_link(metadata: &Metadata) -> bool {
    #[cfg(windows)]
    {
        use cap_fs_ext::OsMetadataExt;
        use windows_sys::Win32::Storage::FileSystem::FILE_ATTRIBUTE_REPARSE_POINT;
        metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0
    }
    #[cfg(not(windows))]
    {
        metadata.is_symlink()
    }
}

fn std_is_reparse(_metadata: &std::fs::Metadata) -> bool {
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        use windows_sys::Win32::Storage::FileSystem::FILE_ATTRIBUTE_REPARSE_POINT;
        _metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0
    }
    #[cfg(not(windows))]
    {
        false
    }
}

fn open_child_dir(parent: &Dir, name: &OsStr, relative: &Path) -> VaultResult<Dir> {
    let dir = parent
        .open_dir_nofollow(name)
        .map_err(|e| open_error(parent, name, relative, e))?;
    #[cfg(windows)]
    if is_link(&dir.dir_metadata()?) {
        return Err(VaultError::InvalidPath(format!(
            "unexpected directory reparse point: {}",
            relative.display()
        )));
    }
    Ok(dir)
}

fn open_error(parent: &Dir, name: &OsStr, path: &Path, error: io::Error) -> VaultError {
    #[cfg(unix)]
    let loop_error = error.raw_os_error() == Some(libc::ELOOP);
    #[cfg(not(unix))]
    let loop_error = false;
    if loop_error
        || parent
            .symlink_metadata(name)
            .is_ok_and(|meta| is_link(&meta))
    {
        return VaultError::InvalidPath(format!(
            "unexpected symlink in physical path: {}",
            path.display()
        ));
    }
    super::map_not_found(path)(error)
}

/// Preserve disk spelling on normalizing/case-insensitive volumes. A native
/// miss may fall back to canonical Unicode equivalence, never to case folding.
fn lookup_component(dir: &Dir, requested: &OsStr) -> VaultResult<Option<(OsString, Metadata)>> {
    let native = match dir.symlink_metadata(requested) {
        Ok(metadata) => Some(metadata),
        Err(error) if error.kind() == ErrorKind::NotFound => None,
        Err(error) => return Err(error.into()),
    };
    let text = requested
        .to_str()
        .ok_or_else(|| VaultError::InvalidPath("vault paths must be UTF-8".into()))?;
    let key = path::canonical_unicode_key(text);
    let mut equivalent = None;
    let mut ambiguous = false;
    #[cfg(test)]
    crate::test_helpers::count_io(&crate::test_helpers::NAME_LOOKUP_SCANS);
    for entry in dir_entries(dir)? {
        let name = entry?.file_name();
        if name == requested {
            return disk_entry(dir, name).map(Some);
        }
        if native.is_none()
            && name
                .to_str()
                .is_some_and(|name| path::canonical_unicode_key(name) == key)
        {
            ambiguous |= equivalent.replace(name).is_some();
        }
    }
    if let Some(native) = native {
        // Filesystems implement their own name equivalence. For example, APFS
        // can resolve long s to S and final sigma to sigma. Use the object the
        // kernel actually looked up to recover the name from this directory.
        let mut matching_link = None;
        let mut ambiguous_fold = false;
        let folded = key.to_lowercase();
        #[cfg(test)]
        crate::test_helpers::count_io(&crate::test_helpers::NAME_LOOKUP_SCANS);
        for entry in dir_entries(dir)? {
            let name = entry?.file_name();
            let metadata = match dir.symlink_metadata(&name) {
                Ok(metadata) => metadata,
                Err(error) if error.kind() == ErrorKind::NotFound => continue,
                Err(error) => return Err(error.into()),
            };
            if same_object(&native, &metadata) {
                if name
                    .to_str()
                    .is_some_and(|text| path::canonical_unicode_key(text).to_lowercase() == folded)
                {
                    // Disambiguates a regular hard link with a different name.
                    ambiguous_fold |= matching_link.replace(name.clone()).is_some();
                }
                ambiguous |= equivalent.replace(name).is_some();
            }
        }
        if let Some(name) = matching_link
            && !ambiguous_fold
        {
            return disk_entry(dir, name).map(Some);
        }
        if equivalent.is_none() {
            return Err(VaultError::NoteNotFound(requested.into()));
        }
    }
    if ambiguous {
        return Err(VaultError::InvalidPath(format!(
            "ambiguous vault filename: {}",
            requested.display()
        )));
    }
    match equivalent {
        Some(name) => disk_entry(dir, name).map(Some),
        None => Ok(None),
    }
}

fn disk_entry(dir: &Dir, name: OsString) -> VaultResult<(OsString, Metadata)> {
    let metadata = dir
        .symlink_metadata(&name)
        .map_err(super::map_not_found(Path::new(&name)))?;
    Ok((name, metadata))
}

/// A writable inode may also be reachable through a denied or external name.
fn require_single_link(file: &File, path: &Path) -> VaultResult<()> {
    #[cfg(unix)]
    let links = { std::os::unix::fs::MetadataExt::nlink(&file.metadata()?) };
    #[cfg(windows)]
    let links = {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Storage::FileSystem::{
            BY_HANDLE_FILE_INFORMATION, GetFileInformationByHandle,
        };
        let mut information = BY_HANDLE_FILE_INFORMATION::default();
        if unsafe { GetFileInformationByHandle(file.as_raw_handle(), &mut information) } == 0 {
            return Err(io::Error::last_os_error().into());
        }
        u64::from(information.nNumberOfLinks)
    };
    #[cfg(not(any(unix, windows)))]
    return Err(VaultError::InvalidPath(
        "authenticated content requires native file identity".into(),
    ));
    #[cfg(any(unix, windows))]
    if links != 1 {
        return Err(VaultError::AccessDenied {
            path: path.to_owned(),
            visible: false,
        });
    }
    Ok(())
}
