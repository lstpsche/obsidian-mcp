//! Protected local files opened relative to retained directory handles.

use std::ffi::{OsStr, OsString};
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use cap_fs_ext::{DirExt, FollowSymlinks, OpenOptionsFollowExt};
use cap_std::fs::{Dir, DirBuilder, OpenOptions};

use super::{AccessError, AccessResult};

#[cfg(windows)]
#[path = "secure_windows.rs"]
mod windows;

#[cfg(target_os = "macos")]
#[path = "secure_macos.rs"]
mod macos;

#[derive(Clone)]
pub(super) struct ProtectedDirectory {
    directory: Arc<Dir>,
    path: PathBuf,
    identity: String,
}

impl ProtectedDirectory {
    pub fn open(path: &Path, create: bool) -> AccessResult<Self> {
        if !path.is_absolute() {
            return Err(AccessError::Configuration(
                "access files require absolute paths".into(),
            ));
        }
        let canonical = canonical_directory(path, create)?;
        let directory = open_canonical_directory(&canonical)?;
        validate_directory(&directory, true)?;
        let identity = object_identity(&directory.try_clone()?.into_std_file())?;
        Ok(Self {
            directory: Arc::new(directory),
            path: canonical,
            identity,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn verify_named(&self) -> AccessResult<()> {
        let named = open_canonical_directory(&self.path)?;
        validate_directory(&named, true)?;
        if object_identity(&named.into_std_file())? != self.identity {
            return Err(AccessError::Configuration(
                "access directory identity changed; reopen the service".into(),
            ));
        }
        Ok(())
    }

    pub fn read(&self, name: &OsStr, maximum: usize) -> AccessResult<Vec<u8>> {
        let mut file = self.open_file(name, false, false)?;
        let len = file.metadata()?.len();
        if len > maximum as u64 {
            return Err(AccessError::Configuration(
                "access file exceeds its size limit".into(),
            ));
        }
        let mut bytes = Vec::with_capacity(len as usize);
        Read::by_ref(&mut file)
            .take(maximum as u64 + 1)
            .read_to_end(&mut bytes)?;
        if bytes.len() > maximum {
            return Err(AccessError::Configuration(
                "access file exceeds its size limit".into(),
            ));
        }
        Ok(bytes)
    }

    pub fn open_file(&self, name: &OsStr, writable: bool, create: bool) -> AccessResult<File> {
        validate_name(name)?;
        let mut options = OpenOptions::new();
        options
            .read(true)
            .write(writable)
            .follow(FollowSymlinks::No);
        if create {
            options.create_new(true);
        }
        #[cfg(unix)]
        {
            use cap_fs_ext::OpenOptionsExt;
            options
                .mode(0o600)
                .custom_flags(libc::O_NONBLOCK | libc::O_NOCTTY);
        }
        let file = self.directory.open_with(name, &options)?.into_std();
        #[cfg(windows)]
        if create {
            windows::protect(&file, false)?;
        }
        #[cfg(target_os = "macos")]
        if create {
            macos::protect(&file)?;
        }
        validate_file(&file)?;
        Ok(file)
    }

    pub fn create(&self, name: &OsStr, bytes: &[u8]) -> AccessResult<()> {
        self.verify_named()?;
        #[cfg(test)]
        fault(FailurePoint::DeliveryCreate)?;
        let mut file = self.open_file(name, true, true)?;
        #[cfg(test)]
        fault(FailurePoint::DeliveryWrite)?;
        file.write_all(bytes)?;
        #[cfg(test)]
        fault(FailurePoint::DeliveryFileSync)?;
        file.sync_all()?;
        #[cfg(test)]
        fault(FailurePoint::DeliveryDirectorySync)?;
        self.sync()?;
        Ok(())
    }

    pub fn publish(&self, name: &OsStr, bytes: &[u8]) -> AccessResult<()> {
        self.verify_named()?;
        let temp_name = OsString::from(format!(".access-{}.tmp", super::random_id()?));
        let mut temporary = Temporary {
            directory: Arc::clone(&self.directory),
            name: temp_name.clone(),
            published: false,
            created: false,
        };
        #[cfg(test)]
        fault(FailurePoint::PolicyCreate)
            .map_err(|error| AccessError::publication(false, error))?;
        let mut file = self.open_file(&temp_name, true, true)?;
        temporary.created = true;
        #[cfg(test)]
        fault(FailurePoint::PolicyWrite).map_err(|error| AccessError::publication(false, error))?;
        file.write_all(bytes)
            .map_err(|error| AccessError::publication(false, error))?;
        #[cfg(test)]
        fault(FailurePoint::PolicyFileSync)
            .map_err(|error| AccessError::publication(false, error))?;
        file.sync_all()
            .map_err(|error| AccessError::publication(false, error))?;
        // Rename through the retained directory, without reopening ambient paths.
        #[cfg(test)]
        fault(FailurePoint::PolicyRename)
            .map_err(|error| AccessError::publication(false, error))?;
        self.directory
            .rename(&temp_name, &self.directory, name)
            .map_err(|error| AccessError::Publication {
                outcome: super::PublicationOutcome::Unknown,
                message: error.to_string(),
            })?;
        temporary.published = true;
        #[cfg(test)]
        fault(FailurePoint::PolicyDirectorySync)
            .map_err(|error| AccessError::publication(true, error))?;
        self.sync()
            .map_err(|error| AccessError::publication(true, error))?;
        Ok(())
    }

    pub fn sync(&self) -> AccessResult<()> {
        #[cfg(unix)]
        self.directory.try_clone()?.into_std_file().sync_all()?;
        // Windows does not support FlushFileBuffers for an ordinary directory
        // handle. File contents are flushed before atomic namespace publication.
        Ok(())
    }
}

struct Temporary {
    directory: Arc<Dir>,
    name: OsString,
    published: bool,
    created: bool,
}

impl Drop for Temporary {
    fn drop(&mut self) {
        if self.created && !self.published {
            let _ = self.directory.remove_file(&self.name);
        }
    }
}

pub(super) fn split_file(
    path: &Path,
    create_parent: bool,
) -> AccessResult<(ProtectedDirectory, OsString)> {
    let name = path
        .file_name()
        .ok_or_else(|| AccessError::Configuration("access file path must name a file".into()))?;
    validate_name(name)?;
    let parent = path.parent().ok_or_else(|| {
        AccessError::Configuration("access file path requires a directory".into())
    })?;
    Ok((
        ProtectedDirectory::open(parent, create_parent)?,
        name.to_owned(),
    ))
}

pub(super) fn validate_name(name: &OsStr) -> AccessResult<()> {
    let path = Path::new(name);
    let mut components = path.components();
    if !matches!(components.next(), Some(Component::Normal(_))) || components.next().is_some() {
        return Err(AccessError::Configuration("invalid access filename".into()));
    }
    #[cfg(windows)]
    {
        let raw = name.to_string_lossy();
        let base = raw.split('.').next().unwrap_or("").to_ascii_uppercase();
        let device = matches!(base.as_str(), "CON" | "PRN" | "AUX" | "NUL")
            || ["COM", "LPT"].iter().any(|prefix| {
                base.strip_prefix(prefix).is_some_and(|number| {
                    matches!(
                        number,
                        "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | "¹" | "²" | "³"
                    )
                })
            });
        if device || raw.ends_with(['.', ' ']) || raw.contains(':') {
            return Err(AccessError::Configuration(
                "ambiguous Windows access filename".into(),
            ));
        }
    }
    Ok(())
}

pub(super) fn validate_directory_path(path: &Path) -> AccessResult<()> {
    if !path.is_absolute()
        || path
            .components()
            .any(|part| matches!(part, Component::ParentDir | Component::CurDir))
    {
        return Err(AccessError::Configuration(
            "access paths must be absolute and normalized".into(),
        ));
    }
    if path.exists() {
        return ProtectedDirectory::open(path, false).map(drop);
    }
    let mut ancestor = path;
    while !ancestor.exists() {
        reject_dangling_alias(ancestor)?;
        validate_name(
            ancestor
                .file_name()
                .ok_or_else(|| AccessError::Configuration("invalid access directory".into()))?,
        )?;
        ancestor = ancestor
            .parent()
            .ok_or_else(|| AccessError::Configuration("invalid access directory".into()))?;
    }
    let directory = open_canonical_directory(&ancestor.canonicalize()?)?;
    validate_directory(&directory, false)?;
    validate_creation_parent(&directory)
}

fn canonical_directory(path: &Path, create: bool) -> AccessResult<PathBuf> {
    if path
        .components()
        .any(|part| matches!(part, Component::ParentDir | Component::CurDir))
    {
        return Err(AccessError::Configuration(
            "access paths must be normalized".into(),
        ));
    }
    if path.exists() {
        return Ok(path.canonicalize()?);
    }
    if !create {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "access directory not found",
        )
        .into());
    }
    let mut missing = Vec::new();
    let mut ancestor = path;
    while !ancestor.exists() {
        reject_dangling_alias(ancestor)?;
        missing.push(
            ancestor
                .file_name()
                .ok_or_else(|| AccessError::Configuration("invalid access directory".into()))?
                .to_owned(),
        );
        ancestor = ancestor
            .parent()
            .ok_or_else(|| AccessError::Configuration("invalid access directory".into()))?;
    }
    let canonical = ancestor.canonicalize()?;
    let mut directory = open_canonical_directory(&canonical)?;
    let mut full = canonical;
    for name in missing.into_iter().rev() {
        validate_creation_parent(&directory)?;
        let builder = private_directory_builder();
        let created = match directory.create_dir_with(&name, &builder) {
            Ok(()) => true,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => false,
            Err(error) => return Err(error.into()),
        };
        directory = directory.open_dir_nofollow(&name)?;
        #[cfg(windows)]
        if created {
            windows::protect(&directory.try_clone()?.into_std_file(), true)?;
        }
        #[cfg(target_os = "macos")]
        if created {
            macos::protect(&directory.try_clone()?.into_std_file())?;
        }
        #[cfg(not(any(windows, target_os = "macos")))]
        let _ = created;
        validate_directory(&directory, true)?;
        full.push(name);
    }
    Ok(full)
}

fn reject_dangling_alias(path: &Path) -> AccessResult<()> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_symlink() => Err(AccessError::Configuration(
            "unresolved access directory alias".into(),
        )),
        Ok(_) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

fn validate_creation_parent(directory: &Dir) -> AccessResult<()> {
    #[cfg(any(windows, target_os = "macos"))]
    let file = directory.try_clone()?.into_std_file();
    #[cfg(target_os = "macos")]
    macos::validate_creation_parent(&file)?;
    #[cfg(windows)]
    windows::validate_creation_parent(&file)?;
    #[cfg(not(any(windows, target_os = "macos")))]
    let _ = directory;
    Ok(())
}

fn private_directory_builder() -> DirBuilder {
    #[cfg(unix)]
    {
        use cap_std::fs::DirBuilderExt;
        let mut builder = DirBuilder::new();
        builder.mode(0o700);
        builder
    }
    #[cfg(not(unix))]
    {
        DirBuilder::new()
    }
}

fn open_canonical_directory(path: &Path) -> AccessResult<Dir> {
    let mut base = PathBuf::new();
    let mut names = Vec::new();
    for part in path.components() {
        match part {
            Component::Prefix(_) | Component::RootDir => base.push(part.as_os_str()),
            Component::Normal(name) => names.push(name),
            _ => {
                return Err(AccessError::Configuration(
                    "access directory is not normalized".into(),
                ));
            }
        }
    }
    let mut directory = Dir::open_ambient_dir(base, cap_std::ambient_authority())?;
    validate_directory(&directory, false)?;
    for name in names {
        validate_name(name)?;
        directory = directory.open_dir_nofollow(name)?;
        validate_directory(&directory, false)?;
    }
    Ok(directory)
}

fn validate_directory(directory: &Dir, private: bool) -> AccessResult<()> {
    let file = directory.try_clone()?.into_std_file();
    let metadata = file.metadata()?;
    if !metadata.is_dir() {
        return Err(AccessError::Configuration(
            "access parent must be a directory".into(),
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let owner = unsafe { libc::geteuid() };
        let mode = metadata.mode();
        let trusted_ancestor = metadata.uid() == 0 || metadata.uid() == owner;
        let sticky_root = metadata.uid() == 0 && mode & u32::from(libc::S_ISVTX) != 0;
        if !trusted_ancestor
            || (mode & 0o022 != 0 && !sticky_root)
            || (private && (metadata.uid() != owner || mode & 0o077 != 0))
        {
            return Err(AccessError::Configuration(
                "access directory must be owner-controlled; use a private directory with mode 0700"
                    .into(),
            ));
        }
    }
    #[cfg(windows)]
    windows::validate(&file, private, true)?;
    #[cfg(target_os = "macos")]
    macos::validate(&file, private)?;
    Ok(())
}

fn validate_file(file: &File) -> AccessResult<()> {
    let metadata = file.metadata()?;
    if !metadata.is_file() {
        return Err(AccessError::Configuration(
            "access file must be a regular file".into(),
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if metadata.uid() != unsafe { libc::geteuid() }
            || metadata.mode() & 0o077 != 0
            || metadata.nlink() != 1
        {
            return Err(AccessError::Configuration(
                "access file must be owned by the service owner with mode 0600 and no hard links"
                    .into(),
            ));
        }
    }
    #[cfg(windows)]
    windows::validate(file, true, false)?;
    #[cfg(target_os = "macos")]
    macos::validate(file, true)?;
    Ok(())
}

pub(super) fn object_identity(file: &File) -> AccessResult<String> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let metadata = file.metadata()?;
        Ok(format!("unix:{}:{}", metadata.dev(), metadata.ino()))
    }
    #[cfg(windows)]
    {
        windows::object_identity(file)
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = file;
        Err(AccessError::Configuration(
            "authenticated access is unsupported on this platform".into(),
        ))
    }
}

#[cfg(test)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum FailurePoint {
    DeliveryCreate,
    DeliveryWrite,
    DeliveryFileSync,
    DeliveryDirectorySync,
    PolicyCreate,
    PolicyWrite,
    PolicyFileSync,
    PolicyRename,
    PolicyDirectorySync,
}
#[cfg(test)]
thread_local! { static FAILURE: std::cell::Cell<Option<FailurePoint>> = const { std::cell::Cell::new(None) }; }
#[cfg(test)]
pub(super) fn fail_next(point: FailurePoint) {
    FAILURE.set(Some(point));
}
#[cfg(test)]
fn fault(point: FailurePoint) -> AccessResult<()> {
    if FAILURE.with(|failure| failure.get() == Some(point)) {
        FAILURE.set(None);
        return Err(std::io::Error::other("injected protected-file failure").into());
    }
    Ok(())
}
