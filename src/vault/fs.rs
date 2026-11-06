//! Rooted filesystem operations for note content and directory entries.

mod root;
#[cfg(windows)]
mod windows;
pub use root::VaultRoot;
pub(crate) use root::is_link;
#[cfg(windows)]
pub(crate) use windows::dir_entries;
#[cfg(windows)]
use windows::{create_child_dir, remove_entry};

use std::ffi::OsStr;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use cap_std::fs::{Dir, OpenOptions};
use globset::Glob;

use crate::error::{VaultError, VaultResult};
use crate::models::FileStat;
use crate::vault::path as vault_path;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoveResult {
    pub from: PathBuf,
    pub to: PathBuf,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum WriteMode {
    Replace,
    Append,
    Create,
}

fn map_not_found(path: &Path) -> impl FnOnce(io::Error) -> VaultError + '_ {
    |e| match e.kind() {
        io::ErrorKind::NotFound => VaultError::NoteNotFound(path.to_owned()),
        _ => VaultError::Io(e),
    }
}

pub(super) fn is_missing_entry(error: &VaultError) -> bool {
    error.is_not_found()
        || matches!(error, VaultError::Io(err) if err.kind() == io::ErrorKind::NotADirectory)
}

/// Only for no-follow opens or stats of a previously validated native name.
/// An invalid physical path here means a component became a symlink or reparse point.
pub(super) fn skip_changed_entry(error: &VaultError) -> bool {
    if matches!(error, VaultError::InvalidPath(_)) {
        tracing::warn!(%error, "skipping a vault entry replaced during traversal");
        return true;
    }
    is_missing_entry(error)
}

fn file_times(
    size: u64,
    created: io::Result<std::time::SystemTime>,
    modified: io::Result<std::time::SystemTime>,
) -> FileStat {
    FileStat {
        size,
        created: created.ok().map(chrono::DateTime::<chrono::Utc>::from),
        modified: modified.ok().map(chrono::DateTime::<chrono::Utc>::from),
    }
}

pub fn stat_from_metadata(metadata: &std::fs::Metadata) -> FileStat {
    file_times(metadata.len(), metadata.created(), metadata.modified())
}

impl VaultRoot {
    fn target_metadata(&self, path: &Path) -> VaultResult<cap_std::fs::Metadata> {
        let relative = self.resolve_existing(path)?.relative;
        let metadata = self.entry_metadata_exact(&relative)?;
        if is_link(&metadata) {
            return Err(VaultError::InvalidPath(format!(
                "physical vault entry changed to a link: {}",
                path.display()
            )));
        }
        Ok(metadata)
    }

    pub fn file_exists(&self, path: &Path) -> bool {
        self.target_metadata(path).is_ok_and(|meta| meta.is_file())
    }

    pub fn file_stat(&self, path: &Path) -> VaultResult<FileStat> {
        let meta = self.target_metadata(path)?;
        Ok(file_times(
            meta.len(),
            meta.created().map(|t| t.into_std()),
            meta.modified().map(|t| t.into_std()),
        ))
    }

    pub fn read_file(&self, path: &Path) -> VaultResult<String> {
        read_text(&mut self.open_file(path)?)
    }

    pub fn read_file_bounded(&self, path: &Path, max_bytes: usize) -> VaultResult<String> {
        read_bounded(&mut self.open_file(path)?, path, max_bytes)
    }

    pub fn read_observed(&self, path: &Path) -> VaultResult<String> {
        self.read_observed_with(path, read_text)
    }

    pub fn read_observed_with_stat(&self, path: &Path) -> VaultResult<(String, FileStat)> {
        self.read_observed_with(path, |file| {
            let content = read_text(file)?;
            Ok((content, stat_from_metadata(&file.metadata()?)))
        })
    }

    pub fn read_exact_bounded(&self, path: &Path, max_bytes: usize) -> VaultResult<String> {
        self.read_observed_with(path, |file| read_bounded(file, path, max_bytes))
    }

    pub(crate) fn read_exact_bounded_with_stat(
        &self,
        path: &Path,
        max_bytes: usize,
    ) -> VaultResult<(String, FileStat)> {
        self.read_observed_with(path, |file| {
            let content = read_bounded(file, path, max_bytes)?;
            Ok((content, stat_from_metadata(&file.metadata()?)))
        })
    }

    fn read_observed_with<T>(
        &self,
        path: &Path,
        read: impl FnOnce(&mut File) -> VaultResult<T>,
    ) -> VaultResult<T> {
        let mut file = self.open_exact(path)?;
        self.verify_observed(path, &file)?;
        let result = read(&mut file)?;
        self.verify_observed(path, &file)?;
        Ok(result)
    }

    pub fn write_file(&self, path: &Path, content: &str) -> VaultResult<PathBuf> {
        self.write(path, content, WriteMode::Replace)
    }

    pub fn create_file(&self, path: &Path, content: &str) -> VaultResult<PathBuf> {
        self.write(path, content, WriteMode::Create)
    }

    pub fn append_file(&self, path: &Path, content: &str) -> VaultResult<PathBuf> {
        self.write(path, content, WriteMode::Append)
    }

    fn write(&self, path: &Path, content: &str, mode: WriteMode) -> VaultResult<PathBuf> {
        let actual = self.resolve_for_write(path)?.relative;
        self.write_resolved(&actual, content, mode)?;
        Ok(actual)
    }

    fn write_resolved(&self, path: &Path, content: &str, mode: WriteMode) -> VaultResult<()> {
        let (parent, name) = self.parent(path, true)?;
        let mut options = OpenOptions::new();
        options.write(true).append(mode == WriteMode::Append);
        match parent.symlink_metadata(name) {
            Ok(_) if mode == WriteMode::Create => {
                return Err(VaultError::AlreadyExists(path.to_owned()));
            }
            Ok(metadata) if is_link(&metadata) || !metadata.is_file() => {
                return Err(VaultError::InvalidPath(format!(
                    "expected a regular vault file: {}",
                    path.display()
                )));
            }
            Ok(_) => (),
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                options.create_new(true);
            }
            Err(e) => return Err(e.into()),
        }
        let mut file =
            root::open_regular(&parent, name, path, options).map_err(|err| match err {
                VaultError::Io(e) if e.kind() == io::ErrorKind::AlreadyExists => {
                    VaultError::AlreadyExists(path.to_owned())
                }
                error => error,
            })?;
        if mode == WriteMode::Replace {
            file.set_len(0)?;
        }
        file.write_all(content.as_bytes())?;
        Ok(())
    }

    /// Unlink the selected entry; an in-vault alias never unlinks its target.
    pub fn delete_file(&self, path: &Path) -> VaultResult<PathBuf> {
        let actual = self.resolve_entry(path, false)?;
        let (parent, name) = self.parent(&actual, false)?;
        let metadata = parent.symlink_metadata(name).map_err(map_not_found(path))?;
        if is_link(&metadata) {
            self.link_target(&actual, path)?;
        }
        remove_entry(&parent, name, metadata.is_dir())?;
        Ok(actual)
    }

    pub fn move_file(&self, from: &Path, to: &Path) -> VaultResult<MoveResult> {
        let source = self.resolve_entry(from, false)?;
        let (source_dir, source_name) = self.parent(&source, false)?;
        // Check the effective target while keeping the original entry as the
        // object to rename. Validate before creating any destination directory.
        let resolved = self.resolve_existing(from)?;
        if !self.entry_metadata_exact(&resolved.relative)?.is_file() {
            return Err(VaultError::InvalidPath(format!(
                "note move requires a regular file: {}",
                from.display()
            )));
        }
        let destination = self.resolve_entry(to, true)?;
        if source.parent() != destination.parent()
            && is_link(&source_dir.symlink_metadata(source_name)?)
            && source_dir.read_link_contents(source_name)?.is_relative()
        {
            return Err(VaultError::InvalidPath(format!(
                "cannot move a relative note symlink to another directory: {}",
                from.display()
            )));
        }
        let (destination_dir, destination_name) = self.parent(&destination, true)?;
        rename_exclusive(&source_dir, source_name, &destination_dir, destination_name).map_err(
            |error| match error.kind() {
                io::ErrorKind::AlreadyExists => VaultError::AlreadyExists(to.to_owned()),
                _ => VaultError::Io(error),
            },
        )?;
        Ok(MoveResult {
            from: source,
            to: destination,
        })
    }

    pub fn list_files(
        &self,
        relative: &Path,
        recursive: bool,
        glob: Option<&str>,
    ) -> VaultResult<Vec<PathBuf>> {
        self.list_entries(relative, recursive, glob, false)
    }

    pub(crate) fn list_regular_files(
        &self,
        relative: &Path,
        recursive: bool,
        glob: Option<&str>,
    ) -> VaultResult<Vec<PathBuf>> {
        self.list_entries(relative, recursive, glob, true)
    }

    fn list_entries(
        &self,
        relative: &Path,
        recursive: bool,
        glob: Option<&str>,
        regular_only: bool,
    ) -> VaultResult<Vec<PathBuf>> {
        let actual = self
            .resolve_existing(relative)
            .map_err(|e| match e {
                VaultError::NoteNotFound(_) => VaultError::DirectoryNotFound(relative.to_owned()),
                error => error,
            })?
            .relative;
        let matcher = glob
            .map(|pattern| {
                Glob::new(pattern)
                    .map(|g| g.compile_matcher())
                    .map_err(|e| VaultError::InvalidPath(format!("invalid glob pattern: {e}")))
            })
            .transpose()?;
        let mut results = Vec::new();
        self.visit_entries_with_types(
            &actual,
            recursive,
            super::exclude::is_visible_path,
            || Ok(()),
            |path, is_file, is_dir| {
                if matcher
                    .as_ref()
                    .is_some_and(|matcher| !matcher.is_match(path))
                {
                    return Ok(());
                }
                let alias_file = if regular_only && !is_file && !is_dir {
                    let metadata = match self.entry_metadata_exact(path) {
                        Ok(metadata) => metadata,
                        Err(error) if error.is_not_found() => return Ok(()),
                        Err(error) => return Err(error),
                    };
                    if is_link(&metadata) {
                        match self.resolve_existing(path) {
                            Ok(target) => self.entry_metadata_exact(&target.relative)?.is_file(),
                            Err(error) if error.is_not_found() => false,
                            Err(error) => return Err(error),
                        }
                    } else {
                        false
                    }
                } else {
                    false
                };
                if !regular_only || is_file || alias_file {
                    results.push(path.to_owned());
                }
                Ok(())
            },
        )
        .map_err(|error| {
            if is_missing_entry(&error) {
                VaultError::DirectoryNotFound(relative.to_owned())
            } else {
                error
            }
        })?;
        results.sort();
        Ok(results)
    }

    /// Walk addressable entries without following links. Queue native relative
    /// paths so a wide directory never holds all its children's handles open.
    pub(super) fn visit_entries(
        &self,
        directory: &Path,
        recursive: bool,
        include_name: impl Fn(&Path) -> bool,
        check: impl Fn() -> VaultResult<()>,
        mut visit: impl FnMut(&Path, bool) -> VaultResult<()>,
    ) -> VaultResult<()> {
        self.visit_entries_with_types(
            directory,
            recursive,
            include_name,
            check,
            |path, file, _| visit(path, file),
        )
    }

    fn visit_entries_with_types(
        &self,
        directory: &Path,
        recursive: bool,
        include_name: impl Fn(&Path) -> bool,
        check: impl Fn() -> VaultResult<()>,
        mut visit: impl FnMut(&Path, bool, bool) -> VaultResult<()>,
    ) -> VaultResult<()> {
        let mut pending = vec![directory.to_owned()];
        while let Some(base) = pending.pop() {
            check()?;
            let current = match self.open_dir_exact(&base) {
                Err(error) if base != directory && skip_changed_entry(&error) => continue,
                other => other?,
            };
            #[cfg(test)]
            crate::test_helpers::count_io(&crate::test_helpers::INDEX_DIRECTORY_VISITS);
            for entry in dir_entries(&current)? {
                check()?;
                let entry = entry?;
                let name = entry.file_name();
                if !include_name(Path::new(&name)) {
                    continue;
                }
                let native = base.join(name);
                let path = match vault_path::portable_native(&native) {
                    Err(error @ VaultError::InvalidPath(_)) => {
                        tracing::warn!(path = %native.display(), %error, "skipping an unrepresentable vault entry");
                        continue;
                    }
                    other => other?,
                };
                let kind = match entry.file_type() {
                    Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
                    other => other?,
                };
                if recursive && kind.is_dir() {
                    pending.push(path.clone());
                }
                visit(&path, kind.is_file(), kind.is_dir())?;
            }
        }
        Ok(())
    }
}

#[cfg(any(
    target_os = "linux",
    target_os = "android",
    target_vendor = "apple",
    target_os = "redox"
))]
fn rename_exclusive(
    source: &Dir,
    old: &std::ffi::OsStr,
    destination: &Dir,
    new: &std::ffi::OsStr,
) -> io::Result<()> {
    rustix::fs::renameat_with(
        source,
        old,
        destination,
        new,
        rustix::fs::RenameFlags::NOREPLACE,
    )
    .map_err(Into::into)
}

#[cfg(all(
    not(windows),
    not(any(
        target_os = "linux",
        target_os = "android",
        target_vendor = "apple",
        target_os = "redox"
    ))
))]
fn rename_exclusive(_: &Dir, _: &std::ffi::OsStr, _: &Dir, _: &std::ffi::OsStr) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "atomic no-replace rename is unavailable on this platform",
    ))
}

#[cfg(windows)]
fn rename_exclusive(source: &Dir, old: &OsStr, destination: &Dir, new: &OsStr) -> io::Result<()> {
    windows::rename(source, old, destination, new, false)
}

#[cfg(not(windows))]
pub(crate) fn dir_entries(dir: &Dir) -> io::Result<cap_std::fs::ReadDir> {
    dir.entries()
}

#[cfg(not(windows))]
fn create_child_dir(dir: &Dir, name: &OsStr) -> io::Result<()> {
    dir.create_dir(name)
}

#[cfg(not(windows))]
fn remove_entry(dir: &Dir, name: &OsStr, directory: bool) -> io::Result<()> {
    if directory {
        dir.remove_dir(name)
    } else {
        dir.remove_file(name)
    }
}

#[cfg(has_embeddings)]
pub(crate) fn replace_child(dir: &Dir, old: &OsStr, new: &OsStr) -> io::Result<()> {
    #[cfg(windows)]
    {
        windows::rename(dir, old, dir, new, true)
    }
    #[cfg(not(windows))]
    {
        dir.rename(old, dir, new)
    }
}

#[cfg(has_embeddings)]
pub(crate) fn remove_temp_file(dir: &Dir, name: &OsStr) -> io::Result<()> {
    remove_entry(dir, name, false)
}

fn read_text(file: &mut File) -> VaultResult<String> {
    let mut text = String::new();
    file.read_to_string(&mut text)?;
    Ok(text)
}

fn read_bounded(file: &mut File, path: &Path, max_bytes: usize) -> VaultResult<String> {
    let read_limit = max_bytes
        .checked_add(1)
        .ok_or_else(|| VaultError::InvalidSourceRange("source byte limit is too large".into()))?;
    let hint = file.metadata()?.len().min(read_limit as u64).min(64 * 1024) as usize;
    let bytes = read_bounded_bytes(file, hint, read_limit)?;
    if bytes.len() > max_bytes {
        return Err(VaultError::SemanticResourceLimit {
            resource: format!("source bytes for {}", path.display()),
            actual: bytes.len(),
            limit: max_bytes,
        });
    }
    String::from_utf8(bytes)
        .map_err(|error| VaultError::Io(io::Error::new(io::ErrorKind::InvalidData, error)))
}

fn read_bounded_bytes(
    reader: &mut impl Read,
    initial_capacity: usize,
    read_limit: usize,
) -> std::io::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(initial_capacity.min(read_limit))
        .map_err(std::io::Error::other)?;
    let mut buffer = [0; 8192];
    while bytes.len() < read_limit {
        let remaining = (read_limit - bytes.len()).min(buffer.len());
        let count = match reader.read(&mut buffer[..remaining]) {
            Ok(count) => count,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        };
        if count == 0 {
            break;
        }
        let needed = bytes.len() + count;
        if needed > bytes.capacity() {
            let capacity = bytes
                .capacity()
                .saturating_mul(2)
                .max(buffer.len())
                .max(needed)
                .min(read_limit);
            bytes
                .try_reserve_exact(capacity - bytes.len())
                .map_err(std::io::Error::other)?;
        }
        bytes.extend_from_slice(&buffer[..count]);
    }
    Ok(bytes)
}

/// Standalone helpers for callers without a retained vault handle.
pub fn resolve_path(vault_root: &Path, relative: &Path) -> VaultResult<PathBuf> {
    Ok(VaultRoot::open(vault_root)?
        .resolve_for_write(relative)?
        .absolute)
}

pub fn file_exists(vault_root: &Path, path: &Path) -> bool {
    VaultRoot::open(vault_root).is_ok_and(|root| root.file_exists(path))
}

pub fn file_stat(vault_root: &Path, path: &Path) -> VaultResult<FileStat> {
    VaultRoot::open(vault_root)?.file_stat(path)
}

pub fn read_file(vault_root: &Path, path: &Path) -> VaultResult<String> {
    VaultRoot::open(vault_root)?.read_file(path)
}

pub fn read_file_bounded(vault_root: &Path, path: &Path, max_bytes: usize) -> VaultResult<String> {
    VaultRoot::open(vault_root)?.read_file_bounded(path, max_bytes)
}

pub fn write_file(vault_root: &Path, path: &Path, content: &str) -> VaultResult<PathBuf> {
    VaultRoot::open(vault_root)?.write_file(path, content)
}

pub fn create_file(vault_root: &Path, path: &Path, content: &str) -> VaultResult<PathBuf> {
    VaultRoot::open(vault_root)?.create_file(path, content)
}

pub fn append_file(vault_root: &Path, path: &Path, content: &str) -> VaultResult<PathBuf> {
    VaultRoot::open(vault_root)?.append_file(path, content)
}

pub fn delete_file(vault_root: &Path, path: &Path) -> VaultResult<PathBuf> {
    VaultRoot::open(vault_root)?.delete_file(path)
}

pub fn move_file(vault_root: &Path, from: &Path, to: &Path) -> VaultResult<MoveResult> {
    VaultRoot::open(vault_root)?.move_file(from, to)
}

pub fn list_files(
    vault_root: &Path,
    dir: &Path,
    recursive: bool,
    glob: Option<&str>,
) -> VaultResult<Vec<PathBuf>> {
    VaultRoot::open(vault_root)?.list_files(dir, recursive, glob)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;
    use unicode_normalization::UnicodeNormalization;

    fn setup_vault() -> TempDir {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("note1.md"), "# Note 1\nHello").unwrap();
        fs::write(dir.path().join("note2.md"), "# Note 2\nWorld").unwrap();
        fs::create_dir_all(dir.path().join("subfolder")).unwrap();
        fs::write(dir.path().join("subfolder/nested.md"), "# Nested\nContent").unwrap();
        fs::create_dir_all(dir.path().join(".obsidian")).unwrap();
        fs::write(
            dir.path().join(".obsidian/config.json"),
            r#"{"key":"value"}"#,
        )
        .unwrap();
        fs::write(dir.path().join(".hidden_file"), "secret").unwrap();
        dir
    }

    #[test]
    fn bounded_reader_growth_never_reserves_past_the_probe_limit() {
        for read_limit in [0, 1, 8191, 8192, 8193, 16_385, 65_537] {
            let data = vec![b'x'; read_limit + 31];
            for hint in [0, 1, read_limit.saturating_add(100)] {
                let mut reader = std::io::Cursor::new(&data);
                let bytes = read_bounded_bytes(&mut reader, hint, read_limit).unwrap();
                assert_eq!(bytes, data[..read_limit]);
                assert!(bytes.capacity() <= read_limit);
                assert_eq!(reader.position(), read_limit as u64);
            }
        }
    }

    #[test]
    fn bounded_source_read_handles_exact_capacity_and_one_byte_over() {
        let dir = TempDir::new().unwrap();
        let path = Path::new("bounded.md");
        for size in [0, 1, 8191, 8192, 8193, 65_537] {
            fs::write(dir.path().join(path), vec![b'x'; size]).unwrap();
            let text = read_file_bounded(dir.path(), path, size).unwrap();
            assert_eq!(text.len(), size);
            assert!(text.capacity() <= size + 1);
            if size > 0 {
                assert!(matches!(
                    read_file_bounded(dir.path(), path, size - 1),
                    Err(VaultError::SemanticResourceLimit { actual, limit, .. })
                        if actual == size && limit == size - 1
                ));
            }
        }
    }

    // ── resolve_path ──

    #[test]
    fn resolve_simple_relative_path() {
        let vault = setup_vault();
        let result = resolve_path(vault.path(), Path::new("note1.md")).unwrap();
        assert!(result.is_file());
        assert!(result.ends_with("note1.md"));
    }

    #[test]
    fn resolve_rejects_parent_escape() {
        let vault = setup_vault();
        let err = resolve_path(vault.path(), Path::new("../etc/passwd")).unwrap_err();
        assert!(matches!(err, VaultError::OutsideVault(_)));
    }

    #[test]
    fn resolve_rejects_nested_parent_escape() {
        let vault = setup_vault();
        let err = resolve_path(vault.path(), Path::new("subfolder/../../etc/passwd")).unwrap_err();
        assert!(matches!(err, VaultError::OutsideVault(_)));
    }

    #[test]
    fn resolve_rejects_absolute_path() {
        let vault = setup_vault();
        let err = resolve_path(vault.path(), Path::new("/etc/passwd")).unwrap_err();
        assert!(matches!(err, VaultError::InvalidPath(_)));
    }

    #[test]
    fn resolve_normalizes_dot() {
        let vault = setup_vault();
        let result = resolve_path(vault.path(), Path::new("./note1.md")).unwrap();
        assert!(result.is_file());
    }

    #[test]
    fn resolve_allows_valid_parent_dir() {
        let vault = setup_vault();
        let result = resolve_path(vault.path(), Path::new("subfolder/../note1.md")).unwrap();
        assert!(result.is_file());
    }

    // ── file_exists ──

    #[test]
    fn exists_returns_true_for_file() {
        let vault = setup_vault();
        assert!(file_exists(vault.path(), Path::new("note1.md")));
    }

    #[test]
    fn exists_returns_false_for_missing() {
        let vault = setup_vault();
        assert!(!file_exists(vault.path(), Path::new("no_such.md")));
    }

    #[test]
    fn exists_returns_false_for_directory() {
        let vault = setup_vault();
        assert!(!file_exists(vault.path(), Path::new("subfolder")));
    }

    #[test]
    fn exists_returns_false_for_traversal() {
        let vault = setup_vault();
        assert!(!file_exists(vault.path(), Path::new("../../etc/passwd")));
    }

    // ── file_stat ──

    #[test]
    fn stat_returns_correct_size() {
        let vault = setup_vault();
        let stat = file_stat(vault.path(), Path::new("note1.md")).unwrap();
        assert_eq!(stat.size, "# Note 1\nHello".len() as u64);
        assert!(stat.modified.is_some());
    }

    #[test]
    fn stat_not_found() {
        let vault = setup_vault();
        let err = file_stat(vault.path(), Path::new("missing.md")).unwrap_err();
        assert!(matches!(err, VaultError::NoteNotFound(_)));
    }

    // ── read_file / write_file round-trip ──

    #[test]
    fn read_existing_file() {
        let vault = setup_vault();
        let content = read_file(vault.path(), Path::new("note1.md")).unwrap();
        assert_eq!(content, "# Note 1\nHello");
    }

    #[test]
    fn read_missing_file() {
        let vault = setup_vault();
        let err = read_file(vault.path(), Path::new("nope.md")).unwrap_err();
        assert!(matches!(err, VaultError::NoteNotFound(_)));
    }

    #[test]
    fn write_and_read_round_trip() {
        let vault = setup_vault();
        let path = Path::new("new_note.md");
        write_file(vault.path(), path, "fresh content").unwrap();
        let content = read_file(vault.path(), path).unwrap();
        assert_eq!(content, "fresh content");
    }

    #[test]
    fn write_creates_parent_dirs() {
        let vault = setup_vault();
        let path = Path::new("deep/nested/dir/note.md");
        write_file(vault.path(), path, "deep").unwrap();
        assert_eq!(read_file(vault.path(), path).unwrap(), "deep");
    }

    #[test]
    fn write_overwrites_existing() {
        let vault = setup_vault();
        let path = Path::new("note1.md");
        write_file(vault.path(), path, "overwritten").unwrap();
        assert_eq!(read_file(vault.path(), path).unwrap(), "overwritten");
    }

    #[test]
    fn write_returns_existing_unicode_normalized_relative_path() {
        let vault = setup_vault();
        let composed = "02_База-знаний/lic1c.md";
        let decomposed: String = composed.nfd().collect();
        let disk_path = PathBuf::from(&decomposed);
        fs::create_dir_all(vault.path().join(disk_path.parent().unwrap())).unwrap();
        fs::write(vault.path().join(&disk_path), "old").unwrap();

        let written = write_file(vault.path(), Path::new(composed), "new").unwrap();

        assert_eq!(written, disk_path);
        assert_eq!(read_file(vault.path(), Path::new(composed)).unwrap(), "new");
    }

    #[test]
    fn write_uses_existing_unicode_normalized_parent() {
        let vault = setup_vault();
        let composed_dir = "02_База-знаний";
        let decomposed_dir: String = composed_dir.nfd().collect();
        fs::create_dir_all(vault.path().join(&decomposed_dir)).unwrap();

        let written =
            write_file(vault.path(), Path::new("02_База-знаний/New.md"), "content").unwrap();

        assert_eq!(written, PathBuf::from(format!("{decomposed_dir}/New.md")));
        assert!(vault.path().join(written).is_file());
    }

    // ── append_file ──

    #[test]
    fn append_creates_new_file() {
        let vault = setup_vault();
        let path = Path::new("appended.md");
        append_file(vault.path(), path, "line1\n").unwrap();
        assert_eq!(read_file(vault.path(), path).unwrap(), "line1\n");
    }

    #[test]
    fn append_adds_to_existing() {
        let vault = setup_vault();
        let path = Path::new("note1.md");
        append_file(vault.path(), path, "\nappended").unwrap();
        assert_eq!(
            read_file(vault.path(), path).unwrap(),
            "# Note 1\nHello\nappended"
        );
    }

    // ── delete_file ──

    #[test]
    fn delete_existing_file() {
        let vault = setup_vault();
        let path = Path::new("note1.md");
        assert!(file_exists(vault.path(), path));
        delete_file(vault.path(), path).unwrap();
        assert!(!file_exists(vault.path(), path));
    }

    #[test]
    fn delete_empty_directory() {
        let vault = setup_vault();
        let dir_path = Path::new("empty_dir");
        fs::create_dir(vault.path().join("empty_dir")).unwrap();
        delete_file(vault.path(), dir_path).unwrap();
        assert!(!vault.path().join("empty_dir").exists());
    }

    #[test]
    fn delete_missing_file() {
        let vault = setup_vault();
        let err = delete_file(vault.path(), Path::new("missing.md")).unwrap_err();
        assert!(matches!(err, VaultError::NoteNotFound(_)));
    }

    // ── move_file ──

    #[test]
    fn move_renames_file() {
        let vault = setup_vault();
        let from = Path::new("note1.md");
        let to = Path::new("renamed.md");
        let result = move_file(vault.path(), from, to).unwrap();
        assert_eq!(result.from, PathBuf::from("note1.md"));
        assert_eq!(result.to, PathBuf::from("renamed.md"));
        assert!(!file_exists(vault.path(), from));
        assert!(file_exists(vault.path(), to));
        assert_eq!(read_file(vault.path(), to).unwrap(), "# Note 1\nHello");
    }

    #[test]
    fn move_returns_existing_unicode_normalized_source() {
        let vault = setup_vault();
        let composed = "02_База-знаний/lic1c.md";
        let decomposed: String = composed.nfd().collect();
        let disk_path = PathBuf::from(&decomposed);
        fs::create_dir_all(vault.path().join(disk_path.parent().unwrap())).unwrap();
        fs::write(vault.path().join(&disk_path), "# License").unwrap();

        let moved = move_file(
            vault.path(),
            Path::new(composed),
            Path::new("Moved/lic1c.md"),
        )
        .unwrap();

        assert_eq!(moved.from, disk_path);
        assert_eq!(moved.to, PathBuf::from("Moved/lic1c.md"));
        assert!(file_exists(vault.path(), Path::new("Moved/lic1c.md")));
    }

    #[test]
    fn move_creates_parent_dirs() {
        let vault = setup_vault();
        let from = Path::new("note1.md");
        let to = Path::new("new_dir/note1.md");
        move_file(vault.path(), from, to).unwrap();
        assert!(file_exists(vault.path(), to));
    }

    #[test]
    fn move_rejects_existing_destination() {
        let vault = setup_vault();
        let err =
            move_file(vault.path(), Path::new("note1.md"), Path::new("note2.md")).unwrap_err();
        assert!(matches!(err, VaultError::AlreadyExists(_)));
    }

    #[test]
    fn move_rejects_missing_source() {
        let vault = setup_vault();
        let err = move_file(vault.path(), Path::new("ghost.md"), Path::new("dest.md")).unwrap_err();
        assert!(matches!(err, VaultError::NoteNotFound(_)));
    }

    // ── list_files ──

    #[test]
    fn list_root_non_recursive() {
        let vault = setup_vault();
        let files = list_files(vault.path(), Path::new(""), false, None).unwrap();
        let names: Vec<String> = files.iter().map(|p| p.display().to_string()).collect();
        assert!(names.contains(&"note1.md".to_string()));
        assert!(names.contains(&"note2.md".to_string()));
        assert!(names.contains(&"subfolder".to_string()));
        assert!(!names.iter().any(|n| n.contains(".obsidian")));
        assert!(!names.iter().any(|n| n.contains(".hidden")));
    }

    #[test]
    fn list_recursive() {
        let vault = setup_vault();
        let files = list_files(vault.path(), Path::new(""), true, None).unwrap();
        let names: Vec<String> = files.iter().map(|p| p.display().to_string()).collect();
        assert!(names.contains(&"note1.md".to_string()));
        assert!(names.iter().any(|n| n.contains("nested.md")));
        assert!(!names.iter().any(|n| n.contains(".obsidian")));
    }

    #[test]
    fn list_with_glob() {
        let vault = setup_vault();
        let files = list_files(vault.path(), Path::new(""), true, Some("**/*.md")).unwrap();
        for f in &files {
            assert!(f.display().to_string().ends_with(".md"));
        }
        assert!(files.len() >= 3);
    }

    #[test]
    fn list_excludes_obsidian_dir() {
        let vault = setup_vault();
        let files = list_files(vault.path(), Path::new(""), true, None).unwrap();
        for f in &files {
            assert!(
                !f.display().to_string().contains(".obsidian"),
                "should exclude .obsidian: {}",
                f.display()
            );
        }
    }

    #[test]
    fn list_excludes_hidden_files() {
        let vault = setup_vault();
        let files = list_files(vault.path(), Path::new(""), false, None).unwrap();
        for f in &files {
            let name = f.file_name().unwrap().to_string_lossy();
            assert!(!name.starts_with('.'), "should exclude hidden: {name}");
        }
    }

    #[test]
    fn list_nonexistent_dir() {
        let vault = setup_vault();
        let err = list_files(vault.path(), Path::new("no_such_dir"), false, None).unwrap_err();
        assert!(matches!(err, VaultError::DirectoryNotFound(_)));
    }

    #[test]
    fn list_subdirectory() {
        let vault = setup_vault();
        let files = list_files(vault.path(), Path::new("subfolder"), false, None).unwrap();
        assert_eq!(files.len(), 1);
        assert!(files[0].display().to_string().contains("nested.md"));
    }

    #[test]
    fn concurrent_creators_never_overwrite_the_winner() {
        let dir = tempfile::tempdir().unwrap();
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(8));
        let workers: Vec<_> = (0..8)
            .map(|id| {
                let root = dir.path().to_path_buf();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    (
                        id,
                        create_file(&root, Path::new("new.md"), &format!("author-{id}")),
                    )
                })
            })
            .collect();
        let mut winners = Vec::new();
        for worker in workers {
            let (id, result) = worker.join().unwrap();
            match result {
                Ok(_) => winners.push(id),
                Err(VaultError::AlreadyExists(_)) => {}
                other => panic!("unexpected create result: {other:?}"),
            }
        }
        assert_eq!(winners.len(), 1);
        assert_eq!(
            read_file(dir.path(), Path::new("new.md")).unwrap(),
            format!("author-{}", winners[0])
        );
    }
}

#[cfg(test)]
mod confinement_tests {
    use super::*;

    #[test]
    fn traversal_continues_when_an_enumerated_directory_disappears() {
        let temp = tempfile::tempdir().unwrap();
        let root = VaultRoot::open(temp.path()).unwrap();
        std::fs::create_dir(temp.path().join("removed")).unwrap();
        root.create_file(Path::new("kept/note.md"), "visible")
            .unwrap();
        let mut files = Vec::new();
        root.visit_entries(
            Path::new(""),
            true,
            |_| true,
            || Ok(()),
            |path, is_file| {
                if path == Path::new("removed") {
                    std::fs::remove_dir(temp.path().join(path))?;
                } else if is_file {
                    files.push(path.to_owned());
                }
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(files, [Path::new("kept/note.md")]);
    }

    #[cfg(unix)]
    #[test]
    fn moving_a_note_alias_preserves_its_target() {
        use std::os::unix::fs::symlink;
        let temp = tempfile::tempdir().unwrap();
        let root = VaultRoot::open(temp.path()).unwrap();
        root.create_file(Path::new("note.md"), "first").unwrap();
        root.create_file(Path::new("other/note.md"), "second")
            .unwrap();
        let alias = Path::new("alias.md");
        symlink("note.md", temp.path().join(alias)).unwrap();
        for destination in ["other/alias.md", "missing/alias.md"] {
            assert!(matches!(
                root.move_file(alias, Path::new(destination)),
                Err(VaultError::InvalidPath(_))
            ));
            assert_eq!(root.read_file(alias).unwrap(), "first");
            assert!(!temp.path().join(destination).is_symlink());
        }
        assert!(!temp.path().join("missing").exists());

        let absolute = Path::new("absolute.md");
        symlink(temp.path().join("note.md"), temp.path().join(absolute)).unwrap();
        let moved = Path::new("other/moved.md");
        root.move_file(absolute, moved).unwrap();
        assert!(temp.path().join(moved).is_symlink());
        assert_eq!(root.read_file(moved).unwrap(), "first");
    }

    #[test]
    fn racing_moves_never_replace_the_destination() {
        let dir = tempfile::tempdir().unwrap();
        let root = VaultRoot::open(dir.path()).unwrap();
        let barrier = std::sync::Barrier::new(4);
        std::thread::scope(|scope| {
            let mut movers = Vec::new();
            for index in 0..4 {
                let name = format!("source-{index}.md");
                let root = &root;
                root.write_file(Path::new(&name), &name).unwrap();
                let barrier = &barrier;
                movers.push(scope.spawn(move || {
                    barrier.wait();
                    (
                        name.clone(),
                        root.move_file(Path::new(&name), Path::new("winner.md")),
                    )
                }));
            }
            let mut winner = None;
            for worker in movers {
                let (name, result) = worker.join().unwrap();
                match result {
                    Ok(_) => {
                        assert!(winner.replace(name.clone()).is_none());
                        assert!(!dir.path().join(name).exists());
                    }
                    Err(VaultError::AlreadyExists(_)) => {
                        assert_eq!(root.read_file(Path::new(&name)).unwrap(), name)
                    }
                    other => panic!("unexpected move result: {other:?}"),
                }
            }
            assert_eq!(
                root.read_file(Path::new("winner.md")).unwrap(),
                winner.unwrap()
            );
        });
    }

    #[cfg(unix)]
    #[test]
    fn a_replaced_parent_or_leaf_cannot_redirect_an_already_resolved_write() {
        use std::os::unix::fs::symlink;
        use std::sync::Barrier;
        let dir = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let root = VaultRoot::open(dir.path()).unwrap();
        let sentinel = outside.path().join("note.md");
        std::fs::write(&sentinel, "outside").unwrap();

        for replace_parent in [false, true] {
            let subdir = if replace_parent { "parent" } else { "leaf" };
            let relative = Path::new(subdir).join("note.md");
            root.write_file(&relative, "inside").unwrap();
            let barrier = Barrier::new(2);
            std::thread::scope(|scope| {
                let worker = scope.spawn(|| {
                    let target = root.resolve_for_write(&relative).unwrap();
                    barrier.wait();
                    barrier.wait();
                    root.write_resolved(&target.relative, "replacement", WriteMode::Replace)
                });
                barrier.wait();
                if replace_parent {
                    std::fs::rename(dir.path().join(subdir), dir.path().join("old-parent"))
                        .unwrap();
                    symlink(outside.path(), dir.path().join(subdir)).unwrap();
                } else {
                    std::fs::remove_file(dir.path().join(&relative)).unwrap();
                    symlink(&sentinel, dir.path().join(&relative)).unwrap();
                }
                barrier.wait();
                assert!(worker.join().unwrap().is_err());
            });
            assert_eq!(std::fs::read_to_string(&sentinel).unwrap(), "outside");
        }
    }

    #[cfg(unix)]
    #[test]
    fn an_alias_swap_during_open_cannot_pair_one_vault_with_another_root_key() {
        use std::os::unix::fs::symlink;
        let temp = tempfile::tempdir().unwrap();
        let first = temp.path().join("first");
        let second = temp.path().join("second");
        let alias = temp.path().join("alias");
        for vault in [&first, &second] {
            std::fs::create_dir(vault).unwrap();
        }
        symlink(&first, &alias).unwrap();
        let dir = Dir::open_ambient_dir(&alias, cap_std::ambient_authority()).unwrap();
        std::fs::remove_file(&alias).unwrap();
        symlink(&second, &alias).unwrap();
        match VaultRoot::from_directory(dir, &alias) {
            Ok(root) => {
                assert_eq!(root.path(), first.canonicalize().unwrap());
                root.create_file(Path::new("note.md"), "first").unwrap();
                assert_eq!(
                    std::fs::read_to_string(first.join("note.md")).unwrap(),
                    "first"
                );
                assert!(!second.join("note.md").exists());
            }
            Err(VaultError::InvalidPath(_)) => (), // no native handle path on this system
            Err(error) => panic!("unexpected root error: {error}"),
        }
    }

    #[cfg(unix)]
    #[test]
    fn a_retained_vault_handle_does_not_follow_its_replaced_root_name() {
        use std::os::unix::fs::symlink;
        let dir = tempfile::tempdir().unwrap();
        let vault = dir.path().join("vault");
        let original = dir.path().join("original");
        let other = dir.path().join("other");
        std::fs::create_dir(&vault).unwrap();
        std::fs::create_dir(&other).unwrap();
        let root = VaultRoot::open(&vault).unwrap();
        std::fs::rename(&vault, &original).unwrap();
        symlink(&other, &vault).unwrap();
        root.write_file(Path::new("note.md"), "original").unwrap();
        assert_eq!(root.read_file(Path::new("note.md")).unwrap(), "original");
        assert_eq!(
            std::fs::read_to_string(original.join("note.md")).unwrap(),
            "original"
        );
        assert!(!other.join("note.md").exists());
    }

    #[cfg(unix)]
    #[test]
    fn in_vault_links_support_content_and_distinguish_entry_operations() {
        use std::os::unix::fs::symlink;
        let vault = tempfile::tempdir().unwrap();
        let root = VaultRoot::open(vault.path()).unwrap();
        root.create_file(Path::new("folder/target.md"), "A")
            .unwrap();
        symlink("folder/target.md", vault.path().join("alias.md")).unwrap();
        symlink(vault.path().join("folder"), vault.path().join("absolute")).unwrap();
        assert_eq!(
            root.append_file(Path::new("alias.md"), "B").unwrap(),
            Path::new("folder/target.md")
        );
        assert_eq!(
            root.read_file(Path::new("absolute/target.md")).unwrap(),
            "AB"
        );
        assert!(root.open_exact(Path::new("alias.md")).is_err());
        assert!(root.open_exact(Path::new("absolute/target.md")).is_err());
        let moved = root
            .move_file(Path::new("alias.md"), Path::new("alias2.md"))
            .unwrap();
        assert_eq!(moved.from, Path::new("alias.md"));
        assert!(vault.path().join("alias2.md").is_symlink());
        root.delete_file(Path::new("alias2.md")).unwrap();
        assert_eq!(root.read_file(Path::new("folder/target.md")).unwrap(), "AB");
        assert!(
            root.move_file(Path::new("absolute"), Path::new("uncreated/nested"))
                .is_err()
        );
        assert!(!vault.path().join("uncreated").exists());
    }

    #[cfg(unix)]
    #[test]
    fn caller_separators_are_normalized_before_traversal_and_native_names_are_preserved() {
        use crate::vault::path::{normalize_relative, relative_from_absolute};
        let vault = tempfile::tempdir().unwrap();
        assert!(matches!(
            normalize_relative(Path::new(r"..\outside")),
            Err(VaultError::OutsideVault(_))
        ));
        assert_eq!(
            normalize_relative(Path::new(r"dir\note.md")).unwrap(),
            Path::new("dir/note.md")
        );
        let hostile = vault.path().join(r"..\note.md");
        std::fs::write(&hostile, "literal filename").unwrap();
        assert!(matches!(
            relative_from_absolute(vault.path(), &hostile),
            Err(VaultError::InvalidPath(_))
        ));
    }

    #[test]
    fn moved_native_spelling_never_reads_a_new_note_under_an_old_key() {
        let vault = tempfile::tempdir().unwrap();
        let root = VaultRoot::open(vault.path()).unwrap();
        let old = Path::new("visible/note.md");
        root.create_file(old, "old body").unwrap();
        let parent = vault.path().join("visible");
        let renamed = vault.path().join("VISIBLE");
        std::fs::rename(&parent, &renamed).unwrap();
        std::fs::write(renamed.join("note.md"), "private after directory rename").unwrap();
        // On a case-insensitive volume opening the old spelling still succeeds.
        // On a case-sensitive volume the open itself rejects the stale key.
        assert!(root.read_exact_bounded(old, 1024).is_err());
        assert!(root.read_observed(old).is_err());
        let fresh = Path::new("VISIBLE/note.md");
        assert_eq!(
            root.read_observed(fresh).unwrap(),
            "private after directory rename"
        );
        std::fs::rename(renamed.join("note.md"), renamed.join("NOTE.md")).unwrap();
        std::fs::write(renamed.join("NOTE.md"), "private after file rename").unwrap();
        assert!(root.read_observed_with_stat(fresh).is_err());
        assert_eq!(
            root.read_file(fresh).is_ok(),
            renamed.join("note.md").exists()
        );
    }

    #[test]
    fn observed_spelling_is_checked_again_after_the_source_read() {
        let vault = tempfile::tempdir().unwrap();
        let root = VaultRoot::open(vault.path()).unwrap();
        let old = Path::new("Ready.md");
        root.create_file(old, "prior source").unwrap();
        let result = root.read_observed_with(old, |file| {
            std::fs::rename(vault.path().join(old), vault.path().join("ready.md"))?;
            std::fs::write(vault.path().join("ready.md"), "newly private")?;
            read_text(file)
        });
        assert!(result.is_err());
    }

    #[test]
    fn native_unicode_aliases_preserve_the_actual_entry_name() {
        let vault = tempfile::tempdir().unwrap();
        let root = VaultRoot::open(vault.path()).unwrap();
        root.create_file(Path::new("S.md"), "source").unwrap();
        let alias = Path::new("ſ.md");
        if vault.path().join(alias).exists() {
            assert_eq!(root.append_file(alias, " text").unwrap(), Path::new("S.md"));
            assert_eq!(root.read_file(Path::new("S.md")).unwrap(), "source text");
        } else {
            assert!(root.read_file(alias).is_err());
        }
    }

    #[cfg(windows)]
    #[test]
    fn stream_names_cannot_change_a_regular_note() {
        let vault = tempfile::tempdir().unwrap();
        let root = VaultRoot::open(vault.path()).unwrap();
        root.create_file(Path::new("note.md"), "intact").unwrap();
        for stream in [
            "note.md:extra",
            "folder/file.md:extra",
            ":alternate",
            "NUL/child.md",
            "COM1.txt",
        ] {
            assert!(root.write_file(Path::new(stream), "new").is_err());
            assert!(
                root.move_file(Path::new("note.md"), Path::new(stream))
                    .is_err()
            );
        }
        assert!(!vault.path().join("folder").exists());
        assert_eq!(root.read_file(Path::new("note.md")).unwrap(), "intact");
    }

    #[cfg(windows)]
    #[test]
    fn directory_operations_use_the_same_root_over_unc() {
        use std::path::{Component, Prefix};

        let scratch = tempfile::tempdir().unwrap();
        let path = scratch.path().canonicalize().unwrap();
        let mut components = path.components();
        let Some(Component::Prefix(prefix)) = components.next() else {
            panic!("expected drive")
        };
        let (Prefix::VerbatimDisk(letter) | Prefix::Disk(letter)) = prefix.kind() else {
            panic!("expected local drive");
        };
        let mut unc = PathBuf::from(format!(r"\\localhost\{}$", char::from(letter)));
        unc.extend(components.filter(|c| matches!(c, Component::Normal(_))));
        // Hosted Windows runners expose their local administrative share.
        // Other hosts can provide a writable UNC test share explicitly.
        let configured = std::env::var_os("OBSIDIAN_TEST_UNC_ROOT");
        let shared;
        if let Some(directory) = configured {
            shared = tempfile::tempdir_in(directory).unwrap();
            unc = shared.path().to_owned();
        } else if let Err(error) = std::fs::read_dir(&unc) {
            assert!(
                std::env::var_os("GITHUB_ACTIONS").is_none(),
                "Windows CI needs its administrative share or OBSIDIAN_TEST_UNC_ROOT: {error}"
            );
            eprintln!("loopback administrative share unavailable; UNC fixture skipped");
            return;
        }
        assert!(
            matches!(unc.components().next(), Some(Component::Prefix(prefix))
            if matches!(prefix.kind(), Prefix::UNC(_, _) | Prefix::VerbatimUNC(_, _))),
            "UNC fixture requires a network share"
        );
        let root = VaultRoot::open(&unc).unwrap();
        root.write_file(Path::new("parent/note.md"), "network root")
            .unwrap();
        assert_eq!(
            root.list_files(Path::new("parent"), true, None).unwrap(),
            [Path::new("parent/note.md")]
        );
        root.move_file(Path::new("parent/note.md"), Path::new("other/moved.md"))
            .unwrap();
        assert_eq!(
            root.read_observed(Path::new("other/moved.md")).unwrap(),
            "network root"
        );
        assert_eq!(
            std::fs::read_to_string(unc.join("other/moved.md")).unwrap(),
            "network root"
        );
        root.delete_file(Path::new("other/moved.md")).unwrap();
        root.delete_file(Path::new("other")).unwrap();
        assert!(!unc.join("other").exists());
    }

    #[test]
    fn normalization_collision_prefers_exact_disk_entries_and_rejects_ambiguity() {
        let dir = tempfile::tempdir().unwrap();
        let first = "a\u{030a}\u{0301}.md";
        let second = "\u{00e5}\u{0301}.md";
        let equivalent = "\u{01fb}.md";
        std::fs::write(dir.path().join(first), "first").unwrap();
        if dir.path().join(second).exists() {
            return;
        } // Native normalization aliases these names.
        std::fs::write(dir.path().join(second), "second").unwrap();
        let root = VaultRoot::open(dir.path()).unwrap();
        assert_eq!(root.read_file(Path::new(first)).unwrap(), "first");
        assert_eq!(root.read_file(Path::new(second)).unwrap(), "second");
        assert!(matches!(
            root.write_file(Path::new(equivalent), "ambiguous"),
            Err(VaultError::InvalidPath(_))
        ));
        assert_eq!(root.read_file(Path::new(first)).unwrap(), "first");
        assert_eq!(root.read_file(Path::new(second)).unwrap(), "second");
    }
}
