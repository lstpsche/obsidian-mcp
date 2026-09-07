//! Filesystem operations: read, write, list, delete, and rename notes.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use globset::Glob;
use walkdir::WalkDir;

use crate::error::{VaultError, VaultResult};
use crate::models::FileStat;
use crate::vault::path as vault_path;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoveResult {
    pub from: PathBuf,
    pub to: PathBuf,
}

fn is_hidden(name: &str) -> bool {
    name.starts_with('.')
}

fn map_not_found(path: &Path) -> impl FnOnce(std::io::Error) -> VaultError + '_ {
    |e| match e.kind() {
        std::io::ErrorKind::NotFound => VaultError::NoteNotFound(path.to_path_buf()),
        _ => VaultError::Io(e),
    }
}

/// Resolve a relative path against vault root, ensuring it stays within vault.
///
/// For existing paths, canonicalization catches symlink escapes.
/// For non-existent paths, manual normalization is sufficient.
pub fn resolve_path(vault_root: &Path, relative: &Path) -> VaultResult<PathBuf> {
    Ok(vault_path::resolve_for_write(vault_root, relative)?.absolute)
}

/// Check if path exists and is a file.
pub fn file_exists(vault_root: &Path, path: &Path) -> bool {
    vault_path::resolve_existing(vault_root, path)
        .map(|p| p.absolute.is_file())
        .unwrap_or(false)
}

/// Get file stat (size, created, modified).
pub fn file_stat(vault_root: &Path, path: &Path) -> VaultResult<FileStat> {
    let abs = vault_path::resolve_existing(vault_root, path)?.absolute;
    let meta = fs::metadata(&abs).map_err(map_not_found(path))?;

    let created = meta
        .created()
        .ok()
        .map(chrono::DateTime::<chrono::Utc>::from);
    let modified = meta
        .modified()
        .ok()
        .map(chrono::DateTime::<chrono::Utc>::from);

    Ok(FileStat {
        size: meta.len(),
        created,
        modified,
    })
}

/// Read a file's content as a UTF-8 string.
pub fn read_file(vault_root: &Path, path: &Path) -> VaultResult<String> {
    let abs = vault_path::resolve_existing(vault_root, path)?.absolute;
    fs::read_to_string(&abs).map_err(map_not_found(path))
}

/// Write content to a file (creates parent dirs if needed). Overwrites if exists.
pub fn write_file(vault_root: &Path, path: &Path, content: &str) -> VaultResult<PathBuf> {
    let resolved = vault_path::resolve_for_write(vault_root, path)?;
    if let Some(parent) = resolved.absolute.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&resolved.absolute, content)?;
    Ok(resolved.relative)
}

/// Create a file without overwriting an existing path, including concurrent creators.
pub fn create_file(vault_root: &Path, path: &Path, content: &str) -> VaultResult<PathBuf> {
    let resolved = vault_path::resolve_for_write(vault_root, path)?;
    if let Some(parent) = resolved.absolute.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&resolved.absolute)
        .map_err(|error| match error.kind() {
            std::io::ErrorKind::AlreadyExists => VaultError::AlreadyExists(path.to_path_buf()),
            _ => VaultError::Io(error),
        })?;
    file.write_all(content.as_bytes())?;
    Ok(resolved.relative)
}

/// Append content to a file. Creates the file if it doesn't exist.
pub fn append_file(vault_root: &Path, path: &Path, content: &str) -> VaultResult<PathBuf> {
    let resolved = vault_path::resolve_for_write(vault_root, path)?;
    if let Some(parent) = resolved.absolute.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&resolved.absolute)?;
    file.write_all(content.as_bytes())?;
    Ok(resolved.relative)
}

/// Delete a file or empty directory.
pub fn delete_file(vault_root: &Path, path: &Path) -> VaultResult<PathBuf> {
    let resolved = vault_path::resolve_existing(vault_root, path)?;
    if resolved.absolute.is_dir() {
        fs::remove_dir(&resolved.absolute).map_err(map_not_found(path))?;
    } else {
        fs::remove_file(&resolved.absolute).map_err(map_not_found(path))?;
    }
    Ok(resolved.relative)
}

/// Move/rename a file. Returns the actual source and destination relative paths.
pub fn move_file(vault_root: &Path, from: &Path, to: &Path) -> VaultResult<MoveResult> {
    let resolved_from = vault_path::resolve_existing(vault_root, from)?;
    let resolved_to = vault_path::resolve_for_write(vault_root, to)?;
    if resolved_to.absolute.exists() {
        return Err(VaultError::AlreadyExists(to.to_path_buf()));
    }

    if let Some(parent) = resolved_to.absolute.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::rename(&resolved_from.absolute, &resolved_to.absolute)?;

    Ok(MoveResult {
        from: resolved_from.relative,
        to: resolved_to.relative,
    })
}

/// List files/dirs in a directory. Returns paths relative to vault root.
///
/// If `recursive` is true, walks the entire subtree.
/// If `glob` is Some, filters results by glob pattern.
/// Hidden files and `.obsidian/` are excluded.
pub fn list_files(
    vault_root: &Path,
    dir: &Path,
    recursive: bool,
    glob: Option<&str>,
) -> VaultResult<Vec<PathBuf>> {
    let abs_dir = vault_path::resolve_existing(vault_root, dir)
        .map_err(|err| match err {
            VaultError::NoteNotFound(_) => VaultError::DirectoryNotFound(dir.to_path_buf()),
            other => other,
        })?
        .absolute;
    if !abs_dir.is_dir() {
        return Err(VaultError::DirectoryNotFound(dir.to_path_buf()));
    }

    let canonical_root = vault_root
        .canonicalize()
        .map_err(|_| VaultError::InvalidPath(vault_root.display().to_string()))?;

    let glob_matcher = glob
        .map(|pattern| {
            Glob::new(pattern)
                .map(|g| g.compile_matcher())
                .map_err(|e| VaultError::InvalidPath(format!("invalid glob pattern: {e}")))
        })
        .transpose()?;

    let mut results = Vec::new();

    let mut try_add = |entry_path: &Path| -> VaultResult<()> {
        let rel = vault_path::relative_from_absolute(&canonical_root, entry_path)?;
        if let Some(ref matcher) = glob_matcher
            && !matcher.is_match(&rel)
        {
            return Ok(());
        }
        results.push(rel);
        Ok(())
    };

    if recursive {
        for entry in WalkDir::new(&abs_dir)
            .min_depth(1)
            .into_iter()
            .filter_entry(|e| {
                e.file_name()
                    .to_str()
                    .map(|name| !is_hidden(name))
                    .unwrap_or(false)
            })
        {
            let entry = entry.map_err(|e| VaultError::Io(std::io::Error::other(e.to_string())))?;
            try_add(entry.path())?;
        }
    } else {
        for entry in fs::read_dir(&abs_dir)? {
            let entry = entry?;
            let name = entry.file_name();
            if is_hidden(&name.to_string_lossy()) {
                continue;
            }
            try_add(&entry.path())?;
        }
    }

    results.sort();
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
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
