//! Vault-relative path normalization and Unicode-aware filesystem resolution.

use std::borrow::Cow;
use std::ffi::OsString;
use std::path::{Component, Path, PathBuf};

use unicode_normalization::UnicodeNormalization;

use super::fs::VaultRoot;
use crate::error::{VaultError, VaultResult};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedPath {
    pub relative: PathBuf,
    pub absolute: PathBuf,
}

/// A logical key in a vault namespace, independent of the client's native OS.
/// Server names are serialized unchanged; only the local adapter applies native
/// filename restrictions.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VaultPath(String);

impl VaultPath {
    pub fn parse(value: &str) -> VaultResult<Self> {
        if value.starts_with(['/', '\\']) || value.chars().any(char::is_control) {
            return Err(VaultError::InvalidPath(
                "expected a vault-relative UTF-8 path".into(),
            ));
        }
        let spelling = value.replace('\\', "/");
        let mut names = Vec::new();
        for name in spelling.split('/') {
            match name {
                "" | "." => {}
                ".." => {
                    if names.pop().is_none() {
                        return Err(VaultError::OutsideVault(value.into()));
                    }
                }
                _ => names.push(name),
            }
        }
        Ok(Self(names.join("/")))
    }

    /// Accept a single name from directory enumeration without reinterpreting it.
    pub fn entry(name: &str) -> VaultResult<Self> {
        if name.is_empty()
            || matches!(name, "." | "..")
            || name.contains(['/', '\\'])
            || name.chars().any(char::is_control)
        {
            return Err(VaultError::InvalidPath(
                "unrepresentable vault directory entry".into(),
            ));
        }
        Ok(Self(name.to_owned()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Use the accepted logical separator spelling without changing name identity.
/// Native Windows joins introduce backslashes, while source indexes store `/`.
/// Avoid native component parsing so drive-like remote names remain exact keys.
pub(crate) fn logical_separators(path: &Path) -> Cow<'_, Path> {
    match path.to_str() {
        Some(raw) if raw.contains('\\') => Cow::Owned(PathBuf::from(raw.replace('\\', "/"))),
        _ => Cow::Borrowed(path),
    }
}

/// Inspect a relative vault key without interpreting a Windows drive prefix.
pub(crate) fn logical_file_name(path: &Path) -> Option<&str> {
    path.to_str()?
        .rsplit(['/', '\\'])
        .next()
        .filter(|name| !name.is_empty())
}

pub(crate) fn logical_file_stem(path: &Path) -> Option<&str> {
    let name = logical_file_name(path)?;
    Some(match name.rsplit_once('.') {
        Some((stem, _)) if !stem.is_empty() => stem,
        _ => name,
    })
}

pub(crate) fn logical_extension(path: &Path) -> Option<&str> {
    let (stem, extension) = logical_file_name(path)?.rsplit_once('.')?;
    (!stem.is_empty()).then_some(extension)
}

pub(crate) fn logical_join(parent: &Path, child: &Path) -> VaultResult<PathBuf> {
    let text = |path: &Path| {
        path.to_str()
            .ok_or_else(|| VaultError::InvalidPath("vault paths must be UTF-8".into()))
            .and_then(VaultPath::parse)
    };
    let parent = text(parent)?;
    let child = text(child)?;
    Ok(PathBuf::from(if parent.as_str().is_empty() {
        child.as_str().to_owned()
    } else if child.as_str().is_empty() {
        parent.as_str().to_owned()
    } else {
        format!("{}/{}", parent.as_str(), child.as_str())
    }))
}

/// Normalize a caller-supplied path, including its accepted separator spelling,
/// before checking traversal. Paths discovered from the filesystem use
/// `relative_from_absolute` and never reinterpret a literal filename.
pub fn normalize_relative(path: &Path) -> VaultResult<PathBuf> {
    let text = path
        .to_str()
        .ok_or_else(|| VaultError::InvalidPath("vault-relative paths must be UTF-8".into()))?;
    let portable = text.replace('\\', "/");
    let mut names = Vec::new();
    for component in Path::new(&portable).components() {
        match component {
            Component::Normal(name) => {
                #[cfg(windows)]
                validate_windows_name(
                    name.to_str()
                        .ok_or_else(|| VaultError::InvalidPath("invalid vault name".into()))?,
                )?;
                names.push(name);
            }
            Component::CurDir => {}
            Component::ParentDir => {
                if names.pop().is_none() {
                    return Err(VaultError::OutsideVault(path.to_path_buf()));
                }
            }
            Component::RootDir | Component::Prefix(_) => {
                return Err(VaultError::InvalidPath(format!(
                    "absolute paths not allowed: {}",
                    path.display()
                )));
            }
        }
    }
    Ok(PathBuf::from(names.join(std::ffi::OsStr::new("/"))))
}

/// Keep the exact disk identity when serializing a native relative name.
pub(super) fn portable_native(path: &Path) -> VaultResult<PathBuf> {
    let mut result = OsString::new();
    for component in path.components() {
        let Component::Normal(name) = component else {
            return Err(VaultError::InvalidPath(format!(
                "expected a physical vault-relative path: {}",
                path.display()
            )));
        };
        let text = name
            .to_str()
            .filter(|name| !name.contains('\\'))
            .ok_or_else(|| {
                VaultError::InvalidPath(format!(
                    "unrepresentable vault filename: {}",
                    name.display()
                ))
            })?;
        #[cfg(windows)]
        validate_windows_name(text)?;
        if !result.is_empty() {
            result.push("/");
        }
        result.push(text);
    }
    Ok(result.into())
}

#[cfg(windows)]
fn validate_windows_name(name: &str) -> VaultResult<()> {
    // Native relative directory creation and rename operate below Win32's DOS
    // name translation. Refuse alternate data streams and device aliases
    // before they can acquire a different meaning in another operation.
    let stem = name
        .split('.')
        .next()
        .unwrap_or(name)
        .trim_end()
        .to_ascii_uppercase();
    let numbered = stem
        .strip_prefix("COM")
        .or_else(|| stem.strip_prefix("LPT"));
    if name.contains(':')
        || matches!(
            stem.as_str(),
            "NUL" | "CON" | "AUX" | "PRN" | "CONIN$" | "CONOUT$"
        )
        || matches!(
            numbered,
            Some("0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | "¹" | "²" | "³")
        )
    {
        return Err(VaultError::InvalidPath(format!(
            "reserved Windows vault filename: {name}"
        )));
    }
    Ok(())
}

pub fn relative_from_absolute(vault_root: &Path, absolute: &Path) -> VaultResult<PathBuf> {
    let relative = absolute
        .strip_prefix(vault_root)
        .map_err(|_| VaultError::OutsideVault(absolute.to_path_buf()))?;
    portable_native(relative)
}

pub fn canonical_unicode_key(s: &str) -> String {
    s.nfc().collect()
}

/// Return the current physical spelling for reporting or index identity.
/// Perform filesystem operations through the rooted handle.
pub fn resolve_existing(vault_root: &Path, relative: &Path) -> VaultResult<ResolvedPath> {
    VaultRoot::open(vault_root)?.resolve_existing(relative)
}

pub fn resolve_for_write(vault_root: &Path, relative: &Path) -> VaultResult<ResolvedPath> {
    VaultRoot::open(vault_root)?.resolve_for_write(relative)
}

pub fn resolve_parent_for_write(
    vault_root: &Path,
    relative: &Path,
) -> VaultResult<(PathBuf, PathBuf)> {
    let root = VaultRoot::open(vault_root)?;
    let resolved = root.resolve_for_write(relative)?;
    let parent = resolved.relative.parent().unwrap_or(Path::new(""));
    Ok((parent.to_owned(), root.path().join(parent)))
}

#[cfg(test)]
mod tests {
    use super::*;

    use tempfile::TempDir;
    use unicode_normalization::UnicodeNormalization;

    #[test]
    fn logical_server_paths_do_not_apply_client_filename_rules() {
        assert_eq!(
            VaultPath::parse("Folder/CON.md").unwrap().as_str(),
            "Folder/CON.md"
        );
        assert_eq!(VaultPath::parse("a:b.md").unwrap().as_str(), "a:b.md");
        assert_eq!(VaultPath::parse("a\\b/../c.md").unwrap().as_str(), "a/c.md");
        assert!(VaultPath::parse("../secret").is_err());
        assert!(VaultPath::parse("/absolute").is_err());
        assert!(VaultPath::parse("\\absolute").is_err());
        assert!(VaultPath::parse("\\\\server\\share").is_err());
        for name in ["..", "a/b", "a\\b", "", "bad\0name"] {
            assert!(VaultPath::entry(name).is_err());
        }
        assert_ne!(
            VaultPath::entry("café.md").unwrap(),
            VaultPath::entry("cafe\u{301}.md").unwrap()
        );
    }

    #[test]
    fn logical_separator_spelling_preserves_exact_names() {
        let native = Path::new("notes").join("re\u{301}sume\u{301}.md");
        assert_eq!(
            logical_separators(&native).as_os_str(),
            std::ffi::OsStr::new("notes/re\u{301}sume\u{301}.md")
        );
        assert_eq!(
            logical_separators(Path::new("notes\\nested\\entry.md")).as_os_str(),
            std::ffi::OsStr::new("notes/nested/entry.md")
        );
        assert_ne!(
            logical_separators(Path::new("a:b.md")).as_os_str(),
            logical_separators(Path::new("A:b.md")).as_os_str()
        );
        assert_ne!(
            logical_separators(Path::new("notes/café.md")).as_os_str(),
            logical_separators(Path::new("notes/cafe\u{301}.md")).as_os_str()
        );
        assert_eq!(
            logical_separators(Path::new("a:b\\CON.md")).as_os_str(),
            std::ffi::OsStr::new("a:b/CON.md")
        );
    }

    #[test]
    fn logical_key_helpers_preserve_server_colons_and_reserved_names() {
        let path = Path::new("a:b.md");
        assert_eq!(logical_file_name(path), Some("a:b.md"));
        assert_eq!(logical_file_stem(path), Some("a:b"));
        assert_eq!(
            logical_join(Path::new("notes"), path).unwrap(),
            PathBuf::from("notes/a:b.md")
        );
        assert_eq!(logical_file_stem(Path::new(".config")), Some(".config"));
        assert!(logical_join(Path::new("notes"), Path::new("../escape")).is_err());
    }

    fn setup_vault() -> TempDir {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir_all(dir.path().join(".obsidian")).unwrap();
        dir
    }

    #[test]
    fn normalize_rejects_parent_escape() {
        let err = normalize_relative(Path::new("../secret.md")).unwrap_err();
        assert!(matches!(err, VaultError::OutsideVault(_)));
    }

    #[test]
    fn normalize_rejects_absolute_path() {
        let err = normalize_relative(Path::new("/tmp/secret.md")).unwrap_err();
        assert!(matches!(err, VaultError::InvalidPath(_)));
    }

    #[test]
    fn resolve_existing_finds_decomposed_filename_from_composed_input() {
        let vault = setup_vault();
        let composed = "02_База-знаний/Сущности/lic1c.md";
        let decomposed: String = composed.nfd().collect();
        let disk_path = PathBuf::from(&decomposed);
        std::fs::create_dir_all(vault.path().join(disk_path.parent().unwrap())).unwrap();
        std::fs::write(vault.path().join(&disk_path), "# License\n").unwrap();

        let resolved = resolve_existing(vault.path(), Path::new(composed)).unwrap();
        assert_eq!(resolved.relative, disk_path);
        assert!(resolved.absolute.is_file());
    }

    #[test]
    fn resolve_for_write_uses_existing_decomposed_parent() {
        let vault = setup_vault();
        let composed_dir = "02_База-знаний";
        let decomposed_dir: String = composed_dir.nfd().collect();
        std::fs::create_dir_all(vault.path().join(&decomposed_dir)).unwrap();

        let resolved = resolve_for_write(vault.path(), Path::new("02_База-знаний/New.md")).unwrap();
        assert_eq!(
            resolved.relative,
            PathBuf::from(format!("{decomposed_dir}/New.md"))
        );
        assert!(
            resolved
                .absolute
                .ends_with(Path::new(&decomposed_dir).join("New.md"))
        );
    }

    #[test]
    fn resolve_for_write_preserves_new_final_component_spelling() {
        let vault = setup_vault();
        let path = Path::new("New Folder/café.md");

        let resolved = resolve_for_write(vault.path(), path).unwrap();
        assert_eq!(resolved.relative, path);
    }

    #[test]
    fn relative_from_absolute_returns_forward_slashes() {
        let vault = setup_vault();
        let absolute = vault.path().join("a").join("b.md");
        std::fs::create_dir_all(absolute.parent().unwrap()).unwrap();
        std::fs::write(&absolute, "# B\n").unwrap();

        let relative = relative_from_absolute(vault.path(), &absolute).unwrap();
        assert_eq!(relative, PathBuf::from("a/b.md"));
    }
}
