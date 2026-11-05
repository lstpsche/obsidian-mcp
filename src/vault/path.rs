//! Vault-relative path normalization and Unicode-aware filesystem resolution.

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
