//! Portable vault configuration and machine-owned data namespaces.

use std::path::{Path, PathBuf};

use super::exclude::{ExcludeSet, ignore_rule, read_ignore};
use super::fs::VaultRoot;
use crate::config::{vault_slug, vault_storage_key};
use crate::error::{VaultError, VaultResult};

const PORTABLE: &str = ".obsidian-mcp";

pub(super) struct DataLocation {
    pub root: VaultRoot,
    pub directory: PathBuf,
}

impl DataLocation {
    fn portable(root: &VaultRoot) -> Self {
        Self {
            root: root.clone(),
            directory: PORTABLE.into(),
        }
    }

    pub fn path(&self) -> PathBuf {
        self.root.path().join(&self.directory)
    }

    fn machine_policy(&self) -> VaultResult<Option<ExcludeSet>> {
        read_ignore(&self.root, &self.directory.join("ignore"))
    }
}

pub(super) fn initialize(
    vault: &VaultRoot,
    external: Option<&Path>,
    restrictions: &[String],
) -> VaultResult<(DataLocation, ExcludeSet)> {
    vault.create_dir_all(Path::new(PORTABLE))?;
    let ignore = Path::new(".obsidian-mcp/ignore");
    match vault.create_file(ignore, "") {
        Ok(_) | Err(VaultError::AlreadyExists(_)) => {}
        Err(err) => return Err(err),
    }
    let data = select_location(vault, external, true)?;
    let rules = load_policy(vault, &data, restrictions)?;
    data.root
        .create_dir_all(&data.directory.join("embeddings"))?;
    Ok((data, rules))
}

pub(super) fn effective_policy(
    vault: &VaultRoot,
    external: Option<&Path>,
    restrictions: &[String],
) -> VaultResult<ExcludeSet> {
    let data = select_location(vault, external, false)?;
    load_policy(vault, &data, restrictions)
}

fn load_policy(
    vault: &VaultRoot,
    data: &DataLocation,
    restrictions: &[String],
) -> VaultResult<ExcludeSet> {
    let mut rules = restrictions.to_vec();
    if let Some(portable) = read_ignore(vault, Path::new(".obsidian-mcp/ignore"))? {
        rules.extend_from_slice(portable.patterns());
    }
    if data.directory != Path::new(PORTABLE) {
        // The same observation authorizes migration and supplies the policy.
        // Reading it again could turn a vanished confirmation into an empty view.
        match data.machine_policy()? {
            Some(machine) => rules.extend_from_slice(machine.patterns()),
            None => {
                if let Some(legacy) = unclaimed_configuration(&data.root, vault.path())? {
                    return Err(VaultError::InvalidPath(format!(
                        "cannot identify the vault that owns the legacy configuration at {}. Place the intended ignore rules, or an explicitly empty ignore file, at {} before indexing",
                        legacy.display(),
                        data.path().join("ignore").display()
                    )));
                }
            }
        }
    }
    ExcludeSet::build(rules)
}

fn select_location(
    vault: &VaultRoot,
    external: Option<&Path>,
    initialize: bool,
) -> VaultResult<DataLocation> {
    let Some(external) = external else {
        return Ok(DataLocation::portable(vault));
    };
    let portable = vault.path().join(PORTABLE);
    if external == portable {
        return Ok(DataLocation::portable(vault));
    }
    if initialize {
        // This is the explicit operator-provided root. Every path beneath it
        // is opened through its capability, including the old namespace.
        std::fs::create_dir_all(external)?;
    }
    let machine = match VaultRoot::open(external) {
        Ok(root) => root,
        Err(err) if err.is_not_found() && !initialize => {
            // A missing machine root is empty. A dangling link is a broken
            // configuration, including when it occurs above the final entry.
            for component in external.ancestors().filter(|p| !p.as_os_str().is_empty()) {
                match std::fs::symlink_metadata(component) {
                    Ok(metadata) if metadata.is_symlink() && component.canonicalize().is_err() => {
                        return Err(VaultError::InvalidPath(format!(
                            "unresolved machine data root {}",
                            component.display()
                        )));
                    }
                    Ok(_) => break,
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                    Err(error) => return Err(error.into()),
                }
            }
            return Ok(DataLocation::portable(vault));
        }
        Err(err) => return Err(err),
    };
    match vault.open_dir(Path::new(PORTABLE)) {
        Ok(directory) if machine.same_directory(&directory)? => {
            return Ok(DataLocation::portable(vault));
        }
        Ok(_) => {}
        Err(err) if err.is_not_found() => {}
        Err(err) => return Err(err),
    }
    Ok(DataLocation {
        root: machine,
        directory: Path::new("vaults-v2").join(vault_storage_key(vault.path())),
    })
}

fn unclaimed_configuration(root: &VaultRoot, vault: &Path) -> VaultResult<Option<PathBuf>> {
    let parent = match root.open_dir(Path::new("vaults")) {
        Ok(dir) => dir,
        Err(err) if err.is_not_found() => return Ok(None),
        Err(err) => return Err(err),
    };
    let slug = vault_slug(vault);
    // Native lookup also recognizes aliases on case-insensitive volumes.
    // Older slugs were bounded in characters and can exceed NAME_MAX bytes.
    if !slug.is_empty() {
        match parent.symlink_metadata(&slug) {
            Ok(_) => {}
            Err(err) if err.kind() == std::io::ErrorKind::NotFound || name_too_long(&err) => {
                return Ok(None);
            }
            Err(err) => return Err(err.into()),
        }
    }
    let relative = Path::new("vaults").join(slug);
    let dir = root.open_dir(&relative)?;
    for item in super::fs::dir_entries(&dir)? {
        let item = item?;
        let name = item.file_name();
        let is_disposable = if name == "ignore" && item.file_type()?.is_file() {
            let reader = std::io::BufReader::new(root.open_exact(&relative.join("ignore"))?);
            legacy_ignore_is_empty(reader)?
        } else if name == "embeddings" && item.file_type()?.is_dir() {
            let mut cache_only = true;
            for cached in super::fs::dir_entries(&root.open_dir(&relative.join("embeddings"))?)? {
                let cached = cached?;
                if cached.file_name() != "embeddings.bin" || !cached.file_type()?.is_file() {
                    cache_only = false;
                    break;
                }
            }
            cache_only
        } else {
            false
        };
        if !is_disposable {
            return Ok(Some(root.path().join(&relative)));
        }
    }
    Ok(None)
}

fn legacy_ignore_is_empty(reader: impl std::io::BufRead) -> std::io::Result<bool> {
    for line in reader.split(b'\n') {
        // Undecodable legacy config needs the same ownership confirmation as a rule.
        if !std::str::from_utf8(&line?).is_ok_and(|text| ignore_rule(text).is_none()) {
            return Ok(false);
        }
    }
    Ok(true)
}

fn name_too_long(error: &std::io::Error) -> bool {
    #[cfg(unix)]
    {
        error.raw_os_error() == Some(libc::ENAMETOOLONG)
    }
    #[cfg(windows)]
    {
        error.raw_os_error() == Some(206)
    } // ERROR_FILENAME_EXCED_RANGE
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::test_config;
    use crate::vault::Vault;

    #[tokio::test]
    async fn vault_merges_and_normalizes_all_ignore_sources() {
        let vault = tempfile::tempdir().unwrap();
        let machine = tempfile::tempdir().unwrap();
        let portable = vault.path().join(PORTABLE);
        std::fs::create_dir(&portable).unwrap();
        std::fs::write(portable.join("ignore"), "Archive/\n# local comment\n").unwrap();
        let specific = machine
            .path()
            .join("vaults-v2")
            .join(vault_storage_key(&vault.path().canonicalize().unwrap()));
        std::fs::create_dir_all(&specific).unwrap();
        std::fs::write(specific.join("ignore"), "Archive/**\nDrafts/\n").unwrap();
        let mut config = test_config(vault.path());
        config.mcp_data_dir = Some(machine.path().to_owned());
        config.exclude_patterns = vec!["Drafts/**".into(), "Temp/".into()];
        let opened = Vault::open(&config).await.unwrap();
        assert_eq!(
            opened.exclude().patterns(),
            ["Archive/**", "Drafts/**", "Temp/**"]
        );
    }

    #[tokio::test]
    async fn colliding_legacy_vaults_require_independent_machine_policies() {
        let temp = tempfile::tempdir().unwrap();
        let first = temp.path().join("a-b/c");
        let second = temp.path().join("a/b-c");
        let machine = temp.path().join("machine");
        let mut configs = Vec::new();
        for root in [&first, &second] {
            std::fs::create_dir_all(root).unwrap();
            std::fs::write(root.join("visible.md"), "one").unwrap();
            std::fs::write(root.join("private.md"), "two").unwrap();
            let mut config = test_config(root);
            config.mcp_data_dir = Some(machine.clone());
            configs.push(config);
        }
        let key_a = first.canonicalize().unwrap();
        let key_b = second.canonicalize().unwrap();
        assert_eq!(vault_slug(&key_a), vault_slug(&key_b));
        let old = machine.join("vaults").join(vault_slug(&key_a));
        std::fs::create_dir_all(&old).unwrap();
        std::fs::write(old.join("ignore"), "private.md\n").unwrap();
        let new = machine.join("vaults-v2");
        let dest_a = new.join(vault_storage_key(&key_a));
        let dest_b = new.join(vault_storage_key(&key_b));
        assert_ne!(dest_a, dest_b);
        for config in configs.iter().rev() {
            let error = Vault::open(config).await.err().unwrap().to_string();
            assert!(error.contains("legacy configuration") && error.contains("vaults-v2"));
        }
        assert!(!new.exists());
        std::fs::create_dir_all(&dest_a).unwrap();
        std::fs::copy(old.join("ignore"), dest_a.join("ignore")).unwrap();
        assert!(Vault::open(&configs[1]).await.is_err());
        std::fs::create_dir_all(&dest_b).unwrap();
        std::fs::write(dest_b.join("ignore"), "").unwrap();
        for _ in 0..2 {
            let (a, b) = tokio::join!(Vault::open(&configs[0]), Vault::open(&configs[1]));
            let a = a.unwrap();
            let b = b.unwrap();
            assert_eq!(a.indexed_paths(), ["visible.md"]);
            assert_eq!(b.indexed_paths().len(), 2);
            assert!(b.get_note_metadata(Path::new("private.md")).is_ok());
            assert_eq!(a.mcp_data(), dest_a.canonicalize().unwrap());
            assert_eq!(b.mcp_data(), dest_b.canonicalize().unwrap());
        }
        assert_eq!(
            std::fs::read_to_string(old.join("ignore")).unwrap(),
            "private.md\n"
        );
    }

    #[tokio::test]
    async fn legacy_cache_only_state_does_not_block_a_cold_open() {
        let temp = tempfile::tempdir().unwrap();
        let vault = temp.path().join("vault");
        std::fs::create_dir(&vault).unwrap();
        let machine = temp.path().join("machine");
        let old = machine
            .join("vaults")
            .join(vault_slug(&vault.canonicalize().unwrap()));
        std::fs::create_dir_all(old.join("embeddings")).unwrap();
        std::fs::write(old.join("embeddings/embeddings.bin"), "old").unwrap();
        std::fs::write(old.join("ignore"), "\u{2003}# user comment\r\n\u{a0}\n").unwrap();
        let mut config = test_config(&vault);
        config.mcp_data_dir = Some(machine);
        let opened = Vault::open(&config).await.unwrap();
        assert!(opened.mcp_data().join("embeddings").is_dir());
        assert!(!opened.mcp_data().join("ignore").exists());
        assert_eq!(
            std::fs::read(old.join("embeddings/embeddings.bin")).unwrap(),
            b"old"
        );
        drop(opened);
        std::fs::write(old.join("ignore"), b"#\xff").unwrap();
        assert!(
            matches!(Vault::open(&config).await, Err(VaultError::InvalidPath(message)) if message.contains("cannot identify"))
        );
        std::fs::write(old.join("ignore"), "").unwrap();
        std::fs::write(old.join("custom-settings"), "owned").unwrap();
        assert!(Vault::open(&config).await.is_err());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn metadata_links_cannot_read_or_create_external_configuration() {
        use std::os::unix::fs::symlink;
        let vault = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let metadata = vault.path().join(".obsidian-mcp");
        let config = test_config(vault.path());
        symlink(outside.path(), &metadata).unwrap();
        assert!(Vault::open(&config).await.is_err());
        assert!(std::fs::read_dir(outside.path()).unwrap().next().is_none());
        std::fs::remove_file(&metadata).unwrap();
        std::fs::create_dir(&metadata).unwrap();
        let sentinel = outside.path().join("private");
        std::fs::write(&sentinel, "PRIVATE CONTENT\n").unwrap();
        symlink(&sentinel, metadata.join("ignore")).unwrap();
        let diagnostic = Vault::open(&config).await.err().unwrap().to_string();
        assert!(!diagnostic.contains("PRIVATE CONTENT"));
        assert_eq!(
            std::fs::read_to_string(&sentinel).unwrap(),
            "PRIVATE CONTENT\n"
        );
        std::fs::remove_file(&sentinel).unwrap();
        assert!(Vault::open(&config).await.is_err());
        assert!(!sentinel.exists());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn explicit_alias_of_portable_data_reuses_the_same_directory() {
        let vault = tempfile::tempdir().unwrap();
        let machine = tempfile::tempdir().unwrap();
        let portable = vault.path().join(PORTABLE);
        std::fs::create_dir(&portable).unwrap();
        std::fs::write(portable.join("ignore"), "private.md\n").unwrap();
        std::fs::write(vault.path().join("private.md"), "excluded").unwrap();
        let alias = machine.path().join("alias");
        std::os::unix::fs::symlink(&portable, &alias).unwrap();
        let mut config = test_config(vault.path());
        config.mcp_data_dir = Some(alias);
        let opened = Vault::open(&config).await.unwrap();
        assert_eq!(opened.mcp_data(), &portable.canonicalize().unwrap());
        assert!(!portable.join("vaults-v2").exists());
        assert_eq!(opened.exclude().patterns(), ["private.md"]);
    }

    #[cfg(unix)]
    #[test]
    fn storage_identity_preserves_non_unicode_and_canonical_aliases() {
        use std::os::unix::{ffi::OsStrExt, fs::symlink};
        let temp = tempfile::tempdir().unwrap();
        let canonical = temp.path().canonicalize().unwrap();
        let a = canonical.join(std::ffi::OsStr::from_bytes(&[b'r', 0x80]));
        let b = canonical.join(std::ffi::OsStr::from_bytes(&[b'r', 0x81]));
        assert_eq!(a.to_string_lossy(), b.to_string_lossy());
        assert_ne!(vault_storage_key(&a), vault_storage_key(&b));
        // Linux permits these native names. APFS rejects invalid UTF-8 at mkdir.
        #[cfg(target_os = "linux")]
        for root in [&a, &b] {
            std::fs::create_dir(root).unwrap();
            assert_eq!(
                vault_storage_key(root),
                vault_storage_key(&root.canonicalize().unwrap())
            );
        }
        let root = temp.path().join("vault");
        std::fs::create_dir(&root).unwrap();
        let alias = temp.path().join("alias");
        symlink(&root, &alias).unwrap();
        assert_eq!(
            vault_storage_key(&root.canonicalize().unwrap()),
            vault_storage_key(&alias.canonicalize().unwrap())
        );
        let long = Path::new("/").join("界".repeat(180));
        assert_eq!(vault_storage_key(&long).len(), 64);
    }
}
