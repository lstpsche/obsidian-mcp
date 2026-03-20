use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use obsidian_mcp::config::Config;
use obsidian_mcp::models::{NotePeriod, PatchOperation, PatchRequest, PatchTargetType};
use obsidian_mcp::vault::Vault;

fn fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("test_vault")
}

fn fixture_config() -> Config {
    Config {
        vault_path: fixture_path(),
        watch: false,
        log_level: "error".into(),
        tantivy: false,
        embeddings: false,
        embeddings_model: String::new(),
    }
}

static VAULT: LazyLock<Vault> = LazyLock::new(|| {
    tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(Vault::open(&fixture_config()))
        .expect("failed to open fixture vault")
});

async fn copy_fixture_to_temp() -> (tempfile::TempDir, Vault) {
    let tmp = tempfile::tempdir().unwrap();
    copy_dir_recursive(&fixture_path(), tmp.path());
    let config = Config {
        vault_path: tmp.path().to_path_buf(),
        watch: false,
        log_level: "error".into(),
        tantivy: false,
        embeddings: false,
        embeddings_model: String::new(),
    };
    let vault = Vault::open(&config)
        .await
        .expect("failed to open temp vault");
    (tmp, vault)
}

fn copy_dir_recursive(src: &Path, dst: &Path) {
    for entry in walkdir::WalkDir::new(src) {
        let entry = entry.unwrap();
        let rel = entry.path().strip_prefix(src).unwrap();
        let target = dst.join(rel);
        if entry.file_type().is_dir() {
            std::fs::create_dir_all(&target).unwrap();
        } else {
            if let Some(parent) = target.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            std::fs::copy(entry.path(), &target).unwrap();
        }
    }
}

// ── Read operations ──────────────────────────────────────────────────────

mod vault_read {
    use super::*;

    #[test]
    fn list_files_root() {
        let files = VAULT.list_files(Path::new(""), true, None).unwrap();
        assert!(
            files.len() >= 7,
            "expected at least 7 files, got {}",
            files.len()
        );
    }

    #[test]
    fn list_files_subdirectory() {
        let files = VAULT
            .list_files(Path::new("Projects"), false, None)
            .unwrap();
        assert!(files.iter().any(|f| f.ends_with("rust-mcp.md")));
        assert!(files.iter().any(|f| f.ends_with("python-tools.md")));
    }

    #[test]
    fn list_files_glob() {
        let files = VAULT
            .list_files(Path::new(""), true, Some("**/*.md"))
            .unwrap();
        assert!(
            files
                .iter()
                .all(|f| f.extension().is_some_and(|e| e == "md"))
        );
        assert!(!files.is_empty());
    }

    #[test]
    fn read_note_content() {
        let content = VAULT.read_note(Path::new("Projects/rust-mcp.md")).unwrap();
        assert!(content.contains("# Rust MCP Server"));
        assert!(content.contains("tags: [rust, mcp, project]"));
    }

    #[test]
    fn read_nested_note() {
        let content = VAULT
            .read_note(Path::new("Notes/deep/nested-note.md"))
            .unwrap();
        assert!(content.contains("# Nested Note"));
    }

    #[test]
    fn note_metadata() {
        let meta = VAULT
            .get_note_metadata(Path::new("Projects/rust-mcp.md"))
            .unwrap();
        assert_eq!(meta.title, "rust-mcp");
        assert!(meta.tags.contains(&"rust".to_string()));
        assert!(meta.tags.contains(&"mcp".to_string()));
        assert!(meta.tags.contains(&"backend".to_string()));
        assert!(!meta.headings.is_empty());
        assert!(!meta.links.is_empty());
        assert!(!meta.block_refs.is_empty());
    }

    #[test]
    fn document_map() {
        let map = VAULT
            .get_document_map(Path::new("Projects/rust-mcp.md"))
            .unwrap();
        assert!(map.headings.iter().any(|h| h.contains("Rust MCP Server")));
        assert!(map.headings.iter().any(|h| h.contains("Architecture")));
        assert!(map.block_refs.contains(&"intro".to_string()));
        assert!(map.block_refs.contains(&"impl".to_string()));
        assert!(map.frontmatter_fields.contains(&"tags".to_string()));
        assert!(map.frontmatter_fields.contains(&"status".to_string()));
    }

    #[test]
    fn vault_stats() {
        let stats = VAULT.vault_stats().unwrap();
        assert!(stats.total_notes >= 7);
        assert!(stats.total_tags > 0);
        assert!(stats.total_links > 0);
    }
}

// ── Search operations ────────────────────────────────────────────────────

mod vault_search {
    use super::*;

    #[test]
    fn search_text_finds_match() {
        let results = VAULT.search_text("quantum entanglement", 40).unwrap();
        assert!(!results.is_empty());
        assert!(results.iter().any(|r| r.path == PathBuf::from("orphan.md")));
    }

    #[test]
    fn search_text_case_insensitive() {
        let results = VAULT.search_text("RUST MCP SERVER", 40).unwrap();
        assert!(
            results
                .iter()
                .any(|r| r.path == PathBuf::from("Projects/rust-mcp.md"))
        );
    }

    #[test]
    fn search_text_no_match() {
        let results = VAULT
            .search_text("xyzzy_nonexistent_term_12345", 40)
            .unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn search_regex() {
        let results = VAULT.search_regex(r"#\w+", 40).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn search_by_tag_exact() {
        let notes = VAULT.search_by_tag("rust").unwrap();
        assert!(
            notes
                .iter()
                .any(|n| n.path == PathBuf::from("Projects/rust-mcp.md"))
        );
    }

    #[test]
    fn search_by_tag_prefix() {
        let notes = VAULT.search_by_tag_prefix("project").unwrap();
        assert!(
            notes
                .iter()
                .any(|n| n.path == PathBuf::from("Projects/rust-mcp.md"))
        );
        assert!(
            notes
                .iter()
                .any(|n| n.path == PathBuf::from("Projects/python-tools.md"))
        );
    }

    #[test]
    fn search_frontmatter_exact() {
        let notes = VAULT
            .search_frontmatter("status", &serde_json::json!("active"))
            .unwrap();
        assert!(
            notes
                .iter()
                .any(|n| n.path == PathBuf::from("Projects/rust-mcp.md"))
        );
    }

    #[test]
    fn search_frontmatter_exists() {
        let notes = VAULT.search_frontmatter_exists("priority").unwrap();
        assert!(
            notes
                .iter()
                .any(|n| n.path == PathBuf::from("Projects/python-tools.md"))
        );
    }

    #[test]
    fn search_frontmatter_contains() {
        let notes = VAULT
            .search_frontmatter_contains("tags", &serde_json::json!("python"))
            .unwrap();
        assert!(
            notes
                .iter()
                .any(|n| n.path == PathBuf::from("Projects/python-tools.md"))
        );
    }
}

// ── Graph operations ─────────────────────────────────────────────────────

mod vault_graph {
    use super::*;

    #[test]
    fn backlinks() {
        let backlinks = VAULT.backlinks(Path::new("Projects/rust-mcp.md")).unwrap();
        let paths: Vec<_> = backlinks.iter().map(|n| &n.path).collect();
        assert!(paths.contains(&&PathBuf::from("Projects/python-tools.md")));
        assert!(paths.contains(&&PathBuf::from("Notes/getting-started.md")));
        assert!(paths.contains(&&PathBuf::from("Daily/2026-03-19.md")));
    }

    #[test]
    fn outgoing_links() {
        let links = VAULT
            .outgoing_links(Path::new("Projects/rust-mcp.md"))
            .unwrap();
        let targets: Vec<_> = links.iter().map(|l| l.target.as_str()).collect();
        assert!(targets.contains(&"getting-started"));
        assert!(targets.contains(&"python-tools"));
    }

    #[test]
    fn broken_links() {
        let broken = VAULT.broken_links().unwrap();
        let broken_targets: Vec<_> = broken.iter().map(|(_, l)| l.target.as_str()).collect();
        assert!(broken_targets.contains(&"nonexistent-page"));
        assert!(broken_targets.contains(&"another-missing-note"));
    }

    #[test]
    fn orphan_notes() {
        let orphans = VAULT.orphan_notes().unwrap();
        assert!(
            orphans.iter().any(|n| n.path == PathBuf::from("orphan.md")),
            "orphan.md should be detected as orphan, got: {:?}",
            orphans.iter().map(|n| &n.path).collect::<Vec<_>>()
        );
    }

    #[test]
    fn link_resolution() {
        let resolved = VAULT.resolve_link("rust-mcp");
        assert_eq!(resolved, Some(PathBuf::from("Projects/rust-mcp.md")));

        let unresolved = VAULT.resolve_link("nonexistent-page");
        assert!(unresolved.is_none());
    }
}

// ── Write operations (temp copies) ───────────────────────────────────────

mod vault_write {
    use super::*;

    #[tokio::test]
    async fn create_and_read() {
        let (_tmp, vault) = copy_fixture_to_temp().await;
        vault
            .create_note(Path::new("new-note.md"), "# New Note\nBody\n", None)
            .unwrap();
        let content = vault.read_note(Path::new("new-note.md")).unwrap();
        assert!(content.contains("# New Note"));
    }

    #[tokio::test]
    async fn create_with_frontmatter() {
        let (_tmp, vault) = copy_fixture_to_temp().await;
        let fm = serde_json::json!({"tags": ["test"], "draft": true});
        vault
            .create_note(Path::new("fm-note.md"), "Body\n", Some(&fm))
            .unwrap();
        let content = vault.read_note(Path::new("fm-note.md")).unwrap();
        assert!(content.starts_with("---\n"));
        assert!(content.contains("Body\n"));
    }

    #[tokio::test]
    async fn create_fails_if_exists() {
        let (_tmp, vault) = copy_fixture_to_temp().await;
        let err = vault
            .create_note(Path::new("Projects/rust-mcp.md"), "dup", None)
            .unwrap_err();
        assert!(
            matches!(err, obsidian_mcp::error::VaultError::AlreadyExists(_)),
            "expected AlreadyExists, got: {err:?}"
        );
    }

    #[tokio::test]
    async fn write_overwrites() {
        let (_tmp, vault) = copy_fixture_to_temp().await;
        vault
            .write_note(Path::new("orphan.md"), "# Replaced\n")
            .unwrap();
        let content = vault.read_note(Path::new("orphan.md")).unwrap();
        assert_eq!(content, "# Replaced\n");
    }

    #[tokio::test]
    async fn append() {
        let (_tmp, vault) = copy_fixture_to_temp().await;
        vault
            .append_note(Path::new("orphan.md"), "\nAppended line\n")
            .unwrap();
        let content = vault.read_note(Path::new("orphan.md")).unwrap();
        assert!(content.ends_with("Appended line\n"));
    }

    #[tokio::test]
    async fn prepend_after_frontmatter() {
        let (_tmp, vault) = copy_fixture_to_temp().await;
        vault
            .prepend_note(Path::new("Projects/rust-mcp.md"), "Prepended\n")
            .unwrap();
        let content = vault.read_note(Path::new("Projects/rust-mcp.md")).unwrap();
        let prepended_pos = content.find("Prepended\n").unwrap();
        let heading_pos = content.find("# Rust MCP Server").unwrap();
        assert!(prepended_pos < heading_pos);
        assert!(content.starts_with("---\n"));
    }

    #[tokio::test]
    async fn delete_note() {
        let (_tmp, vault) = copy_fixture_to_temp().await;
        vault.delete_note(Path::new("orphan.md")).unwrap();
        let err = vault.read_note(Path::new("orphan.md")).unwrap_err();
        assert!(matches!(
            err,
            obsidian_mcp::error::VaultError::NoteNotFound(_)
        ));
    }

    #[tokio::test]
    async fn move_note() {
        let (_tmp, vault) = copy_fixture_to_temp().await;
        let new_path = vault
            .move_note(Path::new("orphan.md"), Path::new("Archive/orphan.md"))
            .unwrap();
        assert_eq!(new_path, PathBuf::from("Archive/orphan.md"));
        let content = vault.read_note(Path::new("Archive/orphan.md")).unwrap();
        assert!(content.contains("Orphan Note"));
        assert!(vault.read_note(Path::new("orphan.md")).is_err());
    }

    #[tokio::test]
    async fn patch_heading_append() {
        let (_tmp, vault) = copy_fixture_to_temp().await;
        vault
            .patch_note(
                Path::new("Projects/rust-mcp.md"),
                &PatchRequest {
                    operation: PatchOperation::Append,
                    target_type: PatchTargetType::Heading,
                    target: "Features".into(),
                    content: "- New feature added\n".into(),
                },
            )
            .unwrap();
        let content = vault.read_note(Path::new("Projects/rust-mcp.md")).unwrap();
        assert!(content.contains("- New feature added\n"));
    }

    #[tokio::test]
    async fn frontmatter_set_and_remove() {
        let (_tmp, vault) = copy_fixture_to_temp().await;
        vault
            .set_frontmatter_field(
                Path::new("orphan.md"),
                "category",
                serde_json::json!("archive"),
            )
            .unwrap();
        let fm = vault.get_frontmatter(Path::new("orphan.md")).unwrap();
        assert_eq!(fm.unwrap()["category"], "archive");

        vault
            .remove_frontmatter_field(Path::new("orphan.md"), "category")
            .unwrap();
        let fm = vault.get_frontmatter(Path::new("orphan.md")).unwrap();
        match fm {
            None => {} // removing last field strips frontmatter entirely
            Some(obj) => assert!(obj.get("category").is_none()),
        }
    }

    #[tokio::test]
    async fn frontmatter_get_existing() {
        let (_tmp, vault) = copy_fixture_to_temp().await;
        let fm = vault
            .get_frontmatter(Path::new("Projects/rust-mcp.md"))
            .unwrap();
        let obj = fm.expect("rust-mcp.md should have frontmatter");
        assert_eq!(obj["status"], "active");
    }

    #[tokio::test]
    async fn write_then_search() {
        let (_tmp, vault) = copy_fixture_to_temp().await;
        vault
            .write_note(
                Path::new("searchme.md"),
                "# Unique\nfindable_xyzzy_content\n",
            )
            .unwrap();
        let results = vault.search_text("findable_xyzzy_content", 40).unwrap();
        assert!(
            results
                .iter()
                .any(|r| r.path == PathBuf::from("searchme.md"))
        );
    }
}

// ── Periodic notes ───────────────────────────────────────────────────────

mod vault_periodic {
    use super::*;

    #[test]
    fn list_recent_daily_notes() {
        let notes = VAULT
            .list_recent_periodic_notes(&NotePeriod::Daily, 10)
            .unwrap();
        assert!(
            notes.iter().any(|p| p.ends_with("2026-03-19.md")),
            "expected to find the daily note, got: {:?}",
            notes
        );
    }

    #[tokio::test]
    async fn create_periodic_note() {
        let (_tmp, vault) = copy_fixture_to_temp().await;
        let date = chrono::NaiveDate::from_ymd_opt(2026, 1, 15).unwrap();
        let path = vault
            .create_periodic_note(&NotePeriod::Daily, Some(date), None)
            .unwrap();
        assert!(path.to_string_lossy().contains("2026-01-15"));
        let content = vault.read_note(&path).unwrap();
        assert!(content.is_empty() || content.contains("2026"));
    }
}
