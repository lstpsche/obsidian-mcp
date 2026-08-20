use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use obsidian_mcp::config::{Config, ToolFilter};
use obsidian_mcp::models::{NotePeriod, PatchOperation, PatchRequest, PatchTargetType};
use obsidian_mcp::vault::Vault;

#[cfg(all(unix, feature = "embeddings"))]
mod common;

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
        transport: obsidian_mcp::config::Transport::Stdio,
        http_host: obsidian_mcp::config::DEFAULT_HTTP_HOST,
        http_port: obsidian_mcp::config::DEFAULT_HTTP_PORT,
        tantivy: false,
        embeddings: false,
        embeddings_model: String::new(),
        hybrid_alpha: 0.25,
        embedding_provider: None,
        tool_filter: ToolFilter::Full,
        mcp_data_dir: None,
        exclude_patterns: vec![],
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
        transport: obsidian_mcp::config::Transport::Stdio,
        http_host: obsidian_mcp::config::DEFAULT_HTTP_HOST,
        http_port: obsidian_mcp::config::DEFAULT_HTTP_PORT,
        tantivy: false,
        embeddings: false,
        embeddings_model: String::new(),
        hybrid_alpha: 0.25,
        embedding_provider: None,
        tool_filter: ToolFilter::Full,
        mcp_data_dir: None,
        exclude_patterns: vec![],
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

// ── Tantivy BM25 search (temp copies with tantivy enabled) ──────────────

mod vault_tantivy_search {
    use super::*;
    use obsidian_mcp::models::SearchField;

    async fn copy_fixture_with_tantivy() -> (tempfile::TempDir, Vault) {
        let tmp = tempfile::tempdir().unwrap();
        copy_dir_recursive(&fixture_path(), tmp.path());
        let config = Config {
            vault_path: tmp.path().to_path_buf(),
            watch: false,
            log_level: "error".into(),
            transport: obsidian_mcp::config::Transport::Stdio,
            http_host: obsidian_mcp::config::DEFAULT_HTTP_HOST,
            http_port: obsidian_mcp::config::DEFAULT_HTTP_PORT,
            tantivy: true,
            embeddings: false,
            embeddings_model: String::new(),
            hybrid_alpha: 0.25,
            embedding_provider: None,
            tool_filter: ToolFilter::Full,
            mcp_data_dir: None,
            exclude_patterns: vec![],
        };
        let vault = Vault::open(&config)
            .await
            .expect("failed to open tantivy vault");
        (tmp, vault)
    }

    #[tokio::test]
    async fn search_text_returns_ranked_results() {
        let (_tmp, vault) = copy_fixture_with_tantivy().await;
        let results = vault.search_text("quantum entanglement", 40).unwrap();
        assert!(!results.is_empty());
        assert!(
            results[0].score.is_some(),
            "Tantivy search should populate scores"
        );

        if results.len() >= 2 {
            let s0 = results[0].score.unwrap();
            let s1 = results[1].score.unwrap();
            assert!(s0 >= s1, "results should be sorted by score descending");
        }
    }

    #[tokio::test]
    async fn search_text_stemming_finds_related_terms() {
        let (_tmp, vault) = copy_fixture_with_tantivy().await;
        // "server" appears in rust-mcp.md; "servers" stems to the same root
        let results = vault.search_text("servers", 40).unwrap();
        assert!(
            !results.is_empty(),
            "stemming should match 'servers' → 'server'"
        );
        assert!(results[0].score.is_some());
    }

    #[tokio::test]
    async fn search_text_with_options_fuzzy() {
        let (_tmp, vault) = copy_fixture_with_tantivy().await;

        vault
            .write_note(
                Path::new("fuzzy_target.md"),
                "# Architecture\nMicroservices architecture patterns.\n",
            )
            .unwrap();

        // "architeture" has a typo (missing 'c')
        let results = vault
            .search_text_with_options("architeture", 40, 10, true, None)
            .unwrap();
        assert!(
            results
                .iter()
                .any(|r| r.path == PathBuf::from("fuzzy_target.md")),
            "fuzzy should find 'architecture' from 'architeture'"
        );
    }

    #[tokio::test]
    async fn search_text_with_options_field_filter() {
        let (_tmp, vault) = copy_fixture_with_tantivy().await;

        vault
            .write_note(
                Path::new("elasticsearch.md"),
                "# Elasticsearch\nDatabase internals and indexing.\n",
            )
            .unwrap();

        // Title field = filename stem = "elasticsearch"
        let title_results = vault
            .search_text_with_options("elasticsearch", 40, 10, false, Some(&[SearchField::Title]))
            .unwrap();
        assert!(
            title_results
                .iter()
                .any(|r| r.path == PathBuf::from("elasticsearch.md"))
        );

        // "indexing" appears only in the body, not title
        let body_results = vault
            .search_text_with_options("indexing", 40, 10, false, Some(&[SearchField::Body]))
            .unwrap();
        assert!(
            body_results
                .iter()
                .any(|r| r.path == PathBuf::from("elasticsearch.md"))
        );
    }

    #[tokio::test]
    async fn search_text_context_snippets_from_tantivy() {
        let (_tmp, vault) = copy_fixture_with_tantivy().await;
        let results = vault.search_text("quantum entanglement", 80).unwrap();

        assert!(!results.is_empty());
        let first = &results[0];
        assert!(!first.matches.is_empty(), "should have context snippets");
        let ctx = &first.matches[0].context;
        let has_any_word = ctx.contains("quantum") || ctx.contains("entanglement");
        assert!(
            has_any_word,
            "context should contain at least one query word"
        );
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

// ── Unicode path normalization ──────────────────────────────────────────

mod unicode_paths {
    use super::*;
    use unicode_normalization::UnicodeNormalization;

    fn unicode_config(vault_root: &Path) -> Config {
        Config {
            vault_path: vault_root.to_path_buf(),
            watch: false,
            log_level: "error".into(),
            transport: obsidian_mcp::config::Transport::Stdio,
            http_host: obsidian_mcp::config::DEFAULT_HTTP_HOST,
            http_port: obsidian_mcp::config::DEFAULT_HTTP_PORT,
            tantivy: true,
            embeddings: false,
            embeddings_model: String::new(),
            hybrid_alpha: 0.25,
            embedding_provider: None,
            tool_filter: ToolFilter::Full,
            mcp_data_dir: None,
            exclude_patterns: vec![],
        }
    }

    #[tokio::test]
    async fn canonically_equivalent_unicode_paths_work_end_to_end() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join(".obsidian")).unwrap();

        let composed = "02_База-знаний/Сущности/lic1c.md";
        let composed_link_target = "02_База-знаний/Сущности/lic1c";
        let decomposed: String = composed.nfd().collect();
        let disk_path = PathBuf::from(&decomposed);
        std::fs::create_dir_all(dir.path().join(disk_path.parent().unwrap())).unwrap();
        std::fs::write(
            dir.path().join(&disk_path),
            "# License\n\ninitial-unicode-token\n\nLinks to [[source]].\n",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("source.md"),
            format!("# Source\n\nLinks to [[{composed_link_target}]].\n"),
        )
        .unwrap();

        let vault = Vault::open(&unicode_config(dir.path())).await.unwrap();

        let content = vault.read_note(Path::new(composed)).unwrap();
        assert!(content.contains("initial-unicode-token"));

        let metadata = vault.get_note_metadata(Path::new(composed)).unwrap();
        assert_eq!(metadata.path, disk_path);

        let backlinks = vault.backlinks(Path::new(composed)).unwrap();
        assert!(
            backlinks
                .iter()
                .any(|note| note.path == PathBuf::from("source.md"))
        );

        let outgoing = vault.outgoing_links(Path::new(composed)).unwrap();
        assert!(outgoing.iter().any(|link| link.target == "source"));

        let initial_results = vault.search_text("initial-unicode-token", 40).unwrap();
        assert_eq!(initial_results.len(), 1);
        assert_eq!(initial_results[0].path, disk_path);

        vault
            .append_note(Path::new(composed), "\nappended-unicode-token\n")
            .unwrap();
        assert!(
            vault
                .read_note(Path::new(composed))
                .unwrap()
                .contains("appended-unicode-token")
        );
        let appended_results = vault.search_text("appended-unicode-token", 40).unwrap();
        assert_eq!(appended_results.len(), 1);
        assert_eq!(appended_results[0].path, disk_path);

        let moved = vault
            .move_note(Path::new(composed), Path::new("Moved/lic1c.md"))
            .unwrap();
        assert_eq!(moved, PathBuf::from("Moved/lic1c.md"));
        assert!(vault.get_note_metadata(Path::new(composed)).is_err());
        assert!(vault.get_note_metadata(&moved).is_ok());

        vault.delete_note(&moved).unwrap();
        assert!(vault.get_note_metadata(&moved).is_err());
        assert!(
            vault
                .search_text("appended-unicode-token", 40)
                .unwrap()
                .is_empty()
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

// ── Tool filtering (integration) ─────────────────────────────────────────

mod tool_filtering {
    use std::collections::HashSet;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    use obsidian_mcp::config::{ALL_TOOL_NAMES, SemanticMode};
    use obsidian_mcp::tools::{ObsidianMcp, SemanticRuntime};

    use super::*;

    fn test_runtime() -> SemanticRuntime {
        SemanticRuntime {
            mode: SemanticMode::Local,
            daemon_client: None,
            daemon_unavailable_reason: None,
            prefetch_count: 50,
            vault_ensured: Arc::new(AtomicBool::new(false)),
        }
    }

    fn filtering_config(vault_root: &Path, filter: ToolFilter) -> Config {
        Config {
            vault_path: vault_root.to_path_buf(),
            watch: false,
            log_level: "error".into(),
            transport: obsidian_mcp::config::Transport::Stdio,
            http_host: obsidian_mcp::config::DEFAULT_HTTP_HOST,
            http_port: obsidian_mcp::config::DEFAULT_HTTP_PORT,
            tantivy: false,
            embeddings: false,
            embeddings_model: String::new(),
            hybrid_alpha: 0.25,
            embedding_provider: None,
            tool_filter: filter,
            mcp_data_dir: None,
            exclude_patterns: vec![],
        }
    }

    async fn build_server(filter: ToolFilter) -> (tempfile::TempDir, ObsidianMcp) {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join(".obsidian")).unwrap();
        let config = filtering_config(tmp.path(), filter);
        let disabled = config.tool_filter.disabled_tools();
        let vault = Vault::open(&config).await.expect("open vault");
        let server = ObsidianMcp::new(vault, 0.25, test_runtime(), disabled);
        (tmp, server)
    }

    #[tokio::test]
    async fn full_profile_exposes_all_18_tools() {
        let (_tmp, server) = build_server(ToolFilter::Full).await;
        let tools = server.tool_router.list_all();
        assert_eq!(
            tools.len(),
            ALL_TOOL_NAMES.len(),
            "full profile should expose all {} tools, got {}",
            ALL_TOOL_NAMES.len(),
            tools.len()
        );
        for name in ALL_TOOL_NAMES {
            assert!(
                server.tool_router.has_route(name),
                "full profile should include '{name}'"
            );
        }
    }

    #[tokio::test]
    async fn core_profile_exposes_14_tools() {
        let (_tmp, server) = build_server(ToolFilter::Profile("core".into())).await;
        let tools = server.tool_router.list_all();
        assert_eq!(tools.len(), 14, "core profile should expose 14 tools");

        assert!(server.tool_router.has_route("note_read"));
        assert!(server.tool_router.has_route("vault_list"));
        assert!(server.tool_router.has_route("search_text"));
        assert!(server.tool_router.has_route("frontmatter"));
        assert!(server.tool_router.has_route("note_inspect"));

        assert!(!server.tool_router.has_route("search_semantic"));
        assert!(!server.tool_router.has_route("wikilinks"));
        assert!(!server.tool_router.has_route("periodic"));
        assert!(!server.tool_router.has_route("open_in_obsidian"));
    }

    #[tokio::test]
    async fn read_profile_exposes_10_tools() {
        let (_tmp, server) = build_server(ToolFilter::Profile("read".into())).await;
        let tools = server.tool_router.list_all();
        assert_eq!(tools.len(), 10, "read profile should expose 10 tools");

        assert!(server.tool_router.has_route("note_read"));
        assert!(server.tool_router.has_route("vault_list"));
        assert!(server.tool_router.has_route("search_text"));
        assert!(server.tool_router.has_route("search_semantic"));
        assert!(server.tool_router.has_route("wikilinks"));

        assert!(!server.tool_router.has_route("note_create"));
        assert!(!server.tool_router.has_route("note_write"));
        assert!(!server.tool_router.has_route("note_delete"));
        assert!(!server.tool_router.has_route("note_move"));
    }

    #[tokio::test]
    async fn minimal_profile_exposes_6_tools() {
        let (_tmp, server) = build_server(ToolFilter::Profile("minimal".into())).await;
        let tools = server.tool_router.list_all();
        assert_eq!(tools.len(), 6, "minimal profile should expose 6 tools");

        let expected = [
            "note_read",
            "note_create",
            "note_write",
            "vault_list",
            "search_text",
            "vault_info",
        ];
        for name in &expected {
            assert!(
                server.tool_router.has_route(name),
                "minimal profile should include '{name}'"
            );
        }
        assert!(!server.tool_router.has_route("search_regex"));
        assert!(!server.tool_router.has_route("wikilinks"));
        assert!(!server.tool_router.has_route("frontmatter"));
    }

    #[tokio::test]
    async fn allow_list_only_listed_tools() {
        let allowed: HashSet<String> = ["note_read", "vault_list"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let (_tmp, server) = build_server(ToolFilter::AllowList(allowed)).await;
        let tools = server.tool_router.list_all();
        assert_eq!(tools.len(), 2, "allow-list should expose only 2 tools");

        assert!(server.tool_router.has_route("note_read"));
        assert!(server.tool_router.has_route("vault_list"));
        assert!(!server.tool_router.has_route("note_create"));
        assert!(!server.tool_router.has_route("search_text"));
    }

    #[tokio::test]
    async fn deny_list_hides_only_listed_tools() {
        let denied: HashSet<String> = ["open_in_obsidian", "wikilinks"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let (_tmp, server) = build_server(ToolFilter::DenyList(denied)).await;
        let tools = server.tool_router.list_all();
        assert_eq!(
            tools.len(),
            ALL_TOOL_NAMES.len() - 2,
            "deny-list should hide 2 tools"
        );

        assert!(!server.tool_router.has_route("open_in_obsidian"));
        assert!(!server.tool_router.has_route("wikilinks"));
        assert!(server.tool_router.has_route("note_read"));
        assert!(server.tool_router.has_route("vault_list"));
        assert!(server.tool_router.has_route("search_text"));
    }
}

// ── Exclusion & metadata folder ─────────────────────────────────────────

mod vault_exclusion {
    use super::*;

    fn config_with_exclusions(vault_root: &Path, patterns: Vec<String>) -> Config {
        Config {
            vault_path: vault_root.to_path_buf(),
            watch: false,
            log_level: "error".into(),
            transport: obsidian_mcp::config::Transport::Stdio,
            http_host: obsidian_mcp::config::DEFAULT_HTTP_HOST,
            http_port: obsidian_mcp::config::DEFAULT_HTTP_PORT,
            tantivy: false,
            embeddings: false,
            embeddings_model: String::new(),
            hybrid_alpha: 0.25,
            embedding_provider: None,
            tool_filter: ToolFilter::Full,
            mcp_data_dir: None,
            exclude_patterns: patterns,
        }
    }

    fn tantivy_config_with_exclusions(vault_root: &Path, patterns: Vec<String>) -> Config {
        Config {
            tantivy: true,
            ..config_with_exclusions(vault_root, patterns)
        }
    }

    #[tokio::test]
    async fn exclusion_filters_index() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join(".obsidian")).unwrap();
        std::fs::create_dir_all(dir.path().join("Active")).unwrap();
        std::fs::create_dir_all(dir.path().join("Archive")).unwrap();
        std::fs::write(dir.path().join("Active/note.md"), "# Active Note\n").unwrap();
        std::fs::write(dir.path().join("Archive/old.md"), "# Old Note\n").unwrap();

        let config = config_with_exclusions(dir.path(), vec!["Archive/".into()]);
        let vault = Vault::open(&config).await.unwrap();

        assert!(
            vault.get_note_metadata(Path::new("Active/note.md")).is_ok(),
            "non-excluded note should be in index"
        );
        assert!(
            vault
                .get_note_metadata(Path::new("Archive/old.md"))
                .is_err(),
            "excluded note should not be in index"
        );

        let stats = vault.vault_stats().unwrap();
        assert_eq!(stats.total_notes, 1, "only non-excluded notes counted");

        let content = vault.read_note(Path::new("Archive/old.md")).unwrap();
        assert!(
            content.contains("Old Note"),
            "direct read of excluded note should still work"
        );
    }

    #[tokio::test]
    async fn exclusion_via_ignore_file() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join(".obsidian")).unwrap();
        std::fs::create_dir_all(dir.path().join(".obsidian-mcp")).unwrap();
        std::fs::create_dir_all(dir.path().join("Active")).unwrap();
        std::fs::create_dir_all(dir.path().join("Archive")).unwrap();
        std::fs::write(dir.path().join("Active/note.md"), "# Active\n").unwrap();
        std::fs::write(dir.path().join("Archive/old.md"), "# Old\n").unwrap();
        std::fs::write(dir.path().join(".obsidian-mcp/ignore"), "Archive/\n").unwrap();

        let config = config_with_exclusions(dir.path(), vec![]);
        let vault = Vault::open(&config).await.unwrap();

        assert!(
            vault.get_note_metadata(Path::new("Active/note.md")).is_ok(),
            "non-excluded note should be in index"
        );
        assert!(
            vault
                .get_note_metadata(Path::new("Archive/old.md"))
                .is_err(),
            "note excluded via ignore file should not be in index"
        );
    }

    #[tokio::test]
    async fn mcp_home_created_on_startup() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join(".obsidian")).unwrap();

        assert!(
            !dir.path().join(".obsidian-mcp").exists(),
            "precondition: .obsidian-mcp should not exist yet"
        );

        let config = config_with_exclusions(dir.path(), vec![]);
        let _vault = Vault::open(&config).await.unwrap();

        assert!(
            dir.path().join(".obsidian-mcp").is_dir(),
            ".obsidian-mcp directory should be created on startup"
        );

        let ignore_path = dir.path().join(".obsidian-mcp/ignore");
        assert!(ignore_path.exists(), "ignore file should be auto-created");
        let content = std::fs::read_to_string(&ignore_path).unwrap();
        assert!(
            content.is_empty(),
            "auto-created ignore file should be empty"
        );
    }

    #[tokio::test]
    async fn mcp_data_external_path() {
        let vault_dir = tempfile::tempdir().unwrap();
        let data_dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(vault_dir.path().join(".obsidian")).unwrap();

        let config = Config {
            mcp_data_dir: Some(data_dir.path().to_path_buf()),
            ..config_with_exclusions(vault_dir.path(), vec![])
        };
        let vault = Vault::open(&config).await.unwrap();

        let slug = obsidian_mcp::config::vault_slug(vault.root());
        let expected = data_dir.path().join("vaults").join(&slug);
        assert!(
            expected.is_dir(),
            "external data dir should contain vaults/{slug}/ structure"
        );
        assert_eq!(vault.mcp_data(), expected);
    }

    #[tokio::test]
    async fn obsidian_mcp_dir_not_indexed() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join(".obsidian")).unwrap();
        std::fs::create_dir_all(dir.path().join(".obsidian-mcp")).unwrap();
        std::fs::write(
            dir.path().join(".obsidian-mcp/test.md"),
            "# Should not be indexed\n",
        )
        .unwrap();
        std::fs::write(dir.path().join("visible.md"), "# Visible\n").unwrap();

        let config = config_with_exclusions(dir.path(), vec![]);
        let vault = Vault::open(&config).await.unwrap();

        assert!(
            vault.get_note_metadata(Path::new("visible.md")).is_ok(),
            "regular note should be indexed"
        );
        assert!(
            vault
                .get_note_metadata(Path::new(".obsidian-mcp/test.md"))
                .is_err(),
            ".obsidian-mcp/ contents should never be indexed"
        );

        let stats = vault.vault_stats().unwrap();
        assert_eq!(stats.total_notes, 1);
    }

    #[tokio::test]
    async fn tantivy_respects_exclusion() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join(".obsidian")).unwrap();
        std::fs::create_dir_all(dir.path().join("Active")).unwrap();
        std::fs::create_dir_all(dir.path().join("Archive")).unwrap();
        std::fs::write(
            dir.path().join("Active/visible.md"),
            "# Visible\nxylophone-unique-test-word content\n",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("Archive/hidden.md"),
            "# Hidden\nxylophone-unique-test-word content\n",
        )
        .unwrap();

        let config = tantivy_config_with_exclusions(dir.path(), vec!["Archive/".into()]);
        let vault = Vault::open(&config).await.unwrap();

        let results = vault.search_text("xylophone-unique-test-word", 40).unwrap();
        assert_eq!(results.len(), 1, "only the non-excluded note should appear");
        assert_eq!(results[0].path, PathBuf::from("Active/visible.md"));
    }

    #[tokio::test]
    async fn vault_info_reports_exclusion_stats() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join(".obsidian")).unwrap();
        std::fs::create_dir_all(dir.path().join("Active")).unwrap();
        std::fs::create_dir_all(dir.path().join("Archive")).unwrap();
        std::fs::write(dir.path().join("Active/note.md"), "# Active Note\n").unwrap();
        std::fs::write(dir.path().join("Archive/old.md"), "# Old Note\n").unwrap();

        let config = config_with_exclusions(dir.path(), vec!["Archive/".into()]);
        let vault = Vault::open(&config).await.unwrap();

        let stats = vault.vault_stats().unwrap();
        assert_eq!(
            stats.excluded_notes, 1,
            "one .md file in Archive/ should be excluded"
        );
        assert_eq!(stats.total_notes, 1, "only non-excluded notes counted");

        let patterns = vault.exclude().patterns();
        assert!(
            patterns.iter().any(|p| p.contains("Archive")),
            "exclude_patterns should contain the Archive pattern, got: {patterns:?}"
        );

        assert_eq!(
            vault.mcp_data(),
            vault.mcp_home(),
            "mcp_data_dir should equal mcp_home when OBSIDIAN_MCP_DATA is not set"
        );
    }

    #[tokio::test]
    async fn move_into_excluded_dir_removes_from_index() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join(".obsidian")).unwrap();
        std::fs::create_dir_all(dir.path().join("Archive")).unwrap();
        std::fs::write(dir.path().join("note.md"), "# Movable Note\n").unwrap();

        let config = config_with_exclusions(dir.path(), vec!["Archive/".into()]);
        let vault = Vault::open(&config).await.unwrap();

        assert!(
            vault.get_note_metadata(Path::new("note.md")).is_ok(),
            "note should be in index before move"
        );

        let new_path = vault
            .move_note(Path::new("note.md"), Path::new("Archive/moved.md"))
            .unwrap();
        assert_eq!(new_path, PathBuf::from("Archive/moved.md"));

        assert!(
            vault
                .get_note_metadata(Path::new("Archive/moved.md"))
                .is_err(),
            "note should NOT be in index after moving to excluded dir"
        );
        assert!(
            vault.get_note_metadata(Path::new("note.md")).is_err(),
            "old path should be gone from index"
        );

        let content = vault.read_note(Path::new("Archive/moved.md")).unwrap();
        assert!(
            content.contains("Movable Note"),
            "file should still be readable on disk via direct access"
        );
    }

    #[tokio::test]
    async fn move_out_of_excluded_dir_adds_to_index() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join(".obsidian")).unwrap();
        std::fs::create_dir_all(dir.path().join("Archive")).unwrap();
        std::fs::write(dir.path().join("Archive/hidden.md"), "# Hidden Note\n").unwrap();

        let config = config_with_exclusions(dir.path(), vec!["Archive/".into()]);
        let vault = Vault::open(&config).await.unwrap();

        assert!(
            vault
                .get_note_metadata(Path::new("Archive/hidden.md"))
                .is_err(),
            "excluded note should NOT be in index"
        );

        let new_path = vault
            .move_note(Path::new("Archive/hidden.md"), Path::new("visible.md"))
            .unwrap();
        assert_eq!(new_path, PathBuf::from("visible.md"));

        assert!(
            vault.get_note_metadata(Path::new("visible.md")).is_ok(),
            "note should be in index after moving out of excluded dir"
        );
        assert!(
            vault
                .get_note_metadata(Path::new("Archive/hidden.md"))
                .is_err(),
            "old excluded path should not be in index"
        );
    }

    #[tokio::test]
    async fn search_text_excludes_excluded_notes() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join(".obsidian")).unwrap();
        std::fs::create_dir_all(dir.path().join("Active")).unwrap();
        std::fs::create_dir_all(dir.path().join("Archive")).unwrap();
        std::fs::write(
            dir.path().join("Active/visible.md"),
            "# Visible\nzebra-platypus-unique-search-term here\n",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("Archive/hidden.md"),
            "# Hidden\nzebra-platypus-unique-search-term here\n",
        )
        .unwrap();

        let config = config_with_exclusions(dir.path(), vec!["Archive/".into()]);
        let vault = Vault::open(&config).await.unwrap();

        let results = vault
            .search_text("zebra-platypus-unique-search-term", 40)
            .unwrap();
        assert_eq!(
            results.len(),
            1,
            "only the non-excluded note should appear in regex search"
        );
        assert_eq!(results[0].path, PathBuf::from("Active/visible.md"));
    }

    #[tokio::test]
    async fn vault_list_includes_excluded_notes() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join(".obsidian")).unwrap();
        std::fs::create_dir_all(dir.path().join("Active")).unwrap();
        std::fs::create_dir_all(dir.path().join("Archive")).unwrap();
        std::fs::write(dir.path().join("Active/note.md"), "# Active\n").unwrap();
        std::fs::write(dir.path().join("Archive/old.md"), "# Old\n").unwrap();

        let config = config_with_exclusions(dir.path(), vec!["Archive/".into()]);
        let vault = Vault::open(&config).await.unwrap();

        let files = vault.list_files(Path::new(""), true, None).unwrap();
        assert!(
            files.iter().any(|f| f == Path::new("Active/note.md")),
            "non-excluded file should appear in listing, got: {files:?}"
        );
        assert!(
            files.iter().any(|f| f == Path::new("Archive/old.md")),
            "excluded file should ALSO appear in listing (vault_list is unfiltered), got: {files:?}"
        );
    }
}

// ── Semantic search (embeddings feature) ────────────────────────────────

#[cfg(feature = "embeddings")]
mod vault_semantic_search {
    use super::*;

    /// Serialize model loading across tests to prevent concurrent fastembed
    /// cache access races.
    static MODEL_LOCK: std::sync::LazyLock<tokio::sync::Mutex<()>> =
        std::sync::LazyLock::new(|| tokio::sync::Mutex::new(()));

    fn embeddings_config(vault_root: &Path) -> Config {
        Config {
            vault_path: vault_root.to_path_buf(),
            watch: false,
            log_level: "error".into(),
            transport: obsidian_mcp::config::Transport::Stdio,
            http_host: obsidian_mcp::config::DEFAULT_HTTP_HOST,
            http_port: obsidian_mcp::config::DEFAULT_HTTP_PORT,
            tantivy: false,
            embeddings: true,
            embeddings_model: "BAAI/bge-small-en-v1.5".into(),
            hybrid_alpha: 0.25,
            embedding_provider: None,
            tool_filter: ToolFilter::Full,
            mcp_data_dir: None,
            exclude_patterns: vec![],
        }
    }

    async fn open_with_embeddings(vault_root: &Path) -> Vault {
        let _guard = MODEL_LOCK.lock().await;
        let config = embeddings_config(vault_root);
        let vault = Vault::open(&config)
            .await
            .expect("open vault with embeddings");
        wait_for_embeddings_ready(&vault).await;
        vault
    }

    async fn wait_for_embeddings_ready(vault: &Vault) {
        tokio::time::timeout(std::time::Duration::from_secs(300), async {
            loop {
                if vault.has_embeddings() {
                    return;
                }
                if let Some(error) = vault.embedding_load_error() {
                    panic!("embedding model failed to load: {error}");
                }
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }
        })
        .await
        .unwrap_or_else(|_| {
            panic!(
                "timed out waiting for embedding readiness: {:?}",
                vault.embedding_status()
            )
        });
    }

    async fn wait_for_semantic_hit(
        vault: &Vault,
        query: &str,
        top_k: usize,
        path: &Path,
    ) -> Vec<(PathBuf, f32)> {
        for _ in 0..20 {
            let results = vault.search_semantic(query, top_k).unwrap();
            if results.iter().any(|(p, _)| p == path) {
                return results;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        vault.search_semantic(query, top_k).unwrap()
    }

    #[tokio::test]
    async fn search_semantic_returns_results() {
        let (_tmp, _vault) = copy_fixture_to_temp().await;
        let vault = open_with_embeddings(_tmp.path()).await;

        let results = vault.search_semantic("programming languages", 5).unwrap();
        assert!(
            !results.is_empty(),
            "semantic search should return results for the fixture vault"
        );
        if results.len() >= 2 {
            assert!(
                results[0].1 >= results[1].1,
                "results should be sorted by descending score"
            );
        }
    }

    #[tokio::test]
    async fn search_semantic_empty_vault_returns_empty() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join(".obsidian")).unwrap();
        let vault = open_with_embeddings(tmp.path()).await;

        let results = vault.search_semantic("anything", 10).unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn search_semantic_disabled_returns_error() {
        let (_tmp, vault) = copy_fixture_to_temp().await;
        let result = vault.search_semantic("test query", 5);
        assert!(
            result.is_err(),
            "search_semantic should fail when embeddings are disabled"
        );
    }

    #[tokio::test]
    async fn search_semantic_syncs_on_write() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join(".obsidian")).unwrap();
        let vault = open_with_embeddings(tmp.path()).await;

        vault
            .write_note(
                Path::new("rust.md"),
                "# Rust\nRust is a systems programming language known for memory safety.\n",
            )
            .unwrap();

        let results =
            wait_for_semantic_hit(&vault, "memory safe programming", 5, Path::new("rust.md")).await;
        assert!(
            results.iter().any(|(p, _)| p == Path::new("rust.md")),
            "newly written note should appear in semantic search"
        );
    }

    #[tokio::test]
    async fn search_semantic_syncs_on_delete() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join(".obsidian")).unwrap();
        let vault = open_with_embeddings(tmp.path()).await;

        vault
            .write_note(
                Path::new("gone.md"),
                "# Ephemeral\nThis note will be deleted soon.\n",
            )
            .unwrap();
        vault.delete_note(Path::new("gone.md")).unwrap();

        let results = vault.search_semantic("ephemeral deleted", 5).unwrap();
        assert!(
            !results.iter().any(|(p, _)| p == Path::new("gone.md")),
            "deleted note should not appear in semantic search"
        );
    }

    // ── hybrid search (E7) ──────────────────────────────────────────

    fn hybrid_config(vault_root: &Path) -> Config {
        Config {
            tantivy: true,
            ..embeddings_config(vault_root)
        }
    }

    async fn open_hybrid(vault_root: &Path) -> Vault {
        let _guard = MODEL_LOCK.lock().await;
        let config = hybrid_config(vault_root);
        let vault = Vault::open(&config)
            .await
            .expect("open vault with tantivy + embeddings");
        wait_for_embeddings_ready(&vault).await;
        vault
    }

    #[tokio::test]
    async fn search_hybrid_returns_results() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join(".obsidian")).unwrap();
        let vault = open_hybrid(tmp.path()).await;

        vault
            .write_note(
                Path::new("rust.md"),
                "# Rust\nRust is a systems programming language known for memory safety.\n",
            )
            .unwrap();
        vault
            .write_note(
                Path::new("python.md"),
                "# Python\nPython is a dynamic language for scripting and data science.\n",
            )
            .unwrap();

        let results = vault
            .search_hybrid("systems programming", 5, 50, 0.4)
            .unwrap();
        assert!(!results.is_empty(), "hybrid search should return results");
        assert!(
            results.iter().any(|(p, _)| p == Path::new("rust.md")),
            "rust.md should be in hybrid results for 'systems programming'"
        );
        if results.len() >= 2 {
            assert!(
                results[0].1 >= results[1].1,
                "results should be sorted by descending combined score"
            );
        }
    }

    #[tokio::test]
    async fn search_hybrid_empty_query_returns_empty() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join(".obsidian")).unwrap();
        let vault = open_hybrid(tmp.path()).await;

        vault
            .write_note(Path::new("note.md"), "# Note\nSome content.\n")
            .unwrap();

        let results = vault.search_hybrid("", 5, 50, 0.4).unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn search_hybrid_without_tantivy_errors() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join(".obsidian")).unwrap();
        let vault = open_with_embeddings(tmp.path()).await;

        let result = vault.search_hybrid("test", 5, 50, 0.4);
        assert!(
            result.is_err(),
            "hybrid search should fail when Tantivy is disabled"
        );
    }

    #[tokio::test]
    async fn search_hybrid_syncs_after_write() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join(".obsidian")).unwrap();
        let vault = open_hybrid(tmp.path()).await;

        vault
            .write_note(
                Path::new("quantum.md"),
                "# Quantum Computing\nQuantum computers use qubits for exponential parallelism.\n",
            )
            .unwrap();

        let results = vault
            .search_hybrid("quantum computing", 5, 50, 0.4)
            .unwrap();
        assert!(
            results.iter().any(|(p, _)| p == Path::new("quantum.md")),
            "newly written note should appear in hybrid search"
        );
    }
}

#[cfg(all(unix, feature = "embeddings"))]
mod semantic_tool_runtime_modes {
    use super::*;
    use std::sync::atomic::AtomicBool;
    use std::sync::{Arc, LazyLock};

    use obsidian_mcp::client::semantic_daemon::{DaemonConnectPolicy, SemanticDaemonClient};
    use obsidian_mcp::config::SemanticMode;
    use obsidian_mcp::daemon::server::IpcEndpoint;
    use obsidian_mcp::tools::SemanticRuntime;
    use obsidian_mcp::tools::search::{SearchSemanticParams, search_semantic};
    use rmcp::model::ErrorCode;

    use crate::common::daemon_test_utils::{DaemonTestServer, create_temp_vault, write_note};

    static MODEL_LOCK: LazyLock<tokio::sync::Mutex<()>> =
        LazyLock::new(|| tokio::sync::Mutex::new(()));
    const MODEL_NAME: &str = "BAAI/bge-small-en-v1.5";

    fn semantic_tool_config(vault_root: &Path, embeddings: bool) -> Config {
        Config {
            vault_path: vault_root.to_path_buf(),
            watch: false,
            log_level: "error".into(),
            transport: obsidian_mcp::config::Transport::Stdio,
            http_host: obsidian_mcp::config::DEFAULT_HTTP_HOST,
            http_port: obsidian_mcp::config::DEFAULT_HTTP_PORT,
            tantivy: false,
            embeddings,
            embeddings_model: MODEL_NAME.to_string(),
            hybrid_alpha: 0.25,
            embedding_provider: None,
            tool_filter: ToolFilter::Full,
            mcp_data_dir: None,
            exclude_patterns: vec![],
        }
    }

    fn extract_text(result: &rmcp::model::CallToolResult) -> &str {
        result.content[0]
            .as_text()
            .expect("expected text content")
            .text
            .as_str()
    }

    async fn wait_for_local_tool_hit(
        vault: &Vault,
        runtime: &SemanticRuntime,
        query: &str,
        expected_path: &str,
    ) -> Vec<serde_json::Value> {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(300);
        loop {
            let result = search_semantic(
                vault,
                SearchSemanticParams {
                    query: query.to_string(),
                    top_k: Some(5),
                    include_content: Some(false),
                    lexical_prefetch: Some(false),
                    alpha: None,
                },
                0.25,
                runtime,
            )
            .await;
            if let Ok(result) = result {
                let parsed: Vec<serde_json::Value> =
                    serde_json::from_str(extract_text(&result)).expect("parse semantic result");
                if parsed.iter().any(|entry| {
                    entry
                        .get("path")
                        .and_then(serde_json::Value::as_str)
                        .is_some_and(|path| path == expected_path)
                }) {
                    return parsed;
                }
            }
            assert!(
                std::time::Instant::now() < deadline,
                "timed out waiting for local semantic hit; status: {:?}",
                vault.embedding_status()
            );
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
    }

    #[tokio::test]
    async fn daemon_mode_preserves_semantic_result_schema() {
        let _guard = MODEL_LOCK.lock().await;
        let server = DaemonTestServer::start(MODEL_NAME).await;

        let vault_dir = create_temp_vault();
        write_note(
            vault_dir.path(),
            "semantic.md",
            "# Semantic\nRust ownership and memory safety for systems programming.",
        );
        let vault = Vault::open(&semantic_tool_config(vault_dir.path(), false))
            .await
            .expect("open vault");

        let runtime = SemanticRuntime {
            mode: SemanticMode::Daemon,
            daemon_client: Some(SemanticDaemonClient::new(
                IpcEndpoint::UnixSocket(server.endpoint_path().to_path_buf()),
                DaemonConnectPolicy::default(),
            )),
            daemon_unavailable_reason: None,
            vault_ensured: Arc::new(AtomicBool::new(false)),
            prefetch_count: 50,
        };

        server
            .ensure_vault_ready(vault_dir.path(), true, std::time::Duration::from_secs(300))
            .await;

        let result = search_semantic(
            &vault,
            SearchSemanticParams {
                query: "memory safe systems".to_string(),
                top_k: Some(5),
                include_content: Some(false),
                lexical_prefetch: Some(false),
                alpha: None,
            },
            0.25,
            &runtime,
        )
        .await
        .expect("daemon semantic search should succeed");
        let parsed: Vec<serde_json::Value> =
            serde_json::from_str(extract_text(&result)).expect("parse semantic result");

        assert!(!parsed.is_empty());
        let first = &parsed[0];
        assert!(first.get("path").is_some(), "path field should exist");
        assert!(first.get("title").is_some(), "title field should exist");
        assert!(first.get("score").is_some(), "score field should exist");
        assert!(first.get("tags").is_some(), "tags field should exist");
        assert!(
            first.get("subpath").is_none(),
            "MCP response should keep legacy schema (no subpath field)"
        );

        server.shutdown().await;
    }

    #[tokio::test]
    async fn auto_mode_falls_back_to_local_backend_when_daemon_unavailable() {
        let _guard = MODEL_LOCK.lock().await;
        let vault_dir = create_temp_vault();
        let vault = Vault::open(&semantic_tool_config(vault_dir.path(), true))
            .await
            .expect("open vault");
        vault
            .write_note(
                Path::new("local.md"),
                "# Local\nOwnership and borrow checker for memory safety.",
            )
            .expect("write local note");

        let runtime = SemanticRuntime {
            mode: SemanticMode::Auto,
            daemon_client: None,
            daemon_unavailable_reason: Some("daemon socket unavailable".to_string()),
            vault_ensured: Arc::new(AtomicBool::new(false)),
            prefetch_count: 50,
        };

        let parsed = wait_for_local_tool_hit(&vault, &runtime, "memory safety", "local.md").await;
        assert!(!parsed.is_empty());
        assert!(
            parsed.iter().any(|entry| {
                entry
                    .get("path")
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|path| path == "local.md")
            }),
            "local backend result should include local.md"
        );
    }

    #[tokio::test]
    async fn daemon_mode_without_client_returns_invalid_request_error() {
        let vault_dir = create_temp_vault();
        let vault = Vault::open(&semantic_tool_config(vault_dir.path(), false))
            .await
            .expect("open vault");

        let runtime = SemanticRuntime {
            mode: SemanticMode::Daemon,
            daemon_client: None,
            daemon_unavailable_reason: Some("not connected".to_string()),
            vault_ensured: Arc::new(AtomicBool::new(false)),
            prefetch_count: 50,
        };

        let result = search_semantic(
            &vault,
            SearchSemanticParams {
                query: "anything".to_string(),
                top_k: Some(3),
                include_content: Some(false),
                lexical_prefetch: Some(false),
                alpha: None,
            },
            0.25,
            &runtime,
        )
        .await;
        let err = result.expect_err("daemon mode should fail without daemon client");
        assert_eq!(err.code, ErrorCode::INVALID_REQUEST);
    }
}

#[cfg(all(unix, feature = "embeddings-api"))]
mod background_embedding_runtime {
    use super::*;
    use std::ffi::OsString;
    use std::process::Stdio;
    use std::sync::{Arc, LazyLock, Mutex};
    use std::time::Duration;

    use axum::extract::State;
    use axum::http::StatusCode;
    use axum::response::{IntoResponse, Response};
    use axum::routing::post;
    use axum::{Json, Router};
    use obsidian_mcp::config::EmbeddingProvider;
    use obsidian_mcp::vault::embedding_runtime::{EmbeddingPhase, EmbeddingRuntimeStatus};
    use serde_json::{Value, json};
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

    static API_TEST_LOCK: LazyLock<tokio::sync::Mutex<()>> =
        LazyLock::new(|| tokio::sync::Mutex::new(()));

    #[derive(Clone)]
    struct ApiState {
        requests: Arc<Mutex<Vec<Vec<String>>>>,
        blocked_markers: Arc<Mutex<Vec<String>>>,
        failure_markers: Arc<Mutex<Vec<String>>>,
        gate: tokio::sync::watch::Sender<bool>,
    }

    struct ControlledEmbeddingApi {
        address: std::net::SocketAddr,
        state: ApiState,
        shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
        task: tokio::task::JoinHandle<std::io::Result<()>>,
    }

    impl ControlledEmbeddingApi {
        async fn start() -> Self {
            let (gate, _) = tokio::sync::watch::channel(true);
            let state = ApiState {
                requests: Arc::new(Mutex::new(Vec::new())),
                blocked_markers: Arc::new(Mutex::new(Vec::new())),
                failure_markers: Arc::new(Mutex::new(Vec::new())),
                gate,
            };
            let app = Router::new()
                .route("/v1/embeddings", post(embedding_api_handler))
                .with_state(state.clone());
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                .await
                .expect("bind controlled embedding API");
            let address = listener.local_addr().expect("embedding API address");
            let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
            let task = tokio::spawn(async move {
                axum::serve(listener, app)
                    .with_graceful_shutdown(async move {
                        let _ = shutdown_rx.await;
                    })
                    .await
            });
            Self {
                address,
                state,
                shutdown_tx: Some(shutdown_tx),
                task,
            }
        }

        fn base_url(&self) -> String {
            format!("http://{}/v1", self.address)
        }

        fn block_inputs_containing(&self, markers: &[&str]) {
            *self
                .state
                .blocked_markers
                .lock()
                .unwrap_or_else(|error| error.into_inner()) =
                markers.iter().map(|marker| (*marker).to_string()).collect();
            self.state.gate.send_replace(false);
        }

        fn release_blocked(&self) {
            self.state.gate.send_replace(true);
        }

        fn fail_inputs_containing(&self, markers: &[&str]) {
            *self
                .state
                .failure_markers
                .lock()
                .unwrap_or_else(|error| error.into_inner()) =
                markers.iter().map(|marker| (*marker).to_string()).collect();
        }

        fn clear_failures(&self) {
            self.state
                .failure_markers
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .clear();
        }

        fn clear_requests(&self) {
            self.state
                .requests
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .clear();
        }

        fn recorded_inputs(&self) -> Vec<String> {
            self.state
                .requests
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .iter()
                .flatten()
                .cloned()
                .collect()
        }

        async fn wait_for_input(&self, marker: &str) {
            tokio::time::timeout(Duration::from_secs(10), async {
                loop {
                    if self
                        .recorded_inputs()
                        .iter()
                        .any(|input| input.contains(marker))
                    {
                        return;
                    }
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            })
            .await
            .unwrap_or_else(|_| {
                panic!(
                    "timed out waiting for embedding input '{marker}'; recorded: {:?}",
                    self.recorded_inputs()
                )
            });
        }

        async fn shutdown(mut self) {
            if let Some(shutdown_tx) = self.shutdown_tx.take() {
                let _ = shutdown_tx.send(());
            }
            self.task
                .await
                .expect("embedding API task should join")
                .expect("embedding API should stop cleanly");
        }
    }

    async fn embedding_api_handler(
        State(state): State<ApiState>,
        Json(body): Json<Value>,
    ) -> Response {
        let inputs = match &body["input"] {
            Value::Array(values) => values
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect::<Vec<_>>(),
            Value::String(value) => vec![value.clone()],
            _ => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json!({"error": "input must be a string or array"})),
                )
                    .into_response();
            }
        };
        state
            .requests
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .push(inputs.clone());

        let should_fail = {
            let markers = state
                .failure_markers
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            inputs
                .iter()
                .any(|input| markers.iter().any(|marker| input.contains(marker)))
        };
        if should_fail {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({
                    "error": "provider echoed FAIL_MARKER and sensitive note content"
                })),
            )
                .into_response();
        }

        let should_block = {
            let markers = state
                .blocked_markers
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            inputs
                .iter()
                .any(|input| markers.iter().any(|marker| input.contains(marker)))
        };
        if should_block {
            let mut gate = state.gate.subscribe();
            while !*gate.borrow() {
                if gate.changed().await.is_err() {
                    break;
                }
            }
        }

        let data = inputs
            .iter()
            .enumerate()
            .map(|(index, input)| {
                let checksum = input
                    .bytes()
                    .fold(0u32, |sum, byte| sum.wrapping_add(u32::from(byte)));
                json!({
                    "index": index,
                    "embedding": [
                        1.0,
                        f64::from(checksum % 997) / 997.0,
                        input.len() as f64 / 1000.0 + 0.01
                    ]
                })
            })
            .collect::<Vec<_>>();
        Json(json!({"data": data})).into_response()
    }

    struct EmbeddingApiEnv {
        previous: Vec<(&'static str, Option<OsString>)>,
    }

    impl EmbeddingApiEnv {
        fn install(base_url: &str) -> Self {
            let values = [
                ("OBSIDIAN_EMBEDDING_API_BASE", Some(base_url)),
                ("OPENAI_BASE_URL", None),
                ("OBSIDIAN_EMBEDDING_API_KEY", Some("integration-test-key")),
                ("OPENAI_API_KEY", None),
                ("OBSIDIAN_EMBEDDING_API_MODEL", None),
                ("OPENAI_MODEL", None),
                ("OBSIDIAN_EMBEDDING_DIM", None),
                ("OBSIDIAN_EMBEDDING_CA_CERT", None),
                ("OBSIDIAN_EMBEDDING_TLS_VERIFY", None),
            ];
            let mut previous = Vec::with_capacity(values.len());
            for (name, value) in values {
                previous.push((name, std::env::var_os(name)));
                unsafe {
                    if let Some(value) = value {
                        std::env::set_var(name, value);
                    } else {
                        std::env::remove_var(name);
                    }
                }
            }
            Self { previous }
        }
    }

    impl Drop for EmbeddingApiEnv {
        fn drop(&mut self) {
            for (name, value) in self.previous.drain(..) {
                unsafe {
                    if let Some(value) = value {
                        std::env::set_var(name, value);
                    } else {
                        std::env::remove_var(name);
                    }
                }
            }
        }
    }

    fn api_config(vault_root: &Path, model: &str, watch: bool) -> Config {
        Config {
            vault_path: vault_root.to_path_buf(),
            watch,
            log_level: "error".into(),
            transport: obsidian_mcp::config::Transport::Stdio,
            http_host: obsidian_mcp::config::DEFAULT_HTTP_HOST,
            http_port: obsidian_mcp::config::DEFAULT_HTTP_PORT,
            tantivy: true,
            embeddings: true,
            embeddings_model: model.into(),
            hybrid_alpha: 0.25,
            embedding_provider: Some(EmbeddingProvider::Api),
            tool_filter: ToolFilter::Full,
            mcp_data_dir: None,
            exclude_patterns: Vec::new(),
        }
    }

    fn create_api_vault() -> tempfile::TempDir {
        let vault = tempfile::tempdir().expect("temporary vault");
        std::fs::create_dir_all(vault.path().join(".obsidian")).expect("create .obsidian");
        vault
    }

    fn cache_path(vault_root: &Path) -> PathBuf {
        vault_root
            .join(".obsidian-mcp")
            .join("embeddings")
            .join("embeddings.bin")
    }

    async fn wait_for_status<F>(vault: &Vault, label: &str, predicate: F) -> EmbeddingRuntimeStatus
    where
        F: Fn(&EmbeddingRuntimeStatus) -> bool,
    {
        tokio::time::timeout(Duration::from_secs(15), async {
            loop {
                let status = vault
                    .embedding_status()
                    .expect("embedding runtime should be configured");
                if predicate(&status) {
                    return status;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap_or_else(|_| {
            panic!(
                "timed out waiting for {label}; status: {:?}",
                vault.embedding_status()
            )
        })
    }

    async fn wait_for_cache_bytes(cache: &Path, previous: Option<&[u8]>) -> Vec<u8> {
        tokio::time::timeout(Duration::from_secs(15), async {
            loop {
                if let Ok(bytes) = std::fs::read(cache)
                    && !bytes.is_empty()
                    && previous.is_none_or(|old| old != bytes)
                {
                    return bytes;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("timed out waiting for cache update: {}", cache.display()))
    }

    #[tokio::test]
    async fn mcp_initialize_and_core_tool_do_not_wait_for_embedding_loader() {
        let _guard = API_TEST_LOCK.lock().await;
        let api = ControlledEmbeddingApi::start().await;
        api.block_inputs_containing(&["dim"]);
        let vault = create_api_vault();
        std::fs::write(vault.path().join("note.md"), "# Note\ncore content\n").unwrap();

        let mut child = tokio::process::Command::new(env!("CARGO_BIN_EXE_obsidian-mcp"))
            .env("OBSIDIAN_VAULT_PATH", vault.path())
            .env("OBSIDIAN_WATCH", "false")
            .env("OBSIDIAN_TANTIVY", "false")
            .env("OBSIDIAN_EMBEDDINGS", "true")
            .env("OBSIDIAN_EMBEDDINGS_MODEL", "api-integration-model")
            .env("OBSIDIAN_EMBEDDING_PROVIDER", "api")
            .env("OBSIDIAN_EMBEDDING_API_BASE", api.base_url())
            .env("OBSIDIAN_EMBEDDING_API_KEY", "integration-test-key")
            .env("OBSIDIAN_SEMANTIC_MODE", "local")
            .env("OBSIDIAN_LOG_LEVEL", "error")
            .env_remove("OBSIDIAN_EMBEDDING_API_MODEL")
            .env_remove("OPENAI_MODEL")
            .env_remove("OBSIDIAN_EMBEDDING_DIM")
            .env_remove("OBSIDIAN_EMBEDDING_CA_CERT")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .expect("spawn MCP binary");
        let mut stdin = child.stdin.take().expect("child stdin");
        let stdout = child.stdout.take().expect("child stdout");
        let mut lines = BufReader::new(stdout).lines();

        api.wait_for_input("dim").await;
        let initialize = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "startup-barrier-test", "version": "0.0.1"}
            }
        });
        stdin
            .write_all(format!("{initialize}\n").as_bytes())
            .await
            .expect("write initialize");
        let initialize_response = tokio::time::timeout(Duration::from_secs(2), lines.next_line())
            .await
            .expect("initialize must not wait for embeddings")
            .expect("read initialize response")
            .expect("initialize response line");
        let initialize_response: Value = serde_json::from_str(&initialize_response).unwrap();
        assert!(initialize_response.get("result").is_some());

        stdin
            .write_all(
                format!(
                    "{}\n",
                    json!({"jsonrpc": "2.0", "method": "notifications/initialized"})
                )
                .as_bytes(),
            )
            .await
            .expect("write initialized notification");
        let core_call = json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "vault_info", "arguments": {}}
        });
        stdin
            .write_all(format!("{core_call}\n").as_bytes())
            .await
            .expect("write core tool call");
        let core_response = tokio::time::timeout(Duration::from_secs(2), lines.next_line())
            .await
            .expect("core tool must not wait for embeddings")
            .expect("read core response")
            .expect("core response line");
        let core_response: Value = serde_json::from_str(&core_response).unwrap();
        assert!(core_response.get("result").is_some(), "{core_response}");

        api.release_blocked();
        stdin.shutdown().await.expect("close MCP stdin");
        drop(stdin);
        if tokio::time::timeout(Duration::from_secs(3), child.wait())
            .await
            .is_err()
        {
            child.kill().await.expect("kill test MCP process");
        }
        api.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn compatible_cache_serves_lkg_while_reconciling_and_prunes_exclusions() {
        let _guard = API_TEST_LOCK.lock().await;
        let api = ControlledEmbeddingApi::start().await;
        let _env = EmbeddingApiEnv::install(&api.base_url());
        let vault_dir = create_api_vault();
        std::fs::create_dir_all(vault_dir.path().join("Archive")).unwrap();
        std::fs::write(
            vault_dir.path().join("active.md"),
            "# Active\nBASELINE_MARKER semantic content\n",
        )
        .unwrap();
        std::fs::write(
            vault_dir.path().join("Archive/hidden.md"),
            "# Hidden\nHIDDEN_MARKER old archive content\n",
        )
        .unwrap();
        let model = "cache-compatibility-model";
        let config = api_config(vault_dir.path(), model, false);

        let first = Vault::open(&config).await.unwrap();
        wait_for_status(&first, "initial ready cache", |status| {
            status.phase == EmbeddingPhase::Ready && status.indexed_notes == 2
        })
        .await;
        let cache = cache_path(vault_dir.path());
        let old_cache = wait_for_cache_bytes(&cache, None).await;
        drop(first);

        std::fs::write(
            vault_dir.path().join("active.md"),
            "# Active\nBLOCK_CHANGED latest semantic content\n",
        )
        .unwrap();
        api.clear_requests();
        api.block_inputs_containing(&["BLOCK_CHANGED"]);
        let mut pruned_config = api_config(vault_dir.path(), model, false);
        pruned_config.exclude_patterns = vec!["Archive/".into()];
        let second = Vault::open(&pruned_config).await.unwrap();
        api.wait_for_input("BLOCK_CHANGED").await;
        let warming = wait_for_status(&second, "queryable last-known-good cache", |status| {
            status.phase == EmbeddingPhase::Warming && status.queryable
        })
        .await;
        assert_eq!(warming.indexed_notes, 1);
        assert_eq!(warming.total_notes, 1);

        let results = second.search_semantic("query-ready", 10).unwrap();
        assert!(
            results
                .iter()
                .any(|(path, _)| path == Path::new("active.md"))
        );
        assert!(
            results
                .iter()
                .all(|(path, _)| path != Path::new("Archive/hidden.md")),
            "newly excluded cached paths must be pruned before publication"
        );

        api.release_blocked();
        wait_for_status(&second, "changed-note reconciliation", |status| {
            status.phase == EmbeddingPhase::Ready && status.indexed_notes == 1
        })
        .await;
        let new_cache = wait_for_cache_bytes(&cache, Some(&old_cache)).await;
        drop(second);

        api.clear_requests();
        let third = Vault::open(&pruned_config).await.unwrap();
        wait_for_status(&third, "unchanged-cache restart", |status| {
            status.phase == EmbeddingPhase::Ready && status.indexed_notes == 1
        })
        .await;
        let restart_inputs = api.recorded_inputs();
        assert!(
            !restart_inputs
                .iter()
                .any(|input| input.contains("BLOCK_CHANGED") || input.contains("HIDDEN_MARKER")),
            "unchanged and excluded notes must not be inferred on restart: {restart_inputs:?}"
        );
        assert_ne!(old_cache, new_cache);
        drop(third);
        api.shutdown().await;
    }

    #[derive(serde::Serialize)]
    struct LegacyCache {
        dim: usize,
        entries: Vec<(String, Vec<f32>)>,
    }

    #[tokio::test]
    async fn invalid_and_wrong_space_caches_rebuild_only_in_background() {
        let _guard = API_TEST_LOCK.lock().await;
        let api = ControlledEmbeddingApi::start().await;
        let _env = EmbeddingApiEnv::install(&api.base_url());

        let seed = create_api_vault();
        std::fs::write(seed.path().join("note.md"), "# Note\nSEED_MARKER\n").unwrap();
        let seed_config = api_config(seed.path(), "seed-model", false);
        let seed_vault = Vault::open(&seed_config).await.unwrap();
        wait_for_status(&seed_vault, "seed cache", |status| {
            status.phase == EmbeddingPhase::Ready
        })
        .await;
        let valid_cache = wait_for_cache_bytes(&cache_path(seed.path()), None).await;
        drop(seed_vault);

        let legacy = bincode::serde::encode_to_vec(
            LegacyCache {
                dim: 3,
                entries: vec![("note.md".into(), vec![1.0, 0.0, 0.0])],
            },
            bincode::config::standard(),
        )
        .unwrap();
        let mut trailing = valid_cache.clone();
        trailing.extend_from_slice(b"trailing-bytes");
        let cases = vec![
            ("legacy", legacy, "seed-model"),
            ("corrupt", vec![0xff; 64], "seed-model"),
            (
                "truncated",
                valid_cache[..valid_cache.len() / 2].to_vec(),
                "seed-model",
            ),
            ("trailing", trailing, "seed-model"),
            ("wrong-model", valid_cache.clone(), "different-model"),
        ];

        for (name, bytes, model) in cases {
            let vault_dir = create_api_vault();
            std::fs::write(
                vault_dir.path().join("note.md"),
                format!("# Note\nCASE_MARKER_{name}\n"),
            )
            .unwrap();
            let cache = cache_path(vault_dir.path());
            std::fs::create_dir_all(cache.parent().unwrap()).unwrap();
            std::fs::write(&cache, bytes).unwrap();
            api.clear_requests();
            api.block_inputs_containing(&["CASE_MARKER"]);

            let vault = tokio::time::timeout(
                Duration::from_secs(2),
                Vault::open(&api_config(vault_dir.path(), model, false)),
            )
            .await
            .unwrap_or_else(|_| panic!("{name} cache blocked Vault::open"))
            .unwrap();
            api.wait_for_input(&format!("CASE_MARKER_{name}")).await;
            let status = vault.embedding_status().unwrap();
            assert!(
                !status.queryable,
                "{name} cache must not become queryable before rebuild: {status:?}"
            );

            api.release_blocked();
            wait_for_status(&vault, &format!("{name} cache rebuild"), |status| {
                status.phase == EmbeddingPhase::Ready && status.queryable
            })
            .await;
            drop(vault);
        }

        api.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn direct_and_watcher_updates_are_latest_wins_during_reconciliation() {
        let _guard = API_TEST_LOCK.lock().await;
        let api = ControlledEmbeddingApi::start().await;
        let _env = EmbeddingApiEnv::install(&api.base_url());
        let vault_dir = create_api_vault();
        std::fs::write(
            vault_dir.path().join("watched.md"),
            "# Watched\nBLOCK_INITIAL stale content\n",
        )
        .unwrap();
        api.block_inputs_containing(&["BLOCK_INITIAL"]);
        let vault = Vault::open(&api_config(vault_dir.path(), "latest-wins-model", true))
            .await
            .unwrap();
        api.wait_for_input("BLOCK_INITIAL").await;

        std::fs::write(
            vault_dir.path().join("watched.md"),
            "# Watched\nWATCH_FINAL authoritative watcher content\n",
        )
        .unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if vault
                    .search_text("WATCH_FINAL", 20)
                    .unwrap()
                    .iter()
                    .any(|result| result.path == Path::new("watched.md"))
                {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("watcher should update the primary index before release");

        vault
            .write_note(Path::new("direct.md"), "# Direct\nDIRECT_OLD\n")
            .unwrap();
        vault
            .write_note(Path::new("direct.md"), "# Direct\nDIRECT_FINAL\n")
            .unwrap();
        vault
            .write_note(Path::new("deleted.md"), "# Deleted\nDELETE_ME\n")
            .unwrap();
        vault.delete_note(Path::new("deleted.md")).unwrap();

        api.release_blocked();
        wait_for_status(&vault, "latest-wins reconciliation", |status| {
            status.phase == EmbeddingPhase::Ready
                && status.indexed_notes == 2
                && status.total_notes == 2
        })
        .await;
        let inputs = api.recorded_inputs();
        assert!(inputs.iter().any(|input| input.contains("WATCH_FINAL")));
        assert!(inputs.iter().any(|input| input.contains("DIRECT_FINAL")));
        assert!(!inputs.iter().any(|input| input.contains("DIRECT_OLD")));
        assert!(!inputs.iter().any(|input| input.contains("DELETE_ME")));

        let results = vault.search_semantic("query-ready", 10).unwrap();
        assert!(
            results
                .iter()
                .any(|(path, _)| path == Path::new("watched.md"))
        );
        assert!(
            results
                .iter()
                .any(|(path, _)| path == Path::new("direct.md"))
        );
        assert!(
            results
                .iter()
                .all(|(path, _)| path != Path::new("deleted.md"))
        );
        drop(vault);
        api.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn failed_refresh_serves_lkg_without_phantoms_then_recovers() {
        let _guard = API_TEST_LOCK.lock().await;
        let api = ControlledEmbeddingApi::start().await;
        let _env = EmbeddingApiEnv::install(&api.base_url());
        let vault_dir = create_api_vault();
        std::fs::write(
            vault_dir.path().join("old.md"),
            "# Old\nOLD_MARKER stable content\n",
        )
        .unwrap();
        let config = api_config(vault_dir.path(), "failure-recovery-model", false);
        let first = Vault::open(&config).await.unwrap();
        wait_for_status(&first, "failure seed", |status| {
            status.phase == EmbeddingPhase::Ready
        })
        .await;
        let cache = cache_path(vault_dir.path());
        let old_cache = wait_for_cache_bytes(&cache, None).await;
        drop(first);

        std::fs::write(
            vault_dir.path().join("old.md"),
            "# Old\nFAIL_MARKER changed existing note\n",
        )
        .unwrap();
        std::fs::write(
            vault_dir.path().join("new.md"),
            "# New\nFAIL_MARKER brand new note\n",
        )
        .unwrap();
        api.clear_requests();
        api.fail_inputs_containing(&["FAIL_MARKER"]);
        let second = Vault::open(&config).await.unwrap();
        let degraded = wait_for_status(&second, "failed refresh", |status| {
            status.phase == EmbeddingPhase::Degraded && status.queryable
        })
        .await;
        assert_eq!(degraded.indexed_notes, 1);
        assert_eq!(degraded.total_notes, 2);
        let error = degraded.last_error.expect("failure should be reported");
        assert!(error.contains("HTTP status 500"));
        assert!(!error.contains("sensitive note content"));

        let lkg_results = second.search_semantic("query-ready", 10).unwrap();
        assert!(
            lkg_results
                .iter()
                .any(|(path, _)| path == Path::new("old.md"))
        );
        assert!(
            lkg_results
                .iter()
                .all(|(path, _)| path != Path::new("new.md")),
            "a failed new note must not gain a phantom vector"
        );

        api.clear_failures();
        wait_for_status(&second, "refresh retry recovery", |status| {
            status.phase == EmbeddingPhase::Ready
                && status.indexed_notes == 2
                && status.last_error.is_none()
        })
        .await;
        wait_for_cache_bytes(&cache, Some(&old_cache)).await;
        drop(second);
        api.shutdown().await;
    }

    #[tokio::test]
    async fn note_read_failure_is_degraded_and_retry_recovers() {
        let _guard = API_TEST_LOCK.lock().await;
        let api = ControlledEmbeddingApi::start().await;
        let _env = EmbeddingApiEnv::install(&api.base_url());
        let vault_dir = create_api_vault();
        std::fs::write(
            vault_dir.path().join("note.md"),
            "# Note\nvalid before loader\n",
        )
        .unwrap();
        api.block_inputs_containing(&["dim"]);
        let vault = Vault::open(&api_config(vault_dir.path(), "read-retry-model", false))
            .await
            .unwrap();
        api.wait_for_input("dim").await;
        std::fs::write(vault_dir.path().join("note.md"), b"\xff\xfe\xfd").unwrap();
        api.release_blocked();

        let degraded = wait_for_status(&vault, "note read degradation", |status| {
            status.phase == EmbeddingPhase::Degraded && !status.queryable
        })
        .await;
        assert_eq!(degraded.indexed_notes, 0);
        assert_eq!(degraded.total_notes, 1);
        assert!(degraded.last_error.is_some());

        std::fs::write(
            vault_dir.path().join("note.md"),
            "# Note\nREAD_RECOVERED valid content\n",
        )
        .unwrap();
        wait_for_status(&vault, "note read retry", |status| {
            status.phase == EmbeddingPhase::Ready
                && status.queryable
                && status.indexed_notes == 1
                && status.last_error.is_none()
        })
        .await;
        assert!(
            api.recorded_inputs()
                .iter()
                .any(|input| input.contains("READ_RECOVERED"))
        );
        drop(vault);
        api.shutdown().await;
    }
}
