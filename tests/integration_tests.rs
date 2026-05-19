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
        }
    }

    async fn open_with_embeddings(vault_root: &Path) -> Vault {
        let _guard = MODEL_LOCK.lock().await;
        let config = embeddings_config(vault_root);
        Vault::open(&config)
            .await
            .expect("open vault with embeddings")
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

        let results = vault.search_semantic("memory safe programming", 5).unwrap();
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
        Vault::open(&config)
            .await
            .expect("open vault with tantivy + embeddings")
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
        }
    }

    fn extract_text(result: &rmcp::model::CallToolResult) -> &str {
        result.content[0]
            .as_text()
            .expect("expected text content")
            .text
            .as_str()
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

        let result = search_semantic(
            &vault,
            SearchSemanticParams {
                query: "memory safety".to_string(),
                top_k: Some(5),
                include_content: Some(false),
                lexical_prefetch: Some(false),
                alpha: None,
            },
            0.25,
            &runtime,
        )
        .await
        .expect("auto mode should fall back to local backend");
        let parsed: Vec<serde_json::Value> =
            serde_json::from_str(extract_text(&result)).expect("parse semantic result");
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
