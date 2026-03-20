//! Text, regex, tag, and frontmatter search tools across vault notes.

use rmcp::model::{CallToolResult, Content, ErrorCode};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::error::VaultError;
use crate::models::SearchField;
use crate::vault::Vault;

// ── search_text ─────────────────────────────────────────────────────

#[derive(Deserialize, JsonSchema, Default)]
pub struct SearchTextParams {
    /// Natural-language search query. Supports stemming (e.g. "program"
    /// matches "programming"). Results are ranked by BM25 relevance.
    pub query: String,
    /// Characters of context around each match (default: 100).
    #[serde(default)]
    pub context_length: Option<usize>,
    /// Maximum number of file results to return (default: 20).
    #[serde(default)]
    pub max_results: Option<usize>,
    /// Enable fuzzy matching with edit distance 1 (tolerates typos). Default: false.
    #[serde(default)]
    pub fuzzy: Option<bool>,
    /// Restrict search to specific note fields. Default: all fields.
    /// Allowed values: `title`, `headings`, `tags`, `body`, `frontmatter`.
    #[serde(default)]
    pub fields: Option<Vec<SearchField>>,
}

pub async fn search_text(
    vault: &Vault,
    params: SearchTextParams,
) -> Result<CallToolResult, rmcp::ErrorData> {
    let context_length = params.context_length.unwrap_or(100);
    let max_results = params.max_results.unwrap_or(20);
    let fuzzy = params.fuzzy.unwrap_or(false);

    let results = if fuzzy || params.fields.is_some() {
        let fields_slice = params.fields.as_deref();
        vault.search_text_with_options(
            &params.query,
            context_length,
            max_results,
            fuzzy,
            fields_slice,
        )?
    } else {
        let all = vault.search_text(&params.query, context_length)?;
        all.into_iter().take(max_results).collect()
    };

    let json = serde_json::to_string_pretty(&results)
        .map_err(|e| VaultError::Other(format!("JSON serialization failed: {e}")))?;
    Ok(CallToolResult::success(vec![Content::text(json)]))
}

// ── search_regex ────────────────────────────────────────────────────

#[derive(Deserialize, JsonSchema, Default)]
pub struct SearchRegexParams {
    /// Regular expression pattern to search for.
    pub pattern: String,
    /// Characters of context around each match (default: 100).
    #[serde(default)]
    pub context_length: Option<usize>,
    /// Maximum number of file results to return (default: 20).
    #[serde(default)]
    pub max_results: Option<usize>,
}

pub async fn search_regex(
    vault: &Vault,
    params: SearchRegexParams,
) -> Result<CallToolResult, rmcp::ErrorData> {
    let context_length = params.context_length.unwrap_or(100);
    let max_results = params.max_results.unwrap_or(20);

    let results = vault.search_regex(&params.pattern, context_length)?;
    let limited: Vec<_> = results.into_iter().take(max_results).collect();

    let json = serde_json::to_string_pretty(&limited)
        .map_err(|e| VaultError::Other(format!("JSON serialization failed: {e}")))?;
    Ok(CallToolResult::success(vec![Content::text(json)]))
}

// ── search_tag ──────────────────────────────────────────────────────

#[derive(Deserialize, JsonSchema, Default)]
pub struct SearchTagParams {
    /// Tag to search for (without the `#` prefix).
    pub tag: String,
    /// If true, also match nested tags (e.g. `inbox` matches `inbox/read`). Default: true.
    #[serde(default)]
    pub include_nested: Option<bool>,
}

pub async fn search_tag(
    vault: &Vault,
    params: SearchTagParams,
) -> Result<CallToolResult, rmcp::ErrorData> {
    let tag = params.tag.strip_prefix('#').unwrap_or(&params.tag);
    let include_nested = params.include_nested.unwrap_or(true);

    let results = if include_nested {
        vault.search_by_tag_prefix(tag)?
    } else {
        vault.search_by_tag(tag)?
    };

    let json = serde_json::to_string_pretty(&results)
        .map_err(|e| VaultError::Other(format!("JSON serialization failed: {e}")))?;
    Ok(CallToolResult::success(vec![Content::text(json)]))
}

// ── search_frontmatter ─────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "snake_case")]
pub enum FrontmatterOperator {
    /// Exact equality (or array-contains for list fields).
    #[default]
    Eq,
    /// Substring match for strings; element membership for arrays.
    Contains,
    /// Field exists regardless of value.
    Exists,
}

#[derive(Deserialize, JsonSchema, Default)]
pub struct SearchFrontmatterParams {
    /// Frontmatter field name to query.
    pub field: String,
    /// Value to compare against. Required for `eq` and `contains`; ignored for `exists`.
    #[serde(default)]
    pub value: Option<serde_json::Value>,
    /// Comparison operator (default: `eq`).
    #[serde(default)]
    pub operator: FrontmatterOperator,
}

pub async fn search_frontmatter(
    vault: &Vault,
    params: SearchFrontmatterParams,
) -> Result<CallToolResult, rmcp::ErrorData> {
    let results = match params.operator {
        FrontmatterOperator::Exists => vault.search_frontmatter_exists(&params.field)?,
        FrontmatterOperator::Eq => {
            let value = params.value.ok_or_else(|| {
                rmcp::ErrorData::new(
                    ErrorCode::INVALID_PARAMS,
                    "'value' is required for 'eq' operator",
                    None::<serde_json::Value>,
                )
            })?;
            vault.search_frontmatter(&params.field, &value)?
        }
        FrontmatterOperator::Contains => {
            let value = params.value.ok_or_else(|| {
                rmcp::ErrorData::new(
                    ErrorCode::INVALID_PARAMS,
                    "'value' is required for 'contains' operator",
                    None::<serde_json::Value>,
                )
            })?;
            vault.search_frontmatter_contains(&params.field, &value)?
        }
    };

    let json = serde_json::to_string_pretty(&results)
        .map_err(|e| VaultError::Other(format!("JSON serialization failed: {e}")))?;
    Ok(CallToolResult::success(vec![Content::text(json)]))
}

// ── search_semantic ──────────────────────────────────────────────────

#[cfg(feature = "embeddings")]
const DEFAULT_PREFETCH_COUNT: usize = 50;
#[cfg(feature = "embeddings")]
const DEFAULT_ALPHA: f32 = 0.4;

#[derive(Deserialize, JsonSchema, Default)]
pub struct SearchSemanticParams {
    /// Natural-language query for semantic search. Does not require exact
    /// keyword matches — conceptually similar notes are returned.
    pub query: String,
    /// Number of results to return (default: 10).
    #[serde(default)]
    pub top_k: Option<usize>,
    /// If true, include the full note content in each result. Default: false.
    #[serde(default)]
    pub include_content: Option<bool>,
    /// When true, first retrieves top candidates via BM25 lexical search,
    /// then re-ranks by combining lexical and semantic scores. Produces
    /// higher-quality results than either approach alone. Requires both
    /// Tantivy and embeddings to be enabled. Default: false.
    #[serde(default)]
    pub lexical_prefetch: Option<bool>,
}

#[cfg(feature = "embeddings")]
#[derive(serde::Serialize, JsonSchema)]
struct SemanticSearchResult {
    path: std::path::PathBuf,
    title: String,
    score: f32,
    tags: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

#[cfg(feature = "embeddings")]
pub async fn search_semantic(
    vault: &Vault,
    params: SearchSemanticParams,
) -> Result<CallToolResult, rmcp::ErrorData> {
    if !vault.has_embeddings() {
        return Err(rmcp::ErrorData::new(
            ErrorCode::INVALID_REQUEST,
            "Embeddings are not enabled. Set OBSIDIAN_EMBEDDINGS=true and build with --features embeddings.",
            None::<serde_json::Value>,
        ));
    }

    let top_k = params.top_k.unwrap_or(10);
    let include_content = params.include_content.unwrap_or(false);
    let lexical_prefetch = params.lexical_prefetch.unwrap_or(false);

    let hits = if lexical_prefetch {
        vault.search_hybrid(&params.query, top_k, DEFAULT_PREFETCH_COUNT, DEFAULT_ALPHA)?
    } else {
        vault.search_semantic(&params.query, top_k)?
    };

    let mut results = Vec::with_capacity(hits.len());
    for (path, score) in hits {
        let meta = vault.get_note_metadata(&path).ok();
        let title = meta.as_ref().map(|m| m.title.clone()).unwrap_or_default();
        let tags = meta.as_ref().map(|m| m.tags.clone()).unwrap_or_default();
        let content = if include_content {
            vault.read_note(&path).ok()
        } else {
            None
        };
        results.push(SemanticSearchResult {
            path,
            title,
            score,
            tags,
            content,
        });
    }

    let json = serde_json::to_string_pretty(&results)
        .map_err(|e| VaultError::Other(format!("JSON serialization failed: {e}")))?;
    Ok(CallToolResult::success(vec![Content::text(json)]))
}

#[cfg(not(feature = "embeddings"))]
pub async fn search_semantic(
    _vault: &Vault,
    _params: SearchSemanticParams,
) -> Result<CallToolResult, rmcp::ErrorData> {
    Err(rmcp::ErrorData::new(
        ErrorCode::INVALID_REQUEST,
        "Semantic search is not available. This binary was compiled without the 'embeddings' feature. Rebuild with: cargo build --features embeddings",
        None::<serde_json::Value>,
    ))
}

// ── tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::config::Config;

    fn test_config(vault_root: &Path) -> Config {
        Config {
            vault_path: vault_root.to_path_buf(),
            watch: false,
            log_level: "error".into(),
            tantivy: false,
            embeddings: false,
            embeddings_model: String::new(),
        }
    }

    fn create_test_vault(dir: &Path) {
        std::fs::create_dir_all(dir.join(".obsidian")).unwrap();
    }

    fn extract_text(result: &CallToolResult) -> &str {
        result.content[0]
            .as_text()
            .expect("expected text content")
            .text
            .as_str()
    }

    async fn setup_search_vault() -> (tempfile::TempDir, Vault) {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());

        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        vault
            .write_note(
                Path::new("rust.md"),
                "---\ntags: [lang, systems]\nstatus: stable\n---\n# Rust\nRust is a systems language.\n",
            )
            .unwrap();
        vault
            .write_note(
                Path::new("python.md"),
                "---\ntags: [lang, scripting]\nstatus: in progress\n---\n# Python\nPython is dynamic.\n",
            )
            .unwrap();
        vault
            .write_note(
                Path::new("notes.md"),
                "# Notes\nSome random notes about #inbox stuff.\n\n#inbox/read #inbox/todo\n",
            )
            .unwrap();
        vault
            .write_note(
                Path::new("empty.md"),
                "# Empty\nNothing interesting here.\n",
            )
            .unwrap();

        (dir, vault)
    }

    // ── search_text ─────────────────────────────────────────────────

    #[tokio::test]
    async fn search_text_finds_match() {
        let (_dir, vault) = setup_search_vault().await;
        let result = search_text(
            &vault,
            SearchTextParams {
                query: "systems".into(),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let text = extract_text(&result);

        assert!(text.contains("rust.md"));
        assert!(!text.contains("python.md"));
    }

    #[tokio::test]
    async fn search_text_limits_results() {
        let (_dir, vault) = setup_search_vault().await;
        let result = search_text(
            &vault,
            SearchTextParams {
                query: "is".into(),
                max_results: Some(1),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let text = extract_text(&result);

        let parsed: Vec<serde_json::Value> = serde_json::from_str(text).unwrap();
        assert_eq!(parsed.len(), 1);
    }

    #[tokio::test]
    async fn search_text_empty_query_returns_empty() {
        let (_dir, vault) = setup_search_vault().await;
        let result = search_text(
            &vault,
            SearchTextParams {
                query: String::new(),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let text = extract_text(&result);

        let parsed: Vec<serde_json::Value> = serde_json::from_str(text).unwrap();
        assert!(parsed.is_empty());
    }

    // ── search_regex ────────────────────────────────────────────────

    #[tokio::test]
    async fn search_regex_valid_pattern() {
        let (_dir, vault) = setup_search_vault().await;
        let result = search_regex(
            &vault,
            SearchRegexParams {
                pattern: r"(?i)python".into(),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let text = extract_text(&result);

        assert!(text.contains("python.md"));
    }

    #[tokio::test]
    async fn search_regex_invalid_pattern_returns_error() {
        let (_dir, vault) = setup_search_vault().await;
        let result = search_regex(
            &vault,
            SearchRegexParams {
                pattern: "[invalid".into(),
                ..Default::default()
            },
        )
        .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn search_regex_limits_results() {
        let (_dir, vault) = setup_search_vault().await;
        let result = search_regex(
            &vault,
            SearchRegexParams {
                pattern: r"\w+".into(),
                max_results: Some(2),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let text = extract_text(&result);

        let parsed: Vec<serde_json::Value> = serde_json::from_str(text).unwrap();
        assert!(parsed.len() <= 2);
    }

    // ── search_tag ──────────────────────────────────────────────────

    #[tokio::test]
    async fn search_tag_exact() {
        let (_dir, vault) = setup_search_vault().await;
        let result = search_tag(
            &vault,
            SearchTagParams {
                tag: "inbox".into(),
                include_nested: Some(false),
            },
        )
        .await
        .unwrap();
        let text = extract_text(&result);

        assert!(text.contains("notes.md"));
        let parsed: Vec<serde_json::Value> = serde_json::from_str(text).unwrap();
        assert_eq!(parsed.len(), 1);
    }

    #[tokio::test]
    async fn search_tag_include_nested() {
        let (_dir, vault) = setup_search_vault().await;
        let result = search_tag(
            &vault,
            SearchTagParams {
                tag: "inbox".into(),
                include_nested: Some(true),
            },
        )
        .await
        .unwrap();
        let text = extract_text(&result);

        assert!(text.contains("notes.md"));
    }

    #[tokio::test]
    async fn search_tag_strips_hash_prefix() {
        let (_dir, vault) = setup_search_vault().await;
        let result = search_tag(
            &vault,
            SearchTagParams {
                tag: "#lang".into(),
                include_nested: Some(false),
            },
        )
        .await
        .unwrap();
        let text = extract_text(&result);

        let parsed: Vec<serde_json::Value> = serde_json::from_str(text).unwrap();
        assert_eq!(parsed.len(), 2);
    }

    // ── search_frontmatter ──────────────────────────────────────────

    #[tokio::test]
    async fn search_frontmatter_eq() {
        let (_dir, vault) = setup_search_vault().await;
        let result = search_frontmatter(
            &vault,
            SearchFrontmatterParams {
                field: "status".into(),
                value: Some(serde_json::json!("stable")),
                operator: FrontmatterOperator::Eq,
            },
        )
        .await
        .unwrap();
        let text = extract_text(&result);

        assert!(text.contains("rust.md"));
        let parsed: Vec<serde_json::Value> = serde_json::from_str(text).unwrap();
        assert_eq!(parsed.len(), 1);
    }

    #[tokio::test]
    async fn search_frontmatter_eq_array_contains() {
        let (_dir, vault) = setup_search_vault().await;
        let result = search_frontmatter(
            &vault,
            SearchFrontmatterParams {
                field: "tags".into(),
                value: Some(serde_json::json!("systems")),
                operator: FrontmatterOperator::Eq,
            },
        )
        .await
        .unwrap();
        let text = extract_text(&result);

        assert!(text.contains("rust.md"));
    }

    #[tokio::test]
    async fn search_frontmatter_contains_substring() {
        let (_dir, vault) = setup_search_vault().await;
        let result = search_frontmatter(
            &vault,
            SearchFrontmatterParams {
                field: "status".into(),
                value: Some(serde_json::json!("progress")),
                operator: FrontmatterOperator::Contains,
            },
        )
        .await
        .unwrap();
        let text = extract_text(&result);

        assert!(text.contains("python.md"));
        let parsed: Vec<serde_json::Value> = serde_json::from_str(text).unwrap();
        assert_eq!(parsed.len(), 1);
    }

    #[tokio::test]
    async fn search_frontmatter_exists() {
        let (_dir, vault) = setup_search_vault().await;
        let result = search_frontmatter(
            &vault,
            SearchFrontmatterParams {
                field: "status".into(),
                value: None,
                operator: FrontmatterOperator::Exists,
            },
        )
        .await
        .unwrap();
        let text = extract_text(&result);

        let parsed: Vec<serde_json::Value> = serde_json::from_str(text).unwrap();
        assert_eq!(parsed.len(), 2); // rust.md + python.md
    }

    #[tokio::test]
    async fn search_frontmatter_exists_missing_field() {
        let (_dir, vault) = setup_search_vault().await;
        let result = search_frontmatter(
            &vault,
            SearchFrontmatterParams {
                field: "nonexistent".into(),
                value: None,
                operator: FrontmatterOperator::Exists,
            },
        )
        .await
        .unwrap();
        let text = extract_text(&result);

        let parsed: Vec<serde_json::Value> = serde_json::from_str(text).unwrap();
        assert!(parsed.is_empty());
    }

    #[tokio::test]
    async fn search_frontmatter_eq_without_value_errors() {
        let (_dir, vault) = setup_search_vault().await;
        let result = search_frontmatter(
            &vault,
            SearchFrontmatterParams {
                field: "status".into(),
                value: None,
                operator: FrontmatterOperator::Eq,
            },
        )
        .await;

        assert!(result.is_err());
    }

    // ── search_text with Tantivy BM25 ──────────────────────────────

    fn tantivy_config(vault_root: &Path) -> Config {
        Config {
            vault_path: vault_root.to_path_buf(),
            watch: false,
            log_level: "error".into(),
            tantivy: true,
            embeddings: false,
            embeddings_model: String::new(),
        }
    }

    async fn setup_tantivy_vault() -> (tempfile::TempDir, Vault) {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());

        let vault = Vault::open(&tantivy_config(dir.path())).await.unwrap();

        vault
            .write_note(
                Path::new("rust.md"),
                "---\ntags: [lang, systems]\nstatus: stable\n---\n# Rust\nRust is a systems programming language.\n",
            )
            .unwrap();
        vault
            .write_note(
                Path::new("python.md"),
                "---\ntags: [lang, scripting]\nstatus: in progress\n---\n# Python\nPython is a dynamic scripting language.\n",
            )
            .unwrap();
        vault
            .write_note(
                Path::new("cooking.md"),
                "# Cooking Tips\nHow to make a great pasta dish.\n",
            )
            .unwrap();

        (dir, vault)
    }

    #[tokio::test]
    async fn search_text_tantivy_returns_scores() {
        let (_dir, vault) = setup_tantivy_vault().await;
        let result = search_text(
            &vault,
            SearchTextParams {
                query: "systems".into(),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let text = extract_text(&result);
        let parsed: Vec<serde_json::Value> = serde_json::from_str(text).unwrap();

        assert!(!parsed.is_empty());
        assert!(text.contains("rust.md"));
        // BM25 results should have a score
        assert!(parsed[0].get("score").is_some());
        assert!(parsed[0]["score"].as_f64().unwrap() > 0.0);
    }

    #[tokio::test]
    async fn search_text_tantivy_ranked_descending() {
        let (_dir, vault) = setup_tantivy_vault().await;
        let result = search_text(
            &vault,
            SearchTextParams {
                query: "language".into(),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let text = extract_text(&result);
        let parsed: Vec<serde_json::Value> = serde_json::from_str(text).unwrap();

        if parsed.len() >= 2 {
            let s0 = parsed[0]["score"].as_f64().unwrap();
            let s1 = parsed[1]["score"].as_f64().unwrap();
            assert!(s0 >= s1, "results should be sorted by score descending");
        }
    }

    #[tokio::test]
    async fn search_text_tantivy_fuzzy() {
        let (_dir, vault) = setup_tantivy_vault().await;
        let result = search_text(
            &vault,
            SearchTextParams {
                query: "pyhton".into(),
                fuzzy: Some(true),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let text = extract_text(&result);

        assert!(
            text.contains("python.md"),
            "fuzzy should match 'pyhton' to 'python'"
        );
    }

    #[tokio::test]
    async fn search_text_tantivy_field_filter() {
        let (_dir, vault) = setup_tantivy_vault().await;
        let result = search_text(
            &vault,
            SearchTextParams {
                query: "cooking".into(),
                fields: Some(vec![SearchField::Title]),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let text = extract_text(&result);

        assert!(
            text.contains("cooking.md"),
            "title search for 'cooking' should find cooking.md"
        );
    }

    #[tokio::test]
    async fn search_text_tantivy_context_snippets() {
        let (_dir, vault) = setup_tantivy_vault().await;
        let result = search_text(
            &vault,
            SearchTextParams {
                query: "pasta".into(),
                context_length: Some(50),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let text = extract_text(&result);
        let parsed: Vec<serde_json::Value> = serde_json::from_str(text).unwrap();

        assert!(!parsed.is_empty());
        let matches = parsed[0]["matches"].as_array().unwrap();
        assert!(!matches.is_empty(), "should have context matches");
        assert!(matches[0]["context"].as_str().unwrap().contains("pasta"));
    }
}
