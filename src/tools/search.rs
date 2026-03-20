//! Text, regex, tag, and frontmatter search tools across vault notes.

use rmcp::model::{CallToolResult, Content, ErrorCode};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::error::VaultError;
use crate::vault::Vault;

// ── search_text ─────────────────────────────────────────────────────

#[derive(Deserialize, JsonSchema, Default)]
pub struct SearchTextParams {
    /// Full-text search query (case-insensitive).
    pub query: String,
    /// Characters of context around each match (default: 100).
    #[serde(default)]
    pub context_length: Option<usize>,
    /// Maximum number of file results to return (default: 20).
    #[serde(default)]
    pub max_results: Option<usize>,
}

pub async fn search_text(
    vault: &Vault,
    params: SearchTextParams,
) -> Result<CallToolResult, rmcp::ErrorData> {
    let context_length = params.context_length.unwrap_or(100);
    let max_results = params.max_results.unwrap_or(20);

    let results = vault.search_text(&params.query, context_length)?;
    let limited: Vec<_> = results.into_iter().take(max_results).collect();

    let json = serde_json::to_string_pretty(&limited)
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
}
