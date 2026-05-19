//! Text, regex, tag, and frontmatter search tools across vault notes.

use rmcp::model::{CallToolResult, Content, ErrorCode};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::config::SemanticMode;
use crate::daemon::protocol;
use crate::error::VaultError;
use crate::models::SearchField;
use crate::vault::Vault;

use super::SemanticRuntime;

const MAX_RESULTS_CAP: usize = 200;
const MAX_CONTEXT_LEN_CAP: usize = 2000;

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
    let context_length = params
        .context_length
        .unwrap_or(100)
        .min(MAX_CONTEXT_LEN_CAP);
    let max_results = params.max_results.unwrap_or(20).min(MAX_RESULTS_CAP);
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
    let context_length = params
        .context_length
        .unwrap_or(100)
        .min(MAX_CONTEXT_LEN_CAP);
    let max_results = params.max_results.unwrap_or(20).min(MAX_RESULTS_CAP);

    let results = vault.search_regex(&params.pattern, context_length)?;
    let limited: Vec<_> = results.into_iter().take(max_results).collect();

    let json = serde_json::to_string_pretty(&limited)
        .map_err(|e| VaultError::Other(format!("JSON serialization failed: {e}")))?;
    Ok(CallToolResult::success(vec![Content::text(json)]))
}

// ── search_metadata ─────────────────────────────────────────────────

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
pub struct SearchMetadataParams {
    /// Type of metadata search: `"tag"` to find notes by tag, or `"frontmatter"` to query by frontmatter field.
    #[serde(rename = "type")]
    pub search_type: String,
    /// Tag to search for (without the `#` prefix). Required when type is `"tag"`.
    #[serde(default)]
    pub tag: Option<String>,
    /// If true, also match nested tags (e.g. `inbox` matches `inbox/read`). Default: true. Only used when type is `"tag"`.
    #[serde(default)]
    pub include_nested: Option<bool>,
    /// Frontmatter field name to query. Required when type is `"frontmatter"`.
    #[serde(default)]
    pub field: Option<String>,
    /// Value to compare against. Required for `eq` and `contains` operators; ignored for `exists`. Only used when type is `"frontmatter"`.
    #[serde(default)]
    pub value: Option<serde_json::Value>,
    /// Comparison operator (default: `eq`). Only used when type is `"frontmatter"`.
    #[serde(default)]
    pub operator: Option<FrontmatterOperator>,
}

pub async fn search_metadata(
    vault: &Vault,
    params: SearchMetadataParams,
) -> Result<CallToolResult, rmcp::ErrorData> {
    let search_type = params.search_type.as_str();

    if search_type.eq_ignore_ascii_case("tag") {
        search_metadata_tag(vault, &params)
    } else if search_type.eq_ignore_ascii_case("frontmatter") {
        search_metadata_frontmatter(vault, &params)
    } else {
        Err(rmcp::ErrorData::new(
            ErrorCode::INVALID_PARAMS,
            format!("Unknown type '{search_type}'. Valid values: \"tag\", \"frontmatter\""),
            None::<serde_json::Value>,
        ))
    }
}

fn search_metadata_tag(
    vault: &Vault,
    params: &SearchMetadataParams,
) -> Result<CallToolResult, rmcp::ErrorData> {
    let tag = params.tag.as_deref().ok_or_else(|| {
        rmcp::ErrorData::new(
            ErrorCode::INVALID_PARAMS,
            "'tag' is required when type is \"tag\"",
            None::<serde_json::Value>,
        )
    })?;
    let tag = tag.strip_prefix('#').unwrap_or(tag);
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

fn search_metadata_frontmatter(
    vault: &Vault,
    params: &SearchMetadataParams,
) -> Result<CallToolResult, rmcp::ErrorData> {
    let field = params.field.as_deref().ok_or_else(|| {
        rmcp::ErrorData::new(
            ErrorCode::INVALID_PARAMS,
            "'field' is required when type is \"frontmatter\"",
            None::<serde_json::Value>,
        )
    })?;
    let operator = params.operator.clone().unwrap_or_default();

    let results = match operator {
        FrontmatterOperator::Exists => vault.search_frontmatter_exists(field)?,
        FrontmatterOperator::Eq => {
            let value = params.value.as_ref().ok_or_else(|| {
                rmcp::ErrorData::new(
                    ErrorCode::INVALID_PARAMS,
                    "'value' is required for 'eq' operator",
                    None::<serde_json::Value>,
                )
            })?;
            vault.search_frontmatter(field, value)?
        }
        FrontmatterOperator::Contains => {
            let value = params.value.as_ref().ok_or_else(|| {
                rmcp::ErrorData::new(
                    ErrorCode::INVALID_PARAMS,
                    "'value' is required for 'contains' operator",
                    None::<serde_json::Value>,
                )
            })?;
            vault.search_frontmatter_contains(field, value)?
        }
    };

    let json = serde_json::to_string_pretty(&results)
        .map_err(|e| VaultError::Other(format!("JSON serialization failed: {e}")))?;
    Ok(CallToolResult::success(vec![Content::text(json)]))
}

// ── search_semantic ──────────────────────────────────────────────────

#[cfg(has_embeddings)]
const DEFAULT_PREFETCH_COUNT: usize = 50;
#[cfg(has_embeddings)]
const SNIPPET_CONTEXT_LEN: usize = 150;
#[cfg(has_embeddings)]
const SNIPPET_FALLBACK_CHARS: usize = 300;

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
    /// Blending weight for hybrid re-ranking: `alpha * BM25 + (1-alpha) * semantic`.
    /// Only used when `lexical_prefetch` is true. Lower values favor semantic similarity.
    /// Overrides the `OBSIDIAN_HYBRID_ALPHA` env var for this query. Range: 0.0–1.0, default: 0.25.
    #[serde(default)]
    pub alpha: Option<f32>,
}

#[derive(serde::Serialize, JsonSchema)]
struct SemanticSearchResult {
    path: std::path::PathBuf,
    title: String,
    score: f32,
    tags: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    snippet: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

pub async fn search_semantic(
    vault: &Vault,
    params: SearchSemanticParams,
    default_alpha: f32,
    runtime: &SemanticRuntime,
) -> Result<CallToolResult, rmcp::ErrorData> {
    let top_k = params.top_k.unwrap_or(10);
    let include_content = params.include_content.unwrap_or(false);
    let lexical_prefetch = params.lexical_prefetch.unwrap_or(false);
    let alpha = params.alpha.unwrap_or(default_alpha).clamp(0.0, 1.0);

    let results = match runtime.mode {
        SemanticMode::Daemon => {
            search_semantic_daemon(
                vault,
                &params,
                top_k,
                include_content,
                lexical_prefetch,
                alpha,
                runtime,
            )
            .await
        }
        SemanticMode::Local => search_semantic_local(
            vault,
            &params.query,
            top_k,
            include_content,
            lexical_prefetch,
            alpha,
        ),
        SemanticMode::Auto => match search_semantic_daemon(
            vault,
            &params,
            top_k,
            include_content,
            lexical_prefetch,
            alpha,
            runtime,
        )
        .await
        {
            Ok(results) => Ok(results),
            Err(err) if local_backend_available(vault) && should_fallback_to_local(&err) => {
                tracing::warn!(error = %err, "semantic daemon unavailable in auto mode; falling back to local embeddings backend");
                runtime
                    .vault_ensured
                    .store(false, std::sync::atomic::Ordering::Relaxed);
                search_semantic_local(
                    vault,
                    &params.query,
                    top_k,
                    include_content,
                    lexical_prefetch,
                    alpha,
                )
            }
            Err(err) => Err(err),
        },
    }
    .map_err(to_semantic_tool_error)?;

    let json = serde_json::to_string_pretty(&results)
        .map_err(|e| VaultError::Other(format!("JSON serialization failed: {e}")))?;
    Ok(CallToolResult::success(vec![Content::text(json)]))
}

async fn search_semantic_daemon(
    vault: &Vault,
    params: &SearchSemanticParams,
    top_k: usize,
    include_content: bool,
    lexical_prefetch: bool,
    alpha: f32,
    runtime: &SemanticRuntime,
) -> Result<Vec<SemanticSearchResult>, VaultError> {
    let Some(client) = runtime.daemon_client.as_ref() else {
        let reason = runtime
            .daemon_unavailable_reason
            .as_deref()
            .unwrap_or("semantic daemon client is not initialized");
        return Err(VaultError::DaemonUnavailable(reason.to_string()));
    };

    if !runtime
        .vault_ensured
        .load(std::sync::atomic::Ordering::Relaxed)
    {
        match client.ensure_vault(vault.root(), true, None).await {
            Ok(_) => {
                runtime
                    .vault_ensured
                    .store(true, std::sync::atomic::Ordering::Relaxed);
            }
            Err(err) => return Err(err),
        }
    }

    let daemon_result = if lexical_prefetch {
        let prefetch_count = runtime.prefetch_count;
        client
            .search_hybrid(
                vault.root(),
                &params.query,
                top_k,
                prefetch_count,
                alpha,
                include_content,
            )
            .await?
    } else {
        client
            .search_semantic(vault.root(), &params.query, top_k, include_content)
            .await?
    };

    Ok(daemon_result
        .results
        .into_iter()
        .map(|hit| SemanticSearchResult {
            path: std::path::PathBuf::from(hit.path),
            title: hit.title,
            score: hit.score,
            tags: hit.tags,
            snippet: hit.snippet,
            content: hit.content,
        })
        .collect())
}

#[cfg(has_embeddings)]
fn search_semantic_local(
    vault: &Vault,
    query: &str,
    top_k: usize,
    include_content: bool,
    lexical_prefetch: bool,
    alpha: f32,
) -> Result<Vec<SemanticSearchResult>, VaultError> {
    if !vault.has_embeddings() {
        let detail = vault
            .embedding_load_error()
            .map(|e| format!("Embedding model failed to load: {e}"))
            .unwrap_or_else(|| {
                "Embeddings not enabled (compile with --features embeddings or embeddings-api, and set OBSIDIAN_EMBEDDINGS=true)".to_string()
            });
        return Err(VaultError::Embedding(detail));
    }

    let hits = if lexical_prefetch {
        vault.search_hybrid(query, top_k, DEFAULT_PREFETCH_COUNT, alpha)?
    } else {
        vault.search_semantic(query, top_k)?
    };

    let word_re = if !include_content {
        compile_query_word_regex(query)
    } else {
        None
    };

    let mut results = Vec::with_capacity(hits.len());
    for (path, score) in hits {
        let meta = vault.get_note_metadata(&path).ok();
        let title = meta.as_ref().map(|m| m.title.clone()).unwrap_or_default();
        let tags = meta.as_ref().map(|m| m.tags.clone()).unwrap_or_default();

        let (content, snippet) = if include_content {
            (vault.read_note(&path).ok(), None)
        } else {
            let snip = vault.read_note(&path).ok().map(|text| {
                let body = crate::vault::frontmatter::get_body(&text);
                if let Some(ref re) = word_re
                    && let Some(found) = re.find(body)
                {
                    let (ctx, _, _, _) = crate::vault::index::extract_match_context(
                        body,
                        found.start(),
                        found.end(),
                        SNIPPET_CONTEXT_LEN,
                    );
                    return ctx;
                }
                body_preview(&text, SNIPPET_FALLBACK_CHARS)
            });
            (None, snip)
        };

        results.push(SemanticSearchResult {
            path,
            title,
            score,
            tags,
            snippet,
            content,
        });
    }

    Ok(results)
}

#[cfg(not(has_embeddings))]
fn search_semantic_local(
    _vault: &Vault,
    _query: &str,
    _top_k: usize,
    _include_content: bool,
    _lexical_prefetch: bool,
    _alpha: f32,
) -> Result<Vec<SemanticSearchResult>, VaultError> {
    Err(VaultError::Embedding(
        "Semantic search is not available. Rebuild with --features embeddings or --features embeddings-api".to_string(),
    ))
}

#[cfg(has_embeddings)]
use crate::vault::search_utils::{body_preview, compile_query_word_regex};

fn local_backend_available(vault: &Vault) -> bool {
    #[cfg(has_embeddings)]
    {
        vault.has_embeddings()
    }
    #[cfg(not(has_embeddings))]
    {
        let _ = vault;
        false
    }
}

fn should_fallback_to_local(err: &VaultError) -> bool {
    match err {
        VaultError::DaemonUnavailable(_)
        | VaultError::DaemonIpc(_)
        | VaultError::DaemonTimeout { .. }
        | VaultError::DaemonBootstrap(_) => true,
        VaultError::DaemonRpc { code, .. } => matches!(
            *code,
            protocol::ERR_DAEMON_UNAVAILABLE
                | protocol::ERR_BOOTSTRAP_REQUIRED
                | protocol::ERR_VAULT_NOT_READY
                | protocol::ERR_INCOMPATIBLE_API_VERSION
        ),
        _ => false,
    }
}

fn to_semantic_tool_error(err: VaultError) -> rmcp::ErrorData {
    if matches!(err, VaultError::Embedding(_)) {
        rmcp::ErrorData::new(
            ErrorCode::INVALID_REQUEST,
            err.to_string(),
            None::<serde_json::Value>,
        )
    } else {
        err.into()
    }
}

// ── tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::path::Path;
    #[cfg(unix)]
    use std::path::PathBuf;

    use super::*;
    use crate::test_helpers::{create_test_vault, extract_text, tantivy_config, test_config};
    #[cfg(unix)]
    use crate::{
        client::semantic_daemon::{DaemonConnectPolicy, SemanticDaemonClient},
        daemon::server::IpcEndpoint,
    };
    #[cfg(unix)]
    use serde_json::json;
    #[cfg(unix)]
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

    #[cfg(unix)]
    fn start_prefetch_capture_server(socket_path: PathBuf) -> tokio::task::JoinHandle<usize> {
        tokio::spawn(async move {
            if socket_path.exists() {
                let _ = std::fs::remove_file(&socket_path);
            }
            let listener = tokio::net::UnixListener::bind(&socket_path).expect("bind unix socket");
            let mut captured_prefetch = 0usize;

            for _ in 0..2 {
                let (stream, _) = listener.accept().await.expect("accept client");
                let (reader, mut writer) = tokio::io::split(stream);
                let mut reader = BufReader::new(reader);
                let mut line = String::new();
                reader.read_line(&mut line).await.expect("read request");
                let request: serde_json::Value =
                    serde_json::from_str(&line).expect("request should be valid JSON");
                let id = request
                    .get("id")
                    .cloned()
                    .expect("request should include id");
                let method = request
                    .get("method")
                    .and_then(serde_json::Value::as_str)
                    .expect("request should include method");

                let response = match method {
                    "ensure_vault" => json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": {
                            "vault_id": "test-vault",
                            "ready": true,
                            "watch_enabled": true,
                            "model_name": "BAAI/bge-small-en-v1.5"
                        }
                    }),
                    "search_hybrid" => {
                        captured_prefetch = request
                            .get("params")
                            .and_then(|params| params.get("prefetch"))
                            .and_then(serde_json::Value::as_u64)
                            .expect("search_hybrid should include prefetch")
                            as usize;
                        json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "result": {
                                "results": []
                            }
                        })
                    }
                    other => panic!("unexpected method in daemon test server: {other}"),
                };

                writer
                    .write_all(
                        format!(
                            "{}\n",
                            serde_json::to_string(&response).expect("serialize response")
                        )
                        .as_bytes(),
                    )
                    .await
                    .expect("write response");
                writer.flush().await.expect("flush response");
            }

            captured_prefetch
        })
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

    // ── search_metadata (tag) ──────────────────────────────────────

    #[tokio::test]
    async fn search_metadata_tag_exact() {
        let (_dir, vault) = setup_search_vault().await;
        let result = search_metadata(
            &vault,
            SearchMetadataParams {
                search_type: "tag".into(),
                tag: Some("inbox".into()),
                include_nested: Some(false),
                ..Default::default()
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
    async fn search_metadata_tag_include_nested() {
        let (_dir, vault) = setup_search_vault().await;
        let result = search_metadata(
            &vault,
            SearchMetadataParams {
                search_type: "tag".into(),
                tag: Some("inbox".into()),
                include_nested: Some(true),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let text = extract_text(&result);

        assert!(text.contains("notes.md"));
    }

    #[tokio::test]
    async fn search_metadata_tag_strips_hash_prefix() {
        let (_dir, vault) = setup_search_vault().await;
        let result = search_metadata(
            &vault,
            SearchMetadataParams {
                search_type: "tag".into(),
                tag: Some("#lang".into()),
                include_nested: Some(false),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let text = extract_text(&result);

        let parsed: Vec<serde_json::Value> = serde_json::from_str(text).unwrap();
        assert_eq!(parsed.len(), 2);
    }

    #[tokio::test]
    async fn search_metadata_tag_missing_tag_errors() {
        let (_dir, vault) = setup_search_vault().await;
        let result = search_metadata(
            &vault,
            SearchMetadataParams {
                search_type: "tag".into(),
                ..Default::default()
            },
        )
        .await;

        assert!(result.is_err());
    }

    // ── search_metadata (frontmatter) ───────────────────────────────

    #[tokio::test]
    async fn search_metadata_frontmatter_eq() {
        let (_dir, vault) = setup_search_vault().await;
        let result = search_metadata(
            &vault,
            SearchMetadataParams {
                search_type: "frontmatter".into(),
                field: Some("status".into()),
                value: Some(serde_json::json!("stable")),
                operator: Some(FrontmatterOperator::Eq),
                ..Default::default()
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
    async fn search_metadata_frontmatter_eq_array_contains() {
        let (_dir, vault) = setup_search_vault().await;
        let result = search_metadata(
            &vault,
            SearchMetadataParams {
                search_type: "frontmatter".into(),
                field: Some("tags".into()),
                value: Some(serde_json::json!("systems")),
                operator: Some(FrontmatterOperator::Eq),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let text = extract_text(&result);

        assert!(text.contains("rust.md"));
    }

    #[tokio::test]
    async fn search_metadata_frontmatter_contains_substring() {
        let (_dir, vault) = setup_search_vault().await;
        let result = search_metadata(
            &vault,
            SearchMetadataParams {
                search_type: "frontmatter".into(),
                field: Some("status".into()),
                value: Some(serde_json::json!("progress")),
                operator: Some(FrontmatterOperator::Contains),
                ..Default::default()
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
    async fn search_metadata_frontmatter_exists() {
        let (_dir, vault) = setup_search_vault().await;
        let result = search_metadata(
            &vault,
            SearchMetadataParams {
                search_type: "frontmatter".into(),
                field: Some("status".into()),
                operator: Some(FrontmatterOperator::Exists),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let text = extract_text(&result);

        let parsed: Vec<serde_json::Value> = serde_json::from_str(text).unwrap();
        assert_eq!(parsed.len(), 2); // rust.md + python.md
    }

    #[tokio::test]
    async fn search_metadata_frontmatter_exists_missing_field() {
        let (_dir, vault) = setup_search_vault().await;
        let result = search_metadata(
            &vault,
            SearchMetadataParams {
                search_type: "frontmatter".into(),
                field: Some("nonexistent".into()),
                operator: Some(FrontmatterOperator::Exists),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let text = extract_text(&result);

        let parsed: Vec<serde_json::Value> = serde_json::from_str(text).unwrap();
        assert!(parsed.is_empty());
    }

    #[tokio::test]
    async fn search_metadata_frontmatter_eq_without_value_errors() {
        let (_dir, vault) = setup_search_vault().await;
        let result = search_metadata(
            &vault,
            SearchMetadataParams {
                search_type: "frontmatter".into(),
                field: Some("status".into()),
                operator: Some(FrontmatterOperator::Eq),
                ..Default::default()
            },
        )
        .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn search_metadata_frontmatter_missing_field_errors() {
        let (_dir, vault) = setup_search_vault().await;
        let result = search_metadata(
            &vault,
            SearchMetadataParams {
                search_type: "frontmatter".into(),
                value: Some(serde_json::json!("test")),
                ..Default::default()
            },
        )
        .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn search_metadata_invalid_type_errors() {
        let (_dir, vault) = setup_search_vault().await;
        let result = search_metadata(
            &vault,
            SearchMetadataParams {
                search_type: "invalid".into(),
                ..Default::default()
            },
        )
        .await;

        assert!(result.is_err());
    }

    // ── search_text with Tantivy BM25 ──────────────────────────────

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

    // ── SearchSemanticParams ────────────────────────────────────────

    #[test]
    fn semantic_params_defaults() {
        let params: SearchSemanticParams = serde_json::from_str(r#"{"query": "test"}"#).unwrap();
        assert_eq!(params.query, "test");
        assert!(params.alpha.is_none());
        assert!(params.lexical_prefetch.is_none());
        assert!(params.top_k.is_none());
    }

    #[test]
    fn semantic_params_with_alpha() {
        let params: SearchSemanticParams =
            serde_json::from_str(r#"{"query": "q", "alpha": 0.7, "lexical_prefetch": true}"#)
                .unwrap();
        assert!((params.alpha.unwrap() - 0.7).abs() < f32::EPSILON);
        assert_eq!(params.lexical_prefetch, Some(true));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn daemon_prefetch_uses_runtime_value_without_forcing_min_50() {
        let (_dir, vault) = setup_search_vault().await;
        let socket_dir = tempfile::tempdir().expect("tempdir");
        let socket_path = socket_dir.path().join("semanticd.sock");
        let server = start_prefetch_capture_server(socket_path.clone());

        let runtime = SemanticRuntime {
            mode: SemanticMode::Daemon,
            daemon_client: Some(SemanticDaemonClient::new(
                IpcEndpoint::UnixSocket(socket_path),
                DaemonConnectPolicy::default(),
            )),
            daemon_unavailable_reason: None,
            prefetch_count: 7,
            vault_ensured: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        };

        let result = search_semantic(
            &vault,
            SearchSemanticParams {
                query: "systems language".to_string(),
                top_k: Some(5),
                include_content: Some(false),
                lexical_prefetch: Some(true),
                alpha: Some(0.25),
            },
            0.25,
            &runtime,
        )
        .await
        .expect("daemon search should succeed");
        let parsed: Vec<serde_json::Value> =
            serde_json::from_str(extract_text(&result)).expect("parse result");
        assert!(parsed.is_empty(), "mock daemon returns empty result set");

        let captured_prefetch = server.await.expect("server join");
        assert_eq!(
            captured_prefetch, 7,
            "runtime prefetch should be used as-is"
        );
    }
}
