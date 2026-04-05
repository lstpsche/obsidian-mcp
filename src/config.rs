//! Configuration: env/CLI config for vault path, watch toggle, log level,
//! and optional search features (Tantivy BM25, embeddings).

use std::path::PathBuf;

const DEFAULT_EMBEDDINGS_MODEL: &str = "BAAI/bge-small-en-v1.5";
const DEFAULT_HYBRID_ALPHA: f32 = 0.25;

#[derive(Debug, Clone)]
pub struct Config {
    pub vault_path: PathBuf,
    pub watch: bool,
    pub log_level: String,
    /// Enable Tantivy BM25 full-text index (`OBSIDIAN_TANTIVY`, default `true`).
    pub tantivy: bool,
    /// Enable semantic embedding search (`OBSIDIAN_EMBEDDINGS`, default `false`).
    pub embeddings: bool,
    /// HuggingFace model name for embeddings (`OBSIDIAN_EMBEDDINGS_MODEL`).
    pub embeddings_model: String,
    /// Hybrid search alpha: `alpha * BM25 + (1-alpha) * semantic` (`OBSIDIAN_HYBRID_ALPHA`, default `0.25`).
    /// Clamped to `[0.0, 1.0]`. Lower values give more weight to semantic similarity.
    pub hybrid_alpha: f32,
}

impl Config {
    /// Load configuration from CLI args and environment variables.
    ///
    /// Priority for vault path: CLI arg (first positional) > `OBSIDIAN_VAULT_PATH` env var.
    pub fn load() -> Result<Self, String> {
        let vault_path = std::env::args()
            .nth(1)
            .or_else(|| std::env::var("OBSIDIAN_VAULT_PATH").ok())
            .map(|raw| normalize_vault_path(&raw))
            .ok_or_else(|| {
                "Vault path required: pass as first argument or set OBSIDIAN_VAULT_PATH".to_string()
            })?;

        let watch = std::env::var("OBSIDIAN_WATCH")
            .unwrap_or_else(|_| "true".into())
            .eq_ignore_ascii_case("true");

        let log_level = std::env::var("OBSIDIAN_LOG_LEVEL").unwrap_or_else(|_| "info".into());

        let tantivy = std::env::var("OBSIDIAN_TANTIVY")
            .unwrap_or_else(|_| "true".into())
            .eq_ignore_ascii_case("true");

        let embeddings = std::env::var("OBSIDIAN_EMBEDDINGS")
            .unwrap_or_else(|_| "false".into())
            .eq_ignore_ascii_case("true");

        let embeddings_model = std::env::var("OBSIDIAN_EMBEDDINGS_MODEL")
            .unwrap_or_else(|_| DEFAULT_EMBEDDINGS_MODEL.into());

        let hybrid_alpha = std::env::var("OBSIDIAN_HYBRID_ALPHA")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(DEFAULT_HYBRID_ALPHA)
            .clamp(0.0, 1.0);

        Ok(Self {
            vault_path,
            watch,
            log_level,
            tantivy,
            embeddings,
            embeddings_model,
            hybrid_alpha,
        })
    }
}

fn normalize_vault_path(raw: &str) -> PathBuf {
    let trimmed = raw.trim();
    let normalized = strip_matching_outer_quotes(trimmed).trim();
    let final_value = if normalized.is_empty() {
        trimmed
    } else {
        normalized
    };
    PathBuf::from(final_value)
}

fn strip_matching_outer_quotes(mut value: &str) -> &str {
    loop {
        let is_double_quoted = value.starts_with('"') && value.ends_with('"');
        let is_single_quoted = value.starts_with('\'') && value.ends_with('\'');
        if (is_double_quoted || is_single_quoted) && value.len() >= 2 {
            value = &value[1..value.len() - 1];
            continue;
        }
        return value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_vault_path_keeps_plain_value() {
        assert_eq!(
            normalize_vault_path("/tmp/my-vault"),
            PathBuf::from("/tmp/my-vault")
        );
    }

    #[test]
    fn normalize_vault_path_strips_double_quotes() {
        assert_eq!(
            normalize_vault_path("\"/tmp/my-vault\""),
            PathBuf::from("/tmp/my-vault")
        );
    }

    #[test]
    fn normalize_vault_path_strips_single_quotes_and_spaces() {
        assert_eq!(
            normalize_vault_path("  '/tmp/my-vault'  "),
            PathBuf::from("/tmp/my-vault")
        );
    }

    #[test]
    fn normalize_vault_path_handles_multiple_quote_layers() {
        assert_eq!(
            normalize_vault_path(" \"'/tmp/my-vault'\" "),
            PathBuf::from("/tmp/my-vault")
        );
    }
}
