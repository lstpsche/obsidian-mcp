//! Configuration: env/CLI config for vault path, watch toggle, log level,
//! and optional search features (Tantivy BM25, embeddings).

use std::path::PathBuf;

pub const DEFAULT_MODEL_NAME: &str = "BAAI/bge-small-en-v1.5";
const DEFAULT_HYBRID_ALPHA: f32 = 0.25;
const DEFAULT_SEMANTIC_CONNECT_TIMEOUT_MS: u64 = 2_000;
const DEFAULT_SEMANTIC_CONNECT_RETRIES: u32 = 2;
const DEFAULT_SEMANTIC_RETRY_BACKOFF_MS: u64 = 250;
const DEFAULT_SEMANTIC_PREFETCH_COUNT: usize = 50;

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
            .unwrap_or_else(|_| DEFAULT_MODEL_NAME.into());

        let hybrid_alpha = std::env::var("OBSIDIAN_SEMANTIC_ALPHA")
            .ok()
            .or_else(|| std::env::var("OBSIDIAN_HYBRID_ALPHA").ok())
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SemanticMode {
    #[default]
    Auto,
    Daemon,
    Local,
}

impl SemanticMode {
    fn parse(raw: &str) -> Option<Self> {
        let normalized = raw.trim();
        if normalized.eq_ignore_ascii_case("auto") {
            Some(Self::Auto)
        } else if normalized.eq_ignore_ascii_case("daemon") {
            Some(Self::Daemon)
        } else if normalized.eq_ignore_ascii_case("local") {
            Some(Self::Local)
        } else {
            None
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Daemon => "daemon",
            Self::Local => "local",
        }
    }
}

#[derive(Debug, Clone)]
pub struct SemanticRuntimeConfig {
    /// Semantic runtime mode (`OBSIDIAN_SEMANTIC_MODE`): `auto`, `daemon`, or `local`.
    pub mode: SemanticMode,
    /// Override shared semantic home path (`OBSIDIAN_SEMANTIC_HOME`).
    pub semantic_home_override: Option<PathBuf>,
    /// Override daemon binary path (`OBSIDIAN_SEMANTIC_DAEMON_PATH`).
    pub daemon_path_override: Option<PathBuf>,
    /// Override daemon endpoint (`OBSIDIAN_SEMANTIC_ENDPOINT`).
    pub daemon_endpoint_override: Option<String>,
    /// Override daemon binary download URL (`OBSIDIAN_SEMANTIC_DAEMON_DOWNLOAD_URL`).
    pub daemon_download_url: Option<String>,
    /// Shared semantic model name (`OBSIDIAN_SEMANTIC_MODEL`).
    pub model_name: String,
    /// Daemon request timeout in milliseconds (`OBSIDIAN_SEMANTIC_CONNECT_TIMEOUT_MS`).
    pub connect_timeout_ms: u64,
    /// Daemon retry count (`OBSIDIAN_SEMANTIC_CONNECT_RETRIES`).
    pub connect_retries: u32,
    /// Daemon retry backoff in milliseconds (`OBSIDIAN_SEMANTIC_RETRY_BACKOFF_MS`).
    pub retry_backoff_ms: u64,
    /// Default lexical prefetch count for hybrid daemon queries (`OBSIDIAN_SEMANTIC_PREFETCH`).
    pub prefetch_count: usize,
}

impl SemanticRuntimeConfig {
    pub fn load_from_env() -> Self {
        Self {
            mode: std::env::var("OBSIDIAN_SEMANTIC_MODE")
                .ok()
                .as_deref()
                .and_then(SemanticMode::parse)
                .unwrap_or_default(),
            semantic_home_override: normalize_optional_path_env("OBSIDIAN_SEMANTIC_HOME"),
            daemon_path_override: normalize_optional_path_env("OBSIDIAN_SEMANTIC_DAEMON_PATH"),
            daemon_endpoint_override: normalize_optional_string_env("OBSIDIAN_SEMANTIC_ENDPOINT"),
            daemon_download_url: std::env::var("OBSIDIAN_SEMANTIC_DAEMON_DOWNLOAD_URL")
                .ok()
                .map(|raw| raw.trim().to_string())
                .filter(|value| !value.is_empty()),
            model_name: std::env::var("OBSIDIAN_SEMANTIC_MODEL")
                .ok()
                .map(|raw| raw.trim().to_string())
                .filter(|value| !value.is_empty())
                .unwrap_or_else(|| DEFAULT_MODEL_NAME.to_string()),
            connect_timeout_ms: parse_u64_env("OBSIDIAN_SEMANTIC_CONNECT_TIMEOUT_MS")
                .unwrap_or(DEFAULT_SEMANTIC_CONNECT_TIMEOUT_MS)
                .clamp(100, 60_000),
            connect_retries: parse_u32_env("OBSIDIAN_SEMANTIC_CONNECT_RETRIES")
                .unwrap_or(DEFAULT_SEMANTIC_CONNECT_RETRIES)
                .clamp(0, 10),
            retry_backoff_ms: parse_u64_env("OBSIDIAN_SEMANTIC_RETRY_BACKOFF_MS")
                .unwrap_or(DEFAULT_SEMANTIC_RETRY_BACKOFF_MS)
                .clamp(50, 60_000),
            prefetch_count: parse_usize_env("OBSIDIAN_SEMANTIC_PREFETCH")
                .unwrap_or(DEFAULT_SEMANTIC_PREFETCH_COUNT)
                .clamp(1, 1_000),
        }
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

fn normalize_optional_path_env(var_name: &str) -> Option<PathBuf> {
    std::env::var(var_name)
        .ok()
        .map(|raw| normalize_vault_path(&raw))
        .filter(|path| !path.as_os_str().is_empty())
}

fn normalize_optional_string_env(var_name: &str) -> Option<String> {
    std::env::var(var_name)
        .ok()
        .map(|raw| raw.trim().to_string())
        .map(|raw| strip_matching_outer_quotes(&raw).trim().to_string())
        .filter(|value| !value.is_empty())
}

fn parse_u64_env(var_name: &str) -> Option<u64> {
    std::env::var(var_name).ok()?.trim().parse::<u64>().ok()
}

fn parse_u32_env(var_name: &str) -> Option<u32> {
    std::env::var(var_name).ok()?.trim().parse::<u32>().ok()
}

fn parse_usize_env(var_name: &str) -> Option<usize> {
    std::env::var(var_name).ok()?.trim().parse::<usize>().ok()
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

    #[test]
    fn semantic_runtime_config_defaults_model() {
        let cfg = SemanticRuntimeConfig::load_from_env();
        assert!(!cfg.model_name.is_empty());
    }

    #[test]
    fn semantic_mode_parse_known_values() {
        assert_eq!(SemanticMode::parse("auto"), Some(SemanticMode::Auto));
        assert_eq!(SemanticMode::parse("daemon"), Some(SemanticMode::Daemon));
        assert_eq!(SemanticMode::parse("local"), Some(SemanticMode::Local));
        assert_eq!(SemanticMode::parse("DAEMON"), Some(SemanticMode::Daemon));
    }

    #[test]
    fn semantic_mode_parse_unknown_value() {
        assert_eq!(SemanticMode::parse("unexpected"), None);
    }

    #[test]
    fn semantic_mode_as_str_roundtrip() {
        assert_eq!(SemanticMode::Auto.as_str(), "auto");
        assert_eq!(SemanticMode::Daemon.as_str(), "daemon");
        assert_eq!(SemanticMode::Local.as_str(), "local");
    }
}
