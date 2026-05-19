//! Configuration: env/CLI config for vault path, watch toggle, log level,
//! transport selection, and optional search features (Tantivy BM25, embeddings).

use std::collections::HashSet;
use std::net::{IpAddr, Ipv4Addr};
use std::path::PathBuf;

pub const DEFAULT_MODEL_NAME: &str = "BAAI/bge-small-en-v1.5";
pub const DEFAULT_HTTP_PORT: u16 = 37842;
pub const DEFAULT_HTTP_HOST: IpAddr = IpAddr::V4(Ipv4Addr::LOCALHOST);

const DEFAULT_HYBRID_ALPHA: f32 = 0.25;
const DEFAULT_SEMANTIC_CONNECT_TIMEOUT_MS: u64 = 2_000;
const DEFAULT_SEMANTIC_CONNECT_RETRIES: u32 = 2;
const DEFAULT_SEMANTIC_RETRY_BACKOFF_MS: u64 = 250;
const DEFAULT_SEMANTIC_PREFETCH_COUNT: usize = 50;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Transport {
    #[default]
    Stdio,
    Http,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingProvider {
    Local,
    Api,
}

impl EmbeddingProvider {
    pub fn parse(raw: &str) -> Option<Self> {
        let normalized = raw.trim();
        if normalized.eq_ignore_ascii_case("local") {
            Some(Self::Local)
        } else if normalized.eq_ignore_ascii_case("api") {
            Some(Self::Api)
        } else {
            None
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::Api => "api",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub vault_path: PathBuf,
    pub watch: bool,
    pub log_level: String,
    pub transport: Transport,
    pub http_host: IpAddr,
    pub http_port: u16,
    /// Enable Tantivy BM25 full-text index (`OBSIDIAN_TANTIVY`, default `true`).
    pub tantivy: bool,
    /// Enable semantic embedding search (`OBSIDIAN_EMBEDDINGS`, default `false`).
    pub embeddings: bool,
    /// HuggingFace model name for embeddings (`OBSIDIAN_EMBEDDINGS_MODEL`).
    pub embeddings_model: String,
    /// Hybrid search alpha: `alpha * BM25 + (1-alpha) * semantic` (`OBSIDIAN_HYBRID_ALPHA`, default `0.25`).
    /// Clamped to `[0.0, 1.0]`. Lower values give more weight to semantic similarity.
    pub hybrid_alpha: f32,
    /// Embedding backend selection (`OBSIDIAN_EMBEDDING_PROVIDER`).
    /// `None` = infer from compiled features (local if `embeddings`, api if `embeddings-api`).
    pub embedding_provider: Option<EmbeddingProvider>,
    /// Tool filter configuration (`OBSIDIAN_TOOLS`): profile name, allow-list, or deny-list.
    pub tool_filter: ToolFilter,
}

// ── Tool Filtering ─────────────────────────────────────────────────

pub const ALL_TOOL_NAMES: &[&str] = &[
    "vault_list",
    "note_read",
    "note_create",
    "note_write",
    "note_insert",
    "note_patch",
    "note_delete",
    "note_move",
    "search_text",
    "search_regex",
    "search_metadata",
    "search_semantic",
    "note_inspect",
    "frontmatter",
    "wikilinks",
    "periodic",
    "vault_info",
    "open_in_obsidian",
];

const PROFILE_CORE: &[&str] = &[
    "vault_list",
    "note_read",
    "note_create",
    "note_write",
    "note_insert",
    "note_patch",
    "note_delete",
    "note_move",
    "search_text",
    "search_regex",
    "search_metadata",
    "note_inspect",
    "frontmatter",
    "vault_info",
];

const PROFILE_READ: &[&str] = &[
    "note_read",
    "vault_list",
    "search_text",
    "search_regex",
    "search_metadata",
    "search_semantic",
    "note_inspect",
    "frontmatter",
    "wikilinks",
    "vault_info",
];

const PROFILE_MINIMAL: &[&str] = &[
    "note_read",
    "note_create",
    "note_write",
    "vault_list",
    "search_text",
    "vault_info",
];

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ToolFilter {
    #[default]
    Full,
    Profile(String),
    AllowList(HashSet<String>),
    DenyList(HashSet<String>),
}

impl ToolFilter {
    pub fn parse(raw: &str) -> Result<Self, String> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Ok(Self::Full);
        }

        if trimmed.contains('!') {
            let tools: HashSet<String> = trimmed
                .split(',')
                .map(|s| s.trim().trim_start_matches('!').to_string())
                .filter(|s| !s.is_empty())
                .collect();
            for name in &tools {
                if !ALL_TOOL_NAMES.contains(&name.as_str()) {
                    tracing::warn!(tool = %name, "unknown tool name in deny-list, ignoring");
                }
            }
            return Ok(Self::DenyList(tools));
        }

        if !trimmed.contains(',') {
            if let Some(profile) = resolve_profile(trimmed) {
                return Ok(Self::Profile(profile));
            }
            if ALL_TOOL_NAMES.contains(&trimmed) {
                return Ok(Self::AllowList(
                    std::iter::once(trimmed.to_string()).collect(),
                ));
            }
            return Err(format!(
                "Unknown profile or tool name '{trimmed}'. Valid profiles: full, core, read, minimal. \
                 For an allow-list of multiple tools, use comma-separated names."
            ));
        }

        let tools: HashSet<String> = trimmed
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        for name in &tools {
            if !ALL_TOOL_NAMES.contains(&name.as_str()) {
                tracing::warn!(tool = %name, "unknown tool name in allow-list, ignoring");
            }
        }
        Ok(Self::AllowList(tools))
    }

    pub fn disabled_tools(&self) -> HashSet<String> {
        let all: HashSet<&str> = ALL_TOOL_NAMES.iter().copied().collect();

        match self {
            Self::Full => HashSet::new(),
            Self::Profile(name) => {
                let enabled = profile_tools(name);
                all.iter()
                    .filter(|t| !enabled.contains(*t))
                    .map(|t| (*t).to_string())
                    .collect()
            }
            Self::AllowList(allowed) => all
                .iter()
                .filter(|t| !allowed.contains(**t))
                .map(|t| (*t).to_string())
                .collect(),
            Self::DenyList(denied) => denied.clone(),
        }
    }
}

fn resolve_profile(name: &str) -> Option<String> {
    let lower = name.to_ascii_lowercase();
    match lower.as_str() {
        "full" | "core" | "read" | "minimal" => Some(lower),
        _ => None,
    }
}

fn profile_tools(name: &str) -> HashSet<&'static str> {
    match name {
        "full" => ALL_TOOL_NAMES.iter().copied().collect(),
        "core" => PROFILE_CORE.iter().copied().collect(),
        "read" => PROFILE_READ.iter().copied().collect(),
        "minimal" => PROFILE_MINIMAL.iter().copied().collect(),
        _ => ALL_TOOL_NAMES.iter().copied().collect(),
    }
}

/// Parsed CLI arguments split into flags/values and positional args.
#[derive(Debug, Default)]
pub struct CliArgs {
    pub http: bool,
    pub port: Option<u16>,
    pub host: Option<IpAddr>,
    pub vault_path: Option<PathBuf>,
}

/// Parse CLI args after the binary name, ignoring --help/--version (handled earlier).
pub fn parse_cli_args() -> CliArgs {
    let mut result = CliArgs::default();
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--http" => result.http = true,
            "--port" => {
                if let Some(val) = args.next() {
                    result.port = val.parse().ok();
                }
            }
            "--host" => {
                if let Some(val) = args.next() {
                    result.host = val.parse().ok();
                }
            }
            s if s.starts_with('-') => {}
            _ => {
                if result.vault_path.is_none() {
                    result.vault_path = Some(normalize_vault_path(&arg));
                }
            }
        }
    }
    result
}

impl Config {
    /// Load configuration from CLI args and environment variables.
    ///
    /// Priority for vault path: CLI positional arg > `OBSIDIAN_VAULT_PATH` env var.
    /// Priority for transport: `--http` flag > `OBSIDIAN_TRANSPORT` env var > default (stdio).
    pub fn load(cli: &CliArgs) -> Result<Self, String> {
        let vault_path = cli
            .vault_path
            .clone()
            .or_else(|| {
                std::env::var("OBSIDIAN_VAULT_PATH")
                    .ok()
                    .map(|raw| normalize_vault_path(&raw))
            })
            .ok_or_else(|| {
                "Vault path required: pass as first argument or set OBSIDIAN_VAULT_PATH".to_string()
            })?;

        let transport = if cli.http {
            Transport::Http
        } else {
            match std::env::var("OBSIDIAN_TRANSPORT").ok().as_deref() {
                Some(v) if v.eq_ignore_ascii_case("http") => Transport::Http,
                _ => Transport::Stdio,
            }
        };

        let http_port = cli
            .port
            .or_else(|| parse_u16_env("OBSIDIAN_HTTP_PORT"))
            .unwrap_or(DEFAULT_HTTP_PORT);

        let http_host = cli
            .host
            .or_else(|| {
                std::env::var("OBSIDIAN_HTTP_HOST")
                    .ok()
                    .and_then(|v| v.trim().parse().ok())
            })
            .unwrap_or(DEFAULT_HTTP_HOST);

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

        let embedding_provider = std::env::var("OBSIDIAN_EMBEDDING_PROVIDER")
            .ok()
            .and_then(|v| {
                let trimmed = v.trim();
                if trimmed.is_empty() {
                    return None;
                }
                match EmbeddingProvider::parse(trimmed) {
                    Some(p) => Some(p),
                    None => {
                        tracing::warn!(
                            value = trimmed,
                            "unknown OBSIDIAN_EMBEDDING_PROVIDER, ignoring"
                        );
                        None
                    }
                }
            });

        let tool_filter = match std::env::var("OBSIDIAN_TOOLS").ok() {
            Some(raw) if !raw.trim().is_empty() => ToolFilter::parse(raw.trim())?,
            _ => ToolFilter::Full,
        };

        Ok(Self {
            vault_path,
            watch,
            log_level,
            transport,
            http_host,
            http_port,
            tantivy,
            embeddings,
            embeddings_model,
            hybrid_alpha,
            embedding_provider,
            tool_filter,
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

fn parse_u16_env(var_name: &str) -> Option<u16> {
    std::env::var(var_name).ok()?.trim().parse::<u16>().ok()
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

    // ── ToolFilter ────────────────────────────────────────────────

    #[test]
    fn tool_filter_empty_is_full() {
        assert_eq!(ToolFilter::parse(""), Ok(ToolFilter::Full));
        assert_eq!(ToolFilter::parse("  "), Ok(ToolFilter::Full));
    }

    #[test]
    fn tool_filter_profile_full() {
        let filter = ToolFilter::parse("full").unwrap();
        assert_eq!(filter, ToolFilter::Profile("full".into()));
        assert!(filter.disabled_tools().is_empty());
    }

    #[test]
    fn tool_filter_profile_core() {
        let filter = ToolFilter::parse("core").unwrap();
        assert_eq!(filter, ToolFilter::Profile("core".into()));
        let disabled = filter.disabled_tools();
        assert_eq!(disabled.len(), ALL_TOOL_NAMES.len() - PROFILE_CORE.len());
        assert!(disabled.contains("search_semantic"));
        assert!(disabled.contains("wikilinks"));
        assert!(disabled.contains("periodic"));
        assert!(disabled.contains("open_in_obsidian"));
        assert!(!disabled.contains("note_read"));
    }

    #[test]
    fn tool_filter_profile_read() {
        let filter = ToolFilter::parse("read").unwrap();
        assert_eq!(filter, ToolFilter::Profile("read".into()));
        let disabled = filter.disabled_tools();
        assert_eq!(disabled.len(), ALL_TOOL_NAMES.len() - PROFILE_READ.len());
        assert!(disabled.contains("note_create"));
        assert!(disabled.contains("note_write"));
        assert!(!disabled.contains("note_read"));
        assert!(!disabled.contains("search_semantic"));
    }

    #[test]
    fn tool_filter_profile_minimal() {
        let filter = ToolFilter::parse("minimal").unwrap();
        assert_eq!(filter, ToolFilter::Profile("minimal".into()));
        let disabled = filter.disabled_tools();
        assert_eq!(disabled.len(), ALL_TOOL_NAMES.len() - PROFILE_MINIMAL.len());
        assert!(disabled.contains("search_regex"));
        assert!(disabled.contains("wikilinks"));
        assert!(!disabled.contains("note_read"));
        assert!(!disabled.contains("vault_list"));
    }

    #[test]
    fn tool_filter_case_insensitive_profile() {
        assert_eq!(
            ToolFilter::parse("CORE").unwrap(),
            ToolFilter::Profile("core".into())
        );
        assert_eq!(
            ToolFilter::parse("Read").unwrap(),
            ToolFilter::Profile("read".into())
        );
    }

    #[test]
    fn tool_filter_allow_list() {
        let filter = ToolFilter::parse("note_read,vault_list").unwrap();
        let expected: HashSet<String> = ["note_read", "vault_list"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(filter, ToolFilter::AllowList(expected));
        let disabled = filter.disabled_tools();
        assert_eq!(disabled.len(), ALL_TOOL_NAMES.len() - 2);
        assert!(!disabled.contains("note_read"));
        assert!(!disabled.contains("vault_list"));
        assert!(disabled.contains("note_create"));
    }

    #[test]
    fn tool_filter_single_valid_tool_is_allow_list() {
        let filter = ToolFilter::parse("note_read").unwrap();
        let expected: HashSet<String> = std::iter::once("note_read".to_string()).collect();
        assert_eq!(filter, ToolFilter::AllowList(expected));
    }

    #[test]
    fn tool_filter_deny_list() {
        let filter = ToolFilter::parse("!open_in_obsidian,!wikilinks").unwrap();
        let expected: HashSet<String> = ["open_in_obsidian", "wikilinks"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(filter, ToolFilter::DenyList(expected));
        let disabled = filter.disabled_tools();
        assert_eq!(disabled.len(), 2);
        assert!(disabled.contains("open_in_obsidian"));
        assert!(disabled.contains("wikilinks"));
    }

    #[test]
    fn tool_filter_unknown_single_word_errors() {
        let result = ToolFilter::parse("nonexistent");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("Valid profiles"));
    }

    #[test]
    fn tool_filter_deny_list_with_unknown_tool() {
        let filter = ToolFilter::parse("!fake_tool,!note_read").unwrap();
        let disabled = filter.disabled_tools();
        assert!(disabled.contains("fake_tool"));
        assert!(disabled.contains("note_read"));
    }

    // ── EmbeddingProvider ─────────────────────────────────────────

    #[test]
    fn embedding_provider_parse_known_values() {
        assert_eq!(
            EmbeddingProvider::parse("local"),
            Some(EmbeddingProvider::Local)
        );
        assert_eq!(
            EmbeddingProvider::parse("api"),
            Some(EmbeddingProvider::Api)
        );
    }

    #[test]
    fn embedding_provider_parse_case_insensitive() {
        assert_eq!(
            EmbeddingProvider::parse("LOCAL"),
            Some(EmbeddingProvider::Local)
        );
        assert_eq!(
            EmbeddingProvider::parse("Api"),
            Some(EmbeddingProvider::Api)
        );
        assert_eq!(
            EmbeddingProvider::parse("API"),
            Some(EmbeddingProvider::Api)
        );
    }

    #[test]
    fn embedding_provider_parse_unknown_returns_none() {
        assert_eq!(EmbeddingProvider::parse("unknown"), None);
        assert_eq!(EmbeddingProvider::parse(""), None);
        assert_eq!(EmbeddingProvider::parse("  "), None);
    }

    #[test]
    fn embedding_provider_as_str_roundtrip() {
        assert_eq!(EmbeddingProvider::Local.as_str(), "local");
        assert_eq!(EmbeddingProvider::Api.as_str(), "api");
        assert_eq!(
            EmbeddingProvider::parse(EmbeddingProvider::Local.as_str()),
            Some(EmbeddingProvider::Local)
        );
        assert_eq!(
            EmbeddingProvider::parse(EmbeddingProvider::Api.as_str()),
            Some(EmbeddingProvider::Api)
        );
    }
}
