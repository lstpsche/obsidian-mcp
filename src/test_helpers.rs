//! Shared test utilities for unit tests across vault and tool modules.

use std::path::Path;

use rmcp::model::CallToolResult;

use crate::config::{Config, ToolFilter};

pub fn test_config(vault_root: &Path) -> Config {
    Config {
        vault_path: vault_root.to_path_buf(),
        watch: false,
        log_level: "error".into(),
        transport: crate::config::Transport::Stdio,
        http_host: crate::config::DEFAULT_HTTP_HOST,
        http_port: crate::config::DEFAULT_HTTP_PORT,
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

pub fn tantivy_config(vault_root: &Path) -> Config {
    Config {
        vault_path: vault_root.to_path_buf(),
        watch: false,
        log_level: "error".into(),
        transport: crate::config::Transport::Stdio,
        http_host: crate::config::DEFAULT_HTTP_HOST,
        http_port: crate::config::DEFAULT_HTTP_PORT,
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

pub fn create_test_vault(dir: &Path) {
    std::fs::create_dir_all(dir.join(".obsidian")).unwrap();
}

pub fn extract_text(result: &CallToolResult) -> &str {
    result.content[0]
        .as_text()
        .expect("expected text content")
        .text
        .as_str()
}
