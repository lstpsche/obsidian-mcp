//! Shared test utilities for unit tests across vault and tool modules.

use std::path::Path;

use rmcp::model::CallToolResult;

use crate::config::Config;

pub fn test_config(vault_root: &Path) -> Config {
    Config {
        vault_path: vault_root.to_path_buf(),
        watch: false,
        log_level: "error".into(),
        tantivy: false,
        embeddings: false,
        embeddings_model: String::new(),
        hybrid_alpha: 0.25,
    }
}

pub fn tantivy_config(vault_root: &Path) -> Config {
    Config {
        vault_path: vault_root.to_path_buf(),
        watch: false,
        log_level: "error".into(),
        tantivy: true,
        embeddings: false,
        embeddings_model: String::new(),
        hybrid_alpha: 0.25,
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
