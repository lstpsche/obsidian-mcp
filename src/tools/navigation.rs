//! Vault listing and navigation tools (`vault_list`, `vault_structure`).

use std::collections::BTreeMap;
use std::path::Path;

use rmcp::model::{CallToolResult, Content};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::error::VaultError;
use crate::vault::Vault;

/// Parameters for the `vault_list` tool.
#[derive(Deserialize, JsonSchema, Default)]
pub struct VaultListParams {
    /// Directory path relative to vault root. Omit or leave empty for vault root.
    pub path: Option<String>,
    /// List files recursively through subdirectories. Defaults to false.
    pub recursive: Option<bool>,
    /// Glob pattern to filter results (e.g., `"*.md"`, `"journal/**"`).
    pub glob: Option<String>,
}

/// List files and directories in the vault. Returns a JSON array of relative paths.
pub fn vault_list(
    vault: &Vault,
    params: VaultListParams,
) -> Result<CallToolResult, rmcp::ErrorData> {
    let dir = params.path.as_deref().unwrap_or("");
    let recursive = params.recursive.unwrap_or(false);
    let files = vault.list_files(Path::new(dir), recursive, params.glob.as_deref())?;

    let paths: Vec<&str> = files.iter().filter_map(|p| p.to_str()).collect();
    let json = serde_json::to_string_pretty(&paths)
        .map_err(|e| VaultError::Other(format!("JSON serialization failed: {e}")))?;

    Ok(CallToolResult::success(vec![Content::text(json)]))
}

/// Parameters for the `vault_structure` tool.
#[derive(Deserialize, JsonSchema, Default)]
pub struct VaultStructureParams {
    /// Directory path relative to vault root. Omit or leave empty for vault root.
    pub path: Option<String>,
    /// Maximum depth to display. Omit for unlimited depth.
    pub max_depth: Option<usize>,
}

/// Get a tree view of the vault structure, formatted like the `tree` command.
pub fn vault_structure(
    vault: &Vault,
    params: VaultStructureParams,
) -> Result<CallToolResult, rmcp::ErrorData> {
    let dir = params.path.as_deref().unwrap_or("");
    let dir_path = Path::new(dir);
    let files = vault.list_files(dir_path, true, None)?;

    let mut root = TreeNode::new();
    for path in &files {
        let relative = path.strip_prefix(dir_path).unwrap_or(path);
        if let Some(max) = params.max_depth
            && relative.components().count() > max
        {
            continue;
        }
        root.insert(relative);
    }

    let label = if dir.is_empty() { "." } else { dir };
    let mut output = label.to_string();
    output.push('\n');
    render_tree(&root, &mut output, "");

    if output.ends_with('\n') {
        output.pop();
    }

    Ok(CallToolResult::success(vec![Content::text(output)]))
}

struct TreeNode {
    children: BTreeMap<String, TreeNode>,
}

impl TreeNode {
    fn new() -> Self {
        Self {
            children: BTreeMap::new(),
        }
    }

    fn insert(&mut self, path: &Path) {
        let mut node = self;
        for component in path.components() {
            let name = component.as_os_str().to_string_lossy().into_owned();
            node = node.children.entry(name).or_insert_with(TreeNode::new);
        }
    }
}

fn render_tree(node: &TreeNode, output: &mut String, prefix: &str) {
    let count = node.children.len();
    for (i, (name, child)) in node.children.iter().enumerate() {
        let is_last = i == count - 1;
        let connector = if is_last { "└── " } else { "├── " };
        output.push_str(prefix);
        output.push_str(connector);
        output.push_str(name);
        output.push('\n');

        if !child.children.is_empty() {
            let child_prefix = if is_last {
                format!("{prefix}    ")
            } else {
                format!("{prefix}│   ")
            };
            render_tree(child, output, &child_prefix);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use super::*;
    use crate::test_helpers::{extract_text, test_config};

    fn create_test_vault(dir: &Path) {
        crate::test_helpers::create_test_vault(dir);
        fs::write(dir.join("readme.md"), "# Readme").unwrap();
        fs::write(dir.join("notes.md"), "# Notes").unwrap();
        fs::create_dir_all(dir.join("journal")).unwrap();
        fs::write(dir.join("journal/2024-01-01.md"), "# Jan 1").unwrap();
        fs::write(dir.join("journal/2024-01-02.md"), "# Jan 2").unwrap();
        fs::create_dir_all(dir.join("projects/alpha")).unwrap();
        fs::write(dir.join("projects/alpha/spec.md"), "# Spec").unwrap();
    }

    // ── vault_list ──

    #[tokio::test]
    async fn list_root_non_recursive() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        let result = vault_list(&vault, VaultListParams::default()).unwrap();
        let text = extract_text(&result);
        let paths: Vec<String> = serde_json::from_str(text).unwrap();

        assert!(paths.contains(&"readme.md".to_string()));
        assert!(paths.contains(&"notes.md".to_string()));
        assert!(paths.contains(&"journal".to_string()));
        assert!(paths.contains(&"projects".to_string()));
        assert!(!paths.iter().any(|p| p.contains(".obsidian")));
        assert!(!paths.iter().any(|p| p.contains("2024")));
    }

    #[tokio::test]
    async fn list_recursive() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        let result = vault_list(
            &vault,
            VaultListParams {
                recursive: Some(true),
                ..Default::default()
            },
        )
        .unwrap();
        let text = extract_text(&result);
        let paths: Vec<String> = serde_json::from_str(text).unwrap();

        assert!(paths.iter().any(|p| p.contains("2024-01-01.md")));
        assert!(paths.iter().any(|p| p.contains("spec.md")));
    }

    #[tokio::test]
    async fn list_with_glob() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        let result = vault_list(
            &vault,
            VaultListParams {
                recursive: Some(true),
                glob: Some("**/*.md".to_string()),
                ..Default::default()
            },
        )
        .unwrap();
        let text = extract_text(&result);
        let paths: Vec<String> = serde_json::from_str(text).unwrap();

        for p in &paths {
            assert!(p.ends_with(".md"), "expected .md file, got: {p}");
        }
        assert!(paths.len() >= 4);
    }

    #[tokio::test]
    async fn list_subdirectory() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        let result = vault_list(
            &vault,
            VaultListParams {
                path: Some("journal".to_string()),
                ..Default::default()
            },
        )
        .unwrap();
        let text = extract_text(&result);
        let paths: Vec<String> = serde_json::from_str(text).unwrap();

        assert_eq!(paths.len(), 2);
        assert!(paths.iter().all(|p| p.contains("journal")));
    }

    #[tokio::test]
    async fn list_nonexistent_dir_errors() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        let result = vault_list(
            &vault,
            VaultListParams {
                path: Some("nonexistent".to_string()),
                ..Default::default()
            },
        );
        assert!(result.is_err());
    }

    // ── vault_structure ──

    #[tokio::test]
    async fn structure_full_tree() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        let result = vault_structure(&vault, VaultStructureParams::default()).unwrap();
        let text = extract_text(&result);

        assert!(text.starts_with('.'));
        assert!(text.contains("├── ") || text.contains("└── "));
        assert!(text.contains("readme.md"));
        assert!(text.contains("journal"));
        assert!(text.contains("spec.md"));
    }

    #[tokio::test]
    async fn structure_max_depth_1() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        let result = vault_structure(
            &vault,
            VaultStructureParams {
                max_depth: Some(1),
                ..Default::default()
            },
        )
        .unwrap();
        let text = extract_text(&result);

        assert!(text.contains("journal"));
        assert!(text.contains("readme.md"));
        assert!(!text.contains("2024-01-01.md"));
        assert!(!text.contains("spec.md"));
    }

    #[tokio::test]
    async fn structure_subdirectory() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        let result = vault_structure(
            &vault,
            VaultStructureParams {
                path: Some("projects".to_string()),
                ..Default::default()
            },
        )
        .unwrap();
        let text = extract_text(&result);

        assert!(text.starts_with("projects"));
        assert!(text.contains("alpha"));
        assert!(text.contains("spec.md"));
        assert!(!text.contains("journal"));
    }

    #[tokio::test]
    async fn structure_nonexistent_dir_errors() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        let result = vault_structure(
            &vault,
            VaultStructureParams {
                path: Some("nonexistent".to_string()),
                ..Default::default()
            },
        );
        assert!(result.is_err());
    }
}
