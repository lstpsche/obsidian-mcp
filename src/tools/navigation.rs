//! Vault listing and navigation tools (`vault_list`).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use rmcp::model::{CallToolResult, Content, ErrorCode};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::VaultError;
use crate::vault::Vault;

/// Parameters for the `vault_list` tool.
#[derive(Deserialize, JsonSchema, Default)]
pub struct VaultListParams {
    /// Directory path relative to vault root. Omit or leave empty for vault root.
    pub path: Option<String>,
    /// List files recursively through subdirectories. Defaults to false. Only used in list mode.
    pub recursive: Option<bool>,
    /// Glob pattern to filter results (e.g., `"*.md"`, `"journal/**"`). Only used in list mode.
    pub glob: Option<String>,
    /// Output format: `"list"` (default) returns a JSON array; `"tree"` returns a tree-formatted string.
    pub format: Option<String>,
    /// Maximum depth to display. In list mode, limits path component count. In tree mode, limits nesting depth.
    pub max_depth: Option<usize>,
    /// Include indexed note metadata in list mode. Defaults to false and is invalid in tree mode.
    pub include_metadata: Option<bool>,
}

#[derive(Serialize)]
struct VaultListEntry {
    path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tags: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    size: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    created: Option<DateTime<Utc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    modified: Option<DateTime<Utc>>,
}

impl VaultListEntry {
    fn path_only(path: &str) -> Self {
        Self {
            path: path.to_owned(),
            title: None,
            tags: None,
            size: None,
            created: None,
            modified: None,
        }
    }
}

/// List files and directories in the vault.
///
/// In `"list"` mode (default): returns a JSON array of relative paths.
/// In `"tree"` mode: returns a tree-formatted string like the `tree` command.
pub fn vault_list(
    vault: &Vault,
    params: VaultListParams,
) -> Result<CallToolResult, rmcp::ErrorData> {
    let format = params.format.as_deref().unwrap_or("list");

    if format.eq_ignore_ascii_case("list") {
        vault_list_flat(vault, &params)
    } else if format.eq_ignore_ascii_case("tree") {
        if params.include_metadata.unwrap_or(false) {
            return Err(rmcp::ErrorData::new(
                ErrorCode::INVALID_PARAMS,
                "`include_metadata` is only valid in list mode",
                None::<serde_json::Value>,
            ));
        }
        vault_list_tree(vault, &params)
    } else {
        Err(rmcp::ErrorData::new(
            ErrorCode::INVALID_PARAMS,
            format!("Unknown format '{format}'. Valid values: \"list\", \"tree\""),
            None::<serde_json::Value>,
        ))
    }
}

fn vault_list_flat(
    vault: &Vault,
    params: &VaultListParams,
) -> Result<CallToolResult, rmcp::ErrorData> {
    let dir = params.path.as_deref().unwrap_or("");
    let recursive = params.recursive.unwrap_or(false);
    let files = vault.list_files(Path::new(dir), recursive, params.glob.as_deref())?;

    let paths: Vec<&str> = files
        .iter()
        .filter(|p| {
            params
                .max_depth
                .is_none_or(|max| p.components().count() <= max)
        })
        .filter_map(|p| p.to_str())
        .collect();

    let json = if params.include_metadata.unwrap_or(false) {
        let entries = paths
            .into_iter()
            .map(|path| {
                let note_path = Path::new(path);
                let is_markdown = note_path
                    .extension()
                    .and_then(|extension| extension.to_str())
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("md"));
                if !is_markdown {
                    return Ok(VaultListEntry::path_only(path));
                }

                match vault.get_note_metadata(note_path) {
                    Ok(metadata) => Ok(VaultListEntry {
                        path: path.to_owned(),
                        title: Some(metadata.title),
                        tags: Some(metadata.tags),
                        size: Some(metadata.stat.size),
                        created: metadata.stat.created,
                        modified: metadata.stat.modified,
                    }),
                    Err(VaultError::NoteNotFound(_)) => Ok(VaultListEntry::path_only(path)),
                    Err(error) => Err(rmcp::ErrorData::from(error)),
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        serde_json::to_string_pretty(&entries)
    } else {
        serde_json::to_string_pretty(&paths)
    }
    .map_err(|e| VaultError::Other(format!("JSON serialization failed: {e}")))?;

    Ok(CallToolResult::success(vec![Content::text(json)]))
}

fn vault_list_tree(
    vault: &Vault,
    params: &VaultListParams,
) -> Result<CallToolResult, rmcp::ErrorData> {
    let dir = params.path.as_deref().unwrap_or("");
    let dir_path = Path::new(dir);
    let files = vault.list_files(dir_path, true, None)?;
    let canonical_dir = if dir.is_empty() {
        PathBuf::new()
    } else {
        vault.canonical_existing_relative_path(dir_path)?
    };

    let mut root = TreeNode::new();
    for path in &files {
        let relative = path.strip_prefix(&canonical_dir).unwrap_or(path);
        if let Some(max) = params.max_depth
            && relative.components().count() > max
        {
            continue;
        }
        root.insert(relative);
    }

    let label = if canonical_dir.as_os_str().is_empty() {
        ".".to_string()
    } else {
        canonical_dir.to_string_lossy().into_owned()
    };
    let mut output = label;
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
    use unicode_normalization::UnicodeNormalization;

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

    #[tokio::test]
    async fn list_metadata_enriches_indexed_notes_and_keeps_other_entries_path_only() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let content = "---\ntags: [alpha, beta]\n---\n# Readme";
        fs::write(dir.path().join("readme.md"), content).unwrap();
        fs::write(dir.path().join("UPPER.MD"), "# Upper").unwrap();
        fs::write(dir.path().join("loose.txt"), "not a note").unwrap();
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        let result = vault_list(
            &vault,
            VaultListParams {
                include_metadata: Some(true),
                ..Default::default()
            },
        )
        .unwrap();
        let entries: Vec<serde_json::Value> = serde_json::from_str(extract_text(&result)).unwrap();

        let readme = entries
            .iter()
            .find(|entry| entry["path"] == "readme.md")
            .unwrap();
        assert_eq!(readme["title"], "readme");
        assert_eq!(readme["tags"], serde_json::json!(["alpha", "beta"]));
        assert_eq!(readme["size"], content.len());
        if let Some(modified) = readme.get("modified") {
            assert!(modified.as_str().is_some());
        }

        let uppercase = entries
            .iter()
            .find(|entry| entry["path"] == "UPPER.MD")
            .unwrap();
        assert_eq!(uppercase["title"], "UPPER");
        assert_eq!(uppercase["tags"], serde_json::json!([]));

        for path in ["journal", "loose.txt"] {
            let entry = entries.iter().find(|entry| entry["path"] == path).unwrap();
            assert_eq!(entry, &serde_json::json!({"path": path}));
        }
    }

    #[tokio::test]
    async fn list_metadata_keeps_excluded_and_newly_unindexed_notes_path_only() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        fs::create_dir_all(dir.path().join("Archive")).unwrap();
        fs::write(dir.path().join("Archive/hidden.md"), "# Hidden").unwrap();
        let config = crate::config::Config {
            exclude_patterns: vec!["Archive/**".into()],
            ..test_config(dir.path())
        };
        let vault = Vault::open(&config).await.unwrap();
        fs::write(dir.path().join("late.md"), "# Late").unwrap();

        let result = vault_list(
            &vault,
            VaultListParams {
                recursive: Some(true),
                include_metadata: Some(true),
                ..Default::default()
            },
        )
        .unwrap();
        let entries: Vec<serde_json::Value> = serde_json::from_str(extract_text(&result)).unwrap();

        for path in ["Archive/hidden.md", "late.md"] {
            let entry = entries.iter().find(|entry| entry["path"] == path).unwrap();
            assert_eq!(entry, &serde_json::json!({"path": path}));
        }
    }

    #[tokio::test]
    async fn list_metadata_preserves_actual_unicode_path_spelling() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let composed = "é.md";
        let decomposed: String = composed.nfd().collect();
        fs::write(dir.path().join(&decomposed), "# Unicode").unwrap();
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        let result = vault_list(
            &vault,
            VaultListParams {
                include_metadata: Some(true),
                ..Default::default()
            },
        )
        .unwrap();
        let entries: Vec<serde_json::Value> = serde_json::from_str(extract_text(&result)).unwrap();

        assert!(entries.iter().any(|entry| entry["path"] == decomposed));
    }

    #[tokio::test]
    async fn list_metadata_false_preserves_default_output_exactly() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        let default = vault_list(&vault, VaultListParams::default()).unwrap();
        let explicitly_disabled = vault_list(
            &vault,
            VaultListParams {
                include_metadata: Some(false),
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(extract_text(&default), extract_text(&explicitly_disabled));
    }

    #[tokio::test]
    async fn list_metadata_preserves_recursive_glob_and_max_depth_membership() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        fs::create_dir_all(dir.path().join("journal/deep")).unwrap();
        fs::write(dir.path().join("journal/deep/hidden.md"), "# Deep").unwrap();
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();
        let params = || VaultListParams {
            recursive: Some(true),
            glob: Some("journal/**".into()),
            max_depth: Some(2),
            ..Default::default()
        };

        let default = vault_list(&vault, params()).unwrap();
        let expected: Vec<String> = serde_json::from_str(extract_text(&default)).unwrap();
        let metadata = vault_list(
            &vault,
            VaultListParams {
                include_metadata: Some(true),
                ..params()
            },
        )
        .unwrap();
        let entries: Vec<serde_json::Value> =
            serde_json::from_str(extract_text(&metadata)).unwrap();
        let actual: Vec<String> = entries
            .iter()
            .map(|entry| entry["path"].as_str().unwrap().to_owned())
            .collect();

        assert_eq!(actual, expected);
        assert!(actual.contains(&"journal/deep".to_string()));
        assert!(!actual.contains(&"journal/deep/hidden.md".to_string()));
    }

    // ── vault_list (tree mode) ──

    #[tokio::test]
    async fn list_tree_format() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        let result = vault_list(
            &vault,
            VaultListParams {
                format: Some("tree".into()),
                ..Default::default()
            },
        )
        .unwrap();
        let text = extract_text(&result);

        assert!(text.starts_with('.'));
        assert!(text.contains("├── ") || text.contains("└── "));
        assert!(text.contains("readme.md"));
        assert!(text.contains("journal"));
        assert!(text.contains("spec.md"));
    }

    #[tokio::test]
    async fn list_tree_max_depth_1() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        let result = vault_list(
            &vault,
            VaultListParams {
                format: Some("tree".into()),
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
    async fn list_tree_subdirectory() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        let result = vault_list(
            &vault,
            VaultListParams {
                format: Some("tree".into()),
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
    async fn list_tree_subdirectory_strips_canonical_unicode_dir() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let composed = "02_База-знаний";
        let decomposed: String = composed.nfd().collect();
        fs::create_dir_all(dir.path().join(&decomposed)).unwrap();
        fs::write(dir.path().join(&decomposed).join("lic1c.md"), "# License").unwrap();
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        let result = vault_list(
            &vault,
            VaultListParams {
                format: Some("tree".into()),
                path: Some(composed.to_string()),
                ..Default::default()
            },
        )
        .unwrap();
        let text = extract_text(&result);

        assert!(text.starts_with(&decomposed));
        assert!(text.contains("lic1c.md"));
        assert!(!text.contains(&format!("{decomposed}/lic1c.md")));
    }

    #[tokio::test]
    async fn list_tree_nonexistent_dir_errors() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        let result = vault_list(
            &vault,
            VaultListParams {
                format: Some("tree".into()),
                path: Some("nonexistent".to_string()),
                ..Default::default()
            },
        );
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn list_tree_rejects_metadata() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        let error = vault_list(
            &vault,
            VaultListParams {
                format: Some("tree".into()),
                include_metadata: Some(true),
                ..Default::default()
            },
        )
        .err()
        .expect("tree mode metadata must be rejected");

        assert_eq!(error.code, ErrorCode::INVALID_PARAMS);
    }

    #[tokio::test]
    async fn list_invalid_format_errors() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        let result = vault_list(
            &vault,
            VaultListParams {
                format: Some("invalid".into()),
                ..Default::default()
            },
        );
        assert!(result.is_err());
    }
}
