//! Backlink, outgoing-link, and graph traversal tools.

use std::path::{Path, PathBuf};

use rmcp::model::{CallToolResult, Content};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::VaultError;
use crate::vault::Vault;

// ── param structs ───────────────────────────────────────────────────

#[derive(Debug, Deserialize, JsonSchema, Default)]
pub struct LinksBacklinksParams {
    /// Path to the note (relative to vault root), e.g. `notes/example.md`.
    pub path: String,
}

#[derive(Debug, Deserialize, JsonSchema, Default)]
pub struct LinksOutgoingParams {
    /// Path to the note (relative to vault root), e.g. `notes/example.md`.
    pub path: String,
}

#[derive(Debug, Deserialize, JsonSchema, Default)]
pub struct LinksBrokenParams {
    /// Optional note path to check. If omitted, checks the entire vault.
    pub path: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema, Default)]
pub struct LinksOrphansParams {}

// ── response types ──────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct BacklinkSource {
    /// Path of the note that contains links to the target.
    pub source_path: PathBuf,
    /// The specific wikilinks in this note that point to the target.
    pub links: Vec<BacklinkRef>,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct BacklinkRef {
    /// Raw wikilink text, e.g. `[[note#heading|alias]]`.
    pub raw: String,
    /// 0-based line number where the link appears.
    pub line: usize,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct OutgoingLink {
    /// Raw wikilink text.
    pub raw: String,
    /// Link target (note name or path).
    pub target: String,
    /// Resolved vault-relative path, or `null` if the link is broken.
    pub resolved_path: Option<PathBuf>,
    /// Heading fragment, if present.
    pub heading: Option<String>,
    /// Block reference, if present.
    pub block_ref: Option<String>,
    /// Display alias, if present.
    pub alias: Option<String>,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct BrokenLink {
    /// Path of the note containing the broken link.
    pub source_path: PathBuf,
    /// Raw wikilink text.
    pub link_raw: String,
    /// Unresolved target.
    pub target: String,
}

// ── handler functions ───────────────────────────────────────────────

fn to_json_text(value: &impl Serialize) -> Result<CallToolResult, rmcp::ErrorData> {
    let json = serde_json::to_string_pretty(value)
        .map_err(|e| VaultError::Other(format!("JSON serialization failed: {e}")))?;
    Ok(CallToolResult::success(vec![Content::text(json)]))
}

/// Find all notes linking TO a given note, with the specific wikilinks used.
pub async fn links_backlinks(
    vault: &Vault,
    params: LinksBacklinksParams,
) -> Result<CallToolResult, rmcp::ErrorData> {
    let path = Path::new(&params.path);
    vault.get_note_metadata(path)?;

    let backlink_notes = vault.backlinks(path)?;

    let result: Vec<BacklinkSource> = backlink_notes
        .iter()
        .filter_map(|source| {
            let matching: Vec<BacklinkRef> = source
                .links
                .iter()
                .filter(|link| vault.resolve_link(&link.target).as_deref() == Some(path))
                .map(|link| BacklinkRef {
                    raw: link.raw.clone(),
                    line: link.line,
                })
                .collect();

            if matching.is_empty() {
                None
            } else {
                Some(BacklinkSource {
                    source_path: source.path.clone(),
                    links: matching,
                })
            }
        })
        .collect();

    to_json_text(&result)
}

/// Find all outgoing links FROM a given note, with resolution status.
pub async fn links_outgoing(
    vault: &Vault,
    params: LinksOutgoingParams,
) -> Result<CallToolResult, rmcp::ErrorData> {
    let path = Path::new(&params.path);
    vault.get_note_metadata(path)?;

    let links = vault.outgoing_links(path)?;

    let result: Vec<OutgoingLink> = links
        .into_iter()
        .map(|link| {
            let resolved_path = vault.resolve_link(&link.target);
            OutgoingLink {
                raw: link.raw,
                target: link.target,
                resolved_path,
                heading: link.heading,
                block_ref: link.block_ref,
                alias: link.alias,
            }
        })
        .collect();

    to_json_text(&result)
}

/// Find all broken (unresolved) wikilinks, optionally filtered to a single note.
pub async fn links_broken(
    vault: &Vault,
    params: LinksBrokenParams,
) -> Result<CallToolResult, rmcp::ErrorData> {
    let result: Vec<BrokenLink> = match params.path.as_deref() {
        Some(p) => {
            let path = Path::new(p);
            vault.get_note_metadata(path)?;
            let links = vault.outgoing_links(path)?;

            links
                .into_iter()
                .filter(|link| {
                    !link.target.is_empty() && vault.resolve_link(&link.target).is_none()
                })
                .map(|link| BrokenLink {
                    source_path: path.to_path_buf(),
                    link_raw: link.raw,
                    target: link.target,
                })
                .collect()
        }
        None => {
            let all = vault.broken_links()?;
            all.into_iter()
                .map(|(source_path, link)| BrokenLink {
                    source_path,
                    link_raw: link.raw,
                    target: link.target,
                })
                .collect()
        }
    };

    to_json_text(&result)
}

/// Find notes with no inbound and no outbound links.
pub async fn links_orphans(
    vault: &Vault,
    _params: LinksOrphansParams,
) -> Result<CallToolResult, rmcp::ErrorData> {
    let orphans = vault.orphan_notes()?;
    let paths: Vec<PathBuf> = orphans.into_iter().map(|n| n.path).collect();
    to_json_text(&paths)
}

#[cfg(test)]
mod tests {
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

        std::fs::write(dir.join("a.md"), "# A\n\nLinks to [[b]] and [[c]].\n").unwrap();
        std::fs::write(dir.join("b.md"), "# B\n\nLinks back to [[a]].\n").unwrap();
        std::fs::write(dir.join("c.md"), "# C\n\nLinks to [[a#heading|alias]].\n").unwrap();
        std::fs::write(
            dir.join("d.md"),
            "# D\n\nLinks to [[nonexistent]] and [[a]].\n",
        )
        .unwrap();
        std::fs::write(dir.join("orphan.md"), "# Orphan\n\nNo links here.\n").unwrap();
    }

    fn extract_text(result: &CallToolResult) -> &str {
        result.content[0]
            .as_text()
            .expect("expected text content")
            .text
            .as_str()
    }

    #[tokio::test]
    async fn backlinks_returns_correct_sources_and_refs() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        let result = links_backlinks(
            &vault,
            LinksBacklinksParams {
                path: "a.md".into(),
            },
        )
        .await
        .unwrap();
        let text = extract_text(&result);
        let backlinks: Vec<serde_json::Value> = serde_json::from_str(text).unwrap();

        let source_paths: Vec<&str> = backlinks
            .iter()
            .filter_map(|bl| bl["source_path"].as_str())
            .collect();
        assert!(source_paths.contains(&"b.md"));
        assert!(source_paths.contains(&"c.md"));
        assert!(source_paths.contains(&"d.md"));

        let b_entry = backlinks
            .iter()
            .find(|bl| bl["source_path"] == "b.md")
            .unwrap();
        let b_links = b_entry["links"].as_array().unwrap();
        assert_eq!(b_links.len(), 1);
        assert!(b_links[0]["raw"].as_str().unwrap().contains("[[a]]"));
    }

    #[tokio::test]
    async fn backlinks_nonexistent_note_errors() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        assert!(
            links_backlinks(
                &vault,
                LinksBacklinksParams {
                    path: "nonexistent.md".into()
                },
            )
            .await
            .is_err()
        );
    }

    #[tokio::test]
    async fn backlinks_note_with_none_returns_empty() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        let result = links_backlinks(
            &vault,
            LinksBacklinksParams {
                path: "orphan.md".into(),
            },
        )
        .await
        .unwrap();
        let text = extract_text(&result);
        let backlinks: Vec<serde_json::Value> = serde_json::from_str(text).unwrap();
        assert!(backlinks.is_empty());
    }

    #[tokio::test]
    async fn outgoing_links_with_resolved_paths() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        let result = links_outgoing(
            &vault,
            LinksOutgoingParams {
                path: "a.md".into(),
            },
        )
        .await
        .unwrap();
        let text = extract_text(&result);
        let links: Vec<serde_json::Value> = serde_json::from_str(text).unwrap();
        assert_eq!(links.len(), 2);

        let b_link = links.iter().find(|l| l["target"] == "b").unwrap();
        assert_eq!(b_link["resolved_path"], "b.md");

        let c_link = links.iter().find(|l| l["target"] == "c").unwrap();
        assert_eq!(c_link["resolved_path"], "c.md");
    }

    #[tokio::test]
    async fn outgoing_links_broken_shown_as_null() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        let result = links_outgoing(
            &vault,
            LinksOutgoingParams {
                path: "d.md".into(),
            },
        )
        .await
        .unwrap();
        let text = extract_text(&result);
        let links: Vec<serde_json::Value> = serde_json::from_str(text).unwrap();

        let broken = links.iter().find(|l| l["target"] == "nonexistent").unwrap();
        assert!(broken["resolved_path"].is_null());

        let resolved = links.iter().find(|l| l["target"] == "a").unwrap();
        assert_eq!(resolved["resolved_path"], "a.md");
    }

    #[tokio::test]
    async fn outgoing_links_include_heading_and_alias() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        let result = links_outgoing(
            &vault,
            LinksOutgoingParams {
                path: "c.md".into(),
            },
        )
        .await
        .unwrap();
        let text = extract_text(&result);
        let links: Vec<serde_json::Value> = serde_json::from_str(text).unwrap();
        assert_eq!(links.len(), 1);
        assert_eq!(links[0]["target"], "a");
        assert_eq!(links[0]["heading"], "heading");
        assert_eq!(links[0]["alias"], "alias");
        assert_eq!(links[0]["resolved_path"], "a.md");
    }

    #[tokio::test]
    async fn outgoing_links_nonexistent_note_errors() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        assert!(
            links_outgoing(
                &vault,
                LinksOutgoingParams {
                    path: "nonexistent.md".into()
                },
            )
            .await
            .is_err()
        );
    }

    #[tokio::test]
    async fn broken_links_vault_wide() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        let result = links_broken(&vault, LinksBrokenParams::default())
            .await
            .unwrap();
        let text = extract_text(&result);
        let broken: Vec<serde_json::Value> = serde_json::from_str(text).unwrap();
        assert!(!broken.is_empty());
        assert!(broken.iter().any(|bl| bl["target"] == "nonexistent"));
        assert!(broken.iter().any(|bl| bl["source_path"] == "d.md"));
    }

    #[tokio::test]
    async fn broken_links_single_note_with_broken() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        let result = links_broken(
            &vault,
            LinksBrokenParams {
                path: Some("d.md".into()),
            },
        )
        .await
        .unwrap();
        let text = extract_text(&result);
        let broken: Vec<serde_json::Value> = serde_json::from_str(text).unwrap();
        assert_eq!(broken.len(), 1);
        assert_eq!(broken[0]["target"], "nonexistent");
    }

    #[tokio::test]
    async fn broken_links_single_note_without_broken() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        let result = links_broken(
            &vault,
            LinksBrokenParams {
                path: Some("a.md".into()),
            },
        )
        .await
        .unwrap();
        let text = extract_text(&result);
        let broken: Vec<serde_json::Value> = serde_json::from_str(text).unwrap();
        assert!(broken.is_empty());
    }

    #[tokio::test]
    async fn broken_links_nonexistent_note_errors() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        assert!(
            links_broken(
                &vault,
                LinksBrokenParams {
                    path: Some("nonexistent.md".into()),
                },
            )
            .await
            .is_err()
        );
    }

    #[tokio::test]
    async fn orphan_notes_detected() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        let result = links_orphans(&vault, LinksOrphansParams {}).await.unwrap();
        let text = extract_text(&result);
        let orphans: Vec<String> = serde_json::from_str(text).unwrap();

        assert!(orphans.contains(&"orphan.md".to_string()));
        assert!(!orphans.contains(&"a.md".to_string()));
        assert!(!orphans.contains(&"b.md".to_string()));
        assert!(!orphans.contains(&"c.md".to_string()));
    }

    #[tokio::test]
    async fn orphan_notes_exclude_notes_with_outgoing_links() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        let result = links_orphans(&vault, LinksOrphansParams {}).await.unwrap();
        let text = extract_text(&result);
        let orphans: Vec<String> = serde_json::from_str(text).unwrap();
        assert!(!orphans.contains(&"d.md".to_string()));
    }
}
