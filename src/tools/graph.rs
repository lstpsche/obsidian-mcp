//! Backlink, outgoing-link, and graph traversal tools.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use rmcp::model::{CallToolResult, ContentBlock, ErrorCode};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::VaultError;
use crate::vault::Vault;

// ── param struct ────────────────────────────────────────────────────

/// Parameters for the `wikilinks` tool.
#[derive(Debug, Deserialize, JsonSchema, Default)]
pub struct WikilinksParams {
    /// Query type: `"backlinks"`, `"outgoing"`, `"broken"`, or `"orphans"`.
    pub query: String,
    /// Note path (relative to vault root). Required for `"backlinks"` and `"outgoing"`, optional for `"broken"`, unused for `"orphans"`.
    pub path: Option<String>,
}

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

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OrphanStatus {
    /// Note has no inbound links and no outbound wikilinks at all.
    NoLinks,
    /// Note has no inbound links and no resolvable outbound links,
    /// but it does contain outgoing wikilinks that are broken.
    BrokenOutgoingOnly,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct OrphanNoteEntry {
    /// Path of the disconnected note.
    pub path: PathBuf,
    /// Why this note is disconnected from the resolvable graph.
    pub status: OrphanStatus,
    /// Broken outgoing targets found in this note (empty for `no_links`).
    pub broken_targets: Vec<String>,
}

// ── handler functions ───────────────────────────────────────────────

fn to_json_text(value: &impl Serialize) -> Result<CallToolResult, rmcp::ErrorData> {
    let json = serde_json::to_string_pretty(value)
        .map_err(|e| VaultError::Other(format!("JSON serialization failed: {e}")))?;
    Ok(CallToolResult::success(vec![ContentBlock::text(json)]))
}

fn has_resolved_target(vault: &Vault, target: &str) -> bool {
    !target.is_empty() && vault.resolve_link(target).is_some()
}

fn is_broken_target(vault: &Vault, target: &str) -> bool {
    !target.is_empty() && vault.resolve_link(target).is_none()
}

/// Query the vault's wikilink graph: backlinks, outgoing links, broken links, or orphan notes.
pub async fn wikilinks(
    vault: &Vault,
    params: WikilinksParams,
) -> Result<CallToolResult, rmcp::ErrorData> {
    if params.query.eq_ignore_ascii_case("backlinks") {
        let p = params.path.as_deref().ok_or_else(|| {
            rmcp::ErrorData::new(
                ErrorCode::INVALID_PARAMS,
                "'path' is required for query 'backlinks'",
                None::<serde_json::Value>,
            )
        })?;
        wikilinks_backlinks(vault, p).await
    } else if params.query.eq_ignore_ascii_case("outgoing") {
        let p = params.path.as_deref().ok_or_else(|| {
            rmcp::ErrorData::new(
                ErrorCode::INVALID_PARAMS,
                "'path' is required for query 'outgoing'",
                None::<serde_json::Value>,
            )
        })?;
        wikilinks_outgoing(vault, p).await
    } else if params.query.eq_ignore_ascii_case("broken") {
        wikilinks_broken(vault, params.path.as_deref()).await
    } else if params.query.eq_ignore_ascii_case("orphans") {
        wikilinks_orphans(vault).await
    } else {
        Err(rmcp::ErrorData::new(
            ErrorCode::INVALID_PARAMS,
            format!(
                "Unknown query '{}'. Valid values: \"backlinks\", \"outgoing\", \"broken\", \"orphans\"",
                params.query
            ),
            None::<serde_json::Value>,
        ))
    }
}

async fn wikilinks_backlinks(
    vault: &Vault,
    note_path: &str,
) -> Result<CallToolResult, rmcp::ErrorData> {
    let path = Path::new(note_path);
    let target_path = vault.canonical_existing_relative_path(path)?;
    vault.get_note_metadata(&target_path)?;

    let backlink_notes = vault.backlinks(&target_path)?;

    let result: Vec<BacklinkSource> = backlink_notes
        .iter()
        .filter_map(|source| {
            let matching: Vec<BacklinkRef> = source
                .links
                .iter()
                .filter(|link| vault.resolve_link(&link.target).as_deref() == Some(&target_path))
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

async fn wikilinks_outgoing(
    vault: &Vault,
    note_path: &str,
) -> Result<CallToolResult, rmcp::ErrorData> {
    let path = Path::new(note_path);
    let actual_path = vault.canonical_existing_relative_path(path)?;
    vault.get_note_metadata(&actual_path)?;

    let links = vault.outgoing_links(&actual_path)?;

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

async fn wikilinks_broken(
    vault: &Vault,
    note_path: Option<&str>,
) -> Result<CallToolResult, rmcp::ErrorData> {
    let result: Vec<BrokenLink> = match note_path {
        Some(p) => {
            let path = Path::new(p);
            let actual_path = vault.canonical_existing_relative_path(path)?;
            vault.get_note_metadata(&actual_path)?;
            let links = vault.outgoing_links(&actual_path)?;

            links
                .into_iter()
                .filter(|link| is_broken_target(vault, &link.target))
                .map(|link| BrokenLink {
                    source_path: actual_path.clone(),
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

async fn wikilinks_orphans(vault: &Vault) -> Result<CallToolResult, rmcp::ErrorData> {
    let mut disconnected: Vec<OrphanNoteEntry> = vault
        .orphan_notes()?
        .into_iter()
        .map(|note| OrphanNoteEntry {
            path: note.path,
            status: OrphanStatus::NoLinks,
            broken_targets: Vec::new(),
        })
        .collect();

    let mut seen_paths: HashSet<PathBuf> = disconnected.iter().map(|e| e.path.clone()).collect();

    let mut broken_by_source: HashMap<PathBuf, HashSet<String>> = HashMap::new();
    for (source_path, link) in vault.broken_links()? {
        broken_by_source
            .entry(source_path)
            .or_default()
            .insert(link.target);
    }

    for (source_path, broken_targets) in broken_by_source {
        if seen_paths.contains(&source_path) {
            continue;
        }

        let has_incoming = !vault.backlinks(&source_path)?.is_empty();
        if has_incoming {
            continue;
        }

        let has_resolved_outgoing = vault
            .outgoing_links(&source_path)?
            .into_iter()
            .any(|link| has_resolved_target(vault, &link.target));
        if has_resolved_outgoing {
            continue;
        }

        let mut broken_targets: Vec<String> = broken_targets.into_iter().collect();
        broken_targets.sort();

        disconnected.push(OrphanNoteEntry {
            path: source_path.clone(),
            status: OrphanStatus::BrokenOutgoingOnly,
            broken_targets,
        });
        seen_paths.insert(source_path);
    }

    disconnected.sort_by(|a, b| a.path.cmp(&b.path));
    to_json_text(&disconnected)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::test_helpers::{extract_text, test_config};
    use unicode_normalization::UnicodeNormalization;

    fn create_test_vault(dir: &Path) {
        crate::test_helpers::create_test_vault(dir);

        std::fs::write(dir.join("a.md"), "# A\n\nLinks to [[b]] and [[c]].\n").unwrap();
        std::fs::write(dir.join("b.md"), "# B\n\nLinks back to [[a]].\n").unwrap();
        std::fs::write(dir.join("c.md"), "# C\n\nLinks to [[a#heading|alias]].\n").unwrap();
        std::fs::write(
            dir.join("d.md"),
            "# D\n\nLinks to [[nonexistent]] and [[a]].\n",
        )
        .unwrap();
        std::fs::write(
            dir.join("broken_only.md"),
            "# Broken Only\n\nLinks to [[still_missing]].\n",
        )
        .unwrap();
        std::fs::write(dir.join("orphan.md"), "# Orphan\n\nNo links here.\n").unwrap();
    }

    fn backlinks_params(path: &str) -> WikilinksParams {
        WikilinksParams {
            query: "backlinks".into(),
            path: Some(path.into()),
            ..Default::default()
        }
    }

    fn outgoing_params(path: &str) -> WikilinksParams {
        WikilinksParams {
            query: "outgoing".into(),
            path: Some(path.into()),
            ..Default::default()
        }
    }

    fn broken_params(path: Option<&str>) -> WikilinksParams {
        WikilinksParams {
            query: "broken".into(),
            path: path.map(Into::into),
            ..Default::default()
        }
    }

    fn orphans_params() -> WikilinksParams {
        WikilinksParams {
            query: "orphans".into(),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn backlinks_returns_correct_sources_and_refs() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        let result = wikilinks(&vault, backlinks_params("a.md")).await.unwrap();
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
    async fn backlinks_accept_canonically_equivalent_unicode_path() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let composed = "02_База-знаний/Сущности/lic1c.md";
        let decomposed: String = composed.nfd().collect();
        let disk_path = PathBuf::from(&decomposed);
        std::fs::create_dir_all(dir.path().join(disk_path.parent().unwrap())).unwrap();
        std::fs::write(dir.path().join(&disk_path), "# License\n").unwrap();
        std::fs::write(
            dir.path().join("source.md"),
            "# Source\n\nLinks to [[02_База-знаний/Сущности/lic1c]].\n",
        )
        .unwrap();
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        let result = wikilinks(&vault, backlinks_params(composed)).await.unwrap();
        let text = extract_text(&result);
        let backlinks: Vec<serde_json::Value> = serde_json::from_str(text).unwrap();

        assert!(
            backlinks
                .iter()
                .any(|entry| entry["source_path"] == "source.md")
        );
    }

    #[tokio::test]
    async fn backlinks_nonexistent_note_errors() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        assert!(
            wikilinks(&vault, backlinks_params("nonexistent.md"))
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn backlinks_note_with_none_returns_empty() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        let result = wikilinks(&vault, backlinks_params("orphan.md"))
            .await
            .unwrap();
        let text = extract_text(&result);
        let backlinks: Vec<serde_json::Value> = serde_json::from_str(text).unwrap();
        assert!(backlinks.is_empty());
    }

    #[tokio::test]
    async fn backlinks_missing_path_errors() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        let result = wikilinks(
            &vault,
            WikilinksParams {
                query: "backlinks".into(),
                path: None,
                ..Default::default()
            },
        )
        .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn outgoing_links_with_resolved_paths() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        let result = wikilinks(&vault, outgoing_params("a.md")).await.unwrap();
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

        let result = wikilinks(&vault, outgoing_params("d.md")).await.unwrap();
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

        let result = wikilinks(&vault, outgoing_params("c.md")).await.unwrap();
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
            wikilinks(&vault, outgoing_params("nonexistent.md"))
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn broken_links_vault_wide() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        let result = wikilinks(&vault, broken_params(None)).await.unwrap();
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

        let result = wikilinks(&vault, broken_params(Some("d.md")))
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

        let result = wikilinks(&vault, broken_params(Some("a.md")))
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
            wikilinks(&vault, broken_params(Some("nonexistent.md")))
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn orphan_notes_detected() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        let result = wikilinks(&vault, orphans_params()).await.unwrap();
        let text = extract_text(&result);
        let orphans: Vec<OrphanNoteEntry> = serde_json::from_str(text).unwrap();

        let orphan_entry = orphans
            .iter()
            .find(|entry| entry.path == PathBuf::from("orphan.md"))
            .expect("expected orphan.md in orphans");
        assert_eq!(orphan_entry.status, OrphanStatus::NoLinks);

        let broken_only_entry = orphans
            .iter()
            .find(|entry| entry.path == PathBuf::from("broken_only.md"))
            .expect("expected broken_only.md in orphans");
        assert_eq!(broken_only_entry.status, OrphanStatus::BrokenOutgoingOnly);
        assert!(
            broken_only_entry
                .broken_targets
                .iter()
                .any(|target| target == "still_missing")
        );

        assert!(
            !orphans
                .iter()
                .any(|entry| entry.path == PathBuf::from("a.md"))
        );
        assert!(
            !orphans
                .iter()
                .any(|entry| entry.path == PathBuf::from("b.md"))
        );
        assert!(
            !orphans
                .iter()
                .any(|entry| entry.path == PathBuf::from("c.md"))
        );
    }

    #[tokio::test]
    async fn orphan_notes_exclude_notes_with_outgoing_links() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        let result = wikilinks(&vault, orphans_params()).await.unwrap();
        let text = extract_text(&result);
        let orphans: Vec<OrphanNoteEntry> = serde_json::from_str(text).unwrap();
        assert!(
            !orphans
                .iter()
                .any(|entry| entry.path == PathBuf::from("d.md"))
        );
    }

    #[tokio::test]
    async fn wikilinks_invalid_query() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        let result = wikilinks(
            &vault,
            WikilinksParams {
                query: "invalid".into(),
                ..Default::default()
            },
        )
        .await;
        assert!(result.is_err());
    }
}
