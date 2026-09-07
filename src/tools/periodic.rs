//! Periodic note tool — unified handler for daily, weekly, monthly, quarterly, yearly notes.

use chrono::NaiveDate;
use rmcp::model::{CallToolResult, ErrorCode};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::models::NotePeriod;
use crate::vault::Vault;

fn parse_date(date_str: &str) -> Result<NaiveDate, rmcp::ErrorData> {
    NaiveDate::parse_from_str(date_str, "%Y-%m-%d").map_err(|_| {
        rmcp::ErrorData::new(
            ErrorCode::INVALID_PARAMS,
            format!("Invalid date '{date_str}'; expected YYYY-MM-DD"),
            None::<serde_json::Value>,
        )
    })
}

#[derive(Deserialize, JsonSchema, Default)]
pub struct PeriodicParams {
    /// Action to perform: `"get"` (read note content), `"create"` (create from template or custom content), or `"list"` (list recent notes).
    pub action: String,
    /// Period type: daily, weekly, monthly, quarterly, yearly.
    pub period: NotePeriod,
    /// ISO date (YYYY-MM-DD). Defaults to today. Used by `"get"` and `"create"`.
    #[serde(default)]
    pub date: Option<String>,
    /// Custom content; overrides template expansion. Only used by `"create"`.
    #[serde(default)]
    pub content: Option<String>,
    /// Maximum number of notes to return (default: 10). Only used by `"list"`.
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Serialize, JsonSchema)]
#[serde(tag = "action", rename_all = "snake_case")]
#[schemars(extend("type" = "object"))]
pub(super) enum PeriodicOutput {
    Get {
        content: String,
    },
    Create {
        path: std::path::PathBuf,
        message: String,
    },
    List {
        results: Vec<PeriodicEntry>,
    },
}

#[derive(Serialize, JsonSchema)]
pub(super) struct PeriodicEntry {
    /// Filename stem in the configured periodic-note date format.
    date: String,
    path: std::path::PathBuf,
}

pub async fn periodic(
    vault: &Vault,
    params: PeriodicParams,
) -> Result<CallToolResult, rmcp::ErrorData> {
    match params.action.to_ascii_lowercase().as_str() {
        "get" => {
            let date = params.date.map(|s| parse_date(&s)).transpose()?;
            let content = vault.get_periodic_note(&params.period, date)?;
            super::output::with_text(
                &PeriodicOutput::Get {
                    content: content.clone(),
                },
                content,
            )
        }
        "create" => {
            let date = params.date.map(|s| parse_date(&s)).transpose()?;
            let path =
                vault.create_periodic_note(&params.period, date, params.content.as_deref())?;
            let message = format!("Created: {}", path.display());
            super::output::with_text(
                &PeriodicOutput::Create {
                    path,
                    message: message.clone(),
                },
                message,
            )
        }
        "list" => {
            let limit = params.limit.unwrap_or(10);
            let paths = vault.list_recent_periodic_notes(&params.period, limit)?;

            let items: Vec<PeriodicEntry> = paths
                .into_iter()
                .map(|p| {
                    let date = p
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or_default()
                        .to_string();
                    PeriodicEntry { path: p, date }
                })
                .collect();

            let text = serde_json::to_string_pretty(&items).map_err(|e| {
                rmcp::ErrorData::new(
                    ErrorCode::INTERNAL_ERROR,
                    e.to_string(),
                    None::<serde_json::Value>,
                )
            })?;
            super::output::with_text(&PeriodicOutput::List { results: items }, text)
        }
        other => Err(rmcp::ErrorData::new(
            ErrorCode::INVALID_PARAMS,
            format!("Unknown action '{other}'. Valid values: \"get\", \"create\", \"list\""),
            None::<serde_json::Value>,
        )),
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::test_helpers::{create_test_vault, extract_text, test_config};

    fn setup_daily_config(dir: &std::path::Path) {
        create_test_vault(dir);
        let daily_dir = dir.join("Daily");
        fs::create_dir_all(&daily_dir).unwrap();
        fs::write(
            dir.join(".obsidian/daily-notes.json"),
            r#"{"format":"YYYY-MM-DD","folder":"Daily"}"#,
        )
        .unwrap();
    }

    #[tokio::test]
    async fn unknown_action_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        setup_daily_config(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        let result = periodic(
            &vault,
            PeriodicParams {
                action: "destroy".into(),
                ..Default::default()
            },
        )
        .await;
        let err = result.unwrap_err();
        assert!(err.message.contains("Unknown action"));
        assert!(err.message.contains("destroy"));
    }

    #[tokio::test]
    async fn action_is_case_insensitive() {
        let dir = tempfile::tempdir().unwrap();
        setup_daily_config(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        let result = periodic(
            &vault,
            PeriodicParams {
                action: "LIST".into(),
                ..Default::default()
            },
        )
        .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn list_returns_empty_array() {
        let dir = tempfile::tempdir().unwrap();
        setup_daily_config(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        let result = periodic(
            &vault,
            PeriodicParams {
                action: "list".into(),
                limit: Some(5),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let items: Vec<serde_json::Value> = serde_json::from_str(extract_text(&result)).unwrap();
        assert!(items.is_empty());
    }

    #[tokio::test]
    async fn create_then_get() {
        let dir = tempfile::tempdir().unwrap();
        setup_daily_config(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        let msg = periodic(
            &vault,
            PeriodicParams {
                action: "create".into(),
                date: Some("2026-01-15".into()),
                content: Some("hello periodic".into()),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert!(extract_text(&msg).contains("Created"));

        let content = periodic(
            &vault,
            PeriodicParams {
                action: "get".into(),
                date: Some("2026-01-15".into()),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert!(extract_text(&content).contains("hello periodic"));
    }
}
