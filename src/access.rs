//! Optional audience policy for fixed-root private memory readers.

use std::path::{Component, Path};

use serde::{Deserialize, Serialize};

use crate::error::{VaultError, VaultResult};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ReadScope {
    pub version: u32,
    pub tenant_id: String,
    pub star_label: String,
    pub workspace_id: String,
    pub audience_kind: String,
    pub audience_id: String,
    pub purpose: String,
    pub scope_id: String,
    pub allowed_prefixes: Vec<String>,
    pub expires_at: f64,
}

impl ReadScope {
    pub fn validate(&self) -> VaultResult<()> {
        if self.version != 1
            || !matches!(self.audience_kind.as_str(), "channel" | "dm")
            || !matches!(
                self.purpose.as_str(),
                "conversation" | "reflection" | "operator"
            )
            || [
                &self.tenant_id,
                &self.star_label,
                &self.workspace_id,
                &self.audience_id,
                &self.scope_id,
            ]
            .iter()
            .any(|s| !valid_id(s))
            || !self.expires_at.is_finite()
            || self.expires_at <= chrono::Utc::now().timestamp_millis() as f64 / 1000.0
            || self.allowed_prefixes.is_empty()
            || self.allowed_prefixes.len() > 64
        {
            return Err(VaultError::Other(
                "invalid private memory read scope".into(),
            ));
        }
        for prefix in &self.allowed_prefixes {
            let path = Path::new(prefix.trim_end_matches('/'));
            if !normalized(path)
                || (!prefix.ends_with('/') && !prefix.ends_with(".md"))
                || !path
                    .components()
                    .any(|part| part.as_os_str() == self.scope_id.as_str())
                || !matches!(
                    path.components()
                        .next()
                        .and_then(|p| p.as_os_str().to_str()),
                    Some("Capture" | "Organize" | "Distill" | "Express")
                )
                || (prefix.starts_with("Capture/") && self.purpose == "conversation")
            {
                return Err(VaultError::Other(
                    "invalid private memory path scope".into(),
                ));
            }
        }
        Ok(())
    }

    pub fn allows_path(&self, path: &Path) -> bool {
        if !normalized(path) {
            return false;
        }
        let Some(path) = path.to_str() else {
            return false;
        };
        self.allowed_prefixes.iter().any(|prefix| {
            if prefix.ends_with('/') {
                path.starts_with(prefix)
            } else {
                path == prefix
            }
        })
    }

    pub fn allows_frontmatter(&self, frontmatter: Option<&serde_json::Value>) -> bool {
        let Some(fm) = frontmatter else {
            return false;
        };
        if fm.get("memory_schema").and_then(|value| value.as_str()) != Some("slack-code/1")
            || fm.get("sensitivity").and_then(|value| value.as_str()) != Some("normal")
        {
            return false;
        }
        [
            ("tenant_id", &self.tenant_id),
            ("star_label", &self.star_label),
            ("workspace_id", &self.workspace_id),
            ("audience_kind", &self.audience_kind),
            ("audience_id", &self.audience_id),
        ]
        .iter()
        .all(|(key, value)| fm.get(key).and_then(|v| v.as_str()) == Some(value.as_str()))
    }
}

/// Keep private provider inputs within the trusted, confirmed curated namespace.
pub(crate) fn private_source_eligible(path: &Path, content: &str) -> bool {
    let Ok(Some(fm)) = crate::vault::frontmatter::parse_frontmatter(content) else {
        return false;
    };
    let field = |key: &str| fm.get(key).and_then(|value| value.as_str());
    let (Some(tenant), Some(star), Some(workspace), Some(kind), Some(audience)) = (
        field("tenant_id"),
        field("star_label"),
        field("workspace_id"),
        field("audience_kind"),
        field("audience_id"),
    ) else {
        return false;
    };
    if ![tenant, star, workspace, audience]
        .iter()
        .all(|value| valid_id(value))
        || !matches!(kind, "channel" | "dm")
        || content.lines().any(|line| {
            line.trim_start()
                .strip_prefix(['-', '*'])
                .is_some_and(|rest| {
                    rest.starts_with(char::is_whitespace)
                        && rest
                            .trim_start()
                            .strip_prefix("[inferred")
                            .is_some_and(|tag| tag.starts_with(char::is_whitespace))
                })
        })
    {
        return false;
    }
    let scope = format!(
        "slack-{}-{kind}-{}",
        workspace.to_ascii_lowercase(),
        audience.to_ascii_lowercase()
    );
    let policy = ReadScope {
        version: 1,
        tenant_id: tenant.into(),
        star_label: star.into(),
        workspace_id: workspace.into(),
        audience_kind: kind.into(),
        audience_id: audience.into(),
        purpose: "conversation".into(),
        scope_id: scope.clone(),
        expires_at: f64::MAX,
        allowed_prefixes: ["Self", "People", "Work", "Topics"]
            .into_iter()
            .map(|section| format!("Organize/{section}/{scope}/"))
            .chain([
                format!("Express/Skills/{scope}/"),
                format!("Distill/{scope}/brief.md"),
                format!("Distill/{scope}/index.md"),
            ])
            .collect(),
    };
    path.extension().is_some_and(|value| value == "md")
        && policy.allows_path(path)
        && policy.allows_frontmatter(Some(&fm))
}

fn valid_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 256
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
}

fn normalized(path: &Path) -> bool {
    let Some(raw) = path.to_str() else {
        return false;
    };
    !raw.is_empty()
        && !raw.contains('\\')
        && !raw.contains('\0')
        && !raw.contains("//")
        && !raw.split('/').any(|p| p.is_empty() || p.starts_with('.'))
        && path
            .components()
            .all(|part| matches!(part, Component::Normal(_)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scope() -> ReadScope {
        ReadScope {
            version: 1,
            tenant_id: "bloomer".into(),
            star_label: "bloom".into(),
            workspace_id: "T1".into(),
            audience_kind: "dm".into(),
            audience_id: "U1".into(),
            purpose: "conversation".into(),
            scope_id: "slack-t1-dm-u1".into(),
            allowed_prefixes: vec!["Distill/slack-t1-dm-u1/index.md".into()],
            expires_at: 4_000_000_000.0,
        }
    }

    #[test]
    fn scope_rejects_expiry_and_noncanonical_or_raw_prefixes() {
        let mut policy = scope();
        assert!(policy.validate().is_ok());
        for prefix in [
            "../Distill/slack-t1-dm-u1/",
            "/Distill/slack-t1-dm-u1/",
            "Distill//slack-t1-dm-u1/",
            "Distill/slack-t1-dm-u1/../other/",
            "Distill/slack-t1-dm-u2/",
            "Capture/chats/slack-t1-dm-u1/",
        ] {
            policy.allowed_prefixes = vec![prefix.into()];
            assert!(policy.validate().is_err(), "{prefix}");
        }
        policy = scope();
        policy.expires_at = 0.0;
        assert!(policy.validate().is_err());
        policy.expires_at = f64::NAN;
        assert!(policy.validate().is_err());
    }

    #[test]
    fn directory_grants_and_exact_hot_files_have_distinct_boundaries() {
        let mut policy = scope();
        assert!(policy.allows_path(Path::new("Distill/slack-t1-dm-u1/index.md")));
        assert!(!policy.allows_path(Path::new("Distill/slack-t1-dm-u1/index.md-extra.md")));
        policy.allowed_prefixes = vec!["Organize/Work/slack-t1-dm-u1/".into()];
        assert!(policy.allows_path(Path::new("Organize/Work/slack-t1-dm-u1/task.md")));
        assert!(!policy.allows_path(Path::new("Organize/Work/slack-t1-dm-u11/task.md")));
        assert!(!policy.allows_frontmatter(None));
    }
}
