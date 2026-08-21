//! Compile-time identity used to preserve and verify installed capabilities.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use super::UpgradeError;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BuildIdentity {
    pub name: String,
    pub version: String,
    pub target: String,
    pub features: BTreeSet<String>,
}

impl BuildIdentity {
    pub fn embedded() -> Self {
        Self {
            name: env!("CARGO_PKG_NAME").to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            target: env!("OBSIDIAN_MCP_BUILD_TARGET").to_string(),
            features: parse_features(env!("OBSIDIAN_MCP_BUILD_FEATURES")),
        }
    }

    pub fn to_json(&self) -> Result<String, UpgradeError> {
        serde_json::to_string(self).map_err(UpgradeError::from)
    }

    pub fn from_json(raw: &[u8]) -> Result<Self, UpgradeError> {
        serde_json::from_slice(raw).map_err(UpgradeError::from)
    }

    pub fn feature_list(&self) -> String {
        if self.features.is_empty() {
            "none".to_string()
        } else {
            self.features.iter().cloned().collect::<Vec<_>>().join(",")
        }
    }
}

fn parse_features(raw: &str) -> BTreeSet<String> {
    raw.split(',')
        .map(str::trim)
        .filter(|feature| !feature.is_empty() && *feature != "default")
        .map(str::to_owned)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_parser_sorts_and_deduplicates() {
        assert_eq!(
            parse_features("embeddings-api,embeddings,embeddings-api,default"),
            BTreeSet::from(["embeddings".to_string(), "embeddings-api".to_string()])
        );
    }

    #[test]
    fn embedded_identity_round_trips_as_json() {
        let identity = BuildIdentity::embedded();
        let encoded = identity.to_json().expect("build identity should serialize");
        let decoded = BuildIdentity::from_json(encoded.as_bytes())
            .expect("build identity should deserialize");
        assert_eq!(decoded, identity);
        assert_eq!(decoded.name, "obsidian-mcp");
        assert!(!decoded.version.is_empty());
        assert!(!decoded.target.is_empty());
    }

    #[test]
    fn feature_list_is_safe_for_human_output() {
        let identity = BuildIdentity {
            name: "obsidian-mcp".into(),
            version: "1.0.0".into(),
            target: "test-target".into(),
            features: BTreeSet::new(),
        };
        assert_eq!(identity.feature_list(), "none");
    }
}
