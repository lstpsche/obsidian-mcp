//! Path exclusion logic: compile glob patterns and test vault-relative paths.

use std::collections::BTreeSet;
use std::path::Path;

use globset::{Glob, GlobSet, GlobSetBuilder};
use sha2::{Digest, Sha256};

use super::fs::VaultRoot;
use crate::error::{VaultError, VaultResult};

/// Whether a vault-relative path belongs to the visible indexing namespace.
pub(crate) fn is_visible_path(path: &Path) -> bool {
    path.to_str()
        .is_some_and(|path| path.split(['/', '\\']).all(|name| !name.starts_with('.')))
}

/// Compiled set of glob patterns for excluding vault paths from indexing.
#[derive(Default)]
pub struct ExcludeSet {
    set: GlobSet,
    patterns: Vec<String>,
}

impl ExcludeSet {
    /// Compile a complete policy. A malformed exclusion cannot grant access
    /// to a path that the operator intended to exclude.
    pub fn build(patterns: Vec<String>) -> VaultResult<Self> {
        Self::from_sources(
            patterns
                .into_iter()
                .enumerate()
                .map(|(i, p)| ("exclusion policy", i + 1, p)),
        )
    }

    fn from_sources<'a>(
        entries: impl IntoIterator<Item = (&'a str, usize, String)>,
    ) -> VaultResult<Self> {
        let mut builder = GlobSetBuilder::new();
        let mut accepted = BTreeSet::new();
        for (source, line, raw) in entries {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                continue;
            }
            let normalized = match trimmed.strip_suffix('/') {
                Some(prefix) => format!("{prefix}/**"),
                None => trimmed.to_owned(),
            };
            if accepted.contains(&normalized) {
                continue;
            }
            let glob = Glob::new(&normalized).map_err(|error| {
                VaultError::InvalidPath(format!(
                    "invalid exclusion pattern in {source} at line {line}: {}",
                    error.kind()
                ))
            })?;
            builder.add(glob);
            accepted.insert(normalized);
        }
        Ok(Self {
            set: builder
                .build()
                .map_err(|err| VaultError::Other(format!("compile exclusion policy: {err}")))?,
            patterns: accepted.into_iter().collect(),
        })
    }

    pub fn merged(&self, extra: &[String]) -> VaultResult<Self> {
        Self::build(self.patterns.iter().chain(extra).cloned().collect())
    }

    pub fn scope_id(&self, canonical_root: &Path) -> String {
        let mut hash = Sha256::new();
        hash.update(b"obsidian-mcp/index-scope/v1\0");
        hash.update(crate::config::vault_storage_key(canonical_root));
        for rule in &self.patterns {
            hash.update((rule.len() as u64).to_be_bytes());
            hash.update(rule.as_bytes());
        }
        crate::config::digest_hex(hash)
    }

    /// Check whether a vault-relative path is excluded.
    pub fn is_excluded(&self, relative_path: &Path) -> bool {
        if self.is_empty() {
            return false;
        }
        self.set.is_match(relative_path)
    }

    /// True when no patterns are configured.
    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }

    /// Active (accepted, normalized) patterns for diagnostics.
    pub fn patterns(&self) -> &[String] {
        &self.patterns
    }
}

/// Parse the content of an ignore file into raw pattern strings.
///
/// Lines starting with `#` (after trimming leading whitespace) are comments.
/// Blank lines and surrounding whitespace are stripped.
pub fn parse_ignore_lines(content: &str) -> Vec<String> {
    ignore_rules(content)
        .map(|(_, rule)| rule.to_owned())
        .collect()
}

fn ignore_rules(content: &str) -> impl Iterator<Item = (usize, &str)> {
    content
        .lines()
        .enumerate()
        .filter_map(|(i, line)| ignore_rule(line).map(|rule| (i + 1, rule)))
}

pub(super) fn ignore_rule(line: &str) -> Option<&str> {
    let rule = line.trim();
    (!rule.is_empty() && !rule.starts_with('#')).then_some(rule)
}

/// Read one optional policy through its originating filesystem capability.
pub(crate) fn read_ignore(root: &VaultRoot, path: &Path) -> VaultResult<Option<ExcludeSet>> {
    let content = match root.read_file(path) {
        Ok(content) => content,
        Err(err) if err.is_not_found() => return Ok(None),
        Err(err) => return Err(err),
    };
    let label = root.path().join(path).display().to_string();
    ExcludeSet::from_sources(
        ignore_rules(&content).map(|(line, rule)| (label.as_str(), line, rule.to_owned())),
    )
    .map(Some)
}

pub(crate) fn daemon_exclusions(root: &VaultRoot) -> VaultResult<ExcludeSet> {
    let machine = crate::config::normalize_optional_path_env("OBSIDIAN_MCP_DATA");
    let extra = crate::config::exclusion_env().map_err(VaultError::InvalidPath)?;
    super::data::effective_policy(root, machine.as_deref(), &extra)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── ExcludeSet::build — normalization ──

    #[test]
    fn build_normalizes_trailing_slash() {
        let set = ExcludeSet::build(vec!["Archive/".into()]).unwrap();
        assert_eq!(set.patterns(), &["Archive/**"]);
    }

    #[test]
    fn build_preserves_explicit_double_star() {
        let set = ExcludeSet::build(vec!["Archive/**".into()]).unwrap();
        assert_eq!(set.patterns(), &["Archive/**"]);
    }

    #[test]
    fn build_normalizes_nested_trailing_slash() {
        let set = ExcludeSet::build(vec!["**/drafts/".into()]).unwrap();
        assert_eq!(set.patterns(), &["**/drafts/**"]);
    }

    #[test]
    fn build_no_normalization_without_trailing_slash() {
        let set = ExcludeSet::build(vec!["*.tmp".into()]).unwrap();
        assert_eq!(set.patterns(), &["*.tmp"]);
    }

    #[test]
    fn build_rejects_an_invalid_rule_without_exposing_it() {
        let err = ExcludeSet::build(vec!["[invalid".into(), "valid/**".into()])
            .err()
            .unwrap();
        assert!(err.to_string().contains("line 1"));
        assert!(!err.to_string().contains("[invalid"));
    }

    #[test]
    fn build_empty_input() {
        let set = ExcludeSet::build(vec![]).unwrap();
        assert!(set.is_empty());
        assert!(set.patterns().is_empty());
    }

    #[test]
    fn build_all_invalid() {
        assert!(ExcludeSet::build(vec!["[bad1".into(), "[bad2".into()]).is_err());
    }

    #[test]
    fn build_trims_whitespace() {
        let set = ExcludeSet::build(vec!["  Archive/  ".into()]).unwrap();
        assert_eq!(set.patterns(), &["Archive/**"]);
    }

    #[test]
    fn build_skips_blank_entries() {
        let set = ExcludeSet::build(vec!["".into(), "  ".into(), "Archive/".into()]).unwrap();
        assert_eq!(set.patterns().len(), 1);
    }

    // ── ExcludeSet::is_excluded ──

    #[test]
    fn is_excluded_matches_file_in_excluded_dir() {
        let set = ExcludeSet::build(vec!["Archive/".into()]).unwrap();
        assert!(set.is_excluded(Path::new("Archive/old-note.md")));
    }

    #[test]
    fn is_excluded_matches_deeply_nested() {
        let set = ExcludeSet::build(vec!["Archive/".into()]).unwrap();
        assert!(set.is_excluded(Path::new("Archive/sub/deep.md")));
    }

    #[test]
    fn is_excluded_rejects_non_matching() {
        let set = ExcludeSet::build(vec!["Archive/".into()]).unwrap();
        assert!(!set.is_excluded(Path::new("Active/note.md")));
    }

    #[test]
    fn is_excluded_rejects_similar_name() {
        let set = ExcludeSet::build(vec!["Archive/".into()]).unwrap();
        assert!(!set.is_excluded(Path::new("Archived-note.md")));
    }

    #[test]
    fn is_excluded_empty_set_always_false() {
        let set = ExcludeSet::build(vec![]).unwrap();
        assert!(!set.is_excluded(Path::new("anything.md")));
    }

    #[test]
    fn is_excluded_wildcard_pattern() {
        let set = ExcludeSet::build(vec!["*.tmp".into()]).unwrap();
        assert!(set.is_excluded(Path::new("scratch.tmp")));
        assert!(!set.is_excluded(Path::new("note.md")));
    }

    #[test]
    fn is_excluded_double_star_pattern() {
        let set = ExcludeSet::build(vec!["**/drafts/".into()]).unwrap();
        assert!(set.is_excluded(Path::new("a/b/drafts/note.md")));
        assert!(set.is_excluded(Path::new("drafts/note.md")));
    }

    #[test]
    fn is_excluded_nested_dir_pattern() {
        let set = ExcludeSet::build(vec!["Resources/Meetings/".into()]).unwrap();
        assert!(set.is_excluded(Path::new("Resources/Meetings/2024-01.md")));
        assert!(!set.is_excluded(Path::new("Resources/Notes/note.md")));
    }

    // ── parse_ignore_lines ──

    #[test]
    fn parse_ignore_lines_strips_comments() {
        let result = parse_ignore_lines("# comment\nArchive/\n# another\n*.tmp");
        assert_eq!(result, vec!["Archive/", "*.tmp"]);
    }

    #[test]
    fn parse_ignore_lines_strips_blank_lines() {
        let result = parse_ignore_lines("Archive/\n\n\n*.tmp");
        assert_eq!(result, vec!["Archive/", "*.tmp"]);
    }

    #[test]
    fn parse_ignore_lines_trims_whitespace() {
        let result = parse_ignore_lines("  Archive/  \n  *.tmp  ");
        assert_eq!(result, vec!["Archive/", "*.tmp"]);
    }

    #[test]
    fn parse_ignore_lines_hash_mid_line_not_comment() {
        let result = parse_ignore_lines("path#with#hashes");
        assert_eq!(result, vec!["path#with#hashes"]);
    }

    #[test]
    fn parse_ignore_lines_indented_comment() {
        let result = parse_ignore_lines("  # indented comment\nArchive/");
        assert_eq!(result, vec!["Archive/"]);
    }

    #[test]
    fn parse_ignore_lines_empty_input() {
        let result = parse_ignore_lines("");
        assert!(result.is_empty());
    }

    #[test]
    fn parse_ignore_lines_only_comments_and_blanks() {
        let result = parse_ignore_lines("# comment\n\n# another\n  ");
        assert!(result.is_empty());
    }

    #[test]
    fn parse_ignore_lines_mixed_content() {
        let input = "\
# Exclusion patterns for obsidian-mcp
# Last updated: 2026-05-29

Archive/
Resources/Meetings/

# Drafts at any depth
**/drafts/*.tmp
";
        let result = parse_ignore_lines(input);
        assert_eq!(
            result,
            vec!["Archive/", "Resources/Meetings/", "**/drafts/*.tmp"]
        );
    }
}
