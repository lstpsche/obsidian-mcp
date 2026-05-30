//! Path exclusion logic: compile glob patterns and test vault-relative paths.

use std::fs;
use std::path::Path;

use globset::{Glob, GlobSet, GlobSetBuilder};

use crate::error::{VaultError, VaultResult};

/// Compiled set of glob patterns for excluding vault paths from indexing.
pub struct ExcludeSet {
    set: GlobSet,
    patterns: Vec<String>,
}

impl ExcludeSet {
    /// Compile a list of raw patterns into a `GlobSet`.
    ///
    /// Each pattern is trimmed, blank entries are skipped, and trailing `/`
    /// is normalized to `/**`. Invalid patterns are logged and skipped.
    pub fn build(patterns: Vec<String>) -> VaultResult<Self> {
        let mut builder = GlobSetBuilder::new();
        let mut accepted = Vec::new();

        for raw in &patterns {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                continue;
            }

            let normalized = if let Some(prefix) = trimmed.strip_suffix('/') {
                format!("{prefix}/**")
            } else {
                trimmed.to_string()
            };

            match Glob::new(&normalized) {
                Ok(glob) => {
                    builder.add(glob);
                    accepted.push(normalized);
                }
                Err(e) => {
                    tracing::warn!(pattern = trimmed, error = %e, "skipping invalid exclude pattern");
                }
            }
        }

        let set = builder
            .build()
            .map_err(|e| VaultError::Other(format!("glob set compile: {e}")))?;

        Ok(Self {
            set,
            patterns: accepted,
        })
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
    content
        .lines()
        .filter_map(|line| {
            let ltrimmed = line.trim_start();
            if ltrimmed.starts_with('#') {
                return None;
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                return None;
            }
            Some(trimmed.to_string())
        })
        .collect()
}

/// Load and merge ignore patterns from both config locations.
///
/// Reads `{mcp_home}/ignore` and (if different) `{mcp_data}/ignore`,
/// merges both lists, sorts, and deduplicates.
pub fn load_ignore_patterns(mcp_home: &Path, mcp_data: &Path) -> Vec<String> {
    let mut patterns = Vec::new();

    if let Ok(content) = fs::read_to_string(mcp_home.join("ignore")) {
        patterns.extend(parse_ignore_lines(&content));
    }

    if mcp_data != mcp_home
        && let Ok(content) = fs::read_to_string(mcp_data.join("ignore"))
    {
        patterns.extend(parse_ignore_lines(&content));
    }

    patterns.sort();
    patterns.dedup();
    patterns
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
    fn build_skips_invalid_pattern() {
        let set = ExcludeSet::build(vec!["[invalid".into(), "valid/**".into()]).unwrap();
        assert_eq!(set.patterns().len(), 1);
        assert_eq!(set.patterns()[0], "valid/**");
    }

    #[test]
    fn build_empty_input() {
        let set = ExcludeSet::build(vec![]).unwrap();
        assert!(set.is_empty());
        assert!(set.patterns().is_empty());
    }

    #[test]
    fn build_all_invalid() {
        let set = ExcludeSet::build(vec!["[bad1".into(), "[bad2".into()]).unwrap();
        assert!(set.is_empty());
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

    // ── load_ignore_patterns ──

    #[test]
    fn load_ignore_patterns_single_location() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join("ignore"), "Archive/\n*.tmp\n").unwrap();

        let result = load_ignore_patterns(dir.path(), dir.path());
        assert_eq!(result, vec!["*.tmp", "Archive/"]);
    }

    #[test]
    fn load_ignore_patterns_both_locations() {
        let home = tempfile::TempDir::new().unwrap();
        let data = tempfile::TempDir::new().unwrap();
        std::fs::write(home.path().join("ignore"), "Archive/\n").unwrap();
        std::fs::write(data.path().join("ignore"), "Drafts/\n").unwrap();

        let result = load_ignore_patterns(home.path(), data.path());
        assert_eq!(result, vec!["Archive/", "Drafts/"]);
    }

    #[test]
    fn load_ignore_patterns_dedup() {
        let home = tempfile::TempDir::new().unwrap();
        let data = tempfile::TempDir::new().unwrap();
        std::fs::write(home.path().join("ignore"), "Archive/\nDrafts/\n").unwrap();
        std::fs::write(data.path().join("ignore"), "Archive/\nMeetings/\n").unwrap();

        let result = load_ignore_patterns(home.path(), data.path());
        assert_eq!(result, vec!["Archive/", "Drafts/", "Meetings/"]);
    }

    #[test]
    fn load_ignore_patterns_missing_files() {
        let dir = tempfile::TempDir::new().unwrap();
        let result = load_ignore_patterns(dir.path(), dir.path());
        assert!(result.is_empty());
    }

    #[test]
    fn load_ignore_patterns_same_path_no_duplicates() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join("ignore"), "Archive/\nDrafts/\n").unwrap();

        let result = load_ignore_patterns(dir.path(), dir.path());
        assert_eq!(result, vec!["Archive/", "Drafts/"]);
    }

    #[test]
    fn load_ignore_patterns_one_missing_one_present() {
        let home = tempfile::TempDir::new().unwrap();
        let data = tempfile::TempDir::new().unwrap();
        std::fs::write(data.path().join("ignore"), "External/\n").unwrap();

        let result = load_ignore_patterns(home.path(), data.path());
        assert_eq!(result, vec!["External/"]);
    }
}
