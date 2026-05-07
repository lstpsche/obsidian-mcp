//! Shared search utility functions used by both MCP tools and the semantic daemon.

use std::path::PathBuf;

/// Strip YAML frontmatter and return a character-limited body preview.
pub(crate) fn body_preview(content: &str, max_chars: usize) -> String {
    let start = if let Some(stripped) = content.strip_prefix("---") {
        stripped
            .find("\n---")
            .map(|idx| {
                // 3 (stripped "---" prefix) + 4 ("\n---" closing delimiter)
                let end = idx + 3 + 4;
                content[end..].find('\n').map_or(end, |nl| end + nl + 1)
            })
            .unwrap_or(0)
    } else {
        0
    };
    let body = content[start..].trim_start();
    body.chars().take(max_chars).collect()
}

/// Build a case-insensitive regex matching any query word.
///
/// Returns `None` for empty queries.
pub(crate) fn compile_query_word_regex(query: &str) -> Option<regex::Regex> {
    let pattern: String = query
        .split_whitespace()
        .map(regex::escape)
        .collect::<Vec<_>>()
        .join("|");
    if pattern.is_empty() {
        return None;
    }
    regex::Regex::new(&format!("(?i){pattern}")).ok()
}

/// Min-max normalize BM25 scores to `[0, 1]`.
///
/// When all scores are identical, each normalized score is `1.0`.
/// Non-finite scores (NaN, infinity) are silently filtered out.
pub(crate) fn normalize_bm25_scores(hits: &[(PathBuf, f32)]) -> Vec<(PathBuf, f32)> {
    if hits.is_empty() {
        return Vec::new();
    }

    let finite_hits: Vec<_> = hits.iter().filter(|(_, s)| s.is_finite()).collect();
    if finite_hits.is_empty() {
        return Vec::new();
    }

    let min = finite_hits
        .iter()
        .map(|(_, score)| *score)
        .fold(f32::INFINITY, f32::min);
    let max = finite_hits
        .iter()
        .map(|(_, score)| *score)
        .fold(f32::NEG_INFINITY, f32::max);
    let range = max - min;

    finite_hits
        .iter()
        .map(|(path, score)| {
            let normalized = if range == 0.0 {
                1.0
            } else {
                (score - min) / range
            };
            ((*path).clone(), normalized)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    // ── body_preview ────────────────────────────────────────────────

    #[test]
    fn body_preview_strips_frontmatter() {
        let content = "---\ntags: [a]\n---\nHello world";
        let preview = body_preview(content, 100);
        assert_eq!(preview, "Hello world");
    }

    #[test]
    fn body_preview_no_frontmatter() {
        let content = "# Title\nSome body text";
        let preview = body_preview(content, 100);
        assert_eq!(preview, "# Title\nSome body text");
    }

    #[test]
    fn body_preview_truncates() {
        let content = "---\nk: v\n---\nABCDEFGHIJ";
        let preview = body_preview(content, 5);
        assert_eq!(preview, "ABCDE");
    }

    #[test]
    fn body_preview_empty_content() {
        let preview = body_preview("", 100);
        assert_eq!(preview, "");
    }

    #[test]
    fn body_preview_unclosed_frontmatter() {
        let content = "---\ntags: [a]\nNo closing delimiter here";
        let preview = body_preview(content, 200);
        assert!(preview.contains("tags:"));
    }

    // ── normalize_bm25_scores ───────────────────────────────────────

    #[test]
    fn normalize_empty() {
        assert!(normalize_bm25_scores(&[]).is_empty());
    }

    #[test]
    fn normalize_single_score() {
        let hits = vec![(PathBuf::from("a.md"), 5.0)];
        let norm = normalize_bm25_scores(&hits);
        assert_eq!(norm.len(), 1);
        assert!(
            (norm[0].1 - 1.0).abs() < 1e-6,
            "single item normalizes to 1.0"
        );
    }

    #[test]
    fn normalize_identical_scores() {
        let hits = vec![
            (PathBuf::from("a.md"), 3.0),
            (PathBuf::from("b.md"), 3.0),
            (PathBuf::from("c.md"), 3.0),
        ];
        let norm = normalize_bm25_scores(&hits);
        for (_, score) in &norm {
            assert!(
                (score - 1.0).abs() < 1e-6,
                "identical scores should all normalize to 1.0"
            );
        }
    }

    #[test]
    fn normalize_min_max_range() {
        let hits = vec![
            (PathBuf::from("high.md"), 10.0),
            (PathBuf::from("mid.md"), 5.0),
            (PathBuf::from("low.md"), 0.0),
        ];
        let norm = normalize_bm25_scores(&hits);

        let high = norm
            .iter()
            .find(|(p, _)| p == Path::new("high.md"))
            .unwrap()
            .1;
        let mid = norm
            .iter()
            .find(|(p, _)| p == Path::new("mid.md"))
            .unwrap()
            .1;
        let low = norm
            .iter()
            .find(|(p, _)| p == Path::new("low.md"))
            .unwrap()
            .1;

        assert!((high - 1.0).abs() < 1e-6);
        assert!((mid - 0.5).abs() < 1e-6);
        assert!(low.abs() < 1e-6);
    }

    #[test]
    fn normalize_preserves_order() {
        let hits = vec![
            (PathBuf::from("first.md"), 8.0),
            (PathBuf::from("second.md"), 4.0),
            (PathBuf::from("third.md"), 2.0),
        ];
        let norm = normalize_bm25_scores(&hits);
        assert!(norm[0].1 > norm[1].1);
        assert!(norm[1].1 > norm[2].1);
    }

    #[test]
    fn normalize_filters_non_finite_scores() {
        let hits = vec![
            (PathBuf::from("valid.md"), 10.0),
            (PathBuf::from("nan.md"), f32::NAN),
            (PathBuf::from("inf.md"), f32::INFINITY),
            (PathBuf::from("neginf.md"), f32::NEG_INFINITY),
            (PathBuf::from("low.md"), 0.0),
        ];
        let norm = normalize_bm25_scores(&hits);
        assert_eq!(norm.len(), 2);
        assert!(norm.iter().any(|(p, _)| p == Path::new("valid.md")));
        assert!(norm.iter().any(|(p, _)| p == Path::new("low.md")));
    }

    #[test]
    fn normalize_all_non_finite_returns_empty() {
        let hits = vec![
            (PathBuf::from("nan.md"), f32::NAN),
            (PathBuf::from("inf.md"), f32::INFINITY),
        ];
        assert!(normalize_bm25_scores(&hits).is_empty());
    }
}
