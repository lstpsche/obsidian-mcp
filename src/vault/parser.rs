//! Markdown parsing: heading extraction, code-block detection, structure analysis.

use std::ops::Range;
use std::sync::LazyLock;

use pulldown_cmark::{Event, HeadingLevel, Options, Parser, Tag, TagEnd};
use regex::Regex;

use crate::models::{DocumentMap, Heading, WikiLink};
use crate::vault::frontmatter;

// ---------------------------------------------------------------------------
// Parser options
// ---------------------------------------------------------------------------

fn parser_options() -> Options {
    Options::ENABLE_HEADING_ATTRIBUTES | Options::ENABLE_YAML_STYLE_METADATA_BLOCKS
}

// ---------------------------------------------------------------------------
// Static regex patterns
// ---------------------------------------------------------------------------

static WIKILINK_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\[\[([^\[\]]+)\]\]").unwrap());

/// Tag after whitespace or line start; first char must be a letter.
static INLINE_TAG_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)(?:^|[ \t])#([a-zA-Z][\w/-]*)").unwrap());

static BLOCK_REF_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)\^([a-zA-Z0-9-]+)\s*$").unwrap());

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Heading with byte offset preserved for range-finding operations.
struct HeadingPos {
    level: u8,
    text: String,
    line: usize,
    offset: usize,
}

fn heading_level_to_u8(level: HeadingLevel) -> u8 {
    match level {
        HeadingLevel::H1 => 1,
        HeadingLevel::H2 => 2,
        HeadingLevel::H3 => 3,
        HeadingLevel::H4 => 4,
        HeadingLevel::H5 => 5,
        HeadingLevel::H6 => 6,
    }
}

fn byte_offset_to_line(content: &str, offset: usize) -> usize {
    content[..offset].bytes().filter(|&b| b == b'\n').count()
}

/// Byte ranges of fenced/indented code blocks, inline code spans, and YAML
/// metadata blocks. Used to filter out regex matches from non-prose regions.
fn code_block_ranges(content: &str) -> Vec<Range<usize>> {
    let parser = Parser::new_ext(content, parser_options());
    let mut ranges = Vec::new();
    let mut block_start: Option<usize> = None;

    for (event, range) in parser.into_offset_iter() {
        match event {
            Event::Start(Tag::CodeBlock(_)) | Event::Start(Tag::MetadataBlock(_)) => {
                block_start = Some(range.start);
            }
            Event::End(TagEnd::CodeBlock) | Event::End(TagEnd::MetadataBlock(_)) => {
                if let Some(start) = block_start.take() {
                    ranges.push(start..range.end);
                }
            }
            Event::Code(_) => {
                ranges.push(range);
            }
            _ => {}
        }
    }

    ranges
}

/// Binary-search sorted, non-overlapping `ranges` to check if `offset` is
/// inside any of them.
fn is_in_code_block(offset: usize, ranges: &[Range<usize>]) -> bool {
    ranges
        .binary_search_by(|r| {
            if offset < r.start {
                std::cmp::Ordering::Greater
            } else if offset >= r.end {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_ok()
}

/// Parse the inner text of a `[[…]]` wikilink into a [`WikiLink`].
fn parse_wikilink_inner(raw: &str, inner: &str, line: usize) -> WikiLink {
    let (link_part, alias) = match inner.split_once('|') {
        Some((l, a)) if !a.is_empty() => (l, Some(a.to_string())),
        Some((l, _)) => (l, None),
        None => (inner, None),
    };

    let (target, fragment) = match link_part.split_once('#') {
        Some((t, f)) => (t.to_string(), Some(f)),
        None => (link_part.to_string(), None),
    };

    let (heading, block_ref) = match fragment {
        Some(f) if f.starts_with('^') => (None, Some(f[1..].to_string())),
        Some(f) => (Some(f.to_string()), None),
        None => (None, None),
    };

    WikiLink {
        raw: raw.to_string(),
        target,
        heading,
        block_ref,
        alias,
        line,
    }
}

/// Shared implementation that returns headings with both line numbers and byte
/// offsets.
fn extract_headings_with_offsets(content: &str) -> Vec<HeadingPos> {
    let parser = Parser::new_ext(content, parser_options());
    let mut headings = Vec::new();
    let mut in_heading = false;
    let mut current_level = 0u8;
    let mut current_text = String::new();
    let mut current_offset = 0usize;

    for (event, range) in parser.into_offset_iter() {
        match event {
            Event::Start(Tag::Heading { level, .. }) => {
                in_heading = true;
                current_level = heading_level_to_u8(level);
                current_text.clear();
                current_offset = range.start;
            }
            Event::Text(ref text) | Event::Code(ref text) if in_heading => {
                current_text.push_str(text);
            }
            Event::SoftBreak if in_heading => {
                current_text.push(' ');
            }
            Event::End(TagEnd::Heading(_)) if in_heading => {
                headings.push(HeadingPos {
                    level: current_level,
                    text: current_text.clone(),
                    line: byte_offset_to_line(content, current_offset),
                    offset: current_offset,
                });
                in_heading = false;
            }
            _ => {}
        }
    }

    headings
}

/// Extract top-level YAML frontmatter keys, delegating to
/// [`frontmatter::extract_raw_frontmatter`] for the actual delimiter parsing.
fn extract_frontmatter_keys(content: &str) -> Vec<String> {
    let (yaml_str, _) = match frontmatter::extract_raw_frontmatter(content) {
        Some(v) => v,
        None => return Vec::new(),
    };

    if yaml_str.trim().is_empty() {
        return Vec::new();
    }

    let value: serde_yaml::Value = match serde_yaml::from_str(yaml_str) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };

    match value {
        serde_yaml::Value::Mapping(map) => map
            .keys()
            .filter_map(|k| k.as_str().map(String::from))
            .collect(),
        _ => Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Extract all headings from markdown content.
pub fn extract_headings(content: &str) -> Vec<Heading> {
    extract_headings_with_offsets(content)
        .into_iter()
        .map(|h| Heading {
            level: h.level,
            text: h.text,
            line: h.line,
        })
        .collect()
}

/// Extract all inline tags (`#tag`, `#nested/tag`) from content.
///
/// Does **not** include frontmatter tags. Skips tags inside code blocks and
/// inline code spans. Returns sorted unique tags.
pub fn extract_inline_tags(content: &str) -> Vec<String> {
    let exclusions = code_block_ranges(content);
    let mut tags: Vec<String> = INLINE_TAG_RE
        .captures_iter(content)
        .filter_map(|cap| {
            let m = cap.get(1)?;
            if is_in_code_block(m.start(), &exclusions) {
                return None;
            }
            Some(m.as_str().to_string())
        })
        .collect();

    tags.sort();
    tags.dedup();
    tags
}

/// Extract all wikilinks from content.
///
/// Handles `[[note]]`, `[[note|alias]]`, `[[note#heading]]`,
/// `[[note#^blockid]]`, `[[#heading]]` (self-reference), etc.
/// Skips wikilinks inside code blocks.
pub fn extract_wikilinks(content: &str) -> Vec<WikiLink> {
    let exclusions = code_block_ranges(content);
    WIKILINK_RE
        .captures_iter(content)
        .filter_map(|cap| {
            let full = cap.get(0)?;
            if is_in_code_block(full.start(), &exclusions) {
                return None;
            }
            let inner = cap.get(1)?.as_str();
            let line = byte_offset_to_line(content, full.start());
            Some(parse_wikilink_inner(full.as_str(), inner, line))
        })
        .collect()
}

/// Extract all block reference IDs (`^blockid` at end of line or standalone).
/// Skips block refs inside code blocks and inline code spans.
pub fn extract_block_refs(content: &str) -> Vec<String> {
    let exclusions = code_block_ranges(content);
    BLOCK_REF_RE
        .captures_iter(content)
        .filter_map(|cap| {
            let m = cap.get(1)?;
            if is_in_code_block(m.start(), &exclusions) {
                return None;
            }
            Some(m.as_str().to_string())
        })
        .collect()
}

/// Build a [`DocumentMap`] for a note (headings, block refs, frontmatter fields).
pub fn build_document_map(content: &str) -> DocumentMap {
    let headings = extract_headings(content)
        .into_iter()
        .map(|h| format!("{} {}", "#".repeat(h.level as usize), h.text))
        .collect();

    let block_refs = extract_block_refs(content);
    let frontmatter_fields = extract_frontmatter_keys(content);

    DocumentMap {
        headings,
        block_refs,
        frontmatter_fields,
    }
}

/// Find the byte range of a heading section.
///
/// `heading_path` supports nested headings separated by `delimiter`,
/// e.g. `"Introduction::Background"` with delimiter `"::"`.
///
/// The range spans from the heading line to just before the next heading of the
/// same or higher level (or EOF / end of the parent section).
pub fn find_heading_range(
    content: &str,
    heading_path: &str,
    delimiter: &str,
) -> Option<(usize, usize)> {
    let headings = extract_headings_with_offsets(content);
    let segments: Vec<&str> = heading_path.split(delimiter).map(str::trim).collect();

    let mut search_start = 0usize;
    let mut search_end = content.len();

    for segment in &segments {
        let idx = headings.iter().position(|h| {
            h.offset >= search_start
                && h.offset < search_end
                && h.text.eq_ignore_ascii_case(segment)
        })?;

        let level = headings[idx].level;
        search_start = headings[idx].offset;

        search_end = headings[idx + 1..]
            .iter()
            .find(|h| h.offset < search_end && h.level <= level)
            .map(|h| h.offset)
            .unwrap_or(search_end);
    }

    Some((search_start, search_end))
}

/// Find the byte range of the line containing a block reference (`^blockid`).
pub fn find_block_ref_range(content: &str, block_id: &str) -> Option<(usize, usize)> {
    let pattern = format!(r"(?m)^.*\^{}\s*$", regex::escape(block_id));
    let re = Regex::new(&pattern).ok()?;
    let m = re.find(content)?;
    Some((m.start(), m.end()))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── extract_headings ─────────────────────────────────────────────

    #[test]
    fn headings_basic() {
        let content = "# Title\n\nSome text.\n\n## Section\n\nMore text.\n\n### Sub\n";
        let h = extract_headings(content);

        assert_eq!(h.len(), 3);
        assert_eq!(h[0].level, 1);
        assert_eq!(h[0].text, "Title");
        assert_eq!(h[0].line, 0);
        assert_eq!(h[1].level, 2);
        assert_eq!(h[1].text, "Section");
        assert_eq!(h[2].level, 3);
        assert_eq!(h[2].text, "Sub");
    }

    #[test]
    fn headings_with_inline_code() {
        let content = "# The `foo` method\n";
        let h = extract_headings(content);

        assert_eq!(h.len(), 1);
        assert_eq!(h[0].text, "The foo method");
    }

    #[test]
    fn headings_with_frontmatter() {
        let content = "---\ntitle: Hello\n---\n\n# Real Heading\n";
        let h = extract_headings(content);

        assert_eq!(h.len(), 1);
        assert_eq!(h[0].text, "Real Heading");
    }

    #[test]
    fn headings_empty_content() {
        assert!(extract_headings("").is_empty());
        assert!(extract_headings("No headings here.").is_empty());
    }

    // ── extract_wikilinks ────────────────────────────────────────────

    #[test]
    fn wikilink_simple() {
        let content = "See [[note]] for details.";
        let links = extract_wikilinks(content);

        assert_eq!(links.len(), 1);
        assert_eq!(links[0].raw, "[[note]]");
        assert_eq!(links[0].target, "note");
        assert!(links[0].heading.is_none());
        assert!(links[0].block_ref.is_none());
        assert!(links[0].alias.is_none());
    }

    #[test]
    fn wikilink_with_alias() {
        let links = extract_wikilinks("Check [[note|display text]].");
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].target, "note");
        assert_eq!(links[0].alias.as_deref(), Some("display text"));
    }

    #[test]
    fn wikilink_with_heading() {
        let links = extract_wikilinks("Go to [[note#section]].");
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].target, "note");
        assert_eq!(links[0].heading.as_deref(), Some("section"));
    }

    #[test]
    fn wikilink_with_block_ref() {
        let links = extract_wikilinks("See [[note#^abc123]].");
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].target, "note");
        assert_eq!(links[0].block_ref.as_deref(), Some("abc123"));
        assert!(links[0].heading.is_none());
    }

    #[test]
    fn wikilink_self_reference() {
        let links = extract_wikilinks("Jump to [[#heading]].");
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].target, "");
        assert_eq!(links[0].heading.as_deref(), Some("heading"));
    }

    #[test]
    fn wikilink_with_path() {
        let links = extract_wikilinks("[[folder/subfolder/note]]");
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].target, "folder/subfolder/note");
    }

    #[test]
    fn wikilink_heading_and_alias() {
        let links = extract_wikilinks("[[note#heading|alias]]");
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].target, "note");
        assert_eq!(links[0].heading.as_deref(), Some("heading"));
        assert_eq!(links[0].alias.as_deref(), Some("alias"));
    }

    #[test]
    fn wikilink_in_code_block_excluded() {
        let content = "Real [[link]]\n\n```\n[[not-a-link]]\n```\n";
        let links = extract_wikilinks(content);

        assert_eq!(links.len(), 1);
        assert_eq!(links[0].target, "link");
    }

    #[test]
    fn wikilink_in_inline_code_excluded() {
        let content = "Real [[link]] and `[[not-a-link]]` here.";
        let links = extract_wikilinks(content);

        assert_eq!(links.len(), 1);
        assert_eq!(links[0].target, "link");
    }

    #[test]
    fn wikilinks_line_numbers() {
        let content = "[[a]]\ntext\n[[b]]\n";
        let links = extract_wikilinks(content);

        assert_eq!(links.len(), 2);
        assert_eq!(links[0].line, 0);
        assert_eq!(links[1].line, 2);
    }

    // ── extract_inline_tags ──────────────────────────────────────────

    #[test]
    fn tags_basic() {
        let content = "Some #rust and #mcp tags.";
        let tags = extract_inline_tags(content);

        assert_eq!(tags, vec!["mcp", "rust"]);
    }

    #[test]
    fn tags_nested() {
        let content = "Tagged #lang/rust here.";
        let tags = extract_inline_tags(content);

        assert_eq!(tags, vec!["lang/rust"]);
    }

    #[test]
    fn tags_at_line_start() {
        let content = "#tag at start\n#another at start";
        let tags = extract_inline_tags(content);

        assert_eq!(tags, vec!["another", "tag"]);
    }

    #[test]
    fn tags_skip_code_block() {
        let content = "#real\n\n```\n#not-a-tag\n```\n\n#also-real\n";
        let tags = extract_inline_tags(content);

        assert_eq!(tags, vec!["also-real", "real"]);
    }

    #[test]
    fn tags_skip_inline_code() {
        let content = "#real and `#fake` here.";
        let tags = extract_inline_tags(content);

        assert_eq!(tags, vec!["real"]);
    }

    #[test]
    fn tags_numeric_rejected() {
        let content = "Issue #123 is not a tag.";
        let tags = extract_inline_tags(content);

        assert!(tags.is_empty());
    }

    #[test]
    fn tags_deduplicated() {
        let content = "#rust is great. #rust again.";
        let tags = extract_inline_tags(content);

        assert_eq!(tags, vec!["rust"]);
    }

    #[test]
    fn tags_not_from_heading_markers() {
        let content = "# Heading\n\n## Another Heading\n";
        let tags = extract_inline_tags(content);

        assert!(tags.is_empty());
    }

    #[test]
    fn tags_skip_frontmatter() {
        let content = "---\ntags: [rust]\n---\n\n#visible\n";
        let tags = extract_inline_tags(content);

        assert_eq!(tags, vec!["visible"]);
    }

    // ── extract_block_refs ───────────────────────────────────────────

    #[test]
    fn block_refs_basic() {
        let content = "Some text ^abc123\nMore text ^def456\n";
        let refs = extract_block_refs(content);

        assert_eq!(refs, vec!["abc123", "def456"]);
    }

    #[test]
    fn block_refs_with_trailing_whitespace() {
        let content = "Text ^myref   \n";
        let refs = extract_block_refs(content);

        assert_eq!(refs, vec!["myref"]);
    }

    #[test]
    fn block_refs_standalone() {
        let content = "^standalone\n";
        let refs = extract_block_refs(content);

        assert_eq!(refs, vec!["standalone"]);
    }

    #[test]
    fn block_refs_with_dashes() {
        let content = "Line ^my-block-ref\n";
        let refs = extract_block_refs(content);

        assert_eq!(refs, vec!["my-block-ref"]);
    }

    #[test]
    fn block_refs_empty_content() {
        assert!(extract_block_refs("").is_empty());
        assert!(extract_block_refs("no refs here").is_empty());
    }

    // ── find_heading_range ──────────────────────────────────────────

    #[test]
    fn heading_range_simple() {
        let content = "# Intro\n\nText here.\n\n# Next\n\nMore.\n";
        let range = find_heading_range(content, "Intro", "::");

        assert!(range.is_some());
        let (start, end) = range.unwrap();
        let section = &content[start..end];
        assert!(section.starts_with("# Intro"));
        assert!(!section.contains("# Next"));
    }

    #[test]
    fn heading_range_nested() {
        let content = "\
# A
## B
Some B content.
## C
Some C content.
# D
";
        let range = find_heading_range(content, "A::B", "::");

        assert!(range.is_some());
        let (start, end) = range.unwrap();
        let section = &content[start..end];
        assert!(section.starts_with("## B"));
        assert!(section.contains("Some B content."));
        assert!(!section.contains("## C"));
    }

    #[test]
    fn heading_range_to_eof() {
        let content = "# Only\n\nContent until end.";
        let range = find_heading_range(content, "Only", "::");

        assert!(range.is_some());
        let (start, end) = range.unwrap();
        assert_eq!(start, 0);
        assert_eq!(end, content.len());
    }

    #[test]
    fn heading_range_not_found() {
        let content = "# Exists\n\nContent.";
        assert!(find_heading_range(content, "Missing", "::").is_none());
    }

    #[test]
    fn heading_range_custom_delimiter() {
        let content = "# A\n## B\nContent.\n# C\n";
        let range = find_heading_range(content, "A/B", "/");

        assert!(range.is_some());
        let (start, end) = range.unwrap();
        let section = &content[start..end];
        assert!(section.starts_with("## B"));
    }

    // ── find_block_ref_range ────────────────────────────────────────

    #[test]
    fn block_ref_range_found() {
        let content = "First line\nImportant text ^myref\nThird line\n";
        let range = find_block_ref_range(content, "myref");

        assert!(range.is_some());
        let (start, end) = range.unwrap();
        let line = &content[start..end];
        assert!(line.contains("Important text"));
        assert!(line.contains("^myref"));
    }

    #[test]
    fn block_ref_range_not_found() {
        let content = "No block refs here.\n";
        assert!(find_block_ref_range(content, "missing").is_none());
    }

    #[test]
    fn block_ref_range_standalone() {
        let content = "Line above\n^solo\nLine below\n";
        let range = find_block_ref_range(content, "solo");

        assert!(range.is_some());
        let (start, end) = range.unwrap();
        assert_eq!(&content[start..end], "^solo");
    }

    // ── build_document_map ──────────────────────────────────────────

    #[test]
    fn document_map_full() {
        let content = "\
---
tags: [rust]
date: 2026-03-19
---

# Introduction

Some text ^ref1

## Details

More text ^ref2
";
        let map = build_document_map(content);

        assert_eq!(map.headings, vec!["# Introduction", "## Details"]);
        assert_eq!(map.block_refs, vec!["ref1", "ref2"]);
        assert!(map.frontmatter_fields.contains(&"tags".to_string()));
        assert!(map.frontmatter_fields.contains(&"date".to_string()));
    }

    #[test]
    fn document_map_no_frontmatter() {
        let content = "# Just a heading\n\nBody text.\n";
        let map = build_document_map(content);

        assert_eq!(map.headings, vec!["# Just a heading"]);
        assert!(map.block_refs.is_empty());
        assert!(map.frontmatter_fields.is_empty());
    }

    #[test]
    fn document_map_empty() {
        let map = build_document_map("");

        assert!(map.headings.is_empty());
        assert!(map.block_refs.is_empty());
        assert!(map.frontmatter_fields.is_empty());
    }

    // ── edge cases ──────────────────────────────────────────────────

    #[test]
    fn empty_content_all_extractors() {
        let content = "";
        assert!(extract_headings(content).is_empty());
        assert!(extract_inline_tags(content).is_empty());
        assert!(extract_wikilinks(content).is_empty());
        assert!(extract_block_refs(content).is_empty());
    }

    #[test]
    fn code_block_only_content() {
        let content = "```\n# not heading\n[[not-link]]\n#not-tag\n^not-ref\n```\n";
        assert!(extract_headings(content).is_empty());
        assert!(extract_wikilinks(content).is_empty());
        assert!(extract_inline_tags(content).is_empty());
        assert!(extract_block_refs(content).is_empty());
    }

    #[test]
    fn block_refs_skip_code_block() {
        let content = "Real text ^real-ref\n\n```\n^fake-ref\n```\n\nAnother ^also-real\n";
        let refs = extract_block_refs(content);
        assert_eq!(refs, vec!["real-ref", "also-real"]);
    }

    #[test]
    fn heading_range_case_insensitive() {
        let content = "# My Title\n\nContent here.\n\n# Other\n";
        let range = find_heading_range(content, "my title", "::");
        assert!(range.is_some());
        let (start, _) = range.unwrap();
        assert_eq!(start, 0);
    }
}
