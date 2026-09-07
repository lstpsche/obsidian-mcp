//! Content patching: insert, replace, and append operations on note bodies.

use std::path::Path;

use serde_yaml::Value as YamlValue;

use super::parser::{HeadingPos, extract_headings_with_offsets};
use crate::error::{VaultError, VaultResult};
use crate::models::{PatchOperation, PatchRequest, PatchTargetType};

const HEADING_DELIMITER: &str = "::";

/// Apply a patch operation to note content. Returns the modified content.
///
/// `path` is used only for error context — all operations are pure string transforms.
pub fn apply_patch(content: &str, request: &PatchRequest, path: &Path) -> VaultResult<String> {
    match request.target_type {
        PatchTargetType::Heading => patch_heading(content, request, path),
        PatchTargetType::Block => patch_block(content, request, path),
        PatchTargetType::Frontmatter => patch_frontmatter(content, request, path),
    }
}

// ============================================================
// Shared helpers
// ============================================================

/// Returns byte offset of the start of each line.
fn line_byte_offsets(content: &str) -> Vec<usize> {
    let mut offsets = vec![0usize];
    for (i, b) in content.bytes().enumerate() {
        if b == b'\n' {
            offsets.push(i + 1);
        }
    }
    offsets
}

fn line_offset(offsets: &[usize], n: usize, content_len: usize) -> usize {
    offsets.get(n).copied().unwrap_or(content_len)
}

/// Iterate lines outside fenced code blocks and frontmatter,
/// calling `f(line_index, line_str)` for each relevant line.
fn for_each_non_code_line<'a>(content: &'a str, mut f: impl FnMut(usize, &'a str)) {
    let mut in_code_block = false;
    let mut fence_char: u8 = b'`';
    let mut fence_len: usize = 0;
    let mut in_frontmatter = false;
    let mut past_frontmatter = false;

    for (idx, line) in content.lines().enumerate() {
        if !past_frontmatter {
            if idx == 0 && line.trim() == "---" {
                in_frontmatter = true;
                continue;
            }
            if in_frontmatter {
                if line.trim() == "---" {
                    past_frontmatter = true;
                }
                continue;
            }
            past_frontmatter = true;
        }

        let stripped = line.trim();

        if in_code_block {
            if !stripped.is_empty()
                && stripped.len() >= fence_len
                && stripped.bytes().all(|b| b == fence_char)
            {
                in_code_block = false;
            }
            continue;
        }

        let left = line.trim_start();
        if left.starts_with("```") || left.starts_with("~~~") {
            fence_char = left.as_bytes()[0];
            fence_len = left.bytes().take_while(|&b| b == fence_char).count();
            in_code_block = true;
            continue;
        }

        f(idx, line);
    }
}

/// Parse an ATX heading line into `(level, text)`.
fn parse_heading(line: &str) -> Option<(u8, &str)> {
    let t = line.trim_start();
    if !t.starts_with('#') {
        return None;
    }
    let level = t.bytes().take_while(|&b| b == b'#').count();
    if level > 6 {
        return None;
    }
    let rest = &t[level..];
    if !rest.is_empty() && !rest.starts_with(' ') {
        return None;
    }
    Some((level as u8, rest.trim()))
}

fn normalize_heading_target_segment(segment: &str) -> &str {
    let trimmed = segment.trim();
    let level = trimmed.bytes().take_while(|&b| b == b'#').count();

    if (1..=6).contains(&level) {
        let rest = &trimmed[level..];
        if rest.chars().next().is_some_and(char::is_whitespace) {
            return rest.trim();
        }
    }

    trimmed
}

fn ensure_trailing_newline(s: &mut String) {
    if !s.is_empty() && !s.ends_with('\n') {
        s.push('\n');
    }
}

// ============================================================
// Heading patching
// ============================================================

/// Resolve the body and section boundary using the same Markdown headings as inspection.
fn resolve_heading_range(
    content: &str,
    headings: &[HeadingPos],
    target: &str,
    path: &Path,
) -> VaultResult<(usize, usize)> {
    let mut search_start = 0;
    let mut search_end = content.len();
    let mut after_heading = 0;

    for segment in target.split(HEADING_DELIMITER) {
        let matches = |candidate: &str| {
            headings.iter().find(|heading| {
                let raw = content[heading.offset..heading.end].lines().next();
                let raw_text = raw.and_then(parse_heading).map(|(_, text)| text);
                heading.offset >= search_start
                    && heading.offset < search_end
                    && (heading.text.eq_ignore_ascii_case(candidate)
                        || raw_text.is_some_and(|text| text.eq_ignore_ascii_case(candidate)))
            })
        };
        let heading = matches(segment.trim())
            .or_else(|| matches(normalize_heading_target_segment(segment)))
            .ok_or_else(|| VaultError::PatchTargetNotFound {
                path: path.to_path_buf(),
                target_type: "heading".into(),
                target: target.into(),
            })?;
        after_heading = heading.end;
        search_start = heading.end;
        search_end = headings
            .iter()
            .find(|next| next.offset > heading.offset && next.level <= heading.level)
            .map_or(content.len(), |next| next.offset)
            .min(search_end);
    }
    Ok((after_heading, search_end))
}

fn patch_heading(content: &str, request: &PatchRequest, path: &Path) -> VaultResult<String> {
    let headings = extract_headings_with_offsets(content);
    let (after_heading, section_end_byte) =
        resolve_heading_range(content, &headings, &request.target, path)?;

    let mut result = String::with_capacity(content.len() + request.content.len() + 2);

    match request.operation {
        PatchOperation::Prepend => {
            result.push_str(&content[..after_heading]);
            ensure_trailing_newline(&mut result);
            result.push_str(&request.content);
            ensure_trailing_newline(&mut result);
            result.push_str(&content[after_heading..]);
        }
        PatchOperation::Append => {
            result.push_str(&content[..section_end_byte]);
            ensure_trailing_newline(&mut result);
            result.push_str(&request.content);
            ensure_trailing_newline(&mut result);
            result.push_str(&content[section_end_byte..]);
        }
        PatchOperation::Replace => {
            result.push_str(&content[..after_heading]);
            ensure_trailing_newline(&mut result);
            result.push_str(&request.content);
            if section_end_byte < content.len() {
                ensure_trailing_newline(&mut result);
            }
            result.push_str(&content[section_end_byte..]);
        }
    }

    Ok(result)
}

// ============================================================
// Block ref patching
// ============================================================

/// Find the first line (outside code blocks / frontmatter) ending with `^{block_id}`.
fn find_block_ref_line(content: &str, block_id: &str) -> Option<usize> {
    let suffix = format!("^{block_id}");
    let mut found = None;
    for_each_non_code_line(content, |idx, line| {
        if found.is_some() {
            return;
        }
        let trimmed = line.trim_end();
        if trimmed.ends_with(&suffix) {
            let before = &trimmed[..trimmed.len() - suffix.len()];
            if before.is_empty() || before.ends_with(' ') {
                found = Some(idx);
            }
        }
    });
    found
}

fn patch_block(content: &str, request: &PatchRequest, path: &Path) -> VaultResult<String> {
    let block_id = request.target.strip_prefix('^').unwrap_or(&request.target);
    let block_line =
        find_block_ref_line(content, block_id).ok_or_else(|| VaultError::PatchTargetNotFound {
            path: path.to_path_buf(),
            target_type: "block".into(),
            target: request.target.clone(),
        })?;

    let offsets = line_byte_offsets(content);
    let line_start = offsets[block_line];
    let line_end = line_offset(&offsets, block_line + 1, content.len());

    let mut result = String::with_capacity(content.len() + request.content.len() + 2);

    match request.operation {
        PatchOperation::Prepend => {
            result.push_str(&content[..line_start]);
            result.push_str(&request.content);
            ensure_trailing_newline(&mut result);
            result.push_str(&content[line_start..]);
        }
        PatchOperation::Append => {
            result.push_str(&content[..line_end]);
            ensure_trailing_newline(&mut result);
            result.push_str(&request.content);
            ensure_trailing_newline(&mut result);
            result.push_str(&content[line_end..]);
        }
        PatchOperation::Replace => {
            let line_text = &content[line_start..line_end];
            let trimmed = line_text.trim_end_matches(['\r', '\n']);
            let ref_marker = format!("^{block_id}");
            let ref_pos =
                trimmed
                    .rfind(&ref_marker)
                    .ok_or_else(|| VaultError::PatchTargetNotFound {
                        path: path.to_path_buf(),
                        target_type: "block".into(),
                        target: request.target.clone(),
                    })?;
            let block_ref_part = &trimmed[ref_pos..];
            let line_ending = &content[line_start + trimmed.len()..line_end];

            result.push_str(&content[..line_start]);
            if !request.content.is_empty() {
                result.push_str(&request.content);
                result.push(' ');
            }
            result.push_str(block_ref_part);
            result.push_str(line_ending);
            result.push_str(&content[line_end..]);
        }
    }

    Ok(result)
}

// ============================================================
// Frontmatter patching
// ============================================================

struct FrontmatterRange {
    yaml_start: usize,
    yaml_end: usize,
    body_start: usize,
}

/// Detect `---` delimited frontmatter at file start.
/// Returns byte offsets: yaml content lives at `content[yaml_start..yaml_end]`,
/// body resumes at `content[body_start..]`.
fn frontmatter_boundaries(content: &str) -> Option<FrontmatterRange> {
    if !content.starts_with("---") {
        return None;
    }
    let first_nl = content.find('\n')?;
    if content[..first_nl].trim_end_matches('\r').trim() != "---" {
        return None;
    }
    let yaml_start = first_nl + 1;

    let mut pos = yaml_start;
    loop {
        if pos >= content.len() {
            return None;
        }
        let next_nl = content[pos..].find('\n');
        let line_end = next_nl.map(|i| pos + i).unwrap_or(content.len());
        let line = content[pos..line_end].trim_end_matches('\r');
        if line.trim() == "---" {
            let body_start = if line_end < content.len() {
                line_end + 1
            } else {
                content.len()
            };
            return Some(FrontmatterRange {
                yaml_start,
                yaml_end: pos,
                body_start,
            });
        }
        match next_nl {
            Some(_) => pos = line_end + 1,
            None => return None,
        }
    }
}

fn parse_yaml_value(s: &str) -> YamlValue {
    serde_yaml::from_str(s).unwrap_or_else(|_| YamlValue::String(s.to_string()))
}

fn yaml_scalar_to_string(v: &YamlValue) -> String {
    match v {
        YamlValue::String(s) => s.clone(),
        YamlValue::Bool(b) => b.to_string(),
        YamlValue::Number(n) => n.to_string(),
        YamlValue::Null => String::new(),
        other => serde_yaml::to_string(other)
            .unwrap_or_default()
            .trim()
            .to_string(),
    }
}

fn patch_frontmatter(content: &str, request: &PatchRequest, path: &Path) -> VaultResult<String> {
    let range = frontmatter_boundaries(content).ok_or_else(|| VaultError::PatchTargetNotFound {
        path: path.to_path_buf(),
        target_type: "frontmatter".into(),
        target: request.target.clone(),
    })?;

    let yaml_str = &content[range.yaml_start..range.yaml_end];
    let mut mapping: serde_yaml::Mapping = if yaml_str.trim().is_empty() {
        serde_yaml::Mapping::new()
    } else {
        serde_yaml::from_str(yaml_str).map_err(|e| VaultError::FrontmatterParse {
            path: path.to_path_buf(),
            source: e,
        })?
    };

    let key = YamlValue::String(request.target.clone());

    match request.operation {
        PatchOperation::Replace => {
            mapping.insert(key, parse_yaml_value(&request.content));
        }
        PatchOperation::Append => {
            let current = mapping.remove(&key);
            let updated = match current {
                Some(YamlValue::Sequence(mut seq)) => {
                    seq.push(parse_yaml_value(&request.content));
                    YamlValue::Sequence(seq)
                }
                Some(YamlValue::String(mut s)) => {
                    s.push_str(&request.content);
                    YamlValue::String(s)
                }
                Some(other) => {
                    let mut s = yaml_scalar_to_string(&other);
                    s.push_str(&request.content);
                    YamlValue::String(s)
                }
                None => YamlValue::String(request.content.clone()),
            };
            mapping.insert(key, updated);
        }
        PatchOperation::Prepend => {
            let current = mapping.remove(&key);
            let updated = match current {
                Some(YamlValue::Sequence(mut seq)) => {
                    seq.insert(0, parse_yaml_value(&request.content));
                    YamlValue::Sequence(seq)
                }
                Some(YamlValue::String(s)) => {
                    let mut new = request.content.clone();
                    new.push_str(&s);
                    YamlValue::String(new)
                }
                Some(other) => {
                    let mut new = request.content.clone();
                    new.push_str(&yaml_scalar_to_string(&other));
                    YamlValue::String(new)
                }
                None => YamlValue::String(request.content.clone()),
            };
            mapping.insert(key, updated);
        }
    }

    let new_yaml = serde_yaml::to_string(&mapping).map_err(|e| VaultError::Other(e.to_string()))?;
    let new_yaml = new_yaml
        .strip_prefix("---\n")
        .or_else(|| new_yaml.strip_prefix("---\r\n"))
        .unwrap_or(&new_yaml);

    let body = &content[range.body_start..];

    let mut result = String::with_capacity(8 + new_yaml.len() + body.len());
    result.push_str("---\n");
    result.push_str(new_yaml);
    if !new_yaml.is_empty() && !new_yaml.ends_with('\n') {
        result.push('\n');
    }
    result.push_str("---\n");
    result.push_str(body);

    Ok(result)
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn req(op: PatchOperation, tt: PatchTargetType, target: &str, content: &str) -> PatchRequest {
        PatchRequest {
            operation: op,
            target_type: tt,
            target: target.into(),
            content: content.into(),
        }
    }

    fn p() -> &'static Path {
        Path::new("test.md")
    }

    const SAMPLE: &str = "\
---
tags:
  - rust
  - mcp
title: Test Note
---
# Introduction
This is the intro.

## Details
Some details here.
Important detail ^block1

### Sub Details
Sub detail content ^block2

## Summary
Summary text.

# Conclusion
Final thoughts.
";

    // ---- Heading: prepend ----

    #[test]
    fn heading_prepend() {
        let r = req(
            PatchOperation::Prepend,
            PatchTargetType::Heading,
            "Introduction",
            "Prepended line.",
        );
        let result = apply_patch(SAMPLE, &r, p()).unwrap();
        let lines: Vec<&str> = result.lines().collect();
        let idx = lines.iter().position(|l| *l == "# Introduction").unwrap();
        assert_eq!(lines[idx + 1], "Prepended line.");
        assert_eq!(lines[idx + 2], "This is the intro.");
    }

    // ---- Heading: append ----

    #[test]
    fn heading_append() {
        let r = req(
            PatchOperation::Append,
            PatchTargetType::Heading,
            "Details",
            "Appended to details.",
        );
        let result = apply_patch(SAMPLE, &r, p()).unwrap();
        let lines: Vec<&str> = result.lines().collect();
        let summary_idx = lines.iter().position(|l| *l == "## Summary").unwrap();
        assert_eq!(lines[summary_idx - 1], "Appended to details.");
    }

    #[test]
    fn heading_marker_prefixed_target_matches() {
        let r = req(
            PatchOperation::Append,
            PatchTargetType::Heading,
            "## Details",
            "Appended to details.",
        );
        let result = apply_patch(SAMPLE, &r, p()).unwrap();
        let lines: Vec<&str> = result.lines().collect();
        let summary_idx = lines.iter().position(|l| *l == "## Summary").unwrap();
        assert_eq!(lines[summary_idx - 1], "Appended to details.");
    }

    #[test]
    fn heading_top_level_marker_prefixed_target_matches() {
        let r = req(
            PatchOperation::Prepend,
            PatchTargetType::Heading,
            "# Introduction",
            "Prepended line.",
        );
        let result = apply_patch(SAMPLE, &r, p()).unwrap();
        let lines: Vec<&str> = result.lines().collect();
        let idx = lines.iter().position(|l| *l == "# Introduction").unwrap();
        assert_eq!(lines[idx + 1], "Prepended line.");
    }

    // ---- Heading: replace ----

    #[test]
    fn heading_replace() {
        let r = req(
            PatchOperation::Replace,
            PatchTargetType::Heading,
            "Summary",
            "New summary content.",
        );
        let result = apply_patch(SAMPLE, &r, p()).unwrap();
        let lines: Vec<&str> = result.lines().collect();
        let idx = lines.iter().position(|l| *l == "## Summary").unwrap();
        assert_eq!(lines[idx + 1], "New summary content.");
        assert!(!result.contains("Summary text."));
        assert!(result.contains("# Conclusion"));
    }

    // ---- Heading: nested path ----

    #[test]
    fn heading_nested_path() {
        let r = req(
            PatchOperation::Prepend,
            PatchTargetType::Heading,
            "Introduction::Details",
            "Nested prepend.",
        );
        let result = apply_patch(SAMPLE, &r, p()).unwrap();
        let lines: Vec<&str> = result.lines().collect();
        let idx = lines.iter().position(|l| *l == "## Details").unwrap();
        assert_eq!(lines[idx + 1], "Nested prepend.");
    }

    #[test]
    fn heading_marker_prefixed_nested_path_matches() {
        let r = req(
            PatchOperation::Prepend,
            PatchTargetType::Heading,
            "# Introduction::## Details",
            "Nested prepend.",
        );
        let result = apply_patch(SAMPLE, &r, p()).unwrap();
        let lines: Vec<&str> = result.lines().collect();
        let idx = lines.iter().position(|l| *l == "## Details").unwrap();
        assert_eq!(lines[idx + 1], "Nested prepend.");
    }

    #[test]
    fn heading_deeply_nested() {
        let r = req(
            PatchOperation::Replace,
            PatchTargetType::Heading,
            "Introduction::Details::Sub Details",
            "Replaced sub content.",
        );
        let result = apply_patch(SAMPLE, &r, p()).unwrap();
        assert!(result.contains("Replaced sub content."));
        assert!(!result.contains("Sub detail content ^block2"));
        assert!(result.contains("## Summary"));
    }

    // ---- Heading: missing ----

    #[test]
    fn heading_missing() {
        let r = req(
            PatchOperation::Append,
            PatchTargetType::Heading,
            "Nonexistent",
            "content",
        );
        let err = apply_patch(SAMPLE, &r, p()).unwrap_err();
        match err {
            VaultError::PatchTargetNotFound { target, .. } => {
                assert_eq!(target, "Nonexistent");
            }
            other => panic!("Expected PatchTargetNotFound, got {other:?}"),
        }
    }

    #[test]
    fn malformed_heading_marker_targets_do_not_match() {
        for target in ["#Details", "####### Details"] {
            let r = req(
                PatchOperation::Append,
                PatchTargetType::Heading,
                target,
                "content",
            );
            assert!(
                matches!(
                    apply_patch(SAMPLE, &r, p()),
                    Err(VaultError::PatchTargetNotFound { .. })
                ),
                "malformed target {target:?} should not match"
            );
        }
    }

    #[test]
    fn heading_hash_text_matches_raw_before_normalized_fallback() {
        let content =
            "# Hash Heading\nWrong target.\n\n# # Hash Heading\nRight target.\n\n# After\n";
        let r = req(
            PatchOperation::Append,
            PatchTargetType::Heading,
            "# Hash Heading",
            "Appended.",
        );
        let result = apply_patch(content, &r, p()).unwrap();
        let wrong_idx = result.find("Wrong target.").unwrap();
        let right_idx = result.find("Right target.").unwrap();
        let appended_idx = result.find("Appended.").unwrap();
        let after_idx = result.find("# After").unwrap();
        assert!(wrong_idx < right_idx);
        assert!(right_idx < appended_idx);
        assert!(appended_idx < after_idx);
    }

    // ---- Block: prepend ----

    #[test]
    fn block_prepend() {
        let r = req(
            PatchOperation::Prepend,
            PatchTargetType::Block,
            "block1",
            "Before block.",
        );
        let result = apply_patch(SAMPLE, &r, p()).unwrap();
        let lines: Vec<&str> = result.lines().collect();
        let idx = lines.iter().position(|l| l.contains("^block1")).unwrap();
        assert_eq!(lines[idx - 1], "Before block.");
    }

    // ---- Block: append ----

    #[test]
    fn block_append() {
        let r = req(
            PatchOperation::Append,
            PatchTargetType::Block,
            "block1",
            "After block.",
        );
        let result = apply_patch(SAMPLE, &r, p()).unwrap();
        let lines: Vec<&str> = result.lines().collect();
        let idx = lines.iter().position(|l| l.contains("^block1")).unwrap();
        assert_eq!(lines[idx + 1], "After block.");
    }

    // ---- Block: replace ----

    #[test]
    fn block_replace() {
        let r = req(
            PatchOperation::Replace,
            PatchTargetType::Block,
            "block1",
            "New block text",
        );
        let result = apply_patch(SAMPLE, &r, p()).unwrap();
        assert!(result.contains("New block text ^block1"));
        assert!(!result.contains("Important detail ^block1"));
    }

    #[test]
    fn block_with_caret_prefix() {
        let r = req(
            PatchOperation::Append,
            PatchTargetType::Block,
            "^block2",
            "After block2.",
        );
        let result = apply_patch(SAMPLE, &r, p()).unwrap();
        let lines: Vec<&str> = result.lines().collect();
        let idx = lines.iter().position(|l| l.contains("^block2")).unwrap();
        assert_eq!(lines[idx + 1], "After block2.");
    }

    #[test]
    fn block_missing() {
        let r = req(
            PatchOperation::Append,
            PatchTargetType::Block,
            "nonexistent",
            "content",
        );
        assert!(matches!(
            apply_patch(SAMPLE, &r, p()),
            Err(VaultError::PatchTargetNotFound { .. })
        ));
    }

    // ---- Block: standalone ref ----

    #[test]
    fn block_ref_standalone_replace() {
        let content = "First line\n^solo\nLast line\n";
        let r = req(
            PatchOperation::Replace,
            PatchTargetType::Block,
            "solo",
            "Replaced",
        );
        let result = apply_patch(content, &r, p()).unwrap();
        assert!(result.contains("Replaced ^solo"));
        assert!(result.contains("First line"));
        assert!(result.contains("Last line"));
    }

    // ---- Frontmatter: replace ----

    #[test]
    fn frontmatter_replace() {
        let r = req(
            PatchOperation::Replace,
            PatchTargetType::Frontmatter,
            "title",
            "New Title",
        );
        let result = apply_patch(SAMPLE, &r, p()).unwrap();
        let fm = frontmatter_boundaries(&result).unwrap();
        let yaml: serde_yaml::Value =
            serde_yaml::from_str(&result[fm.yaml_start..fm.yaml_end]).unwrap();
        assert_eq!(yaml["title"].as_str().unwrap(), "New Title");
        assert!(result.contains("# Introduction"));
    }

    // ---- Frontmatter: append list ----

    #[test]
    fn frontmatter_append_list() {
        let r = req(
            PatchOperation::Append,
            PatchTargetType::Frontmatter,
            "tags",
            "obsidian",
        );
        let result = apply_patch(SAMPLE, &r, p()).unwrap();
        let fm = frontmatter_boundaries(&result).unwrap();
        let yaml: serde_yaml::Value =
            serde_yaml::from_str(&result[fm.yaml_start..fm.yaml_end]).unwrap();
        let tags = yaml["tags"].as_sequence().unwrap();
        let texts: Vec<&str> = tags.iter().filter_map(|v| v.as_str()).collect();
        assert!(texts.contains(&"rust"));
        assert!(texts.contains(&"mcp"));
        assert!(texts.contains(&"obsidian"));
        assert_eq!(*texts.last().unwrap(), "obsidian");
    }

    // ---- Frontmatter: prepend list ----

    #[test]
    fn frontmatter_prepend_list() {
        let r = req(
            PatchOperation::Prepend,
            PatchTargetType::Frontmatter,
            "tags",
            "first",
        );
        let result = apply_patch(SAMPLE, &r, p()).unwrap();
        let fm = frontmatter_boundaries(&result).unwrap();
        let yaml: serde_yaml::Value =
            serde_yaml::from_str(&result[fm.yaml_start..fm.yaml_end]).unwrap();
        let tags = yaml["tags"].as_sequence().unwrap();
        assert_eq!(tags[0].as_str().unwrap(), "first");
    }

    // ---- Frontmatter: append string ----

    #[test]
    fn frontmatter_append_string() {
        let r = req(
            PatchOperation::Append,
            PatchTargetType::Frontmatter,
            "title",
            " (v2)",
        );
        let result = apply_patch(SAMPLE, &r, p()).unwrap();
        let fm = frontmatter_boundaries(&result).unwrap();
        let yaml: serde_yaml::Value =
            serde_yaml::from_str(&result[fm.yaml_start..fm.yaml_end]).unwrap();
        assert_eq!(yaml["title"].as_str().unwrap(), "Test Note (v2)");
    }

    // ---- Frontmatter: prepend string ----

    #[test]
    fn frontmatter_prepend_string() {
        let r = req(
            PatchOperation::Prepend,
            PatchTargetType::Frontmatter,
            "title",
            "Prefix: ",
        );
        let result = apply_patch(SAMPLE, &r, p()).unwrap();
        let fm = frontmatter_boundaries(&result).unwrap();
        let yaml: serde_yaml::Value =
            serde_yaml::from_str(&result[fm.yaml_start..fm.yaml_end]).unwrap();
        assert_eq!(yaml["title"].as_str().unwrap(), "Prefix: Test Note");
    }

    // ---- Frontmatter: new field ----

    #[test]
    fn frontmatter_new_field() {
        let r = req(
            PatchOperation::Replace,
            PatchTargetType::Frontmatter,
            "status",
            "draft",
        );
        let result = apply_patch(SAMPLE, &r, p()).unwrap();
        let fm = frontmatter_boundaries(&result).unwrap();
        let yaml: serde_yaml::Value =
            serde_yaml::from_str(&result[fm.yaml_start..fm.yaml_end]).unwrap();
        assert_eq!(yaml["status"].as_str().unwrap(), "draft");
    }

    // ---- Frontmatter: missing block ----

    #[test]
    fn frontmatter_missing_block() {
        let content = "# No frontmatter\nJust content.\n";
        let r = req(
            PatchOperation::Replace,
            PatchTargetType::Frontmatter,
            "key",
            "value",
        );
        assert!(matches!(
            apply_patch(content, &r, p()),
            Err(VaultError::PatchTargetNotFound { .. })
        ));
    }

    // ---- Edge cases ----

    #[test]
    fn heading_inside_code_block_ignored() {
        let content = "# Real Heading\nSome text.\n```\n# Not A Heading\n```\n# After Code\n";
        let headings = extract_headings_with_offsets(content);
        let texts: Vec<&str> = headings.iter().map(|h| h.text.as_str()).collect();
        assert_eq!(texts, vec!["Real Heading", "After Code"]);
    }

    #[test]
    fn block_inside_code_block_ignored() {
        let content = "Start\n```\nfake ^block9\n```\nReal ^block9\n";
        assert_eq!(find_block_ref_line(content, "block9"), Some(4));
    }

    #[test]
    fn content_preservation() {
        let r = req(
            PatchOperation::Prepend,
            PatchTargetType::Heading,
            "Conclusion",
            "Added.",
        );
        let result = apply_patch(SAMPLE, &r, p()).unwrap();
        assert!(result.contains("This is the intro."));
        assert!(result.contains("Some details here."));
        assert!(result.contains("Summary text."));
        assert!(result.contains("Important detail ^block1"));
    }

    #[test]
    fn heading_at_end_of_file() {
        let content = "# Only Heading\n";
        let r = req(
            PatchOperation::Append,
            PatchTargetType::Heading,
            "Only Heading",
            "Content added.",
        );
        let result = apply_patch(content, &r, p()).unwrap();
        assert!(result.contains("# Only Heading"));
        assert!(result.contains("Content added."));
    }

    #[test]
    fn heading_case_insensitive() {
        let content = "# My Heading\nBody text.\n";
        let r = req(
            PatchOperation::Append,
            PatchTargetType::Heading,
            "my heading",
            "Appended.",
        );
        let result = apply_patch(content, &r, p()).unwrap();
        assert!(result.contains("Appended."));
    }

    #[test]
    fn frontmatter_body_preserved_exactly() {
        let content = "---\nkey: val\n---\nBody with special chars: *bold* [[link]] #tag\n";
        let r = req(
            PatchOperation::Replace,
            PatchTargetType::Frontmatter,
            "key",
            "new",
        );
        let result = apply_patch(content, &r, p()).unwrap();
        assert!(result.contains("Body with special chars: *bold* [[link]] #tag"));
    }

    #[test]
    fn frontmatter_yaml_comment_not_heading() {
        let content = "---\n# yaml comment\nkey: val\n---\n# Real Heading\n";
        let headings = extract_headings_with_offsets(content);
        assert_eq!(headings.len(), 1);
        assert_eq!(headings[0].text, "Real Heading");
    }

    #[test]
    fn heading_replace_preserves_setext_siblings_and_their_underlines() {
        let content = "# Parent\nold\n\nSibling\n=======\nkeep\n";
        let result = apply_patch(
            content,
            &req(
                PatchOperation::Replace,
                PatchTargetType::Heading,
                "Parent",
                "NEW",
            ),
            p(),
        )
        .unwrap();
        assert_eq!(result, "# Parent\nNEW\nSibling\n=======\nkeep\n");
        let result = apply_patch(
            content,
            &req(
                PatchOperation::Replace,
                PatchTargetType::Heading,
                "Sibling",
                "NEW",
            ),
            p(),
        )
        .unwrap();
        assert_eq!(result, "# Parent\nold\n\nSibling\n=======\nNEW");
    }

    #[test]
    fn heading_targets_round_trip_from_inspection_and_ignore_indented_code() {
        let content = "## `code` heading\nold\n\n    # Fake\n    code\n\n## Next\nkeep\n";
        let target = &super::super::parser::build_document_map(content).headings[0];
        for target in [target.as_str(), "`code` heading"] {
            let result = apply_patch(
                content,
                &req(
                    PatchOperation::Replace,
                    PatchTargetType::Heading,
                    target,
                    "NEW",
                ),
                p(),
            )
            .unwrap();
            assert_eq!(result, "## `code` heading\nNEW\n## Next\nkeep\n");
        }
    }

    #[test]
    fn block_append_at_eof_preserves_reference() {
        let result = apply_patch(
            "text ^id",
            &req(PatchOperation::Append, PatchTargetType::Block, "id", "NEW"),
            p(),
        )
        .unwrap();
        assert_eq!(result, "text ^id\nNEW\n");
        assert_eq!(find_block_ref_line(&result, "id"), Some(0));
    }
}
