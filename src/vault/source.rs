//! Revision-bound source observations and bounded context expansion.

use std::path::Path;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{VaultError, VaultResult};
use crate::models::SourceRange;

pub const MAX_SOURCE_BYTES: usize = 16 * 1024 * 1024;
pub const DEFAULT_PAGE_BYTES: usize = 64 * 1024;
pub const MAX_PAGE_BYTES: usize = 256 * 1024;
const DEFAULT_HEADING_COUNT: usize = 100;
const MAX_HEADING_COUNT: usize = 1_000;
const MAX_HEADING_PREVIEW_CHARS: usize = 256;

/// One owned source buffer. All references and hashes describe these exact bytes.
#[derive(Debug, Clone)]
pub struct SourceSnapshot {
    pub content: String,
    pub source_revision: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SourcePage {
    pub content: String,
    pub source_revision: String,
    pub source_bytes: usize,
    pub requested_range: SourceRange,
    pub returned_range: SourceRange,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_range: Option<SourceRange>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SourceHeading {
    pub ordinal: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_ordinal: Option<usize>,
    pub level: u8,
    pub text: String,
    pub text_truncated: bool,
    pub heading_range: SourceRange,
    pub section_range: SourceRange,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SourceHeadingsPage {
    pub source_revision: String,
    pub source_bytes: usize,
    pub total_headings: usize,
    pub headings: Vec<SourceHeading>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_heading_offset: Option<usize>,
}

pub fn source_revision(content: &str) -> String {
    Sha256::digest(content.as_bytes())
        .iter()
        .flat_map(|&byte| hex_pair(byte).map(char::from))
        .collect()
}

#[cfg(has_embeddings)]
pub(crate) fn source_revision_matches(content: &str, expected: &str) -> bool {
    expected.len() == 64
        && Sha256::digest(content.as_bytes())
            .iter()
            .flat_map(|&byte| hex_pair(byte))
            .eq(expected.bytes())
}

fn hex_pair(byte: u8) -> [u8; 2] {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    [HEX[usize::from(byte >> 4)], HEX[usize::from(byte & 15)]]
}

pub fn validate_revision(revision: &str) -> VaultResult<()> {
    if revision.len() == 64
        && revision
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err(VaultError::InvalidSourceRange(
            "expected_revision must contain exactly 64 lowercase hexadecimal digits".into(),
        ))
    }
}

pub fn read_snapshot(vault_root: &Path, path: &Path) -> VaultResult<SourceSnapshot> {
    Ok(SourceSnapshot::new(super::fs::read_file_bounded(
        vault_root,
        path,
        MAX_SOURCE_BYTES,
    )?))
}

impl SourceSnapshot {
    pub fn new(content: String) -> Self {
        let source_revision = source_revision(&content);
        Self {
            content,
            source_revision,
        }
    }

    pub fn check_revision(&self, path: &Path, expected: Option<&str>) -> VaultResult<()> {
        if let Some(expected) = expected {
            validate_revision(expected)?;
            if expected != self.source_revision {
                return Err(VaultError::SourceChanged {
                    path: path.to_owned(),
                    expected_revision: expected.to_owned(),
                    current_revision: self.source_revision.clone(),
                });
            }
        }
        Ok(())
    }

    pub fn body_range(&self) -> SourceRange {
        SourceRange {
            start_byte: super::frontmatter::extract_raw_frontmatter(&self.content)
                .map_or(0, |(_, offset)| offset),
            end_byte: self.content.len(),
        }
    }

    pub fn validate_range(&self, range: SourceRange) -> VaultResult<()> {
        if range.start_byte > range.end_byte || range.end_byte > self.content.len() {
            return Err(VaultError::InvalidSourceRange(format!(
                "{}..{} is outside a {}-byte note",
                range.start_byte,
                range.end_byte,
                self.content.len()
            )));
        }
        if !self.content.is_char_boundary(range.start_byte)
            || !self.content.is_char_boundary(range.end_byte)
        {
            return Err(VaultError::InvalidSourceRange(
                "range endpoints must be UTF-8 character boundaries".into(),
            ));
        }
        Ok(())
    }

    pub fn page(
        &self,
        path: &Path,
        expected_revision: Option<&str>,
        range: Option<SourceRange>,
        max_bytes: Option<usize>,
    ) -> VaultResult<SourcePage> {
        if range.is_some() && expected_revision.is_none() {
            return Err(VaultError::InvalidSourceRange(
                "expected_revision is required with range".into(),
            ));
        }
        self.check_revision(path, expected_revision)?;
        let requested_range = range.unwrap_or(SourceRange {
            start_byte: 0,
            end_byte: self.content.len(),
        });
        self.validate_range(requested_range)?;
        let budget = match max_bytes {
            Some(value) => Some(validate_limit(value, MAX_PAGE_BYTES, "max_bytes")?),
            None if range.is_some() => Some(DEFAULT_PAGE_BYTES),
            None => None,
        };
        let end = match budget {
            Some(budget) => self.content.floor_char_boundary(
                requested_range
                    .start_byte
                    .saturating_add(budget)
                    .min(requested_range.end_byte),
            ),
            None => requested_range.end_byte,
        };
        if end == requested_range.start_byte && end < requested_range.end_byte {
            return Err(VaultError::InvalidSourceRange(
                "max_bytes cannot fit the next UTF-8 character".into(),
            ));
        }
        let returned_range = SourceRange {
            start_byte: requested_range.start_byte,
            end_byte: end,
        };
        Ok(SourcePage {
            content: self.content[returned_range.start_byte..returned_range.end_byte].to_owned(),
            source_revision: self.source_revision.clone(),
            source_bytes: self.content.len(),
            requested_range,
            returned_range,
            next_range: (end < requested_range.end_byte).then_some(SourceRange {
                start_byte: end,
                end_byte: requested_range.end_byte,
            }),
        })
    }

    pub fn headings_page(
        &self,
        path: &Path,
        expected_revision: Option<&str>,
        heading_offset: usize,
        max_headings: Option<usize>,
        max_bytes: Option<usize>,
    ) -> VaultResult<SourceHeadingsPage> {
        if heading_offset > 0 && expected_revision.is_none() {
            return Err(VaultError::InvalidSourceRange(
                "expected_revision is required with a nonzero heading_offset".into(),
            ));
        }
        self.check_revision(path, expected_revision)?;
        let max_headings = validate_limit(
            max_headings.unwrap_or(DEFAULT_HEADING_COUNT),
            MAX_HEADING_COUNT,
            "max_headings",
        )?;
        let max_bytes = validate_limit(
            max_bytes.unwrap_or(DEFAULT_PAGE_BYTES),
            MAX_PAGE_BYTES,
            "max_bytes",
        )?;
        let headings = super::parser::extract_headings_with_offsets(&self.content);
        if heading_offset > headings.len() {
            return Err(VaultError::InvalidSourceRange(format!(
                "heading_offset {heading_offset} exceeds {} headings",
                headings.len()
            )));
        }
        let mut parents = vec![None; headings.len()];
        let mut section_ends = vec![self.content.len(); headings.len()];
        let mut stack: Vec<usize> = Vec::new();
        for (ordinal, heading) in headings.iter().enumerate() {
            while let Some(previous) = stack.last().copied() {
                if headings[previous].level < heading.level {
                    break;
                }
                section_ends[previous] = heading.offset;
                stack.pop();
            }
            parents[ordinal] = stack.last().copied();
            stack.push(ordinal);
        }
        let mut page = SourceHeadingsPage {
            source_revision: self.source_revision.clone(),
            source_bytes: self.content.len(),
            total_headings: headings.len(),
            headings: Vec::new(),
            next_heading_offset: None,
        };
        let mut records_bytes = 0;
        let last = headings
            .len()
            .min(heading_offset.saturating_add(max_headings));
        for ordinal in heading_offset..last {
            let heading = &headings[ordinal];
            let text: String = heading
                .text
                .chars()
                .take(MAX_HEADING_PREVIEW_CHARS)
                .collect();
            let record = SourceHeading {
                ordinal,
                parent_ordinal: parents[ordinal],
                level: heading.level,
                text_truncated: text.len() < heading.text.len(),
                text,
                heading_range: SourceRange {
                    start_byte: heading.offset,
                    end_byte: heading.end,
                },
                section_range: SourceRange {
                    start_byte: heading.offset,
                    end_byte: section_ends[ordinal],
                },
            };
            page.next_heading_offset = (ordinal + 1 < headings.len()).then_some(ordinal + 1);
            let record_bytes = serialized_len(&record)?;
            let envelope_bytes = serialized_len(&SourceHeadingsPage {
                source_revision: page.source_revision.clone(),
                source_bytes: page.source_bytes,
                total_headings: page.total_headings,
                headings: Vec::new(),
                next_heading_offset: page.next_heading_offset,
            })?;
            let candidate_bytes =
                envelope_bytes + records_bytes + record_bytes + page.headings.len();
            if candidate_bytes > max_bytes {
                page.next_heading_offset = Some(ordinal);
                if page.headings.is_empty() {
                    return Err(VaultError::InvalidSourceRange(
                        "max_bytes cannot fit one heading locator".into(),
                    ));
                }
                break;
            }
            records_bytes += record_bytes;
            page.headings.push(record);
        }
        if page.headings.is_empty() && serialized_len(&page)? > max_bytes {
            return Err(VaultError::InvalidSourceRange(
                "max_bytes cannot fit the heading page envelope".into(),
            ));
        }
        Ok(page)
    }
}

fn validate_limit(value: usize, cap: usize, name: &str) -> VaultResult<usize> {
    if value == 0 || value > cap {
        Err(VaultError::InvalidSourceRange(format!(
            "{name} must be between 1 and {cap}"
        )))
    } else {
        Ok(value)
    }
}

fn serialized_len(value: &impl Serialize) -> VaultResult<usize> {
    serde_json::to_vec(value)
        .map(|bytes| bytes.len())
        .map_err(|error| VaultError::Other(format!("JSON serialization failed: {error}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn pages_reconstruct_exact_raw_source_and_reject_stale_continuation() {
        let content = "\u{feff}---\r\ntag: yes\r\n---\r\n# 😀\n\0café";
        let source = SourceSnapshot::new(content.into());
        let mut range = None;
        let mut reconstructed = String::new();
        loop {
            let page = source
                .page(
                    Path::new("note.md"),
                    Some(&source.source_revision),
                    range,
                    Some(5),
                )
                .unwrap();
            reconstructed.push_str(&page.content);
            match page.next_range {
                Some(next) => range = Some(next),
                None => break,
            }
        }
        assert_eq!(reconstructed, content);
        let changed = SourceSnapshot::new(format!("{content}!"));
        assert!(matches!(
            changed.page(
                Path::new("note.md"),
                Some(&source.source_revision),
                range,
                Some(5)
            ),
            Err(VaultError::SourceChanged { .. })
        ));
    }

    #[test]
    fn ranges_require_revision_utf8_boundaries_and_forward_progress() {
        let source = SourceSnapshot::new("😀tail".into());
        let range = SourceRange {
            start_byte: 0,
            end_byte: 8,
        };
        assert!(
            source
                .page(Path::new("n.md"), None, Some(range), None)
                .is_err()
        );
        assert!(
            source
                .page(
                    Path::new("n.md"),
                    Some(&source.source_revision),
                    Some(range),
                    Some(1)
                )
                .is_err()
        );
        assert!(
            source
                .page(
                    Path::new("n.md"),
                    Some(&source.source_revision),
                    Some(SourceRange {
                        start_byte: 1,
                        end_byte: 4
                    }),
                    None
                )
                .is_err()
        );
        assert!(
            source
                .page(
                    Path::new("n.md"),
                    Some(&source.source_revision.to_uppercase()),
                    None,
                    None
                )
                .is_err()
        );
        let empty = source
            .page(
                Path::new("n.md"),
                Some(&source.source_revision),
                Some(SourceRange {
                    start_byte: 8,
                    end_byte: 8,
                }),
                Some(1),
            )
            .unwrap();
        assert!(empty.content.is_empty());
        assert!(empty.next_range.is_none());
    }

    #[test]
    fn bounded_reader_reports_overflow_and_confines_paths() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("note.md"), "abcd").unwrap();
        assert!(matches!(
            super::super::fs::read_file_bounded(dir.path(), Path::new("note.md"), 3),
            Err(VaultError::SemanticResourceLimit {
                actual: 4,
                limit: 3,
                ..
            })
        ));
        assert!(read_snapshot(dir.path(), Path::new("../outside.md")).is_err());
        assert!(matches!(
            read_snapshot(dir.path(), Path::new("missing.md")),
            Err(VaultError::NoteNotFound(_))
        ));
    }

    #[cfg(unix)]
    #[test]
    fn source_reads_reject_outside_symlinks_and_invalid_utf8() {
        let root = tempdir().unwrap();
        let outside = tempdir().unwrap();
        std::fs::write(outside.path().join("secret.md"), "outside").unwrap();
        std::os::unix::fs::symlink(
            outside.path().join("secret.md"),
            root.path().join("alias.md"),
        )
        .unwrap();
        assert!(matches!(
            read_snapshot(root.path(), Path::new("alias.md")),
            Err(VaultError::OutsideVault(_))
        ));
        std::fs::write(root.path().join("invalid.md"), [0xff, 0xfe]).unwrap();
        assert!(matches!(
            read_snapshot(root.path(), Path::new("invalid.md")),
            Err(VaultError::Io(error)) if error.kind() == std::io::ErrorKind::InvalidData
        ));
    }

    #[test]
    fn heading_pages_keep_hierarchy_setext_ranges_and_revision() {
        let text = "---\nx: true\n---\n# Parent\nintro\n## **Child**\nbody\n## Child\nnext\n\nSibling\n=======\nend\n```\n# fake\n```\n";
        let source = SourceSnapshot::new(text.into());
        let page = source
            .headings_page(Path::new("n.md"), None, 0, Some(2), None)
            .unwrap();
        assert_eq!(page.total_headings, 4);
        assert_eq!(page.headings[1].parent_ordinal, Some(0));
        assert_eq!(page.headings[1].text, "Child");
        assert_eq!(page.next_heading_offset, Some(2));
        assert_eq!(
            &text[page.headings[0].section_range.start_byte
                ..page.headings[0].section_range.end_byte],
            &text[text.find("# Parent").unwrap()..text.find("Sibling").unwrap()]
        );
        assert!(
            source
                .headings_page(Path::new("n.md"), None, 2, None, None)
                .is_err()
        );
        let tail = source
            .headings_page(
                Path::new("n.md"),
                Some(&source.source_revision),
                2,
                None,
                None,
            )
            .unwrap();
        assert_eq!(tail.headings[1].text, "Sibling");
        assert!(tail.next_heading_offset.is_none());
        assert!(
            source
                .headings_page(Path::new("n.md"), None, 0, None, Some(1))
                .is_err()
        );
    }

    #[test]
    fn giant_heading_preview_is_explicit_and_page_budget_is_exact() {
        let source = SourceSnapshot::new(format!("# {}\n## next\n", "😀".repeat(1_000)));
        let first = source
            .headings_page(Path::new("n.md"), None, 0, Some(1), None)
            .unwrap();
        let exact_budget = serialized_len(&first).unwrap();
        let page = source
            .headings_page(Path::new("n.md"), None, 0, None, Some(exact_budget))
            .unwrap();
        assert_eq!(page.headings.len(), 1);
        assert!(page.headings[0].text_truncated);
        assert_eq!(page.headings[0].heading_range.start_byte, 0);
        assert_eq!(serialized_len(&page).unwrap(), exact_budget);
    }
}
