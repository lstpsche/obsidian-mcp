//! Complete, source-bound preparation for semantic embeddings.

use std::cell::Cell;
use std::mem::size_of;
use std::{fmt, io};

use serde::de::{self, DeserializeSeed, MapAccess, SeqAccess, Visitor};
use std::path::Path;

use crate::error::{VaultError, VaultResult};
use crate::models::{EvidenceScope, SourceRange};

use super::embeddings::{Embedder, prepared_text_hash};
use super::parser::{MarkdownBlock, MarkdownStructure};
use super::source::{MAX_SOURCE_BYTES, source_revision};
use super::{frontmatter, parser};

pub(crate) const MAX_CHUNKS_PER_NOTE: usize = 4096;
pub(crate) const MAX_CONTEXT_BYTES: usize = 16 * 1024;
#[cfg(test)]
const DEFAULT_PREPARATION_BUDGET: usize = 64 * 1024 * 1024;
const OVERLAP_TOKENS: usize = 32;
const MAX_SPLIT_PROBES: usize = 4096;
const BUDGET_RESOURCE: &str = "embedding preparation budget";

#[derive(Debug, Clone)]
pub(crate) struct PreparedNote {
    pub(crate) source_revision: String,
    pub(crate) source_bytes: usize,
    pub(crate) body_range: SourceRange,
    pub(crate) title: String,
    pub(crate) tags: Vec<String>,
    pub(crate) chunks: Vec<PreparedChunk>,
}

impl PreparedNote {
    /// Retained capacities and space reserved for every eventual chunk vector.
    pub(crate) fn staging_bytes(&self, dimension: usize) -> usize {
        size_of::<Self>()
            .saturating_add(self.source_revision.capacity())
            .saturating_add(self.title.capacity())
            .saturating_add(string_vec_bytes(&self.tags))
            .saturating_add(
                self.chunks
                    .capacity()
                    .saturating_mul(size_of::<PreparedChunk>()),
            )
            .saturating_add(
                self.chunks
                    .iter()
                    .map(|chunk| {
                        chunk
                            .text
                            .capacity()
                            .saturating_add(string_vec_bytes(&chunk.heading_path))
                            .saturating_add(chunk.block_kind.as_ref().map_or(0, String::capacity))
                            .saturating_add(vector_reservation(dimension))
                    })
                    .fold(0usize, usize::saturating_add),
            )
    }
}

#[derive(Debug, Clone)]
pub(crate) struct PreparedChunk {
    pub(crate) text: String,
    pub(crate) input_hash: [u8; 32],
    pub(crate) scope: EvidenceScope,
    pub(crate) range: SourceRange,
    pub(crate) overlap_range: Option<SourceRange>,
    pub(crate) heading_path: Vec<String>,
    pub(crate) section_range: Option<SourceRange>,
    pub(crate) block_kind: Option<String>,
}

struct Section<'a> {
    range: SourceRange,
    heading_path: &'a [&'a str],
    section_range: SourceRange,
    heading_end: usize,
}

type TokenCounter<'a> = &'a dyn Fn(&str) -> VaultResult<usize>;

/// The caller reserves its raw source buffer separately. This budget includes
/// owned preparation workspace, retained output, and future chunk vectors.
pub(crate) fn prepare_note_with_budget(
    path: &Path,
    raw: &str,
    embedder: &dyn Embedder,
    budget_bytes: usize,
) -> VaultResult<PreparedNote> {
    prepare_budgeted(
        path,
        raw,
        embedder.input_policy().max_input_tokens,
        &|text| embedder.count_tokens(text),
        embedder.dimension(),
        budget_bytes,
    )
}

#[cfg(test)]
pub(crate) fn prepare_note(
    path: &Path,
    raw: &str,
    embedder: &dyn Embedder,
) -> VaultResult<PreparedNote> {
    prepare_note_with_budget(path, raw, embedder, DEFAULT_PREPARATION_BUDGET)
}

#[cfg(test)]
fn prepare_with_counter(
    path: &Path,
    raw: &str,
    max_tokens: usize,
    count: TokenCounter<'_>,
) -> VaultResult<PreparedNote> {
    prepare_budgeted(path, raw, max_tokens, count, 1, DEFAULT_PREPARATION_BUDGET)
}

struct PreparationBudget {
    limit: usize,
    used: Cell<usize>,
}

impl PreparationBudget {
    fn check(&self, bytes: usize) -> VaultResult<()> {
        let actual = self.used.get().saturating_add(bytes);
        check_limit(BUDGET_RESOURCE, actual, self.limit)
    }

    fn retain(&self, bytes: usize) -> VaultResult<()> {
        self.check(bytes)?;
        self.used.set(self.used.get() + bytes);
        Ok(())
    }

    fn reserve(&self, bytes: usize) -> VaultResult<WorkspaceReservation<'_>> {
        self.retain(bytes)?;
        Ok(WorkspaceReservation {
            budget: self,
            bytes,
        })
    }
}

struct WorkspaceReservation<'a> {
    budget: &'a PreparationBudget,
    bytes: usize,
}

impl Drop for WorkspaceReservation<'_> {
    fn drop(&mut self) {
        self.budget.used.set(self.budget.used.get() - self.bytes);
    }
}

struct Preparation<'a> {
    max_tokens: usize,
    count: TokenCounter<'a>,
    dimension: usize,
    memory: PreparationBudget,
}

impl Preparation<'_> {
    fn compose(
        &self,
        prefix: &str,
        raw: &str,
        overlap: Option<SourceRange>,
        range: SourceRange,
    ) -> VaultResult<String> {
        let overlap_bytes = overlap.map_or(0, |range| range.end_byte - range.start_byte + 1);
        let bytes = prefix
            .len()
            .saturating_add(overlap_bytes)
            .saturating_add(range.end_byte - range.start_byte);
        self.memory.check(bytes)?;
        let mut text = String::with_capacity(bytes);
        text.push_str(prefix);
        if let Some(overlap) = overlap {
            text.push_str(&raw[overlap.start_byte..overlap.end_byte]);
            text.push('\n');
        }
        text.push_str(&raw[range.start_byte..range.end_byte]);
        Ok(text)
    }

    fn fits(
        &self,
        prefix: &str,
        raw: &str,
        overlap: Option<SourceRange>,
        range: SourceRange,
    ) -> VaultResult<bool> {
        Ok((self.count)(&self.compose(prefix, raw, overlap, range)?)? <= self.max_tokens)
    }
}

fn prepare_budgeted(
    path: &Path,
    raw: &str,
    max_tokens: usize,
    count: TokenCounter<'_>,
    dimension: usize,
    budget_bytes: usize,
) -> VaultResult<PreparedNote> {
    check_limit("semantic source bytes", raw.len(), MAX_SOURCE_BYTES)?;
    let preparation = Preparation {
        max_tokens,
        count,
        dimension,
        memory: PreparationBudget {
            limit: budget_bytes,
            used: Cell::new(0),
        },
    };
    // Source revisions are a fixed 64-byte hexadecimal digest.
    preparation.memory.retain(size_of::<PreparedNote>() + 64)?;
    let title = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .ok_or_else(|| {
            VaultError::InvalidPath(format!("note has no UTF-8 filename: {}", path.display()))
        })?;
    preparation.memory.retain(title.len())?;
    let title = title.to_owned();
    let inline_bytes = parser::inline_tag_allocation_bound(raw);
    let tags = source_tags(path, raw, inline_bytes, &preparation)?;
    let body_start = frontmatter::extract_raw_frontmatter(raw).map_or(0, |(_, start)| start);
    let body_range = SourceRange {
        start_byte: body_start,
        end_byte: raw.len(),
    };
    let prefix_reservation = preparation.memory.reserve(title.len() + 1)?;
    let mut prefix = String::with_capacity(title.len() + 1);
    prefix.push_str(&title);
    prefix.push('\n');
    let full_input = preparation.compose(&prefix, raw, None, body_range)?;
    let mut chunks = Vec::new();
    if count(&full_input)? <= max_tokens {
        append_chunk(
            &mut chunks,
            full_input,
            EvidenceScope::Note,
            body_range,
            None,
            &[],
            None,
            None,
            &preparation,
        )?;
    } else {
        // The full-note trial must not stay alive beside the overflowing draft.
        drop(full_input);
        drop(prefix_reservation);
        drop(prefix);
        let structure_bytes = parser::semantic_structure_allocation_bound(raw);
        let workspace = preparation.memory.reserve(structure_bytes)?;
        let structure = parser::markdown_structure(raw)?;
        prepare_sections(
            raw,
            &title,
            body_range,
            &structure,
            &preparation,
            &mut chunks,
        )?;
        drop(structure);
        drop(workspace);
        validate_coverage(raw, body_range, &chunks)?;
        let note = PreparedNote {
            source_revision: source_revision(raw),
            source_bytes: raw.len(),
            body_range,
            title,
            tags,
            chunks,
        };
        check_limit(BUDGET_RESOURCE, note.staging_bytes(dimension), budget_bytes)?;
        return Ok(note);
    }
    drop(prefix_reservation);
    drop(prefix);
    validate_coverage(raw, body_range, &chunks)?;
    let note = PreparedNote {
        source_revision: source_revision(raw),
        source_bytes: raw.len(),
        body_range,
        title,
        tags,
        chunks,
    };
    check_limit(BUDGET_RESOURCE, note.staging_bytes(dimension), budget_bytes)?;
    Ok(note)
}

/// Count application-owned YAML value storage without retaining that value.
/// Parser-private arenas are bounded separately by the source size limit.
#[derive(Clone, Copy)]
struct ValueAllocationSize<'a> {
    depth: usize,
    available: usize,
    exceeded: &'a Cell<Option<usize>>,
}

impl ValueAllocationSize<'_> {
    fn accept<E: de::Error>(self, bytes: usize) -> Result<usize, E> {
        if bytes > self.available {
            self.exceeded.set(Some(bytes));
            Err(E::custom(
                "frontmatter value exceeds the preparation budget",
            ))
        } else {
            Ok(bytes)
        }
    }

    fn child(self) -> Self {
        Self {
            depth: self.depth + 1,
            ..self
        }
    }
}

impl<'de> DeserializeSeed<'de> for ValueAllocationSize<'_> {
    type Value = usize;

    fn deserialize<D: serde::Deserializer<'de>>(self, deserializer: D) -> Result<usize, D::Error> {
        if self.depth > 128 {
            return Err(de::Error::custom("frontmatter nesting exceeds 128 levels"));
        }
        deserializer.deserialize_any(self)
    }
}

impl<'de> Visitor<'de> for ValueAllocationSize<'_> {
    type Value = usize;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a YAML value")
    }

    fn visit_unit<E: de::Error>(self) -> Result<usize, E> {
        Ok(0)
    }
    fn visit_none<E: de::Error>(self) -> Result<usize, E> {
        Ok(0)
    }
    fn visit_bool<E: de::Error>(self, _: bool) -> Result<usize, E> {
        Ok(0)
    }
    fn visit_i64<E: de::Error>(self, _: i64) -> Result<usize, E> {
        Ok(0)
    }
    fn visit_u64<E: de::Error>(self, _: u64) -> Result<usize, E> {
        Ok(0)
    }
    fn visit_i128<E: de::Error>(self, _: i128) -> Result<usize, E> {
        Ok(0)
    }
    fn visit_u128<E: de::Error>(self, _: u128) -> Result<usize, E> {
        Ok(0)
    }
    fn visit_f64<E: de::Error>(self, _: f64) -> Result<usize, E> {
        Ok(0)
    }
    fn visit_str<E: de::Error>(self, value: &str) -> Result<usize, E> {
        self.accept(value.len())
    }
    fn visit_string<E: de::Error>(self, value: String) -> Result<usize, E> {
        self.accept(value.capacity())
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<usize, A::Error> {
        let mut length = 0usize;
        let mut nested = 0usize;
        while let Some(bytes) = sequence.next_element_seed(self.child())? {
            length += 1;
            nested = nested.saturating_add(bytes);
            self.accept::<A::Error>(
                nested.saturating_add(
                    parser::allocation_capacity_bound(length)
                        .saturating_mul(size_of::<serde_json::Value>()),
                ),
            )?;
        }
        Ok(nested.saturating_add(
            parser::allocation_capacity_bound(length)
                .saturating_mul(size_of::<serde_json::Value>()),
        ))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut mapping: A) -> Result<usize, A::Error> {
        let mut length = 0usize;
        let mut nested = 0usize;
        while let Some(key_bytes) = mapping.next_key_seed(self.child())? {
            let value_bytes = mapping.next_value_seed(self.child())?;
            length += 1;
            nested = nested.saturating_add(key_bytes).saturating_add(value_bytes);
            self.accept::<A::Error>(nested.saturating_add(length.saturating_mul(1024)))?;
        }
        // Covers sparsely occupied BTreeMap nodes, including a one-entry root.
        Ok(nested.saturating_add(length.saturating_mul(1024)))
    }
}

struct SerializedLength(usize);
impl io::Write for SerializedLength {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0 = self.0.saturating_add(bytes.len());
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn frontmatter_tag_bytes(frontmatter: &serde_json::Value) -> VaultResult<usize> {
    let Some(tags) = frontmatter.get("tags") else {
        return Ok(0);
    };
    let (count, text_bytes) = match tags {
        serde_json::Value::String(text) => (
            text.split(',')
                .filter(|part| !part.trim().is_empty())
                .count(),
            text.len(),
        ),
        serde_json::Value::Array(values) => {
            let mut bytes = 0usize;
            let mut count = 0usize;
            for value in values {
                match value {
                    serde_json::Value::Null => continue,
                    serde_json::Value::String(text) => bytes = bytes.saturating_add(text.len()),
                    other => {
                        let mut length = SerializedLength(0);
                        serde_json::to_writer(&mut length, other).map_err(|error| {
                            VaultError::Embedding(format!("cannot size frontmatter tag: {error}"))
                        })?;
                        bytes = bytes.saturating_add(
                            length
                                .0
                                .checked_next_power_of_two()
                                .unwrap_or(usize::MAX)
                                .max(128),
                        );
                    }
                }
                count += 1;
            }
            (count, bytes)
        }
        _ => (0, 0),
    };
    Ok(text_bytes.saturating_add(
        parser::allocation_capacity_bound(count).saturating_mul(size_of::<String>()),
    ))
}

fn source_tags(
    path: &Path,
    raw: &str,
    inline_bytes: usize,
    preparation: &Preparation<'_>,
) -> VaultResult<Vec<String>> {
    let yaml_bytes = if let Some((yaml, _)) = frontmatter::extract_raw_frontmatter(raw) {
        if yaml.trim().is_empty() {
            0
        } else {
            let exceeded = Cell::new(None);
            let measured = ValueAllocationSize {
                depth: 0,
                available: preparation.memory.limit - preparation.memory.used.get(),
                exceeded: &exceeded,
            }
            .deserialize(serde_yaml::Deserializer::from_str(yaml));
            if let Some(bytes) = exceeded.get() {
                return Err(VaultError::SemanticResourceLimit {
                    resource: BUDGET_RESOURCE.to_owned(),
                    actual: preparation.memory.used.get().saturating_add(bytes),
                    limit: preparation.memory.limit,
                });
            }
            measured.map_err(|source| VaultError::FrontmatterParse {
                path: path.to_path_buf(),
                source,
            })?
        }
    } else {
        0
    };
    let yaml_reservation = preparation.memory.reserve(yaml_bytes)?;
    // No YAML value is allocated for a note with no frontmatter.
    let frontmatter = frontmatter::parse_frontmatter(raw)?;
    let tag_bytes = frontmatter
        .as_ref()
        .map(frontmatter_tag_bytes)
        .transpose()?
        .unwrap_or(0);
    let tag_reservation = preparation.memory.reserve(tag_bytes)?;
    let mut tags = frontmatter
        .as_ref()
        .map(frontmatter::extract_frontmatter_tags)
        .unwrap_or_default();
    drop(frontmatter);
    drop(yaml_reservation);
    let inline_reservation = preparation.memory.reserve(inline_bytes)?;
    let inline = parser::extract_inline_tags(raw);
    let required = tags.len().saturating_add(inline.len());
    let extra_headers = required
        .saturating_sub(tags.capacity())
        .saturating_mul(size_of::<String>());
    let merge_reservation = preparation.memory.reserve(extra_headers)?;
    tags.try_reserve_exact(inline.len()).map_err(|error| {
        VaultError::Embedding(format!("cannot allocate semantic tag records: {error}"))
    })?;
    tags.extend(inline);
    tags.sort_unstable();
    tags.dedup();
    let retained = string_vec_bytes(&tags);
    let reserved = tag_bytes
        .saturating_add(inline_bytes)
        .saturating_add(extra_headers);
    check_limit(BUDGET_RESOURCE, retained, reserved)?;
    drop(merge_reservation);
    drop(inline_reservation);
    drop(tag_reservation);
    preparation.memory.retain(retained)?;
    Ok(tags)
}

fn prepare_sections(
    raw: &str,
    title: &str,
    body: SourceRange,
    structure: &MarkdownStructure,
    preparation: &Preparation<'_>,
    chunks: &mut Vec<PreparedChunk>,
) -> VaultResult<()> {
    let first = structure
        .headings
        .partition_point(|heading| heading.offset < body.start_byte);
    let headings = &structure.headings[first..];
    let _ends_reservation = preparation
        .memory
        .reserve(headings.len().saturating_mul(size_of::<usize>()))?;
    let mut subtree_ends = vec![body.end_byte; headings.len()];
    let mut stack = [0usize; 6];
    let mut stack_len = 0usize;
    for (index, heading) in headings.iter().enumerate() {
        while stack_len > 0 && headings[stack[stack_len - 1]].level >= heading.level {
            stack_len -= 1;
            subtree_ends[stack[stack_len]] = heading.offset;
        }
        stack[stack_len] = index;
        stack_len += 1;
    }
    let preamble_end = headings
        .first()
        .map_or(body.end_byte, |heading| heading.offset);
    if preamble_end > body.start_byte || headings.is_empty() {
        let range = SourceRange {
            start_byte: body.start_byte,
            end_byte: preamble_end,
        };
        let section = Section {
            range,
            heading_path: &[],
            section_range: range,
            heading_end: range.start_byte,
        };
        prepare_section(raw, title, &section, &structure.blocks, preparation, chunks)?;
    }
    stack_len = 0;
    for (index, heading) in headings.iter().enumerate() {
        while stack_len > 0 && headings[stack[stack_len - 1]].level >= heading.level {
            stack_len -= 1;
        }
        stack[stack_len] = index;
        stack_len += 1;
        let mut names = [""; 6];
        for (position, &heading_index) in stack[..stack_len].iter().enumerate() {
            names[position] = headings[heading_index].text.as_str();
        }
        let section = Section {
            range: SourceRange {
                start_byte: heading.offset,
                end_byte: headings
                    .get(index + 1)
                    .map_or(body.end_byte, |next| next.offset),
            },
            heading_path: &names[..stack_len],
            section_range: SourceRange {
                start_byte: heading.offset,
                end_byte: subtree_ends[index],
            },
            heading_end: heading.end,
        };
        prepare_section(raw, title, &section, &structure.blocks, preparation, chunks)?;
    }
    Ok(())
}

fn prepare_section(
    raw: &str,
    title: &str,
    section: &Section<'_>,
    blocks: &[MarkdownBlock],
    preparation: &Preparation<'_>,
    chunks: &mut Vec<PreparedChunk>,
) -> VaultResult<()> {
    let context_bytes =
        title
            .len()
            .saturating_add(1)
            .saturating_add(if section.heading_path.is_empty() {
                0
            } else {
                section
                    .heading_path
                    .iter()
                    .map(|text| text.len())
                    .sum::<usize>()
                    .saturating_add(
                        section
                            .heading_path
                            .len()
                            .saturating_sub(1)
                            .saturating_mul(3),
                    )
                    .saturating_add(1)
            });
    check_limit("embedding context bytes", context_bytes, MAX_CONTEXT_BYTES)?;
    let _prefix_reservation = preparation.memory.reserve(context_bytes)?;
    let mut prefix = String::with_capacity(context_bytes);
    prefix.push_str(title);
    prefix.push('\n');
    for (index, heading) in section.heading_path.iter().enumerate() {
        if index > 0 {
            prefix.push_str(" > ");
        }
        prefix.push_str(heading);
    }
    if !section.heading_path.is_empty() {
        prefix.push('\n');
    }
    let entire = preparation.compose(&prefix, raw, None, section.range)?;
    if (preparation.count)(&entire)? <= preparation.max_tokens {
        return append_chunk(
            chunks,
            entire,
            EvidenceScope::Passage,
            section.range,
            None,
            section.heading_path,
            Some(section.section_range),
            None,
            preparation,
        );
    }
    if section.range.start_byte == section.range.end_byte {
        return Err(VaultError::EmbeddingInputTooLong {
            tokens: (preparation.count)(&entire)?,
            max_tokens: preparation.max_tokens,
        });
    }
    drop(entire);
    let mut start = section.range.start_byte;
    while start < section.range.end_byte {
        let block = containing_block(blocks, start);
        let overlap = preceding_overlap(raw, section, block, start, &prefix, preparation)?;
        let candidate_end = fitting_end(
            raw,
            start,
            section.range.end_byte,
            &prefix,
            overlap,
            preparation,
        )?;
        let end = preferred_end(
            raw,
            start,
            candidate_end,
            section,
            blocks,
            &prefix,
            overlap,
            preparation,
        )?;
        let range = SourceRange {
            start_byte: start,
            end_byte: end,
        };
        let block_kind = split_block(blocks, range).map(|block| block.kind);
        let text = preparation.compose(&prefix, raw, overlap, range)?;
        append_chunk(
            chunks,
            text,
            EvidenceScope::Passage,
            range,
            overlap,
            section.heading_path,
            Some(section.section_range),
            block_kind,
            preparation,
        )?;
        start = end;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn append_chunk(
    chunks: &mut Vec<PreparedChunk>,
    text: String,
    scope: EvidenceScope,
    range: SourceRange,
    overlap_range: Option<SourceRange>,
    heading_path: &[&str],
    section_range: Option<SourceRange>,
    block_kind: Option<&str>,
    preparation: &Preparation<'_>,
) -> VaultResult<()> {
    check_limit("chunks per note", chunks.len() + 1, MAX_CHUNKS_PER_NOTE)?;
    let header_growth = if chunks.len() == chunks.capacity() {
        size_of::<PreparedChunk>()
    } else {
        0
    };
    let metadata = heading_path
        .len()
        .saturating_mul(size_of::<String>())
        .saturating_add(
            heading_path
                .iter()
                .map(|heading| heading.len())
                .sum::<usize>(),
        )
        .saturating_add(block_kind.map_or(0, str::len));
    let retained = text
        .capacity()
        .saturating_add(metadata)
        .saturating_add(header_growth)
        .saturating_add(vector_reservation(preparation.dimension));
    preparation.memory.check(retained)?;
    chunks.try_reserve_exact(1).map_err(|error| {
        VaultError::Embedding(format!("cannot allocate embedding chunk record: {error}"))
    })?;
    let mut owned_headings = Vec::with_capacity(heading_path.len());
    owned_headings.extend(heading_path.iter().map(|heading| (*heading).to_owned()));
    chunks.push(PreparedChunk {
        input_hash: prepared_text_hash(&text),
        text,
        scope,
        range,
        overlap_range,
        heading_path: owned_headings,
        section_range,
        block_kind: block_kind.map(str::to_owned),
    });
    preparation.memory.retain(retained)?;
    Ok(())
}

fn string_vec_bytes(strings: &Vec<String>) -> usize {
    strings
        .capacity()
        .saturating_mul(size_of::<String>())
        .saturating_add(
            strings
                .iter()
                .map(String::capacity)
                .fold(0usize, usize::saturating_add),
        )
}

fn vector_reservation(dimension: usize) -> usize {
    size_of::<Option<Vec<f32>>>()
        .saturating_add(size_of::<super::embeddings::ChunkEmbedding>())
        .saturating_add(dimension.saturating_mul(size_of::<f32>()))
        // Coordinator hash indices and bounded request/job pointer collections.
        .saturating_add(256)
}

/// Find a validated advancing prefix without assuming token-count monotonicity.
fn fitting_end(
    raw: &str,
    start: usize,
    section_end: usize,
    prefix: &str,
    overlap: Option<SourceRange>,
    preparation: &Preparation<'_>,
) -> VaultResult<usize> {
    let first_end = next_boundary(raw, start);
    let initial_bytes = preparation.max_tokens.saturating_mul(4).max(4);
    let mut end = raw
        .floor_char_boundary(start.saturating_add(initial_bytes).min(section_end))
        .max(first_end);
    let mut known_fit = None;
    loop {
        let tokens = (preparation.count)(&preparation.compose(
            prefix,
            raw,
            overlap,
            SourceRange {
                start_byte: start,
                end_byte: end,
            },
        )?)?;
        if tokens <= preparation.max_tokens {
            known_fit = Some(end);
            if end == section_end {
                return Ok(end);
            }
            let previous_end = end;
            end = raw.floor_char_boundary(
                start
                    .saturating_add((end - start).saturating_mul(2))
                    .min(section_end),
            );
            end = end.max(next_boundary(raw, previous_end));
        } else if let Some(end) = known_fit {
            return Ok(end);
        } else if end == first_end {
            return probe_advancing_prefix(raw, start, section_end, prefix, overlap, preparation);
        } else {
            end = raw
                .floor_char_boundary(start + (end - start) / 2)
                .max(first_end);
        }
    }
}

fn probe_advancing_prefix(
    raw: &str,
    start: usize,
    section_end: usize,
    prefix: &str,
    overlap: Option<SourceRange>,
    preparation: &Preparation<'_>,
) -> VaultResult<usize> {
    let mut minimum_tokens = usize::MAX;
    for (probes, (offset, character)) in raw[start..section_end].char_indices().enumerate() {
        if probes == MAX_SPLIT_PROBES {
            return Err(VaultError::SemanticResourceLimit {
                resource: "embedding split boundary probes".to_owned(),
                actual: probes + 1,
                limit: MAX_SPLIT_PROBES,
            });
        }
        let end = start + offset + character.len_utf8();
        let tokens = (preparation.count)(&preparation.compose(
            prefix,
            raw,
            overlap,
            SourceRange {
                start_byte: start,
                end_byte: end,
            },
        )?)?;
        if tokens <= preparation.max_tokens {
            return Ok(end);
        }
        minimum_tokens = minimum_tokens.min(tokens);
    }
    Err(VaultError::EmbeddingInputTooLong {
        tokens: minimum_tokens,
        max_tokens: preparation.max_tokens,
    })
}

#[allow(clippy::too_many_arguments)]
fn preferred_end(
    raw: &str,
    start: usize,
    valid_end: usize,
    section: &Section<'_>,
    blocks: &[MarkdownBlock],
    prefix: &str,
    overlap: Option<SourceRange>,
    preparation: &Preparation<'_>,
) -> VaultResult<usize> {
    if valid_end == section.range.end_byte {
        return Ok(valid_end);
    }
    let last_block = blocks.partition_point(|block| block.range.end <= valid_end);
    let block_end = last_block
        .checked_sub(1)
        .map(|index| blocks[index].range.end)
        .unwrap_or(start);
    let mut sentence_end = start;
    let mut line_end = start;
    let mut word_end = start;
    for (offset, character) in raw[start..valid_end].char_indices() {
        let end = start + offset + character.len_utf8();
        if character == '\n' {
            line_end = end;
        }
        if character.is_whitespace() {
            word_end = end;
        }
        if matches!(character, '.' | '!' | '?' | '。' | '！' | '？')
            && raw[end..].chars().next().is_none_or(char::is_whitespace)
        {
            sentence_end = end;
        }
    }
    let minimum_cut = if start == section.range.start_byte
        && raw[section.heading_end..section.range.end_byte]
            .chars()
            .any(|character| !character.is_whitespace())
    {
        section.heading_end
    } else {
        start
    };
    for end in [block_end, sentence_end, line_end, word_end] {
        if end <= minimum_cut || end > valid_end {
            continue;
        }
        if end == valid_end
            || preparation.fits(
                prefix,
                raw,
                overlap,
                SourceRange {
                    start_byte: start,
                    end_byte: end,
                },
            )?
        {
            return Ok(end);
        }
    }
    Ok(valid_end)
}

fn containing_block(blocks: &[MarkdownBlock], offset: usize) -> Option<&MarkdownBlock> {
    let index = blocks.partition_point(|block| block.range.end <= offset);
    blocks
        .get(index)
        .filter(|block| block.range.start <= offset)
}

fn split_block(blocks: &[MarkdownBlock], range: SourceRange) -> Option<&MarkdownBlock> {
    containing_block(blocks, range.start_byte)
        .filter(|block| range.start_byte > block.range.start || range.end_byte < block.range.end)
        .or_else(|| {
            containing_block(blocks, range.end_byte.saturating_sub(1))
                .filter(|block| range.end_byte < block.range.end)
        })
}

fn preceding_overlap(
    raw: &str,
    section: &Section<'_>,
    block: Option<&MarkdownBlock>,
    start: usize,
    prefix: &str,
    preparation: &Preparation<'_>,
) -> VaultResult<Option<SourceRange>> {
    let Some(block) =
        block.filter(|block| start > section.range.start_byte && start > block.range.start)
    else {
        return Ok(None);
    };
    let first_end = next_boundary(raw, start);
    let mut overlap_start = raw.ceil_char_boundary(
        start
            .saturating_sub(OVERLAP_TOKENS * 8)
            .max(block.range.start)
            .max(section.range.start_byte),
    );
    while overlap_start < start {
        let overlap = SourceRange {
            start_byte: overlap_start,
            end_byte: start,
        };
        if (preparation.count)(&raw[overlap_start..start])? <= OVERLAP_TOKENS
            && preparation.fits(
                prefix,
                raw,
                Some(overlap),
                SourceRange {
                    start_byte: start,
                    end_byte: first_end,
                },
            )?
        {
            return Ok(Some(overlap));
        }
        overlap_start = raw.ceil_char_boundary(overlap_start + (start - overlap_start).div_ceil(2));
    }
    Ok(None)
}

fn next_boundary(raw: &str, start: usize) -> usize {
    start + raw[start..].chars().next().map_or(0, char::len_utf8)
}

fn check_limit(resource: &str, actual: usize, limit: usize) -> VaultResult<()> {
    if actual > limit {
        return Err(VaultError::SemanticResourceLimit {
            resource: resource.to_owned(),
            actual,
            limit,
        });
    }
    Ok(())
}

fn validate_coverage(raw: &str, body: SourceRange, chunks: &[PreparedChunk]) -> VaultResult<()> {
    let mut expected_start = body.start_byte;
    for chunk in chunks {
        let range = chunk.range;
        if range.start_byte != expected_start
            || range.end_byte > body.end_byte
            || range.end_byte < range.start_byte
            || (range.start_byte == range.end_byte && body.start_byte != body.end_byte)
            || !raw.is_char_boundary(range.start_byte)
            || !raw.is_char_boundary(range.end_byte)
        {
            return Err(VaultError::Embedding(
                "prepared note does not cover the complete body with valid source ranges".into(),
            ));
        }
        expected_start = range.end_byte;
    }
    if chunks.is_empty() || expected_start != body.end_byte {
        return Err(VaultError::Embedding(
            "prepared note has incomplete body coverage".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn count(text: &str) -> VaultResult<usize> {
        Ok(text.chars().count() + 2)
    }

    fn prepare(raw: &str, limit: usize) -> PreparedNote {
        prepare_with_counter(Path::new("n.md"), raw, limit, &count).unwrap()
    }

    fn assert_complete(raw: &str, note: &PreparedNote, limit: usize) {
        validate_coverage(raw, note.body_range, &note.chunks).unwrap();
        let reconstructed = note
            .chunks
            .iter()
            .map(|chunk| &raw[chunk.range.start_byte..chunk.range.end_byte])
            .collect::<String>();
        assert_eq!(reconstructed, &raw[note.body_range.start_byte..]);
        for chunk in &note.chunks {
            assert!(count(&chunk.text).unwrap() <= limit, "{}", chunk.text);
            assert_eq!(chunk.input_hash, prepared_text_hash(&chunk.text));
            if let Some(overlap) = chunk.overlap_range {
                assert_eq!(overlap.end_byte, chunk.range.start_byte);
                assert!(
                    count(&raw[overlap.start_byte..overlap.end_byte]).unwrap() <= OVERLAP_TOKENS
                );
            }
        }
    }

    #[test]
    fn exact_limit_remains_whole_and_one_over_partitions() {
        let raw = "a".repeat(20);
        let note = prepare(&raw, 24);
        assert_eq!(note.chunks.len(), 1);
        assert_eq!(note.chunks[0].scope, EvidenceScope::Note);
        let overflow = prepare(&format!("{raw}b"), 24);
        assert!(overflow.chunks.len() > 1);
        assert!(
            overflow
                .chunks
                .iter()
                .all(|chunk| chunk.scope == EvidenceScope::Passage)
        );
        assert_complete(&format!("{raw}b"), &overflow, 24);
    }

    #[test]
    fn late_body_is_preserved_in_inputs_and_hashes() {
        let raw = format!(
            "{}\n\n## Tail\nImportant hidden fact.",
            "intro ".repeat(450)
        );
        let note = prepare(&raw, 96);
        assert_complete(&raw, &note, 96);
        assert!(
            note.chunks
                .iter()
                .any(|chunk| chunk.text.contains("Important hidden fact."))
        );
        let changed = prepare(&raw.replace("hidden", "secret"), 96);
        assert_ne!(note.source_revision, changed.source_revision);
        assert_ne!(
            note.chunks.last().unwrap().input_hash,
            changed.chunks.last().unwrap().input_hash
        );
    }

    #[test]
    fn nested_sections_preserve_hierarchy_and_subtree_bounds() {
        let raw = format!(
            "preamble\n# Parent\nintro\n## Child\n{}\n## Child\nsecond\n# Next\nend",
            "details ".repeat(15)
        );
        let note = prepare(&raw, 80);
        assert_complete(&raw, &note, 80);
        let parent = note
            .chunks
            .iter()
            .find(|chunk| chunk.heading_path == ["Parent"])
            .unwrap();
        assert_eq!(
            parent.section_range.unwrap().end_byte,
            raw.find("# Next").unwrap()
        );
        let children = note
            .chunks
            .iter()
            .filter(|chunk| chunk.heading_path == ["Parent", "Child"])
            .collect::<Vec<_>>();
        assert!(children.len() >= 2);
        assert!(children.iter().any(|chunk| chunk.text.contains("second")));
        assert!(note.chunks.first().unwrap().heading_path.is_empty());
    }

    #[test]
    fn frontmatter_and_body_metadata_share_source_revision() {
        let raw = "---\r\ntags: [front]\r\n---\r\n# Parent\r\n#inline\r\n";
        let note = prepare(raw, 100);
        assert_eq!(note.tags, ["front", "inline"]);
        assert!(!note.chunks[0].text.contains("tags:"));
        assert_eq!(note.source_bytes, raw.len());
        assert_complete(raw, &note, 100);
        let changed = prepare(&raw.replace("front", "other"), 100);
        assert_ne!(note.source_revision, changed.source_revision);
        assert_eq!(note.chunks[0].input_hash, changed.chunks[0].input_hash);
        assert_eq!(changed.tags, ["inline", "other"]);
    }

    #[test]
    fn setext_formatting_and_fake_code_headings_are_structural() {
        let raw = format!(
            "Root\n====\n\n## *Child*\n```rust\n# fake\n{}\n```\n",
            "let value = 1;\n".repeat(12)
        );
        let note = prepare(&raw, 100);
        assert_complete(&raw, &note, 100);
        assert!(
            note.chunks
                .iter()
                .any(|chunk| chunk.heading_path == ["Root", "Child"])
        );
        assert!(
            !note
                .chunks
                .iter()
                .any(|chunk| chunk.heading_path.iter().any(|heading| heading == "fake"))
        );
        assert!(
            note.chunks
                .iter()
                .any(|chunk| chunk.block_kind.as_deref() == Some("code"))
        );
    }

    #[test]
    fn unicode_headingless_and_whitespace_sources_progress() {
        for raw in ["🦀漢字".repeat(120), " \r\n\t".repeat(120), "x".repeat(300)] {
            let note = prepare(&raw, 48);
            assert_complete(&raw, &note, 48);
            assert!(note.chunks.len() > 1);
        }
    }

    #[test]
    fn empty_and_heading_only_notes_remain_searchable() {
        for raw in ["", "# Only heading\n", "# A\n## B\n# C\n"] {
            let note = prepare(raw, 128);
            assert_eq!(note.chunks.len(), 1);
            assert_complete(raw, &note, 128);
        }
        let raw = "# A\n## B\n# C\n".repeat(8);
        let note = prepare(&raw, 40);
        assert_complete(&raw, &note, 40);
    }

    #[test]
    fn paragraph_boundaries_are_preferred_when_available() {
        let raw = "first paragraph.\n\nsecond paragraph.\n\nthird paragraph.\n";
        let note = prepare(raw, 45);
        assert_complete(raw, &note, 45);
        assert_eq!(&raw[..note.chunks[0].range.end_byte], "first paragraph.\n");
    }

    #[test]
    fn mandatory_context_and_source_limits_fail_explicitly() {
        let title_error =
            prepare_with_counter(Path::new("very-long-title.md"), "x", 5, &count).unwrap_err();
        assert!(matches!(
            title_error,
            VaultError::EmbeddingInputTooLong { .. }
        ));
        let heading = format!("# {}\nbody", "h".repeat(MAX_CONTEXT_BYTES + 1));
        assert!(matches!(
            prepare_with_counter(Path::new("n.md"), &heading, 20, &count),
            Err(VaultError::SemanticResourceLimit { .. })
        ));
        let raw = "x".repeat(MAX_SOURCE_BYTES + 1);
        assert!(matches!(
            prepare_with_counter(Path::new("n.md"), &raw, 100, &count),
            Err(VaultError::SemanticResourceLimit { .. })
        ));
    }

    #[test]
    fn overlap_is_bounded_and_never_crosses_section_boundaries() {
        let raw = format!("## One\n{}\n## Two\n{}", "a ".repeat(90), "b ".repeat(90));
        let note = prepare(&raw, 80);
        assert_complete(&raw, &note, 80);
        assert!(
            note.chunks
                .iter()
                .any(|chunk| chunk.overlap_range.is_some())
        );
        for chunk in &note.chunks {
            if let Some(overlap) = chunk.overlap_range {
                assert!(overlap.start_byte >= chunk.section_range.unwrap().start_byte);
            }
        }
    }

    #[test]
    fn final_join_is_counted_even_for_nonmonotonic_tokenizer() {
        let counter = |text: &str| -> VaultResult<usize> {
            // A synthetic merge makes some longer prefixes cheaper than shorter ones.
            Ok(text.chars().count() + usize::from(text.ends_with('x')) * 10)
        };
        let raw = "xy".repeat(70);
        let note = prepare_with_counter(Path::new("n.md"), &raw, 30, &counter).unwrap();
        validate_coverage(&raw, note.body_range, &note.chunks).unwrap();
        assert!(
            note.chunks
                .iter()
                .all(|chunk| counter(&chunk.text).unwrap() <= 30)
        );
    }

    #[test]
    fn a_fitting_join_is_not_rejected_by_prefix_only_count() {
        let raw = "# A\nx\n# B\ny";
        let counter = |text: &str| -> VaultResult<usize> {
            Ok(if text == "n\nA\n# A\nx\n" || text == "n\nB\n# B\ny" {
                3
            } else {
                100
            })
        };
        let note = prepare_with_counter(Path::new("n.md"), raw, 4, &counter).unwrap();
        assert_eq!(note.chunks.len(), 2);
        validate_coverage(raw, note.body_range, &note.chunks).unwrap();
    }

    #[test]
    fn a_fitting_intermediate_prefix_can_follow_an_oversized_scalar() {
        let counter = |text: &str| -> VaultResult<usize> {
            let body = text.strip_prefix("n\n").unwrap_or(text);
            Ok(if body == "abc" { 3 } else { body.len() + 10 })
        };
        let raw = "abc".repeat(20);
        let note = prepare_with_counter(Path::new("n.md"), &raw, 4, &counter).unwrap();
        assert_eq!(note.chunks.len(), 20);
        assert!(note.chunks.iter().all(|chunk| chunk.text == "n\nabc"));
        validate_coverage(&raw, note.body_range, &note.chunks).unwrap();
    }

    #[test]
    fn pathological_no_progress_is_a_bounded_search_failure() {
        let counter = |_text: &str| -> VaultResult<usize> { Ok(100) };
        let raw = "x".repeat(MAX_SPLIT_PROBES + 1);
        let error = prepare_with_counter(Path::new("n.md"), &raw, 4, &counter).unwrap_err();
        assert!(matches!(
            error,
            VaultError::SemanticResourceLimit { resource, limit: MAX_SPLIT_PROBES, .. }
                if resource == "embedding split boundary probes"
        ));
    }

    #[test]
    fn preparation_budget_precedes_payload_and_vector_allocations() {
        let calls = Cell::new(0usize);
        let counter = |text: &str| -> VaultResult<usize> {
            calls.set(calls.get() + 1);
            count(text)
        };
        let raw = "plain prose ".repeat(1000);
        let error =
            prepare_budgeted(Path::new("n.md"), &raw, usize::MAX, &counter, 1, 1024).unwrap_err();
        assert!(
            matches!(error, VaultError::SemanticResourceLimit { resource, .. } if resource == BUDGET_RESOURCE)
        );
        assert_eq!(calls.get(), 0);

        let error = prepare_budgeted(Path::new("n.md"), "body", 100, &counter, 1_000_000, 65_536)
            .unwrap_err();
        assert!(
            matches!(error, VaultError::SemanticResourceLimit { resource, .. } if resource == BUDGET_RESOURCE)
        );
        assert!(calls.get() > 0);
    }

    #[test]
    fn large_punctuated_prose_is_not_charged_as_parser_nodes() {
        let raw = "word. ".repeat(350_000);
        let words = |text: &str| -> VaultResult<usize> { Ok(text.split_whitespace().count() + 2) };
        let note =
            prepare_budgeted(Path::new("n.md"), &raw, 8192, &words, 384, 8 * 1024 * 1024).unwrap();
        validate_coverage(&raw, note.body_range, &note.chunks).unwrap();
        assert!(note.chunks.len() > 1);
        assert!(note.staging_bytes(384) < 8 * 1024 * 1024);
        assert!(
            note.chunks
                .iter()
                .all(|chunk| words(&chunk.text).unwrap() <= 8192)
        );
    }

    #[test]
    fn budgeted_metadata_keeps_complex_frontmatter_tags() {
        let raw = "---\ntags: [front, null, 42, true, {nested: value}]\n---\nbody #inline\n";
        let note = prepare_budgeted(Path::new("n.md"), raw, 100, &count, 1, 65_536).unwrap();
        assert_eq!(
            note.tags,
            ["42", "front", "inline", "true", "{\"nested\":\"value\"}"]
        );
        assert!(note.staging_bytes(1) <= 65_536);
    }

    #[test]
    fn yaml_value_preflight_stops_on_budget_pressure_with_a_typed_error() {
        let raw = format!("---\nproperties: [{}]\n---\nbody\n", "1, ".repeat(2000));
        let error = prepare_budgeted(Path::new("n.md"), &raw, 100, &count, 1, 4096).unwrap_err();
        assert!(matches!(error,
            VaultError::SemanticResourceLimit { resource, limit: 4096, .. }
                if resource == BUDGET_RESOURCE));
    }
}
