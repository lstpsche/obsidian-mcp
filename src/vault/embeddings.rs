//! Embedding store and model wrapper for semantic search (Layer 2).
//!
//! Gated behind `#[cfg(has_embeddings)]` (either `embeddings` or `embeddings-api`
//! Cargo feature). Provides:
//! - `EmbeddingStore`: in-memory HashMap of note embeddings with brute-force
//!   cosine similarity search and bounded versioned persistence.
//! - `EmbeddingModel`: backend-agnostic wrapper supporting local fastembed
//!   (`--features embeddings`) and OpenAI-compatible API (`--features embeddings-api`).

use std::collections::{HashMap, HashSet};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};

#[cfg(feature = "embeddings-api")]
use std::sync::Arc;

use super::embedding_input::{self, EmbeddingInputPolicy, EmbeddingRequestLimits, TokenCounter};
use crate::config::EmbeddingProvider;
use crate::error::{VaultError, VaultResult};
use sha2::{Digest, Sha256};

const CACHE_MAGIC: [u8; 8] = *b"OBSMCPEM";
const CACHE_SCHEMA_VERSION: u16 = 2;
pub(crate) const EMBEDDING_INPUT_VERSION: u16 = 2;
const MAX_CACHE_BYTES: u64 = 1024 * 1024 * 1024;
const MAX_CACHE_ENTRIES: usize = 1_000_000;
const MAX_CACHE_PATH_BYTES: usize = 16 * 1024;
const MAX_NOTE_TAGS: usize = 4_096;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) enum EmbeddingBackendKind {
    Local,
    Api,
}

/// Identifies the complete vector space represented by an embedding cache.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct EmbeddingSpaceIdentity {
    pub backend: EmbeddingBackendKind,
    pub model: String,
    pub endpoint_fingerprint: Option<[u8; 32]>,
    pub dimension: usize,
    pub input_version: u16,
    pub tokenizer_fingerprint: [u8; 32],
    pub max_input_tokens: usize,
    pub chunking_version: u16,
}

impl EmbeddingSpaceIdentity {
    #[cfg(feature = "embeddings")]
    fn local(model: String, dimension: usize, policy: &EmbeddingInputPolicy) -> Self {
        Self {
            backend: EmbeddingBackendKind::Local,
            model,
            endpoint_fingerprint: None,
            dimension,
            input_version: EMBEDDING_INPUT_VERSION,
            tokenizer_fingerprint: policy.tokenizer_fingerprint,
            max_input_tokens: policy.max_input_tokens,
            chunking_version: embedding_input::CHUNKING_VERSION,
        }
    }

    #[cfg(feature = "embeddings-api")]
    fn api(model: String, base_url: &str, dimension: usize, policy: &EmbeddingInputPolicy) -> Self {
        Self {
            backend: EmbeddingBackendKind::Api,
            model,
            endpoint_fingerprint: Some(endpoint_fingerprint(base_url)),
            dimension,
            input_version: EMBEDDING_INPUT_VERSION,
            tokenizer_fingerprint: policy.tokenizer_fingerprint,
            max_input_tokens: policy.max_input_tokens,
            chunking_version: embedding_input::CHUNKING_VERSION,
        }
    }
}

pub(crate) trait Embedder: Send + Sync {
    fn dimension(&self) -> usize;
    fn space_identity(&self) -> &EmbeddingSpaceIdentity;
    fn input_policy(&self) -> &EmbeddingInputPolicy;
    fn count_tokens(&self, text: &str) -> VaultResult<usize>;
    fn request_limits(&self) -> EmbeddingRequestLimits;
    /// Plan a prefix that can be dispatched as one bounded request.
    fn plan_batch(&self, texts: &[&str]) -> VaultResult<usize> {
        embedding_input::plan_batch(
            texts,
            self.input_policy(),
            self.request_limits(),
            2,
            |text| self.count_tokens(text),
        )
    }
    fn embed_batch(&self, texts: &[&str]) -> VaultResult<Vec<Vec<f32>>>;
    fn embed_next_batch(&self, texts: &[&str]) -> (usize, VaultResult<Vec<Vec<f32>>>) {
        let count = match self.plan_batch(texts) {
            Ok(count) if count > 0 && count <= texts.len() => count,
            Ok(_) => {
                return (
                    1,
                    Err(VaultError::Embedding(
                        "embedding request planner did not make progress".into(),
                    )),
                );
            }
            Err(error) => return (1, Err(error)),
        };
        (count, self.embed_batch(&texts[..count]))
    }
    fn embed_query(&self, query: &str) -> VaultResult<Vec<f32>> {
        embedding_input::validate_input(query, self.input_policy(), |text| {
            self.count_tokens(text)
        })?;
        let mut vectors =
            validate_embedding_batch(self.embed_batch(&[query])?, 1, self.dimension())?;
        vectors
            .pop()
            .ok_or_else(|| VaultError::Embedding("embedding query returned no vector".into()))
    }
}

// ── Cosine similarity ──────────────────────────────────────────────────

pub(crate) fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    CosineQuery::new(&a[..a.len().min(b.len())]).similarity(b)
}

/// A validated query's norm is shared by every scored chunk.
pub(crate) struct CosineQuery<'a> {
    vector: &'a [f32],
    norm: f64,
}

impl<'a> CosineQuery<'a> {
    pub(crate) fn new(vector: &'a [f32]) -> Self {
        let squared_norm = vector.iter().fold(0.0f64, |sum, &value| {
            let value = f64::from(value);
            sum + value * value
        });
        Self {
            vector,
            norm: squared_norm.sqrt(),
        }
    }

    pub(crate) fn similarity(&self, other: &[f32]) -> f32 {
        let (dot, squared_norm) = self.vector.iter().zip(other).fold(
            (0.0f64, 0.0f64),
            |(dot, squared_norm), (&left, &right)| {
                let (left, right) = (f64::from(left), f64::from(right));
                (dot + left * right, squared_norm + right * right)
            },
        );
        if self.norm == 0.0 || squared_norm == 0.0 {
            return 0.0;
        }
        (dot / (self.norm * squared_norm.sqrt())).clamp(-1.0, 1.0) as f32
    }
}

// ── EmbeddingStore ─────────────────────────────────────────────────────

use crate::models::{EvidenceScope, SourceRange};
use std::sync::Arc as NoteArc;

use super::embedding_chunks::{
    MAX_CHUNKS_PER_NOTE as MAX_NOTE_CHUNKS, MAX_CONTEXT_BYTES, context_prefix_bytes,
};
use super::source::MAX_SOURCE_BYTES as MAX_NOTE_SOURCE_BYTES;
const MAX_NOTE_METADATA_BYTES: usize = 64 * 1024;
const MAX_CACHE_DIMENSION: usize = 65_536;

/// One validated vector and its exact source provenance.
#[derive(Debug, Clone)]
pub(crate) struct ChunkEmbedding {
    pub input_hash: [u8; 32],
    pub vector: Vec<f32>,
    pub scope: EvidenceScope,
    pub range: SourceRange,
    pub overlap_range: Option<SourceRange>,
    pub heading_path: Vec<String>,
    pub section_range: Option<SourceRange>,
    pub block_kind: Option<String>,
}

/// An immutable, complete semantic projection of one raw-note revision.
#[derive(Debug, Clone)]
pub(crate) struct NoteEmbeddingEntry {
    pub source_revision: String,
    pub source_bytes: usize,
    pub body_range: SourceRange,
    pub title: String,
    pub tags: Vec<String>,
    pub chunks: Vec<ChunkEmbedding>,
}

impl NoteEmbeddingEntry {
    /// Tied vectors select the earliest source occurrence deterministically.
    pub(crate) fn best_chunk(&self, query: &[f32]) -> Option<(usize, f32)> {
        self.chunks
            .iter()
            .enumerate()
            .map(|(index, chunk)| (index, cosine_similarity(query, &chunk.vector)))
            .max_by(|(left_index, left), (right_index, right)| {
                left.total_cmp(right)
                    .then_with(|| right_index.cmp(left_index))
            })
    }
}

/// Complete note projections; snapshots clone immutable entry references.
pub struct EmbeddingStore {
    embeddings: HashMap<PathBuf, NoteArc<NoteEmbeddingEntry>>,
    dim: usize,
    identity: Option<EmbeddingSpaceIdentity>,
    first_pass_complete: bool,
    encoded_entries_bytes: usize,
    total_chunks: usize,
}

// Legacy fixture shapes are deliberately unavailable to production writes.
#[cfg(test)]
struct EmbeddingCacheData {
    magic: [u8; 8],
    schema_version: u16,
    identity: EmbeddingSpaceIdentity,
    first_pass_complete: bool,
    entries: Vec<EmbeddingCacheEntry>,
}
#[cfg(test)]
struct EmbeddingCacheEntry {
    path: String,
    content_hash: [u8; 32],
    vector: Vec<f32>,
}

impl EmbeddingStore {
    pub fn new(dim: usize) -> Self {
        Self {
            embeddings: HashMap::new(),
            dim,
            identity: None,
            first_pass_complete: false,
            encoded_entries_bytes: 0,
            total_chunks: 0,
        }
    }

    pub(crate) fn new_with_identity(identity: EmbeddingSpaceIdentity) -> Self {
        Self {
            dim: identity.dimension,
            identity: Some(identity),
            ..Self::new(0)
        }
    }

    pub(crate) fn insert_note(
        &mut self,
        path: PathBuf,
        entry: NoteEmbeddingEntry,
    ) -> VaultResult<()> {
        let path_text = path
            .to_str()
            .ok_or_else(|| cache_error("cache path is not UTF-8"))?;
        validate_cache_path(path_text)?;
        validate_note_entry(&entry, self.dim)?;
        let entry_bytes = encoded_entry_len(path_text, &entry)?;
        let old = self.embeddings.get(&path);
        let old_bytes = old
            .map(|old| encoded_entry_len(path_text, old))
            .transpose()?
            .unwrap_or(0);
        let old_chunks = old.map_or(0, |old| old.chunks.len());
        let chunks = self
            .total_chunks
            .saturating_sub(old_chunks)
            .checked_add(entry.chunks.len())
            .ok_or_else(|| cache_error("embedding chunk count overflow"))?;
        let entries_bytes = self
            .encoded_entries_bytes
            .saturating_sub(old_bytes)
            .checked_add(entry_bytes)
            .ok_or_else(|| cache_error("embedding cache size overflow"))?;
        let notes = self.embeddings.len() + usize::from(old.is_none());
        if chunks > MAX_CACHE_ENTRIES || notes > MAX_CACHE_ENTRIES {
            return Err(cache_error(
                "embedding cache contains too many entries or chunks",
            ));
        }
        let header_size = self
            .identity
            .as_ref()
            .map(cache_header_len)
            .transpose()?
            .unwrap_or(MAX_NOTE_METADATA_BYTES);
        if entries_bytes.saturating_add(header_size) as u64 > MAX_CACHE_BYTES {
            return Err(cache_error("embedding cache is too large to persist"));
        }
        self.embeddings.insert(path, NoteArc::new(entry));
        self.encoded_entries_bytes = entries_bytes;
        self.total_chunks = chunks;
        Ok(())
    }

    pub(crate) fn note(&self, path: &Path) -> Option<NoteArc<NoteEmbeddingEntry>> {
        self.embeddings.get(path).map(NoteArc::clone)
    }

    pub(crate) fn capture_entries(
        &self,
        allowed: &HashSet<PathBuf>,
    ) -> HashMap<PathBuf, NoteArc<NoteEmbeddingEntry>> {
        self.embeddings
            .iter()
            .filter(|(path, _)| allowed.contains(*path))
            .map(|(path, entry)| (path.clone(), NoteArc::clone(entry)))
            .collect()
    }

    #[cfg(test)]
    pub fn insert(&mut self, path: PathBuf, vector: Vec<f32>) {
        if validate_vector(&vector, self.dim).is_err() {
            return;
        }
        let mut entry = fixture_note(&path, [0; 32], vector);
        entry.source_revision.clear();
        self.embeddings.insert(path, NoteArc::new(entry));
        self.first_pass_complete = false;
        self.recount_fixture_entries();
    }

    #[cfg(test)]
    pub(crate) fn insert_hashed(
        &mut self,
        path: PathBuf,
        content_hash: [u8; 32],
        vector: Vec<f32>,
    ) -> VaultResult<()> {
        validate_vector(&vector, self.dim)?;
        self.embeddings.insert(
            path.clone(),
            NoteArc::new(fixture_note(&path, content_hash, vector)),
        );
        self.recount_fixture_entries();
        Ok(())
    }

    #[cfg(test)]
    fn recount_fixture_entries(&mut self) {
        self.total_chunks = self
            .embeddings
            .values()
            .map(|entry| entry.chunks.len())
            .sum();
        self.encoded_entries_bytes = self
            .embeddings
            .iter()
            .filter_map(|(path, entry)| encoded_entry_len(&path.to_string_lossy(), entry).ok())
            .sum();
    }

    pub fn remove(&mut self, path: &Path) -> VaultResult<bool> {
        let Some(old) = self.embeddings.get(path) else {
            return Ok(false);
        };
        let path_text = path
            .to_str()
            .ok_or_else(|| cache_error("cache path is not UTF-8"))?;
        let old_bytes = encoded_entry_len(path_text, old)?;
        let total_chunks = self
            .total_chunks
            .checked_sub(old.chunks.len())
            .ok_or_else(|| cache_error("embedding chunk accounting underflow"))?;
        let encoded_entries_bytes = self
            .encoded_entries_bytes
            .checked_sub(old_bytes)
            .ok_or_else(|| cache_error("embedding cache size accounting underflow"))?;
        self.embeddings.remove(path);
        self.total_chunks = total_chunks;
        self.encoded_entries_bytes = encoded_entries_bytes;
        Ok(true)
    }

    /// Returns the first vector for callers checking entry availability.
    pub fn get(&self, path: &Path) -> Option<&[f32]> {
        self.embeddings
            .get(path)
            .and_then(|entry| entry.chunks.first())
            .map(|chunk| chunk.vector.as_slice())
    }

    #[cfg(test)]
    pub(crate) fn content_hash(&self, path: &Path) -> Option<&[u8; 32]> {
        self.embeddings
            .get(path)
            .and_then(|entry| entry.chunks.first())
            .map(|chunk| &chunk.input_hash)
    }

    pub fn len(&self) -> usize {
        self.embeddings.len()
    }
    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }
    pub fn dim(&self) -> usize {
        self.dim
    }
    #[cfg(test)]
    pub(crate) fn identity(&self) -> Option<&EmbeddingSpaceIdentity> {
        self.identity.as_ref()
    }
    pub(crate) fn first_pass_complete(&self) -> bool {
        self.first_pass_complete
    }
    pub(crate) fn set_first_pass_complete(&mut self, complete: bool) {
        self.first_pass_complete = complete;
    }

    pub(crate) fn retain_paths(&mut self, paths: &HashSet<PathBuf>) -> VaultResult<bool> {
        let removed = self
            .embeddings
            .keys()
            .filter(|path| !paths.contains(*path))
            .cloned()
            .collect::<Vec<_>>();
        for path in &removed {
            self.remove(path)?;
        }
        Ok(!removed.is_empty())
    }

    pub fn query(&self, query_vec: &[f32], top_k: usize) -> Vec<(PathBuf, f32)> {
        let scored = self
            .embeddings
            .iter()
            .filter_map(|(path, entry)| {
                entry
                    .best_chunk(query_vec)
                    .map(|(_, score)| (path.clone(), score))
            })
            .collect();
        Self::rank_scores(scored, top_k)
    }

    #[cfg(test)]
    pub(crate) fn query_paths(
        &self,
        query_vec: &[f32],
        allowed: &HashSet<PathBuf>,
        top_k: usize,
    ) -> Vec<(PathBuf, f32)> {
        let scored = self
            .embeddings
            .iter()
            .filter(|(path, _)| allowed.contains(*path))
            .filter_map(|(path, entry)| {
                entry
                    .best_chunk(query_vec)
                    .map(|(_, score)| (path.clone(), score))
            })
            .collect();
        Self::rank_scores(scored, top_k)
    }

    fn rank_scores(mut scored: Vec<(PathBuf, f32)>, top_k: usize) -> Vec<(PathBuf, f32)> {
        let cmp = |left: &(PathBuf, f32), right: &(PathBuf, f32)| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        };
        if top_k == 0 {
            return Vec::new();
        }
        if top_k < scored.len() {
            scored.select_nth_unstable_by(top_k, cmp);
            scored.truncate(top_k);
        }
        scored.sort_unstable_by(cmp);
        scored
    }

    pub fn save(&self, path: &Path) -> VaultResult<()> {
        Self::persist_cache_bytes(path, &self.encode_cache()?, None).map(|_| ())
    }

    pub(crate) fn encode_cache(&self) -> VaultResult<Vec<u8>> {
        let identity = self
            .identity
            .as_ref()
            .ok_or_else(|| cache_error("embedding store has no vector-space identity"))?;
        validate_space_identity(identity)?;
        if identity.dimension != self.dim {
            return Err(cache_error(
                "embedding store identity has an invalid dimension",
            ));
        }
        if self.embeddings.len() > MAX_CACHE_ENTRIES || self.total_chunks > MAX_CACHE_ENTRIES {
            return Err(cache_error(
                "embedding cache contains too many entries or chunks",
            ));
        }
        let mut entries = self.embeddings.iter().collect::<Vec<_>>();
        entries.sort_unstable_by_key(|(left, _)| *left);
        let mut out = CacheWriter::default();
        write_header(
            &mut out,
            CACHE_MAGIC,
            CACHE_SCHEMA_VERSION,
            identity,
            self.first_pass_complete,
            entries.len(),
        )?;
        for (path, entry) in entries {
            let path = path
                .to_str()
                .ok_or_else(|| cache_error("cache path is not UTF-8"))?;
            validate_cache_path(path)?;
            validate_note_entry(entry, self.dim)?;
            write_entry(&mut out, path, entry)?;
        }
        Ok(out.bytes)
    }

    pub(crate) fn persist_cache_bytes_if_live(
        path: &Path,
        bytes: &[u8],
        live: &AtomicBool,
    ) -> VaultResult<bool> {
        Self::persist_cache_bytes(path, bytes, Some(live))
    }

    fn persist_cache_bytes(
        path: &Path,
        bytes: &[u8],
        live: Option<&AtomicBool>,
    ) -> VaultResult<bool> {
        if bytes.len() as u64 > MAX_CACHE_BYTES {
            return Err(cache_error("embedding cache is too large to persist"));
        }
        if live.is_some_and(|flag| !flag.load(AtomicOrdering::Acquire)) {
            return Ok(false);
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
            let mut temp = tempfile::NamedTempFile::new_in(parent)?;
            temp.write_all(bytes)?;
            temp.flush()?;
            temp.as_file().sync_all()?;
            if live.is_some_and(|flag| !flag.load(AtomicOrdering::Acquire)) {
                return Ok(false);
            }
            temp.persist(path)
                .map_err(|error| VaultError::Io(error.error))?;
            Ok(true)
        } else {
            Err(cache_error("embedding cache path has no parent"))
        }
    }

    pub fn load(path: &Path) -> VaultResult<Self> {
        Self::load_bounded(path, None, MAX_CACHE_ENTRIES)
    }

    pub(crate) fn load_for_space(
        path: &Path,
        expected: &EmbeddingSpaceIdentity,
        current_note_count: usize,
    ) -> VaultResult<Self> {
        Self::load_bounded(
            path,
            Some(expected),
            current_note_count
                .saturating_add(1024)
                .min(MAX_CACHE_ENTRIES),
        )
    }

    fn load_bounded(
        path: &Path,
        expected: Option<&EmbeddingSpaceIdentity>,
        max_entries: usize,
    ) -> VaultResult<Self> {
        let file = std::fs::File::open(path)?;
        let size = file.metadata()?.len();
        if size > MAX_CACHE_BYTES {
            return Err(cache_error("embedding cache is too large"));
        }
        let mut input = CacheReader {
            reader: std::io::BufReader::new(file).take(MAX_CACHE_BYTES + 1),
            remaining: size as usize,
        };
        if input.array::<8>()? != CACHE_MAGIC {
            return Err(cache_error(
                "unsupported legacy embedding cache format or invalid magic",
            ));
        }
        let schema = input.u16()?;
        if schema != CACHE_SCHEMA_VERSION {
            return Err(VaultError::EmbeddingCacheIncompatible(format!(
                "unsupported embedding cache schema version {schema}; rebuilding required"
            )));
        }
        let identity_bytes = input.bytes(MAX_NOTE_METADATA_BYTES)?;
        let identity: EmbeddingSpaceIdentity = serde_json::from_slice(&identity_bytes)
            .map_err(|error| cache_error(format!("cache identity deserialize error: {error}")))?;
        validate_space_identity(&identity)?;
        if let Some(expected) = expected
            && &identity != expected
        {
            return Err(VaultError::EmbeddingCacheIncompatible(
                "embedding cache vector-space identity mismatch".into(),
            ));
        }
        let first_pass_complete = input.flag()?;
        let count = input.count(max_entries, "entries")?;
        let mut store = Self::new_with_identity(identity);
        store.first_pass_complete = first_pass_complete;
        for _ in 0..count {
            let path_text = input.string(MAX_CACHE_PATH_BYTES)?;
            let path = validate_cache_path(&path_text)?;
            // Canonically equivalent spellings can identify distinct native files.
            if store.embeddings.contains_key(&path) {
                return Err(cache_error(format!(
                    "embedding cache contains duplicate path '{path_text}'"
                )));
            }
            let entry = read_entry(&mut input, store.dim, store.total_chunks)?;
            store.insert_note(path, entry)?;
        }
        if input.remaining != 0 {
            return Err(cache_error("embedding cache contains trailing bytes"));
        }
        let mut trailing = [0; 1];
        if input.reader.read(&mut trailing)? != 0 {
            return Err(cache_error("embedding cache contains trailing bytes"));
        }
        Ok(store)
    }
}

fn cache_error(message: impl Into<String>) -> VaultError {
    VaultError::Embedding(message.into())
}

fn validate_range(range: &SourceRange, source_bytes: usize) -> VaultResult<()> {
    if range.start_byte > range.end_byte || range.end_byte > source_bytes {
        return Err(cache_error(
            "embedding cache contains an invalid source range",
        ));
    }
    Ok(())
}

fn validate_note_entry(entry: &NoteEmbeddingEntry, dim: usize) -> VaultResult<()> {
    if entry.source_revision.len() != 64
        || !entry
            .source_revision
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(cache_error(
            "embedding cache entry has an invalid source revision or prepared-text hash",
        ));
    }
    if entry.source_bytes > MAX_NOTE_SOURCE_BYTES {
        return Err(cache_error("embedding source exceeds supported byte limit"));
    }
    validate_range(&entry.body_range, entry.source_bytes)?;
    if entry.body_range.end_byte != entry.source_bytes {
        return Err(cache_error("embedding body range must end at source EOF"));
    }
    validate_note_metadata(&entry.title, &entry.tags)?;
    if entry.chunks.is_empty() || entry.chunks.len() > MAX_NOTE_CHUNKS {
        return Err(cache_error("embedding note has an invalid chunk count"));
    }
    if entry.body_range.start_byte == entry.body_range.end_byte
        && (entry.chunks.len() != 1 || !matches!(entry.chunks[0].scope, EvidenceScope::Note))
    {
        return Err(cache_error(
            "an empty note body requires one whole-note embedding",
        ));
    }
    let mut next_byte = entry.body_range.start_byte;
    for chunk in &entry.chunks {
        validate_vector(&chunk.vector, dim)?;
        validate_range(&chunk.range, entry.source_bytes)?;
        if chunk.range.start_byte != next_byte
            || chunk.range.end_byte < chunk.range.start_byte
            || (chunk.range.start_byte == chunk.range.end_byte
                && entry.body_range.start_byte != entry.body_range.end_byte)
        {
            return Err(cache_error(
                "embedding chunk ranges do not cover the complete body exactly",
            ));
        }
        next_byte = chunk.range.end_byte;
        if let Some(overlap) = &chunk.overlap_range {
            validate_range(overlap, entry.source_bytes)?;
            if overlap.start_byte < entry.body_range.start_byte
                || overlap.end_byte > chunk.range.start_byte
            {
                return Err(cache_error(
                    "embedding overlap is outside its preceding source",
                ));
            }
        }
        if let Some(section) = &chunk.section_range {
            validate_range(section, entry.source_bytes)?;
            if section.start_byte > chunk.range.start_byte
                || section.end_byte < chunk.range.end_byte
            {
                return Err(cache_error("embedding chunk is outside its section"));
            }
        }
        let context_bytes = context_prefix_bytes(
            &entry.title,
            chunk.heading_path.iter().map(String::len).sum(),
            chunk.heading_path.len(),
        );
        if chunk.heading_path.len() > 6 || context_bytes > MAX_CONTEXT_BYTES {
            return Err(cache_error(
                "embedding chunk context exceeds supported limit",
            ));
        }
        if chunk
            .block_kind
            .as_ref()
            .is_some_and(|kind| kind.len() > MAX_CONTEXT_BYTES)
        {
            return Err(cache_error(
                "embedding block metadata exceeds supported limit",
            ));
        }
        if matches!(chunk.scope, EvidenceScope::Note)
            && (entry.chunks.len() != 1
                || chunk.range != entry.body_range
                || !chunk.heading_path.is_empty()
                || chunk.section_range.is_some())
        {
            return Err(cache_error(
                "whole-note embedding has invalid evidence metadata",
            ));
        }
    }
    if next_byte != entry.body_range.end_byte {
        return Err(cache_error("embedding chunks omit body content"));
    }
    Ok(())
}

pub(crate) fn validate_note_metadata(title: &str, tags: &[String]) -> VaultResult<()> {
    if tags.len() > MAX_NOTE_TAGS {
        return Err(VaultError::SemanticResourceLimit {
            resource: "embedding note tags".into(),
            actual: tags.len(),
            limit: MAX_NOTE_TAGS,
        });
    }
    let metadata_bytes = tags
        .iter()
        .fold(title.len(), |total, tag| total.saturating_add(tag.len()));
    if metadata_bytes > MAX_NOTE_METADATA_BYTES {
        return Err(VaultError::SemanticResourceLimit {
            resource: "embedding note metadata bytes".into(),
            actual: metadata_bytes,
            limit: MAX_NOTE_METADATA_BYTES,
        });
    }
    Ok(())
}

fn encoded_entry_len(path: &str, entry: &NoteEmbeddingEntry) -> VaultResult<usize> {
    let mut writer = CacheWriter {
        bytes: Vec::new(),
        counting: true,
        length: 0,
    };
    write_entry(&mut writer, path, entry)?;
    Ok(writer.length)
}

fn cache_header_len(identity: &EmbeddingSpaceIdentity) -> VaultResult<usize> {
    let mut writer = CacheWriter {
        bytes: Vec::new(),
        counting: true,
        length: 0,
    };
    write_header(
        &mut writer,
        CACHE_MAGIC,
        CACHE_SCHEMA_VERSION,
        identity,
        false,
        0,
    )?;
    Ok(writer.length)
}

#[derive(Default)]
struct CacheWriter {
    bytes: Vec<u8>,
    counting: bool,
    length: usize,
}
impl CacheWriter {
    fn put(&mut self, bytes: &[u8]) -> VaultResult<()> {
        let needed = self
            .length
            .checked_add(bytes.len())
            .ok_or_else(|| cache_error("embedding cache size overflow"))?;
        if needed as u64 > MAX_CACHE_BYTES {
            return Err(cache_error("embedding cache is too large to persist"));
        }
        if !self.counting {
            if needed > self.bytes.capacity() {
                let capacity = needed
                    .max(self.bytes.capacity().saturating_mul(2).max(1024))
                    .min(MAX_CACHE_BYTES as usize);
                self.bytes
                    .try_reserve_exact(capacity - self.bytes.len())
                    .map_err(|error| {
                        cache_error(format!("embedding cache allocation failed: {error}"))
                    })?;
            }
            self.bytes.extend_from_slice(bytes);
        }
        self.length = needed;
        Ok(())
    }
    fn number(&mut self, value: usize) -> VaultResult<()> {
        let value = u32::try_from(value)
            .map_err(|_| cache_error("cache length exceeds supported integer range"))?;
        self.put(&value.to_le_bytes())
    }
    fn string(&mut self, text: &str) -> VaultResult<()> {
        self.number(text.len())?;
        self.put(text.as_bytes())
    }
    fn range(&mut self, range: &SourceRange) -> VaultResult<()> {
        self.number(range.start_byte)?;
        self.number(range.end_byte)
    }
    fn optional_range(&mut self, range: Option<&SourceRange>) -> VaultResult<()> {
        self.put(&[u8::from(range.is_some())])?;
        if let Some(range) = range {
            self.range(range)?;
        }
        Ok(())
    }
}

fn write_header(
    out: &mut CacheWriter,
    magic: [u8; 8],
    schema: u16,
    identity: &EmbeddingSpaceIdentity,
    complete: bool,
    count: usize,
) -> VaultResult<()> {
    out.put(&magic)?;
    out.put(&schema.to_le_bytes())?;
    let identity = serde_json::to_vec(identity)
        .map_err(|error| cache_error(format!("cache identity serialization failed: {error}")))?;
    if identity.len() > MAX_NOTE_METADATA_BYTES {
        return Err(cache_error(
            "embedding identity exceeds supported metadata limit",
        ));
    }
    out.number(identity.len())?;
    out.put(&identity)?;
    out.put(&[u8::from(complete)])?;
    out.number(count)
}

fn write_entry(out: &mut CacheWriter, path: &str, entry: &NoteEmbeddingEntry) -> VaultResult<()> {
    out.string(path)?;
    out.string(&entry.source_revision)?;
    out.number(entry.source_bytes)?;
    out.range(&entry.body_range)?;
    out.string(&entry.title)?;
    out.number(entry.tags.len())?;
    for tag in &entry.tags {
        out.string(tag)?;
    }
    out.number(entry.chunks.len())?;
    for chunk in &entry.chunks {
        out.put(&chunk.input_hash)?;
        out.put(&[u8::from(matches!(chunk.scope, EvidenceScope::Passage))])?;
        out.range(&chunk.range)?;
        out.optional_range(chunk.overlap_range.as_ref())?;
        out.number(chunk.heading_path.len())?;
        for heading in &chunk.heading_path {
            out.string(heading)?;
        }
        out.optional_range(chunk.section_range.as_ref())?;
        out.put(&[u8::from(chunk.block_kind.is_some())])?;
        if let Some(kind) = &chunk.block_kind {
            out.string(kind)?;
        }
        out.number(chunk.vector.len())?;
        for value in &chunk.vector {
            out.put(&value.to_le_bytes())?;
        }
    }
    Ok(())
}

struct CacheReader<R> {
    reader: R,
    remaining: usize,
}
impl<R: Read> CacheReader<R> {
    fn take(&mut self, length: usize) -> VaultResult<Vec<u8>> {
        if length > self.remaining {
            return Err(cache_error("cache deserialize error: truncated field"));
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(length)
            .map_err(|error| cache_error(format!("cache allocation failed: {error}")))?;
        bytes.resize(length, 0);
        self.reader
            .read_exact(&mut bytes)
            .map_err(|error| cache_error(format!("cache deserialize error: {error}")))?;
        self.remaining -= length;
        Ok(bytes)
    }
    fn array<const N: usize>(&mut self) -> VaultResult<[u8; N]> {
        if N > self.remaining {
            return Err(cache_error("cache deserialize error: truncated field"));
        }
        let mut bytes = [0; N];
        self.reader
            .read_exact(&mut bytes)
            .map_err(|error| cache_error(format!("cache deserialize error: {error}")))?;
        self.remaining -= N;
        Ok(bytes)
    }
    fn u16(&mut self) -> VaultResult<u16> {
        Ok(u16::from_le_bytes(self.array()?))
    }
    fn number(&mut self) -> VaultResult<usize> {
        Ok(u32::from_le_bytes(self.array()?) as usize)
    }
    fn count(&mut self, maximum: usize, kind: &str) -> VaultResult<usize> {
        let value = self.number()?;
        if value > maximum {
            return Err(cache_error(format!(
                "embedding cache contains too many {kind}: {value} (limit {maximum})"
            )));
        }
        Ok(value)
    }
    fn flag(&mut self) -> VaultResult<bool> {
        match self.array::<1>()?[0] {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(cache_error("cache deserialize error: invalid flag")),
        }
    }
    fn bytes(&mut self, maximum: usize) -> VaultResult<Vec<u8>> {
        let length = self.count(maximum, "field bytes")?;
        self.take(length)
    }
    fn string(&mut self, maximum: usize) -> VaultResult<String> {
        String::from_utf8(self.bytes(maximum)?)
            .map_err(|error| cache_error(format!("cache field is not UTF-8: {error}")))
    }
    fn range(&mut self) -> VaultResult<SourceRange> {
        Ok(SourceRange {
            start_byte: self.number()?,
            end_byte: self.number()?,
        })
    }
    fn optional_range(&mut self) -> VaultResult<Option<SourceRange>> {
        if self.flag()? {
            self.range().map(Some)
        } else {
            Ok(None)
        }
    }
}

fn read_entry<R: Read>(
    input: &mut CacheReader<R>,
    dim: usize,
    stored_chunks: usize,
) -> VaultResult<NoteEmbeddingEntry> {
    let source_revision = input.string(64)?;
    let source_bytes = input.count(MAX_NOTE_SOURCE_BYTES, "source bytes")?;
    let body_range = input.range()?;
    let title = input.string(MAX_NOTE_METADATA_BYTES)?;
    let tag_count = input.count(MAX_NOTE_TAGS, "tags")?;
    let mut metadata_bytes = title.len();
    let mut tags = Vec::new();
    for _ in 0..tag_count {
        let tag = input.string(MAX_NOTE_METADATA_BYTES.saturating_sub(metadata_bytes))?;
        metadata_bytes += tag.len();
        tags.push(tag);
    }
    let count = input.count(
        MAX_NOTE_CHUNKS.min(MAX_CACHE_ENTRIES.saturating_sub(stored_chunks)),
        "chunks",
    )?;
    let mut chunks = Vec::new();
    for _ in 0..count {
        let input_hash = input.array()?;
        let scope = if input.flag()? {
            EvidenceScope::Passage
        } else {
            EvidenceScope::Note
        };
        let range = input.range()?;
        let overlap_range = input.optional_range()?;
        let headings = input.count(6, "ancestor headings")?;
        let mut heading_path = Vec::new();
        let mut context_bytes = context_prefix_bytes(&title, 0, headings);
        if context_bytes > MAX_CONTEXT_BYTES {
            return Err(cache_error(
                "embedding chunk context exceeds supported limit",
            ));
        }
        for _ in 0..headings {
            let heading = input.string(MAX_CONTEXT_BYTES - context_bytes)?;
            context_bytes += heading.len();
            heading_path.push(heading);
        }
        let section_range = input.optional_range()?;
        let block_kind = if input.flag()? {
            Some(input.string(MAX_CONTEXT_BYTES)?)
        } else {
            None
        };
        let vector_length = input.number()?;
        if vector_length != dim {
            return Err(cache_error(format!(
                "embedding dimension mismatch: expected {dim}, got {vector_length}"
            )));
        }
        if vector_length.saturating_mul(4) > input.remaining {
            return Err(cache_error("cache deserialize error: truncated vector"));
        }
        let mut vector = Vec::new();
        vector
            .try_reserve_exact(vector_length)
            .map_err(|error| cache_error(format!("cache vector allocation failed: {error}")))?;
        for _ in 0..vector_length {
            vector.push(f32::from_le_bytes(input.array()?));
        }
        chunks.push(ChunkEmbedding {
            input_hash,
            vector,
            scope,
            range,
            overlap_range,
            heading_path,
            section_range,
            block_kind,
        });
    }
    Ok(NoteEmbeddingEntry {
        source_revision,
        source_bytes,
        body_range,
        title,
        tags,
        chunks,
    })
}

#[cfg(test)]
fn fixture_note(path: &Path, hash: [u8; 32], vector: Vec<f32>) -> NoteEmbeddingEntry {
    NoteEmbeddingEntry {
        source_revision: hash.iter().map(|byte| format!("{byte:02x}")).collect(),
        source_bytes: 1,
        body_range: SourceRange {
            start_byte: 0,
            end_byte: 1,
        },
        title: path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string(),
        tags: Vec::new(),
        chunks: vec![ChunkEmbedding {
            input_hash: hash,
            vector,
            scope: EvidenceScope::Note,
            range: SourceRange {
                start_byte: 0,
                end_byte: 1,
            },
            overlap_range: None,
            heading_path: Vec::new(),
            section_range: None,
            block_kind: None,
        }],
    }
}

fn validate_space_identity(identity: &EmbeddingSpaceIdentity) -> VaultResult<()> {
    if identity.dimension == 0 || identity.dimension > MAX_CACHE_DIMENSION {
        return Err(VaultError::Embedding(format!(
            "embedding dimension is outside supported bounds (1..={MAX_CACHE_DIMENSION})"
        )));
    }
    if identity.model.trim().is_empty() || identity.model.len() > MAX_CACHE_PATH_BYTES {
        return Err(VaultError::Embedding(
            "embedding cache model identity must not be empty".into(),
        ));
    }
    if identity.input_version != EMBEDDING_INPUT_VERSION {
        return Err(VaultError::EmbeddingCacheIncompatible(format!(
            "unsupported embedding input version {}",
            identity.input_version
        )));
    }
    if identity.max_input_tokens == 0 {
        return Err(cache_error("embedding input capacity must be positive"));
    }
    if identity.chunking_version != embedding_input::CHUNKING_VERSION {
        return Err(VaultError::EmbeddingCacheIncompatible(format!(
            "unsupported embedding chunking version {}",
            identity.chunking_version,
        )));
    }
    match (identity.backend, identity.endpoint_fingerprint) {
        (EmbeddingBackendKind::Local, None) | (EmbeddingBackendKind::Api, Some(_)) => Ok(()),
        (EmbeddingBackendKind::Local, Some(_)) => Err(VaultError::Embedding(
            "local embedding cache identity must not contain an API endpoint fingerprint".into(),
        )),
        (EmbeddingBackendKind::Api, None) => Err(VaultError::Embedding(
            "API embedding cache identity is missing its endpoint fingerprint".into(),
        )),
    }
}

fn validate_cache_path(path: &str) -> VaultResult<PathBuf> {
    if path.is_empty() || path.len() > MAX_CACHE_PATH_BYTES || path.contains('\\') {
        return Err(VaultError::Embedding(format!(
            "invalid embedding cache path '{path}'"
        )));
    }
    let original = Path::new(path);
    let normalized = super::path::normalize_relative(original).map_err(|error| {
        VaultError::Embedding(format!("invalid embedding cache path '{path}': {error}"))
    })?;
    if normalized != original || normalized.to_string_lossy() != path {
        return Err(VaultError::Embedding(format!(
            "embedding cache path is not normalized: '{path}'"
        )));
    }
    Ok(normalized)
}

fn validate_vector(vector: &[f32], expected_dim: usize) -> VaultResult<()> {
    if vector.len() != expected_dim {
        return Err(VaultError::Embedding(format!(
            "embedding dimension mismatch: expected {expected_dim}, got {}",
            vector.len()
        )));
    }
    if vector.iter().any(|value| !value.is_finite()) {
        return Err(VaultError::Embedding(
            "embedding vector contains a non-finite value".into(),
        ));
    }
    Ok(())
}

pub(crate) fn prepared_text_hash(text: &str) -> [u8; 32] {
    Sha256::digest(text.as_bytes()).into()
}

#[cfg(feature = "embeddings-api")]
fn endpoint_fingerprint(base_url: &str) -> [u8; 32] {
    let normalized = base_url.trim().trim_end_matches('/');
    Sha256::digest(normalized.as_bytes()).into()
}

#[cfg(feature = "embeddings-api")]
fn short_fingerprint(fingerprint: &[u8; 32]) -> String {
    fingerprint[..6]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

// ── EmbeddingBackend ───────────────────────────────────────────────────

#[cfg(feature = "embeddings")]
const LOCAL_MAX_REQUEST_INPUTS: usize = 32;

enum EmbeddingBackend {
    #[cfg(feature = "embeddings")]
    Local(Box<std::sync::Mutex<fastembed::TextEmbedding>>),

    #[cfg(feature = "embeddings-api")]
    Api(ApiEmbeddingClient),
}

#[cfg(feature = "embeddings-api")]
struct ApiEmbeddingRequest {
    texts: Vec<String>,
    retry_deadline: Option<std::time::Instant>,
    response: std::sync::mpsc::Sender<VaultResult<Vec<Vec<f32>>>>,
}

#[cfg(feature = "embeddings-api")]
struct ApiEmbeddingClient {
    sender: Option<std::sync::mpsc::SyncSender<ApiEmbeddingRequest>>,
    workers: Vec<std::thread::JoinHandle<()>>,
    timeout: std::time::Duration,
}

#[cfg(feature = "embeddings-api")]
impl ApiEmbeddingClient {
    const WORKER_COUNT: usize = 2;
    const QUEUE_CAPACITY: usize = 2;

    fn start(
        client: reqwest::blocking::Client,
        base_url: String,
        model: String,
        api_key: zeroize::Zeroizing<String>,
        timeout: std::time::Duration,
    ) -> VaultResult<Self> {
        let (sender, receiver) =
            std::sync::mpsc::sync_channel::<ApiEmbeddingRequest>(Self::QUEUE_CAPACITY);
        let receiver = Arc::new(std::sync::Mutex::new(receiver));
        let base_url = Arc::new(base_url);
        let model = Arc::new(model);
        let api_key = Arc::new(api_key);
        let mut workers: Vec<std::thread::JoinHandle<()>> = Vec::with_capacity(Self::WORKER_COUNT);
        for worker_index in 0..Self::WORKER_COUNT {
            let client = client.clone();
            let receiver = Arc::clone(&receiver);
            let base_url = Arc::clone(&base_url);
            let model = Arc::clone(&model);
            let api_key = Arc::clone(&api_key);
            let worker = match std::thread::Builder::new()
                .name(format!("obsidian-mcp-embedding-api-{worker_index}"))
                .spawn(move || {
                    loop {
                        let request = receiver
                            .lock()
                            .unwrap_or_else(|error| error.into_inner())
                            .recv();
                        let Ok(request) = request else {
                            break;
                        };
                        let texts = request.texts.iter().map(String::as_str).collect::<Vec<_>>();
                        let result = if request.retry_deadline.is_some() {
                            embed_batch_api_with_policy(
                                &client,
                                base_url.as_str(),
                                model.as_str(),
                                api_key.as_str(),
                                &texts,
                                request.retry_deadline,
                            )
                        } else {
                            embed_batch_api(
                                &client,
                                base_url.as_str(),
                                model.as_str(),
                                api_key.as_str(),
                                &texts,
                            )
                        };
                        let _ = request.response.send(result);
                    }
                }) {
                Ok(worker) => worker,
                Err(error) => {
                    drop(sender);
                    for worker in workers {
                        let _ = worker.join();
                    }
                    return Err(VaultError::Embedding(format!(
                        "failed to start embedding API request worker: {error}"
                    )));
                }
            };
            workers.push(worker);
        }
        Ok(Self {
            sender: Some(sender),
            workers,
            timeout,
        })
    }

    fn embed_batch(&self, texts: &[&str]) -> VaultResult<Vec<Vec<f32>>> {
        self.request(texts, false)
    }

    fn embed_with_retry(&self, texts: &[&str]) -> VaultResult<Vec<Vec<f32>>> {
        self.request(texts, true)
    }

    fn request(&self, texts: &[&str], retry_transient: bool) -> VaultResult<Vec<Vec<f32>>> {
        let sender = self
            .sender
            .as_ref()
            .ok_or_else(|| VaultError::Embedding("embedding API worker unavailable".into()))?;
        let retry_deadline = retry_transient.then(|| std::time::Instant::now() + self.timeout);
        let (response, receiver) = std::sync::mpsc::channel();
        let mut request = ApiEmbeddingRequest {
            texts: texts.iter().map(|text| (*text).to_string()).collect(),
            retry_deadline,
            response,
        };
        if let Some(deadline) = retry_deadline {
            loop {
                match sender.try_send(request) {
                    Ok(()) => break,
                    Err(std::sync::mpsc::TrySendError::Full(pending)) => {
                        request = pending;
                        let remaining =
                            deadline.saturating_duration_since(std::time::Instant::now());
                        if remaining.is_zero() {
                            return Err(api_retry_timeout());
                        }
                        std::thread::sleep(remaining.min(std::time::Duration::from_millis(1)));
                    }
                    Err(std::sync::mpsc::TrySendError::Disconnected(_)) => {
                        return Err(VaultError::Embedding("embedding API worker stopped".into()));
                    }
                }
            }
            receiver
                .recv_timeout(deadline.saturating_duration_since(std::time::Instant::now()))
                .map_err(|error| match error {
                    std::sync::mpsc::RecvTimeoutError::Timeout => api_retry_timeout(),
                    std::sync::mpsc::RecvTimeoutError::Disconnected => {
                        VaultError::Embedding("embedding API worker stopped".into())
                    }
                })?
        } else {
            sender
                .send(request)
                .map_err(|_| VaultError::Embedding("embedding API worker stopped".into()))?;
            receiver
                .recv()
                .map_err(|_| VaultError::Embedding("embedding API worker stopped".into()))?
        }
    }
}

#[cfg(feature = "embeddings-api")]
impl Drop for ApiEmbeddingClient {
    fn drop(&mut self) {
        self.sender.take();
        for worker in self.workers.drain(..) {
            if worker.join().is_err() {
                tracing::warn!("embedding API request worker panicked during shutdown");
            }
        }
    }
}

// ── EmbeddingModel ─────────────────────────────────────────────────────

/// Backend-agnostic embedding model supporting local fastembed and
/// OpenAI-compatible API backends.
pub struct EmbeddingModel {
    backend: EmbeddingBackend,
    dim: usize,
    identity: EmbeddingSpaceIdentity,
    input_policy: EmbeddingInputPolicy,
    request_limits: EmbeddingRequestLimits,
    counter: TokenCounter,
    request_overhead: usize,
}

impl EmbeddingModel {
    /// Load an embedding model using the specified (or inferred) backend.
    ///
    /// `provider` selects the backend explicitly; `None` infers from compiled
    /// features (local preferred when both are available).
    pub async fn load(model_name: &str, provider: Option<EmbeddingProvider>) -> VaultResult<Self> {
        match resolve_provider(provider) {
            EmbeddingProvider::Local => Self::load_local(model_name).await,
            EmbeddingProvider::Api => Self::load_api(model_name).await,
        }
    }

    /// Embed a batch of texts. Returns one vector per input text.
    pub fn embed_batch(&self, texts: &[&str]) -> VaultResult<Vec<Vec<f32>>> {
        <Self as Embedder>::embed_batch(self, texts)
    }

    /// Embed a single text. Convenience wrapper over `embed_batch`.
    pub fn embed_one(&self, text: &str) -> VaultResult<Vec<f32>> {
        let mut results = self.embed_batch(&[text])?;
        results
            .pop()
            .ok_or_else(|| VaultError::Embedding("embed returned empty result".into()))
    }

    /// Embedding dimensionality for the loaded model.
    pub fn dim(&self) -> usize {
        self.dim
    }

    fn embed_with_policy(
        &self,
        texts: &[&str],
        _retry_transient: bool,
    ) -> VaultResult<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        embedding_input::validate_batch(
            texts,
            &self.input_policy,
            self.request_limits,
            self.request_overhead,
            |text| self.counter.count(text),
        )?;
        self.dispatch_validated(texts, _retry_transient)
    }

    fn dispatch_validated(
        &self,
        texts: &[&str],
        _retry_transient: bool,
    ) -> VaultResult<Vec<Vec<f32>>> {
        let vectors = match &self.backend {
            #[cfg(feature = "embeddings")]
            EmbeddingBackend::Local(inner) => {
                let mut model = inner
                    .lock()
                    .map_err(|e| VaultError::Embedding(format!("model lock poisoned: {e}")))?;
                model
                    .embed(texts, Some(LOCAL_MAX_REQUEST_INPUTS))
                    .map_err(|error| VaultError::EmbeddingRequest {
                        message: format!("local embedding inference failed: {error}"),
                        retryable: true,
                        retry_after: None,
                    })?
            }
            #[cfg(feature = "embeddings-api")]
            EmbeddingBackend::Api(client) => {
                if _retry_transient {
                    client.embed_with_retry(texts)?
                } else {
                    client.embed_batch(texts)?
                }
            }
        };
        validate_embedding_batch(vectors, texts.len(), self.dim)
    }

    // ── Local backend (fastembed) ──────────────────────────────────────

    #[cfg(feature = "embeddings")]
    async fn load_local(model_name: &str) -> VaultResult<Self> {
        let model_name = model_name.to_owned();

        tokio::task::spawn_blocking(move || {
            let (model_enum, canonical_model, dim) = resolve_local_model(&model_name)?;
            let options = fastembed::InitOptions::new(model_enum).with_show_download_progress(true);

            let mut inner = fastembed::TextEmbedding::try_new(options)
                .map_err(|e| VaultError::Embedding(format!("model load failed: {e}")))?;
            let max_input_tokens = inner
                .tokenizer
                .get_truncation()
                .map(|params| params.max_length)
                .ok_or_else(|| {
                    VaultError::EmbeddingConfiguration(
                        "local embedding tokenizer has no verified input capacity".into(),
                    )
                })?;
            let (counter, tokenizer_fingerprint) =
                TokenCounter::hugging_face(inner.tokenizer.clone(), true)?;
            inner.tokenizer.with_truncation(None).map_err(|error| {
                VaultError::EmbeddingConfiguration(format!(
                    "cannot disable inference truncation: {error}"
                ))
            })?;
            let input_policy = EmbeddingInputPolicy {
                max_input_tokens,
                tokenizer_fingerprint,
            };
            let identity = EmbeddingSpaceIdentity::local(canonical_model, dim, &input_policy);
            validate_space_identity(&identity)?;
            let request_limits = local_request_limits(max_input_tokens)?;

            Ok(Self {
                backend: EmbeddingBackend::Local(Box::new(std::sync::Mutex::new(inner))),
                dim,
                identity,
                input_policy,
                request_limits,
                counter,
                request_overhead: 2,
            })
        })
        .await
        .map_err(|e| VaultError::Embedding(format!("spawn_blocking join error: {e}")))?
    }

    #[cfg(not(feature = "embeddings"))]
    async fn load_local(_model_name: &str) -> VaultResult<Self> {
        Err(VaultError::Embedding(
            "local embedding backend not compiled (needs --features embeddings)".into(),
        ))
    }

    // ── API backend (OpenAI-compatible) ────────────────────────────────

    #[cfg(feature = "embeddings-api")]
    async fn load_api(model_name: &str) -> VaultResult<Self> {
        let model_name = model_name.to_owned();
        let timeout =
            crate::config::embedding_api_timeout_from_env().map_err(VaultError::Embedding)?;

        tokio::task::spawn_blocking(move || {
            let api_key = zeroize::Zeroizing::new(
                read_env_with_fallback("OBSIDIAN_EMBEDDING_API_KEY", "OPENAI_API_KEY").ok_or_else(
                    || {
                        VaultError::Embedding(
                            "API key required: set OBSIDIAN_EMBEDDING_API_KEY or OPENAI_API_KEY"
                                .into(),
                        )
                    },
                )?,
            );

            let base_url = embedding_input::normalize_api_base(
                &read_env_with_fallback("OBSIDIAN_EMBEDDING_API_BASE", "OPENAI_BASE_URL")
                    .unwrap_or_else(|| "https://api.openai.com/v1".to_string()),
            )?;

            let model = read_env_with_fallback("OBSIDIAN_EMBEDDING_API_MODEL", "OPENAI_MODEL")
                .unwrap_or(model_name);
            let contract = embedding_input::ApiInputContract::from_env(&base_url, &model)?;
            let client = build_api_client(timeout)?;
            let api_client = ApiEmbeddingClient::start(
                client,
                base_url.clone(),
                model.clone(),
                api_key,
                timeout,
            )?;
            Self::from_api_client(
                api_client,
                base_url,
                model,
                contract,
                parse_usize_env("OBSIDIAN_EMBEDDING_DIM"),
            )
        })
        .await
        .map_err(|e| VaultError::Embedding(format!("spawn_blocking join error: {e}")))?
    }

    #[cfg(feature = "embeddings-api")]
    fn from_api_client(
        api_client: ApiEmbeddingClient,
        base_url: String,
        model: String,
        contract: embedding_input::ApiInputContract,
        configured_dimension: Option<usize>,
    ) -> VaultResult<Self> {
        let request_overhead = serde_json::to_vec(
            &serde_json::json!({"model":model,"input":[],"encoding_format":"float"}),
        )
        .map_err(|error| {
            VaultError::Embedding(format!("cannot encode embedding request: {error}"))
        })?
        .len();
        embedding_input::validate_batch(
            &["dim"],
            &contract.policy,
            contract.limits,
            request_overhead,
            |text| contract.counter.count(text),
        )?;

        let dim = match configured_dimension {
            Some(0) => {
                return Err(VaultError::Embedding(
                    "OBSIDIAN_EMBEDDING_DIM must be greater than zero".into(),
                ));
            }
            Some(d) => {
                tracing::info!(dim = d, "using explicit embedding dimension");
                d
            }
            None => {
                tracing::info!("probing embedding API for dimension…");
                probe_api_dimension(&api_client)?
            }
        };
        let identity = EmbeddingSpaceIdentity::api(model.clone(), &base_url, dim, &contract.policy);
        validate_space_identity(&identity)?;
        let endpoint = identity
            .endpoint_fingerprint
            .as_ref()
            .map(short_fingerprint)
            .unwrap_or_default();

        tracing::info!(
            endpoint_fingerprint = %endpoint,
            model = %model,
            dim,
            "API embedding backend ready"
        );

        Ok(Self {
            backend: EmbeddingBackend::Api(api_client),
            dim,
            identity,
            input_policy: contract.policy,
            request_limits: contract.limits,
            counter: contract.counter,
            request_overhead,
        })
    }

    #[cfg(not(feature = "embeddings-api"))]
    async fn load_api(_model_name: &str) -> VaultResult<Self> {
        Err(VaultError::Embedding(
            "API embedding backend not compiled (needs --features embeddings-api)".into(),
        ))
    }
}

impl Embedder for EmbeddingModel {
    fn dimension(&self) -> usize {
        self.dim
    }

    fn space_identity(&self) -> &EmbeddingSpaceIdentity {
        &self.identity
    }

    fn input_policy(&self) -> &EmbeddingInputPolicy {
        &self.input_policy
    }

    fn count_tokens(&self, text: &str) -> VaultResult<usize> {
        self.counter.count(text)
    }

    fn request_limits(&self) -> EmbeddingRequestLimits {
        self.request_limits
    }

    fn plan_batch(&self, texts: &[&str]) -> VaultResult<usize> {
        embedding_input::plan_batch(
            texts,
            &self.input_policy,
            self.request_limits,
            self.request_overhead,
            |text| self.counter.count(text),
        )
    }

    fn embed_batch(&self, texts: &[&str]) -> VaultResult<Vec<Vec<f32>>> {
        self.embed_with_policy(texts, false)
    }

    fn embed_next_batch(&self, texts: &[&str]) -> (usize, VaultResult<Vec<Vec<f32>>>) {
        let count = match self.plan_batch(texts) {
            Ok(count) if count > 0 && count <= texts.len() => count,
            Ok(_) => {
                return (
                    1,
                    Err(VaultError::Embedding(
                        "embedding request planner did not make progress".into(),
                    )),
                );
            }
            Err(error) => return (1, Err(error)),
        };
        (count, self.dispatch_validated(&texts[..count], false))
    }

    fn embed_query(&self, query: &str) -> VaultResult<Vec<f32>> {
        let mut vectors = self.embed_with_policy(&[query], true)?;
        vectors
            .pop()
            .ok_or_else(|| VaultError::Embedding("embedding query returned no vector".into()))
    }
}

#[cfg(feature = "embeddings")]
fn local_request_limits(max_input_tokens: usize) -> VaultResult<EmbeddingRequestLimits> {
    let max_total_tokens = max_input_tokens
        .checked_mul(LOCAL_MAX_REQUEST_INPUTS)
        .ok_or_else(|| {
            VaultError::EmbeddingConfiguration("local embedding batch capacity overflow".into())
        })?;
    Ok(EmbeddingRequestLimits {
        max_inputs: LOCAL_MAX_REQUEST_INPUTS,
        max_total_tokens,
        max_request_bytes: embedding_input::MAX_EMBEDDING_REQUEST_BYTES,
    })
}

#[cfg(feature = "embeddings")]
fn resolve_local_model(
    configured_name: &str,
) -> VaultResult<(fastembed::EmbeddingModel, String, usize)> {
    let configured_name = configured_name.trim();
    let model = match configured_name.parse().ok() {
        Some(model) => model,
        None => {
            let supported = fastembed::TextEmbedding::list_supported_models();
            let mut matches = supported
                .iter()
                .filter(|info| info.model_code.eq_ignore_ascii_case(configured_name))
                .map(|info| info.model.clone())
                .collect::<Vec<_>>();
            if matches.is_empty()
                && let Some(repository_name) = configured_name.split_once('/').map(|(_, name)| name)
            {
                matches = supported
                    .iter()
                    .filter(|info| {
                        info.model_code
                            .split_once('/')
                            .is_some_and(|(_, name)| name.eq_ignore_ascii_case(repository_name))
                    })
                    .map(|info| info.model.clone())
                    .collect();
            }
            match matches.as_slice() {
                [model] => model.clone(),
                [] => {
                    return Err(VaultError::Embedding(format!(
                        "unknown local embedding model '{configured_name}'"
                    )));
                }
                _ => {
                    return Err(VaultError::Embedding(format!(
                        "ambiguous local embedding model '{configured_name}'; use a fastembed model enum name"
                    )));
                }
            }
        }
    };
    let info = fastembed::TextEmbedding::get_model_info(&model).map_err(|error| {
        VaultError::Embedding(format!(
            "embedding metadata unavailable for local model '{configured_name}': {error}"
        ))
    })?;
    let canonical_model = format!("{model:?}");
    let dimension = info.dim;
    Ok((model, canonical_model, dimension))
}

// ── Provider resolution ────────────────────────────────────────────────

fn resolve_provider(explicit: Option<EmbeddingProvider>) -> EmbeddingProvider {
    if let Some(p) = explicit {
        return p;
    }

    let has_local = cfg!(feature = "embeddings");
    let has_api = cfg!(feature = "embeddings-api");

    match (has_local, has_api) {
        (true, _) => EmbeddingProvider::Local,
        (false, true) => EmbeddingProvider::Api,
        (false, false) => unreachable!("embeddings module compiled without any backend"),
    }
}

// ── API client helpers ─────────────────────────────────────────────────

#[cfg(feature = "embeddings-api")]
fn build_api_client(timeout: std::time::Duration) -> Result<reqwest::blocking::Client, VaultError> {
    let mut builder = reqwest::blocking::ClientBuilder::new().timeout(timeout);

    if let Ok(cert_path) = std::env::var("OBSIDIAN_EMBEDDING_CA_CERT") {
        let cert_pem = std::fs::read(&cert_path).map_err(|e| {
            VaultError::Embedding(format!("failed to read CA cert {cert_path}: {e}"))
        })?;
        let cert = reqwest::Certificate::from_pem(&cert_pem)
            .map_err(|e| VaultError::Embedding(format!("invalid CA cert: {e}")))?;
        builder = builder.add_root_certificate(cert);
    }

    if std::env::var("OBSIDIAN_EMBEDDING_TLS_VERIFY")
        .map(|v| v.eq_ignore_ascii_case("false") || v == "0")
        .unwrap_or(false)
    {
        tracing::warn!(
            "TLS verification disabled for embedding API — NOT recommended for production"
        );
        builder = builder.danger_accept_invalid_certs(true);
    }

    builder
        .build()
        .map_err(|e| VaultError::Embedding(format!("failed to build HTTP client: {e}")))
}

#[cfg(feature = "embeddings-api")]
fn probe_api_dimension(client: &ApiEmbeddingClient) -> Result<usize, VaultError> {
    let vecs = client.embed_with_retry(&["dim"])?;
    let first = vecs
        .first()
        .ok_or_else(|| VaultError::Embedding("dimension probe returned empty result".into()))?;
    if first.is_empty() {
        return Err(VaultError::Embedding(
            "dimension probe returned zero-length vector".into(),
        ));
    }
    Ok(first.len())
}

#[cfg(feature = "embeddings-api")]
fn embed_batch_api(
    client: &reqwest::blocking::Client,
    base_url: &str,
    model: &str,
    api_key: &str,
    texts: &[&str],
) -> Result<Vec<Vec<f32>>, VaultError> {
    embed_batch_api_with_policy(client, base_url, model, api_key, texts, None)
}

#[cfg(feature = "embeddings-api")]
fn api_retry_timeout() -> VaultError {
    VaultError::EmbeddingRequest {
        message: "embedding API request timed out within its retry budget".into(),
        retryable: true,
        retry_after: None,
    }
}

#[cfg(feature = "embeddings-api")]
fn embed_batch_api_with_policy(
    client: &reqwest::blocking::Client,
    base_url: &str,
    model: &str,
    api_key: &str,
    texts: &[&str],
    retry_deadline: Option<std::time::Instant>,
) -> Result<Vec<Vec<f32>>, VaultError> {
    let url = format!("{}/embeddings", base_url.trim_end_matches('/'));
    let body = serde_json::json!({
        "model": model,
        "input": texts,
        "encoding_format": "float",
    });

    let mut attempts = 0;
    loop {
        let mut request = client
            .post(&url)
            .header("Authorization", format!("Bearer {api_key}"))
            .json(&body);
        if let Some(deadline) = retry_deadline {
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if remaining.is_zero() {
                return Err(api_retry_timeout());
            }
            request = request.timeout(remaining);
        }
        attempts += 1;
        let (mut message, retryable, retry_after) = match request.send() {
            Ok(response) => {
                let status = response.status();
                if status.is_success() {
                    match response.bytes() {
                        Ok(bytes) => {
                            let resp: serde_json::Value =
                                serde_json::from_slice(&bytes).map_err(|_| {
                                    VaultError::Embedding(
                                        "embedding API returned invalid JSON".into(),
                                    )
                                })?;
                            return parse_embedding_response(&resp, texts.len());
                        }
                        Err(error) => {
                            let detail = if error.is_timeout() {
                                "response body timed out"
                            } else {
                                "response body read failed"
                            };
                            (format!("embedding API {detail}"), true, None)
                        }
                    }
                } else {
                    let retry_after = response
                        .headers()
                        .get("retry-after")
                        .and_then(|value| value.to_str().ok())
                        .and_then(|value| value.parse::<u64>().ok())
                        .map(std::time::Duration::from_secs);
                    (
                        format!("embedding API returned HTTP status {status}"),
                        status.as_u16() == 429 || status.is_server_error(),
                        retry_after,
                    )
                }
            }
            Err(error) => {
                let detail = if error.is_timeout() {
                    "request timed out"
                } else if error.is_connect() {
                    "connection failed"
                } else if error.is_builder() {
                    "request could not be constructed"
                } else {
                    "request failed"
                };
                (format!("embedding API {detail}"), !error.is_builder(), None)
            }
        };
        if retryable
            && attempts <= 3
            && let Some(deadline) = retry_deadline
        {
            let wait =
                retry_after.unwrap_or_else(|| std::time::Duration::from_secs(1 << (attempts - 1)));
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if wait < remaining {
                std::thread::sleep(wait);
                continue;
            }
            message.push_str("; retry deadline cannot accommodate another attempt");
        }
        if attempts > 1 {
            message.push_str(&format!(" after {attempts} attempts"));
        }
        return Err(VaultError::EmbeddingRequest {
            message,
            retryable,
            retry_after,
        });
    }
}

/// Parse an OpenAI-compatible embedding API response into embedding vectors.
///
/// Providers may either omit every `index` and preserve array order, or include
/// a complete unique index set. Mixed or partial responses are rejected.
#[cfg(feature = "embeddings-api")]
fn parse_embedding_response(
    resp: &serde_json::Value,
    expected_count: usize,
) -> Result<Vec<Vec<f32>>, VaultError> {
    let data = resp["data"]
        .as_array()
        .ok_or_else(|| VaultError::Embedding("missing 'data' array in API response".into()))?;
    if data.len() != expected_count {
        return Err(VaultError::Embedding(format!(
            "embedding API returned {} vectors for {expected_count} inputs",
            data.len()
        )));
    }

    let indexed_response = data.first().is_some_and(|item| !item["index"].is_null());

    let mut indexed: Vec<(usize, Vec<f32>)> = Vec::with_capacity(data.len());
    for (array_pos, item) in data.iter().enumerate() {
        let has_index = !item["index"].is_null();
        if has_index != indexed_response {
            return Err(VaultError::Embedding(
                "embedding API returned mixed indexed and unindexed items".into(),
            ));
        }
        let idx = if indexed_response {
            let raw = item["index"].as_u64().ok_or_else(|| {
                VaultError::Embedding("embedding response index must be an unsigned integer".into())
            })?;
            usize::try_from(raw).map_err(|_| {
                VaultError::Embedding("embedding response index is out of range".into())
            })?
        } else {
            array_pos
        };
        if idx >= expected_count {
            return Err(VaultError::Embedding(format!(
                "embedding response index {idx} is out of range for {expected_count} inputs"
            )));
        }
        let vec = item["embedding"]
            .as_array()
            .ok_or_else(|| {
                VaultError::Embedding("missing 'embedding' array in response item".into())
            })?
            .iter()
            .map(|v| {
                v.as_f64()
                    .ok_or_else(|| {
                        VaultError::Embedding("non-numeric value in embedding vector".into())
                    })
                    .and_then(|f| {
                        let value = f as f32;
                        value.is_finite().then_some(value).ok_or_else(|| {
                            VaultError::Embedding("non-finite value in embedding vector".into())
                        })
                    })
            })
            .collect::<Result<Vec<f32>, _>>()?;
        indexed.push((idx, vec));
    }

    if indexed_response {
        indexed.sort_unstable_by_key(|(idx, _)| *idx);
        for (expected, (actual, _)) in indexed.iter().enumerate() {
            if *actual != expected {
                return Err(VaultError::Embedding(format!(
                    "embedding response indices are not unique and contiguous: expected {expected}, got {actual}"
                )));
            }
        }
    }
    Ok(indexed.into_iter().map(|(_, vec)| vec).collect())
}

pub(crate) fn validate_embedding_batch(
    vectors: Vec<Vec<f32>>,
    expected_count: usize,
    expected_dim: usize,
) -> VaultResult<Vec<Vec<f32>>> {
    if vectors.len() != expected_count {
        return Err(VaultError::Embedding(format!(
            "embedding backend returned {} vectors for {expected_count} inputs",
            vectors.len()
        )));
    }
    for vector in &vectors {
        validate_vector(vector, expected_dim)?;
    }
    Ok(vectors)
}

// ── Env var helpers (API backend) ──────────────────────────────────────

#[cfg(feature = "embeddings-api")]
fn read_env_with_fallback(primary: &str, fallback: &str) -> Option<String> {
    let read_trimmed = |var: &str| {
        std::env::var(var)
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
    };
    read_trimmed(primary).or_else(|| read_trimmed(fallback))
}

#[cfg(feature = "embeddings-api")]
fn parse_usize_env(var_name: &str) -> Option<usize> {
    std::env::var(var_name).ok()?.trim().parse::<usize>().ok()
}

// ── Legacy cache relocation ────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LegacyCacheMigration {
    NotFound,
    AlreadyPresent(PathBuf),
    Migrated(PathBuf),
}

pub fn migrate_legacy_cache_to_daemon_store(
    vault_root: &Path,
    semantic_home: &Path,
) -> VaultResult<LegacyCacheMigration> {
    let vault_id = crate::daemon::home::compute_vault_id(vault_root)?;
    let target = semantic_home
        .join("vaults")
        .join(vault_id)
        .join("embeddings.bin");
    let legacy_source = vault_root
        .join(".obsidian")
        .join("obsidian-mcp")
        .join("embeddings.bin");
    let new_source = vault_root
        .join(".obsidian-mcp")
        .join("embeddings")
        .join("embeddings.bin");

    migrate_cache_candidates_to_path(&[new_source, legacy_source], &target)
}

pub(crate) fn migrate_cache_candidates_to_path(
    sources: &[PathBuf],
    target: &Path,
) -> VaultResult<LegacyCacheMigration> {
    if target.exists() {
        return Ok(LegacyCacheMigration::AlreadyPresent(target.to_path_buf()));
    }

    let Some(source) = sources.iter().find(|source| source.is_file()) else {
        return Ok(LegacyCacheMigration::NotFound);
    };

    let source_file = std::fs::File::open(source)?;
    let source_len = source_file.metadata()?.len();
    if source_len > MAX_CACHE_BYTES {
        return Err(VaultError::Embedding(format!(
            "embedding cache is too large to relocate: {source_len} bytes (limit {MAX_CACHE_BYTES})"
        )));
    }

    let parent = target.parent().ok_or_else(|| {
        VaultError::Embedding(format!(
            "embedding cache migration target has no parent: {}",
            target.display()
        ))
    })?;
    std::fs::create_dir_all(parent)?;

    let mut temp = tempfile::NamedTempFile::new_in(parent)?;
    let mut limited_source = source_file.take(MAX_CACHE_BYTES + 1);
    let copied = std::io::copy(&mut limited_source, &mut temp)?;
    if copied > MAX_CACHE_BYTES {
        return Err(VaultError::Embedding(format!(
            "embedding cache is too large to relocate: more than {MAX_CACHE_BYTES} bytes"
        )));
    }
    temp.flush()?;
    temp.as_file().sync_all()?;

    match temp.persist_noclobber(target) {
        Ok(_) => Ok(LegacyCacheMigration::Migrated(target.to_path_buf())),
        Err(error)
            if error.error.kind() == std::io::ErrorKind::AlreadyExists || target.exists() =>
        {
            Ok(LegacyCacheMigration::AlreadyPresent(target.to_path_buf()))
        }
        Err(error) => Err(VaultError::Io(error.error)),
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── cosine_similarity ──────────────────────────────────────────

    #[test]
    fn cosine_similarity_self_is_one() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "self-similarity should be 1.0, got {sim}"
        );
    }

    #[test]
    fn cosine_similarity_orthogonal_is_zero() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(
            sim.abs() < 1e-6,
            "orthogonal vectors should have similarity ~0, got {sim}"
        );
    }

    #[test]
    fn cosine_similarity_opposite_is_negative() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(
            (sim + 1.0).abs() < 1e-6,
            "opposite vectors should be -1.0, got {sim}"
        );
    }

    #[test]
    fn cosine_similarity_zero_vector_returns_zero() {
        let a = vec![1.0, 2.0];
        let zero = vec![0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &zero), 0.0);
        assert_eq!(cosine_similarity(&zero, &a), 0.0);
    }

    // ── EmbeddingStore ─────────────────────────────────────────────

    fn test_identity(dim: usize) -> EmbeddingSpaceIdentity {
        EmbeddingSpaceIdentity {
            backend: EmbeddingBackendKind::Local,
            model: "test-model".to_string(),
            endpoint_fingerprint: None,
            dimension: dim,
            input_version: EMBEDDING_INPUT_VERSION,
            tokenizer_fingerprint: [0; 32],
            max_input_tokens: 512,
            chunking_version: embedding_input::CHUNKING_VERSION,
        }
    }

    fn make_store() -> EmbeddingStore {
        let mut store = EmbeddingStore::new_with_identity(test_identity(3));
        store
            .insert_hashed(
                PathBuf::from("a.md"),
                prepared_text_hash("a"),
                vec![1.0, 0.0, 0.0],
            )
            .unwrap();
        store
            .insert_hashed(
                PathBuf::from("b.md"),
                prepared_text_hash("b"),
                vec![0.0, 1.0, 0.0],
            )
            .unwrap();
        store
            .insert_hashed(
                PathBuf::from("c.md"),
                prepared_text_hash("c"),
                vec![0.7, 0.7, 0.0],
            )
            .unwrap();
        store.set_first_pass_complete(true);
        store
    }

    fn write_cache_data(path: &Path, data: &EmbeddingCacheData) {
        let mut writer = CacheWriter::default();
        write_header(
            &mut writer,
            data.magic,
            data.schema_version,
            &data.identity,
            data.first_pass_complete,
            data.entries.len(),
        )
        .unwrap();
        for entry in &data.entries {
            let note = fixture_note(
                Path::new(&entry.path),
                entry.content_hash,
                entry.vector.clone(),
            );
            write_entry(&mut writer, &entry.path, &note).unwrap();
        }
        std::fs::write(path, writer.bytes).unwrap();
    }

    fn single_entry_cache(
        identity: EmbeddingSpaceIdentity,
        path: &str,
        vector: Vec<f32>,
    ) -> EmbeddingCacheData {
        EmbeddingCacheData {
            magic: CACHE_MAGIC,
            schema_version: CACHE_SCHEMA_VERSION,
            identity,
            first_pass_complete: true,
            entries: vec![EmbeddingCacheEntry {
                path: path.to_string(),
                content_hash: prepared_text_hash("cached text"),
                vector,
            }],
        }
    }

    #[test]
    fn query_returns_top_k_sorted() {
        let store = make_store();
        let query = vec![1.0, 0.0, 0.0];
        let results = store.query(&query, 2);

        assert_eq!(results.len(), 2);
        assert_eq!(
            results[0].0,
            PathBuf::from("a.md"),
            "exact match should rank first"
        );
        assert!(
            results[0].1 > results[1].1,
            "results should be sorted by descending score"
        );
    }

    #[test]
    fn query_top_k_exceeding_store_size() {
        let store = make_store();
        let query = vec![1.0, 0.0, 0.0];
        let results = store.query(&query, 100);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn query_paths_ranks_only_authoritative_members() {
        let store = make_store();
        let allowed = HashSet::from([PathBuf::from("b.md"), PathBuf::from("c.md")]);
        let results = store.query_paths(&[1.0, 0.0, 0.0], &allowed, 10);

        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|(path, _)| allowed.contains(path)));
        assert_eq!(results[0].0, PathBuf::from("c.md"));
    }

    #[test]
    fn insert_remove_updates_results() {
        let mut store = make_store();
        assert_eq!(store.len(), 3);

        store.remove(Path::new("a.md")).unwrap();
        assert_eq!(store.len(), 2);
        assert!(store.get(Path::new("a.md")).is_none());

        let query = vec![1.0, 0.0, 0.0];
        let results = store.query(&query, 10);
        assert!(!results.iter().any(|(p, _)| p == Path::new("a.md")));

        store.insert(PathBuf::from("d.md"), vec![0.9, 0.1, 0.0]);
        assert_eq!(store.len(), 3);
        let results = store.query(&query, 1);
        assert_eq!(results[0].0, PathBuf::from("d.md"));
    }

    #[test]
    fn get_returns_embedding() {
        let store = make_store();
        let vec = store.get(Path::new("a.md")).unwrap();
        assert_eq!(vec, &[1.0, 0.0, 0.0]);
        assert!(store.get(Path::new("nonexistent.md")).is_none());
    }

    #[test]
    fn persistence_roundtrip() {
        let store = make_store();
        let dir = tempfile::tempdir().unwrap();
        let cache_path = dir.path().join("embeddings.bin");

        store.save(&cache_path).unwrap();
        let loaded = EmbeddingStore::load(&cache_path).unwrap();

        assert_eq!(loaded.dim(), store.dim());
        assert_eq!(loaded.len(), store.len());
        assert_eq!(loaded.identity(), store.identity());
        assert!(loaded.first_pass_complete());
        assert_eq!(
            loaded.content_hash(Path::new("a.md")),
            store.content_hash(Path::new("a.md"))
        );

        let query = vec![1.0, 0.0, 0.0];
        let original_results = store.query(&query, 3);
        let loaded_results = loaded.query(&query, 3);

        assert_eq!(original_results.len(), loaded_results.len());
        for (orig, load) in original_results.iter().zip(&loaded_results) {
            assert_eq!(orig.0, load.0);
            assert!((orig.1 - load.1).abs() < 1e-6);
        }
    }

    #[test]
    fn empty_store_query() {
        let store = EmbeddingStore::new(3);
        assert!(store.is_empty());
        let results = store.query(&[1.0, 0.0, 0.0], 10);
        assert!(results.is_empty());
    }

    #[test]
    fn cache_rejects_trailing_bytes() {
        let store = make_store();
        let dir = tempfile::tempdir().unwrap();
        let cache_path = dir.path().join("embeddings.bin");
        store.save(&cache_path).unwrap();
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(&cache_path)
            .unwrap();
        file.write_all(b"trailing").unwrap();

        let error = EmbeddingStore::load(&cache_path).err().unwrap();
        assert!(error.to_string().contains("trailing bytes"));
    }

    #[test]
    fn cache_rejects_legacy_truncated_wrong_magic_and_wrong_schema() {
        let dir = tempfile::tempdir().unwrap();
        let cache_path = dir.path().join("embeddings.bin");

        std::fs::write(&cache_path, b"legacy cache").unwrap();
        assert!(EmbeddingStore::load(&cache_path).is_err());

        let valid = make_store().encode_cache().unwrap();
        std::fs::write(&cache_path, &valid[..valid.len() - 1]).unwrap();
        assert!(EmbeddingStore::load(&cache_path).is_err());

        let mut wrong_magic = single_entry_cache(test_identity(3), "one.md", vec![1.0; 3]);
        wrong_magic.magic = *b"NOTCACHE";
        write_cache_data(&cache_path, &wrong_magic);
        let error = EmbeddingStore::load(&cache_path).err().unwrap();
        assert!(error.to_string().contains("legacy embedding cache"));

        let mut wrong_schema = single_entry_cache(test_identity(3), "one.md", vec![1.0; 3]);
        wrong_schema.schema_version = CACHE_SCHEMA_VERSION + 1;
        write_cache_data(&cache_path, &wrong_schema);
        let error = EmbeddingStore::load(&cache_path).err().unwrap();
        assert!(error.to_string().contains("schema version"));
    }

    #[test]
    fn cache_rejects_invalid_and_non_normalized_paths() {
        let dir = tempfile::tempdir().unwrap();
        let cache_path = dir.path().join("embeddings.bin");
        let invalid_paths = [
            "",
            "/absolute.md",
            "../escape.md",
            "folder/../escape.md",
            "./note.md",
            "folder\\note.md",
            "folder//note.md",
        ];

        for path in invalid_paths {
            let data = single_entry_cache(test_identity(3), path, vec![1.0; 3]);
            write_cache_data(&cache_path, &data);
            assert!(
                EmbeddingStore::load(&cache_path).is_err(),
                "cache path should be rejected: {path:?}"
            );
        }

        let oversized_path = format!("{}.md", "a".repeat(MAX_CACHE_PATH_BYTES));
        let data = single_entry_cache(test_identity(3), &oversized_path, vec![1.0; 3]);
        write_cache_data(&cache_path, &data);
        assert!(EmbeddingStore::load(&cache_path).is_err());
    }

    #[test]
    fn cache_rejects_invalid_identity_and_vector_dimensions() {
        let dir = tempfile::tempdir().unwrap();
        let cache_path = dir.path().join("embeddings.bin");

        let invalid_identities = [
            EmbeddingSpaceIdentity {
                dimension: 0,
                ..test_identity(3)
            },
            EmbeddingSpaceIdentity {
                model: "  ".to_string(),
                ..test_identity(3)
            },
            EmbeddingSpaceIdentity {
                input_version: EMBEDDING_INPUT_VERSION + 1,
                ..test_identity(3)
            },
            EmbeddingSpaceIdentity {
                endpoint_fingerprint: Some([7; 32]),
                ..test_identity(3)
            },
            EmbeddingSpaceIdentity {
                backend: EmbeddingBackendKind::Api,
                endpoint_fingerprint: None,
                ..test_identity(3)
            },
        ];
        for identity in invalid_identities {
            let dim = identity.dimension.max(1);
            let data = single_entry_cache(identity, "one.md", vec![1.0; dim]);
            write_cache_data(&cache_path, &data);
            assert!(EmbeddingStore::load(&cache_path).is_err());
        }

        let data = single_entry_cache(test_identity(3), "one.md", vec![1.0; 2]);
        write_cache_data(&cache_path, &data);
        let error = EmbeddingStore::load(&cache_path).err().unwrap();
        assert!(error.to_string().contains("dimension mismatch"));
    }

    #[test]
    fn cache_policy_versions_require_a_rebuild_while_invalid_dimensions_remain_errors() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("cache.bin");
        let expected = test_identity(3);
        let mut input = expected.clone();
        input.input_version += 1;
        let mut chunking = expected.clone();
        chunking.chunking_version += 1;
        for identity in [input, chunking] {
            write_cache_data(&path, &single_entry_cache(identity, "one.md", vec![1.0; 3]));
            assert!(matches!(
                EmbeddingStore::load_for_space(&path, &expected, 1),
                Err(VaultError::EmbeddingCacheIncompatible(_))
            ));
            assert!(matches!(
                EmbeddingStore::load(&path),
                Err(VaultError::EmbeddingCacheIncompatible(_))
            ));
        }
        let invalid = EmbeddingSpaceIdentity {
            dimension: 0,
            ..expected.clone()
        };
        write_cache_data(&path, &single_entry_cache(invalid, "one.md", vec![1.0; 3]));
        assert!(matches!(
            EmbeddingStore::load_for_space(&path, &expected, 1),
            Err(VaultError::Embedding(_))
        ));
    }

    #[test]
    fn cache_load_is_bounded_by_current_vault_and_file_size() {
        let dir = tempfile::tempdir().unwrap();
        let cache_path = dir.path().join("embeddings.bin");
        let identity = test_identity(1);
        let entries = (0..1025)
            .map(|index| EmbeddingCacheEntry {
                path: format!("{index}.md"),
                content_hash: prepared_text_hash("cached text"),
                vector: vec![1.0],
            })
            .collect();
        write_cache_data(
            &cache_path,
            &EmbeddingCacheData {
                magic: CACHE_MAGIC,
                schema_version: CACHE_SCHEMA_VERSION,
                identity: identity.clone(),
                first_pass_complete: true,
                entries,
            },
        );
        let error = EmbeddingStore::load_for_space(&cache_path, &identity, 0)
            .err()
            .unwrap();
        assert!(error.to_string().contains("too many entries"));

        let file = std::fs::File::create(&cache_path).unwrap();
        file.set_len(MAX_CACHE_BYTES + 1).unwrap();
        let error = EmbeddingStore::load_for_space(&cache_path, &identity, 0)
            .err()
            .unwrap();
        assert!(error.to_string().contains("too large"));
    }

    #[test]
    fn failed_encoding_preserves_previous_cache_and_success_replaces_it() {
        let dir = tempfile::tempdir().unwrap();
        let cache_path = dir.path().join("embeddings.bin");
        let original = make_store();
        original.save(&cache_path).unwrap();
        let original_bytes = std::fs::read(&cache_path).unwrap();

        let mut invalid = EmbeddingStore::new_with_identity(test_identity(3));
        invalid
            .insert_hashed(
                PathBuf::from("../escape.md"),
                prepared_text_hash("bad"),
                vec![1.0, 0.0, 0.0],
            )
            .unwrap();
        assert!(invalid.save(&cache_path).is_err());
        assert_eq!(std::fs::read(&cache_path).unwrap(), original_bytes);

        let mut replacement = make_store();
        replacement
            .insert_hashed(
                PathBuf::from("a.md"),
                prepared_text_hash("replacement"),
                vec![0.0, 0.0, 1.0],
            )
            .unwrap();
        replacement.save(&cache_path).unwrap();
        let loaded = EmbeddingStore::load(&cache_path).unwrap();
        assert_eq!(loaded.get(Path::new("a.md")), Some(&[0.0, 0.0, 1.0][..]));
    }

    #[test]
    fn concurrent_readers_observe_only_complete_atomic_cache_snapshots() {
        let dir = tempfile::tempdir().unwrap();
        let cache_path = dir.path().join("embeddings.bin");
        let original = make_store();
        original.save(&cache_path).unwrap();

        let mut replacement = make_store();
        replacement
            .insert_hashed(
                PathBuf::from("a.md"),
                prepared_text_hash("replacement"),
                vec![0.0, 0.0, 1.0],
            )
            .unwrap();
        let start = std::sync::Arc::new(std::sync::Barrier::new(2));
        let writer_path = cache_path.clone();
        let writer_start = std::sync::Arc::clone(&start);
        let writer = std::thread::spawn(move || {
            writer_start.wait();
            for iteration in 0..20 {
                if iteration % 2 == 0 {
                    replacement.save(&writer_path).unwrap();
                } else {
                    original.save(&writer_path).unwrap();
                }
            }
        });

        start.wait();
        let mut reads = 0;
        loop {
            let loaded = EmbeddingStore::load(&cache_path).unwrap();
            let vector = loaded.get(Path::new("a.md")).unwrap();
            assert!(vector == [1.0, 0.0, 0.0] || vector == [0.0, 0.0, 1.0]);
            reads += 1;
            if writer.is_finished() {
                break;
            }
        }
        writer.join().unwrap();
        assert!(reads > 0);
    }

    #[test]
    fn cache_encoding_rejects_entries_without_hashes() {
        let mut store = EmbeddingStore::new_with_identity(test_identity(3));
        store.insert(PathBuf::from("one.md"), vec![1.0, 0.0, 0.0]);
        let error = store.encode_cache().unwrap_err();
        assert!(error.to_string().contains("prepared-text hash"));
    }

    #[test]
    fn cache_rejects_wrong_vector_space() {
        let store = make_store();
        let dir = tempfile::tempdir().unwrap();
        let cache_path = dir.path().join("embeddings.bin");
        store.save(&cache_path).unwrap();

        let mut expected = test_identity(3);
        expected.model = "different-model".to_string();
        let error = EmbeddingStore::load_for_space(&cache_path, &expected, 3)
            .err()
            .unwrap();
        assert!(error.to_string().contains("identity mismatch"));
    }

    #[test]
    fn every_vector_space_component_participates_in_cache_compatibility() {
        let store = make_store();
        let dir = tempfile::tempdir().unwrap();
        let cache_path = dir.path().join("embeddings.bin");
        store.save(&cache_path).unwrap();

        let mut mismatches = Vec::new();
        let mut model = test_identity(3);
        model.model = "other-model".to_string();
        mismatches.push(model);
        let mut dimension = test_identity(4);
        dimension.model = "test-model".to_string();
        mismatches.push(dimension);
        let mut input = test_identity(3);
        input.input_version += 1;
        mismatches.push(input);
        let mut tokenizer = test_identity(3);
        tokenizer.tokenizer_fingerprint = [1; 32];
        mismatches.push(tokenizer);
        let mut capacity = test_identity(3);
        capacity.max_input_tokens += 1;
        mismatches.push(capacity);
        let mut partition = test_identity(3);
        partition.chunking_version += 1;
        mismatches.push(partition);
        mismatches.push(EmbeddingSpaceIdentity {
            backend: EmbeddingBackendKind::Api,
            model: "test-model".to_string(),
            endpoint_fingerprint: Some([1; 32]),
            dimension: 3,
            input_version: EMBEDDING_INPUT_VERSION,
            tokenizer_fingerprint: [0; 32],
            max_input_tokens: 512,
            chunking_version: embedding_input::CHUNKING_VERSION,
        });

        for expected in mismatches {
            let error = EmbeddingStore::load_for_space(&cache_path, &expected, 3)
                .err()
                .unwrap();
            assert!(error.to_string().contains("identity mismatch"));
        }
    }

    #[test]
    fn cache_rejects_duplicate_exact_normalized_paths() {
        let dir = tempfile::tempdir().unwrap();
        let cache_path = dir.path().join("embeddings.bin");
        let data = EmbeddingCacheData {
            magic: CACHE_MAGIC,
            schema_version: CACHE_SCHEMA_VERSION,
            identity: test_identity(3),
            first_pass_complete: true,
            entries: vec![
                EmbeddingCacheEntry {
                    path: "note.md".to_string(),
                    content_hash: prepared_text_hash("one"),
                    vector: vec![1.0, 0.0, 0.0],
                },
                EmbeddingCacheEntry {
                    path: "note.md".to_string(),
                    content_hash: prepared_text_hash("two"),
                    vector: vec![0.0, 1.0, 0.0],
                },
            ],
        };
        write_cache_data(&cache_path, &data);

        let error = EmbeddingStore::load(&cache_path).err().unwrap();
        assert!(error.to_string().contains("duplicate path"));
    }

    #[test]
    fn admitted_native_unicode_spellings_round_trip_as_distinct_notes() {
        let decomposed = PathBuf::from("Cafe\u{301}.md");
        let composed = PathBuf::from("Caf\u{e9}.md");
        assert_ne!(decomposed, composed);
        let first_hash = prepared_text_hash("1");
        let second_hash = prepared_text_hash("2");
        let mut store = EmbeddingStore::new_with_identity(test_identity(3));
        store
            .insert_note(
                decomposed.clone(),
                fixture_note(&decomposed, first_hash, vec![1.0, 0.0, 0.0]),
            )
            .unwrap();
        store
            .insert_note(
                composed.clone(),
                fixture_note(&composed, second_hash, vec![0.0, 1.0, 0.0]),
            )
            .unwrap();
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("cache.bin");
        store.save(&path).unwrap();

        let loaded = EmbeddingStore::load_for_space(&path, &test_identity(3), 2).unwrap();
        assert_eq!(loaded.len(), 2);
        let first = loaded.note(&decomposed).unwrap();
        let second = loaded.note(&composed).unwrap();
        assert_eq!(
            first.source_revision,
            crate::vault::source::source_revision("1")
        );
        assert_eq!(
            second.source_revision,
            crate::vault::source::source_revision("2")
        );
        assert_ne!(first.source_revision, second.source_revision);
        assert_eq!(first.chunks[0].vector, [1.0, 0.0, 0.0]);
        assert_eq!(second.chunks[0].vector, [0.0, 1.0, 0.0]);
        assert_eq!(loaded.query(&[1.0, 0.0, 0.0], 1)[0].0, decomposed);
        assert_eq!(loaded.query(&[0.0, 1.0, 0.0], 1)[0].0, composed);
    }

    #[test]
    fn cache_rejects_non_finite_vectors() {
        let dir = tempfile::tempdir().unwrap();
        let cache_path = dir.path().join("embeddings.bin");
        let data = EmbeddingCacheData {
            magic: CACHE_MAGIC,
            schema_version: CACHE_SCHEMA_VERSION,
            identity: test_identity(3),
            first_pass_complete: true,
            entries: vec![EmbeddingCacheEntry {
                path: "bad.md".to_string(),
                content_hash: prepared_text_hash("bad"),
                vector: vec![1.0, f32::NAN, 0.0],
            }],
        };
        write_cache_data(&cache_path, &data);

        let error = EmbeddingStore::load(&cache_path).err().unwrap();
        assert!(error.to_string().contains("non-finite"));
    }

    #[test]
    fn cosine_similarity_large_finite_vectors_remains_finite() {
        let vector = [f32::MAX, f32::MAX / 2.0, -f32::MAX];
        assert_eq!(cosine_similarity(&vector, &vector), 1.0);
        let opposite = vector.map(|value| -value);
        assert_eq!(cosine_similarity(&vector, &opposite), -1.0);
    }

    fn passage_note() -> NoteEmbeddingEntry {
        let mut entry = fixture_note(
            Path::new("long.md"),
            prepared_text_hash("revision"),
            vec![1.0, 0.0, 0.0],
        );
        entry.source_bytes = 10;
        entry.body_range = SourceRange {
            start_byte: 2,
            end_byte: 10,
        };
        let first = &mut entry.chunks[0];
        first.scope = EvidenceScope::Passage;
        first.range = SourceRange {
            start_byte: 2,
            end_byte: 6,
        };
        first.heading_path = vec!["First".into()];
        first.section_range = Some(first.range);
        entry.chunks.push(ChunkEmbedding {
            input_hash: prepared_text_hash("second"),
            vector: vec![0.0, 1.0, 0.0],
            scope: EvidenceScope::Passage,
            range: SourceRange {
                start_byte: 6,
                end_byte: 10,
            },
            overlap_range: None,
            heading_path: vec!["Second".into()],
            section_range: Some(SourceRange {
                start_byte: 6,
                end_byte: 10,
            }),
            block_kind: Some("code".into()),
        });
        entry
    }

    #[test]
    fn complete_chunk_sets_round_trip_and_rank_once_per_note() {
        let mut store = EmbeddingStore::new_with_identity(test_identity(3));
        store.insert_note("long.md".into(), passage_note()).unwrap();
        store
            .insert_note(
                "short.md".into(),
                fixture_note(
                    Path::new("short.md"),
                    prepared_text_hash("short"),
                    vec![0.2, 0.8, 0.0],
                ),
            )
            .unwrap();
        assert_eq!(store.query(&[0.0, 1.0, 0.0], 2)[0].0, Path::new("long.md"));
        assert_eq!(store.query(&[0.0, 1.0, 0.0], 2).len(), 2);
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("cache.bin");
        store.save(&path).unwrap();
        let loaded = EmbeddingStore::load_for_space(&path, &test_identity(3), 2).unwrap();
        let note = loaded.note(Path::new("long.md")).unwrap();
        assert_eq!(note.body_range.start_byte, 2);
        assert_eq!(note.chunks.len(), 2);
        assert_eq!(note.chunks[1].block_kind.as_deref(), Some("code"));
        assert_eq!(note.best_chunk(&[0.0, 1.0, 0.0]), Some((1, 1.0)));
    }

    #[test]
    fn cache_context_admission_includes_title_separators_and_bounds_block_metadata_independently() {
        let identity = test_identity(3);
        let mut entry = passage_note();
        entry.chunks[0].heading_path = vec!["h".repeat(MAX_CONTEXT_BYTES - 6)];
        entry.chunks[0].block_kind = Some("paragraph".into());
        let mut store = EmbeddingStore::new_with_identity(identity.clone());
        store.insert_note("long.md".into(), entry.clone()).unwrap();
        entry.title.push('x');
        assert!(store.insert_note("long.md".into(), entry.clone()).is_err());
        assert_eq!(store.note(Path::new("long.md")).unwrap().title, "long");
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("cache.bin");
        let mut writer = CacheWriter::default();
        write_header(
            &mut writer,
            CACHE_MAGIC,
            CACHE_SCHEMA_VERSION,
            &identity,
            false,
            1,
        )
        .unwrap();
        write_entry(&mut writer, "long.md", &entry).unwrap();
        std::fs::write(&path, writer.bytes).unwrap();
        let error = EmbeddingStore::load_for_space(&path, &identity, 1)
            .err()
            .unwrap();
        assert_eq!(
            error.to_string(),
            format!(
                "Embedding error: embedding cache contains too many field bytes: {} (limit {})",
                MAX_CONTEXT_BYTES - 6,
                MAX_CONTEXT_BYTES - 7
            )
        );

        entry.title.pop();
        entry.chunks[0].block_kind = Some("m".repeat(MAX_CONTEXT_BYTES));
        store.insert_note("long.md".into(), entry.clone()).unwrap();
        store.save(&path).unwrap();
        assert_eq!(
            EmbeddingStore::load_for_space(&path, &identity, 1)
                .unwrap()
                .note(Path::new("long.md"))
                .unwrap()
                .chunks[0]
                .block_kind
                .as_ref()
                .unwrap()
                .len(),
            MAX_CONTEXT_BYTES
        );
        entry.chunks[0].block_kind.as_mut().unwrap().push('m');
        assert!(store.insert_note("long.md".into(), entry.clone()).is_err());
        let mut writer = CacheWriter::default();
        write_header(
            &mut writer,
            CACHE_MAGIC,
            CACHE_SCHEMA_VERSION,
            &identity,
            false,
            1,
        )
        .unwrap();
        write_entry(&mut writer, "long.md", &entry).unwrap();
        std::fs::write(&path, writer.bytes).unwrap();
        assert!(EmbeddingStore::load_for_space(&path, &identity, 1).is_err());
    }

    #[test]
    fn captured_entries_survive_complete_generation_replacement() {
        let mut store = EmbeddingStore::new_with_identity(test_identity(3));
        store.insert_note("long.md".into(), passage_note()).unwrap();
        let captured = store.capture_entries(&HashSet::from([PathBuf::from("long.md")]));
        let old_revision = captured[Path::new("long.md")].source_revision.clone();
        let mut replacement = passage_note();
        replacement.source_revision = "a".repeat(64);
        replacement.chunks[1].vector = vec![0.0, 0.0, 1.0];
        store.insert_note("long.md".into(), replacement).unwrap();
        store.remove(Path::new("long.md")).unwrap();
        assert_eq!(captured[Path::new("long.md")].source_revision, old_revision);
        assert_eq!(
            captured[Path::new("long.md")].best_chunk(&[0.0, 1.0, 0.0]),
            Some((1, 1.0))
        );
    }

    #[test]
    fn invalid_replacement_cannot_publish_a_partial_chunk_set() {
        let mut store = EmbeddingStore::new_with_identity(test_identity(3));
        store.insert_note("long.md".into(), passage_note()).unwrap();
        let mut invalid = passage_note();
        invalid.chunks[1].range.start_byte += 1;
        assert!(store.insert_note("long.md".into(), invalid).is_err());
        assert_eq!(
            store.note(Path::new("long.md")).unwrap().chunks[1]
                .range
                .start_byte,
            6
        );
        let mut invalid = passage_note();
        invalid.chunks[1].vector[1] = f32::NAN;
        assert!(store.insert_note("long.md".into(), invalid).is_err());
        assert_eq!(
            store
                .note(Path::new("long.md"))
                .unwrap()
                .best_chunk(&[0.0, 1.0, 0.0]),
            Some((1, 1.0))
        );
    }

    #[test]
    fn remove_reports_accounting_failure_without_removing_the_note() {
        let mut store = EmbeddingStore::new_with_identity(test_identity(3));
        store.insert_note("long.md".into(), passage_note()).unwrap();
        let chunks = store.total_chunks;
        store.encoded_entries_bytes = 0;
        let error = store.remove(Path::new("long.md")).unwrap_err();
        assert!(error.to_string().contains("accounting underflow"));
        assert!(store.note(Path::new("long.md")).is_some());
        assert_eq!(store.total_chunks, chunks);
        assert_eq!(store.encoded_entries_bytes, 0);
    }

    #[test]
    fn removed_and_retained_notes_preserve_exact_cache_size_accounting() {
        let mut store = EmbeddingStore::new_with_identity(test_identity(3));
        for path in ["first.md", "second.md", "third.md"] {
            store.insert_note(path.into(), passage_note()).unwrap();
        }
        assert!(store.remove(Path::new("first.md")).unwrap());
        assert!(!store.remove(Path::new("missing.md")).unwrap());
        assert!(
            store
                .retain_paths(&HashSet::from([PathBuf::from("third.md")]))
                .unwrap()
        );
        assert_eq!(store.total_chunks, 2);
        assert_eq!(
            store.encoded_entries_bytes,
            encoded_entry_len("third.md", &passage_note()).unwrap()
        );
        assert_eq!(
            store.encode_cache().unwrap().len(),
            store.encoded_entries_bytes
                + cache_header_len(store.identity.as_ref().unwrap()).unwrap()
        );
    }

    #[test]
    fn cache_rejects_legacy_schema_before_reading_entries() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("cache.bin");
        let mut bytes = CACHE_MAGIC.to_vec();
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&u32::MAX.to_le_bytes());
        std::fs::write(&path, bytes).unwrap();
        assert!(matches!(
            EmbeddingStore::load(&path),
            Err(VaultError::EmbeddingCacheIncompatible(_))
        ));
    }

    #[test]
    fn cache_rejects_huge_nested_chunk_count_before_vector_allocation() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("cache.bin");
        let entry = passage_note();
        let mut writer = CacheWriter::default();
        write_header(
            &mut writer,
            CACHE_MAGIC,
            CACHE_SCHEMA_VERSION,
            &test_identity(3),
            false,
            1,
        )
        .unwrap();
        writer.string("long.md").unwrap();
        writer.string(&entry.source_revision).unwrap();
        writer.number(entry.source_bytes).unwrap();
        writer.range(&entry.body_range).unwrap();
        writer.string(&entry.title).unwrap();
        writer.number(0).unwrap();
        writer.number(u32::MAX as usize).unwrap();
        std::fs::write(&path, writer.bytes).unwrap();
        let error = EmbeddingStore::load(&path).err().unwrap();
        assert!(error.to_string().contains("too many chunks"));
    }

    #[test]
    fn tied_chunk_and_note_scores_use_source_and_path_order() {
        let mut store = EmbeddingStore::new_with_identity(test_identity(3));
        let mut entry = passage_note();
        entry.chunks[1].vector = entry.chunks[0].vector.clone();
        store.insert_note("z.md".into(), entry.clone()).unwrap();
        store.insert_note("a.md".into(), entry).unwrap();
        let hits = store.query(&[1.0, 0.0, 0.0], 1);
        assert_eq!(hits[0].0, Path::new("a.md"));
        assert_eq!(
            store
                .note(Path::new("a.md"))
                .unwrap()
                .best_chunk(&[1.0, 0.0, 0.0]),
            Some((0, 1.0))
        );
        assert!(store.query(&[1.0, 0.0, 0.0], 0).is_empty());
    }

    #[test]
    fn migrate_legacy_cache_copies_once_and_keeps_source() {
        let vault_root = tempfile::tempdir().expect("temp vault root");
        let semantic_home = tempfile::tempdir().expect("temp semantic home");
        std::fs::create_dir_all(vault_root.path().join(".obsidian")).expect("create .obsidian");

        let source = vault_root
            .path()
            .join(".obsidian")
            .join("obsidian-mcp")
            .join("embeddings.bin");
        std::fs::create_dir_all(source.parent().expect("source parent"))
            .expect("create source dir");
        std::fs::write(&source, b"legacy-cache-bytes").expect("write legacy cache");

        let first = migrate_legacy_cache_to_daemon_store(vault_root.path(), semantic_home.path())
            .expect("first migration should succeed");
        let migrated_path = match first {
            LegacyCacheMigration::Migrated(path) => path,
            other => panic!("expected migrated outcome, got: {other:?}"),
        };
        assert!(source.exists(), "source cache should not be deleted");
        assert!(migrated_path.exists(), "target cache should be created");
        assert_eq!(
            std::fs::read(&source).expect("read source bytes"),
            std::fs::read(&migrated_path).expect("read target bytes")
        );

        let second = migrate_legacy_cache_to_daemon_store(vault_root.path(), semantic_home.path())
            .expect("second migration should succeed");
        assert_eq!(second, LegacyCacheMigration::AlreadyPresent(migrated_path));
    }

    #[test]
    fn migrate_legacy_cache_without_source_is_noop() {
        let vault_root = tempfile::tempdir().expect("temp vault root");
        let semantic_home = tempfile::tempdir().expect("temp semantic home");
        std::fs::create_dir_all(vault_root.path().join(".obsidian")).expect("create .obsidian");

        let outcome = migrate_legacy_cache_to_daemon_store(vault_root.path(), semantic_home.path())
            .expect("migration should succeed");
        assert_eq!(outcome, LegacyCacheMigration::NotFound);
    }

    #[test]
    fn migrate_legacy_cache_rejects_oversized_source_before_copying() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let source = directory.path().join("oversized.bin");
        let target = directory.path().join("target").join("embeddings.bin");
        let source_file = std::fs::File::create(&source).expect("create sparse source");
        source_file
            .set_len(MAX_CACHE_BYTES + 1)
            .expect("extend sparse source");

        let error = migrate_cache_candidates_to_path(&[source], &target)
            .expect_err("oversized migration must fail");

        assert!(error.to_string().contains("too large to relocate"));
        assert!(!target.exists(), "oversized cache must not be published");
    }

    #[test]
    fn migrate_legacy_cache_checks_daemon_store_first() {
        let vault_root = tempfile::tempdir().expect("temp vault root");
        let semantic_home = tempfile::tempdir().expect("temp semantic home");
        let vault_id = crate::daemon::home::compute_vault_id(vault_root.path()).unwrap();
        let target = semantic_home
            .path()
            .join("vaults")
            .join(vault_id)
            .join("embeddings.bin");
        std::fs::create_dir_all(target.parent().expect("target parent"))
            .expect("create target dir");
        std::fs::write(&target, b"daemon-cache-bytes").expect("write target cache");

        let outcome = migrate_legacy_cache_to_daemon_store(vault_root.path(), semantic_home.path())
            .expect("migration should succeed");

        assert_eq!(outcome, LegacyCacheMigration::AlreadyPresent(target));
    }

    #[test]
    fn migrate_legacy_cache_uses_active_local_source() {
        let vault_root = tempfile::tempdir().expect("temp vault root");
        let semantic_home = tempfile::tempdir().expect("temp semantic home");

        let new_source = vault_root
            .path()
            .join(".obsidian-mcp")
            .join("embeddings")
            .join("embeddings.bin");
        std::fs::create_dir_all(new_source.parent().expect("parent")).expect("create new dir");
        std::fs::write(&new_source, b"new-cache-bytes").expect("write new cache");

        let result = migrate_legacy_cache_to_daemon_store(vault_root.path(), semantic_home.path())
            .expect("migration should succeed");
        let migrated_path = match result {
            LegacyCacheMigration::Migrated(path) => path,
            other => panic!("expected Migrated, got: {other:?}"),
        };
        assert!(new_source.exists(), "new source should not be deleted");
        assert_eq!(
            std::fs::read(&new_source).expect("read new source"),
            std::fs::read(&migrated_path).expect("read target"),
        );
    }

    #[test]
    fn migrate_legacy_cache_prefers_active_location_over_older_legacy_location() {
        let vault_root = tempfile::tempdir().expect("temp vault root");
        let semantic_home = tempfile::tempdir().expect("temp semantic home");

        let legacy_source = vault_root
            .path()
            .join(".obsidian")
            .join("obsidian-mcp")
            .join("embeddings.bin");
        std::fs::create_dir_all(legacy_source.parent().expect("parent"))
            .expect("create legacy dir");
        std::fs::write(&legacy_source, b"legacy-bytes").expect("write legacy");

        let new_source = vault_root
            .path()
            .join(".obsidian-mcp")
            .join("embeddings")
            .join("embeddings.bin");
        std::fs::create_dir_all(new_source.parent().expect("parent")).expect("create new dir");
        std::fs::write(&new_source, b"new-bytes").expect("write new");

        let result = migrate_legacy_cache_to_daemon_store(vault_root.path(), semantic_home.path())
            .expect("migration should succeed");
        let migrated_path = match result {
            LegacyCacheMigration::Migrated(path) => path,
            other => panic!("expected Migrated, got: {other:?}"),
        };
        assert_eq!(
            std::fs::read(&migrated_path).expect("read target"),
            b"new-bytes",
            "the active local cache should be preferred over the older legacy location"
        );
    }

    #[test]
    fn concurrent_cache_migrations_publish_one_complete_source_without_overwrite() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let first_source = directory.path().join("first.bin");
        let second_source = directory.path().join("second.bin");
        let target = directory.path().join("target").join("embeddings.bin");
        let first_bytes = vec![0x11; 512 * 1024];
        let second_bytes = vec![0x22; 512 * 1024];
        std::fs::write(&first_source, &first_bytes).expect("write first source");
        std::fs::write(&second_source, &second_bytes).expect("write second source");

        let barrier = std::sync::Arc::new(std::sync::Barrier::new(3));
        let workers = [first_source, second_source]
            .into_iter()
            .map(|source| {
                let barrier = std::sync::Arc::clone(&barrier);
                let target = target.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    migrate_cache_candidates_to_path(&[source], &target)
                        .expect("migration should resolve atomically")
                })
            })
            .collect::<Vec<_>>();

        barrier.wait();
        let outcomes = workers
            .into_iter()
            .map(|worker| worker.join().expect("migration worker should join"))
            .collect::<Vec<_>>();
        assert_eq!(
            outcomes
                .iter()
                .filter(|outcome| matches!(outcome, LegacyCacheMigration::Migrated(_)))
                .count(),
            1
        );
        assert_eq!(
            outcomes
                .iter()
                .filter(|outcome| matches!(outcome, LegacyCacheMigration::AlreadyPresent(_)))
                .count(),
            1
        );

        let published = std::fs::read(&target).expect("read published target");
        assert!(published == first_bytes || published == second_bytes);
    }

    // ── resolve_provider ──────────────────────────────────────────

    #[test]
    fn resolve_provider_explicit_local() {
        let result = resolve_provider(Some(EmbeddingProvider::Local));
        assert_eq!(result, EmbeddingProvider::Local);
    }

    #[test]
    fn resolve_provider_explicit_api() {
        let result = resolve_provider(Some(EmbeddingProvider::Api));
        assert_eq!(result, EmbeddingProvider::Api);
    }

    #[test]
    fn resolve_provider_none_infers_from_features() {
        let result = resolve_provider(None);
        if cfg!(feature = "embeddings") {
            assert_eq!(result, EmbeddingProvider::Local);
        } else if cfg!(feature = "embeddings-api") {
            assert_eq!(result, EmbeddingProvider::Api);
        }
    }

    #[cfg(feature = "embeddings")]
    #[test]
    fn local_requests_bound_chunk_multiplication_independently_of_note_batches() {
        let policy = EmbeddingInputPolicy {
            max_input_tokens: 512,
            tokenizer_fingerprint: [1; 32],
        };
        let limits = local_request_limits(policy.max_input_tokens).unwrap();
        let texts = vec!["prepared chunk"; 56];
        let accepted =
            embedding_input::plan_batch(&texts, &policy, limits, 2, |_| Ok(512)).unwrap();
        assert_eq!(accepted, 32);
        assert_eq!(limits.max_total_tokens, 16_384);
        assert!(embedding_input::validate_batch(&texts, &policy, limits, 2, |_| Ok(512)).is_err());
        assert!(
            embedding_input::validate_batch(&texts[..accepted], &policy, limits, 2, |_| Ok(512))
                .is_ok()
        );
    }

    #[cfg(feature = "embeddings")]
    #[test]
    fn local_model_enum_and_repository_alias_have_one_canonical_identity() {
        let (enum_model, enum_identity, enum_dim) = resolve_local_model("BGESmallENV15").unwrap();
        let (repo_model, repo_identity, repo_dim) =
            resolve_local_model("BAAI/bge-small-en-v1.5").unwrap();

        assert_eq!(enum_model, repo_model);
        assert_eq!(enum_identity, "BGESmallENV15");
        assert_eq!(enum_identity, repo_identity);
        assert_eq!(enum_dim, repo_dim);
        assert_eq!(repo_dim, 384);
    }

    #[cfg(feature = "embeddings")]
    #[test]
    fn unknown_local_model_is_rejected_instead_of_falling_back() {
        let error = resolve_local_model("definitely-not-a-model").err().unwrap();
        assert!(error.to_string().contains("unknown local embedding model"));
    }

    #[cfg(feature = "embeddings")]
    #[test]
    fn ambiguous_local_repository_alias_is_rejected() {
        let error = resolve_local_model("Xenova/all-MiniLM-L12-v2")
            .err()
            .unwrap();
        assert!(
            error
                .to_string()
                .contains("ambiguous local embedding model")
        );
    }

    // ── API response parsing ──────────────────────────────────────

    #[cfg(feature = "embeddings-api")]
    mod api_response_tests {
        use super::*;
        use std::sync::{LazyLock, Mutex};

        static ENV_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

        fn with_env_lock<F: FnOnce()>(f: F) {
            let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            f();
        }

        fn api_fixture(
            responses: Vec<(u16, &'static str, String)>,
        ) -> (String, std::thread::JoinHandle<Vec<serde_json::Value>>) {
            api_fixture_with_content_lengths(
                responses
                    .into_iter()
                    .map(|(status, retry_after, body)| (status, retry_after, body, None))
                    .collect(),
            )
        }

        fn api_fixture_with_content_lengths(
            responses: Vec<(u16, &'static str, String, Option<usize>)>,
        ) -> (String, std::thread::JoinHandle<Vec<serde_json::Value>>) {
            use std::io::Write as _;
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let base = format!("http://{}/v1", listener.local_addr().unwrap());
            let server = std::thread::spawn(move || {
                let mut requests = Vec::new();
                for (status, retry_after, body, declared_bytes) in responses {
                    let (mut stream, _) = listener.accept().unwrap();
                    crate::test_helpers::read_http_request_headers(&mut stream);
                    let request = serde_json::Deserializer::from_reader(&mut stream)
                        .into_iter::<serde_json::Value>()
                        .next()
                        .unwrap()
                        .unwrap();
                    requests.push(request);
                    let response = format!(
                        "HTTP/1.1 {status} Test\r\nContent-Type: application/json\r\nContent-Length: {}\r\nRetry-After: {retry_after}\r\nConnection: close\r\n\r\n{body}",
                        declared_bytes.unwrap_or(body.len()),
                    );
                    stream.write_all(response.as_bytes()).unwrap();
                }
                requests
            });
            (base, server)
        }

        fn api_fixture_model(
            base: &str,
            dimension: Option<usize>,
            timeout: std::time::Duration,
        ) -> VaultResult<EmbeddingModel> {
            let directory = tempfile::tempdir().unwrap();
            let profile = directory.path().join("profile.json");
            std::fs::write(
                &profile,
                serde_json::to_vec(&serde_json::json!({
                    "version": 1,
                    "profile_revision": "synthetic-v1",
                    "api_base": base,
                    "model": "test-model",
                    "tokenizer": {"kind":"cl100k_base"},
                    "max_input_tokens": 512,
                    "max_batch_inputs": 32,
                    "max_batch_tokens": 16384
                }))
                .unwrap(),
            )
            .unwrap();
            let contract =
                embedding_input::ApiInputContract::load(base, "test-model", Some(&profile))?;
            let client = ApiEmbeddingClient::start(
                build_api_client(timeout)?,
                base.to_string(),
                "test-model".into(),
                zeroize::Zeroizing::new("test-key".into()),
                timeout,
            )?;
            EmbeddingModel::from_api_client(
                client,
                base.to_string(),
                "test-model".into(),
                contract,
                dimension,
            )
        }

        #[test]
        fn api_initialization_rejects_explicit_and_probed_unsupported_dimensions() {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            listener.set_nonblocking(true).unwrap();
            let base = format!("http://{}/v1", listener.local_addr().unwrap());
            for dimension in [0, MAX_CACHE_DIMENSION + 1] {
                let error =
                    api_fixture_model(&base, Some(dimension), std::time::Duration::from_secs(2))
                        .err()
                        .unwrap();
                let expected = if dimension == 0 {
                    "OBSIDIAN_EMBEDDING_DIM must be greater than zero"
                } else {
                    "embedding dimension is outside supported bounds (1..=65536)"
                };
                assert!(error.to_string().contains(expected), "{error}");
                assert_eq!(
                    listener.accept().unwrap_err().kind(),
                    std::io::ErrorKind::WouldBlock
                );
            }
            let model = api_fixture_model(
                &base,
                Some(MAX_CACHE_DIMENSION),
                std::time::Duration::from_secs(2),
            )
            .unwrap();
            assert_eq!(model.dim(), MAX_CACHE_DIMENSION);
            let store = EmbeddingStore::new_with_identity(model.identity.clone());
            assert!(store.encode_cache().is_ok());

            let body =
                serde_json::json!({"data":[{"embedding":vec![0.0; MAX_CACHE_DIMENSION + 1]}]})
                    .to_string();
            let (base, server) = api_fixture(vec![(200, "0", body)]);
            let error = api_fixture_model(&base, None, std::time::Duration::from_secs(2))
                .err()
                .unwrap();
            assert!(error.to_string().contains("outside supported bounds"));
            assert_eq!(
                server.join().unwrap()[0]["input"],
                serde_json::json!(["dim"])
            );
        }

        #[test]
        fn api_dimension_probe_and_query_recover_transient_rate_limits() {
            let valid = r#"{"data":[{"embedding":[1.0,0.0]}]}"#.to_string();
            let (base, server) = api_fixture(vec![
                (429, "0", String::new()),
                (200, "0", valid.clone()),
                (429, "0", String::new()),
                (503, "0", String::new()),
                (200, "0", valid),
            ]);
            let model = api_fixture_model(&base, None, std::time::Duration::from_secs(2)).unwrap();
            assert_eq!(model.dim(), 2);
            assert_eq!(
                Embedder::embed_query(&model, "query body").unwrap(),
                vec![1.0, 0.0]
            );
            let requests = server.join().unwrap();
            assert_eq!(requests.len(), 5);
            assert!(
                requests[..2]
                    .iter()
                    .all(|request| request["input"] == serde_json::json!(["dim"]))
            );
            assert!(
                requests[2..]
                    .iter()
                    .all(|request| request["input"] == serde_json::json!(["query body"]))
            );
        }

        #[test]
        fn api_probe_and_query_retry_incomplete_successful_response_bodies() {
            let valid = r#"{"data":[{"embedding":[1.0,0.0]}]}"#.to_string();
            let (base, server) = api_fixture_with_content_lengths(vec![
                (200, "0", "{".into(), Some(100)),
                (200, "0", valid.clone(), None),
                (200, "0", "{".into(), Some(100)),
                (200, "0", valid, None),
            ]);
            let model = api_fixture_model(&base, None, std::time::Duration::from_secs(3)).unwrap();
            assert_eq!(model.dim(), 2);
            assert_eq!(
                Embedder::embed_query(&model, "query body").unwrap(),
                vec![1.0, 0.0]
            );
            let requests = server.join().unwrap();
            assert_eq!(requests.len(), 4);
            assert!(
                requests[..2]
                    .iter()
                    .all(|request| request["input"] == serde_json::json!(["dim"]))
            );
            assert!(
                requests[2..]
                    .iter()
                    .all(|request| request["input"] == serde_json::json!(["query body"]))
            );
        }

        #[test]
        fn incomplete_successful_document_response_leaves_retry_to_the_coordinator() {
            let (base, server) =
                api_fixture_with_content_lengths(vec![(200, "0", "{".into(), Some(100))]);
            let model =
                api_fixture_model(&base, Some(2), std::time::Duration::from_secs(2)).unwrap();
            let (count, result) = Embedder::embed_next_batch(&model, &["document body"]);
            assert_eq!(count, 1);
            let error = result.unwrap_err();
            assert!(
                matches!(
                    &error,
                    VaultError::EmbeddingRequest {
                        retryable: true,
                        ..
                    }
                ),
                "{error}"
            );
            assert!(error.to_string().contains("response body"));
            assert_eq!(server.join().unwrap().len(), 1);
        }

        #[test]
        fn api_query_retry_attempts_are_bounded_and_document_requests_stay_single_attempt() {
            let (base, server) = api_fixture((0..4).map(|_| (429, "0", String::new())).collect());
            let model =
                api_fixture_model(&base, Some(2), std::time::Duration::from_secs(2)).unwrap();
            let error = Embedder::embed_query(&model, "query body").unwrap_err();
            assert!(error.to_string().contains("429"));
            assert!(error.to_string().contains("after 4 attempts"));
            assert_eq!(server.join().unwrap().len(), 4);

            for status in [429, 503] {
                let (base, server) = api_fixture(vec![(status, "0", String::new())]);
                let model =
                    api_fixture_model(&base, Some(2), std::time::Duration::from_secs(2)).unwrap();
                let (count, result) = Embedder::embed_next_batch(&model, &["document body"]);
                assert_eq!(count, 1);
                let error = result.unwrap_err();
                assert!(matches!(
                    error,
                    VaultError::EmbeddingRequest {
                        retryable: true,
                        ..
                    }
                ));
                assert_eq!(server.join().unwrap().len(), 1);
            }
        }

        #[test]
        fn document_rate_limit_preserves_the_full_provider_retry_delay() {
            let (base, server) = api_fixture(vec![(429, "120", String::new())]);
            let model =
                api_fixture_model(&base, Some(2), std::time::Duration::from_secs(2)).unwrap();
            let started = std::time::Instant::now();
            let (count, result) = Embedder::embed_next_batch(&model, &["document body"]);
            assert_eq!(count, 1);
            let error = result.unwrap_err();
            assert!(matches!(error, VaultError::EmbeddingRequest {
                retryable:true, retry_after:Some(delay), ..
            } if delay == std::time::Duration::from_secs(120)));
            assert!(started.elapsed() < std::time::Duration::from_secs(1));
            assert_eq!(server.join().unwrap().len(), 1);
        }

        #[test]
        fn planned_document_dispatch_sends_only_the_validated_prefix() {
            let body = serde_json::json!({"data":(0..32).map(|index| {
                serde_json::json!({"index":index,"embedding":[1.0,0.0]})
            }).collect::<Vec<_>>()})
            .to_string();
            let (base, server) = api_fixture(vec![(200, "0", body)]);
            let model =
                api_fixture_model(&base, Some(2), std::time::Duration::from_secs(2)).unwrap();
            let (count, result) = Embedder::embed_next_batch(&model, &["body"; 33]);
            assert_eq!(count, 32);
            assert_eq!(result.unwrap(), vec![vec![1.0, 0.0]; 32]);
            assert_eq!(
                server.join().unwrap()[0]["input"].as_array().unwrap().len(),
                32
            );

            let oversized = "body ".repeat(600);
            let (count, result) = Embedder::embed_next_batch(&model, &[oversized.as_str()]);
            assert_eq!(count, 1);
            assert!(matches!(
                result,
                Err(VaultError::EmbeddingInputTooLong { .. })
            ));
            assert!(matches!(
                model.embed_batch(&[oversized.as_str()]),
                Err(VaultError::EmbeddingInputTooLong { .. })
            ));
        }

        #[test]
        fn api_query_does_not_retry_permanent_errors_and_respects_retry_deadline() {
            for (status, body) in [
                (401, String::new()),
                (200, "invalid JSON".into()),
                (200, r#"{"data":[{"embedding":[1.0]}]}"#.into()),
            ] {
                let (base, server) = api_fixture(vec![(status, "0", body)]);
                let model =
                    api_fixture_model(&base, Some(2), std::time::Duration::from_secs(2)).unwrap();
                assert!(Embedder::embed_query(&model, "query body").is_err());
                assert_eq!(server.join().unwrap().len(), 1);
            }
            let (base, server) = api_fixture(vec![(429, "1000", String::new())]);
            let model =
                api_fixture_model(&base, Some(2), std::time::Duration::from_millis(100)).unwrap();
            let start = std::time::Instant::now();
            let error = Embedder::embed_query(&model, "query body").unwrap_err();
            assert!(error.to_string().contains("429"));
            assert!(error.to_string().contains("retry deadline"));
            assert!(start.elapsed() < std::time::Duration::from_secs(1));
            assert_eq!(server.join().unwrap().len(), 1);
        }

        #[test]
        fn parse_valid_single_embedding() {
            let resp = serde_json::json!({
                "data": [{"embedding": [0.1, 0.2, 0.3]}]
            });
            let result = parse_embedding_response(&resp, 1).unwrap();
            assert_eq!(result.len(), 1);
            assert_eq!(result[0].len(), 3);
            assert!((result[0][0] - 0.1).abs() < 1e-6);
        }

        #[test]
        fn parse_valid_multiple_embeddings() {
            let resp = serde_json::json!({
                "data": [
                    {"embedding": [0.1, 0.2]},
                    {"embedding": [0.3, 0.4]}
                ]
            });
            let result = parse_embedding_response(&resp, 2).unwrap();
            assert_eq!(result.len(), 2);
            assert_eq!(result[0], vec![0.1f32, 0.2]);
            assert_eq!(result[1], vec![0.3f32, 0.4]);
        }

        #[test]
        fn parse_missing_data_field() {
            let resp = serde_json::json!({"object": "list"});
            let err = parse_embedding_response(&resp, 1).unwrap_err();
            assert!(err.to_string().contains("missing 'data' array"));
        }

        #[test]
        fn parse_missing_embedding_in_item() {
            let resp = serde_json::json!({
                "data": [{"index": 0}]
            });
            let err = parse_embedding_response(&resp, 1).unwrap_err();
            assert!(err.to_string().contains("missing 'embedding' array"));
        }

        #[test]
        fn parse_non_numeric_value_in_vector() {
            let resp = serde_json::json!({
                "data": [{"embedding": [0.1, "bad", 0.3]}]
            });
            let err = parse_embedding_response(&resp, 1).unwrap_err();
            assert!(err.to_string().contains("non-numeric value"));
        }

        #[test]
        fn parse_reorders_by_index_field() {
            let resp = serde_json::json!({
                "data": [
                    {"index": 1, "embedding": [0.3, 0.4]},
                    {"index": 0, "embedding": [0.1, 0.2]}
                ]
            });
            let result = parse_embedding_response(&resp, 2).unwrap();
            assert_eq!(result.len(), 2);
            assert_eq!(result[0], vec![0.1f32, 0.2]);
            assert_eq!(result[1], vec![0.3f32, 0.4]);
        }

        #[test]
        fn parse_falls_back_to_array_order_without_index() {
            let resp = serde_json::json!({
                "data": [
                    {"embedding": [0.1, 0.2]},
                    {"embedding": [0.3, 0.4]}
                ]
            });
            let result = parse_embedding_response(&resp, 2).unwrap();
            assert_eq!(result[0], vec![0.1f32, 0.2]);
            assert_eq!(result[1], vec![0.3f32, 0.4]);
        }

        #[test]
        fn parse_empty_data_array() {
            let resp = serde_json::json!({"data": []});
            let result = parse_embedding_response(&resp, 0).unwrap();
            assert!(result.is_empty());
        }

        #[test]
        fn parse_empty_embedding_vector() {
            let resp = serde_json::json!({
                "data": [{"embedding": []}]
            });
            let result = parse_embedding_response(&resp, 1).unwrap();
            assert_eq!(result.len(), 1);
            assert!(result[0].is_empty());
        }

        #[test]
        fn parse_rejects_partial_response() {
            let resp = serde_json::json!({
                "data": [{"embedding": [0.1, 0.2]}]
            });
            let error = parse_embedding_response(&resp, 2).err().unwrap();
            assert!(error.to_string().contains("1 vectors for 2 inputs"));
        }

        #[test]
        fn parse_rejects_mixed_index_presence() {
            let resp = serde_json::json!({
                "data": [
                    {"index": 0, "embedding": [0.1, 0.2]},
                    {"embedding": [0.3, 0.4]}
                ]
            });
            let error = parse_embedding_response(&resp, 2).err().unwrap();
            assert!(error.to_string().contains("mixed indexed and unindexed"));
        }

        #[test]
        fn parse_rejects_duplicate_indices() {
            let resp = serde_json::json!({
                "data": [
                    {"index": 0, "embedding": [0.1, 0.2]},
                    {"index": 0, "embedding": [0.3, 0.4]}
                ]
            });
            let error = parse_embedding_response(&resp, 2).err().unwrap();
            assert!(error.to_string().contains("not unique and contiguous"));
        }

        #[test]
        fn parse_rejects_out_of_range_index() {
            let resp = serde_json::json!({
                "data": [
                    {"index": 0, "embedding": [0.1, 0.2]},
                    {"index": 2, "embedding": [0.3, 0.4]}
                ]
            });
            let error = parse_embedding_response(&resp, 2).err().unwrap();
            assert!(error.to_string().contains("out of range"));
        }

        #[test]
        fn common_validator_rejects_wrong_dimension_and_non_finite_values() {
            let wrong_dimension = validate_embedding_batch(vec![vec![0.1]], 1, 2)
                .err()
                .unwrap();
            assert!(wrong_dimension.to_string().contains("dimension mismatch"));

            let non_finite = validate_embedding_batch(vec![vec![0.1, f32::INFINITY]], 1, 2)
                .err()
                .unwrap();
            assert!(non_finite.to_string().contains("non-finite"));
        }

        #[test]
        fn api_cache_identity_uses_endpoint_fingerprint_not_plaintext_url() {
            let endpoint = "https://embedding-user:secret@example.invalid/v1/private";
            let identity = EmbeddingSpaceIdentity::api(
                "test-model".to_string(),
                endpoint,
                3,
                &EmbeddingInputPolicy {
                    max_input_tokens: 512,
                    tokenizer_fingerprint: [1; 32],
                },
            );
            let bytes =
                bincode::serde::encode_to_vec(&identity, bincode::config::standard()).unwrap();

            assert!(
                !bytes
                    .windows(endpoint.len())
                    .any(|window| window == endpoint.as_bytes())
            );
            assert!(!String::from_utf8_lossy(&bytes).contains("secret"));
        }

        #[test]
        fn api_http_error_does_not_expose_url_key_body_or_input() {
            use std::io::{Read as _, Write as _};

            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let address = listener.local_addr().unwrap();
            let server = std::thread::spawn(move || {
                let (mut stream, _) = listener.accept().unwrap();
                let mut request = [0u8; 4096];
                let _ = stream.read(&mut request);
                let body = r#"{"error":"provider echoed sensitive note body and api-secret"}"#;
                let response = format!(
                    "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                stream.write_all(response.as_bytes()).unwrap();
            });
            let base_url = format!("http://{address}/secret-url-component");
            let client = build_api_client(std::time::Duration::from_secs(30)).unwrap();

            let error = embed_batch_api(
                &client,
                &base_url,
                "test-model",
                "api-secret",
                &["sensitive note body"],
            )
            .unwrap_err()
            .to_string();

            assert!(error.contains("HTTP status 400"));
            assert!(!error.contains("secret-url-component"));
            assert!(!error.contains("api-secret"));
            assert!(!error.contains("sensitive note body"));
            server.join().unwrap();
        }

        #[test]
        fn api_transport_error_does_not_expose_secret_url() {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let address = listener.local_addr().unwrap();
            let server = std::thread::spawn(move || {
                let (stream, _) = listener.accept().unwrap();
                drop(stream);
            });
            let base_url = format!("http://{address}/secret-url-component");
            let client = build_api_client(std::time::Duration::from_secs(30)).unwrap();

            let error = embed_batch_api(
                &client,
                &base_url,
                "test-model",
                "api-secret",
                &["sensitive note body"],
            )
            .unwrap_err()
            .to_string();

            assert!(error.contains("embedding API"));
            assert!(!error.contains("secret-url-component"));
            assert!(!error.contains("api-secret"));
            assert!(!error.contains("sensitive note body"));
            server.join().unwrap();
        }

        #[test]
        fn api_status_classification_leaves_retry_to_the_coordinator() {
            use std::io::Write as _;
            for (status, retryable) in [(400, false), (401, false), (429, true), (503, true)] {
                let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
                let address = listener.local_addr().unwrap();
                let server = std::thread::spawn(move || {
                    let (mut stream, _) = listener.accept().unwrap();
                    crate::test_helpers::read_http_request_headers(&mut stream);
                    serde_json::Deserializer::from_reader(&mut stream)
                        .into_iter::<serde_json::Value>()
                        .next()
                        .unwrap()
                        .unwrap();
                    let response = format!(
                        "HTTP/1.1 {status} Test\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                    );
                    stream.write_all(response.as_bytes()).unwrap();
                });
                let client = build_api_client(std::time::Duration::from_secs(2)).unwrap();
                let error = embed_batch_api(
                    &client,
                    &format!("http://{address}/v1"),
                    "test",
                    "synthetic-key",
                    &["text"],
                )
                .unwrap_err();
                assert!(
                    matches!(&error, VaultError::EmbeddingRequest {retryable:actual,..} if *actual==retryable)
                );
                assert!(error.to_string().contains(&format!("HTTP status {status}")));
                server.join().unwrap();
            }
        }

        #[test]
        fn api_configured_timeout_applies_to_probe_and_embedding_requests() {
            use std::io::Write as _;
            use std::time::Duration;

            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let address = listener.local_addr().unwrap();
            let server = std::thread::spawn(move || {
                for _ in 0..2 {
                    let (mut stream, _) = listener.accept().unwrap();
                    crate::test_helpers::read_http_request_headers(&mut stream);
                    serde_json::Deserializer::from_reader(&mut stream)
                        .into_iter::<serde_json::Value>()
                        .next()
                        .unwrap()
                        .unwrap();
                    std::thread::sleep(Duration::from_millis(150));
                    let body = r#"{"data":[{"embedding":[1.0,0.0]}]}"#;
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    // A timed-out client may have closed its connection already.
                    let _ = stream.write_all(response.as_bytes());
                }
            });
            let base_url = format!("http://{address}/v1");
            let short_client = ApiEmbeddingClient::start(
                build_api_client(Duration::from_millis(50)).unwrap(),
                base_url.clone(),
                "test-model".to_string(),
                zeroize::Zeroizing::new("test-key".to_string()),
                Duration::from_millis(50),
            )
            .unwrap();
            let error = probe_api_dimension(&short_client).unwrap_err();
            assert!(error.to_string().contains("request timed out"));

            let patient_client = ApiEmbeddingClient::start(
                build_api_client(Duration::from_secs(2)).unwrap(),
                base_url,
                "test-model".to_string(),
                zeroize::Zeroizing::new("test-key".to_string()),
                Duration::from_secs(2),
            )
            .unwrap();
            assert_eq!(
                patient_client.embed_batch(&["note body"]).unwrap(),
                vec![vec![1.0, 0.0]]
            );
            server.join().unwrap();
        }

        #[test]
        fn read_env_with_fallback_primary_wins() {
            with_env_lock(|| {
                unsafe {
                    std::env::set_var("TEST_PRIMARY_KEY_A", "primary_value");
                    std::env::set_var("TEST_FALLBACK_KEY_A", "fallback_value");
                }
                let result = read_env_with_fallback("TEST_PRIMARY_KEY_A", "TEST_FALLBACK_KEY_A");
                assert_eq!(result, Some("primary_value".to_string()));
                unsafe {
                    std::env::remove_var("TEST_PRIMARY_KEY_A");
                    std::env::remove_var("TEST_FALLBACK_KEY_A");
                }
            });
        }

        #[test]
        fn read_env_with_fallback_uses_fallback() {
            with_env_lock(|| {
                unsafe {
                    std::env::remove_var("TEST_PRIMARY_KEY_B");
                    std::env::set_var("TEST_FALLBACK_KEY_B", "fallback_value");
                }
                let result = read_env_with_fallback("TEST_PRIMARY_KEY_B", "TEST_FALLBACK_KEY_B");
                assert_eq!(result, Some("fallback_value".to_string()));
                unsafe {
                    std::env::remove_var("TEST_FALLBACK_KEY_B");
                }
            });
        }

        #[test]
        fn read_env_with_fallback_returns_none_when_both_missing() {
            with_env_lock(|| {
                unsafe {
                    std::env::remove_var("TEST_PRIMARY_KEY_C");
                    std::env::remove_var("TEST_FALLBACK_KEY_C");
                }
                let result = read_env_with_fallback("TEST_PRIMARY_KEY_C", "TEST_FALLBACK_KEY_C");
                assert_eq!(result, None);
            });
        }

        #[test]
        fn read_env_with_fallback_ignores_empty_primary() {
            with_env_lock(|| {
                unsafe {
                    std::env::set_var("TEST_PRIMARY_KEY_D", "  ");
                    std::env::set_var("TEST_FALLBACK_KEY_D", "valid");
                }
                let result = read_env_with_fallback("TEST_PRIMARY_KEY_D", "TEST_FALLBACK_KEY_D");
                assert_eq!(result, Some("valid".to_string()));
                unsafe {
                    std::env::remove_var("TEST_PRIMARY_KEY_D");
                    std::env::remove_var("TEST_FALLBACK_KEY_D");
                }
            });
        }

        #[test]
        fn parse_usize_env_valid() {
            with_env_lock(|| {
                unsafe {
                    std::env::set_var("TEST_DIM_VALID", "384");
                }
                assert_eq!(parse_usize_env("TEST_DIM_VALID"), Some(384));
                unsafe {
                    std::env::remove_var("TEST_DIM_VALID");
                }
            });
        }

        #[test]
        fn parse_usize_env_invalid() {
            with_env_lock(|| {
                unsafe {
                    std::env::set_var("TEST_DIM_INVALID", "not_a_number");
                }
                assert_eq!(parse_usize_env("TEST_DIM_INVALID"), None);
                unsafe {
                    std::env::remove_var("TEST_DIM_INVALID");
                }
            });
        }

        #[test]
        fn parse_usize_env_missing() {
            with_env_lock(|| {
                unsafe {
                    std::env::remove_var("TEST_DIM_MISSING");
                }
                assert_eq!(parse_usize_env("TEST_DIM_MISSING"), None);
            });
        }
    }
}
