//! Embedding store and model wrapper for semantic search (Layer 2).
//!
//! Gated behind the `embeddings` Cargo feature. Provides:
//! - `EmbeddingStore`: in-memory HashMap of note embeddings with brute-force
//!   cosine similarity search and bincode persistence.
//! - `EmbeddingModel`: wrapper around `fastembed::TextEmbedding` with async
//!   model loading and batch/single embedding generation.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use fastembed::ModelTrait;

use crate::error::{VaultError, VaultResult};

// ── Cosine similarity ──────────────────────────────────────────────────

pub(crate) fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let (dot, norm_a, norm_b) = a
        .iter()
        .zip(b)
        .fold((0.0f32, 0.0f32, 0.0f32), |(d, na, nb), (&x, &y)| {
            (d + x * y, na + x * x, nb + y * y)
        });
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a.sqrt() * norm_b.sqrt())
}

// ── EmbeddingStore ─────────────────────────────────────────────────────

/// In-memory store mapping vault-relative note paths to embedding vectors.
///
/// Search is brute-force cosine similarity — O(n * dim). For dim=384 and
/// n=5000 this is ~2M multiply-adds, well under 5ms on modern hardware.
pub struct EmbeddingStore {
    embeddings: HashMap<PathBuf, Vec<f32>>,
    dim: usize,
}

/// Serde-friendly intermediate for bincode persistence.
/// Avoids `PathBuf` encoding issues by converting to `String`.
#[derive(serde::Serialize, serde::Deserialize)]
struct EmbeddingCacheData {
    dim: usize,
    entries: Vec<(String, Vec<f32>)>,
}

impl EmbeddingStore {
    /// Create an empty store for embeddings of the given dimensionality.
    pub fn new(dim: usize) -> Self {
        Self {
            embeddings: HashMap::new(),
            dim,
        }
    }

    /// Insert or replace the embedding for a note.
    pub fn insert(&mut self, path: PathBuf, vec: Vec<f32>) {
        debug_assert_eq!(
            vec.len(),
            self.dim,
            "embedding dimension mismatch: expected {}, got {}",
            self.dim,
            vec.len()
        );
        self.embeddings.insert(path, vec);
    }

    /// Remove a note's embedding.
    pub fn remove(&mut self, path: &Path) {
        self.embeddings.remove(path);
    }

    /// Retrieve a note's embedding vector.
    pub fn get(&self, path: &Path) -> Option<&[f32]> {
        self.embeddings.get(path).map(|v| v.as_slice())
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

    /// Find the `top_k` most similar notes to `query_vec`, sorted by
    /// descending cosine similarity.
    pub fn query(&self, query_vec: &[f32], top_k: usize) -> Vec<(PathBuf, f32)> {
        let mut scored: Vec<(PathBuf, f32)> = self
            .embeddings
            .iter()
            .map(|(path, vec)| (path.clone(), cosine_similarity(query_vec, vec)))
            .collect();

        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
    }

    /// Serialize the store to a binary cache file.
    pub fn save(&self, path: &Path) -> VaultResult<()> {
        let data = EmbeddingCacheData {
            dim: self.dim,
            entries: self
                .embeddings
                .iter()
                .map(|(p, v)| (p.to_string_lossy().into_owned(), v.clone()))
                .collect(),
        };
        let bytes = bincode::serde::encode_to_vec(&data, bincode::config::standard())
            .map_err(|e| VaultError::Embedding(format!("cache serialize error: {e}")))?;

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Deserialize a store from a binary cache file.
    pub fn load(path: &Path) -> VaultResult<Self> {
        let bytes = std::fs::read(path)?;
        let (data, _): (EmbeddingCacheData, _) =
            bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
                .map_err(|e| VaultError::Embedding(format!("cache deserialize error: {e}")))?;

        let mut embeddings = HashMap::with_capacity(data.entries.len());
        for (path_str, vec) in data.entries {
            if vec.len() != data.dim {
                tracing::warn!(
                    path = %path_str,
                    expected = data.dim,
                    got = vec.len(),
                    "skipping cache entry with mismatched embedding dimension"
                );
                continue;
            }
            embeddings.insert(PathBuf::from(path_str), vec);
        }

        Ok(Self {
            embeddings,
            dim: data.dim,
        })
    }
}

// ── EmbeddingModel ─────────────────────────────────────────────────────

/// Wrapper around `fastembed::TextEmbedding` providing thread-safe embedding.
///
/// `TextEmbedding::embed()` takes `&mut self`, so access is serialized via
/// a `Mutex`. The lock is held only during inference calls.
pub struct EmbeddingModel {
    inner: std::sync::Mutex<fastembed::TextEmbedding>,
    dim: usize,
}

impl EmbeddingModel {
    /// Load an embedding model by HuggingFace name (e.g. "BAAI/bge-small-en-v1.5").
    ///
    /// Model initialization downloads weights on first use and loads the ONNX
    /// runtime, both of which are blocking — runs inside `spawn_blocking`.
    pub async fn load(model_name: &str) -> VaultResult<Self> {
        let model_name = model_name.to_owned();

        tokio::task::spawn_blocking(move || {
            let model_enum: fastembed::EmbeddingModel = model_name.parse().unwrap_or_default();

            let dim = fastembed::EmbeddingModel::get_model_info(&model_enum)
                .map(|info| info.dim)
                .unwrap_or(384);

            let options = fastembed::InitOptions::new(model_enum).with_show_download_progress(true);

            let inner = fastembed::TextEmbedding::try_new(options)
                .map_err(|e| VaultError::Embedding(format!("model load failed: {e}")))?;

            Ok(Self {
                inner: std::sync::Mutex::new(inner),
                dim,
            })
        })
        .await
        .map_err(|e| VaultError::Embedding(format!("spawn_blocking join error: {e}")))?
    }

    /// Embed a batch of texts. Returns one vector per input text.
    pub fn embed_batch(&self, texts: &[&str]) -> VaultResult<Vec<Vec<f32>>> {
        let mut model = self
            .inner
            .lock()
            .map_err(|e| VaultError::Embedding(format!("model lock poisoned: {e}")))?;

        model
            .embed(texts, Some(64))
            .map_err(|e| VaultError::Embedding(format!("embed failed: {e}")))
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
}

// ── Text preparation ───────────────────────────────────────────────────

const MAX_BODY_WORDS: usize = 400;

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
    let source = vault_root
        .join(".obsidian")
        .join("obsidian-mcp")
        .join("embeddings.bin");
    if !source.is_file() {
        return Ok(LegacyCacheMigration::NotFound);
    }

    let vault_id = crate::daemon::home::compute_vault_id(vault_root)?;
    let target = semantic_home
        .join("vaults")
        .join(vault_id)
        .join("embeddings.bin");
    if target.exists() {
        return Ok(LegacyCacheMigration::AlreadyPresent(target));
    }

    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::copy(&source, &target)?;
    Ok(LegacyCacheMigration::Migrated(target))
}

/// Prepare text for embedding from note components.
///
/// Format: `"{title}\n{headings joined with " | "}\n{body truncated to 400 words}"`.
/// The body should already have frontmatter stripped.
pub fn prepare_embed_text(title: &str, headings: &[String], body: &str) -> String {
    let headings_line = headings.join(" | ");

    let truncated_body: String = body
        .split_whitespace()
        .take(MAX_BODY_WORDS)
        .collect::<Vec<_>>()
        .join(" ");

    if headings_line.is_empty() {
        format!("{title}\n{truncated_body}")
    } else {
        format!("{title}\n{headings_line}\n{truncated_body}")
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

    fn make_store() -> EmbeddingStore {
        let mut store = EmbeddingStore::new(3);
        store.insert(PathBuf::from("a.md"), vec![1.0, 0.0, 0.0]);
        store.insert(PathBuf::from("b.md"), vec![0.0, 1.0, 0.0]);
        store.insert(PathBuf::from("c.md"), vec![0.7, 0.7, 0.0]);
        store
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
    fn insert_remove_updates_results() {
        let mut store = make_store();
        assert_eq!(store.len(), 3);

        store.remove(Path::new("a.md"));
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

    // ── prepare_embed_text ─────────────────────────────────────────

    #[test]
    fn prepare_embed_text_truncates_body() {
        let long_body: String = (0..600)
            .map(|i| format!("word{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let result = prepare_embed_text("Title", &[], &long_body);

        let word_count = result.lines().last().unwrap().split_whitespace().count();
        assert_eq!(word_count, MAX_BODY_WORDS);
    }

    #[test]
    fn prepare_embed_text_joins_headings() {
        let headings = vec!["Introduction".to_string(), "Summary".to_string()];
        let result = prepare_embed_text("My Note", &headings, "Some body text.");

        assert!(result.starts_with("My Note\n"));
        assert!(result.contains("Introduction | Summary"));
        assert!(result.ends_with("Some body text."));
    }

    #[test]
    fn prepare_embed_text_no_headings() {
        let result = prepare_embed_text("Title", &[], "Body here.");
        assert_eq!(result, "Title\nBody here.");
    }

    #[test]
    fn prepare_embed_text_short_body_unchanged() {
        let body = "Short body with a few words.";
        let result = prepare_embed_text("T", &[], body);
        assert!(result.contains(body));
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
}
