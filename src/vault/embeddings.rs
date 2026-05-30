//! Embedding store and model wrapper for semantic search (Layer 2).
//!
//! Gated behind `#[cfg(has_embeddings)]` (either `embeddings` or `embeddings-api`
//! Cargo feature). Provides:
//! - `EmbeddingStore`: in-memory HashMap of note embeddings with brute-force
//!   cosine similarity search and bincode persistence.
//! - `EmbeddingModel`: backend-agnostic wrapper supporting local fastembed
//!   (`--features embeddings`) and OpenAI-compatible API (`--features embeddings-api`).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[cfg(feature = "embeddings")]
use fastembed::ModelTrait;

use crate::config::EmbeddingProvider;
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
    ///
    /// Vectors with a dimension mismatch are rejected (logged + skipped)
    /// to prevent garbage cosine-similarity results from a misconfigured
    /// API backend.
    pub fn insert(&mut self, path: PathBuf, vec: Vec<f32>) {
        if vec.len() != self.dim {
            tracing::warn!(
                path = %path.display(),
                expected = self.dim,
                got = vec.len(),
                "embedding dimension mismatch — skipping insert"
            );
            return;
        }
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

        let cmp = |a: &(PathBuf, f32), b: &(PathBuf, f32)| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        };

        if top_k < scored.len() {
            scored.select_nth_unstable_by(top_k, cmp);
            scored.truncate(top_k);
            scored.sort_unstable_by(cmp);
        } else {
            scored.sort_unstable_by(cmp);
        }
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

// ── EmbeddingBackend ───────────────────────────────────────────────────

enum EmbeddingBackend {
    #[cfg(feature = "embeddings")]
    Local(Box<std::sync::Mutex<fastembed::TextEmbedding>>),

    #[cfg(feature = "embeddings-api")]
    Api {
        client: reqwest::blocking::Client,
        base_url: String,
        model: String,
        api_key: zeroize::Zeroizing<String>,
    },
}

// ── EmbeddingModel ─────────────────────────────────────────────────────

/// Backend-agnostic embedding model supporting local fastembed and
/// OpenAI-compatible API backends.
pub struct EmbeddingModel {
    backend: EmbeddingBackend,
    dim: usize,
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
        match &self.backend {
            #[cfg(feature = "embeddings")]
            EmbeddingBackend::Local(inner) => {
                let mut model = inner
                    .lock()
                    .map_err(|e| VaultError::Embedding(format!("model lock poisoned: {e}")))?;
                model
                    .embed(texts, Some(64))
                    .map_err(|e| VaultError::Embedding(format!("embed failed: {e}")))
            }
            #[cfg(feature = "embeddings-api")]
            EmbeddingBackend::Api {
                client,
                base_url,
                model,
                api_key,
            } => embed_batch_api(client, base_url, model, api_key, texts),
        }
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

    // ── Local backend (fastembed) ──────────────────────────────────────

    #[cfg(feature = "embeddings")]
    async fn load_local(model_name: &str) -> VaultResult<Self> {
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
                backend: EmbeddingBackend::Local(Box::new(std::sync::Mutex::new(inner))),
                dim,
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

            let base_url = read_env_with_fallback("OBSIDIAN_EMBEDDING_API_BASE", "OPENAI_BASE_URL")
                .unwrap_or_else(|| "https://api.openai.com/v1".to_string());

            let model = read_env_with_fallback("OBSIDIAN_EMBEDDING_API_MODEL", "OPENAI_MODEL")
                .unwrap_or(model_name);

            let client = build_api_client()?;

            let dim = match parse_usize_env("OBSIDIAN_EMBEDDING_DIM") {
                Some(d) => {
                    tracing::info!(dim = d, "using explicit embedding dimension");
                    d
                }
                None => {
                    tracing::info!("probing embedding API for dimension…");
                    probe_api_dimension(&client, &base_url, &model, &api_key)?
                }
            };

            tracing::info!(
                base_url = %base_url,
                model = %model,
                dim,
                "API embedding backend ready"
            );

            Ok(Self {
                backend: EmbeddingBackend::Api {
                    client,
                    base_url,
                    model,
                    api_key,
                },
                dim,
            })
        })
        .await
        .map_err(|e| VaultError::Embedding(format!("spawn_blocking join error: {e}")))?
    }

    #[cfg(not(feature = "embeddings-api"))]
    async fn load_api(_model_name: &str) -> VaultResult<Self> {
        Err(VaultError::Embedding(
            "API embedding backend not compiled (needs --features embeddings-api)".into(),
        ))
    }
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
fn build_api_client() -> Result<reqwest::blocking::Client, VaultError> {
    let mut builder =
        reqwest::blocking::ClientBuilder::new().timeout(std::time::Duration::from_secs(30));

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
fn probe_api_dimension(
    client: &reqwest::blocking::Client,
    base_url: &str,
    model: &str,
    api_key: &str,
) -> Result<usize, VaultError> {
    let vecs = embed_batch_api(client, base_url, model, api_key, &["dim"])?;
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
    let url = format!("{}/embeddings", base_url.trim_end_matches('/'));
    let body = serde_json::json!({
        "model": model,
        "input": texts,
        "encoding_format": "float",
    });

    const MAX_RETRIES: u8 = 3;
    let mut attempt = 0u8;
    loop {
        let response = client
            .post(&url)
            .header("Authorization", format!("Bearer {api_key}"))
            .json(&body)
            .send()
            .map_err(|e| VaultError::Embedding(format!("embedding API request failed: {e}")))?;

        let status = response.status();
        if status.as_u16() == 429 && attempt < MAX_RETRIES {
            let wait = response
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(1u64 << attempt)
                .min(30);
            attempt += 1;
            tracing::warn!(
                retry_after_secs = wait,
                attempt = attempt,
                max_retries = MAX_RETRIES,
                "embedding API rate limited (attempt {attempt}/{MAX_RETRIES})"
            );
            std::thread::sleep(std::time::Duration::from_secs(wait));
            continue;
        }

        if !status.is_success() {
            let body_text = response.text().unwrap_or_default();
            return Err(VaultError::Embedding(format!(
                "embedding API error {status}: {body_text}"
            )));
        }

        let resp: serde_json::Value = response
            .json()
            .map_err(|e| VaultError::Embedding(format!("embedding API parse error: {e}")))?;

        return parse_embedding_response(&resp);
    }
}

/// Parse an OpenAI-compatible embedding API response into embedding vectors.
///
/// Items are sorted by the `index` field when present, falling back to array
/// position for providers that omit it. This ensures correct input→output
/// alignment even when providers return items out of order.
#[cfg(feature = "embeddings-api")]
fn parse_embedding_response(resp: &serde_json::Value) -> Result<Vec<Vec<f32>>, VaultError> {
    let data = resp["data"]
        .as_array()
        .ok_or_else(|| VaultError::Embedding("missing 'data' array in API response".into()))?;

    let mut indexed: Vec<(usize, Vec<f32>)> = Vec::with_capacity(data.len());
    for (array_pos, item) in data.iter().enumerate() {
        let idx = item["index"]
            .as_u64()
            .map(|i| i as usize)
            .unwrap_or(array_pos);
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
                    .map(|f| f as f32)
            })
            .collect::<Result<Vec<f32>, _>>()?;
        indexed.push((idx, vec));
    }

    indexed.sort_by_key(|(idx, _)| *idx);
    Ok(indexed.into_iter().map(|(_, vec)| vec).collect())
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

// ── Shared embedding index builder ────────────────────────────────────

const BATCH_SIZE: usize = 64;

/// Load cached embeddings or rebuild from note entries.
///
/// The caller is responsible for lock acquisition on the index — this
/// function receives pre-extracted note entries to stay decoupled from
/// any particular lock strategy.
pub(crate) fn build_or_load_embedding_store(
    cache_path: &Path,
    vault_root: &Path,
    note_entries: &[(PathBuf, crate::models::NoteMetadata)],
    model: &EmbeddingModel,
) -> VaultResult<EmbeddingStore> {
    if let Ok(store) = EmbeddingStore::load(cache_path) {
        if store.dim() == model.dim() && store.len() == note_entries.len() {
            tracing::info!(
                cache = %cache_path.display(),
                cached = store.len(),
                "loaded embedding cache"
            );
            return Ok(store);
        }
        tracing::info!(
            cache = %cache_path.display(),
            cached = store.len(),
            current = note_entries.len(),
            "embedding cache stale, rebuilding"
        );
    }

    let entries: Vec<(PathBuf, String)> = note_entries
        .iter()
        .filter_map(|(path, meta)| {
            let content = super::fs::read_file(vault_root, path).ok()?;
            let body = super::frontmatter::get_body(&content);
            let heading_texts: Vec<String> = meta.headings.iter().map(|h| h.text.clone()).collect();
            let text = prepare_embed_text(&meta.title, &heading_texts, body);
            Some((path.clone(), text))
        })
        .collect();

    let mut store = EmbeddingStore::new(model.dim());
    for chunk in entries.chunks(BATCH_SIZE) {
        let texts: Vec<&str> = chunk.iter().map(|(_, text)| text.as_str()).collect();
        match model.embed_batch(&texts) {
            Ok(vectors) => {
                for ((path, _), vector) in chunk.iter().zip(vectors) {
                    store.insert(path.clone(), vector);
                }
            }
            Err(err) => {
                tracing::warn!(error = %err, "embedding batch failed, skipping chunk");
            }
        }
    }

    if let Err(err) = store.save(cache_path) {
        tracing::warn!(error = %err, "failed to save embedding cache");
    }

    Ok(store)
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
    let vault_id = crate::daemon::home::compute_vault_id(vault_root)?;
    let target = semantic_home
        .join("vaults")
        .join(vault_id)
        .join("embeddings.bin");
    if target.exists() {
        return Ok(LegacyCacheMigration::AlreadyPresent(target));
    }

    let legacy_source = vault_root
        .join(".obsidian")
        .join("obsidian-mcp")
        .join("embeddings.bin");
    let new_source = vault_root
        .join(".obsidian-mcp")
        .join("embeddings")
        .join("embeddings.bin");

    let source = if legacy_source.is_file() {
        legacy_source
    } else if new_source.is_file() {
        new_source
    } else {
        return Ok(LegacyCacheMigration::NotFound);
    };

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
    fn migrate_legacy_cache_uses_new_source_as_fallback() {
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
    fn migrate_legacy_cache_prefers_legacy_over_new() {
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
            b"legacy-bytes",
            "legacy source should be preferred over new"
        );
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

        #[test]
        fn parse_valid_single_embedding() {
            let resp = serde_json::json!({
                "data": [{"embedding": [0.1, 0.2, 0.3]}]
            });
            let result = parse_embedding_response(&resp).unwrap();
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
            let result = parse_embedding_response(&resp).unwrap();
            assert_eq!(result.len(), 2);
            assert_eq!(result[0], vec![0.1f32, 0.2]);
            assert_eq!(result[1], vec![0.3f32, 0.4]);
        }

        #[test]
        fn parse_missing_data_field() {
            let resp = serde_json::json!({"object": "list"});
            let err = parse_embedding_response(&resp).unwrap_err();
            assert!(err.to_string().contains("missing 'data' array"));
        }

        #[test]
        fn parse_missing_embedding_in_item() {
            let resp = serde_json::json!({
                "data": [{"index": 0}]
            });
            let err = parse_embedding_response(&resp).unwrap_err();
            assert!(err.to_string().contains("missing 'embedding' array"));
        }

        #[test]
        fn parse_non_numeric_value_in_vector() {
            let resp = serde_json::json!({
                "data": [{"embedding": [0.1, "bad", 0.3]}]
            });
            let err = parse_embedding_response(&resp).unwrap_err();
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
            let result = parse_embedding_response(&resp).unwrap();
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
            let result = parse_embedding_response(&resp).unwrap();
            assert_eq!(result[0], vec![0.1f32, 0.2]);
            assert_eq!(result[1], vec![0.3f32, 0.4]);
        }

        #[test]
        fn parse_empty_data_array() {
            let resp = serde_json::json!({"data": []});
            let result = parse_embedding_response(&resp).unwrap();
            assert!(result.is_empty());
        }

        #[test]
        fn parse_empty_embedding_vector() {
            let resp = serde_json::json!({
                "data": [{"embedding": []}]
            });
            let result = parse_embedding_response(&resp).unwrap();
            assert_eq!(result.len(), 1);
            assert!(result[0].is_empty());
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
