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
    /// Stable content hash of each note's prepared embed text, kept in lockstep
    /// with `embeddings`. Lets a cold start reuse a cached vector when a note's
    /// embed input is unchanged (see `build_or_load_embedding_store`).
    hashes: HashMap<PathBuf, u64>,
    dim: usize,
    /// Identity of the model whose vectors this store holds (see
    /// `EmbeddingModel::model_id`). `None` for stores that were never associated
    /// with a model (e.g. unit-test fixtures). Persisted so a cold start can
    /// invalidate the cache on a same-dimension model swap.
    model_id: Option<String>,
}

/// Serde-friendly intermediate for bincode persistence.
/// Avoids `PathBuf` encoding issues by converting to `String`.
///
/// The per-entry `u64` is the prepared-embed-text hash (see `embed_text_hash`).
/// This is a format change from the pre-incremental cache; an older cache fails
/// to deserialize and is treated as absent, forcing one full rebuild that then
/// writes the hashed format (incremental from then on).
#[derive(serde::Serialize, serde::Deserialize)]
struct EmbeddingCacheData {
    dim: usize,
    /// Identity of the model that produced these vectors (see
    /// `EmbeddingModel::model_id`); `None` on stores never bound to a model.
    model_id: Option<String>,
    entries: Vec<(String, u64, Vec<f32>)>,
}

/// Stable 64-bit hash of a note's prepared embed text, used only to detect
/// whether the embedding input changed between runs. Derived from SHA-256
/// (already a dependency) so it is deterministic across processes and std
/// versions — unlike `DefaultHasher`, whose seed is not guaranteed stable.
/// Not security-sensitive: a collision only risks reusing one stale vector.
pub(crate) fn embed_text_hash(text: &str) -> u64 {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(text.as_bytes());
    u64::from_le_bytes(
        digest[..8]
            .try_into()
            .expect("SHA-256 digest is 32 bytes, so [..8] always fits"),
    )
}

impl EmbeddingStore {
    /// Create an empty store for embeddings of the given dimensionality.
    pub fn new(dim: usize) -> Self {
        Self {
            embeddings: HashMap::new(),
            hashes: HashMap::new(),
            dim,
            model_id: None,
        }
    }

    /// Record the identity of the model whose vectors this store holds.
    pub fn set_model_id(&mut self, model_id: impl Into<String>) {
        self.model_id = Some(model_id.into());
    }

    /// The identity of the model whose vectors this store holds, if known.
    pub fn model_id(&self) -> Option<&str> {
        self.model_id.as_deref()
    }

    /// Insert or replace the embedding for a note, recording the hash of the
    /// prepared embed text (see `embed_text_hash`) so a later cold start can
    /// tell whether the note's embedding input changed.
    ///
    /// Vectors with a dimension mismatch are rejected (logged + skipped, and
    /// no hash is recorded) to prevent garbage cosine-similarity results from
    /// a misconfigured API backend.
    pub fn insert(&mut self, path: PathBuf, vec: Vec<f32>, hash: u64) {
        if vec.len() != self.dim {
            tracing::warn!(
                path = %path.display(),
                expected = self.dim,
                got = vec.len(),
                "embedding dimension mismatch — skipping insert"
            );
            return;
        }
        self.hashes.insert(path.clone(), hash);
        self.embeddings.insert(path, vec);
    }

    /// Remove a note's embedding (and its recorded hash).
    pub fn remove(&mut self, path: &Path) {
        self.embeddings.remove(path);
        self.hashes.remove(path);
    }

    /// Retrieve a note's embedding vector.
    pub fn get(&self, path: &Path) -> Option<&[f32]> {
        self.embeddings.get(path).map(|v| v.as_slice())
    }

    /// The recorded embed-text hash for a note, if present.
    pub fn hash_of(&self, path: &Path) -> Option<u64> {
        self.hashes.get(path).copied()
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
            model_id: self.model_id.clone(),
            entries: self
                .embeddings
                .iter()
                .map(|(p, v)| {
                    (
                        p.to_string_lossy().into_owned(),
                        self.hashes.get(p).copied().unwrap_or(0),
                        v.clone(),
                    )
                })
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
        let mut hashes = HashMap::with_capacity(data.entries.len());
        for (path_str, hash, vec) in data.entries {
            if vec.len() != data.dim {
                tracing::warn!(
                    path = %path_str,
                    expected = data.dim,
                    got = vec.len(),
                    "skipping cache entry with mismatched embedding dimension"
                );
                continue;
            }
            let path = PathBuf::from(path_str);
            hashes.insert(path.clone(), hash);
            embeddings.insert(path, vec);
        }

        Ok(Self {
            embeddings,
            hashes,
            dim: data.dim,
            model_id: data.model_id,
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
    /// Stable identity of the loaded model (backend + model name). Written into
    /// the embedding cache so a cold start can detect a model swap even when the
    /// new model has the same dimension, and rebuild rather than reuse vectors
    /// from a different vector space.
    model_id: String,
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

    /// Stable identity of the loaded model (see `model_id` field).
    pub fn model_id(&self) -> &str {
        &self.model_id
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

            // Identity is the fastembed variant. Stable across restarts of a
            // given binary (verified); a variant *rename* forces a safe rebuild.
            // Known limitation: a fastembed upgrade that keeps a variant's name
            // but changes its weights/tokenization would keep this key and reuse
            // stale vectors — clear the cache dir after such an upgrade.
            let model_id = format!("local:{model_enum:?}");
            let options = fastembed::InitOptions::new(model_enum).with_show_download_progress(true);

            let inner = fastembed::TextEmbedding::try_new(options)
                .map_err(|e| VaultError::Embedding(format!("model load failed: {e}")))?;

            Ok(Self {
                backend: EmbeddingBackend::Local(Box::new(std::sync::Mutex::new(inner))),
                dim,
                model_id,
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
            // Identity includes the endpoint: the same model name at a different
            // base URL is a different vector space, so both must key the cache.
            let model_id = format!("api:{base_url}:{model}");

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
                model_id,
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

/// Split the current notes into those whose cached embedding can be reused
/// (embed input unchanged since the cache was written) and those that must be
/// (re-)embedded (new note, or embed text changed). Paths present in `cached`
/// but absent from `prepared` are simply not carried forward (removed notes).
///
/// Pure and model-free so the incremental decision can be unit-tested without
/// a real embedding backend.
#[allow(clippy::type_complexity)]
fn partition_reusable<'a>(
    cached: Option<&EmbeddingStore>,
    prepared: &'a [(PathBuf, String, u64)],
) -> (Vec<(PathBuf, Vec<f32>, u64)>, Vec<&'a (PathBuf, String, u64)>) {
    let mut reuse: Vec<(PathBuf, Vec<f32>, u64)> = Vec::new();
    let mut embed: Vec<&'a (PathBuf, String, u64)> = Vec::new();

    for entry in prepared {
        let (path, _text, hash) = entry;
        let reused = cached.and_then(|c| {
            if c.hash_of(path) == Some(*hash) {
                c.get(path).map(|vec| vec.to_vec())
            } else {
                None
            }
        });
        match reused {
            Some(vec) => reuse.push((path.clone(), vec, *hash)),
            None => embed.push(entry),
        }
    }

    (reuse, embed)
}

/// Load cached embeddings and refresh them incrementally from note entries.
///
/// Only notes that are new or whose prepared embed text changed since the
/// cache was written are (re-)embedded; unchanged notes reuse their cached
/// vector, and removed notes are dropped. This turns a cold start on a vault
/// that changed by a handful of notes from a full re-embed of every note
/// (previously triggered by any note-count change) into embedding only the
/// deltas — the difference between a ~1-minute and a sub-second startup on a
/// large vault. The embed input hash (not raw content) is the reuse key, so a
/// change that doesn't alter the prepared text (e.g. frontmatter-only, or body
/// beyond the truncation limit) correctly reuses the cached vector.
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
    // A cache is reusable only if it was produced by the SAME model — same
    // dimension AND same model identity. A same-dimension model swap would
    // otherwise silently mix vectors from two different vector spaces.
    let cached = EmbeddingStore::load(cache_path)
        .ok()
        .filter(|c| c.dim() == model.dim() && c.model_id() == Some(model.model_id()));

    // Prepare embed text + hash for every current note. Reading all note files
    // is cheap (filesystem cache); the cost this avoids is model inference.
    //
    // A note that is still indexed but whose file read fails *transiently* must
    // NOT be mistaken for a removal — dropping its cached vector would delete a
    // good embedding (and force a re-embed) over a momentary I/O blip. So on a
    // read failure we preserve the cached vector if we have one, and only skip
    // (cannot embed) a note that is both unreadable and uncached.
    let mut prepared: Vec<(PathBuf, String, u64)> = Vec::new();
    let mut carried: Vec<(PathBuf, Vec<f32>, u64)> = Vec::new();
    for (path, meta) in note_entries {
        match super::fs::read_file(vault_root, path) {
            Ok(content) => {
                let body = super::frontmatter::get_body(&content);
                let heading_texts: Vec<String> =
                    meta.headings.iter().map(|h| h.text.clone()).collect();
                let text = prepare_embed_text(&meta.title, &heading_texts, body);
                let hash = embed_text_hash(&text);
                prepared.push((path.clone(), text, hash));
            }
            Err(err) => match cached.as_ref().and_then(|c| {
                c.get(path)
                    .map(|vec| (vec.to_vec(), c.hash_of(path).unwrap_or(0)))
            }) {
                Some((vec, hash)) => {
                    tracing::warn!(path = %path.display(), error = %err,
                        "note read failed at cold start; preserving cached embedding");
                    carried.push((path.clone(), vec, hash));
                }
                None => {
                    tracing::warn!(path = %path.display(), error = %err,
                        "note read failed at cold start and not cached; skipping");
                }
            },
        }
    }

    let (reuse, to_embed) = partition_reusable(cached.as_ref(), &prepared);

    tracing::info!(
        cache = %cache_path.display(),
        total = note_entries.len(),
        reused = reuse.len() + carried.len(),
        to_embed = to_embed.len(),
        had_cache = cached.is_some(),
        "refreshing embedding store"
    );

    let mut store = EmbeddingStore::new(model.dim());
    store.set_model_id(model.model_id());
    for (path, vec, hash) in carried {
        store.insert(path, vec, hash);
    }
    for (path, vec, hash) in reuse {
        store.insert(path, vec, hash);
    }

    for chunk in to_embed.chunks(BATCH_SIZE) {
        let texts: Vec<&str> = chunk.iter().map(|(_, text, _)| text.as_str()).collect();
        match model.embed_batch(&texts) {
            Ok(vectors) => {
                for ((path, _, hash), vector) in chunk.iter().zip(vectors) {
                    store.insert(path.clone(), vector, *hash);
                }
            }
            Err(err) => {
                tracing::warn!(error = %err, "embedding batch failed, skipping chunk");
            }
        }
    }

    // Persist only when the store actually changed vs the loaded cache, so an
    // unchanged vault doesn't rewrite an identical cache file every startup.
    let changed = match &cached {
        None => true,
        Some(c) => !to_embed.is_empty() || c.len() != store.len(),
    };
    if changed
        && let Err(err) = store.save(cache_path)
    {
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
        store.insert(PathBuf::from("a.md"), vec![1.0, 0.0, 0.0], 10);
        store.insert(PathBuf::from("b.md"), vec![0.0, 1.0, 0.0], 20);
        store.insert(PathBuf::from("c.md"), vec![0.7, 0.7, 0.0], 30);
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

        store.insert(PathBuf::from("d.md"), vec![0.9, 0.1, 0.0], 40);
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

    // ── incremental cache: hashing + reuse partition ───────────────

    #[test]
    fn embed_text_hash_is_deterministic_and_content_sensitive() {
        assert_eq!(embed_text_hash("hello world"), embed_text_hash("hello world"));
        assert_ne!(embed_text_hash("hello world"), embed_text_hash("hello  world"));
        assert_ne!(embed_text_hash("a"), embed_text_hash("b"));
    }

    #[test]
    fn insert_records_hash_and_remove_clears_it() {
        let mut store = EmbeddingStore::new(3);
        store.insert(PathBuf::from("a.md"), vec![1.0, 0.0, 0.0], 42);
        assert_eq!(store.hash_of(Path::new("a.md")), Some(42));
        store.remove(Path::new("a.md"));
        assert_eq!(store.hash_of(Path::new("a.md")), None);
        assert!(store.get(Path::new("a.md")).is_none());
    }

    #[test]
    fn dim_mismatch_insert_records_no_hash() {
        let mut store = EmbeddingStore::new(3);
        // Wrong length: rejected, and no hash should linger.
        store.insert(PathBuf::from("bad.md"), vec![1.0, 0.0], 7);
        assert!(store.get(Path::new("bad.md")).is_none());
        assert_eq!(store.hash_of(Path::new("bad.md")), None);
    }

    #[test]
    fn persistence_preserves_hashes() {
        let store = make_store();
        let dir = tempfile::tempdir().unwrap();
        let cache_path = dir.path().join("embeddings.bin");
        store.save(&cache_path).unwrap();
        let loaded = EmbeddingStore::load(&cache_path).unwrap();
        assert_eq!(loaded.hash_of(Path::new("a.md")), Some(10));
        assert_eq!(loaded.hash_of(Path::new("b.md")), Some(20));
        assert_eq!(loaded.hash_of(Path::new("c.md")), Some(30));
    }

    #[test]
    fn persistence_preserves_model_id() {
        let mut store = make_store();
        assert_eq!(store.model_id(), None);
        store.set_model_id("local:BGESmallENV15");
        let dir = tempfile::tempdir().unwrap();
        let cache_path = dir.path().join("embeddings.bin");
        store.save(&cache_path).unwrap();
        let loaded = EmbeddingStore::load(&cache_path).unwrap();
        assert_eq!(loaded.model_id(), Some("local:BGESmallENV15"));
    }

    #[test]
    fn partition_reuses_unchanged_embeds_changed_and_new() {
        // Cache holds a.md (hash 10) and b.md (hash 20).
        let cached = make_store(); // a=10, b=20, c=30
        let prepared = vec![
            (PathBuf::from("a.md"), "text a".to_string(), 10), // unchanged -> reuse
            (PathBuf::from("b.md"), "text b changed".to_string(), 99), // changed -> embed
            (PathBuf::from("new.md"), "brand new".to_string(), 77), // new -> embed
            // c.md absent from prepared -> dropped (removed note)
        ];

        let (reuse, embed) = partition_reusable(Some(&cached), &prepared);

        let reuse_paths: Vec<_> = reuse.iter().map(|(p, _, _)| p.clone()).collect();
        assert_eq!(reuse_paths, vec![PathBuf::from("a.md")]);
        // reused vector comes from the cache, with its hash carried
        assert_eq!(reuse[0].1, vec![1.0, 0.0, 0.0]);
        assert_eq!(reuse[0].2, 10);

        let embed_paths: Vec<_> = embed.iter().map(|(p, _, _)| p.clone()).collect();
        assert_eq!(
            embed_paths,
            vec![PathBuf::from("b.md"), PathBuf::from("new.md")]
        );
    }

    #[test]
    fn partition_without_cache_embeds_everything() {
        let prepared = vec![
            (PathBuf::from("a.md"), "a".to_string(), 1),
            (PathBuf::from("b.md"), "b".to_string(), 2),
        ];
        let (reuse, embed) = partition_reusable(None, &prepared);
        assert!(reuse.is_empty());
        assert_eq!(embed.len(), 2);
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
