//! Daemon-side indexing helpers shared by per-vault contexts and watcher updates.

use std::path::PathBuf;

#[cfg(feature = "embeddings")]
use crate::error::{VaultError, VaultResult};
#[cfg(feature = "embeddings")]
use crate::models::NoteMetadata;
#[cfg(feature = "embeddings")]
use crate::vault::{frontmatter, fs, index::VaultIndex};
#[cfg(feature = "embeddings")]
use std::path::Path;
#[cfg(feature = "embeddings")]
use std::sync::{Arc, RwLock};

#[cfg(feature = "embeddings")]
use crate::vault::embeddings::{self, EmbeddingModel, EmbeddingStore};

#[cfg(feature = "embeddings")]
const BATCH_SIZE: usize = 64;

/// Min-max normalize BM25 scores to `[0, 1]`.
///
/// When all scores are identical, each normalized score is `1.0`.
pub(crate) fn normalize_bm25_scores(hits: &[(PathBuf, f32)]) -> Vec<(PathBuf, f32)> {
    if hits.is_empty() {
        return Vec::new();
    }

    let min = hits
        .iter()
        .map(|(_, score)| *score)
        .fold(f32::INFINITY, f32::min);
    let max = hits
        .iter()
        .map(|(_, score)| *score)
        .fold(f32::NEG_INFINITY, f32::max);
    let range = max - min;

    hits.iter()
        .map(|(path, score)| {
            let normalized = if range == 0.0 {
                1.0
            } else {
                (score - min) / range
            };
            (path.clone(), normalized)
        })
        .collect()
}

#[cfg(feature = "embeddings")]
pub(crate) fn build_or_load_embeddings(
    vault_root: &Path,
    index: &Arc<RwLock<VaultIndex>>,
    model: &EmbeddingModel,
    cache_path: &Path,
) -> VaultResult<EmbeddingStore> {
    let index_guard = index
        .read()
        .map_err(|err| VaultError::Other(format!("daemon index lock poisoned: {err}")))?;
    let note_count = index_guard.notes().len();

    if let Ok(store) = EmbeddingStore::load(cache_path) {
        if store.dim() == model.dim() && store.len() == note_count {
            tracing::info!(
                cache = %cache_path.display(),
                cached = store.len(),
                "loaded daemon embedding cache"
            );
            return Ok(store);
        }
        tracing::info!(
            cache = %cache_path.display(),
            cached = store.len(),
            current = note_count,
            "daemon embedding cache stale, rebuilding"
        );
    }

    let entries: Vec<(PathBuf, String)> = index_guard
        .notes()
        .iter()
        .filter_map(|(path, meta)| {
            let content = fs::read_file(vault_root, path).ok()?;
            let body = frontmatter::get_body(&content);
            let heading_texts: Vec<String> = meta.headings.iter().map(|h| h.text.clone()).collect();
            let text = embeddings::prepare_embed_text(&meta.title, &heading_texts, body);
            Some((path.clone(), text))
        })
        .collect();
    drop(index_guard);

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
                tracing::warn!(error = %err, "daemon embedding batch failed, skipping chunk");
            }
        }
    }

    save_embedding_cache(cache_path, &store);
    Ok(store)
}

#[cfg(feature = "embeddings")]
pub(crate) fn embed_note(
    vault_root: &Path,
    relative: &Path,
    meta: &NoteMetadata,
    model: &EmbeddingModel,
    store: &Arc<RwLock<EmbeddingStore>>,
    cache_path: &Path,
) {
    let Ok(content) = fs::read_file(vault_root, relative) else {
        return;
    };

    let body = frontmatter::get_body(&content);
    let heading_texts: Vec<String> = meta.headings.iter().map(|h| h.text.clone()).collect();
    let text = embeddings::prepare_embed_text(&meta.title, &heading_texts, body);

    match model.embed_one(&text) {
        Ok(vector) => match store.write() {
            Ok(mut store_guard) => {
                store_guard.insert(relative.to_path_buf(), vector);
                save_embedding_cache(cache_path, &store_guard);
            }
            Err(err) => {
                tracing::error!(error = %err, "daemon embedding store lock poisoned");
            }
        },
        Err(err) => {
            tracing::warn!(
                path = %relative.display(),
                error = %err,
                "daemon embedding failed"
            );
        }
    }
}

#[cfg(feature = "embeddings")]
pub(crate) fn remove_note_embedding(
    relative: &Path,
    store: &Arc<RwLock<EmbeddingStore>>,
    cache_path: &Path,
) {
    match store.write() {
        Ok(mut store_guard) => {
            store_guard.remove(relative);
            save_embedding_cache(cache_path, &store_guard);
        }
        Err(err) => {
            tracing::error!(error = %err, "daemon embedding store lock poisoned");
        }
    }
}

#[cfg(feature = "embeddings")]
pub(crate) fn save_embedding_cache(cache_path: &Path, store: &EmbeddingStore) {
    if let Err(err) = store.save(cache_path) {
        tracing::warn!(
            cache = %cache_path.display(),
            error = %err,
            "failed to save daemon embedding cache"
        );
    }
}
