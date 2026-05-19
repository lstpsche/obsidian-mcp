//! Daemon-side indexing helpers shared by per-vault contexts and watcher updates.

#[cfg(has_embeddings)]
use crate::error::{VaultError, VaultResult};
#[cfg(has_embeddings)]
use crate::models::NoteMetadata;
#[cfg(has_embeddings)]
use crate::vault::{frontmatter, fs, index::VaultIndex};
#[cfg(has_embeddings)]
use std::path::Path;
#[cfg(has_embeddings)]
use std::sync::{Arc, RwLock};

#[cfg(has_embeddings)]
use crate::vault::embeddings::{self, EmbeddingModel, EmbeddingStore};

#[cfg(has_embeddings)]
pub(crate) fn build_or_load_embeddings(
    vault_root: &Path,
    index: &Arc<RwLock<VaultIndex>>,
    model: &EmbeddingModel,
    cache_path: &Path,
) -> VaultResult<EmbeddingStore> {
    let index_guard = index
        .read()
        .map_err(|err| VaultError::Other(format!("daemon index lock poisoned: {err}")))?;
    let note_entries: Vec<_> = index_guard
        .notes()
        .iter()
        .map(|(path, meta)| (path.clone(), meta.clone()))
        .collect();
    drop(index_guard);
    embeddings::build_or_load_embedding_store(cache_path, vault_root, &note_entries, model)
}

#[cfg(has_embeddings)]
pub(crate) fn embed_note(
    vault_root: &Path,
    relative: &Path,
    meta: &NoteMetadata,
    model: &EmbeddingModel,
    store: &Arc<RwLock<EmbeddingStore>>,
) -> bool {
    let Ok(content) = fs::read_file(vault_root, relative) else {
        return false;
    };

    let body = frontmatter::get_body(&content);
    let heading_texts: Vec<String> = meta.headings.iter().map(|h| h.text.clone()).collect();
    let text = embeddings::prepare_embed_text(&meta.title, &heading_texts, body);

    match model.embed_one(&text) {
        Ok(vector) => match store.write() {
            Ok(mut store_guard) => {
                store_guard.insert(relative.to_path_buf(), vector);
                true
            }
            Err(err) => {
                tracing::error!(error = %err, "daemon embedding store lock poisoned");
                false
            }
        },
        Err(err) => {
            tracing::warn!(
                path = %relative.display(),
                error = %err,
                "daemon embedding failed"
            );
            false
        }
    }
}

#[cfg(has_embeddings)]
pub(crate) fn remove_note_embedding(relative: &Path, store: &Arc<RwLock<EmbeddingStore>>) -> bool {
    match store.write() {
        Ok(mut store_guard) => {
            store_guard.remove(relative);
            true
        }
        Err(err) => {
            tracing::error!(error = %err, "daemon embedding store lock poisoned");
            false
        }
    }
}

#[cfg(has_embeddings)]
pub(crate) fn save_embedding_cache(cache_path: &Path, store: &EmbeddingStore) {
    if let Err(err) = store.save(cache_path) {
        tracing::warn!(
            cache = %cache_path.display(),
            error = %err,
            "failed to save daemon embedding cache"
        );
    }
}
