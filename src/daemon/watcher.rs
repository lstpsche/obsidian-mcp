//! Daemon-side filesystem watcher for per-vault context synchronization.

use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use notify::RecursiveMode;
use notify_debouncer_mini::{DebounceEventResult, Debouncer, new_debouncer};
use tokio::runtime::Handle;

use crate::error::{VaultError, VaultResult};
use crate::vault::index::VaultIndex;
use crate::vault::tantivy_index::TantivyIndex;

#[cfg(has_embeddings)]
use crate::vault::embeddings::{EmbeddingModel, EmbeddingStore};

#[cfg(has_embeddings)]
use super::indexer;

const DEBOUNCE_TIMEOUT: Duration = Duration::from_millis(500);
const EVENT_CHANNEL_CAPACITY: usize = 256;

#[cfg(has_embeddings)]
pub fn start_watcher(
    vault_root: PathBuf,
    index: Arc<RwLock<VaultIndex>>,
    tantivy: Option<Arc<TantivyIndex>>,
    embedding_model: Arc<EmbeddingModel>,
    embedding_store: Arc<RwLock<EmbeddingStore>>,
    embedding_cache_path: PathBuf,
) -> VaultResult<Debouncer<notify::RecommendedWatcher>> {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<DebounceEventResult>(EVENT_CHANNEL_CAPACITY);
    let rt = Handle::current();

    let mut debouncer = new_debouncer(DEBOUNCE_TIMEOUT, move |result: DebounceEventResult| {
        let tx = tx.clone();
        rt.spawn(async move {
            if let Err(err) = tx.send(result).await {
                tracing::error!("daemon watcher channel closed: {err}");
            }
        });
    })
    .map_err(|err| VaultError::Watcher(err.to_string()))?;

    debouncer
        .watcher()
        .watch(&vault_root, RecursiveMode::Recursive)
        .map_err(|err| {
            VaultError::Watcher(format!("failed to watch {}: {err}", vault_root.display()))
        })?;

    tracing::info!(
        path = %vault_root.display(),
        "daemon watcher started for vault"
    );

    tokio::spawn(async move {
        while let Some(result) = rx.recv().await {
            match result {
                Ok(events) => {
                    let mut tantivy_dirty = false;
                    let mut embedding_dirty = false;
                    for event in events {
                        let (tv, emb) = process_event(
                            &vault_root,
                            &index,
                            tantivy.as_deref(),
                            &embedding_model,
                            &embedding_store,
                            &event.path,
                        );
                        tantivy_dirty |= tv;
                        embedding_dirty |= emb;
                    }
                    if tantivy_dirty
                        && let Some(ref tv) = tantivy
                        && let Err(err) = tv.flush()
                    {
                        tracing::warn!(error = %err, "daemon tantivy batch flush failed");
                    }
                    if embedding_dirty && let Ok(store_guard) = embedding_store.read() {
                        indexer::save_embedding_cache(&embedding_cache_path, &store_guard);
                    }
                }
                Err(err) => {
                    tracing::warn!(error = %err, "daemon watcher error");
                }
            }
        }
        tracing::debug!("daemon watcher event loop exited");
    });

    Ok(debouncer)
}

#[cfg(not(has_embeddings))]
pub fn start_watcher(
    vault_root: PathBuf,
    index: Arc<RwLock<VaultIndex>>,
    tantivy: Option<Arc<TantivyIndex>>,
) -> VaultResult<Debouncer<notify::RecommendedWatcher>> {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<DebounceEventResult>(EVENT_CHANNEL_CAPACITY);
    let rt = Handle::current();

    let mut debouncer = new_debouncer(DEBOUNCE_TIMEOUT, move |result: DebounceEventResult| {
        let tx = tx.clone();
        rt.spawn(async move {
            if let Err(err) = tx.send(result).await {
                tracing::error!("daemon watcher channel closed: {err}");
            }
        });
    })
    .map_err(|err| VaultError::Watcher(err.to_string()))?;

    debouncer
        .watcher()
        .watch(&vault_root, RecursiveMode::Recursive)
        .map_err(|err| {
            VaultError::Watcher(format!("failed to watch {}: {err}", vault_root.display()))
        })?;

    tracing::info!(
        path = %vault_root.display(),
        "daemon watcher started for vault"
    );

    tokio::spawn(async move {
        while let Some(result) = rx.recv().await {
            match result {
                Ok(events) => {
                    let mut tantivy_dirty = false;
                    for event in events {
                        tantivy_dirty |=
                            process_event(&vault_root, &index, tantivy.as_deref(), &event.path);
                    }
                    if tantivy_dirty
                        && let Some(ref tv) = tantivy
                        && let Err(err) = tv.flush()
                    {
                        tracing::warn!(error = %err, "daemon tantivy batch flush failed");
                    }
                }
                Err(err) => {
                    tracing::warn!(error = %err, "daemon watcher error");
                }
            }
        }
        tracing::debug!("daemon watcher event loop exited");
    });

    Ok(debouncer)
}

fn should_process_path(vault_root: &Path, absolute: &Path) -> bool {
    let relative = match absolute.strip_prefix(vault_root) {
        Ok(relative) => relative,
        Err(_) => return false,
    };

    if is_obsidian_dir(relative) {
        return false;
    }

    match absolute.extension().and_then(|ext| ext.to_str()) {
        Some(ext) if ext.eq_ignore_ascii_case("md") => true,
        Some(_) => false,
        None => absolute
            .to_string_lossy()
            .to_ascii_lowercase()
            .ends_with(".md"),
    }
}

fn is_obsidian_dir(relative: &Path) -> bool {
    relative
        .components()
        .next()
        .is_some_and(|component| component.as_os_str() == ".obsidian")
}

/// Returns `(tantivy_touched, embedding_touched)`.
#[cfg(has_embeddings)]
fn process_event(
    vault_root: &Path,
    index: &Arc<RwLock<VaultIndex>>,
    tantivy: Option<&TantivyIndex>,
    embedding_model: &EmbeddingModel,
    embedding_store: &Arc<RwLock<EmbeddingStore>>,
    absolute: &Path,
) -> (bool, bool) {
    if !should_process_path(vault_root, absolute) {
        return (false, false);
    }

    let relative = match absolute.strip_prefix(vault_root) {
        Ok(relative) => relative.to_path_buf(),
        Err(_) => return (false, false),
    };

    let mut tv_touched = false;
    let mut emb_touched = false;

    if absolute.exists() {
        tracing::debug!(
            path = %relative.display(),
            "daemon watcher reindex (create/modify)"
        );
        let meta = match index.write() {
            Ok(mut index_guard) => {
                if let Err(err) = index_guard.reindex_file(vault_root, &relative) {
                    tracing::warn!(path = %relative.display(), error = %err, "daemon reindex failed");
                    return (false, false);
                }
                index_guard.get_note(&relative).cloned()
            }
            Err(err) => {
                tracing::error!(error = %err, "daemon index lock poisoned");
                return (false, false);
            }
        };
        if let Some(tv) = tantivy
            && let Some(ref m) = meta
        {
            if let Err(err) = tv.reindex_file_batch(vault_root, &relative, m) {
                tracing::warn!(path = %relative.display(), error = %err, "daemon tantivy reindex failed");
            } else {
                tv_touched = true;
            }
        }
        if let Some(ref m) = meta {
            emb_touched =
                indexer::embed_note(vault_root, &relative, m, embedding_model, embedding_store);
        }
    } else {
        tracing::debug!(path = %relative.display(), "daemon watcher remove (delete)");
        match index.write() {
            Ok(mut index_guard) => index_guard.remove_file(&relative),
            Err(err) => {
                tracing::error!(error = %err, "daemon index lock poisoned");
                return (false, false);
            }
        }
        if let Some(tv) = tantivy {
            if let Err(err) = tv.remove_file_batch(&relative) {
                tracing::warn!(path = %relative.display(), error = %err, "daemon tantivy remove failed");
            } else {
                tv_touched = true;
            }
        }
        emb_touched = indexer::remove_note_embedding(&relative, embedding_store);
    }

    (tv_touched, emb_touched)
}

/// Returns whether Tantivy was touched.
#[cfg(not(has_embeddings))]
fn process_event(
    vault_root: &Path,
    index: &Arc<RwLock<VaultIndex>>,
    tantivy: Option<&TantivyIndex>,
    absolute: &Path,
) -> bool {
    if !should_process_path(vault_root, absolute) {
        return false;
    }

    let relative = match absolute.strip_prefix(vault_root) {
        Ok(relative) => relative.to_path_buf(),
        Err(_) => return false,
    };

    if absolute.exists() {
        tracing::debug!(
            path = %relative.display(),
            "daemon watcher reindex (create/modify)"
        );
        let meta = match index.write() {
            Ok(mut index_guard) => {
                if let Err(err) = index_guard.reindex_file(vault_root, &relative) {
                    tracing::warn!(path = %relative.display(), error = %err, "daemon reindex failed");
                    return false;
                }
                index_guard.get_note(&relative).cloned()
            }
            Err(err) => {
                tracing::error!(error = %err, "daemon index lock poisoned");
                return false;
            }
        };
        if let Some(tv) = tantivy
            && let Some(ref m) = meta
        {
            if let Err(err) = tv.reindex_file_batch(vault_root, &relative, m) {
                tracing::warn!(path = %relative.display(), error = %err, "daemon tantivy reindex failed");
                return false;
            }
            return true;
        }
        false
    } else {
        tracing::debug!(path = %relative.display(), "daemon watcher remove (delete)");
        match index.write() {
            Ok(mut index_guard) => index_guard.remove_file(&relative),
            Err(err) => {
                tracing::error!(error = %err, "daemon index lock poisoned");
                return false;
            }
        }
        if let Some(tv) = tantivy {
            if let Err(err) = tv.remove_file_batch(&relative) {
                tracing::warn!(path = %relative.display(), error = %err, "daemon tantivy remove failed");
                return false;
            }
            return true;
        }
        false
    }
}
