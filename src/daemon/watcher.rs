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

#[cfg(feature = "embeddings")]
use crate::vault::embeddings::{EmbeddingModel, EmbeddingStore};

#[cfg(feature = "embeddings")]
use super::indexer;

const DEBOUNCE_TIMEOUT: Duration = Duration::from_millis(500);
const EVENT_CHANNEL_CAPACITY: usize = 256;

#[cfg(feature = "embeddings")]
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
                    for event in events {
                        process_event(
                            &vault_root,
                            &index,
                            tantivy.as_deref(),
                            &embedding_model,
                            &embedding_store,
                            &embedding_cache_path,
                            &event.path,
                        );
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

#[cfg(not(feature = "embeddings"))]
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
                    for event in events {
                        process_event(&vault_root, &index, tantivy.as_deref(), &event.path);
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
        Some("md") => true,
        Some(_) => false,
        None => absolute.to_string_lossy().ends_with(".md"),
    }
}

fn is_obsidian_dir(relative: &Path) -> bool {
    relative
        .components()
        .next()
        .is_some_and(|component| component.as_os_str() == ".obsidian")
}

#[cfg(feature = "embeddings")]
fn process_event(
    vault_root: &Path,
    index: &Arc<RwLock<VaultIndex>>,
    tantivy: Option<&TantivyIndex>,
    embedding_model: &EmbeddingModel,
    embedding_store: &Arc<RwLock<EmbeddingStore>>,
    embedding_cache_path: &Path,
    absolute: &Path,
) {
    if !should_process_path(vault_root, absolute) {
        return;
    }

    let relative = match absolute.strip_prefix(vault_root) {
        Ok(relative) => relative.to_path_buf(),
        Err(_) => return,
    };

    if absolute.exists() {
        tracing::debug!(
            path = %relative.display(),
            "daemon watcher reindex (create/modify)"
        );
        match index.write() {
            Ok(mut index_guard) => {
                if let Err(err) = index_guard.reindex_file(vault_root, &relative) {
                    tracing::warn!(path = %relative.display(), error = %err, "daemon reindex failed");
                    return;
                }

                let meta = index_guard.get_note(&relative).cloned();
                if let Some(tantivy_index) = tantivy
                    && let Some(ref metadata) = meta
                    && let Err(err) = tantivy_index.reindex_file(vault_root, &relative, metadata)
                {
                    tracing::warn!(
                        path = %relative.display(),
                        error = %err,
                        "daemon tantivy reindex failed"
                    );
                }

                if let Some(metadata) = meta.as_ref() {
                    indexer::embed_note(
                        vault_root,
                        &relative,
                        metadata,
                        embedding_model,
                        embedding_store,
                        embedding_cache_path,
                    );
                }
            }
            Err(err) => {
                tracing::error!(error = %err, "daemon index lock poisoned");
            }
        }
    } else {
        tracing::debug!(path = %relative.display(), "daemon watcher remove (delete)");
        match index.write() {
            Ok(mut index_guard) => {
                index_guard.remove_file(&relative);

                if let Some(tantivy_index) = tantivy
                    && let Err(err) = tantivy_index.remove_file(&relative)
                {
                    tracing::warn!(
                        path = %relative.display(),
                        error = %err,
                        "daemon tantivy remove failed"
                    );
                }

                indexer::remove_note_embedding(&relative, embedding_store, embedding_cache_path);
            }
            Err(err) => {
                tracing::error!(error = %err, "daemon index lock poisoned");
            }
        }
    }
}

#[cfg(not(feature = "embeddings"))]
fn process_event(
    vault_root: &Path,
    index: &Arc<RwLock<VaultIndex>>,
    tantivy: Option<&TantivyIndex>,
    absolute: &Path,
) {
    if !should_process_path(vault_root, absolute) {
        return;
    }

    let relative = match absolute.strip_prefix(vault_root) {
        Ok(relative) => relative.to_path_buf(),
        Err(_) => return,
    };

    if absolute.exists() {
        tracing::debug!(
            path = %relative.display(),
            "daemon watcher reindex (create/modify)"
        );
        match index.write() {
            Ok(mut index_guard) => {
                if let Err(err) = index_guard.reindex_file(vault_root, &relative) {
                    tracing::warn!(path = %relative.display(), error = %err, "daemon reindex failed");
                    return;
                }

                if let Some(tantivy_index) = tantivy
                    && let Some(meta) = index_guard.get_note(&relative)
                    && let Err(err) = tantivy_index.reindex_file(vault_root, &relative, meta)
                {
                    tracing::warn!(
                        path = %relative.display(),
                        error = %err,
                        "daemon tantivy reindex failed"
                    );
                }
            }
            Err(err) => {
                tracing::error!(error = %err, "daemon index lock poisoned");
            }
        }
    } else {
        tracing::debug!(path = %relative.display(), "daemon watcher remove (delete)");
        match index.write() {
            Ok(mut index_guard) => {
                index_guard.remove_file(&relative);

                if let Some(tantivy_index) = tantivy
                    && let Err(err) = tantivy_index.remove_file(&relative)
                {
                    tracing::warn!(
                        path = %relative.display(),
                        error = %err,
                        "daemon tantivy remove failed"
                    );
                }
            }
            Err(err) => {
                tracing::error!(error = %err, "daemon index lock poisoned");
            }
        }
    }
}
