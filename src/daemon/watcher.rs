//! Daemon-side filesystem watcher for per-vault context synchronization.

use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use notify::RecursiveMode;
use notify_debouncer_mini::{DebounceEventResult, Debouncer, new_debouncer};
use tokio::runtime::Handle;

use crate::error::{VaultError, VaultResult};
use crate::vault::exclude::ExcludeSet;
use crate::vault::index::VaultIndex;
use crate::vault::path as vault_path;
use crate::vault::tantivy_index::TantivyIndex;

#[cfg(has_embeddings)]
use crate::vault::embedding_runtime::EmbeddingRuntime;

const DEBOUNCE_TIMEOUT: Duration = Duration::from_millis(500);
const EVENT_CHANNEL_CAPACITY: usize = 256;

#[cfg(has_embeddings)]
pub fn start_watcher(
    vault_root: PathBuf,
    index: Arc<RwLock<VaultIndex>>,
    tantivy: Option<Arc<TantivyIndex>>,
    embedding_runtime: EmbeddingRuntime,
    exclude: Arc<ExcludeSet>,
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
                        tantivy_dirty |= process_event(
                            &vault_root,
                            &index,
                            tantivy.as_deref(),
                            &embedding_runtime,
                            &event.path,
                            &exclude,
                        );
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

#[cfg(not(has_embeddings))]
pub fn start_watcher(
    vault_root: PathBuf,
    index: Arc<RwLock<VaultIndex>>,
    tantivy: Option<Arc<TantivyIndex>>,
    exclude: Arc<ExcludeSet>,
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
                        tantivy_dirty |= process_event(
                            &vault_root,
                            &index,
                            tantivy.as_deref(),
                            &event.path,
                            &exclude,
                        );
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

fn should_process_path(vault_root: &Path, absolute: &Path, exclude: &ExcludeSet) -> bool {
    let relative = match vault_path::relative_from_absolute(vault_root, absolute) {
        Ok(relative) => relative,
        Err(_) => return false,
    };

    if is_obsidian_dir(&relative) {
        return false;
    }

    if exclude.is_excluded(&relative) {
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
    relative.components().next().is_some_and(|c| {
        let name = c.as_os_str();
        name == ".obsidian" || name == ".obsidian-mcp"
    })
}

/// Returns whether Tantivy was touched.
#[cfg(has_embeddings)]
fn process_event(
    vault_root: &Path,
    index: &Arc<RwLock<VaultIndex>>,
    tantivy: Option<&TantivyIndex>,
    embedding_runtime: &EmbeddingRuntime,
    absolute: &Path,
    exclude: &ExcludeSet,
) -> bool {
    if !should_process_path(vault_root, absolute, exclude) {
        return false;
    }

    let relative = match vault_path::relative_from_absolute(vault_root, absolute) {
        Ok(relative) => relative,
        Err(_) => return false,
    };

    let mut tv_touched = false;

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
            } else {
                tv_touched = true;
            }
        }
        embedding_runtime.submit_upsert(&relative);
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
            } else {
                tv_touched = true;
            }
        }
        embedding_runtime.submit_remove(&relative);
    }

    tv_touched
}

/// Returns whether Tantivy was touched.
#[cfg(not(has_embeddings))]
fn process_event(
    vault_root: &Path,
    index: &Arc<RwLock<VaultIndex>>,
    tantivy: Option<&TantivyIndex>,
    absolute: &Path,
    exclude: &ExcludeSet,
) -> bool {
    if !should_process_path(vault_root, absolute, exclude) {
        return false;
    }

    let relative = match vault_path::relative_from_absolute(vault_root, absolute) {
        Ok(relative) => relative,
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

#[cfg(test)]
mod tests {
    use super::*;
    use unicode_normalization::UnicodeNormalization;

    #[test]
    fn should_process_unicode_markdown_path() {
        let dir = tempfile::tempdir().unwrap();
        let composed = "02_База-знаний/Сущности/lic1c.md";
        let decomposed: String = composed.nfd().collect();
        let absolute = dir.path().join(decomposed);
        let exclude = ExcludeSet::build(vec![]).unwrap();

        assert!(should_process_path(dir.path(), &absolute, &exclude));
    }

    #[cfg(not(has_embeddings))]
    #[test]
    fn process_event_indexes_actual_unicode_relative_path() {
        let dir = tempfile::tempdir().unwrap();
        let composed = "02_База-знаний/Сущности/lic1c.md";
        let decomposed: String = composed.nfd().collect();
        let disk_path = PathBuf::from(&decomposed);
        let absolute = dir.path().join(&disk_path);
        std::fs::create_dir_all(absolute.parent().unwrap()).unwrap();
        std::fs::write(&absolute, "# License\n").unwrap();

        let index = Arc::new(RwLock::new(VaultIndex::empty()));
        let exclude = ExcludeSet::build(vec![]).unwrap();

        let _ = process_event(dir.path(), &index, None, &absolute, &exclude);

        let index = index.read().unwrap();
        assert!(index.get_note(&disk_path).is_some());
        assert!(index.get_note(Path::new(composed)).is_none());
    }

    #[cfg(has_embeddings)]
    #[tokio::test]
    async fn embedding_events_coalesce_to_latest_path_intents_without_inference() {
        use crate::vault::embedding_runtime::PendingKind;
        use crate::vault::embeddings::Embedder;

        let dir = tempfile::tempdir().unwrap();
        let index = Arc::new(RwLock::new(VaultIndex::empty()));
        let runtime = EmbeddingRuntime::spawn(
            dir.path().to_path_buf(),
            Arc::clone(&index),
            dir.path().join("embeddings.bin"),
            async { std::future::pending::<VaultResult<Arc<dyn Embedder>>>().await },
        );
        let exclude = ExcludeSet::build(vec![]).unwrap();
        let old_relative = PathBuf::from("old.md");
        let old_absolute = dir.path().join(&old_relative);

        std::fs::write(&old_absolute, "# First\n").unwrap();
        let _ = process_event(dir.path(), &index, None, &runtime, &old_absolute, &exclude);
        assert_eq!(
            runtime.pending_kind(&old_relative),
            Some(PendingKind::Upsert)
        );

        std::fs::write(&old_absolute, "# Latest\n").unwrap();
        let _ = process_event(dir.path(), &index, None, &runtime, &old_absolute, &exclude);
        assert_eq!(
            runtime.pending_kind(&old_relative),
            Some(PendingKind::Upsert)
        );

        let new_relative = PathBuf::from("new.md");
        let new_absolute = dir.path().join(&new_relative);
        std::fs::rename(&old_absolute, &new_absolute).unwrap();
        let _ = process_event(dir.path(), &index, None, &runtime, &old_absolute, &exclude);
        let _ = process_event(dir.path(), &index, None, &runtime, &new_absolute, &exclude);
        assert_eq!(
            runtime.pending_kind(&old_relative),
            Some(PendingKind::Remove)
        );
        assert_eq!(
            runtime.pending_kind(&new_relative),
            Some(PendingKind::Upsert)
        );

        std::fs::remove_file(&new_absolute).unwrap();
        let _ = process_event(dir.path(), &index, None, &runtime, &new_absolute, &exclude);
        assert_eq!(
            runtime.pending_kind(&new_relative),
            Some(PendingKind::Remove)
        );
    }
}
