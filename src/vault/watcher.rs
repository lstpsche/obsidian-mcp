//! Filesystem watcher: debounced `notify` events that keep the vault index in sync.
//!
//! Uses `notify-debouncer-mini` for 500ms debouncing and bridges events into a
//! spawned tokio task that updates the [`VaultIndex`].
//!
//! `notify-debouncer-mini` 0.5 erases event kinds (create/modify/delete/rename all
//! become `DebouncedEventKind::Any`). We disambiguate by checking the filesystem at
//! event time: path exists → reindex, path gone → remove.

use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use notify::RecursiveMode;
use notify_debouncer_mini::{DebounceEventResult, Debouncer, new_debouncer};
use tokio::runtime::Handle;

use super::index::VaultIndex;
use super::tantivy_index::TantivyIndex;
use crate::error::{VaultError, VaultResult};

const DEBOUNCE_TIMEOUT: Duration = Duration::from_millis(500);
const EVENT_CHANNEL_CAPACITY: usize = 256;

/// Start watching `vault_root` for filesystem changes.
///
/// Returns the [`Debouncer`] handle — the caller **must** keep it alive
/// (e.g. store it in the `Vault` struct) or watching stops.
///
/// Internally spawns a tokio task that receives debounced events, filters
/// irrelevant paths, and calls the appropriate `VaultIndex` mutation.
#[cfg(feature = "embeddings")]
pub fn start_watcher(
    vault_root: PathBuf,
    index: Arc<RwLock<VaultIndex>>,
    tantivy: Option<Arc<TantivyIndex>>,
    embedding_model: Option<Arc<super::embeddings::EmbeddingModel>>,
    embedding_store: Option<Arc<RwLock<super::embeddings::EmbeddingStore>>>,
) -> VaultResult<Debouncer<notify::RecommendedWatcher>> {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<DebounceEventResult>(EVENT_CHANNEL_CAPACITY);
    let rt = Handle::current();

    let mut debouncer = new_debouncer(DEBOUNCE_TIMEOUT, move |result: DebounceEventResult| {
        let tx = tx.clone();
        rt.spawn(async move {
            if let Err(e) = tx.send(result).await {
                tracing::error!("watcher channel closed: {e}");
            }
        });
    })
    .map_err(|e| VaultError::Watcher(e.to_string()))?;

    debouncer
        .watcher()
        .watch(&vault_root, RecursiveMode::Recursive)
        .map_err(|e| {
            VaultError::Watcher(format!("failed to watch {}: {e}", vault_root.display()))
        })?;

    tracing::info!(path = %vault_root.display(), "filesystem watcher started");

    tokio::spawn(async move {
        while let Some(result) = rx.recv().await {
            match result {
                Ok(events) => {
                    let mut tantivy_dirty = false;
                    let mut embedding_dirty = false;
                    for event in events {
                        let (tv_touched, emb_touched) = process_event(
                            &vault_root,
                            &index,
                            tantivy.as_deref(),
                            embedding_model.as_deref(),
                            embedding_store.as_ref(),
                            &event.path,
                        );
                        tantivy_dirty |= tv_touched;
                        embedding_dirty |= emb_touched;
                    }
                    if tantivy_dirty
                        && let Some(ref tv) = tantivy
                        && let Err(e) = tv.flush()
                    {
                        tracing::warn!(error = %e, "tantivy batch flush failed");
                    }
                    if embedding_dirty
                        && let Some(ref store) = embedding_store
                        && let Ok(s) = store.read()
                    {
                        save_embedding_cache(&vault_root, &s);
                    }
                }
                Err(e) => {
                    tracing::warn!("watch error: {e}");
                }
            }
        }
        tracing::debug!("watcher event loop exited");
    });

    Ok(debouncer)
}

/// Start watching `vault_root` for filesystem changes.
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
            if let Err(e) = tx.send(result).await {
                tracing::error!("watcher channel closed: {e}");
            }
        });
    })
    .map_err(|e| VaultError::Watcher(e.to_string()))?;

    debouncer
        .watcher()
        .watch(&vault_root, RecursiveMode::Recursive)
        .map_err(|e| {
            VaultError::Watcher(format!("failed to watch {}: {e}", vault_root.display()))
        })?;

    tracing::info!(path = %vault_root.display(), "filesystem watcher started");

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
                        && let Err(e) = tv.flush()
                    {
                        tracing::warn!(error = %e, "tantivy batch flush failed");
                    }
                }
                Err(e) => {
                    tracing::warn!("watch error: {e}");
                }
            }
        }
        tracing::debug!("watcher event loop exited");
    });

    Ok(debouncer)
}

/// Decide whether a filesystem event should trigger an index update.
///
/// Returns `false` for:
/// - Paths inside `.obsidian/`
/// - Non-`.md` files
fn should_process_path(vault_root: &Path, absolute: &Path) -> bool {
    let relative = match absolute.strip_prefix(vault_root) {
        Ok(r) => r,
        Err(_) => {
            tracing::trace!(path = %absolute.display(), "event path outside vault root, ignoring");
            return false;
        }
    };

    if is_obsidian_dir(relative) {
        return false;
    }

    match absolute.extension().and_then(|e| e.to_str()) {
        Some("md") => true,
        Some(ext) => {
            tracing::trace!(path = %relative.display(), ext, "non-markdown file, ignoring");
            false
        }
        None => {
            // Deleted files may have lost their extension info if the path no longer
            // exists. We still accept extensionless paths and let the index handle
            // the no-op gracefully — `remove_file` on an unknown path is harmless.
            //
            // However, directories also lack extensions and we don't want to index
            // those, so we check if the path *looks* like it had an `.md` extension
            // by inspecting the string directly.
            let path_str = absolute.to_string_lossy();
            if path_str.ends_with(".md") {
                true
            } else {
                tracing::trace!(path = %relative.display(), "no extension, ignoring");
                false
            }
        }
    }
}

/// Check if a vault-relative path is inside the `.obsidian/` config directory.
fn is_obsidian_dir(relative: &Path) -> bool {
    relative
        .components()
        .next()
        .is_some_and(|c| c.as_os_str() == ".obsidian")
}

/// Process a single debounced event. Returns `(tantivy_touched, embedding_touched)`.
#[cfg(feature = "embeddings")]
fn process_event(
    vault_root: &Path,
    index: &Arc<RwLock<VaultIndex>>,
    tantivy: Option<&TantivyIndex>,
    embedding_model: Option<&super::embeddings::EmbeddingModel>,
    embedding_store: Option<&Arc<RwLock<super::embeddings::EmbeddingStore>>>,
    absolute: &Path,
) -> (bool, bool) {
    if !should_process_path(vault_root, absolute) {
        return (false, false);
    }

    let relative = match absolute.strip_prefix(vault_root) {
        Ok(r) => r.to_path_buf(),
        Err(_) => return (false, false),
    };

    let mut tv_touched = false;
    let mut emb_touched = false;

    if absolute.exists() {
        tracing::debug!(path = %relative.display(), "reindexing (create/modify)");
        let meta = match index.write() {
            Ok(mut idx) => {
                if let Err(e) = idx.reindex_file(vault_root, &relative) {
                    tracing::warn!(path = %relative.display(), error = %e, "reindex failed");
                    return (false, false);
                }
                idx.get_note(&relative).cloned()
            }
            Err(e) => {
                tracing::error!("index lock poisoned: {e}");
                return (false, false);
            }
        };
        if let Some(tv) = tantivy
            && let Some(ref m) = meta
        {
            if let Err(e) = tv.reindex_file_batch(vault_root, &relative, m) {
                tracing::warn!(path = %relative.display(), error = %e, "tantivy reindex failed");
            } else {
                tv_touched = true;
            }
        }
        if let (Some(model), Some(store), Some(m)) =
            (embedding_model, embedding_store, meta.as_ref())
        {
            emb_touched = embed_and_insert(vault_root, &relative, m, model, store);
        }
    } else {
        tracing::debug!(path = %relative.display(), "removing (delete)");
        match index.write() {
            Ok(mut idx) => idx.remove_file(&relative),
            Err(e) => {
                tracing::error!("index lock poisoned: {e}");
                return (false, false);
            }
        }
        if let Some(tv) = tantivy {
            if let Err(e) = tv.remove_file_batch(&relative) {
                tracing::warn!(path = %relative.display(), error = %e, "tantivy remove failed");
            } else {
                tv_touched = true;
            }
        }
        if let Some(store) = embedding_store
            && let Ok(mut s) = store.write()
        {
            s.remove(&relative);
            emb_touched = true;
        }
    }

    (tv_touched, emb_touched)
}

#[cfg(feature = "embeddings")]
fn embed_and_insert(
    vault_root: &Path,
    relative: &Path,
    meta: &crate::models::NoteMetadata,
    model: &super::embeddings::EmbeddingModel,
    store: &Arc<RwLock<super::embeddings::EmbeddingStore>>,
) -> bool {
    let Ok(content) = super::fs::read_file(vault_root, relative) else {
        return false;
    };
    let body = super::frontmatter::get_body(&content);
    let heading_texts: Vec<String> = meta.headings.iter().map(|h| h.text.clone()).collect();
    let text = super::embeddings::prepare_embed_text(&meta.title, &heading_texts, body);
    match model.embed_one(&text) {
        Ok(vec) => {
            if let Ok(mut s) = store.write() {
                s.insert(relative.to_path_buf(), vec);
                true
            } else {
                false
            }
        }
        Err(e) => {
            tracing::warn!(path = %relative.display(), error = %e, "embedding failed in watcher");
            false
        }
    }
}

#[cfg(feature = "embeddings")]
fn save_embedding_cache(vault_root: &Path, store: &super::embeddings::EmbeddingStore) {
    let cache_path = vault_root
        .join(".obsidian")
        .join("obsidian-mcp")
        .join("embeddings.bin");
    if let Err(e) = store.save(&cache_path) {
        tracing::warn!(error = %e, "failed to save embedding cache from watcher");
    }
}

/// Process a single debounced event. Returns whether Tantivy was touched.
#[cfg(not(feature = "embeddings"))]
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
        Ok(r) => r.to_path_buf(),
        Err(_) => return false,
    };

    if absolute.exists() {
        tracing::debug!(path = %relative.display(), "reindexing (create/modify)");
        let meta = match index.write() {
            Ok(mut idx) => {
                if let Err(e) = idx.reindex_file(vault_root, &relative) {
                    tracing::warn!(path = %relative.display(), error = %e, "reindex failed");
                    return false;
                }
                idx.get_note(&relative).cloned()
            }
            Err(e) => {
                tracing::error!("index lock poisoned: {e}");
                return false;
            }
        };
        if let Some(tv) = tantivy
            && let Some(ref m) = meta
        {
            if let Err(e) = tv.reindex_file_batch(vault_root, &relative, m) {
                tracing::warn!(path = %relative.display(), error = %e, "tantivy reindex failed");
                return false;
            }
            return true;
        }
        false
    } else {
        tracing::debug!(path = %relative.display(), "removing (delete)");
        match index.write() {
            Ok(mut idx) => idx.remove_file(&relative),
            Err(e) => {
                tracing::error!("index lock poisoned: {e}");
                return false;
            }
        }
        if let Some(tv) = tantivy {
            if let Err(e) = tv.remove_file_batch(&relative) {
                tracing::warn!(path = %relative.display(), error = %e, "tantivy remove failed");
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
    use std::path::PathBuf;

    fn vault() -> PathBuf {
        PathBuf::from("/tmp/test-vault")
    }

    #[test]
    fn filters_obsidian_directory() {
        let root = vault();
        assert!(!should_process_path(
            &root,
            &root.join(".obsidian/plugins/foo.json")
        ));
        assert!(!should_process_path(
            &root,
            &root.join(".obsidian/workspace.json")
        ));
    }

    #[test]
    fn filters_non_markdown_files() {
        let root = vault();
        assert!(!should_process_path(&root, &root.join("image.png")));
        assert!(!should_process_path(&root, &root.join("data.json")));
        assert!(!should_process_path(
            &root,
            &root.join("subfolder/script.js")
        ));
    }

    #[test]
    fn accepts_markdown_files() {
        let root = vault();
        // should_process_path checks extension; the file needn't exist for that check.
        assert!(should_process_path(&root, &root.join("note.md")));
        assert!(should_process_path(
            &root,
            &root.join("subfolder/deep/note.md")
        ));
    }

    #[test]
    fn filters_paths_outside_vault() {
        let root = vault();
        assert!(!should_process_path(
            &root,
            Path::new("/other/place/note.md")
        ));
    }

    #[test]
    fn obsidian_dir_detection() {
        assert!(is_obsidian_dir(Path::new(".obsidian/plugins/foo.json")));
        assert!(is_obsidian_dir(Path::new(".obsidian")));
        assert!(!is_obsidian_dir(Path::new("notes/.obsidian/foo")));
        assert!(!is_obsidian_dir(Path::new("daily/2024-01-01.md")));
    }

    fn call_start_watcher(
        vault_root: PathBuf,
        index: Arc<RwLock<VaultIndex>>,
    ) -> VaultResult<Debouncer<notify::RecommendedWatcher>> {
        #[cfg(feature = "embeddings")]
        {
            start_watcher(vault_root, index, None, None, None)
        }
        #[cfg(not(feature = "embeddings"))]
        {
            start_watcher(vault_root, index, None)
        }
    }

    #[tokio::test]
    async fn watcher_starts_and_stops() {
        let dir = tempfile::tempdir().unwrap();
        let vault_root = dir.path().to_path_buf();
        let index = Arc::new(RwLock::new(VaultIndex::empty()));

        let debouncer = call_start_watcher(vault_root, index);
        assert!(debouncer.is_ok(), "watcher should start without error");

        drop(debouncer.unwrap());
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    #[tokio::test]
    async fn watcher_survives_mixed_file_events() {
        let dir = tempfile::tempdir().unwrap();
        let vault_root = dir.path().to_path_buf();
        let index = Arc::new(RwLock::new(VaultIndex::empty()));

        let _debouncer = call_start_watcher(vault_root.clone(), index).unwrap();

        // Create files the watcher should ignore.
        std::fs::write(vault_root.join("image.png"), b"fake png").unwrap();
        std::fs::create_dir_all(vault_root.join(".obsidian")).unwrap();
        std::fs::write(vault_root.join(".obsidian/workspace.json"), b"{}").unwrap();

        // Create a markdown file the watcher should process.
        std::fs::write(vault_root.join("note.md"), "# Hello\n").unwrap();

        // Modify it.
        std::fs::write(vault_root.join("note.md"), "# Hello\nUpdated.\n").unwrap();

        // Wait for debounce timeout + processing headroom.
        tokio::time::sleep(Duration::from_millis(1500)).await;

        // Delete it.
        std::fs::remove_file(vault_root.join("note.md")).unwrap();

        tokio::time::sleep(Duration::from_millis(1000)).await;

        // The watcher should not have panicked. VaultIndex stubs are no-ops,
        // so we can't assert index state here — Task 3A integration tests will.
    }
}
