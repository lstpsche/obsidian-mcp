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

use super::exclude::ExcludeSet;
use super::index::VaultIndex;
use super::path as vault_path;
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
#[cfg(has_embeddings)]
pub fn start_watcher(
    vault_root: PathBuf,
    index: Arc<RwLock<VaultIndex>>,
    tantivy: Option<Arc<TantivyIndex>>,
    embedding_runtime: Option<super::embedding_runtime::EmbeddingRuntime>,
    exclude: Arc<ExcludeSet>,
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
                        tantivy_dirty |= process_event(
                            &vault_root,
                            &index,
                            tantivy.as_deref(),
                            embedding_runtime.as_ref(),
                            &event.path,
                            &exclude,
                        );
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

/// Start watching `vault_root` for filesystem changes.
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
/// - Paths inside `.obsidian/` or `.obsidian-mcp/`
/// - Non-`.md` files
fn should_process_path(vault_root: &Path, absolute: &Path) -> bool {
    let relative = match vault_path::relative_from_absolute(vault_root, absolute) {
        Ok(r) => r,
        Err(_) => {
            tracing::trace!(path = %absolute.display(), "event path outside vault root, ignoring");
            return false;
        }
    };

    if is_obsidian_dir(&relative) {
        return false;
    }

    match absolute.extension().and_then(|e| e.to_str()) {
        Some(ext) if ext.eq_ignore_ascii_case("md") => true,
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
            if path_str.to_ascii_lowercase().ends_with(".md") {
                true
            } else {
                tracing::trace!(path = %relative.display(), "no extension, ignoring");
                false
            }
        }
    }
}

/// Check if a vault-relative path is inside `.obsidian/` or `.obsidian-mcp/`.
fn is_obsidian_dir(relative: &Path) -> bool {
    relative.components().next().is_some_and(|c| {
        let name = c.as_os_str();
        name == ".obsidian" || name == ".obsidian-mcp"
    })
}

fn normalized_relative_path(vault_root: &Path, absolute: &Path) -> Option<PathBuf> {
    vault_path::relative_from_absolute(vault_root, absolute).ok()
}

fn is_excluded_path(exclude: &ExcludeSet, relative: &Path) -> bool {
    exclude.is_excluded(Path::new(&relative.to_string_lossy().replace('\\', "/")))
}

/// Process a single debounced event. Returns whether Tantivy was touched.
#[cfg(has_embeddings)]
fn process_event(
    vault_root: &Path,
    index: &Arc<RwLock<VaultIndex>>,
    tantivy: Option<&TantivyIndex>,
    embedding_runtime: Option<&super::embedding_runtime::EmbeddingRuntime>,
    absolute: &Path,
    exclude: &ExcludeSet,
) -> bool {
    if !should_process_path(vault_root, absolute) {
        return false;
    }

    let relative = match normalized_relative_path(vault_root, absolute) {
        Some(r) => r,
        None => return false,
    };

    let mut tv_touched = false;

    if is_excluded_path(exclude, &relative) {
        if absolute.exists() {
            tracing::debug!(path = %relative.display(), "tracking excluded note");
            match index.write() {
                Ok(mut idx) => idx.add_excluded_file(&relative),
                Err(e) => {
                    tracing::error!("index lock poisoned: {e}");
                    return false;
                }
            }
        } else {
            tracing::debug!(path = %relative.display(), "removing excluded note tracking");
            match index.write() {
                Ok(mut idx) => idx.remove_file(&relative),
                Err(e) => {
                    tracing::error!("index lock poisoned: {e}");
                    return false;
                }
            }
        }

        if let Some(tv) = tantivy {
            if let Err(e) = tv.remove_file_batch(&relative) {
                tracing::warn!(path = %relative.display(), error = %e, "tantivy remove failed");
            } else {
                tv_touched = true;
            }
        }
        if let Some(runtime) = embedding_runtime {
            runtime.submit_remove(&relative);
        }
        return tv_touched;
    }

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
            } else {
                tv_touched = true;
            }
        }
        if let Some(runtime) = embedding_runtime {
            runtime.submit_upsert(&relative);
        }
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
            } else {
                tv_touched = true;
            }
        }
        if let Some(runtime) = embedding_runtime {
            runtime.submit_remove(&relative);
        }
    }

    tv_touched
}

/// Process a single debounced event. Returns whether Tantivy was touched.
#[cfg(not(has_embeddings))]
fn process_event(
    vault_root: &Path,
    index: &Arc<RwLock<VaultIndex>>,
    tantivy: Option<&TantivyIndex>,
    absolute: &Path,
    exclude: &ExcludeSet,
) -> bool {
    if !should_process_path(vault_root, absolute) {
        return false;
    }

    let relative = match normalized_relative_path(vault_root, absolute) {
        Some(r) => r,
        None => return false,
    };

    if is_excluded_path(exclude, &relative) {
        if absolute.exists() {
            tracing::debug!(path = %relative.display(), "tracking excluded note");
            match index.write() {
                Ok(mut idx) => idx.add_excluded_file(&relative),
                Err(e) => {
                    tracing::error!("index lock poisoned: {e}");
                    return false;
                }
            }
        } else {
            tracing::debug!(path = %relative.display(), "removing excluded note tracking");
            match index.write() {
                Ok(mut idx) => idx.remove_file(&relative),
                Err(e) => {
                    tracing::error!("index lock poisoned: {e}");
                    return false;
                }
            }
        }

        if let Some(tv) = tantivy {
            if let Err(e) = tv.remove_file_batch(&relative) {
                tracing::warn!(path = %relative.display(), error = %e, "tantivy remove failed");
                return false;
            }
            return true;
        }
        return false;
    }

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
    use unicode_normalization::UnicodeNormalization;

    fn vault() -> PathBuf {
        PathBuf::from("/tmp/test-vault")
    }

    #[test]
    fn filters_obsidian_directory() {
        let root = vault();
        assert!(!should_process_path(
            &root,
            &root.join(".obsidian/plugins/foo.json"),
        ));
        assert!(!should_process_path(
            &root,
            &root.join(".obsidian/workspace.json"),
        ));
    }

    #[test]
    fn filters_obsidian_mcp_directory() {
        let root = vault();
        assert!(!should_process_path(
            &root,
            &root.join(".obsidian-mcp/config.json"),
        ));
        assert!(!should_process_path(
            &root,
            &root.join(".obsidian-mcp/ignore"),
        ));
    }

    #[test]
    fn filters_non_markdown_files() {
        let root = vault();
        assert!(!should_process_path(&root, &root.join("image.png")));
        assert!(!should_process_path(&root, &root.join("data.json")));
        assert!(!should_process_path(
            &root,
            &root.join("subfolder/script.js"),
        ));
    }

    #[test]
    fn accepts_markdown_files() {
        let root = vault();
        assert!(should_process_path(&root, &root.join("note.md")));
        assert!(should_process_path(
            &root,
            &root.join("subfolder/deep/note.md"),
        ));
    }

    #[test]
    fn accepts_uppercase_markdown_extension() {
        let root = vault();
        assert!(should_process_path(&root, &root.join("NOTE.MD")));
        assert!(should_process_path(&root, &root.join("Mixed.Md")));
        assert!(should_process_path(&root, &root.join("subfolder/CAPS.MD"),));
    }

    #[test]
    fn filters_paths_outside_vault() {
        let root = vault();
        assert!(!should_process_path(
            &root,
            Path::new("/other/place/note.md"),
        ));
    }

    #[test]
    fn obsidian_dir_detection() {
        assert!(is_obsidian_dir(Path::new(".obsidian/plugins/foo.json")));
        assert!(is_obsidian_dir(Path::new(".obsidian")));
        assert!(!is_obsidian_dir(Path::new("notes/.obsidian/foo")));
        assert!(!is_obsidian_dir(Path::new("daily/2024-01-01.md")));
    }

    #[test]
    fn obsidian_mcp_dir_detection() {
        assert!(is_obsidian_dir(Path::new(".obsidian-mcp/ignore")));
        assert!(is_obsidian_dir(Path::new(".obsidian-mcp")));
        assert!(is_obsidian_dir(Path::new(
            ".obsidian-mcp/embeddings/embeddings.bin"
        )));
        assert!(!is_obsidian_dir(Path::new("notes/.obsidian-mcp/foo")));
    }

    #[test]
    fn accepts_excluded_markdown_paths_for_tracking() {
        let root = vault();
        let exclude = ExcludeSet::build(vec!["Archive/".into()]).unwrap();
        assert!(should_process_path(&root, &root.join("Archive/note.md")));
        assert!(should_process_path(
            &root,
            &root.join("Archive/sub/deep.md")
        ));
        assert!(is_excluded_path(&exclude, Path::new("Archive/note.md")));
        assert!(is_excluded_path(&exclude, Path::new("Archive/sub/deep.md")));
    }

    #[test]
    fn accepts_non_excluded_paths() {
        let root = vault();
        let exclude = ExcludeSet::build(vec!["Archive/".into()]).unwrap();
        assert!(should_process_path(&root, &root.join("Active/note.md"),));
        assert!(should_process_path(
            &root,
            &root.join("Daily/2024-01-01.md"),
        ));
        assert!(!is_excluded_path(&exclude, Path::new("Active/note.md")));
        assert!(!is_excluded_path(
            &exclude,
            Path::new("Daily/2024-01-01.md")
        ));
    }

    #[test]
    fn normalized_relative_path_preserves_actual_unicode_event_spelling() {
        let dir = tempfile::tempdir().unwrap();
        let composed = "02_База-знаний/Сущности/lic1c.md";
        let decomposed: String = composed.nfd().collect();
        let absolute = dir.path().join(&decomposed);

        let relative = normalized_relative_path(dir.path(), &absolute).unwrap();

        assert_eq!(relative, PathBuf::from(decomposed));
    }

    fn call_start_watcher(
        vault_root: PathBuf,
        index: Arc<RwLock<VaultIndex>>,
    ) -> VaultResult<Debouncer<notify::RecommendedWatcher>> {
        let exclude = Arc::new(ExcludeSet::build(vec![]).unwrap());
        #[cfg(has_embeddings)]
        {
            start_watcher(vault_root, index, None, None, exclude)
        }
        #[cfg(not(has_embeddings))]
        {
            start_watcher(vault_root, index, None, exclude)
        }
    }

    fn call_process_event(
        vault_root: &Path,
        index: &Arc<RwLock<VaultIndex>>,
        absolute: &Path,
        exclude: &ExcludeSet,
    ) {
        #[cfg(has_embeddings)]
        {
            let _ = process_event(vault_root, index, None, None, absolute, exclude);
        }
        #[cfg(not(has_embeddings))]
        {
            let _ = process_event(vault_root, index, None, absolute, exclude);
        }
    }

    #[tokio::test]
    async fn excluded_create_event_updates_stats_without_indexing() {
        let dir = tempfile::tempdir().unwrap();
        let vault_root = dir.path();
        std::fs::create_dir_all(vault_root.join("Archive")).unwrap();
        let path = vault_root.join("Archive/hidden.md");
        std::fs::write(&path, "# Hidden\n").unwrap();

        let index = Arc::new(RwLock::new(VaultIndex::empty()));
        let exclude = ExcludeSet::build(vec!["Archive/".into()]).unwrap();

        call_process_event(vault_root, &index, &path, &exclude);

        let idx = index.read().unwrap();
        assert_eq!(idx.stats().excluded_notes, 1);
        assert!(idx.get_note(Path::new("Archive/hidden.md")).is_none());
    }

    #[tokio::test]
    async fn excluded_delete_event_clears_stats_tracking() {
        let dir = tempfile::tempdir().unwrap();
        let vault_root = dir.path();
        std::fs::create_dir_all(vault_root.join("Archive")).unwrap();
        let path = vault_root.join("Archive/hidden.md");
        std::fs::write(&path, "# Hidden\n").unwrap();

        let index = Arc::new(RwLock::new(VaultIndex::empty()));
        let exclude = ExcludeSet::build(vec!["Archive/".into()]).unwrap();

        call_process_event(vault_root, &index, &path, &exclude);
        std::fs::remove_file(&path).unwrap();
        call_process_event(vault_root, &index, &path, &exclude);

        let idx = index.read().unwrap();
        assert_eq!(idx.stats().excluded_notes, 0);
        assert!(idx.get_note(Path::new("Archive/hidden.md")).is_none());
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
