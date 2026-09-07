//! Filesystem watcher: debounced `notify` events that keep the vault index in sync.
//!
//! Uses `notify-debouncer-mini` for 500ms debouncing and bridges events into a
//! spawned tokio task that updates the [`VaultIndex`].
//!
//! `notify-debouncer-mini` 0.7 erases event kinds (create/modify/delete/rename all
//! become `DebouncedEventKind::Any`). We disambiguate by checking the filesystem at
//! event time: path exists → reindex, path gone → remove.

use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use notify::event::{AccessKind, AccessMode};
use notify::{EventHandler, EventKind, RecursiveMode, Watcher};
use notify_debouncer_mini::{
    DebounceEventHandler, DebounceEventResult, Debouncer, new_debouncer_opt,
};
use tokio::runtime::Handle;

use super::exclude::ExcludeSet;
use super::index::VaultIndex;
use super::path as vault_path;
use super::tantivy_index::TantivyIndex;
use crate::error::{VaultError, VaultResult};

/// Native watcher that discards read-only access events before debouncing.
///
/// Linux reports file opens even for indexing reads. The mini debouncer erases
/// event kinds, so filtering afterwards would create a read/reindex feedback loop.
pub struct ChangeWatcher(notify::RecommendedWatcher);

impl Watcher for ChangeWatcher {
    fn new<F: EventHandler>(mut event_handler: F, config: notify::Config) -> notify::Result<Self> {
        notify::RecommendedWatcher::new(
            move |result: notify::Result<notify::Event>| {
                if let Ok(event) = &result
                    && matches!(event.kind, EventKind::Access(kind)
                        if kind != AccessKind::Close(AccessMode::Write))
                    && !event.need_rescan()
                {
                    return;
                }
                event_handler.handle_event(result);
            },
            config,
        )
        .map(Self)
    }

    fn watch(&mut self, path: &Path, recursive_mode: RecursiveMode) -> notify::Result<()> {
        self.0.watch(path, recursive_mode)
    }

    fn unwatch(&mut self, path: &Path) -> notify::Result<()> {
        self.0.unwatch(path)
    }

    fn configure(&mut self, config: notify::Config) -> notify::Result<bool> {
        self.0.configure(config)
    }

    fn kind() -> notify::WatcherKind {
        notify::RecommendedWatcher::kind()
    }
}

pub(crate) fn new_change_debouncer(
    timeout: Duration,
    event_handler: impl DebounceEventHandler,
) -> notify::Result<Debouncer<ChangeWatcher>> {
    new_debouncer_opt(
        notify_debouncer_mini::Config::default().with_timeout(timeout),
        event_handler,
    )
}

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
) -> VaultResult<Debouncer<ChangeWatcher>> {
    let embedding_runtime = embedding_runtime
        .as_ref()
        .map(super::embedding_runtime::EmbeddingRuntime::downgrade);
    let (tx, mut rx) = tokio::sync::mpsc::channel::<DebounceEventResult>(EVENT_CHANNEL_CAPACITY);
    let rt = Handle::current();

    let mut debouncer =
        new_change_debouncer(DEBOUNCE_TIMEOUT, move |result: DebounceEventResult| {
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
) -> VaultResult<Debouncer<ChangeWatcher>> {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<DebounceEventResult>(EVENT_CHANNEL_CAPACITY);
    let rt = Handle::current();

    let mut debouncer =
        new_change_debouncer(DEBOUNCE_TIMEOUT, move |result: DebounceEventResult| {
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

/// Filter individual note events. Directory events are reconciled separately.
pub(crate) fn should_process_path(vault_root: &Path, absolute: &Path) -> bool {
    normalized_relative_path(vault_root, absolute).is_some_and(|relative| {
        super::exclude::is_visible_path(&relative)
            && relative
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("md"))
    })
}

fn normalized_relative_path(vault_root: &Path, absolute: &Path) -> Option<PathBuf> {
    vault_path::relative_from_absolute(vault_root, absolute).ok()
}

#[cfg(test)]
fn is_excluded_path(exclude: &ExcludeSet, relative: &Path) -> bool {
    exclude.is_excluded(relative)
}

/// Reconcile all known and current Markdown descendants of a directory event.
fn event_paths(
    vault_root: &Path,
    index: &VaultIndex,
    absolute: &Path,
) -> VaultResult<Vec<PathBuf>> {
    let relative = vault_path::relative_from_absolute(vault_root, absolute)?;
    if !super::exclude::is_visible_path(&relative) {
        return Ok(Vec::new());
    }
    if should_process_path(vault_root, absolute)
        && (absolute.is_file() || index.get_note(&relative).is_some())
    {
        return Ok(vec![relative]);
    }
    let mut paths = std::collections::BTreeSet::new();
    for path in index
        .tracked_paths()
        .filter(|path| path.starts_with(&relative))
    {
        paths.insert(path.clone());
    }
    if absolute.is_dir() {
        vault_path::resolve_existing(vault_root, &relative)?;
        for entry in walkdir::WalkDir::new(absolute)
            .into_iter()
            .filter_entry(|entry| {
                entry.path() == absolute
                    || super::exclude::is_visible_path(Path::new(entry.file_name()))
            })
        {
            let entry = entry.map_err(|error| {
                VaultError::Watcher(format!("cannot reconcile {}: {error}", relative.display()))
            })?;
            if entry.file_type().is_file() && should_process_path(vault_root, entry.path()) {
                paths.insert(vault_path::relative_from_absolute(
                    vault_root,
                    entry.path(),
                )?);
            }
        }
    } else if should_process_path(vault_root, absolute) {
        paths.insert(relative);
    }
    Ok(paths.into_iter().collect())
}

/// Process one debounced event. Returns whether a Tantivy batch needs flushing.
pub(crate) fn process_event(
    vault_root: &Path,
    index: &Arc<RwLock<VaultIndex>>,
    tantivy: Option<&TantivyIndex>,
    #[cfg(has_embeddings)] embedding_runtime: Option<
        &super::embedding_runtime::EmbeddingRuntimeWeak,
    >,
    absolute: &Path,
    exclude: &ExcludeSet,
) -> bool {
    let mut index = match index.write() {
        Ok(index) => index,
        Err(error) => {
            tracing::error!(%error, "index lock poisoned");
            return false;
        }
    };
    let paths = match event_paths(vault_root, &index, absolute) {
        Ok(paths) => paths,
        Err(error) => {
            tracing::warn!(path = %absolute.display(), %error, "watcher reconciliation failed");
            return false;
        }
    };
    for path in &paths {
        if let Err(error) = index.refresh_file(vault_root, path, exclude, tantivy) {
            tracing::warn!(path = %path.display(), %error, "watcher reindex failed");
        }
        #[cfg(has_embeddings)]
        if let Some(runtime) = embedding_runtime {
            if index.get_note(path).is_some() {
                runtime.submit_upsert(path);
            } else {
                runtime.submit_remove(path);
            }
        }
    }
    tantivy.is_some() && !paths.is_empty()
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
    fn filters_hidden_components_at_any_depth() {
        let root = vault();
        for path in [
            ".secret/note.md",
            "notes/.trash/deleted.md",
            "notes/.hidden.md",
        ] {
            assert!(!should_process_path(&root, &root.join(path)));
        }
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
    ) -> VaultResult<Debouncer<ChangeWatcher>> {
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

    #[test]
    fn watcher_reads_do_not_trigger_more_events() {
        use std::sync::mpsc;

        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        let path = root.join("test.md");
        let (tx, rx) = mpsc::channel();
        let mut debouncer = new_change_debouncer(
            Duration::from_millis(100),
            move |result: DebounceEventResult| {
                for event in result.unwrap() {
                    if event.path.file_name().is_some_and(|name| name == "test.md") {
                        let content = std::fs::read_to_string(&event.path).unwrap();
                        let parsed = super::super::frontmatter::parse_frontmatter(&content);
                        tx.send(parsed.is_ok()).unwrap();
                    }
                }
            },
        )
        .unwrap();
        debouncer
            .watcher()
            .watch(&root, RecursiveMode::Recursive)
            .unwrap();

        // A parse failure must not turn the indexing read into a retry loop.
        std::fs::write(&path, "---\ntags: [unfinished\n---\n").unwrap();
        assert!(!rx.recv_timeout(Duration::from_secs(5)).unwrap());
        assert_eq!(
            rx.recv_timeout(Duration::from_secs(1)),
            Err(mpsc::RecvTimeoutError::Timeout),
            "reading malformed frontmatter must not schedule another update",
        );

        // A subsequent edit must still be delivered and parse block sequences.
        std::fs::write(
            &path,
            "---\ntitle: \"Test Document\"\nauthor: \"Jane Doe\"\ntags:\n  - documentation\n  - yaml-test\n---\n\n# Hello\n\nSome content here.\n",
        )
        .unwrap();
        assert!(rx.recv_timeout(Duration::from_secs(5)).unwrap());
        assert_eq!(
            rx.recv_timeout(Duration::from_secs(1)),
            Err(mpsc::RecvTimeoutError::Timeout),
            "reading valid frontmatter must not schedule another update",
        );
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

    #[tokio::test]
    async fn directory_events_reconcile_descendants_and_excluded_tracking() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        std::fs::create_dir_all(root.join("Old/nested")).unwrap();
        std::fs::write(root.join("Old/nested/note.md"), "quokka").unwrap();
        let exclude = ExcludeSet::build(vec!["Archive/**".into()]).unwrap();
        let index = Arc::new(RwLock::new(VaultIndex::empty()));
        call_process_event(&root, &index, &root.join("Old"), &exclude);
        assert!(
            index
                .read()
                .unwrap()
                .get_note(Path::new("Old/nested/note.md"))
                .is_some()
        );
        std::fs::rename(root.join("Old"), root.join("New")).unwrap();
        call_process_event(&root, &index, &root.join("Old"), &exclude);
        call_process_event(&root, &index, &root.join("New"), &exclude);
        assert!(
            index
                .read()
                .unwrap()
                .get_note(Path::new("Old/nested/note.md"))
                .is_none()
        );
        assert!(
            index
                .read()
                .unwrap()
                .get_note(Path::new("New/nested/note.md"))
                .is_some()
        );
        std::fs::rename(root.join("New"), root.join("Archive")).unwrap();
        call_process_event(&root, &index, &root.join("New"), &exclude);
        call_process_event(&root, &index, &root.join("Archive"), &exclude);
        assert!(index.read().unwrap().notes().is_empty());
        assert_eq!(index.read().unwrap().excluded_notes(), 1);
        std::fs::remove_dir_all(root.join("Archive")).unwrap();
        call_process_event(&root, &index, &root.join("Archive"), &exclude);
        assert_eq!(index.read().unwrap().excluded_notes(), 0);
    }

    #[tokio::test]
    async fn invalid_external_update_invalidates_metadata_and_lexical_search() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        let exclude = ExcludeSet::build(vec![]).unwrap();
        let index = Arc::new(RwLock::new(VaultIndex::empty()));
        let tv = TantivyIndex::build(&root, index.read().unwrap().notes()).unwrap();
        let path = root.join("note.md");
        for content in [
            "---\ntags: [Work]\n---\nquokka",
            "---\ntags: [unclosed\n---\nchanged",
        ] {
            std::fs::write(&path, content).unwrap();
            assert!(process_event(
                &root,
                &index,
                Some(&tv),
                #[cfg(has_embeddings)]
                None,
                &path,
                &exclude
            ));
            tv.flush().unwrap();
        }
        assert!(index.read().unwrap().notes().is_empty());
        assert_eq!(index.read().unwrap().stats().total_tags, 0);
        assert_eq!(index.read().unwrap().stats().total_notes, 0);
        assert!(tv.search("quokka", 10).unwrap().is_empty());
    }
}
