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
use tokio::sync::mpsc;

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

fn protected_index(index: &RwLock<VaultIndex>) -> bool {
    index
        .read()
        .map(|index| index.content_is_restricted())
        .unwrap_or(true)
}

/// A native subscription that collects changes while the initial indexes are being built.
pub(crate) struct PendingWatcher {
    vault_root: PathBuf,
    rescan: bool,
    receiver: mpsc::Receiver<DebounceEventResult>,
    debouncer: Debouncer<ChangeWatcher>,
}

impl PendingWatcher {
    pub(crate) fn register(vault_root: &Path) -> VaultResult<Self> {
        let (sender, receiver) = mpsc::channel(EVENT_CHANNEL_CAPACITY);
        let mut debouncer = new_change_debouncer(DEBOUNCE_TIMEOUT, move |result| {
            // The debouncer invokes this callback on its own thread.
            let _ = sender.blocking_send(result);
        })
        .map_err(|e| VaultError::Watcher(e.to_string()))?;
        debouncer
            .watcher()
            .watch(vault_root, RecursiveMode::Recursive)
            .map_err(|e| {
                VaultError::Watcher(format!("failed to watch {}: {e}", vault_root.display()))
            })?;
        tracing::info!(path = %vault_root.display(), "filesystem watcher started");
        Ok(Self {
            vault_root: vault_root.to_owned(),
            rescan: false,
            receiver,
            debouncer,
        })
    }

    /// Include changes made while this view had no watch-enabled clients.
    pub(crate) fn request_rescan(&mut self) {
        self.rescan = true;
    }

    pub(crate) fn start(
        self,
        index: Arc<RwLock<VaultIndex>>,
        tantivy: Option<Arc<TantivyIndex>>,
        #[cfg(has_embeddings)] embedding_runtime: Option<
            super::embedding_runtime::EmbeddingRuntime,
        >,
        exclude: Arc<ExcludeSet>,
    ) -> Debouncer<ChangeWatcher> {
        let Self {
            vault_root,
            rescan,
            mut receiver,
            debouncer,
        } = self;
        let initial = rescan.then(|| {
            notify_debouncer_mini::DebouncedEvent::new(
                vault_root.clone(),
                notify_debouncer_mini::DebouncedEventKind::Any,
            )
        });
        let protected = protected_index(&index);
        #[cfg(has_embeddings)]
        let embedding_runtime = embedding_runtime
            .as_ref()
            .map(super::embedding_runtime::EmbeddingRuntime::downgrade);
        let reconcile = move |result: DebounceEventResult| match result {
            Ok(events) => {
                reconcile_events(
                    &vault_root,
                    &index,
                    tantivy.as_deref(),
                    #[cfg(has_embeddings)]
                    embedding_runtime.as_ref(),
                    events.iter().map(|event| event.path.as_path()),
                    &exclude,
                    TantivyIndex::publish,
                );
            }
            Err(error) => {
                if protected {
                    tracing::warn!("protected filesystem watch event failed");
                } else {
                    tracing::warn!(%error, "watch error");
                }
            }
        };
        if let Some(event) = initial {
            reconcile(Ok(vec![event]));
        }
        // Drain batches already delivered during the scan before publishing the vault.
        for _ in 0..receiver.len() {
            if let Ok(result) = receiver.try_recv() {
                reconcile(result);
            }
        }
        let reconcile = Arc::new(reconcile);
        tokio::spawn(async move {
            while let Some(result) = receiver.recv().await {
                let reconcile = Arc::clone(&reconcile);
                if let Err(error) = tokio::task::spawn_blocking(move || reconcile(result)).await {
                    if protected {
                        tracing::error!("protected watcher reconciliation task failed");
                    } else {
                        tracing::error!(%error, "watcher reconciliation task failed");
                    }
                }
            }
        });
        debouncer
    }
}

/// Start a watcher for indexes that already exist.
pub fn start_watcher(
    vault_root: PathBuf,
    index: Arc<RwLock<VaultIndex>>,
    tantivy: Option<Arc<TantivyIndex>>,
    #[cfg(has_embeddings)] embedding_runtime: Option<super::embedding_runtime::EmbeddingRuntime>,
    exclude: Arc<ExcludeSet>,
) -> VaultResult<Debouncer<ChangeWatcher>> {
    let filesystem = index
        .write()
        .map_err(|error| VaultError::Watcher(format!("index lock poisoned: {error}")))?
        .bind_watcher(&vault_root, &exclude)?;
    Ok(PendingWatcher::register(filesystem.path())?.start(
        index,
        tantivy,
        #[cfg(has_embeddings)]
        embedding_runtime,
        exclude,
    ))
}

#[cfg(test)]
pub(crate) fn should_process_path(vault_root: &Path, absolute: &Path) -> bool {
    visible_relative_path(vault_root, absolute)
        .ok()
        .flatten()
        .is_some()
}

fn visible_relative_path(vault_root: &Path, absolute: &Path) -> VaultResult<Option<PathBuf>> {
    let relative = vault_path::relative_from_absolute(vault_root, absolute)?;
    Ok(super::exclude::is_visible_path(&relative).then_some(relative))
}

#[cfg(test)]
fn normalized_relative_path(vault_root: &Path, absolute: &Path) -> Option<PathBuf> {
    vault_path::relative_from_absolute(vault_root, absolute).ok()
}

#[cfg(test)]
fn is_excluded_path(exclude: &ExcludeSet, relative: &Path) -> bool {
    exclude.is_excluded(relative)
}

enum Change {
    Refresh,
    Remove,
}

/// A backend can report only the new spelling of a renamed file or directory.
/// Confirm a possible alias on disk so distinct case-sensitive entries remain separate.
fn previous_spelling(
    filesystem: &super::fs::VaultRoot,
    index: &VaultIndex,
    actual: &Path,
) -> Option<(PathBuf, PathBuf)> {
    let mut current = PathBuf::new();
    for name in actual {
        current.push(name);
        for previous in index.watch_paths().matching(&current, false) {
            if previous != &current
                && previous.parent() == current.parent()
                && filesystem
                    .resolve_existing(previous)
                    .is_ok_and(|entry| entry.relative == current)
            {
                return Some((previous.clone(), current));
            }
        }
    }
    None
}

/// A directory inventory supplies the actual native names. Entries absent from
/// that inventory must be removed without reopening their old spelling, which
/// could now resolve to a renamed file on a case-insensitive filesystem.
fn event_paths(
    filesystem: &super::fs::VaultRoot,
    index: &VaultIndex,
    relative: &Path,
) -> VaultResult<std::collections::BTreeMap<PathBuf, Change>> {
    let metadata = match filesystem.entry_metadata_exact(relative) {
        Ok(meta) if !super::fs::is_link(&meta) && (meta.is_dir() || meta.is_file()) => Some(meta),
        Ok(_) => None,
        Err(error) if super::fs::skip_changed_entry(&error) => None,
        Err(error) => return Err(error),
    };
    let actual = match &metadata {
        Some(_) if relative.as_os_str().is_empty() => Some(PathBuf::new()),
        Some(_) => Some(filesystem.resolve_existing(relative)?.relative),
        None => None,
    };
    let mut paths = std::collections::BTreeMap::new();
    if metadata.as_ref().is_some_and(|meta| meta.is_file())
        && actual.as_deref() == Some(relative)
        && index.tracks(relative)
    {
        paths.insert(relative.to_owned(), Change::Refresh);
        return Ok(paths);
    }
    let (previous, actual, metadata) = match actual
        .as_deref()
        .filter(|path| !path.as_os_str().is_empty() && *path == relative)
        .and_then(|path| previous_spelling(filesystem, index, path))
    {
        // Reconcile both subtrees, even when only a descendant was reported.
        Some((previous, current)) => {
            let metadata = filesystem.entry_metadata_exact(&current)?;
            (Some(previous), Some(current), Some(metadata))
        }
        None => (None, actual, metadata),
    };
    let relative = previous.as_deref().unwrap_or(relative);
    for prefix in std::iter::once(relative).chain(actual.as_deref().filter(|p| *p != relative)) {
        for tracked in index.watch_paths().matching(prefix, true) {
            if tracked.starts_with(prefix) && index.tracks(tracked) {
                paths.insert(tracked.clone(), Change::Remove);
            }
        }
    }
    if let (Some(metadata), Some(actual)) = (metadata, actual) {
        if metadata.is_dir() {
            super::index::visit_files(
                filesystem,
                &actual,
                || Ok(()),
                |path| {
                    paths.insert(path.to_owned(), Change::Refresh);
                    Ok(())
                },
            )?;
        } else {
            paths.insert(actual, Change::Refresh);
        }
    }
    Ok(paths)
}

/// Commit a debounced batch once, outside the metadata lock. Embeddings for
/// eligible notes can run once the matching lexical revision is searchable.
fn reconcile_events<'a>(
    vault_root: &Path,
    index: &Arc<RwLock<VaultIndex>>,
    tantivy: Option<&TantivyIndex>,
    #[cfg(has_embeddings)] runtime: Option<&super::embedding_runtime::EmbeddingRuntimeWeak>,
    paths: impl IntoIterator<Item = &'a Path>,
    exclude: &ExcludeSet,
    flush: impl FnOnce(&TantivyIndex) -> VaultResult<tantivy::Opstamp>,
) -> bool {
    let mut dirty = false;
    for path in paths {
        dirty |= stage_event(
            vault_root,
            index,
            tantivy,
            path,
            exclude,
            #[cfg(has_embeddings)]
            runtime,
        );
    }
    if dirty && let Some(tv) = tantivy {
        match flush(tv) {
            Ok(_committed) =>
            {
                #[cfg(has_embeddings)]
                if let Some(runtime) = runtime {
                    index
                        .write()
                        .unwrap_or_else(|error| error.into_inner())
                        .publish_embedding_upserts(_committed, |path| runtime.submit_upsert(path));
                }
            }
            Err(error) => {
                if protected_index(index) {
                    tracing::warn!(
                        category = error.diagnostic_category(),
                        "protected lexical batch publication failed"
                    );
                } else {
                    tracing::warn!(%error, "tantivy batch flush failed");
                }
            }
        }
    }
    dirty
}

#[cfg(test)]
pub(crate) fn process_event(
    vault_root: &Path,
    index: &Arc<RwLock<VaultIndex>>,
    tantivy: Option<&TantivyIndex>,
    #[cfg(has_embeddings)] runtime: Option<&super::embedding_runtime::EmbeddingRuntimeWeak>,
    absolute: &Path,
    exclude: &ExcludeSet,
) -> bool {
    reconcile_events(
        vault_root,
        index,
        tantivy,
        #[cfg(has_embeddings)]
        runtime,
        [absolute],
        exclude,
        TantivyIndex::publish,
    )
}

fn stage_event(
    vault_root: &Path,
    index: &Arc<RwLock<VaultIndex>>,
    tantivy: Option<&TantivyIndex>,
    absolute: &Path,
    exclude: &ExcludeSet,
    #[cfg(has_embeddings)] runtime: Option<&super::embedding_runtime::EmbeddingRuntimeWeak>,
) -> bool {
    let protected = protected_index(index);
    let relative = match visible_relative_path(vault_root, absolute) {
        Ok(Some(relative)) => relative,
        Ok(None) => return false,
        Err(error) => {
            if protected {
                tracing::warn!(
                    category = error.diagnostic_category(),
                    "invalid protected watcher path"
                );
            } else {
                tracing::warn!(%error, "invalid watcher path");
            }
            return false;
        }
    };
    let mut index = match index.write() {
        Ok(index) => index,
        Err(error) => {
            tracing::error!(%error, "index lock poisoned");
            return false;
        }
    };
    let filesystem = match index.filesystem(vault_root) {
        Ok(filesystem) => filesystem,
        Err(error) => {
            if protected {
                tracing::warn!(
                    category = error.diagnostic_category(),
                    "protected watcher source unavailable"
                );
            } else {
                tracing::warn!(%error, "watcher vault unavailable");
            }
            return false;
        }
    };
    let paths = match event_paths(&filesystem, &index, &relative) {
        Ok(paths) => paths,
        Err(error) => {
            // Still invalidate known descendants if an ancestor disappeared or became a link.
            if protected {
                tracing::warn!(
                    category = error.diagnostic_category(),
                    "protected watcher reconciliation failed"
                );
            } else {
                tracing::warn!(path = %absolute.display(), %error, "watcher reconciliation failed");
            }
            index
                .tracked_paths()
                .filter(|path| path.starts_with(&relative))
                .map(|path| (path.to_owned(), Change::Refresh))
                .collect()
        }
    };
    let mut dirty = false;
    for (path, change) in &paths {
        let was_note = index.get_note(path).is_some();
        let result = match change {
            Change::Refresh => index.refresh_file(
                &filesystem,
                path,
                exclude,
                tantivy,
                #[cfg(has_embeddings)]
                runtime.is_some(),
            ),
            Change::Remove => index.remove_projection(path, tantivy),
        };
        match result {
            Ok(changed) => dirty |= changed,
            Err(error) => {
                dirty |= tantivy.is_some() && was_note;
                if protected {
                    tracing::warn!(
                        category = error.diagnostic_category(),
                        "protected watcher reindex failed"
                    );
                } else {
                    tracing::warn!(path = %path.display(), %error, "watcher reindex failed");
                }
            }
        }
        #[cfg(has_embeddings)]
        if let Some(runtime) = runtime
            && let Some(change) = index.embedding_change(path, was_note)
        {
            runtime.submit(path, change);
        }
    }
    dirty
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
    fn accepts_attachment_paths_for_stats() {
        let root = vault();
        assert!(should_process_path(&root, &root.join("image.png")));
        assert!(should_process_path(&root, &root.join("data.json")));
        assert!(should_process_path(
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
        use std::time::Instant;

        use notify_debouncer_mini::DebouncedEventKind;

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
                        tx.send((event.kind, parsed.is_ok())).unwrap();
                    }
                }
            },
        )
        .unwrap();
        debouncer
            .watcher()
            .watch(&root, RecursiveMode::Recursive)
            .unwrap();

        let expect_settled_edit = |expected_parse_result| {
            let deadline = Instant::now() + Duration::from_secs(5);
            loop {
                let remaining = deadline
                    .checked_duration_since(Instant::now())
                    .expect("the edit should settle before the deadline");
                let (kind, parsed) = rx.recv_timeout(remaining).unwrap();
                // An ongoing edit can emit AnyContinuous before its final Any.
                if kind == DebouncedEventKind::Any {
                    assert_eq!(parsed, expected_parse_result);
                    break;
                }
            }
        };

        // A parse failure must not turn the indexing read into a retry loop.
        std::fs::write(&path, "---\ntags: [unfinished\n---\n").unwrap();
        expect_settled_edit(false);
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
        expect_settled_edit(true);
        assert_eq!(
            rx.recv_timeout(Duration::from_secs(1)),
            Err(mpsc::RecvTimeoutError::Timeout),
            "reading valid frontmatter must not schedule another update",
        );
    }

    #[tokio::test]
    async fn a_watched_empty_index_keeps_its_exclusions_for_manual_reindex() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().canonicalize().unwrap();
        let index = Arc::new(RwLock::new(VaultIndex::empty()));
        let policy = Arc::new(ExcludeSet::build(vec!["Private/**".into()]).unwrap());
        let attach = |policy| {
            start_watcher(
                root.clone(),
                Arc::clone(&index),
                None,
                #[cfg(has_embeddings)]
                None,
                policy,
            )
        };
        let _watcher = attach(policy).unwrap();
        assert!(attach(Arc::new(ExcludeSet::default())).is_err());
        std::fs::create_dir(root.join("Private")).unwrap();
        let path = Path::new("Private/note.md");
        std::fs::write(root.join(path), "secret").unwrap();
        let mut index = index.write().unwrap();
        index.reindex_file(&root, path).unwrap();
        assert!(index.get_note(path).is_none());
        assert_eq!(index.excluded_notes(), 1);
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

    async fn wait_for_stats(index: &RwLock<VaultIndex>, notes: usize, files: usize) {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        loop {
            let current = {
                let index = index.read().unwrap();
                (index.stats().total_notes, index.stats().total_files)
            };
            if current == (notes, files) {
                return;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "watcher did not settle: {current:?}"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    #[tokio::test]
    async fn watcher_tracks_visible_notes_and_attachments() {
        let dir = tempfile::tempdir().unwrap();
        let vault_root = dir.path().to_path_buf();
        let index = Arc::new(RwLock::new(VaultIndex::empty()));

        let vault_root = vault_root.canonicalize().unwrap();
        let _debouncer = call_start_watcher(vault_root.clone(), Arc::clone(&index)).unwrap();

        // A visible attachment counts toward stats, hidden settings do not.
        std::fs::write(vault_root.join("image.png"), b"fake png").unwrap();
        std::fs::create_dir_all(vault_root.join(".obsidian")).unwrap();
        std::fs::write(vault_root.join(".obsidian/workspace.json"), b"{}").unwrap();

        // Create a markdown file the watcher should process.
        std::fs::write(vault_root.join("note.md"), "# Hello\n").unwrap();

        // Modify it.
        std::fs::write(vault_root.join("note.md"), "# Hello\nUpdated.\n").unwrap();

        wait_for_stats(&index, 1, 2).await;
        std::fs::remove_file(vault_root.join("note.md")).unwrap();
        wait_for_stats(&index, 0, 1).await;
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
    async fn changes_between_cold_build_and_watcher_activation_are_replayed() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().canonicalize().unwrap();
        std::fs::write(root.join("note.md"), "earlier #before").unwrap();
        let exclude = Arc::new(ExcludeSet::default());
        let pending = PendingWatcher::register(&root).unwrap();
        let index = VaultIndex::build(&root, exclude.clone()).await.unwrap();
        let lexical = Arc::new(TantivyIndex::build(&root, index.notes()).unwrap());
        let index = Arc::new(RwLock::new(index));
        std::fs::write(root.join("note.md"), "later quokka #after").unwrap();
        std::fs::write(root.join("asset.bin"), [0x80, 0xff]).unwrap();
        assert!(index.read().unwrap().notes_with_tag("after").is_empty());
        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        while pending.receiver.is_empty() {
            assert!(
                tokio::time::Instant::now() < deadline,
                "native watcher never delivered its batch"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        let _active = pending.start(
            index.clone(),
            Some(lexical.clone()),
            #[cfg(has_embeddings)]
            None,
            exclude,
        );
        loop {
            let converged = {
                let index = index.read().unwrap();
                index.stats().total_files == 2 && index.notes_with_tag("after").len() == 1
            } && lexical.search("quokka", 1).unwrap().len() == 1;
            if converged {
                break;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "buffered changes were lost"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    #[tokio::test]
    async fn external_attachment_and_folder_changes_match_reopen() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().canonicalize().unwrap();
        let exclude = Arc::new(ExcludeSet::build(vec!["Archive/**".into()]).unwrap());
        let pending = PendingWatcher::register(&root).unwrap();
        let (index, lexical) = VaultIndex::build_ranked(
            super::super::fs::VaultRoot::open(&root).unwrap(),
            exclude.clone(),
        )
        .await
        .unwrap();
        let index = Arc::new(RwLock::new(index));
        let lexical = Arc::new(lexical);
        let _watcher = pending.start(
            index.clone(),
            Some(lexical.clone()),
            #[cfg(has_embeddings)]
            None,
            exclude.clone(),
        );
        std::fs::create_dir(root.join("Folder")).unwrap();
        std::fs::write(root.join("Folder/note.MD"), "needle #tag").unwrap();
        std::fs::write(root.join("Folder/asset.bin"), [0xff, 0x80]).unwrap();
        for (from, to) in [
            ("", "Folder"),
            ("Folder", "Archive"),
            ("Archive", "Moved"),
            ("Moved", ""),
        ] {
            if !from.is_empty() {
                if to.is_empty() {
                    std::fs::remove_dir_all(root.join(from)).unwrap();
                } else {
                    std::fs::rename(root.join(from), root.join(to)).unwrap();
                }
            }
            let cold = VaultIndex::build(&root, exclude.clone()).await.unwrap();
            let expected_stats = serde_json::to_value(cold.stats()).unwrap();
            let expected_paths = cold.notes().keys().cloned().collect::<Vec<_>>();
            let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
            loop {
                let observed = {
                    let live = index.read().unwrap();
                    (
                        serde_json::to_value(live.stats()).unwrap(),
                        live.notes().keys().cloned().collect::<Vec<_>>(),
                    )
                };
                let hits = lexical
                    .search("needle", 5)
                    .unwrap()
                    .into_iter()
                    .map(|(path, _)| path)
                    .collect::<Vec<_>>();
                if observed.0 == expected_stats
                    && observed.1 == expected_paths
                    && hits == expected_paths
                {
                    break;
                }
                assert!(
                    tokio::time::Instant::now() < deadline,
                    "watcher stayed at {observed:?}, lexical paths {hits:?}, expected {expected_stats} ({expected_paths:?})"
                );
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        }
    }

    #[tokio::test]
    async fn stale_observed_spelling_cannot_read_a_newly_excluded_folder() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().canonicalize().unwrap();
        let exclude = Arc::new(ExcludeSet::build(vec!["Private/**".into()]).unwrap());
        let old = root.join("private");
        let renamed = root.join("Private");
        std::fs::create_dir(&old).unwrap();
        std::fs::write(old.join("note.md"), "old public #before").unwrap();
        let (index, lexical) = VaultIndex::build_ranked(
            super::super::fs::VaultRoot::open(&root).unwrap(),
            exclude.clone(),
        )
        .await
        .unwrap();
        let index = Arc::new(RwLock::new(index));
        std::fs::rename(&old, &renamed).unwrap();
        std::fs::write(renamed.join("note.md"), "new private canary #after").unwrap();
        assert!(
            index
                .read()
                .unwrap()
                .search_text(&root, "canary", 20)
                .unwrap()
                .is_empty()
        );

        // On a case-insensitive filesystem the old name still resolves natively.
        // Deliver only a change to the newly spelled child, as a backend may do.
        let first_event = if old.exists() {
            renamed.join("note.md")
        } else {
            root.clone()
        };
        let changed = process_event(
            &root,
            &index,
            Some(&lexical),
            #[cfg(has_embeddings)]
            None,
            &first_event,
            &exclude,
        );
        assert!(changed);
        assert!(index.read().unwrap().notes().is_empty());
        assert_eq!(index.read().unwrap().excluded_notes(), 1);
        assert!(lexical.search("public", 2).unwrap().is_empty());
        assert!(lexical.search("canary", 2).unwrap().is_empty());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_visible_symlink_never_admits_an_excluded_note() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().canonicalize().unwrap();
        let exclude = Arc::new(ExcludeSet::build(vec!["Private/**".into()]).unwrap());
        std::fs::create_dir(root.join("Private")).unwrap();
        std::fs::write(root.join("visible.md"), "public").unwrap();
        std::fs::write(root.join("Private/secret.md"), "secret #exclusion").unwrap();
        let index = Arc::new(RwLock::new(
            VaultIndex::build(&root, exclude.clone()).await.unwrap(),
        ));
        std::fs::remove_file(root.join("visible.md")).unwrap();
        std::os::unix::fs::symlink(root.join("Private/secret.md"), root.join("visible.md"))
            .unwrap();
        // Even before notification arrives, reading an old index key must not follow the new link.
        assert!(
            index
                .read()
                .unwrap()
                .search_text(&root, "secret", 20)
                .unwrap()
                .is_empty()
        );
        call_process_event(&root, &index, &root.join("visible.md"), &exclude);
        assert!(index.read().unwrap().notes().is_empty());
        assert_eq!(index.read().unwrap().excluded_notes(), 1);
    }

    #[cfg(has_embeddings)]
    #[tokio::test]
    async fn lexical_publication_precedes_only_its_current_embedding_upserts() {
        use crate::vault::embedding_runtime::{EmbeddingRuntime, PendingKind};
        use crate::vault::embeddings::Embedder;

        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        let index = Arc::new(RwLock::new(VaultIndex::empty()));
        let tv = TantivyIndex::build(&root, index.read().unwrap().notes()).unwrap();
        let exclude = ExcludeSet::default();
        let runtime = EmbeddingRuntime::spawn(
            root.clone(),
            index.clone(),
            root.join("cache.bin"),
            std::future::pending::<VaultResult<Arc<dyn Embedder>>>(),
        );
        let weak = runtime.downgrade();
        let file = Path::new("note.md");
        let absolute = root.join(file);

        std::fs::write(&absolute, "first").unwrap();
        reconcile_events(
            &root,
            &index,
            Some(&tv),
            Some(&weak),
            [absolute.as_path()],
            &exclude,
            |lexical| {
                assert_eq!(
                    runtime.pending_kind(file),
                    Some(PendingKind::AwaitingLexical)
                );
                assert!(lexical.search("first", 1).unwrap().is_empty());
                let committed = lexical.publish()?;
                // The replacement is newer than the reader being reported to embeddings.
                std::fs::write(&absolute, "second")?;
                assert!(stage_event(
                    &root,
                    &index,
                    Some(&tv),
                    &absolute,
                    &exclude,
                    Some(&weak)
                ));
                Ok(committed)
            },
        );
        assert_eq!(
            runtime.pending_kind(file),
            Some(PendingKind::AwaitingLexical)
        );
        assert!(tv.search("second", 1).unwrap().is_empty());

        // A failed flush withholds both current upserts. A later successful
        // commit for a different path releases them without another source read.
        let other = root.join("other.md");
        std::fs::write(&other, "other").unwrap();
        reconcile_events(
            &root,
            &index,
            Some(&tv),
            Some(&weak),
            [other.as_path()],
            &exclude,
            |_| Err(VaultError::Other("injected publication failure".into())),
        );
        assert_eq!(
            runtime.pending_kind(file),
            Some(PendingKind::AwaitingLexical)
        );
        assert_eq!(
            runtime.pending_kind(Path::new("other.md")),
            Some(PendingKind::AwaitingLexical)
        );
        let third = root.join("third.md");
        std::fs::write(&third, "third").unwrap();
        process_event(&root, &index, Some(&tv), Some(&weak), &third, &exclude);
        assert_eq!(tv.search("second", 1).unwrap()[0].0, file);
        assert_eq!(runtime.pending_kind(file), Some(PendingKind::Upsert));
        assert_eq!(
            runtime.pending_kind(Path::new("other.md")),
            Some(PendingKind::Upsert)
        );

        // A delayed publication cannot undo a newer deletion.
        assert!(stage_event(
            &root,
            &index,
            Some(&tv),
            &absolute,
            &exclude,
            Some(&weak)
        ));
        std::fs::remove_file(&absolute).unwrap();
        process_event(&root, &index, Some(&tv), Some(&weak), &absolute, &exclude);
        assert_eq!(runtime.pending_kind(file), Some(PendingKind::Remove));
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
        }
        assert!(index.read().unwrap().notes().is_empty());
        assert_eq!(index.read().unwrap().stats().total_tags, 0);
        assert_eq!(index.read().unwrap().stats().total_notes, 0);
        assert!(tv.search("quokka", 10).unwrap().is_empty());
    }
}
