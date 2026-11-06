//! In-memory vault index: `NoteMetadata` for every `.md` file, fast lookups by stem/path/tag.

use std::collections::{HashMap, HashSet};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use regex::Regex;

use crate::error::{MutationOutcome, StorageErrorKind, VaultError, VaultResult};
use crate::models::{FileStat, NoteMetadata, SearchMatch, SearchResult, VaultStats, WikiLink};
use crate::vault::wikilink::{LinkResolver, build_link_resolver};
use crate::vault::{frontmatter, fs, parser};

use super::exclude::ExcludeSet;
use super::path::{logical_file_name, logical_file_stem};

/// Native spellings for each case-folded path prefix in the tracked inventory.
/// A directory stays present until its last tracked descendant is removed.
#[derive(Default)]
pub(super) struct WatchPaths {
    spellings: std::collections::BTreeMap<(PathBuf, PathBuf), usize>,
}

impl WatchPaths {
    fn folded(path: &Path) -> PathBuf {
        super::path::canonical_unicode_key(&path.to_string_lossy())
            .to_uppercase()
            .into()
    }

    fn prefixes(path: &Path) -> impl Iterator<Item = (PathBuf, PathBuf)> {
        path.ancestors()
            .take_while(|p| !p.as_os_str().is_empty())
            .map(|p| (Self::folded(p), p.to_owned()))
    }

    fn insert(&mut self, path: &Path) {
        for key in Self::prefixes(path) {
            *self.spellings.entry(key).or_default() += 1;
        }
    }

    fn remove(&mut self, path: &Path) {
        for key in Self::prefixes(path) {
            if let std::collections::btree_map::Entry::Occupied(mut entry) =
                self.spellings.entry(key)
            {
                *entry.get_mut() -= 1;
                if *entry.get() == 0 {
                    entry.remove();
                }
            }
        }
    }

    pub(super) fn matching<'a>(
        &'a self,
        path: &Path,
        descendants: bool,
    ) -> impl Iterator<Item = &'a PathBuf> + use<'a> {
        let folded = Self::folded(path);
        self.spellings
            .range((folded.clone(), PathBuf::new())..)
            .take_while(move |((key, _), _)| {
                if descendants {
                    key.starts_with(&folded)
                } else {
                    key == &folded
                }
            })
            .map(|((_, native), _)| native)
    }
}

pub struct VaultIndex {
    notes: HashMap<OsString, NoteMetadata>,
    source_revisions: HashMap<OsString, String>,
    tags: HashMap<String, HashSet<OsString>>,
    link_resolver: LinkResolver,
    backlinks: HashMap<OsString, HashSet<OsString>>,
    stats: VaultStats,
    attachments: HashMap<OsString, u64>,
    attachment_bytes: u64,
    filesystem: Option<fs::VaultRoot>,
    read_contents: Option<HashMap<OsString, String>>,
    exclude: Arc<ExcludeSet>,
    excluded_note_paths: HashSet<OsString>,
    source_batch: bool,
    watch_paths: OnceLock<WatchPaths>,
    #[cfg(has_embeddings)]
    waiting_for_lexical: HashMap<OsString, tantivy::Opstamp>,
}

impl VaultIndex {
    /// Create an empty index (useful for tests and pre-initialization).
    pub fn empty() -> Self {
        Self {
            notes: HashMap::new(),
            source_revisions: HashMap::new(),
            tags: HashMap::new(),
            link_resolver: build_link_resolver(&[]),
            backlinks: HashMap::new(),
            stats: VaultStats {
                total_notes: 0,
                total_files: 0,
                total_tags: 0,
                total_links: 0,
                vault_size_bytes: 0,
                excluded_notes: 0,
            },
            attachments: HashMap::new(),
            attachment_bytes: 0,
            filesystem: None,
            read_contents: None,
            exclude: Arc::default(),
            excluded_note_paths: HashSet::new(),
            source_batch: false,
            watch_paths: OnceLock::new(),
            #[cfg(has_embeddings)]
            waiting_for_lexical: HashMap::new(),
        }
    }

    /// Construct an index whose entries will arrive as source observations.
    /// Source acquisition belongs to the caller and never occurs under this index's lock.
    pub(crate) fn for_source(exclude: Arc<ExcludeSet>) -> Self {
        Self {
            exclude,
            ..Self::empty()
        }
    }

    /// Defer graph reconstruction while an unqueryable source catalog is loading.
    /// Inventory totals and byte admission remain current after every observation.
    pub(crate) fn begin_source_batch(&mut self) {
        self.source_batch = true;
    }

    /// Complete the graph before publishing the source catalog as queryable.
    pub(crate) fn finish_source_batch(&mut self) {
        if std::mem::take(&mut self.source_batch) {
            self.rebuild_backlinks();
        }
    }

    /// Apply one complete source observation to metadata and lexical search.
    /// Eligible notes require their exact observed content; attachments and excluded
    /// paths need only their stat. Publish the returned dirty lexical batch after
    /// releasing the metadata lock, then release eligible embedding upserts.
    pub(crate) fn apply_observation(
        &mut self,
        path: &Path,
        content: Option<String>,
        stat: FileStat,
        tantivy: Option<&super::tantivy_index::TantivyIndex>,
        embeddings_enabled: bool,
    ) -> VaultResult<bool> {
        let was_note = self.notes.contains_key(path.as_os_str());
        let exclude = Arc::clone(&self.exclude);
        let update = self.update_observation(path, content, stat, &exclude);
        self.finish_refresh(path, was_note, update, tantivy, embeddings_enabled)
    }

    /// Refresh stat-only observations whose raw source still matches admitted metadata.
    /// Pending lexical publication remains pending; this never stages another document
    /// or releases an embedding upsert. Callers still publish their completed batch.
    pub(crate) fn refresh_unchanged_note(
        &mut self,
        path: &Path,
        revision: &str,
        stat: FileStat,
    ) -> VaultResult<bool> {
        if !self.notes.contains_key(path.as_os_str())
            || self
                .source_revisions
                .get(path.as_os_str())
                .map(String::as_str)
                != Some(revision)
        {
            return Ok(false);
        }
        let (total, _) = match self.observed_byte_totals(path, stat.size, false) {
            Ok(totals) => totals,
            Err(error) => {
                self.source_revisions.remove(path.as_os_str());
                return Err(error);
            }
        };
        if let Some(note) = self.notes.get_mut(path.as_os_str()) {
            note.stat = stat;
        }
        self.stats.vault_size_bytes = total;
        Ok(true)
    }

    fn update_observation(
        &mut self,
        path: &Path,
        content: Option<String>,
        stat: FileStat,
        exclude: &ExcludeSet,
    ) -> VaultResult<Option<String>> {
        if !super::exclude::is_visible_path(path) {
            self.remove_file(path);
            return Ok(None);
        }
        match classify(path, exclude) {
            FileClass::Note => {
                let content = content.ok_or_else(|| {
                    VaultError::Other(format!(
                        "missing content for observed note: {}",
                        path.display()
                    ))
                })?;
                if !exclude.allows_source(path, &content) {
                    self.add_excluded_file(path);
                    return Ok(None);
                }
                let note = parse_note_metadata(path, &content, stat)?;
                let (total, _) = self.observed_byte_totals(path, note.stat.size, false)?;
                self.replace_note(path, note, total);
                self.source_revisions.insert(
                    path.as_os_str().to_owned(),
                    super::source::source_revision(&content),
                );
                Ok(Some(content))
            }
            FileClass::Excluded => {
                self.add_excluded_file(path);
                Ok(None)
            }
            FileClass::Attachment => {
                if self.attachments.get(path.as_os_str()) != Some(&stat.size) {
                    let (total, attachment_bytes) =
                        self.observed_byte_totals(path, stat.size, true)?;
                    self.remove_file(path);
                    self.track_watch_path(path);
                    self.attachments
                        .insert(path.as_os_str().to_owned(), stat.size);
                    self.attachment_bytes = attachment_bytes;
                    self.stats.total_files += 1;
                    self.stats.vault_size_bytes = total;
                }
                Ok(None)
            }
        }
    }

    /// Size attributes can come from a remote server. Admit aggregate changes
    /// before replacing any inventory, tag, link or lexical contribution.
    fn observed_byte_totals(
        &self,
        path: &Path,
        size: u64,
        attachment: bool,
    ) -> VaultResult<(u64, u64)> {
        let old_attachment = self.attachments.get(path.as_os_str()).copied().unwrap_or(0);
        let old_size = self
            .notes
            .get(path.as_os_str())
            .map_or(old_attachment, |note| note.stat.size);
        let total = self
            .stats
            .vault_size_bytes
            .checked_sub(old_size)
            .and_then(|remaining| remaining.checked_add(size))
            .ok_or_else(|| inventory_size_error(path))?;
        let attachment_bytes = self
            .attachment_bytes
            .checked_sub(old_attachment)
            .and_then(|remaining| remaining.checked_add(if attachment { size } else { 0 }))
            .ok_or_else(|| inventory_size_error(path))?;
        Ok((total, attachment_bytes))
    }

    /// Build an isolated metadata and graph projection without changing shared state.
    pub(crate) fn read_view(&self, contents: HashMap<OsString, String>) -> Self {
        let mut view = Self::empty();
        view.filesystem = self.filesystem.clone();
        view.exclude = Arc::clone(&self.exclude);
        for (path, metadata) in &self.notes {
            let Some(content) = contents.get(path) else {
                continue;
            };
            let path = Path::new(path);
            if !self.exclude.allows_source(path, content) {
                continue;
            }
            let Ok(mut metadata) = parse_note_metadata(path, content, metadata.stat.clone()) else {
                continue;
            };
            metadata.links.retain(|link| {
                self.link_resolver
                    .resolve(&link.target)
                    .is_none_or(|target| contents.contains_key(target.as_os_str()))
            });
            for tag in &metadata.tags {
                view.tags
                    .entry(tag.to_lowercase())
                    .or_default()
                    .insert(path.as_os_str().to_owned());
            }
            view.source_revisions.insert(
                path.as_os_str().to_owned(),
                super::source::source_revision(content),
            );
            view.notes.insert(path.as_os_str().to_owned(), metadata);
        }
        view.link_resolver = build_link_resolver(
            &view
                .notes
                .values()
                .map(|note| note.path.clone())
                .collect::<Vec<_>>(),
        );
        view.rebuild_backlinks();
        view.recompute_stats();
        view.read_contents = Some(contents);
        view
    }

    /// Build the index by walking the entire vault directory.
    pub async fn build(vault_root: &Path, exclude: Arc<ExcludeSet>) -> VaultResult<Self> {
        Self::build_rooted(fs::VaultRoot::open(vault_root)?, exclude).await
    }

    pub(crate) async fn build_rooted(
        filesystem: fs::VaultRoot,
        exclude: Arc<ExcludeSet>,
    ) -> VaultResult<Self> {
        run_cold_build(move |check| Self::build_sync(filesystem, exclude, check, |_, _| Ok(())))
            .await
    }

    pub(crate) async fn build_ranked(
        filesystem: fs::VaultRoot,
        exclude: Arc<ExcludeSet>,
    ) -> VaultResult<(Self, super::tantivy_index::TantivyIndex)> {
        run_cold_build(move |check| {
            let mut builder = super::tantivy_index::TantivyBuilder::new()?;
            let index = Self::build_sync(filesystem, exclude, check, |note, content| {
                builder.add(&note.path, note, content)
            })?;
            check()?;
            Ok((index, builder.finish()?))
        })
        .await
    }

    fn build_sync(
        filesystem: fs::VaultRoot,
        exclude: Arc<ExcludeSet>,
        check: &dyn Fn() -> VaultResult<()>,
        mut index_content: impl FnMut(&NoteMetadata, &str) -> VaultResult<()>,
    ) -> VaultResult<Self> {
        let mut index = Self::empty();
        let mut observed_bytes = 0_u64;
        visit_files(&filesystem, Path::new(""), check, |path| {
            match classify(path, &exclude) {
                FileClass::Excluded => {
                    index
                        .excluded_note_paths
                        .insert(path.as_os_str().to_owned());
                }
                FileClass::Attachment => {
                    let metadata = match filesystem.entry_metadata_exact(path) {
                        Ok(metadata) => metadata,
                        Err(error) if fs::skip_changed_entry(&error) => return Ok(()),
                        Err(error) => return Err(error),
                    };
                    if metadata.is_file() && !fs::is_link(&metadata) {
                        observed_bytes = observed_bytes
                            .checked_add(metadata.len())
                            .ok_or_else(|| inventory_size_error(path))?;
                        let attachment_bytes = index
                            .attachment_bytes
                            .checked_add(metadata.len())
                            .ok_or_else(|| inventory_size_error(path))?;
                        index
                            .attachments
                            .insert(path.as_os_str().to_owned(), metadata.len());
                        index.attachment_bytes = attachment_bytes;
                    }
                }
                FileClass::Note => match read_indexed_note(&filesystem, path) {
                    Ok((metadata, content)) => {
                        if !exclude.allows_source(path, &content) {
                            index
                                .excluded_note_paths
                                .insert(path.as_os_str().to_owned());
                            return Ok(());
                        }
                        check()?;
                        observed_bytes = observed_bytes
                            .checked_add(metadata.stat.size)
                            .ok_or_else(|| inventory_size_error(path))?;
                        index_content(&metadata, &content)?;
                        for tag in &metadata.tags {
                            index
                                .tags
                                .entry(tag.to_lowercase())
                                .or_default()
                                .insert(path.as_os_str().to_owned());
                        }
                        index.source_revisions.insert(
                            path.as_os_str().to_owned(),
                            super::source::source_revision(&content),
                        );
                        index.notes.insert(path.as_os_str().to_owned(), metadata);
                    }
                    Err(error) => {
                        tracing::warn!(path = %path.display(), %error, "skipping unparseable note during index build")
                    }
                },
            }
            Ok(())
        })?;
        check()?;
        index.link_resolver = build_link_resolver(
            &index
                .notes
                .values()
                .map(|note| note.path.clone())
                .collect::<Vec<_>>(),
        );
        index.rebuild_backlinks();
        index.recompute_stats();
        index.filesystem = Some(filesystem);
        index.exclude = exclude;
        Ok(index)
    }

    pub(crate) fn filesystem(&self, root: &Path) -> VaultResult<fs::VaultRoot> {
        if let Some(filesystem) = &self.filesystem {
            if filesystem.path() == root || filesystem.path() == root.canonicalize()? {
                return Ok(filesystem.clone());
            }
            return Err(VaultError::InvalidPath(
                "index belongs to a different vault".into(),
            ));
        }
        fs::VaultRoot::open(root)
    }

    /// An empty index takes ownership of its root on its first mutation or
    /// watcher registration, just as a cold-built index does at construction.
    fn bind_filesystem(&mut self, root: &Path) -> VaultResult<fs::VaultRoot> {
        let filesystem = self.filesystem(root)?;
        if self.filesystem.is_none() {
            self.filesystem = Some(filesystem.clone());
        }
        Ok(filesystem)
    }

    pub(crate) fn bind_watcher(
        &mut self,
        root: &Path,
        policy: &Arc<ExcludeSet>,
    ) -> VaultResult<fs::VaultRoot> {
        if self.filesystem.is_some() && self.exclude.patterns() != policy.patterns() {
            return Err(VaultError::Watcher(
                "watcher exclusions differ from the index policy".into(),
            ));
        }
        let filesystem = self.bind_filesystem(root)?;
        self.exclude = Arc::clone(policy);
        Ok(filesystem)
    }

    /// Re-index a single file (on create or modify).
    ///
    /// Uses incremental backlink update for content-only changes (file already
    /// in the index). Falls back to full rebuild when a new file is added
    /// (because the `LinkResolver` path set changes, potentially altering
    /// ambiguity for other notes' wikilinks).
    pub fn reindex_file(&mut self, vault_root: &Path, path: &Path) -> VaultResult<()> {
        let filesystem = self.bind_filesystem(vault_root)?;
        let actual = match filesystem.resolve_existing(path) {
            Ok(resolved) => resolved.relative,
            Err(error) => {
                self.remove_file(path);
                return Err(error);
            }
        };
        if actual != path {
            self.remove_file(path);
        }
        let exclude = Arc::clone(&self.exclude);
        let result = self.update_file(&filesystem, &actual, &exclude, true);
        if result.is_err() {
            self.source_revisions.remove(actual.as_os_str());
        }
        if result.is_err()
            && !matches!(
                &result,
                Err(VaultError::Storage {
                    kind: StorageErrorKind::ResourceLimit,
                    ..
                })
            )
        {
            self.remove_file(&actual);
        }
        result.map(|_| ())
    }

    fn replace_note(&mut self, path: &Path, metadata: NoteMetadata, total_bytes: u64) {
        #[cfg(has_embeddings)]
        self.waiting_for_lexical.remove(path.as_os_str());
        let was_existing = self.notes.contains_key(path.as_os_str());
        if !self.tracks(path) {
            self.track_watch_path(path);
        }
        self.excluded_note_paths.remove(path.as_os_str());
        if let Some(size) = self.attachments.remove(path.as_os_str()) {
            self.attachment_bytes -= size;
        }

        let old_note = self.remove_note_contributions(path);

        for tag in &metadata.tags {
            self.tags
                .entry(tag.to_lowercase())
                .or_default()
                .insert(path.as_os_str().to_owned());
        }
        self.stats.total_links += metadata.links.len();
        self.notes.insert(path.as_os_str().to_owned(), metadata);
        self.stats.total_notes = self.notes.len();
        self.stats.total_files = self.notes.len() + self.attachments.len();
        self.stats.total_tags = self.tags.len();
        self.stats.excluded_notes = self.excluded_note_paths.len();
        self.stats.vault_size_bytes = total_bytes;

        if !was_existing {
            self.link_resolver.add_path(path.to_path_buf());
        }
        if !self.source_batch {
            if was_existing {
                self.update_backlinks_for_file(
                    path,
                    old_note.as_ref().map(|note| note.links.as_slice()),
                );
            } else {
                self.rebuild_backlinks();
            }
        }
    }

    /// Remove all contributions for a deleted file.
    pub fn remove_file(&mut self, path: &Path) {
        self.source_revisions.remove(path.as_os_str());
        if self.tracks(path)
            && let Some(paths) = self.watch_paths.get_mut()
        {
            paths.remove(path);
        }
        #[cfg(has_embeddings)]
        self.waiting_for_lexical.remove(path.as_os_str());
        if self.excluded_note_paths.remove(path.as_os_str()) {
            self.stats.excluded_notes -= 1;
        }
        if let Some(size) = self.attachments.remove(path.as_os_str()) {
            self.attachment_bytes -= size;
            self.stats.total_files -= 1;
            self.stats.vault_size_bytes -= size;
        }
        if self.remove_note_contributions(path).is_some() {
            self.link_resolver.remove_path(path);
            self.backlinks.remove(path.as_os_str());
            if !self.source_batch {
                self.rebuild_backlinks();
            }
        }
    }

    pub(crate) fn remove_projection(
        &mut self,
        path: &Path,
        tantivy: Option<&super::tantivy_index::TantivyIndex>,
    ) -> VaultResult<bool> {
        let was_note = self.notes.contains_key(path.as_os_str());
        self.remove_file(path);
        if was_note && let Some(tv) = tantivy {
            tv.remove_file_batch(path)?;
            return Ok(true);
        }
        Ok(false)
    }

    /// Handle a file rename/move.
    pub fn rename_file(&mut self, vault_root: &Path, old: &Path, new: &Path) -> VaultResult<()> {
        self.remove_file(old);
        self.reindex_file(vault_root, new)
    }

    /// Classify a physical filesystem entry and return source only for a live note.
    fn update_file(
        &mut self,
        filesystem: &fs::VaultRoot,
        path: &Path,
        exclude: &ExcludeSet,
        require_existing: bool,
    ) -> VaultResult<Option<String>> {
        if !super::exclude::is_visible_path(path) {
            self.remove_file(path);
            return Ok(None);
        }

        let metadata = match filesystem.entry_metadata_exact(path) {
            Ok(metadata) if metadata.is_file() && !fs::is_link(&metadata) => metadata,
            Ok(_) => {
                self.remove_file(path);
                return Ok(None);
            }
            Err(error) if !require_existing && fs::is_missing_entry(&error) => {
                self.remove_file(path);
                return Ok(None);
            }
            Err(error) => return Err(error),
        };
        let (content, stat) = if matches!(classify(path, exclude), FileClass::Note) {
            let (content, stat) = filesystem.read_observed_with_stat(path)?;
            (Some(content), stat)
        } else {
            (
                None,
                FileStat {
                    size: metadata.len(),
                    created: None,
                    modified: None,
                },
            )
        };
        self.update_observation(path, content, stat, exclude)
    }

    /// Refresh metadata and lexical search from the same source. The returned flag
    /// indicates that the caller must flush a Tantivy batch after releasing the lock.
    pub(crate) fn refresh_file(
        &mut self,
        filesystem: &fs::VaultRoot,
        path: &Path,
        exclude: &ExcludeSet,
        tantivy: Option<&super::tantivy_index::TantivyIndex>,
        #[cfg(has_embeddings)] embeddings_enabled: bool,
    ) -> VaultResult<bool> {
        let was_note = self.notes.contains_key(path.as_os_str());
        let update = self.update_file(filesystem, path, exclude, false);
        #[cfg(not(has_embeddings))]
        let embeddings_enabled = false;
        self.finish_refresh(path, was_note, update, tantivy, embeddings_enabled)
    }

    fn finish_refresh(
        &mut self,
        path: &Path,
        was_note: bool,
        update: VaultResult<Option<String>>,
        tantivy: Option<&super::tantivy_index::TantivyIndex>,
        _embeddings_enabled: bool,
    ) -> VaultResult<bool> {
        // Aggregate admission failed before changing the source observation.
        // Keep the previous complete metadata/lexical projection in that case.
        if matches!(
            update,
            Err(VaultError::Storage {
                kind: StorageErrorKind::ResourceLimit,
                ..
            })
        ) {
            self.source_revisions.remove(path.as_os_str());
            return update.map(|_| false);
        }
        if update.is_err() {
            self.remove_file(path);
        }
        let lexical = match (tantivy, &update, self.notes.get(path.as_os_str())) {
            (Some(tv), Ok(Some(content)), Some(metadata)) => {
                tv.replace_document(path, metadata, content).map(|_stamp| {
                    #[cfg(has_embeddings)]
                    if _embeddings_enabled {
                        self.waiting_for_lexical
                            .insert(path.as_os_str().to_owned(), _stamp);
                    }
                    true
                })
            }
            (Some(tv), _, _) if was_note => tv.remove_file_batch(path).map(|_| true),
            _ => Ok(false),
        };
        if lexical.is_err() {
            self.remove_file(path);
        }
        match (update, lexical) {
            (Err(error), Err(cleanup)) => Err(VaultError::Other(format!(
                "{error}; synchronizing search for {} also failed: {cleanup}",
                path.display()
            ))),
            (Err(error), _) => Err(error),
            (_, lexical) => lexical,
        }
    }

    #[cfg(has_embeddings)]
    pub(super) fn embedding_change(
        &self,
        path: &Path,
        was_note: bool,
    ) -> Option<super::embedding_runtime::PendingKind> {
        use super::embedding_runtime::PendingKind;
        if self.waiting_for_lexical.contains_key(path.as_os_str()) {
            Some(PendingKind::AwaitingLexical)
        } else if self.notes.contains_key(path.as_os_str()) {
            Some(PendingKind::Upsert)
        } else {
            was_note.then_some(PendingKind::Remove)
        }
    }

    /// A later writer can replace a path while a commit/reload is under way.
    /// Release only the current stamps covered by the successfully published reader.
    /// An unrelated successful commit also recovers upserts held by a failed flush.
    #[cfg(has_embeddings)]
    pub(super) fn publish_embedding_upserts(
        &mut self,
        committed: tantivy::Opstamp,
        mut upsert: impl FnMut(&Path),
    ) {
        let mut ready: Vec<_> = self
            .waiting_for_lexical
            .extract_if(|_, stamp| *stamp <= committed)
            .collect();
        ready.sort_unstable_by_key(|(_, stamp)| *stamp);
        for (path, _) in ready {
            if self.notes.contains_key(&path) {
                upsert(Path::new(&path));
            }
        }
    }

    pub(crate) fn tracked_paths(&self) -> impl Iterator<Item = &Path> {
        self.notes
            .keys()
            .chain(&self.excluded_note_paths)
            .chain(self.attachments.keys())
            .map(Path::new)
    }

    pub(super) fn watch_paths(&self) -> &WatchPaths {
        self.watch_paths.get_or_init(|| {
            let mut paths = WatchPaths::default();
            for path in self.tracked_paths() {
                paths.insert(path);
            }
            paths
        })
    }

    fn track_watch_path(&mut self, path: &Path) {
        if let Some(paths) = self.watch_paths.get_mut() {
            paths.insert(path);
        }
    }

    pub(crate) fn tracks(&self, path: &Path) -> bool {
        self.notes.contains_key(path.as_os_str())
            || self.excluded_note_paths.contains(path.as_os_str())
            || self.attachments.contains_key(path.as_os_str())
    }

    // ── query methods ───────────────────────────────────────────────

    pub fn notes(&self) -> &HashMap<OsString, NoteMetadata> {
        &self.notes
    }

    #[cfg(has_embeddings)]
    pub(crate) fn allows_source(&self, path: &Path, content: &str) -> bool {
        self.exclude.allows_source(path, content)
    }

    pub fn get_note(&self, path: &Path) -> Option<&NoteMetadata> {
        self.notes.get(path.as_os_str())
    }

    pub fn notes_with_tag(&self, tag: &str) -> Vec<&NoteMetadata> {
        self.tags
            .get(&tag.to_lowercase())
            .map(|paths| paths.iter().filter_map(|p| self.notes.get(p)).collect())
            .unwrap_or_default()
    }

    /// Match a tag and all its children (e.g. `inbox` matches `inbox/read`, `inbox/todo`).
    pub fn notes_with_tag_prefix(&self, prefix: &str) -> Vec<&NoteMetadata> {
        let prefix = prefix.to_lowercase();
        let nested_prefix = format!("{prefix}/");
        let mut seen = HashSet::new();
        let mut results = Vec::new();
        for (tag, paths) in &self.tags {
            if tag == &prefix || tag.starts_with(&nested_prefix) {
                for path in paths {
                    if seen.insert(path)
                        && let Some(note) = self.notes.get(path)
                    {
                        results.push(note);
                    }
                }
            }
        }
        results
    }

    pub fn backlinks_to(&self, path: &Path) -> Vec<&NoteMetadata> {
        self.backlinks
            .get(path.as_os_str())
            .map(|sources| sources.iter().filter_map(|p| self.notes.get(p)).collect())
            .unwrap_or_default()
    }

    pub fn outgoing_links(&self, path: &Path) -> Vec<&WikiLink> {
        self.notes
            .get(path.as_os_str())
            .map(|note| note.links.iter().collect())
            .unwrap_or_default()
    }

    pub fn broken_links(&self) -> Vec<(PathBuf, WikiLink)> {
        let mut result = Vec::new();
        for note in self.notes.values() {
            for link in &note.links {
                if link.target.is_empty() {
                    continue;
                }
                if !self.link_resolver.is_resolved(&link.target) {
                    result.push((note.path.clone(), link.clone()));
                }
            }
        }
        result
    }

    pub fn resolve_link(&self, target: &str) -> Option<PathBuf> {
        self.link_resolver.resolve(target)
    }

    pub fn orphan_notes(&self) -> Vec<&NoteMetadata> {
        self.notes
            .iter()
            .filter(|(path, note)| {
                let has_incoming = self.backlinks.get(*path).is_some_and(|s| !s.is_empty());
                let has_outgoing = note.links.iter().any(|l| !l.target.is_empty());
                !has_incoming && !has_outgoing
            })
            .map(|(_, note)| note)
            .collect()
    }

    pub fn stats(&self) -> &VaultStats {
        &self.stats
    }

    pub fn excluded_notes(&self) -> usize {
        self.excluded_note_paths.len()
    }

    /// Track a markdown file that exists on disk but is excluded from the index.
    pub fn add_excluded_file(&mut self, path: &Path) {
        if self.excluded_note_paths.contains(path.as_os_str()) {
            return;
        }
        self.remove_file(path);
        if super::exclude::is_visible_path(path) && is_markdown(path) {
            self.track_watch_path(path);
            self.excluded_note_paths.insert(path.as_os_str().to_owned());
            self.stats.excluded_notes += 1;
        }
    }

    // ── search methods ──────────────────────────────────────────────

    /// Case-insensitive full-text search across all indexed notes.
    pub fn search_text(
        &self,
        vault_root: &Path,
        query: &str,
        context_len: usize,
    ) -> VaultResult<Vec<SearchResult>> {
        self.search_text_limited(&self.filesystem(vault_root)?, query, context_len, None)
    }

    pub(crate) fn search_text_limited(
        &self,
        filesystem: &fs::VaultRoot,
        query: &str,
        context_len: usize,
        limit: Option<usize>,
    ) -> VaultResult<Vec<SearchResult>> {
        if query.is_empty() {
            return Ok(Vec::new());
        }
        let pattern = format!("(?i){}", regex::escape(query));
        let re = Regex::new(&pattern).map_err(|e| VaultError::InvalidRegex {
            pattern: query.to_string(),
            source: e,
        })?;
        self.search_with_regex(filesystem, &re, context_len, limit)
    }

    const MAX_REGEX_PATTERN_LEN: usize = 1000;
    const REGEX_SIZE_LIMIT: usize = 1 << 20; // 1 MiB NFA size cap

    /// Regex search across all indexed notes.
    pub fn search_regex(
        &self,
        vault_root: &Path,
        pattern: &str,
        context_len: usize,
        max_results: usize,
    ) -> VaultResult<Vec<SearchResult>> {
        let limit = (max_results != 0).then_some(max_results);
        self.search_regex_limited(&self.filesystem(vault_root)?, pattern, context_len, limit)
    }

    pub(crate) fn search_regex_limited(
        &self,
        filesystem: &fs::VaultRoot,
        pattern: &str,
        context_len: usize,
        limit: Option<usize>,
    ) -> VaultResult<Vec<SearchResult>> {
        if pattern.len() > Self::MAX_REGEX_PATTERN_LEN {
            return Err(VaultError::InvalidRegex {
                pattern: pattern.to_string(),
                source: regex::Error::Syntax(format!(
                    "pattern exceeds maximum length of {} characters",
                    Self::MAX_REGEX_PATTERN_LEN
                )),
            });
        }
        let re = regex::RegexBuilder::new(pattern)
            .size_limit(Self::REGEX_SIZE_LIMIT)
            .build()
            .map_err(|e| VaultError::InvalidRegex {
                pattern: pattern.to_string(),
                source: e,
            })?;
        self.search_with_regex(filesystem, &re, context_len, limit)
    }

    /// Search notes by frontmatter field values.
    /// For array fields, checks whether the value is contained in the array.
    pub fn search_frontmatter(&self, field: &str, value: &serde_json::Value) -> Vec<&NoteMetadata> {
        self.notes
            .values()
            .filter(|note| {
                note.frontmatter
                    .as_ref()
                    .is_some_and(|fm| frontmatter_field_matches(fm, field, value))
            })
            .collect()
    }

    /// Find notes where a frontmatter field exists, regardless of value.
    pub fn search_frontmatter_exists(&self, field: &str) -> Vec<&NoteMetadata> {
        self.notes
            .values()
            .filter(|note| {
                note.frontmatter
                    .as_ref()
                    .is_some_and(|fm| fm.get(field).is_some())
            })
            .collect()
    }

    /// Search notes by frontmatter with "contains" semantics:
    /// arrays → element membership, strings → substring match, otherwise exact.
    pub fn search_frontmatter_contains(
        &self,
        field: &str,
        value: &serde_json::Value,
    ) -> Vec<&NoteMetadata> {
        self.notes
            .values()
            .filter(|note| {
                note.frontmatter
                    .as_ref()
                    .is_some_and(|fm| frontmatter_field_contains(fm, field, value))
            })
            .collect()
    }

    // ── private helpers ─────────────────────────────────────────────

    fn remove_note_contributions(&mut self, path: &Path) -> Option<NoteMetadata> {
        let old_note = self.notes.remove(path.as_os_str())?;
        let mut empty_tags = Vec::new();
        for tag in &old_note.tags {
            if let Some(paths) = self.tags.get_mut(&tag.to_lowercase()) {
                paths.remove(path.as_os_str());
                if paths.is_empty() {
                    empty_tags.push(tag.to_lowercase());
                }
            }
        }
        for tag in &empty_tags {
            self.tags.remove(tag);
        }
        self.stats.total_notes = self.notes.len();
        self.stats.total_files = self.notes.len() + self.attachments.len();
        self.stats.total_tags = self.tags.len();
        self.stats.total_links -= old_note.links.len();
        self.stats.vault_size_bytes -= old_note.stat.size;
        Some(old_note)
    }

    /// Incrementally update backlinks for a single file whose content changed.
    ///
    /// Safe only when the `LinkResolver` path set hasn't changed (i.e. the file
    /// was already in the index before the update).
    fn update_backlinks_for_file(&mut self, path: &Path, old_links: Option<&[WikiLink]>) {
        if let Some(old) = old_links {
            for link in old {
                if link.target.is_empty() {
                    continue;
                }
                if let Some(resolved) = self.link_resolver.resolve(&link.target)
                    && let Some(sources) = self.backlinks.get_mut(resolved.as_os_str())
                {
                    sources.remove(path.as_os_str());
                    if sources.is_empty() {
                        self.backlinks.remove(resolved.as_os_str());
                    }
                }
            }
        }

        if let Some(note) = self.notes.get(path.as_os_str()) {
            for link in &note.links {
                if link.target.is_empty() {
                    continue;
                }
                if let Some(resolved) = self.link_resolver.resolve(&link.target) {
                    self.backlinks
                        .entry(resolved.into_os_string())
                        .or_default()
                        .insert(path.as_os_str().to_owned());
                }
            }
        }
    }

    fn rebuild_backlinks(&mut self) {
        self.backlinks.clear();
        for (source, note) in &self.notes {
            for link in &note.links {
                if link.target.is_empty() {
                    continue;
                }
                if let Some(resolved) = self.link_resolver.resolve(&link.target) {
                    self.backlinks
                        .entry(resolved.into_os_string())
                        .or_default()
                        .insert(source.clone());
                }
            }
        }
    }

    fn recompute_stats(&mut self) {
        let md_bytes: u64 = self.notes.values().map(|n| n.stat.size).sum();
        self.stats = VaultStats {
            total_notes: self.notes.len(),
            total_files: self.notes.len() + self.attachments.len(),
            total_tags: self.tags.len(),
            total_links: self.notes.values().map(|n| n.links.len()).sum(),
            vault_size_bytes: md_bytes + self.attachment_bytes,
            excluded_notes: self.excluded_note_paths.len(),
        };
    }

    fn search_with_regex(
        &self,
        filesystem: &fs::VaultRoot,
        re: &Regex,
        context_len: usize,
        limit: Option<usize>,
    ) -> VaultResult<Vec<SearchResult>> {
        let mut results = Vec::new();
        if limit == Some(0) {
            return Ok(results);
        }

        for path in self.notes.keys() {
            if limit == Some(results.len()) {
                break;
            }

            let observed = match &self.read_contents {
                Some(contents) => contents
                    .get(path)
                    .cloned()
                    .ok_or_else(|| VaultError::NoteNotFound(PathBuf::from(path))),
                None => filesystem.read_observed(Path::new(path)),
            };
            let content = match observed {
                Ok(c) => c,
                Err(_) => continue,
            };

            let matches: Vec<SearchMatch> = re
                .find_iter(&content)
                .map(|m| {
                    let (context, match_start, match_end, line) =
                        extract_match_context(&content, m.start(), m.end(), context_len);
                    SearchMatch {
                        line,
                        context,
                        match_start,
                        match_end,
                    }
                })
                .collect();

            if !matches.is_empty() {
                results.push(SearchResult {
                    path: PathBuf::from(path),
                    matches,
                    score: None,
                });
            }
        }

        Ok(results)
    }
}

// ── module-level helpers ────────────────────────────────────────────

enum FileClass {
    Note,
    Excluded,
    Attachment,
}

fn inventory_size_error(path: &Path) -> VaultError {
    VaultError::Storage {
        operation: "index observation".into(),
        path: path.to_string_lossy().into_owned(),
        kind: StorageErrorKind::ResourceLimit,
        message: "aggregate source byte size exceeds the supported u64 range".into(),
        outcome: MutationOutcome::NotDispatched,
    }
}

fn is_markdown(path: &Path) -> bool {
    logical_file_name(path)
        .and_then(|name| name.rsplit_once('.'))
        .is_some_and(|(stem, ext)| !stem.is_empty() && ext.eq_ignore_ascii_case("md"))
}

fn classify(path: &Path, exclude: &ExcludeSet) -> FileClass {
    if !is_markdown(path) {
        FileClass::Attachment
    } else if exclude.is_excluded(path) {
        FileClass::Excluded
    } else {
        FileClass::Note
    }
}

/// A blocking task cannot be aborted once started. Keep the receiver in the
/// requesting future so dropping that future withdraws the ongoing disk scan.
async fn run_cold_build<T: Send + 'static>(
    build: impl FnOnce(&dyn Fn() -> VaultResult<()>) -> VaultResult<T> + Send + 'static,
) -> VaultResult<T> {
    let (caller, lifetime) = tokio::sync::oneshot::channel::<()>();
    let result = tokio::task::spawn_blocking(move || {
        let check = || {
            if caller.is_closed() {
                Err(VaultError::Other("index build cancelled".into()))
            } else {
                Ok(())
            }
        };
        check()?;
        build(&check)
    })
    .await
    .map_err(|error| VaultError::Other(format!("index build task panicked: {error}")))?;
    drop(lifetime);
    result
}

/// Walk visible physical entries through directory handles. The caller receives
/// native spellings from directory enumeration; symlinks are never descended.
pub(crate) fn visit_files(
    root: &fs::VaultRoot,
    directory: &Path,
    check: impl Fn() -> VaultResult<()>,
    mut visit: impl FnMut(&Path) -> VaultResult<()>,
) -> VaultResult<()> {
    root.visit_entries(
        directory,
        true,
        super::exclude::is_visible_path,
        check,
        |path, is_file| {
            if is_file {
                visit(path)?;
            }
            Ok(())
        },
    )
}

fn read_indexed_note(root: &fs::VaultRoot, path: &Path) -> VaultResult<(NoteMetadata, String)> {
    let (content, stat) = root.read_observed_with_stat(path)?;
    let metadata = parse_note_metadata(path, &content, stat)?;
    Ok((metadata, content))
}

pub(crate) fn parse_note_metadata(
    path: &Path,
    content: &str,
    stat: FileStat,
) -> VaultResult<NoteMetadata> {
    let fm = frontmatter::parse_frontmatter(content).map_err(|error| {
        VaultError::storage(
            "parse frontmatter",
            path.to_string_lossy(),
            StorageErrorKind::InvalidSource,
            error.to_string(),
        )
    })?;
    let fm_tags = fm
        .as_ref()
        .map(frontmatter::extract_frontmatter_tags)
        .unwrap_or_default();

    let headings = parser::extract_headings(content);
    let inline_tags = parser::extract_inline_tags(content);
    let links = parser::extract_wikilinks(content);
    let block_refs = parser::extract_block_refs(content);

    let mut tags = fm_tags;
    tags.extend(inline_tags);
    tags.sort();
    tags.dedup();

    let title = logical_file_stem(path).unwrap_or_default().to_owned();

    Ok(NoteMetadata {
        path: path.to_path_buf(),
        title,
        tags,
        frontmatter: fm,
        headings,
        links,
        block_refs,
        stat,
    })
}

/// Extract a context window around a regex match.
/// Returns `(context_string, char_offset_start, char_offset_end, line_number)`.
pub(crate) fn extract_match_context(
    content: &str,
    match_byte_start: usize,
    match_byte_end: usize,
    context_len: usize,
) -> (String, usize, usize, usize) {
    let ctx_start = content.floor_char_boundary(match_byte_start.saturating_sub(context_len));
    let ctx_end = content.ceil_char_boundary(
        match_byte_end
            .saturating_add(context_len)
            .min(content.len()),
    );

    let line = content[..match_byte_start]
        .bytes()
        .filter(|&b| b == b'\n')
        .count();

    let context = content[ctx_start..ctx_end].to_string();
    let match_start_chars = content[ctx_start..match_byte_start].chars().count();
    let match_len_chars = content[match_byte_start..match_byte_end].chars().count();

    (
        context,
        match_start_chars,
        match_start_chars + match_len_chars,
        line,
    )
}

fn frontmatter_field_matches(
    fm: &serde_json::Value,
    field: &str,
    value: &serde_json::Value,
) -> bool {
    let Some(field_val) = fm.get(field) else {
        return false;
    };

    if field_val == value {
        return true;
    }

    if let serde_json::Value::Array(arr) = field_val {
        return arr.contains(value);
    }

    false
}

fn frontmatter_field_contains(
    fm: &serde_json::Value,
    field: &str,
    value: &serde_json::Value,
) -> bool {
    let Some(field_val) = fm.get(field) else {
        return false;
    };

    match field_val {
        serde_json::Value::Array(arr) => arr.contains(value),
        serde_json::Value::String(haystack) => {
            if let serde_json::Value::String(needle) = value {
                haystack.contains(needle.as_str())
            } else {
                false
            }
        }
        _ => field_val == value,
    }
}

// ── tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs as stdfs;
    use std::sync::Arc;
    use tempfile::TempDir;
    use unicode_normalization::UnicodeNormalization;

    fn empty_exclude() -> Arc<ExcludeSet> {
        Arc::new(ExcludeSet::build(vec![]).unwrap())
    }

    fn observed_stat(size: usize) -> FileStat {
        FileStat {
            size: size as u64,
            created: None,
            modified: None,
        }
    }

    #[test]
    fn unchanged_source_revision_updates_only_stat_inventory() {
        let mut index = VaultIndex::for_source(empty_exclude());
        let lexical = super::super::tantivy_index::TantivyIndex::empty().unwrap();
        let path = Path::new("a:b.md");
        let content = "# Source\nquokka #kept [[target]]";
        let revision = super::super::source::source_revision(content);
        index
            .apply_observation(
                path,
                Some(content.into()),
                observed_stat(content.len()),
                Some(&lexical),
                false,
            )
            .unwrap();
        lexical.flush().unwrap();
        let before = index.get_note(path).unwrap().clone();
        let before_stats = serde_json::to_value(index.stats()).unwrap();
        let updated_stat = FileStat {
            size: content.len() as u64 + 17,
            created: Some(chrono::Utc::now()),
            modified: Some(chrono::Utc::now()),
        };
        assert!(
            index
                .refresh_unchanged_note(path, &revision, updated_stat.clone())
                .unwrap()
        );
        let mut expected = before;
        expected.stat = updated_stat;
        assert_eq!(
            serde_json::to_value(index.get_note(path).unwrap()).unwrap(),
            serde_json::to_value(expected).unwrap()
        );
        let mut expected_stats = before_stats;
        expected_stats["vault_size_bytes"] = serde_json::json!(content.len() + 17);
        assert_eq!(serde_json::to_value(index.stats()).unwrap(), expected_stats);
        assert_eq!(lexical.search("quokka", 10).unwrap().len(), 1);
        assert!(
            !index
                .refresh_unchanged_note(Path::new("A:b.md"), &revision, observed_stat(100))
                .unwrap()
        );
        assert!(
            !index
                .refresh_unchanged_note(
                    path,
                    &super::super::source::source_revision("changed"),
                    observed_stat(100)
                )
                .unwrap()
        );
        assert_eq!(index.stats().vault_size_bytes, content.len() as u64 + 17);
    }

    #[test]
    fn rejected_or_removed_source_observations_clear_revision_shortcuts() {
        let mut index = VaultIndex::for_source(empty_exclude());
        let path = Path::new("note.md");
        let content = "# Note\nquokka";
        let revision = super::super::source::source_revision(content);
        let admit = |index: &mut VaultIndex| {
            index
                .apply_observation(
                    path,
                    Some(content.into()),
                    observed_stat(content.len()),
                    None,
                    false,
                )
                .unwrap();
        };
        admit(&mut index);
        index.remove_file(path);
        assert!(
            !index
                .refresh_unchanged_note(path, &revision, observed_stat(1))
                .unwrap()
        );
        assert!(!index.source_revisions.contains_key(path.as_os_str()));
        for invalid in [Some("---\ntags: [unterminated\n---\n".into()), None] {
            admit(&mut index);
            assert!(
                index
                    .apply_observation(path, invalid, observed_stat(1), None, false)
                    .is_err()
            );
            assert!(
                !index
                    .refresh_unchanged_note(path, &revision, observed_stat(1))
                    .unwrap()
            );
            assert!(!index.source_revisions.contains_key(path.as_os_str()));
        }
        admit(&mut index);
        index
            .apply_observation(
                Path::new("large.bin"),
                None,
                FileStat {
                    size: u64::MAX - content.len() as u64,
                    created: None,
                    modified: None,
                },
                None,
                false,
            )
            .unwrap();
        assert!(matches!(
            index.refresh_unchanged_note(path, &revision, observed_stat(content.len() + 1)),
            Err(VaultError::Storage {
                kind: StorageErrorKind::ResourceLimit,
                ..
            })
        ));
        assert_eq!(
            index.get_note(path).unwrap().stat.size,
            content.len() as u64
        );
        assert_eq!(index.stats().vault_size_bytes, u64::MAX);
        assert!(
            !index
                .refresh_unchanged_note(path, &revision, observed_stat(content.len()))
                .unwrap()
        );
        index.remove_file(Path::new("large.bin"));
        admit(&mut index);
        index
            .apply_observation(
                Path::new("large.bin"),
                None,
                FileStat {
                    size: u64::MAX - content.len() as u64,
                    created: None,
                    modified: None,
                },
                None,
                false,
            )
            .unwrap();
        assert!(
            index
                .apply_observation(
                    path,
                    Some("changed".into()),
                    observed_stat(content.len() + 1),
                    None,
                    false
                )
                .is_err()
        );
        assert!(index.get_note(path).is_some());
        assert!(
            !index
                .refresh_unchanged_note(path, &revision, observed_stat(content.len()))
                .unwrap()
        );
    }

    #[cfg(has_embeddings)]
    #[test]
    fn unchanged_source_revision_preserves_lexical_generation_and_upsert_gate() {
        use super::super::embedding_runtime::PendingKind;

        let mut index = VaultIndex::for_source(empty_exclude());
        let lexical = super::super::tantivy_index::TantivyIndex::empty().unwrap();
        let path = Path::new("remote.md");
        let content = "# Source\nquokka";
        let revision = super::super::source::source_revision(content);
        index
            .apply_observation(
                path,
                Some(content.into()),
                observed_stat(content.len()),
                Some(&lexical),
                true,
            )
            .unwrap();
        let pending = index.waiting_for_lexical[path.as_os_str()];
        assert!(
            index
                .refresh_unchanged_note(path, &revision, observed_stat(content.len() + 1))
                .unwrap()
        );
        assert_eq!(index.waiting_for_lexical[path.as_os_str()], pending);
        assert!(matches!(
            index.embedding_change(path, true),
            Some(PendingKind::AwaitingLexical)
        ));
        assert!(lexical.search("quokka", 10).unwrap().is_empty());
        let mut submitted = Vec::new();
        index.publish_embedding_upserts(lexical.publish().unwrap(), |path| {
            submitted.push(path.to_owned())
        });
        assert_eq!(submitted, [path.to_owned()]);
        assert!(
            index
                .refresh_unchanged_note(path, &revision, observed_stat(content.len() + 2))
                .unwrap()
        );
        assert!(index.waiting_for_lexical.is_empty());
        submitted.clear();
        index.publish_embedding_upserts(lexical.publish().unwrap(), |path| {
            submitted.push(path.to_owned())
        });
        assert!(
            submitted.is_empty(),
            "verified unchanged content must not resubmit an upsert"
        );
        assert_eq!(lexical.search("quokka", 10).unwrap().len(), 1);
    }

    #[tokio::test]
    async fn native_build_and_refresh_record_exact_raw_source_revisions() {
        let directory = TempDir::new().unwrap();
        let path = Path::new("note.md");
        let original = "# Source\noriginal";
        stdfs::write(directory.path().join(path), original).unwrap();
        let mut index = VaultIndex::build(directory.path(), empty_exclude())
            .await
            .unwrap();
        assert!(
            index
                .refresh_unchanged_note(
                    path,
                    &super::super::source::source_revision(original),
                    observed_stat(original.len())
                )
                .unwrap()
        );
        let replacement = "# Source\nreplacement";
        stdfs::write(directory.path().join(path), replacement).unwrap();
        let filesystem = index.filesystem(directory.path()).unwrap();
        index
            .refresh_file(
                &filesystem,
                path,
                &ExcludeSet::default(),
                None,
                #[cfg(has_embeddings)]
                false,
            )
            .unwrap();
        assert!(
            !index
                .refresh_unchanged_note(
                    path,
                    &super::super::source::source_revision(original),
                    observed_stat(original.len())
                )
                .unwrap()
        );
        assert!(
            index
                .refresh_unchanged_note(
                    path,
                    &super::super::source::source_revision(replacement),
                    observed_stat(replacement.len())
                )
                .unwrap()
        );
        stdfs::write(directory.path().join(path), "---\n[invalid\n---\n").unwrap();
        assert!(
            index
                .refresh_file(
                    &filesystem,
                    path,
                    &ExcludeSet::default(),
                    None,
                    #[cfg(has_embeddings)]
                    false,
                )
                .is_err()
        );
        assert!(
            !index
                .refresh_unchanged_note(
                    path,
                    &super::super::source::source_revision(replacement),
                    observed_stat(replacement.len())
                )
                .unwrap()
        );
    }

    #[test]
    fn isolated_read_views_preserve_exact_keys_and_retained_source() {
        let mut index = VaultIndex::for_source(empty_exclude());
        for (path, content) in [("a:b.md", "source #lower"), ("A:b.md", "source #upper")] {
            index
                .apply_observation(
                    Path::new(path),
                    Some(content.into()),
                    observed_stat(content.len()),
                    None,
                    false,
                )
                .unwrap();
        }
        let contents = HashMap::from([
            (OsString::from("a:b.md"), "view #lower".to_owned()),
            (OsString::from("A:b.md"), "view #upper".to_owned()),
        ]);
        let view = index.read_view(contents);
        assert_eq!(view.notes().len(), 2);
        assert_eq!(view.get_note(Path::new("a:b.md")).unwrap().tags, ["lower"]);
        assert_eq!(view.get_note(Path::new("A:b.md")).unwrap().tags, ["upper"]);
        let directory = TempDir::new().unwrap();
        let root = fs::VaultRoot::open(directory.path()).unwrap();
        let hits = view.search_text_limited(&root, "view", 0, None).unwrap();
        assert_eq!(
            hits.len(),
            2,
            "a read projection uses its admitted bytes without native reads"
        );
        assert!(
            view.search_text_limited(&root, "source", 0, None)
                .unwrap()
                .is_empty()
        );
        assert_eq!(index.notes().len(), 2);
        let filtered = index.read_view(HashMap::from([(
            OsString::from("A:b.md"),
            "view #upper".to_owned(),
        )]));
        assert!(filtered.get_note(Path::new("a:b.md")).is_none());
        assert!(filtered.get_note(Path::new("A:b.md")).is_some());
        assert_eq!(filtered.stats().total_notes, 1);
    }

    #[test]
    fn source_metadata_preserves_names_with_native_prefix_meanings() {
        let mut index = VaultIndex::for_source(empty_exclude());
        for (path, title) in [
            ("a:b.md", "a:b"),
            ("a:b/CON.md", "CON"),
            ("notes/a:b.MD", "a:b"),
        ] {
            let path = Path::new(path);
            index
                .apply_observation(path, Some("# Source".into()), observed_stat(8), None, false)
                .unwrap();
            let metadata = index.get_note(path).unwrap();
            assert_eq!(metadata.path, path);
            assert_eq!(metadata.title, title);
        }
        assert_eq!(index.stats().total_notes, 3);
        assert!(!is_markdown(Path::new(".md")));
    }

    #[test]
    fn source_inventory_keys_preserve_exact_prefix_spelling() {
        let mut index = VaultIndex::for_source(empty_exclude());
        for path in ["a:b.md", "A:b.md"] {
            index
                .apply_observation(
                    Path::new(path),
                    Some("#shared".into()),
                    observed_stat(7),
                    None,
                    false,
                )
                .unwrap();
        }
        assert_eq!(index.notes().len(), 2);
        assert_eq!(index.notes_with_tag("shared").len(), 2);
        assert_eq!(index.stats().vault_size_bytes, 14);
        index.remove_file(Path::new("a:b.md"));
        assert!(index.get_note(Path::new("a:b.md")).is_none());
        assert_eq!(index.get_note(Path::new("A:b.md")).unwrap().title, "A:b");
        assert_eq!(index.notes_with_tag("shared").len(), 1);
        assert_eq!(index.stats().vault_size_bytes, 7);

        for path in ["c:asset", "C:asset"] {
            index
                .apply_observation(Path::new(path), None, observed_stat(1), None, false)
                .unwrap();
        }
        assert_eq!(index.attachments.len(), 2);
        index.remove_file(Path::new("c:asset"));
        assert!(index.tracks(Path::new("C:asset")));
        assert_eq!(index.stats().total_files, 2);
        assert_eq!(index.stats().vault_size_bytes, 8);
    }

    #[test]
    fn malformed_source_metadata_reports_a_document_error_and_removes_projection() {
        let mut index = VaultIndex::for_source(empty_exclude());
        let path = Path::new("invalid.md");
        index
            .apply_observation(path, Some("#valid".into()), observed_stat(6), None, false)
            .unwrap();
        for content in ["---\ntags: [unterminated\n---\n", "---\n- scalar\n---\n"] {
            let error = index
                .apply_observation(
                    path,
                    Some(content.into()),
                    observed_stat(content.len()),
                    None,
                    false,
                )
                .unwrap_err();
            assert!(matches!(
                error,
                VaultError::Storage {
                    kind: StorageErrorKind::InvalidSource,
                    path: ref source,
                    ..
                } if source == "invalid.md"
            ));
            assert!(index.get_note(path).is_none());
        }
        index
            .apply_observation(
                path,
                Some("#recovered".into()),
                observed_stat(10),
                None,
                false,
            )
            .unwrap();
        assert_eq!(index.stats().total_notes, 1);
    }

    #[test]
    fn source_batch_defers_graph_work_and_keeps_admission_totals_current() {
        let exclude = Arc::new(ExcludeSet::build(vec!["Private/**".into()]).unwrap());
        let mut index = VaultIndex::for_source(exclude);
        index.begin_source_batch();
        let content = "[[target]] #batch";
        for number in 0..1000 {
            index
                .apply_observation(
                    Path::new(&format!("note-{number}.md")),
                    Some(content.into()),
                    observed_stat(content.len()),
                    None,
                    false,
                )
                .unwrap();
            assert_eq!(index.stats().total_notes, number + 1);
            assert_eq!(index.stats().total_links, number + 1);
            assert_eq!(
                index.stats().vault_size_bytes,
                ((number + 1) * content.len()) as u64
            );
        }
        for (path, content) in [
            ("target.md", "# Target"),
            ("note-0.md", "[[missing]] #updated"),
        ] {
            index
                .apply_observation(
                    Path::new(path),
                    Some(content.into()),
                    observed_stat(content.len()),
                    None,
                    false,
                )
                .unwrap();
        }
        index.remove_file(Path::new("note-1.md"));
        index
            .apply_observation(Path::new("asset.bin"), None, observed_stat(5), None, false)
            .unwrap();
        index
            .apply_observation(
                Path::new("Private/hidden.md"),
                None,
                observed_stat(100),
                None,
                false,
            )
            .unwrap();
        let stats = serde_json::to_value(index.stats()).unwrap();
        let rejected = index.apply_observation(
            Path::new("oversized.bin"),
            None,
            FileStat {
                size: u64::MAX,
                created: None,
                modified: None,
            },
            None,
            false,
        );
        assert!(matches!(
            rejected,
            Err(VaultError::Storage {
                kind: StorageErrorKind::ResourceLimit,
                ..
            })
        ));
        assert_eq!(serde_json::to_value(index.stats()).unwrap(), stats);
        // Bootstrap never reconstructs the growing graph per observation.
        assert!(index.backlinks.is_empty());
        index.finish_source_batch();
        assert_eq!(index.backlinks_to(Path::new("target.md")).len(), 998);
        assert_eq!(index.broken_links().len(), 1);
        assert_eq!(index.stats().total_notes, 1000);
        assert_eq!(index.stats().total_files, 1001);
        assert_eq!(index.stats().total_tags, 2);
        assert_eq!(index.stats().excluded_notes, 1);
        index.recompute_stats();
        assert_eq!(serde_json::to_value(index.stats()).unwrap(), stats);

        // Once published, both content changes and path-set ambiguity update
        // the graph immediately and preserve the incremental statistics.
        index
            .apply_observation(
                Path::new("note-2.md"),
                Some("#after".into()),
                observed_stat(6),
                None,
                false,
            )
            .unwrap();
        assert_eq!(index.backlinks_to(Path::new("target.md")).len(), 997);
        index
            .apply_observation(
                Path::new("nested/target.md"),
                Some(String::new()),
                observed_stat(0),
                None,
                false,
            )
            .unwrap();
        assert!(index.backlinks_to(Path::new("target.md")).is_empty());
        index.remove_file(Path::new("nested/target.md"));
        assert_eq!(index.backlinks_to(Path::new("target.md")).len(), 997);
        let incremental = serde_json::to_value(index.stats()).unwrap();
        index.recompute_stats();
        assert_eq!(serde_json::to_value(index.stats()).unwrap(), incremental);
    }

    #[test]
    fn source_observations_publish_matching_metadata_and_lexical_content() {
        let mut index = VaultIndex::for_source(empty_exclude());
        let lexical = super::super::tantivy_index::TantivyIndex::empty().unwrap();
        let path = Path::new("notes/remote.md");
        let initial = "# First\nquokka #old [[other]]";
        assert!(
            index
                .apply_observation(
                    path,
                    Some(initial.to_owned()),
                    observed_stat(initial.len()),
                    Some(&lexical),
                    false,
                )
                .unwrap()
        );
        lexical.flush().unwrap();
        assert_eq!(index.get_note(path).unwrap().tags, ["old"]);
        assert_eq!(lexical.search("quokka", 10).unwrap().len(), 1);

        let changed = "# Updated\nwombat #new";
        index
            .apply_observation(
                path,
                Some(changed.to_owned()),
                observed_stat(changed.len()),
                Some(&lexical),
                false,
            )
            .unwrap();
        lexical.flush().unwrap();
        assert_eq!(index.get_note(path).unwrap().tags, ["new"]);
        assert!(index.notes_with_tag("old").is_empty());
        assert!(lexical.search("quokka", 10).unwrap().is_empty());
        assert_eq!(lexical.search("wombat", 10).unwrap().len(), 1);

        assert!(index.remove_projection(path, Some(&lexical)).unwrap());
        lexical.flush().unwrap();
        assert!(index.get_note(path).is_none());
        assert!(lexical.search("wombat", 10).unwrap().is_empty());
        assert_eq!(index.stats().total_files, 0);
    }

    #[test]
    fn source_inventory_does_not_require_excluded_content_or_double_count_attachments() {
        let exclusions = Arc::new(ExcludeSet::build(vec!["Private/**".into()]).unwrap());
        let mut index = VaultIndex::for_source(exclusions);
        index
            .apply_observation(
                Path::new("Private/secret.md"),
                None,
                observed_stat(10),
                None,
                false,
            )
            .unwrap();
        for size in [10, 10, 20] {
            index
                .apply_observation(
                    Path::new("image.png"),
                    None,
                    observed_stat(size),
                    None,
                    false,
                )
                .unwrap();
        }
        assert_eq!(index.stats().excluded_notes, 1);
        assert_eq!(index.stats().total_files, 1);
        assert_eq!(index.stats().vault_size_bytes, 20);
        assert!(index.get_note(Path::new("Private/secret.md")).is_none());
        assert!(
            index
                .apply_observation(Path::new("missing.md"), None, observed_stat(3), None, false,)
                .is_err()
        );
    }

    #[test]
    fn source_attachment_byte_overflow_preserves_the_complete_inventory() {
        let mut index = VaultIndex::for_source(empty_exclude());
        let first = Path::new("first.bin");
        let second = Path::new("second.bin");
        index
            .apply_observation(
                first,
                None,
                FileStat {
                    size: u64::MAX,
                    created: None,
                    modified: None,
                },
                None,
                false,
            )
            .unwrap();
        let error = index
            .apply_observation(second, None, observed_stat(1), None, false)
            .unwrap_err();
        assert!(matches!(
            error,
            VaultError::Storage {
                kind: StorageErrorKind::ResourceLimit,
                ..
            }
        ));
        assert_eq!(index.stats().vault_size_bytes, u64::MAX);
        assert_eq!(index.stats().total_files, 1);
        assert_eq!(index.attachments.get(first.as_os_str()), Some(&u64::MAX));
        assert!(!index.tracks(second));

        // Replacing an existing attachment releases its previous reservation.
        index
            .apply_observation(first, None, observed_stat(1), None, false)
            .unwrap();
        index
            .apply_observation(
                second,
                None,
                FileStat {
                    size: u64::MAX - 1,
                    created: None,
                    modified: None,
                },
                None,
                false,
            )
            .unwrap();
        assert_eq!(index.stats().vault_size_bytes, u64::MAX);
        assert!(
            index
                .apply_observation(first, None, observed_stat(2), None, false)
                .is_err()
        );
        assert_eq!(index.attachments.get(first.as_os_str()), Some(&1));
        assert_eq!(index.stats().total_files, 2);
        index.remove_file(first);
        index.remove_file(second);
        assert_eq!(index.stats().vault_size_bytes, 0);
        assert_eq!(index.stats().total_files, 0);
    }

    #[test]
    fn source_note_byte_overflow_does_not_replace_metadata_or_lexical_content() {
        let mut index = VaultIndex::for_source(empty_exclude());
        let lexical = super::super::tantivy_index::TantivyIndex::empty().unwrap();
        let path = Path::new("remote.md");
        index
            .apply_observation(
                Path::new("large.bin"),
                None,
                FileStat {
                    size: u64::MAX - 1,
                    created: None,
                    modified: None,
                },
                Some(&lexical),
                false,
            )
            .unwrap();
        index
            .apply_observation(
                path,
                Some("quokka #old".into()),
                observed_stat(1),
                Some(&lexical),
                false,
            )
            .unwrap();
        lexical.flush().unwrap();
        let error = index
            .apply_observation(
                path,
                Some("wombat #new".into()),
                observed_stat(2),
                Some(&lexical),
                false,
            )
            .unwrap_err();
        assert!(matches!(
            error,
            VaultError::Storage {
                kind: StorageErrorKind::ResourceLimit,
                ..
            }
        ));
        lexical.flush().unwrap();
        assert_eq!(index.get_note(path).unwrap().stat.size, 1);
        assert_eq!(index.get_note(path).unwrap().tags, ["old"]);
        assert!(index.notes_with_tag("new").is_empty());
        assert_eq!(lexical.search("quokka", 10).unwrap().len(), 1);
        assert!(lexical.search("wombat", 10).unwrap().is_empty());
        assert_eq!(index.stats().vault_size_bytes, u64::MAX);
        assert_eq!(index.stats().total_files, 2);

        index.remove_file(Path::new("large.bin"));
        assert_eq!(index.stats().vault_size_bytes, 1);
        index
            .apply_observation(
                path,
                Some("wombat #new".into()),
                observed_stat(2),
                Some(&lexical),
                false,
            )
            .unwrap();
        assert_eq!(index.stats().vault_size_bytes, 2);
    }

    #[test]
    fn source_note_totals_reject_overflow_before_adding_contributions() {
        let mut index = VaultIndex::for_source(empty_exclude());
        let first = Path::new("first.md");
        index
            .apply_observation(
                first,
                Some("quokka #first".into()),
                FileStat {
                    size: u64::MAX,
                    created: None,
                    modified: None,
                },
                None,
                false,
            )
            .unwrap();
        assert!(matches!(
            index.apply_observation(
                Path::new("second.md"),
                Some("wombat #second [[first]]".into()),
                observed_stat(1),
                None,
                false,
            ),
            Err(VaultError::Storage {
                kind: StorageErrorKind::ResourceLimit,
                ..
            })
        ));
        assert_eq!(index.stats().total_notes, 1);
        assert_eq!(index.stats().total_files, 1);
        assert_eq!(index.stats().vault_size_bytes, u64::MAX);
        assert!(index.notes_with_tag("second").is_empty());
        assert!(index.backlinks_to(first).is_empty());
        index.remove_file(first);
        assert_eq!(index.stats().vault_size_bytes, 0);
    }

    #[test]
    fn malformed_source_observation_invalidates_existing_projections() {
        let mut index = VaultIndex::for_source(empty_exclude());
        let lexical = super::super::tantivy_index::TantivyIndex::empty().unwrap();
        let path = Path::new("remote.md");
        let valid = "# Note\nquokka";
        index
            .apply_observation(
                path,
                Some(valid.to_owned()),
                observed_stat(valid.len()),
                Some(&lexical),
                false,
            )
            .unwrap();
        lexical.flush().unwrap();
        let invalid = "---\ntags: [unterminated\n---\nquokka";
        assert!(
            index
                .apply_observation(
                    path,
                    Some(invalid.to_owned()),
                    observed_stat(invalid.len()),
                    Some(&lexical),
                    false,
                )
                .is_err()
        );
        lexical.flush().unwrap();
        assert!(index.get_note(path).is_none());
        assert!(lexical.search("quokka", 10).unwrap().is_empty());
    }

    #[cfg(has_embeddings)]
    #[test]
    fn source_observation_waits_for_matching_lexical_publication() {
        use super::super::embedding_runtime::PendingKind;

        let mut index = VaultIndex::for_source(empty_exclude());
        let lexical = super::super::tantivy_index::TantivyIndex::empty().unwrap();
        let path = Path::new("remote.md");
        let content = "# Note\nquokka";
        index
            .apply_observation(
                path,
                Some(content.to_owned()),
                observed_stat(content.len()),
                Some(&lexical),
                true,
            )
            .unwrap();
        assert!(matches!(
            index.embedding_change(path, false),
            Some(PendingKind::AwaitingLexical)
        ));
        let mut released = Vec::new();
        index.publish_embedding_upserts(lexical.publish().unwrap(), |path| {
            released.push(path.to_owned());
        });
        assert_eq!(released, [path.to_owned()]);
        assert!(matches!(
            index.embedding_change(path, false),
            Some(PendingKind::Upsert)
        ));
    }

    #[tokio::test]
    async fn index_build_and_reindex_yaml_block_sequences() {
        let content = "---\ntitle: \"Test Document\"\nauthor: \"Jane Doe\"\ntags:\n  - documentation\n  - yaml-test\n---\n\n# Hello\n\nSome content here.\n";
        for newline in ["\n", "\r\n"] {
            let dir = TempDir::new().unwrap();
            let path = Path::new("test.md");
            let content = content.replace('\n', newline);
            stdfs::write(dir.path().join(path), &content).unwrap();

            let mut index = VaultIndex::build(dir.path(), empty_exclude())
                .await
                .unwrap();
            let note = index
                .get_note(path)
                .expect("block sequences must be indexed");
            assert_eq!(note.tags, ["documentation", "yaml-test"]);
            let fm = note.frontmatter.as_ref().unwrap();
            assert_eq!(fm["title"], "Test Document");
            assert_eq!(fm["author"], "Jane Doe");
            assert_eq!(
                fm["tags"],
                serde_json::json!(["documentation", "yaml-test"])
            );

            stdfs::write(
                dir.path().join(path),
                content.replace("yaml-test", "updated"),
            )
            .unwrap();
            index.reindex_file(dir.path(), path).unwrap();
            assert_eq!(
                index.get_note(path).unwrap().tags,
                ["documentation", "updated"]
            );
        }
    }

    #[test]
    fn cold_indexes_open_walked_names_without_sibling_lookups() {
        use crate::test_helpers::{EXACT_FILE_OPENS, INDEX_DIRECTORY_VISITS, NAME_LOOKUP_SCANS};
        for siblings in [1, 100] {
            let directory = TempDir::new().unwrap();
            let filesystem = fs::VaultRoot::open(directory.path()).unwrap();
            let folder = directory.path().join("notes");
            stdfs::create_dir(&folder).unwrap();
            // Keep the exact spelling returned by the filesystem, including NFD.
            let target_name = "re\u{301}sume\u{301}.md";
            let source = "quokka #tag [[re\u{301}sume\u{301}]]";
            for i in 0..siblings {
                let name = if i == 0 {
                    target_name.to_owned()
                } else {
                    format!("{i}.md")
                };
                stdfs::write(folder.join(name), source).unwrap();
            }
            stdfs::write(folder.join("asset.bin"), [0xff, 0]).unwrap();
            stdfs::write(folder.join("ignored.md"), "unread").unwrap();
            let excluded = Arc::new(ExcludeSet::build(vec!["notes/ignored.md".into()]).unwrap());
            NAME_LOOKUP_SCANS.set(0);
            INDEX_DIRECTORY_VISITS.set(0);
            EXACT_FILE_OPENS.set(0);
            let mut builder = crate::vault::tantivy_index::TantivyBuilder::new().unwrap();
            let index =
                VaultIndex::build_sync(filesystem, excluded, &|| Ok(()), |note, content| {
                    builder.add(&note.path, note, content)
                })
                .unwrap();
            let lexical = builder.finish().unwrap();
            assert_eq!(lexical.search("quokka", siblings).unwrap().len(), siblings);
            assert_eq!(EXACT_FILE_OPENS.get(), siblings);
            assert_eq!(INDEX_DIRECTORY_VISITS.get(), 2);
            assert_eq!(NAME_LOOKUP_SCANS.get(), 0);
            assert_eq!(index.notes_with_tag("tag").len(), siblings);
            let target = Path::new("notes").join(target_name);
            assert_eq!(index.backlinks_to(&target).len(), siblings);
            assert_eq!(index.stats().total_links, siblings);
            assert_eq!(
                index.stats().vault_size_bytes,
                (source.len() * siblings + 2) as u64
            );
            assert_eq!(index.stats().total_files, siblings + 1);
            assert_eq!(index.excluded_notes(), 1);
        }
    }

    #[tokio::test]
    async fn dropping_the_cold_build_request_stops_source_opens() {
        let temp = TempDir::new().unwrap();
        for i in 0..5 {
            stdfs::write(temp.path().join(format!("{i}.md")), "source").unwrap();
        }
        let root = fs::VaultRoot::open(temp.path()).unwrap();
        let (first_read, started) = tokio::sync::oneshot::channel();
        let (release, wait) = std::sync::mpsc::channel();
        let (worker, stopped) = tokio::sync::oneshot::channel();
        let request = tokio::spawn(run_cold_build(move |check| {
            crate::test_helpers::EXACT_FILE_OPENS.set(0);
            let mut first_read = Some(first_read);
            let result = VaultIndex::build_sync(root, empty_exclude(), check, |_, _| {
                if let Some(first_read) = first_read.take() {
                    first_read.send(()).unwrap();
                    wait.recv().unwrap();
                }
                Ok(())
            });
            worker
                .send((result.is_err(), crate::test_helpers::EXACT_FILE_OPENS.get()))
                .unwrap();
            result.map(|_| ())
        }));
        started.await.unwrap();
        request.abort();
        assert!(request.await.unwrap_err().is_cancelled());
        release.send(()).unwrap();
        let outcome = tokio::time::timeout(std::time::Duration::from_secs(5), stopped)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(outcome, (true, 1));
    }

    #[cfg(unix)]
    #[test]
    fn an_incrementally_built_index_retains_its_original_root() {
        let temp = TempDir::new().unwrap();
        let original = temp.path().canonicalize().unwrap().join("vault");
        let moved = original.with_file_name("moved");
        stdfs::create_dir(&original).unwrap();
        let path = Path::new("note.md");
        stdfs::write(original.join(path), "#before").unwrap();
        let mut index = VaultIndex::empty();
        index.reindex_file(&original, path).unwrap();
        stdfs::rename(&original, &moved).unwrap();
        stdfs::create_dir(&original).unwrap();
        stdfs::write(original.join(path), "#replacement").unwrap();
        stdfs::write(moved.join(path), "#retained").unwrap();
        index.reindex_file(&original, path).unwrap();
        assert_eq!(index.get_note(path).unwrap().tags, ["retained"]);
    }

    #[test]
    fn watched_prefixes_keep_distinct_native_names_until_their_last_file_is_removed() {
        let mut paths = WatchPaths::default();
        let first = Path::new("Café/one.md");
        let second = Path::new("Café/two.png");
        let alternate = Path::new("Cafe\u{301}/three.md");
        let other = Path::new("Cafés/unrelated.md");
        for path in [first, second, alternate, other] {
            paths.insert(path);
        }
        let folder = Path::new("CAFÉ");
        assert_eq!(paths.matching(folder, false).count(), 2);
        assert_eq!(paths.matching(folder, true).count(), 5);
        paths.remove(first);
        assert_eq!(paths.matching(folder, false).count(), 2);
        paths.remove(second);
        assert_eq!(
            paths.matching(folder, false).collect::<Vec<_>>(),
            [alternate.parent().unwrap()]
        );
        paths.remove(alternate);
        assert_eq!(paths.matching(folder, true).count(), 0);
        assert_eq!(paths.matching(other, false).count(), 1);
    }

    #[cfg(unix)]
    #[test]
    fn exact_observed_name_cannot_redirect_after_directory_enumeration() {
        use std::os::unix::fs::symlink;
        let directory = TempDir::new().unwrap();
        let outside = TempDir::new().unwrap();
        let sentinel = outside.path().join("private.md");
        stdfs::write(&sentinel, "outside source").unwrap();
        let file = directory.path().join("visible.md");
        stdfs::write(&file, "inside source").unwrap();
        let root = fs::VaultRoot::open(directory.path()).unwrap();
        visit_files(
            &root,
            Path::new(""),
            || Ok(()),
            |path| {
                stdfs::remove_file(&file)?;
                assert!(fs::is_missing_entry(&root.open_exact(path).unwrap_err()));
                symlink(&sentinel, &file)?;
                assert!(read_indexed_note(&root, path).is_err());
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(stdfs::read_to_string(sentinel).unwrap(), "outside source");
    }

    #[cfg(unix)]
    #[test]
    fn cold_index_skips_attachments_if_their_open_directory_becomes_a_link() {
        use crate::test_helpers::INDEX_DIRECTORY_VISITS;
        use std::cell::Cell;
        let temp = TempDir::new().unwrap();
        let outside = TempDir::new().unwrap();
        let assets = temp.path().join("assets");
        stdfs::create_dir(&assets).unwrap();
        stdfs::write(assets.join("photo.bin"), b"inside").unwrap();
        stdfs::write(outside.path().join("photo.bin"), b"outside content").unwrap();
        stdfs::write(temp.path().join("kept.md"), "# Visible").unwrap();
        let root = fs::VaultRoot::open(temp.path()).unwrap();
        let visits = INDEX_DIRECTORY_VISITS.get();
        let switched = Cell::new(false);
        let check = || {
            // With one subdirectory, the second directory visit owns its iterator.
            if INDEX_DIRECTORY_VISITS.get() == visits + 2 && !switched.replace(true) {
                stdfs::rename(&assets, temp.path().join(".moved"))?;
                std::os::unix::fs::symlink(outside.path(), &assets)?;
            }
            Ok(())
        };
        let index =
            VaultIndex::build_sync(root, Arc::new(ExcludeSet::default()), &check, |_, _| Ok(()))
                .unwrap();
        assert!(switched.get());
        assert!(index.get_note(Path::new("kept.md")).is_some());
        assert!(index.attachments.is_empty());
    }

    fn setup_vault() -> TempDir {
        let dir = TempDir::new().unwrap();
        let root = dir.path();

        stdfs::write(
            root.join("daily.md"),
            "---\ntags: [journal]\n---\n# Daily Note\n\nToday I worked on #daily stuff.\n\nSee [[notes]] for more.\n",
        )
        .unwrap();

        stdfs::create_dir_all(root.join("notes")).unwrap();
        stdfs::write(
            root.join("notes/alpha.md"),
            "---\ntags: [rust, mcp]\n---\n# Alpha\n\nThis is about the #project.\n\nSee also [[beta]].\n",
        )
        .unwrap();

        stdfs::write(
            root.join("notes/beta.md"),
            "# Beta\n\nReferences back to [[alpha]].\n\nSome text ^block1\n",
        )
        .unwrap();

        stdfs::write(
            root.join("notes/gamma.md"),
            "# Gamma\n\nAn isolated note with no links.\n",
        )
        .unwrap();

        stdfs::create_dir_all(root.join("archive")).unwrap();
        stdfs::write(
            root.join("archive/old.md"),
            "# Old Note\n\nThis references [[alpha]] from the archive.\n",
        )
        .unwrap();

        stdfs::create_dir_all(root.join(".obsidian")).unwrap();
        stdfs::write(root.join(".obsidian/config.json"), r#"{"key":"val"}"#).unwrap();

        stdfs::write(root.join(".hidden"), "secret").unwrap();

        stdfs::write(root.join("image.png"), "fake-image-data").unwrap();

        dir
    }

    // ── build tests ─────────────────────────────────────────────────

    #[tokio::test]
    async fn build_indexes_all_notes() {
        let vault = setup_vault();
        let index = VaultIndex::build(vault.path(), empty_exclude())
            .await
            .unwrap();

        assert_eq!(index.notes.len(), 5);
        assert!(index.get_note(Path::new("daily.md")).is_some());
        assert!(index.get_note(Path::new("notes/alpha.md")).is_some());
        assert!(index.get_note(Path::new("notes/beta.md")).is_some());
        assert!(index.get_note(Path::new("notes/gamma.md")).is_some());
        assert!(index.get_note(Path::new("archive/old.md")).is_some());
    }

    #[tokio::test]
    async fn build_keeps_actual_unicode_relative_path() {
        let vault = tempfile::tempdir().unwrap();
        let composed = "02_База-знаний/Сущности/lic1c.md";
        let decomposed: String = composed.nfd().collect();
        let disk_path = PathBuf::from(&decomposed);
        stdfs::create_dir_all(vault.path().join(disk_path.parent().unwrap())).unwrap();
        stdfs::write(vault.path().join(&disk_path), "# License\n").unwrap();

        let index = VaultIndex::build(vault.path(), empty_exclude())
            .await
            .unwrap();

        assert!(index.get_note(&disk_path).is_some());
        assert!(index.get_note(Path::new(composed)).is_none());
    }

    #[tokio::test]
    async fn build_computes_correct_backlinks() {
        let vault = setup_vault();
        let index = VaultIndex::build(vault.path(), empty_exclude())
            .await
            .unwrap();

        let alpha_bl = index.backlinks_to(Path::new("notes/alpha.md"));
        let bl_paths: HashSet<PathBuf> = alpha_bl.iter().map(|n| n.path.clone()).collect();
        assert!(bl_paths.contains(&PathBuf::from("notes/beta.md")));
        assert!(bl_paths.contains(&PathBuf::from("archive/old.md")));
        assert_eq!(alpha_bl.len(), 2);

        let beta_bl = index.backlinks_to(Path::new("notes/beta.md"));
        assert_eq!(beta_bl.len(), 1);
        assert_eq!(beta_bl[0].path, PathBuf::from("notes/alpha.md"));

        assert!(index.backlinks_to(Path::new("notes/gamma.md")).is_empty());
    }

    #[tokio::test]
    async fn build_indexes_both_frontmatter_and_inline_tags() {
        let vault = setup_vault();
        let index = VaultIndex::build(vault.path(), empty_exclude())
            .await
            .unwrap();

        let alpha = index.get_note(Path::new("notes/alpha.md")).unwrap();
        assert!(alpha.tags.contains(&"rust".to_string()));
        assert!(alpha.tags.contains(&"mcp".to_string()));
        assert!(alpha.tags.contains(&"project".to_string()));

        let rust_notes = index.notes_with_tag("rust");
        assert_eq!(rust_notes.len(), 1);
        assert_eq!(rust_notes[0].path, PathBuf::from("notes/alpha.md"));

        let journal_notes = index.notes_with_tag("journal");
        assert_eq!(journal_notes.len(), 1);
        assert_eq!(journal_notes[0].path, PathBuf::from("daily.md"));
    }

    #[tokio::test]
    async fn build_detects_broken_links() {
        let vault = setup_vault();
        let index = VaultIndex::build(vault.path(), empty_exclude())
            .await
            .unwrap();

        let broken = index.broken_links();
        let broken_targets: Vec<&str> = broken.iter().map(|(_, l)| l.target.as_str()).collect();
        assert!(broken_targets.contains(&"notes"));
    }

    #[tokio::test]
    async fn build_detects_orphan_notes() {
        let vault = setup_vault();
        let index = VaultIndex::build(vault.path(), empty_exclude())
            .await
            .unwrap();

        let orphans = index.orphan_notes();
        let orphan_paths: HashSet<&PathBuf> = orphans.iter().map(|n| &n.path).collect();
        assert!(orphan_paths.contains(&PathBuf::from("notes/gamma.md")));
        assert!(!orphan_paths.contains(&PathBuf::from("notes/alpha.md")));
        assert!(!orphan_paths.contains(&PathBuf::from("notes/beta.md")));
        assert!(!orphan_paths.contains(&PathBuf::from("daily.md")));
    }

    #[tokio::test]
    async fn build_computes_stats() {
        let vault = setup_vault();
        let index = VaultIndex::build(vault.path(), empty_exclude())
            .await
            .unwrap();

        let stats = index.stats();
        assert_eq!(stats.total_notes, 5);
        assert_eq!(stats.total_files, 6); // 5 .md + 1 .png
        assert!(stats.total_tags >= 5); // journal, daily, rust, mcp, project
        assert_eq!(stats.total_links, 4); // daily->notes, alpha->beta, beta->alpha, old->alpha
        assert!(stats.vault_size_bytes > 0);
    }

    #[tokio::test]
    async fn build_skips_hidden_and_obsidian() {
        let vault = setup_vault();
        let index = VaultIndex::build(vault.path(), empty_exclude())
            .await
            .unwrap();

        for path in index.notes.keys() {
            let s = Path::new(path).display().to_string();
            assert!(!s.contains(".obsidian"), "indexed .obsidian: {s}");
            assert!(!s.starts_with('.'), "indexed hidden file: {s}");
        }
    }

    #[tokio::test]
    async fn outgoing_links_correct() {
        let vault = setup_vault();
        let index = VaultIndex::build(vault.path(), empty_exclude())
            .await
            .unwrap();

        let links = index.outgoing_links(Path::new("notes/alpha.md"));
        let targets: Vec<&str> = links.iter().map(|l| l.target.as_str()).collect();
        assert!(targets.contains(&"beta"));
    }

    #[tokio::test]
    async fn get_note_missing_returns_none() {
        let vault = setup_vault();
        let index = VaultIndex::build(vault.path(), empty_exclude())
            .await
            .unwrap();

        assert!(index.get_note(Path::new("nonexistent.md")).is_none());
    }

    #[tokio::test]
    async fn notes_with_nonexistent_tag_returns_empty() {
        let vault = setup_vault();
        let index = VaultIndex::build(vault.path(), empty_exclude())
            .await
            .unwrap();

        assert!(index.notes_with_tag("nonexistent").is_empty());
    }

    #[tokio::test]
    async fn note_title_is_file_stem() {
        let vault = setup_vault();
        let index = VaultIndex::build(vault.path(), empty_exclude())
            .await
            .unwrap();

        assert_eq!(
            index.get_note(Path::new("notes/alpha.md")).unwrap().title,
            "alpha"
        );
        assert_eq!(
            index.get_note(Path::new("daily.md")).unwrap().title,
            "daily"
        );
    }

    #[tokio::test]
    async fn block_refs_indexed() {
        let vault = setup_vault();
        let index = VaultIndex::build(vault.path(), empty_exclude())
            .await
            .unwrap();

        let beta = index.get_note(Path::new("notes/beta.md")).unwrap();
        assert!(beta.block_refs.contains(&"block1".to_string()));
    }

    // ── mutation tests ──────────────────────────────────────────────

    #[tokio::test]
    async fn reindex_file_updates_tags_and_links() {
        let vault = setup_vault();
        let mut index = VaultIndex::build(vault.path(), empty_exclude())
            .await
            .unwrap();

        stdfs::write(
            vault.path().join("notes/gamma.md"),
            "# Gamma\n\nNow with #newtag and [[alpha]].\n",
        )
        .unwrap();

        index
            .reindex_file(vault.path(), Path::new("notes/gamma.md"))
            .unwrap();

        let gamma = index.get_note(Path::new("notes/gamma.md")).unwrap();
        assert!(gamma.tags.contains(&"newtag".to_string()));
        assert!(gamma.links.iter().any(|l| l.target == "alpha"));

        assert_eq!(index.notes_with_tag("newtag").len(), 1);

        let alpha_bl = index.backlinks_to(Path::new("notes/alpha.md"));
        let bl_paths: HashSet<PathBuf> = alpha_bl.iter().map(|n| n.path.clone()).collect();
        assert!(bl_paths.contains(&PathBuf::from("notes/gamma.md")));

        let orphans = index.orphan_notes();
        let orphan_paths: HashSet<&PathBuf> = orphans.iter().map(|n| &n.path).collect();
        assert!(!orphan_paths.contains(&PathBuf::from("notes/gamma.md")));
    }

    #[tokio::test]
    async fn reindex_file_handles_new_file() {
        let vault = setup_vault();
        let mut index = VaultIndex::build(vault.path(), empty_exclude())
            .await
            .unwrap();
        let old_count = index.notes.len();

        stdfs::write(
            vault.path().join("brand_new.md"),
            "# Brand New\n\nA fresh note with #fresh tag.\n",
        )
        .unwrap();

        index
            .reindex_file(vault.path(), Path::new("brand_new.md"))
            .unwrap();

        assert_eq!(index.notes.len(), old_count + 1);
        assert!(index.get_note(Path::new("brand_new.md")).is_some());
        assert_eq!(index.notes_with_tag("fresh").len(), 1);
    }

    #[tokio::test]
    async fn reindex_removes_old_tag_contributions() {
        let vault = setup_vault();
        let mut index = VaultIndex::build(vault.path(), empty_exclude())
            .await
            .unwrap();

        assert_eq!(index.notes_with_tag("rust").len(), 1);

        stdfs::write(
            vault.path().join("notes/alpha.md"),
            "# Alpha\n\nRemoved all tags and links.\n",
        )
        .unwrap();

        index
            .reindex_file(vault.path(), Path::new("notes/alpha.md"))
            .unwrap();

        assert!(index.notes_with_tag("rust").is_empty());
        assert!(index.notes_with_tag("mcp").is_empty());
        assert!(index.notes_with_tag("project").is_empty());
    }

    #[tokio::test]
    async fn remove_file_cleans_up_everything() {
        let vault = setup_vault();
        let mut index = VaultIndex::build(vault.path(), empty_exclude())
            .await
            .unwrap();

        assert!(index.get_note(Path::new("notes/alpha.md")).is_some());
        assert!(!index.notes_with_tag("rust").is_empty());

        index.remove_file(Path::new("notes/alpha.md"));

        assert!(index.get_note(Path::new("notes/alpha.md")).is_none());
        assert!(index.notes_with_tag("rust").is_empty());

        let beta_bl = index.backlinks_to(Path::new("notes/beta.md"));
        assert!(
            !beta_bl
                .iter()
                .any(|n| n.path == Path::new("notes/alpha.md")),
            "alpha should no longer appear as a backlink source"
        );
    }

    #[tokio::test]
    async fn remove_file_updates_stats() {
        let vault = setup_vault();
        let mut index = VaultIndex::build(vault.path(), empty_exclude())
            .await
            .unwrap();
        let old_notes = index.stats().total_notes;

        index.remove_file(Path::new("notes/gamma.md"));

        assert_eq!(index.stats().total_notes, old_notes - 1);
    }

    #[tokio::test]
    async fn rename_file_updates_index() {
        let vault = setup_vault();
        let mut index = VaultIndex::build(vault.path(), empty_exclude())
            .await
            .unwrap();

        stdfs::rename(
            vault.path().join("notes/gamma.md"),
            vault.path().join("notes/delta.md"),
        )
        .unwrap();

        index
            .rename_file(
                vault.path(),
                Path::new("notes/gamma.md"),
                Path::new("notes/delta.md"),
            )
            .unwrap();

        assert!(index.get_note(Path::new("notes/gamma.md")).is_none());
        let delta = index.get_note(Path::new("notes/delta.md")).unwrap();
        assert_eq!(delta.title, "delta");
    }

    // ── search tests ────────────────────────────────────────────────

    #[tokio::test]
    async fn search_text_finds_matches() {
        let vault = setup_vault();
        let index = VaultIndex::build(vault.path(), empty_exclude())
            .await
            .unwrap();

        let results = index.search_text(vault.path(), "alpha", 20).unwrap();
        assert!(!results.is_empty());

        let result_paths: HashSet<PathBuf> = results.iter().map(|r| r.path.clone()).collect();
        assert!(result_paths.contains(&PathBuf::from("notes/alpha.md")));
    }

    #[tokio::test]
    async fn search_text_case_insensitive() {
        let vault = setup_vault();
        let index = VaultIndex::build(vault.path(), empty_exclude())
            .await
            .unwrap();

        let lower = index.search_text(vault.path(), "gamma", 10).unwrap();
        let upper = index.search_text(vault.path(), "GAMMA", 10).unwrap();

        assert_eq!(lower.len(), upper.len());
        assert!(!lower.is_empty());
    }

    #[tokio::test]
    async fn search_text_empty_query_returns_empty() {
        let vault = setup_vault();
        let index = VaultIndex::build(vault.path(), empty_exclude())
            .await
            .unwrap();

        assert!(index.search_text(vault.path(), "", 10).unwrap().is_empty());
    }

    #[tokio::test]
    async fn search_text_context_offsets_are_correct() {
        let vault = setup_vault();
        let index = VaultIndex::build(vault.path(), empty_exclude())
            .await
            .unwrap();

        let results = index.search_text(vault.path(), "isolated", 50).unwrap();
        assert!(!results.is_empty());

        for result in &results {
            for m in &result.matches {
                let extracted: String = m
                    .context
                    .chars()
                    .skip(m.match_start)
                    .take(m.match_end - m.match_start)
                    .collect();
                assert_eq!(extracted.to_lowercase(), "isolated");
            }
        }
    }

    #[tokio::test]
    async fn search_regex_works() {
        let vault = setup_vault();
        let index = VaultIndex::build(vault.path(), empty_exclude())
            .await
            .unwrap();

        let results = index
            .search_regex(vault.path(), r"\[\[alpha\]\]", 10, 0)
            .unwrap();
        assert!(!results.is_empty());

        let paths: HashSet<PathBuf> = results.iter().map(|r| r.path.clone()).collect();
        assert!(paths.contains(&PathBuf::from("notes/beta.md")));
        assert!(paths.contains(&PathBuf::from("archive/old.md")));
    }

    #[tokio::test]
    async fn search_regex_invalid_pattern_returns_error() {
        let vault = setup_vault();
        let index = VaultIndex::build(vault.path(), empty_exclude())
            .await
            .unwrap();

        let result = index.search_regex(vault.path(), "[invalid", 10, 0);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            VaultError::InvalidRegex { .. }
        ));
    }

    #[tokio::test]
    async fn search_frontmatter_exact_match() {
        let vault = setup_vault();
        let index = VaultIndex::build(vault.path(), empty_exclude())
            .await
            .unwrap();

        let results = index.search_frontmatter("tags", &serde_json::json!(["journal"]));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].path, PathBuf::from("daily.md"));
    }

    #[tokio::test]
    async fn search_frontmatter_array_contains() {
        let vault = setup_vault();
        let index = VaultIndex::build(vault.path(), empty_exclude())
            .await
            .unwrap();

        let results = index.search_frontmatter("tags", &serde_json::json!("rust"));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].path, PathBuf::from("notes/alpha.md"));
    }

    #[tokio::test]
    async fn search_frontmatter_no_match() {
        let vault = setup_vault();
        let index = VaultIndex::build(vault.path(), empty_exclude())
            .await
            .unwrap();

        assert!(
            index
                .search_frontmatter("tags", &serde_json::json!("nonexistent"))
                .is_empty()
        );
    }

    #[tokio::test]
    async fn search_frontmatter_missing_field() {
        let vault = setup_vault();
        let index = VaultIndex::build(vault.path(), empty_exclude())
            .await
            .unwrap();

        assert!(
            index
                .search_frontmatter("no_such_field", &serde_json::json!("value"))
                .is_empty()
        );
    }

    // ── tag prefix tests ────────────────────────────────────────────

    #[tokio::test]
    async fn notes_with_tag_prefix_includes_nested() {
        let dir = TempDir::new().unwrap();
        stdfs::write(dir.path().join("a.md"), "# A\n\n#inbox\n").unwrap();
        stdfs::write(dir.path().join("b.md"), "# B\n\n#inbox/read\n").unwrap();
        stdfs::write(dir.path().join("c.md"), "# C\n\n#inbox/todo\n").unwrap();
        stdfs::write(dir.path().join("d.md"), "---\ntags: [other]\n---\n# D\n").unwrap();

        let index = VaultIndex::build(dir.path(), empty_exclude())
            .await
            .unwrap();
        let results = index.notes_with_tag_prefix("inbox");
        assert_eq!(results.len(), 3);
    }

    #[tokio::test]
    async fn notes_with_tag_prefix_no_false_prefix_match() {
        let dir = TempDir::new().unwrap();
        stdfs::write(dir.path().join("a.md"), "# A\n\n#inbox\n").unwrap();
        stdfs::write(dir.path().join("b.md"), "# B\n\n#inboxes\n").unwrap();

        let index = VaultIndex::build(dir.path(), empty_exclude())
            .await
            .unwrap();
        let results = index.notes_with_tag_prefix("inbox");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].path, PathBuf::from("a.md"));
    }

    #[tokio::test]
    async fn notes_with_tag_prefix_deduplicates() {
        let dir = TempDir::new().unwrap();
        stdfs::write(dir.path().join("a.md"), "# A\n\n#inbox #inbox/read\n").unwrap();

        let index = VaultIndex::build(dir.path(), empty_exclude())
            .await
            .unwrap();
        let results = index.notes_with_tag_prefix("inbox");
        assert_eq!(results.len(), 1);
    }

    // ── frontmatter exists / contains tests ─────────────────────────

    #[tokio::test]
    async fn search_frontmatter_exists_finds_field() {
        let vault = setup_vault();
        let index = VaultIndex::build(vault.path(), empty_exclude())
            .await
            .unwrap();

        let results = index.search_frontmatter_exists("tags");
        assert_eq!(results.len(), 2); // daily.md + notes/alpha.md
    }

    #[tokio::test]
    async fn search_frontmatter_exists_missing_field_empty() {
        let vault = setup_vault();
        let index = VaultIndex::build(vault.path(), empty_exclude())
            .await
            .unwrap();

        assert!(index.search_frontmatter_exists("nonexistent").is_empty());
    }

    #[tokio::test]
    async fn search_frontmatter_contains_string_substring() {
        let dir = TempDir::new().unwrap();
        stdfs::write(
            dir.path().join("a.md"),
            "---\nstatus: in progress\n---\n# A\n",
        )
        .unwrap();
        stdfs::write(dir.path().join("b.md"), "---\nstatus: done\n---\n# B\n").unwrap();

        let index = VaultIndex::build(dir.path(), empty_exclude())
            .await
            .unwrap();
        let results = index.search_frontmatter_contains("status", &serde_json::json!("progress"));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].path, PathBuf::from("a.md"));
    }

    #[tokio::test]
    async fn search_frontmatter_contains_array_element() {
        let vault = setup_vault();
        let index = VaultIndex::build(vault.path(), empty_exclude())
            .await
            .unwrap();

        let results = index.search_frontmatter_contains("tags", &serde_json::json!("rust"));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].path, PathBuf::from("notes/alpha.md"));
    }

    // ── exclusion filtering tests ────────────────────────────────────

    #[tokio::test]
    async fn build_excludes_matching_notes() {
        let dir = TempDir::new().unwrap();
        stdfs::create_dir_all(dir.path().join("Archive")).unwrap();
        stdfs::write(dir.path().join("Archive/old.md"), "# Old\n").unwrap();
        stdfs::create_dir_all(dir.path().join("Active")).unwrap();
        stdfs::write(dir.path().join("Active/note.md"), "# Note\n").unwrap();

        let exclude = Arc::new(ExcludeSet::build(vec!["Archive/".into()]).unwrap());
        let index = VaultIndex::build(dir.path(), exclude).await.unwrap();

        assert!(index.get_note(Path::new("Active/note.md")).is_some());
        assert!(index.get_note(Path::new("Archive/old.md")).is_none());
        assert_eq!(index.excluded_notes(), 1);
        assert_eq!(index.stats().total_notes, 1);
    }

    #[tokio::test]
    async fn build_excludes_deeply_nested() {
        let dir = TempDir::new().unwrap();
        stdfs::create_dir_all(dir.path().join("Archive/sub")).unwrap();
        stdfs::write(dir.path().join("Archive/sub/deep.md"), "# Deep\n").unwrap();
        stdfs::write(dir.path().join("top.md"), "# Top\n").unwrap();

        let exclude = Arc::new(ExcludeSet::build(vec!["Archive/".into()]).unwrap());
        let index = VaultIndex::build(dir.path(), exclude).await.unwrap();

        assert!(index.get_note(Path::new("Archive/sub/deep.md")).is_none());
        assert!(index.get_note(Path::new("top.md")).is_some());
        assert_eq!(index.excluded_notes(), 1);
    }

    #[tokio::test]
    async fn build_empty_exclude_includes_all() {
        let dir = TempDir::new().unwrap();
        stdfs::create_dir_all(dir.path().join("Archive")).unwrap();
        stdfs::write(dir.path().join("Archive/old.md"), "# Old\n").unwrap();
        stdfs::write(dir.path().join("note.md"), "# Note\n").unwrap();

        let index = VaultIndex::build(dir.path(), empty_exclude())
            .await
            .unwrap();

        assert_eq!(index.notes().len(), 2);
        assert_eq!(index.excluded_notes(), 0);
    }

    #[tokio::test]
    async fn build_exclude_does_not_affect_non_md() {
        let dir = TempDir::new().unwrap();
        stdfs::create_dir_all(dir.path().join("Archive")).unwrap();
        stdfs::write(dir.path().join("Archive/note.md"), "# Note\n").unwrap();
        stdfs::write(dir.path().join("Archive/data.json"), r#"{"k":"v"}"#).unwrap();
        stdfs::write(dir.path().join("keep.md"), "# Keep\n").unwrap();

        let exclude = Arc::new(ExcludeSet::build(vec!["Archive/".into()]).unwrap());
        let index = VaultIndex::build(dir.path(), exclude).await.unwrap();

        assert_eq!(index.stats().total_notes, 1);
        assert_eq!(index.excluded_notes(), 1);
        // Non-md file still counted: 1 md (keep.md) + 1 non-md (data.json)
        assert_eq!(index.stats().total_files, 2);
    }
}
