//! Vault layer — pure filesystem operations on an Obsidian vault.
//!
//! Knows nothing about MCP; provides the data model and I/O primitives
//! that tool handlers delegate to.

mod data;
pub mod exclude;
pub mod frontmatter;
pub mod fs;
pub mod index;
pub mod parser;
pub mod patch;
pub mod path;
pub mod periodic;
pub mod search_utils;
pub mod source;
pub mod tantivy_index;
pub mod watcher;
pub mod wikilink;

#[cfg(has_embeddings)]
pub(crate) mod embedding_chunks;
#[cfg(has_embeddings)]
pub(crate) mod embedding_input;
#[cfg(has_embeddings)]
pub mod embedding_runtime;
#[cfg(has_embeddings)]
pub mod embeddings;
#[cfg(has_embeddings)]
pub(crate) mod semantic_results;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};

use chrono::{Local, NaiveDate};
use notify_debouncer_mini::Debouncer;

use crate::config::Config;
use crate::error::{VaultError, VaultResult};
use crate::models::{
    DocumentMap, FileStat, NoteMetadata, NotePeriod, PatchRequest, SearchField, SearchMatch,
    SearchResult, VaultStats, WikiLink,
};

use self::exclude::ExcludeSet;
use self::index::VaultIndex;
use self::tantivy_index::TantivyIndex;
use self::watcher::ChangeWatcher;

#[cfg(has_embeddings)]
type EmbeddingLoaderFuture = std::pin::Pin<
    Box<
        dyn std::future::Future<Output = VaultResult<Arc<dyn embeddings::Embedder>>>
            + Send
            + 'static,
    >,
>;
#[cfg(not(has_embeddings))]
type EmbeddingLoaderFuture = ();

/// Internal shared state wrapped in `Arc` for cheap cloning.
struct VaultInner {
    filesystem: fs::VaultRoot,
    watch_enabled: bool,
    mcp_home: PathBuf,
    mcp_data: PathBuf,
    exclude: Arc<ExcludeSet>,
    index: Arc<RwLock<VaultIndex>>,
    mutations: Mutex<()>,
    tantivy: Option<Arc<TantivyIndex>>,
    #[cfg(has_embeddings)]
    embedding_runtime: Option<embedding_runtime::EmbeddingRuntime>,
    /// Kept alive to sustain filesystem watching; never accessed after construction.
    /// Wrapped in `Mutex` to guarantee `Sync` (`Debouncer` contains a `mpsc::Sender`
    /// which is `Send` but not `Sync`).
    _watcher: Mutex<Option<Debouncer<ChangeWatcher>>>,
}

/// High-level facade over the vault filesystem, index, and watcher.
///
/// `Vault` is `Clone + Send + Sync` — cloning increments an internal `Arc`.
/// Mutations through shared handles are serialized from reading through indexing.
/// Metadata queries acquire a shared index lock. External editors are independent.
#[derive(Clone)]
pub struct Vault {
    inner: Arc<VaultInner>,
}

/// Indexed fields used by bulk vault listings.
pub(crate) struct NoteListMetadata {
    pub(crate) title: String,
    pub(crate) tags: Vec<String>,
    pub(crate) stat: FileStat,
}

impl Vault {
    /// Open a vault: validate the path, build the index, and optionally start the watcher.
    pub async fn open(config: &Config) -> VaultResult<Self> {
        Self::open_inner(config, None, true).await
    }

    /// Open a vault with a dormant local embedding coordinator.
    /// Mutations remain queued until `activate_embeddings` selects local ownership.
    pub async fn open_deferred_embeddings(config: &Config) -> VaultResult<Self> {
        Self::open_inner(config, None, false).await
    }

    #[cfg(all(test, has_embeddings))]
    async fn open_with_embedding_loader(
        config: &Config,
        embedding_loader: Option<EmbeddingLoaderFuture>,
    ) -> VaultResult<Self> {
        Self::open_inner(config, embedding_loader, true).await
    }

    async fn open_inner(
        config: &Config,
        _embedding_loader: Option<EmbeddingLoaderFuture>,
        _embeddings_active: bool,
    ) -> VaultResult<Self> {
        let filesystem = fs::VaultRoot::open(&config.vault_path)?;
        let root = filesystem.path().to_path_buf();
        let (data, exclude) = data::initialize(
            &filesystem,
            config.mcp_data_dir.as_deref(),
            &config.exclude_patterns,
        )?;
        let mcp_home = root.join(".obsidian-mcp");
        let mcp_data = data.path();
        let exclude = Arc::new(exclude);
        if !exclude.is_empty() {
            tracing::info!(rules = exclude.patterns().len(), "path exclusion active");
        }

        let pending_watcher = config
            .watch
            .then(|| watcher::PendingWatcher::register(&root))
            .transpose()?;
        let (vi, tantivy) = if config.tantivy {
            let (vi, tv) =
                VaultIndex::build_ranked(filesystem.clone(), Arc::clone(&exclude)).await?;
            tracing::info!(notes = vi.notes().len(), "tantivy BM25 index built");
            (vi, Some(Arc::new(tv)))
        } else {
            (
                VaultIndex::build_rooted(filesystem.clone(), Arc::clone(&exclude)).await?,
                None,
            )
        };

        let index = Arc::new(RwLock::new(vi));

        #[cfg(has_embeddings)]
        let embedding_runtime = if config.embeddings {
            let batch_size =
                crate::config::embedding_batch_size_from_env().map_err(VaultError::Embedding)?;
            let model_name = config.embeddings_model.clone();
            let provider = config.embedding_provider;
            let relative = data
                .directory
                .join("embeddings/scopes")
                .join(exclude.scope_id(&root))
                .join("embeddings.bin");
            let cache = embeddings::CacheFile::new(data.root, relative);
            let loader = _embedding_loader.unwrap_or_else(|| {
                Box::pin(async move {
                    let model = embeddings::EmbeddingModel::load(&model_name, provider).await?;
                    Ok(Arc::new(model) as Arc<dyn embeddings::Embedder>)
                })
            });
            Some(
                embedding_runtime::EmbeddingRuntime::spawn_with_scoped_cache(
                    filesystem.clone(),
                    Arc::clone(&index),
                    cache,
                    batch_size,
                    _embeddings_active,
                    loader,
                ),
            )
        } else {
            None
        };

        let watcher_handle = pending_watcher.map(|watcher| {
            watcher.start(
                Arc::clone(&index),
                tantivy.clone(),
                #[cfg(has_embeddings)]
                embedding_runtime.clone(),
                Arc::clone(&exclude),
            )
        });

        Ok(Self {
            inner: Arc::new(VaultInner {
                filesystem,
                watch_enabled: config.watch,
                mcp_home,
                mcp_data,
                exclude,
                index,
                mutations: Mutex::new(()),
                tantivy,
                #[cfg(has_embeddings)]
                embedding_runtime,
                _watcher: Mutex::new(watcher_handle),
            }),
        })
    }

    /// Activate local embeddings once without replacing the watcher submission handle.
    pub fn activate_embeddings(&self) -> bool {
        #[cfg(has_embeddings)]
        {
            self.inner
                .embedding_runtime
                .as_ref()
                .is_some_and(embedding_runtime::EmbeddingRuntime::activate)
        }
        #[cfg(not(has_embeddings))]
        {
            false
        }
    }

    /// Whether the local coordinator has been activated, independently of readiness.
    pub fn embeddings_active(&self) -> bool {
        #[cfg(has_embeddings)]
        {
            self.inner
                .embedding_runtime
                .as_ref()
                .is_some_and(embedding_runtime::EmbeddingRuntime::is_active)
        }
        #[cfg(not(has_embeddings))]
        {
            false
        }
    }

    /// Vault root path (canonicalized).
    pub fn root(&self) -> &Path {
        self.inner.filesystem.path()
    }

    #[cfg(has_embeddings)]
    pub(crate) fn filesystem(&self) -> &fs::VaultRoot {
        &self.inner.filesystem
    }

    pub fn watch_enabled(&self) -> bool {
        self.inner.watch_enabled
    }

    /// Always `{vault_root}/.obsidian-mcp`. Auto-created on startup.
    pub fn mcp_home(&self) -> &Path {
        &self.inner.mcp_home
    }

    /// Resolved data location. Equals `mcp_home` unless `OBSIDIAN_MCP_DATA` is set.
    pub fn mcp_data(&self) -> &Path {
        &self.inner.mcp_data
    }

    /// Active path exclusion set.
    pub fn exclude(&self) -> &ExcludeSet {
        &self.inner.exclude
    }

    /// Access the Tantivy BM25 index (if enabled via `Config::tantivy`).
    pub fn tantivy(&self) -> Option<&TantivyIndex> {
        self.inner.tantivy.as_deref()
    }

    // ── fs delegation ──────────────────────────────────────────────────

    pub fn list_files(
        &self,
        dir: &Path,
        recursive: bool,
        glob: Option<&str>,
    ) -> VaultResult<Vec<PathBuf>> {
        self.inner.filesystem.list_files(dir, recursive, glob)
    }

    pub fn read_note(&self, path: &Path) -> VaultResult<String> {
        self.inner.filesystem.read_file(path)
    }

    /// Observe one bounded raw note revision for source-bound context reads.
    pub fn read_source_snapshot(&self, path: &Path) -> VaultResult<source::SourceSnapshot> {
        Ok(source::SourceSnapshot::new(
            self.inner
                .filesystem
                .read_file_bounded(path, source::MAX_SOURCE_BYTES)?,
        ))
    }

    pub fn file_stat(&self, path: &Path) -> VaultResult<FileStat> {
        self.inner.filesystem.file_stat(path)
    }

    pub fn write_note(&self, path: &Path, content: &str) -> VaultResult<()> {
        let _mutation = self.lock_mutations()?;
        frontmatter::parse_frontmatter(content)?;
        let actual_path = self.inner.filesystem.write_file(path, content)?;
        self.reindex(&actual_path)?;
        Ok(())
    }

    pub fn append_note(&self, path: &Path, content: &str) -> VaultResult<()> {
        let _mutation = self.lock_mutations()?;
        let combined = match self.inner.filesystem.read_file(path) {
            Ok(existing) => existing + content,
            Err(VaultError::NoteNotFound(_)) => content.to_owned(),
            Err(error) => return Err(error),
        };
        frontmatter::parse_frontmatter(&combined)?;
        let actual_path = self.inner.filesystem.append_file(path, content)?;
        self.reindex(&actual_path)?;
        Ok(())
    }

    /// Create a new note. Returns `AlreadyExists` if the path is occupied.
    /// Optionally prepends YAML frontmatter before the body content.
    pub fn create_note(
        &self,
        path: &Path,
        content: &str,
        frontmatter: Option<&serde_json::Value>,
    ) -> VaultResult<()> {
        let _mutation = self.lock_mutations()?;
        if let Some(value) = frontmatter
            && !value.is_object()
        {
            return Err(VaultError::InvalidFrontmatter {
                path: path.to_path_buf(),
                actual: frontmatter::json_type_name(value),
            });
        }
        let full_content = frontmatter::rebuild_content(frontmatter, content);
        frontmatter::parse_frontmatter(&full_content)?;
        let actual_path = self.inner.filesystem.create_file(path, &full_content)?;
        self.reindex(&actual_path)?;
        Ok(())
    }

    /// Prepend content after frontmatter (or at the start if none exists).
    pub fn prepend_note(&self, path: &Path, content: &str) -> VaultResult<()> {
        let _mutation = self.lock_mutations()?;
        let existing = self.inner.filesystem.read_file(path)?;
        let new_content = match frontmatter::extract_raw_frontmatter(&existing) {
            Some((_, body_start)) => {
                let mut result = String::with_capacity(existing.len() + content.len());
                result.push_str(&existing[..body_start]);
                result.push_str(content);
                result.push_str(&existing[body_start..]);
                result
            }
            None => format!("{content}{existing}"),
        };
        frontmatter::parse_frontmatter(&new_content)?;
        let actual_path = self.inner.filesystem.write_file(path, &new_content)?;
        self.reindex(&actual_path)?;
        Ok(())
    }

    pub fn delete_note(&self, path: &Path) -> VaultResult<()> {
        let _mutation = self.lock_mutations()?;
        let actual_path = self.inner.filesystem.delete_file(path)?;
        self.reindex(&actual_path)
    }

    pub fn move_note(&self, from: &Path, to: &Path) -> VaultResult<PathBuf> {
        let _mutation = self.lock_mutations()?;
        let move_result = self.inner.filesystem.move_file(from, to)?;
        let old_result = self.reindex(&move_result.from);
        let new_result = self.reindex(&move_result.to);
        old_result.and(new_result)?;
        Ok(move_result.to)
    }

    // ── patch delegation ───────────────────────────────────────────────

    pub fn patch_note(&self, path: &Path, request: &PatchRequest) -> VaultResult<()> {
        let _mutation = self.lock_mutations()?;
        let content = self.inner.filesystem.read_file(path)?;
        let patched = patch::apply_patch(&content, request, path)?;
        frontmatter::parse_frontmatter(&patched)?;
        let actual_path = self.inner.filesystem.write_file(path, &patched)?;
        self.reindex(&actual_path)?;
        Ok(())
    }

    // ── frontmatter delegation ─────────────────────────────────────────

    pub fn get_frontmatter(&self, path: &Path) -> VaultResult<Option<serde_json::Value>> {
        let content = self.inner.filesystem.read_file(path)?;
        frontmatter::parse_frontmatter(&content)
    }

    pub fn set_frontmatter_field(
        &self,
        path: &Path,
        key: &str,
        value: serde_json::Value,
    ) -> VaultResult<()> {
        let _mutation = self.lock_mutations()?;
        let content = self.inner.filesystem.read_file(path)?;
        let updated = frontmatter::set_frontmatter_field(&content, key, value)?;
        frontmatter::parse_frontmatter(&updated)?;
        let actual_path = self.inner.filesystem.write_file(path, &updated)?;
        self.reindex(&actual_path)?;
        Ok(())
    }

    pub fn remove_frontmatter_field(&self, path: &Path, key: &str) -> VaultResult<()> {
        let _mutation = self.lock_mutations()?;
        let content = self.inner.filesystem.read_file(path)?;
        let updated = frontmatter::remove_frontmatter_field(&content, key)?;
        frontmatter::parse_frontmatter(&updated)?;
        let actual_path = self.inner.filesystem.write_file(path, &updated)?;
        self.reindex(&actual_path)?;
        Ok(())
    }

    // ── index delegation (read-lock) ───────────────────────────────────

    pub(crate) fn indexed_paths(&self) -> Vec<String> {
        self.read_index()
            .notes()
            .keys()
            .map(|path| path.to_string_lossy().into_owned())
            .collect()
    }

    pub fn get_note_metadata(&self, path: &Path) -> VaultResult<NoteMetadata> {
        let actual_path = self.canonical_existing_relative_path(path)?;
        self.read_index()
            .get_note(&actual_path)
            .cloned()
            .ok_or(VaultError::NoteNotFound(actual_path))
    }

    /// Look up metadata for exact index paths under a single read lock.
    ///
    /// This intentionally skips filesystem path resolution and is only for paths
    /// already returned by vault APIs such as [`Vault::list_files`]. Raw caller
    /// paths must continue to use [`Vault::get_note_metadata`].
    pub(crate) fn get_indexed_note_metadata_batch<'a>(
        &self,
        paths: impl IntoIterator<Item = &'a Path>,
    ) -> Vec<Option<NoteListMetadata>> {
        let index = self.read_index();
        paths
            .into_iter()
            .map(|path| {
                index.get_note(path).map(|metadata| NoteListMetadata {
                    title: metadata.title.clone(),
                    tags: metadata.tags.clone(),
                    stat: metadata.stat.clone(),
                })
            })
            .collect()
    }

    pub fn get_document_map(&self, path: &Path) -> VaultResult<DocumentMap> {
        let content = self.inner.filesystem.read_file(path)?;
        Ok(parser::build_document_map(&content))
    }

    pub fn search_text(&self, query: &str, context_len: usize) -> VaultResult<Vec<SearchResult>> {
        match &self.inner.tantivy {
            Some(tv) => self.tantivy_search_with_context(tv, query, context_len, 200, false, None),
            None => self.read_index().search_text_limited(
                &self.inner.filesystem,
                query,
                context_len,
                None,
            ),
        }
    }

    /// Full-text search with additional Tantivy options (fuzzy, field filter).
    ///
    /// Falls back to `VaultIndex::search_text` (ignoring fuzzy/fields) when
    /// Tantivy is disabled.
    pub fn search_text_with_options(
        &self,
        query: &str,
        context_len: usize,
        max_results: usize,
        fuzzy: bool,
        fields: Option<&[SearchField]>,
    ) -> VaultResult<Vec<SearchResult>> {
        match &self.inner.tantivy {
            Some(tv) => {
                self.tantivy_search_with_context(tv, query, context_len, max_results, fuzzy, fields)
            }
            None => self.read_index().search_text_limited(
                &self.inner.filesystem,
                query,
                context_len,
                Some(max_results),
            ),
        }
    }

    pub(crate) fn search_text_with_limit(
        &self,
        query: &str,
        context_len: usize,
        max_results: usize,
        fuzzy: bool,
        fields: Option<&[SearchField]>,
    ) -> VaultResult<Vec<SearchResult>> {
        // The default MCP query validates its syntax even when the caller requests no hits.
        if max_results == 0
            && !fuzzy
            && fields.is_none()
            && let Some(tv) = &self.inner.tantivy
        {
            tv.parse_default_query(query)?;
            return Ok(Vec::new());
        }
        self.search_text_with_options(query, context_len, max_results, fuzzy, fields)
    }

    pub(crate) fn search_regex_with_limit(
        &self,
        pattern: &str,
        context_len: usize,
        max_results: usize,
    ) -> VaultResult<Vec<SearchResult>> {
        self.read_index().search_regex_limited(
            &self.inner.filesystem,
            pattern,
            context_len,
            Some(max_results),
        )
    }

    pub fn search_regex(
        &self,
        pattern: &str,
        context_len: usize,
    ) -> VaultResult<Vec<SearchResult>> {
        self.read_index()
            .search_regex_limited(&self.inner.filesystem, pattern, context_len, None)
    }

    /// Semantic search via embedding cosine similarity (Layer 2).
    ///
    /// Embeds the query, then performs brute-force cosine similarity against
    /// the embedding store. Returns `(path, score)` pairs sorted by descending
    /// similarity.
    #[cfg(has_embeddings)]
    pub fn search_semantic(&self, query: &str, top_k: usize) -> VaultResult<Vec<(PathBuf, f32)>> {
        Ok(self
            .search_semantic_ranked(query, top_k)?
            .into_iter()
            .map(|hit| (hit.path, hit.score))
            .collect())
    }

    #[cfg(has_embeddings)]
    pub(crate) fn search_semantic_ranked(
        &self,
        query: &str,
        top_k: usize,
    ) -> VaultResult<Vec<semantic_results::RankedNote>> {
        if query.is_empty() || top_k == 0 {
            return Ok(Vec::new());
        }
        let runtime = self.inner.embedding_runtime.as_ref().ok_or_else(|| {
            VaultError::Embedding("embeddings not enabled (OBSIDIAN_EMBEDDINGS=false)".into())
        })?;
        let snapshot = runtime.query_snapshot()?;
        let current_paths = self
            .read_index()
            .notes()
            .keys()
            .cloned()
            .collect::<std::collections::HashSet<_>>();
        let entries = snapshot.capture_entries(&current_paths);
        if entries.is_empty() {
            snapshot.validate_query(query)?;
            return Ok(Vec::new());
        }
        let query_vector = snapshot.embed_query(query)?;
        semantic_results::rank_semantic(entries, &query_vector, top_k)
    }

    /// Returns `true` if embeddings are available for semantic search.
    #[cfg(has_embeddings)]
    pub fn has_embeddings(&self) -> bool {
        self.inner
            .embedding_runtime
            .as_ref()
            .is_some_and(|runtime| runtime.status().queryable)
    }

    #[cfg(has_embeddings)]
    pub(crate) fn embeddings_configured(&self) -> bool {
        self.inner.embedding_runtime.is_some()
    }

    /// Returns the error message from a failed embedding model load, if any.
    pub fn embedding_load_error(&self) -> Option<&str> {
        #[cfg(has_embeddings)]
        {
            self.inner
                .embedding_runtime
                .as_ref()
                .and_then(embedding_runtime::EmbeddingRuntime::first_load_error)
        }
        #[cfg(not(has_embeddings))]
        {
            None
        }
    }

    /// Current background embedding lifecycle status, when embeddings are enabled.
    #[cfg(has_embeddings)]
    pub fn embedding_status(&self) -> Option<embedding_runtime::EmbeddingRuntimeStatus> {
        self.inner
            .embedding_runtime
            .as_ref()
            .map(embedding_runtime::EmbeddingRuntime::status)
    }

    /// Hybrid search: BM25 prefetch via Tantivy, then re-rank by combining
    /// normalized BM25 scores with semantic cosine similarity.
    ///
    /// Requires both Tantivy and embeddings to be enabled.
    ///
    /// `alpha` controls the balance: `final = alpha * norm_bm25 + (1-alpha) * cosine_sim`.
    /// Lower alpha = more weight to semantic meaning.
    #[cfg(has_embeddings)]
    pub fn search_hybrid(
        &self,
        query: &str,
        top_k: usize,
        prefetch_count: usize,
        alpha: f32,
    ) -> VaultResult<Vec<(PathBuf, f32)>> {
        Ok(self
            .search_hybrid_ranked(query, top_k, prefetch_count, alpha)?
            .into_iter()
            .map(|hit| (hit.path, hit.score))
            .collect())
    }

    #[cfg(has_embeddings)]
    pub(crate) fn search_hybrid_ranked(
        &self,
        query: &str,
        top_k: usize,
        prefetch_count: usize,
        alpha: f32,
    ) -> VaultResult<Vec<semantic_results::RankedNote>> {
        if query.is_empty() || top_k == 0 {
            return Ok(Vec::new());
        }

        let tv = self.inner.tantivy.as_ref().ok_or_else(|| {
            VaultError::Other("hybrid search requires Tantivy (set OBSIDIAN_TANTIVY=true)".into())
        })?;
        let runtime = self.inner.embedding_runtime.as_ref().ok_or_else(|| {
            VaultError::Embedding("embeddings not enabled (OBSIDIAN_EMBEDDINGS=false)".into())
        })?;
        let snapshot = runtime.query_snapshot()?;
        snapshot.validate_query(query)?;
        let current_paths = self.read_index().notes().keys().cloned().collect();
        let entries = snapshot.capture_entries(&current_paths);
        if entries.is_empty() {
            return Ok(Vec::new());
        }
        let allowed = semantic_results::paths(&entries);
        let prefetch = prefetch_count.max(top_k);
        let bm25_hits = tv.search_for_paths(query, prefetch, Some(&allowed))?;
        if bm25_hits.is_empty() {
            return Ok(Vec::new());
        }

        let query_vec = snapshot.embed_query(query)?;
        semantic_results::rank_hybrid(
            &entries,
            &bm25_hits,
            &query_vec,
            alpha.clamp(0.0, 1.0),
            top_k,
        )
    }

    pub fn search_by_tag(&self, tag: &str) -> VaultResult<Vec<NoteMetadata>> {
        Ok(self
            .read_index()
            .notes_with_tag(tag)
            .into_iter()
            .cloned()
            .collect())
    }

    pub fn search_by_tag_prefix(&self, tag: &str) -> VaultResult<Vec<NoteMetadata>> {
        Ok(self
            .read_index()
            .notes_with_tag_prefix(tag)
            .into_iter()
            .cloned()
            .collect())
    }

    pub fn search_frontmatter(
        &self,
        field: &str,
        value: &serde_json::Value,
    ) -> VaultResult<Vec<NoteMetadata>> {
        Ok(self
            .read_index()
            .search_frontmatter(field, value)
            .into_iter()
            .cloned()
            .collect())
    }

    pub fn search_frontmatter_exists(&self, field: &str) -> VaultResult<Vec<NoteMetadata>> {
        Ok(self
            .read_index()
            .search_frontmatter_exists(field)
            .into_iter()
            .cloned()
            .collect())
    }

    pub fn search_frontmatter_contains(
        &self,
        field: &str,
        value: &serde_json::Value,
    ) -> VaultResult<Vec<NoteMetadata>> {
        Ok(self
            .read_index()
            .search_frontmatter_contains(field, value)
            .into_iter()
            .cloned()
            .collect())
    }

    pub fn backlinks(&self, path: &Path) -> VaultResult<Vec<NoteMetadata>> {
        let actual_path = self.canonical_existing_relative_path(path)?;
        Ok(self
            .read_index()
            .backlinks_to(&actual_path)
            .into_iter()
            .cloned()
            .collect())
    }

    pub fn outgoing_links(&self, path: &Path) -> VaultResult<Vec<WikiLink>> {
        let actual_path = self.canonical_existing_relative_path(path)?;
        Ok(self
            .read_index()
            .outgoing_links(&actual_path)
            .into_iter()
            .cloned()
            .collect())
    }

    pub fn broken_links(&self) -> VaultResult<Vec<(PathBuf, WikiLink)>> {
        Ok(self.read_index().broken_links())
    }

    pub fn resolve_link(&self, target: &str) -> Option<PathBuf> {
        self.read_index().resolve_link(target)
    }

    pub fn orphan_notes(&self) -> VaultResult<Vec<NoteMetadata>> {
        Ok(self
            .read_index()
            .orphan_notes()
            .into_iter()
            .cloned()
            .collect())
    }

    pub fn vault_stats(&self) -> VaultResult<VaultStats> {
        Ok(self.read_index().stats().clone())
    }

    pub(crate) fn canonical_existing_relative_path(&self, path: &Path) -> VaultResult<PathBuf> {
        Ok(self.inner.filesystem.resolve_existing(path)?.relative)
    }

    /// Validate that a relative path doesn't escape the vault root.
    pub fn validate_path(&self, path: &Path) -> VaultResult<()> {
        self.inner.filesystem.resolve_for_write(path)?;
        Ok(())
    }

    // ── periodic delegation ────────────────────────────────────────────

    pub fn get_periodic_note(
        &self,
        period: &NotePeriod,
        date: Option<NaiveDate>,
    ) -> VaultResult<String> {
        let config = periodic::read_config_rooted(&self.inner.filesystem, period)?;
        let date = date.unwrap_or_else(|| Local::now().date_naive());
        let path = periodic::periodic_note_path(&config, &date)?;
        self.inner.filesystem.read_file(&path)
    }

    pub fn create_periodic_note(
        &self,
        period: &NotePeriod,
        date: Option<NaiveDate>,
        content_override: Option<&str>,
    ) -> VaultResult<PathBuf> {
        let _mutation = self.lock_mutations()?;
        let config = periodic::read_config_rooted(&self.inner.filesystem, period)?;
        let date = date.unwrap_or_else(|| Local::now().date_naive());
        let path = periodic::periodic_note_path(&config, &date)?;

        let content = if let Some(custom) = content_override {
            custom.to_owned()
        } else {
            match &config.template {
                Some(tmpl) if !tmpl.is_empty() => {
                    let title = path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or_default();
                    match periodic::expand_rooted(
                        &self.inner.filesystem,
                        Path::new(tmpl),
                        &date,
                        title,
                    ) {
                        Ok(c) => c,
                        Err(VaultError::NoteNotFound(_)) => String::new(),
                        Err(e) => return Err(e),
                    }
                }
                _ => String::new(),
            }
        };

        frontmatter::parse_frontmatter(&content)?;
        let actual_path = self.inner.filesystem.create_file(&path, &content)?;
        self.reindex(&actual_path)?;
        Ok(actual_path)
    }

    pub fn list_recent_periodic_notes(
        &self,
        period: &NotePeriod,
        limit: usize,
    ) -> VaultResult<Vec<PathBuf>> {
        let config = periodic::read_config_rooted(&self.inner.filesystem, period)?;
        periodic::list_rooted(&self.inner.filesystem, &config, limit)
    }

    // ── private helpers ────────────────────────────────────────────────

    /// Two-phase Tantivy search: BM25 ranking then context extraction.
    ///
    /// 1. Rank: Tantivy returns top-K `(path, score)` via BM25.
    /// 2. Context: For each hit, read the file and locate query words with a
    ///    case-insensitive regex to produce `SearchMatch` snippets.
    ///
    /// If the query matched only through stemming (no literal occurrence of
    /// any query word), the `matches` vec will be empty but `score` is populated.
    fn tantivy_search_with_context(
        &self,
        tv: &TantivyIndex,
        query: &str,
        context_len: usize,
        max_results: usize,
        fuzzy: bool,
        fields: Option<&[SearchField]>,
    ) -> VaultResult<Vec<SearchResult>> {
        let hits = if fuzzy || fields.is_some() {
            tv.search_with_options(query, max_results, fuzzy, fields)?
        } else {
            tv.search(query, max_results)?
        };

        let word_re = (context_len > 0 && !hits.is_empty())
            .then(|| search_utils::compile_query_word_regex(query))
            .flatten();

        let mut results = Vec::with_capacity(hits.len());
        for (path, score) in hits {
            let matches = if let Some(re) = &word_re
                && let Ok(content) = self.inner.filesystem.read_observed(&path)
            {
                re.find_iter(&content)
                    .map(|m| {
                        let (context, match_start, match_end, line) =
                            index::extract_match_context(&content, m.start(), m.end(), context_len);
                        SearchMatch {
                            line,
                            context,
                            match_start,
                            match_end,
                        }
                    })
                    .collect()
            } else {
                Vec::new()
            };

            results.push(SearchResult {
                path,
                matches,
                score: Some(score as f64),
            });
        }

        Ok(results)
    }

    fn lock_mutations(&self) -> VaultResult<std::sync::MutexGuard<'_, ()>> {
        self.inner
            .mutations
            .lock()
            .map_err(|error| VaultError::Other(format!("vault mutation lock poisoned: {error}")))
    }

    fn read_index(&self) -> std::sync::RwLockReadGuard<'_, VaultIndex> {
        self.inner.index.read().unwrap_or_else(|e| e.into_inner())
    }

    fn write_index(&self) -> std::sync::RwLockWriteGuard<'_, VaultIndex> {
        self.inner.index.write().unwrap_or_else(|e| e.into_inner())
    }

    fn reindex(&self, path: &Path) -> VaultResult<()> {
        let mut index = self.write_index();
        let was_note = index.get_note(path).is_some();
        let result = index.refresh_file(
            &self.inner.filesystem,
            path,
            &self.inner.exclude,
            self.inner.tantivy.as_deref(),
            #[cfg(has_embeddings)]
            self.inner.embedding_runtime.is_some(),
        );
        #[cfg(has_embeddings)]
        if let Some(runtime) = &self.inner.embedding_runtime
            && let Some(change) = index.embedding_change(path, was_note)
        {
            runtime.submit(path, change);
        }
        drop(index);
        let dirty = result.as_ref().copied().unwrap_or(was_note);
        let flush = self
            .inner
            .tantivy
            .as_ref()
            .filter(|_| dirty)
            .map(|tv| tv.publish())
            .transpose();
        #[cfg(has_embeddings)]
        if let Ok(Some(committed)) = &flush
            && let Some(runtime) = &self.inner.embedding_runtime
        {
            self.write_index()
                .publish_embedding_upserts(*committed, |path| runtime.submit_upsert(path));
        }
        match (result, flush) {
            (Err(error), Err(flush)) => Err(VaultError::Other(format!(
                "{error}; search flush failed: {flush}"
            ))),
            (Err(error), _) | (_, Err(error)) => Err(error),
            _ => Ok(()),
        }
    }

    // ── embedding helpers (feature-gated) ─────────────────────────────
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::models::{PatchOperation, PatchTargetType};
    use crate::test_helpers::{create_test_vault, tantivy_config, test_config};
    use unicode_normalization::UnicodeNormalization;

    const PERIODIC_PERIODS: [NotePeriod; 5] = [
        NotePeriod::Daily,
        NotePeriod::Weekly,
        NotePeriod::Monthly,
        NotePeriod::Quarterly,
        NotePeriod::Yearly,
    ];

    fn configure_periodic_template(root: &Path, period: &NotePeriod, template: &Path) {
        let key = match period {
            NotePeriod::Daily => "daily",
            NotePeriod::Weekly => "weekly",
            NotePeriod::Monthly => "monthly",
            NotePeriod::Quarterly => "quarterly",
            NotePeriod::Yearly => "yearly",
        };
        let entry = serde_json::json!({
            "enabled": true,
            "format": "YYYY-MM-DD",
            "folder": "Periodic",
            "template": template,
        });
        let (config_path, config) = match period {
            NotePeriod::Daily => (root.join(".obsidian/daily-notes.json"), entry),
            _ => (
                root.join(".obsidian/plugins/periodic-notes/data.json"),
                serde_json::json!({(key): entry}),
            ),
        };
        std::fs::create_dir_all(config_path.parent().unwrap()).unwrap();
        std::fs::write(config_path, serde_json::to_vec(&config).unwrap()).unwrap();
    }

    fn periodic_template_fixture() -> (tempfile::TempDir, PathBuf, PathBuf) {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("vault");
        create_test_vault(&root);
        let canary = directory.path().join("canary.md");
        std::fs::write(&canary, "outside template canary").unwrap();
        (directory, root, canary)
    }

    async fn assert_periodic_template_rejected(
        root: &Path,
        canary: &Path,
        period: &NotePeriod,
        template: &Path,
    ) {
        configure_periodic_template(root, period, template);
        let vault = Vault::open(&test_config(root)).await.unwrap();
        let date = NaiveDate::from_ymd_opt(2026, 1, 2).unwrap();

        let result = vault.create_periodic_note(period, Some(date), None);

        assert!(
            matches!(
                result,
                Err(VaultError::OutsideVault(_) | VaultError::InvalidPath(_))
            ),
            "{period:?}: {result:?}"
        );
        assert!(!root.join("Periodic").exists());
        assert_eq!(
            std::fs::read_to_string(canary).unwrap(),
            "outside template canary"
        );
    }

    #[tokio::test]
    async fn periodic_templates_reject_parent_directory_escape() {
        for period in PERIODIC_PERIODS {
            let (_directory, root, canary) = periodic_template_fixture();
            assert_periodic_template_rejected(&root, &canary, &period, Path::new("../canary"))
                .await;
        }
    }

    #[tokio::test]
    async fn periodic_templates_reject_absolute_paths() {
        for period in PERIODIC_PERIODS {
            let (_directory, root, canary) = periodic_template_fixture();
            assert_periodic_template_rejected(&root, &canary, &period, &canary).await;
        }
    }

    #[tokio::test]
    async fn periodic_templates_allow_missing_vault_relative_templates() {
        for period in PERIODIC_PERIODS {
            let (_directory, root, _canary) = periodic_template_fixture();
            configure_periodic_template(&root, &period, Path::new("Templates/missing"));
            let vault = Vault::open(&test_config(&root)).await.unwrap();
            let date = NaiveDate::from_ymd_opt(2026, 1, 2).unwrap();

            let note = vault
                .create_periodic_note(&period, Some(date), None)
                .unwrap();

            assert!(vault.read_note(&note).unwrap().is_empty());
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn periodic_templates_reject_symlinks_outside_the_vault() {
        for period in PERIODIC_PERIODS {
            let (_directory, root, canary) = periodic_template_fixture();
            std::fs::create_dir(root.join("Templates")).unwrap();
            std::os::unix::fs::symlink(&canary, root.join("Templates/linked.md")).unwrap();
            assert_periodic_template_rejected(
                &root,
                &canary,
                &period,
                Path::new("Templates/linked"),
            )
            .await;
        }
    }

    #[tokio::test]
    async fn periodic_templates_resolve_vault_relative_unicode_paths_with_optional_extension() {
        for period in PERIODIC_PERIODS {
            for template in [
                "Templates/Élan",
                "Templates/Élan.md",
                "Templates/E\u{301}lan",
            ] {
                let (_directory, root, _canary) = periodic_template_fixture();
                std::fs::create_dir(root.join("Templates")).unwrap();
                std::fs::write(
                    root.join("Templates/E\u{301}lan.md"),
                    "Created {{date:YYYY-MM-DD}} {{title}}",
                )
                .unwrap();
                configure_periodic_template(&root, &period, Path::new(template));
                let vault = Vault::open(&test_config(&root)).await.unwrap();
                let date = NaiveDate::from_ymd_opt(2026, 1, 2).unwrap();

                let note = vault
                    .create_periodic_note(&period, Some(date), None)
                    .unwrap();

                assert_eq!(note, Path::new("Periodic/2026-01-02.md"));
                assert_eq!(
                    vault.read_note(&note).unwrap(),
                    "Created 2026-01-02 2026-01-02"
                );
            }
        }
    }

    #[tokio::test]
    async fn periodic_templates_follow_native_case_matching() {
        for period in PERIODIC_PERIODS {
            for template in ["templates/daily", "templates/daily.md"] {
                let (_directory, root, _canary) = periodic_template_fixture();
                std::fs::create_dir(root.join("Templates")).unwrap();
                std::fs::write(root.join("Templates/Daily.md"), "Created {{title}}").unwrap();
                let native_match = root.join("templates/daily.md").is_file();
                configure_periodic_template(&root, &period, Path::new(template));
                let vault = Vault::open(&test_config(&root)).await.unwrap();
                let date = NaiveDate::from_ymd_opt(2026, 1, 2).unwrap();

                let note = vault
                    .create_periodic_note(&period, Some(date), None)
                    .unwrap();

                assert_eq!(
                    vault.read_note(&note).unwrap(),
                    if native_match {
                        "Created 2026-01-02"
                    } else {
                        ""
                    }
                );
            }
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn periodic_templates_reject_native_case_matched_symlinks_outside_the_vault() {
        for period in PERIODIC_PERIODS {
            let (_directory, root, canary) = periodic_template_fixture();
            std::fs::create_dir(root.join("Templates")).unwrap();
            std::os::unix::fs::symlink(&canary, root.join("Templates/Linked.md")).unwrap();
            if !root.join("templates/linked.md").is_file() {
                continue;
            }

            assert_periodic_template_rejected(
                &root,
                &canary,
                &period,
                Path::new("templates/linked"),
            )
            .await;
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn periodic_templates_allow_symlinks_to_templates_inside_the_vault() {
        for period in PERIODIC_PERIODS {
            let (_directory, root, _canary) = periodic_template_fixture();
            std::fs::create_dir(root.join("Templates")).unwrap();
            std::fs::write(root.join("Templates/inside.md"), "Inside {{title}}").unwrap();
            std::os::unix::fs::symlink("inside.md", root.join("Templates/linked.md")).unwrap();
            configure_periodic_template(&root, &period, Path::new("Templates/linked"));
            let vault = Vault::open(&test_config(&root)).await.unwrap();
            let date = NaiveDate::from_ymd_opt(2026, 1, 2).unwrap();

            let note = vault
                .create_periodic_note(&period, Some(date), None)
                .unwrap();

            assert_eq!(vault.read_note(&note).unwrap(), "Inside 2026-01-02");
        }
    }

    #[cfg(has_embeddings)]
    struct TestEmbedder {
        identity: embeddings::EmbeddingSpaceIdentity,
        inputs: Mutex<Vec<String>>,
    }

    #[cfg(has_embeddings)]
    impl TestEmbedder {
        fn new() -> Self {
            Self {
                identity: embeddings::EmbeddingSpaceIdentity {
                    backend: embeddings::EmbeddingBackendKind::Local,
                    model: "vault-test".into(),
                    endpoint_fingerprint: None,
                    dimension: 2,
                    input_version: embeddings::EMBEDDING_INPUT_VERSION,
                    tokenizer_fingerprint: [1; 32],
                    max_input_tokens: 8192,
                    chunking_version: crate::vault::embedding_input::CHUNKING_VERSION,
                },
                inputs: Mutex::new(Vec::new()),
            }
        }
    }

    #[cfg(has_embeddings)]
    impl embeddings::Embedder for TestEmbedder {
        fn input_policy(&self) -> &crate::vault::embedding_input::EmbeddingInputPolicy {
            &crate::vault::embedding_input::EmbeddingInputPolicy {
                max_input_tokens: 8192,
                tokenizer_fingerprint: [1; 32],
            }
        }

        fn count_tokens(&self, text: &str) -> VaultResult<usize> {
            Ok(text.chars().count())
        }

        fn request_limits(&self) -> crate::vault::embedding_input::EmbeddingRequestLimits {
            crate::vault::embedding_input::EmbeddingRequestLimits {
                max_inputs: 64,
                max_total_tokens: 8192 * 64,
                max_request_bytes: 16 * 1024 * 1024,
            }
        }

        fn dimension(&self) -> usize {
            self.identity.dimension
        }

        fn space_identity(&self) -> &embeddings::EmbeddingSpaceIdentity {
            &self.identity
        }

        fn embed_batch(&self, texts: &[&str]) -> VaultResult<Vec<Vec<f32>>> {
            self.inputs
                .lock()
                .unwrap()
                .extend(texts.iter().map(|text| (*text).to_string()));
            Ok(texts.iter().map(|_| vec![1.0, 0.0]).collect())
        }
    }

    #[cfg(has_embeddings)]
    fn blocked_embedding_loader() -> EmbeddingLoaderFuture {
        Box::pin(std::future::pending())
    }

    #[cfg(has_embeddings)]
    fn embedding_config(vault_root: &Path) -> Config {
        let mut config = test_config(vault_root);
        config.embeddings = true;
        config
    }

    #[tokio::test]
    async fn vault_open_succeeds() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let config = test_config(dir.path());

        let vault = Vault::open(&config).await;
        assert!(vault.is_ok());
    }

    #[tokio::test]
    async fn vault_open_without_obsidian_dir() {
        let dir = tempfile::tempdir().unwrap();
        let config = test_config(dir.path());

        let vault = Vault::open(&config).await;
        assert!(vault.is_ok(), "should succeed even without .obsidian/");
    }

    #[tokio::test]
    async fn vault_open_nonexistent_path() {
        let config = Config {
            vault_path: PathBuf::from("/nonexistent/path/that/does/not/exist"),
            watch: false,
            log_level: "error".into(),
            transport: crate::config::Transport::Stdio,
            http_host: crate::config::DEFAULT_HTTP_HOST,
            http_port: crate::config::DEFAULT_HTTP_PORT,
            tantivy: false,
            embeddings: false,
            embeddings_model: String::new(),
            hybrid_alpha: 0.25,
            embedding_provider: None,
            tool_filter: crate::config::ToolFilter::Full,
            mcp_data_dir: None,
            exclude_patterns: vec![],
        };

        let result = Vault::open(&config).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn vault_read_write_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());

        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        vault
            .write_note(Path::new("hello.md"), "# Hello\nWorld")
            .unwrap();
        let content = vault.read_note(Path::new("hello.md")).unwrap();
        assert_eq!(content, "# Hello\nWorld");
    }

    #[tokio::test]
    async fn vault_append_note() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        vault.write_note(Path::new("log.md"), "line one\n").unwrap();
        vault
            .append_note(Path::new("log.md"), "line two\n")
            .unwrap();

        let content = vault.read_note(Path::new("log.md")).unwrap();
        assert_eq!(content, "line one\nline two\n");
    }

    #[tokio::test]
    async fn vault_delete_removes_from_index() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        vault
            .write_note(Path::new("ephemeral.md"), "# Gone soon")
            .unwrap();
        assert!(vault.get_note_metadata(Path::new("ephemeral.md")).is_ok());

        vault.delete_note(Path::new("ephemeral.md")).unwrap();
        assert!(vault.get_note_metadata(Path::new("ephemeral.md")).is_err());
    }

    #[tokio::test]
    async fn vault_move_updates_index() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        vault.write_note(Path::new("old.md"), "# Moved").unwrap();
        assert!(vault.get_note_metadata(Path::new("old.md")).is_ok());

        let new_path = vault
            .move_note(Path::new("old.md"), Path::new("subdir/new.md"))
            .unwrap();
        assert_eq!(new_path, PathBuf::from("subdir/new.md"));

        assert!(vault.get_note_metadata(Path::new("old.md")).is_err());
        assert!(vault.get_note_metadata(Path::new("subdir/new.md")).is_ok());

        let content = vault.read_note(Path::new("subdir/new.md")).unwrap();
        assert_eq!(content, "# Moved");
    }

    #[tokio::test]
    async fn vault_metadata_accepts_canonically_equivalent_unicode_path() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let composed = "02_База-знаний/Сущности/lic1c.md";
        let decomposed: String = composed.nfd().collect();
        let disk_path = PathBuf::from(&decomposed);
        std::fs::create_dir_all(dir.path().join(disk_path.parent().unwrap())).unwrap();
        std::fs::write(dir.path().join(&disk_path), "# License\n").unwrap();

        let vault = Vault::open(&test_config(dir.path())).await.unwrap();
        let meta = vault.get_note_metadata(Path::new(composed)).unwrap();

        assert_eq!(meta.path, disk_path);
    }

    #[tokio::test]
    async fn vault_batch_metadata_uses_exact_index_paths_without_resolution() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let composed = "Knowledge/é.md";
        let decomposed: String = composed.nfd().collect();
        let disk_path = PathBuf::from(&decomposed);
        std::fs::create_dir_all(dir.path().join(disk_path.parent().unwrap())).unwrap();
        std::fs::write(dir.path().join(&disk_path), "# Unicode\n").unwrap();

        let vault = Vault::open(&test_config(dir.path())).await.unwrap();
        let metadata = vault.get_indexed_note_metadata_batch([
            disk_path.as_path(),
            Path::new("missing.md"),
            Path::new(composed),
        ]);

        assert_eq!(metadata.len(), 3);
        assert_eq!(
            metadata[0].as_ref().map(|meta| meta.title.as_str()),
            disk_path.file_stem().and_then(|stem| stem.to_str())
        );
        assert!(metadata[1].is_none());
        assert!(metadata[2].is_none());
    }

    #[tokio::test]
    async fn vault_write_delete_and_move_use_unicode_canonical_index_key() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let composed = "02_База-знаний/Сущности/lic1c.md";
        let decomposed: String = composed.nfd().collect();
        let disk_path = PathBuf::from(&decomposed);
        std::fs::create_dir_all(dir.path().join(disk_path.parent().unwrap())).unwrap();
        std::fs::write(dir.path().join(&disk_path), "# Old\n").unwrap();

        let vault = Vault::open(&test_config(dir.path())).await.unwrap();
        let before = vault.vault_stats().unwrap().total_notes;

        vault
            .write_note(Path::new(composed), "# New\nunique-unicode-write\n")
            .unwrap();
        assert_eq!(vault.vault_stats().unwrap().total_notes, before);
        assert_eq!(
            vault
                .search_text("unique-unicode-write", 40)
                .unwrap()
                .first()
                .map(|result| result.path.clone()),
            Some(disk_path.clone())
        );

        let moved = vault
            .move_note(Path::new(composed), Path::new("Moved/lic1c.md"))
            .unwrap();
        assert_eq!(moved, PathBuf::from("Moved/lic1c.md"));
        assert!(vault.get_note_metadata(Path::new(composed)).is_err());
        assert!(vault.get_note_metadata(&moved).is_ok());

        vault.delete_note(&moved).unwrap();
        assert!(vault.get_note_metadata(&moved).is_err());
    }

    #[tokio::test]
    async fn vault_patch_note() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        vault
            .write_note(
                Path::new("patched.md"),
                "# Section\nOriginal content\n# Other\nKeep this\n",
            )
            .unwrap();

        let request = PatchRequest {
            operation: PatchOperation::Replace,
            target_type: PatchTargetType::Heading,
            target: "Section".into(),
            content: "Replaced content\n".into(),
        };
        vault.patch_note(Path::new("patched.md"), &request).unwrap();

        let content = vault.read_note(Path::new("patched.md")).unwrap();
        assert!(content.contains("Replaced content"));
        assert!(content.contains("# Other"));
        assert!(content.contains("Keep this"));
    }

    #[tokio::test]
    async fn vault_frontmatter_operations() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        vault
            .write_note(Path::new("meta.md"), "# Note\nBody text\n")
            .unwrap();

        vault
            .set_frontmatter_field(
                Path::new("meta.md"),
                "status",
                serde_json::Value::String("draft".into()),
            )
            .unwrap();
        vault
            .set_frontmatter_field(
                Path::new("meta.md"),
                "priority",
                serde_json::Value::Number(1.into()),
            )
            .unwrap();

        let fm = vault.get_frontmatter(Path::new("meta.md")).unwrap();
        assert!(fm.is_some());
        let obj = fm.unwrap();
        assert_eq!(obj["status"], "draft");
        assert_eq!(obj["priority"], 1);

        vault
            .remove_frontmatter_field(Path::new("meta.md"), "status")
            .unwrap();

        let fm = vault.get_frontmatter(Path::new("meta.md")).unwrap();
        let obj = fm.unwrap();
        assert!(obj.get("status").is_none());
        assert_eq!(obj["priority"], 1);
    }

    #[tokio::test]
    async fn vault_search_text() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        vault
            .write_note(
                Path::new("searchable.md"),
                "# Rust\nRust is a systems language.\n",
            )
            .unwrap();
        vault
            .write_note(Path::new("other.md"), "# Python\nPython is dynamic.\n")
            .unwrap();

        let results = vault.search_text("Rust", 40).unwrap();
        assert!(!results.is_empty());
        assert!(results.iter().any(|r| r.path == Path::new("searchable.md")));
        assert!(!results.iter().any(|r| r.path == Path::new("other.md")));
    }

    #[tokio::test]
    async fn vault_document_map() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        vault
            .write_note(
                Path::new("mapped.md"),
                "---\ntags: [rust]\n---\n# Heading\n## Sub\nText ^block1\n",
            )
            .unwrap();

        let map = vault.get_document_map(Path::new("mapped.md")).unwrap();
        assert!(map.headings.iter().any(|h| h.contains("Heading")));
        assert!(map.headings.iter().any(|h| h.contains("Sub")));
        assert!(map.block_refs.contains(&"block1".to_string()));
        assert!(map.frontmatter_fields.contains(&"tags".to_string()));
    }

    #[tokio::test]
    async fn vault_tags_and_backlinks() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        vault
            .write_note(Path::new("a.md"), "---\ntags: [project]\n---\n# A\n[[b]]\n")
            .unwrap();
        vault
            .write_note(Path::new("b.md"), "---\ntags: [project]\n---\n# B\n")
            .unwrap();

        let tagged = vault.search_by_tag("project").unwrap();
        assert_eq!(tagged.len(), 2);

        let links = vault.outgoing_links(Path::new("a.md")).unwrap();
        assert!(links.iter().any(|l| l.target == "b"));

        let bl = vault.backlinks(Path::new("b.md")).unwrap();
        assert!(bl.iter().any(|m| m.path == Path::new("a.md")));
    }

    #[tokio::test]
    async fn vault_stats() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        vault.write_note(Path::new("one.md"), "# One").unwrap();
        vault.write_note(Path::new("two.md"), "# Two").unwrap();

        let stats = vault.vault_stats().unwrap();
        assert_eq!(stats.total_notes, 2);
    }

    #[tokio::test]
    async fn vault_create_note_basic() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        vault
            .create_note(Path::new("new.md"), "# New\n", None)
            .unwrap();
        let content = vault.read_note(Path::new("new.md")).unwrap();
        assert_eq!(content, "# New\n");
    }

    #[tokio::test]
    async fn vault_create_note_with_frontmatter() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        let fm = serde_json::json!({"tags": ["test"], "draft": true});
        vault
            .create_note(Path::new("fm.md"), "Body\n", Some(&fm))
            .unwrap();
        let content = vault.read_note(Path::new("fm.md")).unwrap();
        assert!(content.starts_with("---\n"));
        assert!(content.contains("Body\n"));
    }

    #[tokio::test]
    async fn vault_create_note_rejects_non_object_frontmatter_without_writing() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        for (path, frontmatter, actual) in [
            ("array.md", serde_json::json!(["rust", "mcp"]), "array"),
            ("string.md", serde_json::json!("{\"draft\":true}"), "string"),
            ("null.md", serde_json::Value::Null, "null"),
        ] {
            let err = vault
                .create_note(Path::new(path), "body", Some(&frontmatter))
                .unwrap_err();
            assert!(
                matches!(
                    &err,
                    VaultError::InvalidFrontmatter {
                        path: error_path,
                        actual: error_actual
                    } if error_path == Path::new(path) && *error_actual == actual
                ),
                "unexpected error for '{path}': {err:?}"
            );
            assert!(!dir.path().join(path).exists());

            let error_data: rmcp::ErrorData = err.into();
            assert_eq!(error_data.code, rmcp::model::ErrorCode::INVALID_PARAMS);
        }
    }

    #[tokio::test]
    async fn vault_create_note_fails_if_exists() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        vault.create_note(Path::new("dup.md"), "", None).unwrap();
        let err = vault
            .create_note(Path::new("dup.md"), "new", None)
            .unwrap_err();
        assert!(matches!(err, VaultError::AlreadyExists(_)));
    }

    #[tokio::test]
    async fn vault_prepend_after_frontmatter() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        vault
            .write_note(Path::new("pre.md"), "---\ntags: [a]\n---\nExisting body\n")
            .unwrap();
        vault
            .prepend_note(Path::new("pre.md"), "Prepended\n")
            .unwrap();
        let content = vault.read_note(Path::new("pre.md")).unwrap();

        assert!(content.starts_with("---\n"));
        let prepended_pos = content.find("Prepended\n").unwrap();
        let existing_pos = content.find("Existing body\n").unwrap();
        assert!(prepended_pos < existing_pos);
    }

    #[tokio::test]
    async fn vault_prepend_no_frontmatter() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        vault
            .write_note(Path::new("nofm.md"), "Existing\n")
            .unwrap();
        vault
            .prepend_note(Path::new("nofm.md"), "Prepended\n")
            .unwrap();
        let content = vault.read_note(Path::new("nofm.md")).unwrap();
        assert_eq!(content, "Prepended\nExisting\n");
    }

    #[test]
    fn vault_is_send_sync_clone() {
        fn assert_send_sync_clone<T: Send + Sync + Clone>() {}
        assert_send_sync_clone::<Vault>();
    }

    #[tokio::test]
    async fn vault_open_with_tantivy() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());

        std::fs::write(
            dir.path().join("note.md"),
            "# Rust\nRust is a systems language.\n",
        )
        .unwrap();

        let vault = Vault::open(&tantivy_config(dir.path())).await.unwrap();
        assert!(vault.tantivy().is_some());
    }

    #[tokio::test]
    async fn vault_tantivy_syncs_on_write() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&tantivy_config(dir.path())).await.unwrap();

        vault
            .write_note(
                Path::new("alpha.md"),
                "# Alpha\nUnique content about zebras.\n",
            )
            .unwrap();

        let tv = vault.tantivy().unwrap();
        let results = tv.search("zebras", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, PathBuf::from("alpha.md"));
    }

    #[tokio::test]
    async fn vault_tantivy_syncs_on_delete() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&tantivy_config(dir.path())).await.unwrap();

        vault
            .write_note(Path::new("del.md"), "# Deletable\nEphemeral content.\n")
            .unwrap();
        assert_eq!(
            vault
                .tantivy()
                .unwrap()
                .search("ephemeral", 10)
                .unwrap()
                .len(),
            1
        );

        vault.delete_note(Path::new("del.md")).unwrap();
        assert!(
            vault
                .tantivy()
                .unwrap()
                .search("ephemeral", 10)
                .unwrap()
                .is_empty()
        );
    }

    #[tokio::test]
    async fn vault_tantivy_syncs_on_move() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&tantivy_config(dir.path())).await.unwrap();

        vault
            .write_note(Path::new("src.md"), "# Source\nMovable content.\n")
            .unwrap();
        vault
            .move_note(Path::new("src.md"), Path::new("dest.md"))
            .unwrap();

        let tv = vault.tantivy().unwrap();
        let results = tv.search("movable", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, PathBuf::from("dest.md"));
    }

    #[tokio::test]
    async fn vault_tantivy_syncs_on_create() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&tantivy_config(dir.path())).await.unwrap();

        let fm = serde_json::json!({"tags": ["science"]});
        vault
            .create_note(
                Path::new("created.md"),
                "# Created\nBioluminescence in deep sea creatures.\n",
                Some(&fm),
            )
            .unwrap();

        let tv = vault.tantivy().unwrap();
        let results = tv.search("bioluminescence", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, PathBuf::from("created.md"));
    }

    #[tokio::test]
    async fn vault_search_text_tantivy_returns_scores() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&tantivy_config(dir.path())).await.unwrap();

        vault
            .write_note(
                Path::new("a.md"),
                "# Quantum\nQuantum computing is fascinating.\n",
            )
            .unwrap();
        vault
            .write_note(Path::new("b.md"), "# Other\nNothing related.\n")
            .unwrap();

        let results = vault.search_text("quantum", 40).unwrap();
        assert!(!results.is_empty());
        assert!(results[0].score.is_some());
        assert!(results[0].score.unwrap() > 0.0);
        assert_eq!(results[0].path, PathBuf::from("a.md"));
    }

    #[tokio::test]
    async fn vault_search_text_with_options_fuzzy() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&tantivy_config(dir.path())).await.unwrap();

        vault
            .write_note(Path::new("target.md"), "# Algorithm\nSorting algorithms.\n")
            .unwrap();

        // "algorihm" is a typo for "algorithm"
        let results = vault
            .search_text_with_options("algorihm", 40, 10, true, None)
            .unwrap();
        assert!(
            results.iter().any(|r| r.path == Path::new("target.md")),
            "fuzzy search should find 'algorithm' from typo 'algorihm'"
        );
    }

    #[tokio::test]
    async fn vault_search_text_with_options_field_filter() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&tantivy_config(dir.path())).await.unwrap();

        vault
            .write_note(
                Path::new("headingmatch.md"),
                "# Something Else\n## Cryptography Section\nBasic text.\n",
            )
            .unwrap();

        let heading_only = vault
            .search_text_with_options(
                "cryptography",
                40,
                10,
                false,
                Some(&[SearchField::Headings]),
            )
            .unwrap();
        assert!(
            heading_only
                .iter()
                .any(|r| r.path == Path::new("headingmatch.md"))
        );
    }

    #[tokio::test]
    async fn vault_search_text_tantivy_zero_context() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&tantivy_config(dir.path())).await.unwrap();

        vault
            .write_note(
                Path::new("ctx.md"),
                "# Context Test\nSearchable unique phrase here.\n",
            )
            .unwrap();

        let results = vault.search_text("searchable", 0).unwrap();
        assert!(!results.is_empty());
        assert!(results[0].score.is_some());
        assert!(
            results[0].matches.is_empty(),
            "context_length=0 should produce no match snippets"
        );
    }

    #[tokio::test]
    async fn vault_search_text_with_options_fallback_without_tantivy() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let vault = Vault::open(&test_config(dir.path())).await.unwrap();

        vault
            .write_note(
                Path::new("fallback.md"),
                "# Fallback\nFallback search content.\n",
            )
            .unwrap();

        let results = vault
            .search_text_with_options("fallback", 40, 10, true, None)
            .unwrap();
        assert!(
            results.iter().any(|r| r.path == Path::new("fallback.md")),
            "should still find results via regex fallback when tantivy is disabled"
        );
        assert!(
            results[0].score.is_none(),
            "regex fallback should not populate BM25 scores"
        );
    }

    #[cfg(has_embeddings)]
    #[tokio::test]
    async fn deferred_embeddings_reconcile_final_mutations_and_watcher_events() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join(".obsidian")).unwrap();
        std::fs::write(dir.path().join("edit.md"), "OUTDATED_MARKER").unwrap();
        std::fs::write(dir.path().join("delete.md"), "DELETED_MARKER").unwrap();
        let mut config = embedding_config(dir.path());
        config.watch = true;
        config.exclude_patterns = vec!["Hidden/".into()];
        let model = Arc::new(TestEmbedder::new());
        let loader_model = Arc::clone(&model);
        let polls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let loader_polls = Arc::clone(&polls);
        let loader: EmbeddingLoaderFuture = Box::pin(async move {
            loader_polls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(loader_model as Arc<dyn embeddings::Embedder>)
        });
        let vault = Vault::open_inner(&config, Some(loader), false)
            .await
            .unwrap();
        let submitter = vault.inner.embedding_runtime.as_ref().unwrap().downgrade();
        vault
            .write_note(Path::new("edit.md"), "FINAL_EDIT_MARKER")
            .unwrap();
        vault.delete_note(Path::new("delete.md")).unwrap();
        vault
            .create_note(Path::new("created.md"), "FINAL_MOVED_MARKER", None)
            .unwrap();
        vault
            .move_note(Path::new("created.md"), Path::new("moved.md"))
            .unwrap();
        vault
            .write_note(Path::new("Hidden/secret.md"), "EXCLUDED_MARKER")
            .unwrap();
        std::fs::write(dir.path().join("external.md"), "FINAL_EXTERNAL_MARKER").unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while vault.get_note_metadata(Path::new("external.md")).is_err() {
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            }
        })
        .await
        .unwrap();
        assert_eq!(polls.load(std::sync::atomic::Ordering::SeqCst), 0);
        assert!(!vault.embeddings_active());
        assert!(model.inputs.lock().unwrap().is_empty());
        assert!(vault.clone().activate_embeddings());
        assert!(!vault.activate_embeddings());
        // A previously captured watcher handle still reaches the activated coordinator.
        submitter.submit_upsert(Path::new("edit.md"));
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while !vault
                .embedding_status()
                .is_some_and(|status| status.phase == embedding_runtime::EmbeddingPhase::Ready)
            {
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        assert!(vault.embeddings_active());
        assert_eq!(polls.load(std::sync::atomic::Ordering::SeqCst), 1);
        let inputs = model.inputs.lock().unwrap();
        assert_eq!(
            inputs.len(),
            3,
            "only final visible notes require embedding: {inputs:?}"
        );
        for marker in [
            "FINAL_EDIT_MARKER",
            "FINAL_MOVED_MARKER",
            "FINAL_EXTERNAL_MARKER",
        ] {
            assert!(
                inputs.iter().any(|text| text.contains(marker)),
                "missing {marker}: {inputs:?}"
            );
        }
        for marker in ["OUTDATED_MARKER", "DELETED_MARKER", "EXCLUDED_MARKER"] {
            assert!(
                !inputs.iter().any(|text| text.contains(marker)),
                "unexpected {marker}: {inputs:?}"
            );
        }
    }

    #[cfg(has_embeddings)]
    #[tokio::test]
    async fn vault_open_and_core_search_return_while_embedding_loader_is_blocked() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let config = embedding_config(dir.path());

        let vault = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            Vault::open_with_embedding_loader(&config, Some(blocked_embedding_loader())),
        )
        .await
        .expect("Vault::open must not wait for the embedding loader")
        .unwrap();

        assert!(vault.search_text("anything", 20).unwrap().is_empty());
        let error = vault.search_semantic("anything", 5).unwrap_err();
        assert!(
            matches!(error, VaultError::Embedding(ref message) if message.contains("not ready") && message.contains("warming up")),
            "semantic search should report a warming runtime: {error}"
        );
    }

    #[cfg(has_embeddings)]
    #[tokio::test]
    async fn vault_mutations_submit_only_the_latest_embedding_intent() {
        use embedding_runtime::PendingKind;

        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let mut config = embedding_config(dir.path());
        config.exclude_patterns = vec!["Excluded/".into()];
        let vault = Vault::open_with_embedding_loader(&config, Some(blocked_embedding_loader()))
            .await
            .unwrap();
        let runtime = vault.inner.embedding_runtime.as_ref().unwrap();
        let path = Path::new("intent.md");

        vault.create_note(path, "# Intent\nBody\n", None).unwrap();
        assert_eq!(runtime.pending_kind(path), Some(PendingKind::Upsert));

        vault.append_note(path, "Appended\n").unwrap();
        assert_eq!(runtime.pending_kind(path), Some(PendingKind::Upsert));

        vault.prepend_note(path, "Prepended\n").unwrap();
        assert_eq!(runtime.pending_kind(path), Some(PendingKind::Upsert));

        vault.write_note(path, "# Intent\nReplacement\n").unwrap();
        assert_eq!(runtime.pending_kind(path), Some(PendingKind::Upsert));

        vault
            .set_frontmatter_field(path, "state", serde_json::json!("active"))
            .unwrap();
        assert_eq!(runtime.pending_kind(path), Some(PendingKind::Upsert));

        vault.remove_frontmatter_field(path, "state").unwrap();
        assert_eq!(runtime.pending_kind(path), Some(PendingKind::Upsert));

        vault
            .patch_note(
                path,
                &PatchRequest {
                    operation: PatchOperation::Replace,
                    target_type: PatchTargetType::Heading,
                    target: "Intent".into(),
                    content: "Patched\n".into(),
                },
            )
            .unwrap();
        assert_eq!(runtime.pending_kind(path), Some(PendingKind::Upsert));

        vault.delete_note(path).unwrap();
        assert_eq!(runtime.pending_kind(path), Some(PendingKind::Remove));

        let excluded_path = Path::new("Excluded/hidden.md");
        vault.write_note(excluded_path, "# Hidden\n").unwrap();
        assert_eq!(runtime.pending_kind(excluded_path), None);
        let attachment = Path::new("plain.txt");
        vault.write_note(attachment, "attachment").unwrap();
        assert_eq!(runtime.pending_kind(attachment), None);

        let old_path = Path::new("old.md");
        let new_path = Path::new("nested/new.md");
        vault.write_note(old_path, "# Move\n").unwrap();
        vault.move_note(old_path, new_path).unwrap();
        assert_eq!(runtime.pending_kind(old_path), Some(PendingKind::Remove));
        assert_eq!(runtime.pending_kind(new_path), Some(PendingKind::Upsert));

        let now_excluded = Path::new("Excluded/moved.md");
        vault.move_note(new_path, now_excluded).unwrap();
        assert_eq!(runtime.pending_kind(new_path), Some(PendingKind::Remove));
        assert_eq!(runtime.pending_kind(now_excluded), None);
    }

    #[cfg(has_embeddings)]
    #[tokio::test]
    async fn local_semantic_search_filters_paths_removed_from_primary_index() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let path = PathBuf::from("stale.md");
        std::fs::write(dir.path().join(&path), "# Stale\nBody\n").unwrap();

        let model = Arc::new(TestEmbedder::new());
        let mut store = embeddings::EmbeddingStore::new_with_identity(model.identity.clone());
        store
            .insert_hashed(path.clone(), [7; 32], vec![1.0, 0.0])
            .unwrap();
        store.set_first_pass_complete(true);
        let cache_path = dir
            .path()
            .join(".obsidian-mcp")
            .join("embeddings/scopes")
            .join(ExcludeSet::default().scope_id(&dir.path().canonicalize().unwrap()))
            .join("embeddings.bin");
        std::fs::create_dir_all(cache_path.parent().unwrap()).unwrap();
        store.save(&cache_path).unwrap();

        let loader: EmbeddingLoaderFuture =
            Box::pin(async move { Ok(model as Arc<dyn embeddings::Embedder>) });
        let config = embedding_config(dir.path());
        let vault = Vault::open_with_embedding_loader(&config, Some(loader))
            .await
            .unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            while !vault.has_embeddings() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("compatible cache should become queryable");

        vault.write_index().remove_file(&path);

        assert!(vault.search_semantic("stale", 5).unwrap().is_empty());
    }

    #[cfg(has_embeddings)]
    #[tokio::test]
    async fn semantic_facade_keeps_captured_revision_while_query_inference_blocks() {
        use std::sync::mpsc;
        use std::time::Duration;

        struct BlockingQueryEmbedder {
            inner: TestEmbedder,
            entered: mpsc::SyncSender<()>,
            release: Mutex<mpsc::Receiver<()>>,
        }

        impl embeddings::Embedder for BlockingQueryEmbedder {
            fn input_policy(&self) -> &embedding_input::EmbeddingInputPolicy {
                embeddings::Embedder::input_policy(&self.inner)
            }

            fn count_tokens(&self, text: &str) -> VaultResult<usize> {
                embeddings::Embedder::count_tokens(&self.inner, text)
            }

            fn request_limits(&self) -> embedding_input::EmbeddingRequestLimits {
                embeddings::Embedder::request_limits(&self.inner)
            }

            fn dimension(&self) -> usize {
                embeddings::Embedder::dimension(&self.inner)
            }

            fn space_identity(&self) -> &embeddings::EmbeddingSpaceIdentity {
                embeddings::Embedder::space_identity(&self.inner)
            }

            fn embed_batch(&self, texts: &[&str]) -> VaultResult<Vec<Vec<f32>>> {
                if texts.len() == 1 && texts[0] == "REVISION_BOUND_QUERY" {
                    self.entered.send(()).unwrap();
                    self.release
                        .lock()
                        .unwrap()
                        .recv_timeout(Duration::from_secs(5))
                        .unwrap();
                }
                embeddings::Embedder::embed_batch(&self.inner, texts)
            }
        }

        struct ReleaseOnDrop(Option<mpsc::Sender<()>>);

        impl Drop for ReleaseOnDrop {
            fn drop(&mut self) {
                if let Some(sender) = self.0.take() {
                    let _ = sender.send(());
                }
            }
        }

        async fn wait_for_revision(vault: &Vault, expected: &str) {
            let runtime = vault.inner.embedding_runtime.as_ref().unwrap();
            let paths = std::collections::HashSet::from([PathBuf::from("policy.md")]);
            tokio::time::timeout(Duration::from_secs(3), async {
                loop {
                    if let Ok(snapshot) = runtime.query_snapshot()
                        && snapshot
                            .capture_entries(&paths)
                            .get(Path::new("policy.md"))
                            .is_some_and(|entry| entry.source_revision == expected)
                    {
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
            })
            .await
            .expect("complete note generation should commit while query inference is held");
        }

        let directory = tempfile::tempdir().unwrap();
        create_test_vault(directory.path());
        let old_source = "# Policy\nRetain the original source.\n";
        let new_source =
            "---\ntags: [updated]\n---\n# Replacement\nDifferent source and offsets.\n";
        let old_revision = source::source_revision(old_source);
        let new_revision = source::source_revision(new_source);
        std::fs::write(directory.path().join("policy.md"), old_source).unwrap();
        let (entered_tx, entered_rx) = mpsc::sync_channel(1);
        let (release_tx, release_rx) = mpsc::channel();
        let release_guard = ReleaseOnDrop(Some(release_tx));
        let model = Arc::new(BlockingQueryEmbedder {
            inner: TestEmbedder::new(),
            entered: entered_tx,
            release: Mutex::new(release_rx),
        });
        let loader: EmbeddingLoaderFuture =
            Box::pin(async move { Ok(model as Arc<dyn embeddings::Embedder>) });
        let vault =
            Vault::open_with_embedding_loader(&embedding_config(directory.path()), Some(loader))
                .await
                .unwrap();
        wait_for_revision(&vault, &old_revision).await;

        let searching_vault = vault.clone();
        let query = tokio::task::spawn_blocking(move || {
            searching_vault.search_semantic_ranked("REVISION_BOUND_QUERY", 1)
        });
        tokio::task::spawn_blocking(move || entered_rx.recv_timeout(Duration::from_secs(3)))
            .await
            .unwrap()
            .expect("query should reach the held provider after capturing source entries");

        vault
            .write_note(Path::new("policy.md"), new_source)
            .unwrap();
        wait_for_revision(&vault, &new_revision).await;
        drop(release_guard);
        let mut hits = query.await.unwrap().unwrap();
        assert_eq!(hits.len(), 1);
        let hit = hits.remove(0);
        assert_eq!(hit.entry.source_revision, old_revision);
        assert_eq!(hit.evidence.scope, crate::models::EvidenceScope::Note);
        assert_eq!(
            hit.evidence.range,
            crate::models::SourceRange {
                start_byte: 0,
                end_byte: old_source.len(),
            }
        );

        let rendered = semantic_results::render(vault.root(), hit, "REVISION_BOUND_QUERY", false);
        assert_eq!(rendered.source_revision, old_revision);
        assert_eq!(rendered.source_status, crate::models::SourceStatus::Changed);
        let error = rendered.context_error.unwrap();
        assert_eq!(error.code, "source_changed");
        assert_eq!(
            error.current_revision.as_deref(),
            Some(new_revision.as_str())
        );
        assert!(rendered.snippet.is_none());
        assert!(rendered.content.is_none());
        assert!(rendered.preview.is_none());
    }

    #[cfg(has_embeddings)]
    #[tokio::test]
    async fn graceful_degradation_captures_embedding_load_error() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());

        let config = Config {
            vault_path: dir.path().to_path_buf(),
            watch: false,
            log_level: "error".into(),
            transport: crate::config::Transport::Stdio,
            http_host: crate::config::DEFAULT_HTTP_HOST,
            http_port: crate::config::DEFAULT_HTTP_PORT,
            tantivy: false,
            embeddings: true,
            embeddings_model: "nonexistent-model-that-will-fail".into(),
            hybrid_alpha: 0.25,
            embedding_provider: Some(crate::config::EmbeddingProvider::Local),
            tool_filter: crate::config::ToolFilter::Full,
            mcp_data_dir: None,
            exclude_patterns: vec![],
        };

        let vault = Vault::open(&config)
            .await
            .expect("vault should open despite embedding failure");
        assert!(
            !vault.has_embeddings(),
            "embeddings should not be available after load failure"
        );
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            while vault.embedding_load_error().is_none() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("background model load should report its failure");
        let err = vault.embedding_load_error().unwrap();
        assert!(!err.is_empty(), "error message should be descriptive");
    }

    #[cfg(has_embeddings)]
    #[tokio::test]
    async fn disabled_embeddings_do_not_relocate_legacy_cache() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());

        let legacy_dir = dir.path().join(".obsidian").join("obsidian-mcp");
        std::fs::create_dir_all(&legacy_dir).unwrap();
        let legacy_path = legacy_dir.join("embeddings.bin");
        let test_bytes: Vec<u8> = vec![0xDE, 0xAD, 0xBE, 0xEF, 1, 2, 3, 4];
        std::fs::write(&legacy_path, &test_bytes).unwrap();

        let _vault = Vault::open(&test_config(dir.path())).await.unwrap();

        let new_path = dir
            .path()
            .join(".obsidian-mcp")
            .join("embeddings")
            .join("embeddings.bin");
        assert!(
            !new_path.exists(),
            "disabled embeddings must not move caches"
        );
        assert!(
            legacy_path.exists(),
            "disabled embeddings must leave the legacy file untouched"
        );
        assert_eq!(std::fs::read(legacy_path).unwrap(), test_bytes);
    }

    #[cfg(has_embeddings)]
    #[tokio::test]
    async fn legacy_embedding_cache_relocation_waits_for_background_loader() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());

        let legacy_dir = dir.path().join(".obsidian").join("obsidian-mcp");
        std::fs::create_dir_all(&legacy_dir).unwrap();
        std::fs::write(legacy_dir.join("embeddings.bin"), b"old-legacy-data").unwrap();

        let config = embedding_config(dir.path());
        let vault = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            Vault::open_with_embedding_loader(&config, Some(blocked_embedding_loader())),
        )
        .await
        .expect("vault open must not wait for cache relocation")
        .unwrap();

        assert!(
            !dir.path()
                .join(".obsidian-mcp")
                .join("embeddings")
                .join("embeddings.bin")
                .exists(),
            "cache relocation must stay behind the blocked background loader"
        );
        drop(vault);
    }

    #[cfg(has_embeddings)]
    #[tokio::test]
    async fn no_legacy_cache_no_migration_error() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());

        let vault = Vault::open(&test_config(dir.path())).await;
        assert!(vault.is_ok(), "vault should open without legacy cache");

        let new_cache = dir
            .path()
            .join(".obsidian-mcp")
            .join("embeddings")
            .join("embeddings.bin");
        assert!(
            !new_cache.exists(),
            "no cache file should be created when no legacy exists"
        );
    }

    #[tokio::test]
    async fn replacement_at_an_unlinked_source_survives_direct_reconciliation() {
        let directory = tempfile::tempdir().unwrap();
        let vault = Vault::open(&crate::test_helpers::tantivy_config(directory.path()))
            .await
            .unwrap();
        let source = Path::new("source.md");
        let on_disk = directory.path().join(source);
        for moving in [false, true] {
            vault.write_note(source, "departed").unwrap();
            // Hold the metadata lock until the filesystem operation has finished.
            // A watcher can then observe the replacement before the tool gets the lock.
            let mut index = vault.write_index();
            let writer = vault.clone();
            let operation = std::thread::spawn(move || {
                if moving {
                    writer.move_note(source, Path::new("moved.md")).map(|_| ())
                } else {
                    writer.delete_note(source)
                }
            });
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            while on_disk.exists() {
                assert!(
                    std::time::Instant::now() < deadline,
                    "source was never unlinked"
                );
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            std::fs::write(&on_disk, "#replacement presentagain").unwrap();
            assert!(
                index
                    .refresh_file(
                        &vault.inner.filesystem,
                        source,
                        &vault.inner.exclude,
                        vault.inner.tantivy.as_deref(),
                        #[cfg(has_embeddings)]
                        false,
                    )
                    .unwrap()
            );
            drop(index);
            operation.join().unwrap().unwrap();

            assert_eq!(
                vault.get_note_metadata(source).unwrap().tags,
                ["replacement"]
            );
            let hits = vault.search_text("presentagain", 0).unwrap();
            assert_eq!(hits.len(), 1);
            assert_eq!(hits[0].path, source);
            if moving {
                assert_eq!(
                    std::fs::read_to_string(directory.path().join("moved.md")).unwrap(),
                    "departed"
                );
            }
        }
    }

    #[tokio::test]
    async fn note_and_attachment_mutations_match_a_cold_vault() {
        async fn assert_matches_disk(live: &Vault, config: &Config) {
            let cold = Vault::open(config).await.unwrap();
            assert_eq!(
                serde_json::to_value(live.vault_stats().unwrap()).unwrap(),
                serde_json::to_value(cold.vault_stats().unwrap()).unwrap()
            );
            let projection = |vault: &Vault| {
                let index = vault.read_index();
                let mut notes: Vec<_> = index
                    .notes()
                    .iter()
                    .map(|(path, note)| {
                        let mut backlinks: Vec<_> = index
                            .backlinks_to(path)
                            .iter()
                            .map(|n| n.path.clone())
                            .collect();
                        backlinks.sort();
                        (path.clone(), note.tags.clone(), backlinks)
                    })
                    .collect();
                notes.sort();
                drop(index);
                let mut lexical: Vec<_> = vault
                    .search_text("needle", 20)
                    .unwrap()
                    .into_iter()
                    .map(|hit| hit.path)
                    .collect();
                lexical.sort();
                let mut literal: Vec<_> = vault
                    .search_regex("need.e", 20)
                    .unwrap()
                    .into_iter()
                    .map(|hit| hit.path)
                    .collect();
                literal.sort();
                (notes, lexical, literal)
            };
            assert_eq!(projection(live), projection(&cold));
        }
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        let mut config = crate::test_helpers::tantivy_config(root);
        config.exclude_patterns = vec!["Archive/**".into()];
        std::fs::write(root.join("existing.bin"), [0xff, 0, 0x80]).unwrap();
        let live = Vault::open(&config).await.unwrap();
        live.create_note(Path::new("folder/anchor.md"), "#anchor needle", None)
            .unwrap();
        assert!(
            live.move_note(Path::new("folder"), Path::new("unused/subdir"))
                .is_err()
        );
        assert!(root.join("folder/anchor.md").is_file());
        assert!(!root.join("unused").exists());
        live.write_note(Path::new("item.txt"), "needle #tag [[anchor]]")
            .unwrap();
        assert!(live.get_note_metadata(Path::new("item.txt")).is_err());
        assert_matches_disk(&live, &config).await;
        live.append_note(Path::new("item.txt"), " more").unwrap();
        assert_matches_disk(&live, &config).await;
        for (from, to) in [
            ("item.txt", "item.MD"),
            ("item.MD", "Archive/hidden.md"),
            ("Archive/hidden.md", "visible.md"),
            ("visible.md", "last.txt"),
            ("last.txt", ".private/last.txt"),
            (".private/last.txt", "last.txt"),
        ] {
            live.move_note(Path::new(from), Path::new(to)).unwrap();
            assert_matches_disk(&live, &config).await;
        }
        live.delete_note(Path::new("last.txt")).unwrap();
        live.delete_note(Path::new("existing.bin")).unwrap();
        assert_matches_disk(&live, &config).await;
    }

    fn test_config_with_exclusions(vault_root: &Path, patterns: Vec<String>) -> Config {
        Config {
            exclude_patterns: patterns,
            ..test_config(vault_root)
        }
    }

    #[tokio::test]
    async fn move_into_excluded_dir() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let config = test_config_with_exclusions(dir.path(), vec!["Archive/**".into()]);
        let vault = Vault::open(&config).await.unwrap();

        vault
            .write_note(Path::new("active.md"), "# Active")
            .unwrap();
        assert!(vault.get_note_metadata(Path::new("active.md")).is_ok());

        let new_path = vault
            .move_note(Path::new("active.md"), Path::new("Archive/archived.md"))
            .unwrap();
        assert_eq!(new_path, PathBuf::from("Archive/archived.md"));

        assert!(vault.get_note_metadata(Path::new("active.md")).is_err());
        assert!(
            vault
                .get_note_metadata(Path::new("Archive/archived.md"))
                .is_err()
        );
        assert_eq!(vault.vault_stats().unwrap().excluded_notes, 1);

        let content = vault.read_note(Path::new("Archive/archived.md")).unwrap();
        assert_eq!(content, "# Active");
    }

    #[tokio::test]
    async fn move_out_of_excluded_dir() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        std::fs::create_dir_all(dir.path().join("Archive")).unwrap();
        std::fs::write(dir.path().join("Archive/hidden.md"), "# Hidden").unwrap();

        let config = test_config_with_exclusions(dir.path(), vec!["Archive/**".into()]);
        let vault = Vault::open(&config).await.unwrap();

        assert!(
            vault
                .get_note_metadata(Path::new("Archive/hidden.md"))
                .is_err()
        );

        let new_path = vault
            .move_note(Path::new("Archive/hidden.md"), Path::new("visible.md"))
            .unwrap();
        assert_eq!(new_path, PathBuf::from("visible.md"));

        assert!(vault.get_note_metadata(Path::new("visible.md")).is_ok());
        assert!(
            vault
                .get_note_metadata(Path::new("Archive/hidden.md"))
                .is_err()
        );
        assert_eq!(vault.vault_stats().unwrap().excluded_notes, 0);
    }

    #[tokio::test]
    async fn move_between_excluded_dirs() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        std::fs::create_dir_all(dir.path().join("Archive")).unwrap();
        std::fs::write(dir.path().join("Archive/old.md"), "# Old").unwrap();

        let config =
            test_config_with_exclusions(dir.path(), vec!["Archive/**".into(), "Trash/**".into()]);
        let vault = Vault::open(&config).await.unwrap();

        assert!(
            vault
                .get_note_metadata(Path::new("Archive/old.md"))
                .is_err()
        );

        let new_path = vault
            .move_note(Path::new("Archive/old.md"), Path::new("Trash/old.md"))
            .unwrap();
        assert_eq!(new_path, PathBuf::from("Trash/old.md"));

        assert!(
            vault
                .get_note_metadata(Path::new("Archive/old.md"))
                .is_err()
        );
        assert!(vault.get_note_metadata(Path::new("Trash/old.md")).is_err());

        let content = vault.read_note(Path::new("Trash/old.md")).unwrap();
        assert_eq!(content, "# Old");
    }

    #[tokio::test]
    async fn move_into_excluded_dir_with_tantivy() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let config = Config {
            tantivy: true,
            exclude_patterns: vec!["Archive/**".into()],
            ..test_config(dir.path())
        };
        let vault = Vault::open(&config).await.unwrap();

        vault
            .write_note(Path::new("indexed.md"), "# Indexed note with content")
            .unwrap();
        assert!(vault.get_note_metadata(Path::new("indexed.md")).is_ok());

        vault
            .move_note(Path::new("indexed.md"), Path::new("Archive/gone.md"))
            .unwrap();

        assert!(vault.get_note_metadata(Path::new("indexed.md")).is_err());
        assert!(
            vault
                .get_note_metadata(Path::new("Archive/gone.md"))
                .is_err()
        );

        let results = vault.search_text("Indexed note", 100).unwrap();
        assert!(
            results.is_empty(),
            "moved-to-excluded note should not appear in search"
        );
    }

    #[tokio::test]
    async fn write_note_to_excluded_path_does_not_index() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let config = test_config_with_exclusions(dir.path(), vec!["Archive/**".into()]);
        let vault = Vault::open(&config).await.unwrap();

        vault
            .write_note(Path::new("Archive/direct.md"), "# Direct excluded")
            .unwrap();

        assert!(
            vault
                .get_note_metadata(Path::new("Archive/direct.md"))
                .is_err()
        );
        assert_eq!(vault.vault_stats().unwrap().excluded_notes, 1);
        assert_eq!(
            vault.read_note(Path::new("Archive/direct.md")).unwrap(),
            "# Direct excluded"
        );

        vault.delete_note(Path::new("Archive/direct.md")).unwrap();
        assert_eq!(vault.vault_stats().unwrap().excluded_notes, 0);
    }

    #[tokio::test]
    async fn mcp_data_equal_to_mcp_home_uses_default_location() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());
        let mcp_home = dir.path().join(".obsidian-mcp");
        let config = Config {
            mcp_data_dir: Some(mcp_home.clone()),
            ..test_config(dir.path())
        };

        let vault = Vault::open(&config).await.unwrap();

        assert_eq!(vault.mcp_home(), vault.root().join(".obsidian-mcp"));
        assert_eq!(vault.mcp_data(), vault.mcp_home());
        assert!(
            !vault.mcp_home().join("vaults").exists(),
            "default mcp home must not be namespaced under itself"
        );
    }

    #[tokio::test]
    async fn concurrent_transformations_preserve_every_successful_edit() {
        let dir = tempfile::tempdir().unwrap();
        let vault = Vault::open(&tantivy_config(dir.path())).await.unwrap();
        vault
            .write_note(Path::new("shared.md"), &"body\n".repeat(20000))
            .unwrap();
        let barrier = Arc::new(std::sync::Barrier::new(8));
        let workers: Vec<_> = (0..8)
            .map(|id| {
                let vault = vault.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    vault
                        .prepend_note(Path::new("shared.md"), &format!("marker-{id}\n"))
                        .unwrap();
                    vault
                        .set_frontmatter_field(
                            Path::new("shared.md"),
                            &format!("key-{id}"),
                            id.into(),
                        )
                        .unwrap();
                })
            })
            .collect();
        for worker in workers {
            worker.join().unwrap();
        }
        let content = vault.read_note(Path::new("shared.md")).unwrap();
        let fm = vault
            .get_frontmatter(Path::new("shared.md"))
            .unwrap()
            .unwrap();
        for id in 0..8 {
            assert!(content.contains(&format!("marker-{id}\n")));
            assert_eq!(fm[format!("key-{id}")], id);
        }
    }

    #[tokio::test]
    async fn invalid_yaml_write_preserves_file_and_all_indexes() {
        let dir = tempfile::tempdir().unwrap();
        let vault = Vault::open(&tantivy_config(dir.path())).await.unwrap();
        let content = "---\ntags: [Work]\n---\nquokka\n";
        vault.write_note(Path::new("note.md"), content).unwrap();
        let stats = vault.vault_stats().unwrap();
        assert!(
            vault
                .write_note(Path::new("note.md"), "---\ntags: [unclosed\n---\nchanged")
                .is_err()
        );
        assert_eq!(vault.read_note(Path::new("note.md")).unwrap(), content);
        assert_eq!(
            vault.get_note_metadata(Path::new("note.md")).unwrap().tags,
            vec!["Work"]
        );
        assert_eq!(vault.search_text("quokka", 0).unwrap().len(), 1);
        assert_eq!(vault.vault_stats().unwrap().total_notes, stats.total_notes);
    }

    #[tokio::test]
    async fn hidden_writes_and_moves_remain_outside_indexes() {
        let dir = tempfile::tempdir().unwrap();
        let vault = Vault::open(&tantivy_config(dir.path())).await.unwrap();
        vault
            .write_note(Path::new(".secret/hidden.md"), "quokka")
            .unwrap();
        assert!(
            vault
                .get_note_metadata(Path::new(".secret/hidden.md"))
                .is_err()
        );
        vault.write_note(Path::new("visible.md"), "quokka").unwrap();
        vault
            .move_note(Path::new("visible.md"), Path::new("notes/.hidden.md"))
            .unwrap();
        assert!(vault.search_text("quokka", 0).unwrap().is_empty());
        assert_eq!(vault.vault_stats().unwrap().total_notes, 0);
    }

    #[tokio::test]
    async fn tag_identity_is_case_insensitive_across_updates_and_search_fields() {
        let dir = tempfile::tempdir().unwrap();
        let vault = Vault::open(&tantivy_config(dir.path())).await.unwrap();
        vault
            .write_note(Path::new("a.md"), "---\ntags: [Work]\n---\n#WORK/Inbox")
            .unwrap();
        assert_eq!(vault.search_by_tag("work").unwrap().len(), 1);
        assert_eq!(vault.search_by_tag_prefix("wOrK").unwrap().len(), 1);
        assert_eq!(
            vault
                .search_text_with_options("WORK", 0, 10, false, Some(&[SearchField::Tags]))
                .unwrap()
                .len(),
            1
        );
        vault.write_note(Path::new("a.md"), "#work").unwrap();
        assert_eq!(vault.search_by_tag("WORK").unwrap().len(), 1);
        assert!(vault.search_by_tag("work/inbox").unwrap().is_empty());
        vault.delete_note(Path::new("a.md")).unwrap();
        assert!(vault.search_by_tag("work").unwrap().is_empty());
    }
}
