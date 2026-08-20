//! Vault layer — pure filesystem operations on an Obsidian vault.
//!
//! Knows nothing about MCP; provides the data model and I/O primitives
//! that tool handlers delegate to.

pub mod exclude;
pub mod frontmatter;
pub mod fs;
pub mod index;
pub mod parser;
pub mod patch;
pub mod path;
pub mod periodic;
pub mod search_utils;
pub mod tantivy_index;
pub mod watcher;
pub mod wikilink;

#[cfg(has_embeddings)]
pub mod embedding_runtime;
#[cfg(has_embeddings)]
pub mod embeddings;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};

use chrono::{Local, NaiveDate};
use notify_debouncer_mini::Debouncer;

use crate::config::Config;
use crate::error::{VaultError, VaultResult};
use crate::models::{
    DocumentMap, NoteMetadata, NotePeriod, PatchRequest, SearchField, SearchMatch, SearchResult,
    VaultStats, WikiLink,
};

use self::exclude::ExcludeSet;
use self::index::VaultIndex;
use self::tantivy_index::TantivyIndex;

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
    root: PathBuf,
    mcp_home: PathBuf,
    mcp_data: PathBuf,
    exclude: Arc<ExcludeSet>,
    index: Arc<RwLock<VaultIndex>>,
    tantivy: Option<Arc<TantivyIndex>>,
    #[cfg(has_embeddings)]
    embedding_runtime: Option<embedding_runtime::EmbeddingRuntime>,
    /// Kept alive to sustain filesystem watching; never accessed after construction.
    /// Wrapped in `Mutex` to guarantee `Sync` (`Debouncer` contains a `mpsc::Sender`
    /// which is `Send` but not `Sync`).
    _watcher: Mutex<Option<Debouncer<notify::RecommendedWatcher>>>,
}

/// High-level facade over the vault filesystem, index, and watcher.
///
/// `Vault` is `Clone + Send + Sync` — cloning increments an internal `Arc`.
/// All read operations acquire a shared lock on the index; write operations
/// acquire an exclusive lock briefly after the filesystem mutation.
#[derive(Clone)]
pub struct Vault {
    inner: Arc<VaultInner>,
}

impl Vault {
    /// Open a vault: validate the path, build the index, and optionally start the watcher.
    pub async fn open(config: &Config) -> VaultResult<Self> {
        Self::open_with_embedding_loader(config, None).await
    }

    async fn open_with_embedding_loader(
        config: &Config,
        _embedding_loader: Option<EmbeddingLoaderFuture>,
    ) -> VaultResult<Self> {
        let root = config.vault_path.canonicalize().map_err(|_| {
            VaultError::InvalidPath(format!(
                "vault path does not exist: {}",
                config.vault_path.display()
            ))
        })?;

        if !root.join(".obsidian").is_dir() {
            tracing::warn!(
                path = %root.display(),
                "vault has no .obsidian/ directory — this may be a fresh vault"
            );
        }

        let ensure_dir = |p: &Path| -> VaultResult<()> {
            std::fs::create_dir_all(p).map_err(|e| {
                VaultError::Io(std::io::Error::new(
                    e.kind(),
                    format!("failed to create {}: {e}", p.display()),
                ))
            })
        };

        // ── metadata folder setup ──
        let mcp_home = root.join(".obsidian-mcp");
        ensure_dir(&mcp_home)?;

        let mcp_home_canonical = mcp_home.canonicalize().ok();
        let mcp_data = if let Some(ref data_dir) = config.mcp_data_dir {
            let data_dir_canonical = data_dir.canonicalize().ok();
            if data_dir == &mcp_home || data_dir_canonical.as_ref() == mcp_home_canonical.as_ref() {
                tracing::debug!("OBSIDIAN_MCP_DATA resolves to .obsidian-mcp, treating as unset");
                mcp_home.clone()
            } else {
                let slug = crate::config::vault_slug(&root);
                let candidate = data_dir.join("vaults").join(&slug);
                if candidate.starts_with(&root) {
                    tracing::warn!(
                        path = %candidate.display(),
                        "OBSIDIAN_MCP_DATA resolves to inside the vault — \
                         consider a path outside the vault for cloud sync benefits"
                    );
                }
                candidate
            }
        } else {
            mcp_home.clone()
        };

        let ignore_path = mcp_home.join("ignore");
        if !ignore_path.exists()
            && let Err(e) = std::fs::write(&ignore_path, "")
        {
            tracing::warn!(path = %ignore_path.display(), error = %e,
                "failed to create default ignore file");
        }

        if mcp_data != mcp_home {
            ensure_dir(&mcp_data)?;
        }

        ensure_dir(&mcp_data.join("embeddings"))?;

        #[cfg(has_embeddings)]
        {
            let legacy_cache = root
                .join(".obsidian")
                .join("obsidian-mcp")
                .join("embeddings.bin");
            let new_cache = Self::embedding_cache_path(&mcp_data);
            if legacy_cache.is_file() && !new_cache.exists() {
                match std::fs::copy(&legacy_cache, &new_cache) {
                    Ok(bytes) => {
                        tracing::info!(
                            from = %legacy_cache.display(),
                            to = %new_cache.display(),
                            bytes,
                            "migrated legacy embedding cache to new location"
                        );
                    }
                    Err(e) => {
                        tracing::warn!(
                            from = %legacy_cache.display(),
                            to = %new_cache.display(),
                            error = %e,
                            "failed to migrate legacy embedding cache (non-fatal)"
                        );
                    }
                }
            }
        }

        // ── exclusion patterns ──
        let mut patterns = exclude::load_ignore_patterns(&mcp_home, &mcp_data);
        patterns.extend(config.exclude_patterns.iter().cloned());
        patterns.sort();
        patterns.dedup();

        let exclude = Arc::new(exclude::ExcludeSet::build(patterns)?);

        if !exclude.is_empty() {
            tracing::info!(
                patterns = ?exclude.patterns(),
                "path exclusion active"
            );
        }

        let vi = VaultIndex::build(&root, Arc::clone(&exclude)).await?;

        let tantivy = if config.tantivy {
            let tv = TantivyIndex::build(&root, vi.notes())?;
            tracing::info!(notes = vi.notes().len(), "tantivy BM25 index built");
            Some(Arc::new(tv))
        } else {
            None
        };

        let index = Arc::new(RwLock::new(vi));

        #[cfg(has_embeddings)]
        let embedding_runtime = if config.embeddings {
            let model_name = config.embeddings_model.clone();
            let provider = config.embedding_provider;
            let cache_path = Self::embedding_cache_path(&mcp_data);
            let loader = _embedding_loader.unwrap_or_else(|| {
                Box::pin(async move {
                    let model = embeddings::EmbeddingModel::load(&model_name, provider).await?;
                    Ok(Arc::new(model) as Arc<dyn embeddings::Embedder>)
                })
            });
            Some(embedding_runtime::EmbeddingRuntime::spawn(
                root.clone(),
                Arc::clone(&index),
                cache_path,
                loader,
            ))
        } else {
            None
        };

        let watcher_handle = if config.watch {
            #[cfg(has_embeddings)]
            let debouncer = watcher::start_watcher(
                root.clone(),
                Arc::clone(&index),
                tantivy.clone(),
                embedding_runtime.clone(),
                Arc::clone(&exclude),
            )?;
            #[cfg(not(has_embeddings))]
            let debouncer = watcher::start_watcher(
                root.clone(),
                Arc::clone(&index),
                tantivy.clone(),
                Arc::clone(&exclude),
            )?;
            Some(debouncer)
        } else {
            None
        };

        Ok(Self {
            inner: Arc::new(VaultInner {
                root,
                mcp_home,
                mcp_data,
                exclude,
                index,
                tantivy,
                #[cfg(has_embeddings)]
                embedding_runtime,
                _watcher: Mutex::new(watcher_handle),
            }),
        })
    }

    /// Vault root path (canonicalized).
    pub fn root(&self) -> &Path {
        &self.inner.root
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
        fs::list_files(&self.inner.root, dir, recursive, glob)
    }

    pub fn read_note(&self, path: &Path) -> VaultResult<String> {
        fs::read_file(&self.inner.root, path)
    }

    pub fn write_note(&self, path: &Path, content: &str) -> VaultResult<()> {
        let actual_path = fs::write_file(&self.inner.root, path, content)?;
        self.reindex(&actual_path)?;
        Ok(())
    }

    pub fn append_note(&self, path: &Path, content: &str) -> VaultResult<()> {
        let actual_path = fs::append_file(&self.inner.root, path, content)?;
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
        if fs::file_exists(&self.inner.root, path) {
            return Err(VaultError::AlreadyExists(path.to_path_buf()));
        }
        if let Some(value) = frontmatter
            && !value.is_object()
        {
            return Err(VaultError::InvalidFrontmatter {
                path: path.to_path_buf(),
                actual: frontmatter::json_type_name(value),
            });
        }
        let full_content = frontmatter::rebuild_content(frontmatter, content);
        let actual_path = fs::write_file(&self.inner.root, path, &full_content)?;
        self.reindex(&actual_path)?;
        Ok(())
    }

    /// Prepend content after frontmatter (or at the start if none exists).
    pub fn prepend_note(&self, path: &Path, content: &str) -> VaultResult<()> {
        let existing = fs::read_file(&self.inner.root, path)?;
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
        let actual_path = fs::write_file(&self.inner.root, path, &new_content)?;
        self.reindex(&actual_path)?;
        Ok(())
    }

    pub fn delete_note(&self, path: &Path) -> VaultResult<()> {
        let actual_path = fs::delete_file(&self.inner.root, path)?;
        self.write_index().remove_file(&actual_path);
        if let Some(tv) = &self.inner.tantivy {
            tv.remove_file(&actual_path)?;
        }
        #[cfg(has_embeddings)]
        if let Some(runtime) = &self.inner.embedding_runtime {
            runtime.submit_remove(&actual_path);
        }
        Ok(())
    }

    pub fn move_note(&self, from: &Path, to: &Path) -> VaultResult<PathBuf> {
        let move_result = fs::move_file(&self.inner.root, from, to)?;
        let old_path = move_result.from;
        let new_path = move_result.to;

        if self.inner.exclude.is_excluded(&new_path) {
            {
                let mut idx = self.write_index();
                idx.remove_file(&old_path);
                idx.add_excluded_file(&new_path);
                if let Some(tv) = &self.inner.tantivy {
                    tv.remove_file(&old_path)?;
                }
            }
            #[cfg(has_embeddings)]
            if let Some(runtime) = &self.inner.embedding_runtime {
                runtime.submit_remove(&old_path);
                runtime.submit_remove(&new_path);
            }
        } else {
            {
                let mut idx = self.write_index();
                idx.rename_file(&self.inner.root, &old_path, &new_path)?;
                if let Some(tv) = &self.inner.tantivy {
                    tv.remove_file(&old_path)?;
                    if let Some(meta) = idx.get_note(&new_path) {
                        tv.reindex_file(&self.inner.root, &new_path, meta)?;
                    }
                }
            }
            #[cfg(has_embeddings)]
            if let Some(runtime) = &self.inner.embedding_runtime {
                runtime.submit_remove(&old_path);
                runtime.submit_upsert(&new_path);
            }
        }

        Ok(new_path)
    }

    // ── patch delegation ───────────────────────────────────────────────

    pub fn patch_note(&self, path: &Path, request: &PatchRequest) -> VaultResult<()> {
        let content = fs::read_file(&self.inner.root, path)?;
        let patched = patch::apply_patch(&content, request, path)?;
        let actual_path = fs::write_file(&self.inner.root, path, &patched)?;
        self.reindex(&actual_path)?;
        Ok(())
    }

    // ── frontmatter delegation ─────────────────────────────────────────

    pub fn get_frontmatter(&self, path: &Path) -> VaultResult<Option<serde_json::Value>> {
        let content = fs::read_file(&self.inner.root, path)?;
        frontmatter::parse_frontmatter(&content)
    }

    pub fn set_frontmatter_field(
        &self,
        path: &Path,
        key: &str,
        value: serde_json::Value,
    ) -> VaultResult<()> {
        let content = fs::read_file(&self.inner.root, path)?;
        let updated = frontmatter::set_frontmatter_field(&content, key, value)?;
        let actual_path = fs::write_file(&self.inner.root, path, &updated)?;
        self.reindex(&actual_path)?;
        Ok(())
    }

    pub fn remove_frontmatter_field(&self, path: &Path, key: &str) -> VaultResult<()> {
        let content = fs::read_file(&self.inner.root, path)?;
        let updated = frontmatter::remove_frontmatter_field(&content, key)?;
        let actual_path = fs::write_file(&self.inner.root, path, &updated)?;
        self.reindex(&actual_path)?;
        Ok(())
    }

    // ── index delegation (read-lock) ───────────────────────────────────

    pub fn get_note_metadata(&self, path: &Path) -> VaultResult<NoteMetadata> {
        let actual_path = self.canonical_existing_relative_path(path)?;
        self.read_index()
            .get_note(&actual_path)
            .cloned()
            .ok_or(VaultError::NoteNotFound(actual_path))
    }

    pub fn get_document_map(&self, path: &Path) -> VaultResult<DocumentMap> {
        let content = fs::read_file(&self.inner.root, path)?;
        Ok(parser::build_document_map(&content))
    }

    pub fn search_text(&self, query: &str, context_len: usize) -> VaultResult<Vec<SearchResult>> {
        match &self.inner.tantivy {
            Some(tv) => self.tantivy_search_with_context(tv, query, context_len, 200, false, None),
            None => self
                .read_index()
                .search_text(&self.inner.root, query, context_len),
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
            None => {
                let mut results =
                    self.read_index()
                        .search_text(&self.inner.root, query, context_len)?;
                results.truncate(max_results);
                Ok(results)
            }
        }
    }

    pub fn search_regex(
        &self,
        pattern: &str,
        context_len: usize,
    ) -> VaultResult<Vec<SearchResult>> {
        self.read_index()
            .search_regex(&self.inner.root, pattern, context_len, 0)
    }

    /// Semantic search via embedding cosine similarity (Layer 2).
    ///
    /// Embeds the query, then performs brute-force cosine similarity against
    /// the embedding store. Returns `(path, score)` pairs sorted by descending
    /// similarity.
    #[cfg(has_embeddings)]
    pub fn search_semantic(&self, query: &str, top_k: usize) -> VaultResult<Vec<(PathBuf, f32)>> {
        let runtime = self.inner.embedding_runtime.as_ref().ok_or_else(|| {
            VaultError::Embedding("embeddings not enabled (OBSIDIAN_EMBEDDINGS=false)".into())
        })?;
        let snapshot = runtime.query_snapshot()?;
        let mut results = snapshot.semantic_scores(query, usize::MAX)?;
        let index = self.read_index();
        results.retain(|(path, _)| index.get_note(path).is_some());
        results.truncate(top_k);
        Ok(results)
    }

    /// Returns `true` if embeddings are available for semantic search.
    #[cfg(has_embeddings)]
    pub fn has_embeddings(&self) -> bool {
        self.inner
            .embedding_runtime
            .as_ref()
            .is_some_and(|runtime| runtime.status().queryable)
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
        if query.is_empty() {
            return Ok(Vec::new());
        }

        let tv = self.inner.tantivy.as_ref().ok_or_else(|| {
            VaultError::Other("hybrid search requires Tantivy (set OBSIDIAN_TANTIVY=true)".into())
        })?;
        let runtime = self.inner.embedding_runtime.as_ref().ok_or_else(|| {
            VaultError::Embedding("embeddings not enabled (OBSIDIAN_EMBEDDINGS=false)".into())
        })?;
        let snapshot = runtime.query_snapshot()?;

        let prefetch = prefetch_count.max(top_k);
        let bm25_hits = tv.search(query, prefetch)?;
        if bm25_hits.is_empty() {
            return Ok(Vec::new());
        }

        let query_vec = snapshot.embed_query(query)?;
        let index = self.read_index();

        let norm_bm25 = search_utils::normalize_bm25_scores(&bm25_hits);

        let mut combined: Vec<(PathBuf, f32)> = norm_bm25
            .into_iter()
            .filter(|(path, _)| index.get_note(path).is_some())
            .map(|(path, norm_score)| {
                let semantic_score = snapshot.score_for(&path, &query_vec);
                let final_score = alpha * norm_score + (1.0 - alpha) * semantic_score;
                (path, final_score)
            })
            .collect();

        combined
            .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        combined.truncate(top_k);
        Ok(combined)
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
        Ok(path::resolve_existing(&self.inner.root, path)?.relative)
    }

    /// Validate that a relative path doesn't escape the vault root.
    pub fn validate_path(&self, path: &Path) -> VaultResult<()> {
        fs::resolve_path(&self.inner.root, path)?;
        Ok(())
    }

    // ── periodic delegation ────────────────────────────────────────────

    pub fn get_periodic_note(
        &self,
        period: &NotePeriod,
        date: Option<NaiveDate>,
    ) -> VaultResult<String> {
        let config = periodic::read_periodic_config(&self.inner.root, period)?;
        let date = date.unwrap_or_else(|| Local::now().date_naive());
        let path = periodic::periodic_note_path(&config, &date);
        fs::read_file(&self.inner.root, &path)
    }

    pub fn create_periodic_note(
        &self,
        period: &NotePeriod,
        date: Option<NaiveDate>,
        content_override: Option<&str>,
    ) -> VaultResult<PathBuf> {
        let config = periodic::read_periodic_config(&self.inner.root, period)?;
        let date = date.unwrap_or_else(|| Local::now().date_naive());
        let path = periodic::periodic_note_path(&config, &date);

        if fs::file_exists(&self.inner.root, &path) {
            return Err(VaultError::AlreadyExists(path));
        }

        let content = if let Some(custom) = content_override {
            custom.to_owned()
        } else {
            match &config.template {
                Some(tmpl) if !tmpl.is_empty() => {
                    let title = path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or_default();
                    match periodic::expand_template(&self.inner.root, Path::new(tmpl), &date, title)
                    {
                        Ok(c) => c,
                        Err(VaultError::NoteNotFound(_)) => String::new(),
                        Err(e) => return Err(e),
                    }
                }
                _ => String::new(),
            }
        };

        let actual_path = fs::write_file(&self.inner.root, &path, &content)?;
        self.reindex(&actual_path)?;
        Ok(actual_path)
    }

    pub fn list_recent_periodic_notes(
        &self,
        period: &NotePeriod,
        limit: usize,
    ) -> VaultResult<Vec<PathBuf>> {
        let config = periodic::read_periodic_config(&self.inner.root, period)?;
        periodic::list_recent_periodic_notes(&self.inner.root, &config, limit)
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

        let word_re = search_utils::compile_query_word_regex(query);

        let mut results = Vec::with_capacity(hits.len());
        for (path, score) in hits {
            let matches = match (word_re.as_ref(), fs::read_file(&self.inner.root, &path)) {
                (Some(re), Ok(content)) if context_len > 0 => re
                    .find_iter(&content)
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
                    .collect(),
                _ => Vec::new(),
            };

            results.push(SearchResult {
                path,
                matches,
                score: Some(score as f64),
            });
        }

        Ok(results)
    }

    fn read_index(&self) -> std::sync::RwLockReadGuard<'_, VaultIndex> {
        self.inner.index.read().unwrap_or_else(|e| e.into_inner())
    }

    fn write_index(&self) -> std::sync::RwLockWriteGuard<'_, VaultIndex> {
        self.inner.index.write().unwrap_or_else(|e| e.into_inner())
    }

    fn reindex(&self, path: &Path) -> VaultResult<()> {
        let actual_path = self.canonical_existing_relative_path(path)?;
        if self.inner.exclude.is_excluded(&actual_path) {
            self.write_index().add_excluded_file(&actual_path);
            if let Some(tv) = &self.inner.tantivy {
                tv.remove_file(&actual_path)?;
            }
            #[cfg(has_embeddings)]
            if let Some(runtime) = &self.inner.embedding_runtime {
                runtime.submit_remove(&actual_path);
            }
            return Ok(());
        }

        let mut idx = self.write_index();
        idx.reindex_file(&self.inner.root, &actual_path)?;
        if let Some(tv) = &self.inner.tantivy
            && let Some(meta) = idx.get_note(&actual_path)
        {
            tv.reindex_file(&self.inner.root, &actual_path, meta)?;
        }
        drop(idx);
        #[cfg(has_embeddings)]
        if let Some(runtime) = &self.inner.embedding_runtime {
            runtime.submit_upsert(&actual_path);
        }
        Ok(())
    }

    // ── embedding helpers (feature-gated) ─────────────────────────────

    #[cfg(has_embeddings)]
    fn embedding_cache_path(mcp_data: &Path) -> PathBuf {
        mcp_data.join("embeddings").join("embeddings.bin")
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::models::{PatchOperation, PatchTargetType};
    use crate::test_helpers::{create_test_vault, tantivy_config, test_config};
    use unicode_normalization::UnicodeNormalization;

    #[cfg(has_embeddings)]
    struct TestEmbedder {
        identity: embeddings::EmbeddingSpaceIdentity,
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
                },
            }
        }
    }

    #[cfg(has_embeddings)]
    impl embeddings::Embedder for TestEmbedder {
        fn dimension(&self) -> usize {
            self.identity.dimension
        }

        fn space_identity(&self) -> &embeddings::EmbeddingSpaceIdentity {
            &self.identity
        }

        fn embed_batch(&self, texts: &[&str]) -> VaultResult<Vec<Vec<f32>>> {
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
        assert!(
            results
                .iter()
                .any(|r| r.path == PathBuf::from("searchable.md"))
        );
        assert!(!results.iter().any(|r| r.path == PathBuf::from("other.md")));
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
        assert!(bl.iter().any(|m| m.path == PathBuf::from("a.md")));
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
            results.iter().any(|r| r.path == PathBuf::from("target.md")),
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
                .any(|r| r.path == PathBuf::from("headingmatch.md"))
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
            results
                .iter()
                .any(|r| r.path == PathBuf::from("fallback.md")),
            "should still find results via regex fallback when tantivy is disabled"
        );
        assert!(
            results[0].score.is_none(),
            "regex fallback should not populate BM25 scores"
        );
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
        assert_eq!(
            runtime.pending_kind(excluded_path),
            Some(PendingKind::Remove)
        );

        let old_path = Path::new("old.md");
        let new_path = Path::new("nested/new.md");
        vault.write_note(old_path, "# Move\n").unwrap();
        vault.move_note(old_path, new_path).unwrap();
        assert_eq!(runtime.pending_kind(old_path), Some(PendingKind::Remove));
        assert_eq!(runtime.pending_kind(new_path), Some(PendingKind::Upsert));
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
            .join("embeddings")
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
    async fn legacy_embedding_cache_migrated_to_new_location() {
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
            new_path.exists(),
            "migration should copy cache to new location"
        );
        assert_eq!(
            std::fs::read(&new_path).unwrap(),
            test_bytes,
            "migrated file should have identical content"
        );
        assert!(
            legacy_path.exists(),
            "legacy file must not be deleted by migration"
        );
    }

    #[cfg(has_embeddings)]
    #[tokio::test]
    async fn legacy_embedding_migration_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        create_test_vault(dir.path());

        let legacy_dir = dir.path().join(".obsidian").join("obsidian-mcp");
        std::fs::create_dir_all(&legacy_dir).unwrap();
        std::fs::write(legacy_dir.join("embeddings.bin"), b"old-legacy-data").unwrap();

        let new_dir = dir.path().join(".obsidian-mcp").join("embeddings");
        std::fs::create_dir_all(&new_dir).unwrap();
        let new_bytes = b"already-migrated-data";
        std::fs::write(new_dir.join("embeddings.bin"), new_bytes).unwrap();

        let _vault = Vault::open(&test_config(dir.path())).await.unwrap();

        assert_eq!(
            std::fs::read(new_dir.join("embeddings.bin")).unwrap(),
            new_bytes,
            "existing new cache must not be overwritten by legacy"
        );
        assert!(
            legacy_dir.join("embeddings.bin").exists(),
            "legacy file must not be deleted"
        );
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
}
