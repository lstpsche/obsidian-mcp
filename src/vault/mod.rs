//! Vault layer — pure filesystem operations on an Obsidian vault.
//!
//! Knows nothing about MCP; provides the data model and I/O primitives
//! that tool handlers delegate to.

pub mod frontmatter;
pub mod fs;
pub mod index;
pub mod parser;
pub mod patch;
pub mod periodic;
pub mod search_utils;
pub mod tantivy_index;
pub mod watcher;
pub mod wikilink;

#[cfg(feature = "embeddings")]
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

use self::index::VaultIndex;
use self::tantivy_index::TantivyIndex;

/// Internal shared state wrapped in `Arc` for cheap cloning.
struct VaultInner {
    root: PathBuf,
    index: Arc<RwLock<VaultIndex>>,
    tantivy: Option<Arc<TantivyIndex>>,
    #[cfg(feature = "embeddings")]
    embedding_model: Option<Arc<embeddings::EmbeddingModel>>,
    #[cfg(feature = "embeddings")]
    embedding_store: Option<Arc<RwLock<embeddings::EmbeddingStore>>>,
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

        let vi = VaultIndex::build(&root).await?;

        let tantivy = if config.tantivy {
            let tv = TantivyIndex::build(&root, vi.notes())?;
            tracing::info!(notes = vi.notes().len(), "tantivy BM25 index built");
            Some(Arc::new(tv))
        } else {
            None
        };

        let index = Arc::new(RwLock::new(vi));

        #[cfg(feature = "embeddings")]
        let (embedding_model, embedding_store) = if config.embeddings {
            let model = embeddings::EmbeddingModel::load(&config.embeddings_model).await?;
            let model = Arc::new(model);
            let store = Self::build_or_load_embeddings(&root, &index, &model)?;
            let store = Arc::new(RwLock::new(store));
            tracing::info!(
                notes = index
                    .read()
                    .unwrap_or_else(|e| e.into_inner())
                    .notes()
                    .len(),
                dim = model.dim(),
                "embedding store ready"
            );
            (Some(model), Some(store))
        } else {
            (None, None)
        };

        let watcher_handle = if config.watch {
            #[cfg(feature = "embeddings")]
            let debouncer = watcher::start_watcher(
                root.clone(),
                Arc::clone(&index),
                tantivy.clone(),
                embedding_model.clone(),
                embedding_store.clone(),
            )?;
            #[cfg(not(feature = "embeddings"))]
            let debouncer =
                watcher::start_watcher(root.clone(), Arc::clone(&index), tantivy.clone())?;
            Some(debouncer)
        } else {
            None
        };

        Ok(Self {
            inner: Arc::new(VaultInner {
                root,
                index,
                tantivy,
                #[cfg(feature = "embeddings")]
                embedding_model,
                #[cfg(feature = "embeddings")]
                embedding_store,
                _watcher: Mutex::new(watcher_handle),
            }),
        })
    }

    /// Vault root path (canonicalized).
    pub fn root(&self) -> &Path {
        &self.inner.root
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
        fs::write_file(&self.inner.root, path, content)?;
        self.reindex(path)?;
        Ok(())
    }

    pub fn append_note(&self, path: &Path, content: &str) -> VaultResult<()> {
        fs::append_file(&self.inner.root, path, content)?;
        self.reindex(path)?;
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
        let full_content = frontmatter::rebuild_content(frontmatter, content);
        fs::write_file(&self.inner.root, path, &full_content)?;
        self.reindex(path)?;
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
        fs::write_file(&self.inner.root, path, &new_content)?;
        self.reindex(path)?;
        Ok(())
    }

    pub fn delete_note(&self, path: &Path) -> VaultResult<()> {
        fs::delete_file(&self.inner.root, path)?;
        self.write_index().remove_file(path);
        if let Some(tv) = &self.inner.tantivy {
            tv.remove_file(path)?;
        }
        #[cfg(feature = "embeddings")]
        self.remove_embedding(path);
        Ok(())
    }

    pub fn move_note(&self, from: &Path, to: &Path) -> VaultResult<PathBuf> {
        let new_path = fs::move_file(&self.inner.root, from, to)?;
        {
            let mut idx = self.write_index();
            idx.rename_file(&self.inner.root, from, &new_path)?;
            if let Some(tv) = &self.inner.tantivy {
                tv.remove_file(from)?;
                if let Some(meta) = idx.get_note(&new_path) {
                    tv.reindex_file(&self.inner.root, &new_path, meta)?;
                }
            }
        }
        #[cfg(feature = "embeddings")]
        {
            self.remove_embedding(from);
            self.reindex_embedding(&new_path);
        }
        Ok(new_path)
    }

    // ── patch delegation ───────────────────────────────────────────────

    pub fn patch_note(&self, path: &Path, request: &PatchRequest) -> VaultResult<()> {
        let content = fs::read_file(&self.inner.root, path)?;
        let patched = patch::apply_patch(&content, request, path)?;
        fs::write_file(&self.inner.root, path, &patched)?;
        self.reindex(path)?;
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
        fs::write_file(&self.inner.root, path, &updated)?;
        self.reindex(path)?;
        Ok(())
    }

    pub fn remove_frontmatter_field(&self, path: &Path, key: &str) -> VaultResult<()> {
        let content = fs::read_file(&self.inner.root, path)?;
        let updated = frontmatter::remove_frontmatter_field(&content, key)?;
        fs::write_file(&self.inner.root, path, &updated)?;
        self.reindex(path)?;
        Ok(())
    }

    // ── index delegation (read-lock) ───────────────────────────────────

    pub fn get_note_metadata(&self, path: &Path) -> VaultResult<NoteMetadata> {
        self.read_index()
            .get_note(path)
            .cloned()
            .ok_or_else(|| VaultError::NoteNotFound(path.to_path_buf()))
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
    #[cfg(feature = "embeddings")]
    pub fn search_semantic(&self, query: &str, top_k: usize) -> VaultResult<Vec<(PathBuf, f32)>> {
        let model = self.inner.embedding_model.as_ref().ok_or_else(|| {
            VaultError::Embedding("embeddings not enabled (OBSIDIAN_EMBEDDINGS=false)".into())
        })?;
        let store = self
            .inner
            .embedding_store
            .as_ref()
            .ok_or_else(|| VaultError::Embedding("embedding store not initialized".into()))?;

        let query_vec = model.embed_one(query)?;
        let s = store.read().unwrap_or_else(|e| e.into_inner());
        Ok(s.query(&query_vec, top_k))
    }

    /// Returns `true` if embeddings are available for semantic search.
    #[cfg(feature = "embeddings")]
    pub fn has_embeddings(&self) -> bool {
        self.inner.embedding_model.is_some() && self.inner.embedding_store.is_some()
    }

    /// Hybrid search: BM25 prefetch via Tantivy, then re-rank by combining
    /// normalized BM25 scores with semantic cosine similarity.
    ///
    /// Requires both Tantivy and embeddings to be enabled.
    ///
    /// `alpha` controls the balance: `final = alpha * norm_bm25 + (1-alpha) * cosine_sim`.
    /// Lower alpha = more weight to semantic meaning.
    #[cfg(feature = "embeddings")]
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
        let model = self.inner.embedding_model.as_ref().ok_or_else(|| {
            VaultError::Embedding("embeddings not enabled (OBSIDIAN_EMBEDDINGS=false)".into())
        })?;
        let store = self
            .inner
            .embedding_store
            .as_ref()
            .ok_or_else(|| VaultError::Embedding("embedding store not initialized".into()))?;

        let prefetch = prefetch_count.max(top_k);
        let bm25_hits = tv.search(query, prefetch)?;
        if bm25_hits.is_empty() {
            return Ok(Vec::new());
        }

        let query_vec = model.embed_one(query)?;
        let store_guard = store.read().unwrap_or_else(|e| e.into_inner());

        let norm_bm25 = search_utils::normalize_bm25_scores(&bm25_hits);

        let mut combined: Vec<(PathBuf, f32)> = norm_bm25
            .into_iter()
            .map(|(path, norm_score)| {
                let semantic_score = store_guard
                    .get(&path)
                    .map(|emb| embeddings::cosine_similarity(&query_vec, emb))
                    .unwrap_or(0.0);
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
        Ok(self
            .read_index()
            .backlinks_to(path)
            .into_iter()
            .cloned()
            .collect())
    }

    pub fn outgoing_links(&self, path: &Path) -> VaultResult<Vec<WikiLink>> {
        Ok(self
            .read_index()
            .outgoing_links(path)
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

        fs::write_file(&self.inner.root, &path, &content)?;
        self.reindex(&path)?;
        Ok(path)
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
        let mut idx = self.write_index();
        idx.reindex_file(&self.inner.root, path)?;
        if let Some(tv) = &self.inner.tantivy
            && let Some(meta) = idx.get_note(path)
        {
            tv.reindex_file(&self.inner.root, path, meta)?;
        }
        drop(idx);
        #[cfg(feature = "embeddings")]
        self.reindex_embedding(path);
        Ok(())
    }

    // ── embedding helpers (feature-gated) ─────────────────────────────

    #[cfg(feature = "embeddings")]
    fn embedding_cache_path(vault_root: &Path) -> PathBuf {
        vault_root
            .join(".obsidian")
            .join("obsidian-mcp")
            .join("embeddings.bin")
    }

    #[cfg(feature = "embeddings")]
    fn build_or_load_embeddings(
        vault_root: &Path,
        index: &Arc<RwLock<VaultIndex>>,
        model: &embeddings::EmbeddingModel,
    ) -> VaultResult<embeddings::EmbeddingStore> {
        let cache_path = Self::embedding_cache_path(vault_root);
        let idx = index.read().unwrap_or_else(|e| e.into_inner());
        let note_entries: Vec<_> = idx
            .notes()
            .iter()
            .map(|(path, meta)| (path.clone(), meta.clone()))
            .collect();
        drop(idx);
        embeddings::build_or_load_embedding_store(&cache_path, vault_root, &note_entries, model)
    }

    /// Re-embed a single note and update the store. Non-fatal on error.
    #[cfg(feature = "embeddings")]
    fn reindex_embedding(&self, path: &Path) {
        let (Some(model), Some(store)) = (&self.inner.embedding_model, &self.inner.embedding_store)
        else {
            return;
        };

        let Ok(content) = fs::read_file(&self.inner.root, path) else {
            return;
        };

        let idx = self.read_index();
        let Some(meta) = idx.get_note(path) else {
            return;
        };

        let body = frontmatter::get_body(&content);
        let heading_texts: Vec<String> = meta.headings.iter().map(|h| h.text.clone()).collect();
        let text = embeddings::prepare_embed_text(&meta.title, &heading_texts, body);
        drop(idx);

        match model.embed_one(&text) {
            Ok(vec) => {
                let mut s = store.write().unwrap_or_else(|e| e.into_inner());
                s.insert(path.to_path_buf(), vec);
                drop(s);
                self.save_embedding_cache();
            }
            Err(e) => {
                tracing::warn!(path = %path.display(), error = %e, "embedding failed");
            }
        }
    }

    /// Remove a note's embedding from the store.
    #[cfg(feature = "embeddings")]
    fn remove_embedding(&self, path: &Path) {
        if let Some(store) = &self.inner.embedding_store {
            let mut s = store.write().unwrap_or_else(|e| e.into_inner());
            s.remove(path);
            drop(s);
            self.save_embedding_cache();
        }
    }

    /// Persist the embedding cache to disk. Non-fatal on error.
    #[cfg(feature = "embeddings")]
    fn save_embedding_cache(&self) {
        if let Some(store) = &self.inner.embedding_store {
            let s = store.read().unwrap_or_else(|e| e.into_inner());
            let cache_path = Self::embedding_cache_path(&self.inner.root);
            if let Err(e) = s.save(&cache_path) {
                tracing::warn!(error = %e, "failed to save embedding cache");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::models::{PatchOperation, PatchTargetType};
    use crate::test_helpers::{create_test_vault, tantivy_config, test_config};

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
            tantivy: false,
            embeddings: false,
            embeddings_model: String::new(),
            hybrid_alpha: 0.25,
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
}
