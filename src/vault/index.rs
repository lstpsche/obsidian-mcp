//! In-memory vault index: `NoteMetadata` for every `.md` file, fast lookups by stem/path/tag.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use regex::Regex;
use walkdir::WalkDir;

use crate::error::{VaultError, VaultResult};
use crate::models::{NoteMetadata, SearchMatch, SearchResult, VaultStats, WikiLink};
use crate::vault::wikilink::{LinkResolver, build_link_resolver};
use crate::vault::{frontmatter, fs, parser};

pub struct VaultIndex {
    notes: HashMap<PathBuf, NoteMetadata>,
    tags: HashMap<String, HashSet<PathBuf>>,
    link_resolver: LinkResolver,
    backlinks: HashMap<PathBuf, HashSet<PathBuf>>,
    stats: VaultStats,
    non_md_file_count: usize,
    non_md_bytes: u64,
}

impl VaultIndex {
    /// Create an empty index (useful for tests and pre-initialization).
    pub fn empty() -> Self {
        Self {
            notes: HashMap::new(),
            tags: HashMap::new(),
            link_resolver: build_link_resolver(&[]),
            backlinks: HashMap::new(),
            stats: VaultStats {
                total_notes: 0,
                total_files: 0,
                total_tags: 0,
                total_links: 0,
                vault_size_bytes: 0,
            },
            non_md_file_count: 0,
            non_md_bytes: 0,
        }
    }

    /// Build the index by walking the entire vault directory.
    pub async fn build(vault_root: &Path) -> VaultResult<Self> {
        let root = vault_root.to_path_buf();
        tokio::task::spawn_blocking(move || Self::build_sync(&root))
            .await
            .map_err(|e| VaultError::Other(format!("index build task panicked: {e}")))?
    }

    fn build_sync(vault_root: &Path) -> VaultResult<Self> {
        let mut notes = HashMap::new();
        let mut tags: HashMap<String, HashSet<PathBuf>> = HashMap::new();
        let mut non_md_file_count: usize = 0;
        let mut non_md_bytes: u64 = 0;

        let walker = WalkDir::new(vault_root)
            .min_depth(1)
            .into_iter()
            .filter_entry(|e| {
                e.file_name()
                    .to_str()
                    .is_some_and(|name| !name.starts_with('.'))
            });

        for entry in walker {
            let entry = entry.map_err(|e| match e.into_io_error() {
                Some(io_err) => VaultError::Io(io_err),
                None => VaultError::Other("walkdir: directory loop detected".into()),
            })?;

            if !entry.file_type().is_file() {
                continue;
            }

            let abs_path = entry.path();
            let rel = abs_path.strip_prefix(vault_root).map_err(|_| {
                VaultError::Other(format!(
                    "path {} is not under vault root {}",
                    abs_path.display(),
                    vault_root.display()
                ))
            })?;
            let rel_path = PathBuf::from(rel.to_string_lossy().replace('\\', "/"));

            let is_md = abs_path
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("md"));

            if is_md {
                match parse_note_metadata(vault_root, &rel_path) {
                    Ok(metadata) => {
                        for tag in &metadata.tags {
                            tags.entry(tag.clone())
                                .or_default()
                                .insert(rel_path.clone());
                        }
                        notes.insert(rel_path, metadata);
                    }
                    Err(e) => {
                        tracing::warn!(
                            path = %rel_path.display(),
                            error = %e,
                            "skipping unparseable note during index build"
                        );
                    }
                }
            } else {
                non_md_file_count += 1;
                if let Ok(meta) = entry.metadata() {
                    non_md_bytes += meta.len();
                }
            }
        }

        let note_paths: Vec<PathBuf> = notes.keys().cloned().collect();
        let link_resolver = build_link_resolver(&note_paths);

        let mut index = Self {
            notes,
            tags,
            link_resolver,
            backlinks: HashMap::new(),
            stats: VaultStats {
                total_notes: 0,
                total_files: 0,
                total_tags: 0,
                total_links: 0,
                vault_size_bytes: 0,
            },
            non_md_file_count,
            non_md_bytes,
        };

        index.rebuild_backlinks();
        index.recompute_stats();

        Ok(index)
    }

    /// Re-index a single file (on create or modify).
    pub fn reindex_file(&mut self, vault_root: &Path, path: &Path) -> VaultResult<()> {
        self.remove_note_contributions(path);
        self.link_resolver.remove_path(path);

        let metadata = parse_note_metadata(vault_root, path)?;
        for tag in &metadata.tags {
            self.tags
                .entry(tag.clone())
                .or_default()
                .insert(path.to_path_buf());
        }
        self.link_resolver.add_path(path.to_path_buf());
        self.notes.insert(path.to_path_buf(), metadata);

        self.rebuild_backlinks();
        self.recompute_stats();
        Ok(())
    }

    /// Remove a file from the index (on delete).
    pub fn remove_file(&mut self, path: &Path) {
        self.remove_note_contributions(path);
        self.link_resolver.remove_path(path);
        self.backlinks.remove(path);

        self.rebuild_backlinks();
        self.recompute_stats();
    }

    /// Handle a file rename/move.
    pub fn rename_file(&mut self, vault_root: &Path, old: &Path, new: &Path) -> VaultResult<()> {
        self.remove_note_contributions(old);
        self.link_resolver.rename_path(old, new.to_path_buf());
        self.backlinks.remove(old);

        let metadata = parse_note_metadata(vault_root, new)?;
        for tag in &metadata.tags {
            self.tags
                .entry(tag.clone())
                .or_default()
                .insert(new.to_path_buf());
        }
        self.notes.insert(new.to_path_buf(), metadata);

        self.rebuild_backlinks();
        self.recompute_stats();
        Ok(())
    }

    // ── query methods ───────────────────────────────────────────────

    pub fn notes(&self) -> &HashMap<PathBuf, NoteMetadata> {
        &self.notes
    }

    pub fn get_note(&self, path: &Path) -> Option<&NoteMetadata> {
        self.notes.get(path)
    }

    pub fn notes_with_tag(&self, tag: &str) -> Vec<&NoteMetadata> {
        self.tags
            .get(tag)
            .map(|paths| paths.iter().filter_map(|p| self.notes.get(p)).collect())
            .unwrap_or_default()
    }

    /// Match a tag and all its children (e.g. `inbox` matches `inbox/read`, `inbox/todo`).
    pub fn notes_with_tag_prefix(&self, prefix: &str) -> Vec<&NoteMetadata> {
        let nested_prefix = format!("{prefix}/");
        let mut seen = HashSet::new();
        let mut results = Vec::new();
        for (tag, paths) in &self.tags {
            if tag == prefix || tag.starts_with(&nested_prefix) {
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
            .get(path)
            .map(|sources| sources.iter().filter_map(|p| self.notes.get(p)).collect())
            .unwrap_or_default()
    }

    pub fn outgoing_links(&self, path: &Path) -> Vec<&WikiLink> {
        self.notes
            .get(path)
            .map(|note| note.links.iter().collect())
            .unwrap_or_default()
    }

    pub fn broken_links(&self) -> Vec<(PathBuf, WikiLink)> {
        let mut result = Vec::new();
        for (path, note) in &self.notes {
            for link in &note.links {
                if link.target.is_empty() {
                    continue;
                }
                if !self.link_resolver.is_resolved(&link.target) {
                    result.push((path.clone(), link.clone()));
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

    // ── search methods ──────────────────────────────────────────────

    /// Case-insensitive full-text search across all indexed notes.
    pub fn search_text(
        &self,
        vault_root: &Path,
        query: &str,
        context_len: usize,
    ) -> VaultResult<Vec<SearchResult>> {
        if query.is_empty() {
            return Ok(Vec::new());
        }
        let pattern = format!("(?i){}", regex::escape(query));
        let re = Regex::new(&pattern).map_err(|e| VaultError::InvalidRegex {
            pattern: query.to_string(),
            source: e,
        })?;
        self.search_with_regex(vault_root, &re, context_len)
    }

    /// Regex search across all indexed notes.
    pub fn search_regex(
        &self,
        vault_root: &Path,
        pattern: &str,
        context_len: usize,
    ) -> VaultResult<Vec<SearchResult>> {
        let re = Regex::new(pattern).map_err(|e| VaultError::InvalidRegex {
            pattern: pattern.to_string(),
            source: e,
        })?;
        self.search_with_regex(vault_root, &re, context_len)
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

    fn remove_note_contributions(&mut self, path: &Path) {
        if let Some(old_note) = self.notes.remove(path) {
            let mut empty_tags = Vec::new();
            for tag in &old_note.tags {
                if let Some(paths) = self.tags.get_mut(tag) {
                    paths.remove(path);
                    if paths.is_empty() {
                        empty_tags.push(tag.clone());
                    }
                }
            }
            for tag in &empty_tags {
                self.tags.remove(tag);
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
                        .entry(resolved)
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
            total_files: self.notes.len() + self.non_md_file_count,
            total_tags: self.tags.len(),
            total_links: self.notes.values().map(|n| n.links.len()).sum(),
            vault_size_bytes: md_bytes + self.non_md_bytes,
        };
    }

    fn search_with_regex(
        &self,
        vault_root: &Path,
        re: &Regex,
        context_len: usize,
    ) -> VaultResult<Vec<SearchResult>> {
        let mut results = Vec::new();

        for path in self.notes.keys() {
            let content = match fs::read_file(vault_root, path) {
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
                    path: path.clone(),
                    matches,
                    score: None,
                });
            }
        }

        Ok(results)
    }
}

// ── module-level helpers ────────────────────────────────────────────

fn parse_note_metadata(vault_root: &Path, path: &Path) -> VaultResult<NoteMetadata> {
    let content = fs::read_file(vault_root, path)?;
    let stat = fs::file_stat(vault_root, path)?;

    let fm = frontmatter::parse_frontmatter(&content)?;
    let fm_tags = fm
        .as_ref()
        .map(frontmatter::extract_frontmatter_tags)
        .unwrap_or_default();

    let headings = parser::extract_headings(&content);
    let inline_tags = parser::extract_inline_tags(&content);
    let links = parser::extract_wikilinks(&content);
    let block_refs = parser::extract_block_refs(&content);

    let mut tags = fm_tags;
    tags.extend(inline_tags);
    tags.sort();
    tags.dedup();

    let title = path
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();

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
    let ctx_end = content.ceil_char_boundary((match_byte_end + context_len).min(content.len()));

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
    use tempfile::TempDir;

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
        let index = VaultIndex::build(vault.path()).await.unwrap();

        assert_eq!(index.notes.len(), 5);
        assert!(index.get_note(Path::new("daily.md")).is_some());
        assert!(index.get_note(Path::new("notes/alpha.md")).is_some());
        assert!(index.get_note(Path::new("notes/beta.md")).is_some());
        assert!(index.get_note(Path::new("notes/gamma.md")).is_some());
        assert!(index.get_note(Path::new("archive/old.md")).is_some());
    }

    #[tokio::test]
    async fn build_computes_correct_backlinks() {
        let vault = setup_vault();
        let index = VaultIndex::build(vault.path()).await.unwrap();

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
        let index = VaultIndex::build(vault.path()).await.unwrap();

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
        let index = VaultIndex::build(vault.path()).await.unwrap();

        let broken = index.broken_links();
        let broken_targets: Vec<&str> = broken.iter().map(|(_, l)| l.target.as_str()).collect();
        assert!(broken_targets.contains(&"notes"));
    }

    #[tokio::test]
    async fn build_detects_orphan_notes() {
        let vault = setup_vault();
        let index = VaultIndex::build(vault.path()).await.unwrap();

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
        let index = VaultIndex::build(vault.path()).await.unwrap();

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
        let index = VaultIndex::build(vault.path()).await.unwrap();

        for path in index.notes.keys() {
            let s = path.display().to_string();
            assert!(!s.contains(".obsidian"), "indexed .obsidian: {s}");
            assert!(!s.starts_with('.'), "indexed hidden file: {s}");
        }
    }

    #[tokio::test]
    async fn outgoing_links_correct() {
        let vault = setup_vault();
        let index = VaultIndex::build(vault.path()).await.unwrap();

        let links = index.outgoing_links(Path::new("notes/alpha.md"));
        let targets: Vec<&str> = links.iter().map(|l| l.target.as_str()).collect();
        assert!(targets.contains(&"beta"));
    }

    #[tokio::test]
    async fn get_note_missing_returns_none() {
        let vault = setup_vault();
        let index = VaultIndex::build(vault.path()).await.unwrap();

        assert!(index.get_note(Path::new("nonexistent.md")).is_none());
    }

    #[tokio::test]
    async fn notes_with_nonexistent_tag_returns_empty() {
        let vault = setup_vault();
        let index = VaultIndex::build(vault.path()).await.unwrap();

        assert!(index.notes_with_tag("nonexistent").is_empty());
    }

    #[tokio::test]
    async fn note_title_is_file_stem() {
        let vault = setup_vault();
        let index = VaultIndex::build(vault.path()).await.unwrap();

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
        let index = VaultIndex::build(vault.path()).await.unwrap();

        let beta = index.get_note(Path::new("notes/beta.md")).unwrap();
        assert!(beta.block_refs.contains(&"block1".to_string()));
    }

    // ── mutation tests ──────────────────────────────────────────────

    #[tokio::test]
    async fn reindex_file_updates_tags_and_links() {
        let vault = setup_vault();
        let mut index = VaultIndex::build(vault.path()).await.unwrap();

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
        let mut index = VaultIndex::build(vault.path()).await.unwrap();
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
        let mut index = VaultIndex::build(vault.path()).await.unwrap();

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
        let mut index = VaultIndex::build(vault.path()).await.unwrap();

        assert!(index.get_note(Path::new("notes/alpha.md")).is_some());
        assert!(!index.notes_with_tag("rust").is_empty());

        index.remove_file(Path::new("notes/alpha.md"));

        assert!(index.get_note(Path::new("notes/alpha.md")).is_none());
        assert!(index.notes_with_tag("rust").is_empty());

        let beta_bl = index.backlinks_to(Path::new("notes/beta.md"));
        assert!(
            !beta_bl
                .iter()
                .any(|n| n.path == PathBuf::from("notes/alpha.md")),
            "alpha should no longer appear as a backlink source"
        );
    }

    #[tokio::test]
    async fn remove_file_updates_stats() {
        let vault = setup_vault();
        let mut index = VaultIndex::build(vault.path()).await.unwrap();
        let old_notes = index.stats().total_notes;

        index.remove_file(Path::new("notes/gamma.md"));

        assert_eq!(index.stats().total_notes, old_notes - 1);
    }

    #[tokio::test]
    async fn rename_file_updates_index() {
        let vault = setup_vault();
        let mut index = VaultIndex::build(vault.path()).await.unwrap();

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
        let index = VaultIndex::build(vault.path()).await.unwrap();

        let results = index.search_text(vault.path(), "alpha", 20).unwrap();
        assert!(!results.is_empty());

        let result_paths: HashSet<PathBuf> = results.iter().map(|r| r.path.clone()).collect();
        assert!(result_paths.contains(&PathBuf::from("notes/alpha.md")));
    }

    #[tokio::test]
    async fn search_text_case_insensitive() {
        let vault = setup_vault();
        let index = VaultIndex::build(vault.path()).await.unwrap();

        let lower = index.search_text(vault.path(), "gamma", 10).unwrap();
        let upper = index.search_text(vault.path(), "GAMMA", 10).unwrap();

        assert_eq!(lower.len(), upper.len());
        assert!(!lower.is_empty());
    }

    #[tokio::test]
    async fn search_text_empty_query_returns_empty() {
        let vault = setup_vault();
        let index = VaultIndex::build(vault.path()).await.unwrap();

        assert!(index.search_text(vault.path(), "", 10).unwrap().is_empty());
    }

    #[tokio::test]
    async fn search_text_context_offsets_are_correct() {
        let vault = setup_vault();
        let index = VaultIndex::build(vault.path()).await.unwrap();

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
        let index = VaultIndex::build(vault.path()).await.unwrap();

        let results = index
            .search_regex(vault.path(), r"\[\[alpha\]\]", 10)
            .unwrap();
        assert!(!results.is_empty());

        let paths: HashSet<PathBuf> = results.iter().map(|r| r.path.clone()).collect();
        assert!(paths.contains(&PathBuf::from("notes/beta.md")));
        assert!(paths.contains(&PathBuf::from("archive/old.md")));
    }

    #[tokio::test]
    async fn search_regex_invalid_pattern_returns_error() {
        let vault = setup_vault();
        let index = VaultIndex::build(vault.path()).await.unwrap();

        let result = index.search_regex(vault.path(), "[invalid", 10);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            VaultError::InvalidRegex { .. }
        ));
    }

    #[tokio::test]
    async fn search_frontmatter_exact_match() {
        let vault = setup_vault();
        let index = VaultIndex::build(vault.path()).await.unwrap();

        let results = index.search_frontmatter("tags", &serde_json::json!(["journal"]));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].path, PathBuf::from("daily.md"));
    }

    #[tokio::test]
    async fn search_frontmatter_array_contains() {
        let vault = setup_vault();
        let index = VaultIndex::build(vault.path()).await.unwrap();

        let results = index.search_frontmatter("tags", &serde_json::json!("rust"));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].path, PathBuf::from("notes/alpha.md"));
    }

    #[tokio::test]
    async fn search_frontmatter_no_match() {
        let vault = setup_vault();
        let index = VaultIndex::build(vault.path()).await.unwrap();

        assert!(
            index
                .search_frontmatter("tags", &serde_json::json!("nonexistent"))
                .is_empty()
        );
    }

    #[tokio::test]
    async fn search_frontmatter_missing_field() {
        let vault = setup_vault();
        let index = VaultIndex::build(vault.path()).await.unwrap();

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

        let index = VaultIndex::build(dir.path()).await.unwrap();
        let results = index.notes_with_tag_prefix("inbox");
        assert_eq!(results.len(), 3);
    }

    #[tokio::test]
    async fn notes_with_tag_prefix_no_false_prefix_match() {
        let dir = TempDir::new().unwrap();
        stdfs::write(dir.path().join("a.md"), "# A\n\n#inbox\n").unwrap();
        stdfs::write(dir.path().join("b.md"), "# B\n\n#inboxes\n").unwrap();

        let index = VaultIndex::build(dir.path()).await.unwrap();
        let results = index.notes_with_tag_prefix("inbox");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].path, PathBuf::from("a.md"));
    }

    #[tokio::test]
    async fn notes_with_tag_prefix_deduplicates() {
        let dir = TempDir::new().unwrap();
        stdfs::write(dir.path().join("a.md"), "# A\n\n#inbox #inbox/read\n").unwrap();

        let index = VaultIndex::build(dir.path()).await.unwrap();
        let results = index.notes_with_tag_prefix("inbox");
        assert_eq!(results.len(), 1);
    }

    // ── frontmatter exists / contains tests ─────────────────────────

    #[tokio::test]
    async fn search_frontmatter_exists_finds_field() {
        let vault = setup_vault();
        let index = VaultIndex::build(vault.path()).await.unwrap();

        let results = index.search_frontmatter_exists("tags");
        assert_eq!(results.len(), 2); // daily.md + notes/alpha.md
    }

    #[tokio::test]
    async fn search_frontmatter_exists_missing_field_empty() {
        let vault = setup_vault();
        let index = VaultIndex::build(vault.path()).await.unwrap();

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

        let index = VaultIndex::build(dir.path()).await.unwrap();
        let results = index.search_frontmatter_contains("status", &serde_json::json!("progress"));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].path, PathBuf::from("a.md"));
    }

    #[tokio::test]
    async fn search_frontmatter_contains_array_element() {
        let vault = setup_vault();
        let index = VaultIndex::build(vault.path()).await.unwrap();

        let results = index.search_frontmatter_contains("tags", &serde_json::json!("rust"));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].path, PathBuf::from("notes/alpha.md"));
    }
}
