//! Tantivy BM25 full-text index: schema, indexing, and search.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use tantivy::collector::TopDocs;
use tantivy::query::{BooleanQuery, BoostQuery, Occur, QueryParser, TermQuery};
use tantivy::schema::{
    Facet, Field, IndexRecordOption, STORED, STRING, Schema, SchemaBuilder, TextFieldIndexing,
    TextOptions, Value,
};
use tantivy::{Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument, Term, doc};

use super::frontmatter;
use super::fs;
use crate::error::{VaultError, VaultResult};
use crate::models::{NoteMetadata, SearchField};

pub const TITLE_BOOST: f32 = 5.0;
pub const HEADINGS_BOOST: f32 = 3.0;
pub const TAGS_BOOST: f32 = 4.0;
pub const FRONTMATTER_BOOST: f32 = 2.0;

/// Pre-built Tantivy schema with cached field handles for fast document construction.
pub struct TantivySchema {
    pub schema: Schema,
    /// Vault-relative path (stored, not indexed — used as primary key).
    pub f_path: Field,
    /// Note title / filename stem (stored, indexed with `en_stem`, high boost).
    pub f_title: Field,
    /// Concatenated headings (not stored, indexed with `en_stem`, medium boost).
    pub f_headings: Field,
    /// Tags as facets (stored, indexed).
    pub f_tags: Field,
    /// Full note body (not stored, indexed with `en_stem`, base boost).
    pub f_body: Field,
    /// Stringified frontmatter values (not stored, indexed with `en_stem`).
    pub f_frontmatter_text: Field,
}

impl TantivySchema {
    /// Build the Tantivy schema for vault note indexing.
    ///
    /// All indexed text fields use the built-in `en_stem` tokenizer (English stemmer)
    /// with `WithFreqsAndPositions` to support phrase queries and proximity scoring.
    pub fn build() -> Self {
        let mut builder = SchemaBuilder::new();

        let f_path = builder.add_text_field("path", STRING | STORED);

        let stemmed_indexing = TextFieldIndexing::default()
            .set_tokenizer("en_stem")
            .set_index_option(IndexRecordOption::WithFreqsAndPositions);

        let f_title = builder.add_text_field(
            "title",
            TextOptions::default()
                .set_indexing_options(stemmed_indexing.clone())
                .set_stored(),
        );

        let f_headings = builder.add_text_field(
            "headings",
            TextOptions::default().set_indexing_options(stemmed_indexing.clone()),
        );

        let f_tags = builder.add_facet_field("tags", STORED);

        let f_body = builder.add_text_field(
            "body",
            TextOptions::default().set_indexing_options(stemmed_indexing.clone()),
        );

        let f_frontmatter_text = builder.add_text_field(
            "frontmatter_text",
            TextOptions::default().set_indexing_options(stemmed_indexing),
        );

        let schema = builder.build();

        Self {
            schema,
            f_path,
            f_title,
            f_headings,
            f_tags,
            f_body,
            f_frontmatter_text,
        }
    }
}

const WRITER_HEAP_BYTES: usize = 50_000_000;

/// BM25 full-text index backed by Tantivy with an in-RAM directory.
///
/// The index rebuilds from scratch on startup (< 100ms for 5000 notes) so no
/// disk persistence is needed. The `IndexReader` uses `Manual` reload policy;
/// mutations call `reader.reload()` explicitly after each commit.
pub struct TantivyIndex {
    ts: TantivySchema,
    reader: IndexReader,
    writer: Mutex<IndexWriter>,
}

impl TantivyIndex {
    /// Build the full-text index from all notes in the vault.
    ///
    /// Reads each note's content from disk, constructs a Tantivy document
    /// with title/headings/tags/body/frontmatter fields, and commits once.
    pub fn build(vault_root: &Path, notes: &HashMap<PathBuf, NoteMetadata>) -> VaultResult<Self> {
        let ts = TantivySchema::build();
        let index = Index::create_in_ram(ts.schema.clone());
        let mut writer: IndexWriter = index.writer(WRITER_HEAP_BYTES)?;

        for (path, meta) in notes {
            let content = match fs::read_file(vault_root, path) {
                Ok(c) => c,
                Err(e) => {
                    tracing::warn!(
                        path = %path.display(),
                        error = %e,
                        "skipping note during tantivy index build"
                    );
                    continue;
                }
            };
            let doc = build_document(&ts, path, meta, &content);
            writer.add_document(doc)?;
        }

        writer.commit()?;

        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()?;

        Ok(Self {
            ts,
            reader,
            writer: Mutex::new(writer),
        })
    }

    /// Re-index a single file after create or modify (commits immediately).
    pub fn reindex_file(
        &self,
        vault_root: &Path,
        path: &Path,
        meta: &NoteMetadata,
    ) -> VaultResult<()> {
        self.reindex_file_batch(vault_root, path, meta)?;
        self.flush()
    }

    /// Re-index a single file without committing.
    /// Call [`flush`] after processing a batch of updates.
    pub fn reindex_file_batch(
        &self,
        vault_root: &Path,
        path: &Path,
        meta: &NoteMetadata,
    ) -> VaultResult<()> {
        let content = fs::read_file(vault_root, path)?;
        let path_str = path.to_string_lossy();

        let writer = self
            .writer
            .lock()
            .map_err(|e| VaultError::Other(format!("tantivy writer lock poisoned: {e}")))?;

        writer.delete_term(Term::from_field_text(self.ts.f_path, &path_str));
        let doc = build_document(&self.ts, path, meta, &content);
        writer.add_document(doc)?;

        Ok(())
    }

    /// Remove a file from the index (commits immediately).
    pub fn remove_file(&self, path: &Path) -> VaultResult<()> {
        self.remove_file_batch(path)?;
        self.flush()
    }

    /// Remove a file from the index without committing.
    /// Call [`flush`] after processing a batch of updates.
    pub fn remove_file_batch(&self, path: &Path) -> VaultResult<()> {
        let path_str = path.to_string_lossy();

        let writer = self
            .writer
            .lock()
            .map_err(|e| VaultError::Other(format!("tantivy writer lock poisoned: {e}")))?;

        writer.delete_term(Term::from_field_text(self.ts.f_path, &path_str));

        Ok(())
    }

    /// Commit pending writes and reload the reader.
    pub fn flush(&self) -> VaultResult<()> {
        let mut writer = self
            .writer
            .lock()
            .map_err(|e| VaultError::Other(format!("tantivy writer lock poisoned: {e}")))?;
        writer.commit()?;
        drop(writer);
        self.reader.reload()?;
        Ok(())
    }

    /// BM25-ranked full-text search across title, headings, body, and frontmatter.
    ///
    /// Returns `(path, score)` pairs sorted by descending BM25 relevance.
    pub fn search(&self, query: &str, top_k: usize) -> VaultResult<Vec<(PathBuf, f32)>> {
        if query.is_empty() {
            return Ok(Vec::new());
        }

        let searcher = self.reader.searcher();

        let mut query_parser = QueryParser::for_index(
            searcher.index(),
            vec![
                self.ts.f_title,
                self.ts.f_headings,
                self.ts.f_body,
                self.ts.f_frontmatter_text,
            ],
        );
        query_parser.set_field_boost(self.ts.f_title, TITLE_BOOST);
        query_parser.set_field_boost(self.ts.f_headings, HEADINGS_BOOST);
        query_parser.set_field_boost(self.ts.f_body, 1.0);
        query_parser.set_field_boost(self.ts.f_frontmatter_text, FRONTMATTER_BOOST);

        let parsed = query_parser
            .parse_query(query)
            .map_err(|e| VaultError::Other(format!("tantivy query parse error: {e}")))?;

        let top_docs = searcher.search(&parsed, &TopDocs::with_limit(top_k))?;

        let mut results = Vec::with_capacity(top_docs.len());
        for (score, doc_address) in top_docs {
            let retrieved: TantivyDocument = searcher.doc(doc_address)?;
            if let Some(path_str) = retrieved.get_first(self.ts.f_path).and_then(|v| v.as_str()) {
                results.push((PathBuf::from(path_str), score));
            }
        }

        Ok(results)
    }

    /// BM25-ranked search with optional fuzzy matching and field filtering.
    ///
    /// - `fuzzy`: when true, enables edit-distance-1 fuzzy matching on all text fields
    ///   via `QueryParser::set_field_fuzzy`.
    /// - `fields`: when `Some`, restricts the search to the specified fields.
    ///   `Tags` field creates exact facet term queries (not stemmed).
    ///
    /// Delegates to `search()` when neither option is active.
    pub fn search_with_options(
        &self,
        query: &str,
        top_k: usize,
        fuzzy: bool,
        fields: Option<&[SearchField]>,
    ) -> VaultResult<Vec<(PathBuf, f32)>> {
        if query.is_empty() {
            return Ok(Vec::new());
        }

        if !fuzzy && fields.is_none() {
            return self.search(query, top_k);
        }

        let searcher = self.reader.searcher();

        let (text_fields, include_tags) = match fields {
            Some(fs) => {
                let mut tf = Vec::new();
                let mut tags = false;
                for f in fs {
                    match f {
                        SearchField::Title => tf.push((self.ts.f_title, TITLE_BOOST)),
                        SearchField::Headings => tf.push((self.ts.f_headings, HEADINGS_BOOST)),
                        SearchField::Body => tf.push((self.ts.f_body, 1.0)),
                        SearchField::Frontmatter => {
                            tf.push((self.ts.f_frontmatter_text, FRONTMATTER_BOOST));
                        }
                        SearchField::Tags => tags = true,
                    }
                }
                (tf, tags)
            }
            // Tags are not included as facet queries here because they're already
            // covered by `frontmatter_text` (which contains stringified tag values).
            None => (
                vec![
                    (self.ts.f_title, TITLE_BOOST),
                    (self.ts.f_headings, HEADINGS_BOOST),
                    (self.ts.f_body, 1.0),
                    (self.ts.f_frontmatter_text, FRONTMATTER_BOOST),
                ],
                false,
            ),
        };

        let mut subqueries: Vec<(Occur, Box<dyn tantivy::query::Query>)> = Vec::new();

        if !text_fields.is_empty() {
            let field_list: Vec<Field> = text_fields.iter().map(|(f, _)| *f).collect();
            let mut qp = QueryParser::for_index(searcher.index(), field_list);
            for &(field, boost) in &text_fields {
                qp.set_field_boost(field, boost);
                if fuzzy {
                    qp.set_field_fuzzy(field, false, 1, true);
                }
            }
            let parsed = qp
                .parse_query(query)
                .map_err(|e| VaultError::Other(format!("tantivy query parse error: {e}")))?;
            subqueries.push((Occur::Should, parsed));
        }

        if include_tags {
            for word in query.split_whitespace() {
                let components: Vec<&str> = word.split('/').collect();
                let facet = Facet::from_path(components);
                let term = Term::from_facet(self.ts.f_tags, &facet);
                let tq = TermQuery::new(term, IndexRecordOption::Basic);
                let boosted = BoostQuery::new(Box::new(tq), TAGS_BOOST);
                subqueries.push((Occur::Should, Box::new(boosted)));
            }
        }

        if subqueries.is_empty() {
            return Ok(Vec::new());
        }

        let final_query: Box<dyn tantivy::query::Query> = match subqueries.len() {
            1 => subqueries.pop().expect("len checked above").1,
            _ => Box::new(BooleanQuery::new(subqueries)),
        };

        let top_docs = searcher.search(&*final_query, &TopDocs::with_limit(top_k))?;

        let mut results = Vec::with_capacity(top_docs.len());
        for (score, doc_address) in top_docs {
            let retrieved: TantivyDocument = searcher.doc(doc_address)?;
            if let Some(path_str) = retrieved.get_first(self.ts.f_path).and_then(|v| v.as_str()) {
                results.push((PathBuf::from(path_str), score));
            }
        }

        Ok(results)
    }
}

/// Construct a Tantivy document from note metadata and raw file content.
fn build_document(
    ts: &TantivySchema,
    path: &Path,
    meta: &NoteMetadata,
    content: &str,
) -> TantivyDocument {
    let path_str = path.to_string_lossy();

    let headings_text: String = meta
        .headings
        .iter()
        .map(|h| h.text.as_str())
        .collect::<Vec<_>>()
        .join(" | ");

    let body = match frontmatter::extract_raw_frontmatter(content) {
        Some((_, body_start)) => &content[body_start..],
        None => content,
    };

    let fm_text = meta
        .frontmatter
        .as_ref()
        .map(stringify_frontmatter)
        .unwrap_or_default();

    let mut doc = doc!(
        ts.f_path => path_str.as_ref(),
        ts.f_title => meta.title.as_str(),
        ts.f_headings => headings_text,
        ts.f_body => body,
        ts.f_frontmatter_text => fm_text,
    );

    for tag in &meta.tags {
        let components: Vec<&str> = tag.split('/').collect();
        doc.add_facet(ts.f_tags, Facet::from_path(components));
    }

    doc
}

/// Flatten a frontmatter JSON value into a space-separated string of keys and values.
fn stringify_frontmatter(value: &serde_json::Value) -> String {
    let mut parts = Vec::new();
    collect_json_strings(value, &mut parts);
    parts.join(" ")
}

fn collect_json_strings(value: &serde_json::Value, out: &mut Vec<String>) {
    match value {
        serde_json::Value::String(s) => out.push(s.clone()),
        serde_json::Value::Array(arr) => {
            for item in arr {
                collect_json_strings(item, out);
            }
        }
        serde_json::Value::Object(map) => {
            for (k, v) in map {
                out.push(k.clone());
                collect_json_strings(v, out);
            }
        }
        serde_json::Value::Number(n) => out.push(n.to_string()),
        serde_json::Value::Bool(b) => out.push(b.to_string()),
        serde_json::Value::Null => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{FileStat, Heading};

    // ── schema tests ────────────────────────────────────────────────

    #[test]
    fn schema_has_expected_fields() {
        let ts = TantivySchema::build();

        assert_eq!(ts.schema.get_field_name(ts.f_path), "path");
        assert_eq!(ts.schema.get_field_name(ts.f_title), "title");
        assert_eq!(ts.schema.get_field_name(ts.f_headings), "headings");
        assert_eq!(ts.schema.get_field_name(ts.f_tags), "tags");
        assert_eq!(ts.schema.get_field_name(ts.f_body), "body");
        assert_eq!(
            ts.schema.get_field_name(ts.f_frontmatter_text),
            "frontmatter_text"
        );
    }

    #[test]
    fn path_field_is_stored_and_string_indexed() {
        let ts = TantivySchema::build();
        let entry = ts.schema.get_field_entry(ts.f_path);

        assert!(entry.is_stored());
        assert!(entry.is_indexed());
    }

    #[test]
    fn title_field_is_stored_and_stemmed() {
        let ts = TantivySchema::build();
        let entry = ts.schema.get_field_entry(ts.f_title);

        assert!(entry.is_stored());
        assert!(entry.is_indexed());
    }

    #[test]
    fn body_field_is_not_stored() {
        let ts = TantivySchema::build();
        let entry = ts.schema.get_field_entry(ts.f_body);

        assert!(!entry.is_stored());
        assert!(entry.is_indexed());
    }

    #[test]
    fn headings_field_is_not_stored() {
        let ts = TantivySchema::build();
        let entry = ts.schema.get_field_entry(ts.f_headings);

        assert!(!entry.is_stored());
        assert!(entry.is_indexed());
    }

    #[test]
    fn tags_field_is_facet_and_stored() {
        let ts = TantivySchema::build();
        let entry = ts.schema.get_field_entry(ts.f_tags);

        assert!(entry.is_stored());
    }

    #[test]
    fn schema_field_count() {
        let ts = TantivySchema::build();

        assert_eq!(ts.schema.num_fields(), 6);
    }

    // ── TantivyIndex tests ──────────────────────────────────────────

    fn dummy_stat() -> FileStat {
        FileStat {
            size: 100,
            created: None,
            modified: None,
        }
    }

    fn make_meta(path: &str, title: &str, tags: &[&str], headings: &[&str]) -> NoteMetadata {
        NoteMetadata {
            path: PathBuf::from(path),
            title: title.to_string(),
            tags: tags.iter().map(|s| s.to_string()).collect(),
            frontmatter: None,
            headings: headings
                .iter()
                .enumerate()
                .map(|(i, t)| Heading {
                    level: 1,
                    text: t.to_string(),
                    line: i,
                })
                .collect(),
            links: vec![],
            block_refs: vec![],
            stat: dummy_stat(),
        }
    }

    fn setup_vault() -> (tempfile::TempDir, HashMap<PathBuf, NoteMetadata>) {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();

        std::fs::write(
            root.join("rust.md"),
            "---\ntags: [programming, rust]\n---\n# Rust Language\n\nRust is a systems programming language.\n",
        )
        .unwrap();

        std::fs::write(
            root.join("python.md"),
            "# Python\n\nPython is a dynamic programming language.\nGreat for scripting.\n",
        )
        .unwrap();

        std::fs::write(
            root.join("cooking.md"),
            "# Cooking Tips\n\nHow to make a great pasta dish.\n",
        )
        .unwrap();

        let mut notes = HashMap::new();
        notes.insert(
            PathBuf::from("rust.md"),
            NoteMetadata {
                path: PathBuf::from("rust.md"),
                title: "rust".to_string(),
                tags: vec!["programming".to_string(), "rust".to_string()],
                frontmatter: Some(serde_json::json!({"tags": ["programming", "rust"]})),
                headings: vec![Heading {
                    level: 1,
                    text: "Rust Language".to_string(),
                    line: 3,
                }],
                links: vec![],
                block_refs: vec![],
                stat: dummy_stat(),
            },
        );
        notes.insert(
            PathBuf::from("python.md"),
            make_meta("python.md", "python", &[], &["Python"]),
        );
        notes.insert(
            PathBuf::from("cooking.md"),
            make_meta("cooking.md", "cooking", &[], &["Cooking Tips"]),
        );

        (dir, notes)
    }

    #[test]
    fn build_indexes_notes() {
        let (dir, notes) = setup_vault();
        let idx = TantivyIndex::build(dir.path(), &notes).unwrap();

        let searcher = idx.reader.searcher();
        assert_eq!(searcher.num_docs(), 3);
    }

    #[test]
    fn search_finds_by_title() {
        let (dir, notes) = setup_vault();
        let idx = TantivyIndex::build(dir.path(), &notes).unwrap();

        let results = idx.search("rust", 10).unwrap();
        assert!(!results.is_empty());
        assert!(results.iter().any(|(p, _)| p == Path::new("rust.md")));
    }

    #[test]
    fn search_finds_by_body() {
        let (dir, notes) = setup_vault();
        let idx = TantivyIndex::build(dir.path(), &notes).unwrap();

        let results = idx.search("pasta", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, PathBuf::from("cooking.md"));
    }

    #[test]
    fn search_stemming_works() {
        let (dir, notes) = setup_vault();
        let idx = TantivyIndex::build(dir.path(), &notes).unwrap();

        // "programming" is in content; "programs" stems to the same root
        let results = idx.search("programs", 10).unwrap();
        assert!(
            !results.is_empty(),
            "stemming should match 'programs' to 'programming'"
        );
    }

    #[test]
    fn search_empty_query_returns_empty() {
        let (dir, notes) = setup_vault();
        let idx = TantivyIndex::build(dir.path(), &notes).unwrap();

        assert!(idx.search("", 10).unwrap().is_empty());
    }

    #[test]
    fn search_scores_title_higher_than_body() {
        let (dir, notes) = setup_vault();
        let idx = TantivyIndex::build(dir.path(), &notes).unwrap();

        // "python" appears as title for python.md and in body of both python.md and rust.md
        let results = idx.search("python", 10).unwrap();
        assert!(!results.is_empty());
        // python.md should rank first due to title boost
        assert_eq!(results[0].0, PathBuf::from("python.md"));
    }

    #[test]
    fn reindex_file_updates_results() {
        let (dir, mut notes) = setup_vault();
        let idx = TantivyIndex::build(dir.path(), &notes).unwrap();

        // Initially "pasta" only in cooking.md
        assert_eq!(idx.search("pasta", 10).unwrap().len(), 1);

        // Rewrite python.md to mention pasta
        std::fs::write(
            dir.path().join("python.md"),
            "# Python\n\nPython and pasta recipes.\n",
        )
        .unwrap();

        let meta = make_meta("python.md", "python", &[], &["Python"]);
        notes.insert(PathBuf::from("python.md"), meta.clone());
        idx.reindex_file(dir.path(), Path::new("python.md"), &meta)
            .unwrap();

        let results = idx.search("pasta", 10).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn remove_file_removes_from_results() {
        let (dir, notes) = setup_vault();
        let idx = TantivyIndex::build(dir.path(), &notes).unwrap();

        assert_eq!(idx.search("pasta", 10).unwrap().len(), 1);

        idx.remove_file(Path::new("cooking.md")).unwrap();

        assert!(idx.search("pasta", 10).unwrap().is_empty());
    }

    #[test]
    fn stringify_frontmatter_extracts_text() {
        let fm = serde_json::json!({
            "tags": ["rust", "mcp"],
            "status": "draft",
            "priority": 1
        });
        let text = stringify_frontmatter(&fm);
        assert!(text.contains("rust"));
        assert!(text.contains("mcp"));
        assert!(text.contains("draft"));
        assert!(text.contains("1"));
    }

    // ── search_with_options tests ──────────────────────────────────────

    #[test]
    fn search_with_options_delegates_to_search_when_no_options() {
        let (dir, notes) = setup_vault();
        let idx = TantivyIndex::build(dir.path(), &notes).unwrap();

        let plain = idx.search("rust", 10).unwrap();
        let via_opts = idx.search_with_options("rust", 10, false, None).unwrap();
        assert_eq!(plain.len(), via_opts.len());
    }

    #[test]
    fn search_fuzzy_tolerates_typo() {
        let (dir, notes) = setup_vault();
        let idx = TantivyIndex::build(dir.path(), &notes).unwrap();

        // "rast" is one edit away from "rust"
        let strict = idx.search("rast", 10).unwrap();
        let fuzzy = idx.search_with_options("rast", 10, true, None).unwrap();
        assert!(
            fuzzy.len() > strict.len(),
            "fuzzy should find more results than strict for a typo query"
        );
        assert!(fuzzy.iter().any(|(p, _)| p == Path::new("rust.md")));
    }

    #[test]
    fn search_with_field_filter_title_only() {
        let (dir, notes) = setup_vault();
        let idx = TantivyIndex::build(dir.path(), &notes).unwrap();

        let results = idx
            .search_with_options("rust", 10, false, Some(&[SearchField::Title]))
            .unwrap();
        assert!(results.iter().any(|(p, _)| p == Path::new("rust.md")));
        // "cooking" has no title relation to "rust"
        assert!(!results.iter().any(|(p, _)| p == Path::new("cooking.md")));
    }

    #[test]
    fn search_with_field_filter_body_only() {
        let (dir, notes) = setup_vault();
        let idx = TantivyIndex::build(dir.path(), &notes).unwrap();

        // "pasta" appears only in cooking.md body
        let results = idx
            .search_with_options("pasta", 10, false, Some(&[SearchField::Body]))
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, PathBuf::from("cooking.md"));
    }

    #[test]
    fn facet_term_query_roundtrip() {
        let (dir, notes) = setup_vault();
        let idx = TantivyIndex::build(dir.path(), &notes).unwrap();

        let searcher = idx.reader.searcher();
        let facet = Facet::from_path(["programming"]);
        let term = Term::from_facet(idx.ts.f_tags, &facet);
        let tq = TermQuery::new(term, IndexRecordOption::Basic);
        let top_docs = searcher.search(&tq, &TopDocs::with_limit(10)).unwrap();
        assert!(
            !top_docs.is_empty(),
            "direct facet TermQuery should find rust.md"
        );
    }

    #[test]
    fn search_with_field_filter_tags() {
        let (dir, notes) = setup_vault();
        let idx = TantivyIndex::build(dir.path(), &notes).unwrap();

        // "programming" is a tag on rust.md
        let results = idx
            .search_with_options("programming", 10, false, Some(&[SearchField::Tags]))
            .unwrap();
        assert!(
            results.iter().any(|(p, _)| p == Path::new("rust.md")),
            "tag search should find rust.md tagged 'programming'"
        );
    }

    #[test]
    fn search_with_options_empty_query() {
        let (dir, notes) = setup_vault();
        let idx = TantivyIndex::build(dir.path(), &notes).unwrap();

        assert!(
            idx.search_with_options("", 10, true, None)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn search_fuzzy_and_field_combined() {
        let (dir, notes) = setup_vault();
        let idx = TantivyIndex::build(dir.path(), &notes).unwrap();

        // "pythn" is a typo for "python", search only in titles
        let results = idx
            .search_with_options("pythn", 10, true, Some(&[SearchField::Title]))
            .unwrap();
        assert!(
            results.iter().any(|(p, _)| p == Path::new("python.md")),
            "fuzzy + title filter should find python.md for 'pythn'"
        );
    }
}
