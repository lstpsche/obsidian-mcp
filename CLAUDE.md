# CLAUDE.md — Agent Context for obsidian-mcp

> **Self-update rule:** If you learn something new about this project — a design decision,
> a gotcha, a pattern that worked or failed, a dependency quirk, or any other durable fact —
> update this file before finishing your task. Keep entries concise and in the right section.
> Don't duplicate what's already here; amend or refine existing entries instead.

## What This Project Is

**obsidian-mcp** is a Rust MCP (Model Context Protocol) server that gives AI agents full
read/write access to an Obsidian vault through direct filesystem operations.

It replaces two existing projects:
- [mcp-obsidian](https://github.com/MarkusPfundstein/mcp-obsidian) — a Python MCP server that wraps the Obsidian REST API
- [obsidian-local-rest-api](https://github.com/coddingtonbear/obsidian-local-rest-api) — an Obsidian community plugin that exposes a local REST API

**Key difference:** This project talks directly to the vault filesystem. No Obsidian plugins,
no Obsidian running, no HTTP. Just a single static binary that reads/writes markdown files.

## Architecture

```
AI Client (Cursor / Claude Desktop / etc.)
    │ stdio (MCP JSON-RPC)
    ▼
obsidian-mcp binary
    ├── src/tools/       ← MCP tool handlers (rmcp #[tool_router])
    ├── src/vault/       ← vault layer (fs, parsing, index, watcher)
    ├── src/models.rs    ← shared types
    ├── src/config.rs    ← env/CLI config
    └── src/error.rs     ← unified VaultError
    │ filesystem
    ▼
~/vault/  (directory of .md files + .obsidian/ config)
```

### Data flow

1. AI client sends MCP tool call over stdio
2. `rmcp` deserializes into tool params, routes to handler
3. Handler calls `Vault` methods
4. `Vault` delegates to vault sub-modules (fs, parser, frontmatter, index, patch, periodic)
5. Result serialized back as MCP response

### Concurrency model

- The `Vault` struct wraps the index in `Arc<RwLock<VaultIndex>>`
- Tool handlers take `&self` (shared reference) — reads acquire read lock, writes acquire write lock
- The filesystem watcher runs in a background tokio task, updating the index on file changes
- The `Vault` is `Send + Sync + Clone` so rmcp can share it across async tasks

## Technology Stack

| Component | Crate | Purpose |
|-----------|-------|---------|
| MCP protocol | `rmcp` 1.2 | Server, tools, stdio transport |
| Async runtime | `tokio` 1 | async/await, tasks, IO |
| Serialization | `serde`, `serde_json`, `serde_yaml` | JSON + YAML frontmatter |
| JSON Schema | `schemars` 1.0 | Auto-generate schemas for tool params |
| Markdown | `pulldown-cmark` 0.12 | Parse headings, detect code blocks |
| Filesystem | `walkdir` 2, `notify` 7, `globset` 0.4 | Walk dirs, watch changes, glob match |
| Regex | `regex` 1 | Wikilink/tag/block-ref extraction, search |
| Full-text search | `tantivy` 0.22 | BM25 inverted index, stemming, ranked search |
| Dates | `chrono` 0.4 | Periodic notes date handling |
| Errors | `thiserror` 2 | Derive error types |
| Logging | `tracing` 0.1, `tracing-subscriber` 0.3 | Structured logging |

## Obsidian Vault Format Reference

### File structure
- Vault = a directory on disk containing `.md` files
- `.obsidian/` at vault root holds app config (settings, plugin data, themes)
- Notes are plain UTF-8 markdown with optional YAML frontmatter

### Frontmatter (Properties)
```yaml
---
tags: [rust, mcp]
aliases: [obsidian-server]
date: 2026-03-19
custom_field: any value
---
```
- YAML between `---` delimiters at the very start of the file
- The `tags` and `aliases` fields have special meaning to Obsidian
- Values can be text, lists, numbers, booleans, dates

### Wikilinks
| Syntax | Meaning |
|--------|---------|
| `[[note]]` | Link to note (resolved by shortest unique path) |
| `[[note\|alias]]` | Link with display text |
| `[[note#heading]]` | Link to heading in note |
| `[[note#^blockid]]` | Link to block reference |
| `[[#heading]]` | Link to heading in current note |
| `![[note]]` | Embed (transclude) note content |

**Resolution:** Obsidian uses shortest-unique-path matching. `[[foo]]` resolves to the
only `foo.md` in the vault regardless of folder depth. If ambiguous, full path is needed.

### Tags
- Inline: `#tag`, `#nested/tag` — must start with a letter, case-insensitive
- Frontmatter: `tags: [tag1, tag2]`
- Cannot be purely numeric (`#123` is invalid, `#y123` is valid)

### Block references
- `^blockid` at the end of a line or on its own line
- IDs: alphanumeric + dashes
- Referenced via `[[note#^blockid]]`

### Periodic notes config
- Core daily notes: `.obsidian/daily-notes.json`
  ```json
  { "format": "YYYY-MM-DD", "folder": "Daily", "template": "Templates/Daily" }
  ```
- Periodic Notes plugin: `.obsidian/plugins/periodic-notes/data.json`
- Date format uses Moment.js tokens (YYYY, MM, DD, dddd, etc.)

## rmcp Patterns

### Tool definition
```rust
#[derive(Deserialize, JsonSchema, Default)]
struct MyParams {
    /// Description shown to the AI
    field: String,
}

#[tool(name = "tool_name", description = "What the tool does")]
async fn tool_name(
    &self,
    Parameters(params): Parameters<MyParams>,
) -> Result<CallToolResult, ErrorData> {
    // implementation
}
```

### Server setup
```rust
struct ObsidianMcp {
    vault: Vault,
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl ObsidianMcp {
    // #[tool] methods here
    pub fn new(vault: Vault) -> Self {
        Self { tool_router: Self::tool_router(), vault }
    }
}

#[tool_handler]
impl ServerHandler for ObsidianMcp {
    fn get_info(&self) -> ServerInfo { /* ... */ }
}
```

### Error conversion
`VaultError` implements `From<VaultError> for rmcp::ErrorData` so tools can use `?` operator.

### Return types
- `Result<CallToolResult, ErrorData>` — explicit control
- `Result<String, ErrorData>` — auto-wrapped as text content
- `Json<T>` where `T: Serialize + JsonSchema` — structured JSON output

## Project Conventions

### Code style
- Follow `rustfmt` defaults
- `clippy` must pass with no warnings
- No `unwrap()` in library code — use proper error propagation
- `unwrap()` only acceptable in tests and `main()` setup

### File paths
- All public APIs use paths **relative to vault root**
- Internal functions receive `vault_root: &Path` + `relative: &Path` separately
- Always validate that resolved absolute paths don't escape vault root (path traversal prevention)

### Error handling
- All vault operations return `VaultResult<T>` (alias for `Result<T, VaultError>`)
- `VaultError` variants are descriptive (include the path, the target, etc.)
- MCP tools convert to `rmcp::ErrorData` at the boundary

### Naming
- MCP tool names: `snake_case` (e.g., `vault_list`, `note_read`, `search_text`)
- Rust modules: `snake_case`
- Types: `PascalCase`
- Tool parameter structs: `{ToolName}Params` (e.g., `VaultListParams`)

### Module boundaries
- `src/vault/` — knows nothing about MCP; pure vault operations
- `src/tools/` — knows about MCP (rmcp types) and delegates to `Vault`
- `src/models.rs` — shared types used by both layers
- `src/error.rs` — `VaultError` + conversion to `rmcp::ErrorData`

## Development Setup

```bash
# Build
cargo build

# Run (requires vault path)
OBSIDIAN_VAULT_PATH=~/my-vault cargo run

# Check
cargo fmt --check && cargo clippy && cargo test

# Test with MCP Inspector
npx @modelcontextprotocol/inspector cargo run
```

### Environment variables
| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OBSIDIAN_VAULT_PATH` | Yes | — | Absolute path to Obsidian vault root |
| `OBSIDIAN_WATCH` | No | `true` | Enable filesystem watcher |
| `OBSIDIAN_LOG_LEVEL` | No | `info` | Tracing log level |
| `OBSIDIAN_TANTIVY` | No | `true` | Enable Tantivy BM25 full-text index |
| `OBSIDIAN_EMBEDDINGS` | No | `false` | Enable semantic embedding search (requires `embeddings` feature) |
| `OBSIDIAN_EMBEDDINGS_MODEL` | No | `BAAI/bge-small-en-v1.5` | HuggingFace model name for embeddings |

## Known Gotchas & Decisions

<!-- Add entries here as you discover them during implementation -->

- **schemars version:** rmcp 1.2 uses schemars 1.0 (not 0.8). Use `schemars = "1.0"` as direct dep or `use rmcp::schemars;` re-export. The `server` feature activates `dep:schemars` automatically.
- **Self-signed certs:** Not applicable — we don't use HTTP. This is a pure filesystem + stdio project.
- **Wikilink ambiguity:** When multiple files share the same stem, Obsidian considers the link ambiguous and doesn't resolve it. Our `LinkResolver` should return `None` for ambiguous stems.
- **Frontmatter serialization:** `serde_yaml` may reorder keys. This is acceptable — Obsidian doesn't care about key order. But body content after `---` must be preserved byte-for-byte.
- **Code block filtering:** When extracting inline tags and wikilinks, skip content inside fenced code blocks (```...```) and inline code spans (`...`). Missing this causes false positives.
- **notify on macOS:** The `notify` crate uses FSEvents on macOS. Debouncing (~500ms) is essential to avoid duplicate events.
- **Path separators:** Internally use `/` (forward slash) for all vault-relative paths, even on Windows. Convert only at the filesystem boundary.
- **Large vaults:** The in-memory index holds `NoteMetadata` for every `.md` file but does NOT store file content. Content is read on demand for search operations. This keeps memory reasonable for vaults up to ~50k notes.
- **Moment.js date format:** Obsidian uses Moment.js tokens (YYYY, MM, DD, etc.). We convert these to `chrono` format specifiers. The mapping must handle longest-token-first to avoid `MMMM` being partially matched as `MM`. Square-bracket escaping (`[literal]`) must be extracted before token replacement. The quarter token `Q` has no chrono equivalent — we inject a placeholder and resolve it after formatting.
- **Periodic Notes plugin config:** The community plugin stores config in `.obsidian/plugins/periodic-notes/data.json`. Newer versions use a `calendarSets` array with `day`/`week`/`month`/`quarter`/`year` keys; older versions use a flat format with `daily`/`weekly`/`monthly`/`quarterly`/`yearly` keys. Our deserialization uses `#[serde(untagged)]` to handle both. The `LegacyPluginConfig` variant is `Box`ed to satisfy `clippy::large_enum_variant`.
- **Partial date parsing:** When listing periodic notes, filenames for monthly/yearly/quarterly formats don't encode a full date. The `try_parse_date` helper tries appending defaults (day-01, month-01-day-01) to produce a `NaiveDate` for sorting. ISO week formats (`%G-W%V`) need an appended weekday (`%u`) to parse.
- **NaiveDate and time tokens:** `format_date` uses `NaiveDate::format`, which cannot resolve time specifiers (`%H`, `%M`, etc.). Periodic note formats should only contain date tokens. The `expand_template` function handles `{{time}}` separately via `Local::now()`.
- **Edition 2024:** The project uses Rust edition 2024. Ensure your toolchain supports it (`rustup update`).
- **resolve_path two-layer validation:** `resolve_path` first normalizes the relative path manually (rejecting `..` escapes), then canonicalizes if the target exists (catching symlink escapes). For non-existent paths (write targets), manual normalization alone is sufficient since there are no symlinks to follow.
- **tempfile dev-dep:** `tempfile = "3"` is used for test isolation. All fs tests create a `TempDir` vault so they don't touch the real filesystem.
- **Frontmatter-aware scanning:** When scanning for headings or block refs, skip YAML frontmatter at file start (`---…---`). YAML comments (`# comment`) inside frontmatter would otherwise be parsed as ATX headings.
- **Heading path delimiter:** Patch operations use `::` to express nested heading paths (e.g., `"H1::Sub Heading"`). Matching is case-insensitive (`eq_ignore_ascii_case`) consistent with Obsidian's heading link resolution.
- **serde_yaml::to_string format:** `serde_yaml` 0.9 does NOT prefix serialized output with `---`. No stripping needed, but the code handles it defensively.
- **notify-debouncer-mini event erasure:** `notify-debouncer-mini` 0.5 collapses all event kinds (create, modify, delete, rename) into `DebouncedEventKind::Any`. The watcher disambiguates by checking the filesystem at event time: path exists → `reindex_file`, path gone → `remove_file`. Renames decompose into two separate events (old path disappears, new path appears). Also, the error variant in `DebounceEventResult` is a single `notify::Error`, not a `Vec`, despite some docs suggesting otherwise.
- **notify-debouncer-mini tokio bridge:** Version 0.5 has no native async support. The watcher bridges to tokio by capturing `Handle::current()` and using `rt.spawn()` inside the debouncer callback to send events through a `tokio::sync::mpsc` channel. The `Debouncer` must be kept alive (stored in the Vault struct); dropping it stops watching.
- **VaultIndex backlink rebuild:** Mutation methods (`reindex_file`, `remove_file`, `rename_file`) call `rebuild_backlinks()` which recomputes the entire backlinks map from scratch. This is O(total_links) but ensures correctness when a file addition/removal changes the `LinkResolver`'s ambiguity state for other notes' wikilinks. Acceptable for vaults up to ~50k notes since link resolution is O(1) per link.
- **VaultIndex::empty():** An `empty()` constructor exists for creating an uninitialized index, used by the watcher tests and any pre-initialization scenarios. Always prefer `VaultIndex::build()` for production use.
- **Search char offsets:** `SearchMatch.match_start` and `match_end` are **character** offsets within `context`, not byte offsets. This ensures correctness for consumers (AI agents) in languages like Python/JS that use character-based string indexing.
- **Vault struct pattern:** `Vault` uses `Arc<VaultInner>` for `Clone + Send + Sync`. The `index` field is `Arc<RwLock<VaultIndex>>` (shared with the watcher). The `_watcher` field is wrapped in `std::sync::Mutex` because `Debouncer` contains `mpsc::Sender` which is `Send` but not `Sync`; the Mutex is never actually locked after construction.
- **Write-then-reindex pattern:** All write operations (write/append/delete/move/patch/frontmatter) perform the fs mutation first (no lock held), then briefly acquire `index.write()` for the reindex. The watcher also fires for the same change, but `reindex_file` is idempotent so double-updates are harmless.
- **Heading patch scope:** A heading replace operation replaces everything from the heading line to the next heading of **same or higher level**. A `# H1` section includes all its `## H2` sub-sections. To patch only the H1 content without sub-sections, use the nested heading path syntax (`H1::SubHeading`).
- **rmcp 1.2 `Parameters` import:** The `Parameters` extractor is at `rmcp::handler::server::wrapper::Parameters`, NOT `rmcp::handler::server::tool::Parameters` (the latter is a private re-export path). The MCP quickstart docs reference rmcp 0.3 which had a different module layout.
- **rmcp 1.2 `ServerInfo` is `#[non_exhaustive]`:** `ServerInfo` is a type alias for `InitializeResult` which is `#[non_exhaustive]`. Cannot use struct literal syntax. Use builder: `ServerInfo::new(capabilities).with_server_info(...).with_instructions(...)`.
- **rmcp 1.2 `ServiceExt::serve`:** Import `rmcp::ServiceExt` to get the `.serve()` method. For stdio: `.serve(rmcp::transport::io::stdio())`. Returns `RunningService`; call `.waiting().await?` to block until shutdown.
- **lib + bin crate split:** The project has both `src/lib.rs` (re-exports all modules) and `src/main.rs` (thin binary entry point). This enables integration tests in `tests/` to `use obsidian_mcp::*`. The `main.rs` imports from `obsidian_mcp::` (library crate name), not `crate::`.
- **Integration test runtime:** Tests using `#[tokio::test]` must not call `tokio::runtime::Runtime::new()` — that panics with "Cannot start a runtime from within a runtime." Use async helpers instead. The shared read-only `VAULT` static uses `LazyLock` with its own runtime since it initializes outside of any tokio context.
- **Fixture vault:** `tests/fixtures/test_vault/` is a committed static vault used by integration tests. Read tests reference it directly; write tests copy it to a `TempDir` first. The `walkdir` crate (a regular dependency) powers the recursive copy in tests.
- **Integration test location:** Tests live in `tests/integration_tests.rs` (single file with nested `mod` blocks), not a `tests/integration/` directory. 36 tests covering read, search, graph, write, and periodic note operations.
- **`VaultError::InvalidRegex` variant:** The `search_regex` tool validates the regex pattern before searching. Invalid patterns produce `InvalidRegex { pattern, source }` which maps to `ErrorCode::INVALID_PARAMS` at the MCP boundary.
- **`.gitattributes` LF line endings:** The repo includes `* text=auto eol=lf` to enforce LF line endings on all platforms. This is required for Windows CI — without it, `cargo fmt --check` fails due to CRLF mismatches.
- **CI/CD workflows:** Two GitHub Actions workflows: `ci.yml` runs `cargo fmt --check` (Ubuntu only) + `cargo clippy` + `cargo test` on a 3-OS matrix (ubuntu-latest, macos-latest, windows-latest). `release.yml` triggers on `v*` tags, builds 5 targets (x86_64-unknown-linux-gnu, aarch64-unknown-linux-gnu, x86_64-apple-darwin, aarch64-apple-darwin, x86_64-pc-windows-msvc), creates a GitHub Release with checksums, and publishes to crates.io.
- **macOS CI runners:** GitHub deprecated `macos-13`. Both macOS targets (x86_64 and aarch64) now use `macos-latest` in the release workflow. The x86_64 build runs natively (Rosetta) since `macos-latest` is ARM.
- **v0.1.0 release:** The initial version is published on both [crates.io](https://crates.io/crates/obsidian-mcp) and [GitHub Releases](https://github.com/lstpsche/obsidian-mcp/releases).
- **Tantivy BM25 index:** Layer 1 of semantic search. Uses `tantivy` 0.22 with an in-RAM index (`Index::create_in_ram()`). Schema: `path` (STRING|STORED), `title` (en_stem, stored, boost 5), `headings` (en_stem, boost 3), `tags` (FACET, stored, boost 4), `body` (en_stem, boost 1), `frontmatter_text` (en_stem, boost 2). Rebuilds in <100ms for 5000 notes. `TantivySchema` in `src/vault/tantivy_index.rs` holds the schema + cached `Field` handles.
- **TantivyIndex struct:** `TantivyIndex` in `src/vault/tantivy_index.rs` wraps the Tantivy in-RAM index with `build`/`reindex_file`/`remove_file`/`search` methods. Uses `ReloadPolicy::Manual` with explicit `reader.reload()` after each commit for deterministic read-after-write. Writer is behind `std::sync::Mutex` since `IndexWriter::commit` takes `&mut self`. Search uses `QueryParser` with field-level boosts from `TantivySchema` constants.
- **Tantivy Vault integration:** `VaultInner` holds `tantivy: Option<Arc<TantivyIndex>>`, gated by `Config::tantivy` (default `true`). Built during `Vault::open()` from the completed `VaultIndex::notes()`. All write ops (`reindex`/`delete_note`/`move_note`) and the filesystem watcher sync both indices. The watcher receives `Option<Arc<TantivyIndex>>` and updates Tantivy inline after each `VaultIndex` mutation.
- **Tantivy ReloadPolicy:** Tantivy 0.22 renamed `OnCommit` to `OnCommitWithDelay` (polls asynchronously). Only two variants exist: `Manual` and `OnCommitWithDelay`. We use `Manual` + explicit `reload()` for correctness.
- **Tantivy stemming:** The `en_stem` tokenizer uses the Porter stemmer. "programs"/"programming" share the stem "program", but "programmer" stems differently ("programm"). Test stemming with words that share the same root.
- **Tantivy two-phase search:** `search_text` uses a two-phase strategy when Tantivy is enabled: (1) BM25 ranking via Tantivy returns top-K `(path, score)` pairs, (2) for each hit, the file is read and a case-insensitive regex of the original query words extracts context snippets. If a note was matched only via stemming (no literal occurrence of query words), the `matches` array is empty but `score` is populated — the agent uses the score to know the note is relevant and can `note_read` it. Falls back to the regex scan when Tantivy is disabled.
- **Tantivy fuzzy search:** `search_text_with_options` supports `fuzzy: bool` which enables edit-distance-1 fuzzy matching via `QueryParser::set_field_fuzzy`. This tolerates single-character typos (insertions, deletions, substitutions, transpositions).
- **Tantivy field filtering:** `search_text_with_options` supports `fields: Option<&[SearchField]>` to restrict search to specific note fields (title, headings, body, frontmatter, tags). Text fields use `QueryParser` with field-specific boosts. Tags use facet `TermQuery` since they're indexed as Tantivy facets, not text.
- **Tantivy facet encoding:** `Facet::from_text("bare_word")` returns `Err(FacetParseError)` in tantivy 0.22 — it requires a leading `/`. Use `Facet::from_path(components)` instead, which constructs the facet directly from path components. Tags like `"nested/tag"` are split on `/` and passed as components: `Facet::from_path(["nested", "tag"])`.
- **Tantivy tags in default search:** The base `TantivyIndex::search()` (used by `search_text`) does not create facet `TermQuery` for tags — only `search_with_options` does when `fields` explicitly includes `SearchField::Tags`. However, tags are still searchable in the default path because `stringify_frontmatter` flattens the entire frontmatter object (including `tags: [...]`) into the `frontmatter_text` field, which is indexed with `en_stem`. This is by design: facet queries provide exact tag matching, while the text path gives stemmed/fuzzy matching for free.
- **Embedding search (Layer 2):** Opt-in via `OBSIDIAN_EMBEDDINGS=true` + `--features embeddings` Cargo feature. Fully integrated: `EmbeddingStore` (cosine similarity, bincode persistence) and `EmbeddingModel` (fastembed wrapper) are held in `VaultInner`, initialized in `Vault::open()`, and synced on all write operations. Exposed as `search_semantic` MCP tool.
- **fastembed `&mut self`:** `fastembed::TextEmbedding::embed()` takes `&mut self`, so `EmbeddingModel` wraps it in `std::sync::Mutex`. The Mutex is locked only during inference calls.
- **bincode 2 serde compat:** Use `bincode::serde::encode_to_vec` / `decode_from_slice` with `bincode::config::standard()` for embedding cache persistence. The bincode dep requires `features = ["serde"]`.
- **fastembed model resolution:** `fastembed::EmbeddingModel` implements `FromStr`. The config string (e.g. `"BAAI/bge-small-en-v1.5"`) is parsed via `.parse()`, falling back to `Default` (BGESmallENV15, dim=384) on failure. `get_model_info` requires importing `fastembed::ModelTrait`.
- **Embedding cache format:** Binary file at `{vault}/.obsidian/obsidian-mcp/embeddings.bin`. Serialized via bincode serde compat with an `EmbeddingCacheData { dim, entries: Vec<(String, Vec<f32>)> }` struct. Not forward-compatible — cache version changes require rebuild.
- **Embedding Vault integration:** `VaultInner` holds `embedding_model: Option<Arc<EmbeddingModel>>` and `embedding_store: Option<Arc<RwLock<EmbeddingStore>>>`, both `#[cfg(feature = "embeddings")]`. `RwLock` for the store (not `Mutex`) because reads (search queries) vastly outnumber writes (note edits). The model is `Arc` only (thread-safe via its internal Mutex). Cache is persisted to disk after every write that modifies the store.
- **Embedding staleness check:** On startup, if the cached store's note count doesn't match `VaultIndex::notes().len()`, the cache is rebuilt. Per-note content hashing is overkill for MVP — incremental updates handle individual changes after startup.
- **Non-fatal embedding errors in write path:** If embedding fails during `reindex`, the error is logged but doesn't propagate — the note is still written and indexed by VaultIndex and Tantivy. Degraded semantic search is better than a failed write.
- **Watcher feature-gating:** The `start_watcher` and `process_event` functions have two `#[cfg]`-gated versions each (with/without embeddings). This is necessary because `#[cfg]` on function parameters isn't supported in Rust. The watcher tests use a `call_start_watcher` helper that dispatches to the correct variant.
- **rmcp `#[tool_router]` and `#[cfg]`:** The `#[tool_router]` proc macro expands before `cfg` is resolved, so `#[cfg(feature = "...")]` on individual `#[tool]` methods doesn't work — the macro generates references to the method unconditionally. The workaround is to always define the tool method and params, but provide two `#[cfg]`-gated implementations: one that delegates to the real logic, and one that returns an error explaining the feature isn't compiled in.
- **fastembed concurrent model loading:** `fastembed::TextEmbedding::try_new()` has race conditions when multiple instances try to load the same model concurrently (corrupted cache access). Integration tests use a static `tokio::sync::Mutex` to serialize `Vault::open()` calls that trigger model loading.
- **Hybrid re-ranking (E7):** `search_semantic` with `lexical_prefetch: true` runs a two-stage pipeline: Tantivy BM25 retrieves top-50 candidates, then each is re-scored via `alpha * norm_bm25 + (1-alpha) * cosine_sim` (alpha=0.4). BM25 scores are min-max normalized to [0,1] within the result set; when all scores are equal they normalize to 1.0. Notes missing from the embedding store (e.g., embedding failed) get semantic score 0.0. Requires both Tantivy (`OBSIDIAN_TANTIVY=true`) and embeddings (`OBSIDIAN_EMBEDDINGS=true` + `--features embeddings`).

## Task Tracking

The full development plan with task breakdown, dependencies, and parallelization info is in `TODO.md`.

## Links

- **`CallToolResult::structured()` rejects null:** The Cursor MCP client validates `structuredContent` as a record (object). Passing `serde_json::Value::Null` via `CallToolResult::structured()` triggers a validation error. When a tool may return null, use `CallToolResult::success(vec![Content::text("null")])` for the null case and `CallToolResult::structured(value)` for real objects.
- **`create_periodic_note` content override:** The method now accepts `content_override: Option<&str>` as the third parameter. When provided, it skips template expansion entirely. When `None`, missing template files are handled gracefully (`unwrap_or_default`) instead of erroring.

## Links

- [GitHub repository](https://github.com/lstpsche/obsidian-mcp)
- [crates.io](https://crates.io/crates/obsidian-mcp)
- [rmcp docs](https://docs.rs/rmcp/latest/rmcp/)
- [Obsidian help — Properties](https://help.obsidian.md/Editing+and+formatting/Properties)
- [Obsidian help — Internal links](https://help.obsidian.md/Linking+notes+and+files/Internal+links)
- [Obsidian help — Tags](https://help.obsidian.md/Editing+and+formatting/Tags)
- [Obsidian local REST API — OpenAPI spec](https://coddingtonbear.github.io/obsidian-local-rest-api/openapi.yaml) (reference for what the REST API covers)
- [MCP specification](https://modelcontextprotocol.io/)
