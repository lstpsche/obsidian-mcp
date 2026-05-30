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
| MCP protocol | `rmcp` 1.6 | Server, tools, stdio + Streamable HTTP transport |
| HTTP framework | `axum` 0.8 | HTTP routing for Streamable HTTP transport |
| Async runtime | `tokio` 1 | async/await, tasks, IO |
| Serialization | `serde`, `serde_json`, `yaml_serde` 0.10 | JSON + YAML frontmatter |
| JSON Schema | `schemars` 1.0 | Auto-generate schemas for tool params |
| Markdown | `pulldown-cmark` 0.13 | Parse headings, detect code blocks |
| Filesystem | `walkdir` 2, `notify` 8, `globset` 0.4 | Walk dirs, watch changes, glob match |
| Regex | `regex` 1 | Wikilink/tag/block-ref extraction, search |
| Full-text search | `tantivy` 0.26 | BM25 inverted index, stemming, ranked search |
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
- MCP tools convert vault errors to `rmcp::ErrorData` at the boundary via `?`
- Tool-layer parameter validation (invalid action/query/view, missing required param) uses `ErrorData::new(ErrorCode::INVALID_PARAMS, ...)` directly — not `VaultError::Other` which maps to `INTERNAL_ERROR`

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
| `OBSIDIAN_TRANSPORT` | No | `stdio` | Transport: `stdio` or `http` |
| `OBSIDIAN_HTTP_PORT` | No | `37842` | HTTP listen port (only used with `http` transport) |
| `OBSIDIAN_HTTP_HOST` | No | `127.0.0.1` | HTTP bind address (only used with `http` transport) |
| `OBSIDIAN_WATCH` | No | `true` | Enable filesystem watcher |
| `OBSIDIAN_LOG_LEVEL` | No | `info` | Tracing log level |
| `OBSIDIAN_TANTIVY` | No | `true` | Enable Tantivy BM25 full-text index |
| `OBSIDIAN_EMBEDDINGS` | No | `false` | Enable semantic embedding search (requires `embeddings` feature) |
| `OBSIDIAN_EMBEDDINGS_MODEL` | No | `BAAI/bge-small-en-v1.5` | HuggingFace model name for embeddings |
| `OBSIDIAN_HYBRID_ALPHA` | No | `0.25` | Hybrid search blending weight: `alpha * BM25 + (1-alpha) * semantic`. Clamped to [0.0, 1.0]. |
| `OBSIDIAN_EMBEDDING_PROVIDER` | No | *(infer)* | Embedding backend: `local` (fastembed) or `api` (OpenAI-compatible). Omit to infer from compiled features. |
| `OBSIDIAN_TOOLS` | No | *(unset = full)* | Tool filtering: profile name (`full`/`core`/`read`/`minimal`), comma-separated allow-list, or `!`-prefixed deny-list |
| `OBSIDIAN_MCP_DATA` | No | `{vault}/.obsidian-mcp` | External directory for obsidian-mcp data. When set, embeddings and machine-specific config stored under `{value}/vaults/{vault_slug}/`. Config merged with vault-root `.obsidian-mcp/`. |
| `OBSIDIAN_EXCLUDE_PATHS` | No | *(empty)* | Comma-separated glob patterns for paths to exclude from indexing and search |

## Known Gotchas & Decisions

<!-- Add entries here as you discover them during implementation -->

- **schemars version:** rmcp 1.2 uses schemars 1.0 (not 0.8). Use `schemars = "1.0"` as direct dep or `use rmcp::schemars;` re-export. The `server` feature activates `dep:schemars` automatically.
- **Self-signed certs:** Not applicable — we don't use HTTP. This is a pure filesystem + stdio project.
- **Wikilink ambiguity:** When multiple files share the same stem, Obsidian considers the link ambiguous and doesn't resolve it. Our `LinkResolver` should return `None` for ambiguous stems.
- **Frontmatter serialization:** `yaml_serde` (actively maintained fork of deprecated `serde_yaml`) may reorder keys. This is acceptable — Obsidian doesn't care about key order. But body content after `---` must be preserved byte-for-byte. The Cargo.toml uses `serde_yaml = { package = "yaml_serde", version = "0.10" }` so existing `use serde_yaml::` imports work unchanged.
- **Code block filtering:** When extracting inline tags and wikilinks, skip content inside fenced code blocks (```...```) and inline code spans (`...`). Missing this causes false positives.
- **notify on macOS:** The `notify` crate uses FSEvents on macOS. Debouncing (~500ms) is essential to avoid duplicate events.
- **Path separators:** Internally use `/` (forward slash) for all vault-relative paths, even on Windows. Convert only at the filesystem boundary.
- **Large vaults:** The in-memory index holds `NoteMetadata` for every `.md` file but does NOT store file content. Content is read on demand for search operations. This keeps memory reasonable for vaults up to ~50k notes.
- **Vault path quoting:** Some MCP clients may pass the vault path wrapped in literal quotes (for example `"/path/to/vault"` as a single argument value). `Config::load()` now normalizes surrounding matching quotes and whitespace for the vault path before validation, preventing false `InvalidPath` failures.
- **Moment.js date format:** Obsidian uses Moment.js tokens (YYYY, MM, DD, etc.). We convert these to `chrono` format specifiers. The mapping must handle longest-token-first to avoid `MMMM` being partially matched as `MM`. Square-bracket escaping (`[literal]`) must be extracted before token replacement. The quarter token `Q` has no chrono equivalent — we inject a placeholder and resolve it after formatting.
- **Periodic Notes plugin config:** The community plugin stores config in `.obsidian/plugins/periodic-notes/data.json`. Newer versions use a `calendarSets` array with `day`/`week`/`month`/`quarter`/`year` keys; older versions use a flat format with `daily`/`weekly`/`monthly`/`quarterly`/`yearly` keys. Our deserialization uses `#[serde(untagged)]` to handle both. The `LegacyPluginConfig` variant is `Box`ed to satisfy `clippy::large_enum_variant`.
- **Partial date parsing:** When listing periodic notes, filenames for monthly/yearly/quarterly formats don't encode a full date. The `try_parse_date` helper tries appending defaults (day-01, month-01-day-01) to produce a `NaiveDate` for sorting. ISO week formats (`%G-W%V`) need an appended weekday (`%u`) to parse.
- **NaiveDate and time tokens:** `format_date` uses `NaiveDate::format`, which cannot resolve time specifiers (`%H`, `%M`, etc.). Periodic note formats should only contain date tokens. The `expand_template` function handles `{{time}}` separately via `Local::now()`.
- **Edition 2024:** The project uses Rust edition 2024. Ensure your toolchain supports it (`rustup update`).
- **resolve_path two-layer validation:** `resolve_path` first normalizes the relative path manually (rejecting `..` escapes), then canonicalizes if the target exists (catching symlink escapes). For non-existent paths (write targets), manual normalization alone is sufficient since there are no symlinks to follow.
- **tempfile dev-dep:** `tempfile = "3"` is used for test isolation. All fs tests create a `TempDir` vault so they don't touch the real filesystem.
- **Frontmatter-aware scanning:** When scanning for headings or block refs, skip YAML frontmatter at file start (`---…---`). YAML comments (`# comment`) inside frontmatter would otherwise be parsed as ATX headings.
- **Heading path delimiter:** Patch operations use `::` to express nested heading paths (e.g., `"H1::Sub Heading"`). Matching is case-insensitive (`eq_ignore_ascii_case`) consistent with Obsidian's heading link resolution.
- **yaml_serde::to_string format:** `yaml_serde` 0.10 (like its predecessor `serde_yaml` 0.9) does NOT prefix serialized output with `---`. No stripping needed, but the code handles it defensively.
- **notify-debouncer-mini event erasure:** `notify-debouncer-mini` 0.7 collapses all event kinds (create, modify, delete, rename) into `DebouncedEventKind::Any`. The watcher disambiguates by checking the filesystem at event time: path exists → `reindex_file`, path gone → `remove_file`. Renames decompose into two separate events (old path disappears, new path appears). Also, the error variant in `DebounceEventResult` is a single `notify::Error`, not a `Vec`, despite some docs suggesting otherwise.
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
- **Integration test location:** Tests live in `tests/integration_tests.rs` (single file with nested `mod` blocks), not a `tests/integration/` directory. 47 tests covering read, search, graph, write, periodic note, tantivy BM25, and tool filtering operations.
- **Daemon integration test feature gate:** `tests/daemon_integration_tests.rs` is behind `#[cfg(all(unix, feature = "embeddings"))]` (NOT `has_embeddings`) because these tests need the local fastembed backend for actual inference. Running plain `cargo test` compiles this target but executes 0 tests; run `cargo test --features embeddings --test daemon_integration_tests` in CI to actually exercise daemon integration coverage.
- **`VaultError::InvalidRegex` variant:** The `search_regex` tool validates the regex pattern before searching. Invalid patterns produce `InvalidRegex { pattern, source }` which maps to `ErrorCode::INVALID_PARAMS` at the MCP boundary.
- **`.gitattributes` LF line endings:** The repo includes `* text=auto eol=lf` to enforce LF line endings on all platforms. This is required for Windows CI — without it, `cargo fmt --check` fails due to CRLF mismatches.
- **CI/CD workflows:** Two GitHub Actions workflows: `ci.yml` runs `cargo fmt --check` (Ubuntu only) + `cargo clippy` + `cargo test` on a 3-OS matrix (ubuntu-latest, macos-latest, windows-latest). `release.yml` triggers on `v*` tags, builds 5 targets (x86_64-unknown-linux-gnu, aarch64-unknown-linux-gnu, x86_64-apple-darwin, aarch64-apple-darwin, x86_64-pc-windows-msvc), creates a GitHub Release with checksums, and publishes to crates.io.
- **macOS CI runners:** GitHub deprecated `macos-13`. Both macOS targets (x86_64 and aarch64) now use `macos-latest` in the release workflow. The x86_64 build runs natively (Rosetta) since `macos-latest` is ARM.
- **v0.1.0 release:** The initial version is published on both [crates.io](https://crates.io/crates/obsidian-mcp) and [GitHub Releases](https://github.com/lstpsche/obsidian-mcp/releases).
- **Tantivy BM25 index:** Layer 1 of semantic search. Uses `tantivy` 0.26 with an in-RAM index (`Index::create_in_ram()`). Schema: `path` (STRING|STORED), `title` (en_stem, stored, boost 5), `headings` (en_stem, boost 3), `tags` (FACET, stored, boost 4), `body` (en_stem, boost 1), `frontmatter_text` (en_stem, boost 2). Rebuilds in <100ms for 5000 notes. `TantivySchema` in `src/vault/tantivy_index.rs` holds the schema + cached `Field` handles.
- **TantivyIndex struct:** `TantivyIndex` in `src/vault/tantivy_index.rs` wraps the Tantivy in-RAM index with `build`/`reindex_file`/`remove_file`/`search` methods. Uses `ReloadPolicy::Manual` with explicit `reader.reload()` after each commit for deterministic read-after-write. Writer is behind `std::sync::Mutex` since `IndexWriter::commit` takes `&mut self`. Search uses `QueryParser` with field-level boosts from `TantivySchema` constants.
- **Tantivy Vault integration:** `VaultInner` holds `tantivy: Option<Arc<TantivyIndex>>`, gated by `Config::tantivy` (default `true`). Built during `Vault::open()` from the completed `VaultIndex::notes()`. All write ops (`reindex`/`delete_note`/`move_note`) and the filesystem watcher sync both indices. The watcher receives `Option<Arc<TantivyIndex>>` and updates Tantivy inline after each `VaultIndex` mutation.
- **Tantivy ReloadPolicy:** Tantivy renamed `OnCommit` to `OnCommitWithDelay` (polls asynchronously). Only two variants exist: `Manual` and `OnCommitWithDelay`. We use `Manual` + explicit `reload()` for correctness.
- **Tantivy stemming:** The `en_stem` tokenizer uses the Porter stemmer. "programs"/"programming" share the stem "program", but "programmer" stems differently ("programm"). Test stemming with words that share the same root.
- **Tantivy two-phase search:** `search_text` uses a two-phase strategy when Tantivy is enabled: (1) BM25 ranking via Tantivy returns top-K `(path, score)` pairs, (2) for each hit, the file is read and a case-insensitive regex of the original query words extracts context snippets. If a note was matched only via stemming (no literal occurrence of query words), the `matches` array is empty but `score` is populated — the agent uses the score to know the note is relevant and can `note_read` it. Falls back to the regex scan when Tantivy is disabled.
- **Tantivy fuzzy search:** `search_text_with_options` supports `fuzzy: bool` which enables edit-distance-1 fuzzy matching via `QueryParser::set_field_fuzzy`. This tolerates single-character typos (insertions, deletions, substitutions, transpositions).
- **Tantivy field filtering:** `search_text_with_options` supports `fields: Option<&[SearchField]>` to restrict search to specific note fields (title, headings, body, frontmatter, tags). Text fields use `QueryParser` with field-specific boosts. Tags use facet `TermQuery` since they're indexed as Tantivy facets, not text.
- **Tantivy facet encoding:** `Facet::from_text("bare_word")` returns `Err(FacetParseError)` — it requires a leading `/`. Use `Facet::from_path(components)` instead, which constructs the facet directly from path components. Tags like `"nested/tag"` are split on `/` and passed as components: `Facet::from_path(["nested", "tag"])`.
- **Tantivy `CompactDocValue`:** `TantivyDocument::get_first()` returns `Option<CompactDocValue>` (not `Option<OwnedValue>`). To extract a string, use the `Value` trait: `retrieved.get_first(field).and_then(|v| v.as_str())`. Must import `tantivy::schema::Value` for this to work.
- **Tantivy 0.26 `TopDocs` API change:** `TopDocs` no longer directly implements `Collector`. Must call `.order_by_score()` to get a collector: `TopDocs::with_limit(k).order_by_score()`. Without this, `searcher.search()` won't compile.
- **bincode 3.0 is dead:** The bincode 3.0.0 release is a poison pill (maintainer abandoned the project). Stay on bincode 2. The `bincode-next` fork exists but is not yet battle-tested.
- **Tantivy tags in default search:** The base `TantivyIndex::search()` (used by `search_text`) does not create facet `TermQuery` for tags — only `search_with_options` does when `fields` explicitly includes `SearchField::Tags`. However, tags are still searchable in the default path because `stringify_frontmatter` flattens the entire frontmatter object (including `tags: [...]`) into the `frontmatter_text` field, which is indexed with `en_stem`. This is by design: facet queries provide exact tag matching, while the text path gives stemmed/fuzzy matching for free.
- **Embedding search (Layer 2):** Opt-in via `OBSIDIAN_EMBEDDINGS=true` + either `--features embeddings` (local fastembed) or `--features embeddings-api` (OpenAI-compatible API). Both features can be enabled simultaneously. `EmbeddingStore` (cosine similarity, bincode persistence) and `EmbeddingModel` (backend-agnostic wrapper) are held in `VaultInner`, initialized in `Vault::open()`, and synced on all write operations. Exposed as `search_semantic` MCP tool.
- **fastembed `&mut self`:** `fastembed::TextEmbedding::embed()` takes `&mut self`, so `EmbeddingModel` wraps it in `std::sync::Mutex`. The Mutex is locked only during inference calls.
- **bincode 2 serde compat:** Use `bincode::serde::encode_to_vec` / `decode_from_slice` with `bincode::config::standard()` for embedding cache persistence. The bincode dep requires `features = ["serde"]`.
- **fastembed model resolution:** `fastembed::EmbeddingModel` implements `FromStr`. The config string (e.g. `"BAAI/bge-small-en-v1.5"`) is parsed via `.parse()`, falling back to `Default` (BGESmallENV15, dim=384) on failure. `get_model_info` requires importing `fastembed::ModelTrait`.
- **Embedding cache format:** Binary file at `{mcp_data}/embeddings/embeddings.bin` (legacy location: `{vault}/.obsidian/obsidian-mcp/embeddings.bin`). Serialized via bincode serde compat with an `EmbeddingCacheData { dim, entries: Vec<(String, Vec<f32>)> }` struct. Not forward-compatible — cache version changes require rebuild.
- **Embedding Vault integration:** `VaultInner` holds `embedding_model: Option<Arc<EmbeddingModel>>` and `embedding_store: Option<Arc<RwLock<EmbeddingStore>>>`, both `#[cfg(has_embeddings)]`. `RwLock` for the store (not `Mutex`) because reads (search queries) vastly outnumber writes (note edits). The model is `Arc` only (thread-safe via its internal Mutex). Cache is persisted to disk after every write that modifies the store.
- **Local embedding write visibility:** `Vault::reindex_embedding()` offloads embedding generation and cache persistence with `spawn_blocking` when called inside a tokio runtime. Tests that write a note and immediately query semantic search should poll briefly for the expected hit instead of assuming synchronous read-after-write visibility.
- **Embedding staleness check:** On startup, if the cached store's note count doesn't match `VaultIndex::notes().len()`, the cache is rebuilt. Per-note content hashing is overkill for MVP — incremental updates handle individual changes after startup.
- **Non-fatal embedding errors in write path:** If embedding fails during `reindex`, the error is logged but doesn't propagate — the note is still written and indexed by VaultIndex and Tantivy. Degraded semantic search is better than a failed write.
- **Graceful embedding load failure:** If `EmbeddingModel::load()` fails at startup (API unreachable, invalid key, model not found), `Vault::open()` logs a warning and continues with `embedding_model: None`. The error string is stored in `VaultInner::embedding_load_error: Option<String>` (NOT feature-gated to avoid struct literal drift). When `search_semantic` is called, `search_semantic_local()` checks `vault.embedding_load_error()` and returns the specific failure reason via `VaultError::Embedding`. The server remains fully functional for all non-semantic tools.
- **Watcher feature-gating:** The `start_watcher` and `process_event` functions have two `#[cfg]`-gated versions each (with/without embeddings). This is necessary because `#[cfg]` on function parameters isn't supported in Rust. The watcher tests use a `call_start_watcher` helper that dispatches to the correct variant.
- **rmcp `#[tool_router]` and `#[cfg]`:** The `#[tool_router]` proc macro expands before `cfg` is resolved, so `#[cfg(feature = "...")]` on individual `#[tool]` methods doesn't work — the macro generates references to the method unconditionally. The workaround is to always define the tool method and params, but provide two `#[cfg]`-gated implementations: one that delegates to the real logic, and one that returns an error explaining the feature isn't compiled in.
- **fastembed concurrent model loading:** `fastembed::TextEmbedding::try_new()` has race conditions when multiple instances try to load the same model concurrently (corrupted cache access). Integration tests use a static `tokio::sync::Mutex` to serialize `Vault::open()` calls that trigger model loading.
- **Hybrid re-ranking (E7):** `search_semantic` with `lexical_prefetch: true` runs a two-stage pipeline: Tantivy BM25 retrieves top-50 candidates, then each is re-scored via `alpha * norm_bm25 + (1-alpha) * cosine_sim`. Default alpha is 0.25 (configurable via `OBSIDIAN_HYBRID_ALPHA` env var or per-query `alpha` param). BM25 scores are min-max normalized to [0,1] within the result set; when all scores are equal they normalize to 1.0. Notes missing from the embedding store (e.g., embedding failed) get semantic score 0.0. Requires both Tantivy (`OBSIDIAN_TANTIVY=true`) and embeddings (`OBSIDIAN_EMBEDDINGS=true` + `--features embeddings`).
- **Hybrid alpha resolution:** Per-query `alpha` param > `OBSIDIAN_HYBRID_ALPHA` env var > hardcoded default (0.25). Lower values favor semantic similarity; higher values favor keyword frequency. The default was lowered from 0.4 to 0.25 after testing showed the higher value degraded conceptual queries by over-weighting BM25 lexical matches.
- **Semantic search snippets:** `SemanticSearchResult` includes a `snippet` field (populated when `include_content` is false). Snippets are extracted via query-word regex match + `extract_match_context` (150-char window each side, ~300+ chars total). The keyword regex runs on body-only text (frontmatter excluded via `get_body()`) to prevent YAML from polluting snippets. If no regex match (pure semantic hit), a fallback of the first ~300 chars of body is used (frontmatter stripped, title heading skipped since it's redundant with the `title` field).
- **Semantic search context budget:** Cursor agent tool captures show that returning full `content` for every semantic hit can produce 40KB+ single-call outputs and accelerate context rot. Prefer path/title/score/tags plus concise, frontmatter-free snippets by default; reserve full content for explicit follow-up reads or narrowly scoped `include_content` use.
- **`build.rs` and `has_embeddings` cfg alias:** A `build.rs` at the crate root emits `cargo:rustc-cfg=has_embeddings` when either `embeddings` or `embeddings-api` Cargo feature is active. It also emits `cargo:rustc-check-cfg=cfg(has_embeddings)` unconditionally so clippy doesn't warn on the custom cfg. All embedding infrastructure code uses `#[cfg(has_embeddings)]` instead of `#[cfg(any(feature = "embeddings", feature = "embeddings-api"))]`. Only backend-specific internals inside `src/vault/embeddings.rs` use individual feature names (`#[cfg(feature = "embeddings")]` for fastembed, `#[cfg(feature = "embeddings-api")]` for the API client). Do NOT modify the `build.rs` env var checks (`CARGO_FEATURE_EMBEDDINGS`, `CARGO_FEATURE_EMBEDDINGS_API`) — those are Cargo conventions, not cfg names.
- **`EmbeddingProvider` config:** `EmbeddingProvider` enum (`Local` | `Api`) in `src/config.rs`, stored as `Config::embedding_provider: Option<EmbeddingProvider>`. Parsed from `OBSIDIAN_EMBEDDING_PROVIDER` env var (case-insensitive). `None` means infer from compiled features at `EmbeddingModel::load()` time. Unknown values log a warning and fall back to `None`. The enum follows the `SemanticMode` pattern: `parse()` + `as_str()` methods, no `Default` impl.
- **Daemon ignores `OBSIDIAN_EMBEDDING_PROVIDER`:** The semantic daemon's `VaultRegistry::embedding_model()` always passes `None` to `EmbeddingModel::load()`, inferring the backend from compiled features (local preferred). This is intentional — the daemon is a separate binary with its own lifecycle. If a user sets `OBSIDIAN_EMBEDDING_PROVIDER=api` but the daemon binary was compiled with both features, the daemon uses the local backend while the MCP server uses the API backend. This asymmetry is by design; threading provider config through the daemon is a future improvement.
- **`EmbeddingBackend` dual-dispatch enum:** `EmbeddingModel` holds an `EmbeddingBackend` enum with `Local(Box<Mutex<TextEmbedding>>)` and `Api { client, base_url, model, api_key }` variants, each `#[cfg]`-gated by individual feature. The `Local` variant is `Box`ed to satisfy `clippy::large_enum_variant`. `embed_batch()` dispatches via match. `load()` uses `resolve_provider()` which infers from `cfg!()` when no explicit provider is given (local preferred when both features are compiled).
- **API embedding env var chain:** `load_api()` reads env vars with primary/fallback pairs: `OBSIDIAN_EMBEDDING_API_KEY`/`OPENAI_API_KEY` (required), `OBSIDIAN_EMBEDDING_API_BASE`/`OPENAI_BASE_URL` (default `https://api.openai.com/v1`), `OBSIDIAN_EMBEDDING_API_MODEL`/`OPENAI_MODEL` (default: config model name). `OBSIDIAN_EMBEDDING_DIM` skips the dimension probe when set. `OBSIDIAN_EMBEDDING_CA_CERT` adds a custom root cert; `OBSIDIAN_EMBEDDING_TLS_VERIFY=false` disables TLS verification (with warning).
- **`reqwest::blocking` in `spawn_blocking`:** The API backend uses `reqwest::blocking::Client` inside `tokio::task::spawn_blocking` to avoid blocking the async runtime. This is correct because `reqwest::blocking` internally spawns its own tokio runtime (which conflicts with being called from an async context directly). The `embeddings-api` feature activates `reqwest/blocking` and `reqwest/json`.
- **API dimension probing:** When `OBSIDIAN_EMBEDDING_DIM` is not set, `load_api()` sends a single-text embedding request (`"dim"`) to detect the model's output dimensionality at startup. This avoids hardcoding dimension per model. The probe reuses `embed_batch_api()`.
- **API 429 retry:** `embed_batch_api()` retries up to 3 times on HTTP 429 (rate limited). Backoff: `Retry-After` header if present (capped at 30s), otherwise exponential doubling (1s → 2s → 4s). After exhausting retries, the error propagates as `VaultError::Embedding`.
- **API key zeroize:** The `api_key` field in `EmbeddingBackend::Api` uses `zeroize::Zeroizing<String>` to clear memory on drop. `Zeroizing<T>` implements `Deref<Target=T>`, so existing `format!("Bearer {api_key}")` and `&str` deref chains work unchanged. `zeroize` is a direct dep (already a transitive dep via rustls).
- **`parse_embedding_response` shared function:** A single `#[cfg(feature = "embeddings-api")]` function in `embeddings.rs` parses OpenAI-compatible embedding API JSON into `Vec<Vec<f32>>`. Items are sorted by the `index` field when present, falling back to array position for providers that omit it. Used by both `embed_batch_api` (production) and unit tests (directly). Avoids duplicating parsing logic.
- **API `encoding_format`:** `embed_batch_api` explicitly sets `"encoding_format": "float"` in the request body. OpenAI defaults to float, but some OpenAI-compatible providers may not — specifying it prevents silent parse failures from non-float response formats.
- **Non-blocking `reindex_embedding`:** `Vault::reindex_embedding` offloads the blocking embed→store→cache-save sequence to `tokio::task::spawn_blocking` via `Handle::try_current()`. The JoinHandle is dropped (fire-and-forget) since embedding is non-fatal. Falls back to synchronous execution when no tokio runtime is available (e.g. unit tests). This keeps tokio worker threads free for concurrent vault reads during writes.
- **`reindex_embedding` stale-task guard:** `Vault` now tracks per-path embedding task generations in `VaultInner::embedding_task_generation` (`Mutex<HashMap<PathBuf, u64>>`). Async embedding closures only commit when their generation still matches and the note still exists in `VaultIndex`; delete/move paths invalidate old generations before removing embeddings to prevent ghost semantic hits.
- **Rust 2024 env-test serialization:** API helper tests in `src/vault/embeddings.rs` serialize `unsafe` `std::env::set_var/remove_var` usage with a module-level `LazyLock<Mutex<()>>`. Keep process-env mutation tests behind a shared lock to avoid cross-test races on Unix.

## Task Tracking

The full development plan with task breakdown, dependencies, and parallelization info is in `TODO.md`.

- **`CallToolResult::structured()` rejects null:** The Cursor MCP client validates `structuredContent` as a record (object). Passing `serde_json::Value::Null` via `CallToolResult::structured()` triggers a validation error. When a tool may return null, use `CallToolResult::success(vec![Content::text("null")])` for the null case and `CallToolResult::structured(value)` for real objects.
- **`create_periodic_note` content override:** The method now accepts `content_override: Option<&str>` as the third parameter. When provided, it skips template expansion entirely. When `None`, missing template files are handled gracefully (`unwrap_or_default`) instead of erroring.
- **Semantic daemon IPC framing:** `obsidian-semanticd` uses local JSON-RPC 2.0 over UDS (Unix) / named pipes (Windows) with newline-delimited JSON messages (one request/response per line). Keep this framing consistent in all clients/bootstrappers.
- **Shared semantic bootstrap lock + defaults:** Bootstrap serializes install/manifest writes with `<semantic-home>/lock/install.lock` (fs2 exclusive lock). Home defaults are OS-specific (`~/Library/Application Support/obsidian-semantic`, `${XDG_DATA_HOME:-~/.local/share}/obsidian-semantic`, `%APPDATA%/obsidian-semantic`). Default daemon auto-download is version-pinned: `https://github.com/lstpsche/obsidian-mcp/releases/download/v{obsidian-mcp-version}/obsidian-semanticd-{obsidian-mcp-version}-{target}.{tar.gz|zip}` unless `OBSIDIAN_SEMANTIC_DAEMON_DOWNLOAD_URL` is set.
- **Daemon binary $PATH fallback:** `ensure_daemon` checks `$PATH` for `obsidian-semanticd` before downloading from GitHub releases. This ensures that `cargo install obsidian-mcp --features embeddings` (which installs both binaries to `~/.cargo/bin/`) is picked up automatically instead of downloading a release binary that lacks the `embeddings` feature. Resolution order: override env var → manifest path → semantic home bin dir → `$PATH` → download.
- **Windows fs2 lock contention codes:** `FileExt::try_lock_exclusive()` can return raw OS errors `32`/`33` (sharing/lock violation) instead of `ErrorKind::WouldBlock` when another process holds the lock. Treat these as lock contention in `InstallLock::try_acquire()` and return `Ok(None)`.
- **Daemon vault context lifecycle:** `search_semantic`, `search_hybrid`, and `open_hint` now require a prior `ensure_vault` call for that `vault_root`; otherwise the daemon returns `ERR_VAULT_NOT_READY` (`-32030`). The daemon keeps per-vault runtime state in a registry keyed by `vault_id` (SHA-256 of canonical path).
- **Daemon per-vault embedding cache path:** daemon-managed embedding caches live under `<semantic-home>/vaults/<vault_id>/embeddings.bin` (not inside `.obsidian/obsidian-mcp`). Watcher-driven updates persist to this path and keep non-fatal behavior on embedding failures (warn + continue).
- **Daemon query parity rules:** `search_hybrid` keeps explicit empty-query short-circuit (`[]`), while pure `search_semantic` does not short-circuit empty queries. Snippet generation keeps the MCP parity behavior: query-word context first, then frontmatter-stripped body preview fallback.
- **MCP semantic runtime modes:** `OBSIDIAN_SEMANTIC_MODE` controls MCP behavior: `daemon` is strict daemon-only, `local` forces in-process embeddings, and `auto` prefers daemon with local fallback only for availability failures (`DaemonUnavailable`/IPC/timeout/bootstrap and selected daemon RPC availability codes). This keeps non-semantic tools available even when daemon bootstrap fails.
- **MCP daemon startup gate:** MCP only initializes/starts the semantic daemon when `OBSIDIAN_WATCH=true`. With watch disabled, daemon initialization is skipped and semantic behavior falls back to local mode (if available) or returns daemon-unavailable errors in daemon mode.
- **MCP daemon connect policy knobs:** `OBSIDIAN_SEMANTIC_CONNECT_TIMEOUT_MS` (clamped `100..=60000`), `OBSIDIAN_SEMANTIC_CONNECT_RETRIES` (`0..=10`), and `OBSIDIAN_SEMANTIC_RETRY_BACKOFF_MS` (`50..=60000`) drive MCP daemon call retry behavior; retries only apply to transport/timeouts, not protocol errors.
- **Legacy embedding cache migration path:** On daemon initialization with embeddings feature, MCP best-effort copies legacy per-vault cache from `<vault>/.obsidian/obsidian-mcp/embeddings.bin` into `<semantic-home>/vaults/<vault_id>/embeddings.bin`. The copy is idempotent and never deletes the source cache.
- **Daemon stderr logging:** Daemon bootstrap writes daemon stderr to `<semantic-home>/logs/obsidian-semanticd.stderr.log` (append mode) instead of discarding it, while stdout remains null to protect JSON-RPC channels.
- **Shared search utilities:** `src/vault/search_utils.rs` contains `body_preview`, `compile_query_word_regex`, and `normalize_bm25_scores`. These are **not** feature-gated — callers maintain their own `#[cfg(has_embeddings)]` on imports. Do not duplicate these functions in tool or daemon modules.
- **Shared test helpers:** `src/test_helpers.rs` is a `#[cfg(test)] pub(crate) mod` at crate root providing `test_config`, `tantivy_config`, `create_test_vault`, and `extract_text`. Tool test modules import from here and extend with domain-specific setup (e.g. writing fixture notes). Do not duplicate base helpers.
- **Single model name constant:** `config::DEFAULT_MODEL_NAME` is the single source of truth for the default embedding model name (`BAAI/bge-small-en-v1.5`). Both the MCP server and the semantic daemon binary import from `obsidian_mcp::config`. Do not define local model name constants.
- **SemanticRuntime `vault_ensured` flag:** `SemanticRuntime` has a `vault_ensured: Arc<AtomicBool>` field that caches whether `ensure_vault` has been called for the current daemon session. The flag is reset to `false` when the daemon becomes unavailable (fallback path). Always initialize this field when constructing `SemanticRuntime` in new code or tests.
- **VaultRegistry per-vault init lock:** `ensure_vault` uses a per-vault_id `tokio::sync::Mutex<()>` to prevent duplicate `VaultContext` construction under concurrent calls. The pattern is: acquire per-key lock → double-check registry → build if still missing → insert → release.
- **InstallLock timeout:** `InstallLock::acquire` retries `try_lock_exclusive` in a loop with 200ms sleep and 30s timeout instead of blocking indefinitely. Warns after 5s of contention. `acquire_async` wraps this in `spawn_blocking` for use in async contexts (e.g., daemon bootstrap) to avoid blocking the tokio worker thread.
- **Daemon early-exit detection:** After spawning the daemon process, bootstrap waits 50ms and calls `try_wait()` to detect immediate crashes before entering the health probe loop.
- **Incremental backlinks:** `VaultIndex::reindex_file` uses incremental backlink updates (O(links_in_file)) for content-only changes (file already in index). Falls back to full `rebuild_backlinks()` (O(total_links)) only when the path set changes (new file, remove, rename) because `LinkResolver` ambiguity may shift.
- **TantivyIndex batch API:** `reindex_file_batch` / `remove_file_batch` skip commit+reload; call `flush()` once after a batch. Both watchers (MCP and daemon) use these to amortize a single commit per debounced event batch. Direct (non-watcher) mutations still use the committing `reindex_file` / `remove_file` methods.
- **Watcher lock discipline:** Both watchers (MCP `src/vault/watcher.rs` and daemon `src/daemon/watcher.rs`) release the `VaultIndex` write lock before calling Tantivy batch ops or embedding inference. This minimizes read contention on the index lock. The daemon watcher's `indexer::embed_note` and `remove_note_embedding` return `bool` (mutation happened) without persisting the cache; the event loop saves once per batch.
- **Watcher extension matching:** Both watchers use case-insensitive `.md` extension checks (`eq_ignore_ascii_case`) consistent with `VaultIndex::build_sync`. The fallback path for deleted files (no extension from OS) also uses case-insensitive matching.
- **list_files skips canonicalize:** Since `WalkDir` defaults to `follow_links(false)` and the walk root is already canonical, per-entry `canonicalize()` is unnecessary. Entries are stripped against the canonical root directly.
- **EmbeddingStore partial sort:** `query()` uses `select_nth_unstable_by` + sort of the top-k partition instead of sorting the entire store. Significant for vaults with >1k notes.
- **Regex size limits for untrusted input:** `VaultIndex::search_regex` uses `RegexBuilder::new(pattern).size_limit(1 << 20).build()` to cap NFA compilation resources. Patterns exceeding 1000 characters are rejected outright. The Rust `regex` crate is immune to catastrophic backtracking (finite automata) but NOT immune to compilation-time resource exhaustion for pathological patterns. Always use `RegexBuilder` with `size_limit` for user-supplied patterns.
- **Archive extraction entry validation:** `extract_tar_gz_binary` verifies `entry.header().entry_type().is_file()` before unpacking; `extract_zip_binary` checks `!file.is_dir()` and rejects entries containing `..` in the name. The `tar` crate is pinned to `>=0.4.45` (CVE-2026-33056 fix). These are defense-in-depth measures — always validate archive entry types before extraction.
- **Daemon socket permissions:** On Unix, the IPC socket permissions are set to `0600` after `UnixListener::bind()` to prevent other users from connecting. The `set_permissions` call is non-fatal (warns on failure) to avoid breaking setups with unusual filesystem semantics.
- **Windows URI launch:** `launch_uri` uses `explorer.exe` on Windows (not `cmd /C start`) to avoid cmd.exe metacharacter parsing issues. On macOS/Linux, `open`/`xdg-open` are used with a single `.arg(uri)` which bypasses shell interpretation.
- **Search parameter hard caps:** `MAX_RESULTS_CAP` (200) and `MAX_CONTEXT_LEN_CAP` (2000) in `src/tools/search.rs` clamp user-supplied `max_results` and `context_length` to prevent resource exhaustion via unbounded search parameters.
- **Lock poisoning recovery:** All `RwLock`/`Mutex` acquisitions in `src/vault/mod.rs` use `.unwrap_or_else(|e| e.into_inner())` instead of `.expect()`. This recovers from poisoned locks (caused by a panicking thread) instead of cascading the panic into a server crash. The data may be in an inconsistent state, but continued operation with degraded consistency is preferable to total unavailability.
- **CI cargo-audit:** The CI pipeline runs `cargo audit` on Ubuntu to detect known vulnerabilities in dependencies automatically. Transitive-dep advisories that cannot be fixed without major upgrades are suppressed in `.cargo/audit.toml` with comments explaining the risk assessment. Update this file when dependencies are upgraded.
- **CLI flags and subcommands:** Both binaries support `--version`/`-v` and `--help`/`-h`/`help`. The MCP binary also supports `serve`, `stop`, and `restart` subcommands. All are handled by `handle_cli_flags()` at the top of `main()`, before `Config::load()` or tracing init, printing to stdout and calling `std::process::exit(0)`. No CLI framework needed — manual `match` on `std::env::args().nth(1)`.
- **reqwest 0.13 TLS feature rename:** The `rustls-tls` feature was renamed to `rustls` in reqwest 0.13. If reverting reqwest, update the Cargo.toml feature name back.
- **Dual transport (stdio + Streamable HTTP):** The binary supports both stdio (default) and Streamable HTTP transports. Select via `--http` CLI flag or `OBSIDIAN_TRANSPORT=http` env var. HTTP mode uses rmcp's `StreamableHttpService` (tower-compatible) served by axum. `StreamableHttpServerConfig` is `#[non_exhaustive]` — use `Default::default()` + field mutation, not struct literals.
- **HTTP transport config:** `stateful_mode: true` (sessions per MCP Initialize handshake) and `json_response: true` (plain JSON for request-response, no SSE overhead). Each session gets its own `ObsidianMcp` instance via the factory closure, but all share the same `Vault` (Arc-based, thread-safe).
- **`serve` subcommand daemonization:** `obsidian-mcp serve` stops any existing server on the target port before spawning a new detached child with `--http`. Port is resolved via `resolve_port_from_args()`: `--port` CLI arg > `OBSIDIAN_HTTP_PORT` env var > default 37842. The stop logic probes the port via `TcpStream::connect_timeout`, finds the PID via `lsof -ti` (Unix) / `netstat -ano` (Windows), sends SIGTERM / `taskkill`, and polls for up to 2s. Stop actions are logged to stdout. The child's stderr is redirected to a platform-specific log file (macOS: `~/Library/Logs/obsidian-mcp.log`, Linux: `$XDG_STATE_HOME/obsidian-mcp/obsidian-mcp.log`, Windows: `%LOCALAPPDATA%/obsidian-mcp/obsidian-mcp.log`). On Unix, `process_group(0)` detaches the child from the parent's terminal. After spawn, the parent polls `/health` via raw HTTP/1.0 GET (up to 60 ticks × 250ms = 15s), checking `try_wait()` each tick to detect early crashes. Reports version in the success message. No PID file — use OS process management (launchd, systemd) for production daemon setups.
- **`stop` subcommand:** `obsidian-mcp stop [--port PORT]` kills the server on the target port and exits. Reports "no server running" if the port is free. Reuses `stop_existing_server()`.
- **`restart` subcommand:** `obsidian-mcp restart [OPTIONS] [VAULT_PATH]` is equivalent to `stop` + `serve` — stops any existing server, then daemonizes a new one. Shares the `daemonize()` implementation with `serve`.
- **`/health` endpoint:** HTTP mode exposes `GET /health` returning `{"status":"ok","server":"obsidian-mcp","version":"...","notes":<count>}`. When compiled with embeddings, also includes `"embeddings_ready":<bool>` and optionally `"embeddings_error":"<msg>"`. The handler receives a `Vault` clone to query live state. MCP tools are served under `/mcp`.
- **`probe_health` function:** The `daemonize()` startup check uses a raw HTTP/1.0 GET to `/health` (not just TCP connect) and validates the response contains `200` and `"status":"ok"`. This distinguishes "port open but server not ready" from "server healthy".
- **`serve` vs `--http` for process managers:** `serve` daemonizes (parent exits immediately), so launchd/systemd can't track the child process for restarts. Process managers should use `--http` directly with `KeepAlive`/`Restart=on-failure`. `serve` is intended for ad-hoc/manual use only.
- **Graceful HTTP shutdown:** The HTTP server handles both SIGTERM (Unix) and Ctrl+C for clean shutdown. On non-Unix platforms, only Ctrl+C is supported.
- **rmcp 1.6 upgrade:** Bumped from 1.2 to 1.6. No breaking changes to our code. Key additions: Origin/Host validation for HTTP, runtime tool disabling, session store for resumability, init_timeout for streamable-http sessions.
- **Tool consolidation (29→18):** 7 groups of related tools merged into parameterized tools using action/query/type/view string parameters. Consolidation map: `vault_structure`→`vault_list(format:"tree")`, `frontmatter_*`→`frontmatter(action)`, `note_metadata`+`note_document_map`→`note_inspect(view)`, `links_*`→`wikilinks(query)`, `periodic_*`→`periodic(action)`, `note_append`+`note_prepend`→`note_insert(position)`, `search_tag`+`search_frontmatter`→`search_metadata(type)`. Action/type params are `Option<String>` (not enums) for LLM-friendly schema. Validation returns `INVALID_PARAMS` for unknown values or missing required fields.
- **Tool filtering (`OBSIDIAN_TOOLS`):** `ToolFilter` enum in `config.rs` with `Full`/`Profile(String)`/`AllowList(HashSet)`/`DenyList(HashSet)` variants. Detection: contains `!`→deny-list; single word matching profile→profile; single word matching tool name→1-tool allow-list; single unknown word→error; comma-separated→allow-list. `disabled_tools()` returns the complement set for `disable_route()`. Unknown tool names in lists are warned, not errored. Profiles: `full`(18), `core`(14), `read`(10), `minimal`(6). `ALL_TOOL_NAMES` constant is the single source of truth for all 18 post-consolidation names.
- **Tool-layer param validation error codes:** `VaultError::Other(...)` maps to `ErrorCode::INTERNAL_ERROR`, not `INVALID_PARAMS`. For parameter validation in tool handlers (invalid action/query/view/type, missing required param), construct `ErrorData::new(ErrorCode::INVALID_PARAMS, msg, None::<serde_json::Value>)` directly. Only use `VaultError` + `?` for errors originating from vault-layer method calls.
- **`ObsidianMcp::new()` takes `disabled_tools`:** The constructor accepts `HashSet<String>` as a 4th parameter. It creates a mutable `tool_router`, calls `disable_route()` for each name, then stores the router. All callers (stdio and HTTP paths in `main.rs`) must pass this set.
- **HTTP per-session tool filtering (`X-Obsidian-Tools` header):** An axum middleware extracts the header, parses it via `ToolFilter::parse()`, and scopes the resulting disabled set into a `tokio::task_local!`. The `StreamableHttpService` factory closure reads the task-local via `try_with()` and unions it with the server-wide disabled set. This works because the factory runs within the middleware's `next.run(request).await` scope. Malformed headers log a warning and fall back to server default (empty per-session set). The middleware is applied only to the MCP router, not `/health`.
- **`.obsidian-mcp/` metadata folder:** Auto-created at vault root on startup with an empty `ignore` file. Hard-skipped by VaultIndex and both watchers (same treatment as `.obsidian/`). Contains: `ignore` (exclusion patterns), `embeddings/embeddings.bin` (embedding cache). When `OBSIDIAN_MCP_DATA` is set, resolved `mcp_data` is `{OBSIDIAN_MCP_DATA}/vaults/{slug}`; embeddings are stored at `{mcp_data}/embeddings/embeddings.bin`, and the vault-root `ignore` is merged with `{mcp_data}/ignore`.
- **`vault_slug()` namespacing:** Canonical vault path → strip leading `/` or drive prefix → replace separators with `-`. Truncate at 200 chars with 8-char SHA-256 hash suffix. Used for per-vault namespacing under `OBSIDIAN_MCP_DATA`. Implemented in `src/config.rs` with `char_indices().nth()` for UTF-8-safe truncation.
- **Windows vault slugs:** `std::fs::canonicalize` can return extended Windows paths like `\\?\C:\...`. `vault_slug()` must strip the `\\?\` prefix and replace filename-invalid characters (`<>:"|?*`, separators, control chars) with `-` before using the slug as a directory name.
- **Embedding cache relocation:** Cache moved from legacy `{vault}/.obsidian/obsidian-mcp/embeddings.bin` to `{mcp_data}/embeddings/embeddings.bin`. Single point of control: `Vault::embedding_cache_path(mcp_data)`. On startup, if legacy exists and new doesn't, a best-effort copy migrates the cache. Legacy file is never deleted. The MCP watcher's `save_embedding_cache()` receives `mcp_data` (not `vault_root`) to write to the new location. The daemon watcher is unaffected — it uses its own `embedding_cache_path` from `VaultContext`.
- **`ExcludeSet` and `load_ignore_patterns`:** `ExcludeSet` in `src/vault/exclude.rs` compiles glob patterns via `GlobSetBuilder`, normalizes trailing `/` to `/**`, warns+skips invalid patterns. `is_excluded(&Path)` delegates to `GlobSet::is_match`. `load_ignore_patterns(mcp_home, mcp_data)` performs two-level merge from ignore files at both locations, sorted + deduped. Combined with `Config::exclude_patterns` from env var before building the `ExcludeSet`. Patterns are read once at startup — changes require restart.
- **Ignore file two-level merge:** Patterns loaded from `{mcp_home}/ignore` (vault-portable) and `{mcp_data}/ignore` (machine-specific, only if different location). Combined with `OBSIDIAN_EXCLUDE_PATHS` env var. Exclusion scope: VaultIndex, Tantivy, embeddings, watcher, `vault_info` stats. NOT excluded from: direct path access (`note_read`, `note_write`, etc.) and `vault_list`. `vault_info` reports `excluded_notes` count and `exclude_patterns` list for diagnostics.
- **Direct writes to excluded paths:** Direct access tools must remain allowed, but the shared reindex path must not add excluded notes to VaultIndex, Tantivy, or embeddings. `Vault::reindex()` is the central guard: excluded paths are tracked for `excluded_notes` stats and removed from search indexes.
- **Daemon semantic exclusion boundary:** The semantic daemon does not read `.obsidian-mcp/ignore`; MCP-side `search_semantic` must filter daemon and stale local embedding hits through `Vault::get_note_metadata()` before returning them, otherwise excluded notes can leak through semantic search.
- **`OBSIDIAN_MCP_DATA` self-reference:** If `OBSIDIAN_MCP_DATA` resolves to the vault-root `.obsidian-mcp/`, treat it as unset. Do not namespace it under `.obsidian-mcp/vaults/{slug}`.

## Links

- [GitHub repository](https://github.com/lstpsche/obsidian-mcp)
- [crates.io](https://crates.io/crates/obsidian-mcp)
- [rmcp docs](https://docs.rs/rmcp/latest/rmcp/)
- [Obsidian help — Properties](https://help.obsidian.md/Editing+and+formatting/Properties)
- [Obsidian help — Internal links](https://help.obsidian.md/Linking+notes+and+files/Internal+links)
- [Obsidian help — Tags](https://help.obsidian.md/Editing+and+formatting/Tags)
- [Obsidian local REST API — OpenAPI spec](https://coddingtonbear.github.io/obsidian-local-rest-api/openapi.yaml) (reference for what the REST API covers)
- [MCP specification](https://modelcontextprotocol.io/)
