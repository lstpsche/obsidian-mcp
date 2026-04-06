# obsidian-mcp

[![CI](https://github.com/lstpsche/obsidian-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/lstpsche/obsidian-mcp/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/obsidian-mcp.svg)](https://crates.io/crates/obsidian-mcp)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A high-performance [MCP](https://modelcontextprotocol.io) server that gives AI agents full read/write access to [Obsidian](https://obsidian.md) vaults. Written in Rust. Ships as a single binary.

**No Obsidian plugins. No REST API. No Obsidian running. Just your vault on disk.**

## Why?

Existing solutions for AI + Obsidian require the Obsidian app to be running with a community plugin ([obsidian-local-rest-api](https://github.com/coddingtonbear/obsidian-local-rest-api)) and an MCP wrapper ([mcp-obsidian](https://github.com/MarkusPfundstein/mcp-obsidian)) on top. That's fragile, slow, and limited to what the REST API exposes.

obsidian-mcp talks directly to the filesystem. It understands the Obsidian format natively — wikilinks, frontmatter, tags, block references, periodic notes, the full graph. It builds fast in-memory indices on startup and keeps them synced via a filesystem watcher. The result is a single dependency-free binary that works whether Obsidian is open or not.

## Quick Start

```sh
# Install
cargo install obsidian-mcp

# Run
obsidian-mcp /path/to/your/vault
```

Add to your MCP client config (Cursor, Claude Desktop, etc.) and you're done — 29 tools are available immediately.

## What It Can Do

| Category | Capabilities |
|----------|-------------|
| **Navigate** | List files, tree view, vault stats |
| **Read & write** | Create, read, overwrite, append, prepend, move, delete notes |
| **Patch** | Edit individual heading sections, block references, or frontmatter fields without touching the rest of the note |
| **Search** | BM25 full-text (Tantivy) with stemming, fuzzy matching, and per-field filtering. Semantic search via local embeddings. Regex. Tag and frontmatter queries. |
| **Graph** | Backlinks, outgoing links, broken link detection, orphan discovery |
| **Frontmatter** | Get/set/remove fields, query notes by metadata |
| **Periodic notes** | Daily, weekly, monthly, quarterly, yearly — with Obsidian-compatible date formats and template expansion |

All indices (metadata, BM25, embeddings) update in real time via a filesystem watcher.

## Installation

### From crates.io

```sh
cargo install obsidian-mcp
```

For **semantic search local compatibility mode** (`OBSIDIAN_SEMANTIC_MODE=local`), build with embeddings:

```sh
cargo install obsidian-mcp --features embeddings
```

The `embeddings` feature adds ~60 MB to the binary (ONNX Runtime). In daemon mode, model/runtime cache is shared under semantic home; local mode keeps in-process embedding support in the MCP binary.

### Pre-built binaries

Grab the latest from [GitHub Releases](https://github.com/lstpsche/obsidian-mcp/releases/latest):

| Platform | Archive |
|----------|---------|
| Linux x86_64 | `obsidian-mcp-<version>-x86_64-unknown-linux-gnu.tar.gz` |
| Linux ARM64 | `obsidian-mcp-<version>-aarch64-unknown-linux-gnu.tar.gz` |
| macOS Intel | `obsidian-mcp-<version>-x86_64-apple-darwin.tar.gz` |
| macOS Apple Silicon | `obsidian-mcp-<version>-aarch64-apple-darwin.tar.gz` |
| Windows x86_64 | `obsidian-mcp-<version>-x86_64-pc-windows-msvc.zip` |

Semantic daemon release assets are also published per target:

- `obsidian-semanticd-<version>-<target>.tar.gz` (Unix)
- `obsidian-semanticd-<version>-<target>.zip` (Windows)

## Client Setup

### Cursor

`~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "obsidian": {
      "command": "obsidian-mcp",
      "args": ["/path/to/your/vault"]
    }
  }
}
```

### Claude Desktop

```json
{
  "mcpServers": {
    "obsidian": {
      "command": "obsidian-mcp",
      "env": {
        "OBSIDIAN_VAULT_PATH": "/path/to/your/vault",
        "OBSIDIAN_EMBEDDINGS": "true"
      }
    }
  }
}
```

Config file location:
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

### Any MCP client

obsidian-mcp communicates over **stdio** using the standard MCP JSON-RPC protocol. Any client that supports MCP stdio transport will work. Pass the vault path as the first argument or via `OBSIDIAN_VAULT_PATH`.

## Search

Three tiers, each suited to different needs.

### BM25 Full-Text — `search_text`

On by default. Powered by [Tantivy](https://github.com/quickwit-oss/tantivy), rebuilt in-memory on startup (< 100 ms for 5k notes).

- **Stemming** — "deploy" matches "deployment", "deployed", "deploying"
- **Field boosts** — title 5x, tags 4x, headings 3x, frontmatter 2x, body 1x
- **Fuzzy matching** — `fuzzy: true` tolerates single-character typos (edit distance 1)
- **Field filtering** — `fields: ["title", "tags"]` to restrict scope
- Returns ranked results with context snippets and match offsets

### Semantic — `search_semantic`

Uses the shared local semantic daemon by default (`OBSIDIAN_SEMANTIC_MODE=auto`). Results keep the same MCP schema while backend execution can run through daemon or local compatibility mode.

Semantic runtime modes:

- `auto` (default): prefer daemon, fallback to local in-process embeddings when daemon is unavailable and local embeddings are enabled.
- `daemon`: daemon-only path; semantic calls fail clearly if daemon is unavailable.
- `local`: force legacy in-process embeddings path.

Daemon startup policy:

- MCP only initializes/starts the daemon when `OBSIDIAN_WATCH=true`.
- When `OBSIDIAN_WATCH=false`, MCP skips daemon initialization and semantic search must use local mode (if enabled) or returns a clear error.

- Finds notes by **meaning**, not keywords — "making money from software" surfaces notes about monetization
- **Hybrid mode** — `lexical_prefetch: true` combines BM25 candidate retrieval with semantic re-ranking
- **Tunable blending** — `alpha` controls the weight between lexical and semantic scores (default 0.25)
- Daemon cache migration is one-way and non-destructive: legacy `.obsidian/obsidian-mcp/embeddings.bin` is copied into daemon namespace store when available, and never deleted automatically

### Regex — `search_regex`

Always available. Full regex syntax for pattern matching across all notes.

### When to Use What

| Need | Tool |
|------|------|
| Known keyword or phrase | `search_text` |
| Keyword with possible typo | `search_text` + `fuzzy: true` |
| Conceptual — "notes about X" | `search_semantic` |
| Best precision + recall | `search_semantic` + `lexical_prefetch: true` |
| Structural pattern (URLs, IDs) | `search_regex` |
| By tag or metadata | `search_tag` / `search_frontmatter` |

## Tool Reference

<details>
<summary><strong>All 29 tools</strong> (click to expand)</summary>

### Navigation

| Tool | Parameters | Description |
|------|-----------|-------------|
| `vault_list` | `path?`, `recursive?`, `glob?` | List files and directories |
| `vault_structure` | `path?`, `max_depth?` | Tree view of vault structure |
| `vault_info` | — | Aggregate vault statistics |

### Note CRUD

| Tool | Parameters | Description |
|------|-----------|-------------|
| `note_read` | `path` | Read full note content |
| `note_create` | `path`, `content?`, `frontmatter?` | Create a new note |
| `note_write` | `path`, `content` | Overwrite note content |
| `note_append` | `path`, `content` | Append to end of note |
| `note_prepend` | `path`, `content` | Insert after frontmatter |
| `note_patch` | `path`, `operation`, `target_type`, `target`, `content` | Patch a heading section, block ref, or frontmatter field |
| `note_delete` | `path`, `confirm` | Delete a note (requires `confirm: true`) |
| `note_move` | `from`, `to` | Move or rename a note |

### Search

| Tool | Parameters | Description |
|------|-----------|-------------|
| `search_text` | `query`, `fuzzy?`, `fields?`, `context_length?`, `max_results?` | BM25 full-text search with stemming |
| `search_semantic` | `query`, `top_k?`, `include_content?`, `lexical_prefetch?`, `alpha?` | Semantic similarity search |
| `search_regex` | `pattern`, `context_length?`, `max_results?` | Regex pattern search |
| `search_tag` | `tag`, `include_nested?` | Find notes by tag |
| `search_frontmatter` | `field`, `value?`, `operator` | Query by frontmatter field |

### Metadata & Frontmatter

| Tool | Parameters | Description |
|------|-----------|-------------|
| `note_metadata` | `path` | Tags, headings, links, word count, dates |
| `note_document_map` | `path` | List patchable targets (headings, blocks, fields) |
| `frontmatter_get` | `path` | Read frontmatter as JSON |
| `frontmatter_set` | `path`, `key`, `value` | Set a frontmatter field |
| `frontmatter_remove` | `path`, `key` | Remove a frontmatter field |

### Graph & Links

| Tool | Parameters | Description |
|------|-----------|-------------|
| `links_backlinks` | `path` | Notes linking to this note |
| `links_outgoing` | `path` | Outgoing wikilinks with resolution status |
| `links_broken` | `path?` | Broken wikilinks (vault-wide or single note) |
| `links_orphans` | — | Notes with no inbound or outbound links |

### Periodic Notes

| Tool | Parameters | Description |
|------|-----------|-------------|
| `periodic_get` | `period`, `date?` | Read a periodic note |
| `periodic_create` | `period`, `date?`, `content?` | Create a periodic note from template |
| `periodic_list_recent` | `period`, `limit?` | List recent periodic notes |

### Utility

| Tool | Parameters | Description |
|------|-----------|-------------|
| `open_in_obsidian` | `path`, `new_leaf?` | Open note in Obsidian via `obsidian://` URI |

</details>

## Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OBSIDIAN_VAULT_PATH` | Yes* | — | Absolute path to your Obsidian vault |
| `OBSIDIAN_WATCH` | No | `true` | Filesystem watcher for live index updates |
| `OBSIDIAN_LOG_LEVEL` | No | `info` | `trace`, `debug`, `info`, `warn`, `error` |
| `OBSIDIAN_TANTIVY` | No | `true` | BM25 full-text index |
| `OBSIDIAN_EMBEDDINGS` | No | `false` | Semantic embedding search (requires `embeddings` feature) |
| `OBSIDIAN_EMBEDDINGS_MODEL` | No | `BAAI/bge-small-en-v1.5` | HuggingFace model for embeddings |
| `OBSIDIAN_SEMANTIC_MODE` | No | `auto` | Semantic backend mode: `auto`, `daemon`, `local` |
| `OBSIDIAN_SEMANTIC_HOME` | No | OS data-dir default | Shared semantic runtime home for daemon bin/model/cache state |
| `OBSIDIAN_SEMANTIC_DAEMON_PATH` | No | unset | Override daemon binary path used by bootstrap |
| `OBSIDIAN_SEMANTIC_DAEMON_DOWNLOAD_URL` | No | `https://github.com/lstpsche/obsidian-mcp/releases/download/v<version>/obsidian-semanticd-<version>-<target>.<ext>` | Override daemon download URL when binary is missing |
| `OBSIDIAN_SEMANTIC_MODEL` | No | `BAAI/bge-small-en-v1.5` | Default model used by semantic daemon runtime |
| `OBSIDIAN_SEMANTIC_ENDPOINT` | No | `<semantic-home>/ipc/...` | Daemon-only endpoint override (internal/runtime use) |
| `OBSIDIAN_SEMANTIC_CONNECT_TIMEOUT_MS` | No | `2000` | Per-call daemon timeout in milliseconds |
| `OBSIDIAN_SEMANTIC_CONNECT_RETRIES` | No | `2` | Daemon retry attempts after the first failed call |
| `OBSIDIAN_SEMANTIC_RETRY_BACKOFF_MS` | No | `250` | Base retry backoff in milliseconds |
| `OBSIDIAN_SEMANTIC_PREFETCH` | No | `50` | Default lexical prefetch candidate count for hybrid daemon queries |
| `OBSIDIAN_SEMANTIC_ALPHA` | No | `0.25` | Default hybrid blend weight (`alpha * BM25 + (1-alpha) * semantic`) |
| `OBSIDIAN_HYBRID_ALPHA` | No | alias | Backward-compatible alias for `OBSIDIAN_SEMANTIC_ALPHA` |
| `FASTEMBED_CACHE_DIR` | No | `<semantic-home>/model/fastembed-cache` | Daemon-internal shared fastembed cache root |

\* Also accepted as the first CLI argument: `obsidian-mcp /path/to/vault`

## Architecture

```
AI Client (Cursor, Claude Desktop, etc.)
 │ stdio (MCP JSON-RPC)
 ▼
obsidian-mcp
 ├─ tools/          MCP handlers — translate protocol into vault ops
 ├─ vault/
 │  ├─ index        In-memory metadata (tags, links, headings)
 │  ├─ tantivy      BM25 full-text search
 │  ├─ embeddings   Semantic vectors (optional)
 │  └─ watcher      Filesystem events → index sync
 ├─ models.rs       Shared types
 └─ error.rs        VaultError → MCP ErrorData
 │ fs
 ▼
Vault directory (.md files + .obsidian/)
```

The vault layer is a pure Rust library with no knowledge of MCP. The tools layer is a thin adapter. This separation means the vault code is independently testable and reusable.

## Development

```sh
cargo build                          # default build
cargo build --features embeddings    # with semantic search

OBSIDIAN_VAULT_PATH=~/vault cargo run
OBSIDIAN_VAULT_PATH=~/vault OBSIDIAN_EMBEDDINGS=true cargo run --features embeddings

cargo fmt --check && cargo clippy && cargo test

npx @modelcontextprotocol/inspector cargo run   # interactive testing
```

## License

[MIT](LICENSE)
