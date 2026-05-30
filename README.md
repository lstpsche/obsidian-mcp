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

# Run (stdio — for MCP clients that launch the process)
obsidian-mcp /path/to/your/vault

# Run (HTTP — shared server for multiple agents)
obsidian-mcp --http /path/to/your/vault
```

Add to your MCP client config (Cursor, Claude Desktop, etc.) and you're done — 18 tools are available immediately.

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

On Windows, local embeddings builds require a recent MSVC Build Tools toolset for ONNX Runtime. MSVC 14.44+ is known to work; older VS 2019-era 14.2x toolsets may fail to link with `__std_find_trivial_8`. Use `--features embeddings-api` instead if you want semantic search through an external `/v1/embeddings` server without bundling ONNX Runtime.

For **API embedding backend** (OpenAI, Ollama, vLLM, LM Studio, or any `/v1/embeddings`-compatible endpoint):

```sh
cargo install obsidian-mcp --features embeddings-api
```

This adds no model weight to the binary — embeddings are computed by an external server. Both features can be compiled simultaneously; `OBSIDIAN_EMBEDDING_PROVIDER` selects which runs.

#### API backend examples

```sh
# OpenAI
OBSIDIAN_EMBEDDINGS=true
OBSIDIAN_EMBEDDING_PROVIDER=api
OBSIDIAN_EMBEDDING_API_KEY=sk-...
OBSIDIAN_EMBEDDING_API_MODEL=text-embedding-3-small

# Ollama (local, no auth)
OBSIDIAN_EMBEDDINGS=true
OBSIDIAN_EMBEDDING_PROVIDER=api
OBSIDIAN_EMBEDDING_API_BASE=http://localhost:11434/v1
OBSIDIAN_EMBEDDING_API_MODEL=nomic-embed-text
OBSIDIAN_EMBEDDING_API_KEY=unused

# vLLM / LM Studio
OBSIDIAN_EMBEDDINGS=true
OBSIDIAN_EMBEDDING_PROVIDER=api
OBSIDIAN_EMBEDDING_API_BASE=http://localhost:8000/v1
OBSIDIAN_EMBEDDING_API_MODEL=BAAI/bge-small-en-v1.5
OBSIDIAN_EMBEDDING_API_KEY=unused

# OpenRouter (25+ embedding models from multiple providers)
OBSIDIAN_EMBEDDINGS=true
OBSIDIAN_EMBEDDING_PROVIDER=api
OBSIDIAN_EMBEDDING_API_BASE=https://openrouter.ai/api/v1
OBSIDIAN_EMBEDDING_API_MODEL=openai/text-embedding-3-small
OBSIDIAN_EMBEDDING_API_KEY=sk-or-v1-...
```

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

### Install from source (edge)

If you have the repo cloned, you can install the latest `main` build locally:

```sh
./bin/install-edge
```

This pulls the latest `main` and runs `cargo install --path . --features embeddings`.

## Semantic Runtime Compatibility

Current semantic daemon API version: `1` (`DAEMON_API_VERSION` in `src/daemon/protocol.rs`).

| Component | Compatibility contract | Enforcement |
|----------|-------------------------|-------------|
| `obsidian-mcp` | Daemon API version must match exactly | MCP daemon `health` handshake validates `min_api_version`/`max_api_version` against current API version and fails fast on mismatch |
| `obsidian-semantic-search-plugin` | Daemon API version must match exactly | Plugin bootstrap performs daemon `health` handshake and surfaces explicit incompatibility notices |

Release asset compatibility expectations:

- Daemon auto-install clients expect release assets named `obsidian-semanticd-<version>-<target>.{tar.gz|zip}` plus `checksums.sha256`.
- Plugin releases should publish `main.js`, `manifest.json`, and optional `styles.css` as GitHub release assets.

Upgrade guidance:

- Prefer upgrading `obsidian-mcp`, `obsidian-semanticd`, and `obsidian-semantic-search-plugin` together.
- If versions are skewed, startup/handshake fails with explicit API incompatibility errors rather than silent fallback.

## Transport Modes

obsidian-mcp supports two transports. Both are always compiled in — choose the one that fits your workflow.

### stdio (default)

The standard MCP transport. The AI client spawns the process and communicates over stdin/stdout.

**Use when:** your MCP client manages the server process (Cursor, Claude Desktop, most MCP clients).

```sh
obsidian-mcp /path/to/vault
```

Each client connection spawns its own process. Simple, zero-config, works everywhere.

### Streamable HTTP

A persistent HTTP server that multiple clients can connect to simultaneously.

**Use when:** you run multiple headless agents (`cursor agent -p`, parallel Claude sessions, etc.) against the same vault and want to share a single server process instead of spawning one per agent.

```sh
# Foreground (for development / process managers like launchd, systemd)
obsidian-mcp --http /path/to/vault

# Background (ad-hoc daemonize — spawns child, parent exits)
obsidian-mcp serve /path/to/vault
```

The `serve` command daemonizes the server and redirects logs to a platform-specific file:
- **macOS:** `~/Library/Logs/obsidian-mcp.log`
- **Linux:** `$XDG_STATE_HOME/obsidian-mcp/obsidian-mcp.log`
- **Windows:** `%LOCALAPPDATA%/obsidian-mcp/obsidian-mcp.log`

Default: `http://127.0.0.1:37842`. MCP tools are served at `/mcp`, health check at `/health`.

Benefits over stdio for multi-agent setups:
- **Shared index** — one in-memory BM25/embedding index instead of N copies
- **Lower resource usage** — single filesystem watcher, single process
- **Process independence** — server stays up when agents come and go
- **Standard observability** — HTTP health checks, logging, standard networking

Configure the address:

```sh
obsidian-mcp --http --port 9000 --host 0.0.0.0 /path/to/vault
# or via env vars
OBSIDIAN_TRANSPORT=http OBSIDIAN_HTTP_PORT=9000 obsidian-mcp /path/to/vault
```

### Server Management

```sh
obsidian-mcp serve /path/to/vault    # Start HTTP server in background
obsidian-mcp stop                    # Stop running server (default port)
obsidian-mcp stop --port 9000        # Stop server on specific port
obsidian-mcp restart /path/to/vault  # Stop + start (picks up new binary after upgrade)
```

`serve` and `restart` wait for the server to pass a `/health` check before reporting success (up to 15s). If the server fails during startup, the exit code and log path are reported.

## Client Setup

### Cursor (stdio)

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

### Cursor (HTTP — shared server)

Start the server once:

```sh
obsidian-mcp serve /path/to/your/vault
```

Then point Cursor at it:

```json
{
  "mcpServers": {
    "obsidian": {
      "url": "http://127.0.0.1:37842/mcp"
    }
  }
}
```

All Cursor agents (IDE, CLI, headless) share the same server.

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

obsidian-mcp supports both **stdio** and **Streamable HTTP** MCP transports. Any MCP-compatible client can connect via either method. Pass the vault path as the first argument or via `OBSIDIAN_VAULT_PATH`.

## Running as a Service

For always-on HTTP mode, use your OS process manager with `--http` (not `serve`). This lets the process manager handle restarts, logging, and lifecycle — `serve` daemonizes itself, which conflicts with process managers that expect to own the child process.

### macOS (launchd)

Create `~/Library/LaunchAgents/com.obsidian-mcp.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.obsidian-mcp</string>

  <key>ProgramArguments</key>
  <array>
    <string>/Users/YOU/.cargo/bin/obsidian-mcp</string>
    <string>--http</string>
  </array>

  <key>EnvironmentVariables</key>
  <dict>
    <key>OBSIDIAN_VAULT_PATH</key>
    <string>/path/to/your/vault</string>
    <!-- Add embedding env vars here if using API embeddings:
    <key>OBSIDIAN_EMBEDDINGS</key>
    <string>true</string>
    <key>OBSIDIAN_EMBEDDING_PROVIDER</key>
    <string>api</string>
    <key>OBSIDIAN_EMBEDDING_API_KEY</key>
    <string>sk-...</string>
    <key>OBSIDIAN_EMBEDDING_API_BASE</key>
    <string>https://api.openai.com/v1</string>
    <key>OBSIDIAN_EMBEDDING_API_MODEL</key>
    <string>text-embedding-3-small</string>
    -->
  </dict>

  <key>RunAtLoad</key>
  <true/>

  <key>KeepAlive</key>
  <true/>

  <key>StandardOutPath</key>
  <string>/Users/YOU/Library/Logs/obsidian-mcp/launchd.out.log</string>

  <key>StandardErrorPath</key>
  <string>/Users/YOU/Library/Logs/obsidian-mcp/launchd.err.log</string>
</dict>
</plist>
```

> **Important:** launchd does not source your shell profile (`~/.zshrc`, `~/.bashrc`). All environment variables must be defined in the plist's `EnvironmentVariables` section.

```sh
# Load
launchctl load ~/Library/LaunchAgents/com.obsidian-mcp.plist

# Unload (stop)
launchctl unload ~/Library/LaunchAgents/com.obsidian-mcp.plist

# Restart (after upgrade)
launchctl unload ~/Library/LaunchAgents/com.obsidian-mcp.plist && \
  launchctl load ~/Library/LaunchAgents/com.obsidian-mcp.plist
```

### Linux (systemd)

Create `~/.config/systemd/user/obsidian-mcp.service`:

```ini
[Unit]
Description=obsidian-mcp MCP server
After=network.target

[Service]
ExecStart=%h/.cargo/bin/obsidian-mcp --http
Environment=OBSIDIAN_VAULT_PATH=/path/to/your/vault
# Add embedding env vars if needed:
# Environment=OBSIDIAN_EMBEDDINGS=true
# Environment=OBSIDIAN_EMBEDDING_PROVIDER=api
# Environment=OBSIDIAN_EMBEDDING_API_KEY=sk-...
# Environment=OBSIDIAN_EMBEDDING_API_BASE=https://api.openai.com/v1
# Environment=OBSIDIAN_EMBEDDING_API_MODEL=text-embedding-3-small
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

```sh
systemctl --user daemon-reload
systemctl --user enable --now obsidian-mcp

# Check status
systemctl --user status obsidian-mcp

# Restart (after upgrade)
systemctl --user restart obsidian-mcp

# View logs
journalctl --user -u obsidian-mcp -f
```

## Upgrading

```sh
# Install new version
cargo install obsidian-mcp --force
# With features:
cargo install obsidian-mcp --features embeddings,embeddings-api --force

# Restart the running server to pick up the new binary
obsidian-mcp restart /path/to/vault

# Or if using a process manager:
launchctl unload ~/Library/LaunchAgents/com.obsidian-mcp.plist && \
  launchctl load ~/Library/LaunchAgents/com.obsidian-mcp.plist
# systemd: systemctl --user restart obsidian-mcp
```

> `cargo install --force` replaces the binary on disk but does **not** restart any running server. You must restart manually — otherwise the old version continues serving.

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
| By tag or metadata | `search_metadata` |

## Tool Reference

<details>
<summary><strong>All 18 tools</strong> (click to expand)</summary>

### Navigation

| Tool | Parameters | Description |
|------|-----------|-------------|
| `vault_list` | `path?`, `recursive?`, `glob?`, `format?`, `max_depth?` | List files (`format: "list"`) or tree view (`format: "tree"`) |
| `vault_info` | — | Aggregate vault statistics |

### Note CRUD

| Tool | Parameters | Description |
|------|-----------|-------------|
| `note_read` | `path` | Read full note content |
| `note_create` | `path`, `content?`, `frontmatter?` | Create a new note |
| `note_write` | `path`, `content` | Overwrite note content |
| `note_insert` | `path`, `content`, `position?` | Insert at end (`"end"`, default) or beginning (`"beginning"`) |
| `note_patch` | `path`, `operation`, `target_type`, `target`, `content` | Patch a heading section, block ref, or frontmatter field |
| `note_delete` | `path`, `confirm` | Delete a note (requires `confirm: true`) |
| `note_move` | `from`, `to` | Move or rename a note |

### Search

| Tool | Parameters | Description |
|------|-----------|-------------|
| `search_text` | `query`, `fuzzy?`, `fields?`, `context_length?`, `max_results?` | BM25 full-text search with stemming |
| `search_semantic` | `query`, `top_k?`, `include_content?`, `lexical_prefetch?`, `alpha?` | Semantic similarity search |
| `search_regex` | `pattern`, `context_length?`, `max_results?` | Regex pattern search |
| `search_metadata` | `type`, `tag?`, `include_nested?`, `field?`, `value?`, `operator?` | Find by tag (`type: "tag"`) or frontmatter (`type: "frontmatter"`) |

### Introspection & Frontmatter

| Tool | Parameters | Description |
|------|-----------|-------------|
| `note_inspect` | `path`, `view?` | Metadata (`"metadata"`, default) or patchable targets (`"targets"`) |
| `frontmatter` | `action`, `path`, `key?`, `value?` | Get (`"get"`), set (`"set"`), or remove (`"remove"`) frontmatter fields |

### Graph & Links

| Tool | Parameters | Description |
|------|-----------|-------------|
| `wikilinks` | `query`, `path?` | Backlinks, outgoing, broken, or orphans (`query: "backlinks"/"outgoing"/"broken"/"orphans"`) |

### Periodic Notes

| Tool | Parameters | Description |
|------|-----------|-------------|
| `periodic` | `action`, `period`, `date?`, `content?`, `limit?` | Get, create, or list periodic notes (`action: "get"/"create"/"list"`) |

### Utility

| Tool | Parameters | Description |
|------|-----------|-------------|
| `open_in_obsidian` | `path`, `new_leaf?` | Open note in Obsidian via `obsidian://` URI |

</details>

## Tool Filtering

Control which tools are exposed via the `OBSIDIAN_TOOLS` environment variable or per-session `X-Obsidian-Tools` HTTP header.

| Value | Effect |
|-------|--------|
| `full` (or unset) | All 18 tools |
| `core` | 14 tools — drops `search_semantic`, `search_regex`, `periodic`, `open_in_obsidian` |
| `read` | 10 tools — read-only (no create/write/insert/patch/delete/move) |
| `minimal` | 6 tools — `vault_list`, `vault_info`, `note_read`, `note_create`, `note_write`, `search_text` |
| `tool1,tool2,...` | Allow-list — only the named tools |
| `!tool1,!tool2,...` | Deny-list — all tools except the named ones |

## Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OBSIDIAN_VAULT_PATH` | Yes* | — | Absolute path to your Obsidian vault |
| `OBSIDIAN_TRANSPORT` | No | `stdio` | Transport mode: `stdio` or `http` |
| `OBSIDIAN_HTTP_PORT` | No | `37842` | HTTP listen port |
| `OBSIDIAN_HTTP_HOST` | No | `127.0.0.1` | HTTP bind address |
| `OBSIDIAN_WATCH` | No | `true` | Filesystem watcher for live index updates |
| `OBSIDIAN_LOG_LEVEL` | No | `info` | `trace`, `debug`, `info`, `warn`, `error` |
| `OBSIDIAN_TANTIVY` | No | `true` | BM25 full-text index |
| `OBSIDIAN_TOOLS` | No | `full` | Tool filtering: profile name, comma-separated allow-list, or `!`-prefixed deny-list |
| `OBSIDIAN_MCP_DATA` | No | `{vault}/.obsidian-mcp` | External obsidian-mcp data directory. When set, cache/config data is stored under `{value}/vaults/{vault_slug}/` and merged with vault-local config |
| `OBSIDIAN_EXCLUDE_PATHS` | No | *(empty)* | Comma-separated glob patterns excluded from indexing, search, graph, and stats. Merged with `.obsidian-mcp/ignore`; restart after changing ignore files |
| `OBSIDIAN_EMBEDDINGS` | No | `false` | Semantic embedding search (requires `embeddings` or `embeddings-api` feature) |
| `OBSIDIAN_EMBEDDINGS_MODEL` | No | `BAAI/bge-small-en-v1.5` | HuggingFace model for embeddings |
| `OBSIDIAN_EMBEDDING_PROVIDER` | No | *(infer)* | Embedding backend: `local` (fastembed) or `api` (OpenAI-compatible) |
| `OBSIDIAN_EMBEDDING_API_KEY` | When api | Fallback: `OPENAI_API_KEY` | API authentication key |
| `OBSIDIAN_EMBEDDING_API_BASE` | No | `https://api.openai.com/v1` | Embedding API base URL (fallback: `OPENAI_BASE_URL`) |
| `OBSIDIAN_EMBEDDING_API_MODEL` | When api | Fallback: `OPENAI_MODEL` | Model name for embedding API |
| `OBSIDIAN_EMBEDDING_DIM` | No | *(probed)* | Override embedding dimension (skips API probe) |
| `OBSIDIAN_EMBEDDING_CA_CERT` | No | — | Path to PEM CA certificate for API TLS |
| `OBSIDIAN_EMBEDDING_TLS_VERIFY` | No | `true` | Set to `false` to skip TLS verification |
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
AI Client(s)
 │
 ├─ stdio (1:1 process)       ←  default, single-client
 │     OR
 ├─ HTTP POST /mcp (N:1)      ←  shared server, multi-client
 │
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

In HTTP mode, each MCP session gets its own handler instance, but all sessions share a single `Vault` (thread-safe via `Arc<RwLock<...>>`). One filesystem watcher, one BM25 index, one embedding store — regardless of how many agents are connected.

## Development

```sh
cargo build                          # default build
cargo build --features embeddings    # with local semantic search (fastembed)
cargo build --features embeddings-api  # with API semantic search (OpenAI-compatible)

# stdio mode (default)
OBSIDIAN_VAULT_PATH=~/vault cargo run

# HTTP mode
OBSIDIAN_VAULT_PATH=~/vault cargo run -- --http

cargo fmt --check && cargo clippy && cargo test

npx @modelcontextprotocol/inspector cargo run   # interactive testing (stdio)
```

## License

[MIT](LICENSE)
