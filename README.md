# obsidian-mcp

[![CI](https://github.com/lstpsche/obsidian-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/lstpsche/obsidian-mcp/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/obsidian-mcp.svg)](https://crates.io/crates/obsidian-mcp)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A high-performance [MCP](https://modelcontextprotocol.io) server that gives AI agents full read/write access to [Obsidian](https://obsidian.md) vaults. Written in Rust. Ships as a single binary.

**No Obsidian plugins. No REST API. No Obsidian running. Use a local folder or an SFTP vault.**

## Why?

Existing solutions for AI + Obsidian require the Obsidian app to be running with a community plugin ([obsidian-local-rest-api](https://github.com/coddingtonbear/obsidian-local-rest-api)) and an MCP wrapper ([mcp-obsidian](https://github.com/MarkusPfundstein/mcp-obsidian)) on top. That's fragile, slow, and limited to what the REST API exposes.

obsidian-mcp reads and writes vault files directly. It understands the Obsidian format natively — wikilinks, frontmatter, tags, block references, periodic notes, the full graph. Local folders use filesystem notifications; remote folders use SFTP reconciliation. Search indexes and embedding caches stay on the computer running the MCP server. Local vaults need only the binary; SFTP vaults also need a local OpenSSH client.

## Quick Start

```sh
# Install
cargo install obsidian-mcp

# Run (stdio — for MCP clients that launch the process)
obsidian-mcp /path/to/your/vault

# Run (HTTP — shared server for multiple agents)
obsidian-mcp --http /path/to/your/vault

# Remote folder (OpenSSH keys/agent and known host configured locally)
obsidian-mcp sftp://vault-server/home/alice/notes
```

Add to your MCP client config (Cursor, Claude Desktop, etc.). Local vaults expose 19 tools before filtering. Remote vaults expose the same operations except `open_in_obsidian`, which requires a local Obsidian vault. See [remote vault setup and behavior](docs/remote-vaults.md).

## Optional authenticated clients

To give assistants different permissions on one local vault, run:

```sh
obsidian-mcp access setup /path/to/your/vault
```

Setup guides you through reader/editor access and shared folders, then creates a protected token and client configuration snippet. Follow its printed server, reconnect and connection-check commands. It generates Codex and Claude Code header-helper configuration formats. The Codex CLI 0.166.0 local app-server integration supports the generated helper; verify other installed clients separately.

Authentication is enabled only by an explicit `--access-file PATH` or `OBSIDIAN_ACCESS_FILE` setting. Existing unauthenticated HTTP, stdio, daemon and SFTP configurations keep their behavior when these settings are absent. Creating a profile does not change a running server or client configuration.

See [authenticated clients](docs/access-control.md) for permissions, remote connections, restricted stdio, credential rotation and recovery.

## What It Can Do

| Category | Capabilities |
|----------|-------------|
| **Navigate** | List files, tree view, vault stats |
| **Read & write** | Create, read, overwrite, append, prepend, move, delete notes |
| **Patch** | Edit individual heading sections, block references, or frontmatter fields without touching the rest of the note |
| **Search** | BM25 full-text (Tantivy) with stemming, fuzzy matching, and per-field filtering. Semantic search via managed local or API embeddings. Regex. Tag and frontmatter queries. |
| **Graph** | Backlinks, outgoing links, broken link detection, orphan discovery |
| **Frontmatter** | Get/set/remove fields, query notes by metadata |
| **Periodic notes** | Daily, weekly, monthly, quarterly, yearly — with Obsidian-compatible date formats and template expansion |

Local metadata and BM25 indices update through the filesystem watcher. Remote indexes reconcile through periodic scans, and direct note reads fetch remote content. Embeddings reconcile asynchronously through a managed background runtime. Remote initial indexing reports warming until its catalog is available; unknown counts are not reported as an empty vault.

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
OBSIDIAN_EMBEDDING_API_PROFILE=/absolute/path/nomic-profile.json
# Optional tuning for a slow CPU-only embedding server
OBSIDIAN_EMBEDDING_BATCH_SIZE=8
OBSIDIAN_EMBEDDING_API_TIMEOUT_SECS=300
OBSIDIAN_SEMANTIC_CONNECT_TIMEOUT_MS=310000

# vLLM / LM Studio
OBSIDIAN_EMBEDDINGS=true
OBSIDIAN_EMBEDDING_PROVIDER=api
OBSIDIAN_EMBEDDING_API_BASE=http://localhost:8000/v1
OBSIDIAN_EMBEDDING_API_MODEL=BAAI/bge-small-en-v1.5
OBSIDIAN_EMBEDDING_API_KEY=unused
OBSIDIAN_EMBEDDING_API_PROFILE=/absolute/path/bge-profile.json

# OpenRouter (25+ embedding models from multiple providers)
OBSIDIAN_EMBEDDINGS=true
OBSIDIAN_EMBEDDING_PROVIDER=api
OBSIDIAN_EMBEDDING_API_BASE=https://openrouter.ai/api/v1
OBSIDIAN_EMBEDDING_API_MODEL=openai/text-embedding-3-small
OBSIDIAN_EMBEDDING_API_KEY=sk-or-v1-...
OBSIDIAN_EMBEDDING_API_PROFILE=/absolute/path/openrouter-profile.json
```

Smaller batches reduce the work in each embedding request. Increase the API
timeout when a batch can take longer than the default 30 seconds. In daemon mode,
the separate `OBSIDIAN_SEMANTIC_CONNECT_TIMEOUT_MS` deadline covers the complete
daemon call, including query embedding; allow enough time for that request too.
These values are examples, not throughput guarantees.

The shared daemon uses the embedding configuration from its launch environment.
Connecting another MCP server does not replace a running daemon's provider,
endpoint, model, batch size, or timeout settings. Apply changes to the daemon's
launcher and restart it. Use separate semantic homes for different daemon backend
profiles, or `OBSIDIAN_SEMANTIC_MODE=local` for per-process configuration.

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

Current semantic daemon API version: `2` (`DAEMON_API_VERSION` in `src/daemon/protocol.rs`).

Semantic search requires the health capabilities `semantic_source_evidence_v1` and `scoped_indexing_v1`. New clients verify both on the connection that will carry the request, after reconciling an owned daemon. Older v2 clients can still perform their existing health/activation handshake. New clients report an unsupported contract when an external daemon lacks either capability. Restart or upgrade that daemon through its owner. A capability mismatch does not activate a competing local index.

For an owned daemon at the same version, bootstrap may activate an installed sibling that advertises the missing capability through `--__build-info`. A sibling without the capability leaves the running daemon intact and remains explicitly unsupported; newer running versions and external owners are preserved.

| Component | Compatibility contract | Enforcement |
|----------|-------------------------|-------------|
| `obsidian-mcp` | IPC v2 and both semantic capabilities | The client validates health before submitting scope or search data and holds the attachment connection open. |
| `obsidian-semantic-search-plugin` | A release pinned to IPC v1 requires a v1 daemon. | A plugin build connecting to this daemon must negotiate scoped IPC v2, retain its attachment connection and send the issued handle with content requests. |

Daemon API v2 associates document indexing with a live, scoped attachment. Clients with the same effective exclusions on the same daemon share their index and watcher. Different policies have separate caches, counters and search results. The running provider and model remain shared. A document visible to two different scopes can be embedded twice. Each request can narrow eligible paths further using `allowed_paths` before ranking.

Use `.obsidian-mcp/ignore` for restrictions that every client of a vault must honor. The MCP instance’s machine ignore and `OBSIDIAN_EXCLUDE_PATHS` can narrow its own scope. A separately launched daemon may impose additional operator exclusions. Withdrawing the final attachment stops new document submissions from that scope after the daemon observes the disconnect. An already dispatched request can finish. Restart the owners after changing ignore files. A new attachment that observes a changed portable policy invalidates the older views.

During an upgrade, retire the old daemon before relying on scoped exclusions. Older IPC v2 tools can still probe health for activation. Clients performing semantic operations need both capabilities and must attach explicitly. Before downgrading to an unscoped daemon, stop the scoped MCP instances and daemon. Keep embeddings disabled for vaults that depend on these exclusions until a compatible owner is running again.

Release binaries for Linux, Windows, and Apple Silicon macOS include local and API embedding backends. Intel macOS binaries include the API backend only because the current ONNX dependency has no prebuilt Intel macOS library. Cargo users on Intel macOS should install with `--features embeddings-api` instead of `--features embeddings`. Every release archive is smoke-tested on its native target.

The combined ARM64 Linux archives use the Ubuntu 24.04 release runner’s glibc and C++ runtime. Debian 12’s system libraries lack symbols required by the bundled ONNX Runtime. To use a remote embedding provider on Debian 12, build locally with `--no-default-features --features embeddings-api`.

Release asset compatibility expectations:

- Daemon auto-install clients expect release assets named `obsidian-semanticd-<version>-<target>.{tar.gz|zip}` plus `checksums.sha256`.
- Plugin releases should publish `main.js`, `manifest.json`, and optional `styles.css` as GitHub release assets.

Upgrade guidance:

- Use `obsidian-mcp upgrade` for Cargo-installed `obsidian-mcp` and `obsidian-semanticd` binaries; it preserves the compiled feature set and reconciles the locally owned daemon.
- Upgrade `obsidian-semantic-search-plugin` through Obsidian's plugin manager.
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

The same MCP endpoint supports `server/discover` and stateless tool requests with protocol `2026-07-28`, plus the legacy `initialize` and session flow used by `2025-11-25` clients. Modern requests carry the protocol version and client metadata on each call. Reverse proxies must preserve MCP headers and forward an allowed loopback `Host` to the local server.

A stdio client may probe `server/discover` and then select the legacy lifecycle
with `initialize` on the same connection. Discovery alone does not require later
legacy requests to carry modern per-request metadata.

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

`stop`, `serve`, and `restart` target the exact bind address. A missing or invalid `--host` or `--port` value stops the command before listener discovery. The commands verify that the listener belongs to this installation before stopping its process. An unrelated listener or a supervisor that starts a replacement requires action through its owner. On Linux these commands use `lsof` for listener discovery and a kernel PID file descriptor for signaling.

`serve` and `restart` require a fixed port and verify the new child owns a healthy listener before reporting success. Startup errors include the log path. Foreground `--http --port 0` lets the operating system assign a port and reports the bound address in the listening log event.

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

obsidian-mcp supports both **stdio** and **Streamable HTTP** MCP transports for local and SFTP sources. Any MCP-compatible client can connect via either method. Pass the local vault path or SFTP URI as the first argument or via `OBSIDIAN_VAULT_PATH`.

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

# A direct, running user LaunchAgent is restarted automatically by:
obsidian-mcp upgrade
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

# A direct, running user service is restarted automatically by:
obsidian-mcp upgrade

# View logs
journalctl --user -u obsidian-mcp -f
```

## Upgrading

```sh
# Inspect the verified install and exact command without changing anything
obsidian-mcp upgrade --dry-run

# Install the latest crates.io version and activate it
obsidian-mcp upgrade
```

`upgrade` supports official crates.io installations recorded in Cargo's install tracker. It preserves the current target, build profile, and exact compiled feature set; verifies both shipped binaries after Cargo finishes; restarts directly managed running launchd/systemd user services; and reconciles a locally owned semantic daemon without changing its home, model, or caches. Explicit semantic-daemon overrides and PATH-owned daemons are left alone.

Vault settings and service definitions are never rewritten. Stdio clients cannot be restarted from inside their active protocol session, so reconnect them after an update. An ad-hoc background process started with `obsidian-mcp serve` must also be restarted with the same arguments or environment; the upgrader cannot safely reconstruct that launch context. If installation succeeds but activation fails, the command exits non-zero with diagnostics and does not silently roll back the Cargo installation.

Homebrew, Nix, source/path, Git, alternate-registry, and pre-built binary installs are rejected before mutation. Upgrade those through the tool that owns the installation.

See [Upgrading](docs/upgrading.md) for the support matrix, guarantees, outcomes, and recovery guidance.

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

Uses the shared local semantic daemon by default (`OBSIDIAN_SEMANTIC_MODE=auto`). Search returns distinct notes with revision-bound semantic evidence through either the daemon or local backend.

Automatic chunking applies to both embedding backends. The server prepares the complete Markdown body after YAML frontmatter, together with its filename title, and counts the final input using the backend's exact tokenizer. A fitting note uses one embedding. An overflowing note is partitioned along Markdown headings and blocks, with further UTF-8-safe splitting where required. Every piece includes its title and ancestor-heading context. No chunking setting or search mode is needed, and note files are never rewritten.

The local backend uses its loaded tokenizer's effective input capacity, normally 512 tokens. That configured boundary is distinct from a model's theoretical maximum. API budgets come from a verified endpoint/model profile. Word counts and vector dimensions cannot establish either budget. Overlong search queries return an invalid-parameter error with the measured and supported token counts rather than silently losing query content.

Only the official `https://api.openai.com/v1` endpoint and exact model IDs `text-embedding-3-small`, `text-embedding-3-large`, and `text-embedding-ada-002` resolve automatically to `cl100k_base` and the documented limits of 8,192 tokens per input, 2,048 inputs and 300,000 aggregate tokens per request. Other endpoints, including OpenRouter and local servers, require `OBSIDIAN_EMBEDDING_API_PROFILE`. Model aliases and generic provider tokenizer labels are not sufficient to infer an exact tokenizer.

API base URLs must use HTTP(S) with a host and optional path, without embedded credentials, query strings or fragments. Authentication is supplied through `OBSIDIAN_EMBEDDING_API_KEY` or `OPENAI_API_KEY`; profiles contain no credentials.

A profile is strict JSON tied to the exact endpoint and outgoing model. For example, a deployment using OpenRouter's OpenAI model can supply a deliberately single-input request profile after verifying that model contract:

```json
{
  "version": 1,
  "profile_revision": "openrouter-openai-text-embedding-3-small-v1",
  "api_base": "https://openrouter.ai/api/v1",
  "model": "openai/text-embedding-3-small",
  "tokenizer": { "kind": "cl100k_base" },
  "max_input_tokens": 8192,
  "max_batch_inputs": 1,
  "max_batch_tokens": 8192
}
```

For a HuggingFace tokenizer, replace `tokenizer` with `{"kind":"huggingface","path":"tokenizer.json","sha256":"<actual lowercase SHA-256>","add_special_tokens":true}`. The file must export the fully initialized tokenizer, including normalization, added tokens and postprocessing; its special-token behavior must match the provider. Relative paths resolve against the profile directory. Profiles are limited to 1 MiB and tokenizer files to 64 MiB. Unknown fields, mismatched endpoint/model, invalid limits and digest mismatches are configuration errors before inference. `profile_revision` identifies the reviewed deployment contract; it cannot detect a provider secretly replacing model weights under the same alias.

Request item, token and serialized-byte limits are enforced independently. Local inference dispatches at most 32 prepared inputs per request, with an aggregate token ceiling of 32 times the loaded per-input capacity. Reconciliation batch size continues to count notes; a note producing many chunks does not multiply the local inference request size. The daemon reads its own provider/profile environment at startup; later client attachments do not change it. Restart that owner after changing the profile or tokenizer. API-only builds include tokenizer support and use the external model without bundling ONNX or downloading model weights.

API dimension probes and search queries retry transient transport failures, HTTP 429 and server errors for at most four attempts within one `OBSIDIAN_EMBEDDING_API_TIMEOUT_SECS` budget, including queueing, waits and requests. Numeric `Retry-After` delays are honored only when another attempt fits that budget; otherwise the request fails with its cause. Permanent HTTP errors and complete invalid responses are not retried; an incomplete response body remains a transport failure. Document requests make one attempt per dispatch; the indexing coordinator owns their backoff and retained progress, waiting at least as long as any numeric provider retry delay. Transient source-read and validated local-inference failures also use managed backoff. Invalid source, configuration and resource failures remain explicit until the source or owner configuration changes. Provider query failures do not select a competing local owner.

Semantic runtime modes:

- `auto` (default): prefer daemon, fallback to local in-process embeddings only when the daemon is genuinely unavailable and local embeddings are enabled. The local runtime remains dormant while the daemon is healthy or warming. Once activated, local fallback remains selected until the MCP server restarts.
- `daemon`: daemon-only path; semantic calls fail clearly if daemon is unavailable. No local embedding runtime is activated.
- `local`: force the in-process embedding runtime.

A slow daemon query or a warming index does not establish that the daemon stopped
working. Those errors remain visible and do not activate a competing local index.
If bootstrap or the initial health probe times out, later semantic searches resume
startup and complete the handshake before using that daemon. An already started
process retains its endpoint and startup metadata so recovery does not launch a
second process. Passive `vault_info` inspection never resumes bootstrap.

Daemon startup policy:

- In `auto` mode, disabling `OBSIDIAN_WATCH` keeps semantic work local. Explicit `daemon` mode can attach a snapshot with watching disabled. The shared view is watched while any live holder requests watching.

Embedding lifecycle:

| Phase | Queryable | Meaning |
|-------|-----------|---------|
| `warming` | No | The model is loading or the first verified index is being built. A semantic query returns an explicit retryable not-ready error. |
| `warming` | Yes | A same-space last-known-good cache is serving queries while changed notes reconcile in the background. |
| `ready` | Yes | Every currently indexed note has completed reconciliation. An empty vault is also ready. |
| `degraded` | Yes or no | Some work failed. Complete last-known-good note projections can remain usable. Transient provider and persistence failures retry; invalid source, configuration and supported-resource failures require a new source generation or corrected owner configuration. |

Core note, graph, BM25, write, and MCP initialization behavior stays available in every phase. Repeated note changes are coalesced by path, and the newest write or removal wins over in-flight inference.

Use `vault_info` to inspect semantic progress without submitting an embedding
query. Its `embeddings` object identifies `source` (`local` or `daemon`) and
`availability` (`available`, `disabled`, `unavailable`, `unattached`, or
`unsupported`). When a runtime snapshot is available, `status` contains `phase`,
`queryable`, `indexed_notes`, `total_notes`, `pending_notes`, and an optional
`last_error`. Otherwise, the snapshot is omitted and an error can explain the
unavailable state. `available` identifies a status snapshot; check `queryable` to
determine whether it can serve semantic queries. Inspection never attaches a daemon vault or activates local
fallback. Older API v2 daemons without passive status support report `unsupported`.

Existing vectors can remain indexed while awaiting refresh. The counts describe
the current store and pending work, not a completed percentage or ETA. HTTP
`/health` continues to describe the local runtime; use `vault_info` for the selected
semantic backend.

While loading or reconciling, info-level logs report counts and in-flight timing
at most once every 30 seconds in addition to lifecycle changes. Debug-level batch
logs include request size, duration, outcome, and sanitized failures. Ready and
dormant runtimes do not emit idle progress logs.

Embedding cache schema 2 stores complete per-note chunk collections and is atomically replaced. Reuse requires matching backend, canonical model, vector dimension, tokenizer fingerprint, effective input capacity, preparation/partition policy, and—for API backends—the endpoint fingerprint. Schema-1 prefix embeddings are rebuilt automatically from notes rather than relabeled as complete coverage. The first rebuild embeds more text and may increase API requests, token use and cache size. Progress remains visible as warming/degraded until the current reconciliation completes. The local checkpoint is `{mcp_data}/embeddings/scopes/<scope_id>/embeddings.bin`. The daemon checkpoint is `<semantic-home>/vaults/<vault_id>/scopes/<scope_id>/embeddings.bin`. Scoped indexes rebuild from permitted notes on first use and leave unscoped files intact. Checkpoints contain complete note collections, including when the whole vault is still reconciling.

Compatible cached notes whose complete raw source revision is unchanged skip preparation, tokenization and inference. Changed notes reuse exact unchanged chunk inputs while refreshing source revisions and ranges. A later request failure preserves the previous complete note collection. Supported resource limits produce explicit failures and preserve the prior complete note collection; they never turn oversized input into a partially complete success.

<a id="semantic-limits"></a>

| Resource | Supported ceiling |
|----------|-------------------|
| Raw source for semantic preparation or revision-bound reads | 16 MiB, enforced during reading; deliberate legacy `note_read({path})` remains unbounded |
| Structure collected for a note that requires partitioning | 4,096 headings and 65,536 parsed Markdown blocks |
| Chunks per note | 4,096 |
| Prepared title/heading context prefix per chunk | 16 KiB; mandatory context that leaves no model capacity is an error |
| Advancing-boundary search when an initial split cannot fit | 4,096 candidate probes; failure is explicit rather than dropping source |
| Serialized embedding request | 16 MiB, in addition to the backend's input/item/aggregate-token limits |
| Coordinator-owned unpublished preparation and staging | 64 MiB per runtime for owned source/prepared/staged/output buffers, vectors and parser-output collections, including preparation/publication reservations; paused drafts may be evicted and rebuilt to admit other work |
| Serialized embedding cache | 1 GiB; at most 1,000,000 notes and 1,000,000 total chunks; loading also limits notes to the current visible count plus 1,024 reconciliation entries |
| Cached note path, note metadata and vectors | 16 KiB path; 64 KiB title/tag UTF-8 text and 4,096 tags per note; vector dimension at most 65,536 |
| Bounded source page | 64 KiB returned content by default, at most 256 KiB |
| Heading projection | 100 headings/64 KiB serialized projection by default; at most 1,000 headings/256 KiB per page; heading display previews at most 256 characters |

Source/page budgets count UTF-8 bytes, not tokens. Projection byte budgets count the compact projection object. Neither is a total MCP wire-size guarantee because text and structured output can duplicate and escape content. The staging bound covers application-owned coordinator allocations. Private pulldown-cmark, YAML and tokenizer temporary arenas receive bounded inputs/structure, but their APIs do not expose an allocator budget; those internals and ONNX workspaces are outside this cap. It is not a total process-RSS limit. API-profile files are separately capped at 1 MiB and tokenizer exports at 64 MiB. These fixed resource ceilings do not select a chunking mode or override the selected model's exact token capacity.

Each result's `source_revision` is the lowercase SHA-256 of the complete raw note, including frontmatter. `source_bytes` and every half-open UTF-8 byte range refer to that exact raw revision. `evidence.scope` is `note` for a fitting whole-body embedding or `passage` for a matching chunk; `evidence.range`, `heading_path` and optional `section_range` locate it. Pure semantic scores use the winning cosine; hybrid `score` blends lexical and semantic values while `evidence.cosine_similarity` retains the actual cosine. `top_k` counts distinct notes after grouping their embeddings.

`snippet` is a preview capped at 300 characters, with its exact source span in `preview.range`. `evidence.range` identifies the primary source span; contextual title, headings and bounded same-block overlap do not expand that anchor. `preview.complete_evidence` describes whether the preview contains that complete primary span; it does not claim that every relevant rule or exception in the note was included. A fitting multi-topic note has note-level evidence and cannot truthfully identify a semantically winning section from its one vector. `include_content:true` returns the complete raw note instead of a snippet.

The server reads current source once and checks its revision before building a preview or full content. If the source changed or is unreadable, the ranked note remains in results with `source_status:"changed"` or `"unavailable"` and a typed `context_error`; content fields are omitted. It never applies old offsets to newer text or substitutes unrelated content. Last-known-good scores may therefore remain available while their context needs refresh. With watching disabled, external edits require an explicit source/index refresh or service restart; status inspection does not start it.

For a remote API, the identity can prove the configured endpoint and model name, but it cannot detect a provider silently replacing model weights behind the same name. Change the configured model identifier (or remove the derived cache) when a provider makes such a revision.

- Finds notes by **meaning**, not keywords — "making money from software" surfaces notes about monetization
- **Hybrid search** — `lexical_prefetch: true` selects eligible BM25 candidate notes and reranks their actual winning embedding. It cannot retrieve a note absent from lexical preselection.
- **Tunable blending** — `alpha` controls the weight between lexical and semantic scores (default 0.25)

### Regex — `search_regex`

Always available. Full regex syntax for pattern matching across all notes.

### When to Use What

| Need | Tool |
|------|------|
| Known keyword or phrase | `search_text` |
| Keyword with possible typo | `search_text` + `fuzzy: true` |
| Conceptual — "notes about X" | `search_semantic` |
| Lexical candidates with semantic reranking | `search_semantic` + `lexical_prefetch: true` |
| Structural pattern (URLs, IDs) | `search_regex` |
| By tag or metadata | `search_metadata` |

## Tool Reference

<details>
<summary><strong>All 19 tools</strong> (click to expand)</summary>

### Navigation

| Tool | Parameters | Description |
|------|-----------|-------------|
| `vault_list` | `path?`, `recursive?`, `glob?`, `format?`, `max_depth?`, `include_metadata?` | List files (`format: "list"`) or tree view (`format: "tree"`) |
| `vault_info` | — | Aggregate vault statistics and passive semantic backend status |

`vault_list` keeps its default JSON string-array response. With `include_metadata: true` in list mode, each indexed Markdown note instead includes `path`, filename-stem `title`, `tags`, `size`, and available filesystem timestamps from the existing index without loading note bodies. Directories, non-Markdown files, and unindexed or excluded notes remain path-only objects. Metadata is not supported in tree mode.

Listings and indexes expose paths that MCP clients can address. On Unix, discovery skips a filename containing a literal backslash and logs its path. A real note with the corresponding directory separators keeps its own identity.

### Note CRUD

| Tool | Parameters | Description |
|------|-----------|-------------|
| `note_read` | `path`, `expected_revision?`, `range?`, `max_bytes?` | Read full content or a revision-checked source page |
| `note_read_many` | exactly one of `paths` or `dir`; `recursive?`, `glob?`, `max_files?`, `max_bytes?` | Read a bounded batch with per-path skip reasons |
| `note_create` | `path`, `content?`, `frontmatter?` | Create a new note |
| `note_write` | `path`, `content` | Overwrite note content |
| `note_insert` | `path`, `content`, `position?` | Insert at end (`"end"`, default) or beginning (`"beginning"`) |
| `note_patch` | `path`, `operation`, `target_type`, `target`, `content` | Patch a heading section, block ref, or frontmatter field |
| `note_delete` | `path`, `confirm` | Delete a note (requires `confirm: true`) |
| `note_move` | `from`, `to` | Move or rename a note |

`note_move` checks that the source resolves to a regular file before creating destination directories and refuses to overwrite an existing destination. On macOS and Linux the source check and the atomic rename are separate kernel operations. If an external editor replaces the file with a directory between them, the move can rename that directory within the vault.

Moving a note symlink moves the link itself. A relative link can be renamed within its directory. Moving it to a different directory would change where its target is resolved, so that operation returns an error. Absolute links to notes inside the vault can move across directories.

`note_delete` can remove a symlink whose target is missing. Links to existing targets outside the vault still return a path error.

Every tool advertises an object-shaped `outputSchema` and returns matching `structuredContent`. Existing calls retain their text format: Markdown for full note reads, JSON arrays for searches and listings, and confirmation messages for writes. Extended revision/range reads emit their typed page as JSON text as well so text-only clients see revision and continuation metadata. Structured results use `content` for text, `results` for collections, and `message` for confirmations. `vault_info`, `note_inspect`, and `note_read_many` retain their domain-specific fields; `periodic` includes an `action` discriminator.

For `frontmatter`, structured results use `{ "action": "get", "frontmatter": ... }`, with `null` when frontmatter is absent, or `{ "action": "set" | "remove", "message": ... }`. Clients consuming the previous bare structured frontmatter object must read `structuredContent.frontmatter`; the JSON text response is unchanged.

Tools explicitly advertise read-only, destructive, idempotent, and open-world annotations. Mixed-action tools use conservative annotations covering every action: `frontmatter` is destructive, while `periodic` can create notes. Semantic search is open-world because it may use a configured external embedding API. These annotations describe behavior; tool filtering and vault path validation enforce access. After upgrading, refresh the connected app in ChatGPT to reload tool schemas and annotations.

`note_read_many` preserves explicit path order or uses sorted directory order; directory reads are non-recursive by default. It inspects 20 files and returns up to 65,536 content bytes by default, with hard caps of 100 files and 262,144 content bytes. `max_bytes` counts only the UTF-8 bytes in returned note content. The response includes `notes`, bounded `skipped` details (at most 100), total `skipped_count`, and `content_bytes`. Oversized notes are skipped—even when first—and can be fetched deliberately with `note_read`. The typed MCP result is emitted as both `structuredContent` and compatibility JSON text.

To expand a semantic hit, set `note_read.expected_revision` to the hit's `source_revision`, and set `note_read.range` to its `evidence.section_range` (or `evidence.range`). A range is `{ "start_byte": 100, "end_byte": 500 }` and requires the expected revision. The result contains `content`, `source_revision`, `source_bytes`, `requested_range`, `returned_range` and, if unfinished, `next_range`. Continue with the same revision and `next_range`. A mismatch is an explicit source-changed error, never an offset adjustment.

Ranged reads default to 65,536 returned content bytes and allow at most 262,144. `max_bytes` alone starts a bounded full-note read and establishes its revision; `expected_revision` alone performs a deliberate complete checked read. Original `{path}` reads still return the complete raw Markdown with structured `{content}`. Extended snapshots support raw notes up to 16 MiB. Budgets refer to content bytes, not the duplicated/escaped total MCP response size.

For other sections, call `note_inspect` with `fields:["headings"]` and the source revision. Its bounded projection returns heading/parent ordinals, display text, raw heading and section ranges, the revision and `next_heading_offset`. Initial pages can establish the revision; nonzero offsets require it. Defaults are 100 headings and 65,536 serialized projection bytes, capped at 1,000 headings and 262,144 bytes. A giant heading has an explicitly truncated display preview while retaining complete source locators. Whole-section or full-note reading remains necessary when an answer depends on surrounding rules or distant exceptions.

For `note_patch` heading targets, bare heading text such as `"Log"` is canonical. ATX marker-prefixed targets such as `"## Log"` are also accepted, so headings copied from `note_inspect` with `view: "targets"` can be used directly.

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
| `note_inspect` | `path`, `view?`, `fields?`, `expected_revision?`, `heading_offset?`, `max_headings?`, `max_bytes?` | Metadata, patchable targets, or a bounded revision-checked heading projection |
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

`open_in_obsidian` is unavailable for SFTP sources. A remote folder does not identify a local Obsidian vault, and the server does not guess a matching vault name.

</details>

## Tool Filtering

Control which tools are exposed via the `OBSIDIAN_TOOLS` environment variable or the `X-Obsidian-Tools` HTTP header. The header can only further restrict the server-wide tool set; disabled tools are hidden from listings and cannot be called.

Tool filtering is separate from [authenticated client permissions](docs/access-control.md). A client role cannot restore a globally disabled tool, and tool visibility cannot grant access to a denied path or write action. Existing profile definitions remain unchanged.

For legacy HTTP sessions, send the header with `initialize`; that filter lasts for the session. With protocol `2026-07-28`, send it on every request that needs the restriction. Stateless requests do not retain a previous request's filter. A missing header uses the server-wide tool set; invalid profile names or invalid header encoding return HTTP 400.

| Value | Effect |
|-------|--------|
| `full` (or unset) | All supported tools (19 for local vaults; `open_in_obsidian` is unavailable for SFTP) |
| `core` | 15 tools — drops `search_semantic`, `wikilinks`, `periodic`, `open_in_obsidian` |
| `read` | 10 tools — read-only (excludes all mutation tools, including `frontmatter`) |
| `minimal` | 6 tools — `vault_list`, `vault_info`, `note_read`, `note_create`, `note_write`, `search_text` |
| `tool1,tool2,...` | Allow-list — only the named tools |
| `!tool1,!tool2,...` | Deny-list — all tools except the named ones |

The `read` profile exposes frontmatter as part of the raw Markdown returned by `note_read`. For structured frontmatter `get`, `set`, and `remove` actions, use `full`, `core`, or an explicit allow-list containing `frontmatter` (for example, `OBSIDIAN_TOOLS=note_read,frontmatter`). Allowing `frontmatter` enables all three actions.

Private audience isolation is optional and disabled by default. A managed memory
reader can set both `OBSIDIAN_MEMORY_AUTHORIZER_URL` and
`OBSIDIAN_MEMORY_KEY_FILE`, with `OBSIDIAN_SEMANTIC_MODE=local`. The reader
must use a local vault and HTTP bound to loopback; the authorizer must be `http://127.0.0.1:<port>/vault/memory/authorize` (or IPv6
loopback). The key must be an absolute regular file, readable only by its owner
and optionally its group. This mode always restricts MCP to read tools.

Every MCP request must supply `Authorization: Bearer <opaque-grant>`, including
legacy session requests. The reader revalidates the grant with the local
authorizer using `X-Vault-Key` and JSON `{"version":1,"token":"<opaque-grant>"}`.
Unavailable authorization fails closed; a session ID never replaces a grant.
The authorizer must return version 1 with `tenant_id`, `star_label`,
`workspace_id`, `audience_kind` (`channel` or `dm`), `audience_id`, `purpose`
(`conversation`, `reflection` or `operator`), a path-safe `scope_id`,
`allowed_prefixes`, and numeric Unix-seconds `expires_at`.

Allowed paths are relative CODE directories ending in `/`, or exact `.md`
files, and must contain the scope ID as a path component. Conversation grants
cannot include Capture. Notes must carry matching tenant, bot, workspace and
audience frontmatter; unclassified notes are inaccessible. Listing, metadata,
graph, statistics and lexical/semantic candidate selection use the same
immutable audience view. Scoped semantic results require a matching current
source revision. The local `/health` endpoint remains available to operators
and reports `memory_scope_v1` and `memory_authorization_required`, along with
source-wide indexing readiness; it is not an audience-scoped MCP endpoint.

Private indexing admits only schema `slack-code/1` notes marked
`sensitivity: normal`, with valid origin metadata and its matching audience
directory under active Organize sections, Express Skills, or exact Distill
brief/index files. Raw Capture, review material, inferred bullets and
unclassified notes never reach the embedding provider. Direct scoped reads
also require this schema and normal sensitivity; a selected reflection grant
can read its own raw Capture without adding it to the provider corpus.

Private readers reject auto/daemon semantic ownership at startup so an
unscoped pre-existing daemon cannot ingest their source. Ordinary unscoped
clients retain all supported ownership modes.

## Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OBSIDIAN_VAULT_PATH` | Yes* | — | Local vault path or `sftp://[user@]host[:port]/absolute/remote/root` |
| `OBSIDIAN_TRANSPORT` | No | `stdio` | Transport mode: `stdio` or `http` |
| `OBSIDIAN_HTTP_PORT` | No | `37842` | HTTP listen port |
| `OBSIDIAN_HTTP_HOST` | No | `127.0.0.1` | HTTP bind address |
| `OBSIDIAN_WATCH` | No | `true` | Native filesystem watching or SFTP polling for external index updates |
| `OBSIDIAN_LOG_LEVEL` | No | `info` | `trace`, `debug`, `info`, `warn`, `error` |
| `OBSIDIAN_TANTIVY` | No | `true` | BM25 full-text index |
| `OBSIDIAN_TOOLS` | No | `full` | Tool filtering: profile name, comma-separated allow-list, or `!`-prefixed deny-list |
| `OBSIDIAN_ACCESS_FILE` | No | Unset | Explicitly enable authenticated clients with a protected external policy. Equivalent to `--access-file PATH`; absent preserves ordinary access |
| `OBSIDIAN_ACCESS_TOKEN_FILE` | For restricted stdio | Unset | Protected client credential file, equivalent to `--access-token-file PATH`. Requires an access profile; invalid for HTTP servers |
| `OBSIDIAN_MCP_DATA` | No | Source-dependent | Local machine-data root. Local vaults default to `{vault}/.obsidian-mcp`; external local-vault data uses `vaults-v2/<vault_key>/`. SFTP data uses a separate `sources-v1/<source-id>/` namespace and never writes caches into the remote folder |
| `OBSIDIAN_SFTP_TIMEOUT_SECS` | No | `30` | SFTP source-operation deadline, including queueing and eligible recovery; integer from 1 to 86400 |
| `OBSIDIAN_SFTP_POLL_INTERVAL_SECS` | No | `300` | Metadata/tree polling interval while watching is enabled; integer from 1 to 86400 |
| `OBSIDIAN_SFTP_VERIFY_INTERVAL_SECS` | No | `3600` | Full content verification interval; integer from 1 to 86400 and at least the polling interval |
| `OBSIDIAN_EXCLUDE_PATHS` | No | *(empty)* | Comma-separated glob patterns excluded from indexing, search, graph, and stats. Merged with `.obsidian-mcp/ignore`; restart after changing ignore files |
| `OBSIDIAN_EMBEDDINGS` | No | `false` | Semantic embedding search (requires `embeddings` or `embeddings-api` feature) |
| `OBSIDIAN_EMBEDDINGS_MODEL` | No | `BAAI/bge-small-en-v1.5` | HuggingFace model for embeddings |
| `OBSIDIAN_EMBEDDING_PROVIDER` | No | *(infer)* | Embedding backend: `local` (fastembed) or `api` (OpenAI-compatible) |
| `OBSIDIAN_EMBEDDING_API_KEY` | When api | Fallback: `OPENAI_API_KEY` | API authentication key |
| `OBSIDIAN_EMBEDDING_API_BASE` | No | `https://api.openai.com/v1` | Embedding API base URL (fallback: `OPENAI_BASE_URL`) |
| `OBSIDIAN_EMBEDDING_API_MODEL` | When api | Fallback: `OPENAI_MODEL` | Model name for embedding API |
| `OBSIDIAN_EMBEDDING_API_PROFILE` | For custom api | — | Strict JSON tokenizer/capacity profile for the exact endpoint/model; official OpenAI models resolve automatically |
| `OBSIDIAN_EMBEDDING_DIM` | No | *(probed)* | Override embedding dimension (skips API probe) |
| `OBSIDIAN_EMBEDDING_API_TIMEOUT_SECS` | No | `30` | Embedding API request timeout in seconds; integer from 1 to 86400 |
| `OBSIDIAN_EMBEDDING_BATCH_SIZE` | No | `32` | Maximum notes processed per background reconciliation batch; positive integer, for either embedding backend |
| `OBSIDIAN_EMBEDDING_CA_CERT` | No | — | Path to PEM CA certificate for API TLS |
| `OBSIDIAN_EMBEDDING_TLS_VERIFY` | No | `true` | Set to `false` to skip TLS verification |
| `OBSIDIAN_SEMANTIC_MODE` | No | `auto` | Semantic backend mode: `auto`, `daemon`, `local`. SFTP uses the MCP-local runtime for `auto`/`local`. Authenticated local vaults select `local` when unspecified and reject explicit other modes. This selects ownership, not the embedding provider |
| `OBSIDIAN_SEMANTIC_HOME` | No | OS data-dir default | Shared semantic runtime home for daemon bin/model/cache state |
| `OBSIDIAN_SEMANTIC_DAEMON_PATH` | No | unset | Override daemon binary path used by bootstrap |
| `OBSIDIAN_SEMANTIC_DAEMON_DOWNLOAD_URL` | No | `https://github.com/lstpsche/obsidian-mcp/releases/download/v<version>/obsidian-semanticd-<version>-<target>.<ext>` | Override daemon download URL when binary is missing |
| `OBSIDIAN_SEMANTIC_MODEL` | No | `BAAI/bge-small-en-v1.5` | Default model used by semantic daemon runtime |
| `OBSIDIAN_SEMANTIC_ENDPOINT` | No | `<semantic-home>/ipc/...` | Daemon-only endpoint override (internal/runtime use) |
| `OBSIDIAN_SEMANTIC_CONNECT_TIMEOUT_MS` | No | `2000` | Whole daemon-call timeout, including query embedding, in milliseconds; clamped to 100–86400000. Separate from the embedding API timeout |
| `OBSIDIAN_SEMANTIC_CONNECT_RETRIES` | No | `2` | Daemon retry attempts after the first failed call |
| `OBSIDIAN_SEMANTIC_RETRY_BACKOFF_MS` | No | `250` | Base retry backoff in milliseconds |
| `OBSIDIAN_SEMANTIC_PREFETCH` | No | `50` | Default lexical prefetch candidate count for hybrid daemon queries |
| `OBSIDIAN_SEMANTIC_ALPHA` | No | `0.25` | Default hybrid blend weight (`alpha * BM25 + (1-alpha) * semantic`) |
| `OBSIDIAN_HYBRID_ALPHA` | No | alias | Backward-compatible alias for `OBSIDIAN_SEMANTIC_ALPHA` |
| `FASTEMBED_CACHE_DIR` | No | `<semantic-home>/model/fastembed-cache` | Daemon-internal shared fastembed cache root |

\* Also accepted as the first CLI argument: `obsidian-mcp /path/to/vault` or `obsidian-mcp sftp://vault-server/path/to/vault`.

The machine-data key hashes the canonical native vault path. On upgrade, a legacy
directory containing operator configuration can require a decision because different
vaults could have shared its old name. Startup reports the old directory and the
new vault-specific destination before indexing. Create the destination and copy the
intended `ignore` into it. If this vault should have no machine exclusions, create
an empty `ignore` there explicitly. Repeat for each affected vault. Legacy directories
containing only disposable caches or a comment-only ignore can rebuild automatically.
Unreadable or invalid configured exclusions produce an error before indexing.

Invalid explicit embedding API timeout or batch-size settings are reported as
configuration errors. Both binaries read their own process environment; a running
shared daemon keeps its launch settings until it is restarted.

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
 │  ├─ embeddings   Model + strict cache contracts (optional)
 │  ├─ embedding_runtime  Background reconciliation + readiness
 │  └─ watcher      Filesystem events → index sync
 ├─ models.rs       Shared types
 └─ error.rs        VaultError → MCP ErrorData
 │ fs
 ▼
Vault directory (.md files + .obsidian/)
```

The vault layer is a pure Rust library with no knowledge of MCP. The tools layer is a thin adapter. This separation means the vault code is independently testable and reusable.

In HTTP mode, each legacy session or modern stateless request gets its own handler instance, but all handlers share a single `Vault` (thread-safe through shared state). One filesystem watcher, one BM25 index, and one managed embedding coordinator/store serve all connected agents.

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

For suspected vulnerabilities, follow the [security reporting policy](SECURITY.md).
