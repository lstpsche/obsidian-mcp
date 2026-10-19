# Semantic Runtime

This directory defines the shared local semantic-search runtime contract used by:

- `obsidian-mcp` (Rust MCP server)
- `obsidian-semanticd` (Rust daemon)
- Obsidian semantic-search plugin client(s)

The goal is one local runtime, one model cache, and isolated per-vault semantic state.

The daemon shares one embedding provider/model across its vault contexts. Its
launch environment controls the API endpoint, credentials, model, request timeout
(`OBSIDIAN_EMBEDDING_API_TIMEOUT_SECS`, default 30 seconds), and reconciliation
batch size (`OBSIDIAN_EMBEDDING_BATCH_SIZE`, default 32 notes). Later clients reuse
that configuration; their environment does not reconfigure a running daemon.
Restart the daemon after changing its launcher settings. Separate semantic homes
provide separate daemon/backend profiles.

MCP `auto` and `daemon` modes keep local inference dormant while using the daemon.
In `auto`, genuine daemon unavailability may activate a configured local backend;
that fallback stays selected until MCP restarts. Warming and query timeouts do
not activate local inference. `local` mode starts the in-process backend directly.
The existing daemon-call timeout is separate from the provider's API timeout and
must accommodate slow query embedding when using a self-hosted endpoint.

Bootstrap timeouts retain a continuation for later semantic searches. Existing
manifest resolution repeats its ownership/version checks; a newly started daemon
resumes health verification and manifest publication for that exact process
without spawning again. Handler clones share the resolved client. Passive status
only observes resolved clients and never resumes bootstrap or attaches a vault.

## Runtime Topology

```text
Obsidian plugin client(s)  ----\
                                \ local JSON-RPC over IPC
obsidian-mcp client -----------> obsidian-semanticd
                                 - model lifecycle (fastembed)
                                 - per-vault indexing state
                                 - semantic/hybrid query API
                                 - watcher ownership (enabled)
```

## Shared Home Layout

Default semantic home resolution:

- `OBSIDIAN_SEMANTIC_HOME` when set
- otherwise:
  - macOS: `~/Library/Application Support/obsidian-semantic/`
  - Linux: `${XDG_DATA_HOME:-~/.local/share}/obsidian-semantic/`
  - Windows: `%APPDATA%/obsidian-semantic/`

Layout:

```text
<semantic-home>/
  manifest.json
  lock/
    install.lock
  logs/
    obsidian-semanticd.stderr.log
  bin/
    obsidian-semanticd[.exe]
  model/
    fastembed-cache/
  ipc/
    socket / named-pipe metadata
  vaults/
    <vault_id>/
      embeddings.bin
      state.json
```

## Vault Namespace Rule

`vault_id` is:

1. Canonical absolute vault root path
2. Lowercased on Windows only
3. SHA-256 hex digest of the resulting string

This guarantees stable per-vault isolation and prevents cross-vault embedding reads/writes.

## Versioning And Compatibility

- Protocol/API contract version: `2`
- Manifest schema version: `1`
- Daemon must expose `daemon_api_version` in `health`.
- Clients must send accepted API range in `health` handshake:
  - `min_api_version`
  - `max_api_version`

Compatibility policy:

- Same major API version is required.
- If daemon API is outside client-supported range, daemon returns a version mismatch error.
- Clients must not proceed with `ensure_vault` or search calls after mismatch.

## Bootstrap State Machine

Bootstrap is idempotent and shared by all clients.

1. Resolve semantic home
2. Acquire install lock (`lock/install.lock`)
3. Read manifest
4. Probe manifest endpoint via `health`
5. If healthy:
   - reuse external/override runtimes when the API is compatible
   - for locally owned runtimes, compare the running version with the installed sibling
   - activate a newer sibling after an interrupted package upgrade
   - never replace a newer running daemon with an older sibling
6. If unhealthy, missing, or locally owned and older:
   - ensure daemon binary exists
   - prefer `obsidian-semanticd` beside the running `obsidian-mcp`
   - install/download if missing
   - start daemon
   - perform `health` handshake
   - write manifest atomically

Rules:

- Never spawn duplicate daemon when a healthy daemon already exists.
- Exact package versions are required when activating a locally owned sibling; API compatibility remains sufficient for external runtimes.
- Manifest is the authority for endpoint/binary metadata after successful handshake.
- Model download remains daemon-owned and lazy at runtime; bootstrap only ensures daemon availability.
- MCP initializes daemon runtime only when `OBSIDIAN_WATCH=true`; with watch disabled, daemon startup is skipped.

## Contract Documents

- Protocol: `docs/semantic-runtime/protocol-v2.md`
- Manifest: `docs/semantic-runtime/manifest-v1.md`
