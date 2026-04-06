# Semantic Runtime (v1)

This directory defines the shared local semantic-search runtime contract used by:

- `obsidian-mcp` (Rust MCP server)
- `obsidian-semanticd` (Rust daemon)
- Obsidian semantic-search plugin client(s)

The goal is one local runtime, one model cache, and isolated per-vault semantic state.

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

- Protocol/API contract version: `1`
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
5. If healthy: reuse and exit
6. If unhealthy/missing:
   - ensure daemon binary exists
   - install/download if missing
   - start daemon
   - perform `health` handshake
   - write manifest atomically

Rules:

- Never spawn duplicate daemon when a healthy daemon already exists.
- Manifest is the authority for endpoint/binary metadata after successful handshake.
- Model download remains daemon-owned and lazy at runtime; bootstrap only ensures daemon availability.
- MCP initializes daemon runtime only when `OBSIDIAN_WATCH=true`; with watch disabled, daemon startup is skipped.

## Contract Documents

- Protocol: `docs/semantic-runtime/protocol-v1.md`
- Manifest: `docs/semantic-runtime/manifest-v1.md`
