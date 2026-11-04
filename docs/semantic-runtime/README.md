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

Automatic chunking uses one model-owned tokenizer/capacity contract. A complete frontmatter-free note body that fits uses one vector; an overflowing note is partitioned by Markdown structure and further bounded source spans, each with its title/ancestor-heading context. Both backends validate the final input. The local runtime retains its loaded effective capacity, normally 512 tokens; API providers use verified profiles. Chunking has no mode or user-selected size. Distinct-note search results identify their scored source revision and note/passage scope, with revision-checked context expansion available through MCP.

Set `OBSIDIAN_EMBEDDING_API_PROFILE` in the daemon's launch environment for custom API endpoints/models. Its strict JSON binds the exact endpoint and outgoing model to a tokenizer and input/request limits. The only automatic API profiles are exact official OpenAI embedding IDs at `https://api.openai.com/v1`; a proxy or alias requires its own reviewed profile. A profile may select ordinary `cl100k_base` or a digest-pinned, fully initialized HuggingFace tokenizer export with explicit special-token behavior. Profiles contain no credentials. Read the [configuration examples](../../README.md#search) before changing a deployment. Client attachments cannot replace the running daemon's profile, and passive status cannot load it.

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
- New semantic clients also require the additive health capability `semantic_source_evidence_v1`. Validate it against the final daemon after ownership-aware reconciliation; old API2 activation clients remain compatible with the version handshake.
- An owned equal-version daemon missing that capability can be replaced only after the installed sibling advertises support in its additive `--__build-info` capabilities. An old sibling without support is not restarted repeatedly; newer running versions and external owners remain preserved.
- A missing capability on an external/override daemon is an unsupported semantic contract, not evidence of an outage and not permission to start competing local inference.

Embedding cache schema 2 records complete per-note collections, exact source revisions, prepared-input hashes, tokenizer fingerprints and effective capacity/preparation/partition identity. Old schema-1 prefix vectors rebuild automatically; relocation alone does not make them complete. The new cache replaces its predecessor atomically after a durable checkpoint. During the first rebuild, status is warming and API deployments can send substantially more source text than before. No vault files, provider settings, models or launcher definitions are rewritten by cache rebuilding.

Reconciliation publishes one complete note revision at a time. Changed pieces use bounded inference requests; exact unchanged inputs reuse vectors. A later request failure retains the prior complete note collection, and note-level status counters keep their original units. Inactive retry drafts are bounded so failed notes cannot prevent healthy notes/removes from progressing. Permanent profile/source/resource failures are visible and do not trigger an endless submission loop.

The coordinator admits at most 64 MiB of application-owned unpublished source, prepared/staged/output buffers, vectors and parser-output collections per runtime, including preparation and publication reservations. It may evict inactive draft progress so other ready notes and removals can proceed; the committed complete note projection remains intact. Private pulldown-cmark/YAML/tokenizer temporary arenas receive bounded inputs and structure but do not expose allocator budgeting through their APIs. Those internals and ONNX workspaces are outside the owned-allocation cap, which is not a total process-RSS limit. Semantic source and revision-bound reads support 16 MiB of raw UTF-8 source; partitioning supports 4,096 headings, 65,536 Markdown blocks, 4,096 chunks per note and 16 KiB contextual prefixes. Requests support at most 16 MiB serialized bytes as well as their provider token/item budgets. Cache storage supports 1 GiB and 1,000,000 total chunks, with bounded paths, metadata and vector dimensions. See the [complete supported ceilings](../../README.md#semantic-limits) for paging, parser-progress and cache admission details. Unsupported resources fail visibly without silently cutting source or marking a partial note complete.

Search evidence references the indexed revision. The renderer reads live source once, validates that buffer's revision, and returns a preview/full note only when it matches. A changed/unreadable source retains its ranked hit with explicit `source_status` and `context_error`, omitting mismatched context. This allows the established last-known-good score contract while distinguishing it from current content. With watching disabled, external changes require explicit refresh/restart; inspecting status does not reconcile them.

## Bootstrap State Machine

Bootstrap is idempotent and shared by all clients.

1. Resolve semantic home
2. Acquire install lock (`lock/install.lock`)
3. Read manifest
4. Probe manifest endpoint via `health`
5. If healthy:
   - reuse external/override runtimes when the API is compatible
   - for locally owned runtimes, compare the running version with the installed sibling
   - activate a newer sibling after an interrupted package upgrade, or a same-version sibling that advertises a capability missing from the owned running daemon
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
