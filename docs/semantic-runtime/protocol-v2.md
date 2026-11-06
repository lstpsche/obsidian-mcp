# Semantic Daemon Protocol v2

This document defines the local JSON-RPC contract for `obsidian-semanticd`.

The v2 health and shutdown handshake remains compatible with existing upgraders. Semantic clients must also verify `semantic_source_evidence_v1` and `scoped_indexing_v1` after ownership-aware daemon reconciliation. These capabilities cover revision-bound source evidence and document indexing under an explicit exclusion policy. Older content clients must upgrade to attach a vault.

## Transport

- JSON-RPC version: `2.0`
- Encoding: UTF-8 JSON
- Scope: local machine only
- IPC transport:
  - Unix: Unix domain socket
  - Windows: named pipe
- Message framing: newline-delimited JSON objects (one request/response per line)

## Envelope

### Request

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "health",
  "params": {}
}
```

### Success response

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {}
}
```

### Error response

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32602,
    "message": "invalid params",
    "data": null
  }
}
```

## Error Codes

Standard JSON-RPC:

- `-32700` parse error
- `-32600` invalid request
- `-32601` method not found
- `-32602` invalid params
- `-32603` internal error

Daemon-specific server errors:

- `-32010` incompatible API version
- `-32011` required capability unavailable
- `-32020` daemon unavailable
- `-32030` vault warming or degraded
- `-32031` vault not attached
- `-32040` bootstrap required

## Common Types

### SemanticHit

```json
{
  "path": "Projects/rust-mcp.md",
  "title": "rust-mcp",
  "score": 0.82,
  "tags": ["rust", "mcp"],
  "snippet": "optional snippet",
  "content": "optional full note content",
  "subpath": "optional heading-or-blockid",
  "source_revision": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "source_bytes": 1000,
  "source_status": "current",
  "evidence": {
    "scope": "passage",
    "cosine_similarity": 0.82,
    "range": { "start_byte": 200, "end_byte": 600 },
    "heading_path": ["Architecture", "Ownership"],
    "section_range": { "start_byte": 180, "end_byte": 800 }
  },
  "preview": {
    "range": { "start_byte": 200, "end_byte": 400 },
    "complete_evidence": false
  }
}
```

Field rules:

- `path`: vault-relative path using `/` separators
- `title`: note title
- `score`: floating-point ranking score
- `tags`: merged tag list from note metadata
- `snippet`: optional when full content is not included
- `content`: optional full content when requested
- `subpath`: optional heading text or block ref id for deep-open hints
- `source_revision`: lowercase SHA-256 of the exact indexed raw UTF-8 note, including YAML frontmatter; source bytes are never normalized for revision hashing
- `source_bytes`: byte length of that complete raw revision
- `source_status`: `current`, `changed` or `unavailable`, after validating the single source buffer used to construct returned context
- `evidence.scope`: `note` for one fitting whole-body embedding, `passage` for an overflowing note's winning chunk
- `evidence.cosine_similarity`: the raw winning cosine; hybrid `score` can be a blended value
- `evidence.range`: exact primary source span; offsets are raw-note UTF-8 bytes with inclusive start and exclusive end. Contextual title, headings and same-block overlap do not expand this anchor
- `evidence.heading_path` and optional `section_range`: passage ancestry and enclosing section locators; whole-note evidence does not invent a winning heading
- `preview`: optional exact `snippet` span and whether it contains the complete primary evidence span; it is not a promise that every relevant rule in the note was returned

The example illustrates field shape rather than the literal contents of a fixture. `snippet` is at most 300 characters and maps exactly to `preview.range`. A matching chunk is a retrieval location; surrounding or distant context can be read through revision-checked MCP note/heading operations. `include_content:true` retains its full raw-note meaning and omits the snippet/preview.

Supported source and preparation resources are finite: raw notes up to 16 MiB, at most 4,096 chunks per note, 16 KiB contextual prefixes, and overflow-partition structures of at most 4,096 headings/65,536 Markdown blocks. The coordinator bounds owned unpublished preparation/staging to 64 MiB; provider requests and the persistent cache have independent byte/token/count bounds. Resource failure is a meaningful error/degraded status, never permission to silently omit part of a note. See [supported resource ceilings](../../README.md#semantic-limits) for exact cache, vector, split-progress and source-expansion limits. These limits do not add a search mode.

If current source differs from the scored revision, retain the ranked note and evidence with `source_status:"changed"` and `context_error:{"code":"source_changed","message":"...","current_revision":"..."}`. An unreadable/deleted source uses `source_status:"unavailable"` and its typed cause. Omit snippet, preview and content for either case. Never apply indexed offsets to newer bytes, drop the candidate silently, or manufacture empty context. Historical source text is not retained by the semantic daemon.

## Methods

## `health`

Handshake + runtime metadata.

Request params:

```json
{
  "client_name": "obsidian-mcp",
  "client_version": "1.0.1",
  "min_api_version": 2,
  "max_api_version": 2
}
```

Response result:

```json
{
  "daemon_version": "1.0.1",
  "daemon_api_version": 2,
  "pid": 12345,
  "status": "ok",
  "uptime_ms": 1234,
  "model_name": "BAAI/bge-small-en-v1.5",
  "semantic_home": "/path/to/home",
  "capabilities": ["semantic_source_evidence_v1", "scoped_indexing_v1"]
}
```

Behavior:

- Daemon validates client API range.
- If daemon API is outside range, returns `-32010`.
- `pid` identifies the responding process. Lifecycle callers verify executable identity before stopping it.
- A content client checks both required capabilities on the connection that will carry a vault policy or query. A replacement daemon can appear at an existing endpoint, and older v2 implementations can ignore extra JSON fields.
- Accepted capability observations are shared across client clones. Unsupported observations are rechecked, and explicit health probes replace the cached observation, so upgrading the daemon can repair an existing client. These probes do not attach a vault or start inference.

## `ensure_vault`

Attaches an effective indexing policy for one vault. Keep this IPC connection open for the lifetime of the attachment. Queries and status can use separate connections.

Request params:

```json
{
  "vault_root": "/abs/path/to/vault",
  "index_scope": {"version": 1, "exclude_patterns": ["Private/**"]},
  "watch": true,
  "model_name": "BAAI/bge-small-en-v1.5"
}
```

Response result:

```json
{
  "vault_id": "sha256-hex",
  "scope_id": "scope-digest",
  "attachment_id": "opaque-live-handle",
  "ready": true,
  "watch_enabled": true,
  "model_name": "BAAI/bge-small-en-v1.5"
}
```

Rules:

- `vault_root` is an absolute UTF-8 path. The Rust client rejects native paths that cannot be represented in this JSON field.
- `vault_id` derivation follows `README.md` in this directory.
- The descriptor is required. Version 1 accepts up to 4,096 glob patterns totaling at most 256 KiB of UTF-8 text. An empty array accepts the owner’s floor. The owner always adds the portable vault ignore and any daemon-operator restrictions. Invalid or unreadable policy fails before indexing.
- Clients with identical normalized effective policies share the index, progress and watcher. A private scope can index only its own permitted notes. The daemon still shares one configured model and provider across scopes.
- The returned `scope_id` identifies the effective policy and vault. Use the separate `attachment_id` for content calls. It is bound to the vault root and the holder’s lifetime. Repeating the same attachment on its existing connection reuses that holder.
- A cold scan may outlast an MCP request. The MCP runtime retains the acquisition across timed-out or cancelled calls, and later searches resume the same attachment. Closing the last runtime also closes an in-progress attachment.
- On final disconnect the daemon retires that view and stops new document dispatches. An already dispatched provider request may finish. Only a new explicit attachment can reactivate the view. A watcher runs while at least one active holder requests watching.
- Earlier unscoped caches are preserved. The first attachment to scoped storage rebuilds from eligible files.

## `vault_status`

Reads the semantic status of an already attached vault without opening it or
starting indexing or watching.

Request params:

```json
{
  "vault_root": "/abs/path/to/vault",
  "index_scope": {"version": 1, "exclude_patterns": ["Private/**"]}
}
```

Response result:

```json
{
  "phase": "warming",
  "ready": false,
  "indexed_notes": 32,
  "total_notes": 100,
  "pending_notes": 68
}
```

Send exactly one selector. `index_scope` looks up an already active matching view, while `attachment_id` inspects that live holder. The response may include `last_error`. `ready` means a compatible snapshot is
queryable; `phase` independently describes reconciliation as `warming`, `ready`,
or `degraded`. Existing vectors may still need refresh, so `indexed_notes` is not
a completed-work counter.

Counters remain note-based: `indexed_notes` counts complete committed note collections, and `pending_notes` counts distinct note paths. A note whose next revision has only some chunks computed remains pending; partial per-note collections are never published. Transient provider failures use the managed queue's backoff. Permanent input/configuration/resource failures remain visibly degraded until a new source generation or corrected owner configuration.

An unattached vault returns `-32031`; callers inspecting status must not attach it
as a recovery action. This method is additive within API v2. Older v2 daemons can
return `-32601`; clients should surface status as unsupported rather than
substitute a different runtime's progress.

## `shutdown`

Gracefully terminates the shared daemon for an ownership-verified lifecycle operation.

Request params:

```json
{}
```

Response result:

```json
{
  "accepted": true
}
```

Behavior:

- The daemon writes and flushes the success response before stopping its IPC listener.
- Callers must establish local ownership separately; this protocol method does not grant permission to replace an externally managed daemon.
- Lifecycle shutdown requires a compatible health response and verified executable identity.

## `search_semantic`

Pure semantic search.

Request params:

```json
{
  "vault_root": "/abs/path/to/vault",
  "attachment_id": "opaque-live-handle",
  "query": "memory safe programming",
  "top_k": 10,
  "include_content": false
}
```

Response result:

```json
{
  "results": []
}
```

Score semantics:

- `score` is cosine similarity in `[-1.0, 1.0]`.
- A fitting note has one vector; only a note exceeding its backend's verified input budget is partitioned automatically. No chunking/result mode parameter is accepted.
- Results group by original note, select the maximum real chunk cosine, and apply `top_k` after grouping. A long note cannot consume the result count with several passages.
- Results sort by descending score with deterministic path/source-position ties.
- No minimum threshold is applied.
- Queries exceeding the selected tokenizer's effective input budget return `-32602` with measured and supported token counts; queries are not silently truncated or averaged.

## `search_hybrid`

Lexical prefetch + semantic reranking.

Request params:

```json
{
  "vault_root": "/abs/path/to/vault",
  "attachment_id": "opaque-live-handle",
  "query": "memory safe programming",
  "top_k": 10,
  "prefetch": 50,
  "alpha": 0.25,
  "include_content": false
}
```

Response result:

```json
{
  "results": []
}
```

Score semantics:

- Combined score: `alpha * normalized_bm25 + (1 - alpha) * cosine_similarity`.
- `alpha` is clamped to `[0.0, 1.0]`.
- BM25 normalization uses min-max scaling to `[0, 1]` over prefetch candidates.
- When all BM25 scores are equal, normalized BM25 score is `1.0` for all candidates.
- Current/request-visible paths are intersected with complete stored semantic entries before lexical prefetch. Candidates without an embedding do not receive an invented zero cosine or consume the prefetch allowance.
- The semantic component is the note's maximum real chunk cosine. A note absent from lexical preselection is not recovered by hybrid search; pure semantic search remains available.

## `open_hint`

Optional helper to normalize a target for UI open actions.

Request params:

```json
{
  "vault_root": "/abs/path/to/vault",
  "attachment_id": "opaque-live-handle",
  "path": "Projects/rust-mcp.md"
}
```

Response result:

```json
{
  "path": "Projects/rust-mcp.md",
  "exists": true,
  "subpath": null
}
```

## Compatibility Matrix

| Daemon API | Supported client API range |
|---|---|
| `2` | `2..=2` |

Any incompatible combination must fail fast at `health` with `-32010`.

## Search scope and attachment recovery

`search_semantic` and `search_hybrid` accept optional `allowed_paths`, an array of exact vault-relative index paths. Omission searches the attached indexing scope. An empty array permits no results. The daemon intersects request paths with that view before semantic ranking or lexical prefetch. A request can narrow its results without changing the attached view’s document permissions.

An expired holder or unattached view returns `-32031` (`ERR_VAULT_NOT_ATTACHED`). Clients reattach with `ensure_vault` and may retry the query once. Warming or degraded contexts retain `-32030` (`ERR_VAULT_NOT_READY`) and their structured status; they must not trigger a competing local index.

Health and shutdown remain available to old v2 maintenance clients. A content client requiring scoped indexing surfaces a capability error for an older external owner. A capability, policy or version failure cannot trigger a competing local index. After observing an owner with stricter rules than the local vault, MCP also refuses to widen its indexing policy following an outage.
