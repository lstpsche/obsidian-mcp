# Semantic Daemon Protocol v1

This document defines the local JSON-RPC contract for `obsidian-semanticd`.

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
- `-32020` daemon unavailable
- `-32030` vault not ready
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
  "subpath": "optional heading-or-blockid"
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

## Methods

## `health`

Handshake + runtime metadata.

Request params:

```json
{
  "client_name": "obsidian-mcp",
  "client_version": "1.0.1",
  "min_api_version": 1,
  "max_api_version": 1
}
```

Response result:

```json
{
  "daemon_version": "1.0.1",
  "daemon_api_version": 1,
  "status": "ok",
  "uptime_ms": 1234,
  "model_name": "BAAI/bge-small-en-v1.5",
  "semantic_home": "/path/to/home"
}
```

Behavior:

- Daemon validates client API range.
- If daemon API is outside range, returns `-32010`.

## `ensure_vault`

Creates or attaches daemon runtime state for a vault namespace.

Request params:

```json
{
  "vault_root": "/abs/path/to/vault",
  "watch": true,
  "model_name": "BAAI/bge-small-en-v1.5"
}
```

Response result:

```json
{
  "vault_id": "sha256-hex",
  "ready": true,
  "watch_enabled": true,
  "model_name": "BAAI/bge-small-en-v1.5"
}
```

Rules:

- `vault_root` must be absolute.
- `vault_id` derivation follows `README.md` in this directory.
- Multiple clients calling `ensure_vault` for same vault must converge to the same runtime context.

## `search_semantic`

Pure semantic search.

Request params:

```json
{
  "vault_root": "/abs/path/to/vault",
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
- Results sorted by descending score.
- No minimum threshold is applied.

## `search_hybrid`

Lexical prefetch + semantic reranking.

Request params:

```json
{
  "vault_root": "/abs/path/to/vault",
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

## `open_hint`

Optional helper to normalize a target for UI open actions.

Request params:

```json
{
  "vault_root": "/abs/path/to/vault",
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
| `1` | `1..=1` |

Any incompatible combination must fail fast at `health` with `-32010`.
