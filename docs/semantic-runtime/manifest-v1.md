# Semantic Runtime Manifest v1

Manifest path: `<semantic-home>/manifest.json`

The manifest records the currently provisioned daemon runtime and IPC endpoint so all clients can reuse one local runtime.

## Schema

```json
{
  "schema_version": 1,
  "daemon_api_version": 1,
  "daemon_version": "1.0.1",
  "binary_path": "/abs/path/to/obsidian-semanticd",
  "binary_sha256": "optional sha256 hex",
  "ipc": {
    "transport": "unix_socket",
    "endpoint": "/abs/path/to/semanticd.sock"
  },
  "pid": 12345,
  "semantic_home": "/abs/path/to/semantic-home",
  "fastembed_cache_dir": "/abs/path/to/semantic-home/model/fastembed-cache",
  "model_name": "BAAI/bge-small-en-v1.5",
  "created_at": "2026-04-06T10:00:00Z",
  "updated_at": "2026-04-06T10:01:00Z",
  "last_health_check_at": "2026-04-06T10:01:05Z",
  "last_bootstrap_client": "obsidian-mcp",
  "last_bootstrap_client_version": "1.0.1"
}
```

Notes:

- On Windows, `ipc.transport` is `named_pipe` and `ipc.endpoint` is a pipe name.
- `binary_sha256` is optional when checksum is unavailable in development scenarios.
- Timestamps are RFC 3339 UTC strings.

## Field Semantics

- `schema_version`: manifest schema major version.
- `daemon_api_version`: protocol contract version exposed by daemon.
- `daemon_version`: daemon binary version.
- `binary_path`: absolute path to installed daemon binary.
- `ipc`: current daemon IPC endpoint metadata.
- `pid`: daemon process id known at bootstrap time.
- `semantic_home`: resolved runtime home.
- `fastembed_cache_dir`: shared model cache directory used via `FASTEMBED_CACHE_DIR`.
- `model_name`: default runtime embedding model.

## Locking And Atomicity

Manifest writes are serialized by `<semantic-home>/lock/install.lock`.

Required behavior:

1. Acquire lock.
2. Re-read manifest under lock.
3. Apply install/start/reuse decision.
4. Write manifest atomically (temp file + rename).
5. Release lock.

This prevents races where multiple clients attempt install/start simultaneously.

## Lifecycle

### Initial provision

- If manifest missing:
  - install daemon binary if needed
  - start daemon
  - verify `health`
  - write new manifest

### Reuse path

- If manifest exists and endpoint is healthy:
  - reuse existing daemon
  - do not reinstall
  - do not spawn duplicate

### Recovery path

- If manifest exists but endpoint unhealthy:
  - reconcile with actual binary state
  - restart daemon if possible
  - refresh manifest fields after successful health check

## Manifest Authority

The manifest is the source of truth for:

- active IPC endpoint
- active daemon binary path/version
- runtime model/cache location

Clients should prefer manifest authority when it conflicts with ad hoc local discovery, except when a fresh health probe proves the manifest stale.

## Compatibility

- Unknown future fields must be ignored by v1 readers.
- A manifest with unsupported `schema_version` must trigger bootstrap upgrade/failure handling.
