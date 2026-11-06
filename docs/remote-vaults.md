# Remote vaults over SFTP

Run `obsidian-mcp` on your computer against a vault folder on an SFTP server. The remote computer needs an existing SFTP service and permission to access the folder. It does not need Obsidian, an Obsidian plugin, `obsidian-mcp`, remote shell access, or a public MCP endpoint.

The local process reads remote Markdown and configuration files, maintains search indexes, and stores embedding checkpoints locally. It does not synchronize a second editable copy of the vault. Existing synchronization between that remote folder and your Obsidian client can continue separately.

## Setup

Install `obsidian-mcp` locally and make the OpenSSH `ssh` client available to the process that launches it. Windows users may need to enable OpenSSH Client in Optional Features. No local SSH server is needed.

Configure a key or SSH agent and verify the server's host key outside MCP. For example, an entry in `~/.ssh/config` can select the host, account, key and port:

```sshconfig
Host vault-server
    HostName storage.example.com
    User alice
    Port 2222
    IdentityFile ~/.ssh/id_ed25519
```

Verify access with the same local account that will run the MCP service:

```sh
sftp vault-server
```

Check the displayed host-key fingerprint against information from the server operator before accepting it. `obsidian-mcp` uses strict host-key checking and non-interactive authentication. It cannot accept a new key or ask for a password, key passphrase or interactive MFA response through an MCP request. An already unlocked key available to the SSH agent can be used.

Use an absolute path in the server's SFTP namespace:

```sh
obsidian-mcp sftp://vault-server/home/alice/notes
```

Both MCP transports work with this source:

```sh
obsidian-mcp --http sftp://vault-server/home/alice/notes
```

The HTTP listener remains local by default. `--host` and `--port` configure that listener; set an SFTP port in the URI or SSH configuration.

A client that launches an MCP process can use:

```json
{
  "mcpServers": {
    "obsidian": {
      "command": "obsidian-mcp",
      "args": ["sftp://vault-server/home/alice/notes"]
    }
  }
}
```

`OBSIDIAN_VAULT_PATH` accepts the URI too. A positional argument takes precedence. The URI form is `sftp://[user@]host[:port]/absolute/remote/root`; an explicit username or port overrides that part of SSH configuration. Percent-encode spaces and other reserved characters in path components, for example `Work%20Notes`. The URI cannot contain a password, query string, fragment, encoded path separator or parent traversal.

OpenSSH host aliases, identities, known-hosts files and configured jump hosts remain available. The server requests only the SFTP subsystem, disables PTY allocation and inherited forwarding, and does not run commands against vault paths. The owner's SSH configuration and any configured proxy or `Match` helpers are trusted local configuration.

For process managers such as launchd or systemd, put the URI and embedding settings in the service's environment and run foreground `--http`. Shell profiles are not automatically sourced. The service account must have access to the selected key or agent and its known-hosts file.

## Search and local data

Lexical search needs no embedding model. To enable semantic search, use a build with `embeddings` or `embeddings-api` and the same provider configuration used for a local vault. For a build containing the local backend:

```json
{
  "mcpServers": {
    "obsidian": {
      "command": "obsidian-mcp",
      "args": ["sftp://vault-server/home/alice/notes"],
      "env": {
        "OBSIDIAN_EMBEDDINGS": "true",
        "OBSIDIAN_EMBEDDING_PROVIDER": "local"
      }
    }
  }
}
```

A local model may download its model files on first use. With `OBSIDIAN_EMBEDDING_PROVIDER=api`, note content is sent to the configured embedding provider. Local checkpoint storage does not change that provider contract.

For an SFTP source, semantic modes `auto` and `local` select the in-process MCP embedding coordinator. Explicit `daemon` mode is unsupported when embeddings are enabled. The shared local semantic daemon does not receive remote credentials or SFTP URIs. When embeddings are disabled, no semantic backend is required.

One HTTP service shares its source connection, indexes and embedding coordinator across handlers. Separate stdio processes are separate owners and can duplicate downloads or inference. Use a persistent HTTP service when several agents should share one remote index.

Remote machine data is separated by source and effective exclusion policy. Its source identity includes the connection and canonical remote root; a matching folder name alone cannot select another vault's checkpoint. Model/tokenizer/provider identity remains part of cache admission.

`OBSIDIAN_MCP_DATA` selects a local machine-data root. Remote state is stored under its `sources-v1/<source-id>/` namespace. Without an override, the base is the existing semantic home selected by `OBSIDIAN_SEMANTIC_HOME` or its platform default:

| Platform | Default semantic home |
|---|---|
| macOS | `~/Library/Application Support/obsidian-semantic` |
| Linux | `$XDG_DATA_HOME/obsidian-semantic`, or `~/.local/share/obsidian-semantic` |
| Windows | `%APPDATA%\obsidian-semantic` |

Choose an unsynchronized local override if the platform default is synchronized by your environment. Remote source support never uploads embedding/model caches or imports old embedding caches from the remote folder.

An existing remote `.obsidian-mcp/ignore` is read without creating or rewriting it. Its rules are merged with local machine policy and `OBSIDIAN_EXCLUDE_PATHS` before document indexing. An invalid or unreadable policy fails before document submission. Exclusions restrict indexing, search, graph and stats; explicitly addressed note operations retain their existing access contract.

Restart after changing ignore rules. A reconnect that observes a changed portable policy stops indexing rather than silently changing its permissions. A changed effective connection or canonical root also requires reopening the source.

## Freshness and availability

The first scan reads the included note contents over SFTP. MCP negotiation and HTTP process health do not wait for that full scan. Source status distinguishes initialization, availability and catalog reconciliation from semantic readiness. Unknown aggregate counts remain unknown while the initial catalog is incomplete; they are not presented as a known-empty vault.

Transient connection failures during initialization retry independently of the external-change polling interval. Source-facing requests can wake the same retained initializer within their operation deadline. Passive status and process health do not request another connection.

Direct note reads fetch current remote bytes. Revision-checked note pages and semantic context keep their exact-source contract: if a note changed after it was indexed, the result identifies the changed revision and does not apply old offsets to newer text. If the source is unavailable, stale note bodies are not substituted.

Metadata scans find ordinary creations, edits, deletions, directory renames and attachment changes. A complete content verification also runs because SFTP size and modification time alone can miss same-size edits. Only a complete scan can remove catalog entries that were not seen. An incomplete listing or connection failure cannot become a successful empty vault.

A validated reconnect starts a complete verification while retaining a previously complete catalog. Its freshness remains visible through reconciliation status and the last completed scan. An uncertain mutation withdraws catalog authority until repair completes; a changed source identity or portable policy rejects further use and requires reopening the source.

A malformed or unsupported response fails the affected operation and scan. An initialized service retains its reconciliation owner so later work can recover after the fault is resolved; it does not replay a failed mutation. Temporary memory pressure does not turn a valid note into an invalid document or remove its existing projections.

| Setting | Default | Meaning |
|---|---|---|
| `OBSIDIAN_SFTP_TIMEOUT_SECS` | `30` | Source-operation deadline in seconds, including queueing and eligible recovery |
| `OBSIDIAN_SFTP_POLL_INTERVAL_SECS` | `300` | Metadata/tree polling interval |
| `OBSIDIAN_SFTP_VERIFY_INTERVAL_SECS` | `3600` | Full content verification interval |

Values must be integers from 1 to 86400; verification cannot be more frequent than the polling interval. These intervals are targets plus actual scan duration. Disconnection and failed scans make freshness unknown. Full content verification transfers note contents even when metadata does not change; a 20 MiB vault verified every hour transfers about 480 MiB over a continuously running day before other traffic.

Metadata polling also transfers directory listings when nothing changes. Its cost grows with the number and length of file names, including attachments. A synthetic vault with 10,000 notes in 104 directories transferred about 1.30 MB of SFTP data per unchanged metadata scan. At the five-minute default, that is up to about 373 MB per continuously running day, compared with 3.73 GB at a 30-second interval, before SSH and network overhead. Actual scan duration reduces the number of scans; this example is a bandwidth measurement, not a timing guarantee. Full note-content verification adds separate traffic.

The five-minute default limits repeated listing traffic. Choose a shorter interval for faster external-change discovery or a longer one for large vaults and metered connections; direct reads and tool writes do not wait for that interval.

`OBSIDIAN_WATCH=false` disables external polling after initial indexing. Direct tool mutations still refresh local projections, and direct reads remain live. A successful reconnect or an uncertain write still triggers full catalog reconciliation. Passive status does not initiate another connection or activate indexing.

Read-only accounts can use read/search operations without remote directory creation. Select `OBSIDIAN_TOOLS=read` to expose only the read profile. That profile excludes the mixed-action `frontmatter` tool; frontmatter remains available in raw note content.

## Writes and failure outcomes

Creation and movement do not overwrite an existing destination. Replacement operations upload a hidden sibling and publish it using the negotiated OpenSSH POSIX rename extension. If safe replacement is unsupported, the operation fails before changing the destination. There is no truncate-in-place or delete-then-rename recovery path. Permission to edit file bytes does not necessarily grant the directory permissions needed for atomic replacement.

Complete Markdown/YAML validation happens before a tool write is published. Supported required ordinary permission/ownership attributes are preserved before publication; rejected preservation is an error. File replacement cannot promise to preserve arbitrary ACLs, extended attributes or hard-link relationships. Server acknowledgments and optional file `fsync` are not a guarantee of server-directory crash durability.

Each MCP service serializes its own read/transform/write/index operations. External editors, sync clients and other services do not share that serialization. Changes observed during an upload can be reported as conflicts, but the revision check and replacement are separate SFTP operations. Another writer can still change the file between them. This is not an atomic compare-and-swap or a distributed editing lock.

A lost connection after dispatch can leave a write's outcome unknown. Structured storage errors identify the operation, path, failure kind and destination outcome. A confirmed remote commit and a failed local index update are different from a failure before publication. Do not repeat an insertion, patch or another uncertain mutation merely because its response was lost; inspect the current remote note first. Automatic connection recovery does not replay mutations.

Temporary-file cleanup is limited to entries owned by that operation and the remaining operation deadline. A process crash, expired deadline or lost connection can leave a hidden sibling upload. Uploads start at `0600`; destination permissions may already have been preserved on a leftover file before publication. Errors identify a known candidate when available; a lost create acknowledgment can leave its existence and ownership uncertain. Inspect a reported path before removing an orphaned upload. Cleanup is not automatically retried after reconnect, and the server does not delete arbitrary matching files on later startups. It does not queue offline changes or synchronize a permanent editable local mirror.

## Remote path trust

Paths stay relative to the configured vault root in MCP calls. The SFTP backend validates traversal, source types and observed symlink targets, rejects static escapes, and preserves the distinction between an alias entry and its target. Remote POSIX paths are not interpreted using the local machine's Windows filename rules.

Names that cannot be represented by the public path API, including non-UTF-8 names and literal backslashes or control characters, are skipped with a diagnostic. Other representable notes in the directory remain accessible. Malformed server directory entries, such as names containing a path separator, fail the listing rather than becoming a partial successful scan.

Ordinary SSH/SFTP accounts are supported even when they can access other server folders. SFTP uses separate path-based requests; another server process can change a parent or symlink after validation but before the next operation. Client-side checks cannot reproduce the local backend's retained-directory protection against that race. The remote server and other processes that can change its namespace are part of the trust boundary. A server-restricted account can enforce a stronger root boundary without requiring an MCP installation remotely.

`open_in_obsidian` is unavailable for a remote-only source. A remote folder does not identify an installed local Obsidian vault, and matching a vault by basename could open unrelated content.

## Rust consumers

Source I/O through `Vault` is asynchronous. Await source reads, mutations, listing, source-dependent inspection and search calls; pure cached operations retain their synchronous APIs. Data-path getters return `Option<&Path>`: `mcp_data()` is absent until remote initialization resolves local derived state, and remote sources have no vault-local `mcp_home()`. Native low-level filesystem operations still use `VaultRoot` and local daemon IPC remains local-only. MCP tool parameters and normal local result meanings are preserved, while remote status can represent initialization and unknown aggregates. The public Rust source API change requires a major-version migration before publication.

## Native SSH integration checks

The ordinary Unix SFTP MCP tests run packets through the system `sftp-server` using a disposable launcher. They cover source behavior without authenticating an SSH connection. Library protocol fixtures also run on Windows.

A separate prepared test exercises real OpenSSH authentication and host-key verification against a disposable server bound to a high loopback port. It creates temporary host and client keys, known-hosts files, server/client configurations and a vault. It checks rejection of an unknown host, a changed host key and an unauthorized client key, then reads a note through the accepted SSH connection. Builds with `embeddings-api` also index the remote note through a loopback embedding provider and return revision-checked semantic content from the packaged or source-built binary. A test-only native launcher supplies the isolated client configuration; it forwards the product's arguments to the real `ssh` binary. No personal SSH files are read or changed by this connection, and the fixture does not start or reconfigure the system SSH service. On Windows it creates a uniquely named temporary SYSTEM scheduled task to run the isolated server with authentication privileges, and removes that task when the fixture is dropped.

Prepare local OpenSSH client/server tools (`ssh`, `ssh-keygen`, `sshd`) on `PATH`; Unix also recognizes the standard `/usr/bin` and `/usr/sbin` locations. On Linux, install `openssh-server` and ensure its required `/run/sshd` directory exists. macOS includes the tools. The test authenticates as the current account. Unix needs permission to run an unprivileged local SSH server. On Windows, enable OpenSSH Server and run the prepared test from an elevated terminal so it can create its temporary SYSTEM task; the test selects the native tools under `%WINDIR%\System32\OpenSSH` before other tools on `PATH`. It fails with a prerequisite diagnostic rather than silently skipping a missing tool or unusable server.

```sh
OBSIDIAN_TEST_SSH_INTEGRATION=1 cargo test --locked --no-default-features --features embeddings-api --test sftp_integration_tests -- --ignored
```

By default this test is ignored so the normal suite does not start an SSH listener. Set `OBSIDIAN_TEST_MCP_BINARY` to an absolute extracted binary path to test a packaged MCP executable. CI explicitly prepares and selects this test on native Linux, macOS and Windows runners, including every native release archive. Missing server tools, insufficient privileges or failed authentication fail that gate; library protocol checks do not stand in for it. A disposable loopback server does not establish compatibility with a particular storage provider or verify the owner's remote server.
