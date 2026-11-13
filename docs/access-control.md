# Authenticated clients

Client authorization is optional. Existing HTTP and stdio configurations keep their behavior when `--access-file` and `OBSIDIAN_ACCESS_FILE` are absent. Creating an access profile does not enable it, change a running server or edit an MCP client's configuration.

Use authenticated clients when assistants need different access to one local vault. Each client receives its own bearer credential and directory or exact-file permissions. Owners manage clients through the local `access` command; MCP tools cannot change permissions or issue credentials.

## Set up a client

Run this in a terminal:

```sh
obsidian-mcp access setup /path/to/vault
```

Choose a client name, reader or editor access, a shared directory, HTTP or stdio, and a client configuration format. Reader is the default. Whole-vault access requires an explicit choice. Setup previews the permissions and file locations before creating them.

Setup creates a protected profile, a token file and a configuration snippet outside the vault. For the default HTTP connection, it prints three instructions:

1. Start the server with the generated command containing `--access-file`.
2. Merge the generated snippet into the selected MCP client's configuration, then reconnect.
3. Run the generated `access doctor` command to check the endpoint.

A server already using the default port is left alone; setup chooses another available loopback port. An explicitly selected occupied port produces an error. Copy the endpoint printed by setup into any additional client configuration.

For noninteractive setup, provide the client and resources explicitly:

```sh
obsidian-mcp access setup /path/to/vault \
  --name assistant --directory Shared --role editor --client codex
```

Repeat `--directory` or `--file` for additional resources. Use `--deny-directory` or `--deny-file` to exclude a resource. `--directory ""` explicitly selects the whole vault. `--dry-run` checks the proposed setup without creating files. A client with no grants cannot read or edit notes.

The default profile is under the user's application configuration directory, in `obsidian-mcp/access/<vault-key>/access.json`. Its protected `data/` and `tokens/` directories are alongside it. macOS uses `~/Library/Application Support`, Linux uses `$XDG_CONFIG_HOME` or `~/.config`, and Windows uses `%LOCALAPPDATA%`. Setup prints the exact paths; no privileged directory or manual JSON editing is required.

Use `access init /path/to/vault` to create only a profile. An existing profile is managed with `client add`, `grant`, `deny` and the other commands below; setup does not overwrite it.

## Client configuration

The generated HTTP configuration uses the client's native header helper. The helper reads a protected token file, so the client does not depend on an environment variable exported in a terminal. The token itself is absent from the snippet.

| Configuration format | Generated mechanism | Compatibility boundary |
| --- | --- | --- |
| `--client codex` | TOML `http_headers_helper` | Codex CLI 0.166.0 local app-server supports the generated file-backed helper, discovery/catalog and note reads. GUI clients require their own verification. |
| `--client claude-code` | JSON `headersHelper` | Configuration format is generated; actual Claude Code connection behavior must be verified with the installed version. |
| Other applications | No generated adapter | Header or OAuth support must be checked for that application and connection mode. |

These formats do not establish that every release of an application supports them. Generated configuration and direct endpoint checks are separate from actual client acceptance. Reconnect the intended application and use its tools to verify its saved settings. Support in Codex CLI or Claude Code does not imply support in ChatGPT connectors, Claude Desktop/web, Cursor or another client.

This is manually provisioned bearer authentication, not OAuth login. Applications that require OAuth discovery need a separate integration. Do not run an OAuth login command expecting it to exchange this token.

To generate another snippet from an existing token file:

```sh
obsidian-mcp access config assistant \
  --access-file /absolute/private-profile/access.json \
  --token-file /absolute/private-profile/tokens/assistant.token \
  --client codex --url http://127.0.0.1:37842/mcp \
  --out /absolute/private-profile/assistant.toml
```

Replace the profile, token and endpoint with the locations printed during setup. `--out` creates a new file and does not overwrite an existing client configuration. Merge the snippet into the application's normal MCP configuration.

`access headers --token-file PATH` deliberately writes credential-bearing header JSON to stdout for the configured helper. Do not run it to inspect a credential or copy its output into logs. Other administration commands report safe IDs and file locations without printing token values.

## Restricted stdio

A client that launches the server locally can select the same policy and its own protected credential file:

```sh
obsidian-mcp --access-file /absolute/private-profile/access.json \
  --access-token-file /absolute/private-profile/tokens/assistant.token \
  /path/to/vault
```

Use `--transport stdio` with setup/config when generating a stdio snippet. Omit `--url` and `--client-token-file` for this local mode. The profile, token and vault paths belong to the machine running that server process. For a remote vault server, use the authenticated HTTP connection instead of assuming those paths exist on the client.

Restricted stdio revalidates authorization for operations. It does not isolate a launching account that can already read the raw vault or owner policy files through other tools; those operating-system permissions remain authoritative.

## Remote clients

Authenticated HTTP binds to loopback. For another machine, put HTTPS in front of the loopback backend or use a protected SSH forward. Keep the backend endpoint private and preserve the MCP and Authorization headers through the proxy. Native clients can omit Origin; browser-origin requests are rejected by default. Public `/health` reports process status, not vault contents or authorization success.

The backend also validates `Host`. Forward its loopback address and port, rather than the public hostname, while preserving Authorization and every MCP request header. For an existing nginx TLS virtual host, add a location such as:

```nginx
location = /mcp {
    proxy_pass http://127.0.0.1:37842/mcp;
    proxy_http_version 1.1;
    proxy_pass_request_headers on;
    proxy_set_header Host 127.0.0.1:37842;
    proxy_set_header Authorization $http_authorization;
    proxy_set_header Connection "";
    proxy_buffering off;
    proxy_cache off;
}
```

If setup selected another backend port, change both occurrences of `37842`. Keep Origin validation active; native clients omit Origin, and the proxy must not strip it to bypass a browser-origin rejection. Request-header forwarding and the explicit header overrides follow nginx's [proxy module directives](https://nginx.org/en/docs/http/ngx_http_proxy_module.html#proxy_set_header).

A helper runs on the client machine. Its binary and token paths must exist there; a server-side token file is not automatically available remotely. Generate the snippet with `--client-token-file` and `--binary` when those paths differ:

```sh
obsidian-mcp access config assistant \
  --access-file /absolute/private-profile/access.json \
  --token-file /absolute/private-profile/tokens/assistant.token \
  --client claude-code --url https://notes.example.com/mcp \
  --client-token-file /home/assistant/.config/obsidian/credential \
  --binary /home/assistant/.local/bin/obsidian-mcp \
  --out /absolute/private-profile/assistant.json
```

Deliver the token through an existing secure channel into the protected client-local location, and install the binary used by the helper on that machine. The remote helper needs the client token, not the owner policy store or raw vault access. Setup does not transfer secrets, install a reverse proxy, or change account permissions.

## Permissions

| Role | Operations |
| --- | --- |
| Reader | Read permitted notes and use permitted listing, metadata, graph and search operations. |
| Editor | Reader operations plus create, update, move and delete permitted files. |

Directory grants are recursive. Exact-file grants cover only that file; necessary ancestor traversal does not reveal siblings. Grants add authority, and a matching denial always wins, including against a more-specific grant. Resource paths use `/` and are relative to the vault root. `Shared/` never matches `Shared-old/`.

For example, create a whole-vault editor with one private subtree excluded:

```sh
obsidian-mcp access client add assistant \
  --access-file /absolute/private-profile/access.json \
  --directory "" --role editor --deny-directory Private
```

Create another editor for two directories:

```sh
obsidian-mcp access client add shared-helper \
  --access-file /absolute/private-profile/access.json \
  --directory Shared --directory Assistant --role editor
```

`client add` delivers the first token to a protected file and prints its location. A reader can be created with `--directory Shared --role reader`.

Inspect the effective decision before changing a rule:

```sh
obsidian-mcp access check assistant \
  --access-file /absolute/private-profile/access.json \
  --action read --path Private/preferences.md

obsidian-mcp access preview shared-helper \
  --access-file /absolute/private-profile/access.json
```

`check` evaluates policy and explains the matching rule; it does not prove a file exists or is readable on disk. `preview` lists permitted paths without printing note contents or starting embeddings. Use `--max-results` to change its bounded output limit.

The `client list` command prints safe client, rule and credential IDs. Use those IDs to remove one rule with `rules remove CLIENT RULE_ID` or revoke one credential with `token revoke CLIENT CREDENTIAL_ID`. Add a grant with `grant CLIENT --directory PATH --role reader|editor`; add a denial with `deny CLIENT --directory PATH`. `--file` selects an exact file for either command.

Permission changes support `--dry-run`. Use the revision from `--json` output with `--expected-revision NUMBER` when applying a previously reviewed change. A concurrent change produces a revision conflict instead of overwriting another owner's update. Reconnect legacy session clients after changing their permissions.

Denials conservatively match case, Unicode compatibility forms and invisible formatting variants, even when the protected path does not exist yet. Denying `Private/` therefore also denies `PRIVATE/`, including on a case-sensitive filesystem. Grant matching retains exact source spelling and never expands through this deny matching rule. `access check` identifies the responsible denial. Ambiguous Windows device/stream names and nonportable resource paths are rejected.

Dot-prefixed path components are reserved in authenticated mode, including `.obsidian`, `.git` and `.obsidian-mcp`. A whole-vault editor cannot read plugin secrets, change plugin code or access old machine caches. Source aliases and resolved targets must both be authorized. Regular files with multiple hard links are unsupported for authenticated access. Directory moves and deletion are unavailable through restricted note operations.

Permissions are independent of indexing exclusions. An allowed ignored note can still be read explicitly; a denied note cannot. Shared raw Markdown includes its frontmatter and literal links. A reference to a private note grants no access to that target, but text deliberately copied into a shared note is shared text.

The client role is also separate from `OBSIDIAN_TOOLS`. Global and request tool filters can narrow available tools; they cannot grant path or write authority. A reader may use the read actions of mixed tools such as `frontmatter get` and `periodic get/list` when those tools are enabled. Write actions remain denied. `open_in_obsidian` is unavailable to authenticated clients.

## Check the connection

Use the generated command, or select the profile, token and endpoint explicitly:

```sh
obsidian-mcp access doctor \
  --access-file /absolute/private-profile/access.json \
  --token-file /absolute/private-profile/tokens/assistant.token \
  --url http://127.0.0.1:37842/mcp
```

The default check validates the protected profile and token, verifies unauthenticated MCP refusal, and checks authenticated discovery and the tool catalog. It reads no notes and starts no semantic query. A healthy `/health` endpoint alone is insufficient.

Optional checks are explicit:

- `--path Shared/example.md` checks a bounded note read without printing its content.
- `--semantic` runs a semantic query and can initialize embeddings or send a query to the configured API provider.
- `--write-check Shared/access-check.md` creates, reads and deletes a disposable synthetic note. It requires editor access, does not overwrite an existing path, and reports a cleanup failure without claiming success.

Doctor checks an HTTP endpoint directly; it is not a stdio client test. Verify the real application separately after merging its snippet and reconnecting.

## Credentials and recovery

Credentials are generated randomly. The policy contains a digest and safe metadata, not the plaintext token. Multiple credentials can remain valid during a deliberate rotation:

```sh
obsidian-mcp access token create assistant \
  --access-file /absolute/private-profile/access.json \
  --out /absolute/private-profile/tokens/assistant-new.token
```

Update the client's token location/configuration, reconnect and verify the new connection. Then use `token list assistant` to identify and `token revoke assistant OLD_CREDENTIAL_ID` to revoke the old credential. Token creation does not revoke another credential. `--expires-at 2027-01-01T00:00:00Z` sets an optional future expiration. A lost plaintext token is replaced with a new credential; its digest cannot recover it.

Successful token issuance prunes revoked and expired credential metadata to keep the registry bounded. `token list` shows the current credential registry, not a permanent history. Every still-valid overlap credential remains valid until explicitly revoked or expired.

`client disable NAME` suspends the client without removing its rules. `client enable NAME` restores it. `client remove NAME` removes its rules and credentials. All commands require the selected owner profile.

Configured authorization fails closed when the profile is missing, malformed, unsafe or bound to another vault. Restore a valid protected profile and use `access validate --access-file PATH` to check it. Do not remove `--access-file` as an authentication recovery step: that would deliberately start ordinary access.

The profile is bound to both the canonical location and native vault directory identity. After an intentional restore or relocation, review the new binding with `access rebind --access-file PATH --vault NEW_PATH --dry-run`, then apply it and restart the server. An existing source owner never silently inherits the replacement.

A failed policy save can have an uncertain or already-published outcome if replacement succeeded before a later synchronization failed. Preserve any delivered token file and inspect `client list`, `token list` and the policy revision before retrying. Do not assume that a failed command left the previous policy active or revoke the last working credential before checking the replacement.

## Search and supported modes

Authenticated views isolate note metadata, graph resolution, counts and lexical ranking by readable policy. Compatible readable policies can share work; credentials and write roles do not create separate search corpora. The filesystem source and mutation owner remain shared so permitted clients see each other's edits.

| Connection/source | Support |
| --- | --- |
| Ordinary HTTP or stdio, local or SFTP | Existing behavior and semantic owner selection are preserved without access settings. |
| Authenticated HTTP, local vault | Per-request bearer credential; loopback backend with HTTPS proxy or protected SSH forwarding for remote use. |
| Restricted stdio, local vault | Explicit profile plus protected token-file setting; each operation revalidates authorization. |
| Authenticated SFTP | Unsupported; startup rejects it before connecting to the source. |
| Authenticated external semantic daemon | Unsupported; authenticated semantic coordination belongs to the MCP process. |
| Existing private-memory authorizer | Its separate read-scope contract remains available; it cannot be configured simultaneously with generic client authorization. |

The profile selects a protected external machine-data directory automatically. If `OBSIDIAN_MCP_DATA` is explicitly set, it must identify that same directory; conflicting overrides fail with a correction message. Authenticated mode selects MCP-owned semantic coordination when unspecified. An explicit incompatible semantic mode is rejected. Existing embedding enablement and provider/model settings are preserved: setup does not turn embeddings on. `OBSIDIAN_SEMANTIC_MODE=local` describes coordination ownership and can still use an external API embedding provider.

Semantic activation is retained by the service, so a stateless warming response or disconnected client does not abort initialization. Once activated, authorized background reconciliation continues until the client/read-policy authority is withdrawn or the process exits. Disable the client or revoke its credentials to withdraw authority; disconnect alone is not revocation. A separately activated broader client can legitimately submit its own broader eligible notes to the configured provider.

Revocation prevents new unauthorized operation, result and provider-byte release at their authorization boundaries. It cannot recall already released bytes, undo an admitted/committed write, erase earlier model context or retract a previous provider request. A response revoked while streaming can terminate rather than complete. Existing physical cache files do not restore withdrawn authority.

Semantic caches are separated by profile and effective indexing exclusions. The first semantic activation and retirement of an active corpus remove known embedding cache files for readable policies with no remaining enabled client and valid credential. Eligible caches can remain for another process using the same profile. Other profiles and exclusion namespaces are untouched. Cache files left after a crash are reconsidered at the next semantic activation; passive startup and status checks do not start cleanup or inference. Cleanup failures remain visible in owner diagnostics while cached material stays protected. Deletion does not promise secure erasure of storage media or backups, and empty directories can remain.

The OS owner, raw vault/SFTP credentials and unrelated unrestricted tools remain trusted. MCP authorization is not an operating-system sandbox or a guarantee against shared-process timing/resource side channels. When adopting it, also review any old unauthenticated endpoint, SSH forward or delegated owner-wide tool that the same assistants can still use.
