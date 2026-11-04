# Upgrading

`obsidian-mcp upgrade` updates supported Cargo installations without asking users to reconstruct Cargo features or restart known process managers manually.

## Commands

```sh
obsidian-mcp upgrade --dry-run
obsidian-mcp upgrade
```

The command runs before vault configuration is loaded. It does not require a vault path, embedding credentials, or an active Obsidian application.

`--dry-run` verifies ownership and prints the exact Cargo root, target, profile-bearing command, feature set, and supported running services. It does not invoke Cargo, stop a daemon, or mutate a service manager.

## What is preserved

The updater uses the current binary's embedded build identity and Cargo's install tracker to preserve:

- the exact non-default Cargo feature set, including `embeddings` and `embeddings-api`;
- `--no-default-features`, so future defaults cannot silently enable a capability;
- the tracked target triple and build profile;
- both installed binaries, `obsidian-mcp` and `obsidian-semanticd`;
- existing vault/client configuration, environment sources, launchd plists, systemd units, semantic home, model selection, caches, logs, and per-vault state.

It invokes Cargo with separate argv values and `--locked`. It does not use `--force`, edit Cargo's tracker, update Rust, rewrite service definitions, or copy secrets into updater state.

## Supported installations and runtimes

| Installation or runtime | Update | Automatic activation |
|---|---:|---:|
| Cargo-tracked official crates.io install | Yes | Not applicable |
| Running macOS user LaunchAgent with the installed binary as its direct program | Yes | Same plist is booted out and bootstrapped, then exact health is verified |
| Running Linux systemd user service whose `MainPID` is the installed binary | Yes | Same unit is restarted, then its new PID, executable, HTTP configuration, and exact health are verified |
| Locally owned semantic daemon | Yes | Graceful shutdown/restart with the installed sibling and exact daemon health |
| Client-owned stdio session | Binary only | Reconnect the MCP client |
| Explicit/PATH/external semantic daemon | Binary only | Left untouched |
| Inactive launchd/systemd definition | Yes | Left inactive |

Homebrew, Nix, Git/path Cargo installs, alternate registries, manually copied binaries, release archives, system services, Windows Services, containers, and third-party supervisors are not updated in this first version. Use the installation or supervisor that owns them.

An active ad-hoc HTTP process started with `obsidian-mcp serve` is detected on macOS/Linux and blocks replacement before Cargo runs. Stop it with its original lifecycle command, run the upgrade, then start it with the same arguments/environment. The updater will not guess inherited settings or persist secret-bearing environments.

## Activation rules

Package installation and process activation are separate states:

1. The current executable must be exactly `<cargo-root>/bin/obsidian-mcp` and match one unambiguous official crates.io tracker record that owns both binaries.
2. Cargo resolves the newest release with the preserved build settings.
3. Both final binaries must return matching build identities. A feature/target mismatch or version downgrade is an error.
4. If hashes did not change, a healthy runtime already reporting that version is not restarted. A supported old or unhealthy runtime is retried, which repairs an earlier partial activation.
5. Changed binaries trigger reconciliation of locally owned runtimes. A restarted daemon or HTTP service must report the exact installed version, not merely a PID or open port. An already-running daemon with a semantically newer, API-compatible version is preserved rather than downgraded.

On Windows the installed executable cannot replace itself while running. The command validates the installation, starts a uniquely isolated temporary helper, and exits so the helper can perform replacement. The helper inherits terminal output, rejects other active `obsidian-mcp.exe` processes, and prints the final result. A locally owned semantic daemon is stopped before Cargo runs so its sibling executable can be replaced, then restored even when Cargo reports no binary change. The initiating process can report only that handoff succeeded; watch the helper's final terminal message for install/activation success.

## Outcomes and recovery

- **Already up to date:** no binary changed and every supported runtime already reports that version; nothing is restarted on Unix. On Windows, the helper may temporarily restart a locally owned semantic daemon to release its executable lock.
- **Updated:** both binaries and all supported active runtimes were verified. Reconnect stdio clients to use the new executable.
- **Installation failed:** on Unix, running services were not stopped. On Windows, a locally owned semantic daemon stopped for replacement is restored when possible.
- **Partial activation:** binaries are updated, but at least one owned runtime failed to restart or verify. The command exits non-zero and names the owner. Fix the diagnostic and rerun `obsidian-mcp upgrade`; an up-to-date Cargo result still retries an old supported runtime. If the manager left the service inactive, start that same plist or unit first—an inactive definition is never started automatically.
- **Unsupported ownership:** no package mutation occurs. The message identifies the unsupported installation or active runtime without printing its environment.

The updater does not roll back installed files after activation failure because that would make Cargo's tracker disagree with disk state.

Automatic chunking uses embedding cache schema 2. Older prefix-only vectors are rebuilt from the source notes without a manual cache deletion; the first restart can therefore remain warming while it embeds the complete selected bodies. This can increase API token use, request count and cache size. Check passive `vault_info.embeddings` and then an actual semantic query separately from package/runtime health. A successful binary activation does not establish that semantic reconciliation or a particular client query has completed.

The rebuild enforces [supported source, structure, staging and cache ceilings](../README.md#semantic-limits). A note exceeding those bounds remains an explicit semantic failure; the service does not relabel a truncated old vector or partial chunk collection as complete. Resource errors and incompatible provider profiles require the stated source/configuration correction rather than repeated reinstalls or manual deletion of the vault.

Custom API deployments must supply `OBSIDIAN_EMBEDDING_API_PROFILE` in the owning process's existing environment source. The profile pins the exact endpoint/model tokenizer and capacity contract; generic compatible endpoints cannot infer that information from vector dimensions. Exact official OpenAI embedding IDs at the official endpoint have built-in profiles. The updater preserves launcher definitions and does not create a profile, copy credentials or edit environment files. See the [API profile reference](../README.md#search); restart the owning daemon after correcting its configuration.

The daemon remains on API v2 and adds `semantic_source_evidence_v1` to health capabilities. Existing v2 upgraders can still verify the newly installed daemon using their compiled handshake. New clients require the capability after any ownership-verified old-daemon reconciliation. External daemons without it are reported as unsupported and remain under their existing owner. A downgrade can rebuild an older cache projection; cross-schema vector reuse is not guaranteed.

## Manual smoke test for maintainers

Use a temporary Cargo root, vault, semantic home, service definition, and non-default port—never a personal production service.

1. Install an older fixture with a known feature set into the temporary root.
2. Start a dedicated user service and record its definition hash, PID, and `/health.version`.
3. Run the new updater against the controlled fixture/release.
4. Verify the feature/target/profile identity, unchanged definition hash, new PID, and exact HTTP version.
5. If semantic runtime was active, verify its exact version and unchanged semantic-home/cache markers.
6. Run the command again and verify an up-to-date healthy install restarts nothing.
7. Remove only the temporary service, root, vault, and semantic home.

Include the actual previously released updater binary in an isolated old-to-new acceptance run. Verify its API2 activation handshake against the new daemon, then verify the required capability with the new client. Use an oversized synthetic note, confirm note-level ranking plus passage evidence, and expand its source with `note_read` using the returned revision. A source edit before continuation must return a revision conflict. Verify cache rebuilding and custom-provider profile errors separately from the updater's package/process checks.
