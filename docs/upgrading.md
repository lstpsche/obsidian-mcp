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

## Manual smoke test for maintainers

Use a temporary Cargo root, vault, semantic home, service definition, and non-default port—never a personal production service.

1. Install an older fixture with a known feature set into the temporary root.
2. Start a dedicated user service and record its definition hash, PID, and `/health.version`.
3. Run the new updater against the controlled fixture/release.
4. Verify the feature/target/profile identity, unchanged definition hash, new PID, and exact HTTP version.
5. If semantic runtime was active, verify its exact version and unchanged semantic-home/cache markers.
6. Run the command again and verify an up-to-date healthy install restarts nothing.
7. Remove only the temporary service, root, vault, and semantic home.
