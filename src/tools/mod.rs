//! MCP tool handlers — thin wrappers that translate MCP requests into vault operations.

pub mod graph;
pub mod metadata;
pub mod navigation;
pub mod notes;
mod output;
pub mod periodic;
pub mod search;
pub mod utility;

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::tool::schema_for_output;
use rmcp::handler::server::wrapper::{Json, Parameters};
use rmcp::model::{CallToolResult, ErrorData, Implementation, ServerCapabilities, ServerConfig};
use rmcp::{ServerHandler, tool, tool_handler, tool_router};

use crate::client::semantic_daemon::{DaemonConnectPolicy, SemanticDaemonClient, VaultAttachment};
use crate::config::SemanticMode;
use crate::daemon::bootstrap::PendingBootstrap;
use crate::daemon::protocol;
use crate::error::{VaultError, VaultResult};
use crate::vault::Vault;

/// JSON Schema for a value that may be any JSON type.
///
/// `serde_json::Value` deliberately emits an unconstrained schema in Schemars,
/// so tool inputs need this explicit schema to prevent clients from guessing
/// that structured values should be JSON-encoded strings.
pub(crate) fn json_value_schema(_generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
    schemars::json_schema!({
        "type": ["array", "boolean", "null", "number", "object", "string"]
    })
}

/// JSON Schema for an optional object-valued tool input.
pub(crate) fn json_object_schema(_generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
    schemars::json_schema!({
        "type": ["object", "null"],
        "additionalProperties": true
    })
}

/// Deserialize an optional dynamic JSON field while preserving explicit null.
///
/// Serde normally maps both a missing `Option<Value>` and an explicit JSON
/// `null` to `None`. The tool handlers need to distinguish those cases because
/// null is a valid frontmatter value.
pub(crate) fn deserialize_optional_json_value<'de, D>(
    deserializer: D,
) -> Result<Option<serde_json::Value>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    <serde_json::Value as serde::Deserialize>::deserialize(deserializer).map(Some)
}

/// Deserialize an optional object-valued input and reject every other JSON type.
pub(crate) fn deserialize_optional_json_object<'de, D>(
    deserializer: D,
) -> Result<Option<serde_json::Value>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = <Option<serde_json::Value> as serde::Deserialize>::deserialize(deserializer)?;
    match value {
        Some(serde_json::Value::Object(_)) | None => Ok(value),
        Some(_) => Err(<D::Error as serde::de::Error>::custom(
            "expected a JSON object or null",
        )),
    }
}

#[derive(Clone)]
pub struct SemanticRuntime {
    pub mode: SemanticMode,
    pub daemon_client: Option<SemanticDaemonClient>,
    pub pending_daemon_bootstrap: Option<Arc<PendingDaemonBootstrap>>,
    pub daemon_unavailable_reason: Option<String>,
    pub prefetch_count: usize,
    pub attachments: Arc<RuntimeAttachments>,
    pub local_fallback_active: Arc<AtomicBool>,
    pub daemon_health_pending: Arc<AtomicBool>,
}

/// All HTTP handlers for a configured runtime share its daemon holders.
#[derive(Default)]
pub struct RuntimeAttachments {
    scopes: Mutex<AttachmentState>,
}

type AttachmentSlot = Arc<tokio::sync::Mutex<AttachmentWork>>;

#[derive(Default)]
struct AttachmentWork {
    held: Option<Arc<VaultAttachment>>,
    acquisition: Option<tokio::task::JoinHandle<VaultResult<Arc<VaultAttachment>>>>,
}

impl Drop for AttachmentWork {
    fn drop(&mut self) {
        if let Some(task) = &self.acquisition {
            task.abort();
        }
    }
}

#[derive(Default)]
struct AttachmentState {
    slots: HashMap<(PathBuf, String, bool), AttachmentSlot>,
    negotiating: usize,
    stricter_owner_seen: bool,
}

struct Negotiation {
    runtime: std::sync::Weak<RuntimeAttachments>,
    stricter_scope: bool,
}

impl Drop for Negotiation {
    fn drop(&mut self) {
        if let Some(runtime) = self.runtime.upgrade() {
            let mut state = runtime.state();
            state.stricter_owner_seen |= self.stricter_scope;
            state.negotiating -= 1;
        }
    }
}

impl RuntimeAttachments {
    fn state(&self) -> std::sync::MutexGuard<'_, AttachmentState> {
        self.scopes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    async fn handle(
        self: &Arc<Self>,
        vault: &Vault,
        client: &SemanticDaemonClient,
        local_selected: &AtomicBool,
        expired: Option<&str>,
    ) -> VaultResult<Arc<VaultAttachment>> {
        if vault.is_remote() {
            return Err(VaultError::InvalidPath(
                "SFTP vaults cannot attach to the local semantic daemon".into(),
            ));
        }
        let local_scope_id = vault.exclude().scope_id(vault.root());
        let key = (
            vault.root().to_owned(),
            local_scope_id.clone(),
            vault.watch_enabled(),
        );
        let slot = {
            let mut state = self.state();
            if local_selected.load(Ordering::Acquire) {
                return Err(VaultError::DaemonUnavailable(
                    "local fallback is already selected".into(),
                ));
            }
            Arc::clone(state.slots.entry(key).or_default())
        };
        client
            .with_deadline("ensure_vault", async {
                let mut work = slot.lock().await;
                let task = {
                    let mut state = self.state();
                    if local_selected.load(Ordering::Acquire) {
                        return Err(VaultError::DaemonUnavailable(
                            "local fallback is already selected".into(),
                        ));
                    }
                    if let Some(attachment) = &work.held
                        && expired != Some(attachment.result.attachment_id.as_str())
                    {
                        return Ok(Arc::clone(attachment));
                    }
                    work.held = None;
                    work.acquisition.get_or_insert_with(|| {
                        state.negotiating += 1;
                        drop(state);
                        let mut negotiation = Negotiation {
                            runtime: Arc::downgrade(self),
                            stricter_scope: false,
                        };
                        let root = vault.root().to_owned();
                        let watch = vault.watch_enabled();
                        let scope = protocol::IndexScope::new(vault.exclude().patterns());
                        let client = client.clone();
                        tokio::spawn(async move {
                            let attachment =
                                client.attach_vault(&root, &scope, watch, None).await?;
                            negotiation.stricter_scope =
                                attachment.result.scope_id != local_scope_id;
                            drop(negotiation);
                            Ok(Arc::new(attachment))
                        })
                    })
                };
                // A timed-out/cancelled waiter releases the mutex. The runtime
                // still owns the same acquisition task and its open socket.
                let result = task
                    .await
                    .map_err(|error| {
                        VaultError::DaemonProtocol(format!(
                            "daemon attachment task failed: {error}"
                        ))
                    })
                    .and_then(|result| result);
                work.acquisition = None;
                if let Ok(attachment) = &result {
                    work.held = Some(Arc::clone(attachment));
                }
                result
            })
            .await
    }

    fn select_local(&self, selected: &AtomicBool, error: &VaultError) -> bool {
        let mut scopes = self.state();
        if scopes.stricter_owner_seen || scopes.negotiating > 0 {
            return false;
        }
        if !selected.swap(true, Ordering::AcqRel) {
            scopes.slots.clear();
            tracing::warn!(error = %error, "semantic daemon unavailable; local fallback selected until MCP restarts");
        }
        true
    }

    #[cfg(all(test, any(unix, has_embeddings)))]
    fn is_attached(&self) -> bool {
        self.state().slots.values().any(|slot| {
            slot.try_lock().map_or(true, |work| {
                work.held.is_some() || work.acquisition.is_some()
            })
        })
    }
}

/// Shared startup continuation; passive status only observes an initialized client.
pub struct PendingDaemonBootstrap {
    bootstrap: tokio::sync::Mutex<PendingBootstrap>,
    policy: DaemonConnectPolicy,
    client: tokio::sync::OnceCell<SemanticDaemonClient>,
}

impl PendingDaemonBootstrap {
    pub fn new(bootstrap: PendingBootstrap, policy: DaemonConnectPolicy) -> Self {
        Self {
            bootstrap: tokio::sync::Mutex::new(bootstrap),
            policy,
            client: tokio::sync::OnceCell::new(),
        }
    }

    async fn resolve(&self) -> crate::error::VaultResult<&SemanticDaemonClient> {
        let client = self
            .client
            .get_or_try_init(|| async {
                let initialized = self.bootstrap.lock().await.resume().await?;
                Ok::<_, VaultError>(SemanticDaemonClient::with_health(
                    initialized.endpoint,
                    self.policy,
                    initialized.health,
                ))
            })
            .await?;
        // Bootstrap completion and semantic support have different lifetimes.
        // Keep the inspected owner so health can recover without bootstrapping again.
        client.require_semantic_capabilities().await?;
        Ok(client)
    }
}

impl SemanticRuntime {
    /// Inspect a resolved client without resuming bootstrap or starting a daemon.
    pub fn resolved_daemon_client(&self) -> Option<&SemanticDaemonClient> {
        self.daemon_client.as_ref().or_else(|| {
            self.pending_daemon_bootstrap
                .as_ref()
                .and_then(|pending| pending.client.get())
        })
    }

    /// Resume startup only for a request that intends to use semantic search.
    pub async fn resolve_daemon_client(&self) -> crate::error::VaultResult<&SemanticDaemonClient> {
        if let Some(client) = self.resolved_daemon_client() {
            return Ok(client);
        }
        if let Some(pending) = &self.pending_daemon_bootstrap {
            return pending.resolve().await;
        }
        Err(VaultError::DaemonUnavailable(
            self.daemon_unavailable_reason
                .clone()
                .unwrap_or_else(|| "semantic daemon client is not initialized".into()),
        ))
    }

    /// Retry a deferred startup handshake before using the resolved daemon.
    pub async fn ensure_daemon_health(&self) -> crate::error::VaultResult<()> {
        let client = self.resolved_daemon_client().ok_or_else(|| {
            VaultError::DaemonUnavailable("semantic daemon client is not initialized".into())
        })?;
        // The shared client either reuses finalized bootstrap health or completes
        // one deferred handshake. Capability failures retain their typed cause.
        client.require_semantic_capabilities().await?;
        self.daemon_health_pending.store(false, Ordering::Release);
        Ok(())
    }

    pub fn uses_local_backend(&self) -> bool {
        self.mode == SemanticMode::Local
            || (self.mode == SemanticMode::Auto
                && self.local_fallback_active.load(Ordering::Acquire))
    }

    /// Select local ownership for this MCP lifetime after a daemon outage.
    /// A timeout or a warming index does not establish that the daemon stopped.
    pub fn activate_local_fallback(&self, error: &VaultError) -> bool {
        let unavailable = match error {
            VaultError::DaemonUnavailable(_) => true,
            VaultError::DaemonRpc { code, .. } => matches!(
                *code,
                protocol::ERR_DAEMON_UNAVAILABLE | protocol::ERR_BOOTSTRAP_REQUIRED
            ),
            _ => false,
        };
        if self.mode != SemanticMode::Auto || !unavailable {
            return false;
        }
        self.attachments
            .select_local(&self.local_fallback_active, error)
    }
}

pub struct ObsidianMcp {
    vault: Vault,
    hybrid_alpha: f32,
    semantic_runtime: SemanticRuntime,
    pub tool_router: ToolRouter<Self>,
}

#[tool_router]
impl ObsidianMcp {
    pub fn new(
        vault: Vault,
        hybrid_alpha: f32,
        semantic_runtime: SemanticRuntime,
        mut disabled_tools: HashSet<String>,
    ) -> Self {
        if vault.is_remote() {
            disabled_tools.insert("open_in_obsidian".to_owned());
        }
        let mut tool_router = Self::tool_router();
        if !disabled_tools.is_empty() {
            tracing::info!(
                count = disabled_tools.len(),
                "disabling tools per filter config"
            );
            for name in disabled_tools {
                tool_router.disable_route(name);
            }
        }
        Self {
            tool_router,
            vault,
            hybrid_alpha,
            semantic_runtime,
        }
    }

    // ── Navigation ──────────────────────────────────────────────────

    #[tool(
        name = "vault_list",
        annotations(read_only_hint = true, destructive_hint = false, idempotent_hint = true, open_world_hint = false),
        output_schema = schema_for_output::<navigation::VaultListOutput>(),
        description = "List files and directories in the vault. Supports recursive listing, glob filtering, and tree view (format: \"tree\"). List mode returns a JSON array of paths, or objects with indexed title, tags, size, and timestamps when include_metadata is true. Tree mode returns a tree-formatted string."
    )]
    async fn vault_list(
        &self,
        Parameters(params): Parameters<navigation::VaultListParams>,
    ) -> Result<CallToolResult, ErrorData> {
        navigation::vault_list(&self.vault, params).await
    }

    // ── Note CRUD ───────────────────────────────────────────────────

    #[tool(
        name = "note_read",
        annotations(read_only_hint = true, destructive_hint = false, idempotent_hint = true, open_world_hint = false),
        output_schema = schema_for_output::<notes::NoteReadOutput>(),
        description = "Read raw note Markdown. With only path, returns the complete note. Optional expected_revision, range and max_bytes support revision-checked UTF-8 context pages."
    )]
    async fn note_read(
        &self,
        Parameters(params): Parameters<notes::NoteReadParams>,
    ) -> Result<CallToolResult, ErrorData> {
        notes::note_read(&self.vault, params).await
    }

    #[tool(
        name = "note_read_many",
        annotations(
            read_only_hint = true,
            destructive_hint = false,
            idempotent_hint = true,
            open_world_hint = false
        ),
        description = "Read multiple notes in one bounded call. Provide exactly one of `paths` or `dir`; directory reads are non-recursive by default. The server inspects at most 100 files and returns at most 262144 combined content bytes. Oversized or unprocessed notes are reported in `skipped`; use note_read for an intentionally oversized note."
    )]
    async fn note_read_many(
        &self,
        Parameters(params): Parameters<notes::NoteReadManyParams>,
    ) -> Result<Json<notes::NoteReadManyOutput>, ErrorData> {
        notes::note_read_many(&self.vault, params).await
    }

    #[tool(
        name = "note_create",
        annotations(read_only_hint = false, destructive_hint = false, idempotent_hint = false, open_world_hint = false),
        output_schema = schema_for_output::<output::MessageOutput>(),
        description = "Create a new note with optional content and YAML frontmatter. Parent directories are created automatically. Fails if the note already exists."
    )]
    async fn note_create(
        &self,
        Parameters(params): Parameters<notes::NoteCreateParams>,
    ) -> Result<CallToolResult, ErrorData> {
        output::message(notes::note_create(&self.vault, params).await?)
    }

    #[tool(
        name = "note_write",
        annotations(read_only_hint = false, destructive_hint = true, idempotent_hint = true, open_world_hint = false),
        output_schema = schema_for_output::<output::MessageOutput>(),
        description = "Overwrite a note's entire content. The note must already exist."
    )]
    async fn note_write(
        &self,
        Parameters(params): Parameters<notes::NoteWriteParams>,
    ) -> Result<CallToolResult, ErrorData> {
        output::message(notes::note_write(&self.vault, params).await?)
    }

    #[tool(
        name = "note_insert",
        annotations(read_only_hint = false, destructive_hint = false, idempotent_hint = false, open_world_hint = false),
        output_schema = schema_for_output::<output::MessageOutput>(),
        description = "Insert content into an existing note. \
            Position: \"end\" (default) appends after existing content; \
            \"beginning\" inserts after frontmatter (or at the very start if none)."
    )]
    async fn note_insert(
        &self,
        Parameters(params): Parameters<notes::NoteInsertParams>,
    ) -> Result<CallToolResult, ErrorData> {
        output::message(notes::note_insert(&self.vault, params).await?)
    }

    #[tool(
        name = "note_patch",
        annotations(read_only_hint = false, destructive_hint = true, idempotent_hint = false, open_world_hint = false),
        output_schema = schema_for_output::<output::MessageOutput>(),
        description = "Patch a specific section of a note by targeting a heading, block reference, or frontmatter field. Supports append, prepend, and replace operations. Heading targets use bare text such as \"Log\"; ATX marker-prefixed targets such as \"## Log\" are also accepted."
    )]
    async fn note_patch(
        &self,
        Parameters(params): Parameters<notes::NotePatchParams>,
    ) -> Result<CallToolResult, ErrorData> {
        output::message(notes::note_patch(&self.vault, params).await?)
    }

    #[tool(
        name = "note_delete",
        annotations(read_only_hint = false, destructive_hint = true, idempotent_hint = false, open_world_hint = false),
        output_schema = schema_for_output::<output::MessageOutput>(),
        description = "Delete a note from the vault. Requires `confirm: true` as a safety check to prevent accidental data loss."
    )]
    async fn note_delete(
        &self,
        Parameters(params): Parameters<notes::NoteDeleteParams>,
    ) -> Result<CallToolResult, ErrorData> {
        output::message(notes::note_delete(&self.vault, params).await?)
    }

    #[tool(
        name = "note_move",
        annotations(read_only_hint = false, destructive_hint = true, idempotent_hint = false, open_world_hint = false),
        output_schema = schema_for_output::<output::MessageOutput>(),
        description = "Move or rename a note. Parent directories at the destination are created automatically."
    )]
    async fn note_move(
        &self,
        Parameters(params): Parameters<notes::NoteMoveParams>,
    ) -> Result<CallToolResult, ErrorData> {
        output::message(notes::note_move(&self.vault, params).await?)
    }

    // ── Search ──────────────────────────────────────────────────────

    #[tool(
        name = "search_text",
        annotations(read_only_hint = true, destructive_hint = false, idempotent_hint = true, open_world_hint = false),
        output_schema = schema_for_output::<output::Results<Vec<crate::models::SearchResult>>>(),
        description = "BM25-ranked full-text search across all notes. Returns matching files with relevance scores and context snippets. Supports stemming (e.g. 'program' matches 'programming'), optional fuzzy matching for typo tolerance, and field-level filtering."
    )]
    async fn search_text(
        &self,
        Parameters(params): Parameters<search::SearchTextParams>,
    ) -> Result<CallToolResult, ErrorData> {
        search::search_text(&self.vault, params).await
    }

    #[tool(
        name = "search_regex",
        annotations(read_only_hint = true, destructive_hint = false, idempotent_hint = true, open_world_hint = false),
        output_schema = schema_for_output::<output::Results<Vec<crate::models::SearchResult>>>(),
        description = "Search across all notes using a regular expression pattern. Returns matching files with context snippets."
    )]
    async fn search_regex(
        &self,
        Parameters(params): Parameters<search::SearchRegexParams>,
    ) -> Result<CallToolResult, ErrorData> {
        search::search_regex(&self.vault, params).await
    }

    #[tool(
        name = "search_metadata",
        annotations(read_only_hint = true, destructive_hint = false, idempotent_hint = true, open_world_hint = false),
        output_schema = schema_for_output::<output::Results<Vec<crate::models::NoteMetadata>>>(),
        description = "Search notes by metadata. Set type=\"tag\" to find notes with a specific tag (both inline #tags and frontmatter tags), or type=\"frontmatter\" to query by frontmatter field value. For tags: provide `tag` (required) and optional `include_nested`. For frontmatter: provide `field` (required), optional `operator` (eq/contains/exists), and `value` (required for eq/contains)."
    )]
    async fn search_metadata(
        &self,
        Parameters(params): Parameters<search::SearchMetadataParams>,
    ) -> Result<CallToolResult, ErrorData> {
        search::search_metadata(&self.vault, params).await
    }

    #[tool(
        name = "search_semantic",
        annotations(read_only_hint = true, destructive_hint = false, idempotent_hint = true, open_world_hint = true),
        output_schema = schema_for_output::<output::Results<Vec<search::SemanticSearchResult>>>(),
        description = "Find conceptually related notes without requiring exact keyword matches. Results identify whole-note or passage evidence and its source_revision. A passage or preview can omit distant conditions: expand with note_read using expected_revision and the evidence range or section_range, or inspect other headings with note_inspect fields=[\"headings\"]. include_content=true returns complete raw notes when their indexed revision is current."
    )]
    async fn search_semantic(
        &self,
        Parameters(params): Parameters<search::SearchSemanticParams>,
    ) -> Result<CallToolResult, ErrorData> {
        search::search_semantic(
            &self.vault,
            params,
            self.hybrid_alpha,
            &self.semantic_runtime,
        )
        .await
    }

    // ── Metadata ────────────────────────────────────────────────────

    #[tool(
        name = "note_inspect",
        annotations(read_only_hint = true, destructive_hint = false, idempotent_hint = true, open_world_hint = false),
        output_schema = schema_for_output::<metadata::NoteInspectOutput>(),
        description = "Inspect a note. Views: \"metadata\" (default) returns tags, headings, outgoing links, block refs, backlinks count, frontmatter, and file stats. \"targets\" lists patchable headings with Markdown level markers, block refs, and frontmatter fields (use before note_patch)."
    )]
    async fn note_inspect(
        &self,
        Parameters(params): Parameters<metadata::NoteInspectParams>,
    ) -> Result<CallToolResult, ErrorData> {
        metadata::note_inspect(&self.vault, params).await
    }

    #[tool(
        name = "frontmatter",
        annotations(read_only_hint = false, destructive_hint = true, idempotent_hint = false, open_world_hint = false),
        output_schema = schema_for_output::<metadata::FrontmatterOutput>(),
        description = "Read, set, or remove frontmatter fields on a note. Actions: \"get\" returns all frontmatter as JSON (or null), \"set\" upserts a field (requires key + value), \"remove\" deletes a field (requires key)."
    )]
    async fn frontmatter(
        &self,
        Parameters(params): Parameters<metadata::FrontmatterParams>,
    ) -> Result<CallToolResult, ErrorData> {
        metadata::frontmatter(&self.vault, params).await
    }

    // ── Graph / Links ───────────────────────────────────────────────

    #[tool(
        name = "wikilinks",
        annotations(read_only_hint = true, destructive_hint = false, idempotent_hint = true, open_world_hint = false),
        output_schema = schema_for_output::<graph::WikilinksOutput>(),
        description = "Query the vault's wikilink graph. Queries: \"backlinks\" (requires path) finds notes linking TO a note, \"outgoing\" (requires path) finds links FROM a note with resolution status, \"broken\" (optional path) finds unresolved wikilinks, \"orphans\" finds disconnected notes."
    )]
    async fn wikilinks(
        &self,
        Parameters(params): Parameters<graph::WikilinksParams>,
    ) -> Result<CallToolResult, ErrorData> {
        graph::wikilinks(&self.vault, params).await
    }

    // ── Periodic Notes ──────────────────────────────────────────────

    #[tool(
        name = "periodic",
        annotations(read_only_hint = false, destructive_hint = false, idempotent_hint = false, open_world_hint = false),
        output_schema = schema_for_output::<periodic::PeriodicOutput>(),
        description = "Manage periodic notes (daily, weekly, monthly, quarterly, yearly). \
            Actions: \"get\" — read note content (params: period, date?); \
            \"create\" — create from template or custom content (params: period, date?, content?); \
            \"list\" — list recent notes newest-first (params: period, limit?)."
    )]
    async fn periodic(
        &self,
        Parameters(params): Parameters<periodic::PeriodicParams>,
    ) -> Result<CallToolResult, ErrorData> {
        periodic::periodic(&self.vault, params).await
    }

    // ── Utility ─────────────────────────────────────────────────────

    #[tool(
        name = "vault_info",
        annotations(read_only_hint = true, destructive_hint = false, idempotent_hint = true, open_world_hint = false),
        output_schema = schema_for_output::<utility::VaultInfoOutput>(),
        description = "Return aggregate vault statistics and passive semantic backend status. Inspecting status does not start indexing or attach a daemon vault."
    )]
    async fn vault_info(
        &self,
        Parameters(params): Parameters<utility::VaultInfoParams>,
    ) -> Result<CallToolResult, ErrorData> {
        utility::vault_info(&self.vault, &self.semantic_runtime, params).await
    }

    #[tool(
        name = "open_in_obsidian",
        annotations(read_only_hint = false, destructive_hint = false, idempotent_hint = false, open_world_hint = false),
        output_schema = schema_for_output::<output::MessageOutput>(),
        description = "Open a note in the Obsidian desktop app via the obsidian:// URI scheme. Requires Obsidian to be installed."
    )]
    async fn open_in_obsidian(
        &self,
        Parameters(params): Parameters<utility::OpenInObsidianParams>,
    ) -> Result<CallToolResult, ErrorData> {
        utility::open_in_obsidian(&self.vault, params).await
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for ObsidianMcp {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new(
                env!("CARGO_PKG_NAME"),
                env!("CARGO_PKG_VERSION"),
            ))
            .with_instructions(
                "Obsidian vault MCP server. Provides tools to read, write, search, \
                 and navigate your Obsidian notes via direct filesystem access.",
            )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ALL_TOOL_NAMES;
    use crate::test_helpers::{create_test_vault, test_config};
    use crate::vault::Vault;
    use rmcp::ServiceExt;
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

    fn test_runtime() -> SemanticRuntime {
        SemanticRuntime {
            mode: SemanticMode::Local,
            pending_daemon_bootstrap: None,
            daemon_client: None,
            daemon_unavailable_reason: None,
            prefetch_count: 50,
            attachments: Default::default(),
            local_fallback_active: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            daemon_health_pending: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    #[test]
    fn local_fallback_selection_is_shared_and_survives_later_requests() {
        let mut runtime = test_runtime();
        runtime.mode = SemanticMode::Auto;
        let next_handler = runtime.clone();
        assert!(!next_handler.uses_local_backend());
        assert!(runtime.activate_local_fallback(&VaultError::DaemonUnavailable("offline".into())));
        assert!(next_handler.uses_local_backend());
        assert!(
            !next_handler.activate_local_fallback(&VaultError::DaemonTimeout {
                operation: "query".into(),
                timeout_ms: 100,
            })
        );
        assert!(runtime.uses_local_backend());
    }

    #[test]
    fn semantic_policy_failures_and_timeouts_keep_daemon_ownership() {
        let mut runtime = test_runtime();
        runtime.mode = SemanticMode::Auto;
        for error in [
            VaultError::DaemonTimeout {
                operation: "health".into(),
                timeout_ms: 100,
            },
            VaultError::DaemonRpc {
                code: protocol::ERR_VAULT_NOT_READY,
                message: "warming".into(),
                data: None,
            },
            VaultError::DaemonRpc {
                code: protocol::ERR_INCOMPATIBLE_API_VERSION,
                message: "incompatible owner".into(),
                data: None,
            },
            VaultError::DaemonRpc {
                code: protocol::ERR_INVALID_PARAMS,
                message: "invalid exclusion rule".into(),
                data: None,
            },
            VaultError::DaemonBootstrap("invalid owner configuration".into()),
            VaultError::DaemonIpc("owner closed the connection".into()),
        ] {
            assert!(!runtime.activate_local_fallback(&error));
            assert!(!runtime.uses_local_backend());
        }
        runtime.mode = SemanticMode::Daemon;
        assert!(!runtime.activate_local_fallback(&VaultError::DaemonUnavailable("offline".into())));
        assert!(!runtime.uses_local_backend());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn concurrent_handlers_share_initial_and_replacement_attachments() {
        use crate::daemon::server::IpcEndpoint;
        use serde_json::json;

        let directory = tempfile::tempdir().unwrap();
        let socket = directory.path().join("daemon.sock");
        let listener = tokio::net::UnixListener::bind(&socket).unwrap();
        let mut config = test_config(directory.path());
        config.exclude_patterns = vec!["Private/".into()];
        let vault = Vault::open(&config).await.unwrap();
        let expected_scope = vault.exclude().scope_id(vault.root());
        let server = tokio::spawn(async move {
            let mut holders = Vec::new();
            for id in ["rejected", "original", "replacement"] {
                let (stream, _) = listener.accept().await.unwrap();
                let (reader, mut writer) = stream.into_split();
                let mut reader = BufReader::new(reader);
                let request =
                    crate::test_daemon_rpc::scoped_request(&mut reader, &mut writer).await;
                assert_eq!(request["method"], "ensure_vault");
                assert_eq!(
                    request["params"]["index_scope"],
                    json!({"version":1, "exclude_patterns":["Private/**"]})
                );
                assert_eq!(request["params"]["watch"], false);
                if id == "rejected" {
                    let error = json!({"jsonrpc":"2.0","id":request["id"],"error":{
                        "code":protocol::ERR_INVALID_PARAMS,"message":"operator policy requires repair"
                    }});
                    writer
                        .write_all(format!("{error}\n").as_bytes())
                        .await
                        .unwrap();
                    continue;
                }
                let response = json!({"jsonrpc":"2.0", "id":request["id"], "result":{
                    "vault_id":"vault", "scope_id":expected_scope, "attachment_id":id,
                    "watch_enabled":false, "ready":true, "model_name":"model"
                }});
                writer
                    .write_all(format!("{response}\n").as_bytes())
                    .await
                    .unwrap();
                holders.push((reader, writer));
            }
        });
        let client = SemanticDaemonClient::new(IpcEndpoint::UnixSocket(socket), Default::default());
        let mut runtime = test_runtime();
        runtime.mode = SemanticMode::Auto;
        let clone = runtime.clone();
        let error = runtime
            .attachments
            .handle(&vault, &client, &runtime.local_fallback_active, None)
            .await
            .unwrap_err();
        assert!(matches!(
            &error,
            VaultError::DaemonRpc {
                code: protocol::ERR_INVALID_PARAMS,
                ..
            }
        ));
        assert!(!runtime.activate_local_fallback(&error));
        let acquire = async |runtime: &SemanticRuntime, expired: Option<&str>| {
            runtime
                .attachments
                .handle(&vault, &client, &runtime.local_fallback_active, expired)
                .await
                .unwrap()
        };
        let (first, second) = tokio::join!(acquire(&runtime, None), acquire(&clone, None));
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(first.result.attachment_id, "original");
        let expired = Some(first.result.attachment_id.as_str());
        let (first, second) = tokio::join!(acquire(&runtime, expired), acquire(&clone, expired));
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(first.result.attachment_id, "replacement");
        assert!(Arc::ptr_eq(&first, &acquire(&clone, expired).await));
        server.await.unwrap();
        assert!(runtime.activate_local_fallback(&VaultError::DaemonUnavailable("offline".into())));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn slow_attachment_outlives_timed_out_and_cancelled_waiters_until_runtime_drop() {
        use crate::daemon::server::IpcEndpoint;
        use serde_json::json;
        use std::time::Duration;
        use tokio::sync::oneshot;

        for finish in [false, true] {
            let directory = tempfile::tempdir().unwrap();
            let vault = Vault::open(&test_config(directory.path())).await.unwrap();
            let socket = directory.path().join("slow-daemon.sock");
            let listener = tokio::net::UnixListener::bind(&socket).unwrap();
            let (seen, accepted) = oneshot::channel();
            let (release, gate) = oneshot::channel();
            let peer = tokio::spawn(async move {
                let (stream, _) = listener.accept().await.unwrap();
                let (reader, mut writer) = stream.into_split();
                let mut reader = BufReader::new(reader);
                let request =
                    crate::test_daemon_rpc::scoped_request(&mut reader, &mut writer).await;
                assert_eq!(request["method"], "ensure_vault");
                seen.send(()).unwrap();
                let mut line = String::new();
                tokio::select! {
                    closed = reader.read_line(&mut line) => {
                        assert!(!finish, "the sole holder disconnected during its cold scan");
                        assert_eq!(closed.unwrap(), 0);
                    }
                    completed = gate => {
                        completed.unwrap();
                        let reply = json!({"jsonrpc":"2.0","id":request["id"],"result":{
                            "scope_id":"stricter","attachment_id":"retained","vault_id":"vault",
                            "ready":true,"watch_enabled":false,"model_name":"model"
                        }});
                        writer.write_all(format!("{reply}\n").as_bytes()).await.unwrap();
                        assert_eq!(reader.read_line(&mut line).await.unwrap(), 0);
                    }
                }
            });
            let mut runtime = test_runtime();
            runtime.mode = SemanticMode::Auto;
            let clone = runtime.clone();
            let client = SemanticDaemonClient::new(
                IpcEndpoint::UnixSocket(socket),
                DaemonConnectPolicy {
                    timeout: Duration::from_millis(300),
                    retries: 0,
                    retry_backoff: Duration::ZERO,
                },
            );
            let acquire = async |runtime: &SemanticRuntime| {
                runtime
                    .attachments
                    .handle(&vault, &client, &runtime.local_fallback_active, None)
                    .await
            };
            let mut cancelled = Box::pin(acquire(&runtime));
            tokio::select! {
                result = &mut cancelled => panic!("attachment unexpectedly finished: {result:?}"),
                result = accepted => result.unwrap(),
            }
            drop(cancelled);
            drop(runtime);
            for _ in 0..2 {
                assert!(matches!(
                    acquire(&clone).await,
                    Err(VaultError::DaemonTimeout { .. })
                ));
                assert!(
                    !clone
                        .activate_local_fallback(&VaultError::DaemonUnavailable("offline".into()))
                );
            }
            if finish {
                release.send(()).unwrap();
                let attachment = acquire(&clone).await.unwrap();
                assert_eq!(attachment.result.attachment_id, "retained");
                assert!(
                    !clone
                        .activate_local_fallback(&VaultError::DaemonUnavailable("offline".into()))
                );
            }
            drop(clone);
            tokio::time::timeout(Duration::from_secs(3), peer)
                .await
                .expect("dropping the last runtime must withdraw its grant")
                .unwrap();
        }
    }

    async fn call_tool_raw(
        server: ObsidianMcp,
        name: &str,
        arguments: serde_json::Value,
    ) -> serde_json::Value {
        request_raw(
            server,
            "tools/call",
            serde_json::json!({
                "name": name,
                "arguments": arguments
            }),
        )
        .await
    }

    async fn request_raw(
        server: ObsidianMcp,
        method: &str,
        params: serde_json::Value,
    ) -> serde_json::Value {
        let (server_transport, client_transport) = tokio::io::duplex(1024 * 1024);
        let server_handle = tokio::spawn(async move {
            server
                .serve(server_transport)
                .await
                .unwrap()
                .waiting()
                .await
                .unwrap();
        });
        let (client_read, mut client_write) = tokio::io::split(client_transport);
        let mut client_lines = BufReader::new(client_read).lines();

        let mut initialize = serde_json::to_vec(&serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "0.0.1"}
            }
        }))
        .unwrap();
        initialize.push(b'\n');
        client_write.write_all(&initialize).await.unwrap();
        let _initialize_response = client_lines.next_line().await.unwrap().unwrap();

        let mut initialized = serde_json::to_vec(&serde_json::json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }))
        .unwrap();
        initialized.push(b'\n');
        client_write.write_all(&initialized).await.unwrap();

        let mut call = serde_json::to_vec(&serde_json::json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": method,
            "params": params
        }))
        .unwrap();
        call.push(b'\n');
        client_write.write_all(&call).await.unwrap();

        let response =
            serde_json::from_str(&client_lines.next_line().await.unwrap().unwrap()).unwrap();
        client_write.shutdown().await.unwrap();
        drop(client_lines);
        server_handle.await.unwrap();
        response
    }

    #[tokio::test]
    async fn tool_filters_are_enforced_over_protocol() {
        let tmp = tempfile::tempdir().unwrap();
        create_test_vault(tmp.path());
        let vault = Vault::open(&test_config(tmp.path())).await.unwrap();
        for filter in [
            "full",
            "core",
            "read",
            "minimal",
            "note_read,vault_list",
            "!note_delete,!note_write",
        ] {
            let disabled = crate::config::ToolFilter::parse(filter)
                .unwrap()
                .disabled_tools();
            let server = ObsidianMcp::new(vault.clone(), 0.25, test_runtime(), disabled.clone());
            let response = request_raw(server, "tools/list", serde_json::json!({})).await;
            let advertised: HashSet<&str> = response["result"]["tools"]
                .as_array()
                .unwrap()
                .iter()
                .map(|tool| tool["name"].as_str().unwrap())
                .collect();
            let expected: HashSet<&str> = ALL_TOOL_NAMES
                .iter()
                .copied()
                .filter(|name| !disabled.contains(*name))
                .collect();
            assert_eq!(advertised, expected, "filter {filter}");
        }
        let disabled = ALL_TOOL_NAMES.iter().map(|name| name.to_string()).collect();
        let server = ObsidianMcp::new(vault, 0.25, test_runtime(), disabled);
        let response = request_raw(server, "tools/list", serde_json::json!({})).await;
        assert_eq!(response["result"]["tools"], serde_json::json!([]));
    }

    #[tokio::test]
    async fn disabled_calls_are_rejected_before_execution_or_deserialization() {
        let tmp = tempfile::tempdir().unwrap();
        create_test_vault(tmp.path());
        let path = tmp.path().join("note.md");
        std::fs::write(&path, "Keep this note").unwrap();
        let vault = Vault::open(&test_config(tmp.path())).await.unwrap();
        let disabled = crate::config::ToolFilter::parse("read")
            .unwrap()
            .disabled_tools();
        for (name, arguments) in [
            (
                "note_delete",
                serde_json::json!({"path": "note.md", "confirm": true}),
            ),
            ("note_delete", serde_json::json!({})),
            (
                "frontmatter",
                serde_json::json!({"action": "set", "path": "note.md", "key": "tag", "value": "changed"}),
            ),
            (
                "frontmatter",
                serde_json::json!({"action": "remove", "path": "note.md", "key": "tag"}),
            ),
        ] {
            let server = ObsidianMcp::new(vault.clone(), 0.25, test_runtime(), disabled.clone());
            let response = call_tool_raw(server, name, arguments).await;
            assert_eq!(response["error"]["code"], -32602);
            assert_eq!(response["error"]["message"], "tool not found");
            assert_eq!(std::fs::read_to_string(&path).unwrap(), "Keep this note");
        }
        let server = ObsidianMcp::new(vault, 0.25, test_runtime(), disabled);
        let response =
            call_tool_raw(server, "note_read", serde_json::json!({"path": "note.md"})).await;
        assert!(response.get("error").is_none(), "{response}");
        assert_eq!(response["result"]["content"][0]["text"], "Keep this note");
    }

    #[tokio::test]
    async fn frontmatter_remains_available_when_enabled() {
        let tmp = tempfile::tempdir().unwrap();
        create_test_vault(tmp.path());
        let vault = Vault::open(&test_config(tmp.path())).await.unwrap();
        for filter in ["full", "core", "frontmatter"] {
            std::fs::write(tmp.path().join("note.md"), "Keep this note").unwrap();
            let disabled = crate::config::ToolFilter::parse(filter)
                .unwrap()
                .disabled_tools();
            let server = ObsidianMcp::new(vault.clone(), 0.25, test_runtime(), disabled.clone());
            let response = call_tool_raw(
                server,
                "frontmatter",
                serde_json::json!({
                    "action": "set", "path": "note.md", "key": "tag", "value": "saved"
                }),
            )
            .await;
            assert!(response.get("error").is_none(), "{filter}: {response}");
            let server = ObsidianMcp::new(vault.clone(), 0.25, test_runtime(), disabled);
            let response = call_tool_raw(
                server,
                "frontmatter",
                serde_json::json!({
                    "action": "get", "path": "note.md"
                }),
            )
            .await;
            assert!(response.get("error").is_none(), "{filter}: {response}");
            let properties: serde_json::Value =
                serde_json::from_str(response["result"]["content"][0]["text"].as_str().unwrap())
                    .unwrap();
            assert_eq!(properties["tag"], "saved");
        }
    }

    #[tokio::test]
    async fn frontmatter_tool_inputs_publish_explicit_json_types() {
        let tmp = tempfile::tempdir().unwrap();
        create_test_vault(tmp.path());
        let vault = Vault::open(&test_config(tmp.path())).await.unwrap();
        let server = ObsidianMcp::new(vault, 0.25, test_runtime(), HashSet::new());

        let input_schema = |name: &str| {
            let tool = server
                .tool_router
                .get(name)
                .unwrap_or_else(|| panic!("missing tool '{name}'"));
            serde_json::Value::Object(tool.input_schema.as_ref().clone())
        };

        let note_create = input_schema("note_create");
        assert_eq!(
            note_create.pointer("/properties/frontmatter/type"),
            Some(&serde_json::json!(["object", "null"]))
        );
        assert_eq!(
            note_create.pointer("/properties/frontmatter/additionalProperties"),
            Some(&serde_json::json!(true))
        );

        let dynamic_types =
            serde_json::json!(["array", "boolean", "null", "number", "object", "string"]);
        for tool_name in ["frontmatter", "search_metadata"] {
            let schema = input_schema(tool_name);
            assert_eq!(
                schema.pointer("/properties/value/type"),
                Some(&dynamic_types),
                "unexpected value schema for '{tool_name}'"
            );
            assert!(
                !schema["required"]
                    .as_array()
                    .is_some_and(|required| required.contains(&serde_json::json!("value"))),
                "'value' must remain optional for '{tool_name}'"
            );
        }
    }

    #[tokio::test]
    async fn note_read_many_publishes_typed_input_and_output_schemas() {
        let tmp = tempfile::tempdir().unwrap();
        create_test_vault(tmp.path());
        let vault = Vault::open(&test_config(tmp.path())).await.unwrap();
        let server = ObsidianMcp::new(vault, 0.25, test_runtime(), HashSet::new());
        let tool = server
            .tool_router
            .get("note_read_many")
            .expect("missing note_read_many tool");

        let input_schema = serde_json::Value::Object(tool.input_schema.as_ref().clone());
        assert_eq!(
            input_schema.pointer("/properties/paths/maxItems"),
            Some(&serde_json::json!(100))
        );
        assert_eq!(
            input_schema.pointer("/properties/max_files/maximum"),
            Some(&serde_json::json!(100))
        );
        assert_eq!(
            input_schema.pointer("/properties/max_bytes/maximum"),
            Some(&serde_json::json!(262144))
        );

        let output_schema = serde_json::Value::Object(
            tool.output_schema
                .as_ref()
                .expect("note_read_many must advertise outputSchema")
                .as_ref()
                .clone(),
        );
        assert_eq!(output_schema["type"], "object");
        for field in ["notes", "skipped", "skipped_count", "content_bytes"] {
            assert!(
                output_schema["required"]
                    .as_array()
                    .is_some_and(|required| required.contains(&serde_json::json!(field))),
                "output schema must require '{field}'"
            );
        }
    }

    #[tokio::test]
    async fn note_read_many_raw_mcp_returns_matching_text_and_structured_content() {
        let tmp = tempfile::tempdir().unwrap();
        create_test_vault(tmp.path());
        std::fs::write(tmp.path().join("one.md"), "one").unwrap();
        std::fs::write(tmp.path().join("two.md"), "two").unwrap();
        let vault = Vault::open(&test_config(tmp.path())).await.unwrap();
        let server = ObsidianMcp::new(vault, 0.25, test_runtime(), HashSet::new());

        let response = call_tool_raw(
            server,
            "note_read_many",
            serde_json::json!({"paths": ["two.md", "one.md"]}),
        )
        .await;

        let text = response
            .pointer("/result/content/0/text")
            .and_then(serde_json::Value::as_str)
            .expect("missing compatibility text content");
        let text_value: serde_json::Value = serde_json::from_str(text).unwrap();
        assert_eq!(text_value, response["result"]["structuredContent"]);
        assert_eq!(
            text_value.pointer("/notes/0/path"),
            Some(&serde_json::json!("two.md"))
        );
        assert_eq!(
            text_value.pointer("/notes/1/path"),
            Some(&serde_json::json!("one.md"))
        );
        assert_eq!(text_value["skipped_count"], 0);
    }

    #[tokio::test]
    async fn note_create_rejects_stringified_frontmatter_before_writing() {
        let tmp = tempfile::tempdir().unwrap();
        create_test_vault(tmp.path());
        let vault = Vault::open(&test_config(tmp.path())).await.unwrap();
        let server = ObsidianMcp::new(vault, 0.25, test_runtime(), HashSet::new());

        let response = call_tool_raw(
            server,
            "note_create",
            serde_json::json!({
                "path": "invalid.md",
                "content": "body",
                "frontmatter": "{\"tags\":[\"rust\",\"mcp\"]}"
            }),
        )
        .await;
        assert_eq!(response["result"]["isError"], true);
        assert!(
            response["result"]["content"][0]["text"]
                .as_str()
                .unwrap()
                .contains("failed to deserialize parameters")
        );
        assert!(!tmp.path().join("invalid.md").exists());
    }
}
