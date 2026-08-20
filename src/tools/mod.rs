//! MCP tool handlers — thin wrappers that translate MCP requests into vault operations.

pub mod graph;
pub mod metadata;
pub mod navigation;
pub mod notes;
pub mod periodic;
pub mod search;
pub mod utility;

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::{Json, Parameters};
use rmcp::model::{CallToolResult, ErrorData, Implementation, ServerCapabilities, ServerInfo};
use rmcp::{ServerHandler, tool, tool_handler, tool_router};

use crate::client::semantic_daemon::SemanticDaemonClient;
use crate::config::SemanticMode;
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
    pub daemon_unavailable_reason: Option<String>,
    pub prefetch_count: usize,
    pub vault_ensured: Arc<AtomicBool>,
}

pub struct ObsidianMcp {
    vault: Vault,
    hybrid_alpha: f32,
    semantic_runtime: SemanticRuntime,
    #[allow(dead_code)]
    pub tool_router: ToolRouter<Self>,
}

#[tool_router]
impl ObsidianMcp {
    pub fn new(
        vault: Vault,
        hybrid_alpha: f32,
        semantic_runtime: SemanticRuntime,
        disabled_tools: HashSet<String>,
    ) -> Self {
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
        description = "List files and directories in the vault. Supports recursive listing, glob filtering, and tree view (format: \"tree\"). Returns a JSON array of paths (list mode) or a tree-formatted string (tree mode)."
    )]
    async fn vault_list(
        &self,
        Parameters(params): Parameters<navigation::VaultListParams>,
    ) -> Result<CallToolResult, ErrorData> {
        navigation::vault_list(&self.vault, params)
    }

    // ── Note CRUD ───────────────────────────────────────────────────

    #[tool(
        name = "note_read",
        description = "Read the full content of a note. Returns the raw markdown including frontmatter."
    )]
    async fn note_read(
        &self,
        Parameters(params): Parameters<notes::NoteReadParams>,
    ) -> Result<String, ErrorData> {
        notes::note_read(&self.vault, params).await
    }

    #[tool(
        name = "note_read_many",
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
        description = "Create a new note with optional content and YAML frontmatter. Parent directories are created automatically. Fails if the note already exists."
    )]
    async fn note_create(
        &self,
        Parameters(params): Parameters<notes::NoteCreateParams>,
    ) -> Result<String, ErrorData> {
        notes::note_create(&self.vault, params).await
    }

    #[tool(
        name = "note_write",
        description = "Overwrite a note's entire content. The note must already exist."
    )]
    async fn note_write(
        &self,
        Parameters(params): Parameters<notes::NoteWriteParams>,
    ) -> Result<String, ErrorData> {
        notes::note_write(&self.vault, params).await
    }

    #[tool(
        name = "note_insert",
        description = "Insert content into an existing note. \
            Position: \"end\" (default) appends after existing content; \
            \"beginning\" inserts after frontmatter (or at the very start if none)."
    )]
    async fn note_insert(
        &self,
        Parameters(params): Parameters<notes::NoteInsertParams>,
    ) -> Result<String, ErrorData> {
        notes::note_insert(&self.vault, params).await
    }

    #[tool(
        name = "note_patch",
        description = "Patch a specific section of a note by targeting a heading, block reference, or frontmatter field. Supports append, prepend, and replace operations. Heading targets use bare text such as \"Log\"; ATX marker-prefixed targets such as \"## Log\" are also accepted."
    )]
    async fn note_patch(
        &self,
        Parameters(params): Parameters<notes::NotePatchParams>,
    ) -> Result<String, ErrorData> {
        notes::note_patch(&self.vault, params).await
    }

    #[tool(
        name = "note_delete",
        description = "Delete a note from the vault. Requires `confirm: true` as a safety check to prevent accidental data loss."
    )]
    async fn note_delete(
        &self,
        Parameters(params): Parameters<notes::NoteDeleteParams>,
    ) -> Result<String, ErrorData> {
        notes::note_delete(&self.vault, params).await
    }

    #[tool(
        name = "note_move",
        description = "Move or rename a note. Parent directories at the destination are created automatically."
    )]
    async fn note_move(
        &self,
        Parameters(params): Parameters<notes::NoteMoveParams>,
    ) -> Result<String, ErrorData> {
        notes::note_move(&self.vault, params).await
    }

    // ── Search ──────────────────────────────────────────────────────

    #[tool(
        name = "search_text",
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
        description = "Semantic search using daemon-backed runtime (preferred) with local compatibility fallback based on OBSIDIAN_SEMANTIC_MODE. Finds conceptually related notes without requiring exact keyword matches."
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
        description = "Manage periodic notes (daily, weekly, monthly, quarterly, yearly). \
            Actions: \"get\" — read note content (params: period, date?); \
            \"create\" — create from template or custom content (params: period, date?, content?); \
            \"list\" — list recent notes newest-first (params: period, limit?)."
    )]
    async fn periodic(
        &self,
        Parameters(params): Parameters<periodic::PeriodicParams>,
    ) -> Result<String, ErrorData> {
        periodic::periodic(&self.vault, params).await
    }

    // ── Utility ─────────────────────────────────────────────────────

    #[tool(
        name = "vault_info",
        description = "Return aggregate vault statistics: total notes, files, tags, links, and vault size in bytes."
    )]
    async fn vault_info(
        &self,
        Parameters(params): Parameters<utility::VaultInfoParams>,
    ) -> Result<CallToolResult, ErrorData> {
        utility::vault_info(&self.vault, params).await
    }

    #[tool(
        name = "open_in_obsidian",
        description = "Open a note in the Obsidian desktop app via the obsidian:// URI scheme. Requires Obsidian to be installed."
    )]
    async fn open_in_obsidian(
        &self,
        Parameters(params): Parameters<utility::OpenInObsidianParams>,
    ) -> Result<CallToolResult, ErrorData> {
        utility::open_in_obsidian(&self.vault, params).await
    }
}

#[tool_handler]
impl ServerHandler for ObsidianMcp {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
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
            daemon_client: None,
            daemon_unavailable_reason: None,
            prefetch_count: 50,
            vault_ensured: Arc::new(AtomicBool::new(false)),
        }
    }

    async fn call_tool_raw(
        server: ObsidianMcp,
        name: &str,
        arguments: serde_json::Value,
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
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments
            }
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
    async fn no_disabled_tools_exposes_all() {
        let tmp = tempfile::tempdir().unwrap();
        create_test_vault(tmp.path());
        let vault = Vault::open(&test_config(tmp.path())).await.unwrap();
        let server = ObsidianMcp::new(vault, 0.25, test_runtime(), HashSet::new());

        for name in ALL_TOOL_NAMES {
            assert!(
                server.tool_router.has_route(name),
                "expected tool '{name}' to be enabled"
            );
        }
    }

    #[tokio::test]
    async fn disabled_tools_are_hidden() {
        let tmp = tempfile::tempdir().unwrap();
        create_test_vault(tmp.path());
        let vault = Vault::open(&test_config(tmp.path())).await.unwrap();

        let disabled: HashSet<String> = ["open_in_obsidian", "wikilinks", "periodic"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let server = ObsidianMcp::new(vault, 0.25, test_runtime(), disabled);

        assert!(!server.tool_router.has_route("open_in_obsidian"));
        assert!(!server.tool_router.has_route("wikilinks"));
        assert!(!server.tool_router.has_route("periodic"));

        assert!(server.tool_router.has_route("note_read"));
        assert!(server.tool_router.has_route("vault_list"));
        assert!(server.tool_router.has_route("search_text"));
    }

    #[tokio::test]
    async fn disable_all_tools_hides_everything() {
        let tmp = tempfile::tempdir().unwrap();
        create_test_vault(tmp.path());
        let vault = Vault::open(&test_config(tmp.path())).await.unwrap();

        let disabled: HashSet<String> = ALL_TOOL_NAMES.iter().map(|s| s.to_string()).collect();
        let server = ObsidianMcp::new(vault, 0.25, test_runtime(), disabled);

        for name in ALL_TOOL_NAMES {
            assert!(
                !server.tool_router.has_route(name),
                "expected tool '{name}' to be disabled"
            );
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
        assert_eq!(response["error"]["code"], -32602);
        assert!(!tmp.path().join("invalid.md").exists());
    }
}
