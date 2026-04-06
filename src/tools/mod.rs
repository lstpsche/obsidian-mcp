//! MCP tool handlers — thin wrappers that translate MCP requests into vault operations.

pub mod graph;
pub mod metadata;
pub mod navigation;
pub mod notes;
pub mod periodic;
pub mod search;
pub mod utility;

use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ErrorData, Implementation, ServerCapabilities, ServerInfo};
use rmcp::{ServerHandler, tool, tool_handler, tool_router};

use crate::client::semantic_daemon::SemanticDaemonClient;
use crate::config::SemanticMode;
use crate::vault::Vault;

#[derive(Clone)]
pub struct SemanticRuntime {
    pub mode: SemanticMode,
    pub daemon_client: Option<SemanticDaemonClient>,
    pub daemon_unavailable_reason: Option<String>,
    pub prefetch_count: usize,
}

pub struct ObsidianMcp {
    vault: Vault,
    hybrid_alpha: f32,
    semantic_runtime: SemanticRuntime,
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl ObsidianMcp {
    pub fn new(vault: Vault, hybrid_alpha: f32, semantic_runtime: SemanticRuntime) -> Self {
        Self {
            tool_router: Self::tool_router(),
            vault,
            hybrid_alpha,
            semantic_runtime,
        }
    }

    // ── Navigation ──────────────────────────────────────────────────

    #[tool(
        name = "vault_list",
        description = "List files and directories in the vault. Supports recursive listing and glob filtering. Returns a JSON array of relative paths."
    )]
    async fn vault_list(
        &self,
        Parameters(params): Parameters<navigation::VaultListParams>,
    ) -> Result<CallToolResult, ErrorData> {
        navigation::vault_list(&self.vault, params)
    }

    #[tool(
        name = "vault_structure",
        description = "Get a tree view of the vault directory structure, formatted like the `tree` command. Useful for understanding vault organization."
    )]
    async fn vault_structure(
        &self,
        Parameters(params): Parameters<navigation::VaultStructureParams>,
    ) -> Result<CallToolResult, ErrorData> {
        navigation::vault_structure(&self.vault, params)
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
        name = "note_append",
        description = "Append content to the end of an existing note."
    )]
    async fn note_append(
        &self,
        Parameters(params): Parameters<notes::NoteAppendParams>,
    ) -> Result<String, ErrorData> {
        notes::note_append(&self.vault, params).await
    }

    #[tool(
        name = "note_prepend",
        description = "Insert content after the frontmatter block (or at the very start if no frontmatter exists)."
    )]
    async fn note_prepend(
        &self,
        Parameters(params): Parameters<notes::NotePrependParams>,
    ) -> Result<String, ErrorData> {
        notes::note_prepend(&self.vault, params).await
    }

    #[tool(
        name = "note_patch",
        description = "Patch a specific section of a note by targeting a heading, block reference, or frontmatter field. Supports append, prepend, and replace operations."
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
        name = "search_tag",
        description = "Find all notes with a specific tag (both inline #tags and frontmatter tags). Optionally include nested tags."
    )]
    async fn search_tag(
        &self,
        Parameters(params): Parameters<search::SearchTagParams>,
    ) -> Result<CallToolResult, ErrorData> {
        search::search_tag(&self.vault, params).await
    }

    #[tool(
        name = "search_frontmatter",
        description = "Query notes by frontmatter field. Supports exact match (eq), substring/element match (contains), and existence check (exists)."
    )]
    async fn search_frontmatter(
        &self,
        Parameters(params): Parameters<search::SearchFrontmatterParams>,
    ) -> Result<CallToolResult, ErrorData> {
        search::search_frontmatter(&self.vault, params).await
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
        name = "note_metadata",
        description = "Get rich metadata about a note: tags, headings, outgoing links, block references, backlinks count, frontmatter, and file stats."
    )]
    async fn note_metadata(
        &self,
        Parameters(params): Parameters<metadata::NoteMetadataParams>,
    ) -> Result<CallToolResult, ErrorData> {
        metadata::note_metadata(&self.vault, params).await
    }

    #[tool(
        name = "note_document_map",
        description = "List all patch targets in a note: headings (with hierarchy), block references, and frontmatter field names. Use before note_patch to discover valid targets."
    )]
    async fn note_document_map(
        &self,
        Parameters(params): Parameters<metadata::NoteDocumentMapParams>,
    ) -> Result<CallToolResult, ErrorData> {
        metadata::note_document_map(&self.vault, params).await
    }

    #[tool(
        name = "frontmatter_get",
        description = "Get a note's YAML frontmatter as a JSON object, or null if the note has no frontmatter."
    )]
    async fn frontmatter_get(
        &self,
        Parameters(params): Parameters<metadata::FrontmatterGetParams>,
    ) -> Result<CallToolResult, ErrorData> {
        metadata::frontmatter_get(&self.vault, params).await
    }

    #[tool(
        name = "frontmatter_set",
        description = "Set a single frontmatter field on a note (upsert). Creates the frontmatter block if it doesn't exist."
    )]
    async fn frontmatter_set(
        &self,
        Parameters(params): Parameters<metadata::FrontmatterSetParams>,
    ) -> Result<CallToolResult, ErrorData> {
        metadata::frontmatter_set(&self.vault, params).await
    }

    #[tool(
        name = "frontmatter_remove",
        description = "Remove a single frontmatter field from a note. No-op if the field doesn't exist."
    )]
    async fn frontmatter_remove(
        &self,
        Parameters(params): Parameters<metadata::FrontmatterRemoveParams>,
    ) -> Result<CallToolResult, ErrorData> {
        metadata::frontmatter_remove(&self.vault, params).await
    }

    // ── Graph / Links ───────────────────────────────────────────────

    #[tool(
        name = "links_backlinks",
        description = "Find all notes linking TO a given note, with the specific wikilinks used. Useful for discovering how a note is referenced."
    )]
    async fn links_backlinks(
        &self,
        Parameters(params): Parameters<graph::LinksBacklinksParams>,
    ) -> Result<CallToolResult, ErrorData> {
        graph::links_backlinks(&self.vault, params).await
    }

    #[tool(
        name = "links_outgoing",
        description = "Find all outgoing wikilinks FROM a given note, with resolution status showing whether each target exists."
    )]
    async fn links_outgoing(
        &self,
        Parameters(params): Parameters<graph::LinksOutgoingParams>,
    ) -> Result<CallToolResult, ErrorData> {
        graph::links_outgoing(&self.vault, params).await
    }

    #[tool(
        name = "links_broken",
        description = "Find all broken (unresolved) wikilinks in the vault, or optionally within a single note."
    )]
    async fn links_broken(
        &self,
        Parameters(params): Parameters<graph::LinksBrokenParams>,
    ) -> Result<CallToolResult, ErrorData> {
        graph::links_broken(&self.vault, params).await
    }

    #[tool(
        name = "links_orphans",
        description = "Find notes with no inbound and no outbound wikilinks — completely disconnected from the vault graph."
    )]
    async fn links_orphans(
        &self,
        Parameters(params): Parameters<graph::LinksOrphansParams>,
    ) -> Result<CallToolResult, ErrorData> {
        graph::links_orphans(&self.vault, params).await
    }

    // ── Periodic Notes ──────────────────────────────────────────────

    #[tool(
        name = "periodic_get",
        description = "Read the content of a periodic note (daily, weekly, monthly, quarterly, yearly) for a given date. Defaults to today."
    )]
    async fn periodic_get(
        &self,
        Parameters(params): Parameters<periodic::PeriodicGetParams>,
    ) -> Result<String, ErrorData> {
        periodic::periodic_get(&self.vault, params).await
    }

    #[tool(
        name = "periodic_create",
        description = "Create a periodic note for a given date. Uses the configured template unless custom content is provided. Defaults to today."
    )]
    async fn periodic_create(
        &self,
        Parameters(params): Parameters<periodic::PeriodicCreateParams>,
    ) -> Result<String, ErrorData> {
        periodic::periodic_create(&self.vault, params).await
    }

    #[tool(
        name = "periodic_list_recent",
        description = "List recent periodic notes sorted newest-first. Returns paths and dates for the specified period type."
    )]
    async fn periodic_list_recent(
        &self,
        Parameters(params): Parameters<periodic::PeriodicListRecentParams>,
    ) -> Result<String, ErrorData> {
        periodic::periodic_list_recent(&self.vault, params).await
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
