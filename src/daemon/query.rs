//! Daemon query handlers: vault attach, semantic search, and hybrid search.

use std::path::Path;
use std::sync::Arc;

use serde_json::Value;

use super::protocol::{
    self, EnsureVaultParams, EnsureVaultResult, OpenHintParams, OpenHintResult, SearchHybridParams,
    SearchResult, SearchSemanticParams, SemanticHit, SemanticPhase, SemanticStatus,
    VaultStatusParams,
};
use super::vault_context::VaultContext;
use super::vault_registry::{VaultAttachment, VaultRegistry};
use crate::error::VaultError;

const DEFAULT_TOP_K: usize = 10;
const DEFAULT_PREFETCH_COUNT: usize = 50;
const DEFAULT_ALPHA: f32 = 0.25;

#[derive(Debug)]
pub struct QueryError {
    pub code: i64,
    pub message: String,
    pub data: Option<Value>,
}

impl QueryError {
    pub fn new(code: i64, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            data: None,
        }
    }
}

pub type QueryResult<T> = Result<T, QueryError>;

/// Inspect an attached vault without opening it or starting background work.
pub async fn vault_status(
    registry: &VaultRegistry,
    params: VaultStatusParams,
) -> QueryResult<SemanticStatus> {
    let path = Path::new(&params.vault_root);
    let context = match (params.attachment_id.as_deref(), params.index_scope) {
        (Some(id), None) => registry.context_for_holder(path, id),
        (None, Some(scope)) => registry.context_for_policy(path, scope).await,
        _ => {
            return Err(QueryError::new(
                protocol::ERR_INVALID_PARAMS,
                "vault_status requires exactly one of attachment_id or index_scope",
            ));
        }
    }
    .map_err(map_vault_error)?
    .ok_or_else(not_attached)?;
    Ok(semantic_status(&context))
}

pub(crate) async fn ensure_vault(
    registry: &VaultRegistry,
    params: EnsureVaultParams,
    attachment: &mut Option<VaultAttachment>,
) -> QueryResult<EnsureVaultResult> {
    let watch_enabled = params.watch.unwrap_or(true);
    let requested_model = params
        .model_name
        .as_deref()
        .unwrap_or(registry.model_name());
    let (context, new_holder) = registry
        .attach(
            Path::new(&params.vault_root),
            watch_enabled,
            requested_model,
            &params.index_scope,
            attachment.as_ref(),
        )
        .await
        .map_err(map_vault_error)?;

    if let Some(new_holder) = new_holder {
        *attachment = Some(new_holder);
    }
    let holder = attachment.as_ref().ok_or_else(not_attached)?;
    let watch_enabled = context.watch_enabled();
    let status = semantic_status(&context);
    Ok(EnsureVaultResult {
        vault_id: context.vault_id().to_string(),
        scope_id: holder.scope_id.clone(),
        attachment_id: holder.id.clone(),
        ready: status.ready,
        watch_enabled,
        model_name: context.model_name().to_string(),
        phase: Some(status.phase),
        indexed_notes: Some(status.indexed_notes),
        total_notes: Some(status.total_notes),
        pending_notes: Some(status.pending_notes),
        last_error: status.last_error,
    })
}

pub async fn search_semantic(
    registry: &VaultRegistry,
    params: SearchSemanticParams,
) -> QueryResult<SearchResult> {
    let context = require_context(registry, &params.vault_root, &params.attachment_id)?;
    require_semantic_ready(&context)?;
    let top_k = params.top_k.unwrap_or(DEFAULT_TOP_K);
    let include_content = params.include_content.unwrap_or(false);

    let result = tokio::task::spawn_blocking(move || {
        let results = context
            .search_semantic_results(
                &params.query,
                top_k,
                params.allowed_paths.as_deref(),
                include_content,
            )
            .map_err(map_vault_error)?;
        Ok(SearchResult {
            results: results.into_iter().map(SemanticHit::from).collect(),
        })
    })
    .await
    .map_err(|error| {
        QueryError::new(
            protocol::ERR_INTERNAL,
            format!("semantic search worker failed: {error}"),
        )
    })?;
    require_context(registry, &params.vault_root, &params.attachment_id)?;
    result
}

pub async fn search_hybrid(
    registry: &VaultRegistry,
    params: SearchHybridParams,
) -> QueryResult<SearchResult> {
    let context = require_context(registry, &params.vault_root, &params.attachment_id)?;
    if params.query.is_empty() {
        return Ok(SearchResult {
            results: Vec::new(),
        });
    }
    require_semantic_ready(&context)?;

    let top_k = params.top_k.unwrap_or(DEFAULT_TOP_K);
    let include_content = params.include_content.unwrap_or(false);
    let prefetch = params.prefetch.unwrap_or(DEFAULT_PREFETCH_COUNT).max(top_k);
    let alpha = params.alpha.unwrap_or(DEFAULT_ALPHA).clamp(0.0, 1.0);

    let result = tokio::task::spawn_blocking(move || {
        let results = context
            .search_hybrid_results(
                &params.query,
                top_k,
                prefetch,
                alpha,
                params.allowed_paths.as_deref(),
                include_content,
            )
            .map_err(map_vault_error)?;
        Ok(SearchResult {
            results: results.into_iter().map(SemanticHit::from).collect(),
        })
    })
    .await
    .map_err(|error| {
        QueryError::new(
            protocol::ERR_INTERNAL,
            format!("hybrid search worker failed: {error}"),
        )
    })?;
    require_context(registry, &params.vault_root, &params.attachment_id)?;
    result
}

pub async fn open_hint(
    registry: &VaultRegistry,
    params: OpenHintParams,
) -> QueryResult<OpenHintResult> {
    let context = require_context(registry, &params.vault_root, &params.attachment_id)?;
    let (path_part, subpath) = split_subpath(&params.path);
    let relative = Path::new(path_part);
    if relative.is_absolute() {
        return Err(QueryError::new(
            protocol::ERR_INVALID_PARAMS,
            "path must be vault-relative",
        ));
    }

    let canonical_path = match context.canonical_existing_relative_path(relative) {
        Ok(path) => match context.read_note(&path) {
            Ok(_) => Some(path),
            Err(err) => return Err(map_vault_error(err)),
        },
        Err(VaultError::NoteNotFound(_)) => None,
        Err(err) => return Err(map_vault_error(err)),
    };
    let exists = canonical_path.is_some();
    let path = match canonical_path {
        Some(path) => path.to_string_lossy().into_owned(),
        None => path_part.to_string(),
    };
    Ok(OpenHintResult {
        path,
        exists,
        subpath,
    })
}

fn not_attached() -> QueryError {
    QueryError::new(
        protocol::ERR_VAULT_NOT_ATTACHED,
        "vault scope is not attached; call ensure_vault",
    )
}

fn require_context(
    registry: &VaultRegistry,
    vault_root: &str,
    attachment_id: &str,
) -> QueryResult<Arc<VaultContext>> {
    registry
        .context_for_holder(Path::new(vault_root), attachment_id)
        .map_err(map_vault_error)?
        .ok_or_else(not_attached)
}

fn semantic_status(context: &VaultContext) -> SemanticStatus {
    #[cfg(has_embeddings)]
    {
        let status = context.embedding_status();
        SemanticStatus {
            phase: match status.phase {
                crate::vault::embedding_runtime::EmbeddingPhase::Warming => SemanticPhase::Warming,
                crate::vault::embedding_runtime::EmbeddingPhase::Ready => SemanticPhase::Ready,
                crate::vault::embedding_runtime::EmbeddingPhase::Degraded => {
                    SemanticPhase::Degraded
                }
            },
            ready: status.queryable,
            indexed_notes: status.indexed_notes,
            total_notes: status.total_notes,
            pending_notes: status.pending_notes,
            last_error: status.last_error,
        }
    }
    #[cfg(not(has_embeddings))]
    {
        let _ = context;
        SemanticStatus {
            phase: SemanticPhase::Degraded,
            ready: false,
            indexed_notes: 0,
            total_notes: 0,
            pending_notes: 0,
            last_error: Some("daemon binary was compiled without embedding support".into()),
        }
    }
}

fn require_semantic_ready(context: &VaultContext) -> QueryResult<()> {
    let status = semantic_status(context);
    if status.ready {
        return Ok(());
    }

    #[cfg(not(has_embeddings))]
    let (code, message) = (
        protocol::ERR_BOOTSTRAP_REQUIRED,
        "semantic daemon has no embedding backend; use a daemon built with embedding support",
    );
    #[cfg(has_embeddings)]
    let (code, message) = match status.phase {
        SemanticPhase::Warming => (
            protocol::ERR_VAULT_NOT_READY,
            "semantic index is warming; retry the query shortly",
        ),
        SemanticPhase::Degraded => (
            protocol::ERR_VAULT_NOT_READY,
            "semantic index is degraded and unavailable; inspect status and retry after recovery",
        ),
        SemanticPhase::Ready => (
            protocol::ERR_VAULT_NOT_READY,
            "semantic index is not queryable yet; retry the query shortly",
        ),
    };

    let data = serde_json::to_value(&status).unwrap_or_else(|_| serde_json::json!({}));
    Err(QueryError {
        code,
        message: message.into(),
        data: Some(data),
    })
}

fn split_subpath(path: &str) -> (&str, Option<String>) {
    if let Some((base, subpath)) = path.split_once('#') {
        if subpath.is_empty() {
            (base, None)
        } else {
            (base, Some(subpath.to_string()))
        }
    } else {
        (path, None)
    }
}

fn map_vault_error(err: VaultError) -> QueryError {
    let message = err.to_string();
    match err {
        VaultError::DaemonRpc {
            code,
            message,
            data,
        } => QueryError {
            code,
            message,
            data,
        },
        VaultError::InvalidPath(_)
        | VaultError::OutsideVault(_)
        | VaultError::AlreadyExists(_)
        | VaultError::PatchTargetNotFound { .. }
        | VaultError::InvalidRegex { .. }
        | VaultError::EmbeddingInputTooLong { .. }
        | VaultError::InvalidSourceRange(_) => {
            QueryError::new(protocol::ERR_INVALID_PARAMS, message)
        }
        _ => QueryError::new(protocol::ERR_INTERNAL, message),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_inference_errors_do_not_request_daemon_bootstrap() {
        let error = map_vault_error(VaultError::Embedding(
            "embedding API returned an invalid vector".into(),
        ));
        assert_eq!(error.code, protocol::ERR_INTERNAL);
        assert!(error.message.contains("invalid vector"));
    }

    #[cfg(not(has_embeddings))]
    #[tokio::test]
    async fn daemon_without_embedding_backend_requires_bootstrap() {
        let directory = tempfile::tempdir().unwrap();
        let vault_root = directory.path().join("vault");
        std::fs::create_dir(&vault_root).unwrap();
        let context = VaultContext::open(
            "test-vault".into(),
            crate::vault::fs::VaultRoot::open(&vault_root).unwrap(),
            "test-model".into(),
            Arc::new(crate::vault::exclude::ExcludeSet::build(vec![]).unwrap()),
            false,
        )
        .await
        .unwrap();
        let error = require_semantic_ready(&context).unwrap_err();
        assert_eq!(error.code, protocol::ERR_BOOTSTRAP_REQUIRED);
        assert!(error.message.contains("no embedding backend"));
    }
}
