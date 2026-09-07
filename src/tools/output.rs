//! Typed structured results with the existing text representation preserved.

use rmcp::model::{CallToolResult, ContentBlock, ErrorData};
use schemars::JsonSchema;
use serde::Serialize;

use crate::error::VaultError;

#[derive(Serialize, JsonSchema)]
pub struct TextOutput {
    /// Raw note content or rendered directory tree.
    pub content: String,
}

#[derive(Serialize, JsonSchema)]
pub struct MessageOutput {
    /// Confirmation of the completed operation.
    pub message: String,
}

#[derive(Serialize, JsonSchema)]
pub struct Results<T> {
    /// Matching entries, in the same order as the text response.
    pub results: T,
}

pub fn with_text<T: Serialize>(value: &T, text: String) -> Result<CallToolResult, ErrorData> {
    let value = serde_json::to_value(value)
        .map_err(|error| VaultError::Other(format!("JSON serialization failed: {error}")))?;
    let mut result = CallToolResult::success(vec![ContentBlock::text(text)]);
    result.structured_content = Some(value);
    Ok(result)
}

pub fn json<T: Serialize>(value: &T) -> Result<CallToolResult, ErrorData> {
    let text = serde_json::to_string_pretty(value)
        .map_err(|error| VaultError::Other(format!("JSON serialization failed: {error}")))?;
    with_text(value, text)
}

pub fn results<T: Serialize>(value: T) -> Result<CallToolResult, ErrorData> {
    let text = serde_json::to_string_pretty(&value)
        .map_err(|error| VaultError::Other(format!("JSON serialization failed: {error}")))?;
    with_text(&Results { results: value }, text)
}

pub fn content(content: String) -> Result<CallToolResult, ErrorData> {
    with_text(
        &TextOutput {
            content: content.clone(),
        },
        content,
    )
}

pub fn message(message: String) -> Result<CallToolResult, ErrorData> {
    with_text(
        &MessageOutput {
            message: message.clone(),
        },
        message,
    )
}
