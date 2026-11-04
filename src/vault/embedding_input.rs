//! Verified tokenization and request capacity for embedding backends.

#[cfg(feature = "embeddings-api")]
use std::path::{Path, PathBuf};
#[cfg(feature = "embeddings-api")]
use std::sync::Arc;

#[cfg(feature = "embeddings-api")]
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::error::{VaultError, VaultResult};

pub const MAX_EMBEDDING_REQUEST_BYTES: usize = 16 * 1024 * 1024;
pub const CHUNKING_VERSION: u16 = 1;
#[cfg(feature = "embeddings-api")]
const MAX_PROFILE_BYTES: usize = 1024 * 1024;
#[cfg(feature = "embeddings-api")]
const MAX_TOKENIZER_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddingInputPolicy {
    pub max_input_tokens: usize,
    pub tokenizer_fingerprint: [u8; 32],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmbeddingRequestLimits {
    pub max_inputs: usize,
    pub max_total_tokens: usize,
    pub max_request_bytes: usize,
}

#[derive(Clone)]
pub(crate) enum TokenCounter {
    HuggingFace {
        tokenizer: Box<tokenizers::Tokenizer>,
        add_special_tokens: bool,
    },
    #[cfg(feature = "embeddings-api")]
    Cl100k(Arc<tiktoken_rs::CoreBPE>),
}

impl TokenCounter {
    pub(crate) fn count(&self, text: &str) -> VaultResult<usize> {
        match self {
            Self::HuggingFace {
                tokenizer,
                add_special_tokens,
            } => tokenizer
                .encode(text, *add_special_tokens)
                .map(|encoding| encoding.len())
                .map_err(|error| {
                    VaultError::EmbeddingConfiguration(format!("tokenization failed: {error}"))
                }),
            #[cfg(feature = "embeddings-api")]
            Self::Cl100k(tokenizer) => Ok(tokenizer.encode_ordinary(text).len()),
        }
    }

    pub(crate) fn hugging_face(
        mut tokenizer: tokenizers::Tokenizer,
        add_special_tokens: bool,
    ) -> VaultResult<(Self, [u8; 32])> {
        tokenizer.with_padding(None);
        tokenizer.with_truncation(None).map_err(|error| {
            VaultError::EmbeddingConfiguration(format!(
                "cannot disable tokenizer truncation: {error}"
            ))
        })?;
        let serialized = tokenizer.to_string(false).map_err(|error| {
            VaultError::EmbeddingConfiguration(format!("cannot fingerprint tokenizer: {error}"))
        })?;
        // Cargo feature unification can select a different regex implementation.
        let implementation = if cfg!(feature = "embeddings") {
            "tokenizers-0.23.2-onig"
        } else {
            "tokenizers-0.23.2-fancy-regex"
        };
        let mut hash = Sha256::new();
        hash.update(implementation.as_bytes());
        hash.update([u8::from(add_special_tokens)]);
        hash.update(serialized.as_bytes());
        Ok((
            Self::HuggingFace {
                tokenizer: Box::new(tokenizer),
                add_special_tokens,
            },
            hash.finalize().into(),
        ))
    }
}

pub(crate) fn validate_input(
    text: &str,
    policy: &EmbeddingInputPolicy,
    count: impl FnOnce(&str) -> VaultResult<usize>,
) -> VaultResult<usize> {
    if text.is_empty() {
        return Err(VaultError::EmbeddingConfiguration(
            "embedding input cannot be empty".into(),
        ));
    }
    let tokens = count(text)?;
    if tokens > policy.max_input_tokens {
        return Err(VaultError::EmbeddingInputTooLong {
            tokens,
            max_tokens: policy.max_input_tokens,
        });
    }
    Ok(tokens)
}

/// Number of the supplied inputs that fit into the next single request.
/// `empty_request_bytes` includes all framing, including the empty input array.
pub(crate) fn plan_batch(
    texts: &[&str],
    policy: &EmbeddingInputPolicy,
    limits: EmbeddingRequestLimits,
    empty_request_bytes: usize,
    mut count: impl FnMut(&str) -> VaultResult<usize>,
) -> VaultResult<usize> {
    if texts.is_empty() {
        return Ok(0);
    }
    if limits.max_inputs == 0 || limits.max_total_tokens == 0 || limits.max_request_bytes == 0 {
        return Err(VaultError::EmbeddingConfiguration(
            "embedding request limits must be positive".into(),
        ));
    }
    let mut total_tokens = 0usize;
    let mut request_bytes = empty_request_bytes;
    let mut accepted = 0usize;
    for text in texts.iter().take(limits.max_inputs) {
        let tokens = validate_input(text, policy, &mut count)?;
        let encoded_len = serialized_input_len(text)?;
        let next_tokens = total_tokens.checked_add(tokens).ok_or_else(|| {
            VaultError::EmbeddingConfiguration("embedding request token count overflow".into())
        })?;
        let next_bytes = request_bytes
            .checked_add(encoded_len)
            .and_then(|bytes| bytes.checked_add(usize::from(accepted != 0)))
            .ok_or_else(|| {
                VaultError::EmbeddingConfiguration("embedding request byte count overflow".into())
            })?;
        let fits_tokens = next_tokens <= limits.max_total_tokens;
        let fits_bytes = next_bytes <= limits.max_request_bytes;
        if !fits_tokens || !fits_bytes {
            if accepted == 0 {
                let (resource, actual, limit) = if !fits_tokens {
                    ("embedding request tokens", tokens, limits.max_total_tokens)
                } else {
                    (
                        "embedding request bytes",
                        next_bytes,
                        limits.max_request_bytes,
                    )
                };
                return Err(VaultError::SemanticResourceLimit {
                    resource: resource.into(),
                    actual,
                    limit,
                });
            }
            break;
        }
        total_tokens = next_tokens;
        request_bytes = next_bytes;
        accepted += 1;
    }
    Ok(accepted)
}

/// Measure the exact escaped JSON string without allocating its serialized copy.
fn serialized_input_len(text: &str) -> VaultResult<usize> {
    struct CountingWriter(usize);
    impl std::io::Write for CountingWriter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0 = self.0.checked_add(bytes.len()).ok_or_else(|| {
                std::io::Error::other("serialized embedding input length overflow")
            })?;
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut writer = CountingWriter(0);
    serde_json::to_writer(&mut writer, text).map_err(|error| {
        VaultError::Embedding(format!("cannot measure embedding input: {error}"))
    })?;
    Ok(writer.0)
}

pub(crate) fn validate_batch(
    texts: &[&str],
    policy: &EmbeddingInputPolicy,
    limits: EmbeddingRequestLimits,
    empty_request_bytes: usize,
    count: impl FnMut(&str) -> VaultResult<usize>,
) -> VaultResult<()> {
    let accepted = plan_batch(texts, policy, limits, empty_request_bytes, count)?;
    if accepted != texts.len() {
        return Err(VaultError::SemanticResourceLimit {
            resource: "embedding request inputs".into(),
            actual: texts.len(),
            limit: accepted,
        });
    }
    Ok(())
}

#[cfg(feature = "embeddings-api")]
pub(crate) struct ApiInputContract {
    pub policy: EmbeddingInputPolicy,
    pub limits: EmbeddingRequestLimits,
    pub counter: TokenCounter,
}

#[cfg(feature = "embeddings-api")]
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ApiProfile {
    version: u16,
    profile_revision: String,
    api_base: String,
    model: String,
    tokenizer: ApiTokenizer,
    max_input_tokens: usize,
    max_batch_inputs: usize,
    max_batch_tokens: usize,
}

#[cfg(feature = "embeddings-api")]
#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum ApiTokenizer {
    Cl100kBase,
    Huggingface {
        path: PathBuf,
        sha256: String,
        add_special_tokens: bool,
    },
}

#[cfg(feature = "embeddings-api")]
impl ApiInputContract {
    pub(crate) fn from_env(base: &str, model: &str) -> VaultResult<Self> {
        let path = std::env::var_os("OBSIDIAN_EMBEDDING_API_PROFILE").map(PathBuf::from);
        Self::load(base, model, path.as_deref())
    }

    pub(crate) fn load(base: &str, model: &str, profile_path: Option<&Path>) -> VaultResult<Self> {
        let actual_base = normalize_api_base(base)?;
        let (tokenizer, max_input_tokens, max_inputs, max_total_tokens, revision) = if let Some(
            path,
        ) =
            profile_path
        {
            let bytes = read_bounded(path, MAX_PROFILE_BYTES, "embedding API profile")?;
            let profile: ApiProfile = serde_json::from_slice(&bytes).map_err(|error| {
                VaultError::EmbeddingConfiguration(format!(
                    "invalid embedding API profile: {error}"
                ))
            })?;
            if profile.version != 1 || profile.profile_revision.trim().is_empty() {
                return Err(VaultError::EmbeddingConfiguration(
                    "embedding API profile requires version 1 and a nonempty profile_revision"
                        .into(),
                ));
            }
            if normalize_api_base(&profile.api_base)? != actual_base || profile.model != model {
                return Err(VaultError::EmbeddingConfiguration(
                    "embedding API profile does not match the exact configured endpoint and model"
                        .into(),
                ));
            }
            let tokenizer = match profile.tokenizer {
                ApiTokenizer::Cl100kBase => cl100k_counter()?,
                ApiTokenizer::Huggingface {
                    path: tokenizer_path,
                    sha256,
                    add_special_tokens,
                } => {
                    let tokenizer_path = if tokenizer_path.is_absolute() {
                        tokenizer_path
                    } else {
                        path.parent()
                            .unwrap_or_else(|| Path::new("."))
                            .join(tokenizer_path)
                    };
                    let bytes =
                        read_bounded(&tokenizer_path, MAX_TOKENIZER_BYTES, "embedding tokenizer")?;
                    let digest = digest_hex(&bytes);
                    if sha256.len() != 64
                        || !sha256
                            .bytes()
                            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
                        || digest != sha256
                    {
                        return Err(VaultError::EmbeddingConfiguration(
                            "embedding tokenizer SHA-256 mismatch or invalid digest".into(),
                        ));
                    }
                    let tokenizer = tokenizers::Tokenizer::from_bytes(bytes).map_err(|error| {
                        VaultError::EmbeddingConfiguration(format!(
                            "invalid embedding tokenizer: {error}"
                        ))
                    })?;
                    TokenCounter::hugging_face(tokenizer, add_special_tokens)?
                }
            };
            (
                tokenizer,
                profile.max_input_tokens,
                profile.max_batch_inputs,
                profile.max_batch_tokens,
                profile.profile_revision,
            )
        } else {
            if actual_base != "https://api.openai.com/v1"
                || !matches!(
                    model,
                    "text-embedding-3-small" | "text-embedding-3-large" | "text-embedding-ada-002"
                )
            {
                return Err(VaultError::EmbeddingConfiguration(
                        "unknown embedding API tokenizer/capacity; set OBSIDIAN_EMBEDDING_API_PROFILE to a verified profile for this exact endpoint and model".into(),
                    ));
            }
            (
                cl100k_counter()?,
                8192,
                2048,
                300_000,
                "openai-embedding-contract-v1".into(),
            )
        };
        if max_input_tokens == 0 || max_inputs == 0 || max_total_tokens < max_input_tokens {
            return Err(VaultError::EmbeddingConfiguration(
                "embedding API limits must be positive and a request must accommodate one maximum-sized input".into(),
            ));
        }
        let (counter, fingerprint) = tokenizer;
        let mut hash = Sha256::new();
        hash.update(fingerprint);
        hash.update(revision.as_bytes());
        Ok(Self {
            policy: EmbeddingInputPolicy {
                max_input_tokens,
                tokenizer_fingerprint: hash.finalize().into(),
            },
            limits: EmbeddingRequestLimits {
                max_inputs,
                max_total_tokens,
                max_request_bytes: MAX_EMBEDDING_REQUEST_BYTES,
            },
            counter,
        })
    }
}

#[cfg(feature = "embeddings-api")]
fn cl100k_counter() -> VaultResult<(TokenCounter, [u8; 32])> {
    let tokenizer = tiktoken_rs::cl100k_base().map_err(|error| {
        VaultError::EmbeddingConfiguration(format!("cannot initialize cl100k_base: {error}"))
    })?;
    Ok((
        TokenCounter::Cl100k(Arc::new(tokenizer)),
        Sha256::digest(b"tiktoken-rs-0.12.1:cl100k_base:ordinary").into(),
    ))
}

#[cfg(feature = "embeddings-api")]
pub(crate) fn normalize_api_base(base: &str) -> VaultResult<String> {
    let mut url = reqwest::Url::parse(base)
        .map_err(|_| VaultError::EmbeddingConfiguration("invalid embedding API base URL".into()))?;
    if !matches!(url.scheme(), "http" | "https")
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err(VaultError::EmbeddingConfiguration(
            "embedding API base must be an HTTP(S) URL without credentials, query or fragment"
                .into(),
        ));
    }
    let path = url.path().trim_end_matches('/').to_owned();
    url.set_path(&path);
    Ok(url.to_string().trim_end_matches('/').to_owned())
}

#[cfg(feature = "embeddings-api")]
fn digest_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(feature = "embeddings-api")]
fn read_bounded(path: &Path, limit: usize, resource: &str) -> VaultResult<Vec<u8>> {
    use std::io::Read;
    let file = std::fs::File::open(path).map_err(|error| {
        VaultError::EmbeddingConfiguration(format!("cannot open {resource}: {error}"))
    })?;
    let mut bytes = Vec::new();
    file.take(limit as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| {
            VaultError::EmbeddingConfiguration(format!("cannot read {resource}: {error}"))
        })?;
    if bytes.len() > limit {
        return Err(VaultError::SemanticResourceLimit {
            resource: resource.into(),
            actual: bytes.len(),
            limit,
        });
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> EmbeddingInputPolicy {
        EmbeddingInputPolicy {
            max_input_tokens: 4,
            tokenizer_fingerprint: [1; 32],
        }
    }

    fn count(text: &str) -> VaultResult<usize> {
        Ok(text.split_whitespace().count())
    }

    #[test]
    fn final_input_boundary_and_overflow_are_checked() {
        assert_eq!(validate_input("a b c d", &policy(), count).unwrap(), 4);
        assert!(matches!(
            validate_input("a b c d e", &policy(), count),
            Err(VaultError::EmbeddingInputTooLong {
                tokens: 5,
                max_tokens: 4
            })
        ));
    }

    #[test]
    fn requests_bound_items_aggregate_tokens_and_escaped_bytes() {
        let limits = EmbeddingRequestLimits {
            max_inputs: 2,
            max_total_tokens: 5,
            max_request_bytes: 100,
        };
        assert_eq!(
            plan_batch(&["a b c", "d e", "f"], &policy(), limits, 2, count).unwrap(),
            2
        );
        assert_eq!(
            plan_batch(&["a b c", "d e f"], &policy(), limits, 2, count).unwrap(),
            1
        );
        let limits = EmbeddingRequestLimits {
            max_request_bytes: 7,
            ..limits
        };
        assert_eq!(
            plan_batch(&["a", "b"], &policy(), limits, 2, count).unwrap(),
            1
        );
        assert!(plan_batch(&["\"\"\""], &policy(), limits, 2, count).is_err());
    }

    #[test]
    fn escaped_input_is_measured_before_request_allocation() {
        let text = "\u{0001}".repeat(65_536);
        assert_eq!(serialized_input_len(&text).unwrap(), 6 * 65_536 + 2);
        let limits = EmbeddingRequestLimits {
            max_inputs: 1,
            max_total_tokens: 4,
            max_request_bytes: 32,
        };
        assert!(matches!(
            plan_batch(&[&text], &policy(), limits, 2, count),
            Err(VaultError::SemanticResourceLimit { resource, actual, limit: 32 })
                if resource == "embedding request bytes" && actual == 6 * 65_536 + 4
        ));
        let mixed = "quote\" slash\\ newline\n emoji🦀 control\0";
        assert_eq!(
            serialized_input_len(mixed).unwrap(),
            serde_json::to_vec(mixed).unwrap().len()
        );
    }

    fn tiny_hugging_face_tokenizer() -> tokenizers::Tokenizer {
        let model = tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(
                [
                    ("[UNK]".to_string(), 0),
                    ("[CLS]".to_string(), 1),
                    ("[SEP]".to_string(), 2),
                ]
                .into_iter()
                .collect(),
            )
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
        let mut tokenizer = tokenizers::Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(
            tokenizers::pre_tokenizers::whitespace::WhitespaceSplit,
        ));
        tokenizer.with_post_processor(Some(
            tokenizers::processors::template::TemplateProcessing::builder()
                .try_single("[CLS] $A [SEP]")
                .unwrap()
                .special_tokens(vec![("[CLS]", 1), ("[SEP]", 2)])
                .build()
                .unwrap(),
        ));
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: 3,
                ..Default::default()
            }))
            .unwrap();
        tokenizer.with_padding(Some(tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::Fixed(8),
            ..Default::default()
        }));
        tokenizer
    }

    #[test]
    fn hugging_face_counter_removes_truncation_padding_and_counts_special_tokens() {
        let tokenizer = tiny_hugging_face_tokenizer();
        let (counter, fp) = TokenCounter::hugging_face(tokenizer.clone(), true).unwrap();
        assert_eq!(counter.count("a b c d e").unwrap(), 7);
        let (ordinary, ordinary_fp) = TokenCounter::hugging_face(tokenizer, false).unwrap();
        assert_eq!(ordinary.count("a b c d e").unwrap(), 5);
        assert_ne!(fp, ordinary_fp);
    }

    #[cfg(feature = "embeddings-api")]
    #[test]
    fn official_profiles_require_exact_endpoint_and_model() {
        let official =
            ApiInputContract::load("https://api.openai.com/v1/", "text-embedding-3-small", None)
                .unwrap();
        assert_eq!(official.policy.max_input_tokens, 8192);
        assert!(official.counter.count("<|endoftext|>").unwrap() > 1);
        for (base, model) in [
            ("https://example.test/v1", "text-embedding-3-small"),
            ("https://api.openai.com/v1", "openai/text-embedding-3-small"),
            (
                "https://secret:token@api.openai.com/v1",
                "text-embedding-3-small",
            ),
        ] {
            assert!(ApiInputContract::load(base, model, None).is_err());
        }
    }

    #[cfg(feature = "embeddings-api")]
    #[test]
    fn custom_profile_is_strict_and_bound_to_exact_provider() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("profile.json");
        let mut value = serde_json::json!({"version":1,"profile_revision":"test","api_base":"http://127.0.0.1:8123/v1","model":"probe","tokenizer":{"kind":"cl100k_base"},"max_input_tokens":8,"max_batch_inputs":2,"max_batch_tokens":16});
        std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
        assert_eq!(
            ApiInputContract::load("http://127.0.0.1:8123/v1", "probe", Some(&path))
                .unwrap()
                .policy
                .max_input_tokens,
            8
        );
        assert!(ApiInputContract::load("http://127.0.0.1:8123/v1", "other", Some(&path)).is_err());
        value["extra"] = true.into();
        std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
        assert!(ApiInputContract::load("http://127.0.0.1:8123/v1", "probe", Some(&path)).is_err());
    }

    #[cfg(feature = "embeddings-api")]
    #[test]
    fn hugging_face_profile_checks_digest_and_exact_tokenization() {
        let dir = tempfile::tempdir().unwrap();
        let tokenizer = tiny_hugging_face_tokenizer();
        let bytes = tokenizer.to_string(false).unwrap().into_bytes();
        std::fs::write(dir.path().join("tokenizer.json"), &bytes).unwrap();
        let mut profile = serde_json::json!({"version":1,"profile_revision":"test","api_base":"http://127.0.0.1:8123/v1","model":"probe","tokenizer":{"kind":"huggingface","path":"tokenizer.json","sha256":digest_hex(&bytes),"add_special_tokens":true},"max_input_tokens":8,"max_batch_inputs":2,"max_batch_tokens":16});
        let path = dir.path().join("profile.json");
        std::fs::write(&path, serde_json::to_vec(&profile).unwrap()).unwrap();
        let contract =
            ApiInputContract::load("http://127.0.0.1:8123/v1", "probe", Some(&path)).unwrap();
        assert_eq!(contract.counter.count("a b c d e").unwrap(), 7);
        profile["tokenizer"]["sha256"] = "0".repeat(64).into();
        std::fs::write(&path, serde_json::to_vec(&profile).unwrap()).unwrap();
        assert!(ApiInputContract::load("http://127.0.0.1:8123/v1", "probe", Some(&path)).is_err());
    }
}
