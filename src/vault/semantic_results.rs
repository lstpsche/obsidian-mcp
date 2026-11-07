//! Source-bound semantic ranking and result construction shared by both transports.

use std::collections::{HashMap, HashSet};
use std::ffi::OsString;
#[cfg(test)]
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use futures_util::{StreamExt, stream::FuturesUnordered};

use crate::error::{VaultError, VaultResult};
use crate::models::{
    SemanticContextError, SemanticEvidence, SemanticNoteResult, SemanticPreview, SourceRange,
    SourceStatus,
};

use super::embeddings::{CosineQuery, NoteEmbeddingEntry};
use super::search_utils::{compile_query_word_regex, normalize_bm25_scores};
use super::source;

pub(crate) type CapturedEntries = HashMap<OsString, Arc<NoteEmbeddingEntry>>;

/// An immutable semantic observation retained until its source is rendered.
#[derive(Clone)]
pub(crate) struct RankedNote {
    pub path: PathBuf,
    pub score: f32,
    pub evidence: SemanticEvidence,
    pub entry: Arc<NoteEmbeddingEntry>,
}

pub(crate) fn paths(entries: &CapturedEntries) -> HashSet<OsString> {
    entries.keys().cloned().collect()
}

fn best_note(
    path: PathBuf,
    entry: Arc<NoteEmbeddingEntry>,
    query: &CosineQuery<'_>,
) -> VaultResult<RankedNote> {
    let mut winner = None;
    for chunk in &entry.chunks {
        let score = query.similarity(&chunk.vector);
        if !score.is_finite() {
            return Err(VaultError::Embedding(format!(
                "non-finite semantic score for {}",
                path.display()
            )));
        }
        if winner.as_ref().is_none_or(|(previous, previous_start)| {
            score > *previous || (score == *previous && chunk.range.start_byte < *previous_start)
        }) {
            winner = Some((score, chunk.range.start_byte));
        }
    }
    let (score, start) = winner.ok_or_else(|| {
        VaultError::Embedding(format!("no complete semantic input for {}", path.display()))
    })?;
    let chunk = entry
        .chunks
        .iter()
        .find(|chunk| chunk.range.start_byte == start)
        .ok_or_else(|| {
            VaultError::Embedding("semantic winner is absent from its observation".into())
        })?;
    let evidence = SemanticEvidence {
        scope: chunk.scope,
        cosine_similarity: score,
        range: chunk.range,
        heading_path: chunk.heading_path.clone(),
        section_range: chunk.section_range,
    };
    Ok(RankedNote {
        path,
        score,
        evidence,
        entry,
    })
}

fn finish_ranking(mut hits: Vec<RankedNote>, top_k: usize) -> Vec<RankedNote> {
    let compare = |left: &RankedNote, right: &RankedNote| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.path.as_os_str().cmp(right.path.as_os_str()))
    };
    if hits.len() > top_k {
        hits.select_nth_unstable_by(top_k, compare);
        hits.truncate(top_k);
    }
    hits.sort_unstable_by(compare);
    hits
}

pub(crate) fn rank_semantic(
    entries: CapturedEntries,
    query: &[f32],
    top_k: usize,
) -> VaultResult<Vec<RankedNote>> {
    let query = CosineQuery::new(query);
    let hits = entries
        .into_iter()
        .map(|(path, entry)| best_note(PathBuf::from(path), entry, &query))
        .collect::<VaultResult<Vec<_>>>()?;
    Ok(finish_ranking(hits, top_k))
}

pub(crate) fn rank_hybrid(
    entries: &CapturedEntries,
    lexical: &[(PathBuf, f32)],
    query: &[f32],
    alpha: f32,
    top_k: usize,
) -> VaultResult<Vec<RankedNote>> {
    let query = CosineQuery::new(query);
    let mut hits = Vec::with_capacity(lexical.len());
    for (path, lexical_score) in normalize_bm25_scores(lexical) {
        let entry = entries.get(path.as_os_str()).ok_or_else(|| {
            VaultError::Embedding(format!(
                "lexical candidate {} is outside the captured semantic scope",
                path.display()
            ))
        })?;
        let mut hit = best_note(path, Arc::clone(entry), &query)?;
        hit.score = alpha * lexical_score + (1.0 - alpha) * hit.evidence.cosine_similarity;
        hits.push(hit);
    }
    Ok(finish_ranking(hits, top_k))
}

/// Render only text observed at the exact revision whose vector was scored.
pub(crate) fn render_rooted(
    root: &super::fs::VaultRoot,
    hit: RankedNote,
    query: &str,
    include_content: bool,
) -> SemanticNoteResult {
    let snapshot = root
        .read_exact_bounded(&hit.path, source::MAX_SOURCE_BYTES)
        .map(source::SourceSnapshot::new);
    render_snapshot(snapshot, hit, query, include_content)
}

/// Acquire current bytes asynchronously, preserving the scored revision on failure.
async fn render_source(
    observed: &super::storage::ObservedBatch<'_>,
    hit: RankedNote,
    query: &str,
    include_content: bool,
) -> VaultResult<SemanticNoteResult> {
    let raw = observed
        .read_exact_bounded(&hit.path, source::MAX_SOURCE_BYTES)
        .await;
    let query = query.to_owned();
    tokio::task::spawn_blocking(move || {
        render_snapshot(
            raw.map(source::SourceSnapshot::new),
            hit,
            &query,
            include_content,
        )
    })
    .await
    .map_err(|error| VaultError::Other(format!("semantic source rendering task failed: {error}")))
}

const CONCURRENT_SOURCE_READS: usize = 4;
const CONTEXT_DEADLINE_ERROR: &str = "semantic source context deadline exceeded";

type ContextSender = tokio::sync::mpsc::Sender<(usize, SemanticNoteResult)>;

/// Retain dispatched I/O until its own operation deadline or confirmed completion.
async fn acquire_source_context(
    storage: &super::storage::Storage,
    hits: Vec<RankedNote>,
    query: &str,
    include_content: bool,
    deadline: Option<tokio::time::Instant>,
    sender: &ContextSender,
    failure_sender: tokio::sync::oneshot::Sender<VaultError>,
) {
    let finished = || {
        sender.is_closed()
            || deadline.is_some_and(|deadline| tokio::time::Instant::now() >= deadline)
    };
    if finished() {
        return;
    }
    let paths = hits.iter().map(|hit| hit.path.clone()).collect::<Vec<_>>();
    let observed = match storage.observe_batch(&paths).await {
        Ok(observed) => observed,
        Err(error) => {
            let message = error.to_string();
            for (index, hit) in hits.into_iter().enumerate() {
                if finished() {
                    break;
                }
                let result =
                    render_snapshot(Err(VaultError::Other(message.clone())), hit, "", false);
                if sender.send((index, result)).await.is_err() {
                    break;
                }
            }
            return;
        }
    };
    let render = |(index, hit)| {
        let observed = &observed;
        let finished = &finished;
        async move {
            let result = if finished() {
                None
            } else {
                Some(render_source(observed, hit, query, include_content).await)
            };
            (index, result)
        }
    };
    let mut candidates = hits.into_iter().enumerate();
    let mut pending = FuturesUnordered::new();
    for _ in 0..CONCURRENT_SOURCE_READS {
        if finished() {
            break;
        }
        if let Some(candidate) = candidates.next() {
            pending.push(render(candidate));
        }
    }
    let mut failure_sender = Some(failure_sender);
    let mut failed = false;
    while let Some((index, result)) = pending.next().await {
        match result {
            Some(Ok(result)) => {
                if !finished() {
                    // A closed receiver stops admission, but never cancels peers
                    // which already own a remote request or file handle.
                    let _ = sender.send((index, result)).await;
                }
            }
            Some(Err(error)) => {
                failed = true;
                if let Some(failure_sender) = failure_sender.take()
                    && let Err(error) = failure_sender.send(error)
                {
                    tracing::warn!(%error, "semantic context acquisition failed after its caller stopped waiting");
                }
            }
            None => (),
        }
        if !failed
            && !finished()
            && let Some(candidate) = candidates.next()
        {
            pending.push(render(candidate));
        }
    }
}

async fn finish_context_owner(owner: tokio::task::JoinHandle<()>) -> VaultResult<()> {
    owner.await.map_err(|error| {
        VaultError::Other(format!("semantic context acquisition task failed: {error}"))
    })
}

/// Bound result presentation without cancelling already-dispatched source reads.
pub(crate) async fn render_many_source(
    storage: &super::storage::Storage,
    hits: Vec<RankedNote>,
    query: &str,
    include_content: bool,
    timeout: Option<Duration>,
) -> VaultResult<Vec<SemanticNoteResult>> {
    if hits.is_empty() {
        return Ok(Vec::new());
    }
    let deadline = timeout
        .map(|timeout| {
            tokio::time::Instant::now()
                .checked_add(timeout)
                .ok_or_else(|| VaultError::Other("semantic context deadline is too large".into()))
        })
        .transpose()?;
    let storage = storage.clone();
    let observations = hits.clone();
    let query = query.to_owned();
    let source_deadline = super::storage::current_source_deadline();
    let (sender, mut receiver) = tokio::sync::mpsc::channel(CONCURRENT_SOURCE_READS);
    let (failure_sender, mut failure_receiver) = tokio::sync::oneshot::channel();
    // Dropping the join handle detaches this bounded read owner. Receiver closure
    // stops new work while requests and handles already in flight are drained.
    let mut owner = Some(tokio::spawn(async move {
        super::storage::with_source_deadline(
            source_deadline,
            acquire_source_context(
                &storage,
                observations,
                &query,
                include_content,
                deadline,
                &sender,
                failure_sender,
            ),
        )
        .await;
    }));
    let mut results: Vec<Option<SemanticNoteResult>> =
        std::iter::repeat_with(|| None).take(hits.len()).collect();
    let mut failure_open = true;
    loop {
        if deadline.is_some_and(|deadline| tokio::time::Instant::now() >= deadline) {
            break;
        }
        tokio::select! {
            biased;
            failure = &mut failure_receiver, if failure_open => {
                match failure {
                    Ok(error) => return Err(error),
                    Err(_) => failure_open = false,
                }
            }
            received = receiver.recv() => {
                match received {
                    Some((index, result)) => results[index] = Some(result),
                    None => {
                        if let Some(owner) = owner.take() {
                            finish_context_owner(owner).await?;
                        }
                        break;
                    }
                }
            }
            _ = async {
                match deadline {
                    Some(deadline) => tokio::time::sleep_until(deadline).await,
                    None => std::future::pending().await,
                }
            } => break,
        }
    }
    // Closing before inspection makes late failures report at their retained
    // owner instead of racing this caller's final deadline observation.
    receiver.close();
    failure_receiver.close();
    while let Ok((index, result)) = receiver.try_recv() {
        results[index] = Some(result);
    }
    if let Ok(error) = failure_receiver.try_recv() {
        return Err(error);
    }
    if owner.as_ref().is_some_and(|owner| owner.is_finished())
        && let Some(owner) = owner.take()
    {
        finish_context_owner(owner).await?;
    }
    if deadline.is_none() && results.iter().any(Option::is_none) {
        return Err(VaultError::Other(
            "semantic context acquisition finished without all observations".into(),
        ));
    }
    Ok(hits
        .into_iter()
        .zip(results)
        .map(|(hit, result)| {
            result.unwrap_or_else(|| {
                render_snapshot(
                    Err(VaultError::Other(CONTEXT_DEADLINE_ERROR.into())),
                    hit,
                    "",
                    false,
                )
            })
        })
        .collect())
}

/// Build context from exactly one observation, never from mixed source revisions.
pub(crate) fn render_snapshot(
    snapshot: VaultResult<source::SourceSnapshot>,
    hit: RankedNote,
    query: &str,
    include_content: bool,
) -> SemanticNoteResult {
    let mut result = SemanticNoteResult {
        path: hit.path.clone(),
        title: hit.entry.title.clone(),
        score: hit.score,
        tags: hit.entry.tags.clone(),
        snippet: None,
        content: None,
        source_revision: hit.entry.source_revision.clone(),
        source_bytes: hit.entry.source_bytes,
        source_status: SourceStatus::Current,
        evidence: hit.evidence,
        preview: None,
        context_error: None,
    };
    match snapshot {
        Ok(snapshot) if snapshot.source_revision != result.source_revision => {
            result.source_status = SourceStatus::Changed;
            result.context_error = Some(SemanticContextError {
                code: "source_changed".into(),
                message: "The note changed after this semantic observation was indexed.".into(),
                current_revision: Some(snapshot.source_revision),
            });
        }
        Ok(snapshot) => {
            let raw = &snapshot.content;
            let range = result.evidence.range;
            let valid_source = snapshot.validate_range(range).and_then(|()| {
                if result.source_bytes != raw.len() {
                    return Err(VaultError::InvalidSourceRange(
                        "indexed source length does not match the observed revision".into(),
                    ));
                }
                if let Some(section) = result.evidence.section_range {
                    snapshot.validate_range(section)?;
                    if section.start_byte > range.start_byte || section.end_byte < range.end_byte {
                        return Err(VaultError::InvalidSourceRange(
                            "indexed section does not contain its semantic evidence".into(),
                        ));
                    }
                }
                Ok(())
            });
            if let Err(error) = valid_source {
                result.source_status = SourceStatus::Unavailable;
                result.context_error = Some(SemanticContextError {
                    code: "invalid_source_range".into(),
                    message: error.to_string(),
                    current_revision: Some(snapshot.source_revision),
                });
            } else if include_content {
                result.content = Some(snapshot.content);
            } else {
                let preview = preview_range(raw, range, query);
                result.snippet = Some(raw[preview.start_byte..preview.end_byte].to_owned());
                result.preview = Some(SemanticPreview {
                    range: preview,
                    complete_evidence: preview == range,
                });
            }
        }
        Err(error) => {
            result.source_status = SourceStatus::Unavailable;
            result.context_error = Some(SemanticContextError {
                code: "source_unavailable".into(),
                message: error.to_string(),
                current_revision: None,
            });
        }
    }
    result
}

#[cfg(test)]
pub(crate) fn render(
    root: &Path,
    hit: RankedNote,
    query: &str,
    include_content: bool,
) -> SemanticNoteResult {
    render_rooted(
        &super::fs::VaultRoot::open(root).unwrap(),
        hit,
        query,
        include_content,
    )
}

fn preview_range(raw: &str, evidence: SourceRange, query: &str) -> SourceRange {
    const MAX_CHARS: usize = 300;
    let text = &raw[evidence.start_byte..evidence.end_byte];
    let found = compile_query_word_regex(query)
        .and_then(|regex| regex.find(text).map(|matched| matched.start()));
    let target = found.map_or(0, |position| text[..position].chars().count());
    let start_char = target.saturating_sub(MAX_CHARS / 2);
    let start = text
        .char_indices()
        .nth(start_char)
        .map_or(text.len(), |(position, _)| position);
    let end = text[start..]
        .char_indices()
        .nth(MAX_CHARS)
        .map_or(text.len(), |(position, _)| start + position);
    SourceRange {
        start_byte: evidence.start_byte + start,
        end_byte: evidence.start_byte + end,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::EvidenceScope;
    use crate::vault::embeddings::{ChunkEmbedding, EmbeddingStore, prepared_text_hash};

    fn note_entry(raw: &str, ends: &[usize], vectors: &[[f32; 2]]) -> NoteEmbeddingEntry {
        assert_eq!(ends.len(), vectors.len());
        assert_eq!(ends.last().copied(), Some(raw.len()));
        let body_start =
            super::super::frontmatter::extract_raw_frontmatter(raw).map_or(0, |(_, start)| start);
        let body_range = SourceRange {
            start_byte: body_start,
            end_byte: raw.len(),
        };
        let scope = if ends.len() == 1 {
            EvidenceScope::Note
        } else {
            EvidenceScope::Passage
        };
        let mut start_byte = body_start;
        let chunks = ends
            .iter()
            .zip(vectors)
            .map(|(&end_byte, vector)| {
                let range = SourceRange {
                    start_byte,
                    end_byte,
                };
                start_byte = end_byte;
                ChunkEmbedding {
                    input_hash: prepared_text_hash(&raw[range.start_byte..range.end_byte]),
                    vector: vector.to_vec(),
                    scope,
                    range,
                    overlap_range: None,
                    heading_path: if scope == EvidenceScope::Passage {
                        vec!["Details".into()]
                    } else {
                        Vec::new()
                    },
                    section_range: (scope == EvidenceScope::Passage).then_some(body_range),
                    block_kind: None,
                }
            })
            .collect();
        NoteEmbeddingEntry {
            source_revision: source::source_revision(raw),
            source_bytes: raw.len(),
            body_range,
            title: "indexed title".into(),
            tags: vec!["indexed-tag".into()],
            chunks,
        }
    }

    fn capture(notes: Vec<(&str, NoteEmbeddingEntry)>) -> CapturedEntries {
        let mut store = EmbeddingStore::new(2);
        let mut allowed = HashSet::new();
        for (name, entry) in notes {
            let path = PathBuf::from(name);
            allowed.insert(path.as_os_str().to_owned());
            store.insert_note(path, entry).unwrap();
        }
        store.capture_entries(&allowed)
    }

    #[test]
    fn semantic_top_k_counts_distinct_notes_after_max_chunk_scoring() {
        let captured = capture(vec![
            (
                "long.md",
                note_entry(
                    "abcdefgh",
                    &[2, 4, 6, 8],
                    &[[1.0, 0.0], [0.99, 0.1], [0.98, 0.2], [0.97, 0.3]],
                ),
            ),
            ("short.md", note_entry("short", &[5], &[[0.8, 0.6]])),
            ("other.md", note_entry("other", &[5], &[[0.6, 0.8]])),
        ]);
        let hits = rank_semantic(captured.clone(), &[1.0, 0.0], 2).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].path, Path::new("long.md"));
        assert_eq!(hits[1].path, Path::new("short.md"));
        assert_eq!(hits[0].evidence.range.start_byte, 0);
        assert_eq!(hits[0].evidence.cosine_similarity, 1.0);
        assert!(rank_semantic(captured, &[1.0, 0.0], 0).unwrap().is_empty());
    }

    #[test]
    fn tied_notes_and_chunks_choose_path_then_earliest_source_range() {
        let captured = capture(vec![
            ("z.md", note_entry("ab", &[1, 2], &[[1.0, 0.0]; 2])),
            ("a.md", note_entry("cd", &[1, 2], &[[1.0, 0.0]; 2])),
            ("m.md", note_entry("ef", &[1, 2], &[[1.0, 0.0]; 2])),
        ]);
        for _ in 0..3 {
            let hits = rank_semantic(captured.clone(), &[1.0, 0.0], 2).unwrap();
            assert_eq!(hits[0].path, Path::new("a.md"));
            assert_eq!(hits[1].path, Path::new("m.md"));
            assert!(hits.iter().all(|hit| hit.evidence.range.start_byte == 0));
        }
    }

    #[test]
    fn ranking_retains_exact_server_path_spellings() {
        let captured = capture(vec![
            ("a:b.md", note_entry("lower", &[5], &[[1.0, 0.0]])),
            ("A:b.md", note_entry("upper", &[5], &[[1.0, 0.0]])),
        ]);
        assert_eq!(captured.len(), 2);
        let hits = rank_semantic(captured, &[1.0, 0.0], 2).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].path.as_os_str(), "A:b.md");
        assert_eq!(hits[1].path.as_os_str(), "a:b.md");
    }

    #[test]
    fn passage_preview_and_full_content_use_the_scored_source_revision() {
        let dir = tempfile::tempdir().unwrap();
        let prefix = "---\ntags: [indexed-tag]\n---\n# Overview\nfront needle\n";
        let raw = format!(
            "{prefix}## Details\n{}needle{}",
            "界".repeat(250),
            "é".repeat(250)
        );
        std::fs::write(dir.path().join("note.md"), &raw).unwrap();
        let captured = capture(vec![(
            "note.md",
            note_entry(&raw, &[prefix.len(), raw.len()], &[[0.0, 1.0], [1.0, 0.0]]),
        )]);
        let hit = rank_semantic(captured.clone(), &[1.0, 0.0], 1)
            .unwrap()
            .remove(0);
        let result = render(dir.path(), hit, "needle", false);
        assert_eq!(result.source_status, SourceStatus::Current);
        assert_eq!(result.evidence.scope, EvidenceScope::Passage);
        let preview = result.preview.unwrap();
        assert!(preview.range.start_byte >= prefix.len());
        assert!(preview.range.end_byte <= raw.len());
        let snippet = result.snippet.unwrap();
        assert_eq!(
            snippet,
            raw[preview.range.start_byte..preview.range.end_byte]
        );
        assert!(snippet.contains("needle"));
        assert_eq!(snippet.chars().count(), 300);
        assert!(!preview.complete_evidence);
        assert!(result.content.is_none());
        assert!(result.context_error.is_none());

        let hit = rank_semantic(captured, &[1.0, 0.0], 1).unwrap().remove(0);
        let full = render(dir.path(), hit, "needle", true);
        assert_eq!(full.content.as_deref(), Some(raw.as_str()));
        assert_eq!(full.source_revision, source::source_revision(&raw));
        assert!(full.snippet.is_none());
        assert!(full.preview.is_none());
    }

    #[test]
    fn fitting_note_evidence_does_not_invent_a_winning_heading() {
        let dir = tempfile::tempdir().unwrap();
        let raw = "---\ntags: [one]\n---\n# First\none\n## Second\ntwo\n";
        std::fs::write(dir.path().join("fit.md"), raw).unwrap();
        let hit = rank_semantic(
            capture(vec![(
                "fit.md",
                note_entry(raw, &[raw.len()], &[[1.0, 0.0]]),
            )]),
            &[1.0, 0.0],
            1,
        )
        .unwrap()
        .remove(0);
        let result = render(dir.path(), hit, "two", false);
        assert_eq!(result.evidence.scope, EvidenceScope::Note);
        assert!(result.evidence.heading_path.is_empty());
        assert!(result.evidence.section_range.is_none());
        assert_eq!(result.evidence.range.end_byte, raw.len());
        assert!(result.evidence.range.start_byte > 0);
        assert!(result.preview.unwrap().complete_evidence);
        assert!(!result.snippet.unwrap().starts_with("---"));
    }

    #[test]
    fn invalid_section_locators_never_accompany_current_source_content() {
        let dir = tempfile::tempdir().unwrap();
        let raw = "😀tail";
        std::fs::write(dir.path().join("note.md"), raw).unwrap();
        let mut entry = note_entry(raw, &[4, raw.len()], &[[0.0, 1.0], [1.0, 0.0]]);
        // The cache can validate numeric containment without having source bytes.
        // Rendering must additionally reject a section starting inside a UTF-8 scalar.
        entry.chunks[1].section_range = Some(SourceRange {
            start_byte: 1,
            end_byte: raw.len(),
        });
        let captured = capture(vec![("note.md", entry)]);
        for include_content in [false, true] {
            let hit = rank_semantic(captured.clone(), &[1.0, 0.0], 1)
                .unwrap()
                .remove(0);
            let result = render(dir.path(), hit, "tail", include_content);
            assert_eq!(result.source_status, SourceStatus::Unavailable);
            assert_eq!(result.score, 1.0);
            assert!(
                result.content.is_none() && result.snippet.is_none() && result.preview.is_none()
            );
            assert_eq!(result.context_error.unwrap().code, "invalid_source_range");
        }
    }

    #[test]
    fn changed_and_deleted_sources_retain_rank_and_explicit_context_error() {
        let dir = tempfile::tempdir().unwrap();
        let raw = "indexed source";
        let captured = capture(vec![(
            "note.md",
            note_entry(raw, &[raw.len()], &[[1.0, 0.0]]),
        )]);
        std::fs::write(dir.path().join("note.md"), "different current source").unwrap();
        let hit = rank_semantic(captured.clone(), &[1.0, 0.0], 1)
            .unwrap()
            .remove(0);
        let changed = render(dir.path(), hit, "source", true);
        assert_eq!(changed.path, Path::new("note.md"));
        assert_eq!(changed.score, 1.0);
        assert_eq!(changed.title, "indexed title");
        assert_eq!(changed.tags, ["indexed-tag"]);
        assert_eq!(changed.source_status, SourceStatus::Changed);
        assert_eq!(changed.source_revision, source::source_revision(raw));
        assert!(
            changed.snippet.is_none() && changed.content.is_none() && changed.preview.is_none()
        );
        let error = changed.context_error.unwrap();
        assert_eq!(error.code, "source_changed");
        assert_eq!(
            error.current_revision,
            Some(source::source_revision("different current source"))
        );

        std::fs::remove_file(dir.path().join("note.md")).unwrap();
        let hit = rank_semantic(captured, &[1.0, 0.0], 1).unwrap().remove(0);
        let deleted = render(dir.path(), hit, "source", false);
        assert_eq!(deleted.source_status, SourceStatus::Unavailable);
        assert_eq!(deleted.score, 1.0);
        assert!(
            deleted.snippet.is_none() && deleted.content.is_none() && deleted.preview.is_none()
        );
        assert_eq!(deleted.context_error.unwrap().code, "source_unavailable");
    }

    #[tokio::test]
    async fn async_source_rendering_rechecks_the_ranked_revision() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("note.md");
        let original = "# Original\nsource observation";
        std::fs::write(&path, original).unwrap();
        let storage = super::super::storage::Storage::Local(
            super::super::fs::VaultRoot::open(directory.path()).unwrap(),
        );
        let captured = capture(vec![(
            "note.md",
            note_entry(original, &[original.len()], &[[1.0, 0.0]]),
        )]);
        let hit = rank_semantic(captured.clone(), &[1.0, 0.0], 1)
            .unwrap()
            .remove(0);
        let current = render_many_source(&storage, vec![hit], "observation", true, None)
            .await
            .unwrap()
            .remove(0);
        assert_eq!(current.content.as_deref(), Some(original));
        assert_eq!(current.source_status, SourceStatus::Current);

        std::fs::write(&path, "# A later revision").unwrap();
        let hit = rank_semantic(captured.clone(), &[1.0, 0.0], 1)
            .unwrap()
            .remove(0);
        let changed = render_many_source(&storage, vec![hit], "observation", true, None)
            .await
            .unwrap()
            .remove(0);
        assert_eq!(changed.source_status, SourceStatus::Changed);
        assert_eq!(changed.source_revision, source::source_revision(original));
        assert!(changed.content.is_none());
        assert_eq!(changed.context_error.unwrap().code, "source_changed");

        std::fs::remove_file(&path).unwrap();
        let hit = rank_semantic(captured, &[1.0, 0.0], 1).unwrap().remove(0);
        let missing = render_many_source(&storage, vec![hit], "observation", false, None)
            .await
            .unwrap()
            .remove(0);
        assert_eq!(missing.source_status, SourceStatus::Unavailable);
        assert!(missing.content.is_none() && missing.snippet.is_none());
        assert_eq!(missing.context_error.unwrap().code, "source_unavailable");
    }

    #[test]
    fn source_failure_never_uses_historical_content() {
        let raw = "the exact indexed revision";
        let hit = rank_semantic(
            capture(vec![(
                "note.md",
                note_entry(raw, &[raw.len()], &[[1.0, 0.0]]),
            )]),
            &[1.0, 0.0],
            1,
        )
        .unwrap()
        .remove(0);
        let failed = render_snapshot(
            Err(VaultError::Io(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "source request exceeded its deadline",
            ))),
            hit,
            "revision",
            true,
        );
        assert_eq!(failed.score, 1.0);
        assert_eq!(failed.source_status, SourceStatus::Unavailable);
        assert_eq!(failed.source_revision, source::source_revision(raw));
        assert!(failed.content.is_none() && failed.snippet.is_none() && failed.preview.is_none());
        assert!(failed.context_error.unwrap().message.contains("deadline"));
    }

    #[tokio::test]
    async fn concurrent_context_reads_keep_order_and_deadlines_retain_all_hits() {
        let directory = tempfile::tempdir().unwrap();
        let raw = "# Evidence\ncomplete note";
        let names: Vec<String> = (0..8).map(|index| format!("{index}.md")).collect();
        for name in &names {
            std::fs::write(directory.path().join(name), raw).unwrap();
        }
        let entries = capture(
            names
                .iter()
                .map(|name| (name.as_str(), note_entry(raw, &[raw.len()], &[[1.0, 0.0]])))
                .collect(),
        );
        let hits = rank_semantic(entries, &[1.0, 0.0], names.len()).unwrap();
        let storage = super::super::storage::Storage::Local(
            super::super::fs::VaultRoot::open(directory.path()).unwrap(),
        );
        let rendered = render_many_source(&storage, hits.clone(), "evidence", true, None)
            .await
            .unwrap();
        assert_eq!(rendered.len(), names.len());
        for (result, name) in rendered.iter().zip(&names) {
            assert_eq!(result.path, Path::new(name));
            assert_eq!(result.content.as_deref(), Some(raw));
        }

        let expired = render_many_source(&storage, hits, "evidence", true, Some(Duration::ZERO))
            .await
            .unwrap();
        assert_eq!(expired.len(), names.len());
        for (result, name) in expired.iter().zip(&names) {
            assert_eq!(result.path, Path::new(name));
            assert_eq!(result.score, 1.0);
            assert_eq!(result.source_status, SourceStatus::Unavailable);
            assert!(
                result.content.is_none() && result.snippet.is_none() && result.preview.is_none()
            );
            assert_eq!(
                result.context_error.as_ref().unwrap().code,
                "source_unavailable"
            );
        }
    }

    #[test]
    fn hybrid_score_keeps_raw_winner_cosine_and_rejects_missing_candidates() {
        let captured = capture(vec![
            (
                "semantic.md",
                note_entry("ab", &[1, 2], &[[1.0, 0.0], [0.8, 0.6]]),
            ),
            ("lexical.md", note_entry("cd", &[2], &[[0.0, 1.0]])),
        ]);
        let lexical = vec![
            (PathBuf::from("lexical.md"), 10.0),
            (PathBuf::from("semantic.md"), 1.0),
        ];
        let hits = rank_hybrid(&captured, &lexical, &[1.0, 0.0], 0.25, 2).unwrap();
        assert_eq!(hits[0].path, Path::new("semantic.md"));
        assert_eq!(hits[0].evidence.cosine_similarity, 1.0);
        assert_eq!(hits[0].score, 0.75);
        assert_eq!(hits[1].score, 0.25);
        let missing = vec![(PathBuf::from("pending.md"), 1.0)];
        assert!(rank_hybrid(&captured, &missing, &[1.0, 0.0], 0.25, 1).is_err());
    }

    #[test]
    fn non_finite_query_scores_fail_instead_of_becoming_ranked_results() {
        let captured = capture(vec![("note.md", note_entry("x", &[1], &[[1.0, 0.0]]))]);
        assert!(rank_semantic(captured, &[f32::NAN, 0.0], 1).is_err());
    }

    #[test]
    fn preview_is_bounded_and_uses_raw_utf8_offsets() {
        let text = format!(
            "---\na: b\n---\n{}needle{}",
            "界".repeat(250),
            "é".repeat(250)
        );
        let range = SourceRange {
            start_byte: text.find('界').unwrap(),
            end_byte: text.len(),
        };
        let preview = preview_range(&text, range, "needle");
        assert!(preview.start_byte >= range.start_byte);
        assert!(preview.end_byte <= range.end_byte);
        assert_eq!(
            text[preview.start_byte..preview.end_byte].chars().count(),
            300
        );
        assert!(text[preview.start_byte..preview.end_byte].contains("needle"));
    }
}
