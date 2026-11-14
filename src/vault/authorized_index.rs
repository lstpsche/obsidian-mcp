//! Retained readable-corpus projections over one shared source index.

use std::collections::{HashMap, HashSet};
use std::ffi::OsString;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use sha2::{Digest, Sha256};

use crate::access::clients::ClientPolicy;
use crate::error::{VaultError, VaultResult};

use super::index::{AuthorizedSnapshot, VaultIndex};
use super::source::source_revision;
use super::tantivy_index::TantivyIndex;

/// The index and lexical statistics belong to this readable policy alone.
/// Credential and write-role changes do not allocate another corpus.
pub(crate) struct AuthorizedProjection {
    pub index: Arc<RwLock<VaultIndex>>,
    pub tantivy: Option<Arc<TantivyIndex>>,
    ready: Arc<AtomicBool>,
}

impl AuthorizedProjection {
    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }
}

#[derive(Default)]
struct ProjectionState {
    observed_generation: Option<u64>,
    fingerprint: Option<[u8; 32]>,
    revisions: HashMap<OsString, String>,
    contents: Arc<HashMap<OsString, Arc<str>>>,
    projection: Option<Arc<AuthorizedProjection>>,
}

struct ProjectionEntry {
    update: Mutex<ProjectionState>,
    publication: Mutex<()>,
    active: AtomicBool,
    ready: Arc<AtomicBool>,
}

impl Default for ProjectionEntry {
    fn default() -> Self {
        Self {
            update: Mutex::default(),
            publication: Mutex::default(),
            active: AtomicBool::new(true),
            ready: Arc::new(AtomicBool::new(false)),
        }
    }
}

/// Entries are scoped to one source owner, whose indexing exclusions are fixed
/// for its lifetime. All refreshes use exact source keys, not folded path names.
#[derive(Default)]
pub(crate) struct AuthorizedProjectionRegistry {
    entries: Mutex<HashMap<String, Arc<ProjectionEntry>>>,
}

impl AuthorizedProjectionRegistry {
    /// Refresh a retained projection using source observations. The caller must
    /// run this blocking work outside async executor threads and supply an
    /// authorized source reader. Source I/O never holds the source metadata lock.
    pub fn project(
        &self,
        source: &Arc<RwLock<VaultIndex>>,
        policy: &ClientPolicy,
        tantivy_enabled: bool,
        read_source: impl Fn(&Path) -> VaultResult<String>,
    ) -> VaultResult<Arc<AuthorizedProjection>> {
        if !policy.enabled {
            return Err(VaultError::Authorization("client is disabled".into()));
        }
        let policy_key = policy.read_policy_hash();
        let entry = {
            let mut entries = self
                .entries
                .lock()
                .map_err(|_| projection_error("projection registry lock poisoned"))?;
            Arc::clone(entries.entry(policy_key).or_default())
        };
        let mut state = entry
            .update
            .lock()
            .map_err(|_| projection_error("projection update lock poisoned"))?;
        if state
            .projection
            .as_ref()
            .is_some_and(|projection| projection.tantivy.is_some() != tantivy_enabled)
        {
            return Err(projection_error(
                "lexical configuration changed without reopening",
            ));
        }
        let generation = source
            .read()
            .map_err(|_| projection_error("source index lock poisoned"))?
            .generation();
        if state.observed_generation == Some(generation)
            && let Some(projection) = &state.projection
            && projection.is_ready()
        {
            return Ok(Arc::clone(projection));
        }
        let result = Self::refresh(
            &entry,
            &mut state,
            source,
            policy,
            tantivy_enabled,
            read_source,
        );
        if result.is_err() {
            state.observed_generation = None;
            if let Some(projection) = &state.projection {
                projection.ready.store(false, Ordering::Release);
            }
        }
        result
    }

    fn refresh(
        entry: &ProjectionEntry,
        state: &mut ProjectionState,
        source: &Arc<RwLock<VaultIndex>>,
        policy: &ClientPolicy,
        tantivy_enabled: bool,
        read_source: impl Fn(&Path) -> VaultResult<String>,
    ) -> VaultResult<Arc<AuthorizedProjection>> {
        let allows = policy.read_filter();
        let snapshot = source
            .read()
            .map_err(|_| projection_error("source index lock poisoned"))?
            .authorized_snapshot(allows)?;
        let fingerprint = fingerprint(&snapshot)?;
        if state.fingerprint == Some(fingerprint)
            && let Some(projection) = &state.projection
            && projection.is_ready()
        {
            state.observed_generation = Some(snapshot.generation);
            return Ok(Arc::clone(projection));
        }

        let mut contents = HashMap::with_capacity(snapshot.notes.len());
        let mut changed = Vec::new();
        let mut paths = snapshot.notes.keys().collect::<Vec<_>>();
        paths.sort_unstable();
        for key in paths {
            let expected = snapshot
                .revisions
                .get(key)
                .ok_or_else(|| projection_error("authorized source revision is missing"))?;
            let content = if state.revisions.get(key) == Some(expected) {
                state
                    .contents
                    .get(key)
                    .cloned()
                    .ok_or_else(|| projection_error("authorized content snapshot is missing"))?
            } else {
                let path = Path::new(key);
                let content = read_source(path)?;
                let observed = source_revision(&content);
                if &observed != expected {
                    return Err(VaultError::SourceChanged {
                        path: path.to_owned(),
                        expected_revision: expected.clone(),
                        current_revision: observed,
                    });
                }
                changed.push(key.clone());
                Arc::<str>::from(content)
            };
            contents.insert(key.clone(), content);
        }
        let removed = state
            .revisions
            .keys()
            .filter(|path| !snapshot.revisions.contains_key(*path))
            .cloned()
            .collect::<Vec<_>>();
        let contents = Arc::new(contents);
        let next_index = VaultIndex::from_authorized_snapshot(&snapshot, Arc::clone(&contents));
        let projection = match &state.projection {
            Some(projection) => Arc::clone(projection),
            None => Arc::new(AuthorizedProjection {
                index: Arc::new(RwLock::new(VaultIndex::empty())),
                tantivy: tantivy_enabled
                    .then(TantivyIndex::empty)
                    .transpose()?
                    .map(Arc::new),
                ready: Arc::clone(&entry.ready),
            }),
        };

        let was_ready = projection.is_ready();
        if let Some(tantivy) = &projection.tantivy {
            if !was_ready {
                tantivy.replace_all_documents(&snapshot.notes, &contents)?;
            } else if !changed.is_empty() || !removed.is_empty() {
                for key in &removed {
                    tantivy.remove_file_batch(Path::new(key))?;
                }
                for key in &changed {
                    let metadata = snapshot
                        .notes
                        .get(key)
                        .ok_or_else(|| projection_error("authorized metadata is missing"))?;
                    let content = contents
                        .get(key)
                        .ok_or_else(|| projection_error("authorized content is missing"))?;
                    tantivy.replace_document(Path::new(key), metadata, content)?;
                }
                // Reader publication is expensive and occurs outside both the
                // source and authorized metadata locks.
                tantivy.flush()?;
            }
        }
        let _publication = entry
            .publication
            .lock()
            .map_err(|_| projection_error("projection publication lock poisoned"))?;
        if !entry.active.load(Ordering::Acquire) {
            return Err(projection_error("authorized projection was withdrawn"));
        }
        *projection
            .index
            .write()
            .map_err(|_| projection_error("authorized index lock poisoned"))? = next_index;
        state.observed_generation = Some(snapshot.generation);
        state.fingerprint = Some(fingerprint);
        state.revisions = snapshot.revisions;
        state.contents = contents;
        state.projection = Some(Arc::clone(&projection));
        projection.ready.store(true, Ordering::Release);
        Ok(projection)
    }

    /// Reclaim policies no longer admitted by any eligible client. Callers
    /// derive this set from a current protected policy snapshot; source/provider
    /// work is not part of this short authority-maintenance operation.
    pub fn retain_read_policies(&self, retained: &HashSet<String>) -> VaultResult<()> {
        let obsolete = self
            .entries
            .lock()
            .map_err(|_| projection_error("projection registry lock poisoned"))?
            .keys()
            .filter(|key| !retained.contains(*key))
            .cloned()
            .collect::<Vec<_>>();
        for key in obsolete {
            self.retire(&key)?;
        }
        Ok(())
    }

    /// Withdraw an obsolete policy before reclaiming its derived state.
    pub fn retire(&self, policy_key: &str) -> VaultResult<()> {
        let entry = self
            .entries
            .lock()
            .map_err(|_| projection_error("projection registry lock poisoned"))?
            .remove(policy_key);
        if let Some(entry) = entry {
            let _publication = entry
                .publication
                .lock()
                .map_err(|_| projection_error("projection publication lock poisoned"))?;
            entry.active.store(false, Ordering::Release);
            entry.ready.store(false, Ordering::Release);
        }
        Ok(())
    }
}

fn fingerprint(snapshot: &AuthorizedSnapshot) -> VaultResult<[u8; 32]> {
    let mut hash = Sha256::new();
    hash.update(b"obsidian-mcp/authorized-observations/v1\0");
    let mut paths = snapshot.notes.keys().collect::<Vec<_>>();
    paths.sort_unstable();
    for path in paths {
        hash.update(b"note\0");
        hash_part(&mut hash, path.as_encoded_bytes());
        hash_part(
            &mut hash,
            snapshot
                .revisions
                .get(path)
                .ok_or_else(|| projection_error("authorized source revision is missing"))?
                .as_bytes(),
        );
        let stat = serde_json::to_vec(&snapshot.notes[path].stat)
            .map_err(|_| projection_error("authorized source stat could not be encoded"))?;
        hash_part(&mut hash, &stat);
    }
    let mut attachments = snapshot.attachments.iter().collect::<Vec<_>>();
    attachments.sort_unstable_by_key(|(path, _)| *path);
    for (path, size) in attachments {
        hash.update(b"attachment\0");
        hash_part(&mut hash, path.as_encoded_bytes());
        hash.update(size.to_be_bytes());
    }
    let mut excluded = snapshot.excluded.iter().collect::<Vec<_>>();
    excluded.sort_unstable();
    for path in excluded {
        hash.update(b"excluded\0");
        hash_part(&mut hash, path.as_encoded_bytes());
    }
    Ok(hash.finalize().into())
}

fn hash_part(hash: &mut Sha256, value: &[u8]) {
    hash.update((value.len() as u64).to_be_bytes());
    hash.update(value);
}

fn projection_error(message: &str) -> VaultError {
    VaultError::Other(message.to_owned())
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, RwLock};
    use std::time::Duration;

    use crate::access::clients::ClientPolicy;
    use crate::models::FileStat;
    use crate::vault::exclude::ExcludeSet;

    use super::*;

    fn policy(role: &str) -> ClientPolicy {
        serde_json::from_value(serde_json::json!({
            "enabled": true,
            "grants": [{"id": "shared", "directory": "Shared/", "role": role}],
            "denies": [],
            "credentials": []
        }))
        .unwrap()
    }

    fn stat(size: usize) -> FileStat {
        FileStat {
            size: size as u64,
            created: None,
            modified: None,
        }
    }

    fn observe(source: &Arc<RwLock<VaultIndex>>, path: &str, content: &str) {
        source
            .write()
            .unwrap()
            .apply_observation(
                Path::new(path),
                Some(content.to_owned()),
                stat(content.len()),
                None,
                false,
            )
            .unwrap();
    }

    fn source() -> Arc<RwLock<VaultIndex>> {
        Arc::new(RwLock::new(VaultIndex::for_source(Arc::default())))
    }

    fn read<'a>(
        contents: &'a HashMap<PathBuf, String>,
        reads: &'a Cell<usize>,
    ) -> impl Fn(&Path) -> VaultResult<String> + 'a {
        move |path| {
            reads.set(reads.get() + 1);
            contents
                .get(path)
                .cloned()
                .ok_or_else(|| VaultError::NoteNotFound(path.to_owned()))
        }
    }

    #[test]
    fn hidden_documents_cannot_change_visible_scores_or_crowd_out_top_hit() {
        let source = source();
        let registry = AuthorizedProjectionRegistry::default();
        let policy = policy("reader");
        let reads = Cell::new(0);
        let contents = HashMap::from([
            (
                PathBuf::from("Shared/first.md"),
                "rust rust compiler".to_owned(),
            ),
            (PathBuf::from("Shared/second.md"), "rust tooling".to_owned()),
        ]);
        for (path, content) in &contents {
            observe(&source, path.to_str().unwrap(), content);
        }
        let first = registry
            .project(&source, &policy, true, read(&contents, &reads))
            .unwrap();
        let before = first.tantivy.as_ref().unwrap().search("rust", 2).unwrap();
        assert_eq!(reads.get(), 2);
        for index in 0..30 {
            observe(
                &source,
                &format!("Private/{index}.md"),
                "rust rust rust rust rust rust rust rust rust rust",
            );
        }
        let second = registry
            .project(&source, &policy, true, read(&contents, &reads))
            .unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(reads.get(), 2, "hidden changes do not reread visible notes");
        let after = second.tantivy.as_ref().unwrap().search("rust", 2).unwrap();
        assert_eq!(before, after);
        assert_eq!(after.len(), 2);
        assert_eq!(
            second.tantivy.as_ref().unwrap().search("rust", 1).unwrap(),
            before[..1]
        );
    }

    #[test]
    fn hidden_graph_targets_and_duplicate_basenames_do_not_change_visible_graph() {
        let source = source();
        let registry = AuthorizedProjectionRegistry::default();
        let policy = policy("reader");
        let reads = Cell::new(0);
        let contents = HashMap::from([
            (
                PathBuf::from("Shared/source.md"),
                "[[secret]] [[target]]".to_owned(),
            ),
            (PathBuf::from("Shared/target.md"), "# Target".to_owned()),
        ]);
        for (path, content) in &contents {
            observe(&source, path.to_str().unwrap(), content);
        }
        let projection = registry
            .project(&source, &policy, false, read(&contents, &reads))
            .unwrap();
        {
            let index = projection.index.read().unwrap();
            assert_eq!(index.outgoing_links(Path::new("Shared/source.md")).len(), 2);
            assert_eq!(index.broken_links().len(), 1);
            assert_eq!(
                index.resolve_link("target"),
                Some("Shared/target.md".into())
            );
            assert_eq!(index.stats().total_links, 2);
        }
        observe(&source, "Private/secret.md", "private target");
        observe(&source, "Private/target.md", "private duplicate");
        let projection = registry
            .project(&source, &policy, false, read(&contents, &reads))
            .unwrap();
        let index = projection.index.read().unwrap();
        assert_eq!(index.outgoing_links(Path::new("Shared/source.md")).len(), 2);
        assert_eq!(index.broken_links().len(), 1);
        assert_eq!(
            index.resolve_link("target"),
            Some("Shared/target.md".into())
        );
        assert_eq!(index.stats().total_links, 2);
        assert_eq!(reads.get(), 2);
    }

    #[test]
    fn attachment_and_exclusion_statistics_use_authorized_inventory() {
        let exclude = Arc::new(ExcludeSet::build(vec!["**/ignored.md".into()]).unwrap());
        let source = Arc::new(RwLock::new(VaultIndex::for_source(exclude)));
        for (path, bytes) in [
            ("Shared/image.png", 42),
            ("Private/image.png", 9000),
            ("Shared/ignored.md", 11),
            ("Private/ignored.md", 99),
        ] {
            source
                .write()
                .unwrap()
                .apply_observation(Path::new(path), None, stat(bytes), None, false)
                .unwrap();
        }
        observe(&source, "Shared/note.md", "shared");
        let registry = AuthorizedProjectionRegistry::default();
        let projection = registry
            .project(&source, &policy("reader"), false, |_| Ok("shared".into()))
            .unwrap();
        let index = projection.index.read().unwrap();
        assert_eq!(index.stats().total_notes, 1);
        assert_eq!(index.stats().total_files, 2);
        assert_eq!(index.stats().vault_size_bytes, 48);
        assert_eq!(index.stats().excluded_notes, 1);
    }

    #[test]
    fn equal_read_policies_share_projection_and_only_changed_notes_are_read_again() {
        let source = source();
        let registry = AuthorizedProjectionRegistry::default();
        let reads = Cell::new(0);
        let mut contents = HashMap::from([
            (PathBuf::from("Shared/one.md"), "one".to_owned()),
            (PathBuf::from("Shared/two.md"), "two".to_owned()),
        ]);
        for (path, content) in &contents {
            observe(&source, path.to_str().unwrap(), content);
        }
        let reader = registry
            .project(&source, &policy("reader"), true, read(&contents, &reads))
            .unwrap();
        let editor = registry
            .project(&source, &policy("editor"), true, read(&contents, &reads))
            .unwrap();
        assert!(Arc::ptr_eq(&reader, &editor));
        assert_eq!(reads.get(), 2);
        contents.insert("Shared/two.md".into(), "updated".into());
        observe(&source, "Shared/two.md", "updated");
        let updated = registry
            .project(&source, &policy("reader"), true, read(&contents, &reads))
            .unwrap();
        assert_eq!(reads.get(), 3);
        assert_eq!(
            updated
                .tantivy
                .as_ref()
                .unwrap()
                .search("updated", 1)
                .unwrap()[0]
                .0,
            Path::new("Shared/two.md")
        );
        registry
            .project(&source, &policy("reader"), true, read(&contents, &reads))
            .unwrap();
        assert_eq!(reads.get(), 3);
    }

    #[test]
    fn failed_authorized_refresh_is_unavailable_until_complete_recovery() {
        let source = source();
        observe(&source, "Shared/note.md", "old");
        let registry = AuthorizedProjectionRegistry::default();
        let policy = policy("reader");
        let projection = registry
            .project(&source, &policy, true, |_| Ok("old".into()))
            .unwrap();
        observe(&source, "Shared/note.md", "new");
        assert!(
            registry
                .project(&source, &policy, true, |_| {
                    Err(VaultError::Other("visible source unavailable".into()))
                })
                .is_err()
        );
        assert!(!projection.is_ready());
        let recovered = registry
            .project(&source, &policy, true, |_| Ok("new".into()))
            .unwrap();
        assert!(Arc::ptr_eq(&projection, &recovered));
        assert!(recovered.is_ready());
        assert!(
            recovered
                .tantivy
                .as_ref()
                .unwrap()
                .search("old", 1)
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            recovered
                .tantivy
                .as_ref()
                .unwrap()
                .search("new", 1)
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn incremental_refresh_keeps_the_published_corpus_ready() {
        let source = source();
        observe(&source, "Shared/note.md", "originalword");
        let registry = Arc::new(AuthorizedProjectionRegistry::default());
        let policy = policy("reader");
        let projection = registry
            .project(&source, &policy, true, |_| Ok("originalword".into()))
            .unwrap();
        let entry = registry
            .entries
            .lock()
            .unwrap()
            .get(&policy.read_policy_hash())
            .cloned()
            .unwrap();
        let publishing = entry.publication.lock().unwrap();
        observe(&source, "Shared/note.md", "replacementword");
        let worker_registry = Arc::clone(&registry);
        let worker_source = Arc::clone(&source);
        let worker = std::thread::spawn(move || {
            worker_registry.project(&worker_source, &policy, true, |_| {
                Ok("replacementword".into())
            })
        });
        let lexical = projection.tantivy.as_ref().unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        let lexical_published = loop {
            if !lexical.search("replacementword", 1).unwrap().is_empty() {
                break true;
            }
            if worker.is_finished() || std::time::Instant::now() >= deadline {
                break false;
            }
            std::thread::sleep(Duration::from_millis(5));
        };
        let ready_during_refresh = projection.is_ready();
        drop(publishing);
        let refreshed = worker.join().unwrap().unwrap();
        assert!(lexical_published, "refresh must reach held publication");
        assert!(ready_during_refresh, "a routine refresh is not withdrawal");
        assert!(Arc::ptr_eq(&projection, &refreshed));
        assert!(refreshed.is_ready());
    }

    #[test]
    fn publication_failure_invalidates_and_rebuilds_the_complete_lexical_corpus() {
        let source = source();
        observe(&source, "Shared/note.md", "originalword");
        let registry = AuthorizedProjectionRegistry::default();
        let policy = policy("reader");
        let projection = registry
            .project(&source, &policy, true, |_| Ok("originalword".into()))
            .unwrap();
        let poisoned = Arc::clone(&projection);
        assert!(
            std::thread::spawn(move || {
                let _index = poisoned.index.write().unwrap();
                panic!("synthetic metadata publication failure");
            })
            .join()
            .is_err()
        );
        observe(&source, "Shared/note.md", "replacementword");
        let error = registry
            .project(&source, &policy, true, |_| Ok("replacementword".into()))
            .err()
            .unwrap();
        assert!(error.to_string().contains("authorized index lock poisoned"));
        assert!(!projection.is_ready());
        let lexical = projection.tantivy.as_ref().unwrap();
        assert_eq!(lexical.search("replacementword", 1).unwrap().len(), 1);
        let metadata = source
            .read()
            .unwrap()
            .get_note(Path::new("Shared/note.md"))
            .unwrap()
            .clone();
        lexical
            .replace_document(Path::new("Shared/untracked.md"), &metadata, "untrackedword")
            .unwrap();
        lexical.flush().unwrap();
        assert_eq!(lexical.search("untrackedword", 1).unwrap().len(), 1);
        projection.index.clear_poison();
        let recovered = registry
            .project(&source, &policy, true, |_| Ok("replacementword".into()))
            .unwrap();
        assert!(Arc::ptr_eq(&projection, &recovered));
        assert!(recovered.is_ready());
        assert!(lexical.search("untrackedword", 1).unwrap().is_empty());
        assert!(lexical.search("originalword", 1).unwrap().is_empty());
        assert_eq!(lexical.search("replacementword", 1).unwrap().len(), 1);
    }

    #[test]
    fn hidden_source_authority_failure_does_not_degrade_readable_projection() {
        let source = source();
        observe(&source, "Shared/note.md", "shared");
        observe(&source, "Private/note.md", "private");
        let registry = AuthorizedProjectionRegistry::default();
        let policy = policy("reader");
        let first = registry
            .project(&source, &policy, false, |_| Ok("shared".into()))
            .unwrap();
        assert!(
            source
                .write()
                .unwrap()
                .apply_observation(
                    Path::new("Private/note.md"),
                    Some("private".into()),
                    FileStat {
                        size: u64::MAX,
                        created: None,
                        modified: None
                    },
                    None,
                    false,
                )
                .is_err()
        );
        let second = registry
            .project(&source, &policy, false, |_| {
                panic!("hidden failure must not cause visible source rereading")
            })
            .unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        assert!(second.is_ready());
        assert_eq!(second.index.read().unwrap().stats().total_notes, 1);
    }

    #[test]
    fn policy_pruning_withdraws_only_corpora_without_an_eligible_read_policy() {
        let source = source();
        observe(&source, "Shared/note.md", "shared");
        observe(&source, "Private/note.md", "private");
        let registry = AuthorizedProjectionRegistry::default();
        let shared_policy = policy("reader");
        let mut private_policy = policy("editor");
        private_policy.grants[0].directory = Some("Private/".into());
        let shared = registry
            .project(&source, &shared_policy, false, |_| Ok("shared".into()))
            .unwrap();
        let private = registry
            .project(&source, &private_policy, false, |_| Ok("private".into()))
            .unwrap();
        let retained = HashSet::from([policy("editor").read_policy_hash()]);
        registry.retain_read_policies(&retained).unwrap();
        assert!(
            shared.is_ready(),
            "another client's identical read key retains the corpus"
        );
        assert!(!private.is_ready());
        let shared_again = registry
            .project(&source, &shared_policy, false, |_| {
                panic!("retained corpus is current")
            })
            .unwrap();
        assert!(Arc::ptr_eq(&shared, &shared_again));
        let private_again = registry
            .project(&source, &private_policy, false, |_| Ok("private".into()))
            .unwrap();
        assert!(!Arc::ptr_eq(&private, &private_again));
        assert!(
            !private.is_ready(),
            "old handles cannot regain authority on reactivation"
        );
    }

    #[test]
    fn withdrawal_does_not_wait_for_a_source_read_or_allow_late_publication() {
        let source = source();
        observe(&source, "Shared/note.md", "shared");
        let registry = Arc::new(AuthorizedProjectionRegistry::default());
        let policy = policy("reader");
        let policy_key = policy.read_policy_hash();
        let (reading, read_started) = std::sync::mpsc::channel();
        let (resume, resumed) = std::sync::mpsc::channel();
        let worker_registry = Arc::clone(&registry);
        let worker = std::thread::spawn(move || {
            worker_registry.project(&source, &policy, false, |_| {
                reading.send(()).unwrap();
                resumed.recv_timeout(Duration::from_secs(5)).unwrap();
                Ok("shared".into())
            })
        });
        read_started.recv_timeout(Duration::from_secs(5)).unwrap();
        registry.retire(&policy_key).unwrap();
        resume.send(()).unwrap();
        assert!(worker.join().unwrap().is_err());
    }
}
