//! Daemon watcher adapter; vault and daemon indexing share the same event handling.

use crate::error::VaultResult;
use crate::vault::watcher::ChangeWatcher;
use crate::vault::{exclude::ExcludeSet, index::VaultIndex, tantivy_index::TantivyIndex};
use notify_debouncer_mini::Debouncer;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

pub fn start_watcher(
    vault_root: PathBuf,
    index: Arc<RwLock<VaultIndex>>,
    tantivy: Option<Arc<TantivyIndex>>,
    #[cfg(has_embeddings)] embedding_runtime: crate::vault::embedding_runtime::EmbeddingRuntime,
    exclude: Arc<ExcludeSet>,
) -> VaultResult<Debouncer<ChangeWatcher>> {
    crate::vault::watcher::start_watcher(
        vault_root,
        index,
        tantivy,
        #[cfg(has_embeddings)]
        Some(embedding_runtime),
        exclude,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vault::watcher::{process_event, should_process_path};
    #[cfg(not(has_embeddings))]
    use std::path::Path;
    use unicode_normalization::UnicodeNormalization;

    #[test]
    fn should_process_unicode_markdown_path() {
        let dir = tempfile::tempdir().unwrap();
        let composed = "02_База-знаний/Сущности/lic1c.md";
        let decomposed: String = composed.nfd().collect();
        let absolute = dir.path().join(decomposed);
        assert!(should_process_path(dir.path(), &absolute));
    }

    #[cfg(not(has_embeddings))]
    #[test]
    fn process_event_indexes_actual_unicode_relative_path() {
        let dir = tempfile::tempdir().unwrap();
        let composed = "02_База-знаний/Сущности/lic1c.md";
        let decomposed: String = composed.nfd().collect();
        let disk_path = PathBuf::from(&decomposed);
        let absolute = dir.path().join(&disk_path);
        std::fs::create_dir_all(absolute.parent().unwrap()).unwrap();
        std::fs::write(&absolute, "# License\n").unwrap();

        let index = Arc::new(RwLock::new(VaultIndex::empty()));
        let exclude = ExcludeSet::build(vec![]).unwrap();

        let _ = process_event(dir.path(), &index, None, &absolute, &exclude);

        let index = index.read().unwrap();
        assert!(index.get_note(&disk_path).is_some());
        assert!(index.get_note(Path::new(composed)).is_none());
    }

    #[cfg(has_embeddings)]
    #[tokio::test]
    async fn embedding_events_coalesce_to_latest_path_intents_without_inference() {
        use crate::vault::embedding_runtime::PendingKind;
        use crate::vault::embeddings::Embedder;

        let dir = tempfile::tempdir().unwrap();
        let index = Arc::new(RwLock::new(VaultIndex::empty()));
        let runtime = crate::vault::embedding_runtime::EmbeddingRuntime::spawn(
            dir.path().to_path_buf(),
            Arc::clone(&index),
            dir.path().join("embeddings.bin"),
            async { std::future::pending::<VaultResult<Arc<dyn Embedder>>>().await },
        );
        let submitter = runtime.downgrade();
        let exclude = ExcludeSet::build(vec![]).unwrap();
        let old_relative = PathBuf::from("old.md");
        let old_absolute = dir.path().join(&old_relative);

        std::fs::write(&old_absolute, "# First\n").unwrap();
        let _ = process_event(
            dir.path(),
            &index,
            None,
            Some(&submitter),
            &old_absolute,
            &exclude,
        );
        assert_eq!(
            runtime.pending_kind(&old_relative),
            Some(PendingKind::Upsert)
        );

        std::fs::write(&old_absolute, "# Latest\n").unwrap();
        let _ = process_event(
            dir.path(),
            &index,
            None,
            Some(&submitter),
            &old_absolute,
            &exclude,
        );
        assert_eq!(
            runtime.pending_kind(&old_relative),
            Some(PendingKind::Upsert)
        );

        let new_relative = PathBuf::from("new.md");
        let new_absolute = dir.path().join(&new_relative);
        std::fs::rename(&old_absolute, &new_absolute).unwrap();
        let _ = process_event(
            dir.path(),
            &index,
            None,
            Some(&submitter),
            &old_absolute,
            &exclude,
        );
        let _ = process_event(
            dir.path(),
            &index,
            None,
            Some(&submitter),
            &new_absolute,
            &exclude,
        );
        assert_eq!(
            runtime.pending_kind(&old_relative),
            Some(PendingKind::Remove)
        );
        assert_eq!(
            runtime.pending_kind(&new_relative),
            Some(PendingKind::Upsert)
        );

        std::fs::remove_file(&new_absolute).unwrap();
        let _ = process_event(
            dir.path(),
            &index,
            None,
            Some(&submitter),
            &new_absolute,
            &exclude,
        );
        assert_eq!(
            runtime.pending_kind(&new_relative),
            Some(PendingKind::Remove)
        );
    }
}
