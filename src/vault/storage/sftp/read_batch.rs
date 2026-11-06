//! Short-lived raw-path resolution for bounded multi-file reads.

use super::*;
use crate::vault::storage::ResolvedRead;

pub(crate) struct ReadBatch<'a> {
    storage: &'a SftpStorage,
    session: Arc<Session>,
    directories: HashMap<String, HashMap<String, FileAttrs>>,
    name_bytes: usize,
    entry_count: usize,
}

impl<'a> ReadBatch<'a> {
    pub(crate) async fn new(storage: &'a SftpStorage) -> VaultResult<Self> {
        let session = storage
            .operation("read batch", Path::new(""), storage.current_session())
            .await?;
        Ok(Self::with_session(storage, session))
    }

    pub(super) fn with_session(storage: &'a SftpStorage, session: Arc<Session>) -> Self {
        Self {
            storage,
            session,
            directories: HashMap::new(),
            name_bytes: 0,
            entry_count: 0,
        }
    }

    pub(crate) async fn validate(&mut self, path: &Path) -> VaultResult<()> {
        let storage = self.storage;
        storage
            .operation("resolve", path, async {
                self.resolve(path, true).await?;
                Ok(())
            })
            .await
    }

    pub(crate) async fn prepare(&mut self, path: &Path) -> VaultResult<ResolvedRead> {
        let storage = self.storage;
        let deadline = crate::vault::storage::source_deadline(storage.timeout);
        crate::vault::storage::with_source_deadline(
            Some(deadline),
            storage.operation("stat", path, async {
                check_prepared_deadline(Some(deadline), "resolve", path)?;
                let target = self.resolve(path, false).await?;
                Ok(ResolvedRead {
                    relative: target.relative,
                    absolute: target.absolute,
                    deadline: Some(deadline),
                })
            }),
        )
        .await
    }

    pub(crate) async fn stat(&self, target: &ResolvedRead) -> VaultResult<FileStat> {
        let deadline = self.prepared_deadline(target);
        crate::vault::storage::with_source_deadline(
            deadline,
            self.storage.operation("stat", &target.relative, async {
                check_prepared_deadline(deadline, "stat", &target.relative)?;
                file_stat(
                    &self.session.stat(path_text(&target.absolute)?).await?,
                    &target.relative,
                )
            }),
        )
        .await
    }

    pub(crate) async fn read(&self, target: &ResolvedRead, limit: usize) -> VaultResult<String> {
        let deadline = self.prepared_deadline(target);
        crate::vault::storage::with_source_deadline(
            deadline,
            self.storage.operation("read", &target.relative, async {
                check_prepared_deadline(deadline, "read", &target.relative)?;
                self.check_root().await?;
                let absolute = path_text(&target.absolute)?;
                let canonical = self.session.realpath(absolute).await?;
                relative_under(&self.storage.canonical, &canonical)?;
                if canonical != absolute {
                    return Err(VaultError::InvalidPath(
                        "remote read target changed after resolution".into(),
                    ));
                }
                read_source(
                    &self.session,
                    absolute,
                    limit.min(MAX_READ_BYTES),
                    &self.storage.read_budget,
                    &self.storage.read_admission,
                )
                .await
            }),
        )
        .await
    }

    fn prepared_deadline(&self, target: &ResolvedRead) -> Option<tokio::time::Instant> {
        target.deadline.map(|deadline| {
            deadline.min(crate::vault::storage::source_deadline(self.storage.timeout))
        })
    }

    async fn check_root(&self) -> VaultResult<()> {
        if self.session.realpath(&self.storage.canonical).await? != self.storage.canonical {
            return Err(storage_error(
                "resolve",
                "",
                StorageErrorKind::IdentityChanged,
                "remote root changed; reopen the source",
            ));
        }
        Ok(())
    }

    pub(super) async fn resolve(
        &mut self,
        path: &Path,
        allow_missing: bool,
    ) -> VaultResult<ResolvedPath> {
        self.check_root().await?;
        let requested = normalize_relative(path)?;
        let mut actual = String::new();
        let components: Vec<&str> = requested.split('/').filter(|s| !s.is_empty()).collect();
        for (index, name) in components.iter().enumerate() {
            let parent = absolute(&self.storage.canonical, &actual);
            let candidate = match self.find_entry(&parent, name).await? {
                Some((name, _)) => absolute(&parent, &name),
                None if allow_missing => {
                    actual = [actual.as_str(), &components[index..].join("/")]
                        .into_iter()
                        .filter(|s| !s.is_empty())
                        .collect::<Vec<_>>()
                        .join("/");
                    return Ok(resolved(&self.storage.canonical, actual));
                }
                None => return Err(VaultError::NoteNotFound(path.to_owned())),
            };
            let canonical = self.session.realpath(&candidate).await?;
            actual = relative_under(&self.storage.canonical, &canonical)?;
        }
        Ok(resolved(&self.storage.canonical, actual))
    }

    async fn find_entry(
        &mut self,
        parent: &str,
        name: &str,
    ) -> VaultResult<Option<(String, FileAttrs)>> {
        let native_match = match self.session.lstat(&absolute(parent, name)).await {
            Ok(_) => true,
            Err(error) if error.is_not_found() => false,
            Err(error) => return Err(error),
        };
        let cached = self.directories.get(parent);
        // A newly observed native match must not be redirected to a previously
        // listed Unicode or case equivalent. Refresh before selecting its spelling.
        if cached.is_none()
            || (native_match && !cached.is_some_and(|names| names.contains_key(name)))
        {
            self.observe_directory(parent).await?;
        }
        let mut matcher = EntryMatcher::new(name, native_match);
        if let Some(names) = self.directories.get(parent) {
            if let Some(attrs) = names.get(name) {
                return Ok(Some((name.to_owned(), *attrs)));
            }
            for (name, attrs) in names {
                matcher.observe(name.clone(), *attrs);
            }
        }
        matcher.finish()
    }

    async fn observe_directory(&mut self, parent: &str) -> VaultResult<()> {
        if let Some(previous) = self.directories.remove(parent) {
            self.entry_count -= previous.len();
            self.name_bytes -= parent.len() + previous.keys().map(String::len).sum::<usize>();
        }
        let mut name_bytes = self.name_bytes.saturating_add(parent.len());
        let mut entry_count = self.entry_count;
        let mut names = HashMap::new();
        visit_directory(&self.session, parent, |name, attrs| {
            name_bytes = name_bytes.saturating_add(name.len());
            entry_count = entry_count.saturating_add(1);
            if name_bytes > MAX_READ_BYTES || entry_count > 1_000_000 {
                return Err(storage_error(
                    "resolve",
                    parent,
                    StorageErrorKind::ResourceLimit,
                    "batch directory observation budget exceeded",
                ));
            }
            names.insert(name, attrs);
            Ok(())
        })
        .await?;
        if name_bytes > MAX_READ_BYTES {
            return Err(storage_error(
                "resolve",
                parent,
                StorageErrorKind::ResourceLimit,
                "batch directory observation budget exceeded",
            ));
        }
        self.name_bytes = name_bytes;
        self.entry_count = entry_count;
        self.directories.insert(parent.to_owned(), names);
        Ok(())
    }
}

fn check_prepared_deadline(
    deadline: Option<tokio::time::Instant>,
    operation: &str,
    path: &Path,
) -> VaultResult<()> {
    if deadline.is_some_and(|deadline| tokio::time::Instant::now() >= deadline) {
        return Err(storage_error(
            operation,
            &path.display().to_string(),
            StorageErrorKind::Unavailable,
            "prepared note source deadline exceeded",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::tests::{Fault, fake_storage};
    use super::*;

    #[tokio::test]
    async fn repeated_bulk_reads_enumerate_the_shared_parent_once() {
        let (mut storage, files, server) = fake_storage(Fault::Normal).await;
        storage.timeout = Duration::from_secs(5);
        let mut batch = ReadBatch::new(&storage).await.unwrap();
        for _ in 0..100 {
            batch.validate(Path::new("note.md")).await.unwrap();
        }
        for _ in 0..100 {
            let target = batch.prepare(Path::new("note.md")).await.unwrap();
            assert_eq!(target.relative, Path::new("note.md"));
            assert_eq!(batch.read(&target, 3).await.unwrap(), "old");
        }
        {
            let files = files.lock().unwrap();
            assert_eq!(files.opened_directories, 1);
            assert_eq!(files.read_paths.len(), 100);
        }
        server.abort();
    }

    #[tokio::test]
    async fn healthy_bulk_reads_can_outlive_one_source_operation_timeout() {
        let (mut storage, files, server) = fake_storage(Fault::Normal).await;
        storage.timeout = Duration::from_millis(150);
        let mut batch = ReadBatch::new(&storage).await.unwrap();
        let started = tokio::time::Instant::now();
        for index in 0..2 {
            batch.validate(Path::new("note.md")).await.unwrap();
            let target = batch.prepare(Path::new("note.md")).await.unwrap();
            assert_eq!(batch.stat(&target).await.unwrap().size, 3);
            assert_eq!(batch.read(&target, 3).await.unwrap(), "old");
            if index == 0 {
                tokio::time::sleep(Duration::from_millis(180)).await;
            }
        }
        assert!(started.elapsed() > storage.timeout);
        assert!(batch.session.is_healthy());
        assert_eq!(files.lock().unwrap().opened_directories, 1);
        server.abort();
    }

    #[tokio::test]
    async fn prepared_note_metadata_and_body_keep_the_original_deadline() {
        let (mut storage, files, server) = fake_storage(Fault::Normal).await;
        storage.timeout = Duration::from_millis(150);
        let mut batch = ReadBatch::new(&storage).await.unwrap();
        for body in [false, true] {
            let target = batch.prepare(Path::new("note.md")).await.unwrap();
            tokio::time::sleep_until(target.deadline.unwrap() + Duration::from_millis(1)).await;
            let before = files.lock().unwrap().paths.len();
            let result = if body {
                batch.read(&target, 3).await.map(|_| ())
            } else {
                batch.stat(&target).await.map(|_| ())
            };
            assert!(
                matches!(
                    result,
                    Err(VaultError::Storage {
                        kind: StorageErrorKind::Unavailable,
                        ..
                    })
                ),
                "{result:?}"
            );
            assert!(batch.session.is_healthy());
            assert_eq!(files.lock().unwrap().paths.len(), before);
            assert!(files.lock().unwrap().read_paths.is_empty());
        }
        server.abort();
    }

    #[tokio::test]
    async fn prepared_reads_preserve_ambient_source_deadlines() {
        let (storage, files, server) = fake_storage(Fault::Normal).await;
        let mut batch = ReadBatch::new(&storage).await.unwrap();
        let deadline = tokio::time::Instant::now() + Duration::from_millis(100);
        let target = crate::vault::storage::with_source_deadline(
            Some(deadline),
            batch.prepare(Path::new("note.md")),
        )
        .await
        .unwrap();
        assert_eq!(target.deadline, Some(deadline));
        let before = files.lock().unwrap().paths.len();
        let earlier = tokio::time::Instant::now();
        let result =
            crate::vault::storage::with_source_deadline(Some(earlier), batch.stat(&target)).await;
        assert!(
            matches!(
                result,
                Err(VaultError::Storage {
                    kind: StorageErrorKind::Unavailable,
                    ..
                })
            ),
            "{result:?}"
        );
        assert!(batch.session.is_healthy());
        assert_eq!(files.lock().unwrap().paths.len(), before);
        assert!(files.lock().unwrap().read_paths.is_empty());
        server.abort();
    }

    #[tokio::test]
    async fn a_new_native_match_refreshes_a_cached_equivalent() {
        let (storage, files, server) = fake_storage(Fault::Normal).await;
        let mut batch = ReadBatch::new(&storage).await.unwrap();
        batch.directories.insert(
            "/vault".into(),
            HashMap::from([("NOTE.md".into(), FileAttrs::new())]),
        );
        batch.name_bytes = "/vault".len() + "NOTE.md".len();
        batch.entry_count = 1;
        let target = batch.prepare(Path::new("note.md")).await.unwrap();
        assert_eq!(target.relative, Path::new("note.md"));
        assert_eq!(files.lock().unwrap().opened_directories, 1);
        server.abort();
    }
}
