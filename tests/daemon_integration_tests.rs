mod common;

#[cfg(all(unix, feature = "embeddings"))]
mod daemon_integration_tests {
    use std::future::Future;
    use std::sync::LazyLock;
    use std::time::{Duration, Instant};

    use serde_json::json;

    use crate::common::daemon_test_utils::{
        DaemonTestServer, create_temp_vault, rpc_request, write_note, write_note_bytes,
    };

    const MODEL_NAME: &str = "BAAI/bge-small-en-v1.5";

    /// fastembed model loading is not concurrency-safe across many daemon startups.
    static MODEL_LOCK: LazyLock<tokio::sync::Mutex<()>> =
        LazyLock::new(|| tokio::sync::Mutex::new(()));

    async fn wait_until<F, Fut>(timeout: Duration, label: &str, mut check: F)
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = bool>,
    {
        let deadline = Instant::now() + timeout;
        loop {
            if check().await {
                return;
            }
            assert!(Instant::now() < deadline, "timed out waiting for {label}");
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
    }

    #[tokio::test]
    async fn daemon_health_and_open_hint_work() {
        let _guard = MODEL_LOCK.lock().await;
        let server = DaemonTestServer::start(MODEL_NAME).await;

        let api_version = server.health_api_version().await;
        assert_eq!(api_version, 1);

        let vault = create_temp_vault();
        write_note(vault.path(), "note.md", "# Note\nhello world");
        let ensure = server.ensure_vault(vault.path(), false).await;
        assert!(ensure.ready);

        let hint = server.open_hint(vault.path(), "note.md#Heading").await;
        assert_eq!(hint.path, "note.md");
        assert!(hint.exists);
        assert_eq!(hint.subpath.as_deref(), Some("Heading"));

        server.shutdown().await;
    }

    #[tokio::test]
    async fn open_hint_rejects_path_traversal_with_invalid_params() {
        let _guard = MODEL_LOCK.lock().await;
        let server = DaemonTestServer::start(MODEL_NAME).await;

        let vault = create_temp_vault();
        write_note(vault.path(), "note.md", "# Note\nhello world");
        server.ensure_vault(vault.path(), false).await;

        let response = server
            .request_value(
                "open_hint",
                json!({
                    "vault_root": vault.path().display().to_string(),
                    "path": "../outside.md"
                }),
            )
            .await;
        assert_eq!(response["error"]["code"], json!(-32602));

        server.shutdown().await;
    }

    #[tokio::test]
    async fn per_vault_isolation_with_same_note_name() {
        let _guard = MODEL_LOCK.lock().await;
        let server = DaemonTestServer::start(MODEL_NAME).await;

        let vault_a = create_temp_vault();
        let vault_b = create_temp_vault();

        write_note(
            vault_a.path(),
            "shared.md",
            "# Shared\nA_ONLY_MARKER Rust ownership borrow checker and memory safety.",
        );
        write_note(
            vault_b.path(),
            "shared.md",
            "# Shared\nB_ONLY_MARKER Tomato gardening compost basil soil and watering.",
        );

        server.ensure_vault(vault_a.path(), false).await;
        server.ensure_vault(vault_b.path(), false).await;

        let a_results = server
            .search_semantic(vault_a.path(), "tomato basil soil", 5, true)
            .await;
        assert!(!a_results.results.is_empty());
        assert!(
            a_results.results.iter().all(|hit| {
                !hit.content
                    .as_deref()
                    .unwrap_or_default()
                    .contains("B_ONLY_MARKER")
            }),
            "vault A results must not include vault B content"
        );

        let b_results = server
            .search_semantic(vault_b.path(), "borrow checker ownership", 5, true)
            .await;
        assert!(!b_results.results.is_empty());
        assert!(
            b_results.results.iter().all(|hit| {
                !hit.content
                    .as_deref()
                    .unwrap_or_default()
                    .contains("A_ONLY_MARKER")
            }),
            "vault B results must not include vault A content"
        );

        server.shutdown().await;
    }

    #[tokio::test]
    async fn watcher_syncs_create_modify_delete() {
        let _guard = MODEL_LOCK.lock().await;
        let server = DaemonTestServer::start(MODEL_NAME).await;
        let vault = create_temp_vault();

        server.ensure_vault(vault.path(), true).await;

        write_note(
            vault.path(),
            "watched.md",
            "# Watched\nWATCH_CREATE_MARKER vector embeddings retrieval relevance",
        );

        wait_until(
            Duration::from_secs(20),
            "watcher create propagation",
            || async {
                let result = server
                    .search_semantic(vault.path(), "vector embeddings retrieval", 5, true)
                    .await;
                result.results.iter().any(|hit| hit.path == "watched.md")
            },
        )
        .await;

        write_note(
            vault.path(),
            "watched.md",
            "# Watched\nWATCH_UPDATE_MARKER raft consensus leader election log replication",
        );

        wait_until(
            Duration::from_secs(20),
            "watcher modify propagation",
            || async {
                let result = server
                    .search_semantic(vault.path(), "raft consensus leader election", 5, true)
                    .await;
                result.results.iter().any(|hit| {
                    hit.path == "watched.md"
                        && hit
                            .content
                            .as_deref()
                            .unwrap_or_default()
                            .contains("WATCH_UPDATE_MARKER")
                })
            },
        )
        .await;

        std::fs::remove_file(vault.path().join("watched.md")).expect("delete watched note");

        wait_until(
            Duration::from_secs(20),
            "watcher delete propagation",
            || async {
                let result = server
                    .search_semantic(vault.path(), "raft consensus leader election", 5, true)
                    .await;
                !result.results.iter().any(|hit| hit.path == "watched.md")
            },
        )
        .await;

        server.shutdown().await;
    }

    #[tokio::test]
    async fn concurrent_clients_can_attach_and_query() {
        let _guard = MODEL_LOCK.lock().await;
        let server = DaemonTestServer::start(MODEL_NAME).await;
        let vault = create_temp_vault();

        write_note(
            vault.path(),
            "concurrency.md",
            "# Concurrency\nConcurrent clients attach and query semantic daemon.",
        );
        write_note(
            vault.path(),
            "distributed.md",
            "# Distributed\nConsensus and replication across nodes.",
        );

        let endpoint = server.endpoint_path().to_path_buf();
        let vault_root = vault.path().to_path_buf();

        let mut join_set = tokio::task::JoinSet::new();
        for _ in 0..8usize {
            let endpoint = endpoint.clone();
            let vault_root = vault_root.clone();
            join_set.spawn(async move {
                let ensure = rpc_request(
                    &endpoint,
                    "ensure_vault",
                    json!({
                        "vault_root": vault_root.display().to_string(),
                        "watch": false
                    }),
                )
                .await;
                assert!(
                    ensure.get("error").is_none() || ensure["error"].is_null(),
                    "ensure_vault should succeed: {ensure}"
                );

                let search = rpc_request(
                    &endpoint,
                    "search_semantic",
                    json!({
                        "vault_root": vault_root.display().to_string(),
                        "query": "concurrent clients semantic query",
                        "top_k": 5,
                        "include_content": false
                    }),
                )
                .await;
                assert!(
                    search.get("error").is_none() || search["error"].is_null(),
                    "search_semantic should succeed: {search}"
                );
                search["result"]["results"]
                    .as_array()
                    .expect("results should be array")
                    .len()
            });
        }

        let mut completed = 0usize;
        while let Some(joined) = join_set.join_next().await {
            let result_len = joined.expect("task should complete");
            assert!(result_len > 0, "expected non-empty semantic results");
            completed += 1;
        }
        assert_eq!(completed, 8);

        server.shutdown().await;
    }

    #[tokio::test]
    async fn daemon_recovers_after_watcher_reindex_error() {
        let _guard = MODEL_LOCK.lock().await;
        let server = DaemonTestServer::start(MODEL_NAME).await;
        let vault = create_temp_vault();

        server.ensure_vault(vault.path(), true).await;

        write_note_bytes(vault.path(), "broken.md", b"\xff\xfe\xfd");
        tokio::time::sleep(Duration::from_millis(1200)).await;

        write_note(
            vault.path(),
            "recover.md",
            "# Recover\nRECOVERY_MARKER daemon continues serving after watcher error.",
        );

        wait_until(Duration::from_secs(20), "post-error recovery", || async {
            let result = server
                .search_semantic(vault.path(), "daemon recovery marker", 5, true)
                .await;
            result.results.iter().any(|hit| {
                hit.path == "recover.md"
                    && hit
                        .content
                        .as_deref()
                        .unwrap_or_default()
                        .contains("RECOVERY_MARKER")
            })
        })
        .await;

        server.shutdown().await;
    }

    #[tokio::test]
    async fn search_hybrid_keeps_empty_query_short_circuit() {
        let _guard = MODEL_LOCK.lock().await;
        let server = DaemonTestServer::start(MODEL_NAME).await;
        let vault = create_temp_vault();

        write_note(
            vault.path(),
            "hybrid.md",
            "# Hybrid\nhybrid lexical semantic reranking example",
        );
        server.ensure_vault(vault.path(), false).await;

        let result = server
            .search_hybrid(vault.path(), "", 5, 50, 0.25, false)
            .await;
        assert!(result.results.is_empty());

        server.shutdown().await;
    }
}
