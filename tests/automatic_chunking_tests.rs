//! Complete-note context and semantic evidence across automatic input partitioning.

mod context {
    use std::path::Path;

    use obsidian_mcp::config::{Config, ToolFilter, Transport};
    use obsidian_mcp::tools::metadata::{NoteInspectParams, note_inspect};
    use obsidian_mcp::tools::notes::{NoteReadParams, note_read};
    use obsidian_mcp::vault::Vault;
    use serde_json::{Value, json};

    fn config(root: &Path) -> Config {
        Config {
            vault_path: root.to_path_buf(),
            watch: false,
            log_level: "error".into(),
            transport: Transport::Stdio,
            http_host: "127.0.0.1".parse().unwrap(),
            http_port: 37842,
            tantivy: false,
            embeddings: false,
            embeddings_model: String::new(),
            hybrid_alpha: 0.25,
            embedding_provider: None,
            tool_filter: ToolFilter::Full,
            mcp_data_dir: None,
            exclude_patterns: Vec::new(),
        }
    }

    async fn read(vault: &Vault, arguments: Value) -> rmcp::model::CallToolResult {
        let params: NoteReadParams = serde_json::from_value(arguments).unwrap();
        note_read(vault, params).await.unwrap()
    }

    #[tokio::test]
    async fn context_expansion_reconstructs_exact_note_and_rejects_stale_locators() {
        let directory = tempfile::tempdir().unwrap();
        let raw = "---\r\ntags: [retention]\r\n---\r\n# Backups\r\n\r\n## Routine 🦦\r\nDelete local backups after thirty days.\r\n\r\n## Exceptions\r\nRetain backups during active incidents.\r\n";
        std::fs::write(directory.path().join("policy.md"), raw).unwrap();
        let vault = Vault::open(&config(directory.path())).await.unwrap();

        let legacy = read(&vault, json!({"path":"policy.md"})).await;
        assert_eq!(legacy.structured_content.unwrap(), json!({"content":raw}));
        assert_eq!(legacy.content[0].as_text().unwrap().text, raw);

        let first = read(&vault, json!({"path":"policy.md","max_bytes":9})).await;
        let first = first.structured_content.unwrap();
        let revision = first["source_revision"].as_str().unwrap().to_owned();
        let mut reconstructed = first["content"].as_str().unwrap().to_owned();
        let mut next = first["next_range"].clone();
        while !next.is_null() {
            let page = read(
                &vault,
                json!({"path":"policy.md","expected_revision":revision,"range":next,"max_bytes":9}),
            )
            .await;
            let page = page.structured_content.unwrap();
            assert_eq!(page["source_revision"], revision);
            reconstructed.push_str(page["content"].as_str().unwrap());
            next = page["next_range"].clone();
        }
        assert_eq!(
            reconstructed, raw,
            "pagination preserves CRLF and UTF-8 exactly"
        );

        let params: NoteInspectParams = serde_json::from_value(json!({
            "path":"policy.md", "fields":["headings"], "expected_revision":revision,
            "max_headings":2
        }))
        .unwrap();
        let outline = note_inspect(&vault, params)
            .await
            .unwrap()
            .structured_content
            .unwrap();
        assert_eq!(outline["source_revision"], revision);
        assert_eq!(outline["total_headings"], 3);
        let params: NoteInspectParams = serde_json::from_value(json!({
            "path":"policy.md", "fields":["headings"], "expected_revision":revision,
            "heading_offset":outline["next_heading_offset"], "max_headings":2
        }))
        .unwrap();
        let outline = note_inspect(&vault, params)
            .await
            .unwrap()
            .structured_content
            .unwrap();
        let exception = &outline["headings"][0];
        assert_eq!(exception["text"], "Exceptions");
        let expanded = read(
            &vault,
            json!({"path":"policy.md","expected_revision":revision,"range":exception["section_range"]}),
        )
        .await;
        assert!(
            expanded.structured_content.unwrap()["content"]
                .as_str()
                .unwrap()
                .contains("active incidents")
        );

        vault
            .write_note(Path::new("policy.md"), "# Replacement\nChanged source.")
            .unwrap();
        let params: NoteReadParams = serde_json::from_value(json!({
            "path":"policy.md", "expected_revision":revision, "range":exception["section_range"]
        }))
        .unwrap();
        let error = note_read(&vault, params).await.unwrap_err();
        assert_eq!(error.code, rmcp::model::ErrorCode::INVALID_REQUEST);
        assert!(
            error
                .data
                .as_ref()
                .unwrap()
                .get("current_revision")
                .is_some()
        );
    }
}

#[cfg(feature = "embeddings")]
mod local_retrieval {
    use std::path::Path;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    use std::time::{Duration, Instant};

    use obsidian_mcp::config::{Config, EmbeddingProvider, SemanticMode, ToolFilter, Transport};
    use obsidian_mcp::tools::SemanticRuntime;
    use obsidian_mcp::tools::search::{SearchSemanticParams, search_semantic};
    use obsidian_mcp::vault::Vault;
    use serde_json::Value;

    fn config(root: &Path) -> Config {
        Config {
            vault_path: root.to_path_buf(),
            watch: false,
            log_level: "error".into(),
            transport: Transport::Stdio,
            http_host: "127.0.0.1".parse().unwrap(),
            http_port: 37842,
            tantivy: true,
            embeddings: true,
            embeddings_model: "BAAI/bge-small-en-v1.5".into(),
            hybrid_alpha: 0.25,
            embedding_provider: Some(EmbeddingProvider::Local),
            tool_filter: ToolFilter::Full,
            mcp_data_dir: None,
            exclude_patterns: Vec::new(),
        }
    }

    fn runtime() -> SemanticRuntime {
        SemanticRuntime {
            mode: SemanticMode::Local,
            daemon_client: None,
            pending_daemon_bootstrap: None,
            daemon_unavailable_reason: None,
            prefetch_count: 50,
            attachments: Default::default(),
            local_fallback_active: Arc::new(AtomicBool::new(false)),
            daemon_health_pending: Arc::new(AtomicBool::new(false)),
        }
    }

    async fn ready(vault: &Vault) {
        let deadline = Instant::now() + Duration::from_secs(300);
        loop {
            let status = vault.embedding_status().unwrap();
            if status.phase == obsidian_mcp::vault::embedding_runtime::EmbeddingPhase::Ready {
                assert_eq!(status.pending_notes, 0);
                return;
            }
            assert!(status.last_error.is_none(), "indexing failed: {status:?}");
            assert!(
                Instant::now() < deadline,
                "index did not become ready: {status:?}"
            );
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    }

    async fn search(vault: &Vault, runtime: &SemanticRuntime, query: &str) -> Vec<Value> {
        let result = search_semantic(
            vault,
            SearchSemanticParams {
                query: query.into(),
                top_k: Some(3),
                ..Default::default()
            },
            0.25,
            runtime,
        )
        .await
        .unwrap();
        let hits = result.structured_content.unwrap()["results"]
            .as_array()
            .unwrap()
            .clone();
        for hit in &hits {
            let source =
                std::fs::read_to_string(vault.root().join(hit["path"].as_str().unwrap())).unwrap();
            assert_eq!(hit["source_status"], "current");
            assert_eq!(
                hit["source_revision"],
                obsidian_mcp::vault::source::source_revision(&source)
            );
            let start = hit["evidence"]["range"]["start_byte"].as_u64().unwrap() as usize;
            let end = hit["evidence"]["range"]["end_byte"].as_u64().unwrap() as usize;
            assert!(
                source.get(start..end).is_some(),
                "invalid evidence range for {}: {hit}",
                hit["path"]
            );
        }
        println!(
            "local retrieval query={query:?} result_bytes={}",
            serde_json::to_vec(&hits).unwrap().len()
        );
        hits
    }

    async fn orchard_boundary_pair() -> (String, String, usize) {
        tokio::task::spawn_blocking(|| {
            let model = fastembed::TextEmbedding::try_new(
                fastembed::InitOptions::new(fastembed::EmbeddingModel::BGESmallENV15)
                    .with_show_download_progress(true),
            )
            .unwrap();
            let capacity = model.tokenizer.get_truncation().unwrap().max_length;
            let mut tokenizer = model.tokenizer.clone();
            tokenizer.with_padding(None);
            tokenizer.with_truncation(None).unwrap();
            drop(model);

            let count = |title: &str, body: &str| {
                tokenizer
                    .encode(format!("{title}\n{body}"), true)
                    .unwrap()
                    .len()
            };
            let endings = [
                "Apple trees need sunlight.",
                "Young apple trees need sunlight.",
                "Healthy young apple trees need sunlight.",
                "Healthy young apple trees need regular sunlight.",
                "Healthy young apple trees need regular sunlight and water.",
                "Healthy young apple trees need regular sunlight and clean water.",
                "Healthy young apple trees need regular sunlight and clean fresh water.",
                "Healthy young apple trees need regular sunlight and clean fresh water daily.",
            ];
            for repetitions in 1..capacity {
                let observations = "Apple trees require sunlight and water. ".repeat(repetitions);
                for ending in endings {
                    let fitting = format!("{observations}{ending}");
                    if count("orchard-a", &fitting) != capacity {
                        continue;
                    }
                    let overflowing = format!(
                        "{observations}{}",
                        ending.replacen("need ", "need more ", 1)
                    );
                    assert_eq!(count("orchard-b", &overflowing), capacity + 1);
                    return (fitting, overflowing, capacity);
                }
            }
            panic!("cannot construct a coherent exact-capacity orchard fixture");
        })
        .await
        .unwrap()
    }

    #[tokio::test]
    async fn local_model_retrieves_tail_passages_and_retains_note_scope_for_fitting_notes() {
        let directory = tempfile::tempdir().unwrap();
        let (orchard_fitting, orchard_overflowing, capacity) = orchard_boundary_pair().await;
        let weather =
            "The coastal station records rainfall, humidity, wind speed and cloud cover. ";
        let operations = format!(
            "# Operations\n\n## Weather reports\n\n{}\n\n## Backup retention\n\nEncrypted database backups must be preserved until the security incident investigation is closed. Routine copies can be deleted after thirty days.\n",
            weather.repeat(100)
        );
        let transport = format!(
            "# Delivery\n\n## Railway schedules\n\n{}\n\n## Insurance exclusions\n\nCourier insurance does not cover unaccompanied parcels left outside the recipient's home.\n",
            "Passenger trains stop at the central station every morning and evening. ".repeat(100)
        );
        let coffee = "# Coffee\nGrinding beans immediately before brewing preserves aroma. Espresso uses finely ground coffee and high water pressure. Decaffeinated coffee reduces caffeine intake. Annual transit passes renew in January.";
        let music = "A cello is a bowed string instrument. Rosin on the bow helps its hairs grip the strings and produce a clear sustained sound.";
        let garden = format!(
            "# Garden\n{}",
            "Tomatoes need sunlight, compost and regular watering. ".repeat(140)
        );
        for (path, text) in [
            ("operations.md", operations.as_str()),
            ("delivery.md", transport.as_str()),
            ("coffee.md", coffee),
            ("music.md", music),
            ("garden.md", garden.as_str()),
            ("orchard-a.md", orchard_fitting.as_str()),
            ("orchard-b.md", orchard_overflowing.as_str()),
        ] {
            std::fs::write(directory.path().join(path), text).unwrap();
        }
        let vault = Vault::open(&config(directory.path())).await.unwrap();
        ready(&vault).await;
        let runtime = runtime();
        let cases = [
            (
                "When can encrypted copies of the database be removed after a cyberattack?",
                "operations.md",
                "security incident",
            ),
            (
                "Are packages abandoned on a doorstep covered by shipping protection?",
                "delivery.md",
                "unaccompanied parcels",
            ),
            (
                "How do I make a pressurized shot from freshly ground beans?",
                "coffee.md",
                "Espresso",
            ),
            (
                "What lets the hairs of a cello bow grip its strings?",
                "music.md",
                "Rosin",
            ),
        ];
        let mut ranks = Vec::new();
        for (query, expected, answer) in cases {
            let hits = search(&vault, &runtime, query).await;
            let rank = hits
                .iter()
                .position(|hit| hit["path"] == expected)
                .unwrap_or_else(|| {
                    panic!("missing relevant note {expected} for {query}: {hits:?}")
                });
            let hit = &hits[rank];
            assert_eq!(hit["source_status"], "current");
            let source = std::fs::read_to_string(directory.path().join(expected)).unwrap();
            let start = hit["evidence"]["range"]["start_byte"].as_u64().unwrap() as usize;
            let end = hit["evidence"]["range"]["end_byte"].as_u64().unwrap() as usize;
            assert!(
                source[start..end].contains(answer),
                "wrong evidence for {expected}: {hit}"
            );
            if expected == "operations.md" || expected == "delivery.md" {
                assert_eq!(hit["evidence"]["scope"], "passage");
                assert!(start > 400);
            } else {
                assert_eq!(hit["evidence"]["scope"], "note");
                assert!(
                    hit["evidence"]["heading_path"]
                        .as_array()
                        .unwrap()
                        .is_empty()
                );
            }
            assert!(hits.iter().enumerate().all(|(index, hit)| {
                hits[..index]
                    .iter()
                    .all(|previous| previous["path"] != hit["path"])
            }));
            ranks.push((expected, rank + 1));
        }
        let hits = search(
            &vault,
            &runtime,
            "What helps an orchard produce a healthy harvest of apples?",
        )
        .await;
        for (expected, scope) in [("orchard-a.md", "note"), ("orchard-b.md", "passage")] {
            let rank = hits
                .iter()
                .position(|hit| hit["path"] == expected)
                .unwrap_or_else(|| panic!("missing boundary note {expected}: {hits:?}"));
            let hit = &hits[rank];
            assert_eq!(hit["source_status"], "current");
            assert_eq!(hit["evidence"]["scope"], scope);
            let source = std::fs::read_to_string(directory.path().join(expected)).unwrap();
            let start = hit["evidence"]["range"]["start_byte"].as_u64().unwrap() as usize;
            let end = hit["evidence"]["range"]["end_byte"].as_u64().unwrap() as usize;
            assert!(
                source
                    .get(start..end)
                    .unwrap()
                    .to_ascii_lowercase()
                    .contains("apple trees"),
                "wrong boundary evidence for {expected}: {hit}"
            );
            if scope == "note" {
                assert_eq!((start, end), (0, source.len()));
            }
            ranks.push((expected, rank + 1));
        }
        println!(
            "local synthetic retrieval recall@3=1.0 ranks={ranks:?} boundary_tokens={capacity}/{} boundary_result_bytes={}",
            capacity + 1,
            serde_json::to_vec(&hits).unwrap().len()
        );
    }
}
