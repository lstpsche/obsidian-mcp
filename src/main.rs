#[cfg(any(unix, test))]
use std::path::PathBuf;

use rmcp::ServiceExt;
use tracing_subscriber::EnvFilter;

use obsidian_mcp::client::semantic_daemon::{DaemonConnectPolicy, SemanticDaemonClient};
use obsidian_mcp::config::{Config, SemanticMode, SemanticRuntimeConfig};
use obsidian_mcp::daemon::bootstrap::{BootstrapConfig, ensure_daemon};
use obsidian_mcp::daemon::server::IpcEndpoint;
use obsidian_mcp::error::VaultError;
use obsidian_mcp::tools::{ObsidianMcp, SemanticRuntime};
use obsidian_mcp::vault::Vault;

const DAEMON_DISABLED_BY_WATCH_REASON: &str =
    "semantic daemon disabled because OBSIDIAN_WATCH is false";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::load()?;
    let semantic_runtime_config = SemanticRuntimeConfig::load_from_env();

    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new(&config.log_level))
        .with_writer(std::io::stderr)
        .init();

    tracing::info!(vault = %config.vault_path.display(), "starting obsidian-mcp");

    let semantic_runtime = init_semantic_runtime(&config, &semantic_runtime_config).await;
    tracing::info!(
        semantic_mode = semantic_runtime.mode.as_str(),
        daemon_ready = semantic_runtime.daemon_client.is_some(),
        "semantic runtime configured"
    );

    let vault = Vault::open(&config).await?;
    let server = ObsidianMcp::new(vault, config.hybrid_alpha, semantic_runtime)
        .serve(rmcp::transport::io::stdio())
        .await?;

    server.waiting().await?;
    Ok(())
}

struct InitializedDaemonClient {
    client: SemanticDaemonClient,
    #[cfg(feature = "embeddings")]
    semantic_home: Option<PathBuf>,
}

async fn init_semantic_runtime(
    config: &Config,
    runtime_cfg: &SemanticRuntimeConfig,
) -> SemanticRuntime {
    let mut runtime = SemanticRuntime {
        mode: runtime_cfg.mode,
        daemon_client: None,
        daemon_unavailable_reason: None,
        prefetch_count: runtime_cfg.prefetch_count,
    };

    if runtime_cfg.mode == SemanticMode::Local {
        return runtime;
    }
    if !config.watch {
        runtime.daemon_unavailable_reason = Some(DAEMON_DISABLED_BY_WATCH_REASON.to_string());
        tracing::info!("semantic daemon disabled because OBSIDIAN_WATCH=false");
        return runtime;
    }

    match initialize_daemon_client(runtime_cfg).await {
        Ok(initialized) => {
            #[cfg(feature = "embeddings")]
            if let Some(semantic_home) = initialized.semantic_home.as_deref() {
                match obsidian_mcp::vault::embeddings::migrate_legacy_cache_to_daemon_store(
                    &config.vault_path,
                    semantic_home,
                ) {
                    Ok(obsidian_mcp::vault::embeddings::LegacyCacheMigration::Migrated(path)) => {
                        tracing::info!(
                            path = %path.display(),
                            "migrated legacy local embedding cache into daemon namespace store"
                        );
                    }
                    Ok(obsidian_mcp::vault::embeddings::LegacyCacheMigration::AlreadyPresent(
                        path,
                    )) => {
                        tracing::debug!(
                            path = %path.display(),
                            "daemon embedding cache already present; skipping legacy cache migration"
                        );
                    }
                    Ok(obsidian_mcp::vault::embeddings::LegacyCacheMigration::NotFound) => {}
                    Err(err) => {
                        tracing::warn!(
                            error = %err,
                            "failed to migrate legacy embedding cache to daemon namespace"
                        );
                    }
                }
            }

            runtime.daemon_client = Some(initialized.client);
        }
        Err(err) => {
            let reason = err.to_string();
            runtime.daemon_unavailable_reason = Some(reason.clone());
            match runtime_cfg.mode {
                SemanticMode::Daemon => {
                    tracing::error!(
                        error = %reason,
                        "semantic daemon mode requested but daemon is unavailable"
                    );
                }
                SemanticMode::Auto => {
                    tracing::warn!(
                        error = %reason,
                        "semantic daemon unavailable; auto mode may fall back to local backend"
                    );
                }
                SemanticMode::Local => {}
            }
        }
    }

    runtime
}

async fn initialize_daemon_client(
    runtime_cfg: &SemanticRuntimeConfig,
) -> Result<InitializedDaemonClient, VaultError> {
    let policy = DaemonConnectPolicy::from_runtime_config(runtime_cfg);
    let initialized = if let Some(raw_endpoint) = runtime_cfg.daemon_endpoint_override.as_deref() {
        InitializedDaemonClient {
            client: SemanticDaemonClient::new(endpoint_from_override(raw_endpoint), policy),
            #[cfg(feature = "embeddings")]
            semantic_home: None,
        }
    } else {
        let bootstrap_result = ensure_daemon(&BootstrapConfig {
            semantic_home_override: runtime_cfg.semantic_home_override.clone(),
            daemon_path_override: runtime_cfg.daemon_path_override.clone(),
            model_name: runtime_cfg.model_name.clone(),
            download_url_override: runtime_cfg.daemon_download_url.clone(),
            bootstrap_client_name: "obsidian-mcp".to_string(),
            bootstrap_client_version: env!("CARGO_PKG_VERSION").to_string(),
        })
        .await?;
        InitializedDaemonClient {
            client: SemanticDaemonClient::new(bootstrap_result.endpoint, policy),
            #[cfg(feature = "embeddings")]
            semantic_home: Some(bootstrap_result.semantic_home),
        }
    };

    let health = initialized
        .client
        .health("obsidian-mcp", env!("CARGO_PKG_VERSION"))
        .await?;
    tracing::info!(
        daemon_version = %health.daemon_version,
        daemon_api_version = health.daemon_api_version,
        daemon_status = %health.status,
        daemon_semantic_home = %health.semantic_home,
        "semantic daemon connection established"
    );
    Ok(initialized)
}

fn endpoint_from_override(raw: &str) -> IpcEndpoint {
    #[cfg(unix)]
    {
        IpcEndpoint::UnixSocket(PathBuf::from(raw))
    }
    #[cfg(windows)]
    {
        IpcEndpoint::NamedPipe(raw.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn runtime_config(mode: SemanticMode) -> SemanticRuntimeConfig {
        SemanticRuntimeConfig {
            mode,
            semantic_home_override: None,
            daemon_path_override: None,
            daemon_endpoint_override: Some("/tmp/semanticd.sock".to_string()),
            daemon_download_url: None,
            model_name: "BAAI/bge-small-en-v1.5".to_string(),
            connect_timeout_ms: 2_000,
            connect_retries: 2,
            retry_backoff_ms: 250,
            prefetch_count: 50,
        }
    }

    #[tokio::test]
    async fn watch_disabled_skips_daemon_initialization() {
        let config = Config {
            vault_path: PathBuf::from("/tmp/test-vault"),
            watch: false,
            log_level: "error".to_string(),
            tantivy: true,
            embeddings: false,
            embeddings_model: "BAAI/bge-small-en-v1.5".to_string(),
            hybrid_alpha: 0.25,
        };
        let runtime = init_semantic_runtime(&config, &runtime_config(SemanticMode::Daemon)).await;
        assert!(runtime.daemon_client.is_none());
        assert_eq!(
            runtime.daemon_unavailable_reason.as_deref(),
            Some(DAEMON_DISABLED_BY_WATCH_REASON)
        );
    }
}
