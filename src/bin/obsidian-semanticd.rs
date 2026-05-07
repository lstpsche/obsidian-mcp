use tracing_subscriber::EnvFilter;

use obsidian_mcp::config::DEFAULT_MODEL_NAME;
use obsidian_mcp::daemon::home::{self, semantic_home_paths};
use obsidian_mcp::daemon::server::{self, DaemonServerConfig, IpcEndpoint};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let log_level = std::env::var("OBSIDIAN_LOG_LEVEL").unwrap_or_else(|_| "info".to_string());
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new(log_level))
        .with_writer(std::io::stderr)
        .init();

    let semantic_home = home::resolve_semantic_home()?;
    let paths = semantic_home_paths(&semantic_home);
    home::ensure_home_layout(&paths)?;

    let endpoint =
        resolve_endpoint_from_env().unwrap_or_else(|| home::default_ipc_endpoint(&paths));
    let model_name =
        std::env::var("OBSIDIAN_SEMANTIC_MODEL").unwrap_or_else(|_| DEFAULT_MODEL_NAME.to_string());

    tracing::info!(
        endpoint = %endpoint.endpoint_string(),
        semantic_home = %semantic_home.display(),
        "starting obsidian-semanticd"
    );

    let config = DaemonServerConfig {
        endpoint,
        model_name,
        semantic_home,
    };
    server::run(config).await?;
    Ok(())
}

fn resolve_endpoint_from_env() -> Option<IpcEndpoint> {
    let raw = std::env::var("OBSIDIAN_SEMANTIC_ENDPOINT").ok()?;
    if raw.trim().is_empty() {
        return None;
    }
    #[cfg(unix)]
    {
        Some(IpcEndpoint::UnixSocket(std::path::PathBuf::from(raw)))
    }
    #[cfg(windows)]
    {
        Some(IpcEndpoint::NamedPipe(raw))
    }
}
