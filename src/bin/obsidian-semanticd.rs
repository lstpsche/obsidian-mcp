use tracing_subscriber::EnvFilter;

use obsidian_mcp::config::DEFAULT_MODEL_NAME;
use obsidian_mcp::daemon::home::{self, semantic_home_paths};
use obsidian_mcp::daemon::protocol::SEMANTIC_SOURCE_EVIDENCE_CAPABILITY;
use obsidian_mcp::daemon::server::{self, DaemonServerConfig, IpcEndpoint};
use obsidian_mcp::upgrade::BuildIdentity;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    if let Some(code) = handle_cli_flags() {
        std::process::exit(code);
    }

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

fn handle_cli_flags() -> Option<i32> {
    let arg = std::env::args().nth(1)?;
    match arg.as_str() {
        "--__build-info" => {
            match daemon_build_info() {
                Ok(identity) => println!("{identity}"),
                Err(err) => {
                    eprintln!("error: {err}");
                    return Some(1);
                }
            }
            Some(0)
        }
        "--version" | "-v" => {
            println!("obsidian-semanticd {}", env!("CARGO_PKG_VERSION"));
            Some(0)
        }
        "--help" | "-h" | "help" => {
            println!(
                "obsidian-semanticd {version} — semantic search daemon for obsidian-mcp\n\
                 \n\
                 USAGE:\n    \
                     obsidian-semanticd\n\
                 \n\
                 OPTIONS:\n    \
                     -h, --help       Print this help message\n    \
                     -v, --version    Print version\n\
                 \n\
                 ENVIRONMENT VARIABLES:\n    \
                     OBSIDIAN_LOG_LEVEL          Tracing log level           [default: info]\n    \
                     OBSIDIAN_SEMANTIC_HOME      Override semantic home dir\n    \
                     OBSIDIAN_SEMANTIC_ENDPOINT  Override IPC endpoint\n    \
                     OBSIDIAN_SEMANTIC_MODEL     Embedding model name        [default: {model}]\n    \
                     OBSIDIAN_EMBEDDING_PROVIDER Backend: local | api\n    \
                     OBSIDIAN_EMBEDDING_API_BASE Embedding API base URL\n    \
                     OBSIDIAN_EMBEDDING_API_MODEL Exact outgoing API model name\n    \
                     OBSIDIAN_EMBEDDING_API_KEY  API authentication key (or OPENAI_API_KEY)\n    \
                     OBSIDIAN_EMBEDDING_API_PROFILE Verified tokenizer/capacity JSON for custom APIs\n    \
                     OBSIDIAN_EMBEDDING_API_TIMEOUT_SECS API request timeout [default: 30]\n    \
                     OBSIDIAN_EMBEDDING_BATCH_SIZE Reconciliation notes per batch [default: 32]",
                version = env!("CARGO_PKG_VERSION"),
                model = DEFAULT_MODEL_NAME,
            );
            Some(0)
        }
        _ => {
            eprintln!("error: unknown argument '{arg}'; use --help for usage");
            Some(2)
        }
    }
}

fn daemon_build_info() -> Result<String, serde_json::Error> {
    #[derive(serde::Serialize)]
    struct DaemonBuildInfo {
        #[serde(flatten)]
        identity: BuildIdentity,
        capabilities: [&'static str; 1],
    }
    serde_json::to_string(&DaemonBuildInfo {
        identity: BuildIdentity::embedded(),
        capabilities: [SEMANTIC_SOURCE_EVIDENCE_CAPABILITY],
    })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn daemon_capabilities_preserve_legacy_build_identity_parsing() {
        let encoded = daemon_build_info().unwrap();
        assert_eq!(
            BuildIdentity::from_json(encoded.as_bytes()).unwrap(),
            BuildIdentity::embedded()
        );
        let info: serde_json::Value = serde_json::from_str(&encoded).unwrap();
        assert_eq!(
            info["capabilities"],
            serde_json::json!([SEMANTIC_SOURCE_EVIDENCE_CAPABILITY])
        );
    }
}
