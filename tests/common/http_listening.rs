use std::collections::VecDeque;
use std::io;
use std::net::SocketAddr;
use std::time::Duration;

use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Child;

#[path = "isolated_command.rs"]
mod command;
pub use command::isolated_command;

pub async fn ready(child: &mut Child) -> Result<SocketAddr, Box<dyn std::error::Error>> {
    let stderr = child.stderr.take().expect("HTTP fixture must pipe stderr");
    let mut lines = BufReader::new(stderr).lines();
    let mut diagnostics = VecDeque::with_capacity(8);
    let listening = async {
        while let Some(line) = lines.next_line().await? {
            if let Some((_, address)) = line.split_once("HTTP MCP server listening addr=") {
                return address
                    .trim()
                    .parse::<SocketAddr>()
                    .map_err(io::Error::other);
            }
            if diagnostics.len() == 8 {
                diagnostics.pop_front();
            }
            diagnostics.push_back(line.chars().take(512).collect::<String>());
        }
        Err(io::Error::other(
            "HTTP child exited before reporting its address",
        ))
    };
    let address = tokio::time::timeout(Duration::from_secs(15), listening)
        .await
        .map_err(io::Error::other)
        .and_then(|result| result)
        .map_err(|error| io::Error::other(format!("{error}; recent stderr: {diagnostics:?}")))?;
    tokio::spawn(async move {
        let _ = tokio::io::copy(&mut lines.into_inner(), &mut tokio::io::sink()).await;
    });

    let mut health_address = address;
    if address.ip().is_unspecified() {
        health_address.set_ip(if address.is_ipv4() {
            std::net::Ipv4Addr::LOCALHOST.into()
        } else {
            std::net::Ipv6Addr::LOCALHOST.into()
        });
    }
    let response = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(10))
        .build()?
        .get(format!("http://{health_address}/health"))
        .send()
        .await?
        .error_for_status()?;
    let body: serde_json::Value = serde_json::from_str(&response.text().await?)?;
    if body["server"] != "obsidian-mcp" || body["version"] != env!("CARGO_PKG_VERSION") {
        return Err(io::Error::other(format!("unexpected HTTP fixture identity: {body}")).into());
    }
    Ok(address)
}
