//! Minimal, dependency-free health verification for managed HTTP services.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
use std::time::{Duration, Instant};

use serde::Deserialize;

use super::UpgradeError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HealthObservation {
    pub version: String,
}

#[derive(Debug, Deserialize)]
struct HealthBody {
    status: String,
    server: String,
    version: String,
}

pub fn probe(host: &str, port: u16) -> Result<HealthObservation, UpgradeError> {
    let connect_host = connect_host(host);
    let addresses = (connect_host.as_str(), port)
        .to_socket_addrs()
        .map_err(|err| {
            UpgradeError::Activation(format!("cannot resolve {connect_host}:{port}: {err}"))
        })?
        .collect::<Vec<SocketAddr>>();
    let mut last_error = None;
    for address in addresses {
        match probe_address(address, &connect_host) {
            Ok(result) => return Ok(result),
            Err(err) => last_error = Some(err),
        }
    }
    Err(last_error.unwrap_or_else(|| {
        UpgradeError::Activation(format!("no addresses resolved for {connect_host}:{port}"))
    }))
}

pub fn wait_for_version(
    host: &str,
    port: u16,
    expected_version: &str,
    timeout: Duration,
) -> Result<HealthObservation, UpgradeError> {
    let deadline = Instant::now() + timeout;
    loop {
        let last_diagnostic = match probe(host, port) {
            Ok(observation) if observation.version == expected_version => return Ok(observation),
            Ok(observation) => format!(
                "health reported version '{}', expected '{expected_version}'",
                observation.version
            ),
            Err(err) => err.to_string(),
        };
        if Instant::now() >= deadline {
            return Err(UpgradeError::Activation(format!(
                "timed out waiting for http://{}:{port}/health: {last_diagnostic}",
                connect_host(host)
            )));
        }
        std::thread::sleep(Duration::from_millis(200));
    }
}

fn probe_address(
    address: SocketAddr,
    host_header: &str,
) -> Result<HealthObservation, UpgradeError> {
    let timeout = Duration::from_millis(750);
    let mut stream = TcpStream::connect_timeout(&address, timeout).map_err(|err| {
        UpgradeError::Activation(format!("cannot connect to http://{address}/health: {err}"))
    })?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;
    let host_header = format_host_header(host_header, address.port());
    write!(
        stream,
        "GET /health HTTP/1.0\r\nHost: {host_header}\r\nConnection: close\r\n\r\n"
    )?;

    let mut response = Vec::new();
    let mut chunk = [0_u8; 4096];
    while response.len() < 64 * 1024 {
        match stream.read(&mut chunk) {
            Ok(0) => break,
            Ok(read) => response.extend_from_slice(&chunk[..read]),
            Err(err)
                if err.kind() == std::io::ErrorKind::ConnectionReset && !response.is_empty() =>
            {
                break;
            }
            Err(err) => return Err(err.into()),
        }
    }
    let response = String::from_utf8(response)
        .map_err(|err| UpgradeError::Activation(format!("health response is not UTF-8: {err}")))?;
    let (headers, body) = response
        .split_once("\r\n\r\n")
        .or_else(|| response.split_once("\n\n"))
        .ok_or_else(|| UpgradeError::Activation("health response has no HTTP body".into()))?;
    let status = headers.lines().next().unwrap_or_default();
    let status_code = status.split_whitespace().nth(1).unwrap_or_default();
    if status_code != "200" {
        return Err(UpgradeError::Activation(format!(
            "health endpoint returned HTTP {status_code}"
        )));
    }
    let body: HealthBody = serde_json::from_str(body).map_err(|err| {
        UpgradeError::Activation(format!("health endpoint returned invalid JSON: {err}"))
    })?;
    if body.status != "ok" || body.server != "obsidian-mcp" || body.version.trim().is_empty() {
        return Err(UpgradeError::Activation(
            "health endpoint identity is not obsidian-mcp/status=ok".into(),
        ));
    }
    Ok(HealthObservation {
        version: body.version,
    })
}

fn format_host_header(host: &str, port: u16) -> String {
    if host.parse::<std::net::Ipv6Addr>().is_ok() {
        format!("[{host}]:{port}")
    } else {
        format!("{host}:{port}")
    }
}

fn connect_host(host: &str) -> String {
    match host.trim() {
        "" | "0.0.0.0" => "127.0.0.1".to_string(),
        "::" | "[::]" => "::1".to_string(),
        value if value.starts_with('[') && value.ends_with(']') => {
            value[1..value.len() - 1].to_string()
        }
        value => value.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::thread;

    fn one_shot_server(response: &'static str) -> u16 {
        let listener = TcpListener::bind(("127.0.0.1", 0)).expect("listener should bind");
        let port = listener
            .local_addr()
            .expect("listener should have addr")
            .port();
        thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("server should accept");
            let mut request = [0_u8; 512];
            let _ = stream.read(&mut request);
            stream
                .write_all(response.as_bytes())
                .expect("response should write");
        });
        port
    }

    #[test]
    fn probe_requires_exact_health_identity() {
        let port = one_shot_server(
            "HTTP/1.0 200 OK\r\nContent-Type: application/json\r\n\r\n\
             {\"status\":\"ok\",\"server\":\"obsidian-mcp\",\"version\":\"2.5.0\"}",
        );
        assert_eq!(
            probe("0.0.0.0", port).expect("health should pass").version,
            "2.5.0"
        );
    }

    #[test]
    fn probe_rejects_lookalike_json() {
        let port = one_shot_server(
            "HTTP/1.0 200 OK\r\n\r\n\
             {\"status\":\"ok\",\"server\":\"other\",\"version\":\"2.5.0\"}",
        );
        assert!(probe("127.0.0.1", port).is_err());
    }

    #[test]
    fn host_header_brackets_ipv6_literals() {
        assert_eq!(format_host_header("::1", 37842), "[::1]:37842");
        assert_eq!(format_host_header("127.0.0.1", 37842), "127.0.0.1:37842");
    }
}
