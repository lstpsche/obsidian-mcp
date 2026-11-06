use reqwest::{Client, RequestBuilder, Response, StatusCode};
use serde_json::{Value, json};

pub const MODERN: &str = "2026-07-28";

pub fn request(
    client: &Client,
    url: &str,
    method: &str,
    mut params: Value,
    version: &str,
) -> RequestBuilder {
    if version == MODERN {
        params["_meta"] = json!({
            "io.modelcontextprotocol/protocolVersion": MODERN,
            "io.modelcontextprotocol/clientInfo": {"name": "openai-mcp", "version": "1.0.0"},
            "io.modelcontextprotocol/clientCapabilities": {
                "experimental": {"openai/visibility": {"enabled": true}},
                "extensions": {"io.modelcontextprotocol/ui": {"mimeTypes": ["text/html;profile=mcp-app"]}}
            }
        });
    }
    let mut request = client
        .post(format!("{url}/mcp"))
        .header("Content-Type", "application/json")
        .header("Accept", "application/json, text/event-stream")
        .header("Mcp-Protocol-Version", version)
        .header("User-Agent", "openai-mcp/1.0.0");
    if version == MODERN {
        request = request.header("Mcp-Method", method);
        if method == "tools/call" {
            request = request.header("Mcp-Name", params["name"].as_str().unwrap());
        }
    }
    request.body(
        json!({"jsonrpc": "2.0", "id": "transport-test", "method": method, "params": params})
            .to_string(),
    )
}

pub async fn rpc_response(response: Response) -> Value {
    let status = response.status();
    let content_type = response
        .headers()
        .get("content-type")
        .map(|value| value.to_str().unwrap().to_owned());
    let body = response.text().await.unwrap();
    assert_eq!(status, StatusCode::OK, "{body}");
    let value = match content_type.as_deref() {
        Some("application/json") => serde_json::from_str::<Value>(&body).unwrap(),
        Some("text/event-stream") => {
            let messages: Vec<Value> = body
                .lines()
                .filter_map(|line| line.strip_prefix("data: "))
                .filter(|data| !data.is_empty())
                .map(|data| serde_json::from_str(data).unwrap())
                .collect();
            assert_eq!(messages.len(), 1, "{body}");
            messages.into_iter().next().unwrap()
        }
        other => panic!("unexpected content type {other:?}: {body}"),
    };
    assert_eq!(value["id"], "transport-test");
    value
}
