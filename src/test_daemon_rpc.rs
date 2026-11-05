use serde_json::{Value, json};
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncWrite, AsyncWriteExt};

pub(crate) async fn request(reader: &mut (impl AsyncBufRead + Unpin)) -> Value {
    let mut line = String::new();
    reader.read_line(&mut line).await.unwrap();
    serde_json::from_str(&line).unwrap()
}

/// Read the content request after verifying capability negotiation on its connection.
pub(crate) async fn scoped_request(
    reader: &mut (impl AsyncBufRead + Unpin),
    writer: &mut (impl AsyncWrite + Unpin),
) -> Value {
    let probe = request(reader).await;
    assert_eq!(probe["method"], "health");
    assert_eq!(probe["params"]["min_api_version"], 2);
    assert_eq!(probe["params"]["max_api_version"], 2);
    let response = json!({
        "jsonrpc":"2.0", "id":probe["id"],
        "result":{
            "daemon_version":env!("CARGO_PKG_VERSION"), "daemon_api_version":2, "pid":1,
            "status":"ok", "uptime_ms":1, "model_name":"test", "semantic_home":"/tmp/test",
            "capabilities":["semantic_source_evidence_v1", "scoped_indexing_v1"]
        },
    });
    writer
        .write_all(format!("{response}\n").as_bytes())
        .await
        .unwrap();
    request(reader).await
}
