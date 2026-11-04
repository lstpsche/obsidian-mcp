//! Exact capability profiles for isolated embedding API fixtures.

use std::path::{Path, PathBuf};

use serde_json::json;

pub fn write_api_profile(
    directory: &Path,
    api_port: u16,
    model: &str,
    max_input_tokens: usize,
    max_batch_inputs: usize,
) -> PathBuf {
    std::fs::create_dir_all(directory).expect("create profile directory");
    let path = directory.join(format!("{model}-profile.json"));
    let profile = json!({
        "version": 1,
        "profile_revision": "synthetic-v1",
        "api_base": format!("http://127.0.0.1:{api_port}/v1"),
        "model": model,
        "tokenizer": {"kind": "cl100k_base"},
        "max_input_tokens": max_input_tokens,
        "max_batch_inputs": max_batch_inputs,
        "max_batch_tokens": max_input_tokens * max_batch_inputs
    });
    std::fs::write(&path, serde_json::to_vec(&profile).unwrap()).expect("write API profile");
    path
}
