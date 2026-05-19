fn main() {
    let has_local = std::env::var("CARGO_FEATURE_EMBEDDINGS").is_ok();
    let has_api = std::env::var("CARGO_FEATURE_EMBEDDINGS_API").is_ok();

    if has_local || has_api {
        println!("cargo:rustc-cfg=has_embeddings");
    }

    println!("cargo:rustc-check-cfg=cfg(has_embeddings)");
    println!("cargo:rerun-if-changed=build.rs");
}
