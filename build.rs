fn main() {
    let has_local = std::env::var("CARGO_FEATURE_EMBEDDINGS").is_ok();
    let has_api = std::env::var("CARGO_FEATURE_EMBEDDINGS_API").is_ok();

    let mut package_features = std::env::var("CARGO_CFG_FEATURE")
        .unwrap_or_default()
        .split(',')
        .map(str::trim)
        .filter(|feature| !feature.is_empty() && *feature != "default")
        .map(str::to_owned)
        .collect::<Vec<_>>();
    package_features.sort_unstable();
    package_features.dedup();

    let target = std::env::var("TARGET").expect("Cargo must provide TARGET to build scripts");

    if has_local || has_api {
        println!("cargo:rustc-cfg=has_embeddings");
    }

    println!(
        "cargo:rustc-env=OBSIDIAN_MCP_BUILD_FEATURES={}",
        package_features.join(",")
    );
    println!("cargo:rustc-env=OBSIDIAN_MCP_BUILD_TARGET={target}");
    println!("cargo:rustc-check-cfg=cfg(has_embeddings)");
    println!("cargo:rerun-if-changed=build.rs");
}
