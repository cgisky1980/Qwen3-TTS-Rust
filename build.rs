use std::env;
use std::path::PathBuf;

fn main() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();

    if target_os == "linux" {
        // Set the RPATH for the binary
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/runtime");

        // Optional: Automatically link the runtime folder into the target directory
        // so 'cargo run' works without extra RPATHs
        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
        let profile_dir = out_dir.join("../../../"); // Points to target/debug or target/release
        let project_runtime =
            PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("runtime");
        let target_runtime = profile_dir.join("runtime");

        if project_runtime.exists() && !target_runtime.exists() {
            #[cfg(unix)]
            let _ = std::os::unix::fs::symlink(project_runtime, target_runtime);
        }
    }
}
