use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;
use std::fs::File;
use std::io::Write;
use std::path::Path;

pub struct Downloader {
    client: Client,
    model_base_url: String,
}

impl Downloader {
    pub async fn new() -> Self {
        let client = Client::new();
        // Check connectivity to huggingface.co for MODELS
        let hf_base = if Self::check_connectivity(&client, "https://huggingface.co").await {
            "https://huggingface.co"
        } else {
            "https://hf-mirror.com"
        };

        let model_base_url = format!("{}/cgisky/qwen3-tts-custom-gguf/resolve/main", hf_base);

        Self {
            client,
            model_base_url,
        }
    }

    async fn check_connectivity(client: &Client, url: &str) -> bool {
        let result = client
            .head(url)
            .timeout(std::time::Duration::from_secs(3))
            .send()
            .await;
        result.is_ok()
    }

    /// Check and download both models and runtime libraries
    pub async fn check_and_download(
        &self,
        model_dir: &Path,
        quant: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // 1. Check Models
        self.download_models(model_dir, quant).await?;

        // 2. Check Runtime Libraries (DLLs/SOs)
        self.download_runtimes().await?;

        Ok(())
    }

    async fn download_models(
        &self,
        model_dir: &Path,
        quant: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let quant_folder = match quant {
            "q5_k_m" => "gguf_q5_k_m",
            "q8_0" => "gguf_q8_0",
            _ => "gguf",
        };

        let files = vec![
            (
                "onnx/qwen3_tts_decoder.onnx".to_owned(),
                "onnx/qwen3_tts_decoder.onnx".to_owned(),
            ),
            (
                "tokenizer/tokenizer.json".to_owned(),
                "tokenizer/tokenizer.json".to_owned(),
            ),
            (
                format!("{}/qwen3_assets.gguf", quant_folder),
                format!("{}/qwen3_assets.gguf", quant_folder),
            ),
            (
                format!("{}/qwen3_tts_talker.gguf", quant_folder),
                format!("{}/qwen3_tts_talker.gguf", quant_folder),
            ),
            (
                format!("{}/qwen3_tts_predictor.gguf", quant_folder),
                format!("{}/qwen3_tts_predictor.gguf", quant_folder),
            ),
        ];

        for (remote, local) in files {
            let local_path = model_dir.join(&local);
            if !local_path.exists() {
                if let Some(p) = local_path.parent() {
                    std::fs::create_dir_all(p)?;
                }
                let url = format!("{}/{}", self.model_base_url, remote);
                println!("Downloading Model: {}...", local);
                self.download_to_file(&url, &local_path).await?;
            }
        }
        Ok(())
    }

    pub async fn download_runtimes(&self) -> Result<(), Box<dyn std::error::Error>> {
        let runtime_dir = Path::new("runtime");
        if !runtime_dir.exists() {
            std::fs::create_dir_all(runtime_dir)?;
        }

        let os = if cfg!(target_os = "windows") {
            "win"
        } else if cfg!(target_os = "linux") {
            "linux"
        } else if cfg!(target_os = "macos") {
            "osx"
        } else {
            return Err("Unsupported OS for auto-runtime download".into());
        };

        let arch = if cfg!(target_arch = "x86_64") {
            "x64"
        } else if cfg!(target_arch = "aarch64") {
            "arm64"
        } else {
            return Err("Unsupported Architecture for auto-runtime download".into());
        };

        // Determine Backend
        let backend = if cfg!(feature = "cuda") {
            "cuda"
        } else if cfg!(target_os = "macos") {
            "" // macOS uses Metal integrated in the binary
        } else {
            "vulkan"
        };

        // Platform mapping for llama.cpp release assets
        // Windows -> win, Linux -> ubuntu, macOS -> macos
        let release_os = if cfg!(target_os = "windows") {
            "win"
        } else if cfg!(target_os = "macos") {
            "macos"
        } else {
            "ubuntu"
        };

        let release_ext = if cfg!(target_os = "windows") {
            "zip"
        } else {
            "tar.gz"
        };

        // 1. ONNX Runtime
        let ort_version = "1.23.2";
        let ort_ext = if os == "win" { "zip" } else { "tgz" };
        let ort_filename = format!("onnxruntime-{}-{}-{}.{}", os, arch, ort_version, ort_ext);
        let ort_dll_name = if os == "win" {
            "onnxruntime.dll"
        } else if os == "osx" {
            "libonnxruntime.dylib"
        } else {
            "libonnxruntime.so"
        };

        if !runtime_dir.join(ort_dll_name).exists() {
            println!("Downloading ONNX Runtime ({}-{})...", os, arch);
            let ort_url = format!(
                "https://github.com/microsoft/onnxruntime/releases/download/v{}/{}",
                ort_version, ort_filename
            );
            let tmp_zip = Path::new("ort_runtime.tmp");
            self.download_to_file(&ort_url, tmp_zip).await?;

            println!("Extracting ONNX Runtime...");
            if ort_ext == "zip" {
                self.extract_zip(
                    tmp_zip,
                    runtime_dir,
                    &format!("onnxruntime-{}-{}-{}", os, arch, ort_version),
                    "lib",
                )?;
            } else {
                self.extract_targz(
                    tmp_zip,
                    runtime_dir,
                    &format!("onnxruntime-{}-{}-{}", os, arch, ort_version),
                    "lib",
                )?;
            }
            let _ = std::fs::remove_file(tmp_zip);
        }

        // 2. Llama.cpp / GGML (Official Release)
        let llama_dll = if os == "win" {
            "llama.dll"
        } else if os == "osx" {
            "libllama.dylib"
        } else {
            "libllama.so"
        };

        if !runtime_dir.join(llama_dll).exists() {
            println!(
                "Downloading Llama.cpp Runtimes (Official {} vB7885)...",
                if backend.is_empty() { "Metal" } else { backend }
            );
            // URL Patterns:
            // win: llama-b7885-bin-win-vulkan-x64.zip
            // linux: llama-b7885-bin-ubuntu-vulkan-x64.tar.gz
            // macos: llama-b7885-bin-macos-arm64.tar.gz
            let asset_name = if backend.is_empty() {
                format!("llama-b7885-bin-{}-{}.{}", release_os, arch, release_ext)
            } else {
                format!(
                    "llama-b7885-bin-{}-{}-{}.{}",
                    release_os, backend, arch, release_ext
                )
            };

            let llama_url = format!(
                "https://github.com/ggerganov/llama.cpp/releases/download/b7885/{}",
                asset_name
            );

            let tmp_zip = Path::new("llama_runtime.tmp");
            if let Err(e) = self.download_to_file(&llama_url, tmp_zip).await {
                eprintln!(
                    "Warning: Failed to download llama runtime: {}. Please install manually.",
                    e
                );
            } else {
                if release_ext == "zip" {
                    self.extract_zip(tmp_zip, runtime_dir, "", "")?;
                } else {
                    self.extract_targz(tmp_zip, runtime_dir, "", "")?;
                }
                let _ = std::fs::remove_file(tmp_zip);
            }
        }

        Ok(())
    }

    async fn download_to_file(
        &self,
        url: &str,
        path: &Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let res = self.client.get(url).send().await?;
        if !res.status().is_success() {
            return Err(format!("HTTP {} for {}", res.status(), url).into());
        }
        let total = res.content_length().unwrap_or(0);
        let pb = ProgressBar::new(total);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")?
            .progress_chars("#>-"));

        let mut file = File::create(path)?;
        let mut stream = res.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            file.write_all(&chunk)?;
            pb.inc(chunk.len() as u64);
        }
        pb.finish();
        Ok(())
    }

    fn extract_zip(
        &self,
        zip_path: &Path,
        dest: &Path,
        prefix: &str,
        lib_subdir: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::open(zip_path)?;
        let mut archive = zip::ZipArchive::new(file)?;
        for i in 0..archive.len() {
            let mut file = archive.by_index(i)?;
            let outpath = match file.enclosed_name() {
                Some(path) => path.to_owned(),
                None => continue,
            };

            // Filter by prefix and subdirectory if provided
            let path_str = outpath.to_string_lossy();
            if !prefix.is_empty() && !path_str.starts_with(prefix) {
                continue;
            }
            if !lib_subdir.is_empty() && !path_str.contains(lib_subdir) {
                // For ONNX, we only want the DLLs in the lib/ folder
                if !path_str.ends_with(".dll")
                    && !path_str.ends_with(".so")
                    && !path_str.ends_with(".dylib")
                {
                    continue;
                }
            }

            let file_name = outpath.file_name().unwrap();
            let dest_file = dest.join(file_name);

            if file.is_dir() {
                std::fs::create_dir_all(&dest_file)?;
            } else {
                let mut outfile = File::create(&dest_file)?;
                std::io::copy(&mut file, &mut outfile)?;
            }
        }
        Ok(())
    }

    fn extract_targz(
        &self,
        tar_path: &Path,
        dest: &Path,
        prefix: &str,
        _lib_subdir: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let tar_gz = File::open(tar_path)?;
        let tar = flate2::read::GzDecoder::new(tar_gz);
        let mut archive = tar::Archive::new(tar);
        for entry in archive.entries()? {
            let mut entry = entry?;
            let path = entry.path()?.into_owned();
            let path_str = path.to_string_lossy();

            if !prefix.is_empty() && !path_str.starts_with(prefix) {
                continue;
            }

            // Simplified check for libraries
            if path_str.ends_with(".so")
                || path_str.ends_with(".dylib")
                || path_str.ends_with(".dll")
            {
                let file_name = path.file_name().unwrap();
                let dest_file = dest.join(file_name);
                entry.unpack(dest_file)?;
            }
        }
        Ok(())
    }
}
