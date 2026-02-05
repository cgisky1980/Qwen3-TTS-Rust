# Qwen3-TTS Rust

[中文](README.md) | [English](README_EN.md) | [日本語](README_JA.md) | [한국어](README_KO.md) | [Deutsch](README_DE.md) | [Français](README_FR.md) | [Русский](README_RU.md) | [Português](README_PT.md) | [Español](README_ES.md) | [Italiano](README_IT.md)

Rust implementation of Qwen3-TTS, based on ONNX Runtime and llama.cpp (GGUF), designed to provide high-performance, easy-to-integrate text-to-speech capabilities.

## Features
- **High Performance Architecture**: Core logic written in Rust. LLM inference based on **llama.cpp**, supporting **CPU, CUDA, Vulkan** backends and model quantization (Q4/F16).
- **Streaming Decode**: Audio decoding uses **ONNX Runtime (CPU)** for streaming output, enabling ultra-fast response.
- **Voice Cloning**: Supports Zero-shot voice cloning via reference audio.

## Performance

| Device | Quantization | RTF (Real Time Factor) | Avg Time (10 runs) |
|--------|--------------|------------------------|--------------------|
| CPU | Int4 (Q4) | 1.144 | ~4.44s |
| CPU | F16 | 2.664 | ~9.47s |
| CUDA | Int4 (Q4) | 0.608 | ~2.25s |
| CUDA | F16 | 0.715 | ~2.60s |
| Vulkan | Int4 (Q4) | 0.606 | ~2.30s |
| Vulkan | F16 | 0.717 | ~2.87s |

> **Test Environment**: Intel Core i9-13980HX, NVIDIA RTX 2080 Ti. VRAM usage approx. 2GB.
> Data based on Windows platform, average of 10 runs.

## Quick Start

### 1. Prepare Environment (Windows)
You need to place the relevant runtime DLLs in the project directory.
1. Download [ONNX Runtime](https://github.com/microsoft/onnxruntime/releases) (v1.23.2 recommended).
2. Run `assets/download_dlls.ps1` script to automatically download and install ONNX Runtime (CPU version).

### 2. Prepare Models
Run the provided Python script to download pre-trained models:
```bash
python assets/download_models.py
```
Models will be saved to the `models/` directory.

### 3. Voice Management (New)
We recommend extracting voice features and saving them as `.qvoice` files for reuse.

**Extract Voice:**
```powershell
$env:PATH += ";$PWD\runtime"
cargo run --example make_voice --release -- `
    --model_dir ./models `
    --input clone.wav `
    --text "Text content of the reference audio" `
    --output my_voice.qvoice `
    --name "My Custom Voice" `
    --gender "Female" `
    --age "Young" `
    --description "Clear, gentle narration voice"
```

**Generate with Voice Pack:**
```powershell
cargo run --example qwen3-tts --release -- --model_dir ./models --voice my_voice.qvoice --text "Hello, world"
```

### 4. Quick Demo
Use the `run.ps1` script to run the demo (automatically handles DLL paths):
```powershell
.\run.ps1 --input clone.wav --ref_text "Text content of the reference audio" --text "Hello, world"
```

Or run manually (ensure `runtime` is in PATH):
```bash
$env:PATH += ";$PWD\runtime"
cargo run --example qwen3-tts --release -- --model_dir ./models --input clone.wav --ref_text "Text content of the reference audio" --text "Hello, world"
```

## Library Usage
Add to your `Cargo.toml`:
```toml
[dependencies]
qwen3-tts = { path = "../path/to/qwen3-tts-rust" }
```

### Example Code
```rust
use qwen3_tts::TtsEngine;
use std::path::Path;

fn main() -> Result<(), String> {
    // 1. Initialize Engine
    // Specify directory containing models (onnx/gguf) and assets (tokenizer.json etc.)
    let model_dir = Path::new("models");
    let mut engine = TtsEngine::load(model_dir)?;

    // 2. Prepare Input
    let text = "Hello, this is Qwen3-TTS Rust implementation.";
    let ref_audio = Path::new("clone.wav"); // Reference audio path
    let ref_text = "Text content of reference audio."; // Reference audio text

    // 3. Generate Audio
    // Returns AudioSample struct containing samples (Vec<f32>) and sample_rate
    let audio = engine.generate(text, ref_audio, ref_text)?;

    // 4. Save or Play
    audio.save_wav("output.wav")?;
    
    // 5. (Optional) Cleanup
    qwen3_tts::cleanup();
    
    Ok(())
}
```

### API Reference
- **`TtsEngine::load(path)`**: Loads all necessary model resources. Takes a few seconds, recommended to call once at startup.
- **`engine.generate(text, ref_audio, ref_text)`**: Performs inference.
    - `text`: Target text to synthesize.
    - `ref_audio`: Path to reference audio (WAV format, 16kHz+ recommended).
    - `ref_text`: Corresponding text of the reference audio (Critical for accurate voice cloning, especially for Chinese).
- **`audio.save_wav(path)`**: Helper method to save as WAV file.

## To-Do
- [x] Modular code refactoring
- [x] Remove hardcoded paths
- [x] Prepare download scripts
- [x] Prepare conversion scripts
- [ ] Export C API (Optional)

## Acknowledgements
Thanks to the following projects for inspiration and support:
- [Qwen3-TTS-GGUF](https://github.com/HaujetZhao/Qwen3-TTS-GGUF): Referenced its GGUF inference flow.
- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS): Official repository for Qwen3-TTS.

## License
MIT / Apache 2.0
