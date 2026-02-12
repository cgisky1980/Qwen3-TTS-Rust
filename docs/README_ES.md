# Qwen3-TTS Rust

[简体中文](../README.md) | [English](README_EN.md) | [日本語](README_JA.md) | [Korean](README_KO.md) | [Français](README_FR.md) | [Español](README_ES.md) | [Italiano](README_IT.md) | [Deutsch](README_DE.md) | [Русский](README_RU.md) | [Português](README_PT.md)

This project is the ultimate performance implementation of Qwen3-TTS. The core breakthroughs are the deep integration of **"Instruction-Driven"** synthesis and **"Zero-shot Custom Speakers (Cloning)"**. Leveraging Rust's memory safety and the efficient inference of llama.cpp/ONNX, it provides an industrial-grade text-to-speech solution.

## 🚀 Core Features

### 1. Extreme Performance & Streaming
- **Concurrent Streaming Decoding**: Uses a 4-frame (64 codes) granularity for concurrent decoding, determining first-token latency as low as 300ms for a "speech-while-thinking" experience.
- **Hardware Acceleration**: **Vulkan** (Windows/Linux) and **Metal** (macOS) acceleration are enabled by default, significantly boosting inference speed.
- **Automatic Runtime Management**: Zero-config environment; automatically downloads and configures `llama.cpp` (b7885) and `onnxruntime`, ready to use out of the box.

### 2. Flexible Speaker Management
- **Auto-Scan & Cache**: Automatically loads voice files from the `speakers/` directory on startup.
- **Versatile Selection**: Supports flexible speaker selection via CLI arguments `--speaker <name>` or `--voice-file <path>`.
- **Smart Fallback**: Automatically falls back to the default voice (vivian) if the specified speaker is not found, ensuring system stability.

### 3. Precise Instruction Control
- **Instruction-Driven**: Supports embedding emotion tags like `[Happy]`, `[Sad]` in the text to adjust the delivery style in real-time.
- **EOS Alignment**: Perfectly aligned with Qwen3's stop logic, supporting multiple EOS token detections to prevent generation of trailing silence or artifacts.

## 📊 Benchmarks

| Backend | Model (GGUF) | RTF (Real-Time Factor) | Avg Time (ms) | Avg Audio (s) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CUDA** | Q5_K_M | **0.553** | 1162.6 | 2.19 | OK |
| **Vulkan** | Q5_K_M | 0.598 | 1285.4 | 2.19 | OK |
| **CUDA** | Q8_0 | 0.640 | 1523.4 | 2.44 | OK |
| **Vulkan** | Q8_0 | 0.638 | 1502.0 | 2.44 | OK |
| **CPU** | Q5_K_M | 1.677 | 2823.4 | 1.96 | OK |
| **CPU** | Q8_0 | 1.866 | 4160.1 | 2.51 | OK |

- **Test Environment**: Intel Core i9-13980HX, NVIDIA RTX 2080 Ti. VRAM Usage ~0.7-1.5GB.
- **Data Source**: Average of 10 generations on Windows.
- **Best Performance**: RTF 0.553 (CUDA + Q5KM), meaning 1 second of audio takes only 0.553 seconds to generate.

## 🛠️ Quick Start

### 1. Basic Generation
Generate speech using the default speaker:
```powershell
cargo run --example qwen3-tts -- --text "Hello, welcome to use Qwen3-TTS Rust!"
```

### 2. Specify Speaker
Use a preset or custom speaker:
```powershell
# Use name (requires corresponding .json file in speakers/ directory)
cargo run --example qwen3-tts -- --text "The weather is nice today." --speaker dylan

# Use specific file path
cargo run --example qwen3-tts -- --text "I am a custom voice." --voice-file "path/to/my_voice.json"
```

### 3. Clone New Voice
Clone a voice with just 3-10 seconds of reference audio:
```powershell
cargo run --example qwen3-tts -- `
    --ref-audio "ref.wav" `
    --ref-text "The text content corresponding to the reference audio" `
    --save-voice "speakers/my_voice.json" `
    --text "New voice saved, you can use it directly now!"
```

### 4. Advanced Configuration
```powershell
cargo run --example qwen3-tts -- `
    --text "Long text generation test." `
    --max-steps 1024 `    # Adjust max generation length
    --output "output.wav" # 指定 output filename
```

## 📂 Directory Structure

The system automatically builds the following structure on first run:

```text
.
├── models/             # Model files (GGUF, ONNX, Tokenizer)
├── runtime/            # Auto-downloaded dependencies (dll, so, dylib)
└── speakers/           # User custom voices
```

## 📜 License & Acknowledgements

- Based on **MIT / Apache 2.0** license.
- Thanks to [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) official repository for models and technical foundation.
- Thanks to [Qwen3-TTS-GGUF](https://github.com/HaujetZhao/Qwen3-TTS-GGUF) for inspiration on the GGUF inference flow.
