# Qwen3-TTS Rust

[中文](README.md) | [English](docs/README_EN.md) | [日本語](docs/README_JA.md) | [한국어](docs/README_KO.md) | [Deutsch](docs/README_DE.md) | [Français](docs/README_FR.md) | [Русский](docs/README_RU.md) | [Português](docs/README_PT.md) | [Español](docs/README_ES.md) | [Italiano](docs/README_IT.md)


本项目是 Qwen3-TTS 的 Rust 实现，基于 ONNX Runtime 和 llama.cpp (GGUF)，旨在提供高性能、易集成的文本转语音能力。

## 特性
- **高性能架构**: 核心逻辑使用 Rust 编写。LLM 推理基于 **llama.cpp**，支持 **CPU、CUDA、Vulkan** 等多后端及模型量化 (Q4/F16)。
- **流式解码**: 音频解码采用 **ONNX Runtime (CPU)** 进行流式输出，实现极速响应。
- **音色克隆**: 支持通过参考音频进行零样本 (Zero-shot) 音色克隆。

## 性能表现

| 硬件环境 | 模型量化 | RTF (实时率) | 平均耗时 (10轮) |
|--------|---------|--------------|----------------|
| CPU | Int4 (Q4) | 1.144 | ~4.44s |
| CPU | F16 | 2.664 | ~9.47s |
| CUDA | Int4 (Q4) | 0.608 | ~2.25s |
| CUDA | F16 | 0.715 | ~2.60s |
| Vulkan | Int4 (Q4) | 0.606 | ~2.30s |
| Vulkan | F16 | 0.717 | ~2.87s |

> **测试环境**: Intel Core i9-13980HX, NVIDIA RTX 2080 Ti. 显存占用约 2GB.
> 数据基于 Windows 平台 10 轮生成平均值。

## 快速开始

### 1. 准备运行环境 (Windows)
你需要在项目目录下放置相关的运行时 DLL。
1. 下载 [ONNX Runtime](https://github.com/microsoft/onnxruntime/releases) (推荐 v1.23.2)。
2. 运行 `assets/download_dlls.ps1` 脚本自动下载并安装 ONNX Runtime (CPU 版本)。

### 2. 准备模型资源
运行提供的 Python 脚本下载预训练模型：
```bash
python assets/download_models.py
```
模型将保存到 `models/` 目录。

> **注意**：我们会在这几天上传转换好的模型文件，敬请期待。

### 3. 模型转换 (进阶)
如果你有原始的 PyTorch Checkpoint，可以使用 `assets/scripts` 下的工具进行转换：

**1. 导出 ONNX (Codec, Speaker, Decoder):**
```bash
python assets/scripts/export_codec_encoder.py --checkpoint Qwen3-TTS.pt --output models/qwen3_tts_codec_encoder.onnx
python assets/scripts/export_speaker_encoder.py --checkpoint Qwen3-TTS.pt --output models/qwen3_tts_speaker_encoder.onnx
python assets/scripts/export_decoder.py --checkpoint Qwen3-TTS.pt --output models/qwen3_tts_decoder.onnx
```

**2. 转换 GGUF (Talker, Predictor):**
```bash
python assets/scripts/convert_talker_gguf.py --checkpoint Qwen3-TTS.pt --output models/qwen3_tts_talker-q4km.gguf --quantize q4_k_m
python assets/scripts/convert_predictor_gguf.py --checkpoint Qwen3-TTS.pt --output models/qwen3_tts_predictor-q4km.gguf --quantize q4_k_m
```

### 4. 资产转换 (可选)
如果需要将散落的 `.npy` 资产文件打包为单一的 `qwen3_assets.gguf` 文件（推荐）：
1. 安装依赖：`pip install numpy gguf`
2. 运行转换脚本：
```bash
python assets/convert_assets.py --input_dir /path/to/npy/files --output_file models/qwen3_assets.gguf
```
引擎会自动优先加载 `qwen3_assets.gguf`。


### 3. 音色管理 (New)
我们推荐预先提取音色特征保存为 `.qvoice` 文件，以便重复使用。

**提取音色：**
```powershell
    $env:PATH += ";$PWD\runtime"
    cargo run --example make_voice --release -- `
        --model_dir ./models `
        --input clone.wav `
        --text "参考音频的文字内容" `
        --output my_voice.qvoice `
        --name "我的专属音色" `
        --gender "Female" `
        --age "青年" `
        --description "清晰、温柔的解说音色"
```

**使用音色包生成：**
```powershell
cargo run --example qwen3-tts --release -- --model_dir ./models --voice my_voice.qvoice --text "你好，世界"
```

### 4. 快速演示
使用 `run.ps1` 脚本运行演示（自动处理 DLL 路径）：
```powershell
.\run.ps1 --input clone.wav --ref_text "参考音频的文字内容" --text "你好，世界"
```

或者手动运行（需确保 `runtime` 在 PATH 中）：
```bash
$env:PATH += ";$PWD\runtime"
cargo run --example qwen3-tts --release -- --model_dir ./models --input clone.wav --ref_text "参考音频的文字内容" --text "你好，世界"
```

## 作为库引用
在你的 `Cargo.toml` 中增加：
```toml
[dependencies]
qwen3-tts = { path = "../path/to/qwen3-tts-rust" }
```

### 完整示例代码
```rust
use qwen3_tts::TtsEngine;
use std::path::Path;

fn main() -> Result<(), String> {
    // 1. 初始化引擎
    // 需要指定包含 models (onnx/gguf) 和 assets (tokenizer.json 等) 的目录
    let model_dir = Path::new("models");
    let mut engine = TtsEngine::load(model_dir)?;

    // 2. 准备输入
    let text = "你好，我是 Qwen3-TTS 的 Rust 实现。";
    let ref_audio = Path::new("clone.wav"); // 参考音频路径
    let ref_text = "这是参考音频对应的文字内容。"; // 参考音频的文本

    // 3. 生成音频
    // 返回 AudioSample 结构，包含 samples (Vec<f32>) 和 sample_rate
    let audio = engine.generate(text, ref_audio, ref_text)?;

    // 4. 保存或播放
    audio.save_wav("output.wav")?;
    
    // 5. (可选) 清理资源
    qwen3_tts::cleanup();
    
    Ok(())
}
```

### API 说明
- **`TtsEngine::load(path)`**: 加载所有必要的模型资源。耗时较长 (几秒)，建议应用启动时只需一次。
- **`engine.generate(text, ref_audio, ref_text)`**: 执行推理。
    - `text`: 目标合成文本。
    - `ref_audio`: 参考音频路径 (WAV格式, 16kHz+ 推荐)。
    - `ref_text`: 参考音频的对应文本 (对于中文，准确的参考文本对音色还原至关重要)。
- **`audio.save_wav(path)`**: 辅助方法，保存为 WAV 文件。

## 开源准备项
- [x] 代码模块化重构
- [x] 移除硬编码路径
- [x] 准备下载脚本
- [x] 准备转换脚本
- [ ] 导出 C API (可选)

## 致谢
感谢以下项目对本实现的启发和支持：
- [Qwen3-TTS-GGUF](https://github.com/HaujetZhao/Qwen3-TTS-GGUF): 本项目参考了其 GGUF 推理流程。
- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS): Qwen3-TTS 的官方仓库。

## 许可证
MIT / Apache 2.0
