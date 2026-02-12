# Qwen3-TTS Rust

[中文](README.md) | [English](docs/README_EN.md) | [日本語](docs/README_JA.md) | [Korean](docs/README_KO.md) | [Français](docs/README_FR.md) | [Español](docs/README_ES.md) | [Italiano](docs/README_IT.md) | [Deutsch](docs/README_DE.md) | [Русский](docs/README_RU.md) | [Português](docs/README_PT.md)

本项目是 Qwen3-TTS 的极致性能实现，核心突破在于 **“指令驱动 (Instruction-Driven)”** 与 **“零样本自定义音色 (Custom Speakers)”** 的深度集成。通过 Rust 的内存安全特性与 llama.cpp/ONNX 的高效推理，为您提供工业级的文本转语音解决方案。

## 🚀 核心特性

### 1. 极致性能与流式响应
- **并发流式解码**：采用 4 帧 (64 codes) 粒度的并发解码策略，首字延迟低至 300ms，实现“边想边说”的流畅体验。
- **硬件加速**：默认启用 **Vulkan** (Windows/Linux) 和 **Metal** (macOS) 加速，显著提升推理速度。
- **自动运行时管理**：零配置环境，自动下载并配置 `llama.cpp` (b7885) 和 `onnxruntime`，开箱即用。

### 2. 灵活的说话人管理
- **自动扫描与缓存**：启动时自动加载 `speakers/` 目录下的音色文件。
- **多种选择方式**：支持通过 CLI 参数 `--speaker <name>` 或 `--voice-file <path>` 灵活选择说话人。
- **智能回退**：若指定说话人不存在，自动回退至默认音色 (vivian)，确保系统稳定性。

### 3. 精准的指令控制
- **指令驱动**：支持在文本中嵌入 `[高兴]`、`[悲伤]` 等情感指令，实时调整演绎风格。
- **EOS 对齐**：完美对齐 Qwen3 的停止逻辑，支持多种 EOS token 检测，杜绝生成末尾的静音或乱码。

## 📊 性能基准 (Benchmarks)

| Backend | Model (GGUF) | RTF (Real-Time Factor) | Avg Time (ms) | Avg Audio (s) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CUDA** | Q5_K_M | **0.553** | 1162.6 | 2.19 | OK |
| **Vulkan** | Q5_K_M | 0.598 | 1285.4 | 2.19 | OK |
| **CUDA** | Q8_0 | 0.640 | 1523.4 | 2.44 | OK |
| **Vulkan** | Q8_0 | 0.638 | 1502.0 | 2.44 | OK |
| **CPU** | Q5_K_M | 1.677 | 2823.4 | 1.96 | OK |
| **CPU** | Q8_0 | 1.866 | 4160.1 | 2.51 | OK |

- **测试环境**: Intel Core i9-13980HX, NVIDIA RTX 2080 Ti. 显存占用约 0.7-1.5GB.
- **数据来源**: Windows 平台 10 轮生成平均值。
- **最佳性能**: RTF 0.553 (CUDA + Q5KM)，即生成 1 秒音频仅需 0.553 秒。

## 🛠️ 快速上手

### 1. 基础生成
使用默认说话人生成语音：
```powershell
cargo run --bin qwen3_tts -- --text "你好，欢迎使用 Qwen3-TTS Rust！"
```

### 2. 指定说话人
使用预设或自定义说话人：
```powershell
# 使用名称 (需在 speakers/ 目录下存在对应的 .json 文件)
cargo run --bin qwen3_tts -- --text "今天天气不错。" --speaker dylan

# 使用指定文件路径
cargo run --bin qwen3_tts -- --text "我是自定义音色。" --voice-file "path/to/my_voice.json"
```

### 3. 克隆新音色
只需 3-10 秒的参考音频即可克隆音色：
```powershell
cargo run --bin qwen3_tts -- `
    --ref-audio "ref.wav" `
    --ref-text "参考音频对应的文本内容" `
    --save-voice "speakers/my_voice.json" `
    --text "新音色已保存，现在可以直接使用了！"
```

### 4. 风格/情感控制 (Instruction)
通过 `--instruction` 参数实时改变说话语气：
```powershell
# 悲伤语气
cargo run --bin qwen3_tts -- --text "对不起，我不是故意的..." --instruction "悲伤啜泣，非常难过"

# 开心语气
cargo run --bin qwen3_tts -- --text "太棒了！我们成功了！" --instruction "开心激动，语速稍快"
```

### 5. 高级配置
```powershell
cargo run --bin qwen3_tts -- `
    --text "长文本生成测试。" `
    --max-steps 1024 `    # 调整最大生成长度
    --output "output.wav" # 指定输出文件名
```

## 📂 目录结构

系统首次运行会自动构建如下结构：

```text
.
├── models/             # 模型文件 (GGUF, ONNX, Tokenizer)
├── runtime/            # 自动下载的依赖库 (dll, so, dylib)
└── speakers/           # 用户自定义音色
```

## 📜 许可证与致谢

- 基于 **MIT / Apache 2.0** 许可证。
- 感谢 [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) 官方仓库提供的模型与技术基座。
- 感谢 [Qwen3-TTS-GGUF](https://github.com/HaujetZhao/Qwen3-TTS-GGUF) 提供的推理流程启发。
