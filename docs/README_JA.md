# Qwen3-TTS Rust

[中文](../README.md) | [English](README_EN.md) | [日本語](README_JA.md) | [한국어](README_KO.md) | [Deutsch](README_DE.md) | [Français](README_FR.md) | [Русский](README_RU.md) | [Português](README_PT.md) | [Español](README_ES.md) | [Italiano](README_IT.md)

ONNX Runtime と llama.cpp (GGUF) に基づく Qwen3-TTS の Rust 実装です。高性能で統合しやすいテキスト読み上げ機能を提供することを目指しています。

## 特徴
- **高性能アーキテクチャ**: コアロジックは Rust で記述。LLM 推論は **llama.cpp** に基づき、**CPU、CUDA、Vulkan** バックエンドおよびモデル量子化 (Q4/F16) をサポート。
- **ストリーミングデコード**: 音声デコードは **ONNX Runtime (CPU)** を使用してストリーミング出力を行い、超高速応答を実現。
- **音声クローン**: 参照音声を使用したゼロショット (Zero-shot) 音声クローンをサポート。

## パフォーマンス

| デバイス | 量子化 | RTF (リアルタイム係数) | 平均時間 (10回) |
|--------|-------|----------------------|-----------------|
| CPU | Int4 (Q4) | 1.144 | ~4.44s |
| CPU | F16 | 2.664 | ~9.47s |
| CUDA | Int4 (Q4) | 0.608 | ~2.25s |
| CUDA | F16 | 0.715 | ~2.60s |
| Vulkan | Int4 (Q4) | 0.606 | ~2.30s |
| Vulkan | F16 | 0.717 | ~2.87s |

> **テスト環境**: Intel Core i9-13980HX, NVIDIA RTX 2080 Ti. VRAM使用量 約2GB.
> Windows プラットフォームでの 10 回実行の平均値に基づきます。

## クイックスタート

### 1. 実行環境の準備 (Windows)
プロジェクト・ディレクトリに関連するランタイム DLL を配置する必要があります。
1. [ONNX Runtime](https://github.com/microsoft/onnxruntime/releases) (v1.23.2 推奨) をダウンロードします。
2. `../assets/download_dlls.ps1` スクリプトを実行して、ONNX Runtime (CPU 版) を自動的にダウンロードしてインストールします。

### 2. モデルリソースの準備
提供されている Python スクリプトを実行して、事前学習済みモデルをダウンロードします：
```bash
python ../assets/download_models.py
```
モデルは `../models/` ディレクトリに保存されます。

> **注**: 変換済みのモデルファイルは数日中にアップロードされる予定です。ご期待ください。

### 3. 音声管理 (New)
音声特徴を抽出して `.qvoice` ファイルとして保存し、再利用することをお勧めします。

**音声の抽出:**
```powershell
$env:PATH += ";$PWD\runtime"
cargo run --example make_voice --release -- `
    --model_dir ./models `
    --input clone.wav `
    --text "参照音声のテキスト内容" `
    --output my_voice.qvoice `
    --name "私の専用音声" `
    --gender "Female" `
    --age "Young" `
    --description "クリアで優しいナレーション音声"
```

**音声パックを使用して生成:**
```powershell
cargo run --example qwen3-tts --release -- --model_dir ./models --voice my_voice.qvoice --text "こんにちは、世界"
```

### 4. 高速デモ
`run.ps1` スクリプトを使用してデモを実行します（DLL パスを自動処理します）：
```powershell
.\run.ps1 --input clone.wav --ref_text "参照音声のテキスト内容" --text "こんにちは、世界"
```

または手動で実行します（`runtime` が PATH にあることを確認してください）：
```bash
$env:PATH += ";$PWD\runtime"
cargo run --example qwen3-tts --release -- --model_dir ./models --input clone.wav --ref_text "参照音声のテキスト内容" --text "こんにちは、世界"
```

## ライブラリとしての使用
`Cargo.toml` に以下を追加します：
```toml
[dependencies]
qwen3-tts = { path = "../path/to/qwen3-tts-rust" }
```

### サンプルコード
```rust
use qwen3_tts::TtsEngine;
use std::path::Path;

fn main() -> Result<(), String> {
    // 1. エンジンの初期化
    // models (onnx/gguf) と assets (tokenizer.json 等) を含むディレクトリを指定
    let model_dir = Path::new("models");
    let mut engine = TtsEngine::load(model_dir)?;

    // 2. 入力の準備
    let text = "こんにちは、これは Qwen3-TTS の Rust 実装です。";
    let ref_audio = Path::new("clone.wav"); // 参照音声パス
    let ref_text = "これは参照音声に対応するテキスト内容です。"; // 参照音声のテキスト

    // 3. 音声生成
    // AudioSample 構造体を返します（samples (Vec<f32>) と sample_rate を含む）
    let audio = engine.generate(text, ref_audio, ref_text)?;

    // 4. 保存または再生
    audio.save_wav("output.wav")?;
    
    // 5. (オプション) リソースのクリーンアップ
    qwen3_tts::cleanup();
    
    Ok(())
}
```

### API 説明
- **`TtsEngine::load(path)`**: 必要なすべてのモデルリソースを読み込みます。数秒かかるため、アプリケーション起動時に一度だけ呼び出すことを推奨します。
- **`engine.generate(text, ref_audio, ref_text)`**: 推論を実行します。
    - `text`: 合成するターゲットテキスト。
    - `ref_audio`: 参照音声パス (WAV形式, 16kHz+ 推奨)。
    - `ref_text`: 参照音声に対応するテキスト (音声クローンの精度、特に中国語において重要)。
- **`audio.save_wav(path)`**: WAV ファイルとして保存するためのヘルパーメソッド。

## 開発ロードマップ
- [x] コードのモジュール化リファクタリング
- [x] ハードコードされたパスの削除
- [x] ダウンロードスクリプトの準備
- [x] 変換スクリプトの準備
- [ ] C API のエクスポート (オプション)

## 謝辞
本実装のインスピレーションとサポートを頂いた以下のプロジェクトに感謝します：
- [Qwen3-TTS-GGUF](https://github.com/HaujetZhao/Qwen3-TTS-GGUF): GGUF 推論フローの参考にさせていただきました。
- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS): Qwen3-TTS の公式リポジトリ。

## ライセンス
MIT / Apache 2.0

