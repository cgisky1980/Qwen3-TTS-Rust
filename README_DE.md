# Qwen3-TTS Rust

[中文](README.md) | [English](README_EN.md) | [日本語](README_JA.md) | [한국어](README_KO.md) | [Deutsch](README_DE.md) | [Français](README_FR.md) | [Русский](README_RU.md) | [Português](README_PT.md) | [Español](README_ES.md) | [Italiano](README_IT.md)

Rust-Implementierung von Qwen3-TTS, basierend auf ONNX Runtime und llama.cpp (GGUF), entwickelt für leistungsstarke, einfach zu integrierende Text-zu-Sprache-Funktionen.

## Funktionen
- **Hochleistungsarchitektur**: Kernlogik in Rust geschrieben. LLM-Inferenz basiert auf **llama.cpp** und unterstützt **CPU-, CUDA-, Vulkan**-Backends sowie Modellquantisierung (Q4/F16).
- **Streaming-Decodierung**: Die Audiodecodierung verwendet **ONNX Runtime (CPU)** für die Streaming-Ausgabe und ermöglicht eine ultraschnelle Reaktion.
- **Stimmenklonen**: Unterstützt Zero-Shot-Stimmenklonen über Referenzaudio.

## Leistung

| Gerät | Quantisierung | RTF (Echtzeitfaktor) | Durchschnittszeit (10 Läufe) |
|--------|---------------|----------------------|-------------------|
| CPU | Int4 (Q4) | 1.144 | ~4.44s |
| CPU | F16 | 2.664 | ~9.47s |
| CUDA | Int4 (Q4) | 0.608 | ~2.25s |
| CUDA | F16 | 0.715 | ~2.60s |
| Vulkan | Int4 (Q4) | 0.606 | ~2.30s |
| Vulkan | F16 | 0.717 | ~2.87s |

> **Testumgebung**: Intel Core i9-13980HX, NVIDIA RTX 2080 Ti. VRAM-Nutzung ca. 2GB.
> Daten basierend auf Windows-Plattform, Durchschnitt von 10 Läufen.

## Schnellstart

### 1. Umgebung vorbereiten (Windows)
Sie müssen die relevanten Runtime-DLLs im Projektverzeichnis ablegen.
1. Laden Sie [ONNX Runtime](https://github.com/microsoft/onnxruntime/releases) (v1.23.2 empfohlen) herunter.
2. Führen Sie das Skript `assets/download_dlls.ps1` aus, um ONNX Runtime (CPU-Version) automatisch herunterzuladen und zu installieren.

### 2. Modellressourcen vorbereiten
Führen Sie das bereitgestellte Python-Skript aus, um vorab trainierte Modelle herunterzuladen:
```bash
python assets/download_models.py
```
Modelle werden im Verzeichnis `models/` gespeichert.

### 3. Stimmenverwaltung (Neu)
Wir empfehlen, Stimmenmerkmale zu extrahieren und als `.qvoice`-Dateien zur Wiederverwendung zu speichern.

**Stimme extrahieren:**
```powershell
$env:PATH += ";$PWD\runtime"
cargo run --example make_voice --release -- `
    --model_dir ./models `
    --input clone.wav `
    --text "Textinhalt des Referenzaudios" `
    --output my_voice.qvoice `
    --name "Meine eigene Stimme" `
    --gender "Female" `
    --age "Young" `
    --description "Klare, sanfte Erzählstimme"
```

**Generierung mit Stimmenpaket:**
```powershell
cargo run --example qwen3-tts --release -- --model_dir ./models --voice my_voice.qvoice --text "Hallo Welt"
```

### 4. Schnelldemo
Verwenden Sie das Skript `run.ps1`, um die Demo auszuführen (behandelt DLL-Pfade automatisch):
```powershell
.\run.ps1 --input clone.wav --ref_text "Textinhalt des Referenzaudios" --text "Hallo Welt"
```

Oder manuell ausführen (stellen Sie sicher, dass `runtime` im PATH ist):
```bash
$env:PATH += ";$PWD\runtime"
cargo run --example qwen3-tts --release -- --model_dir ./models --input clone.wav --ref_text "Textinhalt des Referenzaudios" --text "Hallo Welt"
```

## Verwendung als Bibliothek
Fügen Sie dies zu Ihrer `Cargo.toml` hinzu:
```toml
[dependencies]
qwen3-tts = { path = "../path/to/qwen3-tts-rust" }
```

### Beispielcode
```rust
use qwen3_tts::TtsEngine;
use std::path::Path;

fn main() -> Result<(), String> {
    // 1. Engine initialisieren
    let model_dir = Path::new("models");
    let mut engine = TtsEngine::load(model_dir)?;

    // 2. Eingabe vorbereiten
    let text = "Hallo, das ist die Qwen3-TTS Rust-Implementierung.";
    let ref_audio = Path::new("clone.wav");
    let ref_text = "Das ist der Text des Referenzaudios.";

    // 3. Audio generieren
    let audio = engine.generate(text, ref_audio, ref_text)?;

    // 4. Speichern
    audio.save_wav("output.wav")?;
    
    // 5. Aufräumen
    qwen3_tts::cleanup();
    
    Ok(())
}
```

## Danksagung
Vielen Dank an die folgenden Projekte für ihre Inspiration und Unterstützung:
- [Qwen3-TTS-GGUF](https://github.com/HaujetZhao/Qwen3-TTS-GGUF): Referenz für den GGUF-Inferenzfluss.
- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS): Offizielles Repository für Qwen3-TTS.

## Lizenz
MIT / Apache 2.0
