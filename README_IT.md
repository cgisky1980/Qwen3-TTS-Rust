# Qwen3-TTS Rust

[中文](README.md) | [English](README_EN.md) | [日本語](README_JA.md) | [한국어](README_KO.md) | [Deutsch](README_DE.md) | [Français](README_FR.md) | [Русский](README_RU.md) | [Português](README_PT.md) | [Español](README_ES.md) | [Italiano](README_IT.md)

Implementazione Rust di Qwen3-TTS, basata su ONNX Runtime e llama.cpp (GGUF), progettata per fornire funzionalità di sintesi vocale ad alte prestazioni e facile integrazione.

## Caratteristiche
- **Architettura ad Alte Prestazioni**: Logica di base scritta in Rust. Inferenza LLM basata su **llama.cpp**, che supporta backend **CPU, CUDA, Vulkan** e quantizzazione del modello (Q4/F16).
- **Decodifica in Streaming**: La decodifica audio utilizza **ONNX Runtime (CPU)** per l'output in streaming, consentendo una risposta ultraveloce.
- **Clonazione Vocale**: Supporta la clonazione vocale Zero-shot tramite audio di riferimento.

## Prestazioni

| Dispositivo | Quantizzazione | RTF (Fattore Tempo Reale) | Tempo Medio (10 esec.) |
|-------------|----------------|---------------------------|------------------------|
| CPU | Int4 (Q4) | 1.144 | ~4.44s |
| CPU | F16 | 2.664 | ~9.47s |
| CUDA | Int4 (Q4) | 0.608 | ~2.25s |
| CUDA | F16 | 0.715 | ~2.60s |
| Vulkan | Int4 (Q4) | 0.606 | ~2.30s |
| Vulkan | F16 | 0.717 | ~2.87s |

> **Ambiente di Test**: Intel Core i9-13980HX, NVIDIA RTX 2080 Ti. Uso VRAM circa 2GB.
> Dati basati su piattaforma Windows, media di 10 esecuzioni.

## Avvio Rapido

### 1. Preparare l'ambiente (Windows)
È necessario posizionare le DLL di runtime pertinenti nella directory del progetto.
1. Scarica [ONNX Runtime](https://github.com/microsoft/onnxruntime/releases) (v1.23.2 consigliato).
2. Esegui lo script `assets/download_dlls.ps1` per scaricare e installare automaticamente ONNX Runtime (versione CPU).

### 2. Preparare i Modelli
Esegui lo script Python fornito per scaricare i modelli pre-addestrati:
```bash
python assets/download_models.py
```
I modelli verranno salvati nella directory `models/`.

### 3. Gestione Vocale (Nuovo)
Si consiglia di estrarre le caratteristiche vocali e salvarle come file `.qvoice` per il riutilizzo.

**Estrarre Voce:**
```powershell
$env:PATH += ";$PWD\runtime"
cargo run --example make_voice --release -- `
    --model_dir ./models `
    --input clone.wav `
    --text "Contenuto testuale dell'audio di riferimento" `
    --output my_voice.qvoice `
    --name "La Mia Voce Personalizzata" `
    --gender "Female" `
    --age "Young" `
    --description "Voce narrante chiara e gentile"
```

**Generare con Pacchetto Vocale:**
```powershell
cargo run --example qwen3-tts --release -- --model_dir ./models --voice my_voice.qvoice --text "Ciao mondo"
```

### 4. Demo Rapida
Usa lo script `run.ps1` per eseguire la demo (gestisce automaticamente i percorsi DLL):
```powershell
.\run.ps1 --input clone.wav --ref_text "Contenuto testuale dell'audio di riferimento" --text "Ciao mondo"
```

Oppure esegui manualmente (assicurati che `runtime` sia nel PATH):
```bash
$env:PATH += ";$PWD\runtime"
cargo run --example qwen3-tts --release -- --model_dir ./models --input clone.wav --ref_text "Contenuto testuale dell'audio di riferimento" --text "Ciao mondo"
```

## Uso come Libreria
Aggiungi questo al tuo `Cargo.toml`:
```toml
[dependencies]
qwen3-tts = { path = "../path/to/qwen3-tts-rust" }
```

### Codice di Esempio
```rust
use qwen3_tts::TtsEngine;
use std::path::Path;

fn main() -> Result<(), String> {
    // 1. Inizializzare Motore
    let model_dir = Path::new("models");
    let mut engine = TtsEngine::load(model_dir)?;

    // 2. Preparare Input
    let text = "Ciao, questa è l'implementazione Rust di Qwen3-TTS.";
    let ref_audio = Path::new("clone.wav");
    let ref_text = "Questo è il testo dell'audio di riferimento.";

    // 3. Generare Audio
    let audio = engine.generate(text, ref_audio, ref_text)?;

    // 4. Salvare
    audio.save_wav("output.wav")?;
    
    // 5. Pulizia
    qwen3_tts::cleanup();
    
    Ok(())
}
```

## Ringraziamenti
Grazie ai seguenti progetti per l'ispirazione e il supporto:
- [Qwen3-TTS-GGUF](https://github.com/HaujetZhao/Qwen3-TTS-GGUF): Riferimento per il flusso di inferenza GGUF.
- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS): Repository ufficiale per Qwen3-TTS.

## Licenza
MIT / Apache 2.0
