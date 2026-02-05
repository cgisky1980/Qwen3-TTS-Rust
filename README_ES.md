# Qwen3-TTS Rust

[中文](README.md) | [English](README_EN.md) | [日本語](README_JA.md) | [한국어](README_KO.md) | [Deutsch](README_DE.md) | [Français](README_FR.md) | [Русский](README_RU.md) | [Português](README_PT.md) | [Español](README_ES.md) | [Italiano](README_IT.md)

Implementación en Rust de Qwen3-TTS, basada en ONNX Runtime y llama.cpp (GGUF), diseñada para proporcionar capacidades de texto a voz de alto rendimiento y fácil integración.

## Características
- **Alto Rendimiento**: Inferencia principal escrita en Rust, soportando aceleración CUDA y DirectML.
- **Clonación de Voz**: Soporta clonación de voz Zero-shot a través de audio de referencia.
- **Salida en Streaming**: (Planeado) Soporte para generación de audio en streaming.
- **Multi-backend**: Soporta `ort` (ONNX) y `llama.cpp` (GGUF).

## Rendimiento

| Dispositivo | Cuantización | RTF (Factor de Tiempo Real) | Tiempo Promedio (10 ejec.) |
|-------------|--------------|-----------------------------|----------------------------|
| CPU | Int4 (Q4) | 1.144 | ~4.44s |
| CPU | F16 | 2.664 | ~9.47s |
| CUDA | Int4 (Q4) | 0.608 | ~2.25s |
| CUDA | F16 | 0.715 | ~2.60s |
| Vulkan | Int4 (Q4) | 0.606 | ~2.30s |
| Vulkan | F16 | 0.717 | ~2.87s |

> **Entorno de prueba**: Intel Core i9-13980HX, NVIDIA RTX 2080 Ti. Uso de VRAM aprox. 2GB.
> Datos basados en la plataforma Windows, promedio de 10 ejecuciones.

## Inicio Rápido

### 1. Preparar Entorno (Windows)
Necesita colocar las DLL de tiempo de ejecución relevantes en el directorio del proyecto.
1. Descargue [ONNX Runtime](https://github.com/microsoft/onnxruntime/releases) (v1.23.2 recomendado).
2. Ejecute el script `assets/download_dlls.ps1` para descargar e instalar automáticamente ONNX Runtime (versión CPU).

### 2. Preparar Modelos
Ejecute el script de Python proporcionado para descargar modelos pre-entrenados:
```bash
python assets/download_models.py
```
Los modelos se guardarán en el directorio `models/`.

### 3. Gestión de Voz (Nuevo)
Recomendamos extraer características de voz y guardarlas como archivos `.qvoice` para su reutilización.

**Extraer Voz:**
```powershell
$env:PATH += ";$PWD\runtime"
cargo run --example make_voice --release -- `
    --model_dir ./models `
    --input clone.wav `
    --text "Contenido textual del audio de referencia" `
    --output my_voice.qvoice `
    --name "Mi Voz Personalizada" `
    --gender "Female" `
    --age "Young" `
    --description "Voz de narración clara y suave"
```

**Generar con Paquete de Voz:**
```powershell
cargo run --example qwen3-tts --release -- --model_dir ./models --voice my_voice.qvoice --text "Hola mundo"
```

### 4. Demostración Rápida
Use el script `run.ps1` para ejecutar la demostración (maneja automáticamente las rutas de DLL):
```powershell
.\run.ps1 --input clone.wav --ref_text "Contenido textual del audio de referencia" --text "Hola mundo"
```

O ejecute manualmente (asegúrese de que `runtime` esté en PATH):
```bash
$env:PATH += ";$PWD\runtime"
cargo run --example qwen3-tts --release -- --model_dir ./models --input clone.wav --ref_text "Contenido textual del audio de referencia" --text "Hola mundo"
```

## Uso como Biblioteca
Agregue esto a su `Cargo.toml`:
```toml
[dependencies]
qwen3-tts = { path = "../path/to/qwen3-tts-rust" }
```

### Código de Ejemplo
```rust
use qwen3_tts::TtsEngine;
use std::path::Path;

fn main() -> Result<(), String> {
    // 1. Inicializar Motor
    let model_dir = Path::new("models");
    let mut engine = TtsEngine::load(model_dir)?;

    // 2. Preparar Entrada
    let text = "Hola, esta es la implementación Rust de Qwen3-TTS.";
    let ref_audio = Path::new("clone.wav");
    let ref_text = "Este es el texto del audio de referencia.";

    // 3. Generar Audio
    let audio = engine.generate(text, ref_audio, ref_text)?;

    // 4. Guardar
    audio.save_wav("output.wav")?;
    
    // 5. Limpiar
    qwen3_tts::cleanup();
    
    Ok(())
}
```

## Agradecimientos
Gracias a los siguientes proyectos por su inspiración y apoyo:
- [Qwen3-TTS-GGUF](https://github.com/HaujetZhao/Qwen3-TTS-GGUF): Referencia para el flujo de inferencia GGUF.
- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS): Repositorio oficial de Qwen3-TTS.

## Licencia
MIT / Apache 2.0
