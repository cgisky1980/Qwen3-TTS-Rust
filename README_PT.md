# Qwen3-TTS Rust

[中文](README.md) | [English](README_EN.md) | [日本語](README_JA.md) | [한국어](README_KO.md) | [Deutsch](README_DE.md) | [Français](README_FR.md) | [Русский](README_RU.md) | [Português](README_PT.md) | [Español](README_ES.md) | [Italiano](README_IT.md)

Implementação em Rust do Qwen3-TTS, baseada em ONNX Runtime e llama.cpp (GGUF), projetada para fornecer capacidades de conversão de texto em fala de alto desempenho e fácil integração.

## Recursos
- **Alto Desempenho**: Inferência principal escrita em Rust, suportando aceleração CUDA e DirectML.
- **Clonagem de Voz**: Suporta clonagem de voz Zero-shot via áudio de referência.
- **Saída em Streaming**: (Planejado) Suporte para geração de áudio em streaming.
- **Multi-backend**: Suporta `ort` (ONNX) e `llama.cpp` (GGUF).

## Desempenho

| Dispositivo | Quantização | RTF (Fator de Tempo Real) | Tempo Médio (10 exec.) |
|-------------|-------------|---------------------------|------------------------|
| CPU | Int4 (Q4) | 1.144 | ~4.44s |
| CPU | F16 | 2.664 | ~9.47s |
| CUDA | Int4 (Q4) | 0.608 | ~2.25s |
| CUDA | F16 | 0.715 | ~2.60s |
| Vulkan | Int4 (Q4) | 0.606 | ~2.30s |
| Vulkan | F16 | 0.717 | ~2.87s |

> **Ambiente de Teste**: Intel Core i9-13980HX, NVIDIA RTX 2080 Ti. Uso de VRAM aprox. 2GB.
> Dados baseados na plataforma Windows, média de 10 execuções.

## Início Rápido

### 1. Preparar Ambiente (Windows)
Você precisa colocar as DLLs de tempo de execução relevantes no diretório do projeto.
1. Baixe o [ONNX Runtime](https://github.com/microsoft/onnxruntime/releases) (v1.23.2 recomendado).
2. Execute o script `assets/download_dlls.ps1` para baixar e instalar automaticamente o ONNX Runtime (versão CPU).

### 2. Preparar Modelos
Execute o script Python fornecido para baixar modelos pré-treinados:
```bash
python assets/download_models.py
```
Os modelos serão salvos no diretório `models/`.

### 3. Gerenciamento de Voz (Novo)
Recomendamos extrair características de voz e salvá-las como arquivos `.qvoice` para reutilização.

**Extrair Voz:**
```powershell
$env:PATH += ";$PWD\runtime"
cargo run --example make_voice --release -- `
    --model_dir ./models `
    --input clone.wav `
    --text "Conteúdo textual do áudio de referência" `
    --output my_voice.qvoice `
    --name "Minha Voz Personalizada" `
    --gender "Female" `
    --age "Young" `
    --description "Voz de narração clara e suave"
```

**Gerar com Pacote de Voz:**
```powershell
cargo run --example qwen3-tts --release -- --model_dir ./models --voice my_voice.qvoice --text "Olá, mundo"
```

### 4. Demonstração Rápida
Use o script `run.ps1` para executar a demonstração (lida automaticamente com caminhos de DLL):
```powershell
.\run.ps1 --input clone.wav --ref_text "Conteúdo textual do áudio de referência" --text "Olá, mundo"
```

Ou execute manualmente (garanta que `runtime` esteja no PATH):
```bash
$env:PATH += ";$PWD\runtime"
cargo run --example qwen3-tts --release -- --model_dir ./models --input clone.wav --ref_text "Conteúdo textual do áudio de referência" --text "Olá, mundo"
```

## Uso como Biblioteca
Adicione isso ao seu `Cargo.toml`:
```toml
[dependencies]
qwen3-tts = { path = "../path/to/qwen3-tts-rust" }
```

### Código de Exemplo
```rust
use qwen3_tts::TtsEngine;
use std::path::Path;

fn main() -> Result<(), String> {
    // 1. Inicializar Motor
    let model_dir = Path::new("models");
    let mut engine = TtsEngine::load(model_dir)?;

    // 2. Preparar Entrada
    let text = "Olá, esta é a implementação Rust do Qwen3-TTS.";
    let ref_audio = Path::new("clone.wav");
    let ref_text = "Este é o texto do áudio de referência.";

    // 3. Gerar Áudio
    let audio = engine.generate(text, ref_audio, ref_text)?;

    // 4. Salvar
    audio.save_wav("output.wav")?;
    
    // 5. Limpar
    qwen3_tts::cleanup();
    
    Ok(())
}
```

## Agradecimentos
Obrigado aos seguintes projetos pela inspiração e apoio:
- [Qwen3-TTS-GGUF](https://github.com/HaujetZhao/Qwen3-TTS-GGUF): Referência para o fluxo de inferência GGUF.
- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS): Repositório oficial do Qwen3-TTS.

## Licença
MIT / Apache 2.0
