# Qwen3-TTS Rust

[中文](../README.md) | [English](README_EN.md) | [日本語](README_JA.md) | [한국어](README_KO.md) | [Deutsch](README_DE.md) | [Français](README_FR.md) | [Русский](README_RU.md) | [Português](README_PT.md) | [Español](README_ES.md) | [Italiano](README_IT.md)

Реализация Qwen3-TTS на Rust, основанная на ONNX Runtime и llama.cpp (GGUF), предназначенная для обеспечения высокопроизводительных и легко интегрируемых возможностей синтеза речи.

## Особенности
- **Высокопроизводительная архитектура**: Основная логика написана на Rust. Инференс LLM основан на **llama.cpp**, поддерживает бэкенды **CPU, CUDA, Vulkan** и квантование моделей (Q4/F16).
- **Потоковое декодирование**: Аудиодекодирование использует **ONNX Runtime (CPU)** для потокового вывода, обеспечивая сверхбыстрый отклик.
- **Клонирование голоса**: Поддерживает Zero-shot клонирование голоса с использованием эталонного аудио.

## Производительность

| Устройство | Квантование | RTF (Фактор реального времени) | Среднее время (10 зап.) |
|------------|-------------|-------------------------------|-------------------------|
| CPU | Int4 (Q4) | 1.144 | ~4.44s |
| CPU | F16 | 2.664 | ~9.47s |
| CUDA | Int4 (Q4) | 0.608 | ~2.25s |
| CUDA | F16 | 0.715 | ~2.60s |
| Vulkan | Int4 (Q4) | 0.606 | ~2.30s |
| Vulkan | F16 | 0.717 | ~2.87s |

> **Тестовая среда**: Intel Core i9-13980HX, NVIDIA RTX 2080 Ti. Использование VRAM ок. 2GB.
> Данные основаны на платформе Windows, среднее значение за 10 запусков.

## Быстрый старт

### 1. Подготовка окружения (Windows)
Вам необходимо поместить соответствующие DLL среды выполнения в каталог проекта.
1. Скачайте [ONNX Runtime](https://github.com/microsoft/onnxruntime/releases) (рекомендуется v1.23.2).
2. Запустите скрипт `../assets/download_dlls.ps1` для автоматической загрузки и установки ONNX Runtime (CPU версия).

### 2. Подготовка моделей
Запустите предоставленный скрипт Python для загрузки предварительно обученных моделей:
```bash
python ../assets/download_models.py
```
Модели будут сохранены в каталоге `../models/`.

> **Примечание**: Мы загрузим конвертированные файлы моделей в ближайшие дни. Следите за обновлениями.

### 3. Конвертация ассетов (Опционально)
Если вам нужно упаковать `.npy` файлы ассетов в один `qwen3_assets.gguf` файл (рекомендуется для чистоты):
1. Установите зависимости: `pip install numpy gguf`
2. Запустите скрипт конвертации:
```bash
python ../assets/convert_assets.py --input_dir /path/to/npy/files --output_file ../models/qwen3_assets.gguf
```
Движок автоматически загрузит `qwen3_assets.gguf`, если он найден.


### 3. Управление голосами (Новое)
Мы рекомендуем извлекать характеристики голоса и сохранять их в виде файлов `.qvoice` для повторного использования.

**Извлечение голоса:**
```powershell
$env:PATH += ";$PWD\runtime"
cargo run --example make_voice --release -- `
    --model_dir ./models `
    --input clone.wav `
    --text "Текстовое содержание эталонного аудио" `
    --output my_voice.qvoice `
    --name "Мой собственный голос" `
    --gender "Female" `
    --age "Young" `
    --description "Чистый, мягкий голос диктора"
```

**Генерация с использованием голосового пакета:**
```powershell
cargo run --example qwen3-tts --release -- --model_dir ./models --voice my_voice.qvoice --text "Привет, мир"
```

### 4. Быстрая демонстрация
Используйте скрипт `run.ps1` для запуска демонстрации (автоматически обрабатывает пути к DLL):
```powershell
.\run.ps1 --input clone.wav --ref_text "Текстовое содержание эталонного аудио" --text "Привет, мир"
```

Или запустите вручную (убедитесь, что `runtime` находится в PATH):
```bash
$env:PATH += ";$PWD\runtime"
cargo run --example qwen3-tts --release -- --model_dir ./models --input clone.wav --ref_text "Текстовое содержание эталонного аудио" --text "Привет, мир"
```

## Использование как библиотеки
Добавьте это в ваш `Cargo.toml`:
```toml
[dependencies]
qwen3-tts = { path = "../path/to/qwen3-tts-rust" }
```

### Пример кода
```rust
use qwen3_tts::TtsEngine;
use std::path::Path;

fn main() -> Result<(), String> {
    // 1. Инициализация движка
    let model_dir = Path::new("models");
    let mut engine = TtsEngine::load(model_dir)?;

    // 2. Подготовка ввода
    let text = "Привет, это реализация Qwen3-TTS на Rust.";
    let ref_audio = Path::new("clone.wav");
    let ref_text = "Это текст эталонного аудио.";

    // 3. Генерация аудио
    let audio = engine.generate(text, ref_audio, ref_text)?;

    // 4. Сохранение
    audio.save_wav("output.wav")?;
    
    // 5. Очистка
    qwen3_tts::cleanup();
    
    Ok(())
}
```

## Благодарности
Спасибо следующим проектам за вдохновение и поддержку:
- [Qwen3-TTS-GGUF](https://github.com/HaujetZhao/Qwen3-TTS-GGUF): Ссылка на поток вывода GGUF.
- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS): Официальный репозиторий Qwen3-TTS.

## Лицензия
MIT / Apache 2.0

