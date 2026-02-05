# Qwen3-TTS Rust

[中文](README.md) | [English](README_EN.md) | [日本語](README_JA.md) | [한국어](README_KO.md) | [Deutsch](README_DE.md) | [Français](README_FR.md) | [Русский](README_RU.md) | [Português](README_PT.md) | [Español](README_ES.md) | [Italiano](README_IT.md)

ONNX Runtime과 llama.cpp (GGUF)를 기반으로 한 Qwen3-TTS의 Rust 구현입니다. 고성능이며 통합하기 쉬운 TTS(Text-to-Speech) 기능을 제공하는 것을 목표로 합니다.

## 특징
- **고성능 아키텍처**: 핵심 로직은 Rust로 작성되었습니다. LLM 추론은 **llama.cpp**를 기반으로 하며, **CPU, CUDA, Vulkan** 백엔드 및 모델 양자화(Q4/F16)를 지원합니다.
- **스트리밍 디코드**: 오디오 디코딩은 **ONNX Runtime (CPU)**을 사용하여 스트리밍 출력을 수행하며, 초고속 응답을 제공합니다.
- **음성 복제**: 참조 오디오를 통한 제로샷(Zero-shot) 음성 복제를 지원합니다.

## 성능

| 장치 | 양자화 | RTF (실시간 비율) | 평균 시간 (10회) |
|--------|-------|-------------------|------------------|
| CPU | Int4 (Q4) | 1.144 | ~4.44s |
| CPU | F16 | 2.664 | ~9.47s |
| CUDA | Int4 (Q4) | 0.608 | ~2.25s |
| CUDA | F16 | 0.715 | ~2.60s |
| Vulkan | Int4 (Q4) | 0.606 | ~2.30s |
| Vulkan | F16 | 0.717 | ~2.87s |

> **테스트 환경**: Intel Core i9-13980HX, NVIDIA RTX 2080 Ti. VRAM 사용량 약 2GB.
> Windows 플랫폼에서 10회 실행 평균 기준 데이터.

## 빠른 시작

### 1. 실행 환경 준비 (Windows)
프로젝트 디렉토리에 관련 런타임 DLL을 배치해야 합니다.
1. [ONNX Runtime](https://github.com/microsoft/onnxruntime/releases) (v1.23.2 권장)을 다운로드합니다.
2. `assets/download_dlls.ps1` 스크립트를 실행하여 ONNX Runtime (CPU 버전)을 자동으로 다운로드하고 설치합니다.

### 2. 모델 리소스 준비
제공된 Python 스크립트를 실행하여 사전 학습된 모델을 다운로드합니다:
```bash
python assets/download_models.py
```
모델은 `models/` 디렉토리에 저장됩니다.

### 3. 음색 관리 (New)
음성 특징을 추출하여 `.qvoice` 파일로 저장하고 재사용하는 것을 권장합니다.

**음색 추출:**
```powershell
$env:PATH += ";$PWD\runtime"
cargo run --example make_voice --release -- `
    --model_dir ./models `
    --input clone.wav `
    --text "참조 오디오의 텍스트 내용" `
    --output my_voice.qvoice `
    --name "나만의 전용 음색" `
    --gender "Female" `
    --age "Young" `
    --description "선명하고 부드러운 내레이션 음색"
```

**음색 팩을 사용하여 생성:**
```powershell
cargo run --example qwen3-tts --release -- --model_dir ./models --voice my_voice.qvoice --text "안녕하세요, 세계"
```

### 4. 빠른 데모
`run.ps1` 스크립트를 사용하여 데모를 실행합니다 (DLL 경로 자동 처리):
```powershell
.\run.ps1 --input clone.wav --ref_text "참조 오디오의 텍스트 내용" --text "안녕하세요, 세계"
```

또는 수동으로 실행합니다 (`runtime`이 PATH에 있는지 확인하십시오):
```bash
$env:PATH += ";$PWD\runtime"
cargo run --example qwen3-tts --release -- --model_dir ./models --input clone.wav --ref_text "참조 오디오의 텍스트 내용" --text "안녕하세요, 세계"
```

## 라이브러리로 사용
`Cargo.toml`에 다음을 추가하십시오:
```toml
[dependencies]
qwen3-tts = { path = "../path/to/qwen3-tts-rust" }
```

### 예제 코드
```rust
use qwen3_tts::TtsEngine;
use std::path::Path;

fn main() -> Result<(), String> {
    // 1. 엔진 초기화
    // models (onnx/gguf) 및 assets (tokenizer.json 등) 포함 디렉토리 지정
    let model_dir = Path::new("models");
    let mut engine = TtsEngine::load(model_dir)?;

    // 2. 입력 준비
    let text = "안녕하세요, 이것은 Qwen3-TTS의 Rust 구현입니다.";
    let ref_audio = Path::new("clone.wav"); // 참조 오디오 경로
    let ref_text = "이것은 참조 오디오에 해당하는 텍스트 내용입니다."; // 참조 오디오 텍스트

    // 3. 오디오 생성
    // AudioSample 구조체 반환 (samples (Vec<f32>) 및 sample_rate 포함)
    let audio = engine.generate(text, ref_audio, ref_text)?;

    // 4. 저장 또는 재생
    audio.save_wav("output.wav")?;
    
    // 5. (선택 사항) 리소스 정리
    qwen3_tts::cleanup();
    
    Ok(())
}
```

### API 설명
- **`TtsEngine::load(path)`**: 필요한 모든 모델 리소스를 로드합니다. 몇 초 정도 걸리므로 애플리케이션 시작 시 한 번만 호출하는 것이 좋습니다.
- **`engine.generate(text, ref_audio, ref_text)`**: 추론을 실행합니다.
    - `text`: 합성할 대상 텍스트.
    - `ref_audio`: 참조 오디오 경로 (WAV 형식, 16kHz+ 권장).
    - `ref_text`: 참조 오디오에 해당하는 텍스트 (음색 복제의 정확성, 특히 중국어에서 중요).
- **`audio.save_wav(path)`**: WAV 파일로 저장하기 위한 보조 메서드.

## 개발 로드맵
- [x] 코드 모듈화 리팩토링
- [x] 하드코딩된 경로 제거
- [x] 다운로드 스크립트 준비
- [x] 변환 스크립트 준비
- [ ] C API 내보내기 (선택 사항)

## 감사의 말
본 구현에 영감과 지원을 주신 다음 프로젝트에 감사드립니다:
- [Qwen3-TTS-GGUF](https://github.com/HaujetZhao/Qwen3-TTS-GGUF): GGUF 추론 흐름을 참고하였습니다.
- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS): Qwen3-TTS의 공식 저장소.

## 라이선스
MIT / Apache 2.0
