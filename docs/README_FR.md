# Qwen3-TTS Rust

[中文](../README.md) | [English](README_EN.md) | [日本語](README_JA.md) | [한국어](README_KO.md) | [Deutsch](README_DE.md) | [Français](README_FR.md) | [Русский](README_RU.md) | [Português](README_PT.md) | [Español](README_ES.md) | [Italiano](README_IT.md)

Implémentation Rust de Qwen3-TTS, basée sur ONNX Runtime et llama.cpp (GGUF), conçue pour fournir des capacités de synthèse vocale haute performance et faciles à intégrer.

## Fonctionnalités
- **Architecture haute performance**: Logique de base écrite en Rust. Inférence LLM basée sur **llama.cpp**, prenant en charge les backends **CPU, CUDA, Vulkan** et la quantification de modèle (Q4/F16).
- **Décodage en streaming**: Le décodage audio utilise **ONNX Runtime (CPU)** pour la sortie en streaming, permettant une réponse ultra-rapide.
- **Clonage de voix**: Prend en charge le clonage de voix Zero-shot via un audio de référence.

## Performance

| Appareil | Quantification | RTF (Facteur temps réel) | Temps moyen (10 exécutions) |
|--------|----------------|--------------------------|-----------------------------|
| CPU | Int4 (Q4) | 1.144 | ~4.44s |
| CPU | F16 | 2.664 | ~9.47s |
| CUDA | Int4 (Q4) | 0.608 | ~2.25s |
| CUDA | F16 | 0.715 | ~2.60s |
| Vulkan | Int4 (Q4) | 0.606 | ~2.30s |
| Vulkan | F16 | 0.717 | ~2.87s |

> **Environnement de test**: Intel Core i9-13980HX, NVIDIA RTX 2080 Ti. Utilisation VRAM env. 2GB.
> Données basées sur la plateforme Windows, moyenne de 10 exécutions.

## Démarrage rapide

### 1. Préparer l'environnement (Windows)
Vous devez placer les DLL d'exécution pertinentes dans le répertoire du projet.
1. Téléchargez [ONNX Runtime](https://github.com/microsoft/onnxruntime/releases) (v1.23.2 recommandé).
2. Exécutez le script `../assets/download_dlls.ps1` pour télécharger et installer automatiquement ONNX Runtime (version CPU).

### 2. Préparer les modèles
Exécutez le script Python fourni pour télécharger les modèles pré-entraînés :
```bash
python ../assets/download_models.py
```
Les modèles seront enregistrés dans le répertoire `../models/`.

> **Remarque**: Nous téléchargerons les fichiers de modèles convertis dans les prochains jours. Restez à l'écoute.

### 3. Gestion des voix (Nouveau)
Nous recommandons d'extraire les caractéristiques vocales et de les enregistrer sous forme de fichiers `.qvoice` pour les réutiliser.

**Extraire une voix :**
```powershell
$env:PATH += ";$PWD\runtime"
cargo run --example make_voice --release -- `
    --model_dir ./models `
    --input clone.wav `
    --text "Contenu textuel de l'audio de référence" `
    --output my_voice.qvoice `
    --name "Ma voix personnalisée" `
    --gender "Female" `
    --age "Young" `
    --description "Voix de narration claire et douce"
```

**Générer avec un pack vocal :**
```powershell
cargo run --example qwen3-tts --release -- --model_dir ./models --voice my_voice.qvoice --text "Bonjour le monde"
```

### 4. Démo rapide
Utilisez le script `run.ps1` pour lancer la démo (gère automatiquement les chemins DLL) :
```powershell
.\run.ps1 --input clone.wav --ref_text "Contenu textuel de l'audio de référence" --text "Bonjour le monde"
```

Ou exécutez manuellement (assurez-vous que `runtime` est dans le PATH) :
```bash
$env:PATH += ";$PWD\runtime"
cargo run --example qwen3-tts --release -- --model_dir ./models --input clone.wav --ref_text "Contenu textuel de l'audio de référence" --text "Bonjour le monde"
```

## Utilisation comme bibliothèque
Ajoutez ceci à votre `Cargo.toml` :
```toml
[dependencies]
qwen3-tts = { path = "../path/to/qwen3-tts-rust" }
```

### Code exemple
```rust
use qwen3_tts::TtsEngine;
use std::path::Path;

fn main() -> Result<(), String> {
    // 1. Initialiser le moteur
    let model_dir = Path::new("models");
    let mut engine = TtsEngine::load(model_dir)?;

    // 2. Préparer l'entrée
    let text = "Bonjour, ceci est l'implémentation Rust de Qwen3-TTS.";
    let ref_audio = Path::new("clone.wav");
    let ref_text = "Ceci est le texte de l'audio de référence.";

    // 3. Générer l'audio
    let audio = engine.generate(text, ref_audio, ref_text)?;

    // 4. Sauvegarder
    audio.save_wav("output.wav")?;
    
    // 5. Nettoyer
    qwen3_tts::cleanup();
    
    Ok(())
}
```

## Remerciements
Merci aux projets suivants pour leur inspiration et leur soutien :
- [Qwen3-TTS-GGUF](https://github.com/HaujetZhao/Qwen3-TTS-GGUF) : Référence pour le flux d'inférence GGUF.
- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) : Dépôt officiel de Qwen3-TTS.

## Licence
MIT / Apache 2.0

