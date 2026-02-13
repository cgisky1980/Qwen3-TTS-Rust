//! Qwen3-TTS Rust Library
//! Optimized and organized structure for open source.

pub mod assets_manager;
pub mod download;
pub mod models;
pub mod tts;
pub mod utils;

// Re-export core types for convenience
pub use tts::engine::SamplerConfig;
pub use tts::engine::TtsEngine;
pub use tts::prompt::PromptBuilder;
pub use utils::audio::AudioSample;
pub use utils::tokenizer::Tokenizer;
pub use utils::voice_file::VoiceFile;

pub fn cleanup() {
    models::llama::cleanup();
}
