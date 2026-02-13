use crate::assets_manager::Assets;
use crate::models::llama::{LlamaBatch, LlamaContext, LlamaModel, LlamaSampler};
use crate::models::onnx::{AudioDecoder, AudioEncoder, SpeakerEncoder};
use crate::tts::prompt::PromptBuilder;
use crate::utils::cache;
use crate::utils::tokenizer::Tokenizer;
use crate::utils::voice_file::VoiceFile;
use crate::AudioSample;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Sampler configuration for TTS generation
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    /// Temperature for sampling (higher = more random, 0.0 = greedy)
    pub temperature: f32,
    /// Top-K sampling (0 = disabled)
    pub top_k: i32,
    /// Top-P (nucleus) sampling (1.0 = disabled)
    pub top_p: f32,
    /// Random seed (None = use system entropy)
    pub seed: Option<u64>,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.5,
            top_k: 50,
            top_p: 1.0,
            seed: None,
        }
    }
}

impl SamplerConfig {
    pub fn new(temperature: f32, top_k: i32, top_p: f32, seed: Option<u64>) -> Self {
        Self {
            temperature,
            top_k,
            top_p,
            seed,
        }
    }
}

/// Main TTS Engine Struct
///
/// IMPORTANT: Field ordering matters for Drop!
/// Rust drops fields in declaration order. Contexts MUST be declared before models
/// because context destructors reference model memory. If models are dropped first,
/// context destructors will access freed memory (ACCESS_VIOLATION).
pub struct TtsEngine {
    assets: Assets,
    tokenizer: Tokenizer,
    // ONNX Models
    encoder: Option<AudioEncoder>,
    speaker_encoder: Option<SpeakerEncoder>,
    // Llama: Contexts MUST be listed before models for correct drop order
    talker_ctx: LlamaContext,
    predictor_ctx: LlamaContext,
    talker_model: LlamaModel,
    predictor_model: LlamaModel,

    // Speakers Cache
    speakers: HashMap<String, VoiceFile>,

    // Config
    model_dir: PathBuf,
    max_steps: usize,
    sampler_config: SamplerConfig,
}

impl TtsEngine {
    /// Initialize the TTS Engine from the specified model directory.
    ///
    /// This function loads all necessary models (GGUF, Onnx, Tokenizer) from the given directory.
    /// It ensures that the essential components for inference are present.
    ///
    /// # Arguments
    ///
    /// * `model_dir` - Path to the directory containing model files.
    /// * `quant` - Quantization level (e.g., "none", "q5_k_m", "q8_0").
    pub async fn new(model_dir: impl AsRef<Path>, quant: &str) -> Result<Self, String> {
        let model_dir = model_dir.as_ref();
        println!("Loading TtsEngine from: {:?} (quant: {})", model_dir, quant);

        // 0. Auto-download check (Models + Runtimes)
        Self::download_models(model_dir, quant).await?;

        let quant_dir = match quant {
            "q5_k_m" => "gguf_q5_k_m",
            "q8_0" => "gguf_q8_0",
            _ => "gguf",
        };

        // 1. Assets
        let assets_path = model_dir.join(quant_dir);
        let assets =
            Assets::load(&assets_path).map_err(|e| format!("Failed to load assets: {}", e))?;

        // 2. Tokenizer
        let tokenizer =
            Tokenizer::load(model_dir).map_err(|e| format!("Failed to load tokenizer: {}", e))?;

        // 3. ONNX Models (Optional for preset mode, but good to have)
        let onnx_dir = model_dir.join("onnx");
        let encoder = AudioEncoder::load(
            &onnx_dir
                .join("qwen3_tts_codec_encoder.onnx")
                .to_string_lossy(),
        )
        .ok();

        let speaker_encoder = SpeakerEncoder::load(
            &onnx_dir
                .join("qwen3_tts_speaker_encoder.onnx")
                .to_string_lossy(),
        )
        .ok();

        // 4. Load GGUF Models
        let talker_path = model_dir.join(quant_dir).join("qwen3_tts_talker.gguf");
        let predictor_path = model_dir.join(quant_dir).join("qwen3_tts_predictor.gguf");

        let talker_model = LlamaModel::load(&talker_path, 99)
            .map_err(|e| format!("Failed to load Talker: {}", e))?;

        let predictor_model = LlamaModel::load(&predictor_path, 99)
            .map_err(|e| format!("Failed to load Predictor: {}", e))?;

        // 5. Create Contexts
        let talker_ctx = LlamaContext::new(&talker_model, 4096, 2048, 1, -1)
            .map_err(|e| format!("Failed to create Talker context: {}", e))?;

        let predictor_ctx = LlamaContext::new(&predictor_model, 512, 32, 0, 4)
            .map_err(|e| format!("Failed to create Predictor context: {}", e))?;

        println!("TtsEngine loaded successfully.");

        let mut engine = Self {
            assets,
            tokenizer,
            encoder,
            speaker_encoder,
            talker_model,
            predictor_model,
            talker_ctx,
            predictor_ctx,
            speakers: HashMap::new(),
            model_dir: model_dir.to_path_buf(),
            max_steps: 512,
            sampler_config: SamplerConfig::default(),
        };

        // 6. Load Speakers
        let speakers_dir = model_dir.join("preset_speakers"); // Default to preset directory
        let speakers_dir = if speakers_dir.exists() {
            speakers_dir
        } else {
            PathBuf::from("speakers")
        };

        if speakers_dir.exists() {
            engine.load_speakers(&speakers_dir)?;
        }

        Ok(engine)
    }

    /// Set the maximum number of generation steps (tokens).
    pub fn set_max_steps(&mut self, steps: usize) {
        self.max_steps = steps;
    }

    /// Set the sampler configuration for generation.
    pub fn set_sampler_config(&mut self, config: SamplerConfig) {
        self.sampler_config = config;
    }

    /// Get the current sampler configuration.
    pub fn get_sampler_config(&self) -> &SamplerConfig {
        &self.sampler_config
    }

    /// Load all speakers from the specified directory.
    pub fn load_speakers(&mut self, speakers_dir: impl AsRef<Path>) -> Result<(), String> {
        let speakers_dir = speakers_dir.as_ref();
        println!("Loading speakers from: {:?}", speakers_dir);

        let entries = std::fs::read_dir(speakers_dir).map_err(|e| e.to_string())?;
        for entry in entries {
            let entry = entry.map_err(|e| e.to_string())?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                if let Ok(voice) = VoiceFile::load(&path) {
                    let id = path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("unknown")
                        .to_string();
                    self.speakers.insert(id, voice);
                }
            }
        }
        println!("Loaded {} speakers.", self.speakers.len());
        Ok(())
    }

    /// Get a speaker by ID or name, with fallback to "vivian".
    pub fn get_speaker(&self, id_or_name: &str) -> &VoiceFile {
        if let Some(v) = self.speakers.get(id_or_name) {
            return v;
        }
        // Fallback to name match
        for v in self.speakers.values() {
            if let Some(ref name) = v.name {
                if name == id_or_name {
                    return v;
                }
            }
        }
        // Final fallback to vivian
        self.speakers.get("vivian").unwrap_or_else(|| {
            // Panic if even vivian is missing and cache is empty
            self.speakers
                .values()
                .next()
                .expect("No speakers loaded in engine!")
        })
    }

    /// Helper to download necessary files before loading.
    pub async fn download_models(model_dir: impl AsRef<Path>, quant: &str) -> Result<(), String> {
        let downloader = crate::download::Downloader::new().await;
        downloader
            .check_and_download(model_dir.as_ref(), quant)
            .await
            .map_err(|e| format!("Download failed: {}", e))
    }

    /// Generate speech from text using a reference audio.
    pub fn generate(
        &mut self,
        text: &str,
        ref_audio_path: impl AsRef<Path>,
        ref_text: &str,
        instruct: Option<&str>,
    ) -> Result<AudioSample, String> {
        let ref_audio_path = ref_audio_path.as_ref();

        // 1. Process Reference Audio
        let (ref_codes, spk_emb) = self.process_reference(ref_audio_path)?;

        // 2. Build Prompt
        // lang_id = 2055 (Chinese) hardcoded for now or parameterize later
        let ref_text_ids = self.tokenizer.encode(ref_text);
        let ref_codes_i32: Vec<i32> = ref_codes.iter().map(|&c| c as i32).collect();

        let prompt_data = PromptBuilder::build_clone_prompt(
            text,
            &self.tokenizer,
            &self.assets,
            &ref_codes_i32,
            &ref_text_ids,
            &spk_emb,
            2055,
            instruct,
        );

        self.run_inference(prompt_data)
    }

    /// Process reference audio to get codes and speaker embedding, using cache if available.
    fn process_reference(&mut self, audio_path: &Path) -> Result<(Vec<i64>, Vec<f32>), String> {
        let cache_path = audio_path.with_extension("cache");
        if cache_path.exists() {
            if let Ok((c, e)) = cache::load_cache(&cache_path) {
                return Ok((c, e));
            }
        }

        let audio = AudioSample::load_wav(audio_path)
            .map_err(|e| format!("Failed to load audio: {}", e))?;

        let ref_codes = self
            .encoder
            .as_mut()
            .ok_or("AudioEncoder not loaded (required for processing raw audio)".to_string())?
            .encode(&audio.samples)
            .map_err(|e| format!("Audio encode failed: {}", e))?;
        let spk_emb = self
            .speaker_encoder
            .as_mut()
            .ok_or("SpeakerEncoder not loaded (required for processing raw audio)".to_string())?
            .encode(&audio.samples)
            .map_err(|e| format!("Speaker extraction failed: {}", e))?;

        let _ = cache::save_cache(&cache_path, &ref_codes, &spk_emb);

        Ok((ref_codes, spk_emb))
    }

    // --- Helpers ---

    fn qwen3_position(start: i32, len: usize) -> Vec<i32> {
        let mut pos = Vec::with_capacity(len * 4);
        let range: Vec<i32> = (start..start + len as i32).collect();
        pos.extend_from_slice(&range); // Temporal
        pos.extend_from_slice(&range); // Height
        pos.extend_from_slice(&range); // Width
        pos.extend(std::iter::repeat_n(0, len)); // Channel
        pos
    }

    fn normal_position(cur_pos: usize, n_tokens: usize) -> Vec<i32> {
        (0..n_tokens).map(|i| (cur_pos + i) as i32).collect()
    }

    /// Create a VoiceFile from a reference audio file and its text.
    ///
    /// Requires that AudioEncoder and SpeakerEncoder are loaded.
    /// The reference audio MUST be 24000Hz.
    pub fn create_voice_file(
        &mut self,
        audio_path: impl AsRef<Path>,
        ref_text: String,
    ) -> Result<crate::utils::voice_file::VoiceFile, String> {
        let encoder = self.encoder.as_mut().ok_or(
            "AudioEncoder not loaded. Please ensure models/onnx/qwen3_tts_codec_encoder.onnx exists.",
        )?;
        let speaker_encoder = self.speaker_encoder.as_mut().ok_or(
            "SpeakerEncoder not loaded. Please ensure models/onnx/qwen3_tts_speaker_encoder.onnx exists.",
        )?;

        // 1. Load Audio
        let mut reader =
            hound::WavReader::open(audio_path).map_err(|e| format!("WAV error: {}", e))?;
        let spec = reader.spec();

        if spec.sample_rate != 24000 {
            return Err(format!(
                "Expected 24000Hz audio, found {}Hz",
                spec.sample_rate
            ));
        }

        let audio: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
            (hound::SampleFormat::Float, 32) => {
                reader.samples::<f32>().map(|s| s.unwrap_or(0.0)).collect()
            }
            (hound::SampleFormat::Int, 16) => reader
                .samples::<i16>()
                .map(|s| (s.unwrap_or(0) as f32) / 32768.0)
                .collect(),
            (hound::SampleFormat::Int, 32) => reader
                .samples::<i32>()
                .map(|s| (s.unwrap_or(0) as f32) / 2147483648.0)
                .collect(),
            _ => {
                return Err(format!(
                    "Unsupported WAV format: {:?} {} bits",
                    spec.sample_format, spec.bits_per_sample
                ))
            }
        };

        // If stereo, take channel 1
        let audio = if spec.channels > 1 {
            audio.chunks(spec.channels as usize).map(|c| c[0]).collect()
        } else {
            audio
        };

        // 2. Run Encoders
        println!("Extracting audio codes...");
        let audio_codes = encoder.encode(&audio).map_err(|e| e.to_string())?;

        println!("Extracting speaker embedding...");
        let speaker_embedding = speaker_encoder.encode(&audio).map_err(|e| e.to_string())?;

        Ok(crate::utils::voice_file::VoiceFile::new(
            ref_text,
            audio_codes,
            speaker_embedding,
        ))
    }

    /// Generate speech using a pre-loaded VoiceFile.
    pub fn generate_with_voice(
        &mut self,
        text: &str,
        voice: &crate::VoiceFile,
        instruct: Option<&str>,
    ) -> Result<AudioSample, String> {
        eprintln!("Debug: generate_with_voice started for text: '{}'", text);

        let prompt_data = if voice.audio_codes.is_empty() {
            eprintln!("Debug: Reference codes empty. Using custom prompt with spk_emb.");
            // Determine if we should use spk_id or spk_emb.
            // VoiceFile currently doesn't store spk_id directly in a structured way that is easily accessible here
            // unless we added it. But VoiceFile has speaker_embedding.
            PromptBuilder::build_core(
                text,
                &self.tokenizer,
                &self.assets,
                Some(2055), // Chinese
                None,       // spk_id (not in VoiceFile)
                Some(&voice.speaker_embedding),
                instruct,
                None,
            )
        } else {
            eprintln!("Debug: Reference codes provided. Using clone prompt.");
            let ref_text_ids = self.tokenizer.encode(&voice.ref_text);
            let ref_codes_i32: Vec<i32> = voice.audio_codes.iter().map(|&c| c as i32).collect();

            PromptBuilder::build_clone_prompt(
                text,
                &self.tokenizer,
                &self.assets,
                &ref_codes_i32,
                &ref_text_ids,
                &voice.speaker_embedding,
                2055,
                instruct,
            )
        };

        eprintln!(
            "Debug: Prompt built ({} embeds). Starting inference...",
            prompt_data.embd.len()
        );
        self.run_inference(prompt_data)
    }

    // Refactor generation logic into a private helper to avoid code duplication
    fn run_inference(
        &mut self,
        prompt_data: crate::tts::prompt::PromptData,
    ) -> Result<AudioSample, String> {
        self.run_inference_stream(prompt_data, None)
    }

    fn run_inference_stream(
        &mut self,
        prompt_data: crate::tts::prompt::PromptData,
        stream_tx: Option<std::sync::mpsc::Sender<Vec<f32>>>,
    ) -> Result<AudioSample, String> {
        let n_tokens_prompt = prompt_data.embd.len();
        let prompt_embeds_flat: Vec<f32> = prompt_data.embd.iter().flatten().copied().collect();
        let talker_embd = self.talker_model.n_embd;
        let predictor_embd = self.predictor_model.n_embd;

        // Talker Prefill
        let mut talker_batch = LlamaBatch::new(4096, talker_embd, 1, 4);
        let pos_arr = Self::qwen3_position(0, n_tokens_prompt);
        talker_batch.set_embd(&prompt_embeds_flat, &pos_arr, 0);

        self.talker_ctx
            .decode(&mut talker_batch)
            .map_err(|e| format!("Talker prefill failed: {}", e))?;

        // Generation Loop
        let mut all_codes: Vec<i32> = Vec::new();
        let mut cur_pos = n_tokens_prompt;

        // Hoisted resources
        let mut predictor_batch = LlamaBatch::new(32, predictor_embd, 1, 1);
        let predictor_sampler = LlamaSampler::greedy(self.predictor_model.n_vocab);

        // Use sampler config for talker
        let seed = self.sampler_config.seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or_else(|_| rand::random())
        });
        let talker_sampler = LlamaSampler::new(
            self.talker_model.n_vocab,
            self.sampler_config.temperature,
            self.sampler_config.top_k,
            self.sampler_config.top_p,
            seed,
        );

        let (tx, rx) = std::sync::mpsc::channel::<(Vec<i64>, bool)>();
        let decoder_model_path = self
            .model_dir
            .join("onnx")
            .join("qwen3_tts_decoder.onnx")
            .to_string_lossy()
            .to_string();

        let decoder_handle = std::thread::spawn(move || {
            let mut local_decoder = match AudioDecoder::load(&decoder_model_path) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("Failed to load decoder in thread: {}", e);
                    return Vec::new();
                }
            };
            let mut full_audio = Vec::new();
            let mut state = AudioDecoder::create_state();
            let mut code_buffer: Vec<i64> = Vec::with_capacity(64);

            while let Ok((codes, is_final)) = rx.recv() {
                code_buffer.extend(codes);
                // Accumulate 4 frames (64 codes) before decoding to balance overhead and latency
                if code_buffer.len() >= 64 || is_final {
                    // Truncate to multiple of 16 (one frame = 16 codes)
                    let valid_len = (code_buffer.len() / 16) * 16;
                    if valid_len > 0 {
                        // Clamp codes to valid range [0, 2047]
                        let safe_codes: Vec<i64> = code_buffer
                            .iter()
                            .take(valid_len)
                            .map(|&c| c.clamp(0, 2047))
                            .collect();
                        if let Ok(samples) = local_decoder.decode(&safe_codes, &mut state, is_final)
                        {
                            if let Some(ref stx) = stream_tx {
                                let _ = stx.send(samples.clone());
                            }
                            full_audio.extend(samples);
                        }
                        // Keep remaining codes (if any) for next iteration
                        let remaining = code_buffer.len() - valid_len;
                        if remaining > 0 && !is_final {
                            code_buffer.drain(0..valid_len);
                        } else {
                            code_buffer.clear();
                        }
                    } else {
                        code_buffer.clear();
                    }
                }
                if is_final {
                    break;
                }
            }
            full_audio
        });

        for step in 0..self.max_steps {
            print!("\r    Generation Step {}/{}...", step + 1, self.max_steps);
            let _ = std::io::Write::flush(&mut std::io::stdout());

            // Talker
            let sample_idx = if cur_pos == n_tokens_prompt {
                (n_tokens_prompt - 1) as i32
            } else {
                0
            };
            let code_0 = talker_sampler.sample(&self.talker_ctx, sample_idx, None, Some(2160));
            // eprintln!(" Debug: Step {} code_0 = {}", step, code_0); // Restore if needed

            if code_0 == 2150 || code_0 == 151673 {
                println!("\n    EOS detected at step {} (code_0={})", step, code_0);
                break;
            }
            all_codes.push(code_0);

            // Predictor
            let emb_idx = if step == 0 { n_tokens_prompt - 1 } else { 0 };
            let m_hidden = self.talker_ctx.get_embedding_at(emb_idx).to_vec();

            let m_h_1024 = self.assets.project(&m_hidden);
            let code_0_1024 = self.assets.get_codec_embedding_1024(0, code_0);

            let mut predictor_input = Vec::with_capacity(2 * predictor_embd);
            predictor_input.extend_from_slice(&m_h_1024);
            predictor_input.extend_from_slice(&code_0_1024);

            self.predictor_ctx.clear_kv_cache();
            predictor_batch.clear();
            let pred_pos = Self::normal_position(0, 2);
            predictor_batch.set_embd(&predictor_input, &pred_pos, 0);

            self.predictor_ctx
                .decode(&mut predictor_batch)
                .map_err(|e| format!("Predictor prefill failed: {}", e))?;

            let mut step_embeds_2048: Vec<Vec<f32>> = Vec::new();
            step_embeds_2048.push(self.assets.get_codec_embedding(0, code_0));

            for q in 1..16 {
                let start_offset = (q - 1) * 2048;
                let end_offset = q * 2048;
                let sampled = predictor_sampler.sample(
                    &self.predictor_ctx,
                    0,
                    Some(start_offset),
                    Some(end_offset),
                );
                let code_q = sampled - start_offset as i32;
                all_codes.push(code_q);

                let emb = self.assets.get_codec_embedding(q, code_q);
                step_embeds_2048.push(emb.to_vec());

                if q < 15 {
                    let next_embed_1024 = self.assets.get_codec_embedding_1024(q, code_q);
                    let next_pos = Self::normal_position(q + 1, 1);
                    predictor_batch.clear();
                    predictor_batch.set_embd(&next_embed_1024, &next_pos, 0);
                    self.predictor_ctx
                        .decode(&mut predictor_batch)
                        .map_err(|e| format!("Predictor decode failed: {}", e))?;
                }
            }

            let frame_codes: Vec<i64> = all_codes
                .iter()
                .rev()
                .take(16)
                .rev()
                .map(|&c| c as i64)
                .collect();
            let _ = tx.send((frame_codes, false));

            let mut feedback = vec![0.0f32; 2048];
            for embed in &step_embeds_2048 {
                for (i, val) in embed.iter().enumerate() {
                    feedback[i] += val;
                }
            }
            for (i, val) in self.assets.tts_pad.iter().enumerate() {
                feedback[i] += val;
            }
            feedback.resize(talker_embd, 0.0);

            let talker_pos = Self::qwen3_position(cur_pos as i32, 1);
            talker_batch.clear();
            talker_batch.set_embd(&feedback, &talker_pos, 0);

            self.talker_ctx
                .decode(&mut talker_batch)
                .map_err(|e| format!("Talker step failed: {}", e))?;

            cur_pos += 1;
        }

        let _ = tx.send((Vec::new(), true));
        drop(tx);

        let audio_samples = decoder_handle
            .join()
            .map_err(|_| "Decoder thread panicked".to_string())?;

        Ok(AudioSample {
            samples: audio_samples,
            sample_rate: 24000,
            channels: 1,
        })
    }
}
