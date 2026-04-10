use crate::assets_manager::Assets;
use crate::models::llama::{LlamaBatch, LlamaContext, LlamaModel, LlamaSampler};
use crate::models::onnx::{init_onruntime, AudioDecoder, AudioEncoder, SpeakerEncoder};
use crate::tts::prompt::PromptBuilder;
use crate::utils::cache;
use crate::utils::tokenizer::Tokenizer;
use crate::utils::voice_file::VoiceFile;
use crate::AudioSample;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Sampler configuration for TTS generation
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    /// Temperature for sampling (higher = more random, 0.0 = greedy)
    pub temperature: f32,
    /// Top-K sampling (0 = disabled)
    pub top_k: i32,
    /// Top-P (nucleus) sampling (1.0 = disabled)
    pub top_p: f32,
    /// Min-P sampling threshold (0.0 = disabled)
    pub min_p: f32,
    /// Repeat penalty (1.0 = disabled)
    pub repeat_penalty: f32,
    /// Frequency penalty (0.0 = disabled)
    pub frequency_penalty: f32,
    /// Presence penalty (0.0 = disabled)
    pub presence_penalty: f32,
    /// Number of recent tokens to consider for penalties
    pub penalty_last_n: usize,
    /// Random seed (None = use system entropy)
    pub seed: Option<u64>,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.9,
            top_k: 50,
            top_p: 1.0,
            min_p: 0.0,
            repeat_penalty: 1.05,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            penalty_last_n: 128,
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
            min_p: 0.0,
            repeat_penalty: 1.05,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            penalty_last_n: 128,
            seed,
        }
    }

    pub fn with_penalties(
        mut self,
        min_p: f32,
        repeat_penalty: f32,
        frequency_penalty: f32,
        presence_penalty: f32,
        penalty_last_n: usize,
    ) -> Self {
        self.min_p = min_p;
        self.repeat_penalty = repeat_penalty;
        self.frequency_penalty = frequency_penalty;
        self.presence_penalty = presence_penalty;
        self.penalty_last_n = penalty_last_n;
        self
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
    decoder: Arc<Mutex<AudioDecoder>>,
    // Llama: Contexts MUST be listed before models for correct drop order
    talker_ctx: LlamaContext,
    predictor_ctx: LlamaContext,
    talker_model: LlamaModel,
    predictor_model: LlamaModel,

    // Speakers Cache
    speakers: HashMap<String, VoiceFile>,

    // Config
    _model_dir: PathBuf,
    max_steps: usize,
    sampler_config: SamplerConfig,
}

impl TtsEngine {
    /// Same values as `LlamaContext::new` for the talker (prefill must respect both).
    const TALKER_N_CTX: usize = 4096;
    const TALKER_N_BATCH: usize = 2048;

    /// Initialize the TTS Engine from the specified model directory.
    ///
    /// This function loads all necessary models (GGUF, Onnx, Tokenizer) from the given directory.
    /// It ensures that the essential components for inference are present.
    ///
    /// # Arguments
    ///
    /// * `model_dir` - Path to the directory containing model files.
    /// * `quant` - Quantization level (e.g., "none", "q5_k_m", "q8_0").
    /// * `n_threads` - Number of threads to use for generation (default: 4 if <= 0).
    pub async fn new(
        model_dir: impl AsRef<Path>,
        quant: &str,
        _n_threads: i32,
    ) -> Result<Self, String> {
        let model_dir = model_dir.as_ref();
        println!("Loading TtsEngine from: {:?} (quant: {})", model_dir, quant);

        // 0. Auto-download check (Models + Runtimes)
        Self::download_models(model_dir, quant).await?;

        let quant_dir = match quant {
            "q5_k_m" => "gguf_q5_k_m",
            "q8_0" => "gguf_q8_0",
            _ => "gguf",
        };

        // 1. Assets - 总是从未量化的 gguf 目录加载（assets 不应该被量化）
        let assets_path = model_dir.join("gguf");
        let assets =
            Assets::load(&assets_path).map_err(|e| format!("Failed to load assets: {}", e))?;

        // 2. Tokenizer
        let tokenizer =
            Tokenizer::load(model_dir).map_err(|e| format!("Failed to load tokenizer: {}", e))?;

        // 3. Initialize ONNX Runtime (must be called before any ONNX session)
        println!("Initializing ONNX Runtime...");
        init_onruntime().map_err(|e| format!("Failed to init ONNX Runtime: {}", e))?;

        // 4. ONNX Models (Optional for preset mode, but good to have)
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

        // 5. Load GGUF Models
        let talker_path = model_dir.join(quant_dir).join("qwen3_tts_talker.gguf");
        let predictor_path = model_dir.join(quant_dir).join("qwen3_tts_predictor.gguf");

        let talker_model = LlamaModel::load(&talker_path, 99)
            .map_err(|e| format!("Failed to load Talker: {}", e))?;

        let predictor_model = LlamaModel::load(&predictor_path, 99)
            .map_err(|e| format!("Failed to load Predictor: {}", e))?;

        // 5. Create Contexts
        // talker: n_ctx=4096, n_batch=2048, embeddings=1, threads=-1 (auto)
        let talker_ctx = LlamaContext::new(&talker_model, 4096, 2048, 1, -1)
            .map_err(|e| format!("Failed to create Talker context: {}", e))?;

        // predictor: n_ctx=512, n_batch=32, embeddings=0, threads=4
        let predictor_ctx = LlamaContext::new(&predictor_model, 512, 32, 0, 4)
            .map_err(|e| format!("Failed to create Predictor context: {}", e))?;

        // 6. 预加载 decoder（预热）
        println!("Pre-loading AudioDecoder...");
        let decoder =
            AudioDecoder::load(&onnx_dir.join("qwen3_tts_decoder.onnx").to_string_lossy())
                .map_err(|e| format!("Failed to load AudioDecoder: {}", e))?;
        let decoder = Arc::new(Mutex::new(decoder));
        println!("AudioDecoder pre-loaded and warmed up.");

        println!("TtsEngine loaded successfully.");

        let mut engine = Self {
            assets,
            tokenizer,
            encoder,
            speaker_encoder,
            decoder,
            talker_model,
            predictor_model,
            talker_ctx,
            predictor_ctx,
            speakers: HashMap::new(),
            _model_dir: model_dir.to_path_buf(),
            max_steps: 4096,
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

    /// Get mutable sampler configuration.
    pub fn get_sampler_config_mut(&mut self) -> &mut SamplerConfig {
        &mut self.sampler_config
    }

    /// Get the speakers map.
    pub fn get_speakers_map(&self) -> &HashMap<String, VoiceFile> {
        &self.speakers
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
        self.generate_with_voice_streaming(text, voice, instruct, None)
    }

    /// Generate speech using a pre-loaded VoiceFile.
    pub fn generate_with_voice_streaming(
        &mut self,
        text: &str,
        voice: &crate::VoiceFile,
        instruct: Option<&str>,
        stream_tx: Option<tokio::sync::mpsc::UnboundedSender<Vec<f32>>>,
    ) -> Result<AudioSample, String> {
        let prompt_data = if voice.audio_codes.is_empty() {
            PromptBuilder::build_core(
                text,
                &self.tokenizer,
                &self.assets,
                Some(2055),
                None,
                Some(&voice.speaker_embedding),
                instruct,
                None,
            )
        } else {
            let ref_text_ids = self.tokenizer.encode(&voice.ref_text);
            let ref_codes_i32: Vec<i32> = voice.audio_codes.iter().map(|&c| c as i32).collect();

            Ok(PromptBuilder::build_clone_prompt(
                text,
                &self.tokenizer,
                &self.assets,
                &ref_codes_i32,
                &ref_text_ids,
                &voice.speaker_embedding,
                2055,
                instruct,
            ))
        }?;

        self.run_inference_stream(prompt_data, stream_tx)
    }

    fn run_inference(
        &mut self,
        prompt_data: crate::tts::prompt::PromptData,
    ) -> Result<AudioSample, String> {
        self.run_inference_stream(prompt_data, None)
    }

    fn run_inference_stream(
        &mut self,
        mut prompt_data: crate::tts::prompt::PromptData,
        stream_tx: Option<tokio::sync::mpsc::UnboundedSender<Vec<f32>>>,
    ) -> Result<AudioSample, String> {
        self.talker_ctx.clear_kv_cache();
        self.predictor_ctx.clear_kv_cache();

        if prompt_data.embd.len() > Self::TALKER_N_CTX {
            eprintln!(
                "WARNING: prompt length {} exceeds talker n_ctx {}. Truncating from the end of the fused prompt.",
                prompt_data.embd.len(),
                Self::TALKER_N_CTX
            );
            prompt_data.embd.truncate(Self::TALKER_N_CTX);
        }

        let n_tokens_prompt = prompt_data.embd.len();
        let prompt_embeds_flat: Vec<f32> = prompt_data.embd.iter().flatten().copied().collect();
        let talker_embd = self.talker_model.n_embd;
        let predictor_embd = self.predictor_model.n_embd;

        let mut talker_batch = LlamaBatch::new(Self::TALKER_N_BATCH, talker_embd, 1, 4);
        let mut talker_prefill_last_idx: i32 = 0;
        let mut prefill_off = 0usize;
        while prefill_off < n_tokens_prompt {
            let chunk_tokens =
                (n_tokens_prompt - prefill_off).min(Self::TALKER_N_BATCH);
            let slice_start = prefill_off * talker_embd;
            let slice_end = (prefill_off + chunk_tokens) * talker_embd;
            let chunk_flat = &prompt_embeds_flat[slice_start..slice_end];
            let pos_arr = Self::qwen3_position(prefill_off as i32, chunk_tokens);
            talker_batch.set_embd(chunk_flat, &pos_arr, 0);
            self.talker_ctx
                .decode(&mut talker_batch)
                .map_err(|e| format!("Talker prefill failed: {}", e))?;
            talker_prefill_last_idx = (chunk_tokens.saturating_sub(1)) as i32;
            prefill_off += chunk_tokens;
        }

        let mut all_codes: Vec<i32> = Vec::new();
        let mut cur_pos = n_tokens_prompt;

        let mut predictor_batch = LlamaBatch::new(32, predictor_embd, 1, 1);
        let predictor_sampler = LlamaSampler::greedy(self.predictor_model.n_vocab);

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

        let tts_pad = self.assets.tts_pad.clone();
        let decoder_arc = self.decoder.clone();

        let decoder_handle = std::thread::spawn(move || {
            let mut full_audio = Vec::new();
            let mut state = AudioDecoder::create_state();

            while let Ok((codes, is_final)) = rx.recv() {
                let n_frames = codes.len() / 16;

                if n_frames == 0 {
                    if is_final {
                        let mut local_decoder = decoder_arc.lock().unwrap();
                        if let Ok(samples) = local_decoder.decode(&[], &mut state, true) {
                            if !samples.is_empty() {
                                if let Some(ref stx) = stream_tx {
                                    let _ = stx.send(samples.clone());
                                }
                                full_audio.extend(samples);
                            }
                        }
                        break;
                    }
                    continue;
                }

                let safe_codes: Vec<i64> = codes.iter().map(|&c| c.clamp(0, 2047)).collect();

                let mut local_decoder = decoder_arc.lock().unwrap();
                if let Ok(samples) = local_decoder.decode(&safe_codes, &mut state, is_final) {
                    if !samples.is_empty() {
                        if let Some(ref stx) = stream_tx {
                            let _ = stx.send(samples.clone());
                        }
                        full_audio.extend(samples);
                    }
                }
            }
            full_audio
        });

        // 连续静音检测参数 (12.5Hz = 80ms/帧)
        const MAX_SILENT_FRAMES: usize = 15; // 1.2秒强制停止
        const SILENT_PENALTY_THRESHOLD: usize = 4; // 0.3秒(4帧)后开始惩罚
        const SILENT_PENALTY_MAX_FRAMES: usize = 8; // 0.6秒(8帧)后极大惩罚
        const SILENT_PENALTY_VALUE: f32 = 2.0;

        // 流式发送参数
        const SEND_INTERVAL: usize = 4;
        let mut consecutive_silent_frames: usize = 0;
        let mut last_sent_frame: usize = 0;

        for step in 0..self.max_steps {
            print!("\r    Generation Step {}/{}...", step + 1, self.max_steps);
            let _ = std::io::Write::flush(&mut std::io::stdout());

            // Talker
            let sample_idx = if cur_pos == n_tokens_prompt {
                talker_prefill_last_idx
            } else {
                0
            };

            let allow_tokens: Vec<i32> = vec![2150, 2148, 2149];

            let code_0 = if consecutive_silent_frames >= SILENT_PENALTY_THRESHOLD {
                let penalty = if consecutive_silent_frames >= SILENT_PENALTY_MAX_FRAMES {
                    100.0 // 极大惩罚，强制退出静音
                } else {
                    SILENT_PENALTY_VALUE
                        * (1.0
                            + (consecutive_silent_frames - SILENT_PENALTY_THRESHOLD) as f32 * 0.5)
                };
                eprintln!(
                    "[Debug] silent {} frames, penalty: {:.1}",
                    consecutive_silent_frames, penalty
                );
                talker_sampler.sample_with_silent_penalty(
                    &self.talker_ctx,
                    sample_idx,
                    Some(0),
                    Some(2048),
                    Some(&allow_tokens),
                    penalty,
                    SILENT_PENALTY_MAX_FRAMES, // 只对前8帧应用惩罚
                )
            } else {
                talker_sampler.sample_with_allow(
                    &self.talker_ctx,
                    sample_idx,
                    Some(0),
                    Some(2048),
                    Some(&allow_tokens),
                )
            };
            eprintln!("[Debug] Sampled code_0={}", code_0);

            if code_0 == 2150 || code_0 == 151673 {
                println!("\n    EOS detected at step {} (code_0={})", step, code_0);
                break;
            }

            let is_silent_frame = code_0 < 100;
            if is_silent_frame {
                consecutive_silent_frames += 1;
                eprintln!(
                    "[Debug] silent frame {}, code_0={}",
                    consecutive_silent_frames, code_0
                );
                if consecutive_silent_frames >= MAX_SILENT_FRAMES {
                    eprintln!(
                        "[Debug] Early stop: {} consecutive silent frames",
                        consecutive_silent_frames
                    );
                    println!(
                        "\n    Early stop: {} consecutive silent frames detected",
                        consecutive_silent_frames
                    );
                    break;
                }
            } else {
                consecutive_silent_frames = 0;
            }

            all_codes.push(code_0);

            // Predictor
            let emb_idx = if step == 0 {
                talker_prefill_last_idx as usize
            } else {
                0
            };
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

            // 流式发送
            let current_frame = all_codes.len() / 16;

            if current_frame > last_sent_frame + SEND_INTERVAL {
                let start_frame = last_sent_frame;
                let end_frame = last_sent_frame + SEND_INTERVAL;

                let start_idx = start_frame * 16;
                let end_idx = end_frame * 16;
                let frame_codes: Vec<i64> = all_codes[start_idx..end_idx]
                    .iter()
                    .map(|&c| c as i64)
                    .collect();

                let _ = tx.send((frame_codes, false));
                last_sent_frame += SEND_INTERVAL;
            }

            let mut feedback = vec![0.0f32; 2048];
            for embed in &step_embeds_2048 {
                for (i, val) in embed.iter().enumerate() {
                    feedback[i] += val;
                }
            }

            let text_vec = &tts_pad;
            for (i, val) in text_vec.iter().enumerate() {
                if i < feedback.len() {
                    feedback[i] += val;
                }
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

        // 发送剩余的帧
        let total_frames = all_codes.len() / 16;

        if total_frames > last_sent_frame {
            let start_frame = last_sent_frame;
            let end_frame = total_frames;

            let start_idx = start_frame * 16;
            let end_idx = end_frame * 16;
            let frame_codes: Vec<i64> = all_codes[start_idx..end_idx]
                .iter()
                .map(|&c| c as i64)
                .collect();

            let _ = tx.send((frame_codes, true));
        } else {
            let _ = tx.send((Vec::new(), true));
        }
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
