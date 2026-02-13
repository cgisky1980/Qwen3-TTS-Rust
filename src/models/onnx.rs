//! 真正的 ONNX 模型推理实现
//! 使用 ort crate 进行 ONNX Runtime 推理
//! 执行提供者优先级: DirectML -> CUDA -> CPU

use ndarray::{Array3, Array4};
use ort::execution_providers::{CUDAExecutionProvider, DirectMLExecutionProvider};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::session::SessionInputValue;
use ort::value::Tensor;
use std::error::Error;

/// 创建 Session，自动选择最佳执行提供者
/// 优先级: DirectML -> CUDA -> CPU
#[allow(dead_code)]
fn create_session_with_best_ep(model_path: &str) -> Result<Session, Box<dyn Error>> {
    // 尝试按优先级注册执行提供者
    // ort 会自动 fallback 到下一个可用的提供者

    println!("  [ONNX] Requesting Execution Providers: CUDA -> DirectML -> CPU");
    let builder = Session::builder()?;
    println!("  [ONNX] Session Builder created.");

    let builder = builder.with_optimization_level(GraphOptimizationLevel::Level3)?;
    println!("  [ONNX] Optimization level set.");

    // Split execution provider configuration for debugging
    let cuda = CUDAExecutionProvider::default().build();
    let dml = DirectMLExecutionProvider::default().build();
    let cpu = ort::execution_providers::CPUExecutionProvider::default().build();

    let builder = builder.with_execution_providers([
        // 优先级 1: CUDA (NVIDIA GPU)
        cuda, // 优先级 2: DirectML (Windows DirectX 12)
        dml,  // CPU 是默认的 fallback
        cpu,
    ])?;
    println!("  [ONNX] Execution Providers configured.");

    let session = builder.commit_from_file(model_path)?;
    println!("  [ONNX] Session committed from file.");

    Ok(session)
}

/// 创建 CPU Session (为了简化部署和减少 DLL 依赖)
fn create_cpu_session(model_path: &str) -> Result<Session, Box<dyn Error>> {
    println!("  [ONNX] Requesting Execution Providers: CPU");
    let builder = Session::builder()?;
    println!("  [ONNX] CPU Session Builder created.");

    let builder = builder.with_optimization_level(GraphOptimizationLevel::Level3)?;

    let cpu = ort::execution_providers::CPUExecutionProvider::default().build();
    let builder = builder.with_execution_providers([cpu])?;
    println!("  [ONNX] CPU Provider configured.");

    let session = builder.commit_from_file(model_path)?;
    println!("  [ONNX] CPU Session committed.");

    Ok(session)
}

/// 打印当前使用的执行提供者信息
fn print_session_info(session: &Session, name: &str) {
    let inputs: Vec<_> = session
        .inputs()
        .iter()
        .map(|i| i.name().to_string())
        .collect();
    let outputs: Vec<_> = session
        .outputs()
        .iter()
        .map(|o| o.name().to_string())
        .collect();
    println!("    Inputs: {:?}", inputs);
    println!("    Outputs: {:?}", outputs);
    println!("  [ONNX] {}: Loaded successfully", name);
}

/// 音频编码器 - 将 24kHz 音频编码为 codec codes
pub struct AudioEncoder {
    session: Session,
}

impl AudioEncoder {
    pub fn load(model_path: &str) -> Result<Self, Box<dyn Error>> {
        println!("  [ONNX] AudioEncoder: Loading {}", model_path);

        // Revert to CPU for stability and minimal dependencies
        let session = create_cpu_session(model_path)?;
        print_session_info(&session, "AudioEncoder");

        Ok(AudioEncoder { session })
    }

    pub fn encode(&mut self, audio: &[f32]) -> Result<Vec<i64>, Box<dyn Error>> {
        println!("  [ONNX] AudioEncoder: Encoding {} samples", audio.len());

        // 使用 (shape, data) 形式创建 Tensor
        let shape = vec![1i64, audio.len() as i64];
        let input_tensor = Tensor::from_array((shape, audio.to_vec()))?;

        let outputs = self
            .session
            .run(ort::inputs!["input_values" => input_tensor])?;

        // 输出: audio_codes [1, T//2000, 16]
        let codes_output = &outputs["audio_codes"];
        let codes_raw = codes_output.try_extract_tensor::<i64>()?;
        println!("  [ONNX] AudioEncoder: Output Shape: {:?}", codes_raw.0);
        let codes: Vec<i64> = codes_raw.1.to_vec();

        let n_frames = codes.len() / 16;
        println!(
            "  [ONNX] AudioEncoder: Encoded {} frames, {} codes",
            n_frames,
            codes.len()
        );
        Ok(codes)
    }
}

/// 说话人编码器 - 提取说话人嵌入
pub struct SpeakerEncoder {
    session: Session,
}

impl SpeakerEncoder {
    pub fn load(model_path: &str) -> Result<Self, Box<dyn Error>> {
        println!("  [ONNX] SpeakerEncoder: Loading {}", model_path);

        // Revert to CPU for stability
        let session = create_cpu_session(model_path)?;
        print_session_info(&session, "SpeakerEncoder");

        Ok(SpeakerEncoder { session })
    }

    pub fn encode(&mut self, audio: &[f32]) -> Result<Vec<f32>, Box<dyn Error>> {
        println!(
            "  [ONNX] SpeakerEncoder: Extracting from {} samples",
            audio.len()
        );

        // 计算 Mel 谱图 (简化版)
        let (n_frames, mels_data) = self.compute_mel(audio);
        let shape = vec![1i64, n_frames as i64, 128i64];
        let mels_tensor = Tensor::from_array((shape, mels_data))?;

        let outputs = self.session.run(ort::inputs!["mels" => mels_tensor])?;

        // 输出: spk_emb [1, 2048]
        let emb_output = &outputs["spk_emb"];
        let emb_raw = emb_output.try_extract_tensor::<f32>()?;
        let emb: Vec<f32> = emb_raw.1.to_vec();

        println!(
            "  [ONNX] SpeakerEncoder: Extracted {} dims embedding",
            emb.len()
        );
        Ok(emb)
    }

    /// 计算 Mel 谱图 (手动实现，完全对齐 Python librosa)
    /// 参数: 24kHz, n_fft=1024, hop=256, n_mels=128, fmin=0, fmax=12000
    fn compute_mel(&self, audio: &[f32]) -> (usize, Vec<f32>) {
        use rustfft::{num_complex::Complex, FftPlanner};

        const SAMPLE_RATE: f32 = 24000.0;
        const N_FFT: usize = 1024;
        const HOP_LENGTH: usize = 256;
        const N_MELS: usize = 128;
        const FMIN: f32 = 0.0;
        const FMAX: f32 = 12000.0;

        // ==================== 1. Mel Filterbank (Slaney style) ====================

        // Hz to Mel (Slaney formula)
        fn hz_to_mel(freq: f32) -> f32 {
            let f_min = 0.0f32;
            let f_sp = 200.0 / 3.0;
            let min_log_hz = 1000.0f32;
            let min_log_mel = (min_log_hz - f_min) / f_sp;
            let logstep = 6.4f32.ln() / 27.0;

            if freq >= min_log_hz {
                min_log_mel + ((freq / min_log_hz).ln() / logstep)
            } else {
                (freq - f_min) / f_sp
            }
        }

        // Mel to Hz (Slaney formula)
        fn mel_to_hz(mel: f32) -> f32 {
            let f_min = 0.0f32;
            let f_sp = 200.0 / 3.0;
            let min_log_hz = 1000.0f32;
            let min_log_mel = (min_log_hz - f_min) / f_sp;
            let logstep = 6.4f32.ln() / 27.0;

            if mel >= min_log_mel {
                min_log_hz * (logstep * (mel - min_log_mel)).exp()
            } else {
                f_min + f_sp * mel
            }
        }

        // Create Mel filterbank (Slaney normalization)
        let n_fft_bins = N_FFT / 2 + 1;
        let mel_min = hz_to_mel(FMIN);
        let mel_max = hz_to_mel(FMAX);

        // Mel bin edges
        let mut mel_edges = Vec::with_capacity(N_MELS + 2);
        for i in 0..=(N_MELS + 1) {
            let mel = mel_min + (mel_max - mel_min) * (i as f32) / ((N_MELS + 1) as f32);
            mel_edges.push(mel_to_hz(mel));
        }

        // FFT bin frequencies
        let fft_freqs: Vec<f32> = (0..n_fft_bins)
            .map(|i| (i as f32) * SAMPLE_RATE / (N_FFT as f32))
            .collect();

        // Build filterbank matrix [n_mels, n_fft_bins]
        let mut mel_filterbank = vec![0.0f32; N_MELS * n_fft_bins];
        for m in 0..N_MELS {
            let f_left = mel_edges[m];
            let f_center = mel_edges[m + 1];
            let f_right = mel_edges[m + 2];

            // Slaney-style normalization: 2.0 / (f_right - f_left)
            let norm = 2.0 / (f_right - f_left);

            for (k, &freq) in fft_freqs.iter().enumerate() {
                let weight = if freq >= f_left && freq <= f_center {
                    (freq - f_left) / (f_center - f_left)
                } else if freq > f_center && freq <= f_right {
                    (f_right - freq) / (f_right - f_center)
                } else {
                    0.0
                };
                mel_filterbank[m * n_fft_bins + k] = weight * norm;
            }
        }

        // ==================== 2. STFT ====================

        // Reflect padding
        let padding = (N_FFT - HOP_LENGTH) / 2;
        let mut padded: Vec<f32> = Vec::with_capacity(padding + audio.len() + padding);

        // Reflect at start
        for i in (1..=padding).rev() {
            if i < audio.len() {
                padded.push(audio[i]);
            } else {
                padded.push(0.0);
            }
        }
        padded.extend_from_slice(audio);
        // Reflect at end
        for i in 1..=padding {
            let idx = audio.len().saturating_sub(1 + i);
            if idx < audio.len() {
                padded.push(audio[idx]);
            } else {
                padded.push(0.0);
            }
        }

        // Hann window
        let hann_window: Vec<f32> = (0..N_FFT)
            .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / N_FFT as f32).cos()))
            .collect();

        // FFT planner
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(N_FFT);

        // Process frames
        let n_frames = (padded.len().saturating_sub(N_FFT)) / HOP_LENGTH + 1;
        let mut all_mels = Vec::with_capacity(n_frames * N_MELS);

        for frame_idx in 0..n_frames {
            let start = frame_idx * HOP_LENGTH;
            if start + N_FFT > padded.len() {
                break;
            }

            // Apply window and convert to complex
            let mut fft_buffer: Vec<Complex<f32>> = (0..N_FFT)
                .map(|i| Complex::new(padded[start + i] * hann_window[i], 0.0))
                .collect();

            // Perform FFT
            fft.process(&mut fft_buffer);

            // Compute magnitudes (add 1e-9 for numerical stability, like Python)
            let magnitudes: Vec<f32> = fft_buffer[..n_fft_bins]
                .iter()
                .map(|c| (c.norm_sqr() + 1e-9).sqrt())
                .collect();

            // Apply Mel filterbank
            for m in 0..N_MELS {
                let mut mel_val = 0.0f32;
                for k in 0..n_fft_bins {
                    mel_val += mel_filterbank[m * n_fft_bins + k] * magnitudes[k];
                }
                // Log compression: log(max(mel, 1e-5))
                let log_mel = mel_val.max(1e-5).ln();
                all_mels.push(log_mel);
            }
        }

        let actual_n_frames = all_mels.len() / N_MELS;
        (actual_n_frames, all_mels)
    }
}

/// 音频解码器 - 将 codec codes 解码为音频波形
pub struct AudioDecoder {
    session: Session,
}

impl AudioDecoder {
    pub fn load(model_path: &str) -> Result<Self, Box<dyn Error>> {
        println!("  [ONNX] AudioDecoder: Loading {}", model_path);

        let session = create_cpu_session(model_path)?;
        print_session_info(&session, "AudioDecoder");

        Ok(AudioDecoder { session })
    }

    pub fn create_state() -> DecoderState {
        DecoderState::new()
    }

    pub fn decode(
        &mut self,
        codes: &[i64],
        state: &mut DecoderState,
        is_final: bool,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        // println!("  [ONNX] AudioDecoder: Decoding {} codes (is_final={})", codes.len(), is_final);

        let n_frames = codes.len() / 16;
        if n_frames == 0 {
            return Ok(vec![]);
        }

        // 1. Prepare Inputs
        let mut inputs_vec: Vec<(std::borrow::Cow<'_, str>, SessionInputValue<'_>)> = Vec::new();

        // audio_codes: [1, N, 16] (if N>0)
        let codes_shape = vec![1i64, n_frames as i64, 16i64];
        let codes_tensor = Tensor::from_array((codes_shape, codes.to_vec()))?;
        inputs_vec.push(("audio_codes".into(), codes_tensor.into()));

        // is_last: [1]
        let is_last_val = if is_final { 1.0f32 } else { 0.0f32 };
        let is_last_tensor = Tensor::from_array((vec![1i64], vec![is_last_val]))?;
        inputs_vec.push(("is_last".into(), is_last_tensor.into()));

        // State Tensors
        let pre_conv_tensor = Tensor::from_array(state.pre_conv_history.clone().into_dyn())?;
        inputs_vec.push(("pre_conv_history".into(), pre_conv_tensor.into()));

        let latent_tensor = Tensor::from_array(state.latent_buffer.clone().into_dyn())?;
        inputs_vec.push(("latent_buffer".into(), latent_tensor.into()));

        let conv_tensor = Tensor::from_array(state.conv_history.clone().into_dyn())?;
        inputs_vec.push(("conv_history".into(), conv_tensor.into()));

        // KV Cache
        for (i, (k, v)) in state.kv_cache.iter().enumerate() {
            let k_tensor = Tensor::from_array(k.clone().into_dyn())?;
            let v_tensor = Tensor::from_array(v.clone().into_dyn())?;
            inputs_vec.push((format!("past_key_{}", i).into(), k_tensor.into()));
            inputs_vec.push((format!("past_value_{}", i).into(), v_tensor.into()));
        }

        // 2. Run Session
        let outputs = self.session.run(inputs_vec)?;

        // 3. Extract Audio
        let wav_output = &outputs["final_wav"];
        let wav_raw = wav_output.try_extract_tensor::<f32>()?;
        // If outputs["valid_samples"] exists, use it. Python says: outputs[1] is valid_samples.
        // Let's check output names in Python: final_wav, valid_samples, pre_conv_new, ...
        // We assume safe to retrieve by name.

        // Handling valid_samples which might be scalar or 1D
        // Handling valid_samples which might be scalar or 1D
        let valid_count = if let Some(valid_out) = outputs.get("valid_samples") {
            let valid_raw = valid_out.try_extract_tensor::<i64>()?;
            valid_raw.1[0] as usize
        } else {
            wav_raw.1.len() // Fallback if not present (should be)
        };

        let audio: Vec<f32> = wav_raw.1.iter().take(valid_count).cloned().collect();

        // 4. Update State (only if not final? Python says "if is_final, state might be dirty but unused")
        // We update unconditionally to match Python logic which returns new_state.

        let extract_3d = |name: &str| -> Result<Array3<f32>, Box<dyn Error>> {
            let out = &outputs[name];
            let raw = out.try_extract_tensor::<f32>()?;
            let shape = raw.0; // &[usize] or similar
                               // Convert shape slices/vecs to exact dimensional array construction
                               // ort 2.0 API might handle shape differently. output shape is typically dynamic.
                               // We assume dimension correctness from model.
                               // ndarray::Array::from_shape_vec is simplest if we know dimensions.
                               // But dimensions change. We use from_iter with shape.
            let shape_ix: (usize, usize, usize) =
                (shape[0] as usize, shape[1] as usize, shape[2] as usize);
            Ok(Array3::from_shape_vec(shape_ix, raw.1.to_vec())?)
        };

        state.pre_conv_history = extract_3d("next_pre_conv_history")?;
        state.latent_buffer = extract_3d("next_latent_buffer")?;
        state.conv_history = extract_3d("next_conv_history")?;

        // Update KV Cache
        // Names: next_key_0, next_value_0...
        for i in 0..8 {
            let k_name = format!("next_key_{}", i);
            let v_name = format!("next_value_{}", i);

            let out_k = &outputs[k_name.as_str()];
            let raw_k = out_k.try_extract_tensor::<f32>()?;
            let shape_k = (
                raw_k.0[0] as usize,
                raw_k.0[1] as usize,
                raw_k.0[2] as usize,
                raw_k.0[3] as usize,
            );
            let arr_k = Array4::from_shape_vec(shape_k, raw_k.1.to_vec())?;

            let out_v = &outputs[v_name.as_str()];
            let raw_v = out_v.try_extract_tensor::<f32>()?;
            let shape_v = (
                raw_v.0[0] as usize,
                raw_v.0[1] as usize,
                raw_v.0[2] as usize,
                raw_v.0[3] as usize,
            );
            let arr_v = Array4::from_shape_vec(shape_v, raw_v.1.to_vec())?;

            state.kv_cache[i] = (arr_k, arr_v);
        }

        Ok(audio)
    }
}

pub struct DecoderState {
    pub pre_conv_history: Array3<f32>,
    pub latent_buffer: Array3<f32>,
    pub conv_history: Array3<f32>,
    pub kv_cache: Vec<(Array4<f32>, Array4<f32>)>,
}

impl Default for DecoderState {
    fn default() -> Self {
        Self::new()
    }
}

impl DecoderState {
    pub fn new() -> Self {
        // Match Python: np.zeros((1, 512, 0)) etc.
        let pre_conv_history = Array3::<f32>::zeros((1, 512, 0));
        let latent_buffer = Array3::<f32>::zeros((1, 1024, 0));
        let conv_history = Array3::<f32>::zeros((1, 1024, 0));

        let mut kv_cache = Vec::with_capacity(8);
        for _ in 0..8 {
            // [1, 16, 0, 64]
            let k = Array4::<f32>::zeros((1, 16, 0, 64));
            let v = Array4::<f32>::zeros((1, 16, 0, 64));
            kv_cache.push((k, v));
        }

        DecoderState {
            pre_conv_history,
            latent_buffer,
            conv_history,
            kv_cache,
        }
    }
}

// 兼容性保留
#[derive(Clone)]
pub struct CodecEmbeddings {
    pub weights: Vec<Vec<f32>>,
    pub bias: Vec<f32>,
}

impl CodecEmbeddings {
    pub fn load_npz(_path: &str) -> Result<Self, Box<dyn Error>> {
        Ok(CodecEmbeddings {
            weights: Vec::new(),
            bias: Vec::new(),
        })
    }
}

pub fn init_onruntime() -> Result<(), Box<dyn Error>> {
    // Explicitly try to load onnxruntime.dll from runtime/ to ensure it's available
    // or set PATH? ort might not use libloading directly for the dll itself in the same way.
    // However, if we load it into the process, it might be found.
    // Alternatively, we can try to set the environment variable path.

    let dll_path = if std::path::Path::new("runtime/onnxruntime.dll").exists() {
        "runtime/onnxruntime.dll"
    } else {
        "onnxruntime.dll"
    };

    // We don't necessarily need to keep the library handle if ort loads it its own way,
    // but loading it here verifies existence and might help with resolution.
    // NOTE: ort 2.0 might have specific configuration for DLL path.
    // For now, let's just print where we think it is.
    println!("  [ONNX] Init: Expecting DLL at {}", dll_path);

    // Attempt to pre-load to see if it fails
    // let _lib = libloading::Library::new(dll_path)?;
    // ^ confusing ownership if we drop it.

    Ok(())
}
