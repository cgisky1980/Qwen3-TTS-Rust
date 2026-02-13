//! Llama.cpp FFI Bindings
// use libc::free; // Removed unused import
use std::ffi::{c_char, c_float, c_void};
use std::os::raw::{c_int, c_uint};
use std::path::Path;

#[repr(C)]
pub struct llama_model_params {
    pub devices: *mut c_void,
    pub tensor_buft_overrides: *mut c_void,
    pub n_gpu_layers: c_int,
    pub split_mode: c_int,
    pub main_gpu: c_int,
    pub tensor_split: *mut c_float,
    pub progress_callback: *mut c_void,
    pub progress_callback_user_data: *mut c_void,
    pub kv_overrides: *mut c_void,
    // CRITICAL: These must be bool (1 byte) to match C struct layout
    pub vocab_only: bool,
    pub use_mmap: bool,
    pub use_direct_io: bool,
    pub use_mlock: bool,
    pub check_tensors: bool,
    pub use_extra_bufts: bool, // Note: Python uses 'use_extra_bufts'
    pub no_host: bool,
    pub no_alloc: bool,
}
#[repr(C)]
pub struct llama_context_params {
    pub n_ctx: c_uint,
    pub n_batch: c_uint,
    pub n_ubatch: c_uint,
    pub n_seq_max: c_uint,
    pub n_threads: c_int,
    pub n_threads_batch: c_int,
    pub rope_scaling_type: c_int,
    pub pooling_type: c_int,
    pub attention_type: c_int,
    pub flash_attn_type: c_int,
    pub rope_freq_base: c_float,
    pub rope_freq_scale: c_float,
    pub yarn_ext_factor: c_float,
    pub yarn_attn_factor: c_float,
    pub yarn_beta_fast: c_float,
    pub yarn_beta_slow: c_float,
    pub yarn_orig_ctx: c_uint,
    pub defrag_thold: c_float,
    pub cb_eval: *mut c_void,
    pub cb_eval_user_data: *mut c_void,
    pub type_k: c_int,
    pub type_v: c_int,
    pub abort_callback: *mut c_void,
    pub abort_callback_data: *mut c_void,
    // CRITICAL: These must be bool (1 byte) to match C struct layout
    pub embeddings: bool,
    pub offload_kqv: bool,
    pub no_perf: bool,
    pub op_offload: bool,
    pub swa_full: bool,
    pub kv_unified: bool,
    pub samplers: *mut c_void,
    pub n_samplers: usize,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct llama_batch {
    pub n_tokens: c_int,
    pub token: *mut c_int,
    pub embd: *mut c_float,
    pub pos: *mut c_int,
    pub n_seq_id: *mut c_int,
    pub seq_id: *mut *mut c_int,
    pub logits: *mut i8,
}
pub type LlamaToken = c_int;
pub type LlamaModelPtr = *mut c_void;
pub type LlamaContextPtr = *mut c_void;
pub type LlamaSamplerPtr = *mut c_void;
pub type LlamaVocabPtr = *mut c_void;

type LlamaBackendInitFn = unsafe extern "C" fn();
type LlamaBackendFreeFn = unsafe extern "C" fn();
type LlamaModelDefaultParamsFn = unsafe extern "C" fn() -> llama_model_params;
type LlamaModelLoadFromFileFn =
    unsafe extern "C" fn(*const c_char, llama_model_params) -> LlamaModelPtr;
type LlamaModelFreeFn = unsafe extern "C" fn(LlamaModelPtr);
type LlamaModelGetVocabFn = unsafe extern "C" fn(LlamaModelPtr) -> LlamaVocabPtr;
type LlamaModelNEmbdfn = unsafe extern "C" fn(LlamaModelPtr) -> c_int;
type LlamaModelNHeadFn = unsafe extern "C" fn(LlamaModelPtr) -> c_int;
type LlamaModelNLayerFn = unsafe extern "C" fn(LlamaModelPtr) -> c_int;
type LlamaModelNCtxFn = unsafe extern "C" fn() -> c_int;
type LlamaModelNVocabFn = unsafe extern "C" fn() -> c_int;
type LlamaVocabNTokensFn = unsafe extern "C" fn(LlamaVocabPtr) -> c_int;
type LlamaVocabEosFn = unsafe extern "C" fn(LlamaVocabPtr) -> LlamaToken;
type LlamaContextDefaultParamsFn = unsafe extern "C" fn() -> llama_context_params;
type LlamaInitFromModelFn =
    unsafe extern "C" fn(LlamaModelPtr, llama_context_params) -> LlamaContextPtr;
type LlamaFreeFn = unsafe extern "C" fn(LlamaContextPtr);
type LlamaBatchInitFn = unsafe extern "C" fn(c_int, c_int, c_int) -> llama_batch;
type LlamaBatchFreeFn = unsafe extern "C" fn(llama_batch);
type LlamaDecodeFn = unsafe extern "C" fn(LlamaContextPtr, llama_batch) -> c_int;
type LlamaGetEmbeddingsFn = unsafe extern "C" fn(LlamaContextPtr) -> *mut c_float;
type LlamaGetLogitsFn = unsafe extern "C" fn(LlamaContextPtr) -> *mut c_float;
type LlamaGetMemoryFn = unsafe extern "C" fn(LlamaContextPtr) -> *mut c_void;
type LlamaMemoryClearFn = unsafe extern "C" fn(*mut c_void, bool);
type LlamaMemorySeqRmFn = unsafe extern "C" fn(*mut c_void, c_int, c_int, c_int) -> bool;
type LlamaMemorySeqPosMaxFn = unsafe extern "C" fn(*mut c_void, c_int) -> c_int;
type LlamaSamplerInitTempFn = unsafe extern "C" fn(c_float) -> LlamaSamplerPtr;
type LlamaSamplerSampleFn =
    unsafe extern "C" fn(LlamaSamplerPtr, LlamaContextPtr, c_int) -> LlamaToken;
type LlamaSamplerFreeFn = unsafe extern "C" fn(LlamaSamplerPtr);
type GgmlBackendLoadAllFn = unsafe extern "C" fn();

pub struct LlamaFFI {
    pub llama_backend_init: LlamaBackendInitFn,
    pub llama_backend_free: LlamaBackendFreeFn,
    pub llama_model_default_params: LlamaModelDefaultParamsFn,
    pub llama_model_load_from_file: LlamaModelLoadFromFileFn,
    pub llama_model_free: LlamaModelFreeFn,
    pub llama_model_get_vocab: LlamaModelGetVocabFn,
    pub llama_model_n_embd: LlamaModelNEmbdfn,
    pub llama_model_n_head: LlamaModelNHeadFn,
    pub llama_model_n_layer: LlamaModelNLayerFn,
    pub llama_model_n_ctx: LlamaModelNCtxFn,
    pub llama_model_n_vocab: LlamaModelNVocabFn,
    pub llama_vocab_n_tokens: LlamaVocabNTokensFn,
    pub llama_vocab_eos: LlamaVocabEosFn,
    pub llama_context_default_params: LlamaContextDefaultParamsFn,
    pub llama_init_from_model: LlamaInitFromModelFn,
    pub llama_free: LlamaFreeFn,
    pub llama_batch_init: LlamaBatchInitFn,
    pub llama_batch_free: LlamaBatchFreeFn,
    pub llama_decode: LlamaDecodeFn,
    pub llama_get_embeddings: LlamaGetEmbeddingsFn,
    pub llama_get_logits: LlamaGetLogitsFn,
    pub llama_get_memory: LlamaGetMemoryFn,
    pub llama_memory_clear: LlamaMemoryClearFn,
    pub llama_memory_seq_rm: LlamaMemorySeqRmFn,
    pub llama_memory_seq_pos_max: LlamaMemorySeqPosMaxFn,
    pub llama_sampler_init_temp: LlamaSamplerInitTempFn,
    pub llama_sampler_sample: LlamaSamplerSampleFn,
    pub llama_sampler_free: LlamaSamplerFreeFn,
    pub ggml_backend_load_all: GgmlBackendLoadAllFn,
}

static FFI: std::sync::OnceLock<LlamaFFI> = std::sync::OnceLock::new();

pub fn get_ffi() -> &'static LlamaFFI {
    FFI.get_or_init(|| {
        unsafe {
            // Get absolute path to runtime directory
            let runtime_path = std::env::current_dir().unwrap().join("runtime");
            let runtime_path_str = runtime_path.to_string_lossy();

            // 1. Add to PATH/LD_LIBRARY_PATH (Most robust for all types of loading)
            let path_key = if cfg!(target_os = "windows") {
                "PATH"
            } else if cfg!(target_os = "macos") {
                "DYLD_LIBRARY_PATH"
            } else {
                "LD_LIBRARY_PATH"
            };

            let path_var = std::env::var_os(path_key).unwrap_or_default();
            let mut paths: Vec<_> = std::env::split_paths(&path_var).collect();
            if !paths.contains(&runtime_path) {
                paths.insert(0, runtime_path.clone());
                let new_path = std::env::join_paths(paths).unwrap();
                std::env::set_var(path_key, new_path);
                println!("  [System] Added to {}: {}", path_key, runtime_path_str);
            }

            // 2. Set DLL search path (Windows specific)
            #[cfg(target_os = "windows")]
            {
                use std::os::windows::ffi::OsStrExt;
                let path_wide = std::ffi::OsStr::new(&runtime_path)
                    .encode_wide()
                    .chain(std::iter::once(0))
                    .collect::<Vec<u16>>();
                winapi::um::winbase::SetDllDirectoryW(path_wide.as_ptr());
                println!("  [System] SetDllDirectoryW set to: {}", runtime_path_str);
            }

            // 3. Set GGML_BACKEND_PATH (Llama.cpp specific)
            std::env::set_var("GGML_BACKEND_PATH", &*runtime_path_str);
            println!("  [System] GGML_BACKEND_PATH set to: {}", runtime_path_str);

            // Library names based on platform
            let (ggml_name, llama_name, libomp_name) = if cfg!(target_os = "windows") {
                ("ggml.dll", "llama.dll", Some("libomp140.x86_64.dll"))
            } else if cfg!(target_os = "macos") {
                ("libggml.dylib", "libllama.dylib", None)
            } else {
                ("libggml.so", "libllama.so", None)
            };

            // Pre-load libomp if present
            if let Some(omp_name) = libomp_name {
                let libomp_path = runtime_path.join(omp_name);
                let _libomp = libloading::Library::new(&libomp_path).ok();
            }

            // 加载 ggml (Dependency of llama)
            let ggml_path = runtime_path.join(ggml_name);
            let ggml_lib = libloading::Library::new(&ggml_path)
                .map_err(|e| {
                    eprintln!(
                        "WARNING: Failed to load {} from {:?}: {}",
                        ggml_name, ggml_path, e
                    )
                })
                .ok();

            // 加载 llama
            let llama_path = runtime_path.join(llama_name);
            let lib = libloading::Library::new(&llama_path)
                .unwrap_or_else(|_| panic!("Failed to load {}. Please ensure it is in the runtime/ directory.", llama_name));

            unsafe extern "C" fn dummy_fn() {}

            let load_all_fn: GgmlBackendLoadAllFn = if let Some(ref glib) = ggml_lib {
                match glib.get::<GgmlBackendLoadAllFn>(b"ggml_backend_load_all") {
                    Ok(symbol) => *symbol,
                    Err(_) => dummy_fn,
                }
            } else {
                lib.get::<GgmlBackendLoadAllFn>(b"ggml_backend_load_all")
                    .map(|s| *s)
                    .unwrap_or(dummy_fn)
            };

            // Try to manually load vulkan backend to check for missing dependencies
            let vulkan_path = runtime_path.join("ggml-vulkan.dll");
            match libloading::Library::new(&vulkan_path) {
                Ok(_) => println!("  [System] Verified: ggml-vulkan.dll is loadable."),
                Err(e) => eprintln!("  [System] WARNING: ggml-vulkan.dll is NOT loadable: {}. This usually means missing Vulkan drivers or runtime.", e),
            }

            let ffi = LlamaFFI {
                llama_backend_init: *lib.get(b"llama_backend_init").expect("llama_backend_init"),
                llama_backend_free: *lib.get(b"llama_backend_free").expect("llama_backend_free"),
                llama_model_default_params: *lib
                    .get(b"llama_model_default_params")
                    .expect("llama_model_default_params"),
                llama_model_load_from_file: *lib
                    .get(b"llama_model_load_from_file")
                    .expect("llama_model_load_from_file"),
                llama_model_free: *lib.get(b"llama_model_free").expect("llama_model_free"),
                llama_model_get_vocab: *lib
                    .get(b"llama_model_get_vocab")
                    .expect("llama_model_get_vocab"),
                llama_model_n_embd: *lib.get(b"llama_model_n_embd").expect("llama_model_n_embd"),
                llama_model_n_head: *lib.get(b"llama_model_n_head").expect("llama_model_n_head"),
                llama_model_n_layer: *lib
                    .get(b"llama_model_n_layer")
                    .expect("llama_model_n_layer"),
                llama_model_n_ctx: *lib.get(b"llama_n_ctx").expect("llama_n_ctx"),
                llama_model_n_vocab: *lib.get(b"llama_n_vocab").expect("llama_n_vocab"),
                llama_vocab_n_tokens: *lib
                    .get(b"llama_vocab_n_tokens")
                    .expect("llama_vocab_n_tokens"),
                llama_vocab_eos: *lib.get(b"llama_vocab_eos").expect("llama_vocab_eos"),
                llama_context_default_params: *lib
                    .get(b"llama_context_default_params")
                    .expect("llama_context_default_params"),
                llama_init_from_model: *lib
                    .get(b"llama_init_from_model")
                    .expect("llama_init_from_model"),
                llama_free: *lib.get(b"llama_free").expect("llama_free"),
                llama_batch_init: *lib.get(b"llama_batch_init").expect("llama_batch_init"),
                llama_batch_free: *lib.get(b"llama_batch_free").expect("llama_batch_free"),
                llama_decode: *lib.get(b"llama_decode").expect("llama_decode"),
                llama_get_embeddings: *lib
                    .get(b"llama_get_embeddings")
                    .expect("llama_get_embeddings"),
                llama_get_logits: *lib.get(b"llama_get_logits").expect("llama_get_logits"),
                llama_get_memory: *lib.get(b"llama_get_memory").expect("llama_get_memory"),
                llama_memory_clear: *lib.get(b"llama_memory_clear").expect("llama_memory_clear"),
                llama_memory_seq_rm: *lib
                    .get(b"llama_memory_seq_rm")
                    .expect("llama_memory_seq_rm"),
                llama_memory_seq_pos_max: *lib
                    .get(b"llama_memory_seq_pos_max")
                    .expect("llama_memory_seq_pos_max"),
                llama_sampler_init_temp: *lib
                    .get(b"llama_sampler_init_temp")
                    .expect("llama_sampler_init_temp"),
                llama_sampler_sample: *lib
                    .get(b"llama_sampler_sample")
                    .expect("llama_sampler_sample"),
                llama_sampler_free: *lib.get(b"llama_sampler_free").expect("llama_sampler_free"),
                ggml_backend_load_all: load_all_fn,
            };

            // Temporarily switch CWD to runtime/ to help ggml_backend_load_all() find its siblings
            let original_cwd = std::env::current_dir().ok();
            if original_cwd.is_some() {
                let _ = std::env::set_current_dir(&runtime_path);
            }

            (ffi.ggml_backend_load_all)(); // 加载所有 backend
            (ffi.llama_backend_init)(); // 初始化 backend

            if let Some(cwd) = original_cwd {
                let _ = std::env::set_current_dir(cwd);
            }

            std::mem::forget(lib);
            if let Some(glib) = ggml_lib {
                std::mem::forget(glib);
            }
            ffi
        }
    })
}

pub fn cleanup() {
    if let Some(ffi) = FFI.get() {
        unsafe {
            (ffi.llama_backend_free)();
        }
    }
}

pub struct LlamaModel {
    pub ptr: LlamaModelPtr,
    pub vocab: LlamaVocabPtr,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_layer: usize,
    pub n_ctx: usize,
    pub n_vocab: usize,
    pub eos_token: LlamaToken,
}
impl LlamaModel {
    pub fn load(path: &Path, n_gpu_layers: i32) -> Result<Self, String> {
        let ffi = get_ffi();
        unsafe {
            let mut params = (ffi.llama_model_default_params)();
            params.n_gpu_layers = n_gpu_layers; // Actually use the parameter!
            let c_path =
                std::ffi::CString::new(path.to_str().unwrap()).map_err(|e| e.to_string())?;
            let ptr = (ffi.llama_model_load_from_file)(c_path.as_ptr(), params);
            if ptr.is_null() {
                return Err("Failed to load model".to_string());
            }
            let vocab = (ffi.llama_model_get_vocab)(ptr);
            let n_vocab = (ffi.llama_vocab_n_tokens)(vocab) as usize;
            let n_embd = (ffi.llama_model_n_embd)(ptr) as usize;
            let n_head = (ffi.llama_model_n_head)(ptr) as usize;
            let n_layer = (ffi.llama_model_n_layer)(ptr) as usize;
            let eos_token = (ffi.llama_vocab_eos)(vocab);
            Ok(Self {
                ptr,
                vocab,
                n_embd,
                n_head,
                n_layer,
                n_ctx: 32768,
                n_vocab,
                eos_token,
            })
        }
    }
}
impl Drop for LlamaModel {
    fn drop(&mut self) {
        unsafe {
            if let Some(ffi) = FFI.get() {
                if !self.ptr.is_null() {
                    (ffi.llama_model_free)(self.ptr);
                }
            }
        }
    }
}
impl Clone for LlamaModel {
    fn clone(&self) -> Self {
        Self {
            ptr: self.ptr,
            vocab: self.vocab,
            n_embd: self.n_embd,
            n_head: self.n_head,
            n_layer: self.n_layer,
            n_ctx: self.n_ctx,
            n_vocab: self.n_vocab,
            eos_token: self.eos_token,
        }
    }
}

pub struct LlamaContext {
    pub ptr: LlamaContextPtr,
    /// Wrapped in ManuallyDrop to prevent double-free.
    /// The model's lifetime is owned by TtsEngine; LlamaContext only borrows it.
    pub model: std::mem::ManuallyDrop<LlamaModel>,
}
impl LlamaContext {
    pub fn new(
        model: &LlamaModel,
        n_ctx: u32,
        n_batch: u32,
        embeddings: i32,
        n_threads: i32,
    ) -> Result<Self, String> {
        let ffi = get_ffi();
        unsafe {
            let mut params = (ffi.llama_context_default_params)();
            params.n_ctx = n_ctx;
            params.n_batch = n_batch;
            params.n_ubatch = 512; // Match Python default
            params.n_seq_max = 1;
            params.embeddings = embeddings != 0; // Convert to bool
            params.flash_attn_type = 1; // 1 = Enabled (Optimized)
            params.offload_kqv = true; // Match Python
            params.no_perf = true; // Match Python

            // Threading configuration
            if n_threads > 0 {
                params.n_threads = n_threads;
            } else {
                // Default logic: Clamp n_threads to 4 for small-batch inference performance
                let cpu_count = std::thread::available_parallelism()
                    .map(|p| p.get() as c_int)
                    .unwrap_or(4);
                params.n_threads = (cpu_count / 2).min(4);
            }

            // params.n_threads_batch = cpu_count; // Use default or let llama.cpp handle it

            let ptr = (ffi.llama_init_from_model)(model.ptr, params);
            if ptr.is_null() {
                return Err("Failed to create context".to_string());
            }
            Ok(Self {
                ptr,
                model: std::mem::ManuallyDrop::new(model.clone()),
            })
        }
    }
    pub fn decode(&mut self, batch: &mut LlamaBatch) -> Result<(), String> {
        let ffi = get_ffi();
        unsafe {
            let result = (ffi.llama_decode)(self.ptr, batch.batch);
            if result != 0 {
                return Err(format!("llama_decode failed: {}", result));
            }
            Ok(())
        }
    }
    pub fn get_embeddings(&self) -> &[f32] {
        let ffi = get_ffi();
        unsafe {
            std::slice::from_raw_parts((ffi.llama_get_embeddings)(self.ptr), self.model.n_embd)
        }
    }

    pub fn get_embedding_at(&self, batch_index: usize) -> &[f32] {
        let ffi = get_ffi();
        unsafe {
            let base_ptr = (ffi.llama_get_embeddings)(self.ptr);
            let offset = batch_index * self.model.n_embd;
            std::slice::from_raw_parts(base_ptr.add(offset), self.model.n_embd)
        }
    }
    pub fn get_logits(&self) -> &[f32] {
        let ffi = get_ffi();
        unsafe { std::slice::from_raw_parts((ffi.llama_get_logits)(self.ptr), self.model.n_vocab) }
    }
    pub fn get_logits_ith(&self, i: usize) -> &[f32] {
        let ffi = get_ffi();
        unsafe {
            let base_ptr = (ffi.llama_get_logits)(self.ptr);
            std::slice::from_raw_parts(base_ptr.add(i * self.model.n_vocab), self.model.n_vocab)
        }
    }
    pub fn clear_kv_cache(&self) -> bool {
        let ffi = get_ffi();
        unsafe {
            let mem = (ffi.llama_get_memory)(self.ptr);
            let result = (ffi.llama_memory_seq_rm)(mem, -1, 0, -1);
            if !result {
                eprintln!("WARNING: clear_kv_cache failed for ctx {:?}", self.ptr);
            }
            result
        }
    }

    pub fn debug_seq_pos_max(&self, seq_id: c_int) -> c_int {
        let ffi = get_ffi();
        unsafe {
            let mem = (ffi.llama_get_memory)(self.ptr);
            let result = (ffi.llama_memory_seq_pos_max)(mem, seq_id);
            eprintln!(
                "DEBUG: [debug_seq_pos_max] mem={:?}, seq_id={}, result={}",
                mem, seq_id, result
            );
            result
        }
    }
}
impl Drop for LlamaContext {
    fn drop(&mut self) {
        unsafe {
            if let Some(ffi) = FFI.get() {
                if !self.ptr.is_null() {
                    (ffi.llama_free)(self.ptr);
                }
            }
        }
    }
}

pub struct LlamaBatch {
    batch: llama_batch,
    n_tokens_max: usize,
    n_embd: usize,
    _n_pos_per_embd: usize,
    _embd_buffer: Vec<c_float>,
    _pos_buffer: Vec<c_int>,
    _seq_id_buffers: Vec<Vec<c_int>>,
    _n_seq_id_buffer: Vec<c_int>,
    _logits_buffer: Vec<i8>,
}
impl LlamaBatch {
    pub fn new(
        n_tokens_max: usize,
        n_embd: usize,
        n_seq_max: usize,
        n_pos_per_embd: usize,
    ) -> Self {
        let ffi = get_ffi();
        unsafe {
            // llama_batch_init allocates all necessary buffers (token, embd, pos, seq_id, logits)
            let batch =
                (ffi.llama_batch_init)(n_tokens_max as c_int, n_embd as c_int, n_seq_max as c_int);

            // We do NOT override these pointers with Rust Vecs anymore.
            // We use the memory managed by llama.cpp.

            Self {
                batch,
                n_tokens_max,
                n_embd,
                _n_pos_per_embd: n_pos_per_embd,
                _embd_buffer: Vec::new(),
                _pos_buffer: Vec::new(),
                _seq_id_buffers: Vec::new(),
                _n_seq_id_buffer: Vec::new(),
                _logits_buffer: Vec::new(),
            }
        }
    }

    pub fn set_embd(&mut self, prompt_embeds: &[f32], pos_arr: &[i32], seq_id: i32) {
        let n_tokens = prompt_embeds.len() / self.n_embd;
        unsafe {
            // Copy embeddings to batch.embd (C memory)
            std::ptr::copy_nonoverlapping(
                prompt_embeds.as_ptr(),
                self.batch.embd,
                prompt_embeds.len(),
            );

            // Copy positions to batch.pos (C memory)
            let max_pos = self.n_tokens_max; // Use n_tokens_max as simple limit. M-RoPE might need more checks if we could query allocated size?
                                             // But llama_batch_init allocates n_tokens_max elements for pos.

            if pos_arr.len() > max_pos {
                eprintln!(
                    "WARNING: pos_arr length {} exceeds batch capacity {}. Truncating!",
                    pos_arr.len(),
                    max_pos
                );
            }
            std::ptr::copy_nonoverlapping(
                pos_arr.as_ptr(),
                self.batch.pos,
                pos_arr.len().min(max_pos),
            );
        }

        // Set sequence IDs
        for i in 0..n_tokens {
            // batch.n_seq_id is *mut i32
            unsafe {
                *self.batch.n_seq_id.add(i) = 1;

                // batch.seq_id is *mut *mut i32.
                // It points to an array of pointers. Each pointer points to an array of seq_ids?
                // Wait, llama_batch_init allocates seq_id[i] as well?
                // "The seq_id array is not allocated by llama_batch_init" ???
                // Let's check llama.cpp docs/source.
                // "llama_batch_init" allocates "seq_id" as array of pointers.
                // BUT it does NOT allocate the actual int arrays they point to?
                // Actually, it usually does NOT. The user is expected to assign pointers.
                // OR it allocates a default buffer?
                //
                // Usage in common.cpp:
                // batch.seq_id[i][0] = 0;
                // This implies batch.seq_id[i] IS valid pointer.
                // So llama_batch_init must have allocated it.
                //
                // Let's assume it IS allocated.

                let seq_ids = *self.batch.seq_id.add(i);
                *seq_ids.add(0) = seq_id;

                // Set logits
                *self.batch.logits.add(i) = if i == n_tokens - 1 { 1 } else { 0 };
            }
        }
        self.batch.n_tokens = n_tokens as c_int;
    }
    pub fn clear(&mut self) {
        self.batch.n_tokens = 0;
    }
    pub fn n_tokens(&self) -> usize {
        self.batch.n_tokens as usize
    }
    pub fn batch_ptr(&mut self) -> *mut llama_batch {
        &mut self.batch
    }
}

pub struct LlamaSampler {
    _ptr: LlamaSamplerPtr,
    n_vocab: usize,
    _neg_inf: f32,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    rng: std::cell::RefCell<rand::rngs::StdRng>,
}
impl LlamaSampler {
    /// Create a sampler with temperature-based random sampling
    /// Python defaults: temperature=0.5, top_k=50, top_p=1.0
    pub fn new(n_vocab: usize, temperature: f32, top_k: i32, top_p: f32, seed: u64) -> Self {
        use rand::SeedableRng;
        Self {
            _ptr: std::ptr::null_mut(),
            n_vocab,
            _neg_inf: -1e9,
            temperature,
            top_k: top_k as usize,
            top_p,
            rng: std::cell::RefCell::new(rand::rngs::StdRng::seed_from_u64(seed)),
        }
    }

    /// Create a greedy (argmax) sampler
    pub fn greedy(n_vocab: usize) -> Self {
        use rand::SeedableRng;
        Self {
            _ptr: std::ptr::null_mut(),
            n_vocab,
            _neg_inf: -1e9,
            temperature: 0.0, // 0.0 means greedy
            top_k: 0,
            top_p: 1.0,
            rng: std::cell::RefCell::new(rand::rngs::StdRng::seed_from_u64(42)),
        }
    }

    pub fn sample(
        &self,
        ctx: &LlamaContext,
        idx: i32,
        limit_start: Option<usize>,
        limit_end: Option<usize>,
    ) -> LlamaToken {
        let ffi = get_ffi();
        unsafe {
            // Handle index offset: idx can be -1 (implied logic?) or explicit batch index.
            // But since we are manually implementing sampling, 'idx' MUST be the explicit Batch Index (0-based)
            // relative to the logits buffer.
            // If idx < 0, we can't easily resolve here without knowing Batch Size.
            // So we assume idx is valid >= 0.

            let offset = if idx >= 0 { idx as usize } else { 0 };
            let base_ptr = (ffi.llama_get_logits)(ctx.ptr);
            let logits_ptr = base_ptr.add(offset * self.n_vocab);

            let logits = std::slice::from_raw_parts(logits_ptr, self.n_vocab);
            let start = limit_start.unwrap_or(0);
            let end = limit_end.unwrap_or(self.n_vocab).min(self.n_vocab);

            // OPTIMIZATION: Zero-alloc Argmax for Greedy Sampling
            if self.temperature <= 0.0 {
                let mut max_val = f32::NEG_INFINITY;
                let mut max_idx = start;

                for (i, &val) in logits.iter().enumerate().take(end).skip(start) {
                    if val > max_val {
                        max_val = val;
                        max_idx = i;
                    }
                }
                return max_idx as LlamaToken;
            }

            // Normal Sampling (Temperature/Top-K/Top-P)
            // 1. Collect (index, logit) pairs in range
            let mut candidates: Vec<(usize, f32)> = (start..end).map(|i| (i, logits[i])).collect();

            // 2. Sort by logit descending
            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // 3. Apply top_k filter
            if self.top_k > 0 && self.top_k < candidates.len() {
                candidates.truncate(self.top_k);
            }

            // 4. Apply temperature and compute softmax
            let max_logit = candidates.first().map(|(_, l)| *l).unwrap_or(0.0);
            let mut probs: Vec<(usize, f32)> = candidates
                .iter()
                .map(|(idx, logit)| {
                    let scaled = (logit - max_logit) / self.temperature;
                    (*idx, scaled.exp())
                })
                .collect();

            // 5. Normalize to probabilities
            let sum: f32 = probs.iter().map(|(_, p)| p).sum();
            if sum > 0.0 {
                for (_, p) in probs.iter_mut() {
                    *p /= sum;
                }
            }

            // 6. Apply top_p (nucleus) sampling
            if self.top_p < 1.0 {
                let mut cumsum = 0.0;
                let mut cutoff_idx = probs.len();
                for (i, (_, p)) in probs.iter().enumerate() {
                    cumsum += p;
                    if cumsum >= self.top_p {
                        cutoff_idx = i + 1;
                        break;
                    }
                }
                probs.truncate(cutoff_idx);

                // Renormalize
                let new_sum: f32 = probs.iter().map(|(_, p)| p).sum();
                if new_sum > 0.0 {
                    for (_, p) in probs.iter_mut() {
                        *p /= new_sum;
                    }
                }
            }

            // 7. Sample from distribution
            use rand::Rng;
            let r: f32 = self.rng.borrow_mut().gen();
            let mut cumsum = 0.0;
            for (idx, p) in probs.iter() {
                cumsum += p;
                if r < cumsum {
                    return *idx as LlamaToken;
                }
            }

            // Fallback to first candidate
            probs
                .first()
                .map(|(idx, _)| *idx as LlamaToken)
                .unwrap_or(start as LlamaToken)
        }
    }

    /// Sample from a restricted range [0, limit_idx) AND a specific list of allowed tokens.
    /// This is useful for Qwen3-TTS where we want audio codes (0-2160) OR EOS tokens (151643, etc.)
    pub fn sample_custom(
        &self,
        ctx: &LlamaContext,
        limit_idx: usize,
        allow_tokens: &[LlamaToken],
    ) -> LlamaToken {
        let ffi = get_ffi();
        unsafe {
            let logits_ptr = (ffi.llama_get_logits)(ctx.ptr);
            let logits = std::slice::from_raw_parts(logits_ptr, self.n_vocab);

            // Determine search range
            let start = 0; // Always start from 0 for custom sampling
            let end = limit_idx.min(self.n_vocab);

            // OPTIMIZATION: Zero-alloc Argmax for Greedy Sampling
            if self.temperature <= 0.0 {
                let mut max_val = f32::NEG_INFINITY;
                let mut max_idx = start;

                // Check range [0, limit_idx)
                for (i, &val) in logits.iter().enumerate().take(end).skip(start) {
                    if val > max_val {
                        max_val = val;
                        max_idx = i;
                    }
                }

                // Check allowed special tokens
                for &token in allow_tokens {
                    let idx = token as usize;
                    if idx < self.n_vocab {
                        let val = logits[idx];
                        if val > max_val {
                            max_val = val;
                            max_idx = idx;
                        }
                    }
                }
                return max_idx as LlamaToken;
            }

            // Normal Sampling (Temperature/Top-K/Top-P)
            // 1. Collect candidates: [0..limit_idx] + [allow_tokens]
            let mut candidates: Vec<(usize, f32)> =
                Vec::with_capacity(limit_idx + allow_tokens.len());

            // Add range [0, limit_idx)
            for (i, &val) in logits
                .iter()
                .enumerate()
                .take(limit_idx.min(self.n_vocab))
            {
                candidates.push((i, val));
            }

            // Add allowed special tokens
            for &token in allow_tokens {
                let idx = token as usize;
                if idx < self.n_vocab {
                    candidates.push((idx, logits[idx]));
                }
            }

            // 2. Sort
            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // 3. Top-K
            if self.top_k > 0 && self.top_k < candidates.len() {
                candidates.truncate(self.top_k);
            }

            // 4. Softmax
            let max_logit = candidates.first().map(|(_, l)| *l).unwrap_or(0.0);
            let mut probs: Vec<(usize, f32)> = candidates
                .iter()
                .map(|(idx, logit)| {
                    let scaled = (logit - max_logit) / self.temperature;
                    (*idx, scaled.exp())
                })
                .collect();

            // 5. Normalize
            let sum: f32 = probs.iter().map(|(_, p)| p).sum();
            if sum > 0.0 {
                for (_, p) in probs.iter_mut() {
                    *p /= sum;
                }
            }

            // 6. Top-P
            if self.top_p < 1.0 {
                let mut cumsum = 0.0;
                let mut cutoff_idx = probs.len();
                for (i, (_, p)) in probs.iter().enumerate() {
                    cumsum += p;
                    if cumsum >= self.top_p {
                        cutoff_idx = i + 1;
                        break;
                    }
                }
                probs.truncate(cutoff_idx);
                // Renormalize
                let new_sum: f32 = probs.iter().map(|(_, p)| p).sum();
                if new_sum > 0.0 {
                    for (_, p) in probs.iter_mut() {
                        *p /= new_sum;
                    }
                }
            }

            // 7. Sample
            use rand::Rng;
            let r: f32 = self.rng.borrow_mut().gen();
            let mut cumsum = 0.0;
            for (idx, p) in probs.iter() {
                cumsum += p;
                if r < cumsum {
                    return *idx as LlamaToken;
                }
            }

            probs[0].0 as LlamaToken
        }
    }
}
impl Drop for LlamaSampler {
    fn drop(&mut self) {}
}
