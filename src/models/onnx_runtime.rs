use libloading::{Library, Symbol};
use std::error::Error;
use std::ffi::{CStr, CString};
use std::fmt;
use std::os::raw::{c_char, c_int, c_void};
use std::path::Path;
use std::ptr::null_mut;

type OrtApiBasePtr = *mut c_void;
type OrtGetApiBaseFn = unsafe extern "C" fn() -> OrtApiBasePtr;
type OrtApiPtr = *mut c_void;
type OrtGetApiFn = unsafe extern "C" fn(u32) -> OrtApiPtr;
type OrtStatusPtr = *mut c_void;
type OrtEnvPtr = *mut c_void;
type OrtSessionPtr = *mut c_void;
type OrtSessionOptionsPtr = *mut c_void;
type OrtMemoryInfoPtr = *mut c_void;
type OrtTensorPtr = *mut c_void;
type OrtValuePtr = *mut c_void;
type OrtAllocatorPtr = *mut c_void;
type OrtRunOptionsPtr = *mut c_void;

pub struct OrtApi;
pub struct OrtEnv;
pub struct OrtSession;
pub struct OrtSessionOptions;
pub struct OrtMemoryInfo;
pub struct OrtAllocator;
pub struct OrtValue;
pub struct OrtRunOptions;

#[derive(Debug)]
pub struct OnnxRuntimeError(String);

impl fmt::Display for OnnxRuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ONNX Runtime Error: {}", self.0)
    }
}

impl Error for OnnxRuntimeError {}

pub struct OnnxRuntime {
    lib: Library,
    get_api_base: Symbol<OrtGetApiBaseFn>,
    get_api: Symbol<OrtGetApiFn>,
    create_env: Symbol<unsafe extern "C" fn(u32, *const c_char, *mut *mut c_void) -> OrtStatusPtr>,
    release_env: Symbol<unsafe extern "C" fn(*mut c_void)>,
    create_session_options: Symbol<unsafe extern "C" fn(*mut *mut c_void) -> OrtStatusPtr>,
    release_session_options: Symbol<unsafe extern "C" fn(*mut c_void)>,
    set_gpu_mem_limit: Symbol<unsafe extern "C" fn(*mut c_void, c_int, c_int) -> OrtStatusPtr>,
    create_session: Symbol<
        unsafe extern "C" fn(
            *const c_char,
            *mut c_void,
            *mut c_void,
            *mut *mut c_void,
        ) -> OrtStatusPtr,
    >,
    release_session: Symbol<unsafe extern "C" fn(*mut c_void)>,
    get_input_count: Symbol<unsafe extern "C" fn(*mut c_void, *mut c_int) -> OrtStatusPtr>,
    get_output_count: Symbol<unsafe extern "C" fn(*mut c_void, *mut c_int) -> OrtStatusPtr>,
    get_input_name: Symbol<
        unsafe extern "C" fn(*mut c_void, c_int, *mut c_void, *mut *mut c_void) -> OrtStatusPtr,
    >,
    get_output_name: Symbol<
        unsafe extern "C" fn(*mut c_void, c_int, *mut c_void, *mut *mut c_void) -> OrtStatusPtr,
    >,
    release_allocator: Symbol<unsafe extern "C" fn(*mut c_void)>,
    create_memory_info: Symbol<
        unsafe extern "C" fn(
            *mut *mut c_void,
            *mut *mut c_void,
            usize,
            c_int,
            c_int,
        ) -> OrtStatusPtr,
    >,
    release_memory_info: Symbol<unsafe extern "C" fn(*mut c_void)>,
    create_tensor_with_data: Symbol<
        unsafe extern "C" fn(
            *mut *mut c_void,
            *const c_void,
            usize,
            *mut c_void,
            c_int,
            *mut *mut c_void,
        ) -> OrtStatusPtr,
    >,
    get_tensor_type: Symbol<unsafe extern "C" fn(*mut c_void, *mut c_int) -> OrtStatusPtr>,
    release_value: Symbol<unsafe extern "C" fn(*mut c_void)>,
    run: Symbol<
        unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            *const *const c_char,
            *const *mut c_void,
            c_int,
            *mut *mut c_void,
        ) -> OrtStatusPtr,
    >,
    get_tensor_shape:
        Symbol<unsafe extern "C" fn(*mut c_void, *mut c_int, *mut c_int) -> OrtStatusPtr>,
    get_string_tensor_data:
        Symbol<unsafe extern "C" fn(*mut c_void, *mut c_void, usize) -> OrtStatusPtr>,
    get_string_tensor_data_length:
        Symbol<unsafe extern "C" fn(*mut c_void, c_int, *mut usize) -> OrtStatusPtr>,
    get_allocator_with_default_options:
        Symbol<unsafe extern "C" fn(*mut *mut c_void) -> OrtStatusPtr>,
}

impl OnnxRuntime {
    fn load_library() -> Result<Library, Box<dyn Error>> {
        let runtime_path = std::env::current_dir()
            .unwrap_or_default()
            .join("runtime");

        #[cfg(target_os = "windows")]
        {
            use std::os::windows::ffi::OsStrExt;
            let path_wide = std::ffi::OsStr::new(&runtime_path)
                .encode_wide()
                .chain(std::iter::once(0))
                .collect::<Vec<u16>>();
            unsafe {
                winapi::um::winbase::SetDllDirectoryW(path_wide.as_ptr());
            }

            let path_key = "PATH";
            let path_var = std::env::var_os(path_key).unwrap_or_default();
            let mut paths: Vec<_> = std::env::split_paths(&path_var).collect();
            if !paths.contains(&runtime_path) {
                paths.insert(0, runtime_path.clone());
                let new_path = std::env::join_paths(paths).unwrap();
                std::env::set_var(path_key, new_path);
            }
        }

        #[cfg(unix)]
        {
            let path_key = if cfg!(target_os = "macos") {
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
            }
        }

        #[cfg(windows)]
        let lib_paths = vec![
            "onnxruntime.dll",
            "runtime/onnxruntime.dll",
            "./runtime/onnxruntime.dll",
        ];
        #[cfg(unix)]
        let lib_paths = vec![
            "libonnxruntime.so",
            "runtime/libonnxruntime.so",
            "./runtime/libonnxruntime.so",
        ];

        let mut last_error = None;
        for path in lib_paths {
            match Library::new(path) {
                Ok(lib) => {
                    println!("Loaded ONNX Runtime from: {}", path);
                    return Ok(lib);
                }
                Err(e) => {
                    last_error = Some(e);
                }
            }
        }

        Err(OnnxRuntimeError(format!(
            "Failed to load ONNX Runtime from any path. Last error: {}",
            last_error.unwrap()
        ))
        .into())
    }

    pub fn new() -> Result<Self, Box<dyn Error>> {
        let lib = Self::load_library()?;

        unsafe {
            let get_api_base: Symbol<OrtGetApiBaseFn> = lib.get(b"OrtGetApiBase")?;
            let get_api: Symbol<OrtGetApiFn> = lib.get(b"OrtGetApi")?;

            let api_base = get_api_base();
            if api_base.is_null() {
                return Err(OnnxRuntimeError("Failed to get ONNX API base".into()).into());
            }

            let api = get_api(16);
            if api.is_null() {
                return Err(OnnxRuntimeError("Failed to get ONNX API v16".into()).into());
            }

            let create_env: Symbol<
                unsafe extern "C" fn(u32, *const c_char, *mut *mut c_void) -> OrtStatusPtr,
            > = lib.get(b"OrtCreateEnv")?;
            let release_env: Symbol<unsafe extern "C" fn(*mut c_void)> =
                lib.get(b"OrtReleaseEnv")?;
            let create_session_options: Symbol<
                unsafe extern "C" fn(*mut *mut c_void) -> OrtStatusPtr,
            > = lib.get(b"OrtCreateSessionOptions")?;
            let release_session_options: Symbol<unsafe extern "C" fn(*mut c_void)> =
                lib.get(b"OrtReleaseSessionOptions")?;
            let set_gpu_mem_limit: Symbol<
                unsafe extern "C" fn(*mut c_void, c_int, c_int) -> OrtStatusPtr,
            > = lib.get(b"OrtSetSessionMemoryAlignment")?;
            let create_session: Symbol<
                unsafe extern "C" fn(
                    *const c_char,
                    *mut c_void,
                    *mut c_void,
                    *mut *mut c_void,
                ) -> OrtStatusPtr,
            > = lib.get(b"OrtCreateSession")?;
            let release_session: Symbol<unsafe extern "C" fn(*mut c_void)> =
                lib.get(b"OrtReleaseSession")?;
            let get_input_count: Symbol<
                unsafe extern "C" fn(*mut c_void, *mut c_int) -> OrtStatusPtr,
            > = lib.get(b"OrtSessionGetInputCount")?;
            let get_output_count: Symbol<
                unsafe extern "C" fn(*mut c_void, *mut c_int) -> OrtStatusPtr,
            > = lib.get(b"OrtSessionGetOutputCount")?;
            let get_input_name: Symbol<
                unsafe extern "C" fn(
                    *mut c_void,
                    c_int,
                    *mut c_void,
                    *mut *mut c_void,
                ) -> OrtStatusPtr,
            > = lib.get(b"OrtSessionGetInputName")?;
            let get_output_name: Symbol<
                unsafe extern "C" fn(
                    *mut c_void,
                    c_int,
                    *mut c_void,
                    *mut *mut c_void,
                ) -> OrtStatusPtr,
            > = lib.get(b"OrtSessionGetOutputName")?;
            let release_allocator: Symbol<unsafe extern "C" fn(*mut c_void)> =
                lib.get(b"OrtReleaseAllocator")?;
            let create_memory_info: Symbol<
                unsafe extern "C" fn(
                    *mut *mut c_void,
                    *mut *mut c_void,
                    usize,
                    c_int,
                    c_int,
                ) -> OrtStatusPtr,
            > = lib.get(b"OrtCreateMemoryInfo")?;
            let release_memory_info: Symbol<unsafe extern "C" fn(*mut c_void)> =
                lib.get(b"OrtReleaseMemoryInfo")?;
            let create_tensor_with_data: Symbol<
                unsafe extern "C" fn(
                    *mut *mut c_void,
                    *const c_void,
                    usize,
                    *mut c_void,
                    c_int,
                    *mut *mut c_void,
                ) -> OrtStatusPtr,
            > = lib.get(b"OrtCreateTensorWithDataAsOrtValue")?;
            let get_tensor_type: Symbol<
                unsafe extern "C" fn(*mut c_void, *mut c_int) -> OrtStatusPtr,
            > = lib.get(b"OrtGetTensorType")?;
            let release_value: Symbol<unsafe extern "C" fn(*mut c_void)> =
                lib.get(b"OrtReleaseValue")?;
            let run: Symbol<
                unsafe extern "C" fn(
                    *mut c_void,
                    *mut c_void,
                    *const *const c_char,
                    *const *mut c_void,
                    c_int,
                    *mut *mut c_void,
                ) -> OrtStatusPtr,
            > = lib.get(b"OrtRun")?;
            let get_tensor_shape: Symbol<
                unsafe extern "C" fn(*mut c_void, *mut c_int, *mut c_int) -> OrtStatusPtr,
            > = lib.get(b"OrtGetTensorShape")?;
            let get_allocator_with_default_options: Symbol<
                unsafe extern "C" fn(*mut *mut c_void) -> OrtStatusPtr,
            > = lib.get(b"OrtGetAllocatorWithDefaultOptions")?;

            Ok(OnnxRuntime {
                lib,
                get_api_base,
                get_api,
                create_env,
                release_env,
                create_session_options,
                release_session_options,
                set_gpu_mem_limit,
                create_session,
                release_session,
                get_input_count,
                get_output_count,
                get_input_name,
                get_output_name,
                release_allocator,
                create_memory_info,
                release_memory_info,
                create_tensor_with_data,
                get_tensor_type,
                release_value,
                run,
                get_tensor_shape,
                get_string_tensor_data,
                get_string_tensor_data_length,
                get_allocator_with_default_options,
            })
        }
    }
}

pub struct OnnxSession {
    runtime: *mut OnnxRuntime,
    session: OrtSessionPtr,
    input_names: Vec<String>,
    output_names: Vec<String>,
}

impl OnnxSession {
    pub fn load(runtime: &OnnxRuntime, model_path: &Path) -> Result<Self, Box<dyn Error>> {
        let mut session: OrtSessionPtr = null_mut();

        unsafe {
            let model_path_cstr = std::ffi::CString::new(model_path.to_string_lossy().as_bytes())?;

            let status = (runtime.create_session)(
                model_path_cstr.as_ptr(),
                runtime as *const _ as *mut c_void,
                null_mut(),
                &mut session,
            );

            if !status.is_null() {
                return Err(OnnxRuntimeError("Failed to create ONNX session".into()).into());
            }

            let mut input_count: c_int = 0;
            let mut output_count: c_int = 0;

            let status = (runtime.get_input_count)(session, &mut input_count);
            if !status.is_null() {
                return Err(OnnxRuntimeError("Failed to get input count".into()).into());
            }

            let status = (runtime.get_output_count)(session, &mut output_count);
            if !status.is_null() {
                return Err(OnnxRuntimeError("Failed to get output count".into()).into());
            }

            let mut allocator: OrtAllocatorPtr = null_mut();
            let status = (runtime.get_allocator_with_default_options)(&mut allocator);
            if !status.is_null() {
                return Err(OnnxRuntimeError("Failed to get default allocator".into()).into());
            }

            let mut input_names = Vec::new();
            for i in 0..input_count {
                let mut name_ptr: *mut c_char = null_mut();
                let status = (runtime.get_input_name)(
                    session,
                    i,
                    allocator,
                    &mut name_ptr as *mut *mut c_char as *mut *mut c_void,
                );
                if !status.is_null() {
                    return Err(OnnxRuntimeError("Failed to get input name".into()).into());
                }
                let name = CStr::from_ptr(name_ptr).to_string_lossy().into_owned();
                input_names.push(name);
                // Note: We are leaking the name string memory here because we don't have OrtAllocatorFree bound
            }

            let mut output_names = Vec::new();
            for i in 0..output_count {
                let mut name_ptr: *mut c_char = null_mut();
                let status = (runtime.get_output_name)(
                    session,
                    i,
                    allocator,
                    &mut name_ptr as *mut *mut c_char as *mut *mut c_void,
                );
                if !status.is_null() {
                    return Err(OnnxRuntimeError("Failed to get output name".into()).into());
                }
                let name = CStr::from_ptr(name_ptr).to_string_lossy().into_owned();
                output_names.push(name);
            }

            Ok(OnnxSession {
                runtime: runtime as *const _ as *mut OnnxRuntime,
                session,
                input_names,
                output_names,
            })
        }
    }

    pub fn run(
        &self,
        runtime: &OnnxRuntime,
        inputs: &[&[f32]],
        output_shapes: &[Vec<i64>],
    ) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
        let mut output_values: Vec<OrtValuePtr> = Vec::new();
        let mut input_values: Vec<OrtValuePtr> = Vec::new();

        let mut allocator: OrtAllocatorPtr = null_mut();
        let mut memory_info: OrtMemoryInfoPtr = null_mut();

        unsafe {
            let status = (runtime.create_memory_info)(&mut memory_info, &mut allocator, 0, 1, 0);
            if !status.is_null() {
                return Err(OnnxRuntimeError("Failed to create memory info".into()).into());
            }

            for (i, input_data) in inputs.iter().enumerate() {
                let shape: [i64; 2] = [1, input_data.len() as i64];
                let mut value: OrtValuePtr = null_mut();
                let status = (runtime.create_tensor_with_data)(
                    &mut value,
                    input_data.as_ptr() as *const c_void,
                    (input_data.len() * std::mem::size_of::<f32>()) as usize,
                    memory_info,
                    2,
                    shape.as_ptr() as *mut c_void,
                );

                if !status.is_null() {
                    return Err(OnnxRuntimeError("Failed to create input tensor".into()).into());
                }

                input_values.push(value);
            }

            let input_names_cstring: Result<Vec<CString>, _> = self
                .input_names
                .iter()
                .map(|s| CString::new(s.as_str()))
                .collect();
            let input_names_cstring = input_names_cstring?;

            let input_names_ptr: Vec<*const c_char> =
                input_names_cstring.iter().map(|cs| cs.as_ptr()).collect();

            let mut run_options: OrtRunOptionsPtr = null_mut();
            let status = (runtime.run)(
                self.session,
                run_options,
                input_names_ptr.as_ptr(),
                input_values.as_ptr() as *const *mut OrtValuePtr,
                input_names_ptr.len() as c_int,
                output_values.as_mut_ptr(),
                output_shapes.len() as i32,
            );

            for value in input_values {
                (runtime.release_value)(value);
            }
            (runtime.release_memory_info)(memory_info);

            if !status.is_null() {
                return Err(OnnxRuntimeError("ONNX run failed".into()).into());
            }

            output_values.set_len(output_shapes.len());

            let mut outputs = Vec::new();
            for (i, shape) in output_shapes.iter().enumerate() {
                let value = output_values[i];
                let mut tensor_type: c_int = 0;
                let status = (runtime.get_tensor_type)(value, &mut tensor_type);
                if !status.is_null() {
                    for v in &output_values {
                        (runtime.release_value)(*v);
                    }
                    return Err(OnnxRuntimeError("Failed to get tensor type".into()).into());
                }

                let mut dims: [c_int; 4] = [0, 0, 0, 0];
                let status = (runtime.get_tensor_shape)(value, dims.as_mut_ptr(), 4);
                if !status.is_null() {
                    for v in &output_values {
                        (runtime.release_value)(*v);
                    }
                    return Err(OnnxRuntimeError("Failed to get tensor shape".into()).into());
                }

                let total_size: usize = shape.iter().product::<i64>() as usize;
                let mut output_data: Vec<f32> = vec![0.0; total_size];

                let data_ptr = output_data.as_mut_ptr() as *mut c_void;
                std::ptr::copy_nonoverlapping(
                    value as *const c_void,
                    data_ptr,
                    total_size * std::mem::size_of::<f32>(),
                );

                outputs.push(output_data);
            }

            for value in output_values {
                (runtime.release_value)(value);
            }

            Ok(outputs)
        }
    }
}

impl Drop for OnnxSession {
    fn drop(&mut self) {
        unsafe {
            if !self.session.is_null() {
                let runtime = &*self.runtime;
                (runtime.release_session)(self.session);
            }
        }
    }
}
