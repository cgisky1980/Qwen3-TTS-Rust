//! Stream module placeholder
pub struct TTSStream;
#[derive(Default)]
pub struct TTSResult {
    pub audio: Vec<f32>,
    pub text: String,
    pub codes: Vec<i64>,
}
