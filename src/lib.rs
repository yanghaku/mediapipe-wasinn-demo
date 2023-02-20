/// process the media input to tensor
pub mod preprocess;

/// do inference using wasi-nn
pub mod inference;

/// process the inference output
pub mod postprocess;

/// mediapipe solutions (such as face detection)
mod solutions;
pub use solutions::*;
