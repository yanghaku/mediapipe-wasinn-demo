use std::fmt::{Display, Formatter};
use wasi_nn;

#[derive(Clone, Debug)]
pub enum InferenceError {
    /// graph file error
    IOError,

    /// get output buf error
    OutputGetLenError {
        expect: u32,
        got: u32,
    },

    InvalidEncoding,

    InvalidArgument,

    MissingMemory,

    Busy,

    RuntimeError,

    UnknownError,
}

impl From<wasi_nn::NnErrno> for InferenceError {
    #[inline(always)]
    fn from(value: wasi_nn::NnErrno) -> Self {
        match value {
            wasi_nn::NN_ERRNO_INVALID_ARGUMENT => Self::InvalidArgument,
            wasi_nn::NN_ERRNO_INVALID_ENCODING => Self::InvalidEncoding,
            wasi_nn::NN_ERRNO_MISSING_MEMORY => Self::MissingMemory,
            wasi_nn::NN_ERRNO_BUSY => Self::Busy,
            wasi_nn::NN_ERRNO_RUNTIME_ERROR => Self::RuntimeError,
            _ => Self::UnknownError,
        }
    }
}

impl std::error::Error for InferenceError {}

impl Display for InferenceError {
    fn fmt(&self, _f: &mut Formatter<'_>) -> std::fmt::Result {
        // todo: impl the error description, or use ```this_error``` crate
        unimplemented!()
    }
}
