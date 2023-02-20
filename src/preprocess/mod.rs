mod image_process;

// other: video stream process? audio process?

use super::inference::{InferenceTensor, InferenceTensorDataLayout};

/// use data to generate a tensor
pub trait ToTensor {
    fn to_tensor(&self, data_layout: InferenceTensorDataLayout) -> InferenceTensor<'static>;
}

/// use data to generate a tensor, and tensor data is a reference of a memory
pub trait ToTensorRef {
    fn to_tensor_ref(&self, data_layout: InferenceTensorDataLayout) -> InferenceTensor;
}
