use wasi_nn;

#[derive(Debug, Clone)]
pub enum InferenceTensorDataLayout {
    NCHW,
    NHWC,
    CHWN,
}

#[derive(Debug, Clone)]
pub enum InferenceTensorType {
    F16,
    F32,
    U8,
    I32,
}

impl Into<wasi_nn::TensorType> for InferenceTensorType {
    fn into(self) -> wasi_nn::TensorType {
        match self {
            InferenceTensorType::F16 => wasi_nn::TENSOR_TYPE_F16,
            InferenceTensorType::F32 => wasi_nn::TENSOR_TYPE_F32,
            InferenceTensorType::U8 => wasi_nn::TENSOR_TYPE_U8,
            InferenceTensorType::I32 => wasi_nn::TENSOR_TYPE_I32,
        }
    }
}

pub enum TensorData<'a> {
    Owned {
        shape: Vec<u32>,
        data: Vec<u8>,
    },
    Ref {
        shape_ref: &'a [u32],
        data_ref: &'a [u8],
    },
}

pub struct InferenceTensor<'a> {
    tp: InferenceTensorType,
    data_layout: InferenceTensorDataLayout,
    data: TensorData<'a>,
}

impl<'a> InferenceTensor<'a> {
    pub fn new_ref(
        tp: InferenceTensorType,
        data_layout: InferenceTensorDataLayout,
        shape_ref: &'a [u32],
        data_ref: &'a [u8],
    ) -> Self {
        Self {
            tp,
            data_layout,
            data: TensorData::Ref {
                shape_ref,
                data_ref,
            },
        }
    }

    pub fn new(
        tp: InferenceTensorType,
        data_layout: InferenceTensorDataLayout,
        shape: Vec<u32>,
        data: Vec<u8>,
    ) -> Self {
        Self {
            tp,
            data_layout,
            data: TensorData::Owned { shape, data },
        }
    }

    pub fn shape_ref(&self) -> &[u32] {
        match &self.data {
            &TensorData::Ref { ref shape_ref, .. } => shape_ref,
            &TensorData::Owned { ref shape, .. } => shape.as_ref(),
        }
    }

    pub fn data_ref(&self) -> &[u8] {
        match &self.data {
            &TensorData::Ref { ref data_ref, .. } => data_ref,
            &TensorData::Owned { ref data, .. } => data.as_ref(),
        }
    }

    pub fn tensor_data_ref(&self) -> (&[u32], &[u8]) {
        match &self.data {
            &TensorData::Ref {
                ref shape_ref,
                ref data_ref,
            } => (shape_ref, data_ref),
            &TensorData::Owned {
                ref shape,
                ref data,
            } => (shape.as_ref(), data.as_ref()),
        }
    }

    pub fn tp(&self) -> InferenceTensorType {
        self.tp.clone()
    }

    pub fn data_layout(&self) -> &InferenceTensorDataLayout {
        &self.data_layout
    }

    pub fn to_owned(self) -> InferenceTensor<'static> {
        todo!()
    }
}
