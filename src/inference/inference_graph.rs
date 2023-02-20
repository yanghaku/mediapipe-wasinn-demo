use super::{InferenceError, InferenceTensor};
use std::path::Path;
use wasi_nn;

/// Inference Graph Encoding
/// with graph encoding, we can chose the backend for wasi-nn
/// now, it can only support tflite!
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum InferenceGraphEncoding {
    Onnx,
    Openvino,
    Tensorflow,
    Pytorch,
    TensorflowLite,
}

impl Into<wasi_nn::GraphEncoding> for InferenceGraphEncoding {
    #[inline(always)]
    fn into(self) -> wasi_nn::GraphEncoding {
        match self {
            InferenceGraphEncoding::Onnx => wasi_nn::GRAPH_ENCODING_ONNX,
            InferenceGraphEncoding::Openvino => wasi_nn::GRAPH_ENCODING_OPENVINO,
            InferenceGraphEncoding::Tensorflow => wasi_nn::GRAPH_ENCODING_TENSORFLOW,
            InferenceGraphEncoding::Pytorch => wasi_nn::GRAPH_ENCODING_PYTORCH,
            InferenceGraphEncoding::TensorflowLite => unsafe { std::mem::transmute(4u8) },
        }
    }
}

/// Inference Graph Device
/// the device for graph to run
/// now, it can only support CPU!
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InferenceGraphDevice {
    CPU,
    GPU,
    TPU,
}

impl Into<wasi_nn::ExecutionTarget> for InferenceGraphDevice {
    #[inline]
    fn into(self) -> wasi_nn::ExecutionTarget {
        match self {
            InferenceGraphDevice::CPU => wasi_nn::EXECUTION_TARGET_CPU,
            InferenceGraphDevice::GPU => wasi_nn::EXECUTION_TARGET_GPU,
            InferenceGraphDevice::TPU => wasi_nn::EXECUTION_TARGET_TPU,
        }
    }
}

/// builder for InferenceGraph
///
/// ### Examples
///
/// #### build a graph with default config ( tflite + cpu )
/// ```
/// use mediapipe_wasinn_demo::inference::InferenceGraphBuilder;
/// let path = "./module.tflite";
/// let graph = InferenceGraphBuilder::default().build_from_file(path)?;
/// ```
#[derive(Debug, Clone)]
pub struct InferenceGraphBuilder {
    encoding: InferenceGraphEncoding,
    device: InferenceGraphDevice,
}

impl Default for InferenceGraphBuilder {
    /// default use tflite and cpu device
    #[inline(always)]
    fn default() -> Self {
        Self::new(
            InferenceGraphEncoding::TensorflowLite,
            InferenceGraphDevice::CPU,
        )
    }
}

impl InferenceGraphBuilder {
    #[inline]
    pub fn new(backend: InferenceGraphEncoding, device: InferenceGraphDevice) -> Self {
        Self {
            encoding: backend,
            device,
        }
    }

    #[inline]
    pub fn encoding(mut self, backend: InferenceGraphEncoding) -> Self {
        self.encoding = backend;
        self
    }

    #[inline]
    pub fn device(mut self, device: InferenceGraphDevice) -> Self {
        self.device = device;
        self
    }

    // todo: new interface to support array input for ```load```

    #[inline]
    pub fn build_from_bytes(self, bytes: Vec<u8>) -> Result<InferenceGraph, InferenceError> {
        let build_info = self.clone();
        let graph_handle =
            unsafe { wasi_nn::load(&[bytes.as_ref()], self.encoding.into(), self.device.into()) }
                .map_err(|e| InferenceError::from(e))?;
        Ok(InferenceGraph {
            build_info,
            graph_handle,
            _graph_content: bytes,
        })
    }

    #[inline(always)]
    pub fn build_from_file(self, file: impl AsRef<Path>) -> Result<InferenceGraph, InferenceError> {
        let bytes = std::fs::read(file).map_err(|_e| InferenceError::IOError)?;
        self.build_from_bytes(bytes)
    }
}

/// InferenceGraph
/// InferenceGraph can create a InferenceGraphExecutor which can do inference.
///
pub struct InferenceGraph {
    build_info: InferenceGraphBuilder,
    graph_handle: wasi_nn::Graph,
    _graph_content: Vec<u8>,
}

impl InferenceGraph {
    #[inline(always)]
    pub fn encoding(&self) -> InferenceGraphEncoding {
        self.build_info.encoding.clone()
    }

    #[inline(always)]
    pub fn device(&self) -> InferenceGraphDevice {
        self.build_info.device.clone()
    }

    #[inline]
    pub fn new_graph_executor(&self) -> Result<InferenceGraphExecutor, InferenceError> {
        let ctx = unsafe { wasi_nn::init_execution_context(self.graph_handle) }
            .map_err(|e| InferenceError::from(e))?;
        Ok(InferenceGraphExecutor {
            graph: self,
            execute_ctx: ctx,
        })
    }
}

impl Drop for InferenceGraph {
    fn drop(&mut self) {
        // todo: destroy this graph?
    }
}

/// Inference Graph Executor
/// Wrapper a graph executor context which can do inference using wasi-nn
///
/// ### Examples
///
/// #### load module and do inference
/// ```
/// use mediapipe_wasinn_demo::inference::{InferenceGraphBuilder, InferenceTensor, InferenceTensorDataLayout, InferenceTensorType};
///
/// let path = "./module.tflite";
/// let graph = InferenceGraphBuilder::default().build_from_file(path)?;
/// let mut executor = graph.new_graph_executor()?;
/// // generate input tensor
/// let input = InferenceTensor::new(InferenceTensorType::F32,InferenceTensorDataLayout::NHWC,Vec::default(),Vec::default());
/// executor.set_inputs_and_run([(0, input)])?;
/// let (output, output_size) = executor.get_output(0, 1024)?;
/// ```
pub struct InferenceGraphExecutor<'a> {
    graph: &'a InferenceGraph,
    execute_ctx: wasi_nn::GraphExecutionContext,
}

impl<'a> InferenceGraphExecutor<'a> {
    #[inline]
    pub fn graph(&self) -> &'a InferenceGraph {
        self.graph
    }

    pub fn set_inputs_and_run(
        &mut self,
        inputs: impl AsRef<[(u32, InferenceTensor<'a>)]>,
    ) -> Result<(), InferenceError> {
        for (ref index, ref input) in inputs.as_ref() {
            let (shape, data) = input.tensor_data_ref();
            let tensor = wasi_nn::Tensor {
                dimensions: shape,
                type_: input.tp().into(),
                data,
            };
            unsafe { wasi_nn::set_input(self.execute_ctx, *index, tensor) }
                .map_err(|e| InferenceError::from(e))?;
        }

        let res = self.run();

        // drop inputs after run
        drop(inputs);
        res
    }

    #[inline(always)]
    fn run(&mut self) -> Result<(), InferenceError> {
        unsafe { wasi_nn::compute(self.execute_ctx) }.map_err(|e| InferenceError::from(e))
    }

    pub fn get_output_with_buffer(
        &mut self,
        index: u32,
        mut buf: impl AsMut<[u8]>,
        buf_size: u32,
    ) -> Result<u32, InferenceError> {
        unsafe {
            wasi_nn::get_output(
                self.execute_ctx,
                index,
                buf.as_mut() as *mut [u8] as *mut u8,
                buf_size,
            )
        }
        .map_err(|e| InferenceError::from(e))
    }

    // todo: opt these api

    pub fn get_output_u8(
        &mut self,
        index: u32,
        expect_len: u32,
    ) -> Result<Vec<u8>, InferenceError> {
        let mut buf: Vec<u8> = vec![0; expect_len as usize];
        let recv = self.get_output_with_buffer(index, &mut buf[..], expect_len)?;
        if recv == expect_len {
            Ok(buf)
        } else {
            Err(InferenceError::OutputGetLenError {
                expect: expect_len,
                got: recv,
            })
        }
    }

    pub fn get_output_f32(
        &mut self,
        index: u32,
        expect_len: u32,
    ) -> Result<Vec<f32>, InferenceError> {
        let mut buf: Vec<f32> = vec![0f32; expect_len as usize];
        let expect_buf_len = expect_len << 2;
        let recv = unsafe {
            wasi_nn::get_output(
                self.execute_ctx,
                index,
                &mut buf[..] as *mut [f32] as *mut u8,
                expect_buf_len,
            )
        }
        .map_err(|e| InferenceError::from(e))?;

        if recv == expect_buf_len {
            Ok(buf)
        } else {
            Err(InferenceError::OutputGetLenError {
                expect: expect_buf_len,
                got: recv,
            })
        }
    }
}

impl<'a> Drop for InferenceGraphExecutor<'a> {
    fn drop(&mut self) {
        // todo: destroy this graph execution context?
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_graph_builder_default() {
        let graph = InferenceGraphBuilder::default();
        assert_eq!(graph.device, InferenceGraphDevice::CPU);
        assert_eq!(graph.encoding, InferenceGraphEncoding::TensorflowLite);
    }
}
