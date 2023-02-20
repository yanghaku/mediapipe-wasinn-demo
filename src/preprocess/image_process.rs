use super::*;
use crate::inference::{InferenceTensorDataLayout as DataLayout, InferenceTensorType};
use image::RgbImage;

/// RgbImage generate a tensor
/// before generate tensor, the image must do resize!
///
/// Image Preprocess Reference:
/// NHWC for tflite: https://github.com/tensorflow/models/blob/4fcd44d71eb15c1c17612bf6cefc646caaf671f1/research/slim/preprocessing/inception_preprocessing.py#L258
///
impl ToTensor for RgbImage {
    fn to_tensor(&self, data_layout: DataLayout) -> InferenceTensor<'static> {
        const MULTIPLY: f32 = 2.0f32 / 255.0f32;

        match data_layout {
            DataLayout::NHWC => {
                let mut data = Vec::with_capacity((self.height() * self.width() * 3 * 4) as usize);
                for p in self.as_ref() {
                    let f = (*p as f32) * MULTIPLY - 1.0f32;
                    for b in f.to_ne_bytes() {
                        data.push(b);
                    }
                }
                InferenceTensor::new(
                    InferenceTensorType::F32,
                    data_layout,
                    vec![1, self.height(), self.width(), 3],
                    data,
                )
            }
            _ => unimplemented!(),
        }
    }
}
