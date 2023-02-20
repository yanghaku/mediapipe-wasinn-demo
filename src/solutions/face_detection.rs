use crate::inference::*;
use crate::postprocess::{ops::Softmax, Box2D, Pointer2D};
use crate::preprocess::ToTensor;
use image::DynamicImage;
use std::cmp::{min, Ordering};

pub struct FaceDetection {
    graph: InferenceGraph,
    min_detection_confidence: f32,
}

impl FaceDetection {
    pub fn new(
        module_selection: FaceDetectionModels,
        min_detection_confidence: f32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let graph =
            InferenceGraphBuilder::default().build_from_file(module_selection.to_model_path())?;
        Ok(Self {
            graph,
            min_detection_confidence,
        })
    }

    // just process once
    pub fn process(
        &self,
        image: &DynamicImage,
    ) -> Result<Vec<FaceDetectionModelOutput>, Box<dyn std::error::Error>> {
        self.generate_processor()?.process_img(image)
    }

    pub fn generate_processor(&self) -> Result<FaceDetectionProcessor, Box<dyn std::error::Error>> {
        let graph_exec = self.graph.new_graph_executor()?;
        Ok(FaceDetectionProcessor {
            face_detection: self,
            graph_exec,
        })
    }
}

/// process a image or stream
pub struct FaceDetectionProcessor<'a> {
    face_detection: &'a FaceDetection,
    graph_exec: InferenceGraphExecutor<'a>,
}

impl<'a> FaceDetectionProcessor<'a> {
    const REGRESSORS_SIZE: u32 = 1 * 896 * 16;
    const SCORES_SIZE: u32 = 1 * 896 * 1;

    pub fn process_img(
        &mut self,
        image: &DynamicImage,
    ) -> Result<Vec<FaceDetectionModelOutput>, Box<dyn std::error::Error>> {
        // generate input
        let input = image
            .resize(128, 128, image::imageops::FilterType::Nearest)
            .to_rgb8()
            .to_tensor(InferenceTensorDataLayout::NHWC);

        // do inference
        self.graph_exec.set_input(0, &input)?;
        self.graph_exec.run()?;

        // get output
        let regressors = self.graph_exec.get_output_f32(0, Self::REGRESSORS_SIZE)?;
        let mut scores = self.graph_exec.get_output_f32(1, Self::SCORES_SIZE)?;

        Ok(FaceDetectionModelOutput::from_with_softmax_threshold(
            &regressors,
            &mut scores,
            self.face_detection.min_detection_confidence,
        ))
    }
}

#[derive(Debug, Clone)]
pub enum FaceDetectionModels {
    ShortRange,
    ShortRangeQuantized,
    FullRangeDense,
    FullRangeSparse,
}

impl FaceDetectionModels {
    fn to_model_path(self) -> &'static str {
        match self {
            FaceDetectionModels::ShortRange => "./assets/face_detection_short_range.tflite",
            _ => {
                unimplemented!()
            }
        }
    }
}

/// Module Info: https://mediapipe.page.link/blazeface-mc
/// Module output can get from [TensorFlow Lite Model Analyzer](https://www.tensorflow.org/lite/guide/model_analyzer)
#[derive(Debug, Clone)]
pub struct FaceDetectionModelOutput {
    pub face: Box2D<f32>,
    pub left_eye: Pointer2D<f32>,
    pub right_eye: Pointer2D<f32>,
    pub nose_tip: Pointer2D<f32>,
    pub mouse: Pointer2D<f32>,
    pub left_eye_tragion: Pointer2D<f32>,
    pub right_eye_tragion: Pointer2D<f32>,
    pub score: f32,
}

impl Eq for FaceDetectionModelOutput {}

impl PartialEq<Self> for FaceDetectionModelOutput {
    fn eq(&self, other: &Self) -> bool {
        // todo
        self.score.eq(&other.score)
    }
}

impl PartialOrd<Self> for FaceDetectionModelOutput {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.score.partial_cmp(&other.score)
    }
}

impl Ord for FaceDetectionModelOutput {
    /// compare using score
    fn cmp(&self, other: &Self) -> Ordering {
        // todo: handle Nan
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Greater)
    }
}

impl FaceDetectionModelOutput {
    pub fn from(value: &[f32], score: f32) -> Self {
        Self {
            face: Box2D::from(value, 1),
            left_eye: Pointer2D {
                x: value[4],
                y: value[5],
            },
            right_eye: Pointer2D {
                x: value[6],
                y: value[7],
            },
            nose_tip: Pointer2D {
                x: value[8],
                y: value[9],
            },
            mouse: Pointer2D {
                x: value[10],
                y: value[11],
            },
            left_eye_tragion: Pointer2D {
                x: value[12],
                y: value[13],
            },
            right_eye_tragion: Pointer2D {
                x: value[14],
                y: value[15],
            },
            score,
        }
    }

    pub fn from_with_threshold(
        regressors: &Vec<f32>,
        scores: &Vec<f32>,
        score_threshold: f32,
    ) -> Vec<Self> {
        let num = min(regressors.len() >> 4, scores.len());
        let mut res = Vec::new();
        for i in 0..num {
            let score = scores[i];
            if score > score_threshold {
                res.push(FaceDetectionModelOutput::from(
                    &regressors[(i << 4)..],
                    score,
                ));
            }
        }
        res
    }

    #[inline]
    pub fn from_with_softmax_threshold(
        regressors: &Vec<f32>,
        scores: &mut Vec<f32>,
        score_threshold: f32,
    ) -> Vec<Self> {
        scores.softmax_inplace();
        Self::from_with_threshold(regressors, scores, score_threshold)
    }
}
