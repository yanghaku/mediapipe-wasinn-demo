use std::cmp::{min, Ordering};
use std::fmt::{Display, Formatter};

use image::DynamicImage;

use crate::inference::*;
use crate::postprocess::{ops::Sigmoid, Anchor, Box2D, Pointer2D, SsdAnchorsGeneratorOptions};
use crate::preprocess::ToTensor;

pub struct FaceDetection {
    graph: InferenceGraph,
    anchors: Vec<Anchor>,
    min_detection_confidence: f32,
}

/// ref: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_detection/face_detection.pbtxt
/// https://github.com/google/mediapipe/blob/master/mediapipe/examples/desktop/autoflip/subgraph/front_face_detection_subgraph.pbtxt
impl FaceDetection {
    pub fn new(
        module_selection: FaceDetectionModels,
        min_detection_confidence: f32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let graph =
            InferenceGraphBuilder::default().build_from_file(module_selection.to_model_path())?;
        let mut anchor_generator_opt =
            SsdAnchorsGeneratorOptions::new(128, 128, 0.1484375, 0.75, 4);
        anchor_generator_opt.aspect_ratios.push(1.0f32);
        anchor_generator_opt.fixed_anchor_size = true;
        anchor_generator_opt.strides = vec![8, 16, 16, 16];
        Ok(Self {
            graph,
            anchors: anchor_generator_opt.generate(),
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
            .resize(128, 128, image::imageops::FilterType::Triangle)
            .to_rgb8()
            .to_tensor(InferenceTensorDataLayout::NHWC);

        // do inference
        self.graph_exec.set_inputs_and_run([(0, input)])?;

        // get output
        let regressors = self.graph_exec.get_output_f32(0, Self::REGRESSORS_SIZE)?;
        let mut scores = self.graph_exec.get_output_f32(1, Self::SCORES_SIZE)?;

        Ok(FaceDetectionModelOutput::from_with_threshold(
            &self.face_detection.anchors,
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
    pub mouse_center: Pointer2D<f32>,
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

impl Display for FaceDetectionModelOutput {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "FaceDetectionModelOutput: {{\n\tScore: {}\n\
            \tFace bound box: {{ xmin: {}, ymin: {}, width: {}, height: {} }}\n\
            \tLeft eye: ({}, {})\n\tRight eye: ({}, {})\n\tNose tip: ({}, {})\n\
            \tMouse center: ({}, {})\n\tLeft eye tragion: ({}, {})\n\tRight eye tragion: ({}, {})\n}}",
            self.score,
            self.face.p.x,
            self.face.p.y,
            self.face.w,
            self.face.h,
            self.right_eye.x,
            self.right_eye.y,
            self.left_eye.x,
            self.left_eye.y,
            self.nose_tip.x,
            self.nose_tip.y,
            self.mouse_center.x,
            self.mouse_center.y,
            self.left_eye_tragion.x,
            self.left_eye_tragion.y,
            self.right_eye_tragion.x,
            self.right_eye_tragion.y,
        )
    }
}

/// https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_detection/face_detection.pbtxt
/// https://github.com/google/mediapipe/blob/master/mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
impl FaceDetectionModelOutput {
    pub fn from(anchor: &Anchor, value: &[f32], score: f32) -> Self {
        // todo: make it to options
        let x_scale = 128.0;
        let y_scale = 128.0;
        let h_scale = 128.0;
        let w_scale = 128.0;

        let mut x_center = value[0];
        let mut y_center = value[1];
        let mut w = value[2];
        let mut h = value[3];

        x_center = x_center / x_scale * anchor.w + anchor.x_center;
        y_center = y_center / y_scale * anchor.h + anchor.y_center;
        h = h / h_scale * anchor.h;
        w = w / w_scale * anchor.w;

        let y_min = y_center - h / 2f32;
        let x_min = x_center - w / 2f32;
        let y_max = y_center + h / 2f32;
        let x_max = x_center + w / 2f32;
        let box2d = Box2D {
            p: Pointer2D { x: x_min, y: y_min },
            w: x_max - x_min,
            h: y_max - y_min,
        };

        let mut key_points = Vec::with_capacity(6);
        for k in 0..6 {
            key_points.push(
                Pointer2D {
                    x: value[4 + (k << 1)],
                    y: value[4 + ((k << 1) | 1)],
                }
                .transform_with_anchor(anchor, x_scale, y_scale),
            );
        }
        Self {
            face: box2d,
            left_eye: key_points[0].clone(),
            right_eye: key_points[1].clone(),
            nose_tip: key_points[2].clone(),
            mouse_center: key_points[3].clone(),
            left_eye_tragion: key_points[4].clone(),
            right_eye_tragion: key_points[5].clone(),
            score,
        }
    }

    fn from_with_threshold(
        anchors: &Vec<Anchor>,
        regressors: &Vec<f32>,
        scores: &mut Vec<f32>,
        score_threshold: f32,
    ) -> Vec<Self> {
        scores.sigmoid_inplace();
        let num = min(regressors.len() >> 4, scores.len());
        let mut res = Vec::new();
        for i in 0..num {
            let score = scores[i];
            if score > score_threshold {
                res.push(FaceDetectionModelOutput::from(
                    &anchors[i],
                    &regressors[(i << 4)..],
                    score,
                ));
            }
        }
        res
    }

    pub fn face_box(&self) -> &Box2D<f32> {
        &self.face
    }
}
