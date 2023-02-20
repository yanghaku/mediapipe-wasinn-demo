use mediapipe_wasinn_demo::inference::{InferenceGraphBuilder, InferenceTensorDataLayout};
use mediapipe_wasinn_demo::preprocess::ToTensor;
use std::path::PathBuf;

#[test]
fn test_inference() {
    let assets = PathBuf::from("./assets");

    // prepare input
    let input = image::open(assets.join("test.jpg"))
        .unwrap()
        .resize(128, 128, image::imageops::FilterType::Triangle)
        .to_rgb8()
        .to_tensor(InferenceTensorDataLayout::NHWC);

    // prepare module
    let graph = InferenceGraphBuilder::default()
        .build_from_file(assets.join("face_detection_short_range.tflite"))
        .unwrap();
    let mut graph_exec = graph.new_graph_executor().unwrap();

    // do inference
    graph_exec.set_inputs_and_run([(0, input)]).unwrap();

    const REGRESSORS_SIZE: u32 = 1 * 896 * 16;
    const SCORES_SIZE: u32 = 1 * 896 * 1;

    // get output
    let regressors = graph_exec.get_output_f32(0, REGRESSORS_SIZE).unwrap();
    let scores = graph_exec.get_output_f32(1, SCORES_SIZE).unwrap();

    assert_eq!(regressors.len(), REGRESSORS_SIZE as usize);
    assert_eq!(scores.len(), SCORES_SIZE as usize);
}
