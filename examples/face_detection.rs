use mediapipe_wasinn_demo::{FaceDetection, FaceDetectionModels};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage {} [image path]", args.get(0).unwrap())
    }

    // read image from file
    let img_path = args.get(1).unwrap();
    let img = image::open(img_path)?;

    // create face detection solution
    let face_detection = FaceDetection::new(FaceDetectionModels::ShortRange, 0.9)?;

    // process a image and get output
    let mut results = face_detection.process(&img)?;

    // sort using score with ascending order
    results.sort();

    // print the results
    println!("The number of results: {}", results.len());
    for r in results {
        println!("{}", r);
    }

    // todo: process the result and draw to image
    Ok(())
}
