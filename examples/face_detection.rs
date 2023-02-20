use mediapipe_wasinn_demo::{FaceDetection, FaceDetectionModels};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage {} [image path] [output image path]",
            args.get(0).unwrap()
        )
    }
    let img_path = args.get(1).unwrap();
    let out_img_path = args.get(2).unwrap();

    // read image from file
    let mut img = image::open(img_path)?;

    // create face detection solution
    let face_detection = FaceDetection::new(FaceDetectionModels::ShortRange, 0.9)?;

    // process a image and get output
    let mut results = face_detection.process(&img)?;

    // print the results
    println!("The number of results: {}", results.len());
    for r in &results {
        println!("{}", r);
    }

    // todo: do nms and draw

    // now we just sort the results with scores, and draw the box with largest score
    if !results.is_empty() {
        results.sort();
        mediapipe_wasinn_demo::postprocess::draw_utils::draw_a_box(
            &mut img,
            results.last().unwrap().face_box(),
        );
        img.save(out_img_path)?;
        println!("Draw the image success! Save to {}", out_img_path);
    }
    Ok(())
}
