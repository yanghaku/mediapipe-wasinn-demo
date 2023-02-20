<div align="center">
  <h1><code>mediapipe-wasinn-demo</code></h1>
  <h2><strong>A Rust library crate for mediapipe models for WasmEdge NN</strong></h2>
  <p>( This project is a demo for <a href="https://github.com/WasmEdge/WasmEdge/issues/2229"> Issue 2229</a> and <a href="https://github.com/WasmEdge/WasmEdge/discussions/2230">Discussion 2230</a> in <a href="https://github.com/WasmEdge/WasmEdge">WasmEdge</a> )</p>
  <p>
    <a href="https://github.com/yanghaku/mediapipe-wasinn-demo/actions?query=workflow%3ACI">
      <img src="https://github.com/yanghaku/mediapipe-wasinn-demo/workflows/CI/badge.svg" alt="CI status"/>
    </a>
  </p>
</div>

## Introduction

This project is a rust library for developers to easily use the mediapipe modules. The crate contains a set of functions
for each model in Mediapipe, and many useful functions to do preprocess and postprocess.

When doing inference, this crate can use [wasi-nn] and run in [WasmEdge] with tflite backend.

## How to Use This Crate

### 1. Add dependency in your ```Cargo.toml```

```toml
[dependencies]
mediapipe-wasinn-demo = "0.1.0-dev"
```

### 2. Use mediapipe-wasinn-demo api in your source code

This is an example, you can find full code in [./examples/face_detection.rs](./examples/face_detection.rs):

```rust
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
    let results = face_detection.process(&img)?;

    // print the results
    println!("The number of results: {}", results.len());
    for r in results {
        println!("{}", r);
    }

    // ......

    Ok(())
}
```

## Requirements

### Run Environment

* WasmEdge with wasi-nn plugin (tflite backend) and tflite C library.

  You can install these things just use the scripts ```./scripts/wasmedge-init.sh```
  ```shell
  ./scripts/wasmedge-init.sh
  ```

### Develop Environment

* Rustup with ```wasm32-wasi``` target installed.

  You can install rust manually or use the scripts in ```./scripts/rust-init.sh```
  ```shell
  ./scripts/rust-init.sh
  ```

## Test Results

* Test Task: face detection, use the model ```face_detection_short_range.tflite```, you can
  click [here](https://mediapipe.page.link/blazeface-mc) to get more information about this model.

* Test Input: ```./assets/test.jpg```

  ![](./assets/test.jpg)

* This crate example's results (with no nms now):

```console
yb$ cargo run --release --example face_detection -- ./assets/test.jpg ./assets/test_result.jpg
    Finished release [optimized] target(s) in 0.01s
     Running `/home/yb/code/tests/mediapipe-wasinn-demo/./scripts/wasmedge-runner.sh target/wasm32-wasi/release/examples/face_detection.wasm ./assets/test.jpg ./assets/test_result.jpg`
The number of results: 2
FaceDetectionModelOutput: {
	Score: 0.9446747
	Face bound box: { xmin: 0.24396126, ymin: 0.3258748, width: 0.47893, height: 0.47885394 }
	Left eye: (0.6110951, 0.45973283)
	Right eye: (0.4164656, 0.47427702)
	Nose tip: (0.5491422, 0.59022015)
	Mouse center: (0.54484063, 0.68035823)
	Left eye tragion: (0.2716728, 0.514828)
	Right eye tragion: (0.6790835, 0.48192626)
}
FaceDetectionModelOutput: {
	Score: 0.9378273
	Face bound box: { xmin: 0.2515205, ymin: 0.33453536, width: 0.4625408, height: 0.46251416 }
	Left eye: (0.6041109, 0.4593312)
	Right eye: (0.41474357, 0.47974306)
	Nose tip: (0.5432431, 0.5890332)
	Mouse center: (0.5382308, 0.67589974)
	Left eye tragion: (0.27871802, 0.5189266)
	Right eye tragion: (0.6761436, 0.4834574)
}
Draw the image success! Save to ./assets/test_result.jpg
```

And this is the image we draw: ![](./assets/test_result.jpg)

* The result of ```mediapipe-python``` (*for comparison*):

```console
$ python3
Python 3.8.15 (default, Nov  4 2022, 20:59:55)
[GCC 11.2.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import cv2
>>> import mediapipe as mp
>>> res=mp.solutions.face_detection.FaceDetection(model_selection=0,min_detection_confidence=0.9).process(cv2.cvtColor(cv2.imread("./assets/test.jpg"),cv2.COLOR_BGR2RGB))
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
>>> res.detections
[label_id: 0
score: 0.944340705871582
location_data {
  format: RELATIVE_BOUNDING_BOX
  relative_bounding_box {
    xmin: 0.24781852960586548
    ymin: 0.32510554790496826
    width: 0.47579383850097656
    height: 0.47572338581085205
  }
  relative_keypoints {
    x: 0.4171816110610962
    y: 0.4769299328327179
  }
  relative_keypoints {
    x: 0.6098904013633728
    y: 0.45966291427612305
  }
  relative_keypoints {
    x: 0.5487101078033447
    y: 0.5954264402389526
  }
  relative_keypoints {
    x: 0.5441946983337402
    y: 0.6810187697410583
  }
  relative_keypoints {
    x: 0.27675142884254456
    y: 0.5163325667381287
  }
  relative_keypoints {
    x: 0.6797348856925964
    y: 0.480962872505188
  }
}
]
>>> 
```

Compare with the standard output from mediapipe-python, we can see that the result is correct.

## Related Links

- [WasmEdge]
- [wasi-nn]

[wasi-nn]: https://github.com/bytecodealliance/wasi-nn

[WasmEdge]: https://github.com/WasmEdge/WasmEdge

## License

This project is licensed under the Apache 2.0 license. See [LICENSE] for more details.

[LICENSE]: LICENSE
