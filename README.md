# Cloud DeepCell - Scaling Image Analysis

This working Repo contains our notes / utilities / info for our cloud [DeepCell](https://github.com/vanvalenlab/deepcell-tf) imaging project. 

Here is the high level workflow for using DeepCell:

![high level workflow](images/deepcell-imaging-highlevel.png)<sub><a href="https://lucid.app/lucidchart/67c3f550-b2aa-4194-b527-56e3592829a3/edit?viewport_loc=-310%2C-595%2C3416%2C1848%2C0_0&invitationId=inv_447a9b8a-7711-43cf-a91f-e978075fc132">lucidchart source</a></sub>

Note that DeepCell itself does not process TIFF files. The TIFF channels must be extracted into Numpy arrays first.

Also note that DeepCell performs its own pre- and post-processing around the TensorFlow prediction. In particular, DeepCell divides the input into 512x512 tiles which it predicts in batches, then reconstructs the overall image.

![tiling process](images/tiling-process.png)

## Goal and Key Links

- **GOAL: Understand and optimize using DeepCell to perform cellular image segmentation on GCP at scale.**
  - KEY LINK #1: our [benchmarking process](benchmarking/deepcell-e2e).
  - KEY LINK #2: our support/testing [notebooks](notebooks).
  - KEY LINK #3: our [project board](https://github.com/users/dchaley/projects/1) & work areas for this project.

## Findings

GPU makes a dramatic difference in model inference time.

![Pixels vs inference time](images/pixels-vs-inference-time.png)

![Pixels vs inference time](images/pixels-vs-postprocess-time.png)

Memory usage increases linearly with number of pixels.

![Pixels vs mem usage](images/pixels-vs-mem-usage.png)

## Optimization opportunities

Here are some areas we've identified:

- Preprocessing
  - DeepCell converts everything to 64bit float. That's memory intensive. Do we actually need to?
- Prediction
  - Benchmark results (GPU, batch size, resolution)
  - Investigate [Google TensorFlow optimizations](https://cloud.google.com/vertex-ai/docs/predictions/optimized-tensorflow-runtime)
- Postprocessing
  - h_maxima: need to ship a [~15x speedup optimization](https://github.com/dchaley/deepcell-imaging/tree/main/benchmarking/h_maxima)

# Local development

## Mac OS x86_64

Nothing special. You just need Python 3.10 at the latest.

```
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Mac OS arm64

Some incantations are needed to work on Apple silicon computers. You also need Python 3.9.

```
python3.9 -m venv venv
source venv/bin/activate
pip install -r requirements-mac-arm64.txt
pip install -r requirements.txt
# Let it fail to install DeepCell, then:
pip install -r requirements.txt --no-deps
```

The issue is that DeepCell depends on `tensorflow` not `tensorflow-macos` which we need for the 2.8 version on arm64 chips.

