# Cloud DeepCell - Scaling Image Analysis

This working Repo contains our notes / utilities / info for our cloud [DeepCell](https://github.com/vanvalenlab/deepcell-tf) imaging project. 

Here is the high level workflow for using DeepCell:

<img src="https://github.com/dchaley/deepcell-imaging/blob/main/images/deepcell-imaging-highlevel.png" width=1000>

Note that DeepCell itself does not process TIFF files. The TIFF channels must be extracted into Numpy arrays first.

Also note that DeepCell performs its own pre- and post-processing around the TensorFlow prediction.

## Goal and Key Links

- **GOAL: Understand and optimize using DeepCell to perform cellular image segmentation on GCP at scale.**
  - KEY LINK #1: our [benchmarking process](benchmarking/deepcell-e2e).
  - KEY LINK #2: our support/testing [notebooks](notebooks).
  - KEY LINK #3: our [project board](https://github.com/users/dchaley/projects/1) & work areas for this project.

## Findings

GPU makes a dramatic difference in model inference time.

![Pixels vs inference time](images/pixels-vs-inference-time.png)

![Pixels vs inference time](images/pixels-vs-postprocess-time.png)

## Optimization opportunities

Here are some areas we've identified:

- Preprocessing
  - DeepCell converts everything to 64bit float. That's memory intensive. Do we actually need to?
- Prediction
  - Benchmark results (GPU, batch size, resolution)
  - Investigate [Google TensorFlow optimizations](https://cloud.google.com/vertex-ai/docs/predictions/optimized-tensorflow-runtime)
- Postprocessing
  - h_maxima: need to ship a [~15x speedup optimization](https://github.com/dchaley/deepcell-imaging/tree/main/benchmarking/h_maxima)

