# Cloud DeepCell – Scaling Image Analysis

This repository contains our tools & research for running [DeepCell](https://github.com/vanvalenlab/deepcell-tf) segmentation and [QuPath](https://qupath.github.io/) measurements on Google Cloud Batch.

Our results show an overall improvement from ~13 hours to ~10 minutes for segmenting & measuring a cell. The starting point was running on a laptop or colo machine, and our work ran on GCP Batch with some cloud-focused enhancements.

The workflow operates on one or more input images, converted from image to `numpy` pixel array. Then DeepCell preprocesses the data (denoising & normalization), runs the segmentation prediction, and postprocesses the predictions into a cell mask. Then, we load the image and mask into QuPath to compute quantitative metrics (size, channel intensity, etc.) for further analysis. For an example downstream usage, see [SpaFlow](https://github.com/dimi-lab/SpaFlow/) (cell clustering & quantification).

Here is the workflow diagram:

![High level workflow](images/deepcell-qupath-overall-process.png)<sub><a href="https://lucid.app/lucidchart/67c3f550-b2aa-4194-b527-56e3592829a3/edit?viewport_loc=-310%2C-595%2C3416%2C1848%2C0_0&invitationId=inv_447a9b8a-7711-43cf-a91f-e978075fc132">lucidchart source</a></sub>

# Running segmentation

## Configuration

You'll need a JSON file available in a cloud bucket, configuring the application environment. Create a file something like this:

```json
{
  "segment_container_image": "$REPOSITORY/benchmarking:batch",
  "quantify_container_image": "$REPOSITORY/qupath-project-initializer:latest",
  "bigquery_benchmarking_table": "$PROJECT.$DATASET.$TABLE",
  "region": "$REGION",
  "networking_interface": {
    "network": "the_network",
    "subnetwork": "the_subnetwork",
    "no_external_ip_address": true
  },
  "service_account": {
    "email": "the_service@account.com"
  }
}
```

You'll need to replace the variables with your environment.

- You can use the public Docker Hub containers, or copy them to your own artifact repository. 
- For the benchmarking, you need to create a dataset & table in a GCP project; or you can omit it or set it to blank to skip collecting benchmarks. The table must be created with the schema specified in [this file](benchmarking/bigquery_benchmarking_schema.json).
- Lastly, specify the GCP region where compute resources will be provisioned. This is not the same as storage buckets, but consider making it the same for efficiency & egress cost reduction.
- The `networking_interface` and `service_account` sections are optional if you want to use default settings.

For example, using the Docker Hub containers & skipping benchmarking & default networking + service account:

```json
{
  "segment_container_image": "dchaley/deepcell-imaging:batch",
  "quantify_container_image": "dchaley/qupath-project-initializer:latest",
  "region": "us-central1"
}
```

Upload this file somewhere to GCP storage. We put ours in the root of our working bucket. You'll pass this GS URI as a parameter to the scripts.

## Run segmentation & measurement

To run DeepCell on input images then compute QuPath measurements, use the helper `scripts/segment-and-measure.py`. There are two ways to run this script: (1) running on a QuPath workspace, and (2) running on explicit paths.

1. QuPath workspace: 

   1. Many QuPath projects are organized something like this:

      ```
      📁 Dataset
      ↳ 📁 OMETIFF
        ↳ 🖼️ SomeTissueSample.ome.tiff
        ↳ 🖼️ AnotherTissueSample.ome.tiff
      ↳ 📁 NPZ_INTERMEDIATE
        ↳ 🔢 SomeTissueSample.npz
        ↳ 🔢 AnotherTissueSample.npz
      ↳ 📁 SEGMASK
        ↳ 🔢 SomeTissueSample_WholeCellMask.tiff
        ↳ 🔢 SomeTissueSample_NucleusMask.tiff
        ↳ 🔢 AnotherTissueSample_WholeCellMask.tiff
        ↳ 🔢 AnotherTissueSample_NucleusMask.tiff
      ↳ 📁 REPORTS
        ↳ 📄 SomeTissueSample_QUANT.tsv
        ↳ 📄 AnotherTissueSample_QUANT.tsv
      ↳ 📁 PROJ
        ↳ 📁 data
          ↳ ...
        ↳ 📄 project.qpproj
      ```

      To generate segmentation masks & quantification reports, run the following command:

      ```shell
      scripts/segment-and-measure.py
        --env_config_uri gs://bucket/path/to/env-config.json
        workspace gs://bucket/path/to/dataset
      ```

      This will enumerate all files in the `OMETIFF` directory that have matching files in `NPZ_INTERMEDIATE`, and run DeepCell segmentation to generate the `SEGMASK` numpy files. Then it will run QuPath measurements to generate the `REPORTS` files.

      If your folder structure is different (for example `OME-TIFF` instead of `OMETIFF`) you can use these parameters to specify the workspace subdirectories: `--images_subdir`, `--npz_subdir`, `--segmasks_subdir`, `--project_subdir`, `--reports_subdir`. Put these parameters *after* the `workspace` command.

2. Explicit paths.

   1. You can also specify all paths explicitly (the files don't have to be organized in a dataset). To do so, run this command:

      ```shell
      scripts/segment-and-measure.py
        --env_config_uri gs://bucket/path/to/env-config.json
        paths
        --images_path gs://bucket/path/to/ometiffs
        --numpy_path gs://bucket/path/to/npzs
        --segmasks_path gs://bucket/path/to/segmasks
        --project_path gs://bucket/path/to/project
        --reports_path gs://bucket/path/to/reports
      ```

In either case, when you download the QuPath project, you'll need to download the OMETIFF files as well. When you open the project it will prompt you to select the base directory containing the OMETIFFs, and from there should automatically remap the image paths.

#### Working on a subset of images

You can use the parameter `--image_filter` to only operate on a subset of the OMETIFFs. For example,

```shell
scripts/segment-and-measure.py
  --env_config_uri gs://.../config.json
  --image_filter SomeTissue
  workspace gs://path/to/workspace
```

This will operate on every file whose name begins with the string `SomeTissue`. This would match `SomeTissueSample`, `SomeTissueImage`, etc. Note that this parameter has to come before the `workspace` or `paths` parameter.

# DeepCell deep dive

## Input data

DeepCell does not process TIFF files. The TIFF channels must be extracted into Numpy arrays first.

## Tiled prediction

DeepCell divides the preprocessed input into 512x512 tiles which it predicts in batches, then recombines into a single image for postprocessing.

![tiling process](images/tiling-process.png)

This makes the prediction very resource-efficient, note however that pre- and post-processing still operate on the entire image. This is particularly problematic for post-processing which is very resource-intensive.

## Post-processing

The prediction step outputs which pixels are most likely to be the center of their cell. The post-processing step runs image analysis algorithms to create the final cell masks. It operates a bit like a "flood fill" to expand the center out.

This uses the h_maxima grayscale reconstruction algorithm, which is (counterintuitively) far slower than prediction itself for large images.

# QuPath deep-dive

Once we have cell predictions, we need to generate quantified metrics for the cells: location, size, channel intensities, and so on. This is crucial for downstream processing & analysis, including in a QuPath desktop environment. For example, a researcher might provide an analyzed & packaged QuPath project to a principal investigator for review.

QuPath is distributed as JAR files. Bioinformaticians typically run Groovy scripts in the embedded QuPath environment, however we don't have a desktop or VM environment for that. Instead we compile Kotlin code with the JARs to run on Google Batch.

The source code for quantifying the metrics plus building the container is located in a different repository: [qupath-project-initializer](https://github.com/dchaley/qupath-project-initializer).

## Image pre-fetching

QuPath measurements are computed a cell at a time. The algorithm re-fetches the image region containing the cell for each cell. This is prohibitively expensive for bulk measurement.

Adding code to prefetch the image into memory, then retrieve subregions from memory, provided a dramatic ~99% speed-up.

# Benchmarking

## Goal and Key Links

- **GOAL: Understand and optimize DeepCell cellular segmentation on GCP at scale.**
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
- Postprocessing
  - h_maxima: need to ship a [~15x speedup optimization](https://github.com/dchaley/deepcell-imaging/tree/main/benchmarking/h_maxima)
- Cost
  - Run the prediction phase only with GPU infrastructure. Run everything else with CPU-only infrastructure.

# Development

## Source code checkout

This repo uses [git-lfs](https://git-lfs.com/) (Git Large File System) to exclude large files (like sample numpy data) in the source history. This process is automatic & transparent, but requires `git-lfs` to be installed beforehand. Please see [these instructions](https://github.com/git-lfs/git-lfs?tab=readme-ov-file#installing).

TLDR,

- on Mac, `brew install git-lfs`.
- on Linux, `sudo [apt-get | yum] install git-lfs`.
- on Windows, `git-lfs` is included in the Git distribution.

## Local development

### Mac OS x86_64

Nothing special. You just need Python 3.10 at the latest.

```
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Mac OS arm64

Some incantations are needed to work on Apple silicon computers. You also need Python 3.9.

DeepCell depends on `tensorflow`, not `tensorflow-macos`. Unfortunately we need `tensorflow-macos` specifically to provide TF2.8 on arm64 chips.

The solution is to install the packages one at a time so that the DeepCell failure doesn't impact the other packages.

```
python3.9 -m venv venv
source venv/bin/activate
pip install -r requirements-mac-arm64.txt
cat requirements.txt | xargs -n 1 pip install

# Let it fail to install DeepCell, then:
pip install -r requirements.txt --no-deps

# Lastly install our own library. Note --no-deps
pip install --editable . --no-deps
```

I think but am not sure that the first `--no-deps` invocation is unnecessary as `pip install` installs dependencies.

