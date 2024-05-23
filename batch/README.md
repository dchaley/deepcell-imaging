# Running DeepCell batch jobs

This directory contains tools to submit DeepCell batch jobs.

## Usage

1. Set up an environment.

   1. The easiest is to just open Cloud Shell in the project.

      Go to console.cloud.google.com, select the project, & click the terminal icon. You can upload files to cloud shell under the "More" menu (three dots).

   2. You can also install `gcloud` to your local computer, run `gcloud auth`, etc.

2. Submit the job:

   1. Pick an input file. This should be a .npz file like the ones we have here: `gs://davids-genomics-data-public/cellular-segmentation/10x-genomics/preview-human-breast-20221103-418mb/input_channels.npz`

   2. From the repo base directory, run the script.
      ```sh
      batch/run-job.py --input_channels_path="$input_npz_file"
      ```

   3. The script will output the job ID plus the output directory.

3. Go to the Batch app in GCP console and wait for termination.

   1. There's no feedback yet! See: https://github.com/dchaley/deepcell-imaging/issues/217

## Todo

This helper script has a few drawbacks compared to manually crafting a job json file as described in [the `container` directory](../container)

* Always visualizes input + predictions.
* Can't customize instance type.
* Can't customize GPU. (Always uses 1 T4)
* Can't customize parallelism, vCPUs, etc.