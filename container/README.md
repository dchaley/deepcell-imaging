# DeepCell benchmarking container

## Building the container

This sets up a container that executes the steps in the benchmark setup notebook. It checks out the repo so in principle we could run the notebook however we haven't tried that yet.

Followed [Google instructions on derivative containers](https://cloud.google.com/deep-learning-containers/docs/derivative-container).

To build/push the container, run: `build-and-push-gce-container.sh`. Make sure to update the environment variables as necessary.

To create the repository, run:

```
gcloud artifacts repositories create $REPOSITORY_NAME \
    --repository-format=docker \
    --location=$LOCATION
gcloud auth configure-docker ${LOCATION}-docker.pkg.dev
```

## Command to run benchmark job on Google Batch

1. Set up an environment.

   1. The easiest is to just open Cloud Shell in the project.

      Go to console.cloud.google.com, select the project, & click the terminal icon. You can upload files to cloud shell under the "More" menu (three dots).

   2. You can also install `gcloud` to your local computer, run `gcloud auth`, etc.

2. Create a JSON file specifying the inputs. See also: [batch.json](batch.json)

   1. Job parameters

      1. Update the input parameter: `input_channels_path`

         This should be a .npz file like the ones we have here: `davids-genomics-data-public/cellular-segmentation/10x-genomics/preview-human-breast-20221103-418mb/input_channels.npz`

      2. Update the `output_path`

         This should be a unique base directory in a cloud bucket somewhere. Example: `gs://deepcell-batch-jobs_us-central1/job-runs/19991231-default-path`

         Remove the visualize flags as desired.

         Files will be written as: `predictions.npz`, `input.png`, `predictions.png`

      3. If you are not using spot instances (the default), update the `provisioning_model` to `standard` (etc) as appropriate.

   2. Batch infra settings

      1. Update memory settings based on number of pixels.
         1. Refer to our [Looker chart](https://lookerstudio.google.com/u/0/reporting/cc9595d1-e639-4b35-a144-0bb8a41df2d0/page/p_rr3yyoz8cd/edit) for memory requirements by pixel count.
         2. Set the `machineType` as appropriate for the number of pixels in the input. 
         3. Set the `memoryMib` as appropriate. Default: `26000` = 25.39 GB.
      2. Change the `accelerators` if desired. Default: 1 Tesla T4 GPU.
      3. Change the `provisioningModel` if desired. Default: `SPOT`.

3. You now have a json file to submit to the Batch API.

   1. Run:

      ```bash
      gcloud batch jobs submit $JOB_ID --location us-central1 --config batch.json
      ```

      Pick some unique value for $JOB_ID. It must start with a letter, and contain only dashes, numbers, and lowercase letters.

4. You can watch the batch job run here: https://console.cloud.google.com/batch/jobs?project=deepcell-on-batch
