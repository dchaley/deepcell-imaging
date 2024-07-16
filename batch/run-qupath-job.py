#!/usr/bin/env python

import argparse
import datetime
import subprocess
import tempfile
import uuid

from deepcell_imaging.numpy_utils import npz_headers

CONTAINER_IMAGE = "us-central1-docker.pkg.dev/deepcell-on-batch/deepcell-benchmarking-us-central1/benchmarking:batch"
REGION = "us-central1"

parser = argparse.ArgumentParser("deepcell-on-batch")
parser.add_argument(
    "--dataset",
    help="Path to the dataset root directory (containing subfolders OMETIFF, SEGMASK, etc)",
    type=str,
    required=True,
)
parser.add_argument(
    "--prefix",
    help="Input prefix, eg ROI01. Input will be: <dataset>/NPZ_INTERMEDIATE/<prefix>.npz",
)
parser.add_argument(
    "--compartment",
    help="Which predictions to generate: whole-cell (default), nuclear, or both",
    type=str,
    choices=["whole-cell", "nuclear", "both"],
    default="whole-cell",
)
parser.add_argument(
    "--bigquery_benchmarking_table",
    help="BigQuery table to write benchmarking results to (empty for none)",
    type=str,
    required=False,
    default="deepcell-on-batch.benchmarking.results_batch",
)
parser.add_argument(
    "--model_path",
    help="Path to the model archive",
    type=str,
    required=False,
    default="gs://genomics-data-public-central1/cellular-segmentation/vanvalenlab/deep-cell/vanvalenlab-tf-model-multiplex-downloaded-20230706/MultiplexSegmentation-resaved-20240710.h5",
)
parser.add_argument(
    "--model_hash",
    help="Hash of the model archive",
    type=str,
    required=False,
    default="56b0f246081fe6b730ca74eab8a37d60",
)

args = parser.parse_args()

dataset = args.dataset.rstrip("/")
prefix = args.prefix
compartment = args.compartment

input_channels_path = "{}/NPZ_INTERMEDIATE/{}.npz".format(dataset, prefix)
job_intermediate_path = "{}/jobs/{}_{}".format(
    dataset, prefix, datetime.datetime.now().isoformat()
)

bigquery_benchmarking_table = args.bigquery_benchmarking_table

tiff_output_uri = "{}/SEGMASK/{}_{}.tiff".format(dataset, prefix, compartment)

# For now, assume there's only one file in the input.
input_file_contents = list(npz_headers(input_channels_path))
if len(input_file_contents) != 1:
    raise ValueError("Expected exactly one array in the input file")
input_image_shape = input_file_contents[0][1]

# A lot of stuff gets hardcoded in this json.
# See the README for limitations.

# Need to escape the curly braces in the JSON template
base_json = """
{{
  "taskGroups": [
    {{
      "taskSpec": {{
        "runnables": [
          {{
            "container": {{
              "imageUri": "{container_image}",
              "entrypoint": "python",
              "commands": [
                "scripts/preprocess.py",
                "--image_uri={input_channels_path}",
                "--benchmark_output_uri={job_intermediate_path}/preprocess_benchmark.json",
                "--output_uri={job_intermediate_path}/preprocessed.npz.gz"
              ]
            }}
          }},
          {{
            "container": {{
              "imageUri": "{container_image}",
              "entrypoint": "python",
              "commands": [
                "scripts/predict.py",
                "--image_uri={job_intermediate_path}/preprocessed.npz.gz",
                "--benchmark_output_uri={job_intermediate_path}/prediction_benchmark.json",
                "--model_path={model_path}",
                "--model_hash={model_hash}",
                "--output_uri={job_intermediate_path}/raw_predictions.npz.gz"
              ]
            }}
          }},
          {{
            "container": {{
              "imageUri": "{container_image}",
              "entrypoint": "python",
              "commands": [
                "scripts/postprocess.py",
                "--raw_predictions_uri={job_intermediate_path}/raw_predictions.npz.gz",
                "--input_rows={input_image_rows}",
                "--input_cols={input_image_cols}",
                "--benchmark_output_uri={job_intermediate_path}/postprocess_benchmark.json",
                "--output_uri={job_intermediate_path}/predictions.npz.gz",
                "--tiff_output_uri={tiff_output_uri}"
              ]
            }}
          }},
          {{
            "container": {{
              "imageUri": "{container_image}",
              "entrypoint": "python",
              "commands": [
                "scripts/gather-benchmark.py",
                "--preprocess_benchmarking_uri={job_intermediate_path}/preprocess_benchmark.json",
                "--prediction_benchmarking_uri={job_intermediate_path}/prediction_benchmark.json",
                "--postprocess_benchmarking_uri={job_intermediate_path}/postprocess_benchmark.json",
                "--bigquery_benchmarking_table={bigquery_benchmarking_table}"
              ]
            }}
          }},
          {{
            "container": {{
              "imageUri": "{container_image}",
              "entrypoint": "python",
              "commands": [
                "scripts/visualize.py",
                "--image_uri={input_channels_path}",
                "--predictions_uri={job_intermediate_path}/predictions.npz.gz",
                "--visualized_input_uri={job_intermediate_path}/visualized_input.png",
                "--visualized_predictions_uri={job_intermediate_path}/visualized_predictions.png"
              ]
            }}
          }}
        ],
        "computeResource": {{
          "memoryMib": 26000
        }},
        "maxRetryCount": 3,
        "lifecyclePolicies": [
          {{
            "action": "RETRY_TASK",
            "actionCondition": {{
              "exitCodes": [50001]
            }}
          }}
        ]
      }},
      "taskCount": 1,
      "parallelism": 1
    }}
  ],
  "allocationPolicy": {{
    "instances": [
      {{
        "installGpuDrivers": true,
        "policy": {{
          "machineType": "n1-standard-8",
          "provisioningModel": "SPOT",
          "accelerators": [
            {{
              "type": "nvidia-tesla-t4",
              "count": 1
            }}
          ]
        }}
      }}
    ],
    "location": {{
      "allowedLocations": [
        "regions/{region}"
      ]
    }}
  }},
  "logsPolicy": {{
    "destination": "CLOUD_LOGGING"
  }}
}}
"""

job_json_str = base_json.format(
    job_intermediate_path=job_intermediate_path,
    tiff_output_uri=tiff_output_uri,
    bigquery_benchmarking_table=bigquery_benchmarking_table,
    model_path=args.model_path,
    model_hash=args.model_hash,
    input_channels_path=input_channels_path,
    container_image=CONTAINER_IMAGE,
    region=REGION,
    input_image_rows=input_image_shape[0],
    input_image_cols=input_image_shape[1],
)

job_json_file = tempfile.NamedTemporaryFile()
with open(job_json_file.name, "w") as f:
    f.write(job_json_str)

# The batch job id must be unique, and can only contain lowercase letters,
# numbers, and hyphens. It must also be 63 characters or fewer.
# We're doing 62 to be safe.
#
# Regex: ^[a-z]([a-z0-9-]{0,61}[a-z0-9])?$
batch_job_id = "deepcell-{}".format(str(uuid.uuid4()))
batch_job_id = batch_job_id[0:62].lower()

cmd = "gcloud batch jobs submit {job_id} --location {location} --config {job_filename}".format(
    job_id=batch_job_id, location=REGION, job_filename=job_json_file.name
)
subprocess.run(cmd, shell=True)

print("Job submitted with ID: {}".format(batch_job_id))
print("Intermediate output: {}".format(job_intermediate_path))
print("Final output: {}".format(tiff_output_uri))
