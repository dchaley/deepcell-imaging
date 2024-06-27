#!/usr/bin/env python

import argparse
import subprocess
import tempfile
import uuid

from deepcell_imaging.numpy_utils import npz_headers

parser = argparse.ArgumentParser("deepcell-on-batch")
parser.add_argument(
    "--input_channels_path",
    help="Path to the input channels npz file",
    type=str,
    required=True,
)

args = parser.parse_args()

job_id = "j" + str(uuid.uuid4())
input_channels_path = args.input_channels_path

CONTAINER_IMAGE = "us-central1-docker.pkg.dev/deepcell-on-batch/deepcell-benchmarking-us-central1/benchmarking:gce"
OUTPUT_BASE_PATH = "gs://deepcell-batch-jobs_us-central1/job-runs"
REGION = "us-central1"

output_path = "{}/{}".format(OUTPUT_BASE_PATH, job_id)

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
                "--benchmark_output_uri={output_path}/preprocess_benchmark.json",
                "--output_uri={output_path}/preprocessed.npz.gz"
              ]
            }}
          }},
          {{
            "container": {{
              "imageUri": "{container_image}",
              "entrypoint": "python",
              "commands": [
                "scripts/predict.py",
                "--image_uri={output_path}/preprocessed.npz.gz",
                "--benchmark_output_uri={output_path}/prediction_benchmark.json",
                "--output_uri={output_path}/raw_predictions.npz.gz"
              ]
            }}
          }},
          {{
            "container": {{
              "imageUri": "{container_image}",
              "entrypoint": "python",
              "commands": [
                "scripts/postprocess.py",
                "--raw_predictions_uri={output_path}/raw_predictions.npz.gz",
                "--input_rows={input_image_rows}",
                "--input_cols={input_image_cols}",
                "--benchmark_output_uri={output_path}/postprocess_benchmark.json",
                "--output_uri={output_path}/predictions.npz.gz"
              ]
            }}
          }},
          {{
            "container": {{
              "imageUri": "{container_image}",
              "entrypoint": "python",
              "commands": [
                "scripts/gather-benchmark.py",
                "--preprocess_benchmarking_uri={output_path}/preprocess_benchmark.json",
                "--prediction_benchmarking_uri={output_path}/prediction_benchmark.json",
                "--postprocess_benchmarking_uri={output_path}/postprocess_benchmark.json",
                "--bigquery_benchmarking_table=deepcell-on-batch.benchmarking.results_batch"
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
                "--predictions_uri={output_path}/predictions.npz.gz",
                "--visualized_input_uri={output_path}/visualized_input.png",
                "--visualized_predictions_uri={output_path}/visualized_predictions.png"
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

#    "labels": {{
#        "goog-batch-job-group": "my-test-pool",
#        "goog-batch-node-pool-max-idle-time": "20min",
#        "goog-batch-node-pool-max-size": "4"
#    }},

job_json_str = base_json.format(
    output_path=output_path,
    input_channels_path=input_channels_path,
    container_image=CONTAINER_IMAGE,
    region=REGION,
    input_image_rows=input_image_shape[0],
    input_image_cols=input_image_shape[1],
)

job_json_file = tempfile.NamedTemporaryFile()
with open(job_json_file.name, "w") as f:
    f.write(job_json_str)

cmd = "gcloud batch jobs submit {job_id} --location {location} --config {job_filename}".format(
    job_id=job_id, location=REGION, job_filename=job_json_file.name
)
subprocess.run(cmd, shell=True)

print("Job submitted with ID: {}".format(job_id))
print("Output: {}".format(output_path))
