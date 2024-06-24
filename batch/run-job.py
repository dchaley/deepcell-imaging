#!/usr/bin/env python

import argparse
import subprocess
import tempfile
import uuid

parser = argparse.ArgumentParser("deepcell-on-batch")
parser.add_argument(
    "--input_channels_path",
    help="Path to the input channels npz file",
    type=str,
    required=True,
)
parser.add_argument(
    "--model_path",
    help="Path to the model archive",
    type=str,
    required=False,
    default="gs://davids-genomics-data-public/cellular-segmentation/deep-cell/vanvalenlab-tf-model-multiplex-downloaded-20230706/MultiplexSegmentation.tar.gz",
)

args = parser.parse_args()

job_id = "j" + str(uuid.uuid4())
input_channels_path = args.input_channels_path

model_path = args.model_path

CONTAINER_IMAGE = "us-central1-docker.pkg.dev/deepcell-on-batch/deepcell-benchmarking-us-central1/benchmarking:gce"
OUTPUT_BASE_PATH = "gs://deepcell-batch-jobs_us-central1/job-runs"
REGION = "us-central1"

output_path = "{}/{}".format(OUTPUT_BASE_PATH, job_id)

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
                            "commands": [
                                "--input_channels_path={input_channels_path}",
                                "--output_path={output_path}",
                                "--model_path={model_path}",
                                "--visualize_input",
                                "--visualize_predictions",
                                "--provisioning_model=spot"
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
    output_path=output_path,
    input_channels_path=input_channels_path,
    container_image=CONTAINER_IMAGE,
    region=REGION,
    model_path=model_path,
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
