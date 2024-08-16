"""
This module contains functions for creating and submitting batch jobs to GCP.
"""

import json
import os

import smart_open

# A lot of stuff gets hardcoded in this json.
# See the README for limitations.

# Note: Need to escape the curly braces in the JSON template
BASE_MULTISTEP_TEMPLATE = """
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
                                "--compartment={compartment}",
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


def make_job_json(
    region: str,
    container_image: str,
    model_path: str,
    model_hash: str,
    bigquery_benchmarking_table: str,
    input_channels_path: str,
    compartment: str,
    working_directory: str,
    tiff_output_uri: str,
    input_image_rows: int,
    input_image_cols: int,
    config: dict = None,
) -> dict:
    json_str = BASE_MULTISTEP_TEMPLATE.format(
        container_image=container_image,
        model_path=model_path,
        model_hash=model_hash,
        input_channels_path=input_channels_path,
        compartment=compartment,
        job_intermediate_path=working_directory,
        tiff_output_uri=tiff_output_uri,
        input_image_rows=input_image_rows,
        input_image_cols=input_image_cols,
        region=region,
        bigquery_benchmarking_table=bigquery_benchmarking_table,
    )

    job_json = json.loads(json_str)

    if config:
        job_json.update(config)

    return job_json


def get_batch_indexed_task(tasks_spec_uri, args_cls):
    with smart_open.open(tasks_spec_uri, "r") as tasks_spec_file:
        tasks_spec = json.load(tasks_spec_file)

    task_index = int(os.environ["BATCH_TASK_INDEX"])
    task = tasks_spec[task_index]

    return args_cls(**task)


def apply_cloud_logs_policy(job: dict) -> None:
    """
    Return a copy of the job definition,
    with the logs policy set to cloud logging.
    """
    job["logsPolicy"] = {"destination": "CLOUD_LOGGING"}
