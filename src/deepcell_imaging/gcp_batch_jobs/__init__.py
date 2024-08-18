"""
This module contains functions for creating and submitting batch jobs to GCP.
"""

import json
import os
import subprocess
import tempfile

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
    ]
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

    apply_allocation_policy(
        job_json,
        region,
        "n1-standard-8",
        "SPOT",
        gpu_type="nvidia-tesla-t4",
        gpu_count=1,
    )
    apply_cloud_logs_policy(job_json)

    return job_json


def get_batch_indexed_task(tasks_spec_uri, args_cls):
    with smart_open.open(tasks_spec_uri, "r") as tasks_spec_file:
        tasks_spec = json.load(tasks_spec_file)

    task_index = int(os.environ["BATCH_TASK_INDEX"])
    task = tasks_spec[task_index]

    return args_cls(**task)


def apply_allocation_policy(
    job: dict,
    region: str,
    machine_type: str,
    provisioning_model: str,
    gpu_type: str = None,
    gpu_count: int = None,
) -> None:
    """
    Apply an allocation policy to the job definition: machine type, provisioning model, and GPU.
    """
    if gpu_type and not gpu_count or gpu_count and not gpu_type:
        raise ValueError("GPU type and GPU count must be set together")
    if provisioning_model not in ["SPOT", "STANDARD"]:
        raise ValueError("Provisioning model must be either SPOT or STANDARD")

    job["allocationPolicy"] = {
        "instances": [
            {
                "policy": {
                    "machineType": machine_type,
                    "provisioningModel": provisioning_model,
                },
            }
        ],
        "location": {"allowedLocations": [f"regions/{region}"]},
    }

    if gpu_type:
        job["allocationPolicy"]["instances"][0]["installGpuDrivers"] = True
        job["allocationPolicy"]["instances"][0]["policy"]["accelerators"] = [
            {"type": gpu_type, "count": gpu_count}
        ]


def apply_cloud_logs_policy(job: dict) -> None:
    """
    Apply a cloud logging policy to the job definition.
    """
    job["logsPolicy"] = {"destination": "CLOUD_LOGGING"}


def submit_job(job: dict, job_id: str, region: str) -> None:
    """
    Submit a job to the Batch service.
    """
    with tempfile.NamedTemporaryFile() as job_json_file:
        with open(job_json_file.name, "w") as f:
            json.dump(job, f)

        cmd = f"gcloud batch jobs submit {job_id} --location {region} --config {job_json_file.name}"
        subprocess.run(cmd, shell=True)
