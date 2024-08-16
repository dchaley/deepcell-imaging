import json
from typing import Optional

import smart_open
from pydantic import BaseModel

from deepcell_imaging.gcp_batch_jobs.types import (
    PreprocessArgs,
    PredictArgs,
    PostprocessArgs,
    GatherBenchmarkArgs,
    VisualizeArgs,
    SegmentationTask,
)
from deepcell_imaging.utils.numpy import npz_headers
from deepcell_imaging.utils.storage import find_matching_npz

# Note: Need to escape the curly braces in the JSON template
BASE_MULTITASK_TEMPLATE = """
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
                                "--tasks_spec_uri={preprocess_tasks_spec_uri}"
                            ]
                        }}
                    }},
                    {{
                        "container": {{
                            "imageUri": "{container_image}",
                            "entrypoint": "python",
                            "commands": [
                                "scripts/predict.py",
                                "--tasks_spec_uri={predict_tasks_spec_uri}"
                            ]
                        }}
                    }},
                    {{
                        "container": {{
                            "imageUri": "{container_image}",
                            "entrypoint": "python",
                            "commands": [
                                "scripts/postprocess.py",
                                "--tasks_spec_uri={postprocess_tasks_spec_uri}"
                            ]
                        }}
                    }},
                    {{
                        "container": {{
                            "imageUri": "{container_image}",
                            "entrypoint": "python",
                            "commands": [
                                "scripts/gather-benchmark.py",
                                "--tasks_spec_uri={gather_benchmark_tasks_spec_uri}"
                            ]
                        }}
                    }},
                    {{
                        "container": {{
                            "imageUri": "{container_image}",
                            "entrypoint": "python",
                            "commands": [
                                "scripts/visualize.py",
                                "--tasks_spec_uri={visualize_tasks_spec_uri}"
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
            "taskCount": {task_count},
            "taskCountPerNode": 1,
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


def make_multitask_job_json(
    region: str,
    container_image: str,
    model_path: str,
    model_hash: str,
    tasks: list[SegmentationTask],
    compartment: str,
    working_directory: str,
    bigquery_benchmarking_table: Optional[str] = None,
    config: dict = None,
) -> dict:
    preprocess_tasks = []
    predict_tasks = []
    postprocess_tasks = []
    gather_benchmark_tasks = []
    visualize_tasks = []

    for index, task in enumerate(tasks):
        task_directory = f"{working_directory}/task_{index}"

        preprocess_tasks.append(
            PreprocessArgs(
                image_uri=task.input_channels_path,
                output_uri=f"{task_directory}/preprocessed.npz.gz",
                benchmark_output_uri=(
                    f"{task_directory}/preprocess_benchmark.json"
                    if bigquery_benchmarking_table
                    else None
                ),
            )
        )
        predict_tasks.append(
            PredictArgs(
                model_path=model_path,
                model_hash=model_hash,
                image_uri=f"{task_directory}/preprocessed.npz.gz",
                output_uri=f"{task_directory}/raw_predictions.npz.gz",
                benchmark_output_uri=(
                    f"{task_directory}/predict_benchmark.json"
                    if bigquery_benchmarking_table
                    else None
                ),
            )
        )
        postprocess_tasks.append(
            PostprocessArgs(
                raw_predictions_uri=f"{task_directory}/raw_predictions.npz.gz",
                output_uri=f"{task_directory}/predictions.npz.gz",
                tiff_output_uri=f"{task.tiff_output_uri}",
                input_rows=task.input_image_rows,
                input_cols=task.input_image_cols,
                compartment=compartment,
                benchmark_output_uri=(
                    f"{task_directory}/postprocess_benchmark.json"
                    if bigquery_benchmarking_table
                    else None
                ),
            )
        )
        gather_benchmark_tasks.append(
            GatherBenchmarkArgs(
                preprocess_benchmarking_uri=(
                    f"{task_directory}/preprocess_benchmark.json"
                ),
                prediction_benchmarking_uri=(
                    f"{task_directory}/predict_benchmark.json"
                ),
                postprocess_benchmarking_uri=(
                    f"{task_directory}/postprocess_benchmark.json"
                ),
                bigquery_benchmarking_table=bigquery_benchmarking_table,
            )
        )
        visualize_tasks.append(
            VisualizeArgs(
                predictions_uri=f"{task_directory}/predictions.npz.gz",
                image_uri=task.input_channels_path,
                visualized_input_uri=f"{task_directory}/visualized_input.png",
                visualized_predictions_uri=f"{task_directory}/visualized_predictions.png",
            )
        )

    # write the json tasks to the working directory
    preprocess_tasks_spec_uri = f"{working_directory}/preprocess_tasks.json"
    predict_tasks_spec_uri = f"{working_directory}/predict_tasks.json"
    postprocess_tasks_spec_uri = f"{working_directory}/postprocess_tasks.json"
    gather_benchmark_tasks_spec_uri = f"{working_directory}/gather_benchmark_tasks.json"
    visualize_tasks_spec_uri = f"{working_directory}/visualize_tasks.json"

    for step_uri, step_tasks in (
        (preprocess_tasks_spec_uri, preprocess_tasks),
        (predict_tasks_spec_uri, predict_tasks),
        (postprocess_tasks_spec_uri, postprocess_tasks),
        (gather_benchmark_tasks_spec_uri, gather_benchmark_tasks),
        (visualize_tasks_spec_uri, visualize_tasks),
    ):
        with smart_open.open(step_uri, "w") as f:
            json.dump([task.model_dump() for task in step_tasks], f)

    json_str = BASE_MULTITASK_TEMPLATE.format(
        container_image=container_image,
        preprocess_tasks_spec_uri=preprocess_tasks_spec_uri,
        predict_tasks_spec_uri=predict_tasks_spec_uri,
        postprocess_tasks_spec_uri=postprocess_tasks_spec_uri,
        gather_benchmark_tasks_spec_uri=gather_benchmark_tasks_spec_uri,
        visualize_tasks_spec_uri=visualize_tasks_spec_uri,
        task_count=len(tasks),
        region=region,
    )

    job_json = json.loads(json_str)

    if config:
        job_json.update(config)

    return job_json


def make_segmentation_tasks(
    image_names, npz_root, npz_blobs, masks_output_root, storage_client=None
):
    matched_images = find_matching_npz(
        image_names, npz_root, npz_blobs, client=storage_client
    )

    for image_name, npz_path in matched_images:
        # FIXME(#298): this needs to depend on compartment.
        tiff_output_uri = f"{masks_output_root}/{image_name}_WholeCellMask.tiff"

        input_file_contents = list(npz_headers(npz_path))
        if len(input_file_contents) != 1:
            raise ValueError("Expected exactly one array in the input file")
        input_image_shape = input_file_contents[0][1]

        yield SegmentationTask(
            input_channels_path=npz_path,
            tiff_output_uri=tiff_output_uri,
            input_image_rows=input_image_shape[0],
            input_image_cols=input_image_shape[1],
        )
