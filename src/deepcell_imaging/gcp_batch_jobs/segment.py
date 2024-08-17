import json
import logging
from typing import Optional

import smart_open

from deepcell_imaging.gcp_batch_jobs import (
    apply_cloud_logs_policy,
    apply_allocation_policy,
)
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
                "runnables": [],
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
    ]
}}
"""


logger = logging.getLogger(__name__)


def create_segmenting_runnable(
    container_image: str,
    phase: str,
    tasks_uris: dict[str, str],
):
    if phase not in [
        "preprocess",
        "predict",
        "postprocess",
        "gather-benchmark",
        "visualize",
    ]:
        raise ValueError(f"Invalid phase: {phase}")

    return {
        "container": {
            "imageUri": container_image,
            "entrypoint": "python",
            "commands": [
                f"scripts/{phase}.py",
                f"--tasks_spec_uri={tasks_uris[phase]}",
            ],
        }
    }


def make_segment_preprocess_tasks(
    tasks: list[SegmentationTask],
    working_directory: str,
    bigquery_benchmarking_table: str,
):
    preprocess_tasks = []
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

    return preprocess_tasks


def make_segment_predict_tasks(
    model_path: str,
    model_hash: str,
    tasks: list[SegmentationTask],
    working_directory: str,
    bigquery_benchmarking_table: str,
):
    predict_tasks = []
    for index, task in enumerate(tasks):
        task_directory = f"{working_directory}/task_{index}"
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

    return predict_tasks


def make_segment_postprocess_tasks(
    tasks: list[SegmentationTask],
    working_directory: str,
    compartment: str,
    bigquery_benchmarking_table: Optional[str] = None,
):
    postprocess_tasks = []
    for index, task in enumerate(tasks):
        task_directory = f"{working_directory}/task_{index}"
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

    return postprocess_tasks


def make_segment_benchmark_tasks(
    tasks: list[SegmentationTask],
    working_directory: str,
    bigquery_benchmarking_table: Optional[str] = None,
):
    gather_benchmark_tasks = []
    for index, task in enumerate(tasks):
        task_directory = f"{working_directory}/task_{index}"
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

    return gather_benchmark_tasks


def make_segment_visualize_tasks(
    tasks: list[SegmentationTask],
    working_directory: str,
    image_array_name: str,
):
    visualize_tasks = []
    for index, task in enumerate(tasks):
        task_directory = f"{working_directory}/task_{index}"
        visualize_tasks.append(
            VisualizeArgs(
                image_uri=task.input_channels_path,
                image_array_name=image_array_name,
                predictions_uri=f"{task_directory}/predictions.npz.gz",
                visualized_input_uri=f"{task_directory}/visualized_input.png",
                visualized_predictions_uri=f"{task_directory}/visualized_predictions.png",
            )
        )

    return visualize_tasks


def upload_tasks(
    working_directory: str,
    preprocess_tasks: list[PreprocessArgs],
    predict_tasks: list[PredictArgs],
    postprocess_tasks: list[PostprocessArgs],
    gather_benchmark_tasks: list[GatherBenchmarkArgs],
    visualize_tasks: list[VisualizeArgs],
):
    preprocess_tasks_spec_uri = f"{working_directory}/preprocess_tasks.json"
    predict_tasks_spec_uri = f"{working_directory}/predict_tasks.json"
    postprocess_tasks_spec_uri = f"{working_directory}/postprocess_tasks.json"
    gather_benchmark_tasks_spec_uri = f"{working_directory}/gather_benchmark_tasks.json"
    visualize_tasks_spec_uri = f"{working_directory}/visualize_tasks.json"

    for tasks, upload_uri in (
        (preprocess_tasks, preprocess_tasks_spec_uri),
        (predict_tasks, predict_tasks_spec_uri),
        (postprocess_tasks, postprocess_tasks_spec_uri),
        (gather_benchmark_tasks, gather_benchmark_tasks_spec_uri),
        (visualize_tasks, visualize_tasks_spec_uri),
    ):
        with smart_open.open(upload_uri, "w") as f:
            json.dump([task.model_dump() for task in tasks], f)

    return {
        "preprocess": preprocess_tasks_spec_uri,
        "predict": predict_tasks_spec_uri,
        "postprocess": postprocess_tasks_spec_uri,
        "gather-benchmark": gather_benchmark_tasks_spec_uri,
        "visualize": visualize_tasks_spec_uri,
    }


def make_segment_job(
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

    preprocess_tasks = make_segment_preprocess_tasks(
        tasks, working_directory, bigquery_benchmarking_table
    )
    predict_tasks = make_segment_predict_tasks(
        model_path, model_hash, tasks, working_directory, bigquery_benchmarking_table
    )
    postprocess_tasks = make_segment_postprocess_tasks(
        tasks, working_directory, compartment, bigquery_benchmarking_table
    )
    gather_benchmark_tasks = make_segment_benchmark_tasks(
        tasks, working_directory, bigquery_benchmarking_table
    )
    visualize_tasks = make_segment_visualize_tasks(
        tasks, working_directory, "input_channels"
    )

    task_uris = upload_tasks(
        working_directory,
        preprocess_tasks,
        predict_tasks,
        postprocess_tasks,
        gather_benchmark_tasks,
        visualize_tasks,
    )

    json_str = BASE_MULTITASK_TEMPLATE.format(
        task_count=len(tasks),
    )

    job = json.loads(json_str)

    job["taskGroups"][0]["taskSpec"]["runnables"] = [
        create_segmenting_runnable(container_image, "preprocess", task_uris),
        create_segmenting_runnable(container_image, "predict", task_uris),
        create_segmenting_runnable(container_image, "postprocess", task_uris),
        create_segmenting_runnable(container_image, "gather-benchmark", task_uris),
        create_segmenting_runnable(container_image, "visualize", task_uris),
    ]

    apply_allocation_policy(
        job,
        region,
        "n1-standard-8",
        "SPOT",
        gpu_type="nvidia-tesla-t4",
        gpu_count=1,
    )
    apply_cloud_logs_policy(job)

    if config:
        job.update(config)

    return job


def make_segmentation_tasks(image_names, npz_root, npz_names, masks_output_root):
    matched_images = find_matching_npz(image_names, npz_root, npz_names)

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
