import json
import logging
from typing import Optional

import smart_open

from deepcell_imaging.gcp_batch_jobs import (
    apply_cloud_logs_policy,
    apply_allocation_policy,
    add_attached_disk,
    add_task_volume,
    set_task_environment_variable,
    add_networking_interface,
    add_service_account,
)
from deepcell_imaging.gcp_batch_jobs.types import (
    PreprocessArgs,
    PredictArgs,
    PostprocessArgs,
    GatherBenchmarkArgs,
    VisualizeArgs,
    SegmentationTask,
    NetworkInterfaceConfig,
    ComputeConfig,
    ServiceAccountConfig,
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
    tasks_uris: dict[str, tuple[list, str]],
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
                f"--tasks_spec_uri={tasks_uris[phase][1]}",
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
                image_name=task.image_name,
                output_uri=f"{task_directory}/preprocessed.npz.gz",
                benchmark_output_uri=(
                    f"{task_directory}/preprocess_benchmark.json"
                    if bigquery_benchmarking_table
                    else ""
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
                    else ""
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
                wholecell_tiff_output_uri=f"{task.wholecell_tiff_output_uri}",
                nuclear_tiff_output_uri=f"{task.nuclear_tiff_output_uri}",
                input_rows=task.input_image_rows,
                input_cols=task.input_image_cols,
                compartment=compartment,
                benchmark_output_uri=(
                    f"{task_directory}/postprocess_benchmark.json"
                    if bigquery_benchmarking_table
                    else ""
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
                bigquery_benchmarking_table=(
                    bigquery_benchmarking_table if bigquery_benchmarking_table else ""
                ),
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
    tasks: dict[str, tuple[list, str]],
):
    for phase_tasks, upload_uri in tasks.values():
        with smart_open.open(upload_uri, "w") as f:
            json.dump([task.model_dump() for task in phase_tasks], f)


def build_segment_job_tasks(
    region: str,
    container_image: str,
    model_path: str,
    model_hash: str,
    tasks: list[SegmentationTask],
    compartment: str,
    working_directory: str,
    bigquery_benchmarking_table: Optional[str] = None,
    networking_interface: NetworkInterfaceConfig = None,
    compute_config: ComputeConfig = None,
    service_account: ServiceAccountConfig = None,
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

    preprocess_tasks_spec_uri = f"{working_directory}/preprocess_tasks.json"
    predict_tasks_spec_uri = f"{working_directory}/predict_tasks.json"
    postprocess_tasks_spec_uri = f"{working_directory}/postprocess_tasks.json"
    gather_benchmark_tasks_spec_uri = f"{working_directory}/gather_benchmark_tasks.json"
    visualize_tasks_spec_uri = f"{working_directory}/visualize_tasks.json"

    phase_task_defs = {
        "preprocess": (preprocess_tasks, preprocess_tasks_spec_uri),
        "predict": (predict_tasks, predict_tasks_spec_uri),
        "postprocess": (postprocess_tasks, postprocess_tasks_spec_uri),
        "gather-benchmark": (gather_benchmark_tasks, gather_benchmark_tasks_spec_uri),
        "visualize": (visualize_tasks, visualize_tasks_spec_uri),
    }

    json_str = BASE_MULTITASK_TEMPLATE.format(task_count=len(tasks))

    job = json.loads(json_str)

    job["taskGroups"][0]["taskSpec"]["runnables"] = [
        create_segmenting_runnable(container_image, "preprocess", phase_task_defs),
        create_segmenting_runnable(container_image, "predict", phase_task_defs),
        create_segmenting_runnable(container_image, "postprocess", phase_task_defs),
        create_segmenting_runnable(
            container_image, "gather-benchmark", phase_task_defs
        ),
        create_segmenting_runnable(container_image, "visualize", phase_task_defs),
    ]

    if not compute_config:
        compute_config = ComputeConfig(
            machine_type="n1-standard-8",
            provisioning_model="SPOT",
            accelerator_count=1,
            accelerator_type="nvidia-tesla-t4",
        )

    apply_allocation_policy(
        job,
        region,
        compute_config,
    )
    apply_cloud_logs_policy(job)

    # Set boot disk size based on the largest input image.
    # The largest intermediate file (raw predictions) has 4 float64s per pixel.
    biggest_pixels = max(
        [task.input_image_rows * task.input_image_cols for task in tasks]
    )
    size_in_bytes = biggest_pixels * 8 * 4

    volume_name = "deepcell-workspace"
    tmp_dir = "/mnt/disks/deepcell-workspace"
    add_attached_disk(job, volume_name, size_in_bytes // 1024 // 1024 // 1024)
    add_task_volume(job, tmp_dir, volume_name)
    set_task_environment_variable(job, "TMPDIR", tmp_dir)

    if networking_interface:
        add_networking_interface(job, networking_interface)

    if service_account:
        add_service_account(job, service_account)

    if config:
        job.update(config)

    return {"job_definition": job, "tasks": phase_task_defs}


def make_segmentation_tasks(image_names, npz_root, npz_names, masks_output_root):
    matched_images = find_matching_npz(image_names, npz_root, npz_names)

    for image_name, npz_path in matched_images:
        wholecell_tiff_output_uri = (
            f"{masks_output_root}/{image_name}_WholeCellMask.tiff"
        )
        nuclear_tiff_output_uri = f"{masks_output_root}/{image_name}_NucleusMask.tiff"

        input_file_contents = list(npz_headers(npz_path))
        if len(input_file_contents) != 1:
            raise ValueError("Expected exactly one array in the input file")
        input_image_shape = input_file_contents[0][1]

        yield SegmentationTask(
            input_channels_path=npz_path,
            image_name=image_name,
            wholecell_tiff_output_uri=wholecell_tiff_output_uri,
            nuclear_tiff_output_uri=nuclear_tiff_output_uri,
            input_image_rows=input_image_shape[0],
            input_image_cols=input_image_shape[1],
        )
