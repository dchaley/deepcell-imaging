from unittest.mock import ANY, patch

from deepcell_imaging.gcp_batch_jobs.segment import (
    build_segment_job_tasks,
    make_segmentation_tasks,
)
from deepcell_imaging.gcp_batch_jobs.types import SegmentationTask


@patch(
    "deepcell_imaging.gcp_batch_jobs.segment.npz_headers",
    return_value=[[[], (123, 456)]],
)
def test_make_segmentation_tasks(_mock_npz_headers):
    tasks = make_segmentation_tasks(
        image_names=["a-prefix", "b-prefix"],
        npz_root="gs://a-dataset/NPZ_INTERMEDIATE",
        npz_names=["a-prefix", "b-prefix"],
        masks_output_root="gs://a-dataset/SEGMASK",
    )

    assert list(tasks) == [
        SegmentationTask(
            input_channels_path="gs://a-dataset/NPZ_INTERMEDIATE/a-prefix.npz",
            wholecell_tiff_output_uri="gs://a-dataset/SEGMASK/a-prefix_WholeCellMask.tiff",
            nuclear_tiff_output_uri="gs://a-dataset/SEGMASK/a-prefix_NuclearMask.tiff",
            input_image_rows=123,
            input_image_cols=456,
        ),
        SegmentationTask(
            input_channels_path="gs://a-dataset/NPZ_INTERMEDIATE/b-prefix.npz",
            wholecell_tiff_output_uri="gs://a-dataset/SEGMASK/b-prefix_WholeCellMask.tiff",
            nuclear_tiff_output_uri="gs://a-dataset/SEGMASK/b-prefix_NuclearMask.tiff",
            input_image_rows=123,
            input_image_cols=456,
        ),
    ]


def test_build_segment_job_tasks():
    job = build_segment_job_tasks(
        region="a-region",
        container_image="an-image",
        model_path="a-model",
        model_hash="a-hash",
        tasks=[
            SegmentationTask(
                input_channels_path="/channels/path",
                wholecell_tiff_output_uri="/tiff/wholecell/path",
                nuclear_tiff_output_uri="/tiff/nuclear/path",
                input_image_rows=123,
                input_image_cols=456,
            )
        ],
        compartment="a-compartment",
        working_directory="a-directory",
        bigquery_benchmarking_table="a-table",
    )

    assert job["job_definition"] == {
        "taskGroups": [
            {
                "taskSpec": {
                    "runnables": [
                        {
                            "container": {
                                "imageUri": "an-image",
                                "entrypoint": "python",
                                "commands": ["scripts/preprocess.py", ANY],
                            }
                        },
                        {
                            "container": {
                                "imageUri": "an-image",
                                "entrypoint": "python",
                                "commands": ["scripts/predict.py", ANY],
                            }
                        },
                        {
                            "container": {
                                "imageUri": "an-image",
                                "entrypoint": "python",
                                "commands": ["scripts/postprocess.py", ANY],
                            }
                        },
                        {
                            "container": {
                                "imageUri": "an-image",
                                "entrypoint": "python",
                                "commands": ["scripts/gather-benchmark.py", ANY],
                            }
                        },
                        {
                            "container": {
                                "imageUri": "an-image",
                                "entrypoint": "python",
                                "commands": ["scripts/visualize.py", ANY],
                            }
                        },
                    ],
                    "computeResource": {
                        "memoryMib": ANY,
                    },
                    "maxRetryCount": ANY,
                    "lifecyclePolicies": [
                        {
                            "action": "RETRY_TASK",
                            "actionCondition": {
                                "exitCodes": [50001],
                            },
                        },
                    ],
                },
                "taskCount": 1,
                "taskCountPerNode": 1,
                "parallelism": 1,
            }
        ],
        "allocationPolicy": {
            "instances": [
                {
                    "installGpuDrivers": True,
                    "policy": {
                        "machineType": "n1-standard-8",
                        "provisioningModel": "SPOT",
                        "accelerators": [
                            {
                                "type": "nvidia-tesla-t4",
                                "count": 1,
                            },
                        ],
                    },
                },
            ],
            "location": {
                "allowedLocations": ["regions/a-region"],
            },
        },
        "logsPolicy": {"destination": "CLOUD_LOGGING"},
    }
