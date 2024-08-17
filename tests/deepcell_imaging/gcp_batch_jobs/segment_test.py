from unittest.mock import ANY, patch

from deepcell_imaging.gcp_batch_jobs.segment import make_segment_job
from deepcell_imaging.gcp_batch_jobs.types import SegmentationTask


@patch("smart_open.open")
def test_make_multitask_job_json(_patched_open):
    job = make_segment_job(
        region="a-region",
        container_image="an-image",
        model_path="a-model",
        model_hash="a-hash",
        tasks=[
            SegmentationTask(
                input_channels_path="/channels/path",
                tiff_output_uri="/tiff/path",
                input_image_rows=123,
                input_image_cols=456,
            )
        ],
        compartment="a-compartment",
        working_directory="a-directory",
        bigquery_benchmarking_table="a-table",
    )

    assert job == {
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
