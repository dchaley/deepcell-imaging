from unittest.mock import ANY, patch

from deepcell_imaging.gcp_batch_jobs.quantify import make_quantify_job
from deepcell_imaging.gcp_batch_jobs.types import QuantifyArgs


@patch("smart_open.open")
def test_make_quantify_job(_patched_open):
    job = make_quantify_job(
        region="a-region",
        container_image="an-image",
        args=QuantifyArgs(
            images_path="/images/path",
            segmasks_path="/segmasks/path",
            project_path="/project/path",
            reports_path="/reports/path",
            image_filter="a-filter",
        ),
    )

    assert job == {
        "taskGroups": [
            {
                "taskSpec": {
                    "runnables": [
                        ANY,
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
                    "environment": {
                        "variables": {"TMPDIR": ANY},
                    },
                    "volumes": ANY,
                },
                "taskCount": 1,
                "taskCountPerNode": 1,
                "parallelism": 1,
            }
        ],
        "allocationPolicy": {
            "instances": [
                {
                    "policy": {
                        "machineType": "n1-standard-8",
                        "provisioningModel": "SPOT",
                        "disks": ANY,
                    },
                },
            ],
            "location": {
                "allowedLocations": ["regions/a-region"],
            },
        },
        "logsPolicy": {"destination": "CLOUD_LOGGING"},
    }
