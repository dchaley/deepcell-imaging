from unittest.mock import ANY, patch

from deepcell_imaging.gcp_batch_jobs.quantify import make_quantify_job, DEFAULT_TMP_DIR
from deepcell_imaging.gcp_batch_jobs.types import EnqueueQuantifyArgs


@patch("smart_open.open")
def test_make_quantify_job(_patched_open):
    job = make_quantify_job(
        region="a-region",
        container_image="an-image",
        args=EnqueueQuantifyArgs(
            images_path="/images/path",
            segmasks_path="/segmasks/path",
            project_path="/project/path",
            reports_path="/reports/path",
            image_filter="a-filter",
            compute_config="my-machine",
        ),
    )

    assert job == {
        "taskGroups": [
            {
                "taskSpec": {
                    "runnables": [
                        {
                            "container": {
                                "imageUri": "an-image",
                                "entrypoint": "java",
                                "commands": [
                                    f"-Djava.io.tmpdir={DEFAULT_TMP_DIR}",
                                    "-jar",
                                    "/app/qupath-measurement-1.0-SNAPSHOT-all.jar",
                                    "--mode=explicit",
                                    "--images-path=/images/path",
                                    "--segmasks-path=/segmasks/path",
                                    "--project-path=/project/path",
                                    "--reports-path=/reports/path",
                                    "--image-filter=a-filter",
                                ],
                            },
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
                    "environment": {
                        "variables": {"TMPDIR": DEFAULT_TMP_DIR},
                    },
                    "volumes": [
                        {
                            "deviceName": "qupath-workspace",
                            "mountPath": DEFAULT_TMP_DIR,
                        }
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
                    "policy": {
                        "machineType": "my-machine",
                        "provisioningModel": "SPOT",
                        "disks": [
                            {
                                "deviceName": "qupath-workspace",
                                "newDisk": ANY,
                            }
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
