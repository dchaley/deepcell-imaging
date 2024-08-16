import sys
from unittest.mock import ANY, patch

from pydantic import BaseModel, Field

from deepcell_imaging.gcp_batch_jobs.segment import make_multitask_job_json
from deepcell_imaging.gcp_batch_jobs.types import SegmentationTask
from deepcell_imaging.utils.cmdline import get_task_arguments


class ArgsForTest(BaseModel):
    images_path: str
    segmasks_path: str
    project_path: str
    reports_path: str
    image_filter: str = Field("")


@patch("smart_open.open")
def test_make_multitask_job_json(patched_open):
    job = make_multitask_job_json(
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
                        ANY,
                        ANY,
                        ANY,
                        ANY,
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
                },
                "taskCount": 1,
                "taskCountPerNode": 1,
                "parallelism": 1,
            }
        ],
        "allocationPolicy": ANY,
        "logsPolicy": {"destination": "CLOUD_LOGGING"},
    }


def test_argv_parsing():
    test_args = [
        "prog",
        "--images_path",
        "gs://bucket/images",
        "--segmasks_path",
        "/folder/segmasks",
        "--project_path",
        "/folder/project",
        "--reports_path",
        "gs://bucket/reports",
    ]
    with patch.object(sys, "argv", test_args):
        result = get_task_arguments("test", ArgsForTest)

    assert dict(result) == {
        "images_path": "gs://bucket/images",
        "segmasks_path": "/folder/segmasks",
        "project_path": "/folder/project",
        "reports_path": "gs://bucket/reports",
        "image_filter": "",
    }


def test_tasks_uri():
    test_args = [
        "prog",
        "--tasks_spec_uri",
        "gs://bucket/tasks.json",
    ]

    task = ArgsForTest(
        images_path="gs://root/images",
        segmasks_path="/root/segmasks",
        project_path="gs://root/project",
        reports_path="/root/reports",
    )

    with patch.object(sys, "argv", test_args):
        with patch(
            "deepcell_imaging.utils.cmdline.get_batch_indexed_task",
            return_value=task,
        ):
            result = get_task_arguments("test", ArgsForTest)

    assert result.images_path == "gs://root/images"
    assert result.segmasks_path == "/root/segmasks"
    assert result.project_path == "gs://root/project"
    assert result.reports_path == "/root/reports"
    assert result.image_filter == ""
