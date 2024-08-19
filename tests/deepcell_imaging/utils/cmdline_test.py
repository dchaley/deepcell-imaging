import sys
from unittest.mock import ANY, patch

from pydantic import BaseModel, Field

from deepcell_imaging.utils.cmdline import get_task_arguments


class ArgsForTest(BaseModel):
    images_path: str
    segmasks_path: str
    project_path: str
    reports_path: str
    image_filter: str = Field("")


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
        result, extra_args = get_task_arguments("test", ArgsForTest)

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
            result, extra_args = get_task_arguments("test", ArgsForTest)

    assert result.images_path == "gs://root/images"
    assert result.segmasks_path == "/root/segmasks"
    assert result.project_path == "gs://root/project"
    assert result.reports_path == "/root/reports"
    assert result.image_filter == ""
