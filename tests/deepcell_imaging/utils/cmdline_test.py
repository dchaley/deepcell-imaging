import sys
from unittest.mock import ANY, patch

from pydantic import BaseModel, Field

from deepcell_imaging.utils.cmdline import get_task_arguments, parse_compute_config


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
        result, env_config = get_task_arguments("test", ArgsForTest)

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
            result, env_config = get_task_arguments("test", ArgsForTest)

    assert result.images_path == "gs://root/images"
    assert result.segmasks_path == "/root/segmasks"
    assert result.project_path == "gs://root/project"
    assert result.reports_path == "/root/reports"
    assert result.image_filter == ""


def test_parse_compute_config():
    compute_config = parse_compute_config("n123-standard-456:ABC+fast-gpu:321")
    assert compute_config.machine_type == "n123-standard-456"
    assert compute_config.provisioning_model == "ABC"
    assert compute_config.accelerator_type == "fast-gpu"
    assert compute_config.accelerator_count == 321

    compute_config = parse_compute_config("")
    assert compute_config.machine_type == "n1-standard-8"
    assert compute_config.provisioning_model == "SPOT"
    assert compute_config.accelerator_type == "nvidia-tesla-t4"
    assert compute_config.accelerator_count == 1

    compute_config = parse_compute_config("a-b-c")
    assert compute_config.machine_type == "a-b-c"
    assert compute_config.provisioning_model == "SPOT"
    assert compute_config.accelerator_type == "nvidia-tesla-t4"
    assert compute_config.accelerator_count == 1

    compute_config = parse_compute_config(":SPOT")
    assert compute_config.machine_type == "n1-standard-8"
    assert compute_config.provisioning_model == "SPOT"
    assert compute_config.accelerator_type == "nvidia-tesla-t4"
    assert compute_config.accelerator_count == 1

    compute_config = parse_compute_config(":SPOT+:7")
    assert compute_config.machine_type == "n1-standard-8"
    assert compute_config.provisioning_model == "SPOT"
    assert compute_config.accelerator_type == "nvidia-tesla-t4"
    assert compute_config.accelerator_count == 7

    compute_config = parse_compute_config("+gpu:3")
    assert compute_config.machine_type == "n1-standard-8"
    assert compute_config.provisioning_model == "SPOT"
    assert compute_config.accelerator_type == "gpu"
    assert compute_config.accelerator_count == 3

    compute_config = parse_compute_config("+gpu")
    assert compute_config.machine_type == "n1-standard-8"
    assert compute_config.provisioning_model == "SPOT"
    assert compute_config.accelerator_type == "gpu"
    assert compute_config.accelerator_count == 1

    compute_config = parse_compute_config("machine+gpu")
    assert compute_config.machine_type == "machine"
    assert compute_config.provisioning_model == "SPOT"
    assert compute_config.accelerator_type == "gpu"
    assert compute_config.accelerator_count == 1
