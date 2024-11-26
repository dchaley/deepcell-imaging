"""
This module contains functions for creating and submitting batch jobs to GCP.
"""

import json
import os
import subprocess
import tempfile

import smart_open

from deepcell_imaging.gcp_batch_jobs.types import (
    NetworkInterfaceConfig,
    ServiceAccountConfig,
    ComputeConfig,
)


def get_batch_indexed_task(tasks_spec_uri, args_cls):
    with smart_open.open(tasks_spec_uri, "r") as tasks_spec_file:
        tasks_spec = json.load(tasks_spec_file)

    task_index = int(os.environ["BATCH_TASK_INDEX"])
    task = tasks_spec[task_index]

    return args_cls(**task)


def apply_allocation_policy(
    job: dict,
    region: str,
    compute_config: ComputeConfig,
) -> None:
    """
    Apply an allocation policy to the job definition: machine type, provisioning model, and GPU.
    """
    if (compute_config.accelerator_type and not compute_config.accelerator_count) or (
        compute_config.accelerator_count and not compute_config.accelerator_type
    ):
        raise ValueError("GPU type and GPU count must be set together")
    if compute_config.provisioning_model not in ["SPOT", "STANDARD"]:
        raise ValueError("Provisioning model must be either SPOT or STANDARD")

    job["allocationPolicy"] = {
        "instances": [
            {
                "policy": {
                    "machineType": compute_config.machine_type,
                    "provisioningModel": compute_config.provisioning_model,
                },
            }
        ],
        "location": {"allowedLocations": [f"regions/{region}"]},
    }

    if compute_config.accelerator_type:
        job["allocationPolicy"]["instances"][0]["installGpuDrivers"] = True
        job["allocationPolicy"]["instances"][0]["policy"]["accelerators"] = [
            {
                "type": compute_config.accelerator_type,
                "count": compute_config.accelerator_count,
            }
        ]


def apply_cloud_logs_policy(job: dict) -> None:
    """
    Apply a cloud logging policy to the job definition.
    """
    job["logsPolicy"] = {"destination": "CLOUD_LOGGING"}


def add_attached_disk(
    job: dict, device_name: str, size_gb: int, disk_type: str = "pd-balanced"
) -> None:
    """
    Set the boot disk size for the job definition.
    """
    attached_disk = {
        "deviceName": device_name,
        "newDisk": {
            "type": disk_type,
            "sizeGb": max(size_gb, 1),
        },
    }

    job["allocationPolicy"]["instances"][0]["policy"]["disks"] = [attached_disk]


def add_task_volume(job: dict, mount_path: str, device_name: str) -> None:
    """
    Add a volume to the job definition.
    """
    job["taskGroups"][0]["taskSpec"]["volumes"] = [
        {
            "mountPath": mount_path,
            "deviceName": device_name,
        }
    ]


def set_task_environment_variable(job: dict, key: str, value: str) -> None:
    """
    Set an environment variable for the task in the job definition.
    """
    env = job["taskGroups"][0]["taskSpec"].setdefault("environment", {})
    env_vars = env.setdefault("variables", {})
    env_vars[key] = value


def add_networking_interface(job: dict, networking_interface: NetworkInterfaceConfig):
    job["allocationPolicy"].setdefault("network", {})["networkInterfaces"] = [
        networking_interface.model_dump()
    ]


def add_service_account(job: dict, service_account: ServiceAccountConfig):
    job["allocationPolicy"]["serviceAccount"] = service_account.model_dump()


def submit_job(job: dict, job_id: str, region: str) -> None:
    """
    Submit a job to the Batch service.
    """
    with tempfile.NamedTemporaryFile() as job_json_file:
        with open(job_json_file.name, "w") as f:
            json.dump(job, f)

        cmd = f"gcloud batch jobs submit {job_id} --location {region} --config {job_json_file.name}"
        subprocess.run(cmd, shell=True)
