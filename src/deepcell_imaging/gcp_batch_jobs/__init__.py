"""
This module contains functions for creating and submitting batch jobs to GCP.
"""

import json
import os
import subprocess
import tempfile

import smart_open


def get_batch_indexed_task(tasks_spec_uri, args_cls):
    with smart_open.open(tasks_spec_uri, "r") as tasks_spec_file:
        tasks_spec = json.load(tasks_spec_file)

    task_index = int(os.environ["BATCH_TASK_INDEX"])
    task = tasks_spec[task_index]

    return args_cls(**task)


def apply_allocation_policy(
    job: dict,
    region: str,
    machine_type: str,
    provisioning_model: str,
    gpu_type: str = None,
    gpu_count: int = None,
) -> None:
    """
    Apply an allocation policy to the job definition: machine type, provisioning model, and GPU.
    """
    if gpu_type and not gpu_count or gpu_count and not gpu_type:
        raise ValueError("GPU type and GPU count must be set together")
    if provisioning_model not in ["SPOT", "STANDARD"]:
        raise ValueError("Provisioning model must be either SPOT or STANDARD")

    job["allocationPolicy"] = {
        "instances": [
            {
                "policy": {
                    "machineType": machine_type,
                    "provisioningModel": provisioning_model,
                },
            }
        ],
        "location": {"allowedLocations": [f"regions/{region}"]},
    }

    if gpu_type:
        job["allocationPolicy"]["instances"][0]["installGpuDrivers"] = True
        job["allocationPolicy"]["instances"][0]["policy"]["accelerators"] = [
            {"type": gpu_type, "count": gpu_count}
        ]


def apply_cloud_logs_policy(job: dict) -> None:
    """
    Apply a cloud logging policy to the job definition.
    """
    job["logsPolicy"] = {"destination": "CLOUD_LOGGING"}


def set_boot_disk_size(job: dict, size_mib: int) -> None:
    """
    Set the boot disk size for the job definition.
    """
    # setdefault is used if the computeResource key does not exist yet
    compute_resource = job["taskGroups"][0]["taskSpec"].setdefault(
        "computeResource", {}
    )
    compute_resource["bootDiskMib"] = size_mib


def submit_job(job: dict, job_id: str, region: str) -> None:
    """
    Submit a job to the Batch service.
    """
    with tempfile.NamedTemporaryFile() as job_json_file:
        with open(job_json_file.name, "w") as f:
            json.dump(job, f)

        cmd = f"gcloud batch jobs submit {job_id} --location {region} --config {job_json_file.name}"
        subprocess.run(cmd, shell=True)
