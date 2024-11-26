import json

from deepcell_imaging.gcp_batch_jobs import (
    apply_allocation_policy,
    apply_cloud_logs_policy,
    add_attached_disk,
    add_task_volume,
    set_task_environment_variable,
    add_networking_interface,
    add_service_account,
)
from deepcell_imaging.gcp_batch_jobs.types import (
    QuantifyArgs,
    ServiceAccountConfig,
    NetworkInterfaceConfig,
    ComputeConfig,
)

# Note: Need to escape the curly braces in the JSON template
BASE_QUANTIFY_JOB_TEMPLATE = """
{{
    "taskGroups": [
        {{
            "taskSpec": {{
                "runnables": [
                    {{
                        "container": {{
                            "imageUri": "{container_image}",
                            "entrypoint": "java",
                            "commands": [
                                "-jar",
                                "/app/qupath-measurement-1.0-SNAPSHOT-all.jar",
                                "--mode=explicit",
                                "--images-path={images_path}",
                                "--segmasks-path={segmasks_path}",
                                "--project-path={project_path}",
                                "--reports-path={reports_path}",
                                "--image-filter={image_filter}"
                            ]
                        }}
                    }}
                ],
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
            "taskCount": 1,
            "parallelism": 1,
            "taskCountPerNode": 1
        }}
    ]
}}
"""


def append_quantify_task(
    job: dict,
    container_image: str,
    args: QuantifyArgs,
    env_config_uri: str = "",
):
    cmd_args = [
        (f"--{flag_name}", arg_value) for flag_name, arg_value in vars(args).items()
    ]
    if env_config_uri:
        cmd_args.append(("--env_config_uri", env_config_uri))

    runnable = {
        "container": {
            "imageUri": container_image,
            "entrypoint": "python",
            "commands": [
                f"scripts/enqueue-qupath-measurement.py",
                *[arg for pair in cmd_args for arg in pair],
            ],
        }
    }
    job["job_definition"]["taskGroups"][0]["taskSpec"]["runnables"].append(runnable)


def make_quantify_job(
    region: str,
    container_image: str,
    args: QuantifyArgs,
    networking_interface: NetworkInterfaceConfig = None,
    compute_config: ComputeConfig = None,
    service_account: ServiceAccountConfig = None,
    config: dict = None,
) -> dict:
    json_str = BASE_QUANTIFY_JOB_TEMPLATE.format(
        container_image=container_image,
        region=region,
        images_path=args.images_path,
        segmasks_path=args.segmasks_path,
        project_path=args.project_path,
        reports_path=args.reports_path,
        image_filter=args.image_filter,
    )

    print(json_str)
    job = json.loads(json_str)

    if not compute_config:
        compute_config = ComputeConfig(
            machine_type="n1-standard-8",
            provisioning_model="SPOT",
        )

    apply_allocation_policy(job, region, compute_config)
    apply_cloud_logs_policy(job)

    # We know this is (probably) way too much– but we don't have accurate
    # sizing yet. Come back to this!
    # https://github.com/dchaley/qupath-project-initializer/issues/40
    size_in_gb = 500

    volume_name = "qupath-workspace"
    tmp_dir = "/mnt/disks/qupath-workspace"
    add_attached_disk(job, volume_name, size_in_gb)
    add_task_volume(job, tmp_dir, volume_name)
    set_task_environment_variable(job, "TMPDIR", tmp_dir)

    if networking_interface:
        add_networking_interface(job, networking_interface)

    if service_account:
        add_service_account(job, service_account)

    if config:
        job.update(config)

    return job
