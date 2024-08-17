import json

from deepcell_imaging.gcp_batch_jobs import (
    apply_allocation_policy,
    apply_cloud_logs_policy,
)
from deepcell_imaging.gcp_batch_jobs.types import QuantifyArgs

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
                                "/app/qupath-project-initializer-1.0-SNAPSHOT-all.jar",
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


def make_quantify_job(
    region: str,
    container_image: str,
    args: QuantifyArgs,
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

    apply_allocation_policy(
        job,
        region,
        "n1-standard-8",
        "SPOT",
    )
    apply_cloud_logs_policy(job)

    if config:
        job.update(config)

    return job
