import json
from typing import Optional

import smart_open
from pydantic import BaseModel

from deepcell_imaging.gcp_batch_jobs.types import (
    PreprocessArgs,
    PredictArgs,
    PostprocessArgs,
    GatherBenchmarkArgs,
    VisualizeArgs,
)

# Note: Need to escape the curly braces in the JSON template
BASE_QUPATH_MEASUREMENTS_TEMPLATE = """
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
    ],
    "allocationPolicy": {{
        "instances": [
            {{
                "policy": {{
                    "machineType": "n1-standard-8",
                    "provisioningModel": "SPOT"
                }}
            }}
        ],
        "location": {{
            "allowedLocations": [
                "regions/{region}"
            ]
        }}
    }},

    "logsPolicy": {{
        "destination": "CLOUD_LOGGING"
    }}
}}
"""


def make_qupath_measurements_job_json(
    region: str,
    container_image: str,
    images_path: str,
    segmasks_path: str,
    project_path: str,
    reports_path: str,
    image_filter: str,
    config: dict = None,
) -> dict:
    json_str = BASE_QUPATH_MEASUREMENTS_TEMPLATE.format(
        container_image=container_image,
        region=region,
        images_path=images_path,
        segmasks_path=segmasks_path,
        project_path=project_path,
        reports_path=reports_path,
        image_filter=image_filter,
    )

    print(json_str)
    job_json = json.loads(json_str)

    if config:
        job_json.update(config)

    return job_json
