"""
This module contains functions for sending notifications to users.
"""

import copy
import json
import requests

TEAMS_NOTIFICATION_TEMPLATE = {
    "type": "message",
    "attachments": [
        {
            "contentType": "application/vnd.microsoft.card.adaptive",
            "contentUrl": None,
            "content": {
                "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                "type": "AdaptiveCard",
                "version": "1.2",
                "body": [
                    {
                        "type": "TextBlock",
                        # The text gets filled in here.
                        # "text": "Hello world"
                    }
                ],
            },
        }
    ],
}


SUCCESS_TEMPLATE = """
## ✅ DeepCell batch job completed

Job id: 12345

Job time: 1h 2m 3s
    
Input npz: url

Output tiff: url
"""

FAILURE_TEMPLATE = """
## ⚠️ DeepCell batch job failed

Job id: 12345

Job time: 1h 2m 3s
    
Input npz: url

Failure reason: ???
"""


# TODO: we need to figure out how to integrate this into a pipeline.
# We don't want a notification for every file being processed:
# we want a notification when the user's batch/collection of files is done.
# Leaving this here now to fill in later.
def send_teams_notification(webhook: str):
    notification_data = copy.deepcopy(TEAMS_NOTIFICATION_TEMPLATE)

    success = True
    if success:
        text = SUCCESS_TEMPLATE.format()
    else:
        text = FAILURE_TEMPLATE.format()

    notification_data["attachments"][0]["content"]["body"][0]["text"] = text

    response = requests.post(
        webhook,
        headers={"Content-Type": "application/json"},
        data=json.dumps(notification_data),
    )
    return response
