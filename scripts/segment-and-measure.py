#!/usr/bin/env python
"""
Segment a collection of images using DeepCell, then measure cells using QuPath.

This script submits a Batch job to run DeepCell. Upon completion, the job submits
the subsequent QuPath job.

ℹ NOTE: This script assumes the input images have already been converted to
intermediate numpy files.
"""
import argparse
import json
import os
import subprocess
import tempfile
import uuid

import deepcell_imaging.gcp_logging
import logging

from deepcell_imaging.gcp_batch_jobs.qupath_measurements import (
    make_qupath_measurements_job_json,
)
from deepcell_imaging.gcp_batch_jobs.types import QupathMeasurementArgs
from deepcell_imaging.utils.cmdline import get_task_arguments

CONTAINER_IMAGE = "us-central1-docker.pkg.dev/deepcell-on-batch/deepcell-benchmarking-us-central1/qupath-project-initializer:latest"
REGION = "us-central1"


def main():
    deepcell_imaging.gcp_logging.initialize_gcp_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser("segment-and-measure")
    parser.add_argument(
        "--image_filter",
        help="Filter for images to process",
        type=str,
        default="",
    )

    subparsers = parser.add_subparsers(help="Mode of operation", dest="mode")

    workspace_parser = subparsers.add_parser("workspace")
    workspace_parser.add_argument(
        "dataset_path",  # positional argument
        help="Path to the dataset",
    )
    workspace_parser.add_argument(
        "--images_subdir",
        help="Subdirectory within the dataset containing the images",
        type=str,
        default="OMETIFF",
    )
    workspace_parser.add_argument(
        "--segmasks_subdir",
        help="Subdirectory within the dataset containing the segmentation masks",
        type=str,
        default="SEGMASK",
    )
    workspace_parser.add_argument(
        "--project_subdir",
        help="Subdirectory within the dataset containing the QuPath project",
        type=str,
        default="PROJ",
    )
    workspace_parser.add_argument(
        "--reports_subdir",
        help="Subdirectory within the dataset containing the QuPath reports",
        type=str,
        default="REPORTS",
    )

    paths_parser = subparsers.add_parser("paths")
    paths_parser.add_argument(
        "--images_path",
        help="Path to the images",
        required=True,
    )
    paths_parser.add_argument(
        "--segmasks_path",
        help="Path to the segmentation masks",
        required=True,
    )
    paths_parser.add_argument(
        "--project_path",
        help="Path to the QuPath project",
        required=True,
    )
    paths_parser.add_argument(
        "--reports_path",
        help="Path to the QuPath reports",
        required=True,
    )

    args = parser.parse_args()

    # For now … do nothing, just print the parsed args.
    print(args)


if __name__ == "__main__":
    main()