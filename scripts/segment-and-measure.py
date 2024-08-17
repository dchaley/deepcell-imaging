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
import logging
import uuid
import datetime

from google.cloud import storage

import deepcell_imaging.gcp_logging
from deepcell_imaging.gcp_batch_jobs.segment import (
    make_segmentation_tasks,
    build_segment_job_tasks,
)
from deepcell_imaging.utils.storage import get_blob_filenames

CONTAINER_IMAGE = "us-central1-docker.pkg.dev/deepcell-on-batch/deepcell-benchmarking-us-central1/qupath-project-initializer:latest"
REGION = "us-central1"


def main():
    deepcell_imaging.gcp_logging.initialize_gcp_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser("segment-and-measure")

    # Common arguments
    parser.add_argument(
        "--image_filter",
        help="Filter for images to process",
        type=str,
        default="",
    )
    parser.add_argument(
        "--model_path",
        help="Path to the model archive",
        type=str,
        default="gs://genomics-data-public-central1/cellular-segmentation/vanvalenlab/deep-cell/vanvalenlab-tf-model-multiplex-downloaded-20230706/MultiplexSegmentation-resaved-20240710.h5",
    )
    parser.add_argument(
        "--model_hash",
        help="Hash of the model archive",
        type=str,
        default="56b0f246081fe6b730ca74eab8a37d60",
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
        "--npz_subdir",
        help="Subdirectory within the dataset containing the image files as numpy arrays",
        type=str,
        default="NPZ_INTERMEDIATE",
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
        "--numpy_path",
        help="Path to the images as numpy arrays",
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

    if args.mode == "workspace":
        image_root = f"{args.dataset_path}/{args.images_subdir}"
        npz_root = f"{args.dataset_path}/{args.npz_subdir}"
        masks_output_root = f"{args.dataset_path}/{args.segmasks_subdir}"
        project_root = f"{args.dataset_path}/{args.project_subdir}"
        reports_root = f"{args.dataset_path}/{args.reports_subdir}"
    elif args.mode == "paths":
        image_root = args.images_path
        npz_root = args.numpy_path
        masks_output_root = args.segmasks_path
        project_root = args.project_path
        reports_root = args.reports_path
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

    client = storage.Client()

    image_paths = get_blob_filenames(image_root, client=client)
    image_paths = [x for x in image_paths if x == args.image_filter]
    npz_paths = get_blob_filenames(npz_root, client=client)

    image_segmentation_tasks = list(
        make_segmentation_tasks(image_paths, npz_root, npz_paths, masks_output_root)
    )

    # The batch job id must be unique, and can only contain lowercase letters,
    # numbers, and hyphens. It must also be 63 characters or fewer.
    # We're doing 62 to be safe.
    #
    # Regex: ^[a-z]([a-z0-9-]{0,61}[a-z0-9])?$
    batch_job_id = "deepcell-{}".format(str(uuid.uuid4()))
    batch_job_id = batch_job_id[0:62].lower()

    if args.mode == "workspace":
        working_directory = f"{args.dataset_path}/jobs/{datetime.datetime.now().isoformat()}_{batch_job_id}"
    else:
        working_directory = (
            f"{npz_root}/jobs/{datetime.datetime.now().isoformat()}_{batch_job_id}"
        )

    job = build_segment_job_tasks(
        region=REGION,
        container_image=CONTAINER_IMAGE,
        model_path=args.model_path,
        model_hash=args.model_hash,
        tasks=image_segmentation_tasks,
        compartment="whole-cell",
        working_directory=working_directory,
    )

    # For now … do nothing, just print the tasks.
    print("Preprocess tasks:")
    print(json.dumps([x.model_dump() for x in job["tasks"]["preprocess"][0]], indent=1))
    print("Predict tasks:")
    print(json.dumps([x.model_dump() for x in job["tasks"]["predict"][0]], indent=1))
    print("Postprocess tasks:")
    print(
        json.dumps([x.model_dump() for x in job["tasks"]["postprocess"][0]], indent=1)
    )
    print("Benchmark tasks:")
    print(
        json.dumps(
            [x.model_dump() for x in job["tasks"]["gather-benchmark"][0]], indent=1
        )
    )
    print("Visualize tasks:")
    print(json.dumps([x.model_dump() for x in job["tasks"]["visualize"][0]], indent=1))


if __name__ == "__main__":
    main()
