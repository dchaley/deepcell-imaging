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

import smart_open
from google.cloud import storage

import deepcell_imaging.gcp_logging
from deepcell_imaging.gcp_batch_jobs import submit_job
from deepcell_imaging.gcp_batch_jobs.quantify import append_quantify_task
from deepcell_imaging.gcp_batch_jobs.segment import (
    make_segmentation_tasks,
    build_segment_job_tasks,
    upload_tasks,
)
from deepcell_imaging.gcp_batch_jobs.types import QuantifyArgs, EnvironmentConfig
from deepcell_imaging.utils.cmdline import add_dataset_parameters, get_dataset_paths
from deepcell_imaging.utils.storage import get_blob_filenames


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
    parser.add_argument(
        "--env_config_uri",
        help="URI to a JSON file containing GCP configuration",
        type=str,
        required=True,
    )

    add_dataset_parameters(parser)

    args = parser.parse_args()

    dataset_paths = get_dataset_paths(args)

    with smart_open.open(args.env_config_uri, "r") as env_config_file:
        env_config_json = json.load(env_config_file)
        env_config = EnvironmentConfig(**env_config_json)

    client = storage.Client()

    logger.info("Fetching images")

    image_paths = get_blob_filenames(dataset_paths["image_root"], client=client)
    image_paths = [
        x for x in image_paths if (not args.image_filter or x == args.image_filter)
    ]

    logger.info("Finding matching npz files")

    npz_paths = get_blob_filenames(dataset_paths["npz_root"], client=client)
    image_segmentation_tasks = list(
        make_segmentation_tasks(
            image_paths,
            dataset_paths["npz_root"],
            npz_paths,
            dataset_paths["masks_output_root"],
        )
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
        working_directory = f"{dataset_paths['npz_root']}/jobs/{datetime.datetime.now().isoformat()}_{batch_job_id}"

    job = build_segment_job_tasks(
        region=env_config.region,
        container_image=env_config.segment_container_image,
        model_path=args.model_path,
        model_hash=args.model_hash,
        tasks=image_segmentation_tasks,
        compartment="both",
        working_directory=working_directory,
        bigquery_benchmarking_table=env_config.bigquery_benchmarking_table,
        networking_interface=env_config.networking_interface,
        service_account=env_config.service_account,
    )

    # Note that we use the SEGMENT container here, not quantify,
    # because we launch the quantify job FROM the segment job.
    append_quantify_task(
        job,
        env_config.segment_container_image,
        QuantifyArgs(
            images_path=dataset_paths["image_root"],
            segmasks_path=dataset_paths["masks_output_root"],
            project_path=dataset_paths["project_root"],
            reports_path=dataset_paths["reports_root"],
            image_filter=args.image_filter,
        ),
        env_config_uri=args.env_config_uri,
    )

    logger.info("Uploading task files")
    upload_tasks(job["tasks"])

    logger.info("Submitting job to Batch")
    job_json = job["job_definition"]
    submit_job(job_json, batch_job_id, env_config.region)

    logger.info("Batch job id: %s", batch_job_id)
    logger.info("Working directory: %s", working_directory)


if __name__ == "__main__":
    main()
