#!/usr/bin/env python

import argparse
import datetime
import json
import os
import subprocess
import tempfile
import uuid

from google.cloud import storage
from google.cloud.storage import Blob

from deepcell_imaging.gcp_batch_jobs.multitask import TaskSpec, make_multitask_job_json
from deepcell_imaging.numpy_utils import npz_headers
from deepcell_imaging.utils.storage import get_blobs, find_matching_npz

CONTAINER_IMAGE = "us-central1-docker.pkg.dev/deepcell-on-batch/deepcell-benchmarking-us-central1/benchmarking:batch"
REGION = "us-central1"

parser = argparse.ArgumentParser("deepcell-on-batch")
parser.add_argument(
    "--dataset",
    help="Path to the dataset root directory (containing subfolders OMETIFF, SEGMASK, etc)",
    type=str,
    required=True,
)
parser.add_argument(
    "--prefix",
    help="Input prefix, eg ROI01. Input will be: <dataset>/NPZ_INTERMEDIATE/<prefix>.npz",
)
parser.add_argument(
    "--compartment",
    help="Which predictions to generate: whole-cell (default), nuclear, or both",
    type=str,
    choices=["whole-cell", "nuclear", "both"],
    default="whole-cell",
)
parser.add_argument(
    "--bigquery_benchmarking_table",
    help="BigQuery table to write benchmarking results to (empty for none)",
    type=str,
    required=False,
    default="deepcell-on-batch.benchmarking.results_batch",
)
parser.add_argument(
    "--model_path",
    help="Path to the model archive",
    type=str,
    required=False,
    default="gs://genomics-data-public-central1/cellular-segmentation/vanvalenlab/deep-cell/vanvalenlab-tf-model-multiplex-downloaded-20230706/MultiplexSegmentation-resaved-20240710.h5",
)
parser.add_argument(
    "--model_hash",
    help="Hash of the model archive",
    type=str,
    required=False,
    default="56b0f246081fe6b730ca74eab8a37d60",
)
parser.add_argument(
    "--configuration",
    help="Path to the Batch configuration file",
    type=str,
    required=False,
)

args = parser.parse_args()

dataset = args.dataset.rstrip("/")
prefix = args.prefix
compartment = args.compartment

image_root = f"{dataset}/OMETIFF"
npz_root = f"{dataset}/NPZ_INTERMEDIATE"
masks_output_root = f"{dataset}/SEGMASK"

client = storage.Client()

image_blobs = set(get_blobs(image_root))
npz_blobs = set(get_blobs(npz_root))

# For each image, with a corresponding npz,
# generate a DeepCell task.
tasks = []

# prefixes = ["mesmer_3", "mesmer_10"]
prefixes = []

for image_name, npz_path in find_matching_npz(
    image_blobs, npz_root, npz_blobs, client=client
):
    if prefixes and image_name not in prefixes:
        continue

    # FIXME: this needs to depend on compartment.
    tiff_output_uri = f"{masks_output_root}/{image_name}_WholeCellMask.tiff"

    # FIXME: this is slow, can we get them in bulk??
    # For now, assume there's only one file in the input.
    input_file_contents = list(npz_headers(npz_path))
    if len(input_file_contents) != 1:
        raise ValueError("Expected exactly one array in the input file")
    input_image_shape = input_file_contents[0][1]

    print("Found image", image_name, "with shape", input_image_shape)

    tasks.append(
        TaskSpec(
            input_channels_path=npz_path,
            tiff_output_uri=tiff_output_uri,
            input_image_rows=input_image_shape[0],
            input_image_cols=input_image_shape[1],
        )
    )

# The batch job id must be unique, and can only contain lowercase letters,
# numbers, and hyphens. It must also be 63 characters or fewer.
# We're doing 62 to be safe.
#
# Regex: ^[a-z]([a-z0-9-]{0,61}[a-z0-9])?$
batch_job_id = "deepcell-{}".format(str(uuid.uuid4()))
batch_job_id = batch_job_id[0:62].lower()

working_directory = (
    f"{dataset}/jobs/{datetime.datetime.now().isoformat()}_{batch_job_id}"
)

bigquery_benchmarking_table = args.bigquery_benchmarking_table

if args.configuration:
    with open(args.configuration, "r") as f:
        config = json.load(f)
else:
    config = {}

job_json = make_multitask_job_json(
    region=REGION,
    container_image=CONTAINER_IMAGE,
    model_path=args.model_path,
    model_hash=args.model_hash,
    bigquery_benchmarking_table=bigquery_benchmarking_table,
    compartment=compartment,
    working_directory=working_directory,
    config=config,
    tasks=tasks,
)

job_json_file = tempfile.NamedTemporaryFile()
with open(job_json_file.name, "w") as f:
    json.dump(job_json, f)

cmd = "gcloud batch jobs submit {job_id} --location {location} --config {job_filename}".format(
    job_id=batch_job_id, location=REGION, job_filename=job_json_file.name
)
subprocess.run(cmd, shell=True)

print("Job submitted with ID: {}".format(batch_job_id))
print("Intermediate output: {}".format(working_directory))
