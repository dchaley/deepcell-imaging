#!/usr/bin/env python

import argparse
import pathlib
import subprocess
import tempfile
import urllib
import uuid

from deepcell_imaging.gcp_batch_jobs import make_job_json
from deepcell_imaging.numpy_utils import npz_headers

parser = argparse.ArgumentParser("deepcell-on-batch")
parser.add_argument(
    "--input_channels_path",
    help="Path to the input channels npz file",
    type=str,
    required=True,
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

args = parser.parse_args()

job_id = "j" + str(uuid.uuid4())
input_channels_path = args.input_channels_path
bigquery_benchmarking_table = args.bigquery_benchmarking_table

CONTAINER_IMAGE = "us-central1-docker.pkg.dev/deepcell-on-batch/deepcell-benchmarking-us-central1/benchmarking:batch"
OUTPUT_BASE_PATH = "gs://deepcell-batch-jobs_us-central1/job-runs"
REGION = "us-central1"

output_path = "{}/{}".format(OUTPUT_BASE_PATH, job_id)

# For now, assume there's only one file in the input.
input_file_contents = list(npz_headers(input_channels_path))
if len(input_file_contents) != 1:
    raise ValueError("Expected exactly one array in the input file")
input_image_shape = input_file_contents[0][1]

parsed_input_path = urllib.parse.urlparse(input_channels_path)
input_file_stem = pathlib.Path(parsed_input_path.path).stem

job_json_str = make_job_json(
    region=REGION,
    container_image=CONTAINER_IMAGE,
    model_path=args.model_path,
    model_hash=args.model_hash,
    bigquery_benchmarking_table=bigquery_benchmarking_table,
    input_channels_path=input_channels_path,
    working_directory=output_path,
    tiff_output_uri="{}/{}.tiff".format(output_path, input_file_stem),
    input_image_rows=input_image_shape[0],
    input_image_cols=input_image_shape[1],
)

job_json_file = tempfile.NamedTemporaryFile()
with open(job_json_file.name, "w") as f:
    f.write(job_json_str)

cmd = "gcloud batch jobs submit {job_id} --location {location} --config {job_filename}".format(
    job_id=job_id, location=REGION, job_filename=job_json_file.name
)
subprocess.run(cmd, shell=True)

print("Job submitted with ID: {}".format(job_id))
print("Output: {}".format(output_path))
