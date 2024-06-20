#!/usr/bin/env python
"""
Script to preprocess an input image for a Mesmer model.

Reads input image from a URI (typically on cloud storage).

Writes preprocessed image to a URI (typically on cloud storage).
"""

import argparse
from deepcell_imaging import benchmark_utils
from google.cloud import bigquery
import io
import json
import smart_open
from tenacity import retry, retry_if_exception_message, wait_random_exponential
import timeit

parser = argparse.ArgumentParser("preprocess")

parser.add_argument(
    "--preprocess_benchmarking_uri",
    help="URI to benchmarking data for the preprocessing step.",
    type=str,
    required=True,
)
parser.add_argument(
    "--prediction_benchmarking_uri",
    help="URI to benchmarking data for the prediction step.",
    type=str,
    required=True,
)
parser.add_argument(
    "--postprocess_benchmarking_uri",
    help="URI to benchmarking data for the postprocessing step.",
    type=str,
    required=True,
)
parser.add_argument(
    "--bigquery_benchmarking_table",
    help="The fully qualified name (project.dataset.table) of the BigQuery table to write benchmarking data to.",
    type=str,
    required=True,
)

args = parser.parse_args()

preprocess_benchmarking_uri = args.preprocess_benchmarking_uri
prediction_benchmarking_uri = args.prediction_benchmarking_uri
postprocess_benchmarking_uri = args.postprocess_benchmarking_uri
bigquery_benchmarking_table = args.bigquery_benchmarking_table

benchmarking_data = {
    "cloud_region": benchmark_utils.get_gce_region(),
}

print("Loading benchmarking data")

t = timeit.default_timer()

for data_uri in [preprocess_benchmarking_uri, prediction_benchmarking_uri, postprocess_benchmarking_uri]:
    with smart_open.open(data_uri, "r") as data_file:
        data = json.load(data_file)
        benchmarking_data.update(data)

data_load_time_s = timeit.default_timer() - t

print("Loaded benchmarking data in %s s" % data_load_time_s)

# Update the overall success to the logical AND of the individual steps
benchmarking_data['success'] = benchmarking_data['preprocessing_success'] and benchmarking_data['prediction_success'] and benchmarking_data['postprocessing_success']

print("Sending data to BigQuery")

t = timeit.default_timer()

bq_client = bigquery.Client()

job_config = bigquery.LoadJobConfig(
    write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
    source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
)

json_str = io.StringIO(json.dumps(benchmarking_data))


@retry(
    wait=wait_random_exponential(multiplier=1, max=60),
    retry=retry_if_exception_message(match=".*403 Exceeded rate limits.*"),
)
def upload_to_bigquery(csv_string, table_id, bq_job_config):
    load_job = bq_client.load_table_from_file(
        csv_string, table_id, job_config=bq_job_config
    )
    load_job.result()  # Waits for the job to complete.


upload_to_bigquery(json_str, bigquery_benchmarking_table, job_config)

bigquery_upload_time_s = timeit.default_timer() - t

print("Send data to BigQuery in %s s" % bigquery_upload_time_s)
