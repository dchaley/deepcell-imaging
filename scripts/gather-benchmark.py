#!/usr/bin/env python
"""
Script to preprocess an input image for a Mesmer model.

Reads input image from a URI (typically on cloud storage).

Writes preprocessed image to a URI (typically on cloud storage).
"""

import argparse
import io
import json
import logging
import timeit
from typing import Optional

import smart_open
from google.cloud import bigquery
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_message, wait_random_exponential

import deepcell_imaging
from deepcell_imaging import benchmark_utils, gcp_logging
from deepcell_imaging.gcp_batch_jobs import get_batch_indexed_task


class GatherBenchmarkArgs(BaseModel):
    preprocess_benchmarking_uri: str
    prediction_benchmarking_uri: str
    postprocess_benchmarking_uri: str
    bigquery_benchmarking_table: Optional[str] = None


def main():
    parser = argparse.ArgumentParser("preprocess")

    deepcell_imaging.gcp_logging.initialize_gcp_logging()
    logger = logging.getLogger(__name__)

    parser.add_argument(
        "--tasks_spec_uri",
        help="URI to a JSON file containing a list of task parameters",
        type=str,
        required=False,
    )

    parsed_args, args_remainder = parser.parse_known_args()

    if parsed_args.tasks_spec_uri:
        if len(args_remainder) > 0:
            raise ValueError("Either pass --tasks_spec_uri alone, or not at all")

        args = get_batch_indexed_task(parsed_args.tasks_spec_uri, GatherBenchmarkArgs)
    else:
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

        parsed_args = parser.parse_args(args_remainder)
        kwargs = {k: v for k, v in vars(parsed_args).items() if v is not None}
        args = GatherBenchmarkArgs(**kwargs)

    preprocess_benchmarking_uri = args.preprocess_benchmarking_uri
    prediction_benchmarking_uri = args.prediction_benchmarking_uri
    postprocess_benchmarking_uri = args.postprocess_benchmarking_uri
    bigquery_benchmarking_table = args.bigquery_benchmarking_table

    if not bigquery_benchmarking_table:
        logger.info("Nothing to do; empty bigquery_benchmarking_table")
        exit()

    benchmarking_data = {
        "cloud_region": benchmark_utils.get_gce_region(),
    }

    logger.info("Loading benchmarking data")

    t = timeit.default_timer()

    for data_uri in [
        preprocess_benchmarking_uri,
        prediction_benchmarking_uri,
        postprocess_benchmarking_uri,
    ]:
        with smart_open.open(data_uri, "r") as data_file:
            data = json.load(data_file)
            benchmarking_data.update(data)

    data_load_time_s = timeit.default_timer() - t

    logger.info("Loaded benchmarking data in %s s" % data_load_time_s)

    # Update the overall success to the logical AND of the individual steps
    benchmarking_data["success"] = (
        benchmarking_data["preprocessing_success"]
        and benchmarking_data["prediction_success"]
        and benchmarking_data["postprocessing_success"]
    )

    logger.info("Sending data to BigQuery")

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

    logger.info("Send data to BigQuery in %s s" % bigquery_upload_time_s)


if __name__ == "__main__":
    main()
