#!/usr/bin/env python
"""
Script to gather & submit benchmarking data for a segmentation run.

Reads each step's benchmarking from URIs (typically on cloud storage).

Then adds in some extra information, and uploads to BigQuery.
"""

import io
import json
import logging
import timeit
import typing

import smart_open
from google.cloud import bigquery
from tenacity import retry, retry_if_exception_message, wait_random_exponential

import deepcell_imaging
from deepcell_imaging import benchmark_utils, gcp_logging
from deepcell_imaging.gcp_batch_jobs.types import GatherBenchmarkArgs
from deepcell_imaging.utils.cmdline import get_task_arguments


def main():
    deepcell_imaging.gcp_logging.initialize_gcp_logging()
    logger = logging.getLogger(__name__)

    args = get_task_arguments("gather-benchmark", GatherBenchmarkArgs)

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

    json_str = json.dumps(benchmarking_data)

    @retry(
        wait=wait_random_exponential(multiplier=1, max=60),
        retry=retry_if_exception_message(match=".*403 Exceeded rate limits.*"),
    )
    def upload_to_bigquery(upload_str: str, table_id, bq_job_config):
        upload_io: typing.IO = io.StringIO(upload_str)
        load_job = bq_client.load_table_from_file(
            upload_io, table_id, job_config=bq_job_config
        )
        load_job.result()  # Waits for the job to complete.

    upload_to_bigquery(json_str, bigquery_benchmarking_table, job_config)

    bigquery_upload_time_s = timeit.default_timer() - t

    logger.info("Send data to BigQuery in %s s" % bigquery_upload_time_s)


if __name__ == "__main__":
    main()
