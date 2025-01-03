#!/usr/bin/env python
"""
Script to run prediction on a preprocessed input image for a Mesmer model.

Reads preprocessed image from a URI (typically on cloud storage).

Writes model output to a URI (typically on cloud storage).

The output npz has 4 arrays in it: names: arr_0, arr_1, arr_2, arr_3.
# arr_0 and arr_1 correspond to whole-cell.
# arr_2 and arr_3 correspond to nuclear.

batch_size : string
"""

import json
import logging
import os
import timeit

import gs_fastcopy
import numpy as np
import smart_open
import tensorflow as tf

from deepcell_imaging.patched_location import Location2D

import deepcell_imaging
from deepcell_imaging import (
    gcp_logging,
    benchmark_utils,
    cached_open,
    mesmer_app,
)
from deepcell_imaging.gcp_batch_jobs.types import PredictArgs
from deepcell_imaging.utils.cmdline import get_task_arguments


def main():
    deepcell_imaging.gcp_logging.initialize_gcp_logging()
    logger = logging.getLogger(__name__)

    args, env_config = get_task_arguments("predict", PredictArgs)

    image_uri = args.image_uri
    batch_size = args.batch_size
    output_uri = args.output_uri
    benchmark_output_uri = args.benchmark_output_uri

    model_remote_path = args.model_path
    model_hash = args.model_hash

    logger.info("Fetching model from: %s" % model_remote_path)

    model_file_name = os.path.basename(model_remote_path)
    model_file_extension = os.path.splitext(model_file_name)[1]

    downloaded_file_path = cached_open.get_file(
        model_file_name,
        model_remote_path,
        file_hash=model_hash,
        extract=(model_file_extension in [".tgz", ".gz", ".zip"]),
        cache_subdir="models",
    )

    # NOTE: what we really mean to do here is identify the extracted
    # contents of the archive, if we downloaded an archive. The tricky
    # thing is that we don't know the name in advance, it depends on
    # the archive. What we *should* do is:
    # - look inside the archive to get the path
    #   - if so, all files must have the same base path
    # - require the model user (aka, here, predict.py) to know the
    #   base path. This means, predict.py callers also need to know.

    # For now, we'll hard-code this for the models we support:
    # - (classic) .tar.gz which removes the .tar.gz extension
    # - (new) .keras which doesn't extract
    model_path = downloaded_file_path.removesuffix(".tar.gz")

    logger.info("Loading model from: {}".format(model_path))

    t = timeit.default_timer()

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"Location2D": Location2D},
    )
    model_load_time_s = timeit.default_timer() - t

    logger.info("Loaded model in %s s" % round(model_load_time_s, 2))

    logger.info("Loading preprocessed image")

    t = timeit.default_timer()

    with gs_fastcopy.read(image_uri) as image_file:
        with np.load(image_file) as loader:
            preprocessed_image = loader["image"]
    input_load_time_s = timeit.default_timer() - t

    logger.info("Loaded preprocessed image in %s s" % round(input_load_time_s, 2))

    logger.info("Running prediction")

    t = timeit.default_timer()
    try:
        model_output = mesmer_app.predict(
            model,
            preprocessed_image,
            batch_size=batch_size,
        )
        success = True
    except Exception as e:
        success = False
        logger.error("Prediction failed with error: %s" % e)

    predict_time_s = timeit.default_timer() - t

    logger.info(
        "Ran prediction in %s s; success: %s" % (round(predict_time_s, 2), success)
    )

    if success:
        logger.info("Saving raw predictions output to %s" % output_uri)

        t = timeit.default_timer()
        with gs_fastcopy.write(output_uri) as output_writer:
            np.savez(
                output_writer,
                arr_0=model_output["whole-cell"][0],
                arr_1=model_output["whole-cell"][1],
                arr_2=model_output["nuclear"][0],
                arr_3=model_output["nuclear"][1],
            )
        output_time_s = timeit.default_timer() - t

        logger.info("Saved output in %s s" % round(output_time_s, 2))
    else:
        logger.warning("Not saving failed prediction output.")
        output_time_s = 0.0

    # Gather & output timing information

    if benchmark_output_uri:
        gpu_info = benchmark_utils.get_gpu_info()

        timing_info = {
            "prediction_instance_type": benchmark_utils.get_gce_instance_type(),
            "prediction_gpu_type": gpu_info[0],
            "prediction_num_gpus": gpu_info[1],
            "prediction_success": success,
            "prediction_peak_memory_gb": benchmark_utils.get_peak_memory_gb(),
            "prediction_is_preemptible": benchmark_utils.get_gce_is_preemptible(),
            "prediction_model_load_time_s": model_load_time_s,
            "prediction_input_load_time_s": input_load_time_s,
            "prediction_batch_size": batch_size,
            "prediction_time_s": predict_time_s,
            "prediction_output_write_time_s": output_time_s,
        }

        with smart_open.open(benchmark_output_uri, "w") as benchmark_output_file:
            json.dump(timing_info, benchmark_output_file)

        logger.info("Wrote benchmarking data to %s" % benchmark_output_uri)


if __name__ == "__main__":
    main()
