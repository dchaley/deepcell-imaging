#!/usr/bin/env python
"""
Script to postprocess raw Mesmer model output back to an image.

Reads raw predictions from a URI (typically on cloud storage).

Writes segmented image npz to a URI (typically on cloud storage).
"""

import json
import logging
import timeit

import gs_fastcopy
import numpy as np
import smart_open
import tifffile

import deepcell_imaging
from deepcell_imaging import gcp_logging, benchmark_utils, mesmer_app
from deepcell_imaging.gcp_batch_jobs.types import PostprocessArgs
from deepcell_imaging.utils.cmdline import get_task_arguments


def main():
    deepcell_imaging.gcp_logging.initialize_gcp_logging()
    logger = logging.getLogger(__name__)

    args, env_config = get_task_arguments("postprocess", PostprocessArgs)

    raw_predictions_uri = args.raw_predictions_uri
    input_rows = args.input_rows
    input_cols = args.input_cols
    compartment = args.compartment
    output_uri = args.output_uri
    tiff_output_uri = args.tiff_output_uri
    benchmark_output_uri = args.benchmark_output_uri

    logger.info("Loading raw predictions")

    t = timeit.default_timer()

    with gs_fastcopy.read(raw_predictions_uri) as raw_predictions_file:
        with np.load(raw_predictions_file) as loader:
            # An array of shape [height, width, channel] containing intensity of nuclear & membrane channels
            raw_predictions = {
                "whole-cell": [loader["arr_0"], loader["arr_1"]],
                "nuclear": [loader["arr_2"], loader["arr_3"]],
            }

    raw_predictions_load_time_s = timeit.default_timer() - t

    logger.info(
        "Loaded raw predictions in %s s" % round(raw_predictions_load_time_s, 2)
    )

    logger.info("Postprocessing raw predictions")

    t = timeit.default_timer()
    try:
        segmentation = mesmer_app.postprocess(
            raw_predictions, (1, input_rows, input_cols, 2), compartment=compartment
        )
        success = True
    except Exception as e:
        logger.error("Postprocessing failed with error: %s" % e)
        success = False

    postprocessing_time_s = timeit.default_timer() - t

    logger.info(
        "Postprocessed raw predictions in %s s; success: %s"
        % (round(postprocessing_time_s, 2), success)
    )

    if success:
        logger.info("Saving postprocessed npz output to %s" % output_uri)
        t = timeit.default_timer()
        with gs_fastcopy.write(output_uri) as output_writer:
            np.savez(output_writer, image=segmentation)

        output_time_s = timeit.default_timer() - t
        logger.info("Saved output in %s s" % round(output_time_s, 2))

        if tiff_output_uri:
            logger.info("Saving postprocessed TIFF output to %s" % tiff_output_uri)
            t = timeit.default_timer()
            segments_int32 = segmentation.astype(np.int32)
            with gs_fastcopy.write(tiff_output_uri) as output_writer:
                tifffile.imwrite(output_writer, segments_int32)

            tiff_output_time_s = timeit.default_timer() - t
            logger.info("Saved tiff output in %s s" % round(tiff_output_time_s, 2))
    else:
        logger.warning("Not saving failed postprocessing output.")
        output_time_s = 0.0

    # Gather & output timing information

    if benchmark_output_uri:
        gpu_info = benchmark_utils.get_gpu_info()

        timing_info = {
            "compartment": compartment,
            "postprocessing_instance_type": benchmark_utils.get_gce_instance_type(),
            "postprocessing_gpu_type": gpu_info[0],
            "postprocessing_num_gpus": gpu_info[1],
            "postprocessing_success": success,
            "postprocessing_peak_memory_gb": benchmark_utils.get_peak_memory_gb(),
            "postprocessing_is_preemptible": benchmark_utils.get_gce_is_preemptible(),
            "postprocessing_input_load_time_s": raw_predictions_load_time_s,
            "postprocessing_time_s": postprocessing_time_s,
            "postprocessing_output_write_time_s": output_time_s,
        }

        with smart_open.open(benchmark_output_uri, "w") as benchmark_output_file:
            json.dump(timing_info, benchmark_output_file)

        logger.info("Wrote benchmarking data to %s" % benchmark_output_uri)


if __name__ == "__main__":
    main()
