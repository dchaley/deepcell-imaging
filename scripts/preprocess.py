#!/usr/bin/env python
"""
Script to preprocess an input image for a Mesmer model.

Reads input image from a URI (typically on cloud storage).

Writes preprocessed image to a URI (typically on cloud storage).
"""

import json
import logging
import timeit
import urllib
from datetime import datetime, timezone
from typing import Optional

import gs_fastcopy
import numpy as np
import smart_open
from pydantic import BaseModel, Field

import deepcell_imaging
from deepcell_imaging import gcp_logging, benchmark_utils, mesmer_app
from deepcell_imaging.utils.cmdline import get_task_arguments


class PreprocessArgs(BaseModel):
    image_uri: str = Field(
        title="Image URI",
        description="URI to input image npz file, containing an array named 'input_channels' by default (see --image-array-name)",
    )
    image_array_name: str = Field(
        default="input_channels",
        title="Image Array Name",
        description="Name of array in input image npz file. Default/blank: input_channels",
    )
    image_mpp: Optional[float] = Field(
        default=None,
        title="Image Microns Per Pixel",
        description="Microns per pixel of input image. Default/blank: use the model's mpp value.",
    )
    output_uri: str = Field(
        title="Output URI",
        description="Where to write preprocessed input npz file containing an array named 'image'",
    )
    benchmark_output_uri: Optional[str] = Field(
        default=None,
        title="Benchmark Output URI",
        description="Where to write preprocessing benchmarking data. Default/blank: don't write benchmarking data.",
    )


def main():
    deepcell_imaging.gcp_logging.initialize_gcp_logging()
    logger = logging.getLogger(__name__)

    args = get_task_arguments("preprocess", PreprocessArgs)

    image_uri = args.image_uri
    image_array_name = args.image_array_name
    image_mpp = args.image_mpp
    output_uri = args.output_uri
    benchmark_output_uri = args.benchmark_output_uri

    # This is hard-coded from the only model-id we support.
    model_input_shape = (None, 256, 256, 2)

    logger.info("Loading input")

    t = timeit.default_timer()
    with gs_fastcopy.read(image_uri) as input_file:
        with np.load(input_file) as loader:
            input_channels = loader[image_array_name]
    input_load_time_s = timeit.default_timer() - t

    logger.info("Loaded input in %s s", round(input_load_time_s, 2))

    logger.info("Preprocessing input")

    t = timeit.default_timer()

    try:
        preprocessed_image = mesmer_app.preprocess_image(
            model_input_shape, input_channels[np.newaxis, ...], image_mpp=image_mpp
        )
        success = True
    except Exception as e:
        success = False
        logger.error("Preprocessing failed with error: %s", e)

    preprocessing_time_s = timeit.default_timer() - t
    logger.info(
        "Preprocessed input in %s s; success: %s"
        % (round(preprocessing_time_s, 2), success)
    )

    if success:
        logger.info("Saving preprocessing output to %s" % output_uri)
        t = timeit.default_timer()

        with gs_fastcopy.write(output_uri) as output_writer:
            np.savez(output_writer, image=preprocessed_image)

        output_time_s = timeit.default_timer() - t

        logger.info("Saved output in %s s" % round(output_time_s, 2))
    else:
        logger.warning("Not saving failed preprocessing output.")
        output_time_s = 0.0

    # Gather & output timing information

    if benchmark_output_uri:
        gpu_info = benchmark_utils.get_gpu_info()

        parsed_url = urllib.parse.urlparse(image_uri)
        filename = parsed_url.path.split("/")[-2]

        # BigQuery datetimes don't have a timezone.
        benchmark_time = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()

        timing_info = {
            "input_file_id": filename,
            "numpy_size_mb": round(input_channels.nbytes / 1e6, 2),
            "pixels_m": input_channels.shape[0] * input_channels.shape[1],
            "benchmark_datetime_utc": benchmark_time,
            "preprocessing_instance_type": benchmark_utils.get_gce_instance_type(),
            "preprocessing_gpu_type": gpu_info[0],
            "preprocessing_num_gpus": gpu_info[1],
            "preprocessing_success": success,
            "preprocessing_peak_memory_gb": benchmark_utils.get_peak_memory_gb(),
            "preprocessing_is_preemptible": benchmark_utils.get_gce_is_preemptible(),
            "preprocessing_input_load_time_s": input_load_time_s,
            "preprocessing_time_s": preprocessing_time_s,
            "preprocessing_output_write_time_s": output_time_s,
        }

        with smart_open.open(benchmark_output_uri, "w") as benchmark_output_file:
            json.dump(timing_info, benchmark_output_file)

        logger.info("Wrote benchmarking data to %s" % benchmark_output_uri)


if __name__ == "__main__":
    main()
