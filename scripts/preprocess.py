#!/usr/bin/env python
"""
Script to preprocess an input image for a Mesmer model.

Reads input image from a URI (typically on cloud storage).

Writes preprocessed image to a URI (typically on cloud storage).
"""

import argparse
from datetime import datetime, timezone
from deepcell_imaging import benchmark_utils, gcloud_storage_utils, mesmer_app
import json
import numpy as np
import smart_open
import timeit
import urllib

parser = argparse.ArgumentParser("preprocess")

parser.add_argument(
    "--image_uri",
    help="URI to input image npz file, containing an array named 'input_channels' by default (see --image-array-name)",
    type=str,
    required=True,
)
parser.add_argument(
    "--image_array_name",
    help="Name of array in input image npz file, default input_channels",
    type=str,
    required=False,
    default="input_channels",
)
parser.add_argument(
    "--image_mpp",
    help="Optional float representing microns per pixel of input image. Leave blank to use model's mpp",
    type=float,
    required=False,
)
parser.add_argument(
    "--output_uri",
    help="Where to write preprocessed input npz file containing an array named 'image'",
    type=str,
    required=True,
)
parser.add_argument(
    "--benchmark_output_uri",
    help="Where to write preprocessing benchmarking data.",
    type=str,
    required=False,
)

args = parser.parse_args()

image_uri = args.image_uri
image_array_name = args.image_array_name
image_mpp = args.image_mpp
output_uri = args.output_uri
benchmark_output_uri = args.benchmark_output_uri

# This is hard-coded from the only model-id we support.
model_input_shape = (None, 256, 256, 2)

print("Loading input")

t = timeit.default_timer()
with np.load(gcloud_storage_utils.fetch_file(image_uri)) as loader:
    input_channels = loader[image_array_name]

input_load_time_s = timeit.default_timer() - t

print("Loaded input in %s s" % round(input_load_time_s, 2))

print("Preprocessing input")

t = timeit.default_timer()

try:
    preprocessed_image = mesmer_app.preprocess_image(
        model_input_shape,
        input_channels[np.newaxis, ...],
        image_mpp=image_mpp
    )
    success = True
except Exception as e:
    success = False
    print("Preprocessing failed with error: %s" % e)

preprocessing_time_s = timeit.default_timer() - t
print("Preprocessed input in %s s; success: %s" % (round(preprocessing_time_s, 2), success))

if success:
    print("Saving preprocessing output to %s" % output_uri)
    t = timeit.default_timer()

    gcloud_storage_utils.write_npz_file(output_uri, image=preprocessed_image)

    output_time_s = timeit.default_timer() - t

    print("Saved output in %s s" % round(output_time_s, 2))
else:
    print("Not saving failed preprocessing output.")
    output_time_s = 0.0

# Gather & output timing information

if benchmark_output_uri:
    gpu_info = benchmark_utils.get_gpu_info()

    parsed_url = urllib.parse.urlparse(image_uri)
    filename = parsed_url.path.split("/")[-2]

    # BigQuery datetimes don't have a timezone.
    benchmark_time = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()

    timing_info = {
        "input_file_id": image_uri,
        "numpy_size_mb": round(input_channels.nbytes / 1e6, 2),
        "pixels_m": input_channels.shape[0] * input_channels.shape[1],
        "benchmark_datetime_utc": benchmark_time,
        "preprocessing_instance_type": benchmark_utils.get_gce_instance_type(),
        "preprocessing_gpu_type": gpu_info[0],
        "preprocessing_num_gpus": gpu_info[1],
        "preprocessing_success": success,
        "preprocessing_peak_memory_gb": benchmark_utils.get_peak_memory(),
        "preprocessing_is_preemptible": benchmark_utils.get_gce_is_preemptible(),
        "preprocessing_input_load_time_s": input_load_time_s,
        "preprocessing_time_s": preprocessing_time_s,
        "preprocessing_output_write_time_s": output_time_s,
    }

    with smart_open.open(benchmark_output_uri, "w") as benchmark_output_file:
        json.dump(timing_info, benchmark_output_file)

    print("Wrote benchmarking data to %s" % benchmark_output_uri)
