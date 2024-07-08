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

import argparse
from deepcell_imaging import (
    benchmark_utils,
    cached_open,
    mesmer_app,
)
import gs_fastcopy
import json
import numpy as np
import os
import smart_open
import tensorflow as tf
import timeit


def main():
    parser = argparse.ArgumentParser("predict")

    parser.add_argument(
        "--image_uri",
        help="URI to preprocessed image npz file, containing an array named 'image'",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--batch_size",
        help="Optional integer representing batch size to use for prediction. Default is 16.",
        type=int,
        required=False,
        default=16,
    )
    parser.add_argument(
        "--output_uri",
        help="Where to write model output npz file containing arr_0, arr_1, arr_2, arr_3",
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
    batch_size = args.batch_size
    output_uri = args.output_uri
    benchmark_output_uri = args.benchmark_output_uri

    # Hard-code remote path & hash based on model_id
    # THIS IS IN US-CENTRAL1
    # If you are running outside us-central1 you should make a copy to avoid egress.
    model_remote_path = "gs://genomics-data-public-central1/cellular-segmentation/vanvalenlab/deep-cell/vanvalenlab-tf-model-multiplex-downloaded-20230706/MultiplexSegmentation.tar.gz"
    model_hash = "a1dfbce2594f927b9112f23a0a1739e0"

    print("Loading model")

    downloaded_file_path = cached_open.get_file(
        "MultiplexSegmentation.tgz",
        model_remote_path,
        file_hash=model_hash,
        extract=True,
        cache_subdir="models",
    )
    # Remove the .tgz extension to get the model directory path
    model_path = os.path.splitext(downloaded_file_path)[0]

    print("Reading model from {}.".format(model_path))

    t = timeit.default_timer()
    model = tf.keras.models.load_model(model_path)
    model_load_time_s = timeit.default_timer() - t

    print("Loaded model in %s s" % round(model_load_time_s, 2))

    print("Loading preprocessed image")

    t = timeit.default_timer()

    with gs_fastcopy.read(image_uri) as image_file:
        with np.load(image_file) as loader:
            preprocessed_image = loader["image"]
    input_load_time_s = timeit.default_timer() - t

    print("Loaded preprocessed image in %s s" % round(input_load_time_s, 2))

    print("Running prediction")

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
        print("Prediction failed with error: %s" % e)

    predict_time_s = timeit.default_timer() - t

    print("Ran prediction in %s s; success: %s" % (round(predict_time_s, 2), success))

    if success:
        print("Saving raw predictions output to %s" % output_uri)

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

        print("Saved output in %s s" % round(output_time_s, 2))
    else:
        print("Not saving failed prediction output.")
        output_time_s = 0.0

    # Gather & output timing information

    if benchmark_output_uri:
        gpu_info = benchmark_utils.get_gpu_info()

        timing_info = {
            "prediction_instance_type": benchmark_utils.get_gce_instance_type(),
            "prediction_gpu_type": gpu_info[0],
            "prediction_num_gpus": gpu_info[1],
            "prediction_success": success,
            "prediction_peak_memory_gb": benchmark_utils.get_peak_memory(),
            "prediction_is_preemptible": benchmark_utils.get_gce_is_preemptible(),
            "prediction_model_load_time_s": model_load_time_s,
            "prediction_input_load_time_s": input_load_time_s,
            "prediction_batch_size": batch_size,
            "prediction_time_s": predict_time_s,
            "prediction_output_write_time_s": output_time_s,
        }

        with smart_open.open(benchmark_output_uri, "w") as benchmark_output_file:
            json.dump(timing_info, benchmark_output_file)

        print("Wrote benchmarking data to %s" % benchmark_output_uri)


if __name__ == "__main__":
    main()
