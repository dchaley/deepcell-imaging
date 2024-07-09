#!/usr/bin/env python
"""
Script to postprocess raw Mesmer model output back to an image.

Reads raw predictions from a URI (typically on cloud storage).

Writes segmented image npz to a URI (typically on cloud storage).
"""

import argparse
from deepcell_imaging import benchmark_utils, mesmer_app
import gs_fastcopy
import json
import numpy as np
import smart_open
import tifffile
import timeit


def main():
    parser = argparse.ArgumentParser("postprocess")

    parser.add_argument(
        "--raw_predictions_uri",
        help="URI to model output npz file, containing 4 arrays: arr_0, arr_1, arr_2, arr_3",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--input_rows",
        help="Number of rows in the input image.",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--input_cols",
        help="Number of columns in the input image.",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--compartment",
        help="Compartment to segment. One of 'whole-cell' (default) or 'nuclear' or 'both'.",
        type=str,
        required=False,
        default="whole-cell",
    )
    parser.add_argument(
        "--output_uri",
        help="Where to write postprocessed segment predictions npz file containing an array named 'image'",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--tiff_output_uri",
        help="Where to write postprocessed segment predictions TIFF file containing a segment number for each pixel",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--benchmark_output_uri",
        help="Where to write preprocessing benchmarking data.",
        type=str,
        required=False,
    )

    args = parser.parse_args()

    raw_predictions_uri = args.raw_predictions_uri
    input_rows = args.input_rows
    input_cols = args.input_cols
    compartment = args.compartment
    output_uri = args.output_uri
    tiff_output_uri = args.tiff_output_uri
    benchmark_output_uri = args.benchmark_output_uri

    print("Loading raw predictions")

    t = timeit.default_timer()

    with gs_fastcopy.read(raw_predictions_uri) as raw_predictions_file:
        with np.load(raw_predictions_file) as loader:
            # An array of shape [height, width, channel] containing intensity of nuclear & membrane channels
            raw_predictions = {
                "whole-cell": [loader["arr_0"], loader["arr_1"]],
                "nuclear": [loader["arr_2"], loader["arr_3"]],
            }

    raw_predictions_load_time_s = timeit.default_timer() - t

    print("Loaded raw predictions in %s s" % round(raw_predictions_load_time_s, 2))

    print("Postprocessing raw predictions")

    t = timeit.default_timer()
    try:
        segmentation = mesmer_app.postprocess(
            raw_predictions, (1, input_rows, input_cols, 2), compartment=compartment
        )
        success = True
    except Exception as e:
        print("Postprocessing failed with error: %s" % e)
        success = False

    postprocessing_time_s = timeit.default_timer() - t

    print(
        "Postprocessed raw predictions in %s s; success: %s"
        % (round(postprocessing_time_s, 2), success)
    )

    if success:
        print("Saving postprocessed npz output to %s" % output_uri)
        t = timeit.default_timer()
        with gs_fastcopy.write(output_uri) as output_writer:
            np.savez(output_writer, image=segmentation)

        output_time_s = timeit.default_timer() - t
        print("Saved output in %s s" % round(output_time_s, 2))

        if tiff_output_uri:
            print("Saving postprocessed TIFF output to %s" % tiff_output_uri)
            t = timeit.default_timer()
            segments_int32 = segmentation.astype(np.int32)
            with gs_fastcopy.write(tiff_output_uri) as output_writer:
                tifffile.imwrite(output_writer, segments_int32)

            tiff_output_time_s = timeit.default_timer() - t
            print("Saved tiff output in %s s" % round(tiff_output_time_s, 2))
    else:
        print("Not saving failed postprocessing output.")
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

        print("Wrote benchmarking data to %s" % benchmark_output_uri)


if __name__ == "__main__":
    main()
