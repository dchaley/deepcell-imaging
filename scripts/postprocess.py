#!/usr/bin/env python
"""
Script to postprocess raw Mesmer model output back to an image.

Reads raw predictions from a URI (typically on cloud storage).

Writes segmented image npz to a URI (typically on cloud storage).
"""

import argparse
from deepcell_imaging import mesmer_app
import numpy as np
import smart_open
import timeit

parser = argparse.ArgumentParser("preprocess")

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

args = parser.parse_args()

raw_predictions_uri = args.raw_predictions_uri
input_rows = args.input_rows
input_cols = args.input_cols
output_uri = args.output_uri

print("Loading raw predictions")

t = timeit.default_timer()
with smart_open.open(raw_predictions_uri, "rb") as raw_predictions_file:
    with np.load(raw_predictions_file) as loader:
        # An array of shape [height, width, channel] containing intensity of nuclear & membrane channels
        raw_predictions = {
            'whole-cell': [loader["arr_0"], loader["arr_1"]],
            'nuclear': [loader["arr_2"], loader["arr_3"]],
        }
raw_predictions_load_time_s = timeit.default_timer() - t

print("Loaded raw predictions in %s s" % raw_predictions_load_time_s)

print("Postprocessing raw predictions")

t = timeit.default_timer()
predictions = mesmer_app.postprocess(
    raw_predictions,
    (1, input_rows, input_cols, 2),
    compartment='whole-cell'
)

postprocessing_time_s = timeit.default_timer() - t

print("Postprocessed raw predictions in %s s" % postprocessing_time_s)

print("Saving output")

t = timeit.default_timer()
with smart_open.open(output_uri, "wb") as output_file:
    np.savez_compressed(output_file, image=predictions)
output_time_s = timeit.default_timer() - t

print("Saved output in %s s" % output_time_s)
