#!/usr/bin/env python
"""
Script to preprocess an input image for a Mesmer model.

Reads input image from a URI (typically on cloud storage).

Writes preprocessed image to a URI (typically on cloud storage).
"""

import argparse
from deepcell_imaging import mesmer_app
import numpy as np
import smart_open
import timeit

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

args = parser.parse_args()

image_uri = args.image_uri
image_array_name = args.image_array_name
image_mpp = args.image_mpp
output_uri = args.output_uri

# This is hard-coded from the only model-id we support.
model_input_shape = (None, 256, 256, 2)

print("Loading input")

t = timeit.default_timer()
with smart_open.open(image_uri, "rb") as image_file:
    with np.load(image_file) as loader:
        # An array of shape [height, width, channel] containing intensity of nuclear & membrane channels
        input_channels = loader[image_array_name]
input_load_time_s = timeit.default_timer() - t

print("Loaded input in %s s" % input_load_time_s)

print("Preprocessing input")

t = timeit.default_timer()
preprocessed_image = mesmer_app.preprocess_image(
    model_input_shape,
    input_channels[np.newaxis, ...],
    image_mpp=image_mpp
)
preprocessing_time_s = timeit.default_timer() - t

print("Preprocessed input in %s s" % preprocessing_time_s)

print("Saving output")

t = timeit.default_timer()
with smart_open.open(output_uri, "wb") as output_file:
    np.savez_compressed(output_file, image=preprocessed_image)
output_time_s = timeit.default_timer() - t

print("Saved output in %s s" % output_time_s)
