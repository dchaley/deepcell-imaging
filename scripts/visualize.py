#!/usr/bin/env python
"""
Script to preprocess an input image for a Mesmer model.

Reads input image from a URI (typically on cloud storage).

Writes preprocessed image to a URI (typically on cloud storage).
"""

import argparse
from deepcell.utils.plot_utils import create_rgb_image, make_outline_overlay
from deepcell_imaging import gcloud_storage_utils
import numpy as np
from PIL import Image
import smart_open
import timeit

parser = argparse.ArgumentParser("visualize")

parser.add_argument(
    "--image_uri",
    help="URI to input image npz file, containing an array named 'input_channels' by default (see --image-array-name)",
    type=str,
    required=True,
)
parser.add_argument(
    "--image_array_name",
    help="Name of array in input image npz file, default: input_channels",
    type=str,
    required=False,
    default="input_channels",
)
parser.add_argument(
    "--predictions_uri",
    help="URI to image predictions npz file, containing an array named 'image'",
    type=str,
    required=True,
)
parser.add_argument(
    "--visualized_input_uri",
    help="Where to write visualized input png file.",
    type=str,
    required=True,
)
parser.add_argument(
    "--visualized_predictions_uri",
    help="Where to write visualized predictions png file.",
    type=str,
    required=True,
)

args = parser.parse_args()

image_uri = args.image_uri
image_array_name = args.image_array_name
predictions_uri = args.predictions_uri
visualized_input_uri = args.visualized_input_uri
visualized_predictions_uri = args.visualized_predictions_uri

print("Loading input")

t = timeit.default_timer()
with gcloud_storage_utils.reader(image_uri) as file:
    with np.load(file) as loader:
        # An array of shape [height, width, channel] containing intensity of nuclear & membrane channels
        input_channels = loader[image_array_name]
input_load_time_s = timeit.default_timer() - t

print("Loaded input in %s s" % input_load_time_s)

print("Loading predictions")

t = timeit.default_timer()
with gcloud_storage_utils.reader(predictions_uri) as file:
    with np.load(file) as loader:
        # An array of shape [height, width, channel] containing intensity of nuclear & membrane channels
        predictions = loader["image"]
predictions_load_time_s = timeit.default_timer() - t

print("Loaded predictions in %s s" % predictions_load_time_s)

nuclear_color = "green"
membrane_color = "blue"

print("Rendering input to %s" % visualized_input_uri)

t = timeit.default_timer()

# Create rgb overlay of image data for visualization
# Note that this normalizes the values from "whatever" to rgb range 0..1
input_rgb = create_rgb_image(
    input_channels[np.newaxis, ...], channel_colors=[nuclear_color, membrane_color]
)[0]

# The png needs to normalize rgb values from 0..1, so normalize to 0..255
im = Image.fromarray((input_rgb * 255).astype(np.uint8))
with smart_open.open(visualized_input_uri, "wb") as input_png_file:
    im.save(input_png_file, mode="RGB")

input_render_time_s = timeit.default_timer() - t

print("Rendered input in %s s" % input_render_time_s)

print("Rendering predictions to %s" % visualized_predictions_uri)

t = timeit.default_timer()
overlay_data = make_outline_overlay(
    rgb_data=input_rgb[np.newaxis, ...],
    predictions=predictions,
)[0]

# The rgb values are 0..1, so normalize to 0..255
im = Image.fromarray((overlay_data * 255).astype(np.uint8))
with smart_open.open(
        visualized_predictions_uri, "wb"
) as predictions_png_file:
    im.save(predictions_png_file, mode="RGB")

predictions_render_time_s = timeit.default_timer() - t

print("Rendered predictions in %s s" % predictions_render_time_s)
