#!/usr/bin/env python
"""
Script to preprocess an input image for a Mesmer model.

Reads input image from a URI (typically on cloud storage).

Writes preprocessed image to a URI (typically on cloud storage).
"""

import logging
import timeit
from typing import Optional

import gs_fastcopy
import numpy as np
import smart_open
from PIL import Image
from deepcell.utils.plot_utils import create_rgb_image, make_outline_overlay
from pydantic import BaseModel, Field

import deepcell_imaging
from deepcell_imaging import gcp_logging
from deepcell_imaging.utils.cmdline import get_task_arguments


class VisualizeArgs(BaseModel):
    image_uri: str = Field(
        title="Image URI",
        description="URI to input image npz file, containing an array named 'input_channels' by default (see --image_array_name)",
    )
    image_array_name: Optional[str] = Field(
        default="input_channels",
        title="Image Array Name",
        description="Name of array in input image npz file, default: input_channels",
    )
    predictions_uri: str = Field(
        title="Predictions URI",
        description="URI to image predictions npz file, containing an array named 'image'",
    )
    visualized_input_uri: str = Field(
        title="Visualized Input URI",
        description="Where to write visualized input png file.",
    )
    visualized_predictions_uri: str = Field(
        title="Visualized Predictions URI",
        description="Where to write visualized predictions png file.",
    )


def main():
    deepcell_imaging.gcp_logging.initialize_gcp_logging()
    logger = logging.getLogger(__name__)

    args = get_task_arguments("visualize", VisualizeArgs)

    image_uri = args.image_uri
    image_array_name = args.image_array_name
    predictions_uri = args.predictions_uri
    visualized_input_uri = args.visualized_input_uri
    visualized_predictions_uri = args.visualized_predictions_uri

    logger.info("Loading input")

    t = timeit.default_timer()
    with gs_fastcopy.read(image_uri) as file:
        with np.load(file) as loader:
            # An array of shape [height, width, channel] containing intensity of nuclear & membrane channels
            input_channels = loader[image_array_name]
    input_load_time_s = timeit.default_timer() - t

    logger.info("Loaded input in %s s" % input_load_time_s)

    logger.info("Loading predictions")

    t = timeit.default_timer()
    with gs_fastcopy.read(predictions_uri) as file:
        with np.load(file) as loader:
            # An array of shape [height, width, channel] containing intensity of nuclear & membrane channels
            predictions = loader["image"]
    predictions_load_time_s = timeit.default_timer() - t

    logger.info("Loaded predictions in %s s" % predictions_load_time_s)

    nuclear_color = "green"
    membrane_color = "blue"

    logger.info("Rendering input to %s" % visualized_input_uri)

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

    logger.info("Rendered input in %s s" % input_render_time_s)

    logger.info("Rendering predictions to %s" % visualized_predictions_uri)

    t = timeit.default_timer()
    overlay_data = make_outline_overlay(
        rgb_data=input_rgb[np.newaxis, ...],
        predictions=predictions[np.newaxis, ...],
    )[0]

    # The rgb values are 0..1, so normalize to 0..255
    im = Image.fromarray((overlay_data * 255).astype(np.uint8))
    with smart_open.open(visualized_predictions_uri, "wb") as predictions_png_file:
        im.save(predictions_png_file, mode="RGB")

    predictions_render_time_s = timeit.default_timer() - t

    logger.info("Rendered predictions in %s s" % predictions_render_time_s)


if __name__ == "__main__":
    main()
