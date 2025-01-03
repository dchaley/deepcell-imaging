#!/usr/bin/env python
"""
Script to generate GeoJSON shapes from a segmentation mask.

Reads post-processed predictions from a URI (typically on cloud storage).

Writes a JSON file containing GeoJSON shapes to a URI (typically on cloud storage).
"""

import json
import logging
import timeit

import gs_fastcopy
import numpy as np
from rasterio import features

import deepcell_imaging
from deepcell_imaging import gcp_logging
from deepcell_imaging.gcp_batch_jobs.types import PredictionsToGeoJsonArgs
from deepcell_imaging.utils.cmdline import get_task_arguments


def main():
    deepcell_imaging.gcp_logging.initialize_gcp_logging()
    logger = logging.getLogger(__name__)

    args, env_config = get_task_arguments(
        "predictions-to-geojson", PredictionsToGeoJsonArgs
    )

    predictions_uri = args.predictions_uri

    logger.info("Loading predictions")

    t = timeit.default_timer()

    with gs_fastcopy.read(predictions_uri) as predictions_file:
        with np.load(predictions_file) as loader:
            # An array of shape [height, width, 1-2] containing intensity of nuclear & membrane channels
            predictions = loader["image"]

    logger.info("Loaded predictions in %s s" % round(timeit.default_timer() - t, 2))

    max_int32 = 2**31 - 1
    if predictions.max() > max_int32:
        raise ValueError(
            "Can only handle up to int32=%d unique labels, not %d"
            % (max_int32, predictions.max())
        )

    predictions = predictions.astype(np.int32, copy=True)

    logger.info("Writing whole cell predictions")
    write_shapes(np.squeeze(predictions[..., 0]), args.whole_cell_output_uri)

    logger.info("Writing nucleus predictions")
    write_shapes(np.squeeze(predictions[..., 1]), args.nucleus_output_uri)


def write_shapes(predictions, output_uri):
    logger = logging.getLogger(__name__)

    logger.info("Detecting predicted shapes")
    t = timeit.default_timer()

    # Don't put a shape around the background.
    mask = predictions != 0

    # Extract the polygon from the result pair.
    shapes = list(x[0] for x in features.shapes(predictions, mask, connectivity=8))

    # Release references to free memory if needed
    predictions = mask = None

    logger.info(
        "Detected %s shapes in %s s",
        len(shapes),
        round(timeit.default_timer() - t, 2),
    )

    logger.info("Writing shapes to %s" % output_uri)
    t = timeit.default_timer()

    with gs_fastcopy.write(output_uri) as output_writer:
        for shape in shapes:
            output_writer.write(json.dumps(shape).encode())
            output_writer.write(b"\n")

    output_json_time_s = timeit.default_timer() - t
    logger.info("Wrote %s shapes in %s s", len(shapes), round(output_json_time_s, 2))


if __name__ == "__main__":
    main()
