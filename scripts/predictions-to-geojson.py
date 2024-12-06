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
    output_uri = args.output_uri

    logger.info("Loading predictions")

    t = timeit.default_timer()

    with gs_fastcopy.read(predictions_uri) as predictions_file:
        with np.load(predictions_file) as loader:
            # An array of shape [height, width, 1] containing intensity of nuclear & membrane channels
            predictions = np.squeeze(loader["image"])

    logger.info("Loaded predictions in %s s" % round(timeit.default_timer() - t, 2))

    logger.info("Getting prediction shapes")

    t = timeit.default_timer()

    max_int32 = 2**31 - 1
    if predictions.max() > max_int32:
        raise ValueError(
            "Can only handle up to int32=%d unique labels, not %d"
            % (max_int32, predictions.max())
        )

    mask = predictions != 0
    shapes = list(
        features.shapes(
            predictions.astype(np.int32, copy=True),
            mask=mask,
            connectivity=8,
        )
    )

    logger.info("Got prediction shapes in %s s" % round(timeit.default_timer() - t, 2))

    logger.info("Writing GeoJSON shapes to %s" % output_uri)

    t = timeit.default_timer()

    geojson_shapes = [x[0] for x in shapes]
    with gs_fastcopy.write(output_uri) as output_writer:
        output_writer.write(json.dumps(geojson_shapes).encode())

    output_json_time_s = timeit.default_timer() - t
    logger.info("Wrote GeoJSON shapes in %s s" % round(output_json_time_s, 2))


if __name__ == "__main__":
    main()
