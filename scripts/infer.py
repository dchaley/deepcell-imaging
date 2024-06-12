#!/usr/bin/env python
"""
Script to run inference on a preprocessed input image for a Mesmer model.

Reads preprocessed image from a URI (typically on cloud storage).

Writes model output to a URI (typically on cloud storage).

The output npz has 4 arrays in it: names: arr_0, arr_1, arr_2, arr_3.
# arr_0 and arr_1 correspond to whole-cell.
# arr_2 and arr_3 correspond to nuclear.

batch_size : string
"""

import argparse
from deepcell_imaging import cached_open, mesmer_app
import numpy as np
import os
import smart_open
import tensorflow as tf
import timeit

parser = argparse.ArgumentParser("preprocess")

parser.add_argument(
    "--image_uri",
    help="URI to preprocessed image npz file, containing an array named 'image'",
    type=str,
    required=True,
)
parser.add_argument(
    "--batch_size",
    help="Optional integer representing batch size to use for inference. Default is 16.",
    type=int,
    required=False,
    default=16
)
parser.add_argument(
    "--output_uri",
    help="Where to write model output npz file containing arr_0, arr_1, arr_2, arr_3",
    type=str,
    required=True,
)

args = parser.parse_args()

image_uri = args.image_uri
batch_size = args.batch_size
output_uri = args.output_uri

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

print("Loaded model in %s s" % model_load_time_s)

print("Loading preprocessed image")

t = timeit.default_timer()
with smart_open.open(image_uri, "rb") as image_file:
    with np.load(image_file) as loader:
        # An array of shape [height, width, channel] containing intensity of nuclear & membrane channels
        preprocessed_image = loader["image"]
input_load_time_s = timeit.default_timer() - t

print("Loaded preprocessed image in %s s" % input_load_time_s)

print("Running inference")

t = timeit.default_timer()
model_output = mesmer_app.infer(
    model,
    preprocessed_image,
    batch_size=batch_size,
)
infer_time_s = timeit.default_timer() - t

print("Ran inference in %s s" % infer_time_s)

print("Saving inference output to %s" % output_uri)

t = timeit.default_timer()
with smart_open.open(output_uri, "wb") as output_file:
    np.savez_compressed(
        output_file,
        arr_0=model_output['whole-cell'][0],
        arr_1=model_output['whole-cell'][1],
        arr_2=model_output['nuclear'][0],
        arr_3=model_output['nuclear'][1],
    )
output_time_s = timeit.default_timer() - t

print("Saved output in %s s" % output_time_s)
