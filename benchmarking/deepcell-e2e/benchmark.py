import logging

try:
    import deepcell
except:
    logging.error("Couldn't load deepcell module. Run the setup notebook.")
    raise

# Imports

import argparse
import csv
from datetime import datetime, timezone
import deepcell
from deepcell.applications import Mesmer
from functools import reduce
import io
from itertools import groupby
import logging
import math
import numpy as np
import os
import platform
import psutil
import re
import resource
import smart_open
import sys
import tensorflow as tf
import timeit
import urllib.parse

# We need to import the cached_open module at the repository root.

# But because we're not shipping the deepcell_imaging module as a proper python package,
# we need to import the module a bit bluntly. This loads the module direct filename,
# relative to this notebook. Then, it's available like usual.

import importlib

file_path = "./deepcell_imaging/__init__.py"
module_name = "deepcell_imaging"
spec = importlib.util.spec_from_file_location(module_name, file_path)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)

from deepcell_imaging import cached_open

parser = argparse.ArgumentParser("benchmark")
parser.add_argument(
    "--custom_job_name",
    help="Vertex AI custom job display name",
    type=str,
    required=True,
)
args = parser.parse_args()

custom_job_name = args.custom_job_name


# This cell is a notebook 'parameters' cell.
# Utilities like Vertex AI or Papermill can override them for an execution.

input_channels_path = "gs://davids-genomics-data-public/cellular-segmentation/deep-cell/vanvalenlab-multiplex-20200810_tissue_dataset/mesmer-sample-3-dev/input_channels.npz"
# 'whole-cell' means predict the cell outlines
# 'nuclear' means predict the cell nucleus outlines
# 'both' does … both
# Set this to 'nuclear' for data that only has a nuclear channel
prediction_compartment = "whole-cell"

# The batch size controls how many tiles are predicted at once.
# Default: 4.
# DeepCell divides each image into tiles, predicts in batches,
# then untiles the result to return overall segmentat predictions.
batch_size = 4

# The location of the TensorFlow model to download.
# The tarball needs to extract to the directory: 'MultiplexSegmentation'
model_remote_path = "gs://davids-genomics-data-public/cellular-segmentation/deep-cell/vanvalenlab-tf-model-multiplex-downloaded-20230706/MultiplexSegmentation.tar.gz"
model_hash = "a1dfbce2594f927b9112f23a0a1739e0"

# The local cache location
model_path = os.path.expanduser("~") + "/.keras/models/MultiplexSegmentation"

predictions_output_path = None
# predictions_output_path = 'gs://davids-genomics-data-public/cellular-segmentation/deep-cell/vanvalenlab-multiplex-20200810_tissue_dataset/mesmer-sample-3-dev/segmentation_predictions.npz'

project_id = "deepcell-401920"
location = "us-west1"
# Set to None for local execution.
notebook_runtime_id = "updateme"

# Set this to True to visualize & save input & prediction.
# If true, the paths may be provided to save the visualizations to storage.
visualize = False
input_png_output_path = None
predictions_png_output_path = None
# input_png_output_path = 'gs://davids-genomics-data-public/cellular-segmentation/deep-cell/vanvalenlab-multiplex-20200810_tissue_dataset/mesmer-sample-3-dev/input.png'
# predictions_png_output_path = 'gs://davids-genomics-data-public/cellular-segmentation/deep-cell/vanvalenlab-multiplex-20200810_tissue_dataset/mesmer-sample-3-dev/segmentation_predictions.png'

# Model warm-up

logger = logging.getLogger()
old_level = logger.getEffectiveLevel()
logger.setLevel(logging.INFO)

downloaded_file_path = cached_open.get_file(
    "MultiplexSegmentation.tgz",
    model_remote_path,
    file_hash=model_hash,
    extract=True,
    cache_subdir="models",
)
# Remove the .tgz extension to get the model directory path
model_path = os.path.splitext(downloaded_file_path)[0]

logging.info("Loading model from {}.".format(model_path))
t = timeit.default_timer()
model = tf.keras.models.load_model(model_path)
logging.info("Loaded model in %s s" % (timeit.default_timer() - t))
app = Mesmer(model=model)

# Need to reset top-level logging for intercept to work.
# I dunno 🤷🏻‍♂️ There's probably a better way to do this...
logger.setLevel(old_level)
logging.basicConfig(force=True)
# %% md
# End-to-end Prediction

# %%
start_time = timeit.default_timer()

# Load inputs

t = timeit.default_timer()
with smart_open.open(input_channels_path, "rb") as input_channel_file:
    with np.load(input_channel_file) as loader:
        # An array of shape [height, width, channel] containing intensity of nuclear & membrane channels
        input_channels = loader["input_channels"]
input_load_time_s = timeit.default_timer() - t

print("Loaded input in %s s" % input_load_time_s)

# Generate predictions

## Intercept log (many shenanigans, such hacking)
logger = logging.getLogger()
old_level = logger.getEffectiveLevel()
logger.setLevel(logging.DEBUG)
logs_buffer = io.StringIO()
buffer_log_handler = logging.StreamHandler(logs_buffer)
logger.addHandler(buffer_log_handler)

## The actual prediction

t = timeit.default_timer()

try:
    # We're only predicting for 1 image, so extract the 1 image's predictions
    segmentation_predictions = app.predict(
        input_channels[np.newaxis, ...],
        image_mpp=0.5,
        compartment=prediction_compartment,
        batch_size=batch_size,
    )[0]
    prediction_success = True

except Exception as e:
    # The exception is nom-nom'd. Safe? We'll see 🤔
    prediction_success = False
    logger.error("Prediction exception: %s", e)

prediction_time_s = timeit.default_timer() - t
total_time_s = timeit.default_timer() - start_time

## Undo log intercept
logger.removeHandler(buffer_log_handler)
logger.setLevel(old_level)

## Wrap up
print("Prediction finished in %s s" % prediction_time_s)
print("Overall operation finished in %s s" % total_time_s)
# %%
# Parse the intercepted debug logs to extract step timing.

debug_logs = logs_buffer.getvalue()
pattern = r"(?sm)Pre-processed data with mesmer_preprocess in (.+?) s.*Model inference finished in (.+?) s.*Post-processed results with mesmer_postprocess in (.+?) s"
match = re.search(re.compile(pattern, re.MULTILINE), debug_logs)

if match:
    preprocess_time_s = float(match.group(1))
    inference_time_s = float(match.group(2))
    postprocess_time_s = float(match.group(3))
else:
    logger.warning("Couldn't parse step timings from debug_logs")
    preprocess_time_s = inference_time_s = postprocess_time_s = math.nan
# %% md
## Save predictions [optional]
# %%
if predictions_output_path:
    with smart_open.open(predictions_output_path, "wb") as predictions_file:
        np.savez_compressed(predictions_file, predictions=segmentation_predictions)
# %% md
# Visualization [optional]
# %% md
## Input channel visualization
# %%
if visualize:
    from deepcell.utils.plot_utils import create_rgb_image
    from PIL import Image

    nuclear_color = "green"
    membrane_color = "blue"

    # Create rgb overlay of image data for visualization
    # Note that this normalizes the values from "whatever" to rgb range 0..1
    input_rgb = create_rgb_image(
        input_channels[np.newaxis, ...], channel_colors=[nuclear_color, membrane_color]
    )[0]

    if input_png_output_path:
        # The png needs to normalize rgb values from 0..1, so normalize to 0..255
        im = Image.fromarray((input_rgb * 255).astype(np.uint8))
        with smart_open.open(input_png_output_path, "wb") as input_png_file:
            im.save(input_png_file, mode="RGB")
        del im

    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(input_rgb)
    ax.set_title("Input")
    plt.show()
# %% md
## Prediction overlay visualization
# %%
if visualize:
    from deepcell.utils.plot_utils import make_outline_overlay

    overlay_data = make_outline_overlay(
        rgb_data=input_rgb[np.newaxis, ...],
        predictions=segmentation_predictions[np.newaxis, ...],
    )[0]

    from PIL import Image

    if predictions_png_output_path:
        # The rgb values are 0..1, so normalize to 0..255
        im = Image.fromarray((overlay_data * 255).astype(np.uint8))
        with smart_open.open(predictions_png_output_path, "wb") as predictions_png_file:
            im.save(predictions_png_file, mode="RGB")
        del im

    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(overlay_data)
    ax.set_title("Predictions")
    plt.show()
# %% md
# Benchmark data
# %%
headers = [
    "Input file id",
    "Image size (MB)",
    "Pixels (M)",
    "Compartment",
    "Benchmark datetime (UTC)",
    "Machine type",
    "GPU type",
    "# GPUs",
    "Batch size",
    "Kernel",
    "Success?",
    "Total time (s)",
    "Peak memory (GB)",
    "Load time (s)",
    "Total prediction time (s)",
    "Prediction overhead (s)",
    "Predict preprocess time (s)",
    "Predict inference time (s)",
    "Predict postprocess time (s)",
    "deepcell-tf version",
]

parsed_url = urllib.parse.urlparse(input_channels_path)
filename = parsed_url.path.split("/")[-2]
image_size = round(input_channels.nbytes / 1000 / 1000, 2)

# Multiply x * y to get pixel count.
pixels = input_channels.shape[0] * input_channels.shape[1]

import ipykernel

try:
    kernel_name = ipykernel.connect.get_connection_info(unpack=True)["kernel_name"]
except:
    try:
        kernel_name = os.environ["HOSTNAME"]
    except:
        kernel_name = "custom_job"

# Get the number of GPUs
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
gpu_details = [tf.config.experimental.get_device_details(gpu) for gpu in gpu_devices]

gpus_by_name = {
    k: list(v) for k, v in groupby(gpu_details, key=lambda x: x["device_name"])
}

gpu_names = list(gpus_by_name.keys())

if len(gpu_names) == 0:
    gpu_name = "None"
    gpu_count = 0
elif len(gpu_names) == 1:
    gpu_name = gpu_names[0]
    gpu_count = len(gpus_by_name[gpu_name])
else:
    raise "Dunno how to handle multiple gpu types"

if custom_job_name:
    # For running on vertex AI:
    try:
        from google.cloud import aiplatform_v1

        client = aiplatform_v1.JobServiceClient()
        parent = "projects/{}/locations/{}".format(project_id, location)
        display_filter = "display_name={}".format(custom_job_name)
        request = aiplatform_v1.ListCustomJobsRequest(
            parent=parent,
            filter=display_filter,
        )
        page_result = client.list_custom_jobs(request=request)
        response = page_result[0]
        machine_type = response.job_spec.worker_pool_specs[0].machine_spec.machine_type
    except Exception as e:
        logging.warning("Error getting machine type: " + str(e))
        machine_type = "error"
else:
    # assume a generic python environment
    # See also:
    # https://docs.python.org/3.10/library/os.html#os.cpu_count
    try:
        num_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        num_cpus = os.cpu_count()
    total_mem = psutil.virtual_memory().total
    machine_type = "local {} CPUs {} GB RAM".format(
        num_cpus, round(total_mem / 1000000000, 1)
    )

# The getrusage call returns different units on mac & linux.
# Get the OS type from the platform library,
# then set the memory unit factor accordingly.
os_type = platform.system()
os_type = os_type[0]
# This is crude and impartial– but it works across my mac & Google Cloud
if "Darwin" == os_type:
    memory_unit_factor = 1000000000
elif "Linux" == os_type:
    memory_unit_factor = 1000000
else:
    # Assume kb like linux
    logging.warning("Couldn't infer machine type from %s", os_type)
    memory_unit_factor = 1000000

peak_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

prediction_overhead_s = (
    prediction_time_s - preprocess_time_s - inference_time_s - postprocess_time_s
)

# Write benchmarking data as CSV:

output = io.StringIO()
writer = csv.writer(output, quoting=csv.QUOTE_NONNUMERIC)
writer.writerow(headers)

deepcell_version = deepcell.__version__

writer.writerow(
    [
        filename,
        image_size,
        round(pixels / 1000000, 2),
        prediction_compartment,
        datetime.now(timezone.utc),
        machine_type,
        gpu_name,
        gpu_count,
        batch_size,
        kernel_name,
        prediction_success,
        round(total_time_s, 2),
        round(peak_mem / memory_unit_factor, 1),
        round(input_load_time_s, 2),
        round(prediction_time_s, 2),
        round(prediction_overhead_s, 2),
        round(preprocess_time_s, 2),
        round(inference_time_s, 2),
        round(postprocess_time_s, 2),
        deepcell_version,
    ]
)

print(output.getvalue())
