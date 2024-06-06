#!/usr/bin/env python

import importlib
import numpy as np
import smart_open
import sys


file_path = "./deepcell_imaging/__init__.py"
module_name = "deepcell_imaging"
spec = importlib.util.spec_from_file_location(module_name, file_path)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)

from deepcell_imaging import mesmer_app

input_channels_path = "sample-data/deepcell/mesmer-sample-7/input_channels.npz"

with smart_open.open(input_channels_path, "rb") as input_channel_file:
    with np.load(input_channel_file) as loader:
        # An array of shape [height, width, channel] containing intensity of nuclear & membrane channels
        input_channels = loader["input_channels"]

predictions = mesmer_app.predict(
    input_channels[np.newaxis, ...],
    image_mpp=0.5,
    compartment="whole-cell",
    batch_size=4,
)

print(input_channels.shape)
print(predictions.shape)
