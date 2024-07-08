#!/usr/bin/env python

import numpy as np
import smart_open
import tensorflow as tf


from deepcell_imaging import mesmer_app

input_channels_path = "sample-data/deepcell/mesmer-sample-7/input_channels.npz"

with smart_open.open(input_channels_path, "rb") as input_channel_file:
    with np.load(input_channel_file) as loader:
        # An array of shape [height, width, channel] containing intensity of nuclear & membrane channels
        input_channels = loader["input_channels"]

model = tf.keras.models.load_model(mesmer_app.model_path)
preprocessed_image = mesmer_app.preprocess_image(
    model.input_shape, input_channels[np.newaxis, ...], image_mpp=None
)
inferred_images = mesmer_app.predict(model, preprocessed_image, batch_size=4)
predictions = mesmer_app.postprocess(
    inferred_images, input_channels[np.newaxis, ...].shape, compartment="whole-cell"
)

print(input_channels.shape)
print(predictions.shape)
