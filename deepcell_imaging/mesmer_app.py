# Copyright 2016-2023 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/deepcell-tf/LICENSE
#
# The Work provided may be used for non-commercial academic purposes only.
# For any other use of the Work, including commercial use, please contact:
# vanvalenlab@gmail.com
#
# Neither the name of Caltech nor the names of its contributors may be used
# to endorse or promote products derived from this software without specific
# prior written permission.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Mesmer application"""


from pathlib import Path

import logging
import numpy as np
import os
import sys
import tensorflow as tf
import timeit

from deepcell_toolbox.deep_watershed import deep_watershed
from deepcell_toolbox.processing import percentile_threshold
from deepcell_toolbox.processing import histogram_normalization

from deepcell.applications import Application
from deepcell.utils import fetch_data, extract_archive

import importlib

file_path = "./deepcell_imaging/__init__.py"
module_name = "deepcell_imaging"
spec = importlib.util.spec_from_file_location(module_name, file_path)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)

from deepcell_imaging import cached_open

MODEL_REMOTE_PATH = (
    "gs://davids-genomics-data-public/cellular-segmentation/deep-cell/vanvalenlab-tf-model-multiplex-downloaded-20230706/MultiplexSegmentation.tar.gz",
)

MODEL_KEY = "models/MultiplexSegmentation-9.tar.gz"
MODEL_NAME = "MultiplexSegmentation"
MODEL_HASH = "a1dfbce2594f927b9112f23a0a1739e0"


downloaded_file_path = cached_open.get_file(
    "MultiplexSegmentation.tgz",
    MODEL_REMOTE_PATH,
    file_hash=MODEL_HASH,
    extract=True,
    cache_subdir="models",
)
# Remove the .tgz extension to get the model directory path
model_path = os.path.splitext(downloaded_file_path)[0]


def predict(input_channels, image_mpp, compartment, batch_size):
    model = tf.keras.models.load_model(model_path)
    app = Mesmer(model=model)

    # In the end result, we don't have app.predict.
    # We have:
    #  (1) preprocess(...)
    #  (2) predict(...)
    #  (3) postprocess(...)

    return app.predict(
        model,
        input_channels,
        image_mpp=image_mpp,
        compartment=compartment,
        batch_size=batch_size,
    )


# pre- and post-processing functions
def mesmer_preprocess(image, **kwargs):
    """Preprocess input data for Mesmer model.

    Args:
        image: array to be processed

    Returns:
        np.array: processed image array
    """

    if len(image.shape) != 4:
        raise ValueError(f"Image data must be 4D, got image of shape {image.shape}")

    output = np.copy(image)
    threshold = kwargs.get("threshold", True)
    if threshold:
        percentile = kwargs.get("percentile", 99.9)
        output = percentile_threshold(image=output, percentile=percentile)

    normalize = kwargs.get("normalize", True)
    if normalize:
        kernel_size = kwargs.get("kernel_size", 128)
        output = histogram_normalization(image=output, kernel_size=kernel_size)

    return output


def format_output_mesmer(output_list):
    """Takes list of model outputs and formats into a dictionary for better readability

    Args:
        output_list (list): predictions from semantic heads

    Returns:
        dict: Dict of predictions for whole cell and nuclear.

    Raises:
        ValueError: if model output list is not len(4)
    """
    expected_length = 4
    if len(output_list) != expected_length:
        raise ValueError(
            "output_list was length {}, expecting length {}".format(
                len(output_list), expected_length
            )
        )

    formatted_dict = {
        "whole-cell": [output_list[0], output_list[1][..., 1:2]],
        "nuclear": [output_list[2], output_list[3][..., 1:2]],
    }

    return formatted_dict


def mesmer_postprocess(
    model_output, compartment="whole-cell", whole_cell_kwargs=None, nuclear_kwargs=None
):
    """Postprocess Mesmer output to generate predictions for distinct cellular compartments

    Args:
        model_output (dict): Output from the Mesmer model. A dict with a key corresponding to
            each cellular compartment with a model prediction. Each key maps to a subsequent dict
            with the following keys entries
            - inner-distance: Prediction for the inner distance transform.
            - outer-distance: Prediction for the outer distance transform
            - fgbg-fg: prediction for the foreground/background transform
            - pixelwise-interior: Prediction for the interior/border/background transform.
        compartment: which cellular compartments to generate predictions for.
            must be one of 'whole_cell', 'nuclear', 'both'
        whole_cell_kwargs (dict): Optional list of post-processing kwargs for whole-cell prediction
        nuclear_kwargs (dict): Optional list of post-processing kwargs for nuclear prediction

    Returns:
        numpy.array: Uniquely labeled mask for each compartment

    Raises:
        ValueError: for invalid compartment flag
    """

    valid_compartments = ["whole-cell", "nuclear", "both"]

    if whole_cell_kwargs is None:
        whole_cell_kwargs = {}

    if nuclear_kwargs is None:
        nuclear_kwargs = {}

    if compartment not in valid_compartments:
        raise ValueError(
            f"Invalid compartment supplied: {compartment}. "
            f"Must be one of {valid_compartments}"
        )

    if compartment == "whole-cell":
        label_images = deep_watershed(model_output["whole-cell"], **whole_cell_kwargs)
    elif compartment == "nuclear":
        label_images = deep_watershed(model_output["nuclear"], **nuclear_kwargs)
    elif compartment == "both":
        label_images_cell = deep_watershed(
            model_output["whole-cell"], **whole_cell_kwargs
        )

        label_images_nucleus = deep_watershed(model_output["nuclear"], **nuclear_kwargs)

        label_images = np.concatenate(
            [label_images_cell, label_images_nucleus], axis=-1
        )

    else:
        raise ValueError(
            f"Invalid compartment supplied: {compartment}. "
            f"Must be one of {valid_compartments}"
        )

    return label_images


def batch_predict(model, tiles, batch_size):
    # list to hold final output
    output_tiles = []

    # loop through each batch
    for i in range(0, tiles.shape[0], batch_size):
        batch_inputs = tiles[i : i + batch_size, ...]

        batch_outputs = model.predict(batch_inputs, batch_size=batch_size)

        # model with only a single output gets temporarily converted to a list
        if not isinstance(batch_outputs, list):
            batch_outputs = [batch_outputs]

        # initialize output list with empty arrays to hold all batches
        if not output_tiles:
            for batch_out in batch_outputs:
                shape = (tiles.shape[0],) + batch_out.shape[1:]
                output_tiles.append(np.zeros(shape, dtype=tiles.dtype))

        # save each batch to corresponding index in output list
        for j, batch_out in enumerate(batch_outputs):
            output_tiles[j][i : i + batch_size, ...] = batch_out

    return output_tiles


class Mesmer(Application):
    """Loads a :mod:`deepcell.model_zoo.panopticnet.PanopticNet` model for
    tissue segmentation with pretrained weights.

    The ``predict`` method handles prep and post processing steps
    to return a labeled image.

    Example:

    .. code-block:: python

        from skimage.io import imread
        from deepcell.applications import Mesmer

        # Load the images
        im1 = imread('TNBC_DNA.tiff')
        im2 = imread('TNBC_Membrane.tiff')

        # Combined together and expand to 4D
        im = np.stack((im1, im2), axis=-1)
        im = np.expand_dims(im,0)

        # Create the application
        app = Mesmer()

        # create the lab
        labeled_image = app.predict(image)

    Args:
        model (tf.keras.Model): The model to load. If ``None``,
            a pre-trained model will be downloaded.
    """

    #: Metadata for the dataset used to train the model
    dataset_metadata = {
        "name": "20200315_IF_Training_6.npz",
        "other": "Pooled whole-cell data across tissue types",
    }

    #: Metadata for the model and training process
    model_metadata = {
        "batch_size": 1,
        "lr": 1e-5,
        "lr_decay": 0.99,
        "training_seed": 0,
        "n_epochs": 30,
        "training_steps_per_epoch": 1739 // 1,
        "validation_steps_per_epoch": 193 // 1,
    }

    def __init__(self, model=None):
        if model is None:
            cache_subdir = "models"
            model_dir = Path.home() / ".deepcell" / "models"
            archive_path = fetch_data(
                asset_key=MODEL_KEY, cache_subdir=cache_subdir, file_hash=MODEL_HASH
            )
            extract_archive(archive_path, model_dir)
            model_path = model_dir / MODEL_NAME
            model = tf.keras.models.load_model(model_path)

        model_mpp = 0.5
        model_image_shape = model.input_shape[1:]
        postprocessing_fn = mesmer_postprocess
        format_model_output_fn = format_output_mesmer
        dataset_metadata = self.dataset_metadata
        model_metadata = self.model_metadata

        self.model = model

        self.model_image_shape = model_image_shape
        # Require dimension 1 larger than model_input_shape due to addition of batch dimension
        # We still need this for _postprocess. When that is inlined, remove.
        self.required_rank = len(self.model_image_shape) + 1

        self.required_channels = self.model_image_shape[-1]

        self.model_mpp = model_mpp
        self.postprocessing_fn = postprocessing_fn
        self.format_model_output_fn = format_model_output_fn
        self.dataset_metadata = dataset_metadata
        self.model_metadata = model_metadata

        self.logger = logging.getLogger(self.__class__.__name__)

        # Test that pre and post processing functions are callable
        if self.postprocessing_fn is not None and not callable(self.postprocessing_fn):
            raise ValueError("Postprocessing_fn must be a callable function.")
        if self.format_model_output_fn is not None and not callable(
            self.format_model_output_fn
        ):
            raise ValueError("Format_model_output_fn must be a callable function.")

    def predict(
        self,
        model,
        image,
        batch_size=4,
        image_mpp=None,
        preprocess_kwargs={},
        compartment="whole-cell",
        pad_mode="constant",
        postprocess_kwargs_whole_cell={},
        postprocess_kwargs_nuclear={},
    ):
        """Generates a labeled image of the input running prediction with
        appropriate pre and post processing functions.

        Input images are required to have 4 dimensions
        ``[batch, x, y, channel]``.
        Additional empty dimensions can be added using ``np.expand_dims``.

        Args:
            image (numpy.array): Input image with shape
                ``[batch, x, y, channel]``.
            batch_size (int): Number of images to predict on per batch.
            image_mpp (float): Microns per pixel for ``image``.
            compartment (str): Specify type of segmentation to predict.
                Must be one of ``"whole-cell"``, ``"nuclear"``, ``"both"``.
            preprocess_kwargs (dict): Keyword arguments to pass to the
                pre-processing function.
            postprocess_kwargs (dict): Keyword arguments to pass to the
                post-processing function.

        Raises:
            ValueError: Input data must match required rank of the application,
                calculated as one dimension more (batch dimension) than expected
                by the model.

            ValueError: Input data must match required number of channels.

        Returns:
            numpy.array: Instance segmentation mask.
        """
        default_kwargs_cell = {
            "maxima_threshold": 0.075,
            "maxima_smooth": 0,
            "interior_threshold": 0.2,
            "interior_smooth": 2,
            "small_objects_threshold": 15,
            "fill_holes_threshold": 15,
            "radius": 2,
        }

        default_kwargs_nuc = {
            "maxima_threshold": 0.1,
            "maxima_smooth": 0,
            "interior_threshold": 0.2,
            "interior_smooth": 2,
            "small_objects_threshold": 15,
            "fill_holes_threshold": 15,
            "radius": 2,
        }

        # overwrite defaults with any user-provided values
        postprocess_kwargs_whole_cell = {
            **default_kwargs_cell,
            **postprocess_kwargs_whole_cell,
        }

        postprocess_kwargs_nuclear = {
            **default_kwargs_nuc,
            **postprocess_kwargs_nuclear,
        }

        # create dict to hold all of the post-processing kwargs
        postprocess_kwargs = {
            "whole_cell_kwargs": postprocess_kwargs_whole_cell,
            "nuclear_kwargs": postprocess_kwargs_nuclear,
            "compartment": compartment,
        }

        # Require dimension 1 larger than model_input_shape due to addition of batch dimension
        required_rank = len(model.input_shape[1:]) + 1

        # Check input size of image
        if len(image.shape) != required_rank:
            raise ValueError(
                f"Input data must have {required_rank} dimensions. "
                f"Input data only has {len(image.shape)} dimensions"
            )

        if image.shape[-1] != self.required_channels:
            raise ValueError(
                f"Input data must have {self.required_channels} channels. "
                f"Input data only has {image.shape[-1]} channels"
            )

        # Resize image, returns unmodified if appropriate
        image = self._resize_input(image, image_mpp)

        # -----------------------------
        # Generate model outputs

        # Preprocessing
        t = timeit.default_timer()
        self.logger.debug(
            "Pre-processing data with %s and kwargs: %s",
            mesmer_preprocess.__name__,
            **preprocess_kwargs,
        )

        image = mesmer_preprocess(image, **preprocess_kwargs)

        self.logger.debug(
            "Pre-processed data with %s in %s s",
            mesmer_preprocess.__name__,
            timeit.default_timer() - t,
        )
        # End preprocessing

        # Tile images, raises error if the image is not 4d
        tiles, tiles_info = self._tile_input(image, pad_mode=pad_mode)

        # Run images through model
        t = timeit.default_timer()
        output_tiles = batch_predict(
            model=self.model, tiles=tiles, batch_size=batch_size
        )
        self.logger.debug(
            "Model inference finished in %s s", timeit.default_timer() - t
        )

        # Untile images
        output_images = self._untile_output(output_tiles, tiles_info)

        # restructure outputs into a dict if function provided
        output_images = self._format_model_output(output_images)

        # -----------------------------

        # Postprocess predictions to create label image
        label_image = self._postprocess(output_images, **postprocess_kwargs)

        # Resize label_image back to original resolution if necessary
        label_image = self._resize_output(label_image, image.shape)

        return label_image
