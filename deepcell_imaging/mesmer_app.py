"""
The Mesmer application, in pieces.

This was extracted from the DeepCell Mesmer Application, and modified to
remove the stateful class object. Instead, each function is a standalone
function that can be called independently.
"""

import logging
import timeit

import numpy as np
from deepcell_toolbox.deep_watershed import deep_watershed
from deepcell_toolbox.processing import histogram_normalization
from deepcell_toolbox.processing import percentile_threshold
from deepcell_toolbox.utils import resize, tile_image, untile_image

MODEL_REMOTE_PATH = "gs://davids-genomics-data-public/cellular-segmentation/deep-cell/vanvalenlab-tf-model-multiplex-downloaded-20230706/MultiplexSegmentation.tar.gz"

MESMER_MODEL_MPP = 0.5


def validate_image(model_input_shape, image):
    # The 1st dimension is the batch dimension, remove it.
    model_image_shape = model_input_shape[1:]

    # Require dimension 1 larger than model_input_shape due to addition of batch dimension
    required_rank = len(model_image_shape) + 1

    # Check input size of image
    if len(image.shape) != required_rank:
        raise ValueError(
            f"Input data must have {required_rank} dimensions. "
            f"Input data only has {len(image.shape)} dimensions"
        )

    if image.shape[-1] != model_input_shape[-1]:
        raise ValueError(
            f"Input data must have {model_input_shape[-1]} channels. "
            f"Input data has {image.shape[-1]} channels"
        )


def preprocess_image(model_input_shape, image, image_mpp):
    logger = logging.getLogger(__name__)
    preprocess_kwargs = {}

    validate_image(model_input_shape, image)

    t = timeit.default_timer()
    logger.debug(
        "Pre-processing data with %s and kwargs: %s",
        mesmer_preprocess.__name__,
        **preprocess_kwargs,
    )

    # Scale the image if mpp defined & different from the model mpp
    if image_mpp not in {None, MESMER_MODEL_MPP}:
        shape = image.shape
        scale_factor = image_mpp / MESMER_MODEL_MPP
        new_shape = (int(shape[1] * scale_factor), int(shape[2] * scale_factor))
        image = resize(image, new_shape, data_format="channels_last")
        logger.debug("Resized input from %s to %s", shape, new_shape)

    image = mesmer_preprocess(image, **preprocess_kwargs)

    logger.debug(
        "Pre-processed data with %s in %s s",
        mesmer_preprocess.__name__,
        timeit.default_timer() - t,
    )

    return image


def predict(model, image, batch_size):
    logger = logging.getLogger(__name__)
    model_image_shape = model.input_shape[1:]
    pad_mode = "constant"

    # TODO: we need to validate the input. But what validations?

    # Tile images, raises error if the image is not 4d
    tiles, tiles_info = tile_input(image, model_image_shape, pad_mode=pad_mode)

    # Run images through model
    t = timeit.default_timer()
    output_tiles = batch_predict(model=model, tiles=tiles, batch_size=batch_size)
    logger.debug("Model prediction finished in %s s", timeit.default_timer() - t)

    # Untile images
    output_images = _untile_output(output_tiles, tiles_info, model_image_shape)

    # restructure outputs into a dict if function provided
    return format_output_mesmer(output_images)


def postprocess(
    output_images,
    input_shape,
    compartment="whole-cell",
    whole_cell_kwargs={},
    nuclear_kwargs={},
):
    logger = logging.getLogger(__name__)

    # TODO: We need to validate the input (the output_images parameter)

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
        **whole_cell_kwargs,
    }

    postprocess_kwargs_nuclear = {
        **default_kwargs_nuc,
        **nuclear_kwargs,
    }

    # create dict to hold all the post-processing kwargs
    postprocess_kwargs = {
        "whole_cell_kwargs": postprocess_kwargs_whole_cell,
        "nuclear_kwargs": postprocess_kwargs_nuclear,
        "compartment": compartment,
    }

    # Postprocess predictions to create label image
    t = timeit.default_timer()
    logger.debug(
        "Post-processing results with %s and kwargs: %s",
        mesmer_postprocess.__name__,
        **postprocess_kwargs,
    )

    label_image = mesmer_postprocess(output_images, **postprocess_kwargs)

    logger.debug(
        "Post-processed results with %s in %s s",
        mesmer_postprocess.__name__,
        timeit.default_timer() - t,
    )

    # Resize label_image back to original resolution if necessary
    # But, remove the 1st dimension: batch num.
    return _resize_output(label_image, input_shape)[0]


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


def tile_input(image, model_image_shape, pad_mode="constant"):
    if len(image.shape) != 4:
        raise ValueError(
            "deepcell_toolbox.tile_image only supports 4d images."
            f"Image submitted for predict has {len(image.shape)} dimensions"
        )

    # Check difference between input and model image size
    x_diff = image.shape[1] - model_image_shape[0]
    y_diff = image.shape[2] - model_image_shape[1]

    # Check if the input is smaller than model image size
    if x_diff < 0 or y_diff < 0:
        # Calculate padding
        x_diff, y_diff = abs(x_diff), abs(y_diff)
        x_pad = (
            (x_diff // 2, x_diff // 2 + 1) if x_diff % 2 else (x_diff // 2, x_diff // 2)
        )
        y_pad = (
            (y_diff // 2, y_diff // 2 + 1) if y_diff % 2 else (y_diff // 2, y_diff // 2)
        )

        tiles = np.pad(image, [(0, 0), x_pad, y_pad, (0, 0)], "reflect")
        tiles_info = {"padding": True, "x_pad": x_pad, "y_pad": y_pad}
    # Otherwise tile images larger than model size
    else:
        # Tile images, needs 4d
        tiles, tiles_info = tile_image(
            image,
            model_input_shape=model_image_shape,
            stride_ratio=0.75,
            pad_mode=pad_mode,
        )

    return tiles, tiles_info


def _untile_output(output_tiles, tiles_info, model_image_shape):
    # If padding was used, remove padding
    if tiles_info.get("padding", False):

        def _process(im, tiles_info):
            ((xl, xh), (yl, yh)) = tiles_info["x_pad"], tiles_info["y_pad"]
            # Edge-case: upper-bound == 0 - this can occur when only one of
            # either X or Y is smaller than model_img_shape while the other
            # is equal to model_image_shape.
            xh = -xh if xh != 0 else None
            yh = -yh if yh != 0 else None
            return im[:, xl:xh, yl:yh, :]

    # Otherwise untile
    else:

        def _process(im, tiles_info):
            out = untile_image(im, tiles_info, model_input_shape=model_image_shape)
            return out

    if isinstance(output_tiles, list):
        output_images = [_process(o, tiles_info) for o in output_tiles]
    else:
        output_images = _process(output_tiles, tiles_info)

    return output_images


def _resize_output(image, original_shape):
    """Rescales input if the shape does not match the original shape
    excluding the batch and channel dimensions.

    Args:
        image (numpy.array): Image to be rescaled to original shape
        original_shape (tuple): Shape of the original input image

    Returns:
        numpy.array: Rescaled image
    """
    if not isinstance(image, list):
        image = [image]

    for i in range(len(image)):
        img = image[i]
        # Compare x,y based on rank of image
        if len(img.shape) == 4:
            same = img.shape[1:-1] == original_shape[1:-1]
        elif len(img.shape) == 3:
            same = img.shape[1:] == original_shape[1:-1]
        else:
            same = img.shape == original_shape[1:-1]

        # Resize if same is false
        if not same:
            # Resize function only takes the x,y dimensions for shape
            new_shape = original_shape[1:-1]
            img = resize(
                img, new_shape, data_format="channels_last", labeled_image=True
            )
        image[i] = img

    if len(image) == 1:
        image = image[0]

    return image


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
