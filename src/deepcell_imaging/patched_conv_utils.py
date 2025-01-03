import types

# Import this, to be able to set its field.
import keras.utils

# Hacky Mc Hackface! Create our own module.
keras.utils.conv_utils = types.ModuleType("conv_utils")


# This is the only function we actually use (via stock deepcell)
# Copied from: https://github.com/tensorflow/addons/blob/d208d752e98c310280938efa939117bf635a60a8/tensorflow_addons/utils/keras_utils.py#L71
# Apache license
def normalize_data_format(value):
    if value is None:
        value = tf.keras.backend.image_data_format()
    data_format = value.lower()
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError(
            "The `data_format` argument must be one of "
            '"channels_first", "channels_last". Received: ' + str(value)
        )


# Patch in the function we use.
keras.utils.conv_utils.normalize_data_format = normalize_data_format
