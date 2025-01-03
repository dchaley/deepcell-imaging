# Force loading our patched conv_utils.
import deepcell_imaging.patched_conv_utils

# Now load the original functions, using our patch.
from deepcell.utils.plot_utils import create_rgb_image, make_outline_overlay
