# Force loading our patched conv_utils.
import deepcell_imaging.patched_conv_utils

# Now load the original module, using our patch.
from deepcell.layers.location import Location2D
