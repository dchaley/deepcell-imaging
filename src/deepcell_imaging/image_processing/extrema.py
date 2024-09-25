"""
This file was copied from:
https://github.com/scikit-image/scikit-image

Specifically:
https://github.com/scikit-image/scikit-image/blob/d15a5f7c8292cb19b84bf8628df28eaf46f60476/skimage/morphology/extrema.py

Licensed under BSD-3
"""

import functools
import numpy as np
import warnings

from skimage.morphology.extrema import _subtract_constant_clip

from deepcell_imaging.image_processing.fast_hybrid import fast_hybrid_reconstruct

# A version of `warnings.warn` with a default stacklevel of 2.
# functool is used so as not to increase the call stack accidentally
warn = functools.partial(warnings.warn, stacklevel=2)


def h_maxima(image, h, footprint=None):
    """Determine all maxima of the image with height >= h.

    The local maxima are defined as connected sets of pixels with equal
    gray level strictly greater than the gray level of all pixels in direct
    neighborhood of the set.

    A local maximum M of height h is a local maximum for which
    there is at least one path joining M with an equal or higher local maximum
    on which the minimal value is f(M) - h (i.e. the values along the path
    are not decreasing by more than h with respect to the maximum's value)
    and no path to an equal or higher local maximum for which the minimal
    value is greater.

    The global maxima of the image are also found by this function.

    Parameters
    ----------
    image : ndarray
        The input image for which the maxima are to be calculated.
    h : unsigned integer
        The minimal height of all extracted maxima.
    footprint : ndarray, optional
        The neighborhood expressed as an n-D array of 1's and 0's.
        Default is the ball of radius 1 according to the maximum norm
        (i.e. a 3x3 square for 2D images, a 3x3x3 cube for 3D images, etc.)

    Returns
    -------
    h_max : ndarray
        The local maxima of height >= h and the global maxima.
        The resulting image is a binary image, where pixels belonging to
        the determined maxima take value 1, the others take value 0.

    See Also
    --------
    skimage.morphology.h_minima
    skimage.morphology.local_maxima
    skimage.morphology.local_minima

    References
    ----------
    .. [1] Soille, P., "Morphological Image Analysis: Principles and
           Applications" (Chapter 6), 2nd edition (2003), ISBN 3540429883.

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.morphology import extrema

    We create an image (quadratic function with a maximum in the center and
    4 additional constant maxima.
    The heights of the maxima are: 1, 21, 41, 61, 81

    >>> w = 10
    >>> x, y = np.mgrid[0:w,0:w]
    >>> f = 20 - 0.2*((x - w/2)**2 + (y-w/2)**2)
    >>> f[2:4,2:4] = 40; f[2:4,7:9] = 60; f[7:9,2:4] = 80; f[7:9,7:9] = 100
    >>> f = f.astype(int)

    We can calculate all maxima with a height of at least 40:

    >>> maxima = extrema.h_maxima(f, 40)

    The resulting image will contain 3 local maxima.
    """

    # Check for h value that is larger then range of the image. If this
    # is True then there are no h-maxima in the image.
    if h > np.ptp(image):
        return np.zeros(image.shape, dtype=np.uint8)

    # Check for floating point h value. For this to work properly
    # we need to explicitly convert image to float64.
    #
    # FIXME: This could give incorrect results if image is int64 and
    #        has a very high dynamic range. The dtype of image is
    #        changed to float64, and different integer values could
    #        become the same float due to rounding.
    #
    #   >>> ii64 = np.iinfo(np.int64)
    #   >>> a = np.array([ii64.max, ii64.max - 2])
    #   >>> a[0] == a[1]
    #   False
    #   >>> b = a.astype(np.float64)
    #   >>> b[0] == b[1]
    #   True
    #
    if np.issubdtype(type(h), np.floating) and np.issubdtype(image.dtype, np.integer):
        if (h % 1) != 0:
            warn(
                "possible precision loss converting image to "
                "floating point. To silence this warning, "
                "ensure image and h have same data type.",
                stacklevel=2,
            )
            image = image.astype(float)
        else:
            h = image.dtype.type(h)

    if h == 0:
        raise ValueError("h = 0 is ambiguous, use local_maxima() " "instead?")

    if np.issubdtype(image.dtype, np.floating):
        # The purpose of the resolution variable is to allow for the
        # small rounding errors that inevitably occur when doing
        # floating point arithmetic. We want shifted_img to be
        # guaranteed to be h less than image. If we only subtract h
        # there may be pixels were shifted_img ends up being
        # slightly greater than image - h.
        #
        # The resolution is scaled based on the pixel values in the
        # image because floating point precision is relative. A
        # very large value of 1.0e10 will have a large precision,
        # say +-1.0e4, and a very small value of 1.0e-10 will have
        # a very small precision, say +-1.0e-16.
        #
        resolution = 2 * np.finfo(image.dtype).resolution * np.abs(image)
        shifted_img = image - h - resolution
    else:
        shifted_img = _subtract_constant_clip(image, h)

    rec_img = fast_hybrid_reconstruct(
        shifted_img, image, method="dilation", footprint=footprint
    )
    residue_img = image - rec_img
    return (residue_img >= h).astype(np.uint8)
