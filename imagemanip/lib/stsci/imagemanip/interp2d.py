from __future__ import division

import numpy as n
from bilinearinterp import bilinearinterp as lininterp

def expand2d(image,outputsize):
    """
    This function expands an input 2D data array to larger dimensions using
    bilinear interpolation.

    Parameters
    ----------
    image : ndarray
        Input image as numpy array
    outputsize : tuple
        Shape tuple describing the size of the output image

    Returns
    -------
    newimage : ndarray
        bilinearly interpolated array with shape specified by outputsize parameter

    """
    if (outputsize[0] >= image.shape[0] and outputsize[1] >= image.shape[1]):
        newimage = n.empty(outputsize,dtype=image.dtype)
        lininterp(image,newimage)
    else:
        raise ValueError,"Output shape must be of larger dimension than input image."

    return newimage
