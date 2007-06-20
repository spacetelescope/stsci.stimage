import numpy as n
from bilinearinterp import bilinearinterp as lininterp

def expand2d(image,outputsize):
    """

    Purpose
    =======
    Given an input 2D data array, expand the array to larger dimensions using
    bilinear interpolation.
    
    """
    if (outputsize[0]> image.shape[0] and outputsize[1] > image.shape[1]):
        newimage = n.empty(image.shape,dtype=image.dtype)
        lininterp(image,newimage)
    else:
        raise ValueError,"Output shape must be of larger dimension than input image."
        
    return newimage
