import numpy as n

def expand2d(image,outputsize):
    """

    Purpose
    =======
    Given an input 2D data array, expand the array to larger dimensions using
    bilinear interpolation.
    
    """
    if (outputsize[0]> image.shape[0] and outputsize[1] > image.shape[1]):
        newimage = n.empty(image.shape,dtype=image.dtype)
    else:
        raise ValueError,"Output shape must be of larger dimension than input image."
        
    return newimage
