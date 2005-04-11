import string, copy, os

import pyfits
import numarray as N
from math import *

import pydrizzle

# Convenience definitions...
yes = True
no = False

#################
#
#
#               Coordinate Transformation Functions
#
#
#################

def XYtoSky(input, pos, idckey='IDCTAB', linear=yes, verbose=no):
    """ Convert input pixel position(s) into RA/Dec position(s).
        Output will be either an (ra,dec) pair or a 'list' of (ra,dec)
        pairs, not a numarray, to correspond with the input position(s).

        Parameter:
            input - Filename with extension specification of image
            pos   - Either a single [x,y] pair or a list of [x,y] pairs.
            idckey - Keyword which points to the IDC table to be used.
            linear - If no, apply distortion correction for image.

    """

    # Start by making sure we have a valid extension specification.
    _insplit = string.split(input,'[')
    if len(_insplit) == 1:
        raise IOError, 'No extension specified for input image!'

    # Now we need to insure that the input is an array:
    if not isinstance(pos,N.NumArray):
        if N.array(pos).getrank() > 1:
            pos = N.array(pos,type=N.Float64)

    # Set up Exposure object
    _exposure = pydrizzle.Exposure(input,idckey=idckey)

    ra,dec = _exposure.geometry.XYtoSky(pos,linear=linear,verbose=verbose)

    if not isinstance(ra,N.NumArray):
        # We are working with a single input, return single values
        return ra,dec
    else:
        # We are working with arrays, so we need to convert them
        # from 2 arrays with RA in one and Dec in the other to 1 array
        # with pairs of RA/Dec values.
        _radec = N.zeros(shape=(len(ra),2),type=ra.type())
        _radec[:,0] = ra
        _radec[:,1] = dec

        return _radec.tolist()
