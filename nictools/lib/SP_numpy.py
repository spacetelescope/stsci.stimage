# This is a numpy-only version of the N.py module distributed with
# ScientificPython v2.7.2.
#
from __future__ import division

from numpy.oldnumeric import *

def int_sum(a, axis=0):
    return add.reduce(a, axis)
def zeros_st(shape, other):
    return zeros(shape, dtype=other.dtype)
