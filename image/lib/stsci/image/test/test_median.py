#!/usr/bin/env python
import numpy as np
import nose
from nose.tools import *
from stsci.image import combine
def test_median1():
    """
    median() nominally computes the median pixels for a stack of
    identically shaped images.

    arrays     specifies a sequence of inputs arrays, which are nominally a
               stack of identically shaped images.

    output     may be used to specify the output array.  If none is specified,
               either arrays[0] is copied or a new array of type 'outtype'
               is created.

    outtype    specifies the type of the output array when no 'output' is
               specified.

    nlow       specifies the number of pixels to be excluded from median
               on the low end of the pixel stack.

    nhigh      specifies the number of pixels to be excluded from median
               on the high end of the pixel stack.

    badmasks   specifies boolean arrays corresponding to 'arrays', where true
               indicates that a particular pixel is not to be included in the
               median calculation.
    """
    a = np.arange(4)
    a = a.reshape((2,2))
    arrays = [a*16, a*4, a*2, a*8]
    result = combine.median(arrays)
    test = np.array([[ 0,  6],
           [12, 18]])
    assert_equal(result.all(),test.all())

def test_median2():
    a = np.arange(4)
    a = a.reshape((2,2))
    arrays = [a*16, a*4, a*2, a*8]
    result = combine.median(arrays, nhigh=1)
    test = np.array([[ 0,  4],
           [ 8, 12]])
    assert_equal(result.all(),test.all())

def test_median3():
    a = np.arange(4)
    a = a.reshape((2,2))
    arrays = [a*16, a*4, a*2, a*8]
    result = combine.median(arrays, nlow=1)
    test = np.array([[ 0,  8],
           [16, 24]])
    assert_equal(result.all(),test.all())

def test_median4():
    a = np.arange(4)
    a = a.reshape((2,2))
    arrays = [a*16, a*4, a*2, a*8]
    result = combine.median(arrays, outtype=np.float32)
    test = np.array([[  0.,   6.],
           [ 12.,  18.]], dtype=np.float32)
    assert_equal(result.all(),test.all())

def test_median5():
    a = np.arange(4)
    a = a.reshape((2,2))
    arrays = [a*16, a*4, a*2, a*8]
    bm = np.zeros((4,2,2), dtype=np.bool8)
    bm[2,...] = 1
    result = combine.median(arrays, badmasks=bm)
    test = np.array([[ 0,  8],
           [16, 24]])
    assert_equal(result.all(),test.all())

def test_median6():
    a = np.arange(4)
    a = a.reshape((2,2))
    arrays = [a*16, a*4, a*2, a*8]
    result = combine.median(arrays, badmasks=combine.threshhold(arrays, high=25))
    test = np.array([[ 0,  6],
           [ 8, 12]])
    assert_equal(result.all(),test.all())
