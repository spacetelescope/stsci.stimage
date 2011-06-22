from __future__ import division

import unittest
import numpy as n
import exceptions,os,sys
from stsci.imagemanip import interp2d

class TestImageManipFunctions(unittest.TestCase):

    def setUp(self):
        # Perform set up actions (if any)
        pass

    def tearDown(self):
        # Perform clean-up actions (if any)
        pass

    def testCopy(self):
        # Test the bilinear interpolation of equal sized array
        a = n.ones((4,4),dtype=n.float32)
        x = interp2d.expand2d(a,(4,4))
        self.assertEqual(a.all(),x.all())

    def testDouble(self):
        # Test the bilinear interpolation routine for doubling
        # the size of an array
        a = n.ones((4,4),dtype=n.float32)
        x = interp2d.expand2d(a,(8,8))
        self.assertEqual(n.ones((8,8),dtype=n.float32).all(),x.all())


if __name__ == '__main__':
    unittest.main()

