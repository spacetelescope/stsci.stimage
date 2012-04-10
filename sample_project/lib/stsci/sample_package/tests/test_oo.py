#
# Here is a sample test based on objects
#
from __future__ import division # confidence high

from ..main import plus2

import unittest

# The object contains a list of methods named "test_*" that are all
# the tests.  For each one, the object gets instatiated, then it calls
# setUp(), test_whatever(), and tearDown().
#
# In nose, you don't get a test result if the __init__ function raises
# an exception, but you do if setUp() does.
#
# In py.test, the class has to be named TestSomething with a capital T

class Test_1(object) :
    a = plus2(4)

    @classmethod
    def setup_class( self ) :
        # called before each test
        print 'test setup'
        self.b = plus2(6)
        self.c = plus2(8)

    # methods named 'test_*' are tests
    def test_a( self ) :
        assert self.a == 6

    def test_b( self ) :
        assert self.b == 8

    def test_c( self ) :
        assert self.c == 10

    def test_d( self ) :
        print "example of a failing test"
        assert self.c == 12

    def test_e( self ) :
        assert plus2(10) == 12

    @classmethod
    def teardown_class( self ) :
        # called after each test
        print 'test teardown'

