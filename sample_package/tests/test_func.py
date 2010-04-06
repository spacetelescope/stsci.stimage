from __future__ import division # confidence high

from sample_package.main import plus2

# Just write a function named test_* that asserts whatever it is you are looking for.

def test_1() :
    assert plus2(2) == 4

def test_2() :
    assert plus2(4) == 6

def test_3() :
    assert plus2(1) == 3
