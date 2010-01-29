from sample_package.main import plus2


def test_1() :
    assert plus2(2) == 4

def test_2() :
    assert plus2(4) == 6

def test_3() :
    assert plus2(1) == 3
