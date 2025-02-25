#!/usr/bin/env python
import os
import sys
import subprocess
import pytest


ROOT = os.path.relpath(os.path.join('build', 'test_c'))
TESTS = [
    'test_cholesky',
    # 'test_geomap',  # FIXME: died with Signals.SIGABRT 6
    'test_lintransform',
    'test_surface',
    'test_triangles',
    'test_xycoincide',
    'test_xysort',
    'test_xyxymatch',
    'test_xyxymatch_triangles'
]


@pytest.mark.parametrize("program", [os.path.join(ROOT, x) for x in TESTS])
def test_runall(program):
    if sys.platform.startswith("win"):
        program += ".exe"

    if not os.path.exists(program):
        pytest.skip("'{}' does not exist. To run the C tests "
                    "execute the following before invoking pytest: "
                    "./waf configure build".format(program))
    returncode = 0
    try:
        returncode = subprocess.check_call(program)
    except subprocess.CalledProcessError as e:
        returncode = e.returncode
        print(e, file=sys.stderr)

    assert returncode == 0
