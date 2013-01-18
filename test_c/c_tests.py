from os.path import join
import subprocess
import sys

TESTS = [
    'cholesky',
    'geomap',
    'lintransform',
    'surface',
    'triangles',
    'xycoincide',
    'xysort',
    'xyxymatch',
    'xyxymatch_triangles'
    ]

def test_generator():
    def run_c_test(name):
        path = join("build", "default", "test_c", "test_%s" % name)
        retcode = subprocess.call(
            path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if retcode != 0:
            raise RuntimeError("Test returned code %d" % retcode)

    for test in TESTS:
        yield run_c_test, test
