#!/usr/bin/env python
import relic.release
from glob import glob
from os import walk
from os.path import abspath, curdir, join
from numpy import get_include as np_include
from setuptools import setup, find_packages, Extension


version = relic.release.get_info()
relic.release.write_template(version, 'lib/stsci/stimage')

sources = []
for root, dirs, files in walk(curdir):
    if 'test_c' in root:
        continue
    for f in files:
        if f.endswith('.c'):
            sources.append(join(abspath(root), f))

setup(
    name = 'stsci.stimage',
    version = version.pep386,
    author = 'Michael Droettboom',
    author_email = 'help@stsci.edu',
    description = 'Various image processing functions',
    url = 'https://github.com/spacetelescope/stsci.stimage',
    classifiers = [
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    install_requires = [
        'nose',
        'numpy',
        'sphinx',
        'stsci.sphinxext'
    ],
    package_dir = {
        '': 'lib'
    },
    packages = find_packages(),
    package_data = {
        '': ['LICENSE'],
    },
    ext_modules=[
        Extension('stsci.stimage._stimage',
            sources,
            include_dirs=[
                np_include(),
                'include',
                'src_wrap',
            ],
        ),
    ],
)
