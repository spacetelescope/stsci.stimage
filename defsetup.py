#!/usr/bin/env python

# Copyright (C) 2008-2010 Association of Universities for Research in Astronomy (AURA)

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

#     1. Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.

#     2. Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials provided
#       with the distribution.

#     3. The name of AURA and its representatives may not be used to
#       endorse or promote products derived from this software without
#       specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY AURA ``AS IS'' AND ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL AURA BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
# TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE.

from __future__ import division, print_function # confidence high

import sys
from distutils.core import setup, Extension
from os.path import join
import os.path

######################################################################
# CONFIGURATION
DEBUG = True
CONTACT = "Michael Droettboom"
EMAIL = "mdroe@stsci.edu"
VERSION = "0.1"

if os.path.exists('stimage'):
    ROOT_DIR = 'stimage'
else:
    ROOT_DIR = '.'

######################################################################
# NUMPY
try:
    import numpy
except ImportError:
    print("numpy must be installed to build stimage.")
    print("ABORTING.")
    raise

try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

######################################################################
# STIMAGE-SPECIFIC AND WRAPPER SOURCE FILES
STIMAGE_SOURCES = [ # List of pure-C files to compile
    'immatch/geomap.c',
    'immatch/xyxymatch.c',
    'immatch/lib/tolerance.c',
    'immatch/lib/triangles.c',
    'immatch/lib/triangles_vote.c',
    'lib/error.c',
    'lib/lintransform.c',
    'lib/polynomial.c',
    'lib/util.c',
    'lib/xybbox.c',
    'lib/xycoincide.c',
    'lib/xysort.c',
    'surface/cholesky.c',
    'surface/fit.c',
    'surface/surface.c',
    'surface/vector.c']
STIMAGE_SOURCES = [join('src', x) for x in STIMAGE_SOURCES]

STIMAGE_WRAP_SOURCES = [
    'stimage_module.c',
    'wrap_util.c',
    'immatch/py_xyxymatch.c',
    'immatch/py_geomap.c'
    ]
STIMAGE_WRAP_SOURCES = [join('src_wrap', x) for x in STIMAGE_WRAP_SOURCES]

######################################################################
# DISTUTILS SETUP
libraries = []
define_macros = []
undef_macros = []
extra_compile_args = []
if DEBUG:
    define_macros.append(('DEBUG', None))
    undef_macros.append('NDEBUG')
    if not sys.platform.startswith('sun') and \
            not sys.platform == 'win32':
        extra_compile_args.extend(["-Wall", "-fno-inline", "-O0", "-g"])
else:
    define_macros.append(('NDEBUG', None))
    undef_macros.append('DEBUG')

pkg = ["stsci.stimage", "stsci.stimage.test"]

setupargs = {
    'version': VERSION,
    'description': "Various image processing functions",
    'author': CONTACT,
    'author_email': EMAIL,
    'url': "http://projects.scipy.org/astropy/astrolib/wiki/WikiStart",
    'platforms': ['unix', 'windows'],
    'ext_modules': [
        Extension(
            'stsci.stimage._stimage',
            STIMAGE_SOURCES + STIMAGE_WRAP_SOURCES,
            include_dirs=[
                numpy_include,
                join(ROOT_DIR, "include"),
                join(ROOT_DIR, "src_wrap")
                ],
            define_macros=define_macros,
            undef_macros=undef_macros,
            extra_compile_args=extra_compile_args,
            libraries=libraries
            )
        ],
    'package_dir' : {'stsci.stimage': 'lib/stsci/stimage', 'stsci.stimage.test' : 'lib/stsci/stimage/test'},
    }
