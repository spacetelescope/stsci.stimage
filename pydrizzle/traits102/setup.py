#!/usr/bin/env python

#--------------------------------------------------------------------------------
#
#  Python setup for the 'traits' package
#
#  Written by: David C. Morrill
#
#  Date: 2/03/2003
#
#  (c) Copyright 2002 by Enthought, Inc.
#
#--------------------------------------------------------------------------------

from distutils.core import setup

setup( name         = 'traits',
       version      = '1.0.2',
       description  = 'Strongly typed Python attributes package',
       author       = 'David C. Morrill',
       author_email = 'dmorrill@enthought.com',
       url          = 'http://www.scipy.org/site_content/traits',
       license      = 'BSD',
       packages     = [ 'traits' ],
       package_dir  = { 'traits': '.' }
)
