#!/usr/bin/env python

#--------------------------------------------------------------------------------
#
#  Python setup for the 'traits' package
#
#  Written by: David C. Morrill
#
#  Date: 12/20/2002
#
#  (c) Copyright 2002 by Enthought, Inc.
#
#--------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

from scipy_distutils.misc_util import default_config_dict

#-------------------------------------------------------------------------------
#  Define the configuration information:
#-------------------------------------------------------------------------------

def configuration ( parent_package = '' ):
    return default_config_dict( 'traits', parent_package )

#-------------------------------------------------------------------------------
#  Do the setup if we are run stand-alone:
#-------------------------------------------------------------------------------

if __name__ == '__main__':
    from scipy_distutils.core import setup
    setup( **configuration() )
