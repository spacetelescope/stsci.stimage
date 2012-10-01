#! /usr/bin/env python
#
# Author: Dave Grumm (based on work by Tomas Dahlen and Eddie Bergeron)
# Program: fsutil.py
# Purpose: utility functions for 'Finesky'
# History: 03/12/08 - first version

from __future__ import division  # confidence high
import sys

__version__ = "0.1 (2008 Mar 12)"

QUIET = 0 # verbosity levels
VERBOSE = 1
VERY_VERBOSE = 2

# default values
verbosity = VERBOSE
thresh = 0.5
medfile = 'PY_Med_image.fits'
callist = '/hal/data2/dev/nicmos_ped/inlist1.lst'

def all_printMsg( message, level=VERBOSE):
    """
    Parameters
    ----------
    message : string
        message to print
    level : int
        verbosity level
    """
    if verbosity >= level:
      print message
      sys.stdout.flush()

def printMsg( message, level=QUIET):
    """
    Parameters
    ----------
    message : string
        message to print
    level : int
        verbosity level
    """
    if verbosity >= level:
        print message
        sys.stdout.flush()

def setVerbosity( verbosity_level):
    """ Copy verbosity to a variable that is global for this file.

    Parameters
    -----------
    verbosity_level : int
        level of verbosity
    """
    global verbosity
    verbosity = verbosity_level

def checkVerbosity( level):
    """
    Parameters
    -----------
    level : int
        level of verbosity

    Returns
    -------
    level : bool
        true if verbosity is at least as great as level.
    """
    return (verbosity >= level)


def setThresh( thresh_value):
    """ Copy thresh to a variable that is global for this file.

    Parameters
    ----------
    thresh_value : float
        level of threshold

    """
    global thresh
    thresh = thresh_value

def setMedfile( medfile_value):
    """ Copy medfile to a variable that is global for this file.

    Parameters
    ----------
    medfile_value : string
        name of output file for masked median image

    """
    global medfile
    medfile = medfile_value


def setCallist( callist_value):
    """ Copy callist to a variable that is global for this file.

    Parameters
    ----------
    callist_value : string
        name of file listing cal files

    """
    global callist
    callist = callist_value
