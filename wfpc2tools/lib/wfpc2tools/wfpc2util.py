#! /usr/bin/env python

from __future__ import division # confidence high
import sys
import time

# Utility functions and parameters for Wfpc2destreak

QUIET = 0 # verbosity levels
VERBOSE = 1
VERY_VERBOSE = 2
HUGE_VAL = 100000. # default for bias_thresh

# default values
verbosity = VERBOSE
group = 4
bias_thresh = HUGE_VAL
row_thresh = 0.1
input_mask = None
niter = 5

def all_printMsg( message, level=VERBOSE):
    """ Print message as verbose message by default

        Parameters
        ----------
        message : string
            message be printed, if verbosity level is appropriate
        level: int [Default: 1 (VERBOSE)]
            integer indicating the level of verbosity for printing this string

    """

    if verbosity >= level:
        print message
        sys.stdout.flush()

def printMsg( message, level=QUIET):
    """ Print message based on verbosity level.

        Parameters
        ----------
        message : string
            message be printed, if verbosity level is appropriate
        level: int [Default: 0 (QUIET)]
            integer indicating the level of verbosity for printing this string

    """

    if verbosity >= level:
        print message
        sys.stdout.flush()

def setGroup( group_value):
        """ Copy group to a variable that is global for this file.

        Parameters
        ----------
        group_value : int
            value of group
        """
        global group
        group = group_value

def setBias_thresh( bias_thresh_value):
        """ Copy bias_thresh to a variable that is global for this file.

        Parameters
        ----------
        bias_thresh_value : float
            value of bias_thresh
        """
        global bias_thresh
        bias_thresh = bias_thresh_value


def setRow_thresh( row_thresh_value):
        """ Copy row_thresh to a variable that is global for this file.

        Parameters
        ----------
        row_thresh_value : float
            value of row_thresh

        """
        global row_thresh
        row_thresh = row_thresh_value

def setInput_mask( input_mask_value):
        """Copy input_mask to a variable that is global for this file.

        Parameters
        ----------
        input_mask_value : string
            value of input_mask

        """
        global input_mask
        input_mask = input_mask_value

def setNiter( niter_value):
        """ Copy niter to a variable that is global for this file.

        Parameters
        ----------
        niter_value : int
            value of niter

        """
        global niter
        niter = niter_value

def setVerbosity( verbosity_level):
        """Copy verbosity to a variable that is global for this file.

        Parameters
        ----------
        verbosity_level: int
            an integer value indicating the level of verbosity

        """

        global verbosity
        verbosity = verbosity_level

def checkVerbosity( level):
    """Return true if verbosity is at least as great as level.

        Parameters
        ----------
        level: int
            level of verbosity to be checked against global value

    """
    return (verbosity >= level)
