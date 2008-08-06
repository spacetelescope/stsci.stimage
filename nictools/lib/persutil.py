#! /usr/bin/env python

import sys
import time

# Utility functions and parameters for nic_rem_persist

QUIET = 0 # verbosity levels
VERBOSE = 1
VERY_VERBOSE = 2   
                                                                                
# default values
verbosity = VERBOSE
persist_lo = 0.5   # only used for pyraf version
used_lo = 0.5   # only used for pyraf version


def all_printMsg( message, level=VERBOSE):

    if verbosity >= level:     
      print message
      sys.stdout.flush()

def printMsg( message, level=QUIET):

    if verbosity >= level:
        print message
        sys.stdout.flush()

def setVerbosity( verbosity_level):
    """Copy verbosity to a variable that is global for this file.                                                              
       argument: verbosity_level -  an integer value indicating the level of verbosity
    """
                                                                                
    global verbosity
    verbosity = verbosity_level

def checkVerbosity( level):
    """Return true if verbosity is at least as great as level."""

    return (verbosity >= level)
