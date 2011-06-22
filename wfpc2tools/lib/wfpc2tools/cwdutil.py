#! /usr/bin/env python

from __future__ import division # confidence high
import sys
import time

# Utility functions and parameters for CalWFPC2Destreak

QUIET = 0 # verbosity levels
VERBOSE = 1
VERY_VERBOSE = 2   
                                                                                
# default value
verbosity = VERBOSE

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


