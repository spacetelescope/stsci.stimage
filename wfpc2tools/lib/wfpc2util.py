#! /usr/bin/env python

import sys
import time

# Utility functions and parameters for Wfpc2destrweak

QUIET = 0 # verbosity levels
VERBOSE = 1
VERY_VERBOSE = 2   
HUGE_VAL = 99999. # default for bias_thresh
                                                                                
# default values
verbosity = VERBOSE
force_alg_type = None
group = 4 
bias_thresh = HUGE_VAL
row_thresh = 0.0

def all_printMsg( message, level=VERBOSE):

    if verbosity >= level:     
      print message
      sys.stdout.flush()

def printMsg( message, level=QUIET):

    if verbosity >= level:
        print message
        sys.stdout.flush()

def setGroup( group_value):
        """ Copy group to a variable that is global for this file.
            @param group_value: value of group
            @type group_value: int
        """
        global group
        group = group_value            

def setBias_thresh( bias_thresh_value):
        """ Copy bias_thresh to a variable that is global for this file.
            @param bias_thresh_value: value of bias_thresh
            @type bias_thresh_value: float
        """
        global bias_thresh
        bias_thresh = bias_thresh_value

def setRow_thresh( row_thresh_value):
        """ Copy row_thresh to a variable that is global for this file.
            @param row_thresh_value: value of row_thresh
            @type row_thresh_value: float
        """
        global row_thresh
        row_thresh = row_thresh_value            

def setForce_alg_type( force_alg_type_value):
        """ Copy force_alg_type to a variable that is global for this file.
            @param force_alg_type_value: value of force_alg_type
            @type force_alg_type_value: string
        """
        global force_alg_type
        force_alg_type = force_alg_type_value                   

def setVerbosity( verbosity_level):
    """Copy verbosity to a variable that is global for this file.                                                              
       argument: verbosity_level -  an integer value indicating the level of verbosity
    """
                                                                                
    global verbosity
    verbosity = verbosity_level

def checkVerbosity( level):
    """Return true if verbosity is at least as great as level."""

    return (verbosity >= level)


