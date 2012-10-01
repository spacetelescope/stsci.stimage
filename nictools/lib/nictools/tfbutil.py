#! /usr/bin/env python

from __future__ import division  # confidence high
import sys, os

# Utility functions and parameters for temp_from_bias

QUIET = 0 # verbosity levels
VERBOSE = 1
VERY_VERBOSE = 2

DO_NOT_WRITE_KEYS = 0 # dry_run levels
DO_WRITE_KEYS = 1

# default values
verbosity = VERBOSE
hdr_key = "TFBTEMP"
err_key = "TFBERR"
edit_type = "RAW"
noclean = False
force = None
dry_run = DO_WRITE_KEYS

nref_par = os.path.expandvars('$nref')

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

        Parameters
        ----------
        verbosity_level L int
            an integer value indicating the level of verbosity
    """

    global verbosity
    verbosity = verbosity_level

def checkVerbosity( level):
    """Return true if verbosity is at least as great as level."""

    return (verbosity >= level)

def setHdr_key( hdr_key_value):
    """Copy hdr_key to a variable that is global for this file.

        Parameters
        ----------
        hdr_key : string
            a string for the keyword name to write
    """

    global hdr_key
    hdr_key = hdr_key_value

def setErr_key( err_key_value):
    """Copy err_key to a variable that is global for this file.

        Parameters
        ----------
        err_key : string
            a string for the keyword for the error estimate
    """

    global err_key
    err_key = err_key_value

def setEdit_type_key( edit_type_value):
    """Copy edit_type_key to a variable that is global for this file.

        Parameters
        ----------
        edit_type_key : string
            a string for the keyword name to write
    """

    global edit_type
    edit_type = edit_type_value

def setNoclean( noclean_value):
    """Copy no_clean to a variable that is global for this file.

        Parameters
        ----------
        no_clean : string
            string that is either True or False
    """

    global noclean
    noclean = noclean_value

def setNref( nref_value):
    """Copy nref to a variable that is global for this file.


        Parameters
        ----------
        nref : string
            string for name of directory containing nonlinearity file
    """

    global nref
    nref = nref_value

def setForce( force_value ):
    """Copy force to a variable that is global for this file.

        Parameters
        ----------
        force : string
            string that is either None, Q, B, or A
    """

    global force
    force = force_value

def setDry_run( dry_run_value):
    """Copy dry_run to a variable that is global for this file.


        Parameters
        ----------
        dry_run : string
            string that is either True or False
    """

    global dry_run
    dry_run = dry_run_value
