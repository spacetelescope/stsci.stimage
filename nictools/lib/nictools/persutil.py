#! /usr/bin/env python

from __future__ import division # confidence high
import sys
import time
import pyfits
from optparse import OptionParser
import os

# Utility functions and parameters for nic_rem_persist

QUIET = 0 # verbosity levels
VERBOSE = 1
VERY_VERBOSE = 2

# default values
verbosity = VERBOSE

PERSIST_LO = 0.5   # only used for pyraf version
USED_LO = 0.5   # only used for pyraf version

nref = os.path.expandvars('$nref')

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

def getPersist_lo( calcfile):
    """ Get value of persist_lo from BEPVALLO in the persistence model file PMODFILE
    or from a default specified by this module.
    This will only be called if user did not specify a value on the command-line.

    Parameters
    ----------
    calcfile : string
        input ped file

    Returns
    -------
    persist_lo :  float

    """

    # get pmodfile from header of calcfile, and get bepvallo from pmodfile
    fh_infile = pyfits.open( calcfile)
    pmodfile =  fh_infile[0].header.get( "PMODFILE" )
    pmodfile = os.path.join( nref, pmodfile.split('nref$')[1] )
    fh_pmod = pyfits.open( pmodfile)
    persist_lo =  fh_pmod[0].header.get( "BEPVALLO" )
    fh_infile.close(); fh_pmod.close()

    if (persist_lo == None ):# if keyword is not in model file, set to module default value
         persist_lo = PERSIST_LO

    return persist_lo


def getUsed_lo( calcfile):
    """  Get value of used_lo from BEPUSELO in the persistence model file PMODFILE
    or from a default specified by this module.
    This will only be called if user did not specify a value on the command-line.

    Parameters
    -----------
    calcfile : string
        input ped file

    Returns
    -------
    used_lo :  float
    """

    # get pmodfile from header of calcfile, and get bepuselo from pmodfile
    fh_infile = pyfits.open( calcfile)
    pmodfile =  fh_infile[0].header.get( "PMODFILE" )
    pmodfile = os.path.join( nref, pmodfile.split('nref$')[1])
    fh_pmod = pyfits.open( pmodfile)
    used_lo =  fh_pmod[0].header.get( "BEPUSELO" )
    fh_infile.close(); fh_pmod.close()

    if (used_lo == None ):    # if keyword is not in model file, set to module default value
         used_lo = USED_LO

    return used_lo


def getPersist_model( calcfile):
    """  Get name of persistence model from PMODFILE in the input file.
    This will only be called if user did not specify a model on the command-line.

    Parameters
    ----------
    calcfile : string
        input ped file

    Returns
    -------
    persist_model : string
    """

    # get pmodfile from header of calcfile
    fh_infile = pyfits.open( calcfile)
    persist_model = fh_infile[0].header.get( "PMODFILE" )
    persist_model = os.path.join( nref, persist_model.split('nref$')[1] )
    fh_infile.close();

    return persist_model


def getPersist_mask( calcfile):
    """ Get name of persistence mask from PMSKFILE in the input file
    This will only be called if user did not specify a model on the command-line.

    Parameters
    ----------
    calcfile : string
        input ped file

    Returns
    -------
    persist_mask : string
    """

    # get pmskfile from header of calcfile
    fh_infile = pyfits.open( calcfile)
    persist_mask = fh_infile[0].header.get( "PMSKFILE" )
    persist_mask = os.path.join( nref,persist_mask.split('nref$')[1])
    fh_infile.close();

    return persist_mask

def getOptions():

    usage = "usage:  %prog [options] calcfile targfile"
    parser = OptionParser( usage)

    parser.set_defaults( verbosity = VERBOSE)
    parser.add_option( "-q", "--quiet", action = "store_const",
                        const = QUIET, dest = "verbosity",
                        help = "quiet, print nothing")
    parser.add_option( "-v", "--verbose", action="store_const",
                        const = VERY_VERBOSE, dest="verbosity",
                        help="very verbose, print lots of information")
    parser.add_option( "-p", "--persist_lo", dest = "persist_lo",
                        help = "minimum allowed value of the persistence")
    parser.add_option( "-u", "--used_lo", dest = "used_lo",
                        help = "minimum allowed value of the fraction of pixels used")
    parser.add_option( "-d", "--persist_model", dest = "persist_model",
                        help = "filename containing persistence model")
    parser.add_option( "-m", "--persist_mask", dest = "persist_mask",
                        help = "filename containing pixel mask")

    (options, args) = parser.parse_args()

    setVerbosity( options.verbosity)
    verbosity = options.verbosity

    return options, args, parser
