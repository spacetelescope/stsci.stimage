#! /usr/bin/env python

import sys
import time
import pyfits
from optparse import OptionParser  

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

def getPersist_lo( input_file, options):
    """
    @param input_file: input ped file
    @type iput_file: string
    @options: array of values to sigma clip
    @type clip:  Float32

    @return: persist_lo
    @rtype:  float    
    """

    if (options.persist_lo == None ):    # get pmodfile from header of input_file, and get bepvallo from pmodfile           
         fh_infile = pyfits.open( input_file)
         pmodfile =  fh_infile[0].header.get( "PMODFILE" )
         fh_pmod = pyfits.open( pmodfile)         
         persist_lo =  fh_pmod[0].header.get( "BEPVALLO" )
         fh_infile.close(); fh_pmod.close()
    else:
         persist_lo = options.persist_lo

    return persist_lo


def getUsed_lo( input_file, options):
    """
    @param input_file: input ped file
    @type iput_file: string
    @options: array of values to sigma clip
    @type clip:  Float32

    @return:  used_lo
    @rtype:  float    
    """
    if (options.used_lo == None ):    # get pmodfile from header of input_file, and get bepuselo from pmodfile           
         fh_infile = pyfits.open( input_file)
         pmodfile =  fh_infile[0].header.get( "PMODFILE" )
         fh_pmod = pyfits.open( pmodfile)         
         used_lo =  fh_pmod[0].header.get( "BEPUSELO" )
         fh_infile.close(); fh_pmod.close()
    else:
         used_lo = options.used_lo

    return used_lo
             

def getPersist_model( input_file, options):
    """
    @param input_file: input ped file
    @type iput_file: string
    @options: array of values to sigma clip
    @type clip:  Float32

    @return: persist_model
    @rtype: string    
    """

    if (options.persist_model == None ):    # get pmodfile from header of input_file
         fh_infile = pyfits.open( input_file)
         persist_model = fh_infile[0].header.get( "PMODFILE" )
         fh_infile.close(); 
    else:
         persist_model = options.persist_model

    return persist_model


def getPersist_mask( input_file, options):
    """
    @param input_file: iput ped file
    @type iput_file: string
    @options: array of values to sigma clip
    @type clip:  Float32

    @return: persist_mask
    @rtype: string    
    """
    if (options.persist_mask == None ):    # get pmskfile from header of input_file
             fh_infile = pyfits.open( input_file)
             persist_mask = fh_infile[0].header.get( "PMSKFILE" )
             fh_infile.close(); 
    else:
             persist_mask = options.persist_mask

    return persist_mask

def getOptions():

    usage = "usage:  %prog [options] inputfile"
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

