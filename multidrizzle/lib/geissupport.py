#
#
#   MODULE:     geissupport.py
#   AUTHOR:     Christopher Hanley
#   HISTORY:    
#       0.1.0 -- 12 July 2004 -- Created -- CJH
#       0.1.1 -- 24 August 2004 -- Modified convertgeis2multifits to append
#               a _c0h.fits extension instead of _c0f.fits -- CJH
#       0.2.0 -- 23 December 2004 -- Added new methods parseWFPC2
#                to support the new MUltidrizzle interface code.
#       0.2.1 -- 03 January 2005 -- Added support for the handling
#                of the zero exposure time input files.  If a file
#                has exptime = 0.0, the functions issues a message
#                stating that the file is not valid input and returns
#                a None to the calling program.

import numarray
import pyfits as P
import readgeis
from readgeis import readgeis
import os
import pydrizzle
from pydrizzle import fileutil

__version__ = '0.2.1'

def convertgeis2multifits(geisfilename):
    """
    The function is used to convert WFPC2 GEIS data files to
    multiextension FITS format files for use by multidrizzle.
    
    This program takes as input the name of a WFPC2 GEIS file
    the user wishes to convert to the multiextension FITs format.
    """

    geis = readgeis(geisfilename)
    
    # Extract the rootname of the inputfile.
    rootname = geisfilename[0:geisfilename.rfind('.')]

    # Extract all the letters, except the last, from the
    #file extension and append the letter 'f'.
    extn=geisfilename[geisfilename.rfind('.')+1:-1]+"h"
    
    # if the file you wish to create already exists, delete it
    dirfiles = os.listdir(os.curdir)
    if (dirfiles.count(rootname+"_"+extn+".fits") > 0):
        os.remove(rootname+"_"+extn+".fits")
        print "! Replacing ",rootname+"_"+extn+".fits"
    
    # Write out the fits file using the new extension name.
    print "! ",geisfilename, " -> ",rootname+"_"+extn+".fits"
    geis.writeto(rootname+"_"+extn+".fits")
    
    # Return the name of the new extension being used
    return rootname+"_"+extn+".fits"
    
def convertgeisinlist(pythonlist):
    """
    Given a python list object as input, covert all
    of the GEIS files in the list to multiextension
    FITS format.  Also, convert any GEIS DQ files 
    as well.
    
    This method returns as output a python list containing
    the names of all of the newly created FITS files.
    """

    flist = []
    
    for geisfile in pythonlist:
        newfile = convertgeis2multifits(geisfile)
        flist.append(newfile)

        # Check to see if there is a DQ file for the geisfile, 
        # if so, convert it to multiextension FITS format
        verifyDQfile(geisfile)
        
    return flist

def verifyDQfile(filename):
    """
    This method takes as geis data file name as input and
    determines if a corresponding DQ file exists.  If the
    file exists it is converted to multiextension FITS format.
    """

    # Extract the rootname of the filename.
    rootname = filename[0:filename.rfind('.')]

    # Get the DQ filename
    dqfilename = findvalidgeisfile(rootname, geistype='dq')       

    # If the DQ file exists, convert it to multiextension
    # FITS format.
    if dqfilename != None:
        newfile = convertgeis2multifits(dqfilename)
    else:
        print "! WARNING: No DQ file found for ", filename
        print "! WARNING: All pixels will be treated as good"
            

def findvalidgeisfile(rootname, geistype='data'):
    """
    Given a rootname for a file, this method returns the
    full name of the GEIS file in that directory if 
    it exists.  The geistype parameter is used to specify the
    type of GEIS file being search.
    
    geistype = "data": Looking for .c0h or .hhh file
    geistype = "dq"  : Looking for .c1h file
    """
    
    # Create a list of the files in the working directory
    dirfiles = os.listdir(os.curdir)
    
    if (geistype.lower() == 'data'):
        geisextnlist = ['.hhh','.c0h']
    elif (geistype.lower() == 'dq'):
        geisextnlist = ['.c1h']
    else:
        raise ValueError, "FIND VALID GEIS FILE DOES NOT SUPPORT THAT GEISTYPE VALUE"
        
    for extn in geisextnlist:
        if (dirfiles.count(rootname+extn) > 0):
            return (rootname+extn) 
    return None
    
def iswaiverfits(inputname):
    """
    Function that determines if the file named as input
    is in wavered fits format.
    
    iswaveredfits returns boolean True or False as output.
    """
    
    pfobj = P.open(inputname)
    
    if (pfobj[0].data != None):
        return True
    
    return False

        
def parseWFPC2(filename):
    """
    FUNCTION: parseWFPC2
    PURPOSE : The parswWFPC2 function accepts as input a string representing a GEIS format file.
              The function extracts the rootname of the input file.  Determines if the file is
              a support GEIS format file.  If it is a supported file type, it is converted to
              a multiextension FITS file.
    INPUT   : filename - string object
    OUTPUT  : newfilename - string representing the name of the file to be processed.
              parseWFPC2flag -          

    """
    # Initialize variables
    parseWFPC2flag = False
    
    # Check to see if the file has an EXPTIME value of zero.  If so return
    # no new file name and a parseWFPC2flag value of False
    if float(fileutil.getKeyword(filename,'EXPTIME')) == 0.0:
        msgstr =  "####################################\n"
        msgstr += "#                                  #\n"
        msgstr += "# WARNING:                         #\n"
        msgstr += "#  EXPTIME keyword value of 0.0 in #\n"
        msgstr += "         " + str(filename) +"\n"
        msgstr += "#  has been detected.  Images with #\n"
        msgstr += "#  no exposure time will not be    #\n"
        msgstr += "#  used during processing.  If you #\n"
        msgstr += "#  wish this file to be used in    #\n"
        msgstr += "#  processing please give EXPTIME  #\n"
        msgstr += "#  a valid non-zero value.         #\n"
        msgstr += "#                                  #\n"
        msgstr += "####################################\n"
        print msgstr   
        return None,False
    
    # Verify that input is not Waiver FITS format.  Perform this test if the file
    # is FITS format.
    if (filename.rfind('.fits') != -1):
        if (iswaiverfits(filename) == True):
            errormsg =  "###################################\n"
            errormsg += "#                                 #\n"
            errormsg += "# ERROR:                          #\n"
            errormsg += "#  Input image:                   #\n"
            errormsg += str(filename)+"\n"
            errormsg += "#  is a waivered FITS image       #\n"
            errormsg += "#                                 #\n"
            errormsg += "#  Multidrizzle does not support  #\n"
            errormsg += "#  this file format.  Please      #\n"
            errormsg += "#  convert this file to either    #\n"
            errormsg += "#  GEIS format or multi extension #\n"
            errormsg += "#  FITS format.                   #\n"
            errormsg += "#                                 #\n"
            errormsg += "###################################\n"
            raise ValueError, errormsg
    
    # Extract the rootname of the filename.
    rootname = filename[0:filename.rfind('.')]
    
    # Find a valid GEIS file on disk for the rootname provided.
    validgeisfile = findvalidgeisfile(rootname,geistype='data')
    
    # if the file provided is a GEIS file convert it to a multi
    # extension FITS format file.
    if validgeisfile != None:
        # Set the boolean flag 
        parseWFPC2flag = True
        newfilename = convertgeis2multifits(validgeisfile)
        
        # Check to see if there is a DQ file for the geisfile, 
        # if so, convert it to multiextension FITS format
        verifyDQfile(validgeisfile)
    else:
        newfilename = filename
    return newfilename,parseWFPC2flag
        
