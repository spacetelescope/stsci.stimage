#
#
#   MODULE:     geissupport.py
#   AUTHOR:     Christopher Hanley
#   HISTORY:    
#       0.1.0 -- 12 July 2004 -- Created -- CJH
#       0.1.1 -- 24 August 2004 -- Modified convertgeis2multifits to append
#               a _c0h.fits extension instead of _c0f.fits -- CJH
#
#
#
import numarray
import pyfits as P
import readgeis
from readgeis import readgeis
import os

__version__ = '0.1.1'

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
    FITS format.  Also, convert at GEIS DQ files 
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
    
def iswaveredfits(inputname):
    """
    Function that determines if the file named as input
    is in wavered fits format.
    
    iswaveredfits returns boolean True or False as output.
    """
    
    pfobj = P.open(inputname)
    
    if (pfobj[0].data != None) and (len(pfobj[0].data.shape) == 3):
        return True
    
    return False

        
