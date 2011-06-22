# deriv.py
from __future__ import division # confidence high

__version__ = '0.1.0'

import pyfits
import quickDeriv

def run(fname):
    """ Runs quickDeriv on a Python list of input files."""
    print "Running quickDeriv on ", fname

    # OPEN THE INPUT IMAGE IN READ ONLY MODE 
    img = pyfits.open(fname,mode='readonly', memmap=0)

    # calling qderiv with the assumption that the 
    # input file is a simple FITS file.
    absderiv = quickDeriv.qderiv(img["PRIMARY"].data)

    # Generate output filename
    _indx = fname.find('.fits')
    if _indx < -1: _indx = len(fname)
    outfilename = fname[:_indx]+'_deriv.fits'

    # WRITE THE OUTPUT IMAGE TO A FITS FILE
    outfile = pyfits.open(outfilename,'append')
    outhdu = pyfits.PrimaryHDU(data=absderiv)
    outfile.append(outhdu)

    # CLOSE THE IMAGE FILES
    outfile.close()
    img.close()
    del outfile
    del img
        
        
def test(inputFile):
    # SPECIFY THE INPUT FILE LIST
    inputFile = '@list.txt'
    outfilename = 'sample_deriv.fits'

    if inputFile[0] == "@":
        # OPEN THE INPUT FILE LIST AND GET INPUT FILENAME
        fileList = open(inputFile, "r")
        numFiles = fileList.readlines()
        # CLOSE THE INPUT FILE LIST
        fileList.close()
    else:
        numFiles = inputFile

    for line in numFiles:
        # STRIP THE CARRIAGE RETURN CHARACTER
        line = line.rstrip()
        if line == "": break

        print "WORKING ON FILE ", line

        # OPEN THE INPUT IMAGE IN READ ONLY MODE
        img = pyfits.open(line,mode='readonly', memmap=0)

        absderiv = quickDeriv.qderiv(img["SCI"].data)

        # WRITE THE OUTPUT IMAGE TO A FITS FILE
        outfile = pyfits.open(outfilename,'append')
        outhdu = pyfits.PrimaryHDU(data=absderiv)
        outfile.append(outhdu)

        # CLOSE THE IMAGE FILES
        outfile.close()
        img.close()
        del outfile
        del img

    print "EXITING PROGRAM..."

