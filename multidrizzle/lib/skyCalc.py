# SkyCalc  --  Python version of gsky.
#
#  IRAF HISTORY
#  ------------
#  26-Mar-97:  Task created (I. Busko)
#  21-Aug-97:  Directory for temporary files (IB)
#  27-Oct-97:  Renamed 'sky'. The former gsky task was renamed 'ogsky' and
#              hidden (IB)
#  11-Nov-97:  Checks if sky-subtracted by crrej (IB)
#  09-Feb-98:  New design (IB)
#  17-Apr-98:  Report all groups in no-subtract, no-verbose mode (IB)
#  20-Oct-98:  Truncate string passed to imcalc (IB)
#  25-Feb-99:  Add expname to process countrate images (JC Hsu)
#  19-May-99:  Add bunit to process countrate images (JC Hsu)
#  15-Jun-99:  change width from int to real (JC Hsu)
#  03-Feb-2000: Initialize kwrd and spell out wkstr's (JC Hsu)
#  24-Jul-2002: Corrected the imcalc 'str(pval)' error (WJ Hack)
#
#  PYTHON HISTORY
#  --------------
#  12-Oct-2003: Converted the basic functionality of this task to PYTHON (C.J. HANLEY)
#  03-Dec-2003: 1) Refactored exisiting code to reduce complexity
#                               2) Removed dependence on specific input header keywords.  Will now work
#                                  with any comma seperated string of keywords.
#                               (C.J. Hanley)
#

# External Modules
import numarray
import pytools.imagestats
from pytools.imagestats import ImageStats as imStat

# Software Version
__version__ = "0.1.1"

class SkyCalc:
    """Class to do the sky subtraction of an astronomical image and update its header information."""

    def __init__(self,
            imgArray,              # {"Input image as a numarray object"}
            header    = None,      # {"Input image header as a pyfits header object"}
            maskArray = None,      # {"Numarray mask object to apply to the input image"}
            lower     = -99.,      # {"Lower limit of usable data"}
            upper     = 4096.,     # {"Upper limit of usable data "}
            nclip     = 5,         # {"Number of sky clipping iterations to compute"}
            lsigma    = 4,         # {"Lower sigma clipping value"}
            usigma    = 4,         # {"Upper sigma clipping value"}
            binwidth  = 0.5,       # {"Binning width for histogram used in calcualtion of the mode"}
            stat      = 'mode',    # {"Sky correction statistics',enum='mean|mode|median"}
            skyname   = None       # {"Header parameter to be updated with sky"}
            ):

        # Define program input variables
        self.__lower = lower
        self.__upper = upper
        self.__nclip = nclip
        self.__lsigma = lsigma
        self.__usigma = usigma
        self.__binwidth = binwidth
        self.__skyname = skyname
        self.__imgArray = imgArray
        self.__header = header
        self.__maskArray = maskArray
        self.__stat = stat

        # Define program output Variables
        self.sky= 0

        # Apply the image mask if available
        if ( self.__maskArray == None ):
            __tmpImgArray = self.__imgArray.copy()
        else:
            # Assume that the mask is 0's and 1's with 1 = good
            # Make a copy of the input image array to appy the mask to
            __tmpImgArray = self.__imgArray.copy()
            # Casting the self.upper as an array of Float32 is a cludge to get around a casing error in numarray.
            # This fix should be removed when the bug fix is included in a future version of numarray. CJH 11/3/03
            numarray.putmask(__tmpImgArray, numarray.logical_not(self.__maskArray), numarray.array((self.__upper)+1, type='Float32'))

        #Compute the image statistics
        """ The Python version of imstats, the program imagestats.py, is used to compute the sky.   """
        self.__imgArrayStats = imStat(__tmpImgArray, lower = self.__lower, upper = self.__upper, nclip = self.__nclip,
                                        lsig = self.__lsigma, usig= self.__usigma, binwidth = self.__binwidth, fields=self.__stat)

        # Delete the "__tmpImgArray" object
        del __tmpImgArray

        self.sky = self.__extractSkyValue()

    def __extractSkyValue(self):
        # Extract the computed sky mode value
        if (self.__stat.lower() =="mode"):
            return self.__imgArrayStats.mode
        elif (self.__stat.lower() == "mean"):
            return self.__imgArrayStats.mean
        else:
            return self.__imgArrayStats.median

    # I/O routines
    def printSky(self):
        """ Print the computed value for the sky """
        print "The ",self.__stat.lower()," of the sky is ",self.sky,"."

    # Header Update Routines
    def updateHeaderSkyValue(self):
        """   Add the computed sky value to the current background level and update the header keyword   """
        # Test for the existence of a header
        if (self.__header != None):
            # If the "skyname" keyword does not exist add it to the header
            for __keyword in self.__skyname.split(','):
                if ( not self.__header.has_key(__keyword) ):
                    # Add th keyword value pair
                    self.__header.update(__keyword, self.sky)
                else: # Otherwise extract the initial background level of the image.
                    __initialBackground = self.__header[__keyword]
                    self.__header[__keyword] = __initialBackground + self.sky
        else:
            print "ABORT: NO HEADER PROVIDED!"

    # Sky Subtraction Routines
    def subtractSky(self):
        """   Subtract the computed sky value from the imput image and add
        the header keyword name = value pair "SUBSKY = T"         """
        numarray.subtract(self.__imgArray,self.sky,self.__imgArray)

        if ( self.__header != None ):
            # Update the header keyword name = value pair "SUBSKY =  T"
            self.__header.update('SUBSKY','T')
