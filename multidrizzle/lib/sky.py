# SKY  --  Python version of sky task in STSDAS/DITHER.
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
import skyCalc

# Sotware Version
__version__ = "0.1.1"

class Sky:
    """Interface class for driving the functionality of the SkyCalc class"""

    def __init__(self,
            imgArray,               # {'Input image as a numarray object'}
            header,                 # {'Input image header as a pyfits header object'}
            maskArray = None,       # {'Numarray mask object to apply to the input image'}
            lower     = -99.,       # {'Lower limit of usable data'}
            upper     = 4096.,      # {'Upper limit of usable data '}
            nclip     = 5,          # {"Number of sky clipping iterations to compute"}
            lsigma    = 4,          # {"Lower sigma clipping value"}
            usigma    = 4,          # {"Upper sigma clipping value"}
            subsky    = 'yes',      # {'Subtract sky from input images ?'}
            binwidth  = 0.5,        # {'Binning width for histogram used in calcualtion of the mode'}
            stat      = 'mode',     # {prompt='Sky correction statistics',enum='mean|mode|median'}
            skyname   = 'SKYVAL'    # {'Header parameter to be updated with sky'}
            ):

        # Define program input variables
        self.__lower = lower
        self.__upper = upper
        self.__subsky = subsky
        self.__nclip = nclip
        self.__lsigma = lsigma
        self.__usigma = usigma
        self.__binwidth = binwidth
        self.__stat = stat
        self.__skyname = skyname
        self.__imgArray = imgArray
        self.__header = header
        self.__maskArray = maskArray

        # Define program output variables
        self.skyvalue = 0

        # Test skyname to determine if the background keyword is currently supported.
        if ( self.__skyname == None ):
            raise ValueError, 'valid background keywords not supplied, exiting program...'

        # Test stat to verify 'mode' or 'mean' as the desired calculation
        if ( (self.__stat.upper() != 'MEAN') and (self.__stat.upper() != 'MODE')
         and (self.__stat.upper() != 'MEDIAN')):
            print 'Sky correction statistics can only be computed using a MEAN, MODE, or MEDIAN'

        # Compute the sky value
        __SkyCalcObj = skyCalc.SkyCalc(self.__imgArray, self.__header, maskArray = self.__maskArray,
                        lower = self.__lower, upper = self.__upper, nclip = self.__nclip,
                        lsigma = self.__lsigma, usigma=self.__usigma, binwidth = self.__binwidth,
                        stat = self.__stat, skyname = self.__skyname)
        self.skyvalue = __SkyCalcObj.sky

        # Update the header with new sky value
        __SkyCalcObj.updateHeaderSkyValue()

        # Subtract the sky value from image data
        if ( self.__subsky.lower() == 'yes' ):
            print 'Subtracting sky value of ', self.skyvalue
            __SkyCalcObj.subtractSky();

        del __SkyCalcObj

        print 'Sky exiting...'
