"""
puftcorr: Module for estimating and removing "Mr. Staypuft" signal from
          a NICMOS exposure.

Usage:    Normally used via the STSDAS task puftcorr in the nicmos package.
          To use as pure python, just invoke the clean method:
          >>> puftcorr.clean('inputfile.fits','outputfile.fits')

For more information:
          Additional user information, including parameter definitions and more
          examples, can be found in the help file for the STSDAS puftcorr task,
          located in nicmos$doc/puftcorr.hlp.

          The algorithm and IDL prototype were developed by L.Bergeron, but
          never made publicly available.
          
Dependencies:
          numpy v1.0.2dev3534 or higher
          pyfits v1.1b4 or higher
          convolve (version ?? or higher)
          ndimage  (version ?? or higher)

"""
from puftcorr import *
__version__="0.15"
__vdate__="2007-02-05"
