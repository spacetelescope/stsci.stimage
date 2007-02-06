"""
saaclean: Module for estimating and removing persistent CR signal due to a
          prior SAA passage.

Usage:    Normally used via the STSDAS task saaclean in the nicmos package.
          To use as pure python, create a params object to override any of
          the default parameters if desired, then invoke clean:
          >>> mypars=saaclean.params(thresh=0.23)
          >>> saaclean.clean('inputfile.fits','outputfile.fits',pars=mypars)

For more information:
          Additional user information, including parameter definitions and more
          examples, can be found in the help file for the STSDAS saaclean task,
          located in nicmos$doc/saaclean.hlp.

          The algorithm and IDL prototype are described in the NICMOS
          ISR 2003-009, by Bergeron and Dickinson, available through the NICMOS
          webpage.

Dependencies:
          numpy 1.0.2.dev3534 or higher
          pyfits v1.1b4 or higher
          imagestats v1.1.0 or higher

"""
from saaclean import * #reveals everything
__version__="1.0d1"
__vdate__="2007-02-05"
