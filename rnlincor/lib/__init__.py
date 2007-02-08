"""
rnlincor: Module to correct for the countrate-dependent nonlinearity in 
          a NICMOS image.

Usage:    Normally used via the STSDAS task rnlincor in the nicmos package.
          To use as pure python, just invoke the run method:
          >>> rnlincor.run('inputfile.fits','outputfile.fits')
          It may also be run from the shell:
          % rnlincor.py infile.fits [outfile.fits] [--nozpcorr]
          

For more information:
          Additional user information, including parameter definitions and more
          examples, can be found in the help file for the STSDAS rnlincor task,
          located in nicmos$doc/rnlincor.hlp.

          This task is based on prototype code developed by R. de Jong. The
          algorithm is described in more detail in ISR NICMOS 2006-003 by
          de Jong.

Dependencies:
          numpy v1.0.2dev3534 or higher
          pyfits v1.1b4 or higher

"""
from rnlincor import *
__version__="0.6"
__vdate__="2007-02-08"
