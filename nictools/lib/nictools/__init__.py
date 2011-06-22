"""The nictools package holds Python tasks useful for analyzing NICMOS data.
Although these tasks are normally run via the STSDAS interface, they may be
used directly.

These tasks include:
  saaclean.py - for removing SAA persistence from an SAA-affected exposure
  puftcorr.py - for estimating and removing "Mr. Staypuft" artifact
  rnlincor.py - for correcting the countrate-dependent nonlinearity in
                a NICMOS image.

This release also includes alpha versions of the following tasks:
  CalTempFromBias: calculate the temperature from the bias present in an image
  NicRemPersist: remove a general persistence signal using a medianed
                 persistence model.

Utility and library functions used by these tasks are also included in this
module. 
"""
