#
#   Authors: Christopher Hanley
#   Program: stis_input.py
#   Purpose: Class used to model STIS specific instrument data.
#   History:
#           Version 0.1.0, ----------- Created
__version__ = '0.1.0'

import pydrizzle
from pydrizzle import fileutil

import makewcs

from input_image import InputImage

def checkSTIS(files):

    """ Checks that MAKEWCS is run on any STIS image in 'files' list. """

    for p in files:
        if fileutil.getKeyword(p,'instrume') == 'STIS':
            print('\nNote: Synchronizing STIS WCS to specified distortion coefficients table\n')
            # Update the CD matrix using the new IDCTAB
            # Not perfect, but it removes majority of errors...
            makewcs.run(image=p)


class STISInputImage (InputImage):

    SEPARATOR = '_'

    def __init__(self, input,dqname,memmap=1):
        InputImage.__init__(self,input,dqname,memmap=1)
        # define the cosmic ray bits value to use in the dq array
        self.cr_bits_value = 4096
        # Effective gain to be used in the driz_cr step.  Since the
        # ACS images have already benn converted to electons per
        # second, the effective gain is 1.
        self._effGain = 1

    def doUnitConversions(self):
        self._effGain = 1
        
    def setInstrumentParameters(self, instrpars, pri_header):
        """ This method overrides the superclass to set default values into
            the parameter dictionary, in case empty entries are provided.
        """
        if self._isNotValid (instrpars['gain'], instrpars['gnkeyword']):
            instrpars['gnkeyword'] = 'ATODGAIN'
        if self._isNotValid (instrpars['rdnoise'], instrpars['rdnkeyword']):
            instrpars['rdnkeyword'] = 'READNSE'
        if self._isNotValid (instrpars['exptime'], instrpars['expkeyword']):
            instrpars['expkeyword'] = 'EXPTIME'
        if instrpars['crbit'] == None or instrpars['crbit'] == 0:
            instrpars['crbit'] = self.cr_bits_value

        self._gain      = self.getInstrParameter(instrpars['gain'], pri_header,
                                                 instrpars['gnkeyword'])
        self._rdnoise   = self.getInstrParameter(instrpars['rdnoise'], pri_header,
                                                 instrpars['rdnkeyword'])
        self._exptime   = self.getInstrParameter(instrpars['exptime'], pri_header,
                                                 instrpars['expkeyword'])
        self._crbit     = instrpars['crbit']

        if self._gain == None or self._rdnoise == None or self._exptime == None:
            print 'ERROR: invalid instrument task parameter'
            raise ValueError

#        InputImage.setInstrumentParameters(self, instrpars, pri_header)

class CCDInputImage (STISInputImage):

    def __init__(self, input, dqname, memmap=1):
        STISInputImage.__init__(self,input,dqname,memmap=1)
        self.instrument = 'STIS/CCD'
        self.full_shape = (1024,1024)


