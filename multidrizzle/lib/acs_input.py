#
#   Authors: Warren Hack, Ivo Busko, Christopher Hanley
#   Program: acs_input.py
#   Purpose: Class used to model ACS specific instrument data.
#   History:
#           Version 0.0.0, ----------- Created
#           Version 0.1.6, 03/23/04 -- Added the attibute cr_bits_value to
#               indicate what bit value a cosmic rate hit will represent in
#               the data quality files. -- CJH
#           Version 0.1.7, 04/19/04 -- Added code to set instrument-dependent
#               parameters. -- IB
#           Version 0.1.8, 04/22/04 -- Added a new attribute for effective
#               gain, 'effgain'.  Value of effgain is 1 for ACS images.  --CJH
#           Version 0.1.9  05/08/04 -- Refactored code to removed hard coded gain
#               and readnoise values.  Now inherit those attributes from InputImage
#               class. -- CJH
#           Version 0.1.10 05/27/04 -- Modified setInstrumentParameters inheritence in
#               order to support ACS/SBC data. -- CJH/WJH/IB
#           Version 0.1.11 06/24/04 -- Moved _isNotValid method to InputImage 
#               super class so that it can be inherited by all instrument
#               specific modules. -- CJH
#           Version 0.1.12 07/02/04 -- Revised to call 'makewcs'.
#           Version 0.1.13 07/08/04 -- Updated names of key in dictionaries. -- CJH
__version__ = '0.1.13'

import pydrizzle
from pydrizzle import fileutil

import makewcs

from input_image import InputImage

def checkACS(files):

    """ Checks that MAKEWCS is run on any ACS image in 'files' list. """

    for p in files:
        if fileutil.getKeyword(p,'instrume') == 'ACS':
            print('\nNote: Synchronizing ACS WCS to specified distortion coefficients table\n')
            # Update the CD matrix using the new IDCTAB
            # Not perfect, but it removes majority of errors...
            makewcs.run(image=p)


class ACSInputImage (InputImage):

    SEPARATOR = '_'

    def __init__(self, input,dqname,memmap=1):
        InputImage.__init__(self,input,dqname,memmap=1)
        # define the cosmic ray bits value to use in the dq array
        self.cr_bits_value = 4096
        # Effective gain to be used in the driz_cr step.  Since the
        # ACS images have already benn converted to electons per
        # second, the effective gain is 1.
        self._effGain = 1

    def _isSubArray(self):
        _subarray = False
        _ltv1 = float(fileutil.getKeyword(parlist['data'],'LTV1'))
        _ltv2 = float(fileutil.getKeyword(parlist['data'],'LTV2'))
        if (_ltv1 != 0.) or (_ltv2 != 0.):
            _subarray = True
        _naxis1 = float(fileutil.getKeyword(parlist['data'],'NAXIS1'))
        _naxis2 = float(fileutil.getKeyword(parlist['data'],'NAXIS2'))
        if (_naxis1 < self.full_shape[0]) or (_naxis2 < self.full_shape[0]):
            _subarray = True
        return _subarray

    def setInstrumentParameters(self, instrpars, pri_header):
        """ This method overrides the superclass to set default values into
            the parameter dictionary, in case empty entries are provided.
        """
        if self._isNotValid (instrpars['gain'], instrpars['gnkeyword']):
            instrpars['gnkeyword'] = 'ATODGNA,ATODGNB,ATODGNC,ATODGND'
        if self._isNotValid (instrpars['rdnoise'], instrpars['rdnkeyword']):
            instrpars['rdnkeyword'] = 'READNSEA,READNSEB,READNSEC,READNSED'
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

class WFCInputImage (ACSInputImage):

    def __init__(self, input, dqname, memmap=1):
        ACSInputImage.__init__(self,input,dqname,memmap=1)
        self.instrument = 'ACS/WFC'
        self.full_shape = (4096,2048)


class HRCInputImage (ACSInputImage):

    def __init__(self, input, dqname, memmap=1):
        ACSInputImage.__init__(self, input, dqname, memmap=1)
        self.instrument = 'ACS/HRC'        
        self.full_shape = (1024,1024)


class SBCInputImage (ACSInputImage):

    def __init__(self, input, dqname, memmap=1):
        ACSInputImage.__init__(self,input,dqname,memmap=1)
        self.full_shape = (1024,1024)
        self.instrument = 'ACS/SBC'

    def setInstrumentParameters(self, instrpars, pri_header):
        """ Sets the instrument parameters.
        """
        if self._isNotValid (instrpars['exptime'], instrpars['expkeyword']):
            instrpars['expkeyword'] = 'EXPTIME'
        if instrpars['crbit'] == None or instrpars['crbit'] == 0:
            instrpars['crbit'] = self.cr_bits_value

        self._gain      = 1
        self._rdnoise   = 0
        self._exptime   = self.getInstrParameter(instrpars['exptime'], pri_header,
                                                 instrpars['expkeyword'])
        self._crbit     = instrpars['crbit']

        if self._exptime == None:
            print 'ERROR: invalid instrument task parameter'
            raise ValueError
