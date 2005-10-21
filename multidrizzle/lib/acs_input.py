#
#   Authors: Warren Hack, Ivo Busko, Christopher Hanley, David Grumm
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
#           Version 0.1.14 07/29/04 -- Plate scale is now defined by the Pydrizzle
#               exposure class by use of a plate scale variable passed in through
#               the constructor.
#           Version 0.1.15 09/15/04 -- Modified SBCInputImage for overloading of
#               gain and readnoise values by user input.  Also modified handling
#               of an input CR bit value of 0 to be converted to a None type.  This
#               allows the DQ array update step to be turned off in DRIZ_CR.
#           Version 1.0.0 06/02/05 -- Calculates direction of CTE tail for cosmic rays
#               for each ACS instrument, which may depend on chip and/or amp  

__version__ = '1.0.0'

import pydrizzle
from pydrizzle import fileutil


from input_image import InputImage


class ACSInputImage (InputImage):

    SEPARATOR = '_'

    def __init__(self, input,dqname,platescale,memmap=0):
        InputImage.__init__(self,input,dqname,platescale,memmap=0)
        # define the cosmic ray bits value to use in the dq array
        self.cr_bits_value = 4096
        self.platescale = platescale
        
        # Effective gain to be used in the driz_cr step.  Since the
        # ACS images have already been converted to electrons,
        # the effective gain is 1.
        self._effGain = 1

    def doUnitConversions(self):
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
        if self._isNotValid (instrpars['rdnoise'], instrpars['rnkeyword']):
            instrpars['rnkeyword'] = 'READNSEA,READNSEB,READNSEC,READNSED'
        if self._isNotValid (instrpars['exptime'], instrpars['expkeyword']):
            instrpars['expkeyword'] = 'EXPTIME'
        if instrpars['crbit'] == None:
            instrpars['crbit'] = self.cr_bits_value
         
        self._gain      = self.getInstrParameter(instrpars['gain'], pri_header,
                                                 instrpars['gnkeyword'])
        self._rdnoise   = self.getInstrParameter(instrpars['rdnoise'], pri_header,
                                                 instrpars['rnkeyword'])
        self._exptime   = self.getInstrParameter(instrpars['exptime'], pri_header,
                                                 instrpars['expkeyword'])
        self._crbit     = instrpars['crbit']

        if self._gain == None or self._rdnoise == None or self._exptime == None:
            print 'ERROR: invalid instrument task parameter'
            raise ValueError

class WFCInputImage (ACSInputImage):

    def __init__(self, input, dqname, platescale, memmap=0):
        ACSInputImage.__init__(self,input,dqname,platescale,memmap=0)
        self.instrument = 'ACS/WFC'
        self.full_shape = (4096,2048)
        self.platescale = platescale

        if ( self.extn == 'sci,1') : # get cte direction, which depends on which chip but is independent of amp 
            self.cte_dir = -1    
        if ( self.extn == 'sci,2') : 
            self.cte_dir = 1   

class HRCInputImage (ACSInputImage):

    def __init__(self, input, dqname, platescale, memmap=0):
        ACSInputImage.__init__(self, input, dqname, platescale,memmap=0)
        self.instrument = 'ACS/HRC'        
        self.full_shape = (1024,1024)
        self.platescale = platescale

        if ( self.amp == 'A' or self.amp == 'B' ) : # cte direction depends on amp (but is independent of chip)
             self.cte_dir = 1   
        if ( self.amp == 'C' or self.amp == 'D' ) :
             self.cte_dir = -1   

class SBCInputImage (ACSInputImage):

    def __init__(self, input, dqname, platescale, memmap=0):
        ACSInputImage.__init__(self,input,dqname,platescale,memmap=0)
        self.full_shape = (1024,1024)
        self.platescale = platescale
        self.instrument = 'ACS/SBC'

        # no cte correction for SBC so set cte_dir=0.
        print('\nWARNING: No cte correction will be made for this SBC data.\n')
        self.cte_dir = 0       

    def setInstrumentParameters(self, instrpars, pri_header):
        """ Sets the instrument parameters.
        """
        if self._isNotValid (instrpars['gain'], instrpars['gnkeyword']):
            instrpars['gnkeyword'] = None
        if self._isNotValid (instrpars['rdnoise'], instrpars['rnkeyword']):
            instrpars['rnkeyword'] = None
        if self._isNotValid (instrpars['exptime'], instrpars['expkeyword']):
            instrpars['expkeyword'] = 'EXPTIME'
        if instrpars['crbit'] == None:
            instrpars['crbit'] = self.cr_bits_value
      
        self._exptime   = self.getInstrParameter(instrpars['exptime'], pri_header,
                                                 instrpars['expkeyword'])
        self._crbit     = instrpars['crbit']

        if self._exptime == None:
            print 'ERROR: invalid instrument task parameter'
            raise ValueError

        # We need to treat Read Noise and Gain as a special case since it is 
        # not populated in the SBC primary header for the MAMA
        if (instrpars['rnkeyword'] != None):
            self._rdnoise   = self.getInstrParameter(instrpars['rdnoise'], pri_header,
                                                     instrpars['rnkeyword'])                                                 
        else:
            self._rdnoise = None
        if (instrpars['gnkeyword'] != None):
            self._gain = self.getInstrParameter(instrpars['gain'], pri_header,
                                                     instrpars['gnkeyword'])
        else:
            self._gain = None
 

        if self._exptime == None:
            print 'ERROR: invalid instrument task parameter'
            raise ValueError

        # We need to determine if the user has used the default readnoise/gain value
        # since if not, they will need to supply a gain/readnoise value as well                
        usingDefaultGain = False
        usingDefaultReadnoise = False
        if (instrpars['gnkeyword'] == None):
            usingDefaultGain = True
        if (instrpars['rnkeyword'] == None):
            usingDefaultReadnoise = True

        # Set the default readnoise or gain values based upon the amount of user input given.
        
        # Case 1: User supplied no gain or readnoise information
        if usingDefaultReadnoise and usingDefaultGain:
            # Set the default gain and readnoise values
            self._setSBCchippars()
        # Case 2: The user has supplied a value for gain
        elif usingDefaultReadnoise and not usingDefaultGain:
            # Set the default readnoise value
            self._setDefaultSBCReadnoise()
        # Case 3: The user has supplied a value for readnoise 
        elif not usingDefaultReadnoise and usingDefaultGain:
            # Set the default gain value
            self._setDefaultSBCGain()
        else:
            # In this case, the user has specified both a gain and readnoise values.  Just use them as is.
            pass

    def _setSBCchippars(self):
        self._setDefaultSBCGain()
        self._setDefaultSBCReadnoise()
     
    def _setDefaultSBCGain(self):
        self._gain = 1

    def _setDefaultSBCReadnoise(self):
        self._rdnoise = 0
