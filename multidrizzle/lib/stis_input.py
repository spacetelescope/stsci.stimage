#
#   Authors: Christopher Hanley
#   Program: stis_input.py
#   Purpose: Class used to model STIS specific instrument data.
#   History:
#           Version 0.1.0, ----------- Created
#           Version 0.1.1 07/29/04 -- Plate scale is now defined by the Pydrizzle
#               exposure class by use of a plate scale variable passed in through
#               the constructor.
#           Version 0.2.0 08/30/04 -- Added support for NUV and FUV MAMAs.
#           Version 0.2.1 09/14/-4 -- Modified the setInstrumentParameters methods
#                to allow for user input.  -- CJH
#           Version 0.2.2 09/15/04 -- Modified the setInstrumentParameters to treat
#               a user cr bit input value of zero as a None.  This allows the
#               user to turn off the DQ array update during the Driz_CR step. -- CJH

__version__ = '0.2.2'

import pydrizzle
from pydrizzle import fileutil
import numarray as N

from input_image import InputImage


class STISInputImage (InputImage):

    SEPARATOR = '_'

    def __init__(self, input,dqname,platescale,memmap=1):
        InputImage.__init__(self,input,dqname,platescale,memmap=1)
        
        # define the cosmic ray bits value to use in the dq array
        self.cr_bits_value = 8192
        self.platescale = platescale
        
        # Effective gain to be used in the driz_cr step.  Since the
        # STIS images have already benn converted to electons
        # the effective gain is 1.
        self._effGain = 1
        
        
    def setInstrumentParameters(self, instrpars, pri_header):
        """ This method overrides the superclass to set default values into
            the parameter dictionary, in case empty entries are provided.
        """
        if self._isNotValid (instrpars['gain'], instrpars['gnkeyword']):
            instrpars['gnkeyword'] = 'ATODGAIN'
        if self._isNotValid (instrpars['rdnoise'], instrpars['rnkeyword']):
            instrpars['rnkeyword'] = 'READNSE'
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


    def updateMDRIZSKY(self,filename=None):
    
        if (filename == None):
            filename = self.name
            
        try:
            _handle = fileutil.openImage(filename,mode='update',memmap=0)
        except:
            raise IOError, "Unable to open %s for sky level computation"%filename
        try:
            try:
                # Assume MDRIZSKY lives in primary header
                print "Updating MDRIZSKY in %s with %f / %f = %f"%(filename,self.getSubtractedSky(),
                        self.getGain(),
                        self.getSubtractedSky() / self.getGain()
                        )
                _handle[0].header['MDRIZSKY'] = self.getSubtractedSky() / self.getGain()
            except:
                print "Cannot find keyword MDRIZSKY in %s to update"%filename
                print "Adding MDRIZSKY keyword to primary header with value %f"%self.getSubtractedSky()
                _handle[0].header.update('MDRIZSKY',self.getSubtractedSky()/self.getGain(), 
                    comment="Sky value subtracted by Multidrizzle")
        finally:
            _handle.close()

    def doUnitConversions(self):
        self._convert2electrons()

    def _convert2electrons(self):
        # Image information
        __handle = fileutil.openImage(self.name,mode='update',memmap=0)
        __sciext = fileutil.getExtn(__handle,extn=self.extn)

        # Multiply the values of the sci extension pixels by the gain.
        print "Converting %s from COUNTS to ELECTRONS"%(self.name)
        N.multiply(__sciext.data,self.getGain(),__sciext.data)        

        __handle.close()
        del __handle
        del __sciext

    def _setMAMAchippars(self):
        self._setDefaultGain()
        self._setDefaultReadnoise()
     
    def _setMAMADefaultGain(self):
        self._gain = 1

    def _setMAMADefaultReadnoise(self):
        self._rdnoise = 0

class CCDInputImage(STISInputImage):

    def __init__(self, input, dqname, platescale, memmap=1):
        STISInputImage.__init__(self,input,dqname,platescale,memmap=1)
        self.instrument = 'STIS/CCD'
        self.full_shape = (1024,1024)
        self.platescale = platescale

class NUVInputImage(STISInputImage):
    def __init__(self, input, dqname, platescale, memmap=1):
        STISInputImage.__init__(self,input,dqname,platescale,memmap=1)
        self.instrument = 'STIS/NUV-MAMA'
        self.full_shape = (1024,1024)
        self.platescale = platescale

    def setInstrumentParameters(self, instrpars, pri_header):
        """ This method overrides the superclass to set default values into
            the parameter dictionary, in case empty entries are provided.
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
        # not populated in the STIS primary header for the MAMAs
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
            self._setMAMAchippars()
        # Case 2: The user has supplied a value for gain
        elif usingDefaultReadnoise and not usingDefaultGain:
            # Set the default readnoise value
            self._setMAMADefaultReadnoise()
        # Case 3: The user has supplied a value for readnoise 
        elif not usingDefaultReadnoise and usingDefaultGain:
            # Set the default gain value
            self._setMAMADefaultGain()
        else:
            # In this case, the user has specified both a gain and readnoise values.  Just use them as is.
            pass

class FUVInputImage(STISInputImage):
    def __init__(self, input, dqname, platescale, memmap=1):
        STISInputImage.__init__(self,input,dqname,platescale,memmap=1)
        self.instrument = 'STIS/FUV-MAMA'
        self.full_shape = (1024,1024)
        self.platescale = platescale

    def setInstrumentParameters(self, instrpars, pri_header):
        """ This method overrides the superclass to set default values into
            the parameter dictionary, in case empty entries are provided.
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
        # not populated in the STIS primary header for the MAMAs
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
            self._setMAMAchippars()
        # Case 2: The user has supplied a value for gain
        elif usingDefaultReadnoise and not usingDefaultGain:
            # Set the default readnoise value
            self._setMAMADefaultReadnoise()
        # Case 3: The user has supplied a value for readnoise 
        elif not usingDefaultReadnoise and usingDefaultGain:
            # Set the default gain value
            self._setMAMADefaultGain()
        else:
            # In this case, the user has specified both a gain and readnoise values.  Just use them as is.
            pass
