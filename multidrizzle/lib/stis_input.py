#
#   Authors: Christopher Hanley
#   Program: stis_input.py
#   Purpose: Class used to model STIS specific instrument data.
#   History:
#           Version 0.1.0, ----------- Created
__version__ = '0.1.0'

import pydrizzle
from pydrizzle import fileutil
import numarray as N

from input_image import InputImage


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

class CCDInputImage (STISInputImage):

    def __init__(self, input, dqname, memmap=1):
        STISInputImage.__init__(self,input,dqname,memmap=1)
        self.instrument = 'STIS/CCD'
        self.full_shape = (1024,1024)


