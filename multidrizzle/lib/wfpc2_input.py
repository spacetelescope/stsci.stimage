#
#   Authors: Warren Hack, Ivo Busko, Christopher Hanley
#   Program: wfpc2_input.py
#   Purpose: Class used to model WFPC2 specific instrument data.
#   History:
#           Version 0.0.0, ----------- Created -- CJH
#           Version 0.0.1, Updated key names in dictionaries -- CJH -- 07/08/04

__version__ = '0.0.1'

import pydrizzle
from pydrizzle import fileutil
from input_image import InputImage
import numarray as N

class WFPC2InputImage (InputImage):

    SEPARATOR = '_'

    def __init__(self, input,dqname,memmap=1):
        InputImage.__init__(self,input,dqname,memmap=1)
        # define the cosmic ray bits value to use in the dq array
        self.cr_bits_value = 4096
        
        # Effective gain to be used in the driz_cr step.  Since the
        # images are arlready to have been convered to electrons
        self._effGain = 1

        # Attribute defining the pixel dimensions of WFPC2 chips.
        self.full_shape = (800,800)

    def setInstrumentParameters(self, instrpars, pri_header):
        """ This method overrides the superclass to set default values into
            the parameter dictionary, in case empty entries are provided.
        """
        if self._isNotValid (instrpars['gain'], instrpars['gnkeyword']):
            instrpars['gnkeyword'] = 'ATODGAIN'

#       We will not be reading the read noise in from the header.  It is 
#       necessary to hard code those values for each WFPC2 chip.
            
        if self._isNotValid (instrpars['exptime'], instrpars['expkeyword']):
            instrpars['expkeyword'] = 'EXPTIME'

        if instrpars['crbit'] == None or instrpars['crbit'] == 0:
            instrpars['crbit'] = self.cr_bits_value

        self._headergain      = self.getInstrParameter(instrpars['gain'], pri_header,
                                                 instrpars['gnkeyword'])
        self._exptime   = self.getInstrParameter(instrpars['exptime'], pri_header,
                                                 instrpars['expkeyword'])
        self._crbit     = instrpars['crbit']

        if self._headergain == None or self._exptime == None:
            print 'ERROR: invalid instrument task parameter'
            raise ValueError
        self._setchippars()
        
    def _setchippars(self):
        pass

    def getComputedSky(self):
        return self._computedsky

    def setComputedSky(self,newValue):
        self._computedsky = newValue
        
    def getSubtractedSky(self):
        return self._subtractedsky
        
    def setSubtractedSky(self,newValue):
        self._subtractedsky = newValue
        
    def getEffGain(self):
        return self._effGain

    def getGain(self):
        return self._gain

    def _getCalibratedGain(self):
        return self._gain

    def getReadNoise(self):
        return self._rdnoise

    def getExpTime(self):
        return self._exptime

    def getCRbit(self):
        return self._crbit

    def getRootname(self,name):
        _indx = name.rfind(self.sep)
        if _indx < 0: _indx = len(name)
        return name[:_indx]

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
                        self.getGain() / self.getSubtractedSky()
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
        print "Converting %s from DN to ELECTRONS"%(self.name)
        N.multiply(__sciext.data,self.getGain(),__sciext.data)        

        __handle.close()
        del __handle
        del __sciext


class WF2InputImage (WFPC2InputImage):

    def __init__(self, input, dqname, memmap=1):
        WFPC2InputImage.__init__(self,input,dqname,memmap=1)
        self.instrument = 'WFPC2/WF2'

    def _setchippars(self):
        if self._headergain == 7:
            self._gain    = 7.12
            self._rdnoise = 5.51  
        elif self._headergain == 15:
            self._gain    = 14.50
            self._rdnoise = 7.84
        else:
            raise ValueError, "! Header gain value is not valid for WFPC2"

class WF3InputImage (WFPC2InputImage):

    def __init__(self, input, dqname, memmap=1):
        WFPC2InputImage.__init__(self, input, dqname, memmap=1)
        self.instrument = 'WFPC2/WF3'

    def _setchippars(self):
        if self._headergain == 7:
            self._gain    = 6.90
            self._rdnoise = 5.22  
        elif self._headergain == 15:
            self._gain    = 13.95
            self._rdnoise = 6.99
        else:
            raise ValueError, "! Header gain value is not valid for WFPC2"

class WF4InputImage (WFPC2InputImage):

    def __init__(self, input, dqname, memmap=1):
        WFPC2InputImage.__init__(self, input, dqname, memmap=1)
        self.instrument = 'WFPC2/WF4'

    def _setchippars(self):
        if self._headergain == 7:
            self._gain    = 7.10
            self._rdnoise = 5.19  
        elif self._headergain == 15:
            self._gain    = 13.95
            self._rdnoise = 8.32
        else:
            raise ValueError, "! Header gain value is not valid for WFPC2"

class PCInputImage (WFPC2InputImage):

    def __init__(self, input, dqname, memmap=1):
        WFPC2InputImage.__init__(self,input,dqname,memmap=1)
        self.instrument = 'WFPC2/PC'

    def _setchippars(self):
        if self._headergain == 7:
            self._gain    = 7.12
            self._rdnoise = 5.24  
        elif self._headergain == 15:
            self._gain    = 13.99
            self._rdnoise = 7.02
        else:
            raise ValueError, "! Header gain value is not valid for WFPC2"
