#
#   Authors: Warren Hack, Ivo Busko, Christopher Hanley
#   Program: wfpc2_input.py
#   Purpose: Class used to model WFPC2 specific instrument data.
#   History:
#           Version 0.0.0, ----------- Created -- CJH
__version__ = '0.0.0'

import pydrizzle
from pydrizzle import fileutil
from input_image import InputImage


class WFPC2InputImage (InputImage):

    SEPARATOR = '_'

    def __init__(self, input,dqname,memmap=1):
        InputImage.__init__(self,input,dqname,memmap=1)
        # define the cosmic ray bits value to use in the dq array
        self.cr_bits_value = 4096
        
        # Effective gain to be used in the driz_cr step.  Since the
        # images are arlready to have been convered to d 
        self._effGain = 1

        # Attribute defining the pixel dimensions of WFPC2 chips.
        self.full_shape = (800,800)


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

    def setInstrumentParameters(self, instrpars, pri_header):
        """ This method overrides the superclass to set default values into
            the parameter dictionary, in case empty entries are provided.
        """
        if self._isNotValid (instrpars['gain'], instrpars['gainkw']):
            instrpars['gainkw'] = 'ATODGAIN'
            self._getCalibratedGain()

#       We will no be reading the read noise in from the header.  It is 
#       necessary to hard code those values for each WFPC2 chip.
        if self._isNotValid (instrpars['rdnoise'], instrpars['rdnoisekw']):
            instrpars['rdnoisekw'] = None
            
        if self._isNotValid (instrpars['exptime'], instrpars['exptimekw']):
            instrpars['exptimekw'] = 'EXPTIME'
        if instrpars['crbit'] == None or instrpars['crbit'] == 0:
            instrpars['crbit'] = self.cr_bits_value

        self._gain      = self.getInstrParameter(instrpars['gain'], pri_header,
                                                 instrpars['gainkw'])
        self._rdnoise   = self.getInstrParameter(instrpars['rdnoise'], pri_header,
                                                 instrpars['rdnoisekw'])
        self._exptime   = self.getInstrParameter(instrpars['exptime'], pri_header,
                                                 instrpars['exptimekw'])
        self._crbit     = instrpars['crbit']

        if self._gain == None or self._rdnoise == None or self._exptime == None:
            print 'ERROR: invalid instrument task parameter'
            raise ValueError


class WF2InputImage (WFPC2InputImage):

    def __init__(self, input, dqname, memmap=1):
        WFPC2InputImage.__init__(self,input,dqname,memmap=1)
        self.instrument = 'WFPC2/WF2'


class WF3InputImage (WFPC2InputImage):

    def __init__(self, input, dqname, memmap=1):
        WFPC2InputImage.__init__(self, input, dqname, memmap=1)
        self.instrument = 'WFPC2/WF3'

class WF4InputImage (WFPC2InputImage):

    def __init__(self, input, dqname, memmap=1):
        WFPC2InputImage.__init__(self, input, dqname, memmap=1)
        self.instrument = 'WFPC2/WF4'

class PCInputImage (WFPC2InputImage):

    def __init__(self, input, dqname, memmap=1):
        WFPC2InputImage.__init__(self,input,dqname,memmap=1)
        self.instrument = 'WFPC2/PC'
