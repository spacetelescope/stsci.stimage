#
#   Authors: Warren Hack, Ivo Busko, Christopher Hanley
#   Program: wfpc2_input.py
#   Purpose: Class used to model WFPC2 specific instrument data.
#   History:
#           Version 0.0.0, ----------- Created -- CJH
#           Version 0.0.1, Updated key names in dictionaries -- CJH -- 07/08/04
#           Version 0.1.0, Have modified the sky value get and set functions to
#               take into account the plate scale differences between the WFPC2
#               chips when comparing the sky values in deciding which minimum to
#               use.  The MDRIZSKY keyword will be populated with a value based
#               upon the WF3 chip plate scale.
#           Version 0.1.1 07/29/04 -- Plate scale is now defined by the Pydrizzle
#               exposure class by use of a plate scale variable passed in through
#               the constructor.
#           Version 0.1.2 09/09/04 -- The setInstrumentParameters was modifed so
#               that user input of gain and readnoise parameters would not be
#               ignored. -- CJH
               
__version__ = '0.1.2'

import pydrizzle
from pydrizzle import fileutil
from input_image import InputImage
import numarray as N

class WFPC2InputImage (InputImage):

    SEPARATOR = '_'

    def __init__(self, input,dqname,platescale,memmap=1):
        InputImage.__init__(self,input,dqname,platescale,memmap=1)
        # define the cosmic ray bits value to use in the dq array
        self.cr_bits_value = 4096
        
        self.platescale = platescale
        
        # Effective gain to be used in the driz_cr step.  Since the
        # images are arlready to have been convered to electrons
        self._effGain = 1

        # Attribute defining the pixel dimensions of WFPC2 chips.
        self.full_shape = (800,800)
        
        # Reference Plate Scale used for updates to MDRIZSKY
        self.refplatescale = 0.0996 # arcsec / pixel

    def setInstrumentParameters(self, instrpars, pri_header):
        """ This method overrides the superclass to set default values into
            the parameter dictionary, in case empty entries are provided.
        """
                
        if self._isNotValid (instrpars['gain'], instrpars['gnkeyword']):
            instrpars['gnkeyword'] = 'ATODGAIN'
    
        if self._isNotValid (instrpars['rdnoise'], instrpars['rnkeyword']):
            instrpars['rnkeyword'] = None
            
        if self._isNotValid (instrpars['exptime'], instrpars['expkeyword']):
            instrpars['expkeyword'] = 'EXPTIME'

        if instrpars['crbit'] == None or instrpars['crbit'] == 0:
            instrpars['crbit'] = self.cr_bits_value

        self._headergain      = self.getInstrParameter(instrpars['gain'], pri_header,
                                                 instrpars['gnkeyword'])    
        self._exptime   = self.getInstrParameter(instrpars['exptime'], pri_header,
                                                 instrpars['expkeyword'])
        self._crbit     = instrpars['crbit']
        
        # We need to treat Read Noise as a special case since it is 
        # not populated in the WFPC2 primary header
        if (instrpars['rnkeyword'] != None):
            self._rdnoise   = self.getInstrParameter(instrpars['rdnoise'], pri_header,
                                                     instrpars['rnkeyword'])                                                 
        else:
            self._rdnoise = None

        if self._headergain == None or self._exptime == None:
            print 'ERROR: invalid instrument task parameter'
            raise ValueError

        # We need to determine if the user has used the default readnoise/gain value
        # since if not, they will need to supply a gain/readnoise value as well        
        
        usingDefaultGain = False
        usingDefaultReadnoise = False
        if (instrpars['gnkeyword'] == 'ATODGAIN'):
            usingDefaultGain = True
        if (instrpars['rnkeyword'] == None):
            usingDefaultReadnoise = True

            
        # If the user has specified either the readnoise or the gain, we need to make sure
        # that they have actually specified both values.  In the default case, the readnoise
        # of the system depends on what the gain
        
        print "Default Readnoise = ",usingDefaultReadnoise, " Default Gain = ",usingDefaultGain              
        if usingDefaultReadnoise and usingDefaultGain:
            self._setchippars()
        elif usingDefaultReadnoise and not usingDefaultGain:
            raise ValueError, "ERROR: You need to supply readnoise information\n when not using the default gain for WFPC2."
        elif not usingDefaultReadnoise and usingDefaultGain:
            raise ValueError, "ERROR: You need to supply gain information when\n not using the default readnoise for WFPC2." 
        else:
            # In this case, the user has specified both a gain and readnoise values.  Just use them as is.
            self._gain = self._headergain
            print "Using user defined values for gain and readnoise"
                    
    def _setchippars(self):
        pass

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

    def getComputedSky(self):
        return (self._computedsky * (self.refplatescale / self.platescale)**2 )

    def setComputedSky(self,newValue):
        self._computedsky = newValue
        
    def getSubtractedSky(self):
        return self._subtractedsky
        
    def setSubtractedSky(self,newValue):
        self._subtractedsky = (newValue / (self.refplatescale /  self.platescale)**2)
            
    def subtractSky(self):
        try:
            try:
                _handle = fileutil.openImage(self.name,mode='update',memmap=0)
                _sciext = fileutil.getExtn(_handle,extn=self.extn)
                print "%s (computed sky,subtracted sky)  : (%f,%f)"%(self.name,self.getComputedSky(),self.getSubtractedSky()*(self.refplatescale / self.platescale)**2)
                N.subtract(_sciext.data,self.getSubtractedSky(),_sciext.data)
            except:
                raise IOError, "Unable to open %s for sky subtraction"%self.name
        finally:
            _handle.close()
            del _sciext,_handle

    def updateMDRIZSKY(self,filename=None):
    
        if (filename == None):
            filename = self.name
            
        try:
            _handle = fileutil.openImage(filename,mode='update',memmap=0)
        except:
            raise IOError, "Unable to open %s for sky level computation"%filename

         # Compute the sky level subtracted from all the WFPC2 detectors based upon the reference plate scale.
        skyvalue = (self.getSubtractedSky()  * (self.refplatescale/self.platescale)**2) / self.getGain()
        
        try:
            try:
                # Assume MDRIZSKY lives in primary header
                print "Updating MDRIZSKY in %s with %f"%(filename,skyvalue)
                _handle[0].header['MDRIZSKY'] = skyvalue
            except:
                print "Cannot find keyword MDRIZSKY in %s to update"%filename
                print "Adding MDRIZSKY keyword to primary header with value %f"%skyvalue
                _handle[0].header.update('MDRIZSKY',skyvalue, comment="Sky value subtracted by Multidrizzle")
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

    def __init__(self, input, dqname, platescale, memmap=1):
        WFPC2InputImage.__init__(self,input,dqname,platescale, memmap=1)
        self.instrument = 'WFPC2/WF2'
        self.platescale = platescale #0.0996 #arcsec / pixel
        
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

    def __init__(self, input, dqname, platescale, memmap=1):
        WFPC2InputImage.__init__(self, input, dqname, platescale, memmap=1)
        self.instrument = 'WFPC2/WF3'
        self.platescale = platescale #0.0996 #arcsec / pixel

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

    def __init__(self, input, dqname, platescale, memmap=1):
        WFPC2InputImage.__init__(self, input, dqname, platescale, memmap=1)
        self.instrument = 'WFPC2/WF4'
        self.platescale = platescale #0.0996 #arcsec / pixel

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

    def __init__(self, input, dqname, platescale, memmap=1):
        WFPC2InputImage.__init__(self,input,dqname,platescale,memmap=1)
        self.instrument = 'WFPC2/PC'
        self.platescale = platescale #0.0455 #arcsec / pixel

    def _setchippars(self):
        if self._headergain == 7:
            self._gain    = 7.12
            self._rdnoise = 5.24  
        elif self._headergain == 15:
            self._gain    = 13.99
            self._rdnoise = 7.02
        else:
            raise ValueError, "! Header gain value is not valid for WFPC2"
