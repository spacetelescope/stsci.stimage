#
#   Authors: Christopher Hanley, David Grumm
#   Program: nicmos_input.py
#   Purpose: Class used to model NICMOS specific instrument data.

from pytools import fileutil
from nictools import readTDD
import numpy as N

from ir_input import IRInputImage
from input_image import InputImage


class NICMOSInputImage(IRInputImage):

    SEPARATOR = '_'

    def __init__(self, input,dqname,platescale,memmap=0):
        IRInputImage.__init__(self,input,dqname,platescale,memmap=0)
        
        # define the cosmic ray bits value to use in the dq array
        self.cr_bits_value = 4096
        self.platescale = platescale
         
        # no cte correction for NICMOS so set cte_dir=0.
        print('\nWARNING: No cte correction will be made for this NICMOS data.\n')
        self.cte_dir = 0   
        self._effGain = 1
        
    def setInstrumentParameters(self, instrpars, pri_header):
        """ This method overrides the superclass to set default values into
            the parameter dictionary, in case empty entries are provided.
        """
        if self._isNotValid (instrpars['gain'], instrpars['gnkeyword']):
            instrpars['gnkeyword'] = 'ADCGAIN'
        if self._isNotValid (instrpars['rdnoise'], instrpars['rnkeyword']):
            instrpars['rnkeyword'] = None
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

        if self._gain == None or self._exptime == None:
            print 'ERROR: invalid instrument task parameter'
            raise ValueError

        # We need to treat Read Noise as a special case since it is 
        # not populated in the NICMOS primary header
        if (instrpars['rnkeyword'] != None):
            self._rdnoise   = self.getInstrParameter(instrpars['rdnoise'], pri_header,
                                                     instrpars['rnkeyword'])                                                 
        else:
            self._rdnoise = None


        # We need to determine if the user has used the default readnoise/gain value
        # since if not, they will need to supply a gain/readnoise value as well        
        
        usingDefaultReadnoise = False
        if (instrpars['rnkeyword'] == None):
            usingDefaultReadnoise = True

            
        # Set the default readnoise values based upon the amount of user input given.
        
        # User supplied no readnoise information
        if usingDefaultReadnoise:
            # Set the default gain and readnoise values
            self._setchippars()

    def _setchippars(self):
        self._setDefaultReadnoise()
                
    def getflat(self):
        """

        Purpose
        =======
        Method for retrieving a detector's flat field.
        
        This method will return an array the same shape as the
        image.

        :units: cps

        """

        # The keyword for NICMOS flat fields in the primary header of the flt
        # file is FLATFILE.  This flat file is not already in the required 
        # units of electrons.
        
        filename = self.header['FLATFILE']
        
        try:
            handle = fileutil.openImage(filename,mode='readonly',memmap=0)
            hdu = fileutil.getExtn(handle,extn=self.grp)
            data = hdu.data[self.ltv2:self.size2,self.ltv1:self.size1]
        except:
            try:
                handle = fileutil.openImage(filename[5:],mode='readonly',memmap=0)
                hdu = fileutil.getExtn(handle,extn=self.grp)
                data = hdu.data[self.ltv2:self.size2,self.ltv1:self.size1]
            except:
                data = N.ones(self.image_shape,dtype=self.image_dtype)
                str = "Cannot find file "+filename+".  Treating flatfield constant value of '1'.\n"
                print str

        flat = (1.0/data) # The flat field is normalized to unity.

        return flat
        

    def getdarkcurrent(self):
        """
        
        Purpose
        =======
        Return the dark current for the NICMOS detectors.
        
        :units: cps
        
        """
                
        try:
            darkcurrent = self.header['exptime'] * (self.darkrate/self.getGain())
            
        except:
            str =  "#############################################\n"
            str += "#                                           #\n"
            str += "# Error:                                    #\n"
            str += "#   Cannot find the value for 'EXPTIME'     #\n"
            str += "#   in the image header.  NICMOS input      #\n"
            str += "#   images are expected to have this header #\n"
            str += "#   keyword.                                #\n"
            str += "#                                           #\n"
            str += "#Error occured in the NICMOSInputImage class#\n"
            str += "#                                           #\n"
            str += "#############################################\n"
            raise ValueError, str
        
        
        return darkcurrent
        
    def getdarkimg(self):
        """
        
        Purpose
        =======
        Return an array representing the dark image for the detector.
        
        :units: cps
        
        """

        # Read the temperature dependeant dark file.  The name for the file is taken from
        # the TEMPFILE keyword in the primary header.
        tddobj = readTDD.fromcalfile(self.name)

        if tddobj == None:
            return N.ones(self.image_shape,dtype=self.image_dtype)*self.getdarkcurrent()
        else:
            # Create Dark Object from AMPGLOW and Lineark Dark components
            darkobj = tddobj.getampglow() + tddobj.getlindark()
                        
            # Return the darkimage taking into account an subarray information available
            return darkobj[self.ltv2:self.size2,self.ltv1:self.size1]
        
    
class NIC1InputImage(NICMOSInputImage):

    def __init__(self, input, dqname, platescale, memmap=0):
        NICMOSInputImage.__init__(self,input,dqname,platescale,memmap=0)
        self.instrument = 'NICMOS/1'
        self.full_shape = (256,256)
        self.platescale = platescale
        self.darkrate = 0.08 #electrons/s

    def _setDefaultReadnoise(self):
        self._rdnoise = 26.0/self.getGain() #ADU

class NIC2InputImage(NICMOSInputImage):
    def __init__(self, input, dqname, platescale, memmap=0):
        NICMOSInputImage.__init__(self,input,dqname,platescale,memmap=0)
        self.instrument = 'NICMOS/2'
        self.full_shape = (256,256)
        self.platescale = platescale
        self.darkrate = 0.08 #electrons/s

    def _setDefaultReadnoise(self):
        self._rdnoise = 26.0/self.getGain() #ADU

class NIC3InputImage(NICMOSInputImage):
    def __init__(self, input, dqname, platescale, memmap=0):
        NICMOSInputImage.__init__(self,input,dqname,platescale,memmap=0)
        self.instrument = 'NICMOS/3'
        self.full_shape = (256,256)
        self.platescale = platescale
        self.darkrate = 0.15 #electrons/s

    def _setDefaultReadnoise(self):
        self._rdnoise = 29.0/self.getGain() #ADU
