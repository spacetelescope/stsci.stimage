#
#   Authors: Christopher Hanley
#   Program: wfc3_input.py
#   Purpose: Class used to model WFC3 specific instrument data.

from pytools import fileutil
import numpy as n
from acs_input import ACSInputImage
from nicmos_input import NICMOSInputImage

class UVISInputImage(ACSInputImage):

    def __init__(self, input, dqname, platescale, memmap=0):
        ACSInputImage.__init__(self,input,dqname,platescale,memmap=0)
        self.instrument = 'WFC3/UVIS'
        self.full_shape = (4096,2048)
        self.platescale = platescale

        if ( self.extn == 'sci,1') : # get cte direction, which depends on which chip but is independent of amp 
            self.cte_dir = -1    
        if ( self.extn == 'sci,2') : 
            self.cte_dir = 1   

class IRInputImage(NICMOSInputImage):

    def __init__(self, input, dqname, platescale, memmap=0):
        NICMOSInputImage.__init__(self,input,dqname,platescale,memmap=0)
        self.instrument = 'WFC3/IR'
        self.full_shape = (1000,1000)
        self.platescale = platescale
        self.darkrate = 0.01 #electrons/second

    def _setDefaultReadnoise(self):
        self._rdnoise = 22.0 # electrons

    def _convert2electrons(self):
        # Image information
        __handle = fileutil.openImage(self.name,mode='update',memmap=0)
        __sciext = fileutil.getExtn(__handle,extn=self.extn)        

        N.multiply(__sciext.data,self.getExpTime(),__sciext.data)        
        
        __handle.close()
        del __handle
        del __sciext

    def updateMDRIZSKY(self,filename=None):
    
        if (filename == None):
            filename = self.name
            
        try:
            _handle = fileutil.openImage(filename,mode='update',memmap=0)
        except:
            raise IOError, "Unable to open %s for sky level computation"%filename
        try:
        
            # Get the exposure time for the image.  If the exposure time of the image
            # is 0, set the MDRIZSKY value to 0.  Otherwise update the MDRIZSKY value
            # in units of electrons per second.
            if (self.getExpTime() == 0.0):
                str =  "*******************************************\n"
                str += "*                                         *\n"
                str += "* ERROR: Image EXPTIME = 0.               *\n"
                str += "* MDRIZSKY header value cannot be         *\n"
                str += "* converted to units of 'electrons/s'        *\n"
                str += "* MDRIZSKY will be set to a value of '0'  *\n"
                str += "*                                         *\n"
                str =  "*******************************************\n"
                _handle[0].header['MDRIZSKY'] = 0
                print str
            else:
                # Assume the MDRIZSKY keyword is in the primary header.  Try to update
                # the header value
                try:
                    # We need to handle the updates of the header for data in 
                    # original units of either counts per second or counts
                    if (_handle[0].header['UNITCORR'].strip() == 'PERFORM'):
                        print "Updating MDRIZSKY in %s with %f / %f = %f"%(filename,
                                self.getSubtractedSky(),
                                self.getExpTime(),
                                self.getSubtractedSky() / self.getExpTime()
                                )
                        _handle[0].header['MDRIZSKY'] = self.getSubtractedSky() / self.getExpTime()
                    else:
                        print "Updating MDRIZSKY in %s with %f"%(filename,
                                self.getSubtractedSky()
                                )
                        _handle[0].header['MDRIZSKY'] = self.getSubtractedSky()
                        
                # The MDRIZSKY keyword was not found in the primary header.  Add the
                # keyword to the header and populate it with the subtracted sky value.
                except:
                    print "Cannot find keyword MDRIZSKY in %s to update"%filename
                    if (_handle[0].header['UNITCORR'].strip() == 'PERFORM'):
                        print "Adding MDRIZSKY keyword to primary header with value %f"%(self.getSubtractedSky()/self.getExpTime())
                        _handle[0].header.update('MDRIZSKY',self.getSubtractedSky()/self.getExpTime(), 
                            comment="Sky value subtracted by Multidrizzle")
                    else:
                        print "Adding MDRIZSKY keyword to primary header with value %f"%(self.getSubtractedSky())
                        _handle[0].header.update('MDRIZSKY',self.getSubtractedSky(), 
                            comment="Sky value subtracted by Multidrizzle")
                    
        finally:
            _handle.close()
