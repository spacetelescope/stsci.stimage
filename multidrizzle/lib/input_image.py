#
#   Authors: Warren Hack, Ivo Busko, Christopher Hanley
#   Program: input_image.py
#   Purpose: Super class used to model astronomical data from observatory instruments.

from __future__ import division # confidence high

import pyfits
from pytools import fileutil

import imagestats
from imagestats import ImageStats

from pytools import imageiter
from pytools.imageiter import ImageIter

import numpy as np
import quickDeriv
import driz_cr

DEFAULT_SEPARATOR = '_'

class InputImage(object):
    '''The InputImage class is the base class for all of the various
       types of images
    '''

    def __init__(self, input,dqname,platescale,memmap=0,proc_unit="native"):
        # These will always be populated by the appropriate
        # sub-class, however, this insures that these attributes
        # are not overlooked/forgotten.
        self.name = input
        self.memmap = memmap
        if not self.SEPARATOR:
            self.sep = DEFAULT_SEPARATOR
        else:
            self.sep = self.SEPARATOR

        self.instrument = None
        _fname,_extn = fileutil.parseFilename(input)
        self.dqfile_fullname = dqname
        self.dqfile_name,self.dqfile_extn = fileutil.parseFilename(dqname)
        self.extn     = _extn
        self.grp      = fileutil.parseExtn(_extn)[1]
        self.rootname = self.getRootname(_fname)
        self.datafile = _fname
        self.cr_bits_value = None
        self._effGain = None
        self.static_badval = 64
        self.static_mask = None
        self.cte_dir = 1 
                
        # read amplifier to be used for HRC or STIS/CCD.
        try:  
            self.amp = fileutil.getKeyword(input,'CCDAMP')
        # set default if keyword missing
        except KeyError:
            # STIS default should be 'D' but 'C' and 'D' have the same readout direction so it's okay
            self.amp = 'C'  
        
        # Define the platescale and reference plate scale for the detector.
        self.platescale = platescale
        self.refplatescale = platescale # Default is to make each chip it's own reference value
        
        # Image information
        handle = fileutil.openImage(self.name,mode='readonly',memmap=self.memmap)
        sciext = fileutil.getExtn(handle,extn=self.extn)
        self.image_shape = sciext.data.shape
        self.image_type = sciext.data.dtype.name
        self.image_dtype= sciext.data.dtype.name
        
        # Retrieve a combined primary and extension header
        self.header = fileutil.getHeader(input,handle=handle)
        del sciext
        handle.close()
        del handle

        # Initialize sky background information keywords
        self._subtractedsky = 0.
        self._computedsky = None

        # Get image size information for possible subarray use
        try:
            self.ltv1 = self.header['LTV1'] * -1
            self.ltv2 = self.header['LTV2'] * -1
        except KeyError:
            self.ltv1 = 0
            self.ltv2 = 0
        self.size1 = self.header['NAXIS1'] + self.ltv1
        self.size2 = self.header['NAXIS2'] + self.ltv2
        
        # Set Units used for processing.  Options are "native" or "electrons"
        self.proc_unit = proc_unit

    def setInstrumentParameters(self, instrpars, pri_header):
        """ 
        Sets the instrument parameters.
        """
        pass
    
    def doUnitConversions(self):
        """
        Convert the sci extension pixels to electrons
        """
        pass

    def getInstrParameter(self, value, header, keyword):
        """ This method gets a instrument parameter from a
            pair of task parameters: a value, and a header keyword.

            The default behavior is:
              - if the value and header keyword are given, raise an exception.
              - if the value is given, use it.
              - if the value is blank and the header keyword is given, use
                the header keyword.
              - if both are blank, or if the header keyword is not
                found, return None.
        """
        if (value != None and value != '')  and (keyword != None and keyword.strip() != ''):
            exceptionMessage = "ERROR: Your input is ambiguous!  Please specify either a value or a keyword.\n  You specifed both " + str(value) + " and " + str(keyword) 
            raise ValueError, exceptionMessage
        elif value != None and value != '':
            return self._averageFromList(value)
        elif keyword != None and keyword.strip() != '':
            return self._averageFromHeader(header, keyword)
        else:
            return None

    def _averageFromHeader(self, header, keyword):
        """ Averages out values taken from header. The keywords where
            to read values from are passed as a comma-separated list.
        """
        _list = ''
        for _kw in keyword.split(','):
            if header.has_key(_kw):
                _list = _list + ',' + str(header[_kw])
            else:
                return None
        return self._averageFromList(_list)

    def _averageFromList(self, param):
        """ Averages out values passed as a comma-separated
            list, disregarding the zero-valued entries.
        """
        _result = 0.0
        _count = 0

        for _param in param.split(','):
            if _param != '' and float(_param) != 0.0:
                _result = _result + float(_param)
                _count  += 1

        if _count >= 1:
            _result = _result / _count
        return _result

    def getreferencesky(self):
        return (self._subtractedsky * (self.refplatescale / self.platescale)**2)

    def getComputedSky(self):
        return self._computedsky

    def setComputedSky(self,newValue):
        self._computedsky = newValue
        
    def getSubtractedSky(self):
        return self._subtractedsky
        
    def setSubtractedSky(self,newValue):
        self._subtractedsky = newValue
        
    def getGain(self):
        return self._gain

    def getExpTime(self):
            return self._exptime

    def getCRbit(self):
        return self._crbit

    def getRootname(self,name):
        _indx = name.rfind(self.sep)
        if _indx < 0: _indx = len(name)
        return name[:_indx]

    def _isSubArray(self):
        """ Instrument specific method to determine whether input is
            a subarray or not.
        """
        pass

    def _isNotValid(self, par1, par2):
        """ Method used to determine if a value or keyword is supplied as 
            input for instrument specific parameters.
        """
        if (par1 == None or par1 == '') and (par2 == None or par2 == ''):
            return True
        else:
            return False

    def updateStaticMask(self, static_mask):

        """ This method updates a static mask passed as a parameter,
            with mask info derived from the [SCI] array. It also
            keeps a private static mask array appropriate for
            use with the [SCI] array when doing sky processing
            later on.
        """
        # Open input image and get pointer to SCI data
        handle = fileutil.openImage(self.name,mode='readonly',memmap=self.memmap)
        sciext = fileutil.getExtn(handle,extn=self.extn)

        # Add SCI array to static mask
        static_mask.addMember(sciext.data, self.signature())
        self.static_mask = static_mask

        # Close input image filehandle
        handle.close()
        del sciext,handle

    def signature(self):

        """ Generates a signature unique to this image. """

        # Shape is taken from PyFITS object. Subclasses that
        # depend on other types of files must override this.
        return (self.instrument, self.image_shape, self.grp)

    def computeSky(self, skypars):

        """ Compute the sky value based upon the sci array of the chip"""

        # Open input image and get pointer to SCI data
        #
        #
        try:
            _handle = fileutil.openImage(self.name,mode='update',memmap=self.memmap)
            _sciext = fileutil.getExtn(_handle,extn=self.extn)
        except:
            raise IOError, "Unable to open %s for sky level computation"%self.name

        _tmp = ImageStats(_sciext.data,
                fields      = skypars['skystat'],
                lower       = skypars['skylower'],
                upper       = skypars['skyupper'],
                nclip       = skypars['skyclip'],
                lsig        = skypars['skylsigma'],
                usig        = skypars['skyusigma'],
                binwidth    = skypars['skywidth']
                )

        self._computedsky = self._extractSkyValue(_tmp,skypars['skystat'].lower())
        print "Computed sky value for ",self.name," : ",self._computedsky

        # Close input image filehandle
        _handle.close()
        del _sciext,_handle

    def _extractSkyValue(self,ImageStatsObject,skystat):
        if (skystat =="mode"):
            return ImageStatsObject.mode
        elif (skystat == "mean"):
            return ImageStatsObject.mean
        else:
            return ImageStatsObject.median

    def subtractSky(self):
        try:
            try:
                _handle = fileutil.openImage(self.name,mode='update',memmap=self.memmap)
                _sciext = fileutil.getExtn(_handle,extn=self.extn)
                print "%s (computed sky,subtracted sky) : (%f,%f)"%(self.name,self.getComputedSky(),self.getSubtractedSky())
                np.subtract(_sciext.data,self.getSubtractedSky(),_sciext.data)
            except:
                raise IOError, "Unable to open %s for sky subtraction"%self.name
        finally:
            _handle.close()
            del _sciext,_handle
                
    def updateMDRIZSKY(self,filename=None):
    
        if (filename == None):
            filename = self.name
            
        try:
            _handle = fileutil.openImage(filename,mode='update',memmap=self.memmap)
        except:
            raise IOError, "Unable to open %s for sky level computation"%filename
        try:
            try:
                # Assume MDRIZSKY lives in primary header
                print "Updating MDRIZSKY in %s with %f"%(filename,self.getSubtractedSky())
                _handle[0].header['MDRIZSKY'] = self.getSubtractedSky()
            except:
                print "Cannot find keyword MDRIZSKY in %s to update"%filename
                print "Adding MDRIZSKY keyword to primary header with value %f"%self.getSubtractedSky()
                _handle[0].header.update('MDRIZSKY',self.getSubtractedSky(), 
                    comment="Sky value subtracted by Multidrizzle")
        finally:
            _handle.close()
        
    def runDrizCR(self, blotted_array, mask_array, drizcrpars, skypars, corr_file, cr_file):
        """ Run 'deriv' and 'driz_cr' to create cosmic-ray mask for this image. """

        _deriv_array = None
        
        print "Working on image ",self.datafile,"..."
        _deriv_array = quickDeriv.qderiv(blotted_array)

        # Open input image and get pointer to SCI data
        handle = fileutil.openImage(self.name,mode='readonly',memmap=self.memmap)
        scihdu = fileutil.getExtn(handle,extn=self.extn)
        
        tmpDriz_cr = driz_cr.DrizCR(scihdu.data,
                        scihdu.header,
                        blotted_array,
                        _deriv_array,
                        mask_array,
                        gain = self.getEffGain(),
                        grow = drizcrpars['driz_cr_grow'],                                             
                        ctegrow = drizcrpars['driz_cr_ctegrow'],
                        ctedir = self.cte_dir, 
                        amp = self.amp,
                        rn = self.getReadNoise(),
                        SNR = drizcrpars['driz_cr_snr'],
                        backg = self.getSubtractedSky(),
                        scale = drizcrpars['driz_cr_scale'])


        # If the user provided a None value for the cr bit, don't
        # update the dqarray
        if (self.getCRbit() != 0):
            # Update the dq information if there is a dq array to update.
            # In the case of WFPC2 input, no DQ file may have been provided.
            # For ACS, there will always be DQ array information in the FLT file.
            if fileutil.findFile(self.dqfile_fullname):
                dqhandle = fileutil.openImage(self.dqfile_name,mode='update',memmap=self.memmap)
                dqarray = fileutil.getExtn(dqhandle,extn=self.dqfile_extn)
                tmpDriz_cr.updatedqarray(dqarray.data,self.getCRbit())
                dqhandle.close()
        else:
            print "  CR bit value of 0 specified.  Skipping DQ array updates."

        if  (corr_file != None):
            tmpDriz_cr.createcorrfile(corr_file,self.header)
        if (cr_file != None):
            tmpDriz_cr.createcrmaskfile(cr_file,self.header)

        del tmpDriz_cr

        # Close input image filehandle
        if handle:
            del scihdu
            handle.close()
            del handle
        if _deriv_array != None:
            del _deriv_array

    def getflat(self):
        """

        Purpose
        =======
        Placeholder method for retrieving a detector's flat field.
        This is an abstract method since each detector will be 
        handled differently.
        
        This method will return an array the same shape as the
        image.
        """
        pass

    def getEffGain(self):
        """
        
        Purpose
        =======
        Method used to return the effective gain of a instrument's
        detector.
        
        This method returns a single floating point value.

        """

        return self._effGain
    
    def getReadNoise(self):
        """
        
        Purpose
        =======
        Method for trturning the readnoise of a detector (in electrons).
        
        :units: electrons
        
        """
        return self._rdnoise
        
    def getReadNoiseImage(self):
        """
        
        Purpose
        =======
        Method for returning the readnoise image of a detector 
        (in electrons).  
        
        The method will return an array of the same shape as the image.
        
        :units: electrons
        
        """
        return np.ones(self.image_shape,dtype=self.image_dtype) * self._rdnoise

    def getdarkcurrent(self):
        """
        
        Purpose
        =======
        Return the dark current for the detector.  This value
        will be contained within an instrument specific keyword.
        The value in the image header will be converted to units
        of electrons.
        
        :units: electrons
        
        """
        pass


    def getdarkimg(self):
        """
        
        Purpose
        =======
        Return an array representing the dark image for the detector.
        
        :units: electrons
        
        """
        return np.ones(self.image_shape,dtype=self.image_dtype)*self.getdarkcurrent()
    
    def getskyimg(self):
        """
        
        Purpose
        =======
        Return an array representing the sky image for the detector.  The value
        of the sky is what would actually be subtracted from the exposure by
        the skysub step.
        
        :units: electrons
        
        """
        return np.ones(self.image_shape,dtype=self.image_dtype)*self.getSubtractedSky()

    
