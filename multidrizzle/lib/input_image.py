#
#   Authors: Warren Hack, Ivo Busko, Christopher Hanley
#   Program: input_image.py
#   Purpose: Super class used to model astronomical data from observatory instruments.
#   History:
#           Version 0.1.22, 03/23/04 -- Added the attibute cr_bits_value to
#               indicate what bit value a cosmic rate hit will represent in
#               the data quality files. -- CJH
#           Version 0.1.23, 04/19/04 -- Added setInstrumentParameters method. -- IB
#           Version 0.1.24, 04/22/04 -- Modified added new attribute call 'effgain'.
#               The effective gain is now used by driz_cr.  -- CJH
#           Version 0.1.25, 05/08/04 -- Eliminated public gain and readnoise attributes
#               to take advantage of new access methods created by Ivo.  Added new attributes
#               for subtracted sky and computer sky values. -- CJH
#           Version 0.1.26, 05/13/04 -- Added code to support MDRIZTAB. -- IB
#           Version 0.1.27, 05/20/04 -- Added support for cr mask creation. -- CJH
#           Version 0.1.28, 05/26/04 -- Fixed bug in call to driz_cr.  I am now passing the readnoise
#               parameter the readnoise, instead of the gain.  -- CJH
#           Version 0.1.29 05/27/04 -- Modified setInstrumentParameters inheritence in
#               order to support ACS/SBC data.  - CJH/WJH/IB
#           Version 0.1.30 06/07/04 -- Turned memory maping back on by default.  -- CJH

__version__ = '0.1.30'

import pyfits

import pydrizzle
from pydrizzle import fileutil

import pytools.imagestats
from pytools.imagestats import ImageStats

import pytools.imageiter
from pytools.imageiter import ImageIter

import numarray as N
import quickDeriv
import driz_cr

DEFAULT_SEPARATOR = '_'

class InputImage:
    '''The InputImage class is the base class for all of the various
       types of images
    '''

    def __init__(self, input,dqname,memmap=1):
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
        self.dqfile_name,self.dqfile_extn = fileutil.parseFilename(dqname)
        self.extn     = _extn
        self.grp      = fileutil.parseExtn(_extn)
        self.rootname = self.getRootname(_fname)
        self.datafile = _fname
        self.cr_bits_value = None
        self._effGain = None
        self.static_badval = 64
        self.static_mask = None

        # Image information
        __handle = fileutil.openImage(self.name,mode='readonly',memmap=self.memmap)
        __sciext = fileutil.getExtn(__handle,extn=self.extn)
        self.image_shape = __sciext.data.shape
        self.image_type = __sciext.data.type()
        self.header = __sciext.header.copy()
        del __sciext
        __handle.close()
        del __handle

        # Initialize sky background information keywords
        self._subtractedsky = 0.
        self._computedsky = None

    def setInstrumentParameters(self, instrpars, pri_header):
        """ Sets the instrument parameters.
        """
        pass
        
    def getInstrParameter(self, value, header, keyword):
        """ This method gets a instrument parameter from a
            pair of task parameters: a value, and a header keyword.

            The default behavior is:
              - if the value is given, use it.
              - if the value is blank and the header keyword is given, use
                the header keyword.
              - if both are blank, or if the header keyword is not
                found, return None.
        """
        if value != None and value != '':
            return self._averageFromList(value)
        elif keyword != None and keyword != '':
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
        __handle = fileutil.openImage(self.name,mode='readonly',memmap=1)
        __sciext = fileutil.getExtn(__handle,extn=self.extn)

        # Add SCI array to static mask
        static_mask.addMember(__sciext.data, self.signature())
        self.static_mask = static_mask

        # Close input image filehandle
        __handle.close()
        del __sciext,__handle

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

        try:
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

        except:
            raise SystemError, "Unable to compute sky level for %s"%self.name

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
                _handle = fileutil.openImage(self.name,mode='update',memmap=0)
                _sciext = fileutil.getExtn(_handle,extn=self.extn)
                print "%s (computed sky,subtracted sky) : (%f,%f)"%(self.name,self.getComputedSky(),self.getSubtractedSky())
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
        try:
            _deriv_array = quickDeriv.qderiv(blotted_array)

            # Open input image and get pointer to SCI data
            __handle = fileutil.openImage(self.name,mode='readonly',memmap=1)
            __scihdu = fileutil.getExtn(__handle,extn=self.extn)
            __dqhandle = fileutil.openImage(self.dqfile_name,mode='update',memmap=1)
            __dqarray = fileutil.getExtn(__dqhandle,extn=self.dqfile_extn)

            __tmpDriz_cr = driz_cr.DrizCR(__scihdu.data,
                            __scihdu.header,
                            blotted_array,
                            _deriv_array,
                            mask_array,
                            gain = self.getEffGain(),
                            rn = self.getReadNoise(),
                            SNR = drizcrpars['snr'],
                            backg = self.getSubtractedSky(),
                            scale = drizcrpars['scale'])

            __tmpDriz_cr.updatedqarray(__dqarray.data,self.cr_bits_value)

            if  (corr_file != None):
                __tmpDriz_cr.createcorrfile(corr_file)
            if (cr_file != None):
                __tmpDriz_cr.createcrmaskfile(cr_file)

            del __tmpDriz_cr

        finally:
            # Close input image filehandle
            if __handle:
                del __scihdu
                __handle.close()
                del __handle
            if _deriv_array != None:
                del _deriv_array
