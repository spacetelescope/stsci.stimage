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
#           Version 0.1.30 06/07/04 -- Turned memory mapping back on by default.  -- CJH
#           Version 0.1.31 06/29/04 -- Modified import of imagestats. -- CJH
#           Version 0.1.32 06/29/04 -- Modified imports to remove dependence on pytools package -- CJH
#           Version 0.1.33 07/08/04 -- pdated Dictionary key names -- CJH
#           Version 0.1.34 07/15/04 -- Modified the the driz_cr calls to handle the case of WFPC2 data
#               where no DQ file was provided. -- CJH
#           Version 0.1.35 07/29/04 -- Added plate scale as an input to the constructor.  Added plate scale
#               and reference plate scale as parameters.  Added a new method that returns a sky value based
#               upon the reference chip.  This is used in the create median step for WFPC2.  Currently
#               for ACS and STIS data is returns the same value that getSubtractedSky would.
#           Version 0.1.36 09/09/04 -- The getInstrParameter method was modified to raise a value error
#               if a user were to specify both a header keyword and a value for a specific parameter.
#               This type of input is ambiguous.  Previously the value would be used and the header
#               keyword silently ignored. -- CJH
#           Version 0.1.37 09/15/04 -- The runDrizCR step was modified to skip the DQ array update if the
#               cr bit value being used is None.  -- CJH
#           Version 0.1.38 09/29/04 -- Modified getExptime to return a value of 1 if exptime is 0.  This is
#               to address an issue with ACS data that can have exptimes = 0.  This may or may not be ACS 
#               specific-- CJH
#           Version 0.1.39 09/30/04 -- Modified the doDrizCR method to pass the input file's primary + extension
#               header to the cor and cr mask file creation methods for inclusion in the resulting fits file.
#           Version 1.0.0 11/03/04 -- Removed the modification to getExptime that was added in version 0.1.38.
#               This fix to this problem will now be addressed points in multidrizzle where scaling occurs.  This
#               allows for a more accurate treatment of image weighting by pydrizzle.
#           Version 1.1.0 06/02/05 -- Added parameters driz_cr_grow and driz_cr_ctegrow for CTE masking of cosmic
#               rays. -- DMG

__version__ = '1.1.0'

import pyfits
import fileutil

import imagestats
from imagestats import ImageStats

import imageiter
from imageiter import ImageIter

import numerix as N
import quickDeriv
import driz_cr

DEFAULT_SEPARATOR = '_'

class InputImage:
    '''The InputImage class is the base class for all of the various
       types of images
    '''

    def __init__(self, input,dqname,platescale,memmap=0):
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
        self.grp      = fileutil.parseExtn(_extn)
        self.rootname = self.getRootname(_fname)
        self.datafile = _fname
        self.cr_bits_value = None
        self._effGain = None
        self.static_badval = 64
        self.static_mask = None
        self.cte_dir = 1 

        try:  # read amplifier to be used for HRC or STIS/CCD.
           self.amp = fileutil.getKeyword(input,'CCDAMP')
        except:  # set default if keyword missing    
           self.amp = 'C'  # STIS default should be 'D' but 'C' and 'D' have the same readout direction so it's okay
        
        # Define the platescale and reference plate scale for the detector.
        self.platescale = platescale
        self.refplatescale = platescale # Default is to make each chip it's own reference value
        
        # Image information
        handle = fileutil.openImage(self.name,mode='readonly',memmap=self.memmap)
        sciext = fileutil.getExtn(handle,extn=self.extn)
        self.image_shape = sciext.data.shape
        self.image_type = sciext.data.dtype.name
        # Retrieve a combined primary and extension header
        self.header = fileutil.getHeader(input,handle=handle)
        del sciext
        handle.close()
        del handle

        # Initialize sky background information keywords
        self._subtractedsky = 0.
        self._computedsky = None

    def setInstrumentParameters(self, instrpars, pri_header):
        """ Sets the instrument parameters.
        """
        pass

    def doUnitConversions(self):
        """
        Convert the sci extensions pixels to electrons
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
        return (self._subtractedsky * (self.refplatescale / self.platescale)**2 )

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
                _handle = fileutil.openImage(self.name,mode='update',memmap=self.memmap)
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
        try:
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

        finally:
            # Close input image filehandle
            if handle:
                del scihdu
                handle.close()
                del handle
            if _deriv_array != None:
                del _deriv_array
