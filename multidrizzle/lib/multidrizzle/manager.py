#
#   Authors: Warren Hack, Ivo Busko, Christopher Hanley, David Grumm
#   Program: manager.py
#   Purpose: Class Used to drive all algorithmic operations within Multidrizzle.

from __future__ import division # confidence medium

# Import numpy functionality
import numpy as np

# Import file utilities
import pyfits
import shutil, os

# Import Pydrizzle
from pydrizzle import pydrizzle
from pydrizzle import drutil,buildmask
from stsci.tools import fileutil, wcsutil

# Import support for specific HST Instruments
from acs_input import WFCInputImage, HRCInputImage, SBCInputImage
from wfpc2_input import WFPC2InputImage, PCInputImage, WF2InputImage, WF3InputImage, WF4InputImage
from stis_input import STISInputImage, CCDInputImage, FUVInputImage, NUVInputImage
from nicmos_input import NICMOSInputImage, NIC1InputImage, NIC2InputImage, NIC3InputImage
from wfc3_input import WFC3UVISInputImage, WFC3IRInputImage

# Import general tools
from stsci.imagestats import ImageStats
import minmed
from minmed import minmed
import static_mask
from static_mask import StaticMask
from stsci.image.numcombine import numCombine
from stsci.tools import nimageiter
from stsci.tools.nimageiter import ImageIter,computeBuffRows,FileIter,computeNumberBuff
from stsci.tools import iterfile
from stsci.tools.iterfile import IterFitsFile

DEFAULT_ORIG_SUFFIX = '_OrIg'

def modifyRootname(rootname,suffix=None):

    """ Inserts '_copy' into a rootname. """
    # This list of extensions includes:
    #   '.fits','.hhh', and '.c0h'
    if not suffix: suffix = DEFAULT_ORIG_SUFFIX
    _extlist = fileutil.EXTLIST
    _indx = None
    # Start out by searching for known extensions...
    for _extn in _extlist:
        _indx = rootname.find(_extn)
        if _indx > 0: break
    # If we do not find a recognizable extension/suffix,
    # raise an Exception.
    if not _indx: raise ValueError,"Rootname %s not recognized!"%rootname

    return rootname[:_indx]+suffix+rootname[_indx:]

class ImageManager(object):
    """ The ImageManager class handles most of the file I/O operations for the input
        images being processed.  Multiple InputImage objects can point to the
        same ImageManager object without worrying about opening or trying to close
        the same image more than once.
    """
    def __init__(self, assoc, context, instrpars, workinplace, static_file, proc_unit="native"):  
        self.context = context
        self.assoc = assoc
        self.output = self.assoc.output
        self.workinplace = workinplace
        self.static_file = static_file
        self.proc_unit = proc_unit
        
        # Establish a default memory mapping behavior
        self.memmap = 0

        # Global static mask.
        self.static_mask = None

        # Keep track of open input images here
        #   single_handles --> singly-drizzled output science images
        #   weight_handles --> singly-drizzled output weight images
        self.single_handles = []
        self.weight_handles = []

        # Keep track of array lists:
        #    single_list --> singly-drizzled output science arrays
        #    weight_list --> singly-drizzled weight arrays
        #    weight_mask_list --> masks from singly-drizzled weight arrays
        # If weight_mask_list stays as None, then no masks are requested or made
        #
        self.single_list = []
        self.weight_list = []
        self.weight_mask_list = None

        # Generate filename for median file
        _medextn = '_drz.fits'
        for extn in ['_drz.fits','_drc.fits']:
            if extn in self.output: 
                _medextn = extn
                break
        
        _indx = self.output.find(_medextn)
        if _indx < 0:
            _indx = len(self.output)
        self.medianfile = self.output[:_indx] + '_med.fits'

        for p in assoc.parlist:

            # Make copies of the input science images
            # Need to simply call 'removeINPUTcopies' to delete these files
            self.setupInputCopies(p,self.workinplace)

            # Setup InputImage objects for this association
            p['image'] = self._getInputImage(p['data'],p)

            # Setup Multidrizzle-specific parameters
            p['datafile'] = p['image'].datafile
            grp = str(p['image'].grp)
            rootname = p['image'].rootname

            # Setup file for mask array,
            #    creating one if it doesn't already exist.
            if not p['driz_mask']:
                # No array has been created yet, so initialize it now...
                _mask_array = None # np.ones(p['image'].image_shape,dtype=np.uint8)
                # We also need to create a name for the mask array file
                p['image'].maskname = p['exposure'].masklist[0]
                # Build the mask image
                self._buildMaskImage(p['image'].maskname,_mask_array)
                del _mask_array
            else:
                p['image'].maskname = p['driz_mask']

            # Create the name of the single drizzle mask file if it doesn't exist.
            
            # This is done in the default case in which the user has not provided a
            # static mask file
            p['image'].singlemaskname = p['exposure'].singlemaskname

            # In the case of the user having provided a static mask file, we need to
            # make certain that pydrizzle has created a mask file for the single drizzle
            # step.  It is possible that no mask was created for single drizzle in the
            # case that the 'driz_sep_bits' parameter was INDEF (None).
            if (self.static_file != None):
                # We need to apply a user supplied static mask file.  We need to make
                # certain that pydrizzle created a mask from the data quality information.
                # If not mask was created, we will need to make one that the user's static
                # file can be applied to.
                #
                # If p['image'].singlemaskname == None at this point, no mask was created.
                if (p['image'].singlemaskname == None):
                    # Since Pydrizzle didn't create a mask, we need to figure out what name
                    # the mask would have gotten if the mask had been created.
                    p['image'].singlemaskname = p['exposure'].masklist[1]
                    p['exposure'].singlemaskname = p['exposure'].masklist[1]
                    
                    # Now create the mask file on disk
                    _mask_array = None # np.ones(p['image'].image_shape,dtype=np.uint8)
                    self._buildMaskImage(p['image'].singlemaskname,_mask_array)
                    del _mask_array
                    
                # Now that we have created a mask for the single drizzle step, we will need
                # to apply the user supplied static mask file to it and the mask for the 
                # final drizzle step.
                self._applyUserStaticFile(p['image'].singlemaskname, 
                                                        static_file, 
                                                        p['image'].grp
                                                        )
                self._applyUserStaticFile(p['image'].maskname, 
                                                    static_file, 
                                                    p['image'].grp
                                                    )

            #
            # Rename some of the variables defined by pydrizzle:
            #
            if not self.context:
                p['outcontext'] = ''

            p['outblot'] = rootname + '_sci' + grp + '_blt.fits'
            #
            p['fillval'] = 'INDEF'
            #
            # Add some new variables:
            p['rootname'] = rootname
            p['rootname_sci'] = rootname + '_sci' + grp

            #p['crsingle'] = rootname + '_chip' + grp + '_crsingle.pl'
            p['dq_mask'] = p['driz_mask']
            p['dxsh'] = 0.
            p['dysh'] = 0.
            p['drot'] = 0.

            # Used to keep track of full mask filename
            # when it needs to be replaced by static_mask
            p['full_mask'] = None

        # This is done in a separate loop over the parlist so that
        # instrument parameters can also be reset from the mdriztab
        # table by an external call.
        self.setInstrumentParameters(instrpars)
                

    def _applyUserStaticFile(self, static_mask, user_static_mask, imageExtNumber):
    
        """
        Method:     _applyUserStaticFile
        Purpose:    Apply the user provided static mask file to the mask created by the
                    static mask step.
                    
        Input:      user_static_file - user specified mask file.  The file is assumed to be 
                    a multi extension FITS file.  Pixel values of 1 are considered "good"
                    and 0 "bad".  There needs to be one "MASK" extension for each extension
                    of the input science image.  This means a static mask file for WFPC2
                    input should have 4 "MASK" extensions while a static mask file for
                    ACS WFC dat would have 2 "MASK" extensions.
                    
                    static_mask - name of static mask file for single and final drizzle steps created
                    by Multidrizzle/Pydrizzle.
                                        
                    imageExtNumber - extension of the inputimage currently being processed
                    by the static_mask step.
                    
        Output:     None
        
        """

        # Define integer values for good and bad pixels
        
        static_goodval = 1
        static_badval = 0

        # Build the appropriate extension name to use on the user static mask file
        extn = "MASK,"+str(imageExtNumber)

        # Open the user supplied static mask file and extract the approprate extension
        file = fileutil.openImage(user_static_mask, memmap=0, mode='readonly')
        input_user_static_mask = fileutil.getExtn(file,extn)

        # Open the static mask file
        static_mask_file = fileutil.openImage(static_mask,memmap=0, mode='update')
        final_static_mask = fileutil.getExtn(static_mask_file,str(0))

        # Apply the user supplied static mask to the static mask by Pydrizzle/Multidrizzle
        tmpArray = np.where( np.equal(input_user_static_mask.data, static_goodval), final_static_mask.data, static_badval)
        final_static_mask.data = tmpArray.copy()

        # Clean up
        del(tmpArray)

        file.close()
        static_mask_file.close()

        print "STATICFILE: ",user_static_mask," applied to ",static_mask             

    def setInstrumentParameters(self, instrpars):
        """ Sets intrument parameters into all image instances.
        """

        # A reference to the primary header is also required.

        for p in self.assoc.parlist:
            p['image'].setInstrumentParameters (instrpars, p['exposure'].header)
            
    def setupInputCopies(self,p,workinplace = False ):
        """ Make copies of all input science files, keeping track of
            the names of the copies and originals.
            Entries in parlist will be:
                ['data']      - copy of input science image
                ['orig_filename'] - original input science image
        """

        _img_root,_img_extn = fileutil.parseFilename(p['data'])

        if not workinplace:
            # Make copies of input images
            _copy = modifyRootname(_img_root)

            # Update parlist entries with pointers to new filename
            p['orig_filename'] = _copy
        else:
            p['orig_filename'] = _img_root
        


    # This is called after 'doFinalDriz'...
    def removeInputCopies(self):
        """ Delete copies of input science images.
        """
        for p in self.assoc.parlist:
            if int(p['group']) == 1:
                _img = p['image'].datafile
                shutil.move(p['orig_filename'],_img)


    def removeMDrizProducts(self):
        """ Remove all intermediate products. """

        # Remove all PyDrizzle intermediate products
        self.assoc.clean(coeffs=True,final=False)

        # Remove median file created by MultiDrizzle
        if os.path.exists(self.medianfile):
            os.remove(self.medianfile)

    def _getInputImage (self, input, plist):
        """ Factory function to return appropriate InputImage class type"""

        # Extract the instrument name for the data that is being processed by Multidrizzle
        _instrument = plist['exposure'].header['INSTRUME']
        
        # Determine the instrument detector in use.  NICMOS is a special case because it does
        # not use the 'DETECTOR' keyword.  It instead used 'CAMERA' to identify which of it's
        # 3 camera's is in use.  All other instruments support the 'DETECTOR' keyword.
        if (_instrument == 'NICMOS'):
            _detector = plist['exposure'].header['CAMERA']
        else:
            _detector = plist['exposure'].header['DETECTOR']
            
        # Extract the plate scale in use by the detector
        _platescale = plist['exposure'].pscale
        if _platescale == None:
            raise ValueError, 'The plate scale has a value of -- None -- '
        
        # Extract the dq array designation
        _dqname = plist['exposure'].dqname
        if _instrument != 'WFPC2':
            _dq_root,_dq_extn = fileutil.parseFilename(_dqname)
            _dqname = plist['orig_filename']+'['+_dq_extn+']'

        if _instrument == 'ACS':
            if _detector == 'HRC': return HRCInputImage(input,_dqname,_platescale,memmap=0,proc_unit=self.proc_unit)
            if _detector == 'WFC': return WFCInputImage(input,_dqname,_platescale,memmap=0,proc_unit=self.proc_unit)
            if _detector == 'SBC': return SBCInputImage(input,_dqname,_platescale,memmap=0,proc_unit=self.proc_unit)
        if _instrument == 'WFPC2':
            if _detector == 1: return PCInputImage(input,_dqname,_platescale,memmap=0,proc_unit=self.proc_unit)
            if _detector == 2: return WF2InputImage(input,_dqname,_platescale,memmap=0,proc_unit=self.proc_unit)
            if _detector == 3: return WF3InputImage(input,_dqname,_platescale,memmap=0,proc_unit=self.proc_unit)
            if _detector == 4: return WF4InputImage(input,_dqname,_platescale,memmap=0,proc_unit=self.proc_unit)
        if _instrument == 'STIS': 
            if _detector == 'CCD': return CCDInputImage(input,_dqname,_platescale,memmap=0,proc_unit=self.proc_unit)
            if _detector == 'FUV-MAMA': return FUVInputImage(input,_dqname,_platescale,memmap=0,proc_unit=self.proc_unit)
            if _detector == 'NUV-MAMA': return NUVInputImage(input,_dqname,_platescale,memmap=0,proc_unit=self.proc_unit)
        if _instrument == 'NICMOS':
            if _detector == 1: return NIC1InputImage(input,_dqname,_platescale,memmap=0,proc_unit=self.proc_unit)
            if _detector == 2: return NIC2InputImage(input,_dqname,_platescale,memmap=0,proc_unit=self.proc_unit)
            if _detector == 3: return NIC3InputImage(input,_dqname,_platescale,memmap=0,proc_unit=self.proc_unit)
        if _instrument == 'WFC3':
            if _detector == 'UVIS': return WFC3UVISInputImage(input,_dqname,_platescale,memmap=0,proc_unit=self.proc_unit)
            if _detector == 'IR': return WFC3IRInputImage(input,_dqname,_platescale,memmap =0,proc_unit=self.proc_unit)

        # If a supported instrument is not detected, print the following error message
        # and raise an exception.
        msg = 'Instrument: ' + str(_instrument) + '/' + str(_detector) + ' not yet supported!'
        raise ValueError, msg

    def _buildMaskImage(self,maskname, mask_array):
        """ Build a default 'weight' array image to use for keeping track of
            the mask information built up by MultiDrizzle in case PyDrizzle
            does not create one itself.
        """
        # If an old version of the maskfile was present,
        # remove it and rebuild it.
        if fileutil.findFile(maskname):
            fileutil.removeFile(maskname)

        _file = pyfits.open(maskname,mode='append')
        _phdu = pyfits.PrimaryHDU(data=mask_array)

        _file.append(_phdu)
        _file.close()
        del _file, _phdu


    def createStatic(self, static_sig):

        """ Create the static bad-pixel mask from input images."""
        
        #Print paramater values supplied through the interface
        print "USER PARAMETERS:"
        print "static     =  True"
        print "static_sig = ",static_sig
        print "\n"
                 
        self.static_mask = StaticMask(goodval = 1, badval = 0, staticsig=static_sig)

        for p in self.assoc.parlist:
            p['image'].updateStaticMask(self.static_mask)

        # For each input, we now need to update the driz_mask with the
        # values from the static mask
        # Combine in_mask array with static mask.
        # The mask_array should always be present, created by
        # ImageManager if PyDrizzle does not create one.
        for p in self.assoc.parlist:
            handle = fileutil.openImage(p['image'].maskname,mode='update')
            static_array = self.static_mask.getMask(p['image'].signature())

            # Print progress message to alert the user that the mask file
            # is now being updated with the static mask information
            # If verbose mode is implemented, this could be included.
            print 'Updating mask file: ',p['image'].maskname,' with static mask.'

            if static_array != None:
                # Account for case that no mask has been created yet
                if handle[0].data is None:
                    handle[0].data = np.ones(p['image'].image_shape,dtype=np.uint8)
                handle[0].data = np.bitwise_and(handle[0].data,static_array)

            handle.close()
            del handle
        
            # If there is going to be a separate mask for the single drizzle step and it is different
            # from the mask used for the final drizzle step it will also need to be updated with the 
            # static mask information
            if ( ((p['image'].singlemaskname != None) and (p['image'].singlemaskname != '')) and (p['image'].singlemaskname != p['image'].maskname) ):
                handle = fileutil.openImage(p['image'].singlemaskname,mode='update')
                static_array = self.static_mask.getMask(p['image'].signature())

                # Print progress message to alert the user that the mask file
                # is now being updated with the static mask information
                # If verbose mode is implemented, this could be included.
                print 'Updating mask file: ',p['image'].singlemaskname,' with static mask.'

                if static_array != None:
                    # Account for case that no mask has been created yet
                    if handle[0].data is None:
                        handle[0].data = np.ones(p['image'].image_shape,dtype=np.uint8)
                    handle[0].data = np.bitwise_and(handle[0].data,static_array)

                handle.close()
                del handle
                
    def doSky(self, skypars, skysub):
    
        # Print out the parameters provided by the interface
        print "USER PARAMETERS:"
        print "skysub    = ",skysub
        print "skywidth  = ",skypars['skywidth']
        print "skystat   = ",skypars['skystat']
        print "skylower  = ",skypars['skylower']
        print "skyupper  = ",skypars['skyupper']
        print "skyclip   = ",skypars['skyclip']
        print "skylsigma = ",skypars['skylsigma']
        print "skyusigma = ",skypars['skyusigma']
        print "skyuser   = ",skypars['skyuser'] 
        print "\n"

        """ Processes sky in input images."""
        if (skypars['skyuser'] != ''):
            # User Subtraction Case, User has done own sky subtraction, we use image header value for _subtractedsky value
            print "User has done own sky subtraction, updating MDRIZSKY with supplied value..."
            for p in self.assoc.parlist:
                if int(p['group']) == 1:
                    try:
                        _handle = fileutil.openImage(p['image'].datafile,mode='update',memmap=0)
                        _userSkyValue = _handle[0].header[skypars['skyuser']]
                        _handle.close()
                    except:
                        print "**************************************************************"
                        print "*"
                        print "*  Cannot find keyword ",skypars['skyuser']," in ",p['image'].datafile," to update"
                        print "*"
                        print "**************************************************************\n\n\n"
                        raise KeyError
                p['image'].setSubtractedSky(_userSkyValue)
        elif (skysub):
            # Compute our own sky values and subtract them from the image copies.
            print "Subtracting sky..."
            _imageMinDict = {}
            _currentImageName = "no match"
            for p in self.assoc.parlist:
                p['image'].computeSky(skypars)
                _computedImageSky = p['image'].getComputedSky()
                if (p['rootname'] != _currentImageName):
                    _currentMinSky = _computedImageSky
                    _currentImageName = p['rootname']
                    _imageMinDict[_currentImageName]=_currentMinSky
                else:
                    if (_computedImageSky < _imageMinDict[p['rootname']]):
                        _imageMinDict[p['rootname']]=_computedImageSky

            for p in self.assoc.parlist:
                # We need to account for the fact that STIS associations contain
                # separate exposures for the same chip within the same file.
                # In those cases, we do not want to use the minimum sky value
                # for the set of chips, but rather use each chip's own value.
                # NOTE: This can be generalized later with changes to PyDrizzle
                #       to provide an attribute that specifies whether each member
                #       associated with file is a separate exposure or not.
                #   WJH/CJH 
                if (p['exposure'].header['INSTRUME'] != 'STIS'):
                    p['image'].setSubtractedSky(_imageMinDict[p['rootname']])
                else:
                    p['image'].setSubtractedSky(p['image'].getComputedSky())
                p['image'].subtractSky()

        else:
            # Default Case, sky subtraction is turned off.  No sky subtraction done to image.
            print "No sky subtraction requested, MDRIZSKY set to a value of 0..."
            for p in self.assoc.parlist:
                p['image'].setSubtractedSky(0)

        #Update the MDRIZSKY KEYWORD
        for p in self.assoc.parlist:
            if int(p['group']) == 1:
                p['image'].updateMDRIZSKY(p['orig_filename'])
                #p['image'].updateMDRIZSKY(p['image'].datafile)

    def _setOutputFrame(self, pars):

        """ Set up user-specified output frame using a SkyField object."""        
        _sky_field = None

        if pars['refimage'] != '' and pars['refimage'] != None:
            # Use the following if the refimage isn't actually going to be
            # drizzled, we just want to set up the pydrizzle object
            #
            _refimg = wcsutil.WCSObject(pars['refimage'])
            refimg_wcs = _refimg.copy()

            # If the user also specified a rotation to be applied,
            # apply that as well...
            if pars['rot']:
                _orient = pars['rot']
            else:
                _orient = refimg_wcs.orientat

           # Now, build output WCS using the SkyField class
            # and default product's WCS as the initial starting point.
            #
            _sky_field = pydrizzle.SkyField(wcs=refimg_wcs)
            # Update with user specified scale and rotation
            _sky_field.set(psize=pars['scale'],orient=_orient)

        elif pars['rot']   != None  or \
             pars['scale'] != None or \
             pars['ra']    != None or \
             pars['outnx'] != None:

            _sky_field = pydrizzle.SkyField()

            if pars['rot'] == None:
                _orient = self.assoc.observation.product.geometry.wcslin.orient
            else:
                _orient = pars['rot']

            # Need to account for non-existent specification of shape
            # when setting up output field parameters.
            if pars['outnx'] == None: _shape = None
            else: _shape = (pars['outnx'],pars['outny'])
            
            print 'Default orientation for output: ',_orient,'degrees'

            _sky_field.set(psize=pars['scale'], orient=_orient,
                           ra=pars['ra'], dec=pars['dec'], shape=_shape)

        # Now that we have built the output frame, let the user know
        # what was built...
        if _sky_field != None:
            print ('\n Image parameters computed from reference image WCS: \n')
            print _sky_field.wcs

        # Apply user-specified output to ASN using the resetPars method.
        # If field==None, it will simply reset to default case.
        #
        self.assoc.resetPars(field=_sky_field,
                            pixfrac=pars['pixfrac'],
                            kernel=pars['kernel']) 

    def doDrizSeparate(self, pars):

        """ Drizzle separate input images. """
        
        # Start by applying input parameters to redefine
        # the output frame as necessary
        for p in self.assoc.parlist:
            
            # First do some cleaning up, in case you are restarting...
            fileutil.removeFile(p['outsingle'])
            if (p['outsweight'] != ''):
                fileutil.removeFile(p['outsweight'])

            # NB DO NOT USE "tophat" unless pixfrac is sufficiently
            # large (> sqrt(2))
            
            p['fillval'] = pars['fillval']
            
            # Pass in the new wt_scale value
            p['wt_scl'] = pars['wt_scl']

            if (p['single_driz_mask'] == None and self.static_mask != None):
                p['single_driz_mask'] = self.static_mask.getMask(p['image'].signature())
                
            # 'in_units' will always be counts given that the input is converted to 'electrons' in call cases
            p['in_units'] = 'counts'
            
            print("\ndrizzle data='"+p['data']+"' outdata='"+p['outsingle']+"' outweig='"+p['outsweight']+
                "' in_mask='static_mask"+"' kernel='"+p['kernel']+
                "' outnx="+str(p['outnx'])+" outny="+str(p['outny'])+" xsh="+str(p['xsh'])+" ysh="+str(p['ysh'])+
                " scale="+str(p['scale'])+" pixfrac="+str(p['pixfrac'])+" rot="+str(p['rot'])+
                " coeffs='"+p['coeffs']+"' wt_scl='"+str(p['wt_scl'])+"' align='center' shft_fr='output' shft_un='output'"+
                " out_un='"+p['units']+"' expkey='"+"EXPTIME"+"' fillval='"+str(p['fillval'])+"'"+
                " xgeoim='"+p['xgeoim']+"' ygeoim='"+p['ygeoim']+"'\n")

        # Perform separate drizzling now that all parameters have been setup...
        self.assoc.run(single=True,save=True,build=False)

        # Now that we are done with the static mask, delete it...
        del self.static_mask


    def createMedian(self, medianpars):
    
        # Print out the parameters provided by the interface
        print "USER PARAMETERS:"
        print "median          =  True"
        print "median_newmasks  = ",medianpars['newmasks']
        print "combine_maskpt  = ",medianpars['maskpt']
        print "combine_type    = ",medianpars['type']  
        print "combine_nsigma  = ",medianpars['nsigma1']," ",medianpars['nsigma2']
        print "combine_nlow    = ",medianpars['nlow']
        print "combine_nhigh   = ",medianpars['nhigh']
        print "combine_lthresh = ",medianpars['lthresh']
        print "combine_hthresh = ",medianpars['hthresh']
        print "combine_grow    = ",medianpars['grow']
        print "\n"
                    
        newmasks = medianpars['newmasks']
        comb_type = medianpars['type']
        nsigma1 = medianpars['nsigma1']
        nsigma2 = medianpars['nsigma2']
        nlow = medianpars['nlow']
        nhigh = medianpars['nhigh']
        
        # Convert the units of the threshold values if necessary
        native_units = self.assoc.parlist[1]['image'].native_units
        proc_units = self.assoc.parlist[1]['image'].proc_unit
        det_gain = self.assoc.parlist[1]['image'].getGain()
        img_exptime = self.assoc.parlist[1]['image'].getExpTime()

        if (medianpars['lthresh'] == None):
            lthresh = None
        else:
            lthresh = float(medianpars['lthresh'])
            if proc_units.lower() == 'native':
                if native_units.lower() == "counts":
                    lthresh = lthresh * det_gain
                    if native_units.lower() == "counts/s":
                        lthresh = lthresh * img_exptime

        if (medianpars['hthresh'] == None):
            hthresh = None
        else:
            hthresh = float(medianpars['hthresh'])
            if proc_units.lower() == 'native':
                if native_units.lower() == "counts":
                    htrhest = hthresh * det_gain
                    if native_units.lower() == "counts/s":
                        hthresh = hthresh * img_exptime


        grow = medianpars['grow']
        maskpt = medianpars['maskpt']
        
        """ Builds combined array from single drizzled images."""
        # Start by removing any previous products...
        fileutil.removeFile(self.medianfile)

        # Compute the mean value of each wht image
        _wht_mean = []

        # Define lists for instrument specific parameters.
        readnoiseList = []
        exposureTimeList = []
        backgroundValueList = []

        for p in self.assoc.parlist:
            # Extract the single drizzled image.
            if p['group'] == 1:
                _file = IterFitsFile(p['outsingle'])
                self.single_handles.append(_file)
                #self.single_list.append(_file[0].data)

                # If it exists, extract the corresponding weight images
                if (fileutil.findFile(p['outsweight'])):

                    _weight_file = IterFitsFile(p['outsweight'])
                    self.weight_handles.append(_weight_file)
                    tmp_mean_value = ImageStats(_weight_file.data,lower=1e-8,lsig=None,usig=None,fields="mean",nclip=0)
                    
                    _wht_mean.append(tmp_mean_value.mean * maskpt)
                    # Clear the memory used by reading in the whole data array for
                    # computing the mean.  This requires that subsequent access to
                    # the data values happens through the 'section' attribute of the HDU.
                    #del _weight_file.data

                # Extract instrument specific parameters and place in lists
                
                # Check for 0 exptime values.  If an image has zero exposure time we will
                # redefine that value as '1'.  Although this will cause inaccurate scaling
                # of the data to occur in the 'minmed' combination algorith, this is a 
                # necessary evil since it avoids divide by zero exceptions.  It is more
                # important that the divide by zero exceptions not cause Multidrizzle to
                # crash in the pipeline than it is to raise an exception for this obviously
                # bad data even though this is not the type of data you would wish to process
                # with Multidrizzle.
                #
                # Get the exposure time from the InputImage object
                imageExpTime = p['image'].getExpTime() 
                exposureTimeList.append(imageExpTime)

                # Extract the sky value for the chip to be used in the model
                backgroundValueList.append(p['image'].getreferencesky())
                # Extract the readnoise value for the chip
                readnoiseList.append(p['image'].getReadNoise())

                print "reference sky value for image ",p['image'].rootname," is ", p['image'].getreferencesky()
            #
            # END Loop over input image list
            #
                
        # create an array for the median output image
        medianOutputImage = np.zeros(self.single_handles[0].shape,dtype=self.single_handles[0].type())

        # create the master list to be used by the image iterator
        masterList = []
        masterList.extend(self.single_handles)
        masterList.extend(self.weight_handles)

        print '\n'

        # Specify the location of the medianOutputImage section in the masterList
        #medianOutputImageSection = 0

        # Specify the location of the drz image sections
        startDrz = 0
        endDrz = len(self.single_handles)+startDrz

        # Specify the location of the wht image sections
        startWht = len(self.single_handles)+startDrz
        endWht = startWht + len(self.weight_handles)

        # Fire up the image iterator
        #
        # The overlap value needs to be set to 2*grow+1 in order to 
        # avoid edge effects when scrolling down the image, and to
        # insure that the last section returned from the iterator
        # has enough row to span the kernel used in the boxcar method
        # within minmed.  
        _overlap = 2*int(grow)+1
        
        #Start by computing the buffer size for the iterator
        _imgarr = masterList[0].data
        _bufsize = nimageiter.BUFSIZE
        _imgrows = _imgarr.shape[0]
        _nrows = computeBuffRows(_imgarr)
        _niter,_nrows = computeNumberBuff(_imgrows,_nrows,_overlap)
        _lastrows = _imgrows - (_niter*_nrows) 
                
        masterList[0].close()
        del _imgarr

        for imageSectionsList,prange in FileIter(masterList,overlap=_overlap,bufsize=_bufsize):
            if newmasks:
                """ Build new masks from single drizzled images. """
                self.weight_mask_list = []
                listIndex = 0
                for _weight_arr in imageSectionsList[startWht:endWht]:
                    # Initialize an output mask array to ones
                    # This array will be reused for every output weight image
                    _weight_mask = np.zeros(_weight_arr.shape,dtype=np.uint8)

                    """ Generate new pixel mask file for median step.
                    This mask will be created from the single-drizzled
                    weight image for this image.

                    The mean of the weight array will be computed and all
                    pixels with values less than 0.7 of the mean will be flagged
                    as bad in this mask.  This mask will then be used when
                    creating the median image.
                    """
                    # Compute image statistics
                    _mean = _wht_mean[listIndex]

                    # 0 means good, 1 means bad here...
                    np.putmask(_weight_mask, np.less(_weight_arr,_mean), 1)
                    #_weight_mask.info()
                    self.weight_mask_list.append(_weight_mask)
                    listIndex += 1

            # Do MINMED
            if ( comb_type.lower() == "minmed"):
                # Issue a warning if minmed is being run with newmasks turned off.
                if (self.weight_mask_list == None):
                    print('\nWARNING: Creating median image without the application of bad pixel masks!\n')


                # Create the combined array object using the minmed algorithm
                result = minmed(imageSectionsList[startDrz:endDrz],  # list of input data to be combined.
                                    imageSectionsList[startWht:endWht],# list of input data weight images to be combined.
                                    readnoiseList,                         # list of readnoise values to use for the input images.
                                    exposureTimeList,                      # list of exposure times to use for the input images.
                                    backgroundValueList,                   # list of image background values to use for the input images
                                    weightMaskList = self.weight_mask_list,  # list of imput data weight masks to use for pixel rejection.
                                    combine_grow = grow,                   # Radius (pixels) for neighbor rejection
                                    combine_nsigma1 = nsigma1,             # Significance for accepting minimum instead of median
                                    combine_nsigma2 = nsigma2              # Significance for accepting minimum instead of median
                                )
  #              medianOutput[prange[0]:prange[1],:] = result.out_file1
  #             minOutput[prange[0]:prange[1],:] = result.out_file2

            # DO NUMCOMBINE
            else:
                # Create the combined array object using the numcombine task
                result = numCombine(imageSectionsList[startDrz:endDrz],
                                        numarrayMaskList=self.weight_mask_list,
                                        combinationType=comb_type.lower(),
                                        nlow=nlow,
                                        nhigh=nhigh,
                                        upper=hthresh,
                                        lower=lthresh
                                    )
                                    
            # We need to account for any specified overlap when writing out
            # the processed image sections to the final output array.
            if prange[1] != _imgrows:
                medianOutputImage[prange[0]:prange[1]-_overlap,:] = result.combArrObj[:-_overlap,:]
            else:
                medianOutputImage[prange[0]:prange[1],:] = result.combArrObj
            
            del result
            del self.weight_mask_list
            self.weight_mask_list = None

        # Write out the combined image
        self._writeCombinedImage(medianOutputImage, self.medianfile)

        #finally:
            # Always close any files opened to produce median image; namely,
            # single drizzle images and singly-drizzled weight images
            #
        self._closeMedianInput()
        del masterList
        del medianOutputImage
#        del medianOutput,minOutput

    def _writeCombinedImage(self, array, filename):
        """ Writes out the result of the combination step.
            The header of the first 'outsingle' file in the
            association parlist is used as the header of the
            new image.
        """

        _fname = self.assoc.parlist[0]['outsingle']
        _file = pyfits.open(_fname, mode='readonly')
        _prihdu = pyfits.PrimaryHDU(header=_file[0].header,data=array)

        _pf = pyfits.HDUList()
        _pf.append(_prihdu)
        _pf.writeto(filename)

        _file.close()
        del _pf


    def _closeMedianInput(self):
        """ Close mask files created from singly-drizzled weight images."""

        # Close all singly drizzled images used to create median image.
        for img in self.single_handles:
            img.close()
        self.single_list = []

        # Close all singly drizzled weight images used to create median image.
        for img in self.weight_handles:
            img.close()
        self.weight_list = []

        # If new median masks was turned on, close those files
        if self.weight_mask_list:
            for arr in self.weight_mask_list:
                del arr
            self.weight_mask_list = None


    def doBlot(self,blotpars):
        """ Blot back combined image into input image pixels. """

        origexptime = {}

        for p in self.assoc.parlist:
            fileutil.removeFile(p['outblot'])
            p['orig_single'] = p['outsingle']
            p['outsingle'] = self.medianfile


            print("\nblot data='"+p['outsingle']+"' outdata='"+p['outblot']+"' scale="+str(p['scale'])+
                " coeffs='"+p['coeffs']+"' xsh="+str(p['xsh'])+" ysh="+str(p['ysh'])+
                " rot="+str(p['rot'])+" outnx="+str(p['blotnx'])+" outny="+str(p['blotny'])+
                " align='center' shft_un='input' shft_fr='input' in_un='"+p['in_units']+"' out_un='"+p['units']+
                " interpol='"+blotpars['interp']+" sinscl='"+str(blotpars['sinscl'])+
                "' expout="+str(p['exptime'])+" expkey='"+"EXPTIME"+"' fillval=0.0\n")

        self.assoc.run(blot=True,save=True,interp=blotpars['interp'],
                        sinscl=blotpars['sinscl'])

        # Restore original outsingle filenames to parlist
        # so that PyDrizzle can remove them as necessary
        for p in self.assoc.parlist:
            p['outsingle'] = p['orig_single']

    def doDrizCR(self, drizcrpars, skypars):
        """ Runs deriv and driz_cr to create cosmic-ray masks. """
        
        # Print out the parameters provided by the interface
        print "USER PARAMETERS:"
        print "driz_cr         =  True"
        print "driz_cr_corr    = ",drizcrpars['driz_cr_corr']
        print "driz_cr_snr     = ",drizcrpars['driz_cr_snr']
        print "driz_cr_scale   = ",drizcrpars['driz_cr_scale']
        print "driz_cr_grow    = ",drizcrpars['driz_cr_grow']      
        print "driz_cr_ctegrow = ",drizcrpars['driz_cr_ctegrow']   

        print "\n"
        
        for p in self.assoc.parlist:
            # If cor_file is desired, then build name for file
            if drizcrpars['driz_cr_corr']:
                # Build Name for cor file
                _corr_file= p['rootname'] + '_sci' + str(p['group']) + '_cor.fits'
                # Build Name for cr file
                _cr_file = p['rootname'] + '_sci' + str(p['group']) + '_crmask.fits'

                # If corr_file and cr_file already exists, delete the old one so it can
                # be replaced cleanly with the new one...
                if fileutil.findFile(_corr_file):
                    fileutil.removeFile(_corr_file)
                if fileutil.findFile(_cr_file):
                    fileutil.removeFile(_cr_file)
            else:
                _corr_file = None
                _cr_file = None

            blot_handle = fileutil.openImage(p['outblot'], memmap=0, mode='readonly')
            mask_handle = fileutil.openImage(p['image'].maskname, mode='update') #, memmap=0)

            p['image'].runDrizCR(blot_handle[0].data, mask_handle[0].data,
                                drizcrpars, skypars, _corr_file, _cr_file)

            # Close outblot file now that we are done with it...
            blot_handle.close()
            mask_handle.close()
            del mask_handle, blot_handle

    def doFinalDriz(self, drizpars):
        """ Performs the final drizzle step. """

        if drizpars['outnx'] != None or drizpars['outny'] != None:
            _final_shape = (drizpars['outnx'],drizpars['outny'])
        else:
            _final_shape = None
        _new_field = None

        # Make sure parameters are set to original values
        self.assoc.resetPars()   

        if drizpars['refimage'] != '' and drizpars['refimage'] != None:
            # Use the following if the refimage isn't actually going to be
            # drizzled, we just want to set up the pydrizzle object
            #
            _refimg = wcsutil.WCSObject(drizpars['refimage'])
            refimg_wcs = _refimg.copy()

            # If the user also specified a rotation to be applied,
            # apply that as well...
            if drizpars['rot']:
                _orient = drizpars['rot']
            else:
                _orient = refimg_wcs.orientat

            # Now, build output WCS using the SkyField class
            # and default product's WCS as the initial starting point. 
            _new_field = pydrizzle.SkyField(wcs=refimg_wcs)

            # Update with user specified scale and rotation
            _new_field.set(psize=drizpars['scale'],orient=_orient)
        
        elif _final_shape != None or \
            drizpars['scale'] != None or \
            drizpars['rot'] != None or\
            drizpars['ra'] != None:

            _new_field = pydrizzle.SkyField(shape=_final_shape)

            _new_field.set(psize=drizpars['scale'], orient=drizpars['rot'],
                            ra=drizpars['ra'], dec=drizpars['dec'])

        if _new_field != None:
            # Before resetting the parameters, make a copy of the 'image' parameters
            # in the parlist
            _plist_orig = []
            for p in self.assoc.parlist:
                _plist_orig.append(p['image'])

        # Now, reset parameters to final values
        # We always want to insure that pixfrac and kernel are reset
        self.assoc.resetPars(field=_new_field, 
                    pixfrac=drizpars['pixfrac'], 
                    kernel=drizpars['kernel'], units=drizpars['units'] ) 
                    
        if _new_field != None:
            # Restore the 'image' parameters to the newly updated parlist
            for _nimg in xrange(len(self.assoc.parlist)):
                self.assoc.parlist[_nimg]['image'] = _plist_orig[_nimg]

        for p in self.assoc.parlist:
            # These should be taken care of with .resetPars
            #if outnx != None: p['outnx'] = final_outnx
            #if outny != None: p['outny'] = final_outny
            #p['kernel']  = drizpars['kernel']
            #p['pixfrac'] = drizpars['pixfrac']
            p['fillval'] = drizpars['fillval']
            p['wt_scl'] = drizpars['wt_scl']
            if not self.context:
                p['outcontext'] = ''
            p['in_units'] = 'counts'


        print("drizzle.outnx = "+str(self.assoc.parlist[0]['outnx']))
        print("drizzle.outny = "+str(self.assoc.parlist[0]['outny']))
        print("drizzle.scale = "+str(self.assoc.parlist[0]['scale']))
        print("drizzle.pixfrac = "+str(self.assoc.parlist[0]['pixfrac']))
        print("drizzle.shft_fr = 'output'")
        print("drizzle.shft_un = 'output'")
        print("drizzle.in_un = '"+str(self.assoc.parlist[0]['in_units'])+"'")
        print("drizzle.out_un = '"+self.assoc.parlist[0]['units']+"'")
        print("drizzle.align = 'center'")
        print("drizzle.expkey = 'EXPTIME'")
        print("drizzle.fillval = "+str(self.assoc.parlist[0]['fillval']))
        print("drizzle.outcont = '"+self.assoc.parlist[0]['outcontext']+"'")
        print("drizzle.kernel = '"+self.assoc.parlist[0]['kernel']+"'")
        print("\n")
    
        runfile = drizpars['runfile']
        if runfile == '':
            _runextn = '_drz.fits'
            for extn in [_runextn,'_drc.fits']:
                if extn in self.output:
                    _runextn = extn
                    break
            runfile = self.output[:self.output.find(_runextn)]+'.run'
            drizpars['runfile'] = runfile
        
        runlog = open(runfile,'w')

        runlog.write("drizzle.outnx = "+str(self.assoc.parlist[0]['outnx'])+"\n")
        runlog.write("drizzle.outny = "+str(self.assoc.parlist[0]['outny'])+"\n")
        runlog.write("drizzle.scale = "+str(self.assoc.parlist[0]['scale'])+"\n")
        runlog.write("drizzle.pixfrac = "+str(self.assoc.parlist[0]['pixfrac'])+"\n")
        runlog.write("drizzle.shft_fr = 'output'\n")
        runlog.write("drizzle.shft_un = 'output'\n")
        runlog.write("drizzle.in_un = "+str(self.assoc.parlist[0]['in_units'])+"\n")
        runlog.write("drizzle.out_un = '"+self.assoc.parlist[0]['units']+"'\n")
        runlog.write("drizzle.align = 'center'\n")
        runlog.write("drizzle.expkey = 'EXPTIME'\n")
        runlog.write("drizzle.fillval = "+str(self.assoc.parlist[0]['fillval'])+"\n")
        runlog.write("drizzle.outcont = '"+self.assoc.parlist[0]['outcontext']+"'\n")
        runlog.write("drizzle.kernel = '"+self.assoc.parlist[0]['kernel']+"'\n")
        runlog.write("\n")

        for p in self.assoc.parlist:
            if p['image'].maskname:
                if drizpars['wht_type'] == 'IVM':
                    self._applyIVM(p)
                elif drizpars['wht_type'] == 'ERR':
                    self._applyERR(p)
                else:
                    pass
                p['driz_mask'] = p['image'].maskname

            xsh_str = "%.4f"  % p['xsh']
            ysh_str = "%.4f"  % p['ysh']
            rot_str = "%.5f"  % p['rot']

            print("\ndrizzle "+p['data']+" "+p['outdata']+
                  " in_mask="+p['driz_mask']+" outweig="+p['outweight']+
                  " xsh="+xsh_str+" ysh="+ysh_str+" rot="+rot_str+
                  " coeffs='"+p['coeffs']+"' wt_scl='"+str(p['wt_scl'])+"'"+
                  " xgeoim='"+p['xgeoim']+"' ygeoim='"+p['ygeoim']+"'\n")
            runlog.write("drizzle "+p['data']+" "+p['outdata']+
                         " in_mask="+p['driz_mask']+" outweig="+p['outweight']+
                         " xsh="+xsh_str+" ysh="+ysh_str+" rot="+rot_str+
                         " coeffs='"+p['coeffs']+"' wt_scl='"+str(p['wt_scl'])+"'"+
                         " xgeoim='"+p['xgeoim']+"' ygeoim='"+p['ygeoim']+"'\n")


        # Close the "runfile" log
        if runlog != None:
            runlog.close()

        self.assoc.run(save=True,build=drizpars['build'])
        
        self.updateMdrizskyHistory(drizpars['build'])
        
        if self.proc_unit.lower() == 'native':
            if self.assoc.parlist[0]['image'].native_units.lower().find('counts') != -1:
                _plist = self.assoc.parlist[0]
                if drizpars['build']: 
                    _output = _plist['output']
                    sci_ext = 'sci'
                else: 
                    _output = _plist['outdata']
                    sci_ext = 0
                    
                fhdu = pyfits.open(_output,mode='update')
                fhdu[sci_ext].data = fhdu[sci_ext].data / _plist['image'].getGain()                
                fhdu[sci_ext].header['photflam'] *= _plist['image'].getGain()
                fhdu[sci_ext].header['bunit'] = fhdu[sci_ext].header['bunit'].upper().replace("ELECTRONS","COUNTS")
                fhdu.close()

        
    def updateMdrizskyHistory(self,build):
        """ Update the output SCI image with HISTORY cards
            that document what MDRIZSKY value was applied to each
            input image.
        """
        _plist = self.assoc.parlist[0]
        if build == True: _sky_output = _plist['output']
        else: _sky_output = _plist['outdata']
        
        fhdu = pyfits.open(_sky_output,mode='update')
        prihdr = fhdu[0].header
        
        for sky in self._getMdrizskyValues():
            sky_str = sky[0]+' MDRIZSKY = '+str(sky[1])
            prihdr.add_history(sky_str)
            
        fhdu.close()
        del fhdu
        
    def updateMdrizVerHistory(self,build,versions):
        """ Update the output SCI image with HISTORY cards
            that document what version of MultiDrizzle was used.
        """
        _plist = self.assoc.parlist[0]
        if build == True: _output = _plist['output']
        else: _output = _plist['outdata']
        
        fhdu = pyfits.open(_output,mode='update')
        prihdr = fhdu[0].header
        
        ver_str = "MultiDrizzle product generated using: "
        prihdr.add_history(ver_str)
        
        for key in versions:
            if versions[key].find('\n') < 0:
                prihdr.add_history(key+versions[key])
            else:
                # This will accomodate multi-line comments
                _ver_str = versions[key].split('\n')
                prihdr.add_history(key)
                for val in _ver_str:
                    if val.strip() != '':
                        prihdr.add_history(val)
                     
        #ver_str = '    MultiDrizzle Version '+str(version)
        #prihdr.add_history(ver_str)
            
        fhdu.close()
        del fhdu

    def _getMdrizskyValues(self):
        """ Builds a list of MDRIZSKY values used for each unique
            input exposure, not chip.
        """
        mdict = {}
        mlist = []
        for member in self.assoc.parlist:
            fname = member['image'].datafile
            if fname not in mdict: 
                mlist.append((fname, member['image'].getSubtractedSky()))
                mdict[fname] = 1
                
        return mlist

                 
    def _applyIVM(self,parlistentry):
        """
        
        Purpose
        =======
        Private method of the ImageMagager class that
        is used to apply IVM files.  
        
        Algorithm
        =========
        If the user provides IVM files they are applied
        to the final masks.  Otherwise they are automatically 
        generated.
        
        Automatically generated IVM files are specific to the 
        instrument being used and the details of the detector 
        used to acquire the image.

        
        """

        if parlistentry['ivmname'] != None:
            print "Applying user supplied IVM files..."

            #Parse the input file name to get the extension we are working on
            sciextn = parlistentry['image'].extn
            index = sciextn.find(',')   
            extn = "IVM,"+sciextn[index+1:]
            
            #Open the mask image for updating and the IVM image
            mask = fileutil.openImage(parlistentry['image'].maskname,mode='update')
            ivm =  fileutil.openImage(parlistentry['ivmname'],mode='readonly')

            ivmfile = fileutil.getExtn(ivm,extn)
            
            # Multiply the IVM file by the input mask in place.        
            mask[0].data = ivmfile.data * mask[0].data

            mask.close()
            ivm.close()

            # Update 'wt_scl' parameter to match use of IVM file
            parlistentry['wt_scl'] = pow(parlistentry['exptime'],2)/pow(parlistentry['scale'],4)
                        
        else:
                        
            imageobj = parlistentry['image']
            
            print "Automatically creating IVM files..."
            # If no IVM files were provided by the user we will 
            # need to automatically generate them based upon 
            # instrument specific information.
            
            flat = imageobj.getflat()
            RN = imageobj.getReadNoiseImage()
            darkimg = imageobj.getdarkimg()
            skyimg = imageobj.getskyimg()
            
            ivm = (flat)**2/(darkimg+(skyimg*flat)+RN**2)

            #Open the mask image for updating
            mask = fileutil.openImage(parlistentry['image'].maskname,mode='update')
            
            # Multiply the IVM file by the input mask in place.        
            mask[0].data = ivm * mask[0].data
            mask.close()
            
            # Update 'wt_scl' parameter to match use of IVM file
            parlistentry['wt_scl'] = pow(parlistentry['exptime'],2)/pow(parlistentry['scale'],4)


    def _applyERR(self,parlistentry):

        """
        _applyERR(self,parlistentry):  A method of the ImageManager class that applies
        the ERROR image as an IVM like file to the input image.
        
        """
        # Not all input images will have an 'ERR' extension.  We must be prepared for the
        # failure of the open of the err array.  

        # Parse the input file name to get the extension we are working on
        sciextn = parlistentry['image'].extn
        index = sciextn.find(',')   
        extn = "ERR,"+sciextn[index+1:]
        fname,fextn = fileutil.parseFilename(parlistentry['data'])

        #Open the mask image for updating
        mask = fileutil.openImage(parlistentry['image'].maskname,mode='update')
        try:
            # Attempt to open the ERR image.
            err =  fileutil.openImage(fname,mode='readonly')

            print "Applying ERR file ",fname+'['+extn+']'," to mask file ",parlistentry['image'].maskname
            errfile = fileutil.getExtn(err,extn)

            # Multiply the scaled ERR file by the input mask in place.        
            mask[0].data = 1/(errfile.data)**2 * mask[0].data

            mask.close()
            err.close()

            # Update 'wt_scl' parameter to match use of IVM file
            parlistentry['wt_scl'] = pow(parlistentry['exptime'],2)/pow(parlistentry['scale'],4)

        except:
            if (parlistentry['image'].instrument.find('WFPC2') > -1 ):
            # If we were unable to find an 'ERR' extension to apply, one possible reason was that
            # the input was a 'standard' WFPC2 data file that does not actually contain an error array.
            # Test for this condition and issue a Warning to the user and continue on to the final
            # drizzle.   
                errstr =  "*******************************************\n"
                errstr += "*                                         *\n"
                errstr += "* WARNING: No ERR weighting will be       *\n"
                errstr += "* applied to the mask used in the final   *\n"
                errstr += "* drizzle step!  Weighting will be only   *\n"
                errstr += "* by exposure time.                       *\n"
                errstr += "*                                         *\n"
                errstr += "* The WFPC2 data provided as input does   *\n"
                errstr += "* not contain ERR arrays.  WFPC2 data is  *\n"
                errstr += "* not supported by this weighting type.   *\n"
                errstr += "*                                         *\n"
                errstr += "* A workaround would be to create inverse *\n"
                errstr += "* variance maps and use 'IVM' as the      *\n"
                errstr += "* final_wht_type.  See the HELP file for  *\n"
                errstr += "* more details on using inverse variance  *\n"
                errstr += "* maps.                                   *\n" 
                errstr += "*                                         *\n"
                errstr =  "*******************************************\n"
                print errstr
                print "\n Continue with final drizzle step..."
            else:
            # We cannot find an 'ERR' extension and the data isn't WFPC2.  Print a generic warning message
            # and continue on with the final drizzle step.
                generrstr =  "*******************************************\n"
                generrstr += "*                                         *\n"
                generrstr += "* WARNING: No ERR weighting will be       *\n"
                generrstr += "* applied to the mask used in the final   *\n"
                generrstr += "* drizzle step!  Weighting will be only   *\n"
                generrstr += "* by exposure time.                       *\n"
                generrstr += "*                                         *\n"
                generrstr += "* The data provided as input does not     *\n"
                generrstr += "* contain an ERR extension.               *\n"
                generrstr += "*                                         *\n"
                generrstr =  "*******************************************\n"
                print generrstr
                print "\n Continue with final drizzle step..."
            # Ensure that the mask file has been closed.
            mask.close()
            
