#
#   Authors: Warren Hack, Ivo Busko, Christopher Hanley
#   Program: manager.py
#   Purpose: Class Used to drive all algorithmic operations within Multidrizzle.
#   History:
#           Version 0.1.23, 04/01/04 -- Bug fixes -- CJH
#           Version 0.1.30, 05/10/04 -- Read MDRIZTAB -- IB
#
#           Version 0.1.30, 05/11/04 -- Fix logic when not applying static mask. -- WJH/CJH
#           Version 0.1.31, 05/12/04 -- Implemented new sky subtraction algorithm -- CJH
#           Version 0.1.32, 05/15/04 -- Bug fixes for working on copies -- CJH/WJH
#           Version 0.1.33, 05/17/04 -- Moved MDRIZTAB code to its own module -- IB
#           Version 0.1.34, 05/20/04 -- Added support for creating a cr mask file.  -- CJH
#           Version 0.1.35, 06/01/04 -- In create median step, the "lthresh" and "hthresh" parameters
#               needed to be recast from type string to type float.  -- CJH
#           Version 0.1.36, 06/23/04 -- Added support for working on input files instead of 
#               making copies of the inputs.
#           Version 0.1.37, 06/28/04 -- Modified some print statements in median step.  -- CJH
#           Version 0.1.38, 06/29/04 -- Modified import of imagestats and minmed. -- CJH
#           Version 0.1.39, 06/29/04 -- Modified imports to remove dependence on pytools package -- CJH
#           Version 0.1.40, 07/08/04 -- Updated Dictionary key names -- CJH
#           Version 0.1.41, 07/16/04 -- Modified the _getInputImage method to support WFPC2 data -- CJH
#           Version 0.1.42, 07/20/04 -- Added support for Stis Images -- CJH
#           Version 0.1.43, 07/29/04 -- Modified call to InputImage class to pass the plate scale.  The 
#               create median step now gets its' sky value with the getreferencesky method.  -- CJH

# Import Numarray functionality
import numarray.image.combine as combine
import numarray as N

# Import file utilities
import pyfits
import shutil, os

# Import Pydrizzle
import pydrizzle
from pydrizzle import fileutil,drutil,buildmask

# Import support for specific HST Instruments
from acs_input import WFCInputImage, HRCInputImage, SBCInputImage
from wfpc2_input import WFPC2InputImage, PCInputImage, WF2InputImage, WF3InputImage, WF4InputImage
from stis_input import CCDInputImage

# Import general tools
import imagestats
from imagestats import ImageStats
import numcombine
from numcombine import numCombine
import minmed
from minmed import minmed
import static_mask
from static_mask import StaticMask
import nimageiter
from nimageiter import ImageIter

__version__ = '0.1.43'

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

class ImageManager:
    """ The ImageManager class handles most of the file I/O operations for the input
        images being processed.  Multiple InputImage objects can point to the
        same ImageManager object without worrying about opening or trying to close
        the same image more than once.
    """
    def __init__(self, assoc, context, instrpars, workinplace):
        self.context = context
        self.assoc = assoc
        self.output = self.assoc.output
        self.workinplace = workinplace

        # Establish a default memory mapping behavior
        self.memmap = 1

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
        _indx = self.output.find('_drz.fits')
        if _indx < 0:
            _indx = len(self.output)
        self.medianfile = self.output[:_indx] + '_med.fits'

        for p in assoc.parlist:

            # Make copies of the input science images
            # Need to simply call 'removeINPUTcopies' to delete these files
            self.setupInputCopies(p,self.workinplace)

            # Setup InputImage objects for this association
            p['image'] = self._getInputImage(p['data'],p)

            # Setup Multidrizzle-specific arameters
            p['datafile'] = p['image'].datafile
            grp = str(p['image'].grp)
            rootname = p['image'].rootname

            # Setup file for mask array,
            #    creating one if it doesn't already exist.
            if not p['driz_mask']:
                # No array has been created yet, so initialize it now...
                _mask_array = N.ones(p['image'].image_shape,N.UInt8)
                # We also need to create a name for the mask array file
                p['image'].maskname = buildmask.buildMaskName(p['image'].rootname,p['image'].grp)

                self._buildMaskImage(p['image'].maskname,_mask_array)
                del _mask_array
            else:
                p['image'].maskname = p['driz_mask']

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
            p['group'] = grp
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

    def setInstrumentParameters(self, instrpars):
        """ Sets intrument parameters into all image instances.
        """

        # A reference to the primary header is also required.

        for p in self.assoc.parlist:
            p['image'].setInstrumentParameters (instrpars, p['exposure'].header)

    def doUnitConversions(self):
        for p in self.assoc.parlist:
            p['image'].doUnitConversions()

    # This is called upon initialization of this class...
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

        _instrument = plist['exposure'].header['INSTRUME']
        _detector = plist['exposure'].header['DETECTOR']
        _platescale = plist['exposure'].pscale
        
        # Extract the dq array designation
        _dqname = plist['exposure'].dqname
        _dq_root,_dq_extn = fileutil.parseFilename(_dqname)
        _dqname = plist['orig_filename']+'['+_dq_extn+']'
#        print "DQ name being build: ",_dqname

        if _instrument == 'ACS':
            if _detector == 'HRC': return HRCInputImage(input,_dqname,_platescale,memmap=0)
            if _detector == 'WFC': return WFCInputImage(input,_dqname,_platescale,memmap=0)
            if _detector == 'SBC': return SBCInputImage(input,_dqname,_platescale,memmap=0)
        if _instrument == 'WFPC2':
            if _detector == 1: return PCInputImage(input,_dqname,_platescale,memmap=0)
            if _detector == 2: return WF2InputImage(input,_dqname,_platescale,memmap=0)
            if _detector == 3: return WF3InputImage(input,_dqname,_platescale,memmap=0)
            if _detector == 4: return WF4InputImage(input,_dqname,_platescale,memmap=0)
        if _instrument == 'STIS': 
            if _detector == 'CCD': return CCDInputImage(input,_dqname_platescale,memmap=0)

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


    def createStatic(self, static_file, static_sig):

        """ Create the static bad-pixel mask from input images."""
        
        #Print paramater values supplied through the interface
        print "USER PARAMETERS:"
        print "static     =  True"
        print "staticfile = ",static_file
        print "static_sig = ",static_sig
        print "\n"
                 
        self.static_mask = StaticMask(goodval = 1, badval = 0, staticsig=static_sig)

        for p in self.assoc.parlist:
            p['image'].updateStaticMask(self.static_mask)

        if (static_file != None):
            self._applyUserStaticFile (static_file, self.static_mask)

        # For each input, we now need to update the driz_mask with the
        # values from the static mask
        # Combine in_mask array with static mask.
        # The mask_array should always be present, created by
        # ImageManager if PyDrizzle does not create one.
        for p in self.assoc.parlist:
            __handle = fileutil.openImage(p['image'].maskname,mode='update')
            __static_array = self.static_mask.getMask(p['image'].signature())

            # Print progress message to alert the user that the mask file
            # is now being updated with the static mask information
            # If verbose mode is implemented, this could be included.
            print 'Updating mask file: ',p['image'].maskname,' with static mask.'

            if __static_array != None:
                __handle[0].data = N.bitwise_and(__handle[0].data,__static_array)

            __handle.close()
            del __handle


    def _applyUserStaticFile(self, static_file, static_mask):

        try:
            file = fileutil.openImage(static_file, memmap=1, mode='readonly')
            input_static_mask = file[0].data

            static_mask.applyStaticFile(input_static_mask, p['image'].signature())

        finally:
            file.close()

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
                p['image'].setSubtractedSky(_imageMinDict[p['rootname']])
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
            _refimg = pydrizzle.wcsutil.WCSObject(pars['refimage'])
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
             pars['ra']    != None:

            _sky_field = pydrizzle.SkyField()

            if pars['rot'] == None:
                _orient = self.assoc.observation.product.geometry.wcslin.orient
            else:
                _orient = pars['rot']

            print 'Default orientation for output: ',_orient,'degrees'

            _sky_field.set(psize=pars['scale'], orient=_orient,
                           ra=pars['ra'], dec=pars['dec'])

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
        self._setOutputFrame(pars)

        for p in self.assoc.parlist:

            # First do some cleaning up, in case you are restarting...
            fileutil.removeFile(p['outsingle'])
            if (p['outsweight'] != ''):
                fileutil.removeFile(p['outsweight'])

            # NB DO NOT USE "tophat" unless pixfrac is sufficiently
            # large (> sqrt(2))
            
            p['fillval'] = pars['fillval']
            # Copy out filename for 'driz_mask' and
            # replace with static_mask Numarray object
            p['full_mask'] = p['driz_mask']
            if (self.static_mask != None):
                p['driz_mask'] = self.static_mask.getMask(p['image'].signature())
            else:
                p['driz_mask'] = None

            print("\ndrizzle data='"+p['data']+"' outdata='"+p['outsingle']+"' outweig='"+p['outsweight']+
                "' outcont='"+p['outscontext']+"' in_mask='static_mask"+"' kernel='"+p['kernel']+
                "' outnx="+str(p['outnx'])+" outny="+str(p['outny'])+" xsh="+str(p['xsh'])+" ysh="+str(p['ysh'])+
                " scale="+str(p['scale'])+" pixfrac="+str(p['pixfrac'])+" rot="+str(p['rot'])+
                " coeffs='"+p['coeffs']+"' wt_scl='"+p['wt_scl']+"' align='center' shft_fr='output' shft_un='output'"+
                " out_un='"+p['units']+"' expkey='"+"EXPTIME"+"' fillval='"+str(p['fillval'])+"'\n")

        # Perform separate drizzling now that all parameters have been setup...
        try:
            self.assoc.run(single=True,save=True,build=False)
        except:
            print 'Could not complete (drizSeparate) processing.'
            raise RuntimeError

        # Restore reference to mask file
        for p in self.assoc.parlist:
            p['driz_mask'] = p['full_mask']

        # Now that we are done with the static mask, delete it...
        del self.static_mask


    def createMedian(self, medianpars):
    
        # Print out the parameters provided by the interface
        print "USER PARAMETERS:"
        print "median          =  True"
        print "median_newmaks  = ",medianpars['newmasks']
        print "combine_type    = ",medianpars['type']  
        print "combine_nsigma  = ",medianpars['nsigma1']," ",medianpars['nsigma2']
        print "combine_nlow    = ",medianpars['nlow']
        print "combine_nhigh   = ",medianpars['nhigh']
        print "combine_lthresh = ",medianpars['lthresh']
        print "combine_hthresh = ",medianpars['hthresh']
        print "combine_grow    = ",medianpars['grow']
        print "\n"
                    
        __newmasks = medianpars['newmasks']
        __type = medianpars['type']
        __nsigma1 = medianpars['nsigma1']
        __nsigma2 = medianpars['nsigma2']
        __nlow = medianpars['nlow']
        __nhigh = medianpars['nhigh']
        if (medianpars['lthresh'] == None):
            __lthresh = None
        else:
            __lthresh = float(medianpars['lthresh'])
        if (medianpars['hthresh'] == None):
            __hthresh = None
        else:
            __hthresh = float(medianpars['hthresh'])
        __grow = medianpars['grow']

        """ Builds combined array from single drizzled images."""
        # Start by removing any previous products...
        fileutil.removeFile(self.medianfile)
        #try:
        # Open all inputs necessary for creating the median image;
        # namely, all unique outsingle images created by PyDrizzle.

        # Compute the mean value of each wht image
        _wht_mean = []

        # Define lists for instrument specific parameters.
        __readnoiseList = []
        __exposureTimeList = []
        __backgroundValueList = []

        for p in self.assoc.parlist:
            # Extract the single drizzled image.
            if p['group'] == '1':
                #_file = fileutil.openImage(p['outsingle'], memmap=1, mode='readonly')
                _file = fileutil.openImage(p['outsingle'], mode='readonly')
                self.single_handles.append(_file)
                #self.single_list.append(_file[0].data)

                # If it exists, extract the corresponding weight images
                if (fileutil.findFile(p['outsweight'])):
                    #_weight_file = fileutil.openImage(p['outsweight'], memmap=1, mode='readonly')
                    _weight_file = fileutil.openImage(p['outsweight'], mode='readonly')
                    #print 'Outsweight filename: ',p['outsweight']
                    self.weight_handles.append(_weight_file)
                    #self.weight_list.append(_weight_file[0].data)
                    _wht_mean.append(_weight_file[0].data.mean() * 0.7)
                    # Clear the memory used by reading in the whole data array for
                    # computing the mean.  This requires that subsequent access to
                    # the data values happens through the 'section' attribute of the HDU.
                    del _weight_file[0].data

                # Extract instrument specific parameters and place in lists
                __readnoiseList.append(p['image'].getReadNoise())
                __exposureTimeList.append(p['image'].getExpTime())
#                __backgroundValueList.append(p['image'].getSubtractedSky())
#                print "subtracted sky value for image ",p['image'].rootname," is ", p['image'].getSubtractedSky()
                __backgroundValueList.append(p['image'].getreferencesky())
                print "reference sky value for image ",p['image'].rootname," is ", p['image'].getreferencesky()


        # create an array for the median output image
        __medianOutputImage = N.zeros(self.single_handles[0][0].data.shape,self.single_handles[0][0].data.type())
        # create the master list to be used by the image iterator
        __masterList = []
        #__masterList.append(__medianOutputImage)
        for _element in self.single_handles:
            __masterList.append(_element[0])
        for _element in self.weight_handles:
            __masterList.append(_element[0])

        print '\n'

        # Specify the location of the medianOutputImage section in the masterList
        #__medianOutputImageSection = 0

        # Specify the location of the drz image sections
        __startDrz = 0
        __endDrz = len(self.single_handles)+__startDrz

        # Specify the location of the wht image sections
        __startWht = len(self.single_handles)+__startDrz
        __endWht = __startWht + len(self.weight_handles)

        # We only want to print this out once...
#        if __newmasks:
#            print('\nCreating pixel mask files for the median step...\n')

  #      __medianOutput = N.zeros(self.single_handles[0][0].data.shape,self.single_handles[0][0].data.type())
  #      __minOutput = N.zeros(self.single_handles[0][0].data.shape,self.single_handles[0][0].data.type())

        # Fire up the image iterator
        for __imageSectionsList,__prange in ImageIter(__masterList):
            #print 'processing rows in range: ',__prange

            # For section syntax, it returns a numarray object of the
            # exact values pulled from the file, so we need to explicitly
            # perform byteswapping to account for FITS convention little-endian
            # platforms (such as Intel/Linux).
            #for img in __imageSectionsList:
            #    img.byteswap()

            if __newmasks:
                """ Build new masks from single drizzled images. """
                self.weight_mask_list = []
                listIndex = 0
                for _weight_arr in __imageSectionsList[__startWht:__endWht]:
                    # Initialize an output mask array to ones
                    # This array will be reused for every output weight image
                    _weight_mask = N.zeros(_weight_arr.shape,N.UInt8)

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
                    N.putmask(_weight_mask, N.less(_weight_arr,_mean), 1)
                    #_weight_mask.info()
                    self.weight_mask_list.append(_weight_mask)
                    listIndex += 1

            # Do MINMED
            if ( __type.lower() == "minmed"):
                # Issue a warning if minmed is being run with __newmasks turned off.
                if (self.weight_mask_list == None):
                    print('\nWARNING: Creating median image without the application of bad pixel masks!\n')


                # Create the combined array object using the minmed algorithm
                __result = minmed(__imageSectionsList[__startDrz:__endDrz],  # list of input data to be combined.
                                    __imageSectionsList[__startWht:__endWht],# list of input data weight images to be combined.
                                    __readnoiseList,                         # list of readnoise values to use for the input images.
                                    __exposureTimeList,                      # list of exposure times to use for the input images.
                                    __backgroundValueList,                   # list of image background values to use for the input images
                                    weightMaskList = self.weight_mask_list,  # list of imput data weight masks to use for pixel rejection.
                                    combine_grow = __grow,                   # Radius (pixels) for neighbor rejection
                                    combine_nsigma1 = __nsigma1,             # Significance for accepting minimum instead of median
                                    combine_nsigma2 = __nsigma2              # Significance for accepting minimum instead of median
                                )
  #              __medianOutput[__prange[0]:__prange[1],:] = __result.out_file1
  #             __minOutput[__prange[0]:__prange[1],:] = __result.out_file2

            # DO NUMCOMBINE
            else:
                # Create the combined array object using the numcombine task
                #print "length of imageSectionsList = ", len(__imageSectionsList)
                #print "length of weight_mask_list = ", len(self.weight_mask_list)
                #print "startdrz,endriz = " , __startDrz,",",__endDrz
                #print __imageSectionsList[__endDrz].info()
                #print __imageSectionsList[__endDrz]
                #print self.weight_mask_list[1]
                __result = numCombine(__imageSectionsList[__startDrz:__endDrz],
                                        numarrayMaskList=self.weight_mask_list,
                                        combinationType=__type.lower(),
                                        nlow=__nlow,
                                        nhigh=__nhigh,
                                        upper=__hthresh,
                                        lower=__lthresh
                                    )
            __medianOutputImage[__prange[0]:__prange[1],:] = __result.combArrObj
            del __result


            del self.weight_mask_list
            self.weight_mask_list = None

        # Write out the combined image
        #self._writeCombinedImage(__masterList[__medianOutputImageSection], self.medianfile)
        self._writeCombinedImage(__medianOutputImage, self.medianfile)
 #       self._writeCombinedImage(__medianOutput,'test_median_file.fits')
 #       self._writeCombinedImage(__minOutput,'test_minimum_file.fits')

        #finally:
            # Always close any files opened to produce median image; namely,
            # single drizzle images and singly-drizzled weight images
            #
        self._closeMedianInput()
        del __masterList
        del __medianOutputImage
#        del __medianOutput,__minOutput

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

        for p in self.assoc.parlist:
            fileutil.removeFile(p['outblot'])
            p['orig_single'] = p['outsingle']
            p['outsingle'] = self.medianfile

            print("\nblot data='"+p['outsingle']+"' outdata='"+p['outblot']+"' scale="+str(p['scale'])+
                " coeffs='"+p['coeffs']+"' xsh="+str(p['xsh'])+" ysh="+str(p['ysh'])+
                " rot="+str(p['rot'])+" outnx="+str(p['blotnx'])+" outny="+str(p['blotny'])+
                " align='center' shft_un='input' shft_fr='input' in_un='"+p['units']+"' out_un='counts'"+
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
        print "driz_cr       =  True"
        print "driz_cr_corr  = ",drizcrpars['driz_cr_corr']
        print "driz_cr_snr   = ",drizcrpars['driz_cr_snr']
        print "driz_cr_scale = ",drizcrpars['driz_cr_scale']
        print "\n"
        
        try:
            for p in self.assoc.parlist:
                try:
                    # If cor_file is desired, then build name for file
                    if drizcrpars['driz_cr_corr']:
                        # Build Name for cor file
                        _corr_file= p['rootname'] + '_sci' + p['group'] + '_cor.fits'
                        # Build Name for cr file
                        _cr_file = p['rootname'] + '_sci' + p['group'] + '_crmask.fits'

                        # If corr_file and cr_file already exists, delete the old one so it can
                        # be replaced cleanly with the new one...
                        if fileutil.findFile(_corr_file):
                            fileutil.removeFile(_corr_file)
                        if fileutil.findFile(_cr_file):
                            fileutil.removeFile(_cr_file)
                    else:
                        _corr_file = None
                        _cr_file = None

                    __blot_handle = fileutil.openImage(p['outblot'], memmap=1, mode='readonly')
                    __mask_handle = fileutil.openImage(p['image'].maskname, mode='update') #, memmap=1)

                    p['image'].runDrizCR(__blot_handle[0].data, __mask_handle[0].data,
                                        drizcrpars, skypars, _corr_file, _cr_file)

                finally:
                    # Close outblot file now that we are done with it...
                    __blot_handle.close()
                    __mask_handle.close()
                    del __mask_handle, __blot_handle
        except:
            print 'Could not complete (doDrizCR) processing.'
            raise RuntimeError

    def doFinalDriz(self, drizpars, runfile):
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
            _refimg = pydrizzle.wcsutil.WCSObject(drizpars['refimage'])
            refimg_wcs = _refimg.copy()

            # If the user also specified a rotation to be applied,
            # apply that as well...
            if drizpars['rot']:
                _orient = drizpars['rot']
            else:
                _orient = refimg_wcs.orientat

           # Now, build output WCS using the SkyField class
            # and default product's WCS as the initial starting point.
            #
            _new_field = pydrizzle.SkyField(wcs=refimg_wcs)
            # Update with user specified scale and rotation
            _new_field.set(psize=drizpars['scale'],orient=_orient)
        
        elif _final_shape != None or \
            drizpars['scale'] != None or \
            drizpars['rot'] != None or\
            drizpars['ra'] != None:

            _new_field = pydrizzle.SkyField(shape=_final_shape)

            #_orientat = self.assoc.observation.product.geometry.wcslin.orient
            #_new_field.set(psize=scale, orient=_orientat + rot)
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
                    kernel=drizpars['kernel'])
                    
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

        print("drizzle.outnx = "+str(self.assoc.parlist[0]['outnx']))
        print("drizzle.outny = "+str(self.assoc.parlist[0]['outny']))
        print("drizzle.scale = "+str(self.assoc.parlist[0]['scale']))
        print("drizzle.pixfrac = "+str(self.assoc.parlist[0]['pixfrac']))
        print("drizzle.shft_fr = 'output'")
        print("drizzle.shft_un = 'output'")
        print("drizzle.in_un = 'counts'")
        print("drizzle.out_un = '"+self.assoc.parlist[0]['units']+"'")
        print("drizzle.align = 'center'")
        print("drizzle.expkey = 'EXPTIME'")
        print("drizzle.fillval = "+str(self.assoc.parlist[0]['fillval']))
        print("drizzle.outcont = '"+self.assoc.parlist[0]['outcontext']+"'")
        print("drizzle.kernel = '"+self.assoc.parlist[0]['kernel']+"'")
        print("\n")

        if runfile != '':
            runlog = open(runfile,'w')
        else:
            runlog = None

        runlog.write("drizzle.outnx = "+str(self.assoc.parlist[0]['outnx'])+"\n")
        runlog.write("drizzle.outny = "+str(self.assoc.parlist[0]['outny'])+"\n")
        runlog.write("drizzle.scale = "+str(self.assoc.parlist[0]['scale'])+"\n")
        runlog.write("drizzle.pixfrac = "+str(self.assoc.parlist[0]['pixfrac'])+"\n")
        runlog.write("drizzle.shft_fr = 'output'\n")
        runlog.write("drizzle.shft_un = 'output'\n")
        runlog.write("drizzle.in_un = 'counts'\n")
        runlog.write("drizzle.out_un = '"+self.assoc.parlist[0]['units']+"'\n")
        runlog.write("drizzle.align = 'center'\n")
        runlog.write("drizzle.expkey = 'EXPTIME'\n")
        runlog.write("drizzle.fillval = "+str(self.assoc.parlist[0]['fillval'])+"\n")
        runlog.write("drizzle.outcont = '"+self.assoc.parlist[0]['outcontext']+"'\n")
        runlog.write("drizzle.kernel = '"+self.assoc.parlist[0]['kernel']+"'\n")
        runlog.write("\n")

        for p in self.assoc.parlist:
            if p['image'].maskname:
                p['driz_mask'] = p['image'].maskname
            else:
                p['driz_mask'] = ''


            xsh_str = "%.4f"  % p['xsh']
            ysh_str = "%.4f"  % p['ysh']
            rot_str = "%.5f"  % p['rot']

            print("\ndrizzle "+p['data']+" "+p['outdata']+
                  " in_mask="+p['driz_mask']+" outweig="+p['outweight']+
                  " xsh="+xsh_str+" ysh="+ysh_str+" rot="+rot_str+
                  " coeffs='"+p['coeffs']+"' wt_scl='"+p['wt_scl']+"'\n")
            runlog.write("drizzle "+p['data']+" "+p['outdata']+
                         " in_mask="+p['driz_mask']+" outweig="+p['outweight']+
                         " xsh="+xsh_str+" ysh="+ysh_str+" rot="+rot_str+
                         " coeffs='"+p['coeffs']+"' wt_scl='"+p['wt_scl']+"'\n")

        # Close the "runfile" log
        if runlog != None:
            runlog.close()

        #try:
        self.assoc.run(save=True,build=drizpars['build'])
        #except:
        #    print 'Error during final drizzling.'
        #    raise RuntimeError
