from pytools import fileutil, wcsutil
import imtype, buildmask
from exposure import Exposure
import numpy as N
import drutil

#from pydrizzle import __version__
__version__ = '6.xxx'
from pytools.fileutil import RADTODEG, DEGTORAD
DEFAULT_PARITY = [[1.0,0.0],[0.0,1.0]]
yes = True
no = False

#################
#
#
#               Custom Parameter Dictionary Class
#
#
#################

class ParDict(dict):
    def __init__(self):
        dict.__init__(self)

    def __str__(self):
        if self['xsh'] == None: _xsh = repr(self['xsh'])
        else: _xsh = '%0.4f'%self['xsh']
        if self['ysh'] == None: _ysh = repr(self['ysh'])
        else: _ysh = '%0.4f'%self['ysh']

        # Insure that valid strings are available for
        # 'driz_mask' and 'single_driz_mask'
        if self['driz_mask'] == None: _driz_mask = ''
        else: _driz_mask = self['driz_mask']
        if self['single_driz_mask'] == None: _sdriz_mask = ''
        else: _sdriz_mask = self['single_driz_mask']


        _str = ''
        _str += 'Parameters for input chip: '+self['data']+'\n'
        _str += '  Shifts:       %s      %s\n'%(_xsh,_ysh)
        _str += '  Output size:  %d      %d \n'%(self['outnx'],self['outny'])
        _str += '  Rotation:     %0.4g deg.      Scale:  %0.4f \n'%(self['rot'],self['scale'])
        _str += '  pixfrac: %0.4f      Kernel: %s      Units: %s\n'%(self['pixfrac'],self['kernel'],self['units'])
        _str += '  ORIENTAT:   %0.4g deg.   Pixel Scale:  %0.4f arcsec/pix\n'%(self['orient'],self['outscl'])
        _str += '  Center at RA:  %0.9g      Dec:  %0.9f \n'%(self['racen'],self['deccen'])
        _str += '  Output product:  %s\n'%self['output']
        _str += '  Coeffs:          '+self['coeffs']+'    Geomode: '+self['geomode']+'\n'
        _str += '  XGeoImage       : '+self['xgeoim']+'\n'
        _str += '  YGeoImage       : '+self['ygeoim']+'\n'
        _str += '  Single mask file: '+_sdriz_mask+'\n'
        _str += '  Final mask file : '+_driz_mask+'\n'
        _str += '  Output science  : '+self['outdata']+'\n'
        _str += '  Output weight   : '+self['outweight']+'\n'
        _str += '  Output context  : '+self['outcontext']+'\n'
        _str += '  Exptime -- total: %0.4f     single: %0.4f\n'%(self['texptime'],self['exptime'])
        _str += '     start: %s         end: %s\n'%(repr(self['expstart']),repr(self['expend']))
        _str += '  Single image products--  output:  %s\n'%self['outsingle']
        _str += '     weight:  %s  \n '%self['outsweight']
        _str += '  Blot output: %s \n'%self['outblot']
        _str += '  Size of original image to blot: %d %d \n'%(self['blotnx'],self['blotny'])

        _str += '\n'

        return _str


class Pattern(object):
    """
     Set default values for these to be overridden by
     instrument specific class variables as necessary.
    """
    IDCKEY = 'IDCTAB'
    PARITY = {'detector':DEFAULT_PARITY}
    REFDATA = {'detector':[[1.,1.],[1.,1.]]}

    NUM_IMSET = 3               # Number of extensions in an IMSET
    PA_KEY = 'PA_V3'
    DETECTOR_NAME = 'detector'
    COPY_SUFFIX = '.orig'       # suffix to use for filename of copy

    def __init__(self, filename, output=None, pars=None):
        # Set these up for use...
        self.members = []
        self.pars = pars
        self.output = output
        self.name = filename

        # Set default value, to be reset as needed by sub-class
        self.nmembers = 1
        self.nimages = 1

        # Extract bit values to be used for this instrument
        self.bitvalue = self.pars['bits']

        # Set IDCKEY, if specified by user...
        if self.pars['idckey'] == '':
            self.idckey = self.IDCKEY
        else:
            self.idckey = self.pars['idckey']
        self.idcdir = self.pars['idcdir']

        # Keyword which provides PA_V3 value; ie., orientation of
        #       telescope at center axis relative to North.
        # Each instrument has their own pre-defined keyword; mostly, PA_V3
        self.pa_key = self.PA_KEY

        self.exptime = None

        self.binned = 1
        
        # These attributes are used for keeping track of the reference
        # image used for computing the shifts, and the shifts computed
        # for each observation, respectively.
        self.offsets = None
        self.v2com = None
        self.v3com = None

        # Read in Primary Header to reduce the overhead of getting
        # keyword values and setup PyFITS object
        # as self.header and self.image_handle
        #
        image_handle = self.getHeaderHandle()

        if self.pars['section'] != None:
            # If a section was specified, check the length of the list
            # for the number of groups specified...
            self.nmembers = len(self.pars['section'])

        # Determine type of input image and syntax needed to read data
        self.imtype = imtype.Imtype(filename,handle=self.image_handle,
                                    dqsuffix=self.pars['dqsuffix'])

        # Keep file I/O localized to same method/routine
        if image_handle:
            image_handle.close()

        if self.header and self.header.has_key(self.DETECTOR_NAME):
            self.detector = self.header[self.DETECTOR_NAME]
        else:
            self.detector = 'detector'

    def getHeaderHandle(self):
        """ Sets up the PyFITS image handle and Primary header
            as self.image_handle and self.header.

            When Pattern being used for output product, filename will be
            set to None and this returns None for header and image_handle.
        """

        _numsci = 0
        if self.name:
            _handle = fileutil.openImage(self.name,mode='readonly',memmap=self.pars['memmap'])
            _fname,_extn = fileutil.parseFilename(self.name)
            _hdr = _handle['PRIMARY'].header.copy()
            # Count number of SCI extensions
            for _fext in _handle:
                if _fext.header.has_key('extname') and _fext.header['extname'] == 'SCI':
                    _numsci += 1

            if _extn > 0:
                # Append correct extension/chip/group header to PRIMARY...
                for _card in fileutil.getExtn(_handle,_extn).header.ascardlist():
                    _hdr.ascard.append(_card)
        else:
            # Default to None
            _handle = None
            _hdr = None

        # Set attribute to point to these products
        self.image_handle = None
        self.header = _hdr
        self.nmembers = _numsci

        return _handle

    def closeHandle(self):
        """ Closes image_handle. """
        if self.image_handle:
            self.image_handle.close()
            #print 'Closing file handle for image: ',self.name
        self.image_handle = None


    def addMembers(self,filename):
        """ Build rootname for each SCI extension, and
            create the mask image from the DQ extension.
            It would then append a new Exposure object to 'members'
            list for each extension.
        """

        self.detector = detector = str(self.header[self.DETECTOR_NAME])

        if self.pars['section'] == None:
            self.pars['section'] = [None]*self.nmembers
        # Build rootname here for each SCI extension...
        for i in range(self.nmembers):
            _sciname = self.imtype.makeSciName(i+1,section=self.pars['section'][i])
            _dqname = self.imtype.makeDQName(i+1)
            _extname = self.imtype.dq_extname

            # Build mask files based on input 'bits' parameter values
            _masklist = []
            _masknames = []
            #
            # If we have a valid bits value...
            # Creat the name of the output mask file
            _maskname = buildmask.buildMaskName(_dqname,i+1)
            _masknames.append(_maskname)
            # Create the actual mask file now...
            outmask = buildmask.buildMaskImage(_dqname,self.bitvalue[0],_maskname,extname=_extname,extver=i+1)
            _masklist.append(outmask)

            #
            # Check to see if a bits value was provided for single drizzling...
            # Different bits value specified for single drizzle step
            # create new filename for single_drizzle mask file
            _maskname = _maskname.replace('final_mask','single_mask')
            _masknames.append(_maskname)

            # Create new mask file with different bit value.
            outmask = buildmask.buildMaskImage(_dqname,self.bitvalue[1],_maskname,extname=_extname,extver=i+1)
            # Add new name to list for single drizzle step
            _masklist.append(outmask)
            _masklist.append(_masknames)
        
            self.members.append(Exposure(_sciname, idckey=self.idckey, dqname=_dqname,
                    mask=_masklist, pa_key=self.pa_key, parity=self.PARITY[detector],
                    idcdir=self.pars['idcdir'], group_indx = i+1,
                    handle=self.image_handle,extver=i+1,exptime=self.exptime[0], mt_wcs=self.pars['mt_wcs']))
 
    def setBunit(self,value=None):
        """Set the bunit attribute for each member.
            Default value defined in Exposure class is 'ELECTRONS'
            If a new value is given as input, it will override the default.
        """
        if value is not None:
            for member in self.members:
                member.set_bunit(value)

    def getProductCorners(self):
        """ Compute the product's corner positions based on input exposure's
            corner positions.
        """
        _prodcorners = []
        for member in self.members:
            _prodcorners += member.corners['corrected'].tolist()

        self.product.corners['corrected'] = N.array(_prodcorners,dtype=N.float64)

    def buildProduct(self,filename,output):
        """
        Create Exposure object for meta-chip product after applying
        distortion model to input members.
        """
        #output_wcs = self._buildMetachip(psize=_psize,orient=_rot)
        output_wcs = self.buildMetachip()

        # For each member, update WCSLIN to account for chip-to-chip
        # offsets, [X/Y]DELTA in refpix.


        # Update the final output size with the meta-chip size
        self.size = (output_wcs.naxis1,output_wcs.naxis2)

        # Need to compute shape for dither product here based on output_shapes
        # and reference pixels for each observation.
        self.product = Exposure(self.rootname,wcs=output_wcs, new=yes)

        self.product.exptime = self.exptime

        # Preserve the original WCS as WCSLIN...
        self.product.geometry.wcslin = self.product.geometry.wcs.copy()

        # Combine all corner positions for each input into product's corner
        self.getProductCorners()

    def buildMetachip(self,update=yes):
        """ Build up the new metashape based on the
         corrected size and position for each Exposure.
         (Pattern method)
        """
        # Build an initial value for size of meta-chip
        _geo_ref = self.members[0].geometry
        _wcs_ref = _geo_ref.wcslin
        _model_ref = _geo_ref.model

        # Get pscale from linearized WCS
        pscale = _wcs_ref.pscale

        _shape = (_wcs_ref.naxis1,_wcs_ref.naxis2,pscale)
        # Build new WCS for output metachip here
        # It will be based on the undistorted version of members[0] WCS
        meta_wcs = _wcs_ref.copy()

        # Now, check to see if members have subarray offsets, but was
        # taken as a full image...
        if len(self.members) > 1 and _geo_ref.wcs.subarray == yes:
            for member in self.members:
                member.geometry.wcslin.subarray = no
                member.geometry.model.refpix['centered'] = no

        #Determine range of pixel values for corrected image
        # Using verbose=yes will return additional info on range calculation
        meta_range = drutil.getRange(self.members,meta_wcs,verbose=no)

        # Update WCS based on new size
        xsize = int(meta_range['xsize'])
        ysize = int(meta_range['ysize'])
        meta_wcs.naxis1 = xsize
        meta_wcs.naxis2 = ysize
        cen = ((xsize/2.),(ysize/2.))
        meta_wcs.recenter()

        _nref = meta_range['nref']
        for member in self.members:
            member.corners['corrected'] -= (_nref[0]/2.,_nref[1]/2.)

        if update:
            # Shifts position of CRPIX to reflect new size
            # Instead of being centered on (0.,0.) like the original guess.
            # CRVAL for this position remains the same, though, as it is still
            # the same point in the sky/image.
            #
            # We need this in order to correctly handle WFPC2 data where
            # the XDELTA/YDELTA values are computed relative to the image center
            # already, so we don't need this correction.
            if not _model_ref.refpix['centered']:
                _nref = meta_range['nref']

                for member in self.members:
                    _refpix = member.geometry.model.refpix
                    # Update XDELTA,YDELTA (zero-point of coefficients) to adjust for
                    # uncentered output
                    _refpix['XDELTA'] -= _nref[0]/2.
                    _refpix['YDELTA'] -= _nref[1]/2.
        #
        # TROLL computation not needed, as this get corrected for both
        # in 'recenter()' and in 'wcsfit'...
        # WJH 19-Feb-2004
        #
        # Do a full fit between the input WCS and meta_wcs now...
        #
        # Rigorously compute the orientation changes from WCS
        # information using algorithm provided by R. Hook from WDRIZZLE.
        abxt,cdyt = drutil.wcsfit(self.members[0].geometry, meta_wcs)
        #Compute the rotation between input and reference from fit coeffs.
        _delta_rot = RADTODEG(N.arctan2(abxt[1],cdyt[0]))
        _crpix = (meta_wcs.crpix1 + abxt[2], meta_wcs.crpix2 + cdyt[2])

        meta_wcs.crval1,meta_wcs.crval2 = meta_wcs.xy2rd(_crpix)
        # Insure output WCS has exactly orthogonal CD matrix
        #meta_wcs.rotateCD(meta_wcs.orient+_delta_rot)
        meta_wcs.updateWCS(orient=meta_wcs.orient+_delta_rot)

        return meta_wcs

    def transformMetachip(self,ref):
        """
        This method transforms this Exposure's WCS to be consistent
        with the provided reference WCS 'ref'.  This method only
        operates on the product MetaChip, with the original WCS
        being preserved as 'wcslin'.

        Primarily, this transformation involves scaling and rotating
        the chip to match the reference frame values. Also, any specified
        size for the output frame would replace the default rotated/scaled
        size. All rotations would be about the center, and the reference
        pixel position gets shifted to accomodate this rotation.

        """
        # Start by getting product WCS
        # Use values from 'wcslin' as they will keep track of the
        # default situation
        _in_wcs = self.product.geometry.wcslin
        _out_wcs = self.product.geometry.wcs

        # Make sure that wcslin has all the WCS information
        if _in_wcs.rootname == 'New':
            # Copy the original WCS data into 'wcslin' to preserve the values
            _in_wcs = self.product.geometry.wcs.copy()

        #_dcrpix = (_in_wcs.naxis1/2.- _in_wcs.crpix1,_in_wcs.naxis2/2.- _in_wcs.crpix2)

        # Check to see if there is any rotation needed
        if ref.orient != None:
            _angle = _in_wcs.orient - ref.wcs.orient
            _ref_orient = ref.wcs.orient
        else:
            _angle = 0.
            _ref_orient = _in_wcs.orient

        # Apply any pixel scale changes to delta_crpix and size of axes
        if ref.psize != None:
            _scale = _in_wcs.pscale / ref.wcs.pscale
            _ref_pscale = ref.wcs.pscale
        else:
            _scale = 1.0
            _ref_pscale = _in_wcs.pscale

        if ref.wcs.naxis1 != 0 and ref.wcs.naxis2 != 0:
            _naxis1 = ref.wcs.naxis1
            _naxis2 = ref.wcs.naxis2
            # Delta between same-scaled frames
            _delta_cens = (ref.wcs.naxis1/_scale - _in_wcs.naxis1,ref.wcs.naxis2/_scale - _in_wcs.naxis2)
        else:
            _delta_cens = (0.,0.)
            _naxis1 = _in_wcs.naxis1
            _naxis2 = _in_wcs.naxis2

        _delta_cen = (_naxis1 - _in_wcs.naxis1, _naxis2 - _in_wcs.naxis2)

        # Rotate axes as necessary
        if _angle != 0.:
            # Rotate axes to find default rotated size and new center
            _xrange,_yrange = drutil.getRotatedSize(self.product.corners['corrected'],_angle)
            _range = [_xrange[1] - _xrange[0] + _delta_cens[0]/2.,_yrange[1] - _yrange[0] + _delta_cens[1]/2.]
            #_dcrpix = ((_xrange[0] + _xrange[1])/2.,(_yrange[0] + _yrange[1])/2.)
        else:
            _range = [_naxis1,_naxis2]

        # Now update CD matrix and reference position
        _out_wcs.naxis1 = int(_range[0])
        _out_wcs.naxis2 = int(_range[1])

        _crpix = (_in_wcs.crpix1 + _delta_cen[0]/2.,
                  _in_wcs.crpix2 + _delta_cen[1]/2.)

        if _scale != 0.:
            _out_wcs.updateWCS(orient=_ref_orient,pixel_scale=_ref_pscale)
            _delta_crpix = (_naxis1 - _out_wcs.naxis1, _naxis2 - _out_wcs.naxis2)
        else:
            _out_wcs.rotateCD(_ref_orient)
            _delta_crpix = (0.,0.)

        _out_wcs.crpix1 = _crpix[0] - _delta_crpix[0]/2.
        _out_wcs.crpix2 = _crpix[1] - _delta_crpix[1]/2.

        _out_wcs.recenter()
        """
        # Update the size and rotated position of reference pixel
        _cen = (_out_wcs.naxis1/2.,_out_wcs.naxis2/2.)

        _out_wcs.crval1,_out_wcs.crval2 = _out_wcs.xy2rd(_cen)
        _out_wcs.crpix1 = _cen[0]
        _out_wcs.crpix2 = _cen[1]
        """
        
    def translateShifts(self):
        """
        Translate the shifts specified in the ASNDICT (as read in from the 
        shiftfile) into offsets in the sky, so they can be translated back
        into the WCS of the PyDrizzle output product.
        
        NOTE:  Only works with 'delta' shifts now, and 
                    requires that a 'refimage' be specified.
        """
        asndict = self.pars['asndict']
                
        # for each set of shifts, translate them into delta(ra,dec) based on refwcs
        for img in asndict['order']:

            xsh = asndict['members'][img]['xshift']
            ysh = asndict['members'][img]['yshift']

            if xsh == 0.0 and ysh == 0.0:
                delta_ra = 0.0
                delta_dec = 0.0
            else:
                #check the units for the shifts...
                if asndict['members'][img]['shift_units'] == 'pixels':
                    # Initialize the reference WCS for use in translation
                    # NOTE: This assumes that a 'refimage' has been specified for
                    #       every set of shifts.
                    refwcs = wcsutil.WCSObject(asndict['members'][img]['refimage'])
                    cp1 = refwcs.crpix1
                    cp2 = refwcs.crpix2

                    nra,ndec = refwcs.xy2rd((cp1+xsh,cp2+ysh))
                    
                    delta_ra = refwcs.crval1-nra
                    delta_dec = refwcs.crval2-ndec
                else:
                    # Shifts already in units of RA/Dec (decimal degrees)
                    # No conversion necessary
                    delta_ra = xsh
                    delta_dec = ysh
            
            asndict['members'][img]['delta_ra'] = delta_ra
            asndict['members'][img]['delta_dec'] = delta_dec
        
    def getShifts(self,member,wcs):
        """
        Translate the delta's in RA/Dec for each image's shift into a 
        shift of the undistorted image.
 
        Input:
            member - Exposure class for chip
            wcs    - PyDrizzle product WCS object
 
        Output:
            [xsh, ysh, rot, scale] - 
                  Returns the full set of shift information as a list
 
        """
        asndict=self.pars['asndict']
        
        mname = None
        for img in asndict['order']: 
            if member.name.find(img) > -1: 
                mname = img
                break
        
        row = asndict['members'][img]
        if row['delta_ra'] == 0.0 and row['delta_dec'] == 0.0:
            xsh = 0.0
            ysh = 0.0
            drot = 0.0
            dscale = 1.0
        else:     
            # translate delta's into shifts
            ncrpix1,ncrpix2 = wcs.rd2xy((wcs.crval1+row['delta_ra'],
                                         wcs.crval2+row['delta_dec']))

            xsh = ncrpix1 - wcs.crpix1
            ysh = ncrpix2 - wcs.crpix2
            drot= -row['rot']
            dscale = row['scale']

        return [xsh,ysh,drot,dscale]
        
    # This method would use information from the product class and exposure class
    # to build the complete parameter list for running the 'drizzle' task
    def buildPars(self,ref=None):
        """This method would build a list of parameters to run 'drizzle'
         one a single input image.
            The reference image info 'ref' will be passed as a SkyField object.
            The default output reference frame will be passed as 'def_wcs'
            for comparison to the user's selected object 'ref'.
        """
        # pars contains the drizzle parameters for each input(in order):
        #  data,outdata,outnx,outny,scale,xsh,ysh,rot
        parlist = []

        # Now define the default WCS for this product
        def_wcs = self.product.geometry.wcslin.copy()

        if ref != None:

            # Extract the total exptime for this output object
            if ref.exptime == None:
                _texptime = self.exptime
            else:
                _texptime = ref.exptime
            #
            # Initialize the SkyField Object with the
            # user settings.
            _field = ref

            # Transform self.product.geometry.wcs to match ref.wcs
            #self.transformMetachip(_field)

            _out_wcs = self.product.geometry.wcs.copy()

            #if not ref.dither or ref.dither == None:
            # Check to make sure we have a complete WCS
            # if not, fill in using the product's default WCS
            _field.mergeWCS(_out_wcs)
            _field.wcs.rootname=def_wcs.rootname

            self.product.geometry.wcslin = _out_wcs
            self.product.geometry.wcs = _field.wcs.copy()

            # Set reference for computing shifts to be transformed
            # product's WCS
            ref_wcs = _field.wcs

        else:
            #
            # Single observation case...
            #
            # Define a default WCS
            ref_wcs = self.product.geometry.wcslin.copy()

            # Update product WCS to reflect default value again
            self.product.geometry.wcs = ref_wcs.copy()

            # Extract the total exposure time
            _texptime = self.product.exptime

        # Insure that reference WCS is properly centered in order to
        # correctly fit to each input WCS
        ref_wcs.recenter()

        # Convert shifts into delta RA/Dec values
        self.translateShifts()
        
        for member in self.members:
            in_wcs = member.geometry.wcslin
            in_wcs_orig = member.geometry.wcs

            #_delta_rot = in_wcs.orient - ref_wcs.orient
            #_scale = ref_wcs.pscale / in_wcs.pscale

            # Compute offset based on shiftfile values
            xsh,ysh,drot,dscale = self.getShifts(member,ref_wcs)

            # Rigorously compute the orientation changes from WCS
            # information using algorithm provided by R. Hook from WDRIZZLE.
            abxt,cdyt = drutil.wcsfit(member.geometry, ref_wcs)

            # Compute the rotation between input and reference from fit coeffs.
            _delta_roty = _delta_rot = RADTODEG(N.arctan2(abxt[1],cdyt[0]))
            _delta_rotx = RADTODEG(N.arctan2(abxt[0],cdyt[1]))
            # Compute scale from fit to allow WFPC2 (and similar) data to be handled correctly
            _scale = 1./N.sqrt(abxt[0]**2 + abxt[1]**2)

            # Correct for additional shifts from shiftfile now
            _delta_x = abxt[2]
            _delta_y = cdyt[2]

            # Start building parameter dictionary for this chip
            parameters = ParDict()
            parameters['data'] = member.name
            parameters['output'] = self.output
            parameters['exposure'] = member
            parameters['group'] = member.group_indx

            parameters['instrument'] = self.instrument
            parameters['detector'] = self.detector

            parameters['driz_mask'] = member.maskname
            parameters['single_driz_mask'] = member.singlemaskname

            # Setup parameters for special cases here...
            parameters['outsingle'] = self.outsingle
            parameters['outsweight'] = self.outsweight
            parameters['outscontext'] = self.outscontext
            parameters['outblot'] = member.outblot
            parameters['blotnx'] = member.naxis1
            parameters['blotny'] = member.naxis2

            #Setup parameters for normal operations
            parameters['outdata'] = self.outdata
            parameters['outweight'] = self.outweight
            parameters['outcontext'] = self.outcontext

            parameters['outnx'] = ref_wcs.naxis1
            parameters['outny'] = ref_wcs.naxis2

            parameters['xsh'] =  _delta_x + xsh
            parameters['ysh'] =  _delta_y + ysh

            parameters['alpha'] = member.geometry.alpha
            parameters['beta'] = member.geometry.beta
            
            # Calculate any rotation relative to the orientation
            # AFTER applying ONLY the distortion coefficients without
            # applying any additional rotation...
            parameters['rot'] = _delta_rot - drot

            # Keep track of both the exposure information and
            # the combined product exposure time information.
            # Single exposure information will be used for 'single=yes'
            # header updates...
            parameters['exptime'] = self.exptime[0]
            parameters['expstart'] = self.exptime[1]
            parameters['expend'] = self.exptime[2]
            parameters['texptime'] = _texptime[0]
            parameters['texpstart'] = _texptime[1]
            parameters['texpend'] = _texptime[2]
            # Need to revise how this gets computed...
            # The pixel scale of the product corresponds to the
            # desired output pixel scale, and the model pscale for
            # the member represents the un-scaled pixel size for the input.
            parameters['scale'] = _scale * dscale

            # Parameters only used by 'wdrizzle'
            parameters['geomode'] = 'wcs'
            parameters['racen'] = ref_wcs.crval1
            parameters['deccen'] = ref_wcs.crval2
            parameters['orient'] = ref_wcs.orient
            parameters['outscl'] = ref_wcs.pscale
            parameters['gpar_xsh'] = member.geometry.gpar_xsh - xsh
            parameters['gpar_ysh'] = member.geometry.gpar_ysh - ysh
            parameters['gpar_rot'] = member.geometry.gpar_rot - drot

            # Insure that the product WCS applied to each exposure gets set
            member.product_wcs = ref_wcs
            # Set up the idcfile for use by 'drizzle'
            member.writeCoeffs()
            parameters['coeffs'] = member.coeffs

            parameters['plam'] = member.plam

            # Set up the distortion image names as parameters
            parameters['xgeoim'] = member.xgeoim
            parameters['ygeoim'] = member.ygeoim

            # Now pass along the remainder of the user specified parameters
            if self.pars['units'] != None:
                parameters['units'] = self.pars['units']
            else:
                parameters['units'] = 'cps'

            if self.pars['in_units'] != None:
                parameters['in_units'] = self.pars['in_units']
            else:
                parameters['in_units'] = 'counts'

            if self.pars['pixfrac'] != None:
                parameters['pixfrac'] = self.pars['pixfrac']
            else:
                parameters['pixfrac'] = 1.0

            if self.pars['kernel'] != None:
                parameters['kernel'] = self.pars['kernel']
            else:
                parameters['kernel'] = 'square'

            if self.pars['wt_scl'] != None:
                parameters['wt_scl'] = self.pars['wt_scl']
            else:
                parameters['wt_scl'] = 'exptime'

            if self.pars['fillval'] != None:
                parameters['fillval'] = str(self.pars['fillval'])
            else:
                parameters['fillval'] = 'INDEF'

            # Parameters useful for header keywords
            parameters['version'] = 'PyDrizzle Version '+__version__
            parameters['driz_version'] = ''
            parameters['nimages'] = self.nimages
            parameters['bunit'] = member.bunit

            parlist.append(parameters)

        # Now, combine them for a complete set of pars for all exposures
        # in this pattern/observation.
        #
        return parlist

    def computeCubicCoeffs(self):
        """
        Method for converting cubic and Trauger coefficients tables
        into a usable form.  It also replaces 'computeOffsets' for
        those tables as well.
        """
        # For each chip in the observation...
        _pscale1 = None
        for img in self.members:
            _chip = img.chip
            _detector = str(img.header[self.DETECTOR_NAME])
            # scale all chips to first chip plate scale...
            if _pscale1 == None or img.chip == '1':
                _pscale1 = self.REFDATA[_detector]['psize']
                _reftheta = self.REFDATA[_detector]['theta']

        _v2ref = 0.
        _v3ref = 0.
        _nmembers = 0
        for img in self.members:
            # ... get the model and type of coefficients table used...
            _model = img.geometry.model
            _ikey = img.geometry.ikey
            _chip = img.chip
            _detector = str(img.header[self.DETECTOR_NAME])
            _refdata = self.REFDATA[_detector]

            # ... determine the plate scale and scaling factor between chips...
            if img.chip == '1':
                _pscale = _refdata['psize']
                _ratio = 1.0
            else:
                _pscale = _refdata['psize']
                _ratio = _refdata['psize'] / _pscale1
            # Record the plate scale for each chip's model that was used
            # to compute the coefficients, not the plate scale from the
            # image header.
            _model.pscale = _pscale
            _model.refpix['PSCALE'] = _pscale

            if _ikey == 'trauger':
                _ratio = 1.

                # V2REF/V3REF was not available in idc file, so we
                # must use our own data...
                _model.refpix['V2REF'] = _refdata['xoff']
                _model.refpix['V3REF'] = _refdata['yoff']

            else:
                _model.refpix['V2REF'] = _refdata['xoff'] * _pscale
                _model.refpix['V3REF'] = _refdata['yoff'] * _pscale

            # Correct the coefficients for the differences in plate scales
            _model.cx = _model.cx * N.array([_model.pscale/_ratio],dtype=N.float64)
            _model.cy = _model.cy * N.array([_model.pscale/_ratio],dtype=N.float64)
            _model.refpix['XREF'] = self.REFPIX['x']
            _model.refpix['YREF'] = self.REFPIX['y']

            # Correct the offsets for the plate scales as well...
            _model.refpix['XDELTA'] = _refdata['xoff']
            _model.refpix['YDELTA'] = _refdata['yoff']

            _model.refpix['centered'] = yes

            if _ikey != 'trauger':
                _model.refpix['V2REF'] = _model.refpix['V2REF'] / _ratio
                _model.refpix['V3REF'] = _model.refpix['V3REF'] / _ratio
            _v2ref += _model.refpix['V2REF']
            _v3ref += _model.refpix['V3REF']
            _nmembers += 1


    def computeOffsets(self,parity=None,refchip=None):
        """
        This version of 'computeOffsets' calculates the zero-point
        shifts to be included in the distortion coefficients table
        used by 'drizzle'.
        It REQUIRES a parity matrix to convert from
        V2/V3 coordinates into detector image X/Y coordinates. This
        matrix will be specific to each detector.
        """
        vref = []

        # Check to see if any chip-to-chip offsets need to be computed at all
        if len(self.members) == 1:
            refp = self.members[0].geometry.model.refpix
            refp['XDELTA'] = 0.
            refp['YDELTA'] = 0.
            #refp['centered'] = yes
            return

        # Set up the parity matrix here for a SINGLE chip
        if parity == None:
            # Use class defined dictionary as default
            parity = self.PARITY
        # Get reference chip information
        ref_model=None
        for member in self.members:
            if not refchip or refchip == int(member.chip):
                ref_model = member.geometry.model
                ref_scale = ref_model.refpix['PSCALE']
                ref_v2v3 = N.array([ref_model.refpix['V2REF'],ref_model.refpix['V3REF']])
                ref_theta = ref_model.refpix['THETA']
                if ref_theta == None: ref_theta = 0.0
                ref_pmat = N.dot(fileutil.buildRotMatrix(ref_theta), member.parity)
                ref_xy = (ref_model.refpix['XREF'],ref_model.refpix['YREF'])
                break

        if not ref_model:
            ref_scale = 1.0
            ref_theta = 0.0
            ref_v2v3 = [0.,0.]
            ref_xy = [0.,0.]
            ref_pmat = N.array([[1.,0.],[0.,1.0]])

        # Now compute the offset for each chip
        # Compute position of each chip's common point relative
        # to the output chip's reference position.
        for member in self.members:
            in_model = member.geometry.model
            refp = in_model.refpix
            pscale = in_model.pscale
            memwcs = member.geometry.wcs

            v2v3 = N.array([in_model.refpix['V2REF'],in_model.refpix['V3REF']])
            scale = refp['PSCALE']
            theta = refp['THETA']
            if theta == None: theta = 0.0

            chipcen = ( (memwcs.naxis1/2.) + memwcs.offset_x,
                        (memwcs.naxis2/2.) + memwcs.offset_y)
            xypos = N.dot(ref_pmat,v2v3-ref_v2v3) / scale + ref_xy
            chiprot = fileutil.buildRotMatrix(theta - ref_theta)

            offcen = ((refp['XREF'] - chipcen[0]), (refp['YREF'] - chipcen[1]))

            # Update member's geometry model with computed
            # reference position...
            #refp['XDELTA'] = vref[i][0] - v2com + chip.geometry.delta_x
            #refp['YDELTA'] = vref[i][1] - v3com + chip.geometry.delta_y
            offset_xy = N.dot(chiprot,xypos-offcen)*scale/ref_scale
            
            refp['XDELTA'] = offset_xy[0]
            refp['YDELTA'] = offset_xy[1]

            # Only set centered to yes for full exposures...
            if member.geometry.wcs.subarray != yes:
                refp['centered'] = no
            else:
                refp['centered'] = yes


    def setNames(self,filename,output):
        """
        Define standard name attibutes:
                outname     - Default final output name
                outdata     - Name for drizzle science output
                outsingle   - Name for output for single image
        """
        self.rootname = filename
        self.outname = output

        # Define FITS output filenames for intermediate products
        # to be used when 'build=no'
        self.outdata = fileutil.buildNewRootname(output,extn='_sci.fits')
        self.outweight = fileutil.buildNewRootname(output,extn='_weight.fits')
        self.outcontext = fileutil.buildNewRootname(output,extn='_context.fits')

        # Define output file names for separate output for each input
        self.outsingle = fileutil.buildNewRootname(filename,extn='_single_sci.fits')
        self.outsweight = fileutil.buildNewRootname(filename,extn='_single_wht.fits')
        self.outscontext = None
        #self.outscontext = fileutil.buildNewRootname(filename,extn='_single_ctx.fits')

    ########
    #
    # User interface methods
    #
    ########
    def getWCS(self):
        return self.members[0].getWCS()

    def getMember(self,memname):
        """ Return the class instance for the member with name memname."""
        member = None
        for mem in self.members:
            if mem.name == memname:
                member = mem
        return member

    def getMemberNames(self):
        """ Return the names of all members for this Class.
            Output: [{self.name:[list of member names]}]
        """
        memlist = []
        for member in self.members:
            memlist.append(member.name)
        return [{self.name:memlist}]

    def getExptime(self):
        _exptime = float(self.header['EXPTIME'])
        if _exptime == 0.: _exptime = 1.0

        if self.header.has_key('EXPSTART'):
            _expstart = float(self.header['EXPSTART'])
            _expend = float(self.header['EXPEND'])
        else:
            _expstart = 0.
            _expend = _exptime

        return (_exptime,_expstart,_expend)

    def DeltaXYtoOffset(self,delta):
        """
        Converts provided delta(x,y) pixel offset into a
        delta(RA,Dec) offset in arcseconds.
        """
        _wcs = self.product.getWCS()
        _geom = self.product.geometry

        new_rd = _geom.XYtoSky((_wcs.crpix1 - delta[0],_wcs.crpix2 - delta[1]))
        delta_ra = (_wcs.crval1 - new_rd[0]) * 3600.
        delta_dec = (_wcs.crval2 - new_rd[1]) * 3600.

        return (delta_ra,delta_dec)


