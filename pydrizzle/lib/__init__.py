import string,os,types,sys
import shutil

#from pyraf import iraf
#from pyraf.iraf import stsdas,analysis,dither
# Import drizzle/blot Python callable module
import arrdriz

# Import PyDrizzle utility modules
import buildmask, fileutil, wcsutil, drutil
import outputimage, imtype
from exposure import Exposure

import numarray as N
import pyfits

#Add buildasn/updateasn to namespace for use by other programs
import buildasn, updateasn
import dqpars,traits102

yes = True  # 1
no = False  # 0

# List of supported instruments/detectors
INSTRUMENT = ["ACS","WFPC2","STIS","NICMOS"]
DEXPTIME = 'EXPTIME'
DEFAULT_PARITY = [[1.0,0.0],[0.0,1.0]]

from drutil import DEFAULT_IDCDIR
from fileutil import RADTODEG, DEGTORAD
from math import *

# Pass along name of module which contains the DQPars classes
DQPARS = 'dqpars'


# Version
__version__ = "5.2.6 (5-October-2004)"

# For History of changes and updates, see 'History'

def _ptime():
    import time

    # Format time values for keywords IRAF-TLM, and DATE
    _ltime = time.localtime(time.time())
    tlm_str = time.strftime('%H:%M:%S (%d/%m/%Y)',_ltime)
    #date_str = time.strftime('%Y-%m-%dT%H:%M:%S',_ltime)
    return tlm_str

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
        _str += '  XGeoImage      : '+self['xgeoim']+'\n'
        _str += '  YGeoImage      : '+self['ygeoim']+'\n'
        _str += '  Input mask file: '+self['driz_mask']+'\n'
        _str += '  Output science : '+self['outdata']+'\n'
        _str += '  Output weight  : '+self['outweight']+'\n'
        _str += '  Output context : '+self['outcontext']+'\n'
        _str += '  Exptime -- total: %0.4f     single: %0.4f\n'%(self['texptime'],self['exptime'])
        _str += '     start: %s         end: %s\n'%(repr(self['expstart']),repr(self['expend']))
        _str += '  Single image products--  output:  %s\n'%self['outsingle']
        _str += '     weight:  %s  \n '%self['outsweight']
        _str += '  Blot output: %s \n'%self['outblot']
        _str += '  Size of original image to blot: %d %d \n'%(self['blotnx'],self['blotny'])

        _str += '\n'

        return _str

class Pattern:
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
    DQCLASS = 'DQPars'
    COPY_SUFFIX = '.orig'       # suffix to use for filename of copy

    def __init__(self, filename, output=None, pars=None):
        # Set these up for use...
        self.members = []
        self.pars = pars
        self.output = output
        self.name = filename

        # Set default value, to be reset as needed by sub-class
        self.nmembers = 1

        # Extract bit values to be used for this instrument
        self.bitvalue = self.getBits(bits=self.pars['bits'])

        # Set IDCKEY, if specified by user...
        if self.pars['idckey'] == None:
            self.idckey = self.IDCKEY
        else:
            self.idckey = self.pars['idckey']
        self.idcdir = self.pars['idcdir']

        # Keyword which provides PA_V3 value; ie., orientation of
        #       telescope at center axis relative to North.
        # Each instrument has their own pre-defined keyword; mostly, PA_V3
        self.pa_key = self.PA_KEY

        self.exptime = None

        # These attributes are used for keeping track of the reference
        # image used for computing the shifts, and the shifts computed
        # for each observation, respectively.
        self.refimage = None
        self.offsets = None
        self.v2com = None
        self.v3com = None

        # Read in Primary Header to reduce the overhead of getting
        # keyword values and setup PyFITS object
        # as self.header and self.image_handle
        #
        self.getHeaderHandle()

        # Determine type of input image and syntax needed to read data
        self.imtype = imtype.Imtype(filename,handle=self.image_handle,
                                    dqsuffix=self.pars['dqsuffix'])

        if self.header and self.header.has_key(self.DETECTOR_NAME):
            self.detector = self.header[self.DETECTOR_NAME]
        else:
            self.detector = 'detector'

        # Read in date-obs from primary header
        if self.header:
            if self.header.has_key('date-obs'):
                self.dateobs = self.header['date-obs']
            elif self.header.has_key('date_obs'):
                self.dateobs = self.header['date_obs']
            else:
                self.dateobs = None
        else:
            self.dateobs = None

    def getHeaderHandle(self):
        """ Sets up the PyFITS image handle and Primary header
            as self.image_handle and self.header.

            When Pattern being used for output product, filename will be
            set to None and this returns None for header and image_handle.
        """

        if self.name:
            _handle = fileutil.openImage(self.name,mode='readonly',memmap=self.pars['memmap'])
            _fname,_extn = fileutil.parseFilename(self.name)
            _hdr = _handle['PRIMARY'].header.copy()

            if _extn > 0:
                # Append correct extension/chip/group header to PRIMARY...
                for _card in fileutil.getExtn(_handle,_extn).header.ascardlist():
                    _hdr.ascard.append(_card)
        else:
            # Default to None
            _handle = None
            _hdr = None

        # Set attribute to point to these products
        self.image_handle = _handle
        self.header = _hdr

    def closeHandle(self):
        """ Closes image_handle. """
        if self.image_handle:
            self.image_handle.close()
        self.image_handle = None


    def getBits(self,bits=None):
        """ Method for extracting the bits value set through DQPars."""
        if self.DQCLASS:
            _a = eval(DQPARS+'.'+self.DQCLASS)()
            if bits != None:
                _a.update(bits)
            _bits = _a.bits
            del _a
        else:
            _bits = None
        return _bits

    def addMembers(self,filename):
        """ Build rootname for each SCI extension, and
            create the mask image from the DQ extension.
            It would then append a new Exposure object to 'members'
            list for each extension.
        """

        self.detector = detector = str(self.header[self.DETECTOR_NAME])

        # Build rootname here for each SCI extension...
        for i in range(self.nmembers):
            _sciname = self.imtype.makeSciName(i+1,section=self.pars['section'])
            _dqname = self.imtype.makeDQName(i+1)
            _extname = self.imtype.dq_extname

            _maskname = buildmask.buildMaskImage(_dqname,self.bitvalue,extname=_extname,extver=i+1)

            self.members.append(Exposure(_sciname, idckey=self.idckey, dqname=_dqname,
                    mask=_maskname, pa_key=self.pa_key, parity=self.PARITY[detector],
                    dateobs=self.dateobs, idcdir=self.pars['idcdir'],
                    handle=self.image_handle,extver=i+1))

    def applyAsnShifts(self):
        """ Apply ASN Shifts to each member and the observations product. """
        _geo = self.product.geometry.wcslin
        _geo0 = self.product.geometry.wcslin.copy()

        _crval0 = (_geo.crval1,_geo.crval2)
        _rot0 = _geo.orient
        self._applyShifts(_geo)

        _dcrval = (_crval0[0] - _geo.crval1,_crval0[1] - _geo.crval2)

        _delta_rot = _rot0 - _geo.orient
        _crpixlin = _geo.rd2xy(_crval0)
        _delta_crpix = (_geo.crpix1 - _crpixlin[0],_geo.crpix2 - _crpixlin[1])

        #
        # Do NOT apply AsnShifts to product.geometry.wcs as it serves
        # as the default WCS for computing the output frame, allowing
        # the images to be shifted relative to original WCS.  This will
        # result in output image where inputs may both be shifted off-center
        # but presumably those shifts correct errors in WCS. WJH 28-Apr-03
        #self._applyShifts(self.product.geometry.wcs)
        #
        # Update each members WCS with the new shifts, to allow for
        # accurate RA/Dec measurements using internal methods.
        for member in self.members:
            # Compute chip specific shifts now...
            _mem_geo = member.geometry
            _mem_wcs = member.geometry.wcs
            _wcs_copy = _mem_wcs.copy()
            _mem_orient = member.geometry.wcs.orient

            #if _delta_crpix[0] == 0. and _delta_crpix[1] == 0.: continue

            # Remember secondary shifts from ASN table
            member.geometry.gpar_xsh = _delta_crpix[0]
            member.geometry.gpar_ysh = _delta_crpix[1]
            member.geometry.gpar_rot = _delta_rot

            # Implement 'WBACKWCS' to update input WCS with shift computed
            # in the output/reference WCS frame.
            #
            # Compute the pixel position of the input reference image in
            # reference WCS
            _oxy1 = _mem_geo.wtraxy((_mem_wcs.crpix1   ,_mem_wcs.crpix2   ),_geo0)
            _oxy2 = _mem_geo.wtraxy((_mem_wcs.crpix1+1.,_mem_wcs.crpix2   ),_geo0)
            _oxy3 = _mem_geo.wtraxy((_mem_wcs.crpix1   ,_mem_wcs.crpix2+1.),_geo0)

            # Compute new CRPIX values for original WCS model to use for
            # removing change in orientation from this shift. WJH
            _oxy1o = _mem_geo.wtraxy((_mem_wcs.crpix1, _mem_wcs.crpix2),_geo)

            # Use 'xy2rd' to convert these pixel positions to RA/Dec
            ra1,dec1 = _geo.xy2rd(_oxy1)
            ra2,dec2 = _geo.xy2rd(_oxy2)
            ra3,dec3 = _geo.xy2rd(_oxy3)

            # Convert these numbers into a new WCS
            _mem_wcs.crval1 = ra1
            _mem_wcs.crval2 = dec1

            """
            Now update CD matrix itself
            We will have a lever-arm effect in place given the shift in
            reference position.

            """
            # Rigorously work out the new CD matrix, via the tangent plane
            ra1 = DEGTORAD(ra1)
            ra2 = DEGTORAD(ra2)
            ra3 = DEGTORAD(ra3)
            dec1 = DEGTORAD(dec1)
            dec2 = DEGTORAD(dec2)
            dec3 = DEGTORAD(dec3)

            bot2 = sin(dec2)*sin(dec1) + cos(dec2)*cos(dec1)*cos(ra2-ra1)
            bot3 = sin(dec3)*sin(dec1) + cos(dec3)*cos(dec1)*cos(ra3-ra1)

            # Go to tangent plane positions
            _mem_wcs.cd11 = RADTODEG(cos(dec2) * sin(ra2-ra1) / bot2 )
            _mem_wcs.cd12 = RADTODEG(cos(dec3) * sin(ra3-ra1) / bot3 )
            _mem_wcs.cd21 = RADTODEG( (sin(dec2)*cos(dec1) - cos(dec2)*sin(dec1)*cos(ra2-ra1)) / bot2 )
            _mem_wcs.cd22 = RADTODEG((sin(dec3)*cos(dec1) - cos(dec3)*sin(dec1)*cos(ra3-ra1)) / bot3 )

            _mem_wcs.orient = fileutil.RADTODEG(N.arctan2(_mem_wcs.cd12,_mem_wcs.cd22))
            _mem_wcs.pscale = N.sqrt(N.power(_mem_wcs.cd11,2) +N.power(_mem_wcs.cd21,2))*3600.
            #
            # Compute change in orientation due to change in CRPIX.
            # This delta gets introduced due to the use of the updated
            # undistorted WCS with the distorted WCS's reference positions.
            # It must be taken out in order to get proper alignment for
            # observations with large delta shifts. WJH 5-May-2004
            #
            _ref_wcs = _mem_wcs.copy()
            _ref_wcs.crpix1 += _oxy1o[0] - _oxy1[0]
            _ref_wcs.crpix2 += _oxy1o[1] - _oxy1[1]
            _ref_wcs.recenter()
            _mem_wcs.rotateCD(_mem_wcs.orient + (_ref_wcs.orient - _mem_wcs.orient))
            """
            Finish updating CD matrix
            """
            # Make sure updates get translated back to undistorted WCS's
            _mem_geo.undistortWCS()

            # Update Member corner positions
            member.corners['corrected'] += _delta_crpix

        # Now update product corners
        self.getProductCorners()

    def _applyShifts(self,in_wcs):
        """ This method updates each member's WCS to reflect any
            offsets/corrections specified in the ASN table.

            This method converts shifts given in output pixels
            into the input frame by using a reference image's
            WCS.  This reference image must exist and have an
            header with a valid WCS; specifically, one which can
            be read using WCSObject.

        """
        # Check to see if there are any offsets given for this member...
        # pars['shift'] will be yes if absolute shifts are given, or
        # if rotations and/or scale changes are provided with/without shifts
        #

        # First, are there any shifts at all..
        if not self.pars.has_key('xshift'):
            return

        # Next, are they non-zero...
        if not self.pars['abshift'] and not self.pars['dshift']:
            return
        #
        # We have shifts, so apply them...
        #
        # Directly apply the shifts provided by the user in the ASN
        # table to the original CRVAL values.
        #
        # Start by converting pixel shifts to units of arcseconds
        # in RA and Dec.
        #
        #in_wcs = self.product.geometry.wcs
        _crval = (in_wcs.crval1,in_wcs.crval2)
        _crpix = (in_wcs.crpix1,in_wcs.crpix2)
        #
        # Setup tuples for containing the final delta's in input units
        #converted from the given shifts regardless of the frame or type.
        #
        _dcrval = (0.,0.)
        _dcrpix = (0.,0.)

        _refimage = False
        # If we are working with shifts provided in output frame...
        if self.pars['refimage'] != '' and self.pars['refimage'] != None:
            # Extract the reference image's WCS for use in converting
            # the shifts into input pixels
            # Only necessary for 'pixel' shifts...
            #
            _out_wcs = wcsutil.WCSObject(self.pars['refimage'])
            _out_wcs.recenter()
            _refimage = True

        """
         Each product now has 'refimage' and 'offsets' attributes
         refimage = {'pix_shift':(),'ra_shift':(),'name':'','val':0.}
         offsets = {'pixels':(),'arcseconds':()}

         PyDrizzle measures ALL shifts relative to the center of the final
         output frame.  The user's shifts will be measured relative to a
         reference image's reference point.  The difference between the
         two frames must be accounted for when comparing shifts measured
         in the two frames.

         This offset needs to be subtracted from the shift provided by
         the user in order to apply it to the PyDrizzle shifts.
        """
        _ra_offset = (self.offsets['arcseconds'][0] - self.refimage['ra_shift'][0],
                    self.offsets['arcseconds'][1] - self.refimage['ra_shift'][1])

        _rotmat = drutil.buildRotMatrix(self.pars['delta_rot'])

        if self.pars['abshift']:
            # Run 'computeOffsets()' to determine default shifts
            # Extract shifts into numarray object and shift to input pixels
            # Compute delta between abshift and 'offsets' as _delta_x,_delta_y
            _delta_x = self.pars['xshift']
            _delta_y = self.pars['yshift']

            # Insure that 'pixel' shifts are in 'input' units
            if self.pars['shift_units'] == 'pixels':
                if _refimage:
                    # Compute delta RA,Dec for output frame ref point
                    # using given pixel shifts
                    _dcrpix = N.dot((_out_wcs.crpix1 - _delta_x,_out_wcs.crpix2 - _delta_y),_rotmat)

                    # CRVAL of output reference frame user shifts were measured from
                    _refcrval = _out_wcs.xy2rd((_dcrpix[0],_dcrpix[1]))
                    # Total arcsecond shift for this image
                    _ra_shift = ( (_refcrval[0] - _out_wcs.crval1),
                                  (_refcrval[1] - _out_wcs.crval2))

                else:
                    # Tested against WFPC2 CADC ASNs...
                    _dcrpix = (_crpix[0] - _delta_x,_crpix[1] - _delta_y)

                    _refcrval = in_wcs.xy2rd(_dcrpix)
                    # Total arcsecond shift for this image
                    _ra_shift = ((_refcrval[0] - in_wcs.crval1),(_refcrval[1] - in_wcs.crval2))
            else:
                #
                # Arcsecond shifts
                #
                # Directly apply shift in arcseconds of RA/Dec to CRVALs
                _ra_shift = (_delta_x/3600.,_delta_y/3600.)

            # Delta RA/Dec based on output pixel shift for this image
            _dcrval = (_ra_shift[0] -_ra_offset[0], _ra_shift[1] -_ra_offset[1])

        else:
            #
            # Delta Shifts
            # Work with deltas from ASN table directly here
            _delta_x = self.pars['delta_x']
            _delta_y = self.pars['delta_y']

            if self.pars['shift_units'] == 'pixels':
                if _refimage:
                    # Compute delta RA,Dec for output frame ref point
                    # using given pixel shifts
                    _dcrpix = N.dot((_out_wcs.crpix1-_delta_x, _out_wcs.crpix2-_delta_y),_rotmat)

                    _refcrval = _out_wcs.xy2rd((_dcrpix[0],_dcrpix[1]))
                    _dcrval = ((_out_wcs.crval1 - _refcrval[0] ),
                               (_out_wcs.crval2 - _refcrval[1] ))
                else:
                    _dcrpix = (_crpix[0] - _delta_x,_crpix[1] - _delta_y)
                    _refcrval = in_wcs.xy2rd(_dcrpix)
                    _dcrval = (_crval[0] - _refcrval[0],_crval[1] - _refcrval[1])
            else:
                #
                # Arcsecond Shifts
                #
                _dcrval = (_delta_x/3600.,_delta_y/3600.)

        # Now, apply computed delta shift, _dcrval, to input WCS
        # and recenter WCS...
        # Update current CRVAL with delta we have computed
        in_wcs.crval1 -= _dcrval[0]
        in_wcs.crval2 -= _dcrval[1]

        # Also, update orientation and plate scale
        _orient = None
        if self.pars['delta_rot'] != 0.:
            _orient = in_wcs.orient + self.pars['delta_rot']

        _scale = None
        if self.pars['delta_scale'] != 0.:
            _scale = in_wcs.pscale * self.pars['delta_scale']

        # Compute new CRVAL for current CRPIX position
        in_wcs.crval1,in_wcs.crval2 = in_wcs.xy2rd(_crpix)
        in_wcs.crpix1 = _crpix[0]
        in_wcs.crpix2 = _crpix[1]

        #if not _refimage:
        # Update product WCS with the new values
        in_wcs.updateWCS(orient=_orient,pixel_scale=_scale)

    def getProductCorners(self):
        """ Compute the product's corner positions based on input exposure's
            corner positions.
        """
        _prodcorners = []
        for member in self.members:
            _prodcorners += member.corners['corrected'].tolist()

        self.product.corners['corrected'] = N.array(_prodcorners,type=N.Float64)

    def buildProduct(self,filename,output):
        """
        Create Exposure object for meta-chip product after applying
        distortion model to input members.
        """
        #output_wcs = self._buildMetachip(psize=_psize,orient=_rot)
        output_wcs = self.buildMetachip()

        # For each member, update WCSLIN to account for chip-to-chip
        # offsets, [X/Y]DELTA in refpix.
        for member in self.members:
            member.geometry.undistortWCS(shape=(output_wcs.naxis1,output_wcs.naxis2))
        output_wcs = self.buildMetachip(update=no)

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
                    # Update corner positions based on corrected DELTAs
                    #member.corners['corrected'] -= (_nref[0]/2.,_nref[1]/2.)
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
            self.transformMetachip(_field)

            if not ref.dither or ref.dither == None:
            # Check to make sure we have a complete WCS
            # if not, fill in using the product's default WCS
                _out_wcs = self.product.geometry.wcs.copy()

                _field.mergeWCS(_out_wcs)
                _field.wcs.rootname=def_wcs.rootname

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

        for member in self.members:
            in_wcs = member.geometry.wcslin
            in_wcs_orig = member.geometry.wcs

            #_delta_rot = in_wcs.orient - ref_wcs.orient
            _scale = ref_wcs.pscale / in_wcs.pscale

            # Rigorously compute the orientation changes from WCS
            # information using algorithm provided by R. Hook from WDRIZZLE.
            abxt,cdyt = drutil.wcsfit(member.geometry, ref_wcs)

            # Compute the rotation between input and reference from fit coeffs.
            _delta_roty = _delta_rot = RADTODEG(N.arctan2(abxt[1],cdyt[0]))
            _delta_rotx = RADTODEG(N.arctan2(abxt[0],cdyt[1]))

            # Correct for additional shifts from shiftfile now
            #_delta_x = abxt[2] + member.geometry.gpar_xsh
            # _delta_y = cdyt[2] + member.geometry.gpar_ysh
            _delta_x = abxt[2]
            _delta_y = cdyt[2]

            # Start building parameter dictionary for this chip
            parameters = ParDict()
            parameters['data'] = member.name
            parameters['output'] = self.output
            parameters['exposure'] = member

            parameters['instrument'] = self.instrument
            parameters['detector'] = self.detector

            # Check to see if a mask was created at all...
            # if not, set it to ''
            if member.maskname == None: _maskname = ''
            else: _maskname = member.maskname
            parameters['driz_mask'] = _maskname

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

            parameters['xsh'] =  _delta_x
            parameters['ysh'] =  _delta_y

            # Calculate any rotation relative to the orientation
            # AFTER applying ONLY the distortion coefficients without
            # applying any additional rotation...
            parameters['rot'] = _delta_rot

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
            parameters['scale'] = _scale

            # Parameters only used by 'wdrizzle'
            parameters['geomode'] = 'wcs'
            parameters['racen'] = ref_wcs.crval1
            parameters['deccen'] = ref_wcs.crval2
            parameters['orient'] = ref_wcs.orient
            parameters['outscl'] = ref_wcs.pscale
            parameters['gpar_xsh'] = member.geometry.gpar_xsh
            parameters['gpar_ysh'] = member.geometry.gpar_ysh
            parameters['gpar_rot'] = member.geometry.gpar_rot

            # coord for image array reference pixel in full chip coords
            _xref = None
            _yref = None
            _delta = yes
            # If we have a subarray, pass along the offset reference position
            if in_wcs.subarray:
                _xref = in_wcs_orig.offset_x + in_wcs_orig.crpix1
                _yref = in_wcs_orig.offset_y + in_wcs_orig.crpix2
                _delta = no
            else:
                # Pass along the reference position assumed by Drizzle
                # based on 'align=center' according to the conventions
                # described in the help page for 'drizzle'.  26Mar03 WJH
                _xref = int(in_wcs_orig.naxis1/2.) + 1.0
                _yref = int(in_wcs_orig.naxis2/2.) + 1.0

            # Set up the idcfile for use by 'drizzle'
            indx = string.rfind(member.name,'.')
            coeffs = member.name[:indx]+'_coeffs'+member.chip+'.dat'
            member.geometry.model.convert(coeffs,xref=_xref,yref=_yref,delta=_delta)
            parameters['coeffs'] = coeffs

            parameters['plam'] = member.plam

            # Set up the distortion image names as parameters
            parameters['xgeoim'] = member.xgeoim
            parameters['ygeoim'] = member.ygeoim

            # Now pass along the remainder of the user specified parameters
            if self.pars['units'] != None:
                parameters['units'] = self.pars['units']
            else:
                parameters['units'] = 'cps'

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
            _model.cx = _model.cx * N.array([_model.pscale/_ratio],type=N.Float64)
            _model.cy = _model.cy * N.array([_model.pscale/_ratio],type=N.Float64)
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
            refp['centered'] = yes
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
                ref_pmat = N.dot(drutil.buildRotMatrix(ref_theta), member.parity)
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

            chipcen = ( memwcs.naxis1/2. + memwcs.offset_x,
                        memwcs.naxis2/2. + memwcs.offset_y)
            xypos = N.dot(ref_pmat,v2v3-ref_v2v3) / scale + ref_xy
            chiprot = drutil.buildRotMatrix(theta - ref_theta)

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

class GenericObservation(Pattern):
    """
        This class defines an observation stored in a Simple FITS format;
        i.e., only a Primary header and image without extensions.
    """

    DQCLASS = None
    REFPIX = {'x':512.,'y':512.}
    DETECTOR_NAME = 'INSTRUME'

    def __init__(self, filename, output, pars=None):

        # Now initialize Pattern with all member exposures...
        Pattern.__init__(self, filename, output=output, pars=pars)

        # Determine the instrument...
        if self.header.has_key('INSTRUME'):
            _instrument = self.header['INSTRUME']
        else:
            _instrument = self.DETECTOR_NAME
        self.instrument = _instrument

        if self.header.has_key('crpix1'):
            self.REFPIX['x'] = self.header['crpix1']
            self.REFPIX['y'] = self.header['crpix2']
        else:
            self.REFPIX['x'] = self.header['naxis1'] /2.
            self.REFPIX['y'] = self.header['naxis2'] /2.

        # build output rootnames here...
        self.setNames(filename,output)

        # Set EXPTIME for exposure
        self.exptime = self.getExptime()

        self.nmembers = 1

        # Now, build list of members and initialize them
        self.addMembers(filename)

        _ikey = self.members[0].geometry.ikey
        if  _ikey != 'idctab' and _ikey != 'wcs' :
            # Correct distortion coefficients to match output pixel scale
            self.computeCubicCoeffs()
        else:
            self.computeOffsets()

        # Set up the input members and create the product meta-chip
        self.buildProduct(filename, output)

class ACSObservation(Pattern):
    """This class defines an observation with information specific
       to ACS WFC exposures, including knowledge of how to mosaic both
       chips."""

    DQCLASS = 'ACSPars'
    # Define a class variable for the gap between the chips
    PARITY = {'WFC':[[1.0,0.0],[0.0,-1.0]],'HRC':[[-1.0,0.0],[0.0,1.0]],'SBC':[[-1.0,0.0],[0.0,1.0]]}

    def __init__(self, filename, output, pars=None):

        # Now, initialize Pattern with all member Exposures...
        Pattern.__init__(self, filename, output=output, pars=pars)

        self.instrument = 'ACS'

        # build required name attributes:
        #  outname, output, outsingle
        self.setNames(filename,output)

        if self.pars['section'] == None:
            self.nmembers = int(self.header['NEXTEND']) / self.NUM_IMSET
            # In case not all expected extensions are present in the ACS image...
            if self.nmembers == 0: self.nmembers = 1
        else:
            # If a section was specified, then there is only 1 member
            # for this observation.
            self.nmembers = 1

        # Set EXPTIME for exposure
        self.exptime = self.getExptime()

        # Build up list of chips in observation
        self.addMembers(filename)

        # Only need to worry about IDC tables for ACS...
        self.computeOffsets()

        # Set up the input members and create the product meta-chip
        self.buildProduct(filename, output)

class STISObservation(Pattern):
    """This class defines an observation with information specific
       to STIS exposures.
    """

    # Default coefficients table to use for this instrument
    IDCKEY = 'cubic'

    DQCLASS = 'STISPars'

    __theta = 0.0
    __parity = drutil.buildRotMatrix(__theta) * N.array([[-1.,1.],[-1.,1.]])
    PARITY = {'CCD':__parity,'NUV-MAMA':__parity,'FUV-MAMA':__parity}

    # The dictionaries 'REFDATA' and 'REFPIX' are required for use with
    # cubic and Trauger coefficients tables in 'computeCubicCoeffs'.
    #
    # This information provides the absolute relationship between the chips
    # Latest plate scales: 0.05071, 0.0246
    REFDATA = {'CCD':{'psize':0.05,'xoff':0.0,'yoff':0.0,'v2':-213.999,'v3':-224.897,'theta':__theta},
              'NUV-MAMA':{'psize':0.024,'xoff':0.0,'yoff':0.0,'v2':-213.999,'v3':-224.897,'theta':__theta},
              'FUV-MAMA':{'psize':0.024,'xoff':0.0,'yoff':0.0,'v2':-213.999,'v3':-224.897,'theta':__theta}}
    REFPIX = {'x':512.,'y':512.}

    def __init__(self, filename, output, pars=None):

        # Now initialize Pattern with all member exposures...
        Pattern.__init__(self, filename, output=output, pars=pars)

        self.instrument = 'STIS'

        # build output rootnames here...
        self.setNames(filename,output)

        # Set EXPTIME for exposure
        self.exptime = self.getExptime()

        if self.pars['section'] == None:
            self.nmembers = int(self.header['NEXTEND']) / self.NUM_IMSET
        else:
            # If a section was specified, then there is only 1 member
            # for this observation.
            self.nmembers = 1

        # Now, build list of members and initialize them
        self.addMembers(filename)

        if self.members[0].geometry.ikey != 'idctab':
            # Correct distortion coefficients to match output pixel scale
            self.computeCubicCoeffs()
        else:
            self.computeOffsets()

        # Set up the input members and create the product meta-chip
        self.buildProduct(filename, output)

    def getExptime(self):

        header = fileutil.getHeader(self.name+'[sci,1]')
        _exptime = float(header['EXPTIME'])
        if _exptime == 0.: _exptime = 1.0

        if header.has_key('EXPSTART'):
            _expstart = float(header['EXPSTART'])
            _expend = float(header['EXPEND'])
        else:
            _expstart = 0.
            _expend = _exptime

        return (_exptime,_expstart,_expend)




class NICMOSObservation(Pattern):
    """This class defines an observation with information specific
       to NICMOS exposures.
    """

    # Default coefficients table to use for this instrument
    IDCKEY = 'cubic'

    DQCLASS = 'NICMOSPars'

    DETECTOR_NAME = 'camera'
    NUM_IMSET = 5

    __theta = 0.0
    __parity = drutil.buildRotMatrix(__theta) * N.array([[-1.,1.],[-1.,1.]])
    PARITY = {'1':__parity,'2':__parity,'3':__parity}

    # The dictionaries 'REFDATA' and 'REFPIX' are required for use with
    # cubic and Trauger coefficients tables in 'computeCubicCoeffs'.
    #
    # This information provides the absolute relationship between the chips
    # Latest plate scales: 0.05071, 0.0246
    REFDATA = {'1':{'psize':0.0432,'xoff':0.0,'yoff':0.0,'v2':-296.9228,'v3':290.1827,'theta':__theta},
              '2':{'psize':0.076,'xoff':0.0,'yoff':0.0,'v2':-319.9464,'v3':311.8579,'theta':__theta},
              '3':{'psize':0.0203758,'xoff':0.0,'yoff':0.0,'v2':-249.8170,'v3':235.2371,'theta':__theta}}
    REFPIX = {'x':128.,'y':128.}

    def __init__(self, filename, output, pars=None):

        # Now initialize Pattern with all member exposures...
        Pattern.__init__(self, filename, output=output, pars=pars)

        self.instrument = 'NICMOS'

        # build output rootnames here...
        self.setNames(filename,output)

        # Set EXPTIME for exposure
        self.exptime = self.getExptime()

        if self.pars['section'] == None:
            self.nmembers = int(self.header['NEXTEND']) / self.NUM_IMSET
        else:
            # If a section was specified, then there is only 1 member
            # for this observation.
            self.nmembers = 1

        # Now, build list of members and initialize them
        self.addMembers(filename)

        if self.members[0].geometry.ikey != 'idctab':
            # Correct distortion coefficients to match output pixel scale
            self.computeCubicCoeffs()
        else:
            self.computeOffsets()

        # Set up the input members and create the product meta-chip
        self.buildProduct(filename, output)


class WFPCObservation(Pattern):
    """This class defines an observation with information specific
       to WFPC2 exposures, including knowledge of how to mosaic the
       chips."""

    # Default coefficients table to use for this instrument
    IDCKEY = 'cubic'
    DQCLASS = 'WFPC2Pars'

    # This parity is the multiplication of PC1 rotation matrix with
    # a flip in X for output image.
    #__theta = 44.67
    __pmat = N.array([[-1.,0.],[0.,1.]])
    __refchip = 3
    PARITY = {'1':__pmat,'2':__pmat,'3':__pmat,'4':__pmat,'WFPC':__pmat}

    NUM_IMSET = 1

    # The dictionaries 'REFDATA' and 'REFPIX' are required for use with
    # cubic and Trauger coefficients tables in 'computeCubicCoeffs'.
    #
    # This information provides the absolute relationship between the chips
    #REFDATA ={'1':{'psize':0.04554,'xoff':354.356,'yoff':343.646,'theta':__theta},
    #          '2':{'psize':0.0996,'xoff':-371.27125,'yoff':350.50803,'theta':314.388},
    #          '3':{'psize':0.0996,'xoff':-369.014827,'yoff':-352.74708,'theta':44.698},
    #          '4':{'psize':0.0996,'xoff':353.39406,'yoff':-354.18689,'theta':135.258}}
    REFDATA ={'1':{'psize':0.04554,'xoff':354.356,'yoff':343.646,'v2':2.374,'v3':-30.268,'theta':224.8480},
              '2':{'psize':0.0996,'xoff':345.7481,'yoff':375.28818,'v2':-51.368,'v3':-5.698,'theta':314.3520},
              '3':{'psize':0.0996,'xoff':366.56876,'yoff':354.79435,'v2':0.064,'v3':48.692,'theta':44.67},
              '4':{'psize':0.0996,'xoff':355.85016,'yoff':351.29183,'v2':55.044,'v3':-6.098,'theta':135.2210}}
    REFPIX = {'x':400.,'y':400.}

    def __init__(self, filename, output, pars=None):

        # Now initialize Pattern with all member exposures...
        Pattern.__init__(self, filename, output=output, pars=pars)

        self.instrument = 'WFPC2'

        gcount = None

        # build output rootnames here...
        self.setNames(filename,output)

        # Determine how many 'chips' make up the observation
        if self.pars['section'] == None:
            if gcount == None:
                self.nmembers = int(self.header['NEXTEND']) / self.NUM_IMSET
            else:
                self.nmembers = int(gcount)
        else:
            # If a section was specified, then there is only 1 member
            # for this observation.
            self.nmembers = 1

        # Set EXPTIME for exposure
        self.exptime = self.getExptime()

        # Now, build list of members and initialize them
        self.addMembers(filename)

        if self.members[0].geometry.ikey != 'idctab':
            # Correct distortion coefficients to match output pixel scale
            self.computeCubicCoeffs()
        else:
            self.computeOffsets(refchip=self.__refchip)

        # Determine desired orientation of product
        self.setOrient()

        # Set up the input members and create the product meta-chip
        self.buildProduct(filename, output)


    def addMembers(self,filename):

        # The PC chip defines the orientation of the metachip, so use
        # it for the PARITY as well.
        self.detector = 'WFPC'

        _chip1_rot = None
        # Build rootname here for each SCI extension...
        for i in range(self.nmembers):
            _extname = self.imtype.makeSciName(i+1,section=self.pars['section'])

            _detnum = fileutil.getKeyword(_extname,self.DETECTOR_NAME,handle=self.image_handle)

            # Start by looking for the corresponding WFPC2 'c1h' files
            _dqfile = self._findDQFile()

            # Reset dqfile name in ImType class to point to new file
            self.imtype.dqfile = _dqfile

            # Set the DQ extname to that used by WFPC2 C1H images
            if _dqfile.find('.fits') > 0:
                self.imtype.dq_extname = 'sdq'
                self.imtype.dq_extn = '[sdq,1]'

            # Build mask file for this member chip
            _dqname = self.imtype.makeDQName(extver=_detnum)

            _maskname = buildmask.buildShadowMaskImage(_dqname,_detnum,bitvalue=self.bitvalue)

            self.members.append(Exposure(_extname, idckey=self.idckey, dqname=_dqname,
                mask=_maskname, parity=self.PARITY[str(i+1)],
                idcdir=self.pars['idcdir'], dateobs=self.dateobs,
                rot=_chip1_rot, handle=self.image_handle, extver=_detnum))

            if self.idckey != 'idctab':
                _chip1_rot = self.members[0].geometry.def_rot

    def _findDQFile(self):
        """ Find the DQ file which corresponds to the input WFPC2 image. """
        if self.name.find('.fits') < 0:
            # Working with a GEIS image...
            dqfile = self.name[:-2]+'1h'
        else:
            # Looking for c1f FITS DQ file...
            dqfile = self.name.replace('0h.fits','1h.fits')

        return dqfile

    def setOrient(self):

        """ Determine desired orientation of product."""
        meta_orient = None
        for exp in self.members:
            if int(exp.chip) == 1:
                meta_orient = exp.geometry.wcslin.orient

        if meta_orient == None:
            meta_orient = self.members[0].geometry.wcslin.orient

        # Set orient for all groups
        # Dither coefficients rotate all chips to chip 1 orientation
        # if not using Trauger coefficients
        for exp in self.members:
            exp.geometry.wcs.orient = meta_orient


class DitherProduct(Pattern):
    """
    Builds an object for a set of dithered inputs, each of which
    will be one of the Observation objects.
    """
    def __init__(self, prodlist, pars=None):

        # Build temporary output drizzle product name
        output = fileutil.buildNewRootname(string.lower(prodlist['output']),extn='_drz.fits')

        # Setup a default exposure to contain the results
        Pattern.__init__(self, None, output=output, pars=pars)

        self.pars = prodlist['members']
        self.nmembers = len(prodlist['members'])
        self.offsets = None

        self.addMembers(prodlist,pars,output)

        if len(self.members) == 0:
            print 'No suitable inputs from ASN table. Quitting PyDrizzle...'
            raise Exception

        self.exptime = self.getExptime()

        self.buildProduct(output)

    def closeHandle(self):
        """ Close image handle for each member."""
        for member in self.members:
            member.closeHandle()

    def buildProduct(self,output):
        # Build default Metachip based on unmodified header WCS values
        output_wcs = self.buildMetachip()

        self.size = (output_wcs.naxis1,output_wcs.naxis2)
        self.product = Exposure(output,wcs=output_wcs,new=yes)

        # Compute default offsets between all inputs...
        # This will be used when applying relative shifts from ASN table
        # or shiftfile.
        self.computeOffsets()

        # Apply any additional shifts from ASN table/shiftfile
        self.applyAsnShifts(output)

        # Preserve default DitherProduct Metachip WCS as wcslin
        self.product.exptime = self.exptime
        # Update the corners arrays for the product now...
        self.product.setCorners()
        #self.product.corners['corrected'] = self.product.corners['raw']
        _prodcorners = []
        for prod in self.members:
            _prodcorners +=  prod.product.corners['corrected'].tolist()
        self.product.corners['corrected'] = N.array(_prodcorners,type=N.Float64)

    def applyAsnShifts(self,output):
        """
        Apply any shifts read in from the ASN table /shiftfile to
        the WCS of each input's product.

        In the case there are no shifts to apply, copy the product
        WCS into WCSLIN as the default case.

        This method should (eventually?) support updating the shifts
        without starting from the beginning with these datasets, similar
        to the 'resetPars' functionality.

        """
        # If there were any shifts to be applied, update input
        # image's WCS with the shifts, then re-build the final
        # product Metachip using corrected product WCS values.
        if self.pars['abshift'] or self.pars['dshift']:
            for prod in self.members:
                prod.applyAsnShifts()

            #If we have shifts of any sort,
            # recompute the DitherProduct's meta-chip with corrected values
            output_wcs = self.buildMetachip()
            self.size = (output_wcs.naxis1,output_wcs.naxis2)
            self.product = Exposure(output,wcs=output_wcs,new=yes)

        self.product.geometry.wcslin = self.product.geometry.wcs.copy()

    def computeOffsets(self):
        """
        This method will rely on final product's 'rd2xy' method
        to compute offsets between the different input chips.
        """
        ref_wcs = self.product.geometry.wcs
        ref_crv= (ref_wcs.crval1,ref_wcs.crval2)

        _tot_ref = {'pix_shift':(0.,0.),'ra_shift':(0.,0.),
                    'val':999999999.,'name':''}

        _member_num = 0
        for member in self.members:
            in_wcs = member.product.geometry.wcs

            x,y = ref_wcs.rd2xy((in_wcs.crval1,in_wcs.crval2))
            xoff = x - ref_wcs.crpix1
            yoff = y - ref_wcs.crpix2

            raoff = (in_wcs.crval1 - ref_wcs.crval1,in_wcs.crval2 - ref_wcs.crval2)

            _delta_rot = in_wcs.orient - ref_wcs.orient
            # Determine which image has the smallest offset
            _tot = N.sqrt(N.power(xoff,2)+N.power(yoff,2))

            # Use first image as reference
            if _member_num == 0:
                _tot_ref['val'] = _tot
                _tot_ref['pix_shift'] = (xoff,yoff)
                _tot_ref['name'] = member.rootname
                _tot_ref['ra_shift'] = raoff
                # This will be used primarily for manual verification
                _tot_ref['wcs'] = in_wcs

            _member_num += 1

            # Keep track of the results for later use/comparison
            member.offsets = {'pixels':(xoff,yoff),'arcseconds':raoff}

        for member in self.members:
            member.refimage = _tot_ref


    def getExptime(self):
        """
        Add up the exposure time for all the members in
        the pattern, since 'drizzle' doesn't have the necessary
        information to correctly set this itself.
        """
        _exptime = 0.
        _start = []
        _end = []
        for member in self.members:
            _memexp = member.product.exptime
            _exptime = _exptime + _memexp[0]
            _start.append(_memexp[1])
            _end.append(_memexp[2])
        _expstart = min(_start)
        _expend = max(_end)

        return (_exptime,_expstart,_expend)

    def addMembers(self,prodlist,pars,output):
        """
        For each entry in prodlist, append the appropriate type
        of Observation to the members list.
        """
        for memname in prodlist['order']:
            pardict = self.pars[memname]
            pardict.update(pars)
            filename = fileutil.buildRootname(memname)
            if filename:
                self.members.append(selectInstrument(filename,output,pars=pardict))
            else:
                print 'No recognizable input! Not building parameters for ',memname


    def buildPars(self,ref=None):

        # If an output field is specified, update product WCS to match
        if ref != None:
            _field = ref
            _field.mergeWCS(self.product.geometry.wcslin)
            #_field.exptime = self.exptime

            # Update product WCS based on input
            self.transformMetachip(_field)
            #_field.mergeWCS(self.product.geometry.wcs)
        else:
            _field = SkyField()
            _field.mergeWCS(self.product.geometry.wcslin)
            # Reset Product WCS to reflect use of default WCS
            self.product.geometry.wcs = self.product.geometry.wcslin.copy()
        _field.exptime = self.exptime
        # Specify that this product comes represents a dither product
        # not just a single observation product.
        _field.dither = yes

        parlist = []
        for member in self.members:
            parlist = parlist + member.buildPars(ref=_field)

        return parlist

    def buildMetachip(self):
        """
        This method combines the results of the member's buildMetachip()
        methods into a composite WCS.
        (DitherProduct method)
        """
        prodlist = []

        _ref = self.members[0].product.geometry

        # Get pscale from model
        pscale = _ref.model.pscale

        for member in self.members:
            # merge the corrected shapes into a corrected meta-chip here
            # Start by computing the corected positions for reference points
            prodlist.append(member.product)

        # Use first input image WCS as initial reference WCS
        meta_wcs = _ref.wcs.copy()

        ref_crval = (meta_wcs.crval1,meta_wcs.crval2)
        ref_crpix = (meta_wcs.crpix1,meta_wcs.crpix2)

        # Specify the output orientation angle
        ref_orient = meta_wcs.orient

        _delta_x = _delta_y = 0.0
        _xr,_yr,_dpx,_dpy = [],[],[],[]
        # Compute offsets for each input relative to this reference WCS
        for member in prodlist:
            _wcs = member.geometry.wcs

            # Rigorously compute the orientation changes from WCS
            # information using algorithm provided by R. Hook from WDRIZZLE.
            abxt,cdyt = drutil.wcsfit(member.geometry, meta_wcs)

            # Compute the rotation between input and reference from fit coeffs.
            _angle = RADTODEG(N.arctan2(abxt[1],cdyt[0]))
            _dpos = (abxt[2],cdyt[2])

            _delta_x += _dpos[0]
            _delta_y += _dpos[1]
            _dpx.append(_dpos[0])
            _dpy.append(_dpos[1])

            # Compute the range of pixels each input spans in the
            # output meta-frame coordinates
            # Each product in this list could have a different plate scale.
            # apply
            _scale = meta_wcs.pscale / _wcs.pscale

            # Now, account for any rotation difference between
            # input and output frames
            #_angle = meta_wcs.orient - _wcs.orient
            _xrange,_yrange = drutil.getRotatedSize(member.corners['corrected'],_angle)

            #_range = [(_xrange[1] - _xrange[0]),(_yrange[1] - _yrange[0])]
            #_range[0] *= _scale
            #_range[1] *= _scale

            _xr.append(_dpos[0] + _xrange[0]*_scale)
            _xr.append(_dpos[0] + _xrange[1]*_scale)
            _yr.append(_dpos[1] + _yrange[0]*_scale)
            _yr.append(_dpos[1] + _yrange[1]*_scale)

        # Determine the full size of the metachip
        _xmin = N.minimum.reduce(_xr)
        _ymin = N.minimum.reduce(_yr)
        _xmax = N.maximum.reduce(_xr)
        _ymax = N.maximum.reduce(_yr)

        _dxmin = N.minimum.reduce(_dpx)
        _dymin = N.minimum.reduce(_dpy)
        _dxmax = N.maximum.reduce(_dpx)
        _dymax = N.maximum.reduce(_dpy)

        _nimg = len(prodlist)
        _delta_x /= _nimg
        _delta_y /= _nimg

        # Computes the offset from center of the overall set of shifts
        # as applied to the pixels.  This insures that the visible pixels
        # are centered in the output.
        _delta_dx = ((_xmax - _delta_x) + (_xmin - _delta_x))/2.
        _delta_dy = ((_ymax - _delta_y) + (_ymin - _delta_y))/2.

        if _xmin > 0.: _xmin = 0.
        if _ymin > 0.: _ymin = 0.

        # Need to account for overall offset in images.
        # by adding in average offset induced by shifts to
        # output size: delta_dx/dy.
        # This accounts for relative offsets that are not centered
        # on the final output image.
        #xsize = int(_xmax - _xmin + 2.0 + abs(_delta_dx) )
        #ysize = int(_ymax - _ymin + 2.0 + abs(_delta_dy) )
        xsize = int(_xmax - _xmin)
        ysize = int(_ymax - _ymin)

        # Compute difference between new size and
        # original size of meta_wcs frame.
        _dxsize = xsize - meta_wcs.naxis1
        _dysize = ysize - meta_wcs.naxis2

        # Update reference WCS for new size
        meta_wcs.naxis1 =  xsize
        meta_wcs.naxis2 =  ysize
        _cen = (float(xsize)/2.,float(ysize)/2.)

        # Determine offset of centers from center of image.
        # d[x/y]size  : difference between input and output frame sizes
        # delta_[x/y] : average of all shifts applied to input images
        # delta_d[x/y]: amount off-center of all shifts
        #
        # Need to take into account overall offset in shifts, such that
        # minimum, not maximum, shift is always 0.0. This is needed to allow
        # subarrays to align properly with full frame observations without
        # introducing an overall offset to the output.  WJH 13 Sept 2004.
        #
        _nref = ( _dxsize/2. - _delta_x - _delta_dx, _dysize/2. - _delta_y - _delta_dy)

        # Have to adjust the CRPIX by how much the center needs to shift
        # to fit into the reference frame.
        meta_wcs.crpix1 = meta_wcs.crpix1 + _nref[0]
        meta_wcs.crpix2 = meta_wcs.crpix2 + _nref[1]
        meta_wcs.recenter()

        return meta_wcs

#
#
# Set up default pars dictionary for external calls
_default_pars = {'psize':None,'rot':None,'idckey':None}

def selectInstrument(filename,output,pars=_default_pars):
    """
    Method which encapsulates the logic for determining
    which class to instantiate for each file.

    """
    # Determine the instrument...
    instrument = fileutil.getKeyword(filename+'[0]','INSTRUME')

    #try:
    # ... then create an appropriate object.
    if instrument == INSTRUMENT[0]:
        member = ACSObservation(filename,output,pars=pars)
    elif instrument == INSTRUMENT[1]:
        member = WFPCObservation(filename,output,pars=pars)
    elif instrument == INSTRUMENT[2]:
        member = STISObservation(filename,output,pars=pars)
    elif instrument == INSTRUMENT[3]:
        member = NICMOSObservation(filename,output,pars=pars)
    else:
        #raise AttributeError, "Instrument '%s' not supported now."%instrument
        member = GenericObservation(filename,output,pars=pars)
    #except:
    #    raise IOError,"Image %s could not be processed."%filename

    return member

class SkyField:
    """
    An class for specifying the parameters and building a WCS object
        for a user-specified drizzle product.
    The user may optionally modify the values for:
        psize       - size of image's pixels in arc-seconds
        orient      - value of ORIENTAT for the field
        shape       - tuple containing the sizes of the field's x/y axes
        ra,dec      - position of the center of the field
                      decimal (124.5678) or
                      sexagesimal string _in quotes_ ('hh:mm:ss.ss')
        crpix       - tuple for pixel position of reference point in
                        output image
    Usage:
        To specify a new field with a fixed output size of 1024x1024:
        --> field = pydrizzle.SkyField(shape=(1024,1024))

        The 'set()' method modifies one of the parameters listed above
        without affecting the remainder of the parameters.
        --> field.set(psize=0.1,orient=0.0)
        --> field.set(ra=123.45678,dec=0.1000,crpix=(521,576))

        View the WCS or user-specified values for this object:
        --> print field

    """
    def __init__(self,shape=None,psize=None,wcs=None):

        self.shape = shape
        self.ra = None
        self.dec = None
        self.orient = None
        self.psize = psize

        self.crpix = (None,None)

        # Set up proper shape tuple for WCSObject
        if shape != None and psize != None:
            wshape = (shape[0],shape[1],psize)
        else:
            wshape = None
            if wcs:
                self.crpix = (wcs.crpix1,wcs.crpix2)

        # Specify 'new=yes' with given rootname to unambiguously create
        #   a new WCS from scratch.
        if wcs == None:
            self.wcs = wcsutil.WCSObject("New",shape=wshape,new=yes)
        else:
            self.wcs = wcs.copy()
            self.wcs.recenter()
            self.set(shape=wshape)

        # Set this to keep track of total exposure time for
        # output frame... Not user settable.
        self.exptime = None
        # Keep track of whether this is for a Dither product or not
        self.dither = None

    def mergeWCS(self,wcs,overwrite=yes):
        """ Sets up the WCS for this object based on another WCS.
            This method will NOT update object attributes other
            than WCS, as all other attributes reflect user-settings.
        """
        #
        # Start by making a copy of the input WCS...
        #
        if self.wcs.rootname == 'New':
            self.wcs = wcs.copy()
        else:
            return
        self.wcs.recenter()

        if self.ra == None:
            _crval = None
        else:
            _crval = (self.ra,self.dec)

        if self.psize == None:
            _ratio = 1.0
            _psize = None
            # Need to resize the WCS for any changes in pscale
        else:
            _ratio = wcs.pscale / self.psize
            _psize = self.psize

        if self.orient == None:
            _orient = None
            _delta_rot = 0.
        else:
            _orient = self.orient
            _delta_rot = wcs.orient - self.orient

        _mrot = drutil.buildRotMatrix(_delta_rot)

        if self.shape == None:
            _corners = N.array([[0.,0.],[wcs.naxis1,0.],[0.,wcs.naxis2],[wcs.naxis1,wcs.naxis2]])
            _corners -= (wcs.naxis1/2.,wcs.naxis2/2.)
            _range = drutil.getRotatedSize(_corners,_delta_rot)
            shape = ((_range[0][1] - _range[0][0])*_ratio,(_range[1][1]-_range[1][0])*_ratio)
            old_shape = (wcs.naxis1*_ratio,wcs.naxis2*_ratio)

            _cen = (shape[0]/2., shape[1]/2.)

            #if _delta_rot == 0.:
            #    _crpix = (self.wcs.crpix1,self.wcs.crpix2)
            #else:
                # Rotate original scaled crpix position to new orientation
                #_crpix = N.dot((wcs.crpix1*_ratio - _cen[0],wcs.crpix2*_ratio -_cen[1]),_mrot)+_cen
            _crpix = _cen
        else:
            shape = self.shape
            if self.crpix == None:
                _crpix = (self.shape[0]/2.,self.shape[1]/2.)
            else:
                _crpix = self.crpix

        # Set up the new WCS based on values from old one.
        self.wcs.updateWCS(pixel_scale=_psize,orient=_orient,refpos=_crpix,refval=_crval)
        self.wcs.naxis1 =  int(shape[0])
        self.wcs.naxis2 =  int(shape[1])

    def set(self,psize=None,orient=None,ra=None,dec=None,
            shape=None,crpix=None):
        """
        Modifies the attributes of the SkyField and
            updates it's WCS when appropriate.
        """
        # Converts(if necessary), then updates the RA and Dec.
        _ra,_dec = None,None
        if ra != None:
            if string.find(repr(ra),':') > 0:
                _hms = string.split(repr(ra)[1:-1],':')
                if _hms[0][0] == '-': _sign = -1
                else: _sign = 1

                for i in range(len(_hms)): _hms[i] = float(_hms[i])
                _ra = _sign * (_hms[0] + ((_hms[1] + _hms[2]/60.) / 60.)) * 15.
            else:
                _ra = float(ra)
            self.ra = _ra

        if dec != None:
            if string.find(repr(dec),':') > 0:
                _dms = string.split(repr(dec)[1:-1],':')
                if _dms[0][0] == '-': _sign = -1
                else: _sign = 1

                for i in range(len(_dms)): _dms[i] = float(_dms[i])
                _dec = _sign * (_dms[0] + ((_dms[1] + _dms[2]/60.) / 60.))
            else:
                _dec = float(dec)
            self.dec = _dec

        if self.ra != None and self.dec != None:
            _crval = (self.ra,self.dec)
        else:
            _crval = None

        # Updates the shape, and reference position,
        # only if a new value is specified.
        _crpix = None
        if crpix == None:
            if shape != None:
                self.shape = shape
                _crpix = (self.shape[0]/2.,self.shape[1]/2.)
        else:
            _crpix = crpix

        self.crpix=_crpix

        if psize != None:
            self.psize = psize

        if orient != None:
            self.orient = orient

        # Updates the WCS with all the changes, if there is enough info.
        self.wcs.updateWCS(pixel_scale=psize,orient=orient,refpos=_crpix,
                                refval=_crval,size=self.shape)

    def __str__(self):
        """ Prints the WCS information set for this object.
        """
        if self.psize != None and self.orient != None:
            block =  self.wcs.__str__()
        else:
            block = 'User parameters for SkyField object: \n'
            block = block + '    psize = '+repr(self.psize)+' \n'
            block = block + '   orient = '+repr(self.orient)+' \n'
            block = block + '    shape = '+repr(self.shape)+' \n'
            block = block + '       ra = '+repr(self.ra)+' \n'
            block = block + '      dec = '+repr(self.dec)+' \n'
            block = block + '   No WCS.\n'

        return block

    def help(self):
        """ Creates and prints usage information for this class.
        """
        print self.__doc__

class PyDrizzle:
    """
Program to process and/or dither-combine image(s) using (t)drizzle.
To create an object named 'test' that corresponds to a drizzle product:
    --> test = pydrizzle.PyDrizzle(input)
where input is the FULL filename of an ACS observation or ASN table.
This computes all the parameters necessary for running drizzle on all
the input images.  Once this object is created, you can run drizzle using:
    --> test.run()

The 'clean()' method can be used to remove files which would interfere with
running Drizzle again using the 'run()' method after a product has already
been created.

Optional parameters:
    output      User-specified name for output products
    field       User-specified parameters for output image
                includes: psize, orient, ra, dec, shape
    units       Units for final product: 'counts' or 'cps'(DEFAULT)
    section     Extension/group to be drizzled: FITS extension or group
                syntax ('1' or 'sci,1') or None (DEFAULT: Use all chips).
    kernel      Specify which kernel to use in TDRIZZLE
                'square'(default),'point','gaussian','turbo','tophat'
    pixfrac     drizzle pixfrac value (Default: 1.0)
    idckey      User-specified keyword for determining IDCTAB filename
                'IDCTAB'(ACS default),'TRAUGER'(WFPC2),'CUBIC'(WFPC2)
    idcdir      User-specified directory for finding coeffs files:
                'drizzle$coeffs' (default)

Optional Parameters for '.run()':
    build       create multi-extension output: yes (Default) or no
    save        keeps the individual inputs from drizzle: yes or no (Default)
    single      drizzle to separate output for each input: yes or no (Default)
    blot        run blot on drizzled products: yes or no (Default)
    clean       remove coeffs and static mask files: yes or no (Default)

Optional Parameters for '.clean()':
    coeffs      Removes coeffs and static mask files: yes or no (Default)
    final       Removes final product: yes or no (Default)

Usage of optional parameters:
    --> test = pydrizzle.PyDrizzle('test_asn.fits',units='counts')
To keep the individual 'drizzle' output products:
    --> test.run(save=yes)

Output frame parameters can be modified 'on-the-fly' using 'resetPars'.
Given an already drizzled image 'refimg_drz.fits' as a reference,
reset drizzle parameters using:
    --> wcsref = pydrizzle.wcsutil.WCSObject('refimg_drz.fits[sci,1]')
    --> f = pydrizzle.SkyField(wcs=wcsref)
Use either:
    --> test.resetPars(wcsref)
Or:
    --> test.resetPars(f)
Return to default parameters using no parameters at all:
    --> test.resetPars()
More help on SkyField objects and their parameters can be obtained using:
    --> f.help()
    """
    def __init__(self, input, output=None, field=None, units=None, section=None,
        kernel=None,pixfrac=None,bits=None,wt_scl='exptime',fillval=0.,idckey=None,
        idcdir=DEFAULT_IDCDIR,memmap=1,dqsuffix=None,prodonly=yes):

        if idcdir == None: idcdir = DEFAULT_IDCDIR

        print 'Starting PyDrizzle Version ',__version__,' at ', _ptime()

        # Do a quick sanity check on the input filename
        # Can it be found in the current directory?
        if not fileutil.findFile(input):
            raise IOError,'Can not find input file in current directory!'

        # Does it have a recognizable extension
        if string.find(input,'.') < 0:
            raise ValueError,"Please specify extension (i.e., .fits) for input '%s' "%input
        # These parameters are needed for buildPars()
        self.input = input

        # Extract user-specified parameters, if any have been set...
        # 'field' will be a SkyField object...
        if field != None:
            psize = field.psize
            orient = field.orient
        else:
            psize = None
            orient = None

        # These can also be set by the user.
        # Minimum set needed: psize, rot, and idckey
        # bitvalue handled by DQPars classes.
        self.pars = {'psize':psize,'units':units,'kernel':kernel,'rot':orient,
            'pixfrac':pixfrac,'idckey':idckey,'wt_scl':wt_scl,
            'fillval':fillval,'section':section, 'idcdir':idcdir+os.sep,
            'memmap':memmap,'dqsuffix':dqsuffix, 'bits':bits}

        # Check to see if user-supplied output name is complete
        # Append .FITS suffix to output name if necessary
        self.output = output
        if output != None:
            _indx = string.find(output,'.')
            if _indx < 0: output = output + '.fits'

        # Watch out for any errors.
        # If they arise, all open files need to be closed...
        self.observation = None
        self.parlist = None
        #try:

        # Decipher input to determine whether we are working with
        # an ASN table or single image...
        #
        # Should this check for ASN table and
        if input.find('_asn') < 0 and input.find('_asc') < 0:
            # Start with single image case...
            # Check to see if an output name was provided.
            if output == None:
                # We need to build a default output name...
                output = fileutil.buildNewRootname(input,extn='_drz.fits')
                print 'Setting up default output name: ',output

            self.observation = selectInstrument(input,output,pars=self.pars)
        else:
            # We are dealing with an ASN table...
            # 'input' - full filename for ASN table
            asndict = fileutil.readAsnTable(input,output=output,prodonly=prodonly)
            # Build output filename
            if output == None:
                output = fileutil.buildNewRootname(asndict['output'],extn='_drz.fits')
                print 'Setting up output name: ',output

            if len(asndict['members'].keys()) > 1:
                self.observation = DitherProduct(asndict,pars=self.pars)
            else:
                inroot = asndict['members'].keys()[0]
                pardict = asndict['members'][inroot]
                infile = fileutil.buildRootname(inroot)
                if infile == None:
                    raise IOError,'No product found for association table.'
                # Append user specified parameters to shifts dictionary
                pardict.update(self.pars)
                self.observation = selectInstrument(infile,output,pars=pardict)
        self.output = output

        # This call puts together the parameters for the input image
        # with those for the output to create a final parameter list
        # for running 'drizzle'.
        # It relies on the buildPars() methods for each exposure to
        # generate a complete set of parameters for all inputs
        #
        self.parlist = self.observation.buildPars(ref=field)

        #finally:
            # Something went wrong, so we need to make sure all file
            # handles get closed...
            #self.close()
            #print 'PyDrizzle could not initialize due to errors.'
        self.observation.closeHandle()

        # Let the user know parameters have been successfully calculated
        print 'Drizzle parameters have been calculated. Ready to .run()...'
        print 'Finished calculating parameters at ',_ptime()

    def clean(self,coeffs=no,final=no):
        """ Removes intermediate products from disk. """
        for img in self.parlist:
            fileutil.removeFile([img['outdata'],img['outcontext'],img['outweight']])
            fileutil.removeFile([img['outsingle'],img['outsweight']])
            #fileutil.removeFile([img['outsingle'],img['outsweight'],img['outscontext']])
            fileutil.removeFile(img['outblot'])
            if coeffs:
                os.remove(img['coeffs'])
                if img['driz_mask'] != '':
                    fileutil.removeFile(img['driz_mask'])
        if final:
            fileutil.removeFile(self.output)


    # Run 'drizzle' here...
    #
    def run(self,save=no,build=yes,blot=no,single=no,clean=no,interp='linear',sinscl=1.0):
        """Perform drizzle operation on input to create output.
         This method would rely on the buildPars() method for
         the output product to produce correct parameters
         based on the inputs. The output for buildPars() is a list
         of dictionaries, one for each input, that matches the
         primary parameters for an IRAF drizzle task.

         This method would then loop over all the entries in the
         list and run 'drizzle' for each entry. """

        print 'PyDrizzle: drizzle task started at ',_ptime()
        _memmap = self.pars['memmap']

        # Check for existance of output file.
        if single == no and build == yes and fileutil.findFile(self.output):
            print 'Removing previous output product...'
            os.remove(self.output)

        # Set parameters for each input and run drizzle on it here.

        if blot:
            #
            # Run blot on data...
            #

            plist = self.parlist[0]
            _insci = N.zeros((plist['outny'],plist['outnx']),N.Float32)
            _outsci = N.zeros((plist['blotny'],plist['blotnx']),N.Float32)
            _hdrlist = []

            for plist in self.parlist:

                _hdrlist.append(plist)
                # Open input image as PyFITS object
                if plist['outsingle'] != plist['outdata']:
                    _data = plist['outsingle']
                else:
                    _data = plist['outdata']

                # PyFITS can be used here as it will always operate on
                # output from PyDrizzle (which will always be a FITS file)
                # Open the input science file
                _fname,_sciextn = fileutil.parseFilename(_data)
                _inimg = fileutil.openImage(_fname)

                # Return the PyFITS HDU corresponding to the named extension
                _scihdu = fileutil.getExtn(_inimg,_sciextn)
                _insci = _scihdu.data.copy()

                # Read in the distortion correction arrays, if specified
                _pxg,_pyg = plist['exposure'].getDGEOArrays()

                # Now pass numarray objects to callable version of Blot...
                #runBlot(plist)
                build=no
                misval = 0.0
                kscale = 1.0

                xmin = 1
                xmax = plist['outnx']
                ymin = 1
                ymax = plist['outny']
                #
                # ARRDRIZ.TBLOT needs to be updated to support 'poly5' interpolation,
                # and exptime scaling of output image.
                #
                """
                #
                # This call to 'arrdriz.tdriz' uses the F2PY syntax
                #
                arrdriz.tblot(_insci, _outsci,xmin,xmax,ymin,ymax,
                            plist['xsh'],plist['ysh'],
                            plist['rot'],plist['scale'], kscale, _pxg, _pyg,
                            'center',interp, plist['coeffs'], plist['exptime'],
                            misval, sinscl, 1)
                #
                # End of F2PY syntax
                #
                """
                #
                # This call to 'arrdriz.tdriz' uses the F2C syntax
                #
                t = arrdriz.tblot(_insci, _outsci,xmin,xmax,ymin,ymax,
                            plist['xsh'],plist['ysh'],
                            plist['rot'],plist['scale'], kscale, _pxg, _pyg,
                            'center',interp, plist['coeffs'], plist['exptime'],
                            misval, sinscl, 1)

                #
                # End of F2C syntax
                #

                # Write output Numarray objects to a PyFITS file
                # Blotting only occurs from a drizzled SCI extension
                # to a blotted SCI extension...
                _header = fileutil.getHeader(plist['data'])
                _wcs = wcsutil.WCSObject(plist['data'],header=_header)

                _outimg = outputimage.OutputImage(_hdrlist, build=no, wcs=_wcs, blot=yes)
                _outimg.outweight = None
                _outimg.outcontext = None
                _outimg.writeFITS(plist['data'],_outsci,None)

                #_buildOutputFits(_outsci,None,plist['outblot'])
                _insci *= 0.
                _outsci *= 0.
                _inimg.close()
                del _inimg
                _hdrlist = []

                del _pxg,_pyg

            del _insci,_outsci

        else:
            #
            # Perform drizzling...
            #
            _wcs = self.observation.product.geometry.wcs

            _numctx = {'all':len(self.parlist)}
            if single:
                # Determine how many chips make up each single image
                for plist in self.parlist:
                    plsingle = plist['outsingle']
                    if _numctx.has_key(plsingle): _numctx[plsingle] += 1
                    else: _numctx[plsingle] = 1
            #
            # A image buffer needs to be setup for converting the input
            # arrays (sci and wht) from FITS format to native format
            # with respect to byteorder and byteswapping.
            # This buffer should be reused for each input.
            #
            plist = self.parlist[0]
            _outsci = N.zeros((plist['outny'],plist['outnx']),N.Float32)
            _outwht = N.zeros((plist['outny'],plist['outnx']),N.Float32)
            _inwcs = N.zeros([8],N.Float64)

            # Compute how many planes will be needed for the context image.
            _nplanes = int((_numctx['all']-1) / 32) + 1
            # For single drizzling or when context is turned off,
            # minimize to 1 plane only...
            if single or self.parlist[0]['outcontext'] == '':
                _nplanes = 1

            # Always initialize context images to a 3-D array
            # and only pass the appropriate plane to drizzle as needed
            _outctx = N.zeros((_nplanes,plist['outny'],plist['outnx']),N.Int32)

            # Keep track of how many chips have been processed
            # For single case, this will determine when to close
            # one product and open the next.
            _numchips = 0
            _nimg = 0
            _hdrlist = []

            for plist in self.parlist:

                # Read in the distortion correction arrays, if specified
                _pxg,_pyg = plist['exposure'].getDGEOArrays()

                _hdrlist.append(plist)
                # Open the SCI image
                _expname = plist['data']
                _handle = fileutil.openImage(_expname,mode='readonly',memmap=0)
                _fname,_extn = fileutil.parseFilename(_expname)
                _sciext = fileutil.getExtn(_handle,extn=_extn)

                # Make a local copy of SCI array and WCS info
                #_insci = _sciext.data.copy()
                _inwcs = drutil.convertWCS(wcsutil.WCSObject(_fname,header=_sciext.header),_inwcs)

                # Compute what plane of the context image this input would
                # correspond to:
                _planeid = int(_numchips /32)

                # Check to see whether there is a mask_array at all to use...
                if isinstance(plist['driz_mask'],types.StringType):
                    if plist['driz_mask'] != None and plist['driz_mask'] != '':
                        _wht_handle = fileutil.openImage(plist['driz_mask'],mode='readonly',memmap=0)
                        _inwht = _wht_handle[0].data.astype(N.Float32)
                        _wht_handle.close()
                        del _wht_handle
                    else:
                        print 'No weight or mask file specified!  Assuming all pixels are good.'
                        _inwht = N.ones((plist['blotny'],plist['blotnx']),N.Float32)
                elif plist['driz_mask'] != None:
                    _inwht = plist['driz_mask'].astype(N.Float32)
                else:
                    print 'No weight or mask file specified!  Assuming all pixels are good.'
                    _inwht = N.ones((plist['blotny'],plist['blotnx']),N.Float32)

                # Set additional parameters needed by 'drizzle'
                _expin = _wtscl = plist['exptime']
                _in_un = 'counts'
                _shift_fr = 'output'
                _shift_un = 'output'
                _uniqid = _numchips + 1
                ystart = 0
                nmiss = 0
                nskip = 0
                _vers = plist['driz_version']

                _con = yes
                _imgctx = _numctx['all']
                if single or plist['outcontext'] == '':
                    _con = no
                    _imgctx = _numctx[plist['outsingle']]

                """
                #
                # This call to 'arrdriz.tdriz' uses the F2PY syntax
                #
                #_dny = plist['blotny']
                # Call 'drizzle' to perform image combination
                tdriz,nmiss,nskip,_vers = arrdriz.tdriz(
                            _sciext.data,_inwht, _outsci, _outwht,
                            _outctx[_planeid], _con, _uniqid, ystart, 1, 1,
                            plist['xsh'],plist['ysh'], 'output','output',
                            plist['rot'],plist['scale'], _pxg,_pyg,
                            'center', plist['pixfrac'], plist['kernel'],
                            plist['coeffs'], 'counts', _expin,_wtscl,
                            plist['fillval'], _inwcs, 1, nmiss, nskip,_vers)
                #
                # End of F2PY syntax
                #
                """
                #
                # This call to 'arrdriz.tdriz' uses the F2C syntax
                #
                _dny = plist['blotny']
                # Call 'drizzle' to perform image combination
                _vers,nmiss,nskip = arrdriz.tdriz(_sciext.data,_inwht, _outsci, _outwht,
                            _outctx[_planeid], _uniqid, ystart, 1, 1, _dny,
                            plist['xsh'],plist['ysh'], 'output','output',
                            plist['rot'],plist['scale'], _pxg,_pyg,
                            'center', plist['pixfrac'], plist['kernel'],
                            plist['coeffs'], 'counts', _expin,_wtscl,
                            plist['fillval'], _inwcs, nmiss, nskip, 1)
                #
                # End of F2C syntax
                #

                plist['driz_version'] = _vers

                if nmiss > 0:
                    print '! Warning, ',nmiss,' points were outside the output image.'
                if nskip > 0:
                    print '! Note, ',nskip,' input lines were skipped completely.'
                # Close image handle
                _handle.close()
                del _handle,_fname,_extn,_sciext
                del _inwht

                del _pxg,_pyg

                if _nimg == 0:
                    # Only update the WCS from drizzling the
                    # first image in the list, just like IRAF DRIZZLE
                    drutil.updateWCS(_inwcs,_wcs)

                # Increment number of chips processed for single output
                _numchips += 1
                if _numchips == _imgctx:
                    #
                    # Write output arrays to FITS file(s) and reset chip counter
                    #
                    _outimg = outputimage.OutputImage(_hdrlist, build=build, wcs=_wcs, single=single)
                    _outimg.writeFITS(plist['data'],_outsci,_outwht,ctxarr=_outctx)
                    del _outimg
                    #
                    # Reset chip counter for next output image...
                    #
                    _numchips = 0
                    _nimg = 0
                    N.multiply(_outsci,0.,_outsci)
                    N.multiply(_outwht,0.,_outwht)
                    N.multiply(_outctx,0,_outctx)

                    _hdrlist = []
                else:
                    _nimg += 1

            del _outsci,_outwht,_inwcs,_outctx, _hdrlist

        # Remove temp files, if desired
        # Files to be removed are:
        #   parlist['coeffs']
        #   parlist['driz_mask']
        if not save and clean:
            for img in self.parlist:
                fileutil.removeFile(img['coeffs'])
                if img['driz_mask'] != '':
                    fileutil.removeFile(img['driz_mask'])

        print 'PyDrizzle drizzling completed at ',_ptime()


    def resetPars(self,field=None,pixfrac=None,kernel=None,units=None):
        """
        Recompute the output parameters based on a new
            SkyField or WCSObject object.
        """
        if field and not isinstance(field, SkyField):
            if isinstance(field, wcsutil.WCSObject):
                _ref = SkyField(wcs=field)
            else:
                raise TypeError, 'No valid WCSObject or SkyField object entered...'
        else:
            _ref = field
        # Create new version of the parlist with the new values of the
        # parameters based on the reference image.
        new_parlist = self.observation.buildPars(ref=_ref)

        # Copy the parameters from the new parlist into the existing
        # parlist (self.parlist) to preserve any changes/updates made
        # to the parlist externally to PyDrizzle.
        for i in xrange(len(self.parlist)):
            for key in new_parlist[i]:
                self.parlist[i][key] = new_parlist[i][key]
        del new_parlist

        if pixfrac or kernel or units:
            #print 'resetting additional parameter(s)...'
            for p in self.parlist:
                if kernel:
                    p['kernel'] = kernel
                if pixfrac:
                    p['pixfrac'] = pixfrac
                if units:
                    if units == 'counts' or units == 'cps':
                        p['units'] = units
                    else:
                        print 'Units ',units,' not valid! Parameter not reset.'
                        print 'Please use either "cps" or "counts".'

    def help(self):
        """
                Run the module level help function to provide syntax information.
        """
        help()

    def printPars(self,pars='xsh,ysh,rot,scale,outnx,outny,data',format=no):
        """ Prints common parameters for review. """
        if format:
            _title = pars.replace(',','    ')
            print _title
            print '-'*72

        _pars = pars.split(',')
        for pl in self.parlist:
            for _p in _pars: print pl[_p],
            print ''

def _buildOutputFits(sci,wht,fname,ctx=None,extlist=['SCI','ERR','DQ']):

    # Remove previous output products, rather than appending to them
    if fileutil.findFile(fname):
        fileutil.removeFile(fname)

    # Setup primary header as an HDU ready for appending to output FITS file
    prihdu = pyfits.PrimaryHDU()
    scihdu = pyfits.ImageHDU(data=sci,name=extlist[0])
    whthdu = pyfits.ImageHDU(data=wht,name=extlist[1])
    ctxhdu = pyfits.ImageHDU(data=ctx,name=extlist[2])

    fimg = pyfits.open(fname,mode='append')
    fimg.append(prihdu)
    fimg.append(scihdu)
    fimg.append(whthdu)
    fimg.append(ctxhdu)
    fimg.close()

# End of 'runDrizzle'
def help():
    print PyDrizzle.__doc__
