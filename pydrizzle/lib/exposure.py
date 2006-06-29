#   Author:     Warren Hack
#   Program:    exposure.py
#   Purpose:
#   This class will provide the basic functionality for keeping
#   track of an exposure's parameters, including distortion model,
#   WCS information, and metachip shape.
#
#   Version:
#           0.1.0 -- Created -- Warren Hack
#           0.1.1 -- DGEOFILE logic set to work with N/A as input
#
#           0.1.2 -- Removed diagnostic print statements related to
#                   the use of DGEO files.  --  CJH
#           0.1.3 -- Removed expansion of DGEOFILE path to avoid reporting
#                   pipeline directories in output drizzle keywords. -- WJH
#           0.1.4 -- Implemented support for providing subarray sections
#                   of DGEOFILEs for subarray exposures.
#
import os
import buildmask, fileutil, drutil,wcsutil,arrdriz
from obsgeometry import ObsGeometry

from math import ceil,floor

import numarray as N

yes = True  # 1
no = False  # 0

__version__ = '0.1.4'

#################
#
#
#               Exposure Classes
#
#
#################
class Exposure:
    """
    This class will provide the basic functionality for keeping
    track of an exposure's parameters, including distortion model,
    WCS information, and metachip shape.
    """

    def __init__(self,expname, handle=None, dqname=None, idckey=None,
                    new=no,wcs=None,mask=None,pa_key=None, parity=None,
                    idcdir=None, rot=None, extver=1, exptime=None):

        # This name should be formatted for use in image I/O
        self.name = fileutil.osfn(expname)

        # osfn() will expand '.' unnecessarily, potentially
        # creating a string-length problem for 'drizzle', which
        # is limited to strings of 80 chars.
        _path,_name = os.path.split(self.name)
        # if path for this filename is the same as the current dir,
        # then there is no need to pass along the path.
        if _path == os.getcwd(): self.name = _name


        # Keep track of any associated mask file created for
        # this exposure from its DQ file, or other mask file.
        _fname,_extn = fileutil.parseFilename(expname)
        _open = False

        # Make sure we have an open file handle to use for getting the
        # header and data arrays.
        if not handle and not new:
            handle = fileutil.openImage(expname)
            _open = True

        # If no extension was specified, try to interrogate the file
        # to find whether the SCI array is in the Primary
        # (as in Simple FITS) or first extension (as in MEF).
        if handle and _extn == None:
            if handle[0].data == None:
                # Primary extension specified and no data present.
                # Try looking for data in next extension.
                if len(handle) > 1 and handle[1].data != None:
                    _extn = 1
                    expname += '[1]'
                else:
                    raise IOError, "No valid image data in %s.\n"%expname
            else:
                _extn = 0

        self.dgeoname = None
        self.xgeoim = ""
        self.ygeoim = ""
        self.exptime = exptime
        if not new:
            # Read in a copy of the header for this exposure/group/extension
            _header = fileutil.getHeader(expname,handle=handle)
            _chip = drutil.getChipId(_header)
            self.chip = str(_chip)
            # Keep track of any distortion correction images provided
            # for this chip
            self.dgeoname = fileutil.getKeyword(expname,'DGEOFILE',handle=handle)
            self.xgeoim,self.ygeoim = self.getDGEOExtn()
            if self.exptime == None:
                self.exptime = float(_header['EXPTIME'])
                if self.exptime == 0.: self.exptime = 1.0
            #
            # Extract photometric transformation keywords
            #    If they do not exist, use default values of 0 and 1
            #
            self.plam = float(fileutil.getKeyword(expname,'PHOTPLAM',handle=handle)) / 10.
            if self.plam == None:
                # Setup a default value in case this keyword does not exist
                self.plam = 555.
            self.photzpt = float(fileutil.getKeyword(expname,'PHOTZPT',handle=handle))
            if self.photzpt == None: self.photzpt = 0.0
            self.photflam = float(fileutil.getKeyword(expname,'PHOTFLAM',handle=handle))
            if self.photflam == None: self.photflam = 1.0

            # Read in date-obs from primary header
            if _header:
                if _header.has_key('date-obs'):
                    self.dateobs = _header['date-obs']
                elif _header.has_key('date_obs'):
                    self.dateobs = _header['date_obs']
                else:
                    self.dateobs = None
            else:
                self.dateobs = None
        else:
            _chip = 1
            _header = None
            self.chip = str(_chip)
            # Set a default value for pivot wavelength
            self.plam = 555.
            self.photzpt = 0.0
            self.photflam = 1.0
            self.dateobs = None
            if self.exptime == None:
                self.exptime = 1.

        self.parity = parity
        self.header = _header
        self.extver = extver

        # Create a pointer to the mask file's data array
        # and the name of the original input DQ file
        self.maskname = None
        self.singlemaskname = None
        self.masklist = None
        if mask != None:
            # Specifies filenames to be used if created.
            self.maskname = mask[0]
            self.singlemaskname = mask[1]
            self.masklist = mask[2]

        self.dqname = dqname

        # Remember the name of the coeffs file generated for this chip
        self.coeffs = self.buildCoeffsName()

        # Read the name of idcfile from image header if not explicitly
        # provided by user.
        if idckey != None and idckey.lower() != 'wcs':
            _indx = expname.find('[')
            if  _indx > -1:
                _idc_fname = expname[:_indx]+'[0]'
            else: _idc_fname = expname+'[0]'

            idcfile, idctype = drutil.getIDCFile(self.header,keyword=idckey,
                                        directory=idcdir)
        else:
            idcfile = None
            idctype = None

        if (idckey != None)  and (idckey.lower() == 'header'):
            idckey = idctype

        # Get distortion model and WCS info.
        self.geometry = ObsGeometry(expname, idcfile, idckey=idckey,
                chip=_chip, new=new, header=self.header,
                pa_key=pa_key, rot=rot, date=self.dateobs)

        # Remember the name and type of the IDC file used...
        self.idcfile = idcfile
        self.idctype = idctype

        # Remember the names of the filters used for the exposure
        self.filters = self.geometry.filter1+','+self.geometry.filter2

        # Define shape here...
        # nx,ny,pixel scale
        #
        if wcs != None:
            # We have been passed a WCS to use
            self.geometry.wcs = wcs
            self.geometry.model.pscale = wcs.pscale
            if expname != None:
                self.geometry.wcs.rootname = expname

        self.naxis1 = self.geometry.wcs.naxis1
        self.naxis2 = self.geometry.wcs.naxis2
        self.pscale = self.geometry.wcs.pscale
        self.shape = (self.naxis1,self.naxis2,self.pscale)

        # Keep track of the positions of the corners of the exposure
        # both for the RAW image and the
        # distortion-corrected, unscaled, unrotated image
        self.corners = {'raw':N.zeros((4,2),type=N.Float64),'corrected':N.zeros((4,2),type=N.Float64)}
        self.setCorners()

        # Generate BLOT output name specific to this Exposure
        _blot_extn = '_sci'+repr(extver)+'_blt.fits'
        self.outblot = fileutil.buildNewRootname(self.name,extn=_blot_extn)

        # Keep track of undistorted frame's position relative to metachip
        # Zero-point offset for chip relative to meta-chip product
        # These values get computed using 'setSingleOffsets' from 'writeCoeffs'
        # to insure that the final XDELTA/YDELTA values have been computed.
        self.product_wcs = self.geometry.wcslin
        self.xzero = 0.
        self.yzero = 0.
        self.chip_shape = (0.,0.)
        self.xsh2 = 0.
        self.ysh2 = 0.

        if _open:
            handle.close()
            del handle

    def setCorners(self):
        """ Initializes corners for the raw image. """
        self.corners['raw'] = N.array([(1.,1.),(1.,self.naxis2),(self.naxis1,1.),(self.naxis1,self.naxis2)],type=N.Float64)

    def setSingleOffsets(self):
        """ Computes the zero-point offset and shape of undistorted single chip relative
            to the full final output product metachip.
        """
        _wcs = self.geometry.wcs
        _corners = N.array([(1.,1.),(1.,_wcs.naxis2),(_wcs.naxis1,1.),(_wcs.naxis1,_wcs.naxis2)])
        _wc = self.geometry.wtraxy(_corners,self.product_wcs)

        _xrange = (_wc[:,0].min(),_wc[:,0].max())
        _yrange = (_wc[:,1].min(),_wc[:,1].max())

        self.xzero = int(_xrange[0] - 1)
        self.yzero = int(_yrange[0] - 1)
        if self.xzero < 0: self.xzero = 0
        if self.yzero < 0: self.yzero = 0

        _out_naxis1 = int(ceil(_xrange[1]) - floor(_xrange[0]))
        _out_naxis2 = int(ceil(_yrange[1]) - floor(_yrange[0]))
        _max_x = _out_naxis1 + self.xzero
        _max_y = _out_naxis2 + self.yzero
        if _max_x > self.product_wcs.naxis1: _out_naxis1 -= (_max_x - self.product_wcs.naxis1)
        if _max_y > self.product_wcs.naxis2: _out_naxis2 -= (_max_y - self.product_wcs.naxis2)

        self.chip_shape = (_out_naxis1,_out_naxis2)
        self.xsh2 = int(ceil((self.product_wcs.naxis1 - _out_naxis1)/2.)) - self.xzero
        self.ysh2 = int(ceil((self.product_wcs.naxis2 - _out_naxis2)/2.)) - self.yzero

    def getShape(self):
        """
        This method gets the shape after opening and reading
        the input image. This version will be the default
        way of doing this, but each instrument may override
        this method with one specific to their data format.
        """
        return self.shape

    def setShape(self,size,pscale):
        """
        This method will set the shape for a new file if
        there is no image header information.

        Size will be defined as (nx,ny) and pixel size
        """
        self.shape = (size[0],size[1],pscale)


    def buildCoeffsName(self):
        """ Define the name of the coeffs file used for this chip. """
        indx = self.name.rfind('.')
        return self.name[:indx]+'_coeffs'+self.chip+'.dat'

    def writeCoeffs(self):
        """ Write out coeffs file for this chip. """
        # Pass along the reference position assumed by Drizzle
        # based on 'align=center' according to the conventions
        # described in the help page for 'drizzle'.  26Mar03 WJH
        #
        # coord for image array reference pixel in full chip coords
        #
        if not self.geometry.model.refpix.has_key('empty_model'):
            _xref = self.geometry.wcs.chip_xref
            _yref = self.geometry.wcs.chip_yref
        else:
            _xref = None
            _yref = None

        # If we have a subarray, pass along the offset reference position
        _delta = not (self.geometry.wcslin.subarray)

        # Set up the idcfile for use by 'drizzle'
        self.geometry.model.convert(self.coeffs,xref=_xref,yref=_yref,delta=_delta)

        # set exposure zero-point, now that all values are computed...
        self.setSingleOffsets()

    def getDGEOExtn(self):
        """ Builds filename with extension to access distortion
            correction image extension appropriate to each chip.
        """
        # If no DGEOFILE has been given, then simply return blanks
        # and 'drizzle' will not use any.
        if not self.dgeoname or self.dgeoname == 'N/A':
            return '',''

        # Open file for introspection.
        fimg = fileutil.openImage(self.dgeoname)
        dx_extver = None
        dy_extver = None
        # Find which extensions match this chip ID
        # We need to identify both a DX and DY EXTNAME extension
        for hdu in fimg:
            hdr = hdu.header
            if not hdr.has_key('CCDCHIP'):
                _chip = 1
            else:
                _chip = int(hdr['CCDCHIP'])

            if hdr.has_key('EXTNAME'):
                _extname = hdr['EXTNAME'].lower()

                if _chip == int(self.chip):
                    if _extname == 'dx':
                        dx_extver = hdr['EXTVER']

                    if _extname == 'dy':
                        dy_extver = hdr['EXTVER']

        fimg.close()
        del fimg

        # Set the name for each extension here...
        _dxgeo = self.dgeoname+'[DX,'+str(dx_extver)+']'
        _dygeo = self.dgeoname+'[DY,'+str(dy_extver)+']'

        return _dxgeo,_dygeo

    def getDGEOArrays(self):
        """ Return numarray objects for the distortion correction
            image arrays.

            If no DGEOFILE is specified, it will return
            empty 2x2 arrays.
        """

        # Instantiate array objects for distortion correction image arrays
        if self.xgeoim == '':
            # No distortion image specified.
            # Defaulting to empty 2x2 array.
            xgdim = ygdim = 2
            _pxg = N.zeros((ygdim,xgdim),N.Float32)
            _pyg = N.zeros((ygdim,xgdim),N.Float32)
        else:
            # Open distortion correction FITS file
            _xgfile = fileutil.openImage(self.xgeoim)
            #_xgfile.info()

            # Access the extensions which correspond to this Exposure
            _xgname,_xgext = fileutil.parseFilename(self.xgeoim)
            _ygname,_ygext = fileutil.parseFilename(self.ygeoim)

            _pxgext = fileutil.getExtn(_xgfile,extn=_xgext)
            _pygext = fileutil.getExtn(_xgfile,extn=_ygext)

            # Copy out the numarray objects for output
            _ltv1 = int(self.geometry.wcs.offset_x)
            _ltv2 = int(self.geometry.wcs.offset_y)
            if _ltv1 != 0. or _ltv2 != 0.:
                # subarray section only
                _pxg = _pxgext.data[_ltv2:_ltv2+self.naxis2,_ltv1:_ltv1+self.naxis1].copy()
                _pyg = _pygext.data[_ltv2:_ltv2+self.naxis2,_ltv1:_ltv1+self.naxis1].copy()
            else:
                # full array
                _pxg = _pxgext.data.copy()
                _pyg = _pygext.data.copy()

            # Close file handles now...
            _xgfile.close()
            del _xgfile

        return _pxg,_pyg


    def applyDeltaWCS(self,dcrval=None,dcrpix=None,drot=None,dscale=None):
        """
        Apply shifts to the WCS of this exposure.

        Shifts are always relative to the current value and in the
        same reference frame.
        """
        in_wcs = self.geometry.wcs
        in_wcslin = self.geometry.wcslin

        if dcrpix:
            _crval = in_wcs.xy2rd((in_wcs.crpix1-dcrpix[0],in_wcs.crpix2-dcrpix[1]))
        elif dcrval:
            _crval = (in_wcs.crval1 - dcrval[0],in_wcs.crval2 - dcrval[1])
        else:
            _crval = None

        _orient = None
        _scale = None
        if drot:
            _orient = in_wcs.orient + drot
        if dscale:
            _scale = in_wcs.pscale * dscale

        _crpix = (in_wcs.crpix1,in_wcs.crpix2)

        in_wcs.updateWCS(pixel_scale=_scale,orient=_orient,refval=_crval,refpos=_crpix)
        in_wcslin.updateWCS(pixel_scale=_scale,orient=_orient,refval=_crval,refpos=_crpix)


    def calcNewEdges(self,pscale=None):
        """
        This method will compute arrays for all the pixels around
        the edge of an image AFTER applying the geometry model.

        Parameter: delta - offset from nominal crpix center position

        Output:   xsides - array which contains the new positions for
                          all pixels along the LEFT and RIGHT edges
                  ysides - array which contains the new positions for
                          all pixels along the TOP and BOTTOM edges
        The new position for each pixel is calculated by calling
        self.geometry.apply() on each position.
        """
        # build up arrays for pixel positions for the edges
        # These arrays need to be: array([(x,y),(x1,y1),...])
        numpix = self.naxis1*2 + self.naxis2 * 2
        border = N.zeros(shape=(numpix,2),type=N.Float64)

        # Now determine the appropriate values for this array
        # We also need to account for any subarray offsets
        xmin = 1.
        xmax = self.naxis1
        ymin = 1.
        ymax = self.naxis2

        # Build range of pixel values for each side
        # Add 1 to make them consistent with pixel numbering in IRAF
        # Also include the LTV offsets to represent position in full chip
        #   since the model works relative to full chip positions.
        xside = N.arange(self.naxis1) + xmin
        yside = N.arange(self.naxis2) + ymin

        #Now apply them to the array to generate the appropriate tuples
        #bottom
        _range0 = 0
        _range1 = self.naxis1
        border[_range0:_range1,0] = xside
        border[_range0:_range1,1] = ymin
        #top
        _range0 = _range1
        _range1 = _range0 + self.naxis1
        border[_range0:_range1,0] = xside
        border[_range0:_range1,1] = ymax
        #left
        _range0 = _range1
        _range1 = _range0 + self.naxis2
        border[_range0:_range1,0] = xmin
        border[_range0:_range1,1] = yside
        #right
        _range0 = _range1
        _range1 = _range0 + self.naxis2
        border[_range0:_range1,0] = xmax
        border[_range0:_range1,1] = yside

        # calculate new edge positions
        border[:,0],border[:,1] = self.geometry.apply(border,pscale=pscale)

        #print 'Calculated corrected border positions at ',_ptime()
        #Now apply any chip-to-chip offset from REFPIX
        _refpix = self.geometry.model.refpix
        if _refpix != None:
            _ratio = pscale / _refpix['PSCALE']
            _delta = (_refpix['XDELTA']/_ratio, _refpix['YDELTA']/_ratio)
        else:
            _delta = (0.,0.)

        return border + _delta

    def getWCS(self):
        return self.geometry.wcs
    def showWCS(self):
        print self.geometry.wcs

    def runDriz(self,pixfrac=1.0,kernel='turbo',fillval='INDEF'):
        """ Runs the 'drizzle' algorithm on this specific chip to create
            a numarray object of the undistorted image.

            The resulting drizzled image gets returned as a numarray object.
        """
        #
        # Perform drizzling...
        #
        _wcs = self.geometry.wcs
        _wcsout = self.product_wcs

        # Rigorously compute the orientation changes from WCS
        # information using algorithm provided by R. Hook from WDRIZZLE.
        abxt,cdyt = drutil.wcsfit(self.geometry, self.product_wcs)

        # Compute the rotation and shifts between input and reference WCS.
        xsh = abxt[2]
        ysh = cdyt[2]
        rot = fileutil.RADTODEG(N.arctan2(abxt[1],cdyt[0]))
        scale = self.product_wcs.pscale / self.geometry.wcslin.pscale

        # Now, trim the final output to only match this chip
        _out_naxis1,_out_naxis2 = self.chip_shape

        #
        # Insure that the coeffs file was created
        #
        if not os.path.exists(self.coeffs):
            self.writeCoeffs()

        # A image buffer needs to be setup for converting the input
        # arrays (sci and wht) from FITS format to native format
        # with respect to byteorder and byteswapping.
        # This buffer should be reused for each input.
        #
        _outsci = N.zeros((_out_naxis2,_out_naxis1),N.Float32)
        _outwht = N.zeros((_out_naxis2,_out_naxis1),N.Float32)
        _inwcs = N.zeros([8],N.Float64)

        # Only one chip will ever be drizzled using this method, so
        # the context image will only ever contain 1 bit-plane
        _outctx = N.zeros((_out_naxis2,_out_naxis1),N.Int32)

        # Read in the distortion correction arrays, if specifij8cw08n4q_raw.fitsed
        _pxg,_pyg = self.getDGEOArrays()

        # Open the SCI image
        _expname = self.name
        _handle = fileutil.openImage(_expname,mode='readonly',memmap=0)
        _fname,_extn = fileutil.parseFilename(_expname)
        _sciext = fileutil.getExtn(_handle,extn=_extn)

        # Make a local copy of SCI array and WCS info
        #_insci = _sciext.data.copy()
        _inwcs = drutil.convertWCS(wcsutil.WCSObject(_fname,header=_sciext.header),_inwcs)

        # Compute what plane of the context image this input would
        # correspond to:
        _planeid = 1

        # Select which mask needs to be read in for drizzling
        _inwht = N.ones((self.naxis2,self.naxis1),N.Float32)

        # Default case: wt_scl = exptime
        _wtscl = self.exptime

        # Set additional parameters needed by 'drizzle'
        _expin = self.exptime
        #_in_un = 'counts'
        #_shift_fr = 'output'
        #_shift_un = 'output'
        _uniqid = 1
        ystart = 0
        nmiss = 0
        nskip = 0
        _vers = ''

        #
        # This call to 'arrdriz.tdriz' uses the F2C syntax
        #
        _dny = self.naxis2
        # Call 'drizzle' to perform image combination
        _vers,nmiss,nskip = arrdriz.tdriz(_sciext.data.copy(),_inwht,
                    _outsci, _outwht, _outctx,
                    _uniqid, ystart, 1, 1, _dny,
                    xsh,ysh, 'output','output', rot,scale,
                    self.xsh2,self.ysh2, 1.0, 1.0, 0.0, 'output',
                    _pxg,_pyg,
                    'center', pixfrac, kernel,
                    self.coeffs, 'counts', _expin,_wtscl,
                    fillval, _inwcs, nmiss, nskip, 1)
        #
        # End of F2C syntax
        #

        if nmiss > 0:
            print '! Warning, ',nmiss,' points were outside the output image.'
        if nskip > 0:
            print '! Note, ',nskip,' input lines were skipped completely.'

        # Close image handle
        _handle.close()
        del _handle,_fname,_extn,_sciext
        del _inwht

        del _pxg,_pyg

        del _outwht,_outctx

        return _outsci
