import string, copy, os

import pyfits
import numarray as N
from math import *

import pydrizzle, drutil, fileutil

# Convenience definitions...
yes = True
no = False

DEGTORAD = fileutil.DEGTORAD
RADTODEG = fileutil.RADTODEG
DIVMOD = fileutil.DIVMOD

#
# History
#
# 30-Mar-2002 WJH: Added separate XYtoSky interface function.
# 19-Apr-2002 WJH: Corrected 'ddtohms' for error in converting neg. dec.
# 20-Sept-2002 WJH: replaced all references to 'keypar' with calls to 'hselect'
#                   This avoids any parameter writes in the pipeline.
# 03-Dec-2002 WJH: Added 'new' parameter to WCSObject to make creating an
#                   object from scratch unambiguous and free from filename
#                   collisions with user files.
# 23-Apr-2003 WJH: Enhanced to search entire file for header with WCS keywords
#                   if no extension was specified with filename.
# 6-Oct-2003 WJH:  Modified to use the 'fileutil.getHeader' function or
#                   accept a PyFITS/readgeis header object.  Removed
#                   any explicit check on whether the image was FITS or
#                   not.
# 5-Feb-2004 WJH:  Added 'recenter' method to rigorously shift the WCS from
#                   an off-center reference pixel position to the frame center
#
#



#################
#
#
#               Coordinate Transformation Functions
#
#
#################

def XYtoSky(input, pos, idckey='IDCTAB', linear=yes, verbose=no):
    """ Convert input pixel position(s) into RA/Dec position(s).
        Output will be either an (ra,dec) pair or a 'list' of (ra,dec)
        pairs, not a numarray, to correspond with the input position(s).

        Parameter:
            input - Filename with extension specification of image
            pos   - Either a single [x,y] pair or a list of [x,y] pairs.
            idckey - Keyword which points to the IDC table to be used.
            linear - If no, apply distortion correction for image.

    """

    # Start by making sure we have a valid extension specification.
    _insplit = string.split(input,'[')
    if len(_insplit) == 1:
        raise IOError, 'No extension specified for input image!'

    # Now we need to insure that the input is an array:
    if not isinstance(pos,N.NumArray):
        if N.array(pos).getrank() > 1:
            pos = N.array(pos,type=N.Float64)

    # Set up Exposure object
    _exposure = pydrizzle.Exposure(input,idckey=idckey)

    ra,dec = _exposure.geometry.XYtoSky(pos,linear=linear,verbose=verbose)

    if not isinstance(ra,N.NumArray):
        # We are working with a single input, return single values
        return ra,dec
    else:
        # We are working with arrays, so we need to convert them
        # from 2 arrays with RA in one and Dec in the other to 1 array
        # with pairs of RA/Dec values.
        _radec = N.zeros(shape=(len(ra),2),type=ra.type())
        _radec[:,0] = ra
        _radec[:,1] = dec

        return _radec.tolist()


def ddtohms(xsky,ysky,verbose=no):

    """ Convert sky position(s) from decimal degrees to HMS format."""

    xskyh = xsky /15.
    xskym = (xskyh - N.floor(xskyh)) * 60.
    xskys = (xskym - N.floor(xskym)) * 60.

    yskym = (N.abs(ysky) - N.floor(N.abs(ysky))) * 60.
    yskys = (yskym - N.floor(yskym)) * 60.

    if isinstance(xskyh,N.NumArray):
        rah,dech = [],[]
        for i in xrange(len(xskyh)):
            rastr = repr(int(xskyh[i]))+':'+repr(int(xskym[i]))+':'+repr(xskys[i])
            decstr = repr(int(ysky[i]))+':'+repr(int(yskym[i]))+':'+repr(yskys[i])
            rah.append(rastr)
            dech.append(decstr)
            if verbose:
                print 'RA = ',rastr,', Dec = ',decstr
    else:
        rastr = repr(int(xskyh))+':'+repr(int(xskym))+':'+repr(xskys)
        decstr = repr(int(ysky))+':'+repr(int(yskym))+':'+repr(yskys)
        rah = rastr
        dech = decstr
        if verbose:
            print 'RA = ',rastr,', Dec = ',decstr

    return rah,dech


def troll(roll, dec, v2, v3):
    """ Computes the roll angle at the target position based on:
            the roll angle at the V1 axis(roll),
            the dec of the target(dec), and
            the V2/V3 position of the aperture (v2,v3) in arcseconds.

        Based on the algorithm provided by Colin Cox that is used in
        Generic Conversion at STScI.
    """
    # Convert all angles to radians
    _roll = DEGTORAD(roll)
    _dec = DEGTORAD(dec)
    _v2 = DEGTORAD(v2 / 3600.)
    _v3 = DEGTORAD(v3 / 3600.)

    # compute components
    sin_rho = sqrt((pow(sin(_v2),2)+pow(sin(_v3),2)) - (pow(sin(_v2),2)*pow(sin(_v3),2)))
    rho = asin(sin_rho)
    beta = asin(sin(_v3)/sin_rho)
    if _v2 < 0: beta = pi - beta
    gamma = asin(sin(_v2)/sin_rho)
    if _v3 < 0: gamma = pi - gamma
    A = pi/2. + _roll - beta
    B = atan2( sin(A)*cos(_dec), (sin(_dec)*sin_rho - cos(_dec)*cos(rho)*cos(A)))

    # compute final value
    troll = RADTODEG(pi - (gamma+B))

    return troll

#################
#
#
#               Coordinate System Class
#
#
#################

class WCSObject:
    """ This class should contain the WCS information from the
            input exposure's header and provide conversion functionality
            from pixels to RA/Dec and back.

        It knows how to update the CD matrix for the new solution.

        rootname: this needs to be in a format supported by IRAF
        and 'hselect', specifically:
                filename.hhh[group] or filename.fits[ext]

        Setting 'new=yes' will create a WCSObject from scratch
            regardless of any input rootname.  This avoids unexpected
            filename collisions.
    """
    def __init__(self, rootname,header=None,shape=None,pa_key='PA_V3',new=no):
        # Initialize wcs dictionaries:
        #   wcsdef - default values for new images
        #   wcstrans - translation table from header keyword to attribute
        self.wcsdef = {'crpix1':0.0,'crpix2':0.0,'crval1':0.0,'crval2':0.0,'cd11':1.0,
                'cd12':1.0,'cd21':1.0,'cd22':1.0,'orient':1.0,'naxis1':0,'naxis2':0,'pscale':1.0,
                'postarg1':0.0,'postarg2':0.0,'pa_obs':0.0,
                'ctype1':'RA---TAN','ctype2':'DEC--TAN'}
        self.wcstrans = {'CRPIX1':'crpix1','CRPIX2':'crpix2','CRVAL1':'crval1','CRVAL2':'crval2',
            'CD1_1':'cd11','CD1_2':'cd12','CD2_1':'cd21','CD2_2':'cd22',
            'ORIENTAT':'orient', 'NAXIS1':'naxis1','NAXIS2':'naxis2',
            'pixel scale':'pscale','CTYPE1':'ctype1','CTYPE2':'ctype2'}
        # Now, read in the CRPIX1/2, CRVAL1/2, CD1/2_1/2 keywords.
        # Simplistic, but easy to understand what you are asking for.

        _exists = yes
        if rootname != None:
            self.rootname = rootname
        else:
            self.rootname = 'New'
            new = yes
            _exists = no

        # Initialize attribute for GEIS image name, just in case...
        self.geisname = None

        # Look for extension specification in rootname
        _indx = _section = string.find(self.rootname,'[')
        # If none are found, use entire rootname
        if _indx < 0:
            _indx = len(self.rootname)

        # Determine whether we are working with a new image or not.
        _dir,_rootname = os.path.split(fileutil.osfn(self.rootname[:_indx]))
        if _dir:
            _filename = _dir+os.sep+_rootname
        else:
            _filename = _rootname
        self.filename = _filename

        if not new:
            _exists = fileutil.checkFileExists(_rootname,directory=_dir)

        else:
            _exists = no

        # If no header has been provided, get the PRIMARY and the
        # specified extension header... This call uses the fully
        # expanded version of the filename, plus any sections listed by
        # by the user in the original rootname.
        if not header and _exists:
            _header = fileutil.getHeader(_filename+self.rootname[_indx:])
        else:
            # Otherwise, simply use the header already read into memory
            # for this exposure/chip.
            _header = header

        if _exists:
            # Initialize WCS object with keyword values...
            try:
                _dkey = 'orientat'
                if _header.has_key('orientat'):
                    self.orient = _header['orientat']
                else:
                    self.orient = None

                for key in self.wcstrans.keys():
                    _dkey = self.wcstrans[key]
                    if _dkey != 'pscale' and _dkey != 'orient':
                        self.__dict__[_dkey] = _header[key]
                self.new = no

            except:
                print 'Could not find WCS keyword: ',_dkey
                raise IOError,'Image %s does not contain all required WCS keywords!' % self.rootname
            self
            # Now, try to read in POSTARG keyword values, if they exist...
            try:
                self.postarg1 = _header['postarg1']
                self.postarg2 = _header['postarg2']
            except:
                # If these keywords, don't exist set defaults...
                self.postarg1 = 0.0
                self.postarg2 = 0.0
            try:
                self.pa_obs = _header[pa_key]
            except:
                # If no such keyword exists, use orientat value later
                self.pa_obs = None

        else:
            # or set default values...
            self.new = yes
            for key in self.wcsdef.keys():
                self.__dict__[key] = self.wcsdef[key]
            if shape != None:
                # ... and update with user values.
                self.naxis1 = int(shape[0])
                self.naxis2 = int(shape[1])
                self.pscale = float(shape[2])

        # Make sure reported 'orient' is consistent with CD matrix
        # while preserving the original 'ORIENTAT' keyword value
        if self.orient:
            self.orientat = self.orient

        self.orient = RADTODEG(N.arctan2(self.cd12,self.cd22))

        # If no keyword provided pa_obs value (PA_V3), then default to
        # image orientation from CD matrix.
        if self.pa_obs == None:
            self.pa_obs = self.orient

        if shape == None:
            self.pscale = N.sqrt(N.power(self.cd11,2)+N.power(self.cd21,2)) * 3600.
            # Use Jacobian determination of pixel scale instead of X or Y separately...
            #self.pscale = N.sqrt(abs(self.cd11*self.cd22 - self.cd12*self.cd21))*3600.

        # Establish an attribute for the linearized orient
        # defined as the orientation of the CD after applying the default
        # distortion correction.
        self._orient_lin = 0.

        # attribute to define format for printing WCS
        self.__format__=yes

        # Keep track of the keyword names used as the backup keywords
        # for the original WCS values
        self.backup = {}
        self.prepend = None

    # You never know when you want to print out the WCS keywords...
    def __str__(self):
        block = 'WCS Keywords for ' + self.rootname + ': \n'
        if not self.__format__:
            for key in self.wcstrans.keys():
                _dkey = self.wcstrans[key]
                strn = string.upper(key) + " = " + repr(self.__dict__[_dkey]) + '\n'
                block = block + strn
        else:
            block += 'CD_11  CD_12: '+repr(self.cd11)+'  '+repr(self.cd12) +'\n'
            block += 'CD_21  CD_22: '+repr(self.cd21)+'  '+repr(self.cd22) +'\n'
            block += 'CRVAL       : '+repr(self.crval1)+'  '+repr(self.crval2) + '\n'
            block += 'CRPIX       : '+repr(self.crpix1)+'  '+repr(self.crpix2) + '\n'
            block += 'NAXIS       : '+repr(int(self.naxis1))+'  '+repr(int(self.naxis2)) + '\n'
            block += 'Plate Scale : '+repr(self.pscale)+'\n'
            block += 'ORIENTAT    : '+repr(self.orient)+'\n'
            block += 'CTYPE       : '+repr(self.ctype1)+'  '+repr(self.ctype2)+'\n'

        block += 'PA Telescope: '+repr(self.pa_obs)+'\n'
        return block

    def __repr__(self):
        return repr(self.__dict__)

    def updateWCS(self, pixel_scale=None, orient=None,refpos=None,refval=None,size=None):
        """
        Create a new CD Matrix from the absolute pixel scale
        and reference image orientation.
        """
        # Set up parameters necessary for updating WCS
        # Check to see if new value is provided,
        # If not, fall back on old value as the default

        _updateCD = no
        if orient != None and orient != self.orient:
            pa = DEGTORAD(orient)
            self.orient = orient
            self._orient_lin = orient
            _updateCD = yes
        else:
            # In case only pixel_scale was specified
            pa = DEGTORAD(self.orient)

        if pixel_scale != None and pixel_scale != self.pscale:
            _ratio = pixel_scale / self.pscale
            self.pscale = pixel_scale
            _updateCD = yes
        else:
            # In case, only orient was specified
            pixel_scale = self.pscale
            _ratio = None

        # If a new plate scale was given,
        # the default size should be revised accordingly
        # along with the default reference pixel position.
        # Added 31 Mar 03, WJH.
        if _ratio != None:
            self.naxis1 /= _ratio
            self.naxis2 /= _ratio
            self.crpix1 = self.naxis1/2.
            self.crpix2 = self.naxis2/2.

        # However, if the user provides a given size,
        # set it to use that no matter what.
        if size != None:
            self.naxis1 = size[0]
            self.naxis2 = size[1]

        # Insure that naxis1,2 always return as integer values.
        self.naxis1 = int(self.naxis1)
        self.naxis2 = int(self.naxis2)

        if refpos != None:
            self.crpix1 = refpos[0]
            self.crpix2 = refpos[1]
        if self.crpix1 == None:
            self.crpix1 = self.naxis1/2.
            self.crpix2 = self.naxis2/2.

        if refval != None:
            self.crval1 = refval[0]
            self.crval2 = refval[1]

        # Reset WCS info now...
        if _updateCD:
            # Only update this should the pscale or orientation change...
            pscale = pixel_scale / 3600.

            self.cd11 = -pscale * N.cos(pa)
            self.cd12 = pscale * N.sin(pa)
            self.cd21 = self.cd12
            self.cd22 = -self.cd11

    def xy2rd(self,pos):
        """
        This method would apply the WCS keywords to a position to
        generate a new sky position.

        The algorithm comes directly from 'imgtools.xy2rd'

        translate (x,y) to (ra, dec)
        """
        if self.ctype1.find('TAN') < 0 or self.ctype2.find('TAN') < 0:
            print 'XY2RD only supported for TAN projections.'
            raise TypeError

        if isinstance(pos,N.NumArray):
            # If we are working with an array of positions,
            # point to just X and Y values
            posx = pos[:,0]
            posy = pos[:,1]
        else:
            # Otherwise, we are working with a single X,Y tuple
            posx = pos[0]
            posy = pos[1]

        xi = self.cd11 * (posx - self.crpix1) + self.cd12 * (posy - self.crpix2)
        eta = self.cd21 * (posx - self.crpix1) + self.cd22 * (posy - self.crpix2)

        xi = DEGTORAD(xi)
        eta = DEGTORAD(eta)
        ra0 = DEGTORAD(self.crval1)
        dec0 = DEGTORAD(self.crval2)

        ra = N.arctan((xi / (N.cos(dec0)-eta*N.sin(dec0)))) + ra0
        dec = N.arctan( ((eta*N.cos(dec0)+N.sin(dec0)) /
                (N.sqrt((N.cos(dec0)-eta*N.sin(dec0))**2 + xi**2))) )

        ra = RADTODEG(ra)
        dec = RADTODEG(dec)
        ra = DIVMOD(ra, 360.)

        # Otherwise, just return the RA,Dec tuple.
        return ra,dec


    def rd2xy(self,skypos,hour=no):
        """
        This method would use the WCS keywords to compute the XY position
        from a given RA/Dec tuple (in deg).

        NOTE: Investigate how to let this function accept arrays as well
        as single positions. WJH 27Mar03

        """
        if self.ctype1.find('TAN') < 0 or self.ctype2.find('TAN') < 0:
            print 'RD2XY only supported for TAN projections.'
            raise TypeError

        det = self.cd11*self.cd22 - self.cd12*self.cd21

        if det == 0.0:
            raise ArithmeticError,"singular CD matrix!"

        cdinv11 = self.cd22 / det
        cdinv12 = -self.cd21 / det
        cdinv21 = -self.cd12 / det
        cdinv22 = self.cd11 / det

        # translate (ra, dec) to (x, y)


        ra0 = DEGTORAD(self.crval1)
        dec0 = DEGTORAD(self.crval2)
        if hour:
            skypos[0] = skypos[0] * 15.
        ra = DEGTORAD(skypos[0])
        dec = DEGTORAD(skypos[1])

        bottom = float(N.sin(dec)*N.sin(dec0) + N.cos(dec)*N.cos(dec0)*N.cos(ra-ra0))
        if bottom == 0.0:
            raise ArithmeticError,"Unreasonable RA/Dec range!"

        xi = RADTODEG((N.cos(dec) * N.sin(ra-ra0) / bottom))
        eta = RADTODEG((N.sin(dec)*N.cos(dec0) - N.cos(dec)*N.sin(dec0)*N.cos(ra-ra0)) / bottom)

        x = cdinv11 * xi + cdinv12 * eta + self.crpix1
        y = cdinv21 * xi + cdinv22 * eta + self.crpix2

        return x,y

    def rotateCD(self,orient):
        """ Rotates WCS CD matrix to new orientation given by 'orient'
        """
        # Determine where member CRVAL position falls in ref frame
        # Find out whether this needs to be rotated to align with
        # reference frame.

        _delta = self.orient - orient
        if _delta == 0.:
            return

        # Start by building the rotation matrix...
        _rot = drutil.buildRotMatrix(_delta)
        # ...then, rotate the CD matrix and update the values...
        _cd = N.array([[self.cd11,self.cd12],[self.cd21,self.cd22]],type=N.Float64)
        _cdrot = N.dot(_cd,_rot)
        self.cd11 = _cdrot[0][0]
        self.cd12 = _cdrot[0][1]
        self.cd21 = _cdrot[1][0]
        self.cd22 = _cdrot[1][1]
        self.orient = orient

    def recenter(self):
        """
        Reset the reference position values to correspond to the center
        of the reference frame.
        Algorithm used here developed by Colin Cox - 27-Jan-2004.
        """
        if self.ctype1.find('TAN') < 0 or self.ctype2.find('TAN') < 0:
            print 'WCS.recenter() only supported for TAN projections.'
            raise TypeError

        # Check to see if WCS is already centered...
        if self.crpix1 == self.naxis1/2. and self.crpix2 == self.naxis2/2.:
            # No recentering necessary... return without changing WCS.
            return

        # This offset aligns the WCS to the center of the pixel, in accordance
        # with the 'align=center' option used by 'drizzle'.
        #_drz_off = -0.5
        _drz_off = 0.
        _cen = (self.naxis1/2.+ _drz_off,self.naxis2/2. + _drz_off)

        # Compute the RA and Dec for center pixel
        _cenrd = self.xy2rd(_cen)
        _cd = N.array([[self.cd11,self.cd12],[self.cd21,self.cd22]],type=N.Float64)
        _ra0 = DEGTORAD(self.crval1)
        _dec0 = DEGTORAD(self.crval2)
        _ra = DEGTORAD(_cenrd[0])
        _dec = DEGTORAD(_cenrd[1])

        # Set up some terms for use in the final result
        _dx = self.naxis1/2. - self.crpix1
        _dy = self.naxis2/2. - self.crpix2

        _dE,_dN = DEGTORAD(N.dot(_cd,(_dx,_dy)))
        _dE_dN = 1 + N.power(_dE,2) + N.power(_dN,2)
        _cosdec = N.cos(_dec)
        _sindec = N.sin(_dec)
        _cosdec0 = N.cos(_dec0)
        _sindec0 = N.sin(_dec0)

        _n1 = N.power(_cosdec,2) + _dE*_dE + _dN*_dN*N.power(_sindec,2)
        _dra_dE = (_cosdec0 - _dN*_sindec0)/_n1
        _dra_dN = _dE*_sindec0 /_n1

        _ddec_dE = -_dE*N.tan(_dec) / _dE_dN
        _ddec_dN = (1/_cosdec) * ((_cosdec0 / N.sqrt(_dE_dN)) - (_dN*N.sin(_dec) / _dE_dN))

        # Compute new CD matrix values now...
        _cd11n = _cosdec * (self.cd11*_dra_dE + self.cd21 * _dra_dN)
        _cd12n = _cosdec * (self.cd12*_dra_dE + self.cd22 * _dra_dN)
        _cd21n = self.cd11 * _ddec_dE + self.cd21 * _ddec_dN
        _cd22n = self.cd12 * _ddec_dE + self.cd22 * _ddec_dN

        _new_orient = RADTODEG(N.arctan2(_cd12n,_cd22n))
        _new_pscale = N.sqrt(N.power(_cd11n,2)+N.power(_cd21n,2)) * 3600.

        # Update the values now...
        self.crpix1 = _cen[0]
        self.crpix2 = _cen[1]
        self.crval1 = RADTODEG(_ra)
        self.crval2 = RADTODEG(_dec)

        # Keep the same plate scale, only change the orientation
        self.rotateCD(_new_orient)

        # These would update the CD matrix with the new rotation
        # ALONG with the new plate scale which we do not want.
        self.cd11 = _cd11n
        self.cd12 = _cd12n
        self.cd21 = _cd21n
        self.cd22 = _cd22n
        self.pscale = _new_pscale

    def write(self,fitsname=None):
        """ Write out the values of the WCS keywords to the
            specified image.

            If it is a GEIS image and 'fitsname' has been provided,
            it will automatically make a multi-extension
            FITS copy of the GEIS and update that file. Otherwise, it
            throw an Exception if the user attempts to directly update
            a GEIS image header.
        """
        image = self.rootname
        _fitsname = fitsname

        if image.find('.fits') < 0 and _fitsname != None:
            # A non-FITS image was provided, and openImage made a copy
            # Update attributes to point to new copy instead
            self.geisname = image
            image = self.rootname = _fitsname

        # Open image as writable FITS object
        fimg = fileutil.openImage(image, mode='update', fitsname=_fitsname)

        _root,_iextn = fileutil.parseFilename(image)
        _extn = fileutil.getExtn(fimg,_iextn)

        # Write out values to header...
        for key in self.wcstrans.keys():
            _dkey = self.wcstrans[key]
            if _dkey != 'pscale':
                _extn.header[key] = self.__dict__[_dkey]

        # Close the file
        fimg.close()
        del fimg

    def savecopy(self,prepend,fitsname=None):
        """ Saves a copy of the WCS keywords from the image header
            as new keywords with the user-supplied 'prepend'
            character(s) prepended to the old keyword names.

            If the file is a GEIS image and 'fitsname' != None, create
            a FITS copy and update that version; otherwise, raise
            an Exception and do not update anything.

        """
        _fitsname = fitsname

        # Insure that the 'prepend' string is not too long: >2 chars
        # Otherwise final modified keyword would be too long for FITS card.
        if len(prepend) > 2:
            _prepend = prepend[:2]
            print 'WARNING: Trimming prepend string to: ',_prepend
        else:
            _prepend = prepend

        # Remember this prepend string for use in 'restoreWCS'
        self.prepend = _prepend

        # Open image in update mode
        #    Copying of GEIS images handled by 'openImage'.
        fimg = fileutil.openImage(self.rootname,mode='update',fitsname=_fitsname)
        if self.rootname.find('.fits') < 0 and _fitsname != None:
            # A non-FITS image was provided, and openImage made a copy
            # Update attributes to point to new copy instead
            self.geisname = self.rootname
            self.rootname = _fitsname

        # extract the extension ID being updated
        _root,_iextn = fileutil.parseFilename(self.rootname)
        _extn = fileutil.getExtn(fimg,_iextn)

        # Write out values to header...
        for key in self.wcstrans.keys():
            _dkey = self.wcstrans[key]
            if _dkey != 'pscale':
                # Extract the value for the original keyword
                _value = _extn.header[key]

                # Extract any comment string for the keyword as well
                _indx_key = _extn.header.ascard.index_of(key)
                _full_key = _extn.header.ascard[_indx_key]
                _indx_comment = _full_key.ascardimage().find('/')
                if _indx_comment < 0:
                    _comment = None
                else:
                    _comment = _full_key.ascardimage()[_indx_comment+1:].strip()

                # Update header with new keyword
                _new_key = self._buildNewKeyname(key,_prepend)

                # Keep a record of the new keywords and what original
                # keywords they corresponded to...
                self.backup[_new_key] = key

                _extn.header.update(_new_key,_value,comment=_comment)

                # Print out history keywords to record when these keywords
                # were backed up.
                _extn.header.update('WCSCDATE',fileutil.getLTime(),
                        comment = "Time WCS keywords were copied.")

        # Close the now updated image
        fimg.close()
        del fimg

    def restoreWCS(self,prepend=None):
        """ Resets the WCS values to the original values stored in
            the backup keywords recorded in self.backup.
        """
        # Open header for image
        image = self.rootname

        if prepend: _prepend = prepend
        elif self.prepend: _prepend = self.prepend
        else: _prepend = None

        # Open image as writable FITS object
        fimg = fileutil.openImage(image, mode='update')
        # extract the extension ID being updated
        _root,_iextn = fileutil.parseFilename(self.rootname)
        _extn = fileutil.getExtn(fimg,_iextn)

        if len(self.backup) > 0:
            # If it knows about the backup keywords already,
            # use this to restore the original values to the original keywords
            for newkey in self.backup.keys():
                _orig_key = self.backup[newkey]
                _extn.header[_orig_key] = _extn.header[newkey]
            fimg.close()
            del fimg
        elif _prepend:
            for key in self.wcstrans.keys():
                # Get new keyword name based on old keyname
                #    and prepend string
                if key != 'pixel scale':
                    _okey = self._buildNewKeyname(key,_prepend)

                    if _extn.header.has_key(_okey):
                        _extn.header[key] = _extn.header[_okey]
                    else:
                        print 'No original WCS values found. Exiting...'
                        fimg.close()
                        del fimg
                        break
        else:
            print 'No original WCS values found. Exiting...'
            fimg.close()
            del fimg

    def _buildNewKeyname(self,key,prepend):
        """ Builds a new keyword based on original keyword name and
            a prepend string.
        """

        if len(prepend+key) <= 8: _new_key = prepend+key
        else: _new_key = str(prepend+key)[:8]

        return _new_key


    def copy(self,deep=yes):
        """ Makes a (deep)copy of this object for use by other objects.
        """
        if deep:
            return copy.deepcopy(self)
        else:
            return copy.copy(self)
