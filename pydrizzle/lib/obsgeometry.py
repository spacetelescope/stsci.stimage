import types,string,os,copy
from math import ceil,floor

# Import PyDrizzle utility modules
import fileutil, wcsutil, drutil

import numpy as N
from drutil import combin

yes = True
no = False

#################
#
#
#               Geometry/Distortion Classes
#
#
#################

class GeometryModel:
    """
    Base class for Distortion model.
    There will be a separate class for each type of
    model/filetype used with drizzle, i.e., IDCModel and
    DrizzleModel.

    Each class will know how to apply the distortion to a
    single point and how to convert coefficients to an input table
    suitable for the drizzle task.

    Coefficients will be stored in CX,CY arrays.
    """
    #
    #
    #
    #
    #
    #
    #
    NORDER = 3

    def __init__(self):
        "  This will open the given file and determine its type and norder."

        #       Method to read in coefficients from given table and
        #       populate the n arrays 'cx' and 'cy'.
        #       This will be different for each type of input file,
        #       IDCTAB vs. drizzle table.

        # Set these up here for all sub-classes to use...
        # But, calculate norder and cx,cy arrays in detector specific classes.
        self.cx = None
        self.cy = None
        self.refpix = None
        self.norder = self.NORDER
        # Keep track of computed zero-point for distortion coeffs
        self.x0 = None
        self.y0 = None

        # default values for these attributes
        self.direction = 'forward'

        self.pscale = 1.0

    def shift(self,cx,cy,xs,ys):
        """
        Shift reference position of coefficients to new center
        where (xs,ys) = old-reference-position - subarray/image center.
        This will support creating coeffs files for drizzle which will
        be applied relative to the center of the image, rather than relative
        to the reference position of the chip.
        """

        _cxs = N.zeros(shape=cx.shape,dtype=cx.dtype)
        _cys = N.zeros(shape=cy.shape,dtype=cy.dtype)
        _k = self.norder + 1
        # loop over each input coefficient
        for m in xrange(_k):
            for n in xrange(_k):
                if m >= n:
                    # For this coefficient, shift by xs/ys.
                    _ilist = range(m, _k)
                    # sum from m to k
                    for i in _ilist:
                        _jlist = range(n, i - (m-n)+1)
                        # sum from n to i-(m-n)
                        for j in _jlist:
                            _cxs[m,n] += cx[i,j]*combin(j,n)*combin((i-j),(m-n))*pow(xs,(j-n))*pow(ys,((i-j)-(m-n)))
                            _cys[m,n] += cy[i,j]*combin(j,n)*combin((i-j),(m-n))*pow(xs,(j-n))*pow(ys,((i-j)-(m-n)))
        return _cxs,_cys

    def convert(self, tmpname, xref=None,yref=None,delta=yes):
        """
         Open up an ASCII file, output coefficients in drizzle
          format after converting them as necessary.
        First, normalize these coefficients to what drizzle expects
        Normalize the coefficients by the MODEL/output plate scale.

        16-May-2002:
        Revised to work with higher order polynomials by John Blakeslee.
        27-June-2002:
            Added ability to shift coefficients to new center for support
                of subarrays.
        """
        cx = self.cx/self.pscale
        cy = self.cy/self.pscale
        x0 = self.refpix['XDELTA'] + cx[0,0]
        y0 = self.refpix['YDELTA'] + cy[0,0]
        #xr = self.refpix['XREF']
        #yr = self.refpix['YREF']
        xr = self.refpix['CHIP_XREF']
        yr = self.refpix['CHIP_YREF']



        '''
        if xref != None:
            # Shift coefficients for use with drizzle
            _xs = xref - self.refpix['XREF'] + 1.0
            _ys = yref - self.refpix['YREF'] + 1.0


            if _xs != 0 or _ys != 0:
                cxs,cys= self.shift(cx, cy, _xs, _ys)
                cx = cxs
                cy = cys

                # We only want to apply this shift to coeffs
                # for subarray images.
                if delta == no:
                    cxs[0,0] = cxs[0,0] - _xs
                    cys[0,0] = cys[0,0] - _ys

                # Now, apply only the difference introduced by the distortion..
                # i.e., (undistorted - original) shift.
                x0 += cxs[0,0]
                y0 += cys[0,0]
        '''
        self.x0 = x0 #+ 1.0
        self.y0 = y0 #+ 1.0

        # Now, write out the coefficients into an ASCII
        # file in 'drizzle' format.
        lines = []


        lines.append('# ACS polynomial distortion coefficients\n')
        lines.append('# Extracted from "%s" \n'%self.name)
        lines.append('refpix %f %f \n'%(xr,yr))
        if self.norder==3:
            lines.append('cubic\n')
        elif self.norder==4:
            lines.append('quartic\n')
        elif self.norder==5:
            lines.append('quintic\n')
        else:
            raise ValueError, "Drizzle cannot handle poly distortions of order %d"%self.norder

        str = '%16.8f %16.8g %16.8g %16.8g %16.8g \n'% (x0,cx[1,1],cx[1,0],cx[2,2],cx[2,1])
        lines.append(str)
        str = '%16.8g %16.8g %16.8g %16.8g %16.8g \n'% (cx[2,0],cx[3,3],cx[3,2],cx[3,1],cx[3,0])
        lines.append(str)
        if self.norder>3:
            str = '%16.8g %16.8g %16.8g %16.8g %16.8g \n'% (cx[4,4],cx[4,3],cx[4,2],cx[4,1],cx[4,0])
            lines.append(str)
        if self.norder>4:
            str = '%16.8g %16.8g %16.8g %16.8g %16.8g %16.8g \n'% (cx[5,5],cx[5,4],cx[5,3],cx[5,2],cx[5,1],cx[5,0])
            lines.append(str)
        lines.append("\n")

        str = '%16.8f %16.8g %16.8g %16.8g %16.8g \n'% (y0,cy[1,1],cy[1,0],cy[2,2],cy[2,1])
        lines.append(str)
        str = '%16.8g %16.8g %16.8g %16.8g %16.8g \n'% (cy[2,0],cy[3,3],cy[3,2],cy[3,1],cy[3,0])
        lines.append(str)
        if self.norder>3:
            str = '%16.8g %16.8g %16.8g %16.8g %16.8g \n'% (cy[4,4],cy[4,3],cy[4,2],cy[4,1],cy[4,0])
            lines.append(str)
        if self.norder>4:
            str = '%16.8g %16.8g %16.8g %16.8g %16.8g %16.8g \n'% (cy[5,5],cy[5,4],cy[5,3],cy[5,2],cy[5,1],cy[5,0])
            lines.append(str)

        output = open(tmpname,'w')
        output.writelines(lines)
        output.close()


    def apply(self, pixpos,scale=1.0,order=None):
        """
         Apply coefficients to a pixel position or a list of positions.
          This should be the same for all coefficients tables.
        Return the geometrically-adjusted position
        in arcseconds from the reference position as a tuple (x,y).

        Compute delta from reference position
        """
        if self.cx == None:
            return pixpos[:,0],pixpos[:,1]

        if order is None:
            order = self.norder

        # Apply in the same way that 'drizzle' would...
        _cx = self.cx / (self.pscale * scale)
        _cy = self.cy / (self.pscale * scale)
        _convert = no
        _p = pixpos

        # Do NOT include any zero-point terms in CX,CY here
        # as they should not be scaled by plate-scale like rest
        # of coeffs...  This makes the computations consistent
        # with 'drizzle'.  WJH 17-Feb-2004
        _cx[0,0] = 0.
        _cy[0,0] = 0.

        if isinstance(_p,types.ListType) or isinstance(_p,types.TupleType):
            _p = N.array(_p,dtype=N.float64)
            _convert = yes

        dxy = _p - (self.refpix['XREF'],self.refpix['YREF'])
        # Apply coefficients from distortion model here...
        c = _p * 0.
        for i in range(order+1):
            for j in range(i+1):
                c[:,0] = c[:,0] + _cx[i][j] * pow(dxy[:,0],j) * pow(dxy[:,1],(i-j))
                c[:,1] = c[:,1] + _cy[i][j] * pow(dxy[:,0],j) * pow(dxy[:,1],(i-j))
        xc = c[:,0]
        yc = c[:,1]

        # Convert results back to same form as original input
        if _convert:
            xc = xc.tolist()
            yc = yc.tolist()
            # If a single tuple was input, return just a single tuple
            if len(xc) == 1:
                xc = xc[0]
                yc = yc[0]

        return xc,yc

    def setPScaleCoeffs(self,pscale):
        self.cx[1,1] = pscale
        self.cy[1,0] = pscale

        self.refpix['PSCALE'] = pscale
        self.pscale = pscale


class IDCModel(GeometryModel):
    """
    This class will open the IDCTAB, select proper row based on
    chip/direction and populate cx,cy arrays.
    We also need to read in SCALE, XCOM,YCOM, XREF,YREF as well.
    """
    def __init__(self, idcfile, date=None, chip=1, direction='forward',
                filter1='CLEAR1',filter2='CLEAR2',offtab=None):
        GeometryModel.__init__(self)
        #
        # Norder must be derived from the coeffs file itself,
        # then the arrays can be setup. Thus, it needs to be
        # done in the sub-class, not in the base class.
        # Read in table.
        # Populate cx,cy,scale, and other variables here.
        #
        self.name = idcfile
        self.cx,self.cy,self.refpix,self.norder = fileutil.readIDCtab(idcfile,
                        chip=chip,direction=direction,filter1=filter1,filter2=filter2,
                date=date, offtab=offtab)
        self.pscale = self.refpix['PSCALE']

class WCSModel(GeometryModel):
    """
    This class sets up a distortion model based on coefficients
    found in the image header.
    """
    def __init__(self,header,rootname):
        GeometryModel.__init__(self)


        if header.has_key('rootname'):
            self.name = header['rootname']
        else:
            self.name = rootname
        # Initialize all necessary distortion arrays with
        # default model...
        #self.cx,self.cy,self.refpix,self.order = fileutil.defaultModel()

        # Read in values from header, and update distortion arrays.
        self.cx,self.cy,self.refpix,self.norder = fileutil.readWCSCoeffs(header)

        self.pscale = self.refpix['PSCALE']



class DrizzleModel(GeometryModel):
    """
    This class will read in an ASCII Cubic
    drizzle coeffs file and populate the cx,cy arrays.
    """

    def __init__(self, idcfile):
        GeometryModel.__init__(self)
        #
        # We now need to read in the file, populate cx,cy, and
        # other variables as necessary.
        #
        self.name = idcfile
        self.cx,self.cy,self.refpix,self.norder = drutil.readCubicTable(idcfile)
        self.pscale = self.refpix['PSCALE']


class TraugerModel(GeometryModel):
    """
    This class will read in the ASCII Trauger coeffs
    file, convert them to SIAF coefficients, then populate
    the cx,cy arrays.
    """
    NORDER = 3

    def __init__(self, idcfile,lam):
        GeometryModel.__init__(self)
        self.name = idcfile
        self.cx,self.cy,self.refpix,self.norder = drutil.readTraugerTable(idcfile,lam)
        self.pscale = self.refpix['PSCALE']
        #
        # Read in file here.
        # Populate cx,cy, and other variables.
        #


class ObsGeometry:
    """
        Base class for Observation's geometric information.

        This class must know how to recognize the different
        types of distortion coefficients tables and instantiate
        the correct class for it.
    """

    def __init__(self, rootname, idcfile, idckey=None, chip=1, direction="forward",
                header=None, pa_key=None, new=None, date=None,
                rot=None):
        """
         We need to setup the proper object for the GeometryModel
          based on the format of the provided idctab.

         We need to trap attempts to address values of 'chip' that
         are not part of the observation; such as in sub-arrays.
        """
        self.header = header

        _offtab = None
        _filt1 = None
        _filt2 = None

        if self.header and (idckey == None or idckey.lower() == 'idctab'):
            # Try to read in filter names for use in ObsGeometry
            try:
                _filtnames = fileutil.getFilterNames(self.header)
                _filters = _filtnames.split(',')

                _filt1 = _filters[0]
                if len(_filters) > 1: _filt2 = _filters[1]
                else: _filt2 = ''

                if _filt1.find('CLEAR') > -1: _filt1 = _filt1[:6]
                if _filt2.find('CLEAR') > -1: _filt2 = _filt2[:6]
                if _filt1.strip() == '': _filt1 = 'CLEAR1'
                if _filt2.strip() == '': _filt2 = 'CLEAR2'

                # Also check to see if an OFFTAB file has been specified
                if self.header.has_key('OFFTAB'):
                    _offtab = self.header['offtab']

            except:
                print '! Warning: Using default filter settings of CLEAR.'

        if _filt1 == None:
            _filt1 = 'CLEAR1'
            _filt2 = 'CLEAR2'

        self.idcfile = idcfile
        self.direction = direction
        self.filter1 = _filt1
        self.filter2 = _filt2
        self.ikey = None
        self.date = date

        # Default values for VAFACTOR correction
        self.vafactor = 1.0
        self.delta_x = 0.0
        self.delta_y = 0.0

        # Default values for secondary shift (tweak-shift values)
        # These pixel shifts will ALWAYS be in terms of 'input' pixels
        # Set in 'applyAsnShifts'
        self.gpar_xsh = 0.0
        self.gpar_ysh = 0.0
        self.gpar_rot = 0.0

        self.def_rot = None

        if not new:
            self.wcs = wcsutil.WCSObject(rootname,header=self.header)
            self.wcs.recenter()
            self.wcslin = self.wcs.copy()

            # Based on the filetype, open the correct geometry model
            if idckey != None:
                ikey = string.lower(idckey)
            else:
                ikey = 'idctab'

            self.ikey = ikey

            if ikey == 'idctab':
                self.model =  IDCModel(self.idcfile,
                    chip=chip, direction=self.direction, date=self.date,
                    filter1=_filt1, filter2=_filt2, offtab=_offtab)
            elif ikey == 'cubic':
                self.model = DrizzleModel(self.idcfile)
                _chip_rot = fileutil.RADTODEG(N.arctan2(self.model.cy[1][0],self.model.cx[1][0]))
                if rot != None:
                    _theta = _chip_rot - rot
                    _cx = drutil.rotateCubic(self.model.cx,_theta)
                    _cy = drutil.rotateCubic(self.model.cy,_theta)
                    self.model.cx = _cx
                    self.model.cy = _cy
                    self.def_rot = rot
                else:
                    self.def_rot = _chip_rot

            elif ikey == 'trauger':
                _lam = self.header['PHOTPLAM']
                self.model = TraugerModel(self.idcfile,float(_lam)/10.)

            elif ikey == 'wcs':
                self.model = WCSModel(self.header, rootname)

            else:
                raise ValueError, "Unknown type of coefficients table %s"%idcfile

            if self.idcfile == None:
                #Update default model with WCS plate scale
                self.model.setPScaleCoeffs(self.wcs.pscale)

            # Insure that a default model pscale has been set
            if self.model.refpix['PSCALE'] == None:
                self.model.pscale = 1.0
                self.model.refpix['PSCALE']  = self.model.pscale
            if self.model.refpix['XREF'] == None:
                self.model.refpix['XREF'] = self.wcs.crpix1
                self.model.refpix['YREF'] = self.wcs.crpix2

            # Determine whether there is any offset for the image
            # as in the case of subarrays (based on LTV keywords)
            _ltv1,_ltv2 = drutil.getLTVOffsets(rootname,header=self.header)

            if float(_ltv1) != 0. or float(_ltv2) != 0.:
                self.wcs.offset_x = self.wcslin.offset_x = -float(_ltv1)
                self.wcs.offset_y = self.wcslin.offset_y = -float(_ltv2)
                _delta_refx =  (self.wcs.crpix1 + self.wcs.offset_x) - self.model.refpix['XREF']
                _delta_refy =  (self.wcs.crpix2 + self.wcs.offset_y) - self.model.refpix['YREF']

                self.wcs.delta_refx = self.wcslin.delta_refx = _delta_refx
                self.wcs.delta_refy = self.wcslin.delta_refy = _delta_refy
                self.wcs.subarray = self.wcslin.subarray = yes
                self.wcs.chip_xref = self.wcs.offset_x + self.wcs.crpix1
                self.wcs.chip_yref = self.wcs.offset_y + self.wcs.crpix2

                # CHIP_X/YREF will be passed as refpix to drizzle through
                # the coefficients files. This is necessary to account for the
                # fact that drizzle applies the model in the image frame
                # while refpix in the the model is defined in the full frame.
                # This affects subarrays and polarized observations.

                self.model.refpix['CHIP_XREF'] = self.wcs.crpix1 - self.wcs.delta_refx
                self.model.refpix['CHIP_YREF'] = self.wcs.crpix2 - self.wcs.delta_refy
            else:
                self.wcs.offset_x = self.wcslin.offset_x = 0.
                self.wcs.offset_y = self.wcslin.offset_y = 0.
                # Need to account for off-center reference pixel in distortion coeffs
                self.wcs.delta_refx = self.wcslin.delta_refx = (self.wcs.naxis1/2.0) - self.model.refpix['XREF']
                self.wcs.delta_refy = self.wcslin.delta_refy = (self.wcs.naxis2/2.0) - self.model.refpix['YREF']
                self.wcs.subarray = self.wcslin.subarray = no
                self.wcs.chip_xref = self.wcs.naxis1/2.
                self.wcs.chip_yref = self.wcs.naxis2/2.
                self.model.refpix['CHIP_XREF'] = self.model.refpix['XREF']
                self.model.refpix['CHIP_YREF'] = self.model.refpix['YREF']


            #
            # Apply VAFACTOR if present in header.
            #
            # Check to see if Velocity aberration correction can be applied.
            # Convert parameters to 'float()' later to allow for 'None'
            # values when keywords do not exist yet in the headers, since
            # you can't convert 'None' with float().

            try:
                _vafactor = self.header['VAFACTOR']
            except KeyError:
                _vafactor = None

            try:
                _wcscorr = self.header['WCSCORR']
            except KeyError:
                _wcscorr = None

            if _wcscorr == 'DONE':
                # Turn off applying VAFACTOR.
                _vafactor == None

            # If no distortion model has been specified,
            # do NOT perform correction for VAFactor either...
            """
            if _vafactor and idcfile != None:
                if float(_vafactor) != 1.0:
                    _ra_targ = self.header['RA_TARG']
                    _dec_targ = self.header['DEC_TARG']

                    if _ra_targ and _dec_targ:
                        # apply vafactor to coefficients now.
                        self.applyVAFactor(float(_vafactor), float(_ra_targ), float(_dec_targ))
                    else:
                        print 'Velocity aberration correction also requires RA_TARG and DEC_TARG.'
                        print 'No correction applied.'
            """
            # Generate linear WCS to linear CD matrix
            self.undistortWCS()
        else:
            # For new images with no distortion, CD matrix is sufficient.
            self.wcs = wcsutil.WCSObject(None)
            self.wcslin = wcsutil.WCSObject(None)
            self.model = GeometryModel()
            self.wcs.offset_x = self.wcslin.offset_x = 0.
            self.wcs.offset_y = self.wcslin.offset_y = 0.
            self.wcs.delta_refx = self.wcslin.delta_refx = 0.
            self.wcs.delta_refy = self.wcslin.delta_refy = 0.
            self.wcs.subarray = self.wcslin.subarray = no


    def apply(self, pixpos,delta=None,pscale=None,verbose=no,order=None):
        """
         This method applies the model to a pixel position
          to calculate the new position.
          Depending on the value of direction, this could mean
          going from distorted/raw to a corrected positions or
          the other way around.

          If a specific pixel scale is provided, this will be
          used to determine the final output position.

        """
        if delta == None:
            # Use default from table.
            deltax = 0.
            deltay = 0.
        else:
            deltax = delta[0]
            deltay = delta[1]

        if pscale == None:
            pscale = self.model.pscale

        _ratio = pscale / self.model.pscale

        # Put input positions into full frame coordinates...
        pixpos = pixpos + N.array((self.wcs.offset_x,self.wcs.offset_y),dtype=N.float64)
        #v2,v3 = self.model.apply(pixpos, scale=pscale)
        v2,v3 = self.model.apply(pixpos,scale=_ratio,order=order)

        # If there was no distortion applied to
        # the pixel position, simply shift by new
        # reference point.
        if self.model.cx == None:
            if self.model.refpix != None:
                if self.model.refpix['XREF'] == None:
                    refpos = (self.wcs.crpix1,self.wcs.crpix2)
                else:
                    refpos = (self.model.refpix['XREF'],self.model.refpix['YREF'])
            else:
                refpos = (self.wcs.crpix1,self.wcs.crpix2)

            xpos = v2 - refpos[0] + deltax - self.wcs.offset_x/_ratio
            ypos = v3 - refpos[1] + deltay - self.wcs.offset_y/_ratio
        else:
            # For sub-arrays, we need to account for the offset
            # between the CRPIX of the sub-array and the model reference
            # position in the full frame coordinate system.
            # In addition, the LTV value (offset_x,offset_y) needs to be
            # removed again as well. WJH 12 Sept 2004
            # For full images, this correction will be ZERO.
            # This offset, though, has to be scaled by the relative plate-scales.
            #
            #v2 = v2 / pscale
            #v3 = v3 / pscale
            
            xpos = v2  + deltax - (self.wcs.delta_refx / _ratio)
            ypos = v3  + deltay - (self.wcs.delta_refy / _ratio)

        # Return the geometrically-adjusted position as a tuple (x,y)
        return xpos,ypos


    def applyVAFactor(self, vafactor, ra_targ, dec_targ):
        """ Correct the distortion coefficients for the effects of velocity
            aberration, IF the VAFACTOR value is present.
            This method relies on RA_TARG and DEC_TARG to provide the information
            on where the telescope was pointing, the point which serves as the center
            of the radial velocity aberration.
        """
        if vafactor != 1.0 and vafactor: self.vafactor = vafactor

        # Convert ra_targ, dec_targ into X,Y position relative to this chip
        targ_x,targ_y = self.wcslin.rd2xy((ra_targ,dec_targ))
        delta_x = targ_x - self.wcslin.crpix1
        delta_y = targ_y - self.wcslin.crpix2

        #self.delta_x = delta_x*(vafactor - 1.0)*self.wcslin.pscale
        #self.delta_y = delta_y*(vafactor - 1.0)*self.wcslin.pscale

        self.delta_x = delta_x*vafactor
        self.delta_y = delta_y*vafactor
        # Now, shift coefficients to the X/Y position of the target aperture
        _xcs,_ycs = self.model.shift(self.model.cx,self.model.cy,delta_x,delta_y)
        #_xc = self.model.cx
        #_yc = self.model.cy

        # Apply VAFACTOR
        for i in range(self.model.norder+1):
            for j in range(i+1):
                _xcs[i][j] *= N.power(vafactor,i)
                _ycs[i][j] *= N.power(vafactor,i)

        _xc,_yc = self.model.shift(_xcs,_ycs,-delta_x,-delta_y)

        # Update coefficients with these values
        self.model.cx = _xc
        self.model.cy = _yc

    def wtraxy(self,pixpos,wcs,verbose=False):
        """
        Converts input pixel position 'pixpos' into an X,Y position in WCS.
        Made this function compatible with list input, as well as single
        tuple input..apply
        """

        # Insure that input wcs is centered for proper results
        # Added 1-Dec-2004.
        wcs.recenter()

        _ab,_cd = drutil.wcsfit(self,wcs)
        _orient = fileutil.RADTODEG(N.arctan2(_ab[1],_cd[0]))
        _scale = wcs.pscale/self.wcslin.pscale
        _xoff = _ab[2]
        _yoff = _cd[2]

        # changed from self.wcs.naxis[1/2]
        _naxis = (wcs.naxis1,wcs.naxis2)
        _rot_mat = fileutil.buildRotMatrix(_orient)

        if isinstance(pixpos, types.TupleType):
            pixpos = [pixpos]

        _delta_x,_delta_y = self.apply(pixpos)
        if verbose:
            print 'Raw corrected position: ',_delta_x,_delta_y
        _delta_x += self.model.refpix['XDELTA']
        _delta_y += self.model.refpix['YDELTA']
        if verbose:
            print 'Fully corrected position: ',_delta_x,_delta_y

        _delta = N.zeros((len(pixpos),2),dtype=N.float32)
        _delta[:,0] = _delta_x
        _delta[:,1] = _delta_y

        # Need to 0.5 offset to xp,yp to compute the offset in the same way that
        # 'drizzle' computes it.
        _xp = _naxis[0]/2. + 0.5
        _yp = _naxis[1]/2. + 0.5
        _xt = _xoff + _xp
        _yt = _yoff + _yp

        if verbose:
            print 'XSH,YSH: ',_xoff,_yoff
            print 'XDELTA,YDELTA: ',self.model.refpix['XDELTA'],self.model.refpix['YDELTA']
            print 'XREF,YREF: ',self.model.refpix['XREF'],self.model.refpix['YREF']
            print 'xt,yt: ',_xt,_yt,' based on xp,yp: ',_xp,_yp

        _xy_corr = N.dot(_delta,_rot_mat) / _scale
        _delta[:,0] = _xy_corr[:,0] + _xt
        _delta[:,1] = _xy_corr[:,1] + _yt

        if len(pixpos) == 1:
            return _delta[0]
        else:
            return _delta
        #return (_x_out,_y_out)

    def invert(self,pixpos,error=None,maxiter=None):
        """
        This method is the reverse of "apply" - it finds a position
        which, when distorted, maps to "pixpos". The method is iterative
        and is modelled on that in "tranback" in the dither package.

        pixpos is an x,y pixel position tuple.

        pixpos is assumed to be in the appropriate sub-array coordinates.

        Richard Hook, ST-ECF/STScI, January 2003
        """

        # Set an appropriate value of error, if not supplied
        if error == None:
            error=0.001

        # Set a sensible number of iterations as default
        if maxiter == None:
            maxiter=10

        # Put input positions into full frame coordinates...
        # Also convert X,Y tuple to a numarray
        pp = N.array([pixpos]) + N.array((self.wcs.offset_x,self.wcs.offset_y),dtype=N.float64)

        # We are going to work with three x,y points as a NumArray
        pos=N.zeros((3,2),dtype=N.float64)

        # We also need a matrix of the shifts
        shift=N.zeros((2,2),dtype=N.float64)

        # Setup an initial guess - just the first pixel
        pos[0]=[self.wcs.crpix1,self.wcs.crpix2]

        # Loop around until we get close enough (determined by the optional error value)
        for i in range(maxiter):

            # Offset by 1 in X and Y
            pos[1]=pos[0]+[1.0,0.0]
            pos[2]=pos[0]+[0.0,1.0]

            # Apply the forward transform
            tout=self.apply(pos)

            # Convert back to NumArray
            out=N.array(tout,dtype=N.float64).apply
            out.transpose()

            # Work out the shifts matrix
            shift[0]=out[1]-out[0]
            shift[1]=out[2]-out[0]

            # Invert the matrix (this probably should be a separate method)
            a=shift[0,0]
            b=shift[0,1]
            c=shift[1,0]
            d=shift[1,1]
            det=a*d-b*c
            if det==0.0: raise "Singular matrix!"
            oshift=shift
            shift[0,0]=d/det
            shift[0,1]=-b/det
            shift[1,0]=-c/det
            shift[1,1]=a/det

            # Determine the X and Y errors
            errors=pp-out[0]

            # Keep the old position before updating it
            old=pos[0].copy()

            # Update the position
            pos[0]=[old[0]+errors[0,0]*shift[0,0]+errors[0,1]*shift[1,0], \
                 old[1]+errors[0,0]*shift[0,1]+errors[0,1]*shift[1,1]]

            # Check the size of the error
            ev=N.sqrt(N.power(pos[0,0]-old[0],2) + N.power(pos[0,1]-old[1],2))

            # Stop looping if close enough
            if ev < error: break

        # Return the new position as a tuple
        return pos[0,0],pos[0,1]

    def undistortWCS(self,shape=None):
        """
        This method applies the distortion to a 1x1 box at the reference
        position and updates the WCS based on the results.

        This method is based directly on the 'drizzle' subroutine 'UPWCS'
        written by R. Hook.
        """
        # Check to see if we have a valid model to apply to the WCS
        if self.model.cx == None:
            return

        # define the reference point
        _cpix1 = self.wcs.crpix1
        _cpix2 = self.wcs.crpix2

        if not shape:
            _cen = (_cpix1, _cpix2)
        else:
            _cen = (shape[0]/2. + 0.5,shape[1]/2. + 0.5)

        _xy = N.array([(_cpix1,_cpix2),(_cpix1+1.,_cpix2),(_cpix1,_cpix2+1.)],dtype=N.float64)

        #
        _xdelta = self.model.refpix['XDELTA']
        _ydelta = self.model.refpix['YDELTA']

        # apply the distortion to them
        _xc,_yc = self.apply(_xy)
        _xc += _xdelta + _cen[0]
        _yc += _ydelta + _cen[1]

        # Now, work out the effective CD matrix of the transformation
        _am = _xc[1] - _xc[0]
        _bm = _xc[2] - _xc[0]
        _cm = _yc[1] - _yc[0]
        _dm = _yc[2] - _yc[0]
        _cd_mat = N.array([[_am,_bm],[_cm,_dm]],dtype=N.float64)

        # Check the determinant for singularity
        _det = (_am * _dm) - (_bm * _cm)
        if ( _det == 0.0):
            print 'Matrix is singular! Can NOT update WCS.'
            return

        _cd_inv = N.linalg.inv(_cd_mat)
        _a = _cd_inv[0,0]
        _b = _cd_inv[0,1]
        _c = _cd_inv[1,0]
        _d = _cd_inv[1,1]

        self.wcslin.cd11 = _a * self.wcs.cd11 + _c * self.wcs.cd12
        self.wcslin.cd21 = _a * self.wcs.cd21 + _c * self.wcs.cd22
        self.wcslin.cd12 = _b * self.wcs.cd11 + _d * self.wcs.cd12
        self.wcslin.cd22 = _b * self.wcs.cd21 + _d * self.wcs.cd22
        self.wcslin.orient = N.arctan2(self.wcslin.cd12,self.wcslin.cd22) * 180./N.pi
        self.wcslin.pscale = N.sqrt(N.power(self.wcslin.cd11,2)+N.power(self.wcslin.cd21,2))*3600.

        # Compute new size and reference pixel position...
        _wcorners = self.calcNewCorners()

        _x0 = int(floor(N.minimum.reduce(_wcorners[:,0])))
        if _x0 > 0: _xmin = 0.0
        else: _xmin = _x0
        _y0 = int(floor(N.minimum.reduce(_wcorners[:,1])))
        if _y0 > 0: _ymin = 0.0
        else: _ymin = _y0

        _naxis1 = int(ceil(N.maximum.reduce(_wcorners[:,0]))) - _xmin
        _naxis2 = int(ceil(N.maximum.reduce(_wcorners[:,1]))) - _ymin
        self.wcslin.naxis1 = int(_naxis1)
        self.wcslin.naxis2 = int(_naxis2)

        self.wcslin.crpix1 = self.wcslin.naxis1/2.
        self.wcslin.crpix2 = self.wcslin.naxis2/2.

        # Compute the position of the distortion-corrected image center
        _center = self.apply([(self.wcs.crpix1,self.wcs.crpix2)])
        self.wcslin.cenx = self.wcslin.crpix1 - _center[0]
        self.wcslin.ceny = self.wcslin.crpix2 - _center[1]


    def XYtoSky(self, pos,verbose=no,linear=no):
        """
        This method applies the distortion model to a pixel position
         and calculates the sky position in RA/Dec.
        """
        if not linear and self.model.refpix != None:
            # Perform full solution including distortion
            # Now we need to insure that the input is an array:
            if not isinstance(pos,N.ndarray):
                if N.array(pos).ndim > 1:
                    pos = N.array(pos,dtype=N.float64)
                else:
                    pos = N.array([pos],dtype=N.float64)

            dcx,dcy = self.apply(pos,verbose=yes)

            if dcx != None:
                # Account for displacement of center for sub-arrays
                _cpos = (dcx +self.wcslin.cenx, dcy + self.wcslin.ceny )

                # Now apply linear CD matrix appropriate to model
                xsky,ysky = self.wcslin.xy2rd(_cpos)
            else:
                xsky,ysky = self.wcs.xy2rd(pos)

        else:
            print 'RA/Dec positions without using distortion coefficients:'
            xsky,ysky = self.wcs.xy2rd(pos)

        # Format the results for easy understanding, if desired...
        if verbose:
            rastr,decstr = wcsutil.ddtohms(xsky,ysky,verbose=verbose)
            print 'RA: ',rastr,'  Dec: ',decstr

        # Return the skypos as a tuple (x,y)
        return xsky,ysky

    def SkytoXY(self, skypos, verbose=no, hour=no):
        """
         This method applies the model to an RA/Dec
          and calculates the pixel position.
          RA and Dec need to be in decimal form!

        This needs to be expanded to include full distortion model
        as well, i.e. inverse distortion solution.
        """
        x,y = self.wcs.rd2xy(skypos,hour=hour)

        if verbose:
            print 'X = ',x,' Y = ',y

        # Return the pixel position as a tuple (x,y)
        # return pos
        return x,y

    def calcNewCorners(self):
        """
        This method will compute a new shape based on the positions of
        the corners AFTER applying the geometry model.

        These new position for each corner should be calculated by calling
        self.geometry.apply() on each corner position.
        This should also take into account the output scale as well.

        Values for the corners must go from 0, not 1, since it is a Python array.
            WJH, 17-Mar-2005
        """
        corners = N.zeros(shape=(4,2),dtype=N.float64)
        xin = [0] * 4
        yin = [0] * 4

        xin[0]=1.
        xin[1]=1.
        xin[2]=self.wcs.naxis1
        xin[3]=self.wcs.naxis1
        yin[0]=1.
        yin[1]=self.wcs.naxis2
        yin[2]=self.wcs.naxis2
        yin[3]=1.

        corners[:,0] = xin
        corners[:,1] = yin
        corners[:,0],corners[:,1] = self.apply(corners)
        corners += (self.model.refpix['XDELTA'],self.model.refpix['YDELTA'])
        return corners
