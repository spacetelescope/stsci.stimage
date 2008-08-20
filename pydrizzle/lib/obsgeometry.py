import types,string,os,copy
from math import ceil,floor

# Import PyDrizzle utility modules
from pytools import fileutil, wcsutil
from distortion import models,mutil

import numpy as N
import drutil


yes = True
no = False

class ObsGeometry:
    """
        Base class for Observation's geometric information.

        This class must know how to recognize the different
        types of distortion coefficients tables and instantiate
        the correct class for it.
    """

    def __init__(self, rootname, idcfile, idckey=None, chip=1, direction="forward",
                header=None, pa_key=None, new=None, date=None,
                rot=None, ref_pscale=1.0,binned=1, mt_wcs=None):
        """
         We need to setup the proper object for the GeometryModel
          based on the format of the provided idctab.

         We need to trap attempts to address values of 'chip' that
         are not part of the observation; such as in sub-arrays.
        """
        self.header = header
        self.wcs = mt_wcs
        
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
        self.tddcorr = False

        # Default values for VAFACTOR correction
        #self.vafactor = 1.0
        #self.delta_x = 0.0
        #self.delta_y = 0.0

        # Default values for secondary shift (tweak-shift values)
        # These pixel shifts will ALWAYS be in terms of 'input' pixels
        # Set in 'applyAsnShifts'
        self.gpar_xsh = 0.0
        self.gpar_ysh = 0.0
        self.gpar_rot = 0.0

        self.def_rot = None

        if not new:
            if self.wcs == None:
                self.wcs = wcsutil.WCSObject(rootname,header=self.header)
            else:
                self.wcs = self.wcs[str(chip)]

            self.wcs.recenter()
            self.wcslin = self.wcs.copy()
            # Implement time-dependent skew for those cases where it is needed
            # Compute the time dependent distrotion skew terms

            # default date of 2004.5 = 2004-7-1
            if (self.header.has_key('WFCTDD') and self.header['WFCTDD'] == 'T') or \
                (self.header.has_key('TDDCORR') and self.header['TDDCORR'] == 'PERFORM'):
                self.alpha,self.beta = mutil.compute_wfc_tdd_coeffs(self.header['date-obs'])
                self.tddcorr = True
            else:
                self.alpha = 0
                self.beta  = 0

            # Based on the filetype, open the correct geometry model
            if idckey != None:
                ikey = string.lower(idckey)
            else:
                ikey = 'idctab'

            self.ikey = ikey

            if ikey == 'idctab':
                self.model =  models.IDCModel(self.idcfile,
                    chip=chip, direction=self.direction, date=self.date,
                    filter1=_filt1, filter2=_filt2, offtab=_offtab, binned=binned,
                    tddcorr=self.tddcorr)
            elif ikey == 'cubic':
                scale = self.wcs.pscale / ref_pscale
                self.model = models.DrizzleModel(self.idcfile, scale=scale)
                self.model.pscale = scale
                self.model.refpix['PSCALE']  = self.model.pscale
                self.model.refpix['XREF'] = self.wcs.naxis1/2.
                self.model.refpix['YREF'] = self.wcs.naxis2/2.
                
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
                self.model = models.TraugerModel(self.idcfile,float(_lam)/10.)

            elif ikey == 'wcs':
                if self.header.has_key('IDCSCALE'):
                    self.model = models.WCSModel(self.header, rootname)
                else:
                    print 'WARNING: Not all SIP-related keywords found!'
                    print '         Reverting to use of IDCTAB for distortion.'
                    
                    self.model =  models.IDCModel(self.idcfile,
                        chip=chip, direction=self.direction, date=self.date,
                        filter1=_filt1, filter2=_filt2, offtab=_offtab, binned=binned)

            else:
                raise ValueError, "Unknown type of coefficients table %s"%idcfile
            
            if self.idcfile == None and ikey != 'wcs':
                #Update default model with WCS plate scale
                self.model.setPScaleCoeffs(self.wcs.pscale)

            # Insure that a default model pscale has been set
            if self.model.refpix['PSCALE'] == None:
                self.model.pscale = 1.0
                self.model.refpix['PSCALE']  = self.model.pscale
            if self.model.refpix['XREF'] == None:
                self.model.refpix['XREF'] = self.wcs.naxis1/2.
                self.model.refpix['YREF'] = self.wcs.naxis2/2.
                
            
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

            # Generate linear WCS to linear CD matrix
            self.undistortWCS()
        else:
            # For new images with no distortion, CD matrix is sufficient.
            self.wcs = wcsutil.WCSObject(None)
            self.wcslin = wcsutil.WCSObject(None)
            self.model = models.GeometryModel()
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
        
        """
        verify that pscale is wcs.pscale ...
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
        #_xp = _naxis[0]/2. + 0.5
        #_yp = _naxis[1]/2. + 0.5
        _xp = _naxis[0]/2. 
        _yp = _naxis[1]/2. 
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

    def invert(self,pixpos,outwcs,error=None,maxiter=None):
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
        # Also convert X,Y tuple to a numpy
        pp = N.array([pixpos]) + N.array((self.wcs.offset_x,self.wcs.offset_y),dtype=N.float64)

        # We are going to work with three x,y points as a Numpy array
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

            # Apply the forward transform for this chip
            tout=self.wtraxy(pos,outwcs)
            # Convert back to Numpy
            out=N.array(tout,dtype=N.float64)

            # Work out the shifts matrix
            shift[0]=out[1]-out[0]
            shift[1]=out[2]-out[0]

            # Invert the matrix (this probably should be a separate method)
            shift = N.linalg.inv(shift)

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

            dcx,dcy = self.apply(pos,verbose=no)

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

        xin[0]=0.
        xin[1]=0.
        xin[2]=self.wcs.naxis1
        xin[3]=self.wcs.naxis1
        yin[0]=0.
        yin[1]=self.wcs.naxis2
        yin[2]=self.wcs.naxis2
        yin[3]=0.

        corners[:,0] = xin
        corners[:,1] = yin
        corners[:,0],corners[:,1] = self.apply(corners)
        corners += (self.model.refpix['XDELTA'],self.model.refpix['YDELTA'])
        return corners
