import numarray, math
import displaydev

try:
    import geotrans
except ImportError:
    geotrans = None
#
# Version 0.1-alpha: Initial release 
#       WJH 7-Oct-2003
#
NITER = 0
KREJ = 0

def ipyfitLine(pix,krej,niter):
    _niter = krej
    _krej = niter

    xscale = 2. / (len(pix)-1)
    
    _pix = pix
    
    npix = _pix.getshape()[0]
     
    normx = numarray.array(numarray.arange(npix)) * 2./(npix-1) - 1
    sumxsqr = numarray.sum(numarray.power(_pix,2))
    sumxz = numarray.sum(_pix * normx)
    sumz = numarray.sum(_pix)

    z0 = sumz / npix
    dz = sumxz / sumxsqr        

    for iter in range(_niter):
    
        # Subtract fitted line from data array
        flat = _pix - (normx * dz + z0)

        # Compute k-sigma rejection threshhold
        mean = sumz / npix
        sigma = math.sqrt( float((sumxsqr/(npix-1.)) - (sumz * sumz)/(npix * (npix-1.)) ) )
        thresh = sigma * _krej
        
        # Clip bad pixels from array
        # Unfortunately, this routine does not 'grow' the bad pixels
        # and this makes a difference compared to the IRAF routine.
        #
        # Clip upper outliers
        flat = numarray.choose(numarray.greater(flat,thresh),(flat,0.0))
        # Clip lower outliers
        flat = numarray.choose(numarray.less(flat,-thresh),(flat,0.0))
                
        # remove all bad pixels
        _pix = numarray.compress(_pix,flat)
        _pmin = numarray.minimum.reduce(_pix)
        _pmax = numarray.maximum.reduce(_pix)
        
        normx = numarray.array(numarray.arange(len(_pix))) * 2./(len(_pix)-1) - 1. 
        sumxsqr = numarray.sum(numarray.power(_pix,2))
        sumxz = numarray.sum(_pix * normx)
        sumz = numarray.sum(_pix)
        
        npix = len(_pix)
        z0 = sumz  / npix 
        dz = (_pmax - _pmin) / npix
        #dz = sumxz / sumxsqr

    zstart = z0 - dz
    #zslope = dz * xscale
    zslope = dz

    return npix,zstart,zslope 
    
# This function converts the input image into a byte-array 
# with the z1/z2 values being mapped from 1 - 200.
# It also returns a status message to report on success or failure.
def bscaleImage(image, iz1, iz2):
    
    _pmin = 1
    _pmax = 200
    _ny,_nx = image.getshape()        
    
    bimage = numarray.zeros((_nx,_ny),numarray.UInt8)
            
    if iz2 == iz1: 
        status = "Image scaled to all zeros! Z1: 0, Z2: 0"
        return bimage
    else:
        scale = _pmax / (iz2 - iz1)
        
    # Now we can scale the pixels using a linear scale only (for now)
    bimage = numarray.clip(((image - iz1) * scale),_pmin,_pmax).astype(numarray.UInt8)

    status = 'Image scaled to Z1: '+repr(iz1)+' Z2: '+repr(iz2)+'...'
    return bimage


class NumDisplay:
    def __init__(self,imtdev=None):
        self.frame = 1
        
        # Attributes used to scale image nearly arbitrarily
        # scale : name of numarray ufunc to apply to array
        # factor: string of mathematical operation and scalar to apply to array
        #         i.e. '*1.96e-18', or '/1200.' or '+1.0' 
        self.scale = None
        self.factor = None
        
        # default values for attributes used to determine pixel range values
        self.zscale = 0
        self.stepline = 6
        self.contrast = 1
        self.nlines = 256
        
        # If zrange != 0, use user-specified min/max values
        self.zrange = False
        
        # Scale the image based on input pixel range...
        self.z1 = None
        self.z2 = None
        
        self.name = None
        try:
            self.view = displaydev.ImageDisplayProxy()
            # Now, open the view...
            self.open(imtdev=imtdev)
        except IOError, error:
            raise IOError("No display device (ds9/ximtool) available.")          

    def open(self,imtdev=None):
        """ Open a display device. """
        self.view.open(imtdev=imtdev)
        
    def close(self):
        """ Close the display device entry."""
        self.view.close()
        
    def set(self,frame=None,z1=None,z2=None,scale=None,factor=None):
        
        """ Allows user to set multiple parameters at one time. """
        
        if frame:
            self.frame = frame
        if z1 != None:
            self.z1 = z1
            self.zrange = True
        if z2 != None:
            self.z2 = z2
            self.zrange = True
            
        if scale:
            self.scale = scale
        if factor:
            self.factor = factor
        
        if z1 == None:
            self.zrange = False
            self.z1 = None
            self.z2 = None
        
        # Perform consistency checks here...
        if self.scale != None:
            if (self.scale.find('log') or self.scale.find('sqrt')) and self.z1 <= 0.:
                print 'Minimum pixel value of ',self.z1,' inconsistent with scale of ',self.scale
                raise ValueError,'Please adjust Z1 value...'
            
    def transformImage(self,pix,fbwidth,fbheight):
        
        # Get the image parameters
        _ny,_nx = pix.shape
        
        if _nx > fbwidth or _ny > fbheight:

            # Compute the starting pixel of the image section to be displayed.
            _lx = (_nx / 2) - (fbwidth / 2)
            _ly = (_ny / 2) - (fbheight / 2) 
            # We need to determine the region of the image to be put in frame
            _nx = min(_nx,fbwidth)
            _ny = min(_ny,fbheight)

        else:
            _lx = 0
            _ly = 0
        # Determine pixel range for image (sub)section
        # Insure it does not go beyond actual image array data bounds
        _xstart = max(_lx,0)
        _xend = max( (_lx + _nx),_nx)
        _ystart = max(_ly, 0)
        _yend = max( (_ly + _ny), _ny)    

        # Return bytescaled, frame-buffer trimmed image            
        if (_xstart == 0 and _xend == pix.shape[0]) and (_ystart == 0 and _yend == pix.shape[1]):
            return bscaleImage(pix, self.z1,self.z2)
        else:
            return bscaleImage(pix[_ystart:_yend,_xstart:_xend],self.z1,self.z2)

    def scaleImage(self, pix):
    
        """ Apply user-specified scaling to image. """
        
        # If, for some reason, no scaling is provide, return original array
        if not self.factor and not self.scale:
            return pix

        # If factor is specified, start building eval string to apply it.
        if self.factor:
            _ifac = 'pix'+self.factor
        else:
            _ifac = 'pix'

        if self.zrange == True:
            _ifac = 'numarray.clip('+_ifac+','+str(self.z1)+','+str(self.z2)+')'
            
        # If scaling is specified, build eval string to apply it.
        if self.scale:
            _iscale = 'numarray.'+self.scale+'('+_ifac+')'
        else:
            _iscale = _ifac
        
        # Rescale the image according to user settings...       
        return eval(_iscale)

    
        
    def getZscale (self,image):
        _niter = KREJ
        _krej = NITER
        
        _x = image.shape[1]
        _y = image.shape[0]

        # how pixels are to be extracted
        _ystep = _y / self.nlines

        # start by extracting pixel values from image to determine min/max
        pix = self.scaleImage(image[::_ystep,::self.stepline].copy())
        
        # Remove all pixels which are identically zero.
        pix = numarray.compress(numarray.not_equal(pix.getflat(),0),pix.getflat())
        npix = pix.getshape()[0]

        center_pixel = max(1,(npix+1)/2)
        # This can take the most time depending on the number of elements in pix
        pix.sort()   
        zmin = pix[1]
        zmax = pix[-1]
        zsum = numarray.sum(pix)
        
        # Use full range of pixel values...
        if (self.contrast == 0) or (self.zscale == 1):
           return zmin, zmax

        # Compute the mean of the pixel values
        med = numarray.add.reduce(pix,0)/npix
        
        # Fit a line to the pixel values
        if (npix > 1):
            if geotrans:
                ngood,zmin,zslope = geotrans.pyfitLine(pix,_krej,_niter)
            else:
                ngood,zmin,zslope = ipyfitLine(pix,2.5,3)
        else:
            ngood = 0
            zslope = 0.0
            zmin = 0.0
                
        if (ngood < self.nlines):
            oz1 = zmin
            oz2 = zmax
        else:
            if (self.contrast > 0):
                zslope = zslope / self.contrast

            oz1 = max (zmin, med - (center_pixel - 1) * zslope)
            oz2 = min (zmax, med + (npix - center_pixel) * zslope)
            
        return oz1,oz2

    def display(self,pix,imtdev=None,name=None,bufname=None):
        
        """ Displays byte-scaled (UInt8) numarray to XIMTOOL device. 
            If input is not byte-scaled, it will perform scaling using 
            set values/defaults.
        """
        
        # Initialize the display device
        #displaydev.open(imtdev=imtdev)
        #_d = displaydev._display._display
        _d = self.view._display
        
        if self.z1 == None:
            self.z1 = numarray.minimum.reduce(pix.flat)
            self.z2 = numarray.maximum.reduce(pix.flat)
            
        # If the user has not selected a specific buffer for the display,
        # select and set the frame buffer size based on input image size.
        if bufname != None:
            _d.setFBconfig(None,bufname=bufname)
        else:
            _ny,_nx = pix.getshape()
            _d.selectFB(_nx,_ny,reset=1)

        # Initialize the specified frame buffer
        _d.setFrame(self.frame)
        _d.eraseFrame()

        # Apply user specified scaling to image, returns original
        # if none are specified.
        bpix = self.scaleImage(pix)        
        
        # If image has been rescaled, then recompute the default pixel range
        # based on rescaled pixel values
        if (self.factor or self.scale) and self.zrange == True:
            self.z1 = numarray.minimum.reduce(bpix.flat)
            self.z2 = numarray.maximum.reduce(bpix.flat)
            
        bpix = self.transformImage(bpix,_d.fbwidth,_d.fbheight)

        _wcsinfo = displaydev.ImageWCS(bpix,z1=self.z1,z2=self.z2,name=name)
        # Update the WCS to match the frame buffer being used.
        _d.syncWCS(_wcsinfo)   
        
        # write out WCS to frame buffer, then erase buffer
        _d.writeWCS(_wcsinfo)
        
        # Now, send the trimmed image (section) to the display device
        _d.writeImage(bpix,_wcsinfo)
        #displaydev.close()

    def readcursor(self):
        """ Return the cursor position from the image display. """
        return self.view.readCursor()

view = NumDisplay()

# create aliases for PyDisplay methods
open = view.open
close = view.close

set = view.set
display = view.display
readcursor = view.readcursor
