"""numdisplay: Package for displaying numarray arrays in IRAF-compatible
                image display tool such as DS9 or XIMTOOL.
    
    Displaying a numarray array object involves:
        1.  Opening the connection to a usable display tool (such as DS9).
        2.  Setting the display parameters for the array, such as min and
            max array value to be used for min and max grey scale level,
            the offset to be applied to the array before displaying, and 
            the scaling of array values within the min/max range.
        3.  Building the byte-scaled version of the array and sending it
            to the display tool.
    
        This package provides several methods for controlling the display
        of the numarray array; namely,
        
            open(imtdev=None): 
                open the default display device or the device specified 
                in 'imtdev'
            
            close():
                close the display device handle
            
            set(z1=None,z2=None,scale=None,factor=None,frame=None):
                convenience method for setting display attributes where
                z1,z2  -- minimum/maximum pixel value to display (float)
                          Explicitly setting 'z1=None' resets the range 
                          to the full range values of the input array.
                scale  -- numarray ufunc name to use for scaling array (string)
                factor -- additive/multiplicative factor to apply to array
                          before scaling (string)
                frame  -- image buffer frame number in which to display array
                            (integer)
                            
            display(pix, name=None, bufname=None):
                display the scaled array in display tool (ds9/ximtool/...)
                name -- optional name to pass along for identifying array
                bufname -- name of buffer to use for displaying array
                            to best match size of array (such as 'imt1024')
                            [default: 512x512 buffer named 'imt512']
                
            readcursor():
                return a single cursor position from the image display
            
            help():
                print Version ID and this message.

    Example:            
        The user starts with a 1024x1024 array in the variable 'fdata'.
        This array has min pixel value of -157.04 and a max pixel value 
        of 111292.02.  The display tool DS9 has already been started from
        the host level and awaits the array for display.  Displaying the
        array requires:
            >>> import numdisplay
            >>> numdisplay.open()
            >>> numdisplay.display(fdata)
        To bring out the fainter features, a 'log' scaling can be applied
        to the positive array values using:
            >>> numdisplay.set(z1=1.,scale='log')
            >>> numdisplay.display(fdata)
        The full pixel range can be displayed with 'log' scaling with:
            >>> numdisplay.set(z1=None,factor='+158.0')
            >>> numdisplay.display(fdata)
        The value of 'z1' is reset to 'None' to force the full range of
        pixel values in the original array to be used again.
        
"""

import numarray, math
import displaydev

try:
    import geotrans
except ImportError:
    geotrans = None

__version__ = "0.1alpha (9-Oct-2003)"
#
# Version 0.1-alpha: Initial release 
#       WJH 7-Oct-2003
#
    
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
    """ Class to manage the attributes and methods necessary for displaying
        the array in the image display tool.
        
        This class contains the methods:
            open(imtdev=None): 
            
            close():
            
            set(z1=None,z2=None,scale=None,factor=None,frame=None):
                            
            display(pix, name=None, bufname=None):
                
            readcursor():
            
    """
    def __init__(self):
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
        
        self.frame = frame
        
        if z1 != None:
            self.z1 = z1
            self.zrange = True
        else:
            self.z1=None

        if z2 != None:
            self.z2 = z2
            self.zrange = True
            
        self.scale = scale
        self.factor = factor
        
        if self.z1 == None:
            self.zrange = False
            self.z2 = None
        
        # Perform consistency checks here...
        if self.scale != None:
            if (self.scale.find('log') or self.scale.find('sqrt')) and self.z1 <= 0.:
                print 'Minimum pixel value of ',self.z1,' MAY be inconsistent with scale of ',self.scale
            
    def _transformImage(self,pix,fbwidth,fbheight):
        
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

    def _scaleImage(self, pix):
    
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

    
    def display(self, pix, name=None, bufname=None):
        
        """ Displays byte-scaled (UInt8) numarray to XIMTOOL device. 
            If input is not byte-scaled, it will perform scaling using 
            set values/defaults.
        """
        
        # Initialize the display device
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
        bpix = self._scaleImage(pix)        
        
        # If image has been rescaled, then recompute the default pixel range
        # based on rescaled pixel values
        if (self.factor or self.scale) or self.zrange == True:
            self.z1 = numarray.minimum.reduce(bpix.flat)
            self.z2 = numarray.maximum.reduce(bpix.flat)
        
        bpix = self._transformImage(bpix,_d.fbwidth,_d.fbheight)

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
    
# Help facility
def help():
    """ Print out doc string with syntax and example. """
    print 'numdisplay --- Version ',__version__
    print __doc__


view = NumDisplay()

# create aliases for PyDisplay methods
open = view.open
close = view.close

set = view.set
display = view.display
readcursor = view.readcursor

