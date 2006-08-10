"""numdisplay: Package for displaying numarray arrays in IRAF-compatible
                image display tool such as DS9 or XIMTOOL.

    Displaying a numarray array object involves:
        1.  Opening the connection to a usable display tool (such as DS9).

        2.  Setting the display parameters for the array, such as min and
            max array value to be used for min and max grey scale level, along
            with any offset, scale factor and/or transformation function to be
            applied to the array.
        3.  Applying any transformation to the input array.  This transformation
            could be a simple numarray ufunc or a user-defined function that
            returns a modified array.

        4.  Building the byte-scaled version of the transformed array and
            sending it to the display tool.  The image sent to the display
            device will be trimmed to fit the image buffer defined by the
            'imtdev' device from the 'imtoolrc' or the 'stdimage'
            variable under IRAF. If the image is smaller than the buffer,
            it will be centered in the display device.

            All pixel positions reported back will be relative to the
            full image size.

        This package provides several methods for controlling the display
        of the numarray array; namely,

            open(imtdev=None):
                Open the default display device or the device specified
                in 'imtdev', such as 'inet:5137' or 'fifo:/dev/imt1o'.

            close():
                Close the display device defined by 'imtdev'. This must
                be done before resetting the display buffer to a new size.

            display(pix, name=None, bufname=None, z1=None, z2=None,
                    transform=None, scale=None, offset=None, frame=None):
                Display the scaled array in display tool (ds9/ximtool/...).
                name -- optional name to pass along for identifying array

                bufname -- name of buffer to use for displaying array
                            to best match size of array (such as 'imt1024')
                            [default: 512x512 buffer named 'imt512']

                z1,z2  -- minimum/maximum pixel value to display (float)
                          Not specifying values will default
                          to the full range values of the input array.

                transform -- Python function to apply to array (function)

                scale  -- multiplicative scale factor to apply to array (float/int)

                offset -- additive factor to apply to array before scaling (float/int)

                frame  -- image buffer frame number in which to display array
                            (integer)

                The display parameters set here will ONLY apply to the display
                of the current array.

            readcursor(sample=0):
                Return a single cursor position from the image display.
                By default, this operation will wait for a keystroke before
                returning the cursor position. If 'sample' is set to 1,
                then it will NOT wait to read the cursor.
                This will return a string containing: x,y,frame and key.

            help():
                print Version ID and this message.

    Example:
        The user starts with a 1024x1024 array in the variable 'fdata'.
        This array has min pixel value of -157.04 and a max pixel value
        of 111292.02.  The display tool DS9 has already been started from
        the host level and awaits the array for display.  Displaying the
        array requires:
            >>> import numdisplay
            >>> numdisplay.display(fdata)
        If there is a problem connecting to the DS9 application, the connection
        can be manually started using:
            >>> numdisplay.open()
        To bring out the fainter features, an offset value of 158 can be added
        to the array to allow a 'log' scaling can be applied to the array values
        using:
            >>> numdisplay.display(fdata,transform=numarray.log,offset=158.0)
        To redisplay the image with default full-range scaling:
            >>> numdisplay.display(fdata)

"""

import numarray, math, string
import displaydev

try:
    import geotrans
except ImportError:
    geotrans = None

__version__ = "1.1 (30-Nov-2004)"
#
# Version 0.1-alpha: Initial release
#       WJH 7-Oct-2003
# Version 0.2.0: Removed need to explicitly call 'open()'.
#       RW 30-Nov-2004
#


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
        # transform: name of Python function to operate on array
        #           default is to do nothing (apply self._noTransform)
        # scale : multiplicative factor to apply to input array
        # offset: additive factor to apply to input array

        self.transform = self._noTransform
        self.scale = None
        self.offset = None

        # default values for attributes used to determine pixel range values
        self.zscale = 0
        self.stepline = 6
        self.contrast = 1    # Not implemented yet!
        self.nlines = 256    # Not implemented yet!

        # If zrange != 0, use user-specified min/max values
        self.zrange = 0  # 0 == False

        # Scale the image based on input pixel range...
        self.z1 = None
        self.z2 = None

        self.name = None
        self.view = displaydev._display

    def open(self,imtdev=None):
        """ Open a display device. """
        self.view.open(imtdev=imtdev)

    def close(self):
        """ Close the display device entry."""
        self.view.close()

    def set(self,frame=None,z1=None,z2=None,contrast=None,transform=None,scale=None,offset=None):

        """ Allows user to set multiple parameters at one time. """

        self.frame = frame
        self.zrange = 0  # 0 == False

        if contrast != None:
            self.contrast = contrast
        else:
            self.contrast = None

        if z1 != None:
            self.z1 = z1
            self.zrange = 1  # 1 == True
        else:
            self.z1=None

        if z2 != None:
            self.z2 = z2
            self.zrange = 1  # 1 == True
        else:
            self.z2=None


        if transform:
            self.transform = transform
        else:
            self.transform = self._noTransform

        if scale != None:
            self.scale = scale
        else:
            self.scale = None

        if offset:
            self.offset = offset
        else:
            self.offset = None

    def _noTransform(self, image):
        """ Applies NO transformation to image. Returns original.
            This will be the default operation when None is specified by user.
        """
        return image

    def _bscaleImage(self, image):
        """
        This function converts the input image into a byte-array
         with the z1/z2 values being mapped from 1 - 200.
         It also returns a status message to report on success or failure.

        """
        _pmin = 1.
        _pmax = 200.
        _ny,_nx = image.shape

        bimage = numarray.zeros((_ny,_nx),numarray.UInt8)
        iz1 = self.z1
        iz2 = self.z2

        if iz2 == iz1:
            status = "Image scaled to all one pixel value!"
            return bimage
        else:
            scale =  _pmax / (iz2 - iz1 + 1)

        # Now we can scale the pixels using a linear scale only (for now)
        # Add '1' to clip value to account for zero indexing
        bimage = numarray.clip(((image - iz1+1) * scale),_pmin,_pmax).astype(numarray.UInt8)

        status = 'Image scaled to Z1: '+repr(iz1)+' Z2: '+repr(iz2)+'...'
        return bimage


    def _fbclipImage(self,pix,fbwidth,fbheight):

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
            return self._bscaleImage(pix)
        else:
            return self._bscaleImage(pix[_ystart:_yend,_xstart:_xend])

    def _transformImage(self, pix):
        """ Apply user-specified scaling to the input array. """

        if isinstance(pix,numarray.numarraycore.NumArray):

            if self.zrange:
                zpix = pix.copy()
                zpix = numarray.clip(pix,self.z1,self.z2)
            else:
                zpix = pix
        else:
            zpix = pix
        # Now, what kind of multiplicative scaling should be applied
        if self.scale:
            # Apply any additive offset to array
            if self.offset:
                return self.transform( (zpix+self.offset)*self.scale)
            else:
                return self.transform( zpix*self.scale)
        else:
            if self.offset:
                return self.transform (zpix + self.offset)
            else:
                return self.transform(zpix)

    def display(self, pix, name=None, bufname=None, z1=None, z2=None,
             transform=None, scale=None, offset=None, frame=None):

        """ Displays byte-scaled (UInt8) numarray to XIMTOOL device.
            This method uses the IIS protocol for displaying the data
            to the image display device, which requires the data to be
            byte-scaled.
            If input is not byte-scaled, it will perform scaling using
            set values/defaults.
        """

        # If any of the display parameters are specified here, apply them
        #if z1 or z2 or transform or scale or offset or frame:
        self.set(frame=frame, z1=z1, z2=z2,
                transform=transform, scale=scale, offset=offset)

        # Initialize the display device
        if not self.view._display:
            self.open()
        _d = self.view._display

        # If no user specified values are provided, interrogate the array itself
        # for the full range of pixel values
        if self.z1 == None:
            self.z1 = numarray.minimum.reduce(numarray.ravel(pix))
        if self.z2 == None:
            self.z2 = numarray.maximum.reduce(numarray.ravel(pix))

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

        bpix = self._transformImage(pix)

        # Recompute the pixel range of (possibly) transformed array
        _z1 = self._transformImage(self.z1)
        _z2 = self._transformImage(self.z2)

        # If there was a problem in the transformation, then restore the original
        # array as the one to be displayed, even though it may not be ideal.
        if _z1 == _z2:
            print 'Error encountered during transformation. No transformation applied...'
            bpix = pix
            self.z1 = numarray.minimum.reduce(numarray.ravel(bpix))
            self.z2 = numarray.maximum.reduce(numarray.ravel(bpix))
            # Failsafe in case input image is flat:
            if self.z1 == self.z2:
                self.z1 -= 1.
                self.z2 += 1.
        else:
            # Reset z1/z2 values now so that image gets displayed with
            # correct range.  Also, when displaying transformed images
            # this allows the input pixel value to be displayed, rather
            # than the transformed pixel value.
            self.z1 = _z1
            self.z2 = _z2

        _wcsinfo = displaydev.ImageWCS(bpix,z1=self.z1,z2=self.z2,name=name)
        print 'Image displayed with Z1: ',self.z1,' Z2:',self.z2

        bpix = self._fbclipImage(bpix,_d.fbwidth,_d.fbheight)

        # Update the WCS to match the frame buffer being used.
        _d.syncWCS(_wcsinfo)

        # write out WCS to frame buffer, then erase buffer
        _d.writeWCS(_wcsinfo)

        # Now, send the trimmed image (section) to the display device
        _d.writeImage(bpix,_wcsinfo)
        #displaydev.close()

    def readcursor(self,sample=0):
        """ Return the cursor position from the image display. """
        return self.view.readCursor(sample=sample)

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
