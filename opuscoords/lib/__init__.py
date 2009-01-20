""" opuscoords - Coordinate conversion package from RA/Dec to Galactic/Ecliptic

This package implements a Python interface to 2 coordinate transformation routines used in Generic Conversion in the HST calibration pipeline. 

Description of Algorithm
========================
The functions are described in the code as follows:  
*       Converts (right ascension, declination) to Galactic (longitude,
*       latitude), System II assuming J2000 equinox. This routine was
*       rewritten (OPR.52126) to use the most precise calculation based on
*       definitional parameters for the Galactic Coordinate system II found
*       on the Web at http://ledas-cxc.star.le.ac/uk/udocs/PG/html/node76.html.
*       The values obtained from this html file were:
*       Galactic center (0,0) in J2000 RA and DEC is at 
*       (17 45 37.20 -28 56 10.22) which when converted to degrees is
*       (266.4050000 -28.9361722). The Galactic N Pole in J2000 RA and DEC is
*       (12 51 26.28 +27 07 41.70), which when converted to degrees is
*       (192.8595000 27.1282500). These two sets of coordinates can be 
*       converted to orthogonal unit vectors and the crossproduct provides the
*       vector for the third direction. Because of rounding error for the input
*       values the normalized crossproduct and the galactic center vector can be
*       used in another cross-product that gives an accurate orthonormal set of
*       vectors within the full precision of these calculations. This conversion
*       matrix has been calulated and the result is recorded to 14 digits (in 
*       excess of the accuracy of the input data). Any imprecision is due to the
*       input parameters not the conversion matrix. 
More information about the system used for the output values can be found at:

http://cxc.harvard.edu/ciao/ahelp/prop-coords.html
http://cxc.harvard.edu/contrib/jcm/ncoords.ps

instead of the URL listed in the code comments above.


This package uses the same (unaltered) C functions used in Generic Conversion and compiles them into a Python sharable object for this interface.  

Syntax
=======
The routines translate RA and Dec in J2000 to either galactic or ecliptic coordinates using the following Python signatures:
  
  >>> import opuscoords
  >>> # Convert RA,Dec to Galactic coordinates
  >>> glat,glon = opuscoords.radec2glatlon(ra,dec)
  >>> # Convert RA,Dec to Ecliptic coordinates
  >>> elat,elon = opuscoords.radec2elatlon(ra,dec)

"""

__version__ = "1.0.0"
__version_date__ = "20 Jan 2009"

from GCcoords import radec2elatlon,radec2glatlon

def help():
  print __doc__
