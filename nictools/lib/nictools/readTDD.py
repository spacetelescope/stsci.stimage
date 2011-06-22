"""

The readTDD.py is a helper module used to extract the linear dark
and amp glow components from a NICMOS time dependent dark file.


:author: Christopher Hanley

:dependencies: stsci.tools.fileutil

"""

__docformat__ = 'restructuredtext'


from stsci.tools.fileutil import openImage

class darkobject(object):
    def __init__(hdulist):
        """ 
        darkobject:  This class takes as input a pyfits hdulist object.
        The linear dark and amp glow noise componenets are then extracted
        from the hdulist.
        """
        
        self.lindark = hdulist['LIN']
        self.ampglow = hdulist['AMPGLOW']
        
    def getlindark(self):
        """
        getlindark: darkobject method which is used to return the linear
        dark component from a NICMOS temperature dependent dark file.
        """
        return self.lindata.data
        
    def getampglow(self):
        """
        getampglow: darkobject method which us used to return the amp
        glow component from a NICMOS temperature dependent dark file.
        """
        return self.ampglow.data

    def getlindarkheader(self):
        """
        getlindarkheader: darkobject method used to return the header
        information of the linear dark entension of a TDD file.
        """
        return self.lindata.header
    
    def getampglowheader(self):
        """
        getampglowheader: darkobject method used to return the header
        information of the amp glow entension of a TDD file.
        """
        return self.ampglow.header
    
    
def fromcalfile(filename):
    """
    fromcalfile: function that returns a darkobject instance given the
    name of a cal.fits file as input.  If there is no TEMPFILE keyword
    in the primary header of the cal.fits file or if the file specified
    by TEMPFILE cannot be found, a None object is returned.
    """
    hdulist = openImage(filename)
    
    if hdulist[0].header.has_key('TEMPFILE'):
        if tddfile == 'N/A':
            return None
        else:
            tddfile = hdulist[0].header['TEMPFILE']
            tddhdulist = openImage(tddfile)
            return darkobject(tddhdulist)
    else:
        return None
        
