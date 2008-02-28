from pytools.fileutil import openImage

class darkobject(object):
    def __init__(hdulist):
        self.lindark = hdulist['LIN']
        self.ampglow = hdulist['AMPGLOW']
        
    def getlindark(self):
        return self.lindata.data
        
    def getampglow(self):
        return self.ampglow.data

    def getlindarkheader(self):
        return self.lindata.header
    
    def getampglowheader(self):
        return self.ampglow.header
    
    
def fromcalfile(filename):
    hdulist = openImage(filename)
    
    if hdulist[0].header.has_key('TEMPFILE'):
        tddfile = hdulist[0].header['TEMPFILE']
        
    if tddfile == 'N/A':
        return None
    else:
        tddhdulist = openImage(tddfile)
        return darkobject(tddhdulist)
