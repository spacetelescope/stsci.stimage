from input_image import InputImage

class STISInputImage (InputImage):

    def __init__(self, input):
        InputImage.__init__(self,input)
        self.instrument = 'STIS/CCD'
        
        self.gain = 1.
        self.readnoise = 5.

    def subtractSky(self, parlist, skytype, skyname, skywidth, skystat, skylower, skyupper):
        """ Subtract background sky from SCI array. """
        skytype = 'single'
        if skystat == 'median':  skystat='mode'

        InputImage.subtractSky(self,parlist, skytype, skyname, skywidth, skystat, skylower, skyupper)

