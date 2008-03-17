#
#   Authors: Christopher Hanley
#   Program: ir_input.py
#   Purpose: Class used to model IR specific instrument data.

from input_image import InputImage

class IRInputImage(InputImage):
    """
    
    IRInputImage
    ------------
    
    The IRInputImage class is the parent class for all of 
    the IR based instrument classes.

    """

    def __init__(self,input,dqname,platescale,memmap=0):
        """
        Constructor for IRInputImage class object.
        """
        InputImage.__init__(self,input,dqname,platescale,memmap=0)
        
    def isCountRate(self):
        """
        isCountRate: Method or IRInputObject used to indicate if the
        science data is in units of counts or count rate.  This method
        assumes that the keyword 'BUNIT' is in the header of the input
        FITS file.
        """
        
        if self.header.has_key('BUNIT'):       
            if self.header['BUINT'].find("/") != -1:
                return True
            else:
                return False
        else:
        
