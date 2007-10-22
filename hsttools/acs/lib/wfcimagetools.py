import pyfits as p
import numpy as n
from pytools import parseinput


def stitch(inputfilename, extn, inputregion, outputfilename, outputsize, outputregion):
    """
    Purpose
    =======
    Extract an image section from an input image and place it in a new image file in 
    a given region.  All other pixel values of the output image will be zero.
    
    :inputfilename: String representing the name of the input file.

    :extn: Integer representing the extension number of the inputfile to extract.

    :inputregion: Tuple representing the region to extract.  It will be of the form (i1,i2,j1,j2)
    
    :outputfilename: String representing the name of the output file.
    
    :outputsize:  Tuple representing the dimensions of the output image.  It will be of the from (x,y)
    
    :outputregion: Tuple representing the location to place the image extract in the new array.
                    It will be of the form (i1,i2,i3,i4).
                    
    Calling Sequence Sample
    =======================
    imagemanip.stitch("test.fits",1,(100,150,100,150),"test2.fits",(200,200),(100,150,100,150))
    
    """
    
    #Open the input file.
    inputHDUlist = p.open(inputfilename)
    
    #Extract the region of interest.
    input = inputHDUlist[extn].data[inputregion[0]:inputregion[1],inputregion[2]:inputregion[3]]
    input2 = inputHDUlist[extn+1].data[inputregion[0]:inputregion[1],inputregion[2]:inputregion[3]]
    input3 = inputHDUlist[extn+2].data[inputregion[0]:inputregion[1],inputregion[2]:inputregion[3]]
     
    #Create reference to original image data.
    outHDUlist = inputHDUlist
    outHDUlist.append(inputHDUlist[1])
    outHDUlist.append(inputHDUlist[2])
    outHDUlist.append(inputHDUlist[3])
    
    #Create output array
    outHDUlist[1].data = n.zeros(outputsize,dtype=inputHDUlist[extn].data.dtype)
    outHDUlist[2].data = n.zeros(outputsize,dtype=inputHDUlist[extn+1].data.dtype)
    outHDUlist[3].data = n.zeros(outputsize,dtype=inputHDUlist[extn+2].data.dtype)
    outHDUlist[4].data = n.zeros(outputsize,dtype=inputHDUlist[extn].data.dtype)
    outHDUlist[5].data = n.zeros(outputsize,dtype=inputHDUlist[extn+1].data.dtype)
    outHDUlist[6].data = n.zeros(outputsize,dtype=inputHDUlist[extn+2].data.dtype)
    
    #Write out the new file
    outHDUlist.writeto(outputfilename)
    outHDUlist.close()
     
    #Open the file in update mode
    outHDUlist = p.open(outputfilename,mode="update")
            
    #Assign the image extract to the region of interest in the output array.
    outdata = n.zeros(outputsize,dtype=inputHDUlist[extn].data.dtype)
    outdata2 = n.zeros(outputsize,dtype=inputHDUlist[extn+1].data.dtype)
    outdata3 = n.zeros(outputsize,dtype=inputHDUlist[extn+2].data.dtype)
    outdata[outputregion[0]:outputregion[1],outputregion[2]:outputregion[3]] = input
    outdata2[outputregion[0]:outputregion[1],outputregion[2]:outputregion[3]] = input2
    outdata3[outputregion[0]:outputregion[1],outputregion[2]:outputregion[3]] = input3
    
    #Overwrite the existing image extension with the new output array. 
    if inputHDUlist[1].header['ccdchip'] == 2:
        e = 1
    else:
        e = 4
    outHDUlist[e].data = outdata
    outHDUlist[e+1].data = outdata2
    outHDUlist[e+2].data = outdata3
    outHDUlist[4].header['EXTVER'] = 2
    outHDUlist[5].header['EXTVER'] = 2
    outHDUlist[6].header['EXTVER'] = 2
    
    #Write out the new file
    outHDUlist.close()

    #Cleanup
    inputHDUlist.close()


def createfullframe(input):
    """
    Purpose
    =======
    Given ACS WFC subarray data, place the subarray in its location of the full frame. 
    """

    inputlist,outfilename  = parseinput.parseinput(input)
    for file in inputlist:
        print "Working on file: ",file
        h = p.open(file)
        header = h["sci"].header
        h.close()
        print header['naxis1'],header['ltv1'],header['naxis2'],header['ltv2']
        stitch(file,1,(0,header['naxis2'],0,header['naxis1']),"ff_"+file,\
               (2048,4096),\
               (0-header['ltv2'],header['naxis2']-header['ltv2'],\
                0-header['ltv1'],header['naxis1']-header['ltv1']))
    
