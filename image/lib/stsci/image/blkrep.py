from __future__ import division

import pyfits
import numpy as np
import math
import sys
from stsci.tools import fileutil, wcsutil

def blkrep(infiles,outfiles,extension_num=None,b1=1,b2=1,b3=1):
    """Performs a block replication across the input image(s),
       based on the input block dimensions designated by the 
       user, and outputs the result to a new FITS file, or files. 
       The value of each pixel is replicated in x and y pixel 
       space based on the user-defined block of pixels, and  
       assigned to all values within the block for the output image.

    Parameters
    ----------
    infiles: string
             List of images to be block replicated.

    outfiles: string
            List of output image names. If the output image 
            name is the same as the input image name, then the 
            block averaged image replaces the input image.

    b1: int
        The number of rows to be produced in block replication 
        (dimension 1, or x).
    
    b2: int
        The number of columns to be produced in block replication 
        (dimension 2, or y).

    b3: int
        The number of layers to be produced in block replication 
        (dimension 1, or z).

    Returns
    -------
    result: FITS files named according to outfiles parameter; 
            block replicated based on b1(x), b2(y), and b3(z) 
            dimensional parameters.
    
    Example
    -------
    >>> import blkrep
    >>> blkrep.blkrep('imageIn1.fits,imageIn2.fits','imageOut1.fits,imageOut2.fits',1,5,5)
    
    """

    infile_split = infiles.split(',')
    outfile_split = outfiles.split(',')

    file_cnt = 0
    for infile in infile_split:

        hdulist = pyfits.open(infile)
        numext = fileutil.countExtn(infile)

        if numext == 0 and (extension_num == None or extension_num == 0):
            prihdr=pyfits.getheader(infile,numext)
            pri_cards=prihdr[0:]

            new_cards=update_hdr(pri_cards,b1,b2,b3)
            outhdr=pyfits.Header(new_cards)

            scidata = hdulist[numext].data
            
        elif numext > 0 and extension_num <= numext:
            prihdr=pyfits.getheader(infile,0)
            scihdr=pyfits.getheader(infile,'SCI',extension_num)

            prihdr_keys=prihdr[0:].keys()
            prihdr_keys=np.asarray(prihdr_keys)
            pri_loc1=np.where(prihdr_keys == 'NAXIS')
            pri_cards1=prihdr[0:pri_loc1[0][0]+1]
            pri_loc2=np.min(np.where(prihdr_keys == 'HISTORY'))
            pri_cards2=prihdr[pri_loc1[0][0]+1:pri_loc2]
            
            scihdr_keys=scihdr[0:].keys()
            scihdr_keys=np.asarray(scihdr_keys)
            sci_loc1=np.where(scihdr_keys == 'NAXIS1')
            sci_cards1=scihdr[sci_loc1[0][0]:sci_loc1[0][0]+2]
            sci_loc2=np.where(scihdr_keys == 'ROOTNAME')
            sci_cards2=scihdr[sci_loc2[0][0]:]

            total_cards=pyfits.CardList(pri_cards1 + sci_cards1 + pri_cards2 + sci_cards2)

            new_total_cards=update_hdr(total_cards,b1,b2,b3)

            outhdr=pyfits.Header(new_total_cards)

            scidata=hdulist['SCI',extension_num].data

        else:
            print '\n******* ERROR *******'
            print 'Incorrect, or no FITS extension specified for '+infile+'!'
            print 'Check specification of parameter "extension_num". '
            print 'Quitting...'
            print '******* ***** *******\n'

        hdulist.close()

        if b3 > 1:
            x_arr_dim = scidata.shape[2]
            x_bin = int(math.ceil(float(x_arr_dim) * b1))

            y_arr_dim = scidata.shape[1]
            y_bin = int(math.ceil(float(y_arr_dim) * b2))

            z_arr_dim = scidata.shape[0]
            z_bin = int(math.ceil(float(z_arr_dim) * b3))

            outarr = np.zeros((z_bin,y_bin,x_bin))

        else:
            x_arr_dim = scidata.shape[1]
            x_bin = int(math.ceil(float(x_arr_dim) * b1))

            y_arr_dim = scidata.shape[0]
            y_bin = int(math.ceil(float(y_arr_dim) * b2))

            outarr = np.zeros((y_bin,x_bin))

        # Loop through the x, y, and z dimensions of the
        # data, as applicable, and bin the pixels based
        # the user designated block size.
        for i,x1 in enumerate(range(0,x_bin,b1)):
            
            x2 = x1 + b1

            # If the end of row is reached,
            # shorten the number of pixels 
            # used in the x_direction
            if x2 > x_bin:
                x2 = x_bin

            for j,y1 in enumerate(range(0,y_bin,b2)):

                y2 = y1 + b2

                # If the end of column is reached,
                # shorten the number of pixels 
                # used in the y_direction
                if y2 > y_bin:
                    y2 = y_bin
                
                if b3 == 1:
                    
                    outarr[y1:y2,x1:x2] = scidata[j,i]

                else:

                    for k,z1 in enumerate(range(0,z_bin,b3)):

                        z2 = z1 + b3

                        # If the end of block is reached,
                        # shorten the number of pixels 
                        # used in the z_direction
                        if z2 > z_bin:
                            z2 = z_bin

                        outarr[z1:z2,y1:y2,x1:x2] = scidata[k,j,i]

        outfile=outfile_split[file_cnt]
        pyfits.writeto(outfile,outarr,outhdr)

        file_cnt = file_cnt + 1


def update_hdr(card_list,b1,b2,b3):

    # Extract necessary header keyword values from output image
    VAFACTOR = card_list['vafactor'].value
    NAXIS1 = card_list['naxis1'].value
    NAXIS2 = card_list['naxis2'].value
    CD1_1 = card_list['cd1_1'].value
    CD1_2 = card_list['cd1_2'].value
    CD2_1 = card_list['cd2_1'].value
    CD2_2 = card_list['cd2_2'].value
    LTV1 = card_list['ltv1'].value
    LTV2 =  card_list['ltv2'].value
    LTM1_1 = card_list['ltm1_1'].value
    LTM2_2 =  card_list['ltm2_2'].value

    # Calculate new header keyword values based on block dimensions

    ltv1 = (1. - np.median(np.array([1,b1]))) + (LTV1 * b1)
    ltv2 = (1. - np.median(np.array([1,b2]))) + (LTV1 * b2)
    ratio_x = b1 / VAFACTOR
    ratio_y = b2 / VAFACTOR
    naxis1 = NAXIS1 * ratio_x
    naxis2 = NAXIS2 * ratio_y
    crpix1 = (naxis1 / 2.) + ltv1
    crpix2 = (naxis2 / 2.) + ltv2
    cd11 = CD1_1 / ratio_x
    cd12 = CD1_2 / ratio_y
    cd21 = CD2_1 / ratio_x
    cd22 = CD2_2 / ratio_y
    ltm1_1 = LTM1_1 * ratio_x
    ltm2_2 = LTM2_2 * ratio_y

    # Set header keywords to new values
    card_list['naxis1'].value = int(math.ceil(naxis1))
    card_list['naxis2'].value = int(math.ceil(naxis2))
    card_list['crpix1'].value = crpix1
    card_list['crpix2'].value = crpix2
    card_list['cd1_1'].value = cd11
    card_list['cd1_2'].value = cd12
    card_list['cd2_1'].value = cd21
    card_list['cd2_2'].value = cd22
    card_list['ltv1'].value = ltv1
    card_list['ltv2'].value = ltv2
    card_list['ltm1_1'].value = ltm1_1
    card_list['ltm2_2'].value = ltm2_2

    # Add new header keywords to the end of the header
    if b3 > 1:
        wcsdim=3
    else:
        wcsdim=2

    card_list.append(pyfits.Card('wcsdim', wcsdim))
    card_list.append(pyfits.Card('wat0_001', 'system=image'))
    card_list.append(pyfits.Card('wat1_001', 'wtype=tan axtype=ra'))
    card_list.append(pyfits.Card('wat2_001', 'wtype=tan axtype=dec'))
    
    return card_list
