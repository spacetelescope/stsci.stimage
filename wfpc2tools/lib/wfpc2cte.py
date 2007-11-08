""" WFPC2CTE - Module for computing the CTE degradation for WFPC2 images

This module updates the header of the input WFPC2 image with standardized
computations of the effect of CTE based on the algorithm published by Dolphin
(2004, http://purcell.as.arizona.edu/wfpc2_calib).  

ASSUMPTIONS for the COMPUTATION: 
    1. The CTE gets computed for a source at the chip center.
    2. The background gets defined by the clipped mode of the 
        central 200x200 pixels from the image.
    3. The source is assumed to have 100 electrons, 1000 electrons and 
        10000 electrons in the aperture.
    4. The reported CTE is the sum of the XCTE and YCTE computed from 
        Dolphin's algorithm.

INPUT:
The sole input for this task is the filename of the WFPC2 image. 
    
    If the input image is in GEIS format, it will convert it to
    a multi-extension FITS formatted file, then update the FITS file while
    leaving the GEIS image un-modified.  
    
    If the input image is already multi-extension FITS, it will update the
    header directly.
    
    If the input image is waivered FITS, it will quit with a message 
    telling the user to first convert the file to GEIS. The user can then
    provide the GEIS image as input.

OUTPUT:    
The keywords which get updated are:
    CTE100    - CTE for a source with an intensity of 100 electrons 
    CTE1000   - CTE for a source with an intensity of 100 electrons
    CTE10000  - CTE for a source with an intensity of 100 electrons

SYNTAX:
This task can be run on an input WFPC2 image using either of the following 
calls:
    wfpc2cte.compute_CTE(filename,quiet=True)
  -or-
    wfpc2cte.run(filename,quiet=True)

where the filename is the name of the input WFPC2 image.


EXAMPLE:

The syntax for running this task on a WFPC2 file named 'u40x0102m.c0h':
    import wfpc2cte
    wfpc2cte.run('u40x0102m.c0h')
    
The command to print out this help file:
    wfpc2cte.help() 
"""
import numpy as np
import pyfits
import pytools
from pytools import readgeis,fileutil
import imagestats
__version__ = '0.1.0 (6-Nov-2007)'

# This contains the default values in electrons for the CTE sources
DEFAULT_COUNTS = np.array([100,1000,10000],np.float32)

def compute_chip_values(extn):
    chip_values = {}
    
    # Read in required keywords from image headers
    photflam = extn.header['photflam']
    photzpt = extn.header['photzpt']

    # Compute magnitudes for all the cases
    chip_values['mags'] = -2.5 * np.log10(DEFAULT_COUNTS * photflam) + photzpt
    chip_values['lct'] = np.log(DEFAULT_COUNTS) - 7

    # Determine the center of the chip as (Y,X)
    chip_shape = extn.data.shape
    chip_center = (chip_shape[0]/2.,chip_shape[1]/2.)
    chip_values['center'] = (chip_center[0] / 800.,chip_center[1]/800.)
    center_region = extn.data[chip_center[0]-100.:chip_center[0]+100.,
                              chip_center[1]-100.:chip_center[1]+100.]

    # Compute the background as the clipped mode of the region.
    chip_stats = imagestats.ImageStats(center_region,fields='mode',nclip=3)
    chip_background = np.sqrt(np.power(chip_stats.mode,2) + 1)
    chip_values['lbg'] = np.log(chip_background) - 1
    chip_values['bg'] = chip_background - 10    
    
    return chip_values
    
    
def compute_XCTE(xpos,bg):

    return 0.0021*np.exp(-0.234*bg)*xpos/800.
    
def compute_YCTE(chip_values,yr,xcte):

    lbg = chip_values['lbg']
    bg = chip_values['bg']                                
    ypos = chip_values['center'][0]

    #adjust original source counts by XCTE
    lct = chip_values['lct']
    lct += 0.921*xcte

    c1 = 0.0114*(0.670*np.exp(-0.246*lbg)+0.330*np.exp(-0.0359*bg))*(1.+0.335*yr-0.0074*yr*yr)*ypos/800.
    c2 = 3.55*np.exp(-0.474*lct)
    ycte = np.log(np.exp(c1)*(1+c2)-c2)/0.436 

    return ycte
    
    
def update_CTE_keywords(hdr, cte,quiet=False):
    hdr.update('CTE100',cte[0])
    hdr.update('CTE1000',cte[1],after='CTE100')
    hdr.update('CTE10000',cte[2],after='CTE1000')
    if not quiet:
        print 'CTE100   = ',cte[0]
        print 'CTE1000  = ',cte[1]
        print 'CTE10000 = ',cte[2]
    
def compute_CTE(filename,quiet=True):

    newname = None
    if filename.find('.fits') < 0:
        # We are working with a GEIS image
        newname = fileutil.buildFITSName(filename)
        update_mode = 'readonly'
    else:
        # We are working with a FITS image, so update it directly
        newname = filename
        update_mode = 'update'
        
    # Open the image in update mode.
    # If it is a GEIS image on input, convert to FITS using 'newname'   
    fimg = fileutil.openImage(filename,mode=update_mode,fitsname=newname)
    if isinstance(fimg[1],pyfits.TableHDU):
        fimg.close()
        print 'Input image is in the unsupported "waivered" FITS format. '
        print '   Please convert to GEIS or multi-extension FITS format.'
        print 'Exiting...'
        return
         
    obs_date = fimg[0].header['expstart']
    obs_mjd = (obs_date - 50193)/365.25

    obs_gain = fimg[0].header['atodgain']

    for extn in fimg[1:]:
        if extn.header['extname'] == 'SCI':
            chip_values = compute_chip_values(extn)
                
            # Compute XCTE and YCTE for all sources
            xcte = compute_XCTE(chip_values['center'][1],chip_values['bg'])
            ycte = compute_YCTE(chip_values, obs_mjd,xcte)
            total_cte = xcte + ycte
            
            update_CTE_keywords(extn.header, total_cte,quiet=quiet)

    if not quiet:
        print 'Updating keywords in: ',newname
        
    # If a GEIS image was used as the original input, update the new FITS file
    if update_mode == 'readonly':
        fimg.writeto(newname)

    # Close the file
    fimg.close()
           
def run(filename):
    compute_CTE(filename)

def help():
    print __doc__
