""" WFPC2CTE - Module for computing the CTE degradation for WFPC2 images

This module updates the header of the input WFPC2 image with standardized
computations of the effect of CTE based on the algorithm published by Dolphin
(2004, http://purcell.as.arizona.edu/wfpc2_calib/2004_12_20.html).

:ASSUMPTIONS for the COMPUTATION:

    1. The CTE gets computed for a source at the chip center.
    2. The background (in electrons) gets defined by the clipped mode of the
        central 200x200 pixels from the image.
    3. The source is assumed to have 100 electrons, 1000 electrons and
        10000 electrons in the aperture.
    4. The reported CTE is the sum of the XCTE and YCTE computed from
        Dolphin's algorithm.

:INPUT:

  The sole input for this task is the filename of the WFPC2 image.

    If the input image is in GEIS format, it will convert it to
    a multi-extension FITS formatted file, then update the FITS file while
    leaving the GEIS image un-modified.

    If the input image is already multi-extension FITS, it will update the
    header directly.

    If the input image is waivered FITS, it will quit with a message
    telling the user to first convert the file to GEIS. The user can then
    provide the GEIS image as input.

:OUTPUT:

  The keywords which get updated are::

    CTE_1E2  - CTE for a source with an intensity of 100 electrons
    CTE_1E3  - CTE for a source with an intensity of 1000 electrons
    CTE_1E4  - CTE for a source with an intensity of 10000 electrons

:SYNTAX:

  This task can be run on an input WFPC2 image using either of the following calls::

    wfpc2cte.compute_CTE(filename,quiet=True)
    -or-
    wfpc2cte.run(filename,quiet=True)

  where the filename is the name of the input WFPC2 image.


:EXAMPLE:

  The syntax for running this task on a WFPC2 file named 'u40x0102m.c0h'::

    import wfpc2cte
    wfpc2cte.run('u40x0102m.c0h')

  The command to print out this help file::

    wfpc2cte.help()

:FUNCTIONS:

"""
from __future__ import division # confidence medium

import numpy as np
import pyfits
import stsci.tools
from stsci.tools import readgeis, fileutil
import stsci.imagestats as imagestats
__version__ = '1.2.4 (2-July-2008)'

# This contains the default values in electrons for the CTE sources
DEFAULT_COUNTS = np.array([100,1000,10000],np.float32)

chip_bg_factor = 10
chip_lbg_factor = 1
chip_lct_factor = 7

def compute_chip_values(extn,gain,nclip=3):
    chip_values = {}

    # Determine the center of the chip as (Y,X)
    chip_shape = extn.data.shape
    chip_center = (chip_shape[0]/2.,chip_shape[1]/2.)
    chip_values['center'] = (chip_center[0] / 800.,chip_center[1]/800.)
    center_slice = [slice(chip_center[0]-100.,chip_center[0]+100.),
                    slice(chip_center[1]-100.,chip_center[1]+100.)]
    center_region = extn.data[center_slice]
    # Compute the background as the clipped mode of the region.
    chip_stats = imagestats.ImageStats(center_region,fields='mode',
                                    lower=-99.0,upper=4096.,
                                    nclip=nclip,binwidth=0.01)
    if chip_stats.mode >= 0:
        chip_background = np.sqrt(np.power(chip_stats.mode,2) + 1)
    else:
        chip_background = 1.0
    chip_background *= gain
    chip_values['bg_raw'] = chip_background
    chip_values['lbg'] = np.log(chip_background) - chip_lbg_factor
    chip_values['bg'] = chip_background - chip_bg_factor

    # Compute e^(counts) for all the cases
    chip_values['lct'] = np.log(DEFAULT_COUNTS) - chip_lct_factor

    return chip_values

def compute_XCTE(xpos,bg):

    return 0.0021*np.exp(-0.234*bg)*xpos

def compute_YCTE(chip_values,yr,xcte):

    lbg = chip_values['lbg']
    bg = chip_values['bg']
    ypos = chip_values['center'][0]

    #adjust original source counts by XCTE
    lct = chip_values['lct']
    lct += 0.921*xcte

    c1 = 0.0114*(0.670*np.exp(-0.246*lbg)+0.330*np.exp(-0.0359*bg))*(1.+0.335*yr-0.0074*yr*yr)*ypos
    c2 = 3.55*np.exp(-0.474*lct)
    ycte = np.log(np.exp(c1)*(1+c2)-c2)/0.436

    return ycte


def update_CTE_keywords(hdr, cte,quiet=False,update=True):
    # Start by checking to see if the keywords to be updated already exist
    # If not, print a warning and insure quiet=False so the results get
    # reported to STDOUT at the very least.
    if not hdr.has_key('CTE_1E2'):
        print "WARNING: CTE keywords not found in %s,%d header."%(hdr['extname'],hdr['extver'])
        print "         Adding the keywords exclusively to "
        print "         the FITS file's extension header."
        quiet = False

    if update:
        hdr.update('CTE_1E2',cte[0])
        hdr.update('CTE_1E3',cte[1],after='CTE_1E2')
        hdr.update('CTE_1E4',cte[2],after='CTE_1E3')

    if not quiet:
        print 'CTE_1E2   = ',cte[0]
        print 'CTE_1E3  = ',cte[1]
        print 'CTE_1E4 = ',cte[2]

def compute_CTE(filename,quiet=True,nclip=3,update=True):
    """
    Compute the CTE correction for a 100, 1000 and 1e+4 DN source in a WFPC2 chip.
    These correction values will be written to the WFPC2 image header as the
    CTE_1E2, CTE_1E3 and CTE_1E4 keywords respectively.

    Parameters
    ----------
    filename : str
        Name of WFPC2 image
    quiet : bool, optional [Default: True]
        Specifies whether or not to print verbose messages during processing
    nclip : int [Default: 3]
        Number of clipping iterations for computing the chip's pixel values
    update : bool [Default: True]
        Specifies whether or not to update the input image header with the
        computed CTE correction values

    """

    newname = None
    if filename.find('.fits') < 0:
        # We are working with a GEIS image
        newname = fileutil.buildFITSName(filename)
    else:
        # We are working with a FITS image, so update it directly
        newname = filename

    update_mode = 'update'
    # Open the image in update mode.
    # If it is a GEIS image on input, convert to FITS using 'newname'
    # then update the FITS file only...
    fimg = fileutil.openImage(filename,mode=update_mode,fitsname=newname)
    fimg.info()
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
            chip_values = compute_chip_values(extn,obs_gain,nclip=nclip)

            # Compute XCTE and YCTE for all sources
            xcte = compute_XCTE(chip_values['center'][1],chip_values['bg'])
            ycte = compute_YCTE(chip_values, obs_mjd,xcte)
            if not quiet:
                print 'Background computed to be: ',chip_values['bg_raw']

            # Based on Workshop 2002 paper, after equation 8...
            total_cte = xcte + ycte

            update_CTE_keywords(extn.header, total_cte,quiet=quiet,update=update)

    if not quiet and update:
        print 'Updating keywords in: ',newname

    fimg.flush()

    # Close the file
    fimg.close()

def run(filename,quiet=True,nclip=3):
    compute_CTE(filename,quiet=quiet,nclip=nclip)

def help():
    print __doc__
