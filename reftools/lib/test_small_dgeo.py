#!/usr/bin/env python

from __future__ import division # confidence high

import  pyfits, pywcs
from pywcs import WCS, DistortionLookupTable
import numpy as np
from matplotlib import pyplot as pl
#pl.hold(False)
from stwcs import wcsutil,updatewcs
import numdisplay
from pytools import fileutil
import wtraxyutils
import os
import imagestats

__version__ = '0.3.1'
__vdate__ = '2010-04-16'

#import gc  #call gc.collect() occasionally

def help():
    print run.__doc__

def build_grid_arrays(nx,ny,step):
    grid = [nx,ny,step]
    print 'grid of : ',nx,'x',ny,' by ',step
        
    #grid=[4096,2048,1]
    xpts = np.array(range(1,grid[0]+1,grid[2]),np.float32)
    ypts = np.array(range(1,grid[1]+1,grid[2]),np.float32)
    xygrid = np.meshgrid(xpts,ypts)
    xarr = xygrid[0].flatten()
    yarr = xygrid[1].flatten()

    return xarr,yarr

def transform_d2im_dgeo(img,extver,xarr,yarr,verbose=False):

    print 'setting up WCS object for ',img,'[sci,',str(extver),']'

    w = wcsutil.HSTWCS(img, ext=('sci',extver))
    d2imx,d2imy = w.det2im(xarr,yarr,1)
    if verbose:
        print 'Det2im results span: '
        print d2imx.min(),d2imx.max(),d2imx.shape
        print d2imy.min(),d2imy.max(),d2imy.shape
    
    xout, yout = w.p4_pix2foc(d2imx,d2imy,1)
    if verbose:
        print 'pix2foc results span: '
        print xout.min(),xout.max(),xout.shape
        print yout.min(),yout.max(),yout.shape

    return xout,yout

def run(scifile,dgeofile=None,output=False,match_sci=False,update=True,vmin=None,vmax=None):
    """ 
        This routine compares how well the sub-sampled DGEOFILE (generated 
        using the 'makesmall' module) corrects the input science image as 
        opposed to the full-size DGEOFILE.  
        
        SYNTAX:
            import test_small_dgeo
            test_small_dgeo.run(scifile,dgeofile=None,output=False)
        
        where:
            scifile   - name of science image
            dgeofile  - name of full-sized DGEOFILE if not in DGEOFILE keyword
            output    - if True, write out differences to FITS file(s)

        The user can either specify the full-size DGEOFILE reference filename
        as the 'dgeofile' parameter or the code will look for the 'DGEOFILE' 
        keyword in the primary header for the name of the full-sized reference
        file.  
        
        The primary output will be a series of plots showing the difference images
        with the mean and stddev of the differences in the label of the image display.
        
        If the 'output' parameter is set to True, these differences 
        will then be written out to FITS files based on the input science image
        rootname. Both the DX and DY differences for a single chip will be written
        out to the same file, with a separate file for each chip. 
        
    """
    if update:
        # update input SCI file to be consistent with reference files in header
        print 'Updating input file ',scifile,' to be consistent with reference files listed in header...'
        updatewcs.updatewcs(scifile)
    # Now, get the original NPOLFILE and overwrite the data in the scifile
    # WCSDVARR extensions to remove the scaling by the linear terms imposed by 
    # the SIP convention
    npolfile = fileutil.osfn(pyfits.getval(scifile,'NPOLFILE'))
    npolroot = os.path.split(npolfile)[1]
    dxextns = []
    for extn in pyfits.open(npolfile):
        if extn.header.has_key('extname') and extn.header['extname'] in ['DX','DY']:
            dxextns.append([extn.header['extname'],extn.header['extver']])
    #dxextns = [['dx',1],['dy',1],['dx',2],['dy',2]]
    ndxextns = len(dxextns)
    # Update input file with NPOLFILE arrays now
    print 'Updating input file ',scifile,' with original '
    print '    NPOLFILE arrays from ',npolfile
    fsci =pyfits.open(scifile,mode='update')
    try:
        next = fsci.index_of(('wcsdvarr',1))
    except KeyError:
        fsci.close()
        print '====='
        print 'ERROR: No WCSDVARR extensions found!'
        print '       Please make sure NPOLFILE is specified and run this task with "update=True".'
        print '====='
        return
    # Replace WCSDVARR arrays here...
    for dxe,wextn in zip(dxextns,range(1,ndxextns+1)):
        fsci['wcsdvarr',wextn].data = pyfits.getdata(npolfile,dxe[0],dxe[1])
    # Now replace the NPOLEXT keyword value with a new one so that it will automatically
    # update with the correct file next time updatewcs is run.
    fsci['sci',1].header['npolext'] = npolroot
    print 'Updated NPOLEXT with ',npolroot
    fsci.close()
    print '\n====='
    print 'WARNING: Updated file ',scifile,' NO LONGER conforms to SIP convention!'
    print '         This file will need to be updated with updatewcs before using with MultiDrizzle.'
    print '=====\n'
    
    # Get info on full-size DGEOFILE
    if dgeofile is None:
        # read in full dgeofile from header
        fulldgeofile = pyfits.getval(scifile,'DGEOFILE')
    else:
        fulldgeofile = dgeofile
        
    print 'Opening full-size DGEOFILE ',fulldgeofile,' for comparison.'
    fulldgeofile = fileutil.osfn(fulldgeofile)
    full_shape = [pyfits.getval(fulldgeofile,'NAXIS2','DX',1),pyfits.getval(fulldgeofile,'NAXIS1','DX',1)]
    
    filter_names = fileutil.getFilterNames(pyfits.getheader(scifile))

    detector = pyfits.getval(fulldgeofile,'DETECTOR')
    # count the number of chips in DGEOFILE 
    xyfile = pyfits.open(scifile)
    numchips = 0
    ccdchip = []
    extname = xyfile[1].header['EXTNAME']
    for extn in xyfile:
        if extn.header.has_key('extname') and extn.header['extname'] == extname:
            numchips += 1
            if extn.header.has_key('ccdchip'):
                ccdchip.append(extn.header['ccdchip'])
            else:
                ccdchip.append(1)
    if not match_sci:
        ltv1 = 0
        ltv2 = 0  
        nx = full_shape[1]
        ny = full_shape[0]
    else:
        nx = xyfile['sci',1].header['NAXIS1']
        ny = xyfile['sci',1].header['NAXIS2']
        ltv1 = xyfile['sci',1].header['ltv1']
        ltv2 = xyfile['sci',1].header['ltv2']

    grid = [nx,ny,1]
    print 'grid of : ',nx,ny
    xyfile.close()

    xarr,yarr = build_grid_arrays(nx,ny,1)
    xgarr = xarr.reshape(grid[1],grid[0])
    ygarr = yarr.reshape(grid[1],grid[0])

    # initialize plot here
    pl.clf()
    pl.gray()
    
    for chip,det in zip(range(1,numchips+1),ccdchip):

        xout,yout = transform_d2im_dgeo(scifile,chip,xarr,yarr)
        
        dgeochip = 1
        dgeo = pyfits.open(fulldgeofile)
        for e in dgeo:
            if not e.header.has_key('ccdchip'):
                continue
            else:
                if e.header['ccdchip'] == det:
                    dgeochip = e.header['extver']
                    break
        dgeo.close()
        
        print 'Matching sci,',chip,' with DX,',dgeochip
        dx= (xout-xarr).reshape(grid[1],grid[0])
        fulldatax = pyfits.getdata(fulldgeofile,'DX',dgeochip)
        diffx=(dx-fulldatax[-ltv2:-ltv2+ny,-ltv1:-ltv1+nx]).astype(np.float32)
        
        pl.imshow(diffx,vmin=vmin,vmax=vmax)
        pl.title('dx-full_x: %s %s(DX,%d) with %g +/- %g'%(filter_names,detector,dgeochip,diffx.mean(),diffx.std()))
        pl.colorbar()
        
        raw_input("Press 'ENTER' to close figure and plot DY...")
        pl.close()

        dy= (yout-yarr).reshape(grid[1],grid[0])
        fulldatay = pyfits.getdata(fulldgeofile,'DY',dgeochip)
        diffy=(dy-fulldatay[-ltv2:-ltv2+ny,-ltv1:-ltv1+nx]).astype(np.float32)

        pl.imshow(diffy,vmin=vmin,vmax=vmax)
        pl.title('dy-full_y: %s %s(DY,%d) with %g +/- %g '%(filter_names,detector,dgeochip,diffy.mean(),diffy.std()))
        pl.colorbar()

        raw_input("Press 'ENTER' to close figure and show next chip...")
        pl.close()
        if output:
            # parse out rootname from input file if user wants results written to file
            outroot = fileutil.buildNewRootname(scifile)
            #
            # setup DGEOFILE ref file as template for each chip's output results
            # we only need dx,1 and dy,1 since each chip will be written out
            # to a separate file and since we will use this template for 
            # writing out 2 different results files
            #
            fhdulist = pyfits.open(fulldgeofile)
            hdulist = pyfits.HDUList()
            hdulist.append(fhdulist[0])
            hdulist.append(fhdulist['dx',1])
            hdulist.append(fhdulist['dy',1])
            fhdulist.close()
            
            outname = outroot+'_sci'+str(chip)+'_dgeo_diffxy.match'
            if os.path.exists(outname): os.remove(outname)
            wtraxyutils.write_xy_file(outname,[xgarr[::32,::32].flatten(),
                                                ygarr[::32,::32].flatten(),
                                                (xgarr+diffx)[::32,::32].flatten(),
                                                (ygarr+diffy)[::32,::32].flatten()],format="%20.8f",append=True)
            
            outname = outroot+'_sci'+str(chip)+'_newfull_dxy.fits'
            if os.path.exists(outname): os.remove(outname)

            hdulist['dx',1].data = dx
            hdulist['dy',1].data = dy
            hdulist.writeto(outname)
            
            outname = outroot+'_sci'+str(chip)+'_diff_dxy.fits'
            if os.path.exists(outname): os.remove(outname)
            hdulist['dx',1].data = diffx
            hdulist['dy',1].data = diffy
            hdulist.writeto(outname)
            print 'Created output file with differences named: ',outname

        del dx,dy,diffx,diffy
            
    if output:
        hdulist.close()
        
def compare_sub_to_full_sci(subarray,full_sci,output=False,update=True):
    if update:
        # update input SCI file to be consistent with reference files in header
        print 'Updating input file ',subarray,' to be consistent with reference files listed in header...'
        updatewcs.updatewcs(subarray)
        print 'Updating input file ',full_sci,' to be consistent with reference files listed in header...'
        updatewcs.updatewcs(full_sci)

    fulldgeofile = fileutil.osfn(pyfits.getval(subarray,'ODGEOFIL'))
    # parse out rootname from input file if user wants results written to file
    if output:
        soutroot = fileutil.buildNewRootname(subarray)
        foutroot = fileutil.buildNewRootname(full_sci)
        hdulist = pyfits.open(fulldgeofile)

    detector = pyfits.getval(fulldgeofile,'DETECTOR')
    filter_names = fileutil.getFilterNames(pyfits.getheader(subarray))

    # count the number of chips in subarray image 
    xyfile = pyfits.open(subarray)
    numchips = 0
    ccdchip = []
    extname = xyfile[1].header['EXTNAME']
    for extn in xyfile:
        if extn.header.has_key('extname') and extn.header['extname'] == extname:
            numchips += 1
            if extn.header.has_key('ccdchip'):
                ccdchip.append([extn.header['ccdchip'],extn.header['extver']])
            else:
                ccdchip.append([1,1])

    snx = xyfile['sci',1].header['NAXIS1']
    sny = xyfile['sci',1].header['NAXIS2']
    ltv1 = xyfile['sci',1].header['ltv1']
    ltv2 = xyfile['sci',1].header['ltv2']
    xyfile.close()

    # build grid of points for full-size image for 
    #    chips corresponding to subarray 
    xyfile = pyfits.open(full_sci)
    fullchip = []
    for extn in xyfile:
        if (extn.header.has_key('extname') and extn.header['extname'] == extname) and \
        extn.header['ccdchip'] == ccdchip[0][0]:
            fullchip.append([extn.header['ccdchip'],extn.header['extver']])
    xyfile.close()

    sxarr,syarr = build_grid_arrays(snx,sny,1)
    full_range = [slice(-ltv2,-ltv2+sny),slice(-ltv1,-ltv1+snx)]


    fnx = pyfits.getval(full_sci,'NAXIS1','sci',1)
    fny = pyfits.getval(full_sci,'NAXIS2','sci',1)
    fxarr,fyarr = build_grid_arrays(fnx,fny,1)
    
    # initialize plot here
    pl.clf()
    pl.gray()
    
    for chip,det,fext in zip(range(1,numchips+1),ccdchip,fullchip):
        # Compute the correction imposed by the D2IM+DGEO corrections 
        #   on the subarray
        sxout,syout = transform_d2im_dgeo(subarray,det[1],sxarr,syarr)
        sdx= (sxout-sxarr).reshape(sny,snx)
        sdy= (syout-syarr).reshape(sny,snx)
        # Compute the correction imposed by the D2IM+DGEO corrections 
        #    on the full sized SCI image
        fxout,fyout = transform_d2im_dgeo(full_sci,fext[1],fxarr,fyarr)
        fdx= (fxout-fxarr).reshape(fny,fnx)
        fdy= (fyout-fyarr).reshape(fny,fnx)

        # determine the difference
        diffx = (sdx - fdx[full_range[0],full_range[1]]).astype(np.float32)
        pl.imshow(diffx)
        pl.title('sub_dx-full_x: %s %s[%d:%d,%d:%d] with %g +/- %g'%
            (filter_names,detector,full_range[0].start,full_range[0].stop,
             full_range[1].start,full_range[1].stop,
             diffx.mean(),diffx.std()))
        pl.colorbar()
        
        raw_input("Press 'ENTER' to close figure and plot DY...")
        pl.close()
        
        # determine the difference
        diffy = (sdy - fdy[full_range[0],full_range[1]]).astype(np.float32)
        pl.imshow(diffy)
        pl.title('sub_dy-full_y: %s %s[%d:%d,%d:%d] with %g +/- %g'%
            (filter_names,detector,full_range[0].start,full_range[0].stop,
             full_range[1].start,full_range[1].stop,
             diffy.mean(),diffy.std()))
        pl.colorbar()
        
        raw_input("Press 'ENTER' to close figure and exit...")
        pl.close()
        
        if output:
            outname = foutroot+'_sci'+str(chip)+'_newfull_dxy.fits'
            if os.path.exists(outname): os.remove(outname)
            hdulist['dx',chip].data = fdx
            hdulist['dy',chip].data = fdy
            hdulist.writeto(outname)
            outname = soutroot+'_sci'+str(chip)+'_newsub_dxy.fits'
            if os.path.exists(outname): os.remove(outname)
            hdulist['dx',chip].data = sdx
            hdulist['dy',chip].data = sdy
            hdulist.writeto(outname)

            """
            outname = outroot+'_sci'+str(chip)+'_diff_dxy.fits'
            if os.path.exists(outname): os.remove(outname)
            hdulist['dx',chip].data = diffx
            hdulist['dy',chip].data = diffy
            hdulist.writeto(outname)
            """
            print 'Created output file with differences named: ',outname
    if output:
        hdulist.close()

def dgeo_function1(x,y):
    """ returns the dgeofile array value for pixel(s) x,y 
        for a simple basic function.
        This will be used to generate a full-sized artificial DGEOFILE that can be
        used to compare with an NPOLFILE generated from this test file.
    """
    return 0.01 + 0.005*(x/4096.0) + 0.005*(y/2048.0)

def dgeo_function2(x,y):
    """ returns the dgeofile array value for pixel(s) x,y 
        for a simple basic function.
        This will be used to generate a full-sized artificial DGEOFILE that can be
        used to compare with an NPOLFILE generated from this test file.
    """
    return -0.01 - 0.005*(x/4096.0) + 0.01*(y/2048.0)

def create_testdgeo(template,output,xfunc=dgeo_function1,yfunc=dgeo_function2):
    """ Generate an artificial DGEOFILE with very simple functions that will
        allow for a simpler comparison with NPOLFILE.
    """
    # start by creating x and y arrays
    xarr = np.fromfunction(xfunc,[2048,4096]).astype(np.float32)
    yarr = np.fromfunction(yfunc,[2048,4096]).astype(np.float32)
    # Open template
    f = pyfits.open(template,mode='readonly')
    # create new file in memory
    o = pyfits.HDUList()
    for e in f:
        o.append(e)
    f.close()
    # update arrays with generated arrays
    for chip in [1,2]:
        o['dx',chip].data = xarr
        o['dy',chip].data = yarr
    # write out new file
    if os.path.exists(output):os.remove(output)
    o.writeto(output)
    o.close()
    del xarr,yarr



    

    