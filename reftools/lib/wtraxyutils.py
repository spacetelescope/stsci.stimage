from __future__ import division # confidence high

import sys,os
import numpy as np

import pyfits
import pylab as pl
import imagestats 
from pytools import fileutil
import datetime


""" 
Code for interpreting ASCII files
"""

def clean_separators(line,sep,niter=10):
    for i in range(niter):
        lnew = line.replace(sep*2,sep)
        if lnew == line: break
        else: line = lnew
    return lnew


def write_xycols(infile,outfile,cols=[1,2,4,5],scale=None,center=[2048,1024],wxin=True):
  
    fin = open(infile,'r')
    fout = open(outfile,'a+')
    nlines = 0
    for l in fin.readlines():
        if l.find("Xin,Yin") > -1 or not wxin:
            lspl = clean_separators(l.strip()," ").split(" ")
            xo = float(lspl[cols[2]])
            yo = float(lspl[cols[3]])
            if scale is not None:
                xo = (xo - center[0])/scale + center[0]
                yo = (yo - center[1])/scale + center[1]
                
            xi = float(lspl[cols[0]])
            yi = float(lspl[cols[1]])
        
            fout.write(" %10.5f %10.5f    %10.5f %10.5f\n"%(xi,yi,xo,yo))
            #fout.write(" %10.5f %10.5f    %10.5f %10.5f\n"%(float(lspl[cols[0]]),float(lspl[cols[1]]),float(lspl[cols[2]]),float(lspl[cols[3]])))
            nlines += 1
  
    fin.close()
    fout.close()
    print '[wtraxyutils.write_xycols] wrote out ',nlines,' to ',outfile

def DEGTORAD(deg):
    return (deg * np.pi / 180.)

def RADTODEG(rad):
    return (rad * 180. / np.pi)

def buildRotMatrix(theta):
    _theta = DEGTORAD(theta)
    _mrot = np.zeros(shape=(2,2),dtype=np.float64)
    _mrot[0] = (np.cos(_theta),np.sin(_theta))
    _mrot[1] = (-np.sin(_theta),np.cos(_theta))

    return _mrot

def write_xy_file(outname,xydata,append=False,format="%20.6f"):
    if not append:
        if os.path.exists(outname): os.remove(outname)
    fout1 = open(outname,'a+')
    for row in range(len(xydata[0])):
        outstr = ""
        for col in range(len(xydata)):
            outstr += format%(xydata[col][row])
        fout1.write(outstr+"\n")
    fout1.close()
    print 'wrote XY data to: ',outname  
 
def apply_poly(cx,cy,order,pos=[[0.,0.],[0.,1.],[1.,0.],[1.,1.]]):
    ''' Apply a polynomial to a unit box.
    '''
    pos = np.array(pos,np.float32)
    c = pos * 0.0
    pos[:,0] -= (pos[:,0].max() - pos[:,0].min())/2.
    pos[:,1] -= (pos[:,1].max() - pos[:,1].min())/2.
    
    for i in range(order+1):
        for j in range(i+1):
            c[:,0] = c[:,0] + cx[i][j] * pow(pos[:,0],j) * pow(pos[:,1],(i-j))
            c[:,1] = c[:,1] + cy[i][j] * pow(pos[:,0],j) * pow(pos[:,1],(i-j))
    
    return c

def rot_poly(cx,cy,rot):
    ''' Rotate poly coeffs by 'rot' degrees.
    '''
    mrot = buildRotMatrix(rot)
    rcxy = np.dot(mrot,[cx.ravel(),cy.ravel()])
    rcx = rcxy[0]
    rcy = rcxy[1]
    rcx.shape = cx.shape
    rcy.shape = cy.shape
    
    return rcx,rcy

def apply_tdd_coeffs(x,y,alpha,beta,theta,xoff=None,yoff=None,recenter=False,direct_tdd=True):
    
    if direct_tdd:
        xoff = 2048.0
        yoff = 2048.0
        print '[apply_tdd_coeffs] alpha,beta: ',alpha,beta
        fx = (x - xoff) / xoff
        fy = (y - yoff) / yoff
        xd = x + beta*fx + alpha*fy
        yd = y - beta*fy + alpha*fx
        print '[apply_tdd_coeffs] new values: ',xd[0],yd[0],yd[-1],xd[-1]
    else:
        alpha /= 2048.0
        beta /= 2048.0
        if recenter:
            x -= xoff
            y -= yoff
        xd = (1+(alpha*np.sin(2*theta))+(beta*np.cos(2*theta)))*x + (alpha*np.cos(2*theta) - beta*np.sin(2*theta))*y
        yd = (alpha*np.cos(2*theta) - beta*np.sin(2*theta))*x + (1+(alpha*np.sin(2*theta)) - (beta*np.cos(2*theta)))*y
        if recenter:
            xd += xoff
            yd += yoff

    return xd,yd
def apply_dq_values(infile,outfile,dqcol=3,cols=[0,1],dqlimit=-10.0,corr=[0.,0.]):
    ''' raw xy -> cleaned xy[dqlimit]
    
        Apply a dq mask based on what data was retained
        during TDD correction by Vera, where dq > 0.
    '''
    fin = open(infile,'r')
    fout = open(outfile,'a+')
    for l in fin.readlines():
        lspl = clean_separators(l.strip()," ").split(" ")
        valid = (lspl[dqcol].find('*') < 0) and (float(lspl[dqcol]) > dqlimit)
        if valid:
            fout.write(" %10.5f %10.5f \n"%(float(lspl[cols[0]])+corr[0],float(lspl[cols[1]])+corr[1]))
    fin.close()
    fout.close()
        
    
def readcols(infile,cols=[0,1,2,3]):

    fin = open(infile,'r')
    outarr = []
    for l in fin.readlines():
        l = l.strip()
        if len(l) == 0 or len(l.split()) < len(cols) or (len(l) > 0 and l[0] == '#' or (l.find("INDEF") > -1)): continue
        for i in range(10):
            lnew = l.replace("  "," ")
            if lnew == l: break
            else: l = lnew
        lspl = lnew.split(" ")

        if len(outarr) == 0:
            for c in range(len(cols)): outarr.append([])

        for c,n in zip(cols,range(len(cols))):
            outarr[n].append(float(lspl[c]))
    fin.close()
    for n in range(len(cols)):
        outarr[n] = np.array(outarr[n],np.float64)
    return outarr            
   
def parse_phot_line(line):
    """ Converts line with photometry into a list of values without blanks
    """
    lspl = line.strip().split(' ')
    # remove all blanks from list
    nblanks = lspl.count('')
    while nblanks > 0:
        lspl.remove('')
        nblanks = lspl.count('')
    return lspl
def read_phot(photfile,aperture=None,col=1):
    """ Read in the photometry results as recorded in a photometry file written
    out by daophot.phot.  If None, it will pick the first non-1 aperture listed
    for each star. 
    """            
    if os.path.exists(photfile):
        pfile = open(photfile,'r')
        photlines = pfile.readlines()
        pfile.close()
        
        out = []
        filename = None
        indx = 0
        for line in photlines:
            # start by ignoring comment lines
            if line[0] == '#':
                continue
            if filename is None:
                filename = line[:20]
            if line[:20] == filename:
                indx = 0
                continue
            lspl = parse_phot_line(line)
            if indx == 0:
                if lspl[-2] in ['NoError','BigShift']: err = 1
                else: err = 0
                
                coords = [float(lspl[0]),float(lspl[1]),err]
            # skip the next 2 lines
            if indx < 3: 
                indx += 1
                continue
            # Ignore radius 1 photometry
            if float(lspl[0]) == 1.0:
                continue

            if aperture is not None:
                if float(lspl[0]) != aperture:
                    continue
                else:                    
                    coords.append(float(lspl[col]))
                    out.append(coords)
            else:
                aperture = float(lspl[0])
                coords.append(float(lspl[col]))
                out.append(coords)

    return np.array(out,dtype=np.float32)

def get_db_fit(dbfile,fit=None):
    """ Get the matrix solution found by geomap as stored in a database file.
        The 'fit' parameter(1-based, not zero based) specifies which fit from 
        a file to read, with the default being the last one.
    """
    if os.path.exists(dbfile):
        db = open(dbfile,'r')
        dblines = db.readlines()
        db.close()
        
        # Record the location of all fits in db file
        fitnum = []
        for line,nline in zip(dblines,range(len(dblines))):
            if line.find('begin') > -1: 
                fitnum.append(nline)
        fitnum.append(len(dblines)+1)

        # Find which line corresponds to the desired fit
        if fit is None:
            nfit = fitnum[-1]
        else:
            nfit = fitnum[fit]

        # Now, read in appropriate values
        out = []
        for line in dblines[nfit-5:nfit-2]:
            vals = clean_separators(line.strip(),'\t').split('\t')
            out.append(float(vals[0]))
            out.append(float(vals[-1]))
        
    return out[0],out[1],np.array([[out[2],out[3]],[out[4],out[5]]])

def apply_db_fit(data,fit,xsh=0.0,ysh=0.0):
    xy1x = data[0]
    xy1y = data[1]
    numpts = xy1x.shape[0]
    if fit is not None:
        xy1 = np.zeros((xy1x.shape[0],2),np.float64)
        xy1[:,0] = xy1x 
        xy1[:,1] = xy1y 
        xy1 = np.dot(xy1,fit)
        xy1x = xy1[:,0] + xsh
        xy1y = xy1[:,1] + ysh
    return xy1x,xy1y

    
def read_center_file(infile):
    x = []
    y = []
    e = []
    f = open(infile)
    for l in f.readlines():
        if l[0] != 'j' and l[0] != '#':
            lspl = clean_separators(l.strip()," ").split(" ")
            x.append(float(lspl[0]))
            y.append(float(lspl[1]))
            if lspl[-1] == 'NoError':
                ev= 1
            else:
                ev=0
            e.append(ev)
    f.close()
    return x,y,e


def transform_dgeo_for_sip(fltimage,update=True):
    """ Applies inverse transform of linear distortion to DGEOFILE for use with
        WDRIZZLE in SIP mode. This is only intended to be used for testing the
        validity of this form of the DGEOFILE for use with SIP headers.
    """
    
    flt_phdr = pyfits.getheader(fltimage)
    # Start by parsing the input filename and creating the output name
    if flt_phdr.has_key('ODGEOFIL'):
        dxyfname = fileutil.osfn(flt_phdr['ODGEOFIL'])
    else:
        dxyfname = fileutil.osfn(flt_phdr['DGEOFILE'])
    print 'Starting with original DGEOFILE of: ',dxyfname
    
    dxypath,dxyname = os.path.split(dxyfname)
    dxyoutname = dxyname.replace('_dxy.fits','_sipdxy.fits')
    if dxyfname.find('sipdxy.fits') > -1:
        if os.path.exists(dxyoutname):
            return
        else:
            update=False
    # Read in the input file
    dxyin = pyfits.open(dxyfname)
    numchips = 0
    for extn in dxyin:
        if extn.header.has_key('extname') and extn.header['extname'] == 'DX':
            numchips += 1

    ## Updated values from makewcs 1.1.1
    ## chip 1
    ##-WCSLIN X/YC,X/YS: [ .9966187132,1.0055060926, .0298351033,-.0392596721]
    ## chip 2
    ##-WCSLIN X/YC,X/YS: [ .9844883922, .9716717277, .0426647491,-.0453650945]
    #These values can be derived using:
    # np.dot(np.linalg.inv(R.cd),cd)
    # where R.cd is the reference chip CD matrix without distortion 
    #               as computed in makewcs
    # and cd is the final CD matrix for each chip.
    # These values are not even used anymore, but they provide documentation of
    # the typical values for an ACS/WFC image
    #if cdmat is None:
    #    cdmat = {1:np.array([[.9966187132,.0392596721],[.0298351033,1.0055060926]]), 
    #         2:np.array([[.9844883922,.0453650945],[.0426647491,.9716717277]])
    #        }

    for chip in range(1,numchips+1):
        # Read in the DX and DY arrays
        dx = dxyin['dx',chip].data
        dy = dxyin['dy',chip].data
        hdr = pyfits.getheader(fltimage,extname='sci',extver=chip)
        cdmat_inv = np.linalg.inv(np.array([[hdr['ocx11'],hdr['ocx10']],
                                            [hdr['ocy11'],hdr['ocy10']]])/hdr['idcscale'])
        del hdr
        #cdmat_inv = np.linalg.inv(cdmat[chip])

        # apply inverse matrix to dx/dy images
        cdx,cdy = np.dot(cdmat_inv,[dx.ravel(),dy.ravel()])
        cdx.shape = dx.shape
        cdy.shape = dy.shape
        
        print 'Transforming chip #',chip
        # Copy in the new values to the FITS structure
        dxyin['dx',chip].data = cdx
        dxyin['dy',chip].data = cdy
        
    if os.path.exists(dxyoutname): os.remove(dxyoutname)
    print 'Writing out transformed DGEOFILE as: ',dxyoutname
    dxyin.writeto(dxyoutname)
    
    # if specified by user, update input image with new DGEOFILE 
    if update:
        flt = pyfits.open(fltimage,mode='update')
        # store original DGEOFILE in a new keyword 
        if (flt[0].header.has_key('ODGEOFIL') and flt[0].header['ODGEOFIL'] in ['N/A',""]) or not flt[0].header.has_key('ODGEOFIL'):
            flt[0].header.update('ODGEOFIL',flt[0].header['DGEOFILE'])
            flt[0].header['DGEOFILE'] = dxyoutname
        flt.flush()
        flt.close()
        print 'Input file ',fltimage,' updated with new DGEOFILE.'
        print '==> Old DGEOFILE archived in ODGEOFIL keyword'
        
def compare_matched_pos(matchfile,img1,img2, width=10,grid=[4,3],cmap=pl.gray):
    ''' Display image slices from each image around each object for visual
        verification of matches. 
    '''
    # Read in arrays for each image
    arr1 = pyfits.getdata(img1,0)
    arr2 = pyfits.getdata(img2,0)
    # Read in match file
    x1,y1,x2,y2 = readcols(matchfile,cols=[0,1,2,3])
                
    # Initialize colormap
    cmap()
    nplots = grid[0]*2*grid[1]
    indx = 0
    indx2 = 0
    for i in range(len(x1)):
        # compute slices
        cenx1 = int(x1[i]+0.5)
        ceny1 = int(y1[i]+0.5)
        slice1 = arr1[ceny1-width:ceny1+width,cenx1-width:cenx1+width]

        cenx2 = int(x2[i]+0.5)
        ceny2 = int(y2[i]+0.5)
        slice2 = arr2[ceny2-width:ceny2+width,cenx2-width:cenx2+width]
        
        # Display pairs of slices as sub-plots
        if indx == 0:
            pl.clf()
        if indx+1 <= nplots:
            indx += 1
            pl.subplot(grid[0],grid[1]*2,indx)
            pl.imshow(slice1)
            pl.title('Obj %d at %0.2f,%0.2f'%((i+1),x1[i],y1[i]))
            indx += 1
            pl.subplot(grid[0],grid[1]*2,indx)
            pl.imshow(slice2)
            pl.title('Obj %d at %0.2f,%0.2f'%((i+1),x2[i],y2[i]))
        else:
            indx = 0
            raw_input("Enter to continue to next set of images...")

def find_min_match_cols(matchroot,xcen,ycen,matchextn=['match','db'],fit=None,
                        cols1=[0,1],cols2=[2,3],limit=0.5):
    ''' Select out those rows from the match file which fall within a
        specified limit after applying the fit in the geomap database file.
        The coordinates are also shifted back into the image frame by applying
        the CRPIX1,CRPIX2 of the output frame (given as xcen,ycen). 
    '''
    match = matchroot+'.'+matchextn[0]
    dbfile = matchroot+'.'+matchextn[1]
    if not os.path.exists(match):
        print 'No matchfile "%s" found...'%match
        return
    # Read in XY columns from ".match" file generated by 'xyxymatch'
    xycols = readcols(match,cols=cols1+cols2)
    # Read in fit from 'geomap' ".db" file
    xsh,ysh,drot = get_db_fit(dbfile,fit=fit)
    
    # Compute residuals after applying fit
    xy1xf,xy1yf = apply_db_fit(xycols,drot,xsh=xsh,ysh=ysh)
    dx = xycols[2] - xy1xf
    dy = xycols[3] - xy1yf
    resids = np.sqrt(dx**2 + dy**2)
    pl.clf()
    pl.plot(xycols[0],resids,'b.')
    # find those points corresponding to limit
    gr = np.where(resids <= limit)[0]
    if len(gr) == 0:
        print 'No matches found within limit of ',limit,' for ',match
        return
    else:
        print 'Found ',len(gr),' matches with limit of ',limit,' for ',match
    
    # generate output name and write out values found with resid <= limit
    outname = matchroot+'_limit'+str(limit).replace('.','')+'.coo'
    write_xy_file(outname,[xycols[0][gr]+xcen,xycols[1][gr]+ycen,xycols[2][gr]+xcen,xycols[3][gr]+ycen])
    
    
def match_cols(xin,yin,xref,yref,outfile,threshold=1.0):
    f = open(outfile,'a+')
    for x,y in zip(xin,yin):
        for xr,yr in zip(xref,yref):
            r = np.sqrt((x-xr)**2 + (y-yr)**2)
            if r <= threshold:
                f.write("%0.6f  %0.6f  %0.6f  %0.6f\n"%(x,y,xr,yr))
                break
    f.close()
    

def ordered_match_cols(file1,file2,output,matchcol1=[3,4],matchcol2=[3,4],cols1=[0,1],cols2=[0,1]):
    ''' Extract xy positions from file 1 and file2 where the xy positions from matchcol1 in file1 
        is the same as the xy positions from matchcol2 in file2.
    '''
    xcol1,ycol1,mxcol1,mycol1 = readcols(file1,cols=cols1+matchcol1)
    xcol2,ycol2,mxcol2,mycol2 = readcols(file2,cols=cols2+matchcol2)
    len1 = len(xcol1)
    len2 = len(xcol2)
    # keep track of matching entries from each file in a list of indices for each file
    c1indx = []
    c2indx = []
    for c1 in range(len1):
        for c2 in range(len2):
            if ((mxcol1[c1] == mxcol2[c2]) and (mycol1[c1] == mycol2[c2])): 
                c1indx.append(c1)
                c2indx.append(c2)
                break
    write_xy_file(output,[xcol1[c1indx],ycol1[c1indx],xcol2[c2indx],ycol2[c2indx]])
    
            
def decimal_date(dateobs,timeobs=None):
    """ Convert DATE-OBS (and optional TIME-OBS) into a decimal year.
    """    
    year,month,day = dateobs.split('-')
    if timeobs is not None: 
        hr,min,sec = timeobs.split(':')
    else:
        hr,min,sec = 0,0,0
    rdate = datetime.datetime(int(year),int(month),int(day),int(hr),int(min),int(sec))
    dday = (float(rdate.strftime("%j")) + rdate.hour/24.0 + rdate.minute/(60.*24) + rdate.second/(3600*24.))/365.25
    ddate = int(year) + dday
    
    return ddate

def plot_fltndrz_linkfile(rootname,fltdrzcols=[8,9],drzcols=[0,1],limit=1.0,vector=False,scale=5,title=None):
    # Name of file generated by Jay's program fltNdrz.e which contains
    # the positions derived from the FLT image, DRZ image, and transformed FLT positions
    fltdrzname = 'LOG.'+rootname+'.link'
    
    #rawx,rawy = readcols(fltdrzname,cols=[4,5])
    drzx,drzy = readcols(fltdrzname,cols=fltdrzcols)
    drzimgx,drzimgy = readcols(fltdrzname,cols=drzcols)
    xy_wxy_file = rootname+'_xyall_vs_wxy.coo'
    
    make_vector_plot(xy_wxy_file,data=[drzx,drzy,drzimgx,drzimgy],limit=limit,vector=vector,scale=scale,title=title)
    
    
def make_vector_plot(coordfile,columns=[0,1,2,3],data=None,title=None, axes=None, every=1,
                    limit=None, xlower=None, ylower=None, output=None, headl=4,headw=3,
                    xsh=0.0,ysh=0.0,fit=None,vector=True,scale=5,append=False,linfit=False,rms=True):
    """ Convert a XYXYMATCH file into a vector plot. """
    if data is None:
        data = readcols(coordfile,cols=columns)

    xy1x = data[0]
    xy1y = data[1]
    xy2x = data[2]
    xy2y = data[3]
    numpts = xy1x.shape[0]
    if fit is not None:
        xy1x,xy1y = apply_db_fit(data,fit,xsh=xsh,ysh=ysh)
        fitstr = '-Fit applied'
        dx = xy2x - xy1x
        dy = xy2y - xy1y
    else:
        dx = xy2x - xy1x - xsh
        dy = xy2y - xy1y - ysh

    print 'Total # points: ',len(dx)
    if limit is not None:
        indx = (np.sqrt(dx**2 + dy**2) <= limit)
        dx = dx[indx].copy()
        dy = dy[indx].copy()
        xy1x = xy1x[indx].copy()
        xy1y = xy1y[indx].copy()
    if xlower is not None:
        xindx = (np.abs(dx) >= xlower)
        dx = dx[xindx].copy()
        dy = dy[xindx].copy()
        xy1x = xy1x[xindx].copy()
        xy1y = xy1y[xindx].copy()
    print '# of points after clipping: ',len(dx)
    
    if output is not None:
        write_xy_file(output,[xy1x,xy1y,dx,dy])
        
    if not append:
        pl.clf()
    if vector: 
        dxs = imagestats.ImageStats(dx.astype(np.float32))
        dys = imagestats.ImageStats(dy.astype(np.float32))
        minx = xy1x.min()
        maxx = xy1x.max()
        miny = xy1y.min()
        maxy = xy1y.max()
        xrange = maxx - minx
        yrange = maxy - miny

        pl.quiver(xy1x[::every],xy1y[::every],dx[::every],dy[::every],\
                  units='y',headwidth=headw,headlength=headl)
        pl.text(minx+xrange*0.01, miny-yrange*(0.005*scale),'DX: %f to %f +/- %f'%(dxs.min,dxs.max,dxs.stddev))
        pl.text(minx+xrange*0.01, miny-yrange*(0.01*scale),'DY: %f to %f +/- %f'%(dys.min,dys.max,dys.stddev))
        pl.title(r"$Vector\ plot\ of\ %d/%d\ residuals:\ %s$"%(xy1x.shape[0],numpts,title))
    else:
        plot_defs = [[xy1x,dx,"X (pixels)","DX (pixels)"],\
                    [xy1y,dx,"Y (pixels)","DX (pixels)"],\
                    [xy1x,dy,"X (pixels)","DY (pixels)"],\
                    [xy1y,dy,"Y (pixels)","DY (pixels)"]]
        if axes is None:
            # Compute a global set of axis limits for all plots
            minx = xy1x.min()
            maxx = xy1x.max()
            miny = dx.min()
            maxy = dx.max()
            
            if xy1y.min() < minx: minx = xy1y.min()
            if xy1y.max() > maxx: maxx = xy1y.max()
            if dy.min() < miny: miny = dy.min()
            if dy.max() > maxy: maxy = dy.max()
        else:
            minx = axes[0][0]
            maxx = axes[0][1]
            miny = axes[1][0]
            maxy = axes[1][1]
        xrange = maxx - minx
        yrange = maxy - miny 
        
        for pnum,plot in zip(range(1,5),plot_defs):
            ax = pl.subplot(2,2,pnum)
            ax.plot(plot[0],plot[1],'.')
            if title is None:
                ax.set_title("Residuals [%d/%d]: No FIT applied"%(xy1x.shape[0],numpts))
            else:
                # This definition of the title supports math symbols in the title
                ax.set_title(r"$"+title+"$")
            pl.xlabel(plot[2])
            pl.ylabel(plot[3])
            lx=[ int((plot[0].min()-500)/500) * 500,int((plot[0].max()+500)/500) * 500]
            pl.plot([lx[0],lx[1]],[0.0,0.0],'k')
            pl.axis([minx,maxx,miny,maxy])
            if rms:
                pl.text(minx+xrange*0.01, maxy-yrange*(0.01*scale),'RMS(X) = %f, RMS(Y) = %f'%(dx.std(),dy.std()))
            if linfit:
                lxr = int((lx[-1] - lx[0])/100)
                lyr = int((plot[1].max() - plot[1].min())/100)
                A = np.vstack([plot[0],np.ones(len(plot[0]))]).T
                m,c = np.linalg.lstsq(A,plot[1])[0]
                yr = [m*lx[0]+c,lx[-1]*m+c]
                pl.plot([lx[0],lx[-1]],yr,'r')
                pl.text(lx[0]+lxr,plot[1].max()+lyr,"%0.5g*x + %0.5g [%0.5g,%0.5g]"%(m,c,yr[0],yr[1]),color='r')
        
