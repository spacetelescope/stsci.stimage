import numpy as N
import ndimage as ND

import imagestats
from fileutil import DEGTORAD
from pydrizzle.drutil import buildRotMatrix
import expandArray

def canny_edge(image, alpha=0.1,thin=False):
    """ Canny edge detector algorithm implementation.  """
    nx1 = 10
    ny1 = 10
    sigma_x1 = 1.0
    sigma_y1 = 1.0
    # X axis direction edge detection
    x_filter = gauss_edge_kernel(nx1,ny1,sigma_x1,sigma_y1,90.0)
    Ix = ND.convolve(image,x_filter)

    nx2 = 10
    ny2 = 10
    sigma_x2 = 1.0
    sigma_y2 = 1.0
    # Y axis direction edge detection
    y_filter = gauss_edge_kernel(nx2,ny2,sigma_x2,sigma_y2,0.0)
    Iy = ND.convolve(image,y_filter)

    # Compute the norm of the gradient 
    #  by combining the X and Y directional derivatives
    Norm_grad = N.sqrt(Ix*Ix + Iy*Iy)

    # Threshholding
    I_max = Norm_grad.max()
    I_min = Norm_grad.min()
    #level = (alpha * (I_max - I_min)) + I_min
    #Thresh_grad = N.maximum(Norm_grad, level+N.zeros(Norm_grad.shape))

    lstats = imagestats.ImageStats(Norm_grad,nclip=3)
    level = lstats.mean + 3*lstats.stddev
    Thresh_grad = N.where(Norm_grad >= level, Norm_grad, level)
    
    if thin:
        # Thinning 
        # Use interpolation to find the pixels where the norms of the
        # gradient are local maximum.
        ny,nx = Norm_grad.shape
        Xk = N.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        Yk = N.array([[-1,-1,-1],[0,0,0],[1,1,1]])
        I_temp = N.zeros(Norm_grad.shape,dtype=N.float32)
        for j in range(2,ny-1):
            for i in range(2,nx-1):
                if Norm_grad[j,i] > level:
                    Zk = Norm_grad[j-1:j+1,i-1:i+1]
                    XI = [Ix[j,i]/Norm_grad[j,i], -Ix[j,i]/Norm_grad[j,i]]
                    YI = [Iy[j,i]/Norm_grad[j,i], -Iy[j,i]/Norm_grad[j,i]]
                    ZI = interp2(Xk,Yk,Zk,XI,YI)
                    if Thresh_grad[j,i] >= ZI[0] and Thresh_grad[j,i] >= ZI[1]:
                        I_temp[j,i] = I_max
                    else:
                        I_temp[j,i] = I_min
                else:
                    I_temp[j,i] = I_min
    else:   
        I_temp = None    
    return I_temp,Norm_grad

def gauss_edge_kernel(nx,ny,sigma_x,sigma_y,theta):
    """ Computes the 2D edge detector (first order derivative
        of 2D Gaussian function) with size n1*n2.  Theta is the 
        angle the detector rotated counter clockwise, while
        sigma_x and sigma_y are the stddev of the Gaussian functions.
    """
  
    r = buildRotMatrix(theta)
    h = N.zeros((ny,nx),dtype=N.float32)
    for y in xrange(ny):
        for x in xrange(nx):
            pos = N.dot(r,[x-((nx)/2),y-((ny)/2)])
            h[y,x] = gauss(pos[0],sigma_x) * dgauss(pos[1],sigma_y)

    h = h / N.sqrt((N.abs(h)*N.abs(h)).sum())

    return h

def gauss_kernel(nx,ny=None,sigma_x=1.0,sigma_y=None):
    """ Computes the 2D Gaussian with size n1*n2.  
        Sigma_x and sigma_y are the stddev of the Gaussian functions.
        The kernel will be normalized to a sum of 1.
    """
    if ny == None: ny = nx
    if sigma_y == None: sigma_y = sigma_x
    
    h = N.zeros((ny,nx),dtype=N.float32)
    for y in xrange(ny):
        for x in xrange(nx):
            pos = [x-((nx)/2),y-((ny)/2)]
            h[y,x] = gauss(pos[0],sigma_x) * gauss(pos[1],sigma_y)

    h = h / N.abs(h).sum()

    return h
  
def gauss(x,sigma):
    """ Compute gaussian."""
    return N.exp(-N.power(x,2)/(2*N.power(sigma,2))) / (sigma*N.sqrt(2*N.pi))

def dgauss(x,sigma):
    """ Compute first order derivative of gaussian function."""
    return -x * gauss(x,sigma) / N.power(sigma,2)

def LoG_mask(sigma=1.0,cutoff=3.9,peak=1.0):
    """ Returns a Laplacian-of-Gaussian kernel [Ref. 1] with
        a mean of 0 and a peak specified by the user.
        If no width is provided by the user, the size
        of the kernel will be automatically determined 
        based on C_log/sigma = 3.9 to include 99.83% of
        the energy.
        References: 
        (1)Chen, et al, IEEE PAMI, PAMI-9, No. 4, July 1987, pg 584-590.
        
    """
    from math import ceil
    
    C_log = cutoff * sigma
    log_width = int(ceil(C_log))
    if (log_width%2) == 0: log_width += 1

    cen = int(log_width/2) + 1
    
    lk = N.zeros((log_width,log_width),dtype=N.float32)
    for y in xrange(log_width):
        for x in xrange(log_width):
            xc = (x+1) - cen
            yc = (y+1) - cen 
            r = (xc*xc + yc*yc)
            lk[y,x] = (2- (pow(r,2)/pow(sigma,2)))*N.exp(-pow(r,2)/(2*pow(sigma,2))) 
  
    
    # Scale kernel to mean of 0 and all integer values
    N.subtract(lk,(lk.sum()/(lk.shape[0]*lk.shape[1])),lk)

    return lk*(peak/lk.max())
    
    

def interp2(xk,yk,zk,xint,yint):
    """ Compute the 2D interpolated value. """

    xk = N.array(xk)
    yk = N.array(yk)
    # Insure that input Z values are in array form    
    z_in = N.array(zk)
    
    # Start by interpolating X and Y value(s)
    t = N.array(interp_bilinear1d(xk,xint)).ravel()
    u = N.array(interp_bilinear1d(yk,yint)).ravel()

    # Find where the input X and Y values fall
    # in the X/Y arrays such that:
    #     xk[i-1] < xint[0] < xk[i]
    #
    x_ind = find_index(xk,xint)  
    y_ind = find_index(yk,yint)

    # Create output arrays
    z_out = N.zeros(t.shape,dtype=z_in.dtype)
    if z_out.size == 1: 
        z_out = N.reshape(z_out,(1,))

    # For these positions, compute the appropriate Z value
    for i in xrange(len(t)):
        j = x_ind[i]
        k = y_ind[i]

        # Insure that all indices for accessing z_in 
        # are within the bounds of z_in
        max_x_indx = len(z_in[0,:]) - 1
        max_y_indx = len(z_in[:,0]) - 1
        if j > max_x_indx: j = max_x_indx
        if k > max_y_indx: k = max_y_indx
        j1 = j+1
        k1 = k+1
        if j1 > max_x_indx: j1 = max_x_indx
        if k1 > max_y_indx: k1 = max_y_indx        

        # Compute bilinear interpolation value here...
        z_out[i] = (1-t[i])* (1-u[i]) * z_in[k,j] + \
                      t[i] * (1-u[i]) * z_in[k,j1] + \
                      t[i] *    u[i]  * z_in[k1,j1] + \
                   (1-t[i])*    u[i]  * z_in[k1,j]

    # If only 1 input, output only 1 value
    if z_out.size == 1:
        return z_out[0]
    else:
        return z_out
        
    
def interp_bilinear1d(x,xi):
    """ Compute the bilinear interpolated value(s) of xi within x."""
    x = N.array(x).ravel()
    x_in = N.array(xi,dtype=N.float32).ravel()

    x_out = N.zeros(x_in.shape,dtype=N.float32)
    x_ind = find_index(x,xi)

    max_indx = x.size - 1
    
    # For each input value
    for i in xrange(x_in.size):
        j = x_ind[i]
        
        # Account for lower boundary condition
        if j < 0:
            j = 0
            j1 = 0
        else:
            j1 = j+1                            
        if j1 > max_indx: j1 = max_indx

        diff_x = x[j1] - x[j]
        if diff_x != 0.:            
            # Compute interpolated value                
            x_out[i] = (x_in[i] - x[j])/diff_x
        else:
            x_out[i] = 1.0

    # If a single scalar was given as input, 
    # it should return a single scalar as output
    if x_in.size == 1:
        return x_out[0]
    else:
        return x_out

def find_index(array,values):
    """ Return the index into array that corresponds to each value. """
    val = N.array(values).ravel()

    arr = N.ravel(array)
    sout = N.array(N.searchsorted(arr,val)).ravel()
    max_indx = array.size - 1

    for i in xrange(len(sout)):
        if sout[i] > max_indx: sout[i] = max_indx
        
        int_val = arr[sout[i]] - val[i]
        
        if int_val > 0 and sout[i] > 0:
            sout[i] -= 1
        elif sout[i] == 0 and int_val > 0:
            # Flag lower out of bounds condition
            sout[i] = -1

    
    return sout

def signum(array):
    """ Return the signum values for the input array,
        where the signum has the standard definition:
            signum(x = array(i,j)) = -1 for x < 0

                                      0 for x == 0
                                      1 for x > 0
    """
    signum = (array > 0).astype(N.int8)
    N.subtract(array < 0,signum,signum)
    
    return signum    
    
    
def find_edge_points(array):
    """ Detects the edge points from a LoG image
        using the criteria:
        - for each zero-crossing pixel, it is an
        edge point iff
        LoG(i,j) <= max[N(i,i)*signum(-LoG(i,j))]
        
        where N(i,j) is the 8-neighborhood values for i,j and
              Log(i,j) is the LoG value of the pixel i,j
        Ref: Dai & Khorram (1999), IEEE Trans. GeoSci. and
             Remote Sensing, Vol 37, No 5, 2351-2362.
    """
    # Compute threshold level for zero-crossing
    stats = imagestats.ImageStats(array)
    # Start by extracting zero-crossing pixels
    mask = array > stats.mean+stats.stddev
    N.subtract(ND.binary_dilation(mask),mask,mask)
    
    # Compute the edge-point criteria to each contour
    maxarr = compute_max_neighbor(array)
    
    # Select zero-crossing pixels which meet the criteria
    N.putmask(mask,(array <= maxarr),mask)
    del maxarr
    
    return mask

def compute_max_neighbor(array):
    """ Compute max[N(i,j)*signum(-1*array). """
    # Compute the edge-point criteria to each contour
    footp = N.array([[1,1,1],[1,0,1],[1,1,1]],N.UInt8)
    maxarr = ND.maximum_filter(array,footprint=footp)
    del footp
    marray = maxarr * signum(-1*array)
    del maxarr
    
    return marray


def compute_edge_strength(array,zero_cross):
    """ Compute the edge strength map for a LoG of an image
        by considering the slopes along both the x and y 
        directions.
        based on Hui Li et al., A contour based approach to
        Multisensor Image Registration
    """
    xkernel = N.array([[0,0,0],[1,0,-1],[0,0,0]])
    ykernel = N.array([[0,1,0],[0,0,0],[0,-1,0]])
    k_x = ND.convolve(array,xkernel)
    k_y = ND.convolve(array,ykernel)
    k_x2 = N.power(k_x,2)
    del k_x
    k_y2 = N.power(k_y,2)
    del k_y
    
    es = N.sqrt(k_x2 + k_y2)
    del k_x2,k_y2
    
    return N.where(zero_cross, es, 0)
    
def compute_LoG_image(image, k_d, k_sigma, gsigma, gauss_sigma):

    """

    Computes Laplassian of Gaussian of an image by
    
    - run a gaussian filter on a decimated image with a decimation factor k_d
    - create an LoG mask
    - convolve the LoG mask with the filtered image
    - expand the image to the original size using bilinear interpolation
    
    """
    logk = LoG_mask( (gauss_sigma/(k_d*k_sigma)) )

    gchip = ND.gaussian_filter(image[::k_d,::k_d],gsigma)
    lgchip = ND.convolve(gchip,logk)
    del gchip
    logchip = expandArray.expandArrayF32(lgchip,k_d,1,0,0)
    del lgchip
    lgchip = logchip[:image.shape[0],:image.shape[1]].copy()
    del logchip
    
    return lgchip
    
def find_LoG_zeros(image,esigma=3, clean=True, name=None):
    #
    # Edge detection using fast Lagrangian-of-Gaussian
    #    as described in Chen et al(1987).
    #
    k_d = 3
    k_sigma = 1.25
    gsigma = 2.4
    gauss_sigma = 4.0
    if not clean:
        import pyfits as p
    
    lgchip = compute_LoG_image(image, k_d, k_sigma, gsigma, gauss_sigma)
    
    if not clean:

        phdu=p.PrimaryHDU(lgchip.astype(N.uint8))
        phdu.writeto(name+'LoG.fits', clobber=True)
        
    #
    # Remove edge-effects from derivative associated with chip edges
    #
    echip = ND.binary_dilation(image==0,iterations=(int(gauss_sigma*k_d*2)))
    gclip = N.where(echip == 0, lgchip, 0)
    
    if not clean:
        phdu = p.PrimaryHDU(echip.astype(N.uint8))
        phdu.writeto(name+'BinDil.fits', clobber=True)
        phdu=p.PrimaryHDU(gclip)
        phdu.writeto(name+'BinDilClipped.fits', clobber=True)
    
    #del echip

    #
    # Find zero-crossings: actually, find the contours corresponding
    #    to mean+stddev level (for the minimum).
    #
    if (gclip.min() > 0):
        gstats = imagestats.ImageStats(gclip,fields='mean,min,max,stddev',nclip=1,usig=5.0,lsig=5.0)
        min_clip = gstats.mean + 3*gstats.stddev
    else: 
        min_clip = 0.
    
    eclip = gclip > min_clip
    del gclip
    
    lgedge = eclip - ND.binary_erosion(eclip)
    
    if not clean:
	phdu = p.PrimaryHDU(eclip.astype(N.uint8))
        phdu.writeto(name+'eclip.fits', clobber=True)
	phdu = p.PrimaryHDU(ND.binary_erosion(eclip).astype(N.uint8))
        phdu.writeto(name+'BinEro.fits', clobber=True)
        phdu = p.PrimaryHDU(lgedge.astype(N.uint8))
        phdu.writeto(name+'edge.fits', clobber=True)
    del eclip

    es = compute_edge_strength(lgchip,lgedge)
    del lgchip
    #
    # Extract edges for zero-crossings
    #
    eslabel,esnum = ND.label(es,structure=ND.generate_binary_structure(2,2))
    #estats = imagestats.ImageStats(es)
    estats = imagestats.ImageStats(es,nclip=1,lsig=5.0,usig=5.0)

    #
    # Implement 2-level threshold for contours 
    #     min level set by edge_min
    #     max level set by edge_max
    # This insures that 'good' contours represent the
    #    strongest edges, not those with weak spots in them. 
    #
    esl_slices = ND.find_objects(eslabel) 
    #edge_max = estats.mean + esigma*estats.stddev
    edge_max = estats.max
    edge_min = estats.mean

    bad_edges = range(esnum)
    for e in bad_edges[-1::-1]:
        extrema = ND.extrema(es[esl_slices[e]],eslabel[esl_slices[e]],e+1)
        if ((extrema[1] >= edge_max) and (extrema[0] >= edge_min)):
            bad_edges.remove(e)
    del es,esnum,estats
    
    #print 'Number of good contours: ',esnum - len(bad_edges)
    # At this point, we have removed the index value for all contours
    # with a maximum strength >= to the limit and min strength >= edge_min,
    # leaving only the weakest contours in the list 'bad_edges'.
    # We use this list to then zero out the contours with indices in the 
    # 'bad_edges' list.
    for e in bad_edges:
        N.subtract(lgedge[esl_slices[e]],eslabel[esl_slices[e]] == e+1, lgedge[esl_slices[e]])
    ##
    # At this point, we have the contours of the zero-crossing 
    #    edges.  
    ##

    #
    # Remove all sources connected to the edge
    #
    a1struct = ND.generate_binary_structure(2,1)
    a1edge = ND.binary_dilation(N.zeros(lgedge.shape),a1struct,-1,1-(echip+lgedge),border_value=1)
    a1mask = ND.binary_dilation(N.zeros(lgedge.shape),a1struct,-1,1-a1edge,border_value=1)
    a1trim = ND.binary_dilation(1-a1mask,a1struct,1)
    N.subtract(a1trim, (1-a1mask),a1trim)
    N.subtract(lgedge, a1trim, lgedge)
    lgedge = N.where(lgedge < 0, 0, lgedge)
    del a1struct, a1edge, a1mask, a1trim, echip

    return lgedge
