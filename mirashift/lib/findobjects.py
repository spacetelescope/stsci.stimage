import numpy as N
import ndimage as ND

import num_pymorph as NP

import imagestats
from math import ceil
import numdisplay
import morph

def DEGTORAD(deg):
    return (deg * N.pi / 180.)

def RADTODEG(rad):
    return (rad * 180. / N.pi)
    
def get_positions(input,sigma=2.0,size=None,offset=None,region=None,thin=False):
    """ Process the input array to return the list of positions that correspond
        to all objects.
        
        This function relies on Numarray nd_image module for its operations.
        Syntax:
            poslist,object,raw = get_positions(input,offset=(0.,0.),region=None)
            
        Input: 
            input   :  numpy array of the (wavelet scaled?) science data
                        this array should correspond to slice specified in 'region'
            sigma   : sigma for gaussian for source/edge detection in image
            size    :  detection limit for objects, typically the size of the kernel
                        used to filter input image. If None, no minimum size
                        will be imposed. Default: None
            offset  : pixel offset of chip relative to final output frame
            region  : slice from full frame array to be used for object detection
            thin    : use homotropic thinning algorithm to extract line segments 
                        corresponding to detected edges, rather than border of 
                        identified edge region. This algorithm currently runs
                        MUCH SLOWER than border extraction.
                    
        Output:
            poslist,objects,rawlist
            where
                poslist: list of positions for identified objects
                objects: list of slices corresponding to identified objects
                rawlist: list of positions for identified objects from array
                
            poslist and rawlist are of the form:
                [id, position, mag]
            where:
                id      :  target number from chip (integer)
                position:  list of position(s) given as x,y pairs (list of lists)
                counts  :  photometry for position(s)(float): 
                            sum of masked region for sources, mean value for edges
            The positions reported in these lists correspond either to a single 
            position for a positive source (star,small galaxy,...) or a list of 
            x,y pixels which correspond to edge features in image.
    """
    # If an object has been provided, break out slice
    #   and offset.  Offset values are used to return positions
    #   in terms of full image coordinates at all times.
    # If no object has been specified, default to full frame
    #   and no offset.
    if region != None:
        oslice = region
        _offset = [max(0,oslice[1].start),max(0,oslice[0].start)]
    else:
        oslice = (slice(None,None,None),slice(None,None,None))
        _offset = [0,0]
    if offset!= None:
        _offset = [_offset[0]+offset[0],_offset[1]+offset[1]]
        
    wchip = input[oslice]
    #
    # Edge detection using Lagrangian-of-Gaussian-like
    #    function from Numarray's nd_image package.
    #
    ggrad = ND.gaussian_gradient_magnitude(wchip,2)
    gstats = imagestats.ImageStats(ggrad,nclip=1)
    clip = N.where(ggrad > gstats.mean + 2*gstats.stddev,ggrad,0)
    del ggrad

    # Now convert it to a Boolean mask
    gclip = gclip > 0
    del clip

    # 
    # Remove objects associated with chip edge
    #
    echip = ND.binary_dilation(wchip==0,iterations=10)
    gclip = gclip - morph.recon_by_dilation(echip,gclip)
    del echip
    #
    # Label detected objects
    #
    wlabels,nobjs = ND.label(gclip)
    # Extract slices for each object 
    wobjs = ND.find_objects(wlabels)
    
    # Trim source detections to only those comparable in size to the kernel
    # Also, delete any object which is smaller than smoothing kernel size.
    nobj_final = nobjs
    for obj in xrange(nobjs):
      if wobjs[obj] != None:
        reg = wobjs[obj]
        reg_xsize = reg[1].stop - reg[1].start
        reg_ysize = reg[0].stop - reg[0].start
         
        if (size != None and (reg_xsize < size[0] and reg_ysize < size[1])) or \
            (reg_xsize <= 1  or reg_ysize <= 1):
             wlabels[wobjs[obj]] = 0
             wobjs[obj] = None
             nobj_final -= 1
                
    #
    # 
    pos_list = []
    raw_list = []
    obj_list = []
        
    print_pos = 0
    obj_id = 0
    print '-- Extracting positions from ',len(wobjs),' regions.'
    for i in xrange(len(wobjs)):
        if wobjs[i] != None:
            #
            # Determine the type of source object: positive source or edge object
            #            
            stype = discriminate_source(wchip[wobjs[i]],wlabels[wobjs[i]],i+1)
            if stype == None: continue
            
            if stype == 1:
                # We have a point source...
                # Start by finding the pixel with maximum value
                maxpos = ND.maximum_position(wchip[wobjs[i]],wlabels[wobjs[i]],i+1)
                maxpos = [maxpos[0]+wobjs[i][0].start+1,maxpos[1]+wobjs[i][1].start+1]
                # Cut out slice centered on max value pixel
                ##
                ## May need to worry about out-of-bounds conditions here
                ##
                #
                pslice = input[maxpos[0]-11:maxpos[0]+12,maxpos[1]-11:maxpos[1]+12].copy()
                plabel = wlabels[maxpos[0]-11:maxpos[0]+12,maxpos[1]-11:maxpos[1]+12].copy()
                # Mask with label mask
                pslice = N.where(plabel == i+1, pslice, 0)
                # Now find center using 'imcntr' algorithm
                cenmass = find_center(pslice)
                # Shift position to final output frame position
                cenpos = [cenmass[1]+maxpos[1]-10+_offset[0], cenmass[0]+maxpos[0]-10+_offset[1]]
                # Compute photometry for source now
                wcounts = ND.sum(wchip,wlabels,i+1)
                # Add source information to target list
                f_list = [cenpos]
                r_list = [[cenpos[0]-offset[0],cenpos[1]-offset[1]]]
                  
                del pslice,plabel
                
            else:        
                #
                #  extract edges using...
                #
                
                if thin == False:
                    edge_clwchip = ND.binary_dilation(gclip[wobjs[i]]) - gclip[wobjs[i]]
                else:
                    edge_clwchip = NP.mmthin(gclip[wobjs[i]])
                    #edge_clwchip = NP.mmthin(edge_clwchip,NP.mmendpoints(),4)                    
                wcounts = ND.mean(wchip,wlabels,i+1) 
                
                nzero_y,nzero_x = N.nonzero(edge_clwchip)            
                xzero = wobjs[i][1].start+_offset[0]+1
                yzero = wobjs[i][0].start+_offset[1]+1
                # For each region, extract the position(s) of the source
                f_list = []
                r_list = []
                for p in xrange(len(nzero_x)):
                    f_list.append([nzero_x[p]+xzero,nzero_y[p]+yzero])
                    r_list.append([nzero_x[p]+xzero-offset[0],nzero_y[p]+yzero-offset[1]])
                """
                cenmass = ND.center_of_mass(wchip,wlabels,i+1)
                cenpos = [cenmass[1]+_offset[0],cenmass[0]+_offset[1]]
                wcounts = ND.mean(wchip,wlabels,i+1) 
                f_list = [[i+1,cenpos,wcounts,stype]]
                r_list = [[i+1,[cenpos[0]-offset[0],cenpos[1]-offset[1]],wcounts,stype]]
                """

            # Keep each regions detections separate, and in sync with 
            # obj_list list of regions. 
            pos_list.append([obj_id,f_list,wcounts,stype])
            raw_list.append([obj_id,r_list,wcounts,stype])
            obj_list.append(wobjs[i])
            obj_id += 1
            del f_list,r_list
                        
    return pos_list,obj_list,raw_list
    
def discriminate_source(array, labels, idnum):
    """ For source number 'idnum' from labels, determine whether
        it is a positive feature or the edge of a larger source.
        It will return 0 for an edge and 1 for a source.
    """
    if (array.shape[0] != labels.shape[0]) or (array.shape[1] != labels.shape[1]):
        return None
    maxpos = ND.maximum_position(array,labels,idnum)
    cenpos = ND.center_of_mass(array,labels,idnum)
    posy = int(maxpos[0])
    posx = int(maxpos[1])
    xslice = [posx-1,posx+2,posx-2,posx+3]
    yslice = [posy-1,posy+2,posy-2,posy+3]
    for i in xrange(len(xslice)):
        if xslice[i] < 0: xslice[i] = 0
        if xslice[i] > array.shape[1]-1: xslice[i] = array.shape[1]-1
    for i in xrange(len(yslice)):
        if yslice[i] < 0: yslice[i] = 0
        if yslice[i] > array.shape[0]-1: i = array.shape[0]-1

    reg1 = array[yslice[0]:yslice[1],xslice[0]:xslice[1]].sum()
    reg2 = array[yslice[2]:yslice[3],xslice[2]:xslice[3]].sum()
    nreg1 = array[yslice[0]:yslice[1],xslice[0]:xslice[1]].nelements()
    nreg2 = array[yslice[2]:yslice[3],xslice[2]:xslice[3]].nelements()

    # Use ND version to get properly masked value of stddev
    svar = ND.standard_deviation(array,labels,idnum)
    smean = ND.mean(array,labels,idnum)
    # number of pixels in larger slice than smaller slice
    dpix = nreg2 - nreg1

    if reg2 - reg1 > (smean+3*svar)*dpix: 
        stype = 1
    else:
        stype = 0

    return stype
    
    
def LOG_function(x,y,s):
    """ Return Laplacian-of-Gaussian for a given position
        with a width of sigma s.
    """
    #D2H_r = -((r*r - s*s)/(s*s*s*s)) * numarray.exp(-((r*r)/(2*s*s)))
    #D2H_xy = (((x*x) + (y*y) -(2*s*s))/(s*s*s*s))*numarray.exp(-(((x*x) + (y*y))/(2*s*s)))
    h = (pow(x,2)+pow(y,2))/(2*pow(s,2))
    D2H_xy = -(1/(N.pi*pow(s,4)))*(1-h)*N.exp(-h)
    return D2H_xy
    
def build_LOGKernel(size,sigma):
    """ Build an LoG kernel of arbitrary size
    """    
    logk = N.zeros(size,dtype=N.float32)
    xcen = int(size[1]/2)
    ycen = int(size[0]/2)
    nelem = logk.nelements()
    for y in xrange(size[0]):
        for x in xrange(size[1]):
            logk[y,x] = LOG_function(x-xcen,y-ycen,sigma)
    lfactor = logk.sum()/nelem
    print lfactor
    
    return N.subtract(logk,lfactor)
    #return logk
    
    
    
def center1d(region):
    """ Compute the center of gravity of a 1-d array.
        Based on 'mpc_getcenter' from IRAF imutil task 'center'
        in the cl.proto package.
        
    """
    mean = region.mean()
    rclip = N.clip(region > mean, 0, region) * (region - mean)
    posn_arr = N.array(range(region.nelements())).reshape(region.shape)
    sum1 =  (posn_arr*rclip).sum()
    sum2 = rclip.sum()
    
    if sum2 > 0.:
        vc = sum1/sum2
    else:
        vc = None
    
    return vc
    
def find_center(region):
    """ Compute the center of a star using MPC algorithm. 
        Based on 'mpc_cntr' from IRAF imutil task 'center'
        in the cl.proto package.
        
        Syntax:
            center = find_center(region)
        Input:
            region - slice of array around target star
        Output: 
            center - array position of center as (y,x)
                        relative to region origin
    """
    maxpos = ND.maximum_position(region)
    rowsum = N.sum(region,axis=0)
    colsum = N.sum(region,axis=1)
    
    ycen = center1d(colsum)
    xcen = center1d(rowsum)
    
    if xcen != None and ycen != None:
        center = [ycen,xcen]
    else:
        center = maxpos

    return center
    
