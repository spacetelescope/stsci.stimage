import numpy as N
import ndimage as ND

import chainMoments
import edge_detect
import imagestats
import numdisplay

from math import log10

__version__ = '0.1.0 (30-November-2004)'

Chain_kernel = [0.1,0.2,0.4,0.2,0.1]

def sumSlices(a,b):
    """ Updates slice 'a' to be consistent with the 
        indexing used for slice 'b'.
        For example, if slice 'a' was derived from slice 'b',
        the sum would return slice 'a' in the frame of 'b'.
    """
    if b[0].start == None and b[1].start == None:
        return a

    yslice = slice(a[0].start+b[0].start,a[0].stop+b[0].start,a[0].step)
    xslice = slice(a[1].start+b[1].start,a[1].stop+b[1].start,a[1].step)

    return (yslice,xslice)

def center1d(region):
    """ Compute the center of gravity of a 1-d array.
        Based on 'mpc_getcenter' from IRAF imutil task 'center'
        in the cl.proto package.
        
    """
    mean = region.mean()
    rclip = N.clip(region > mean, 0, region) * (region - mean)
    posn_arr = N.array(range(region.size)).reshape(region.shape)
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
    

class ObjectList:
    """ 
    This class manages the properties of detected objects
    from multi-scale wavelet transformed image stacks. 
    Methods include:
        getObjects(scale=0)
        getPositions(scale=0)
        getRawPositions(scale=0)
        getSlices(scale=0)
   
    """
    def __init__(self, image, scale=0, offset=(0.,0.), clean=True, name=None):
        self.clean = clean
        self.name = name    
        self.scale = scale
        # If we do not provide a list of objects, 
        # setup empty list for members
        self.objectlist = {}
        self.addObjects(image, scale)
        self.chip_delta = offset

        

    def addObjects(self, image, scale,slices=None):
        self.objectlist[scale] = self.buildObjectlist(image,slices=None)
        
    def getScales(self):
        _scales = []
        for s in self.objectlist.keys():
            _scales.append(int(s))
        return _scales
    
    def buildObjectlist(self, image, slices=None):
        """ Adds members to objectlist."""
        if slices == None:
            _s = [(slice(None,None,None),slice(None,None,None))]
        else:
            _s = slices

        edges = edge_detect.find_LoG_zeros(image, clean=self.clean, name=self.name)
        
        #
        # Label detected objects
        #
        objectlist = []
        for region in _s:
            # Guard against 'contours' which can not and should not
            # be 'filled' or used in general.
            if ( (edges[region].shape[0] <= 3) or (edges[region].shape[1] <= 3) ):
                continue  
            _edge = edges[region]
            _img = image[region]
            
            wlabels,nobjs = ND.label(_edge,structure=ND.generate_binary_structure(2,2))
            # Extract slices for each object 
            wobjs = ND.find_objects(wlabels)
            n = 0
            for wn in wobjs:
                _slice = sumSlices(wn,region)
                if ( _edge[wn].shape[0] <= 3 or _edge[wn].shape[1] <= 3):
                    continue
                # Convert contours from 'edges' into masks for each
                # detected object.  Use mask on '_img' for computing
                # gray-scale moments, not just binary edge moments.                
                _fedge = ND.binary_fill_holes(_edge[wn])
                _masked_img = N.multiply(_img[wn],_fedge)
                objectlist.append(Object(_masked_img,_edge[wn],_slice,n))
                n += 1

                del _slice,_fedge,_masked_img
                
            del wlabels,nobjs,wobjs,_edge,_img

        del edges,_s
        return objectlist
                   
    def getObjects(self,scale=0):
        """
        Return FULL list of object instances for a given scale.
        """
                
        try:
            self.verifyScale(scale)
        except ValueError:
            return None

        return self.objectlist[scale]        
        
    def getSlices(self,scale=0):
        """ 
        Returns list of slices for member objects at the given scale.

        This method ALWAYS returns a list, even if it only has 1 member.
        """
        try:
            self.verifyScale(scale)
        except ValueError:
            return None

        slicelist = []
        for member in self.objectlist[scale]:
            slicelist.append(member.region)

        return slicelist
            
    def getPositions(self,scale=0):
        """ 
        Returns positions (center-of-gravity) for the scale specified.

        This method ALWAYS returns a list, even if it only has 1 member.
        """

        try:
            self.verifyScale(scale)
        except ValueError:
            return None

        poslist = []
        for member in self.objectlist[scale]:
#            poslist.append(member.center)
            poslist.append(member.cenmass)
        pos = N.array(poslist) + self.chip_delta
        del poslist
        
        return pos

    def getFluxes(self,scale=0):
        """ 
        Returns fluxes/total counts for each object at the scale specified.

        This method ALWAYS returns a list, even if it only has 1 member.
        """

        try:
            self.verifyScale(scale)
        except ValueError:
            return None

        fluxlist = []
        for member in self.objectlist[scale]:
            fluxlist.append(member.flux)
        
        return fluxlist

    def getMoments(self,scale=0):
        """ 
        Returns computed invariant moments for the scale specified.

        This method ALWAYS returns a list, even if it only has 1 member.
        """
        try:
            self.verifyScale(scale)
        except ValueError:
            return None

        poslist = []
        for member in self.objectlist[scale]:
            poslist.append(member.moments)

        return poslist
            
    def getChainCodes(self,scale=0):
        """ 
        Returns extracted chain codes for the scale specified.

        This method ALWAYS returns a list, even if it only has 1 member.
        """
        try:
            self.verifyScale(scale)
        except ValueError:
            return None
            
        poslist = []
        clenlist = []
        for member in self.objectlist[scale]:
            poslist.append(member.chaincode)
            clenlist.append(member.codelength)
            
        return poslist,clenlist
        
    def verifyScale(self,scale):
        keys= self.objectlist.keys()
        if scale not in keys:
            print 'Valid scales are: ',keys
            print 'Please specify one of these...'
            raise ValueError
        

class Object:

    def __init__(self,image, edges, region, index, binary=False):
        # This will serve as the ID for this object.
        self.index = index
        self.region = region        

        # Numarray center_of_mass does not take into account
        # the flux at each pixel
        # Center based on flux distribution 
        cenxy = find_center(image)
        self.center = (cenxy[1]+region[1].start,cenxy[0]+region[0].start)
            
        self.array = None
        self.flux = ND.sum(image)
        
        self.ycen_mass,self.xcen_mass = ND.center_of_mass(image)
        self.cenmass = (self.xcen_mass+region[1].start, self.ycen_mass+region[0].start)
        self.chaincode,self.codelength = self.computeChainCode(edges)
        if binary:
            self.moments = self.computeBinaryMoments(edges)
        else:
            self.moments = self.computeMoments(image)
        
        # Keep track of whether this object was successfully 
        # matched against an object from another chip/observation
        # The policy should be to KEEP but IGNORE any objects
        # that do not match.
        self.matched = False
        
    def computeChainCode(self,cpix):
        """ 
        Compute modified chain-code for each contour
        """
        # get pixels in contour
        npix = len(N.nonzero(cpix)[0])
        ccode = chainMoments.getChainCode(cpix.astype(N.float32),npix)
        gcode = ND.convolve(ccode.astype(N.float32),Chain_kernel)
        # compute and subtract the mean from the chain-code
        gcode -= N.average(gcode)

        del ccode,cpix
    
        return gcode,npix

    def computeBinaryMoments(self, edge):
        """
        # For this contour/object, compute the 7 invariant moments
        """
        # Pre-compute necessary central moments needed 
        #     for computing the 7 invariant moments
        #
        ynz,xnz = N.nonzero(edge)
        xnz -= self.xcen_mass
        ynz -= self.ycen_mass
        mu00 = chainMoments.compute_binary_moment_pq(xnz,ynz,0,0)
        mu11 = chainMoments.compute_binary_moment_pq(xnz,ynz,1,1)
        mu12 = chainMoments.compute_binary_moment_pq(xnz,ynz,1,2)
        mu20 = chainMoments.compute_binary_moment_pq(xnz,ynz,2,0)
        mu02 = chainMoments.compute_binary_moment_pq(xnz,ynz,0,2)
        mu21 = chainMoments.compute_binary_moment_pq(xnz,ynz,2,1)
        mu30 = chainMoments.compute_binary_moment_pq(xnz,ynz,3,0)
        mu03 = chainMoments.compute_binary_moment_pq(xnz,ynz,0,3)
        # Now compute normalized invariant moments
        """
        psi1 = (mu20 + mu02) / (mu00*mu00)
        psi2 = ( pow((mu20 - mu02),2) + 4*pow(mu11,2) ) / pow(mu00,4)
        psi3 = ( pow((mu30 - 3*mu12),2) + pow((3*mu21 - mu03),2)) / pow(mu00,5)
        psi4 = ( pow((mu30 + mu12),2) + pow((mu21 + mu03),2) ) / pow(mu00,5)
        psi5 = ( (mu30 - 3*mu12)*(mu30 + mu12)* 
                 (pow((mu30 + mu12),2) - 3*pow((mu21+mu03),2)) -
                 (3*mu21 - mu03) * (mu21 + mu03)* 
                 (pow((mu21 + mu03),2) - 3*(pow((mu30 + mu12),2)))
                ) / pow(mu00,10)
        psi6 = ( (mu20 - mu02)* (pow((mu30+mu12),2) - pow((mu21+mu03),2)) -
                 4*(mu30+mu12) * (mu21+mu03)
                ) / pow(mu00,7)
        psi7 = (  (3*mu21 - mu03)*(mu30+mu12) *
                ( pow((mu30 +mu12),2) - 3*pow((mu21+mu03),2)) -
                (mu30 - mu12) * (mu21+mu03) * 
                (3*pow((mu30 + mu12),2) - pow((mu21+mu03),2))
                ) / pow(mu00,10)
        """
        # Now compute un-normalized invariant moments
        psi1 = (mu20 + mu02)
        psi2 = ( pow((mu20 - mu02),2) + 4*pow(mu11,2) ) 
        psi3 = ( pow((mu30 - 3*mu12),2) + pow((3*mu21 - mu03),2)) 
        psi4 = ( pow((mu30 + mu12),2) + pow((mu21 + mu03),2) ) 
        psi5 = ( (mu30 - 3*mu12)*(mu30 + mu12)* 
                 (pow((mu30 + mu12),2) - 3*pow((mu21+mu03),2)) -
                 (3*mu21 - mu03) * (mu21 + mu03)* 
                 (pow((mu21 + mu03),2) - 3*(pow((mu30 + mu12),2)))
                )
        psi6 = ( (mu20 - mu02)* (pow((mu30+mu12),2) - pow((mu21+mu03),2)) -
                 4*(mu30+mu12) * (mu21+mu03)
                ) 
        psi7 = (  (3*mu21 - mu03)*(mu30+mu12) *
                ( pow((mu30 +mu12),2) - 3*pow((mu21+mu03),2)) -
                (mu30 - mu12) * (mu21+mu03) * 
                (3*pow((mu30 + mu12),2) - pow((mu21+mu03),2))
                ) 
        psi = []
        psi.append(psi1)
        psi.append(psi2)
        psi.append(psi3)
        psi.append(psi4)
        psi.append(psi5)
        psi.append(psi6)
        psi.append(psi7)
        psi.append(float(self.codelength))
        
        return psi        
        

    def computeMoments(self, image):
        """
        # For this contour/object, compute the 7 invariant moments
        """
        xcen = self.xcen_mass
        ycen = self.ycen_mass
        # Pre-compute necessary central moments needed 
        #     for computing the 7 invariant moments
        #
        mu00 = chainMoments.compute_moment_pq(image,xcen,ycen,0,0)
        mu11 = chainMoments.compute_moment_pq(image,xcen,ycen,1,1)
        mu12 = chainMoments.compute_moment_pq(image,xcen,ycen,1,2)
        mu20 = chainMoments.compute_moment_pq(image,xcen,ycen,2,0)
        mu02 = chainMoments.compute_moment_pq(image,xcen,ycen,0,2)
        mu21 = chainMoments.compute_moment_pq(image,xcen,ycen,2,1)
        mu30 = chainMoments.compute_moment_pq(image,xcen,ycen,3,0)
        mu03 = chainMoments.compute_moment_pq(image,xcen,ycen,0,3)
        # Now compute scaled invariant moments
        """
        psi1 = (mu20 + mu02) / (mu00*mu00)
        psi2 = ( pow((mu20 - mu02),2) + 4*pow(mu11,2) ) / pow(mu00,4)
        psi3 = ( pow((mu30 - 3*mu12),2) + pow((3*mu21 - mu03),2)) / pow(mu00,5)
        psi4 = ( pow((mu30 + mu12),2) + pow((mu21 + mu03),2) ) / pow(mu00,5)
        psi5 = ( (mu30 - 3*mu12)*(mu30 + mu12)* 
                 (pow((mu30 + mu12),2) - 3*pow((mu21+mu03),2)) -
                 (3*mu21 - mu03) * (mu21 + mu03)* 
                 (pow((mu21 + mu03),2) - 3*(pow((mu30 + mu12),2)))
                ) / pow(mu00,10)
        psi6 = ( (mu20 - mu02)* (pow((mu30+mu12),2) - pow((mu21+mu03),2)) -
                 4*(mu30+mu12) * (mu21+mu03)
                ) / pow(mu00,7)
        psi7 = (  (3*mu21 - mu03)*(mu30+mu12) *
                ( pow((mu30 +mu12),2) - 3*pow((mu21+mu03),2)) -
                (mu30 - mu12) * (mu21+mu03) * 
                (3*pow((mu30 + mu12),2) - pow((mu21+mu03),2))
                ) / pow(mu00,10)
        """
        # Now compute invariant moments
        psi1 = (mu20 + mu02)
        psi2 = ( pow((mu20 - mu02),2) + 4*pow(mu11,2) )
        psi3 = ( pow((mu30 - 3*mu12),2) + pow((3*mu21 - mu03),2))
        """
        psi4 = ( pow((mu30 + mu12),2) + pow((mu21 + mu03),2) ) 
        psi5 = ( (mu30 - 3*mu12)*(mu30 + mu12)* 
                 (pow((mu30 + mu12),2) - 3*pow((mu21+mu03),2)) -
                 (3*mu21 - mu03) * (mu21 + mu03)* 
                 (pow((mu21 + mu03),2) - 3*(pow((mu30 + mu12),2)))
                ) 
        psi6 = ( (mu20 - mu02)* (pow((mu30+mu12),2) - pow((mu21+mu03),2)) -
                 4*(mu30+mu12) * (mu21+mu03)
                ) 
        psi7 = (  (3*mu21 - mu03)*(mu30+mu12) *
                ( pow((mu30 +mu12),2) - 3*pow((mu21+mu03),2)) -
                (mu30 - mu12) * (mu21+mu03) * 
                (3*pow((mu30 + mu12),2) - pow((mu21+mu03),2))
                )
        """
        psi = []
        psi.append(log10(psi1))
        psi.append(log10(psi2))
        psi.append(log10(psi3))
        psi.append(0.0)
        psi.append(0.0)
        psi.append(0.0)
        psi.append(0.0)
        psi.append(float(self.codelength))
        
        return psi        
