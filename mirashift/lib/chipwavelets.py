import os
from math import floor,ceil,sqrt,pow

import numpy as N

import fileutil
import imagestats

        

import expandArray, chainMoments

import atrous,findobjects,objectlist,edge_detect,linearfit


__version__ = '0.3.1 (22-Dec-2005)'

def convert_1d(index,shape):
    ypos = int(index/shape[1])
    xpos = index - (ypos*shape[1])
    return xpos,ypos    

def find_cog(array, xpos,ypos,limit):
    cog_list = []
    for i in range(array.shape[0]):
        p = array[i]
        if abs(p[0] - xpos) <= limit and abs(p[1] - ypos) <= limit:
            cog_list.append([i,p[0],p[1]])
    return cog_list
    
def compute_ccode_matrix(img_ccode, ref_ccode, Lccode = 0.1, Tccode= 0.8):
    ##################
    # DEVELOPMENT NOTE:
    #
    # This code needs to be sped up dramatically somehow. 
    #
    ##################
    #
    # Initialize chain-code matching matrix
    ccode_matrix = N.zeros((len(img_ccode),len(ref_ccode)),dtype=N.float32)
    # Lccode: Chain code length ratio limit
    #
    # Tccode: Threshold for chain-code matching
    #
    # For each image chain_code, match to every 
    # reference image chain_code
    i = 0
    for icode in img_ccode:
        j = 0
        for rcode in ref_ccode:
            #
            # Resample reference chain-code to same length as 
            # image chain-code.
            # The convolution needs to be done to smooth out
            # pixel-to-pixel jumps introduced by the resampling.
            # 
            """
            # This step takes quite a while to run... can it be made more efficient?
            #
            lratio = (float)(len(rcode) - len(icode))/len(rcode)
            #
            # Check to see if chain-codes are within 10% of each other in length
            # This can be done due to distortion correction performed on inputs
            # to bring them all to nearly the same scale already.
            #
            if abs(lratio) > Lccode: 
                ccode_matrix[i][j] = 0.0
            else:
                ccode = chainMoments.computeChainMatch(icode,rcode, 3, objectlist.Chain_kernel)
                # If matching code less than given criteria, then set to 0. 
                # so that it will not be considered a valid match at all.
                if ccode < Tccode: ccode = 0.

                ccode_matrix[i][j] = ccode
            """
            ccode = chainMoments.computeChainMatch(icode,rcode, 3, objectlist.Chain_kernel,Lccode)
            # If matching code less than given criteria, then set to 0. 
            # so that it will not be considered a valid match at all.
            #print 'ccode = ', ccode
            
            if ccode < Tccode: ccode = 0.

            ccode_matrix[i][j] = ccode
            # Increment index over reference chain-codes
            j += 1

        # Increment index over image chain-codes                
        i += 1
    return ccode_matrix


def perform_ImageMatch(imgobs,refobs,scale,T_ccode=0.8,T_moment=0.1,general=False):
    """ Implement the Steps 2-6 of the ImageMatch algorithm 
        on an input Observation object, img_obs, and 
        a reference Observation, ref_obs.
        
        T_ccode: threshold for chain-code matching
        T_moment: threshold for invariant moment matching
        general:  whether to fit rot and scale along with shifts
    """
                
    # for each input image in the observation list, match 
    # against the reference image
    #
    moment_matrix,ccode_matrix,image_cog,ref_cog = _step2(imgobs,refobs,scale)

    _step3(moment_matrix,ccode_matrix,T_moment,T_ccode)

    psi_matrix,min_elems = _step4(moment_matrix,ccode_matrix,image_cog,ref_cog,scale=scale)

    initial_fit = _step5(image_cog,ref_cog,min_elems,general=general)
    print 'Initial fit from _step5: '
    print initial_fit
        
    # RMSE error limit in pixels
    E_r = 2*scale
    #gcp_match = _step6(initial_fit,image_cog,ref_cog,E_r)
    
    # Perform final fit using all matched points
            
    return initial_fit


def _step2(imgobs,refobs,scale):

    # Step 2 of ImageMatch Algorithm:
    # Compute the invariant-moment matrix and
    # chain-code matching matrix, along with getting the
    # center-of-gravity matrix for the image and the reference obs 
    mmatrix,cmatrix = imgobs.computeFeatureMatrices(refobs,scale=scale)
    image_cog = N.array(imgobs.getPositions(scale=scale))
    ref_cog = N.array(refobs.getPositions(scale=scale))
    
    return mmatrix,cmatrix,image_cog,ref_cog

def _step3(mmatrix,cmatrix,T_moment,T_ccode):

    # Step 3 of ImageMatch Algorithm:
    #  Trim possible matches based on detection thresholds
    mmask = mmatrix < T_moment
    cmask = cmatrix > T_ccode
    # Compute mask of all pairs which have non-zero values for both 
    # moments and ccode values after trimming each based on thresholds.
    reject_mask = mmask * cmask
    N.multiply(mmatrix, reject_mask, mmatrix)
    N.multiply(cmatrix, reject_mask, cmatrix)
    del reject_mask,mmask,cmask
    # This step has been tested to work... WJH 3Nov2005


def _step4(mmatrix,cmatrix,image_cog,ref_cog,scale=None):

    # Step 4 of ImageMatch Algorithm: 
    #   generate reduced-potential match set.
    psi_matrix = mmatrix * (1 - cmatrix)
    psi_max = psi_matrix.max() + 1.0
    rad_matrix = psi_matrix.copy() * 0.
    rad_matrix -= 1.0

    # Now, apply radial distance minimization and clustering to
    # work where only those pixels which 
    i = 0
    for ipos in image_cog:
      j = 0
      for rpos in ref_cog:
        if (psi_matrix[i][j] > 0):
            dist = sqrt(pow((ipos[0]-rpos[0]),2)+pow((ipos[1]-rpos[1]),2))
            rad_matrix[i][j] = dist 
        j += 1
      i += 1
    # Set the bin width based on scale
    # Min value: 0.5
    #
    # NOTE:
    #   The algorithm for setting the bin widths may need to be re-evaluated.
    #
    if scale == 0 or scale==None: rscale = 0.5
    else: rscale = scale
    rad_width = 0.5 * (2*rscale)

    # Create histogram of separations between matched sources 
    #   (sources where psi_matrix < psi_max)
    radhist = imagestats.histogram1d(rad_matrix.flat,int(rad_matrix.max())+1,rad_width,0.)

    #
    # Reset distance matrix values for non-matched sources from -1. to max distance
    #  This prevents them from being picked up by the separation limit when
    #  the peak is near a shift of 0.
    #
    radhmax = radhist.histogram.max()
    rad_matrix[N.where(rad_matrix == -1.0)] = rad_matrix.max()
    # Pick out distance bin with largest number of matched sources
    #rad_peak = radhist.centers[N.where(radhist.histogram == radhmax)[0]][0]
    rad_peak = radhist.getCenters()[N.where(radhist.histogram == radhmax)[0][0]]
    # Identify sources from distance matrix with those in peak bin
    nradpeak = N.where(N.abs(rad_matrix - rad_peak) > rad_width)

    psi_matrix[nradpeak] = psi_max 
    psi_elems = N.where(psi_matrix < psi_max)
    
    del rad_matrix
    """
    #
    # This section sorts the matched, trimmed set of pairs 
    #  based on psi_matrix values.
    # This would be necessary if only the first N were found to be 
    # sufficient for fitting.
    #
    # translate 1-d positions back to 2-d positions
    if nelems != None:
        psi_nelems = nelems
    else:
        psi_nelems = len(N.where(psi_matrix < psi_max)[0])
    
    flat_elems = N.argsort(psi_matrix.flat,kind='quicksort')[:psi_nelems]
    min_xref = []
    min_yimg = []
    for e in flat_elems:
        ypos = int(e/psi_matrix.shape[1])
        xpos = e - (ypos*psi_matrix.shape[1])
        min_xref.append(xpos)
        min_yimg.append(ypos)

    min_elems = (min_yimg,min_xref)
    #psi_matrix[min_elems] = psi_max
    
    del min_xref,min_yimg,flat_elems
    """
    return psi_matrix,psi_elems

def _step5(image_cog,ref_cog,min_elems,general=False):
    # Use distance-minimization to limit initial fit to only
    # those points which have the most similar separation
    #
    # Some sort of sigma-clipping and iteration may need to be implemented 
    # here to further clean the fit.  This would require use of the apply_fit()
    # function as demonstrated in _step6().  
        
    # Step 5: 
    #  Use the 3 initial matched pairs for a first fit
    if general:
        initial_fit = linearfit.fit_arrays(image_cog[min_elems[0]],ref_cog[min_elems[1]])
    else:
        initial_fit = linearfit.fit_shifts(image_cog[min_elems[0]],ref_cog[min_elems[1]])
    return initial_fit

def _step6(initial_fit,image_cog,ref_cog,E_r):
    # Step 6:
    # apply fit to CoG values for remainder of potential 
    # matched pairs in psi_matrix and identify those within
    # the RMSE Error limit.
    x_new,y_new = linearfit.apply_fit(image_cog,initial_fit['coeffs'])
    gcp_match = N.sqrt(N.power(x_new - ref_cog[:,0],2) + N.power(y_new - ref_cog[:,1],2))
    if E_r != None:
        gcp_match[N.where(gcp_match >= E_r)] = -1.0

    del x_new,y_new
    
    return gcp_match

class Chip:
    """
    This class keeps track of all the wavelet transformations
    for a chip, and performs the object finding on those transformed
    images.
    
    Input:
        imagename   - full image name complete with extension
                        such as 'test_flt.fits[sci,1]'.
        imagearray  - numpy object containing the science data for image
        offset      - zero-point offset of chip relative to final output frame
        pyasn       - PyDrizzle object relating image to output frame
                        if None, perform no distortion correction in positions
        scale       - Number of wavelet transformations to apply to image
        form        - form of wavelet interpolation: spline or linear (default)
        photzpt     - photometric zero-point appropriate for this chip
        photflam    - photometric conversion factor to convert counts to flux
    Methods:
        getPositions(scale=0)       
            - returns list of undistorted positions for objects 
                identified at specified wavelet transformation scale
        getRawPositions(scale=0)    
            - returns list of original positions for objects identified
                at specified wavelet transformation scale
    """
    def __init__(self, exposure, keep_wavelets=False, 
                 scale=2, form='linear',median=1, clean=True):
            
           
        # Initialize attributes based on inputs
        self.image = exposure.name
        self.scale = scale
        self.form = form
        self.photzpt = exposure.photzpt
        self.photflam = exposure.photflam
        self.median = median        

        self.waveplanes = None
        self.wavechip = None      
        self.maxkernel = None
        self.clean = clean
        self.rootname = ""
        self.chip = ""

        # Keep track of delta shift for this chip
        self.chip_delta = (0.,0.)
        
        # Keep track of offset of chip into final product
        self.xzero = int(exposure.xzero + self.chip_delta[0] )
        self.yzero = int(exposure.yzero + self.chip_delta[1] )
        
        # Remove distortion from input chip
        imagearray = exposure.runDriz()
        
        # Generate compressed mask for this chip
        self.mask = expandArray.collapseMask(imagearray)        

        self.shape = imagearray.shape

        # Keep track of range of pixels this chip spans
        self.corners = self.computeRange()
        
        # Build up multi-scale transformations of input
        print '- Creating ',self.scale,' multi-scale views of this image'
        self.waveplanes,self.wavechip,self.maxkernel = atrous.multimed(imagearray,maxscale=self.scale,median=self.median)
        
        #if requested - write out intermediate files - fr debugging
        if not self.clean:
            import pyfits as p
            hdulist = p.HDUList()
            phdu = p.PrimaryHDU()
            hdulist.append(phdu)
            for w in self.waveplanes:
                ehdu = p.ImageHDU(w)
                hdulist.append(ehdu)
                hdulist.close()
            self.rootname = exposure.name.split('_')[0]
            self.chip = exposure.chip
            name = self.rootname+str(exposure.chip)+'_waveplanes.fits'
            hdulist.writeto(name, clobber=True)
            phdu = p.PrimaryHDU(self.wavechip)
            name = self.rootname+str(exposure.chip)+'_wavechip.fits'
            phdu.writeto(name, clobber=True)
            del phdu, hdulist, ehdu, p 
        # Finished with original input, so delete it
        del imagearray
        
        # Build list of objects
        print '- Building list of detected objects from all scales'
        self.objectlist = self.__setObjectList()
                
        # Cleanup of variables
        self.cleanWavelets()
                    
    def cleanWavelets(self):
        del self.waveplanes
        del self.wavechip
        self.waveplanes = None
        self.wavechip = None
        
    def __setObjectList(self):
        """ 
        Build full object list for all wavelet scales. 
        """

        #############
        # Edge detection using fast Lagrangian-of-Gaussian
        #    as described in Chen et al(1987).
        ### AND  ### 
        # Find zero-crossings: actually, find the contours corresponding
        #    to mean+stddev level (for the minimum).
        #############
        ##
        # At this point, we have the contours of the zero-crossing 
        #    edges.  
        ##
        # Initialize ObjectList
        print 'Defining all objects for scale: ',self.scale
        objlist = objectlist.ObjectList(self.wavechip, scale=self.scale,offset=(self.xzero,self.yzero), clean=self.clean, name=self.rootname+str(self.chip))
        # For each scale, starting at the top...
        # This loop goes from scale to 0 by 1, i.e. 4-3-2-1-0
        for s in xrange(self.scale-1,-1,-1):
            # back out the wavelet transform for this scale
            # we will reuse the same array to minimize memory
            sc = atrous.atrous_restore(self.waveplanes,self.wavechip,scale=s)
            sobj = objlist.getSlices(scale=s+1)

            # Get all objects identified for this scale which
            # were also identified at a previous scale (lower resolution)
            print 'Adding objects for scale: ',s
            objlist.addObjects(sc,s,sobj)

            del sc,sobj
            
        return objlist
        
    def getFluxes(self,scale=0,units='mag'):
        """ Returns fluxes for all objects.  
            If units='mag', fluxes will be returned as magnitudes
            rather than electrons/counts/ADUs based on photometric
            keywords read in for this chip. 
        """
        flux_list = self.objectlist.getFluxes(scale=scale)
        if units.lower() == 'mag':
            olist = self._convertCountsToMag(flux_list)
        else:
            olist = flux_list
        return olist
        
    def getMask(self):
        """ Return expanded version of mask for chip in output frame."""
        return expandArray.inflateMask(self.mask)
        
    def computeRange(self):
        corners = [(self.xzero,self.yzero),(self.xzero+self.shape[1],self.yzero+self.shape[0])]
        return corners
                
    def setDelta(self,delta):
        self.chip_delta = delta
        self.objectlist.chip_delta = delta
        
        # Now update corner positions based on new delta
        self.corners = self.computeRange()

    def addDelta(self,delta):
        self.chip_delta = (self.chip_delta[0]+delta[0],self.chip_delta[1]+delta[1])
        self.objectlist.chip_delta = (self.objectlist.chip_delta[0]+delta[0], self.objectlist.chip_delta[1]+delta[1])

        # Now update corner positions based on new delta
        self.corners = self.computeRange()
        
    def outputPositions(self,output,scale=0,clean=True):
        """ Writes extracted undistorted positions to output ASCII file."""
        objlist = self.getPositions(scale=scale)
        
        # If user specifies a clean output file, and
        # a file with that name already exists, remove it.
        if clean == True and os.path.exists(output) == True:
            print 'removing previous product...'
            os.remove(output)
            
        ofile = open(output,'w')
        ofile.write('#\n# Undistorted positions extracted from: '+self.image+'\n')
        ofile.write('# based on wavelet transformation scale: '+str(scale)+'\n#\n')
        ofile.write('# Column names:\n')
        #ofile.write('# ID   X           Y              Weight      Mag           Type\n')    
        ofile.write('# ID   X           Y\n') 
        n = 0
        for obj in objlist:
            #for val in obj[1]:
                #ofile.write('%s    %0.6f %0.6f    %0.6f  %0.6f    %d\n'%(obj[0],val[0],val[1],obj[4],obj[2],obj[3]))
            ofile.write('%s    %0.6f %0.6f    \n'%(n,obj[0],obj[1]))
        ofile.close()        


    def _convertCountsToMag(self,objlist):
        """ Iterates through an objlist and converts the 
            listed counts to a magnitude. 
            It also computes weighting factor which can be 
            used in matching sources, and appends it to each object.
            The weighting was designed to return values from 
            -15 (bright) to 0 (faint).
        """        
        for pos in objlist:
            pos.append(-5 - 2.5* N.log10(pos[2]))
            pos[2] = self.photzpt - 2.5 * N.log10(self.photflam*pos[2])
        return objlist
        

class Observation:

    def __init__(self,name):
        rname,extn = fileutil.parseFilename(name)
        self.name = rname
        self.chiplist = []
        self.updated_range = False
        self.mask = None

        self.shape = None
        self.corners = None

        self.xzero = 0
        self.yzero = 0
        self.obs_delta = (0.,0.)
        
    def addChip(self,chip):
        self.chiplist.append(chip)
        self.updated_range = False

    def computeRange(self):
        """ compute range of pixels spanned by entire observation
        in output frame.
         
        This NEEDS to be run once all member chips have been added
        to this object.
        """
        if self.updated_range == False:
            rlist = []
            for chip in self.chiplist:
                rlist.extend(chip.corners)
            rarr = N.array(rlist)

            corners =[ (int(floor(rarr[:,0].min())),int(floor(rarr[:,1].min()))),
                       (int(ceil(rarr[:,0].max())),int(ceil(rarr[:,1].max()))) ]

            del rarr,rlist

            if self.shape == None:
                self.xzero = corners[0][0]
                self.yzero = corners[0][1]
                self.shape = (int((corners[1][1] - corners[0][1])+1),int((corners[1][0] - corners[0][0])+1))

            self.corners = corners
            self.updated_range = True
            
    def computeFeatureMatrices(self,refobs,scale=0,order=3):
        """ Return the feature matrices of the input Observation
            relative to the reference Observation; specifically,
            the invariant-moment distance matrix, the chain-code
            matching matrix, and a center-of-gravity matrix for
            each. 
        """
        
        # Compute the Moment matrix
        img_mmat = self.getMoments(scale=scale)
        ref_mmat = refobs.getMoments(scale=scale)
        print ' -Computing moment matrix...'
        moment_matrix = chainMoments.getMomentMatrix(img_mmat,ref_mmat)
        """
        # Compute matrix based solely on 2nd moment
        #
        i = 0
        moment_matrix = N.zeros((len(img_mmat),len(ref_mmat)), N.Float32)
        for imom in img_mmat:
            j = 0
            for rmom in ref_mmat:
                mom_ratio = imom[7]/rmom[7]
                if abs(1 - mom_ratio) < 0.2:
                    moment_matrix[i][j] = abs(imom[1] - rmom[1])
                else:
                    moment_matrix[i][j] = -1.0
                j += 1
            i += 1
        del img_mmat, ref_mmat
        """
        # Compute the chain-code matrix
        img_ccode,img_clen = self.getChainCodes(scale=scale)
        ref_ccode,ref_clen = refobs.getChainCodes(scale=scale)
        
        print ' -Computing chain-code matrix... for self.name'
        # Initialize chain-code matching matrix
        ccode_matrix = compute_ccode_matrix(img_ccode, ref_ccode, Lccode = 0.1, Tccode= 0.8)

        return moment_matrix,ccode_matrix

    def createMask(self):
        ''' Creates mask of entire observation in output field.'''
        if self.mask == None:
            self.mask = N.zeros(self.shape,dtype=N.uint8)

            for chip in self.chiplist:
                # We need to put the chip corners, 
                #    expressed in terms of the final output frame,
                # into the observation frame by subtracting the
                # observations zero-point.
                c = N.array(chip.corners) - (self.xzero,self.yzero)
                cmask = chip.getMask()
                smask = self.mask[c[0][1]:c[1][1],c[0][0]:c[1][0]]
                N.bitwise_or(smask, cmask,smask)
                del cmask,smask
        
    def getCoords(self,scale=0):
        """ Returns positions, weight, max, and type for all Objects
            detected at the specified scale. 
            
            USED ONLY BY .writeCoords() method of WaveShifts object.
            The type/list of values returned could still be modified.  
        """
        return None
        
    def setDelta(self,delta):
        """ Set global delta for entire observation. """
        self.obs_delta = delta
        for chip in self.chiplist:
            chip.setDelta(delta)
        
    def getMoments(self,scale=0):
        olist = []
        for chip in self.chiplist:
            olist.extend(chip.objectlist.getMoments(scale=scale))
        return olist
        
    def getChainCodes(self,scale=0):
        olist = []
        llist = []
        for chip in self.chiplist:
            ccodelist,clenlist = chip.objectlist.getChainCodes(scale=scale)
            olist.extend(ccodelist)
            llist.extend(clenlist)
            del ccodelist,clenlist        
        return olist,llist
        
    def getPositions(self,scale=0):
        olist = []
        for chip in self.chiplist:
            pos = chip.objectlist.getPositions(scale=scale)+self.obs_delta
            olist.extend(pos.tolist())            
        return olist
    
    def getSlices(self, scale=0):
        olist = []
        for chip in self.chiplist:
            olist.extend(chip.objectlist.getSlices(scale=scale))
        return olist

    def getFluxes(self, scale=0):
        olist = []
        for chip in self.chiplist:
            olist.extend(chip.objectlist.getFluxes(scale=scale))
        return olist
        
    def getScales(self):
        if len(self.chiplist) > 0:
            return self.chiplist[0].objectlist.getScales()
        else:
            return None

class ReferenceObs(Observation):
    """ Class used as reference observation for iterating the fit.   
    """
    def __init__(self,wcs):
        self.wcs = wcs

        Observation.__init__(self, wcs.rootname)
        self.shape = (wcs.naxis2,wcs.naxis1)
        self.mask =  N.zeros(self.shape,dtype=N.uint8)
               
    def checkOverlap(self,obs):
        ''' Determine whether this observation overlaps the current
            reference mask. 
        '''
        obs.computeRange()
        obs.createMask()

        smask = self.mask[obs.corners[0][1]:obs.corners[1][1], obs.corners[0][0]:obs.corners[1][0]]
        N.bitwise_or(smask,obs.mask[:smask.shape[0],:smask.shape[1]],smask)
        
        if len(N.nonzero(smask)[0]) > 0:
            overlap = True
        else:
            overlap = False
        
        return overlap
        
    def overlapMask(self,obs):
        ''' Updates mask of entire observation in output field,
            IF it is found to overlap observations already in reference mask.
            
            It returns a flag denoting whether the observation overlapped
            or not and, therefore, whether the mask was updated or not.
            
            It works on entire observations, rather than just chip-by-chip.
        '''

        overlap = self.checkOverlap(obs)
        if overlap == True:
            smask = self.mask[obs.corners[0][1]:obs.corners[1][1],obs.corners[0][0]:obs.corners[1][0]]
            N.bitwise_or(smask, obs.mask[:smask.shape[0],:smask.shape[1]],smask)
            del obs.mask
            obs.mask = None
        return  overlap
                          
    def addChip(self,obs):
    
        # Not only add the chips from this observation to the chip list...
        self.chiplist.extend(obs.chiplist)
        # ... but also update the reference mask to account for them as well.
        self.overlapMask(obs)        

    def writeShifts(self,filename):
        pass
    
