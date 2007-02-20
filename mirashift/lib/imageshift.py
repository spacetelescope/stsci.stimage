""" This module implements a multi-scale transform based method
    for determining the offset between 2 images.  
    
    The original algorithm 
    was implemented by ESO for their automated image registration in
    the ESI Imaging Survey image processing pipeline, as described by 
    B. Vandame (2002, SPIE 4847, p. 123).  The developers are grateful 
    for the cooperation of the ESO developers in providing a copy of some
    of the EIS software for review during the development of this package.
    
    
    Version 0.1 (Initial version) - WJH (3-Dec-2004)
    Version 0.2 - WJH (15-Dec-2005)
        
"""
import parseinput
import chipwavelets,linearfit

import numpy as N
import os

import pydrizzle
import fileutil


__version__ = '0.2 (15 December 2005)'

class ImageShift:
    """ The ImageShift class serves as the primary interface for
        computing offsets between images using the multi-scale
        median-smoothing transform. 
        
        =================================
        DEVELOPMENT NOTE:  
        This code may eventually need to support fitting to a reference
        image or coordinate list from a reference frame.
        =================================

        Syntax:
            ImageShift(input,output='shifts',reference=None, coeffs='header',
                       scale=2,form='linear')
                        
        Inputs:
            input       - list of input filenames
            output      - name of output shiftfile
                            if nothing is specified, defaults to 'shifts'
            reference   - user-specified reference image
                            if None (default), first image from 
                            input will be used
            coeffs      - parameter used to specify source (if any) of 
                            distortion model to be applied to input images.
                            This corresponds directly to MultiDrizzle
                            'coeffs' and PyDrizzle 'idckey' parameters.
            scale       - number of wavelet transforms to apply: 4 (default)
            form        - form of interpolation kernel to use with wavelet
                            transforms: spline or linear (default)
            median      - radius (in pixels) of initial median 
                            filter: 2 (default)
                             
        Methods:
            .run(verbose=False,min_match=10):
                --> Computes shifts and writes them to output.
                radius  - object matching threshold in pixels
                min_match   - only compute shift if image has 
                                at least 'min_match' objects 
                verbose - print computed shifts interactively

            .writeShiftFile(shift_list = None, output=None):
                --> Writes out shifts to output file.
                shift_list  - list of computed shifts for each image in form of:
                        (filename,((xshift,yshift),rotation,(scale,xscale,yscale))
                            If no shift_list is provided, it writes out values
                            in 'self.shifts'.
                output      - name of output shift file
                            If none is provided, defaults to name specified
                            when class was initialized.                        

    """
    def __init__(self,input, output='shifts',reference=None, coeffs='header',
                 scale=2,form='linear',median=2):  

        self.input = input
        self.scale = scale
        self.form = form        
        self.output = output
        self.overwrite = True
        self.median = median
        
        # Set up shifts attribute to store computed shifts
        self.shifts = None
                
        # Parse out input to insure we only have a list of input filenames
        # Initial assumption: input does not specify an '@'-file which 
        #                   contains IVM filenames
        self.files,asnout = parseinput.parseinput(input)
        
        # 
        # Start processing of images, starting with the reference image
        #
        # Set up PyDrizzle object using:
        #       bits = None : assume all pixel are good (initially)
        #       kernel='turbo' : just a quick drizzling will suffice
        #
        self.pyasn = pydrizzle.PyDrizzle(self.files,output=asnout,
                    kernel='turbo', bits_final=None,bits_single=None,idckey='idctab')
                            
        self.obslist = self._buildObservationList() 
    
        # Set up reference image specification
        if reference == None:
            # If no reference was specified, choose first image in list
            # as reference
            self.refimage = self.files[0]
            # ... and remove it from the list of inputs
            self.files.remove(self.refimage)

            self.refobs = chipwavelets.ReferenceObs(self.pyasn.observation.product.getWCS())
            self.refobs.addChip(self.obslist[0])
        else:
            # Set up reference image specified by user
            self.refimage = reference
            # Check to see if it was in the input list...
            if self.refimage in self.files: 
                # If it is present, remove it from input list
                self.files.remove(self.refimage)
            self.refobs = chipwavelets.ReferenceObs(wcsutil.WCSObject(reference))
            ### ??????????????
            ###     Need to figure out how to setup a user-specified
            ###     image as an Observation object.
            ### ??????????????
            self.refobs.addChip(chipwavelets.Observation(reference)) 
               
        
    def _buildObservationList(self):
        """ Build a composite list of all Observation objects for
            all input images.
        """
        obslist = []
        obs = None
        # Loop over parlist to search for entries matching the given
        # image name
        for pdict in self.pyasn.parlist:
            chip = chipwavelets.Chip(pdict['exposure'],scale=self.scale,median=self.median)
            obsname,obsext = fileutil.parseFilename(pdict['data']) 
            if obs == None:
                obs = chipwavelets.Observation(obsname)       
            elif obsname != obs.name:
                obslist.append(obs)
                del obs
                obs = chipwavelets.Observation(obsname)
            obs.addChip(chip)
            del chip
        #    
        obslist.append(obs)
                
        return obslist

    def getPositionArrays(self,objlist):
        """ Return detected positions as numpy arrays """
        arr = []
        for pos in objlist:
            arr.append(pos[1])
        return N.array(arr)
        
    def writeCoords(self,scale=0):
        """ Write out coordinate files for all input observations.
        """
        for obs in self.obslist:
            self.writeCoordFile(obs.name,scale,obs.getCoords(scale=scale))
    
    def writeCoordFile(self, imagename, scale, objlist):
        """ Write out object list to the coordinate file. """
        if objlist != None:
            image,extn = fileutil.parseFilename(imagename)
            indx = image.rfind('.')
            output = image[:indx]+'_output.coord'
            print '- Writing out coordinates from ',image,' to coordfile: ',output

            clean = self.overwrite
            # If user specifies a clean output file, and
            # a file with that name already exists, remove it.
            if clean == True and os.path.exists(output) == True:
                print '- Removing previous coordinate output file.'
                os.remove(output)

            ofile = open(output,'w')
            ofile.write('#\n# Undistorted positions extracted from: '+image+'\n')
            ofile.write('# based on multi-scale level: '+str(scale)+'\n#\n')
            ofile.write('# Column names:\n')
            ofile.write('# ID   X           Y              Weight      Mag           Type\n')    
            for obj in objlist:
                for val in obj[1]:
                    ofile.write('%s    %0.6f %0.6f    %0.6f  %0.6f    %d\n'%(obj[0],val[0],val[1],obj[4],obj[2],obj[3]))
            ofile.close()        
        
            
    def writeShiftFile(self,shift_list=None,output=None):
        """ Write out a shiftfile."""
        if shift_list == None:
            shift_list = self.shifts
            
        if output == None:
            output = self.output
        if os.path.exists(output):
            if self.overwrite: os.remove(output)
            else: raise IOError,"Output shifts file already exists. Not writing results!"
            
        hdr_str  = "# units: pixels \n"
        hdr_str += "# form: delta \n"
        hdr_str += "# frame: output\n"
        hdr_str += "# reference: "+self.refimage
        
        shift_str ="\n"
        for img in shift_list:
            offsets = img[1]
            if offsets != None:
                shift_str += "%-s    %f  %f  %f\n"%\
                            (img[0],offsets[0][0],offsets[0][1],offsets[1])
            else:
                shift_str += "%-s    %f  %f  %f\n"% (img[0],0.0,0.0,0.0)
            
        f = open(output,'w')
        f.write(hdr_str)
        f.write(shift_str)
        f.close()

    def run(self,verbose=False,min_match=10,overwrite=True,Tccode = 0.8,Tmoment = 0.2,general_fit=False):
        """ Perform the matching between the position lists, then perform
            a generalized linear fit to find the shifts.  These shifts
            then get written out to a shiftfile.
            
            The 'verbose' parameter turns on/off output of compute shifts to
            STDOUT, with the default being turned off (quiet mode).
            
            =================================
            DEVELOPMENT NOTE:
                This needs to be expanded to support iteration over all
                resolution scales, and for all input images.
            =================================
            
            ImageMatch Algorithm:
            Xiaolong Dai, Siamak Khorram, "A Feature-based Image Registration
            Algorithm Using Improved Chain-code Representation Combined with
            Invariant Moments", IEEE Trans. Geo. and Remote Sensing,
            Vol 37, No. 5, 2351-2362, September 1999. 
        """
        self.overwrite = overwrite
                
        # Initialize the output shifts list        
        shift_list = []
        # Start by appending reference image 
        shift_list.append((self.refimage,None))
        if verbose:
            print "Image             Xshift      Yshift      Rot       \n"
            print "===================================================="
            print "%-s    %f  %f  %f\n"% (self.refimage,0.0,0.0,0.0)
        
        ######################################
        # NOTE:
        #
        # Need to iterate over the different scales to improve the 
        # registration. Start with only 1 scale.
        ######################################
        scale = self.scale
        for img in self.obslist:
            # Check to be sure we are not trying to match the
            # reference image to itself
            if img.name == self.refimage:
                continue 
                
            final_fit = chipwavelets.perform_ImageMatch(img,self.refobs,scale,T_ccode=Tccode,T_moment=Tmoment,general=general_fit)
            """
            =================================
            DEVELOPMENT NOTE:
                Shifts need to be converted to a form that can 
                be written out for all input images.
            =================================
            
            # Compute the shifts only for those images which contain 
            # more than 'min_match' objects in common with the reference image.
            if len(poslist1) >= min_match:
                # then fit the matched lists
                offsets = linearfit.fit_arrays(poslist1,poslist2)
                shift_list.append((imgname,offsets))
                
                if verbose:
                    shift_str = "%-40s  %10.4f  %10.4f  %10.4f\n"%\
                        (imgname,offsets[0][0],offsets[0][1],offsets[1])
            else:
                # If no fit is done, return an value of None for this image
                shift_list.append((imgname,None))
                
                if verbose:
                    print 'Warning: Not enough targets for fit!'
                    print '         No shifts computed for:',imgname
        
            """
        # Assign computed shifts to shifts attribute for external 
        # inspection and use.
        self.shifts = shift_list        
        
        # Write out the shift file 
        self.writeShiftFile()        
