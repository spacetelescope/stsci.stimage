"""
saaclean: Module for estimating and removing persistent CR signal due to a prior
          SAA passage.

Usage:    Normally used via the STSDAS task saaclean in the nicmos package.
          To use as pure python, create a params object to override any of
          the default parameters if desired, then invoke clean:
          >>> mypars=saaclean.params(thresh=0.23)
          >>> saaclean.clean('inputfile.fits','outputfile.fits',pars=mypars)

For more information:
          Additional user information, including parameter definitions and more
          examples, can be found in the help file for the STSDAS saaclean task,
          located in nicmos$doc/saaclean.hlp.

          The algorithm and IDL prototype are described in the NICMOS
          ISR 2003-009, by Bergeron and Dickinson, available through the NICMOS
          webpage.
          
Dependencies:
          numarray v0.6 or higher
          pyfits v0.6 or higher
          imagestats v0.2.1

"""
# The above text is duplicated in the __init__ file for the package, since
#that's where it shows up for the user.

import os 
import exceptions
import numarray,pyfits
from imagestats import ImageStats as imstat #pyssg lib
import SP_LeastSquares as LeastSquares #Excerpt from Hinsen's Scientific Python

__version__="0.86dev"
### Warning warning warning, this is listed in the __init__.py ALSO.
### Change it in both places!!!!!!

#History:
# Enhancements, 20 Jan 06, Laidler
#   - replaced infile by calcfile and targfile
#   - allow applying correction to a file other than that on which it
#     was computed
#   - added "clobber" parameter to control clobber behavior on all
#     output files.
# Enhancement, 21 Dec 05, Laidler
#  use header value of ADCGAIN in place of GAINPLOT
#  add new CRTHRESH, NOISETHRESH, BINSIGFRAC parameters
# Enhancement, 21 Dec 05, Laidler
#    - add chi2 output to header keyword set. This involved changing
#      the signature of the parabola_min function to return the chi2.
# Enhancement, 19 Dec 05, Laidler
#    - use DQ extension for bad pixels
# Bugfix,21 Jan 05, Laidler
#    - make all paths & filenames more robust via osfn
# Bugfix,20 Jan 05, Laidler:
#    - correct handling of middle column/row in Exposure.pedskyish()
# Bugfixes, 3 Aug 04, Laidler:
#    - make filename construction more robust via os.path.abspath on the directory
#    - Ensure directory specified in pars.darkpath exists
#    - Fall back to extension 0 if extension 1 doesn't exist (Exposure class)
#    - Support noise reduction in high domain only
# Initial python implementation: Dec 2003, Laidler
# Based on IDL implementation by Bergeron

#Notes for future improvement:
# - possibly make saaper its own class
# - the crthreshholding code is kind of tacky
# - possibly make filename its own class so it can have a method for nref
#   instead of using the osfn helper function
#........................................................................
#Class definitions
#.........................................................................

class params:
    def __init__(self,scale=0.54,wf1=0.7,wf2=0.3,
                 stepsize=0.008,thresh=None,hirange=0.4,lorange=0.25,dofit=1,
                 crthresh=0.3, noisethresh=1.0, binsigfrac=0.3,
                 readsaaper='False',writesaaper='True',saaperfile='saaper.fits',
                 clobber='False',
                 flatsaaper='True',flatsaaperfile=None,
                 maskfile=None,darkpath=None,diagfile=None):
        self.scale=scale
        self.wf1=wf1
        self.wf2=wf2
        self.writesaaper=writesaaper
        self.readsaaper=readsaaper
        self.saaperfile=osfn(saaperfile)
        self.clobber=clobber
        self.flatsaaper=flatsaaper
        self.flatsaaperfile=osfn(flatsaaperfile)
        self.maskfile=osfn(maskfile)
        self.stepsize=stepsize
        self.thresh=thresh
        self.hirange=hirange
        self.lorange=lorange
        self.dofit=dofit

        self.crthresh=crthresh
        self.noisethresh=noisethresh
        self.binsigfrac=binsigfrac
        
        self.darkpath=osfn(darkpath)
        self.diagfile=osfn(diagfile)

        self.appstring=None   # Might be needed later.
        
class Domain:
    """ Stores a list of pixels for a (typically high or low) signal domain"""
    
    def __init__(self,name,pixellist,range):
        self.name=name
        self.pixlist=pixellist
        self.range=range
        self.npix=len(self.pixlist[0])
        #Because the pixlist is created with a "where" statement, it's a
        #2 element array, 1 for x & 1 for y. Thus to get the number of
        #pixels, we need the length in one of the elements.

      

    def striplowerthan(self,factor):
        """self.pp is defined in Exposure.getscales
        It contains the (bin, stddev, mode) for the statistical analysis.
        striplowerthan(factor) examines the stddev column only, and replaces
        all values of the stddev that are less than factor*the zeroth bin,
        with the maximum stddev.
        """
        p1=self.pp[1,:]
        uu=numarray.where(p1 < factor*p1[0])
        if uu[0].nelements != 0:
            p1[uu]=p1.max()
            self.pp[1,:]=p1

    def getmin(self):
        ubest=numarray.where(self.pp[1,:] == self.pp[1,:].min())[0][0]
        umode=numarray.where(self.pp[2,:] == self.pp[2,:].min())[0][0]
        return ubest, umode

    def writeto(self,filename,clobber=False):
        if not clobber:
            if os.path.exists(filename):
                raise IOError, "%s already exists: aborting\n"%filename
        #if clobber=True or file does not exist, proceed anyhow    
        f=open(filename,'w')
        f.write('# '+self.name+'\n')
        f.write('# Pixels in this domain: '+`len(self.pixlist)`+'\n')
        f.write('#  1  scale factor  \n')
        f.write('#  2  sigma   \n')
        f.write('#  3  mode  \n')
        for i in range(len(self.pp[0])):
            f.write('%f   %f    %f\n' % (self.pp[0,i],self.pp[1,i],self.pp[2,i]))
        f.close()
                   
    
class Exposure:
    """ Stores a collection of keywords and the image data for an exposure. """
    
    def __init__(self,imgfile,nickname=None):
        
        self.filename=osfn(imgfile)
        if nickname is None:
            self.nickname=self.filename
        else:
            self.nickname=nickname
        f=pyfits.open(self.filename)
        self.f=f
        h=f[0].header
        self.h=h
        try: #Assume data is in extension 1. if not, fall back to extension 0.
            self.data=f[1].data   #.astype('Float32')
            self.extnum=1
        except IndexError:
            self.data=f[0].data
            self.extnum=0
        self.exptime=h['exptime']
        self.camera=h['camera']
        self.saa_time=h['saa_time']
        self.badfile=osfn(h['maskfile'])
        self.gainplot=h['adcgain']
 
        self.inq1=slice(10,118),slice(10,118)              
        self.inq2=slice(10+128,118+128),slice(10,118)      
        self.inq3=slice(10+128,118+128),slice(10+128,118+128)
        self.inq4=slice(10,118),slice(10+128,118+128)

        self.q1=slice(0,128),slice(0,128)
        self.q2=slice(128,256),slice(0,128)
        self.q3=slice(128,256),slice(128,256)
        self.q4=slice(0,128),slice(128,256)


        print self.nickname, ": using DQ extension for badpix"
        try:
            self.dq=f['dq',1].data
            if self.dq is not None:
                dqmask = 1+16+32+64+128+256 #selected values
                self.nonsourcemask=numarray.bitwise_and(self.dq, dqmask+1024) #exclude sources
                self.nonsourceidx=numarray.where(self.nonsourcemask == 0)
                self.nonsource=self.data[self.nonsourceidx]
                self.badpix=numarray.bitwise_and(self.dq, dqmask)
            else:
                self.badpix=None
        except KeyError,e:
            print e
            print 'DQ extension not found for %s'%imgfile
            print 'defaulting to maskfile'
            self.badpix = None

 #       print "self.badpix.shape = ",self.badpix.shape
            
        if self.badpix is None:
            print "failing over to ",self.badfile
            try:
                f2=pyfits.open(self.badfile)
                self.badpix=f2['dq',1].data
                f2.close()
            except IOError,e:
                print e
                print "Bad pixel image not read"
                print "Bad pixel image filename obtained from ",self.filename
                self.badpix=None

    def writeto(self,outname,clobber=False):
        f=pyfits.open(self.filename)
        f[self.extnum].data=self.data
        f[0].header=self.h #update the primary header
        f.writeto(outname,clobber=clobber)

    def dark_subtract(self,dark):
        self.data=(self.data-dark)/self.exptime

        
    def pedskyish(self):
        """ Performs something like the IRAF pedsky task, but with a bit more
        sophistication in handling the central row and column"""
        
        #Compute the median for each quadrant independently
        m=numarray.array([imstat(self.data[self.inq1],nclip=1,binwidth=0.01,fields='median').median,
                          imstat(self.data[self.inq2],nclip=1,binwidth=0.01,fields='median').median,
                          imstat(self.data[self.inq3],nclip=1,binwidth=0.01,fields='median').median,
                          imstat(self.data[self.inq4],nclip=1,binwidth=0.01,fields='median').median])
        #print "file ",self.filename
        #print "raw m",m
        temp=imstat(m,nclip=1,binwidth=0.01,fields='median')
        #print "stats: mean/median/mode ",temp.mean,temp.median,temp.mode
        m=m-temp.median
        #print "after sub",m        

        #Subtract the median from each quadrant
        self.data[self.q1]=self.data[self.q1]-m[0]
        self.data[self.q2]=self.data[self.q2]-m[1]
        self.data[self.q3]=self.data[self.q3]-m[2]
        self.data[self.q4]=self.data[self.q4]-m[3]


        #"special handling of middle col/row"
        if self.camera < 3:
            temp=imstat( (self.data[:,127]-self.data[:,126]),
                         nclip=1,binwidth=0.01,fields='median')
            self.data[:,127]=self.data[:,127]-temp.median
        elif self.camera==3:
            temp=imstat( (self.data[127,:]-self.data[126,:]),
                         nclip=1,binwidth=0.01,fields='median')
            self.data[127,:]=self.data[127,:]-temp.median
        else:
            raise ValueError, "Bad camera value"

##...................................................................................
## Original code that I think is wrong:
##        transcribed parens from idl code incorrectly
##...................................................................................
##         #Camera 3 is special: treat its middle column in a similar way
##         if self.camera < 3:
##             temp=imstat(self.data[:,127],nclip=1,binwidth=0.01,fields='median')
##         #    print "line 127 median is ",temp.median
##         #    print "line 127 mean is ",self.data[:,127].mean()
##             self.data[:,127]=self.data[:,127]-temp.median-self.data[:,126]
##         elif self.camera==3:
##             temp=imstat(self.data[127,:],nclip=1,binwidth=0.01,fields='median')
##             self.data[127,:]=self.data[127,:]-temp.median-self.data[126,:]
##         else:
##             raise ValueError, "Bad camera value"

        
    def getmask(self,dim=256,border=3,writename='mask.dat',clobber=False):
        """Computes a mask to use for pixels to omit"""
        mask=numarray.zeros((dim,dim),'Float32')
        badmask=numarray.ones((dim,dim),'Float32')
        if self.badpix is not None:
            u=numarray.where(self.badpix != 0)
            mask[u]=1
            badmask[u]=0
        # Always Mask out central "cross" chipgap
        mask[(dim/2)-1,:]=1
        mask[:,(dim/2)-1]=1
        # and the very edges
        mask[0:16,:]=1    #apparently the bottom edge is different
        mask[dim-border:dim,:]=1
        mask[:,0:border+1]=1
        mask[:,dim-border:dim]=1

        if writename:
            writeimage(mask,writename,clobber=clobber)
        return mask,badmask

    def apply_mask(self,mask):
        goodpix=numarray.where(mask == 0)
        self.masked_data = self.data[goodpix]
        

    def getscales(self,saaper,mask,pars):
        
        cal=self.data*self.exptime
        acc=saaper*self.exptime

        for dom in self.domains.values():
            sz1=int(dom.range/pars.stepsize)+1
            stepval=[pars.stepsize*i for i in xrange(sz1)]

            #there's got to be a better way to do this!
            #Make a mask & fill it all with ones
            fitmask=numarray.ones(mask.shape)
            #Then make the pixels we want be set to zero
            fitmask[dom.pixlist]=0
            #Then set the mask-defined bad pixels to one so we don't use them
            #(Notice there's no use of "self.badpix" here, wonder why not?)
            #Ah! It's because self.badpix was already used in *making* that mask. OK.
            badpix=numarray.where(mask == 1)
            fitmask[badpix]=1
            #Finally, choose only those pixels where it's set to zero.
            umask=numarray.where(fitmask == 0)

            dom.pp=numarray.zeros((3,int(dom.range/pars.stepsize)+1),'Float32')
            index=0
            for i in stepval:
                dif=cal-(acc*i)
                temp=imstat(dif[umask],binwidth=0.01,nclip=3,fields='stddev,mode') #sigma=100
                dom.pp[:,index]=i,temp.stddev,temp.mode
                index+=1
            dom.striplowerthan(pars.binsigfrac)
            if pars.diagfile:
                dom.writeto(pars.diagfile+'_'+dom.name+'_signal_domain.dat',clobber=pars.clobber)
            ubest,umode=dom.getmin()
            best=dom.pp[0,ubest]

            print "\nResults summary for %s domain:"%dom.name
            if pars.dofit:
                minx=max(ubest-5,0)
                maxx=min(ubest+5,len(dom.pp[0])-1)
                thedata=[(dom.pp[0,i],dom.pp[1,i]) for i in range(minx,maxx+1)]

                best,dom.chi2=parabola_min(thedata,best)
               # best=parabola1(dom.pp[0,minx:maxx],pp[1,minx:maxx],minguess=best)
               # best=parabola1(dom.pp[0,minx:maxx],pp[1,minx:maxx],minguess=best)

            dom.nr=(1.0-dom.pp[1,ubest]/dom.pp[1,0])*100
            dom.scale=best
            dom.bestloc=ubest


            #print "   zero-mode scale factor is       : ",dom.pp[0,umode]
            print "   min-noise (best) scale factor is: ",dom.scale
            print "   effective noise at this factor (electrons at gain %f): %f"%(self.gainplot,dom.pp[1,ubest]*self.gainplot)
            print "   noise reduction (percent)       : ",dom.nr


    def apply_domains(self,saaper,badmask,noisethresh,appimage=None):
        if appimage is not None: 
            final=appimage
        else:
            final=self.data.copy()
        hdom,ldom=self.domains['high'],self.domains['low']
        self.update=1
        if hdom.nr >= noisethresh and ldom.nr >= noisethresh:
            print "\n Applying noise reduction in both domains "
            self.appstring='both'
            final[ldom.pixlist]= self.data[ldom.pixlist]-(saaper[ldom.pixlist]*ldom.scale*badmask[ldom.pixlist])
            final[hdom.pixlist]= self.data[hdom.pixlist]-(saaper[hdom.pixlist]*hdom.scale*badmask[hdom.pixlist])
        elif hdom.nr > noisethresh and ldom.nr < noisethresh:
            print "\n Applying noise reduction in high domain only "            
            self.appstring='high only'
            final[hdom.pixlist]= self.data[hdom.pixlist]-(saaper[hdom.pixlist]*hdom.scale*badmask[hdom.pixlist])
            
        elif hdom.nr < noisethresh and ldom.nr >= noisethresh:
            print "\n...Noise reduction in high domain < 1%: applying low scale everywhere"
            self.appstring='low everywhere'
            final=self.data-(saaper*ldom.scale*badmask)
        elif hdom.nr < noisethresh and ldom.nr < noisethresh:
            print "\n*** Noise reduction < 1 %, not applying"
            self.appstring='none'
            self.update=0
        else:
            raise ValueError,"Huh?? hi_nr, lo_nr: %f %f"%(hdom.nr,ldom.nr) 
            
        return final

    def update_header(self,pars,tag='default'):
        """ Update the FITS header with all this good stuff we've done"""

        #Start with the last keyword, for ease of applying.
        
        #Describe what was applied
        lastkey='SCNAPPLD'
        self.h.update(lastkey,
                      self.appstring,
                      'to which domains was SAA cleaning applied',
                      after='SAACRMAP')

        
        #Then work forward from the beginning of the section:
        
        #First put in a comment card as a separator
        self.h.add_blank('',before=lastkey)
        self.h.add_blank('      / SAA_CLEAN output keywords',before=lastkey)
        self.h.add_blank('',before=lastkey)
        
        #Then describe the persistence image:
        self.h.update('SAAPERS',
                      pars.saaperfile,
                      'SAA persistence image',
                      before=lastkey)
        if not pars.readsaaper:
            self.h.update('SCNPSCL',
                          pars.scale,
                          'scale factor used to construct persistence img',
                          before=lastkey)
            self.h.update('SCNPMDN',
                          pars.saaper_median,
                          'median used in flatfielding persistence image',
                          before=lastkey)
        self.h.add_blank('',before=lastkey)
        
        #Describe the domains
        self.h.update('SCNTHRSH',
                      self.thresh,
                      'Threshold dividing high & low signal domains',
                      before=lastkey)
        self.h.update('SCNHNPIX',
                      self.domains['high'].npix,
                      'Number of pixels in high signal domain (HSD)',
                      before=lastkey)
        self.h.update('SCNLNPIX',
                      self.domains['low'].npix,
                      'Number of pixels in low signal domain (LSD)',
                      before=lastkey)
        self.h.add_blank('',before=lastkey)
        
        #Describe the results in each domain
##         self.h.update('SCNGAIN',
##                       self.gainplot,
##                       'gain used for effective noise calculations',
##                       before=lastkey)
        for k in self.domains:
            HorL=k[0].upper()
            self.h.update('SCN%sCHI2'%HorL,
                          self.domains[k].chi2,
                          '%sSD chi squared for parabola fit'%HorL,
                     before=lastkey)
            self.h.update('SCN%sSCL'%HorL,
                          self.domains[k].scale,
                          '%sSD scale factor for min noise'%HorL,
                      before=lastkey)
            bestloc=self.domains[k].bestloc
            self.h.update('SCN%sEFFN'%HorL,
                          self.domains[k].pp[1,bestloc]*self.gainplot,
                          '%sSD effective noise at SCNGAIN'%HorL,
                          before=lastkey)
            self.h.update('SCN%sNRED'%HorL,
                          self.domains[k].nr,
                          '%sSD  noise reduction (percent)'%HorL,
                          before=lastkey)
##................................................................       
## Only needed for testing: removed for release
##................................................................       
##             self.h.update('SCNTAG',
##                           tag,
##                           'Tag/description of this version',
##                           
##................................................................       
        
#..........................................................................
# Exception definitions
class NoPersistError(exceptions.Exception):
    pass
class BadThreshError(exceptions.Exception):
    pass
class NegScaleError(exceptions.Exception):
    pass
class InsuffImprovement(exceptions.Exception):
    pass
#..........................................................................
#Helper functions:
#-............................................................................
def osfn(filename):
    """Return a filename with iraf syntax and os environment names substituted out"""
    if filename is None:
        return filename
    
    #Baby assumptions: suppose that the env variables will be in front.
   
    if filename.startswith('$'):  #we need to translate a logical
        symbol,rest=filename.split('/',1)
    elif '$' in filename: #we need to fix the iraf syntax
        symbol,rest=filename.split('$',1)
    else:
        return filename
    newfilename=os.environ[symbol]+'/'+rest    
    return newfilename

def writeimage(image, filename, comment=None,clobber=False):
  hdulist=pyfits.HDUList()
  hdu=pyfits.PrimaryHDU()
  hdu.data=image
  if (comment is not None):
    hdu.header.add_comment(comment)
  hdulist.append(hdu)
  hdulist.writeto(filename,clobber=clobber)

#..........................................................................
# Math functions
def parabola_model(coeffs,t):
    r=coeffs[0]*(t-coeffs[1])**2 + coeffs[2]
    return r

def parabola_min(thedata, startguess):
    #We may not need to rescale the data
    guesscoeff=(100,startguess,0.1)
    fitcoeff,chi2=LeastSquares.leastSquaresFit(parabola_model,guesscoeff,thedata)
    print "chi2 for parabola fit = ",chi2
    return fitcoeff[1],chi2
#..................................................................................
#Not tested or used anywhere yet
def gausspoly_model(coeffs,t):
    import math
    z=(t-coeffs[1])/coeffs[2]
    r=coeffs[0]*math.exp(-(z**2)/2) + coeffs[3] + coeffs[4]*t + coeffs[5]*t**2
    return r

def gaussfit(thedata,startguess):
    guesscoeff=(100,startguess,0.1) #probably wrong
    fitcoeff,chi2=LeastSquares.leastSquaresFit(gausspoly_model,guesscoeff,thedata)
    return r
#..............................................................................
# General functions
#..........................................................................
def get_postsaa_darks(imgfile):
    """ Return the filenames containing the post-saa dark exposures, if
    present. Otherwise raise an exception and exit. """

    #Get the science header
    inpath=os.path.dirname(osfn(imgfile))
    if inpath != '':
        inpath+= '/'
    f=pyfits.open(imgfile)
    h=f[0].header
    saa_asn=h['saa_dark']
    f.close()
    if saa_asn == 'N/A':
        raise NoPersistError, """This data was not taken in an SAA-impacted orbit.
        No correction needed. Exiting."""
    else:
        #Get the files out of that set
        saa_files=[]
        f2=pyfits.open(inpath+saa_asn.lower()+'_asn.fits')
        for i in [0,1]:
            name=f2[1].data[i]
            saa_files.append(inpath+name.field(0).lower()+'_raw.fits')
        f2.close()
        return saa_files

def getdark(camera,darkpath):
    """ Get the right dark file for a given NICMOS camera.
    This is definitely not the right way to do this."""
    dfile={1:'c1_saadarkref_drk.fits',
           2:'c2_saadarkref_drk.fits',
           3:'c3_saadarkref_drk.fits'}
    thefile=os.path.abspath(darkpath)+'/'+dfile[camera]
    f=pyfits.open(thefile)
    ans= f[1].data
    f.close()
    return ans

def make_saaper(im1,im2,dark,pars,crthresh=1):
    #Process the data
    for im in [im1,im2]:
        im.dark_subtract(dark)
        im.pedskyish()
    #Combine the data
    saaper=((im1.data*pars.wf1) + (im2.data/pars.scale)*pars.wf2)
    #Correct for CRs
    if pars.crthresh:
        a=im1.data-(im2.data/pars.scale)
        u1=numarray.where(a > pars.crthresh)
        saaper[u1]=im2.data[u1]/pars.scale

        a=(im2.data/pars.scale) - im1.data
        u2=numarray.where(a > pars.crthresh)
        saaper[u2]=im1.data[u2]
    if pars.writesaaper and pars.saaperfile:
        writeimage(saaper,pars.saaperfile,clobber=pars.clobber)        
    return saaper


def get_dark_data(imgfile,darkpath):
    saafiles=get_postsaa_darks(imgfile)
    im1=Exposure(saafiles[0],nickname='saa_im1')
    im2=Exposure(saafiles[1],nickname='saa_im2')
    dark=getdark(im1.camera,darkpath)
    return im1,im2,dark

def flat_saaper(saaper,img):
    mm=imstat(saaper,nclip=1,binwidth=0.01,fields='median').median
    #Use median, or mode? which is better?
    if img.h['flatdone'] == 'PERFORMED':
        flatname=osfn(img.h['flatfile'])
##         if flatname.startswith('nref$'):
##             prefix,root=flatname.split('$',1)
##             flatname=iraf.osfn(prefix+'$')+root
        flat=Exposure(flatname,nickname='flatfile')

        print "median used in flatfielding: ",mm
        saaper=((saaper-mm)*flat.data) + mm
    return saaper,mm

        


#....................................................................
# The "main" program
#....................................................................
def clean(usr_calcfile,usr_targfile,usr_outfile,pars=None):
    print "saaclean version %s"%__version__
    print "Input files: %s %s"%(usr_calcfile,usr_targfile)
    imgfile=osfn(usr_calcfile)
    targfile=osfn(usr_targfile)
    if imgfile != targfile:
        #then we'll need the data from the targfile
        tfile=pyfits.open(targfile)
        appimage=tfile['sci'].data.copy()
    else:
        #we'll apply it to the same file
        appimage=None
        
    outfile=osfn(usr_outfile)
    if pars is None:
        pars=params()
    if pars.readsaaper:
        saaper=pyfits.open(pars.saaperfile)[0].data
    else:
        im1,im2,dark=get_dark_data(imgfile,pars.darkpath)
        saaper=make_saaper(im1,im2,dark,pars)
        print "Using scale factor of ",pars.scale," to construct persistence image"

    img=Exposure(imgfile,nickname='sci image')
    mask,badmask=img.getmask(writename=pars.maskfile,clobber=pars.clobber)
    saaper,mm=flat_saaper(saaper,img)
    pars.saaper_median=mm
    
    if pars.flatsaaperfile:
        writeimage(saaper,pars.flatsaaperfile,clobber=pars.clobber)

    mask,badmask=img.getmask(writename=None)
    img.apply_mask(mask)

    if pars.thresh is None:
        #Define threshold *on persistence image* as
        # (mean + 3.5*stddev)
        saaperstat = imstat(saaper,binwidth=0.01,nclip=10,fields='stddev,mean')
        img.thresh=saaperstat.mean + 3.5*saaperstat.stddev  
    else:
        img.thresh=pars.thresh

    #Apply threshold *to persistence image*
    img.domains={'high':Domain('high',
                               numarray.where(saaper > img.thresh),
                               pars.hirange),
                 'low' :Domain('low',
                               numarray.where(saaper <= img.thresh),
                               pars.lorange)
                 }

    print "Threshold for hi/lo: ",img.thresh
    print "Npixels hi/lo: ",len(img.domains['high'].pixlist[0]),len(img.domains['low'].pixlist[0])
    if (img.domains['high'].npix == 0):
        raise BadThreshError,"Zero pixels found in high signal domain"
    img.getscales(saaper,mask,pars)

    final=img.apply_domains(saaper,badmask,pars.noisethresh,appimage=appimage)

  
    if 1: #img.update:
        img.data=final
        img.update_header(pars,tag='%s: addmean'%__version__)
        img.writeto(outfile,clobber=pars.clobber)

    return saaper,img
