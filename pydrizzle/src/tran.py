"""

TRAN - a wrapper for traxy/tranback to transform a position (x,y) in pixels
       on an input, distorted image to/from an output image.

There are two methods: tran.f for forward transforms and tran.b for
the reverse.

The syntax is:

 tran.f(original_image,drizzled_image,x,y)
  --or--
 tran.f(original_image,drizzled_image,List='list')

and

 tran.b(drizzled_image,original_image,x,y)
  --or--
 tran.b(drizzled_image,original_image,List='list')

In the 'list' case the list is a normal text file with two, free-
format columns of X,Y pixel positions.

All the information is extracted from the header by searching for
the drizzle header records. The coefficients file must be present and
have the same name. This is in "drizzle" format, as produced by PyDrizzle,
and not the IDCTAB.

Note - the 'original_image' name must match the string written to
the header by drizzle, exactly.

It is assumed that this script is invoked from Pyraf and that the
traxy and tranback IRAF tasks are available. They are in the dither
package of STSDAS 3.1.

Example:

--> import tran

--forwards---

--> tran.f('j8c0c1011_crj.fits[sci,1]','f606w_z61c1_drz.fits[1]',136.109,371.455)
Running Tran Version  0.11 (May 2004)
-Reading drizzle keywords from header...
-Image  j8c0c1011_crj.fits[sci,1]  was # 3
-to be drizzled onto output  f606w_z61c1_drz.fits[1]
-Transforming position...
 Xin,Yin:    136.109   371.455 Xout,Yout:    123.000   432.000

--backwards---

Running Tran Version  0.11 (May 2004)
--> tran.b('f606w_z61c1_drz.fits[1]','j8c0c1011_crj.fits[sci,1]',123,432)
-Reading drizzle keywords from header...
-Image  j8c0c1011_crj.fits[sci,1]  was # 3
-to be drizzled onto output  f606w_z61c1_drz.fits[1]
-Transforming position...
 Xin,Yin:    136.109   371.455 Xout,Yout:    123.000   432.000

Richard Hook, ST-ECF/STScI, April 2003
Added "List" feature and other small improvements, November 2003

Added trap for error messages in "list" form, May 2004.

Comments: rhook@eso.org

"""

from string import *
from math import *
import iraf
import sys

# Some convenient definitions
yes=iraf.yes
no=iraf.no
MaxImages=999
True=1
False=0

__version__ = '0.11 (May 2004)'

# A class for drizzle geometrical parameters
class DrizGeoPars:

    # Constructor, set to drizzle default values
    def __init__(self,image=None,inimage=None):

        if image == None:
            self.scale=1.0
            self.coeffs=None
            self.lam=555.0
            self.xsh=0.0
            self.ysh=0.0
            self.rot=0.0
            self.shft_un="input"
            self.shft_fr="input"
            self.align="center"
            self.d2xscale=0.0
            self.d2yscale=0.0
            self.d2xsh=0.0
            self.d2ysh=0.0
            self.d2rot=0.0
            self.d2shft_fr="output"
        else:

            # Read geometric parameters from a header using an image name as
            # the key

            found=False

            # First search for the entry for this image
            i=1
            while i < MaxImages:
                datkey = 'D%3iDATA' % i
                datkey=datkey.replace(' ','0')

                iraf.keypar(image,datkey,silent='yes')

                # If we can't read this no point considering
                if iraf.keypar.value == '':
                    break

                # If we have a match set flag and leave
                if iraf.keypar.value == inimage:
                    found=True
                    break

                i += 1

            if found:
                print "-Reading drizzle keywords from header..."
                print "-Image ",inimage," was #",i
                print "-to be drizzled onto output ",image
            else:
                raise "Failed to get keyword information from header"

            # Now we know that the selected image is present we can
            # get all the other parameters - we don't check whether this
            # succeeds, if it doesn't let it crash
            stem=datkey[:4]

            iraf.keypar(image,stem+"SCAL",silent='yes')
            self.scale=float(iraf.keypar.value)

            iraf.keypar(image,stem+"COEF",silent='yes')
            self.coeffs=iraf.keypar.value

            iraf.keypar(image,stem+"LAM",silent='yes')
            self.lam=float(iraf.keypar.value)

            iraf.keypar(image,stem+"XSH",silent='yes')
            self.xsh=float(iraf.keypar.value)

            iraf.keypar(image,stem+"YSH",silent='yes')
            self.ysh=float(iraf.keypar.value)

            iraf.keypar(image,stem+"ROT",silent='yes')
            self.rot=float(iraf.keypar.value)

            iraf.keypar(image,stem+"SFTU",silent='yes')
            self.shft_un=iraf.keypar.value

            iraf.keypar(image,stem+"SFTF",silent='yes')
            self.shft_fr=iraf.keypar.value

            # The case of the "align" parameter is more tricky, we
            # have to deduce it from INXC keyword
            iraf.keypar(image,stem+"INXC",silent='yes')
            inxc=float(iraf.keypar.value)

            # Need the X and Y dimensions as well - both input and
            # output
            iraf.keypar(inimage,'i_naxis1',silent='yes')
            xdim=int(iraf.keypar.value)
            iraf.keypar(inimage,'i_naxis2',silent='yes')
            ydim=int(iraf.keypar.value)

            self.nxin=xdim
            self.nyin=ydim

            iraf.keypar(image,'i_naxis1',silent='yes')
            xdim=int(iraf.keypar.value)
            iraf.keypar(image,'i_naxis2',silent='yes')
            ydim=int(iraf.keypar.value)

            self.nxout=xdim
            self.nyout=ydim

            if abs(inxc-float(xdim/2)-0.5) < 1e-4:
                self.align='corner'
            else:
                self.align='center'

            # Check for the presence of secondary parameters
            iraf.keypar(image,stem+"SECP",silent='yes')
            if iraf.keypar.value == "yes":
                raise "! Sorry, this version does NOT support secondary parameters"
            else:
                self.secp=False

# Main TRAN methods - f for forward and b for back
#
# inimage - the input image which is to have its WCS updated
# drizimage - the reference image, assumed to contain the drizzle parameters
#         in its header
#
# x,y - a single position for transformation
#
# List - a text file name containing x y pairs
#
def f(origimage,drizimage,x=None,y=None,List=None):

    # Get the parameters from the header
    GeoPar=DrizGeoPars(drizimage,origimage)

    # Use traxy, along with all the parameters specified above, to
    # transform to the output image
    iraf.traxy.nxin=GeoPar.nxin
    iraf.traxy.nyin=GeoPar.nyin
    iraf.traxy.nxout=GeoPar.nxout
    iraf.traxy.nyout=GeoPar.nyout
    iraf.traxy.scale=GeoPar.scale
    iraf.traxy.xsh=GeoPar.xsh
    iraf.traxy.ysh=GeoPar.ysh
    iraf.traxy.rot=GeoPar.rot
    iraf.traxy.coeffs=GeoPar.coeffs
    iraf.traxy.shft_un=GeoPar.shft_un
    iraf.traxy.shft_fr=GeoPar.shft_fr
    iraf.traxy.align=GeoPar.align
    iraf.traxy.lam=GeoPar.lam

    if List != None:
        f=open(List)
        lines=f.readlines()

        print "   Xin         Yin         Xout        Yout"

        for line in lines:
            x=float(split(line)[0])
            y=float(split(line)[1])
            str=iraf.traxy(x,y,mode='h',Stdout=1)

            # Just show the lines of interest
            for line in str:
                if line[0:1] == '!':
                    print line
                    sys.exit()

                if line[0:3] == ' Xi':
                    xin = float(split(line)[1])
                    yin = float(split(line)[2])
                    xout = float(split(line)[4])
                    yout = float(split(line)[5])
                    print "%10.3f %10.3f %10.3f %10.3f" % (xin,yin,xout,yout)

    else:

        # Transform and display the result
        print "-Transforming position..."
        str=iraf.traxy(x,y,mode='h',Stdout=1)

        # Just show the lines of interest
        for line in str:

            if line[0:1] == '!':
                print line

            if line[0:3] == ' Xi':
                print line

def b(drizimage,origimage,x=None,y=None,List=None):

    # Announce version
    print "+ TRAN.B V0.2 (November 2003)"

    # Get the parameters from the header
    GeoPar=DrizGeoPars(drizimage,origimage)

    # Use tranback, along with all the parameters specified above, to
    # transform to the output image
    iraf.tranback.nxin=GeoPar.nxin
    iraf.tranback.nyin=GeoPar.nyin
    iraf.tranback.nxout=GeoPar.nxout
    iraf.tranback.nyout=GeoPar.nyout
    iraf.tranback.scale=GeoPar.scale
    iraf.tranback.xsh=GeoPar.xsh
    iraf.tranback.ysh=GeoPar.ysh
    iraf.tranback.rot=GeoPar.rot
    iraf.tranback.coeffs=GeoPar.coeffs
    iraf.tranback.shft_un=GeoPar.shft_un
    iraf.tranback.shft_fr=GeoPar.shft_fr
    iraf.tranback.align=GeoPar.align
    iraf.tranback.lam=GeoPar.lam

    if List != None:
        f=open(List)
        lines=f.readlines()

        print "   Xin         Yin         Xout        Yout"

        for line in lines:
            x=float(split(line)[0])
            y=float(split(line)[1])
            str=iraf.tranback(x,y,mode='h',Stdout=1)

            # Just show the lines of interest
            for line in str:
                if line[0:1] == "!":
                    print line
                    sys.exit()

                if line[0:3] == ' Xi':
                    xin = float(split(line)[1])
                    yin = float(split(line)[2])
                    xout = float(split(line)[4])
                    yout = float(split(line)[5])
                    print "%10.3f %10.3f %10.3f %10.3f" % (xin,yin,xout,yout)

    else:

        # Transform and display the result
        print "-Transforming position..."
        str=iraf.tranback(x,y,mode='h',Stdout=1)

        # Just show the lines of interest
        for line in str:
            if line[0:1] == '!':
                print line
                sys.exit()

            if line[0:3] == ' Xi':
                print line
