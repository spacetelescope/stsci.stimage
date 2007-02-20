import numpy as N
import ndimage as ND


def gauss(sigma,dist):
    return N.exp(-0.5 * N.power((dist/sigma),2))

def makegauss(shape,sigma,norm):
    m = N.zeros(shape,dtype=N.float32)
    for i in range(shape[0]):
        for j in range(shape[1]):
            dist = N.sqrt(N.power(j-(shape[1]/2),2) + N.power(i-(shape[0]/2),2))
            m[i,j] = norm * gauss(sigma,dist)
    return m

"""
All algorithms in this modules based on Soille (2002) 
come from:
  P. Soille, 2002, "Morphological Image Analaysis: 
    Principles and Applications (2nd Edition)" (Springer-Verlag:Berlin)
    ISBN 3540429883
"""

def point_minimum(array, mask):
    """ Return a point-wise minimum array based 
        on the mask image. """
    return N.minimum(array,mask)

def point_maximum(array, mask):
    """ Return a point-wise maximum array based 
        on the mask image. """

    return N.maximum(array, mask)

def geodesic_erosion1d(marker,mask):
    """ Perform Geodesic erosion (in 1D) based on
        algorithm on p. 168, P. Soille, 2002. """ 

    marker_erosion = ND.minimum_filter1d(marker,2)
    return N.where(mask >= marker_erosion, mask, marker_erosion)

def geodesic_erosion(marker,mask):
    """ Perform Geodesic erosion based on
        algorithm on p. 168, P. Soille, 2002. 
    """     
    
    # Compute the size for the filter based on the shape
    # of the marker array
    mshape = len(marker.shape)
    msize = (3,) * mshape
        
    marker_erosion = ND.minimum_filter(marker,size=msize)
    return N.where(mask >= marker_erosion, mask, marker_erosion)
    
def geodesic_dilation(marker,mask):
    """ Perform geodesic dilation based on algorithm
        on p. 185 of Soille (2002).
    
    """
    # Compute the size for the filter based on the shape
    # of the marker array
    mshape = len(marker.shape)
    msize = (3,) * mshape
    
    marker_dilation = ND.maximum_filter(marker, size=msize)
    return N.where(mask <= marker_dilation, mask, marker_dilation)

def opening_by_recon(image,size=3):
    """ Implements opening by reconstruction of size n of image 
        as defined on p. 210, P. Soille, 2002.
        
        opening = recon_by_dilation (erosion(image,size),image)
    """
    _size = (size,)*len(image.shape)
    return recon_by_dilation(ND.grey_erosion(image,_size),image)

def closing_by_recon(image,size=3):
    """ Implements closing by reconstruction of size n of image 
        as defined on p. 211, P. Soille, 2002.
        
        closing = recon_by_erosion (dilation(image,size),image)
    """
    _size = (size,)*len(image.shape)
    return recon_by_erosion(ND.grey_dilation(image,_size),image)

    

def recon_by_erosion(marker,mask,max_iter=10):
    """ Iterate over geodesic erosion operations
        until erosion(j+1) = erosion(j).
        Perform Geodesic erosion based on
        algorithm on p. 191-192, P. Soille, 2002. 
        This can be referenced as:
          R^epsilon_mask(marker)
    """
    
    erosion_i = marker
    for i in xrange(max_iter):
        erosion_i1 = geodesic_erosion(erosion_i,mask)
        if i>0 and N.sum(N.equal(erosion_i.ravel(),erosion_i1.ravel())) == erosion_i.size:
            break     
        else:
            erosion_i = erosion_i1.copy()
    del erosion_i1
    return erosion_i

def recon_by_dilation(marker,mask,max_iter=10):
    """ Iterate over geodesic dilation operations
        until dilation(j+1) = dilation(j).
        Perform Geodesic dilation based on
        algorithm on p. 190-191, P. Soille, 2002. 
        This can be referenced as:
          R^delta_mask(marker)
    """
    
    dilation_i = marker
    for i in xrange(max_iter):
        dilation_i1 = geodesic_dilation(dilation_i,mask)
        if i > 0 and N.sum(N.equal(dilation_i.ravel(),dilation_i1.ravel())) == dilation_i.size:
            break     
        else:
            dilation_i = dilation_i1.copy()
    del dilation_i1
    return dilation_i
         
def transform_concave(marker,interval=10):
    """ Perform h-concave transformation using algorithm
        specified on pg. 203 of Soille (2002).
        The algorithm can be stated as:
        
            HMIN_h(f) = recon_by_erosion(f+h,f)  
            HCONCAVE_h(f) = HMIN_h(f) - f
        
        where f = marker, h = interval.        
    """
    hmin = recon_by_erosion(marker+interval,marker)
    return hmin - marker

def transform_convex(marker, interval=10):
    """ Perform h-convex transformation using algorithm
        specified on pg. 203 of Soille (2002).
        The algorithm can be stated as:
        
            HMAX_h(f) = recon_by_dilation(f+h,f)  
            HCONVEX_h(f) = f - HMAX_h(f)
        
        where f = marker, h = interval.        
    """
    hmax = recon_by_dilation(marker+interval,marker)
    return marker - hmax
    

def grey_chm_transform(array, fg, bg, origin=0):
    """
    Perform Grey-scale constrained-hit-or-miss transform
       based on algorithm from pg. 145, Soille (2002).
       Example based on Fig 5.4 (pg 144, Soille 2002).

    Input:
       array: gray-scale array (1-D or 2-D)
       fg   : foreground mask array (UInt8 only)
       bg   : background mask array (UInt8 only)

     Example:
       f = N.array([3, 3, 3, 0, 1, 6, 2, 1, 7, 5, 1, 0, 4, 6, 7, 3, 3, 3], 
                   type=Int8)
       B = N.array([0,0,1,1],N.UInt8)
       Bc = N.array([1,1,0,0,1,1],N.UInt8)
       chm = morph.grey_chm_transform(f,B,Bc)
       print chm
       array([0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 2, 0, 0, 0, 0], type=Int8)
    """
    
    #Determine which mode to use based on where the origin is located
    #
    # if origin == 0, then it is centered and in the foreground
    if origin == 0:
        # We only need to compute erosion_fg(f) and dilation_bg(f)
        # then perform comparisons
        f = ND.grey_erosion(array,footprint = fg)
        df = ND.grey_dilation(array,footprint = bg)
    else:
        f = ND.grey_erosion(array,footprint = bg)
        df = ND.grey_dilation(array, footprint= fg)
        
    return N.maximum(f - df, 0) 
    
def grey_uhm_transform(array,fg,bg):
    """
    Perform grey-scale unconstrained hit-or-miss transform
    based on the algorithm from p 143 of Soille 2002.

    Input:
       array: gray-scale array (1-D or 2-D)
       fg   : foreground mask array (UInt8 only)
       bg   : background mask array (UInt8 only)

     Example:
       f = N.array([3, 3, 3, 0, 1, 6, 2, 1, 7, 5, 1, 0, 4, 6, 7, 3, 3, 3], 
                   type=Int8)
       B = N.array([0,0,1,1],N.UInt8)
       Bc = N.array([1,1,0,0,1,1],N.UInt8)
       chm = morph.grey_uchm_transform(f,B,Bc)
       print chm
       array([0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 2, 0, 0, 0, 0], type=Int8)
    """   
 
    f = ND.grey_erosion(array,footprint = fg)
    df = ND.grey_dilation(array,footprint = bg)
    return N.maximum(f-df, 0)
#
# =====================================================================
#
#   limits (copied from 'PyMorph')
#
# =====================================================================
def limits(f):
    """
        - Purpose
            Get the possible minimum and maximum of an image.
        - Synopsis
            y = limits(f)
        - Input
            f: Unsigned gray-scale (uint8 or uint16), signed (int32) or
               binary image.
        - Output
            y: Vector, the first element is the infimum, the second, the
               supremum.
        - Description
            The possible minimum and the possible maximum of an image depend
            on its data type. These values are important to compute many
            morphological operators (for instance, negate of an image). The
            output is a vector, where the first element is the possible
            minimum and the second, the possible maximum.
        - Examples
            #
            print limits(mmbinary([0, 1, 0]))
            print limits(uint8([0, 1, 2]))
    """
    from numpy import array

    code = f.dtype.name
    if   code == 'bool_': y=array([0,1],dtype=numpy.bool_)
    elif code == 'uint8': y=array([0,255],dtype=numpy.uint8) # UInt8
    elif code == 'uint16': y=array([0,65535],dtype=numpy.uint16) #UInt16
    elif code == 'int16': y =array([-32767,32767],dtype=numpy.int16) #Int16
    elif code == 'int32': y=array([-2147483647,2147483647],dtype=numpy.int32) #Int32
    else:
        assert 0,'Does not accept this typecode:'+code
    return y
#
# =====================================================================
#
#   mmneg
#
# =====================================================================
def numneg(f):
    """
        - Purpose
            Negate an image.
        - Synopsis
            y = numneg(f)
        - Input
            f: Unsigned gray-scale (uint8 or uint16), signed (int32) or
               binary image.
        - Output
            y: Unsigned gray-scale (uint8 or uint16), signed (int32) or
               binary image.
        - Description
            mmneg returns an image y that is the negation (i.e., inverse or
            involution) of the image f . In the binary case, y is the
            complement of f .
        - Examples
            #
            #   example 1
            #
            f=uint8([255, 255, 0, 10, 20, 10, 0, 255, 255])
            print numneg(f)
            print numneg(uint8([0, 1]))
            print numneg(int32([0, 1]))
            #
            #   example 2
            #
            a = mmreadgray('gear.tif')
            b = nummneg(a)
            mmshow(a)
            mmshow(b)
            #
            #   example 3
            #
            c = mmreadgray('astablet.tif')
            d = nummneg(c)
            mmshow(c)
            mmshow(d)
    """
    
    y = limits(f)[0] + limits(f)[1] - f
    y = y.astype(f.dtype)
    return y
    
#
# =====================================================================
#
#   closehole - as translated/copied from 'PyMorph'
#
# =====================================================================
def closehole(f):
    """
        - Purpose
            Close holes of binary and gray-scale images.
        - Synopsis
            y = closehole(f, Bc=None)
        - Input
            f:  Gray-scale (uint8 or uint16) or binary image.
            Bc: Structuring Element Default: None (3x3 elementary cross). (
                connectivity).
        - Output
            y: (same datatype of f ).
        - Description
            mmclohole creates the image y by closing the holes of the image
            f , according with the connectivity defined by the structuring
            element Bc .The images can be either binary or gray-scale.
        - Examples
            #
            #   example 1
            #
            a = mmreadgray('pcb1bin.tif')
            b = closehole(a)
            mmshow(a)
            mmshow(b)
            #
            #   example 2
            #
            a = mmreadgray('boxdrill-B.tif')
            b = closehole(a)
            mmshow(a)
            mmshow(b)
    """

    #if Bc is None: Bc = mmsecross()
    #if Bc is None: Bc = ND.generate_binary_structure(2,1)
    
    #delta_f = mmframe(f)
    delta_f = N.ones(f.shape,N.UInt8)
    delta_f[1:-1,1:-1] = 0
    #y = mmneg( mminfrec( delta_f, mmneg(f), Bc))
    # DOES NOT WORK YET!!! 30-Aug-2005 WJH
    y = numneg(recon_by_dilation(delta_f,f,max_iter=10))
    
    return y
