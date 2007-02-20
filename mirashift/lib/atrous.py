import numpy as N
import convolve
import ndimage as ND

KERNELS = {'linear': N.array([1., 2., 1.])/4.,
           'spline': N.array([1.,4.,6.,4.,1.])/16.,
           'edge': N.array([0.,1.,4.,6.,4.,1.])/16. }
               
def atrous2d(arr,maxscale=1,kernel='linear'):
    """ 
        Compute a' trous wavelet transforms of 2-d numpy
        object up to the 'scale' specified by the user. 
        
        This method supports 'linear', 'spline', and 'edge' kernels.
        
        Syntax:
            waveplanes,wimage = atrous2d(arr,maxscale=1,kernel='linear')

        Input:
            arr        - numpy array for input image
            maxscale   - number wavelet transformations to apply
            kernel     - kernel for wavelet transformation:
                            'linear'(default),'spline','leftedge','rightedge'
        Output:
            waveplanes - stack of wavelet difference images
            wimage     - wavelet transformed image for scale 'scale'
                    
    """        
    if KERNELS.has_key(kernel): 
        kernel1d = KERNELS[kernel]
        wkernel = N.outerproduct(kernel1d,kernel1d)
    else:
        print 'Kernel: ',kernel,' not supported!'
        raise ValueError
    #
    # Initialize variables necessary for keeping track of the
    # kernel size at each scale
    #
    j = 1
    jkernel = wkernel.copy()
    # Define output array containing all the wavelet planes
    waveplanes = N.zeros((maxscale,arr.shape[0],arr.shape[1]),dtype=arr.dtype)
        
    # Perform convolutions
    c0 = arr
    #cj = arr.copy()
    _fft_mode = 0
    for level in xrange(maxscale):
        cj = convolve.convolve2d(c0,kernel=jkernel,fft=_fft_mode,mode='nearest',cval=0.0)
        waveplanes[level] = c0 - cj
        
        # Expand the kernel
        order = 2**j
        jkernel_shape = (wkernel.shape[0]*order, wkernel.shape[1]*order)
        jkernel = N.zeros(jkernel_shape, dtype=wkernel.dtype)
        jkernel[::order,::order] = wkernel
        if jkernel_shape[0] >= 15 or jkernel_shape[1] >= 15:
            _fft_mode = 1

        j += 1
         
        c0 = cj.copy()
        
    del c0
    
    return waveplanes,cj,jkernel_shape  
    
def atrousmed(arr,scale=1,kernel='linear',median=1):
    """ 
        Compute a' trous wavelet transforms of 2-d numpy
        object up to the 'scale' specified by the user, applying
        the median filter at each scale to remove artifacts.
        
        This method supports 'linear', 'spline', and 'edge' kernels.
        
        Syntax:
            waveplanes,wimage = atrous2d(arr,scale=1,kernel='linear')

        Input:
            arr        - numpy array for input image
            scale      - scale for final wavelet transformation
            kernel     - kernel for wavelet transformation:
                            'linear'(default),'spline','leftedge','rightedge'
        Output:
            waveplanes - stack of wavelet difference images
            wimage     - wavelet transformed image for scale 'scale'
                    
    """        
    if KERNELS.has_key(kernel): 
        kernel1d = KERNELS[kernel]
        wkernel = N.outerproduct(kernel1d,kernel1d)
    else:
        print 'Kernel: ',kernel,' not supported!'
        raise ValueError
    #
    # Initialize variables necessary for keeping track of the
    # kernel size at each scale
    #
    j = 1
    jkernel = wkernel.copy()
    # Define output array containing all the wavelet planes
    waveplanes = N.zeros((scale,arr.shape[0],arr.shape[1]),dtype=arr.dtype)
        
    # Perform convolutions
    c0 = arr
    #cj = arr.copy()
    
    _fft_mode = 0
    for level in xrange(scale):
        order = 2**j
        med_size = jkernel.shape
        print 'med_size: ',med_size,' for scale: ',j
        cj = ND.median_filter(c0,size=med_size,mode='nearest',cval=0.0)
        cj = convolve.convolve2d(cj,kernel=jkernel,fft=_fft_mode,mode='nearest',cval=0.0).astype(arr.dtype)
        waveplanes[level] = c0 - cj
        
        # Expand the kernel
        order = 2**j
        jkernel_shape = (wkernel.shape[0]*order, wkernel.shape[1]*order)
        jkernel = N.zeros(jkernel_shape, dtype=wkernel.dtype)
        jkernel[::order,::order] = wkernel
        if jkernel_shape[0] >= 15 or jkernel_shape[1] >= 15:
            _fft_mode = 1
        j += 1
         
        c0 = cj.copy()
        
    del c0
    
    return waveplanes,cj,jkernel_shape 
    
    
def multimed(arr,maxscale=2,median=2):
    """ 
        Compute multiscale median transforms of 2-d numpy
        object up to the 'maxscale' specified by the user, applying
        the median filter at each scale to remove artifacts.
                
        Syntax:
            waveplanes,wimage = multimed(arr,maxscale=1)

        Input:
            arr        - numpy array for input image
            maxscale   - number wavelet transformations to apply

        Output:
            waveplanes - stack of wavelet difference images
            wimage     - wavelet transformed image for scale 'scale'
                    
    """        
    #
    # Initialize variables necessary for keeping track of the
    # kernel size at each scale
    #
    j = 1
    # Define output array containing all the wavelet planes
    waveplanes = N.zeros((maxscale,arr.shape[0],arr.shape[1]),dtype=arr.dtype)
        
    # Perform convolutions
    c0 = arr.copy()
    #cj = arr.copy()
    med_size = 2*median+1 
    # Insure median filter remains an odd-size
    if (med_size%2) == 0: med_size += 1
    
    _fft_mode = 0
    
    for level in xrange(maxscale):
        cj = ND.median_filter(c0,size=med_size,mode='nearest',cval=0.0)
        waveplanes[level] = c0 - cj
        
        j += 1
        
        del c0
        c0 = cj.copy()
        # Increase the median filter size for next step
        med_size = med_size*2 + 1
        
    del c0
    
    return waveplanes,cj,med_size


def atrous_restore(wavelet_planes, scaled_arr,scale=0):
    """ Converts the final scaled atrous array and the stack of 
        differences back into the original input image. 
    
        If a scale is provided, it will return the wavelet 
        transformed image corresponding to that scale.
        The higher the scale value, the lower the resolution,
        with a scale=0 corresponding to full resolution.   
    """

    _scale = scale
    # Perform some sanity bounds checking on input value
    if _scale < 0: 
        _scale = 0
    
    return scaled_arr+N.add.reduce(wavelet_planes[_scale:])

def atrous_diff(wavelet_planes, scaled_arr,scale=0):
    """ Converts the final scaled atrous array and the stack of 
        differences back into the original input image. 
    
        If a scale is provided, it will return the wavelet 
        transformed image corresponding to that scale.
        The higher the scale value, the lower the resolution,
        with a scale=0 corresponding to full resolution.   
    """

    _scale = scale
    # Perform some sanity bounds checking on input value
    if _scale < 0: 
        _scale = 0

    return N.subtract(scaled_arr,N.add.reduce(wavelet_planes[_scale:]))
