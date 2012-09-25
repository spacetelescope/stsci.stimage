.. stsci.ndimage documentation master file, created by
   sphinx-quickstart on Wed Sep 12 14:53:00 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to NDImage's documentation!
======================================

This package contains various functions for multi-dimensional image
processing.

Modules:

.. toctree::
   :maxdepth: 2

   doccer
   filters
   fourier
   interpolation
   io
   measurements
   morphology


Functions (partial list)
------------------------

.. currentmodule:: stsci.ndimage

.. autosummary::
   :nosignatures: 
   
   interpolation.affine_transform 
   measurements.center_of_mass 
   filters.convolve 
   filters.convolve1d 
   filters.correlate 
   filters.correlate1d 
   measurements.extrema 
   measurements.find_objects 
   filters.generic_filter 
   filters.generic_filter1d 
   interpolation.geometric_transform
   measurements.histogram 
   io.imread 
   measurements.label 
   filters.laplace 
   interpolation.map_coordinates 
   measurements.mean 
   filters.median_filter 
   filters.percentile_filter 
   filters.rank_filter 
   interpolation.rotate 
   interpolation.shift 
   filters.uniform_filter 
   filters.uniform_filter1d 
   interpolation.zoom 

Note: the above is less than half the functions available in this
package      

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

