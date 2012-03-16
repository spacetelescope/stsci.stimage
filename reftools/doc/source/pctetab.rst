Make PCTETAB Reference File
===========================

The PCTETAB reference file contains data in both its primary header
and in several table extensions. The parameters that go into the primary header
are single numbers that must be specified in the call to `MakePCTETab`.

The data that goes into the table extensions is kept in text files. The
names of these text files are given to `MakePCTETab`, which reads them to
populate the table extensions. See the documentation for `MakePCTETab` for
more detailed descriptions, argument names, and default values.

Primary Header Parameters
-------------------------

* Number of times the readout is simulated to arrive at the corrected image.
* Number of times the pixels are shifted per readout simulation.
* The read noise, in electrons, of the image. This is technically different
  for each amp but here we pick a single representative value.
* The default value selecting a model for how read noise is handled
  before CTE correction.
* Threshold for re-correcting over-subtracted pixels. Sometimes the CTE
  correction removes too much flux from a trail leaving a large divot.

Table Extensions
----------------



.. currentmodule:: reftools.pctetab

.. automodule:: reftools.pctetab
   :members:
