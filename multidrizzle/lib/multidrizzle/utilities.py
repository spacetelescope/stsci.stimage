#!/usr/bin/env python

from __future__ import division # confidence high
"""

    Purpose
    =======
    A collection of general utility functions that are used by MultiDrizzle.

   :author: Christopher Hanley

"""

def electrons2dn(value, gain=1):
    """
    
    Purpose
    =======
    Convert data from units of electrons to counts
    
    Argumens
    ========
    :value: Array or value to be converted
    :gain: Value given in units of e-/ADU
    
    """
    return value /gain


def dn2electrons(value,gain=1):
    """
    
    Purpose
    =======
    Convert data from units of counts to electrons

    Argumens
    ========
    :value: Array or value to be converted
    :gain: Value given in units of e-/ADU

    """
    return value * gain

