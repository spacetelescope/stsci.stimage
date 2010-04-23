#
#   Authors: Warren Hack, Ivo Busko, Christopher Hanley
#   Program: mdzhandler.py
#   Purpose: Module that handles the MDRIZTAB reference file.
from __future__ import division # confidence high

import pyfits
from pytools import fileutil


def getMultidrizzleParameters(files):
    """ Gets entry in MDRIZTAB where task parameters live.
        This method returns a record array mapping the selected
        row.
    """

    # Get the MDRIZTAB table file name from the primary header.
    # It is gotten from the first file in the input list. No
    # consistency checks are performed.
    _fileName = files[0]
    _header = fileutil.getHeader(_fileName)
    if _header.has_key('MDRIZTAB'):
        _tableName = _header['MDRIZTAB']
    else:
        raise KeyError, "No MDRIZTAB found in file " + _fileName

    _tableName = fileutil.osfn(_tableName)

    # Now get the filters from the primary header.
    _filters = fileutil.getFilterNames(_header)

    # Open MDRIZTAB file.
    try:
        _mdriztab = pyfits.open(_tableName)
    except:
        raise IOError,"MDRIZTAB table '%s' not valid!" % _tableName

    # Look for matching rows based on filter name. If no
    # match, pick up rows for the default filter.
    _rows = _getRowsByFilter(_mdriztab, _filters)
    if _rows == []:
        _rows = _getRowsByFilter(_mdriztab, 'ANY')

    # Now look for the row that matches the number of images.
    # The logic below assumes that rows for a given filter
    # are arranged in ascending order of the 'numimage' field.
    _nimages = len(files)
    _row = 0
    for i in _rows:
        _numimages = _mdriztab[1].data.field('numimages')[i]
        if _nimages >= _numimages:
            _row = i
    print '- MDRIZTAB: MultiDrizzle parameters read from row %s.'%(_row+1)

    mpars = _mdriztab[1].data[_row]
    _mdriztab.close() 

    return mpars

def _getRowsByFilter(table, filters):
    rows = []
    for i in xrange(table[1].data.shape[0]):
        _tfilters = table[1].data.field('filter')[i]
        if _tfilters == filters:
            rows.append(i)
    return rows
