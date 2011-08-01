from __future__ import division # confidence high

import sys

pkg =  ["stwcs", 'stwcs.updatewcs', 'stwcs.wcsutil', 'stwcs.distortion']

setupargs = {
    'version' :         "0.8",
    'description' :		"Recomputes the WCS of an HST observation and puts all istortion corrections in the headers.",
    'package_dir': {'stwcs':'lib/stwcs', 'stwcs.updatewcs': 'lib/stwcs/updatewcs',
                    'stwcs.wcsutil': 'lib/stwcs/wcsutil', 'stwcs.distortion': 'lib/stwcs/distortion'},

    'author' :		    "Nadia Dencheva, Warren Hack",
    'author_email' :    "help@stsci.edu",
    'license' :		    "http://www.stsci.edu/resources/software_hardware/pyraf/LICENSE",
    'platforms' :	    ["Linux","Solaris","Mac OS X", "Windows"],
}
