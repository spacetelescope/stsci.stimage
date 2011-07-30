from __future__ import division # confidence high

import sys

pkg =  "multidrizzle"

setupargs = {
    'version' :         "3.3.5dev",
    'description' :		"Automated process for HST image combination and cosmic-ray rejection",
    'author' :		    "Warren Hack, Christopher Hanley, Ivo Busko, Robert Jedrzejewski, and Anton Koekemoer",
    'author_email' :    "help@stsci.edu",
    'license' :		    "http://www.stsci.edu/resources/software_hardware/pyraf/LICENSE",
    'platforms' :	    ["Linux","Solaris","Mac OS X", "Windows"],
    'data_files' :		 [ ('multidrizzle',['LICENSE.txt']) ],
    'package_dir' :     { 'multidrizzle':'lib/multidrizzle', },

}

