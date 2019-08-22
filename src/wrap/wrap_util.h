/*
Copyright (C) 2008-2010 Association of Universities for Research in Astronomy (AURA)

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer in the documentation and/or other materials provided
      with the distribution.

    3. The name of AURA and its representatives may not be used to
      endorse or promote products derived from this software without
      specific prior written permission.

THIS SOFTWARE IS PROVIDED BY AURA ``AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL AURA BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
*/

/*
 Author: Michael Droettboom
         mdroe@stsci.edu
*/

#ifndef __STIMAGE_WRAP_UTIL_H__
#define __STIMAGE_WRAP_UTIL_H__

#define PY_ARRAY_UNIQUE_SYMBOL pywcs_numpy_api

#include <Python.h>
#include <numpy/arrayobject.h>

#include "immatch/xyxymatch.h"
#include "immatch/geomap.h"
#include "lib/util.h"
#include "lib/xybbox.h"

extern char* SIZE_T_D;

int
to_coord_t(
        const char* const name,
        PyObject* o,
        coord_t* const c);

int
from_coord_t(
        const coord_t* const c,
        PyObject** o);

int
to_bbox_t(
        const char* const name,
        PyObject* o,
        bbox_t* const b);

int
to_xyxymatch_algo_e(
        const char* const name,
        const char* const s,
        xyxymatch_algo_e* const e);

int
to_geomap_fit_e(
        const char* const name,
        const char* const s,
        geomap_fit_e* const e);

int
from_geomap_fit_e(
        const geomap_fit_e e,
        PyObject** o);

int
to_surface_type_e(
        const char* const name,
        const char* const s,
        surface_type_e* const e);

int
from_surface_type_e(
        const surface_type_e e,
        PyObject** o);

int
to_xterms_e(
        const char* const name,
        const char* const s,
        xterms_e* const e);

int
from_xterms_e(
        const xterms_e e,
        PyObject** o);

#endif
