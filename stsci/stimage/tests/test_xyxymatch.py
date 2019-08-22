# Copyright (C) 2008-2010 Association of Universities for Research in Astronomy (AURA)

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

#     1. Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.

#     2. Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials provided
#       with the distribution.

#     3. The name of AURA and its representatives may not be used to
#       endorse or promote products derived from this software without
#       specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY AURA ``AS IS'' AND ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL AURA BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
# TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE.

from __future__ import print_function

import numpy as np
import stsci.stimage as stimage

def test_same():
    np.random.seed(0)
    x = np.random.random((512, 2))
    y = x[:]

    r = stimage.xyxymatch(x, y, algorithm='tolerance',
                          tolerance=0.01,
                          separation=0.0, nmatch=0, maxratio=0, nreject=0)

    print(r.dtype)
    print(r.shape)

    assert len(r) == 512

    for i in range(512):
        assert r['input_x'][i] == r['ref_x'][i]
        assert r['input_y'][i] == r['ref_y'][i]
        assert r['input_idx'][i] == r['ref_idx'][i]
        assert r['input_idx'][i] < 512

def test_different():
    np.random.seed(0)
    x = np.random.random((512, 2))
    y = np.random.random((512, 2))

    r = stimage.xyxymatch(x, y, algorithm='tolerance', tolerance=0.01,
                          separation=0.0)


    assert len(r) < 512 and len(r) > 0
    for i in range(len(r)):
        x0, y0 = r['input_x'][i], r['input_y'][i]
        x1, y1 = r['ref_x'][i], r['ref_y'][i]
        dx = x1 - x0
        dy = y1 - y0
        distance = dx*dx + dy*dy
        assert distance < 0.01 * 0.01
        assert r['input_idx'][i] < 512
        assert r['ref_idx'][i] < 512


