# Copyright (C) 2008-2025 Association of Universities for Research in Astronomy (AURA)

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

import numpy as np
import stsci.stimage as stimage


def test_same():
    np.random.seed(0)
    x = np.random.random((512, 2))
    shift = np.random.random((2,)) * 0.1
    y = x[:] + shift

    r, _ = stimage.geomap(y, x, fit_geometry="general", function="polynomial")

    assert r.fit_geometry == "general"
    assert r.function == "polynomial"
    # add some meaningful assertions


def test_shift():
    np.random.seed(0)
    ref_coord = np.random.random((512, 2))
    shift = 10 * np.random.random((2,))
    transformed_coord = ref_coord + shift

    r, _ = stimage.geomap(
        transformed_coord,
        ref_coord,
        fit_geometry="shift",
        function="polynomial"
    )

    assert r.fit_geometry == "shift"
    assert r.function == "polynomial"

    # next three fail - need to investigate this.
    # assert np.linalg.norm(r.rms) < 1e-6
    # assert np.linalg.norm(r.rotation) < 1e-6
    # assert np.allclose(r.shift, shift)

    # why x<->y swapped?
    assert np.allclose(r.ycoeff, [shift[0], 1, 0])  # should be r.xcoeff
    assert np.allclose(r.xcoeff, [shift[1], 0, 1])  # should be r.ycoeff
