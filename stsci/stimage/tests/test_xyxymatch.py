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

from __future__ import print_function

import numpy as np
import stsci.stimage as stimage


def rotation_matrix(theta):
    """Compute matrix entries for a 2-D rotation matrix."""
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    return cos_theta, sin_theta


def rotate_points(x, y, rot_mat):
    """Using rotation matrix rotate a set of points."""
    cos_th, sin_th = rot_mat
    rx = [cos_th * x[k] - sin_th * y[k] for k in range(len(x))]
    ry = [sin_th * x[k] + cos_th * y[k] for k in range(len(x))]
    return rx, ry


def translate_points(x, y, point):
    """Translate by points."""
    tx = [x[k] + point[0] for k in range(len(x))]
    ty = [y[k] + point[0] for k in range(len(x))]
    return tx, ty


def magnify_points(x, y, mag):
    """Magnification of points."""
    mx = [el * mag for el in x]
    my = [el * mag for el in y]
    return mx, my


def base_xy_15():
    """Basees to use for comparison before and after transformation."""
    x = [ -3.43295,  3.83868, -4.14375,  1.48054,  0.46832,
           2.04648, -2.04506,  3.08643,  0.52985, -0.03765,
           2.79590, -1.28407,  3.87151,  3.25369,  4.41219,]
    y = [ -4.26100, -4.34275,  3.16412,  3.38923, -2.43702,
           0.96338, -0.44943,  2.59602,  0.19762,  0.76262,
           3.29153,  4.70583, -0.09266,  3.20780, -0.31309,]
    return x, y


def extra_xy_5():
    """Extra points to be tacked on that should probably not be matched."""
    x1 = [  3.53592,  3.58649,  0.30056,  2.61534,  3.81013,]
    y1 = [  4.09539, -4.77615, -2.45605, -0.34375,  1.50182,]
    x2 = [  3.16396,  2.51080, -4.33387,  2.29841,  1.07123,]
    y2 = [  1.57800,  2.99725,  2.01904,  0.16218,  2.56969,]
    return x1, y1, x2, y2


def test_same_15_points():
    """
    The input and reference lists will have the same first 15 points.  An
    additional 5 random points are added, for a total of 20 points in each
    list of points.
    """
    x, y = base_xy_15()
    tx = x.copy()
    ty = y.copy()

    x1, y1, x2, y2 = extra_xy_5()
    x += x1; y += y1
    tx += x2; ty += y2

    inp = np.zeros(shape=(20, 2), dtype=float)
    inp[:, 0] = np.array(x)
    inp[:, 1] = np.array(y)

    ref = np.zeros(shape=(20, 2), dtype=float)
    ref[:, 0] = np.array(tx)
    ref[:, 1] = np.array(ty)

    # import ipdb; ipdb.set_trace()

    r = stimage.xyxymatch(inp, ref, algorithm="triangles", tolerance=0.01)
    print(r)


def test_same():
    import ipdb; ipdb.set_trace()
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


