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

import pytest

import math
import numpy as np
import stsci.stimage as stimage


def rotation_matrix(theta):
    """Compute matrix entries for a 2-D rotation matrix."""
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    return cos_theta, sin_theta


def rotate_points_np(x, y, theta_deg):
    """Using rotation matrix rotate a set of points."""
    rot_theta = theta_deg * np.pi / 180.0
    cos_th, sin_th = rotation_matrix(rot_theta)
    rx = cos_th * x - sin_th * y
    ry = sin_th * x + cos_th * y
    return rx, ry


def translate_points_np(x, y, point):
    """Translate by points."""
    tx = x + point[0]
    ty = y + point[1]
    return tx, ty


def magnify_points_np(x, y, mag):
    """Magnification of points."""
    mx = x * mag
    my = y * mag
    return mx, my


def flip_points_np(x, y):
    """Flip around the X-axis of points."""
    my = y * -1.0
    return x, my


def base_xy_15():
    """Basees to use for comparison before and after transformation."""

    # fmt: off
    x = [-303.84609, -246.00600,  420.31010, -407.02170,   47.92501,
          449.99497,  317.32335,  -78.08896,  345.70714,  150.20207,
          458.55107, -343.01210, -306.99155, -145.12925,  183.15312,]
    y = [-305.98824, -464.58525, -146.35067,  185.79842,  253.11053,
          300.24990, -262.25222,  235.43031,  320.96704, -139.03478,
          233.38413, -316.93320,  202.85420, -252.86989,  382.40623,]
    # fmt: on

    return x, y


def extra_xy_5():
    """Extra points to be tacked on that should probably not be matched."""
    # fmt: off
    x1 = [ 426.96851, -388.88402,  108.22633,  170.85110,  435.57264,]
    y1 = [-423.38909, -447.35147,  255.00272,  -47.67573,  492.17773,]

    x2 = [ 166.09336, -137.12012,  131.59430,  469.28326,  375.83601,]
    y2 = [ 219.05374,  269.97358, -432.54307,  -53.96007,  341.31046,]
    # fmt: on

    return x1, y1, x2, y2


def all_transforms_np(x, y, transform, ttype):
    """Do all transforms."""
    x, y = rotate_points_np(x, y, transform["deg"])
    x, y = magnify_points_np(x, y, transform["mag"])
    if "triangles" == ttype:
        x, y = flip_points_np(x, y)
    x, y = translate_points_np(x, y, transform["trans"])
    return x, y


def get_ndarrays(transform=None, args=None, ttype="triangles"):
    """Create input and reference arrays for tests."""
    x, y = base_xy_15()
    tx = x.copy()
    ty = y.copy()

    x1, y1, x2, y2 = extra_xy_5()
    x += x1
    y += y1
    tx += x2
    ty += y2

    x, y, tx, ty = np.array(x), np.array(y), np.array(tx), np.array(ty)

    if transform is not None:
        if "rotation" == transform:
            degs = args
            x, y = rotate_points_np(x, y, degs)
        if "translate" == transform:
            point = args
            x, y = translate_points_np(x, y, point)
        if "flip" == transform:
            x, y = flip_points_np(x, y)
        if "magnify" == transform:
            mag = args
            x, y = magnify_points_np(x, y, mag)
        if "all" == transform:
            x, y = all_transforms_np(x, y, args, ttype)

    inp = np.zeros(shape=(20, 2), dtype=float)
    inp[:, 0] = x
    inp[:, 1] = y

    ref = np.zeros(shape=(20, 2), dtype=float)
    ref[:, 0] = tx
    ref[:, 1] = ty

    return inp, ref, 15


def test_triangles_15_points_same():
    """
    The input and reference lists will have the same first 15 points.  An
    additional 5 random points are added, for a total of 20 points in each
    list of points.
    """
    inp, ref, elen = get_ndarrays()

    r = stimage.xyxymatch(inp, ref, algorithm="triangles", tolerance=0.01)

    # All 15 base points should be matched.  The 5 extra random points should not be.
    assert len(r) == elen


@pytest.mark.parametrize("theta_deg", [45.0, 60.0, 90.0, 150.0])
def test_triangles_15_points_rotated(theta_deg):
    """
    The input and reference lists will have the same first 15 points.

    An additional 5 random points are added, for a total of 20 points
    in each list of points.

    The input points will be rotated by various degrees.
    """
    inp, ref, elen = get_ndarrays("rotation", theta_deg)

    r = stimage.xyxymatch(inp, ref, algorithm="triangles", tolerance=0.01)

    # All 15 base points should be matched.  The 5 extra random points should not be.
    assert len(r) == elen


def test_triangles_15_points_translated():
    """
    The input and reference lists will have the same first 15 points.

    An additional 5 random points are added, for a total of 20 points
    in each list of points.

    The input points will be translated by (-12, 21)
    """
    inp, ref, elen = get_ndarrays("translate", (-12.0, 21.0))

    r = stimage.xyxymatch(inp, ref, algorithm="triangles", tolerance=0.01)

    # All 15 base points should be matched.  The 5 extra random points should not be.
    assert len(r) == elen


def test_triangles_15_points_flipped():
    """
    The input and reference lists will have the same first 15 points.

    An additional 5 random points are added, for a total of 20 points
    in each list of points.

    The input points will be flipped around the X-axis.
    """
    inp, ref, elen = get_ndarrays("flip")

    r = stimage.xyxymatch(inp, ref, algorithm="triangles", tolerance=0.01)

    # All 15 base points should be matched.  The 5 extra random points should not be.
    assert len(r) == elen


@pytest.mark.parametrize("mag", [0.2, 0.5, 10.0])
def test_triangles_15_points_magnified(mag):
    """
    The input and reference lists will have the same first 15 points.

    An additional 5 random points are added, for a total of 20 points
    in each list of points.

    The input points will be magnified by mag
    """
    inp, ref, elen = get_ndarrays("magnify", mag)

    r = stimage.xyxymatch(inp, ref, algorithm="triangles", tolerance=0.01)

    # All 15 base points should be matched.  The 5 extra random points should not be.
    if mag > 0.3:
        # All 15 base points should be matched.  The 5 extra random points should not be.
        assert len(r) == elen
    else:
        # Due to separation tolerances, few are matched
        assert len(r) < elen


def test_triangles_15_points_all_transforms():
    """
    The input and reference lists will have the same first 15 points.

    An additional 5 random points are added, for a total of 20 points
    in each list of points.

    The input points will be rotated, magnified, flipped, and translated.
    """
    transforms = {"deg": 150.0, "mag": 10.0, "trans": (-12.0, 21.0)}
    inp, ref, elen = get_ndarrays("all", transforms)

    r = stimage.xyxymatch(inp, ref, algorithm="triangles", tolerance=0.01)

    # All 15 base points should be matched.  The 5 extra random points should not be.
    assert len(r) == elen


def test_tolerance_15_points_same():
    """
    The input and reference lists will have the same first 15 points.  An
    additional 5 random points are added, for a total of 20 points in each
    list of points.
    """
    inp, ref, elen = get_ndarrays()

    r = stimage.xyxymatch(inp, ref, algorithm="tolerance", tolerance=0.01)

    # All 15 base points should be matched.  The 5 extra random points should not be.
    assert len(r) == elen


@pytest.mark.parametrize("theta_deg", [45.0, 60.0, 90.0, 150.0])
def test_tolerance_15_points_rotated(theta_deg):
    ref, inp, elen = get_ndarrays("rotation", theta_deg)

    rotation = (theta_deg, theta_deg)
    r = stimage.xyxymatch(
        inp, ref, algorithm="tolerance", tolerance=0.01, rotation=rotation
    )

    # All 15 base points should be matched.  The 5 extra random points should not be.
    assert len(r) == elen


def test_tolerance_15_points_translated():
    ref, inp, elen = get_ndarrays("translate", (-12.0, 21.0))

    r = stimage.xyxymatch(
        inp, ref, algorithm="tolerance", tolerance=0.01, ref_origin=(-12.0, 21.0)
    )

    # All 15 base points should be matched.  The 5 extra random points should not be.
    assert len(r) == elen


@pytest.mark.parametrize("mag", [0.2, 0.5, 10.0])
def test_tolerance_15_points_magnified(mag):
    ref, inp, elen = get_ndarrays("magnify", mag)

    in_mag = [mag, mag]
    r = stimage.xyxymatch(inp, ref, algorithm="tolerance", tolerance=0.01, mag=in_mag)

    if mag > 0.3:
        # All 15 base points should be matched.  The 5 extra random points should not be.
        assert len(r) == elen
    else:
        # Due to separation tolerances, few are matched
        assert len(r) < elen


def test_tolerance_15_points_all_transforms():
    theta_deg, mag, point = 45.0, 10.0, [-12.0, 21.0]
    transforms = {"deg": theta_deg, "mag": mag, "trans": point}
    ref, inp, elen = get_ndarrays("all", transforms, ttype="tolerance")

    rotation = [theta_deg, theta_deg]
    in_mag = [mag, mag]
    r = stimage.xyxymatch(
        inp,
        ref,
        algorithm="tolerance",
        tolerance=0.01,
        mag=in_mag,
        rotation=rotation,
        ref_origin=point,
    )

    # All 15 base points should be matched.  The 5 extra random points should not be.
    assert len(r) == elen
