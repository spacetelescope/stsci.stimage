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
from stsci.stimage._stimage import GeomapResults


def base_xy():
    """Basees to use for comparison before and after transformation."""

    x = [
        -303.84609,
        -246.00600,
        420.31010,
        -407.02170,
        47.92501,
        449.99497,
        317.32335,
        -78.08896,
        345.70714,
        150.20207,
        458.55107,
        -343.01210,
        -306.99155,
        -145.12925,
        183.15312,
        426.96851,
        -388.88402,
        108.22633,
        170.85110,
        435.57264,
    ]
    y = [
        -305.98824,
        -464.58525,
        -146.35067,
        185.79842,
        253.11053,
        300.24990,
        -262.25222,
        235.43031,
        320.96704,
        -139.03478,
        233.38413,
        -316.93320,
        202.85420,
        -252.86989,
        382.40623,
        -423.38909,
        -447.35147,
        255.00272,
        -47.67573,
        492.17773,
    ]

    return x, y


def input_reference_points():
    """Get base input and reference points."""
    x, y = base_xy()
    tx = x.copy()
    ty = y.copy()

    length = len(x)

    inp = np.zeros(shape=(length, 2), dtype=float)
    inp[:, 0] = np.array(x)
    inp[:, 1] = np.array(y)

    ref = np.zeros(shape=(length, 2), dtype=float)
    ref[:, 0] = np.array(tx)
    ref[:, 1] = np.array(ty)

    return inp, ref


def assert_check_result(check, result):
    assert check.fit_geometry == result.fit_geometry
    assert check.function == result.function

    tol = 1.0e-4
    np.testing.assert_allclose(check.rms, result.rms, rtol=tol, atol=tol)
    np.testing.assert_allclose(check.mean_ref, result.mean_ref, rtol=tol, atol=tol)
    np.testing.assert_allclose(check.mean_input, result.mean_input, rtol=tol, atol=tol)

    np.testing.assert_allclose(check.shift, result.shift, rtol=tol, atol=tol)
    np.testing.assert_allclose(check.mag, result.mag, rtol=tol, atol=tol)
    np.testing.assert_allclose(check.rotation, result.rotation, rtol=tol, atol=tol)

    np.testing.assert_allclose(check.xcoeff, result.xcoeff, rtol=tol, atol=tol)
    np.testing.assert_allclose(check.ycoeff, result.ycoeff, rtol=tol, atol=tol)


def check_test_same():
    check = GeomapResults()

    fit_geometry = "general"
    function = "polynomial"

    rms = [645.2651, 859.8201]
    mean_ref = [64.7903, 2.7475]
    mean_input = [64.7903, 2.7475]

    shift = [-1.8641052e07, -8.7458640e06]
    mag = [3.1327, 0.1623]
    rotation = [135.0, 37.2223]

    xcoeff = [-1.86410498e07, -2.21516146e00, 9.81723195e-02]
    ycoeff = [-8.74586261e06, -1.03555084e00, 1.29232998e-01]

    check.fit_geometry = fit_geometry
    check.function = function

    check.rms = np.array(rms, dtype=np.float32)
    check.mean_ref = np.array(mean_ref, dtype=np.float32)
    check.mean_input = np.array(mean_input, dtype=np.float32)

    check.shift = np.array(shift, dtype=np.float32)
    check.mag = np.array(mag, dtype=np.float32)
    check.rotation = np.array(rotation, dtype=np.float32)

    check.xcoeff = np.array(xcoeff, dtype=float)
    check.ycoeff = np.array(ycoeff, dtype=float)

    return check


@pytest.mark.xfail(reason="geomap not supported")
def test_same():
    """Test same points."""
    inp, ref = input_reference_points()

    r = stimage.geomap(inp, ref, fit_geometry="general", function="polynomial")
    result = r[0]

    check = check_test_same()
    assert_check_result(check, result)


def rotation_matrix(theta):
    """Compute matrix entries for a 2-D rotation matrix."""
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    return cos_theta, sin_theta


def rotate_points(points, rot_mat):
    """Using rotation matrix rotate a set of points."""
    cos_th, sin_th = rot_mat
    new_points = np.zeros(shape=points.shape, dtype=points.dtype)
    new_points[:, 0] = cos_th * points[:, 0] - sin_th * points[:, 1]
    new_points[:, 1] = sin_th * points[:, 0] + cos_th * points[:, 1]
    return new_points


def check_test_rotate_45():
    check = GeomapResults()

    fit_geometry = "rotate"
    function = "polynomial"

    rms = [356.9341, 363.1235]
    mean_ref = [43.8709, 47.7565]
    mean_input = [64.7903, 2.7475]

    shift = [60.9537, -60.5476]
    mag = [0.9827, 1.0]
    rotation = [135.0, 45.985]

    xcoeff = [60.9294, -0.6948, 0.7192]
    ycoeff = [-61.9859, 0.7192, 0.6948]

    check.fit_geometry = fit_geometry
    check.function = function

    check.rms = np.array(rms, dtype=np.float32)
    check.mean_ref = np.array(mean_ref, dtype=np.float32)
    check.mean_input = np.array(mean_input, dtype=np.float32)

    check.shift = np.array(shift, dtype=np.float32)
    check.mag = np.array(mag, dtype=np.float32)
    check.rotation = np.array(rotation, dtype=np.float32)

    check.xcoeff = np.array(xcoeff, dtype=float)
    check.ycoeff = np.array(ycoeff, dtype=float)

    return check


@pytest.mark.xfail(reason="geomap not supported")
@pytest.mark.parametrize("deg", [45])
def test_rotation(deg):
    inp, ref = input_reference_points()

    # Rotate input points by theta_deg degrees
    rot_theta = float(deg) * np.pi / 180.0
    rot_mat = rotation_matrix(rot_theta)
    ref = rotate_points(ref, rot_mat)

    r = stimage.geomap(inp, ref, fit_geometry="rotate", function="polynomial")
    result = r[0]

    if deg == 45:
        check = check_test_rotate_45()
    else:
        print(f"Degree {deg} is invalid and isn't tested.")
        return
    assert_check_result(check, result)


def check_test_translate():
    check = GeomapResults()

    fit_geometry = "shift"
    function = "polynomial"

    rms = [5.1299, 9.2338]
    mean_ref = [69.7903, 11.7475]
    mean_input = [64.7903, 2.7475]

    shift = [-8.0, -3.0]
    mag = [0.0, 1.0]
    rotation = [0.0, 90.0]

    xcoeff = [-9.0, 0.0, 1.0]
    ycoeff = [-5.0, 1.0, 0.0]

    check.fit_geometry = fit_geometry
    check.function = function

    check.rms = np.array(rms, dtype=np.float32)
    check.mean_ref = np.array(mean_ref, dtype=np.float32)
    check.mean_input = np.array(mean_input, dtype=np.float32)

    check.shift = np.array(shift, dtype=np.float32)
    check.mag = np.array(mag, dtype=np.float32)
    check.rotation = np.array(rotation, dtype=np.float32)

    check.xcoeff = np.array(xcoeff, dtype=float)
    check.ycoeff = np.array(ycoeff, dtype=float)

    return check


@pytest.mark.xfail(reason="geomap not supported")
def test_translate():
    inp, ref = input_reference_points()

    # Translate input points by {.x=5.0, .y=9.0};
    ref[:, 0] += 5.0
    ref[:, 1] += 9.0

    r = stimage.geomap(inp, ref, fit_geometry="shift", function="polynomial")
    result = r[0]

    check = check_test_translate()
    assert_check_result(check, result)


def check_test_magnify():
    check = GeomapResults()

    fit_geometry = "rscale"
    function = "polynomial"

    rms = [1.2594e-13, 8.3502e-14]
    mean_ref = [323.9514, 13.7377]
    mean_input = [64.7903, 2.7475]

    shift = [2.0000e-01, -1.3323e-14]
    mag = [0.2828, 0.2]
    rotation = [315.0, 360.0]

    xcoeff = [1.4211e-14, 2.0000e-01, -4.8986e-17]
    ycoeff = [-1.3323e-14, 4.8986e-17, 2.0000e-01]

    check.fit_geometry = fit_geometry
    check.function = function

    check.rms = np.array(rms, dtype=np.float32)
    check.mean_ref = np.array(mean_ref, dtype=np.float32)
    check.mean_input = np.array(mean_input, dtype=np.float32)

    check.shift = np.array(shift, dtype=np.float32)
    check.mag = np.array(mag, dtype=np.float32)
    check.rotation = np.array(rotation, dtype=np.float32)

    check.xcoeff = np.array(xcoeff, dtype=float)
    check.ycoeff = np.array(ycoeff, dtype=float)

    return check


def check_test_magnify_legendre():
    check = GeomapResults()

    fit_geometry = "rscale"
    function = "legendre"

    rms = [700111.549164, 748496.099203]
    mean_ref = [323.951435, 13.737667]
    mean_input = [64.790287, 2.747533]

    shift = [-407.121700, 13.796240]
    mag = [0.282843, 0.200000]
    rotation = [315.0, 360.0]

    xcoeff = [2.5765e01, 4.3289e02, -1.1719e-13]
    ycoeff = [1.3796e01, 1.0603e-13, 4.7848e02]

    check.fit_geometry = fit_geometry
    check.function = function

    check.rms = np.array(rms, dtype=np.float32)
    check.mean_ref = np.array(mean_ref, dtype=np.float32)
    check.mean_input = np.array(mean_input, dtype=np.float32)

    check.shift = np.array(shift, dtype=np.float32)
    check.mag = np.array(mag, dtype=np.float32)
    check.rotation = np.array(rotation, dtype=np.float32)

    check.xcoeff = np.array(xcoeff, dtype=float)
    check.ycoeff = np.array(ycoeff, dtype=float)

    return check


def check_test_magnify_chebyshev():
    check = GeomapResults()

    fit_geometry = "rscale"
    function = "chebyshev"

    rms = [700111.56, 748496.1]
    mean_ref = [323.9514, 13.7377]
    mean_input = [64.7903, 2.7475]

    shift = [-407.1217, 13.7962]
    mag = [0.2828, 0.2]
    rotation = [315.0, 360.0]

    xcoeff = [2.5765e01, 4.3289e02, -1.1719e-13]
    ycoeff = [1.3796e01, 1.0603e-13, 4.7848e02]

    check.fit_geometry = fit_geometry
    check.function = function

    check.rms = np.array(rms, dtype=np.float32)
    check.mean_ref = np.array(mean_ref, dtype=np.float32)
    check.mean_input = np.array(mean_input, dtype=np.float32)

    check.shift = np.array(shift, dtype=np.float32)
    check.mag = np.array(mag, dtype=np.float32)
    check.rotation = np.array(rotation, dtype=np.float32)

    check.xcoeff = np.array(xcoeff, dtype=float)
    check.ycoeff = np.array(ycoeff, dtype=float)

    return check


@pytest.mark.xfail(reason="geomap not supported")
@pytest.mark.parametrize("poly", ["polynomial", "legendre", "chebyshev"])
def test_magnification(poly):
    """Test magnification fitting."""
    inp, ref = input_reference_points()

    # Magnify inputs by
    ref *= 5.0

    r = stimage.geomap(inp, ref, fit_geometry="rscale", function=poly)
    result = r[0]

    if poly == "polynomial":
        check = check_test_magnify()
    elif poly == "legendre":
        check = check_test_magnify_legendre()
    elif poly == "chebyshev":
        check = check_test_magnify_chebyshev()

    assert_check_result(check, result)
