from __future__ import division         # confidence unknown

def saaModel (model):
    """Get vertices for SAA model number `model`.

    This was copied from UVMpdbelem.cgi, downloaded from:
    http://www.sesd.stsci.edu/prd/files/UVMpdbelem.cgi?ELEM=pdb/svdf.dat

    See http://www.sesd.stsci.edu/prd/icd/icd26_pt3_10_svdf.html for
    a description of the SAA vertex description file.

    Parameters
    ----------
    model: int
        The SAA model number (0 - 32, inclusive).

    Returns
    -------
    list
        List of (latitude, longitude) tuples, one for each vertex.
    """

    saa_models = {
        # RADIO FREQUENCY INTERFERENCE CONTOUR FOR MULTI ACCESS 4/24/89
        # model 0
        0: [(30.0,         62.0),
            (23.0,         78.0),
            (23.0,         86.0),
            (35.0,         86.0)],

        # RADIO FREQ INTERFERENCE CONTOUR FOR SBAND SINGLE ACCESS 4/24/89
        # model 1
        1: [(30.0,         62.0),
            (23.0,         78.0),
            (14.0,        102.0),
            (20.0,        112.0),
            (32.0,        116.0)],

        # SAA MODEL 02 FGS EMPIRICALLY OBTAINED CONTOUR
        # EASTERN BOUNDARY REDEFINED -HDOS REQUEST-REFERENCE HSTAR 1939
        # CONTOUR REDEFINED FOR 14.5 Mv GUIDE STARS - J. ESPER 9/24/93
        # UPDATE: WESTERN HALF SHIFTED WEST 6DEG.  PR 65147, 5/13/10
        # UPDATE: EASTERN HALF ADJUSTED.  PR 65227, 5/24/10
        # flux density 130.0
        2: [(-29.0,          2.0),
            (-26.1,          1.0),
            (-23.0,        358.0),
            (-19.3,        353.0),
            (-15.6,        347.0),
            (-12.0,        340.0),
            (-09.9,        331.4),
            (-09.1,        318.8),
            (-10.0,        308.0),
            (-11.9,        297.2),
            (-14.9,        286.1),
            (-17.0,        283.0),
            (-19.1,        279.9),
            (-21.3,        277.5),
            (-23.7,        276.5),
            (-26.0,        276.4),
            (-29.0,        276.7)],

        # SAA MODEL 03 - STIS MAMA
        # see FOC memo 8/23/90
        # BASED ON IN-ORBIT DATA DURING SAA TESTING- revised 2/19/91
        # 6/17/09: WAS FOC PERFORMANCE CONTOUR. CLONE MODEL 25 STIS MAMA PR62902
        # UPDATE: SHIFTED WEST 6DEG.  PR 65147, 5/13/10
        # flux density 125.0
        3: [(-28.3,         14.0),
            (-27.5,         15.0),
            (-26.1,         13.0),
            (-19.8,          1.5),
            ( -9.6,        341.0),
            ( -7.6,        330.4),
            ( -6.0,        318.8),
            ( -7.9,        297.2),
            (-12.0,        286.1),
            (-17.1,        279.9),
            (-20.3,        277.5),
            (-23.5,        276.5),
            (-26.0,        276.4),
            (-28.6,        276.7)],

        # SAA MODEL 04 - ACS MAMA
        # CARD 2.4.2.19 HAS BEEN DELETED
        # FLUX &gt; 200 ELECTRONS/CM*2-S   ENERGY &gt; 3 MEV  7/20/89
        # 6/17/09: WAS FOC HEALTH AND SAFETY. CLONE MODEL 28 ACS MAMA, PR 62902
        # UPDATE: SHIFTED WEST 6DEG.  PR 65147, 5/13/10
        # flux density 15.0
        4: [(-28.5,         19.0),
            (-16.0,          1.0),
            ( -6.5,        345.0),
            ( -2.0,        335.0),
            (  1.0,        312.0),
            ( -3.0,        294.0),
            ( -7.0,        284.0),
            (-10.0,        278.0),
            (-15.0,        272.0),
            (-20.0,        267.0),
            (-30.0,        269.0)],

        # SAA MODEL 05 - ASTROMETRY CONTOUR
        # SUPERSET OF 10 MEV PROTONS AND 0.5 MEV ELECTRONS CONTOURS
        # FORMERLY CARD 3.2.3.4 NOW IN OPS LIMITATIONS DOCUMENT (O.L.D) 2.2.3.10
        # For   FGS     FLUX &gt;   10 PROTONS/CM*2-S ENERGY &gt; 10MEV
        # FLUX &gt; 1000 ELECT/CM*2-S   ENERGY &gt; 0.5 MEV  01/05/87
        # UPDATE: SHIFTED WEST 6DEG.  PR 65147, 5/13/10
        # flux density 10.0
        5: [(-50.0,        294.0),
            (-30.0,         39.0),
            (-25.0,         34.0),
            (-21.0,         24.0),
            (-16.0,          9.0),
            (-10.2,        354.0),
            ( -2.0,        335.0),
            (  1.0,        312.0),
            ( -3.0,        294.0),
            ( -8.0,        277.0),
            (-20.0,        267.0),
            (-30.0,        269.0)],

        # SAA MODEL 06 - COS NUv (MAMA)
        # FOS memo 2/04/92 Measured FOS Flux 0.04 counts/sec/diode
        # 6/17/09: WAS FOS NORMAL CONTOUR. CLONE MODEL 32 COS NUV (MAMA) PR62902
        # UPDATE: SHIFTED WEST 6DEG.  PR 65147, 5/13/10
        # flux density 125.0
        6: [(-28.3,         14.0),
            (-27.5,         15.0),
            (-26.1,         13.0),
            (-19.8,          1.5),
            ( -9.6,        341.0),
            ( -7.6,        330.4),
            ( -6.0,        318.8),
            ( -7.9,        297.2),
            (-12.0,        286.1),
            (-17.1,        279.9),
            (-20.3,        277.5),
            (-23.5,        276.5),
            (-26.0,        276.4),
            (-28.6,        276.7)],

        # SAA MODEL 07 - GHRS OBSERVED CONTOUR
        # GHRS MEMO 09/04/90 REVISE EASTERN BOUNDARY 03/13/92
        # flux density 100.0
        7: [(-50.0,        300.0),
            (-41.0,        349.0),
            (-23.0,          5.0),
            ( -2.0,        341.0),
            (  1.0,        318.0),
            ( -3.0,        300.0),
            ( -8.0,        283.0),
            (-20.0,        273.0),
            (-30.0,        275.0)],

        # SAA MODEL 08 CONTINUOUS OPERATION CONTOUR BENEATH 30 DEG SOUTH
        # no longer used
        # flux density 160.0
        8: [(-33.0,        325.0),
            (-32.0,        320.0),
            (-31.0,        307.0),
            (-32.0,        301.0),
            (-35.0,        299.0),
            (-37.0,        300.0),
            (-38.0,        305.0),
            (-38.0,        310.0),
            (-36.0,        320.0)],

        # SAA MODEL 09 - FOC NORMAL CONTOUR                  NO LONGER USED
        # no longer used
        # FLUX &gt; 10 PROTONS / CM*2-S  ENERGY &gt; 50 MEV
        # flux density 30.0
        9: [(-48.0,        300.0),
            (-30.0,         43.0),
            (-23.0,         31.0),
            (-16.0,         14.0),
            ( -5.0,        345.0),
            ( -3.0,        339.0),
            (  0.0,        317.0),
            ( -9.0,        285.0),
            (-20.0,        276.0),
            (-30.0,        276.0)],

        # SAA MODEL 10 - FOC HEALTH AND SAFETY CONTOUR       NO LONGER USED
        # no longer used
        # FLUX &gt; 100 ELECTRONS / CM*2-S  ENERGY &gt; 3 MEV
        # flux density 140.0
        10: [(-38.0,        335.0),
             (-30.0,        350.0),
             (-26.0,        355.0),
             (-21.0,        359.0),
             (-20.0,        345.0),
             (-24.0,        325.0),
             (-27.0,        315.0),
             (-35.2,        300.0),
             (-39.0,        295.0),
             (-42.0,        292.0),
             (-43.0,        300.0),
             (-42.0,        321.0)],

        # SAA MODEL 11 CONTINUOUS OPERATION CONTOUR BENEATH 30 DEG SOUTH
        # flux density 160.0
        11: [(-33.0,        325.0),
             (-32.0,        320.0),
             (-31.0,        307.0),
             (-32.0,        301.0),
             (-35.0,        299.0),
             (-37.0,        300.0),
             (-38.0,        305.0),
             (-38.0,        310.0),
             (-36.0,        320.0)],

        # SAA MODEL 12 - FLUX &gt; 1300 PROTONS/CM*2-S  ENERGY &gt; 50MEV
        # no longer used
        # FROM STASSINOPOULOS THEORETICAL MODEL AT 600KM - SOLAR MIN EPOCH 1992.
        # flux density 90.0
        12: [(-33.0,        336.0),
             (-31.0,        340.0),
             (-28.0,        345.0),
             (-24.0,        350.0),
             (-23.0,        347.0),
             (-22.0,        343.0),
             (-18.0,        329.0),
             (-16.0,        318.0),
             (-21.0,        300.0),
             (-23.0,        296.0),
             (-25.0,        294.0),
             (-30.0,        296.0),
             (-38.0,        300.0)],

        # SAA MODEL 13 - FGS EMPIRICALLY OBTAINED CONTOUR    NO LONGER USED
        # no longer used
        # EASTERN BOUNDARY REDEFINED -HDOS REQUEST-REFERENCE HSTAR 1939 2/19/91
        # flux density 130.0
        13: [(-30.0,        349.0),
             (-26.0,        351.0),
             (-24.0,        350.5),
             (-18.0,        349.0),
             (-14.0,        340.0),
             (-12.0,        330.0),
             (-12.0,        310.0),
             (-13.0,        300.0),
             (-13.8,        297.0),
             (-16.0,        293.0),
             (-19.5,        288.0),
             (-24.0,        284.5),
             (-26.0,        284.0),
             (-30.0,        285.0)],

        # SAA MODEL 14 - FLUX &gt; 800 PROTONS/CM*2-S   ENERGY &gt; 50MEV
        # no longer used
        # FROM STASSINOPOULOS THEORETICAL MODEL AT 600KM - SOLAR MIN EPOCH 1992.
        # flux density 70.0
        14: [(-32.0,          0.0),
             (-23.0,        353.0),
             (-19.0,        350.0),
             (-14.0,        340.0),
             (-12.0,        334.0),
             (-11.2,        328.0),
             (-11.0,        326.0),
             (-12.0,        310.0),
             (-15.0,        300.0),
             (-16.0,        298.0),
             (-20.0,        294.0),
             (-25.0,        289.0),
             (-30.0,        290.0),
             (-38.0,        296.0),
             (-42.0,        301.0),
             (-36.0,        351.0)],

        # SAA MODEL 15 - FLUX &gt; 300 PROTONS/CM*2-S   ENERGY &gt; 50MEV
        # no longer used
        # FROM STASSINOPOULOS THEORETICAL MODEL AT 600KM - SOLAR MIN EPOCH 1992.
        # flux density 60.0
        15: [(-31.0,         10.0),
             (-25.0,         15.0),
             (-20.0,          2.0),
             (-15.0,        350.0),
             (-11.0,        340.0),
             (-10.0,        337.0),
             ( -9.0,        334.0),
             ( -8.0,        330.0),
             (-10.0,        304.0),
             (-11.0,        299.0),
             (-12.0,        297.0),
             (-20.0,        286.0),
             (-25.0,        283.0),
             (-30.0,        285.0),
             (-36.0,        290.0),
             (-45.0,        300.0)],

        # SAA MODEL 16 - FLUX &gt; 100 PROTONS/CM*2-S   ENERGY &gt; 50MEV
        # no longer used
        # FROM STASSINOPOULOS THEORETICAL MODEL AT 600KM - SOLAR MIN EPOCH 1992.
        # flux density 40.0
        16: [(-31.0,         30.0),
             (-26.0,         33.0),
             (-25.0,         30.0),
             (-18.0,         10.0),
             (-10.0,        350.0),
             ( -7.0,        343.0),
             ( -5.0,        334.0),
             ( -5.0,        318.0),
             ( -6.0,        304.0),
             ( -7.0,        298.0),
             (-12.0,        290.0),
             (-20.0,        282.0),
             (-26.0,        279.0),
             (-30.0,        282.0),
             (-47.0,        300.0)],

        # SAA MODEL 17 - FLUX &gt; 50 PROTONS/CM*2-S ENERGY &gt; 100MEV
        # no longer used
        # FROM STASSINOPOULOS THEORETICAL MODEL AT 600KM - SOLAR MIN EPOCH 1992.
        # flux density 50.0
        17: [(-44.0,          0.0),
             (-30.0,         26.0),
             (-25.0,         27.0),
             (-17.0,          6.0),
             ( -8.0,        344.0),
             ( -6.0,        314.0),
             ( -7.0,        302.0),
             (-17.0,        286.0),
             (-25.0,        282.0),
             (-30.0,        284.0),
             (-46.0,        300.0)],

        # SAA MODEL 18 - FOS LOW BACKGROUND CONTOUR    cf FOS memo 2/04/92
        # no longer used
        # Measured FOS Flux 0.02 counts/sec/diode
        # flux density 20.0
        18: [(-28.0,        013.5),
             (-23.9,        012.8),
             (-18.3,        007.2),
             (  0.0,        341.0),
             (  3.0,        309.0),
             (  0.0,        288.5),
             ( -9.4,        272.6),
             (-17.8,        268.2),
             (-25.8,        269.1),
             (-29.0,        275.0)],

        # SAA MODEL 19 - FOS NEW RED CONTOUR           cf FOS memo 3/19/93
        # no longer used
        # flux density 120.0
        19: [(-30.0,        283.0),
             (-29.0,        357.0),
             (-27.0,        359.0),
             (-25.0,        359.0),
             (-21.0,        357.0),
             (-19.0,        355.0),
             ( -1.0,        329.0),
             ( -1.0,        317.0),
             ( -5.0,        293.0),
             (-13.0,        281.0),
             (-15.0,        279.0),
             (-27.0,        279.0),
             (-29.0,        281.0)],

        # SAA MODEL 20 - FOS NEW BLUE CONTOUR          cf FOS memo 3/19/93
        # no longer used
        # flux density 120.0
        20: [(-30.0,        285.0),
             (-29.0,        347.0),
             (-27.0,        351.0),
             (-21.0,        351.0),
             (-17.0,        349.0),
             ( -1.0,        325.0),
             ( -1.0,        323.0),
             ( -3.0,        307.0),
             ( -5.0,        299.0),
             ( -9.0,        292.0),
             (-13.0,        285.0),
             (-23.0,        279.0),
             (-25.0,        279.0),
             (-29.0,        283.0)],

        # SAA MODEL 21 - SUPERSET OF FOC PERFORMANCE #3 &amp; FOS RED SIDE #19
        # no longer used
        # flux density 120.0
        21: [(-42.0,        294.4),
             (-42.8,        301.4),
             (-29.0,        357.0),
             (-27.0,        359.0),
             (-20.9,        359.0),
             ( -1.0,        332.0),
             ( -1.0,        317.0),
             ( -5.0,        293.0),
             (-13.0,        281.0),
             (-15.0,        279.0),
             (-27.0,        279.0),
             (-32.7,        282.6)],

        # SAA M0DEL 22 - SUPERSET OF FOC PERFORMANCE #3 &amp; FOS BLUE SIDE #20
        # no longer used
        # flux density 120.0
        22: [(-42.0,        294.4),
             (-42.8,        301.4),
             (-30.0,        350.0),
             (-20.9,        358.4),
             ( -4.9,        335.8),
             ( -1.0,        325.0),
             ( -1.0,        323.0),
             ( -3.0,        307.0),
             ( -5.0,        299.0),
             ( -9.0,        292.0),
             (-13.0,        285.0),
             (-22.0,        279.0),
             (-25.0,        279.0),
             (-32.7,        282.6)],

        # SAA MODEL 23 - NICMOS
        # INITIAL ENTRY: CLONE OF MODEL 5.  CCR 27058, 11/18/96
        # UPDATE: SHIFTED WEST 6DEG.  PR 65147, 5/13/10
        # flux density 10.0
        23: [(-50.0,        294.0),
             (-30.0,         39.0),
             (-25.0,         34.0),
             (-21.0,         24.0),
             (-16.0,          9.0),
             (-10.2,        354.0),
             ( -2.0,        335.0),
             (  1.0,        312.0),
             ( -3.0,        294.0),
             ( -8.0,        277.0),
             (-20.0,        267.0),
             (-30.0,        269.0)],

        # SAA MODEL 24 - STIS CCD
        # INITIAL ENTRY: CLONE OF MODEL 5.  CCR 27058, 11/18/96
        # UPDATE: STIS HV RECON CONTOUR. PR 37266, 10/12/98
        # UPDATE: SHIFTED WEST 6DEG.  PR 65147, 5/13/10
        # flux density 125.0
        24: [(-28.3,         14.0),
             (-27.5,         15.0),
             (-26.1,         13.0),
             (-19.8,          1.5),
             ( -9.6,        341.0),
             ( -7.6,        330.4),
             ( -6.0,        318.8),
             ( -7.9,        297.2),
             (-12.0,        286.1),
             (-17.1,        279.9),
             (-20.3,        277.5),
             (-23.5,        276.5),
             (-26.0,        276.4),
             (-28.6,        276.7)],

        # SAA MODEL 25 STIS MAMA
        # INITIAL ENTRY: CLONE OF MODEL 7.  CCR 27058, 11/18/96
        # UPDATE: CLONE OF MODEL 5.  CCR 27083, 4/17/97
        # UPDATE: STIS HV RECON CONTOUR.  PR 37266, 10/12/98
        # UPDATE: SHIFTED WEST 6DEG.  PR 65147, 5/13/10
        # flux density 125.0
        25: [(-28.3,         14.0),
             (-27.5,         15.0),
             (-26.1,         13.0),
             (-19.8,          1.5),
             ( -9.6,        341.0),
             ( -7.6,        330.4),
             ( -6.0,        318.8),
             ( -7.9,        297.2),
             (-12.0,        286.1),
             (-17.1,        279.9),
             (-20.3,        277.5),
             (-23.5,        276.5),
             (-26.0,        276.4),
             (-28.6,        276.7)],

        # SAA MODEL 26 - WFPC2 EMPIRICALLY DETERMINED CONTOUR
        # INITIAL ENTRY: MODIFICATION OF MODEL 5.  PR 37907, 11/09/98
        # flux density 15.0
        26: [(-28.5,         25.0),
             (-16.0,          7.0),
             ( -6.5,        351.0),
             ( -2.0,        341.0),
             (  1.0,        318.0),
             ( -3.0,        300.0),
             ( -7.0,        290.0),
             (-10.0,        284.0),
             (-15.0,        278.0),
             (-20.0,        273.0),
             (-30.0,        275.0)],

        # SAA MODEL 27 - ACS CCDs
        # INITIAL ENTRY: CLONE OF MODEL 26, OPR 38805, 4/2/1999
        # UPDATE: SHIFTED WEST 6DEG.  PR 65147, 5/13/10
        # flux density 15.0
        27: [(-28.5,         19.0),
             (-16.0,          1.0),
             ( -6.5,        345.0),
             ( -2.0,        335.0),
             (  1.0,        312.0),
             ( -3.0,        294.0),
             ( -7.0,        284.0),
             (-10.0,        278.0),
             (-15.0,        272.0),
             (-20.0,        267.0),
             (-30.0,        269.0)],

        # SAA MODEL 28 - ACS MAMA
        # INITIAL ENTRY: CLONE OF MODEL 7, OPR 38805, 4/2/1999
        # UPDATE: CLONE OF MODEL 25.  PR 41631, 5/11/2000
        # UPDATE: CLONE OF MODEL  5.  PR 61893, 2/10/2009
        # UPDATE: CLONE OF MODEL 27.  PR 62292, 3/27/2009
        # UPDATE: SHIFTED WEST 6DEG.  PR 65147, 5/13/2010
        # flux density 15.0
        28: [(-28.5,         19.0),
             (-16.0,          1.0),
             ( -6.5,        345.0),
             ( -2.0,        335.0),
             (  1.0,        312.0),
             ( -3.0,        294.0),
             ( -7.0,        284.0),
             (-10.0,        278.0),
             (-15.0,        272.0),
             (-20.0,        267.0),
             (-30.0,        269.0)],

        # SAA MODEL 29 - WFC3 UVIS CCD
        # INITIAL ENTRY: CLONE OF MODEL 26.  PR 45488,       03/22/02
        # UPDATE: SHIFTED WEST 6DEG.  PR 65147, 5/13/10
        # flux density 15.0
        29: [(-28.5,         19.0),
             (-16.0,          1.0),
             ( -6.5,        345.0),
             ( -2.0,        335.0),
             (  1.0,        312.0),
             ( -3.0,        294.0),
             ( -7.0,        284.0),
             (-10.0,        278.0),
             (-15.0,        272.0),
             (-20.0,        267.0),
             (-30.0,        269.0)],

        # SAA MODEL 30 - WFC3 IR detector
        # INITIAL ENTRY: CLONE OF MODEL 26.  PR 45488,       03/22/02
        # UPDATE: SHIFTED WEST 6DEG.  PR 65147, 5/13/10
        # flux density 15.0
        30: [(-28.5,         19.0),
             (-16.0,          1.0),
             ( -6.5,        345.0),
             ( -2.0,        335.0),
             (  1.0,        312.0),
             ( -3.0,        294.0),
             ( -7.0,        284.0),
             (-10.0,        278.0),
             (-15.0,        272.0),
             (-20.0,        267.0),
             (-30.0,        269.0)],

        # SAA MODEL 31 COS FUV (XDL)
        # INITIAL ENTRY: CLONE OF MODEL 25.  PR 45488,       03/22/02
        # UPDATE:  SHIFTED WEST 6DEG.  PR 65147, 5/13/10
        # flux density 125.0
        31: [(-28.3,         14.0),
             (-27.5,         15.0),
             (-26.1,         13.0),
             (-19.8,          1.5),
             ( -9.6,        341.0),
             ( -7.6,        330.4),
             ( -6.0,        318.8),
             ( -7.9,        297.2),
             (-12.0,        286.1),
             (-17.1,        279.9),
             (-20.3,        277.5),
             (-23.5,        276.5),
             (-26.0,        276.4),
             (-28.6,        276.7)],

        # SAA MODEL 32 COS NUV (MAMA)
        # INITIAL ENTRY: CLONE OF MODEL 25. PR 45488,        03/22/02
        # UPDATE: SHIFTED WEST 6DEG.  PR 65147, 5/13/10
        # flux density 125.0
        32: [(-28.3,         14.0),
             (-27.5,         15.0),
             (-26.1,         13.0),
             (-19.8,          1.5),
             ( -9.6,        341.0),
             ( -7.6,        330.4),
             ( -6.0,        318.8),
             ( -7.9,        297.2),
             (-12.0,        286.1),
             (-17.1,        279.9),
             (-20.3,        277.5),
             (-23.5,        276.5),
             (-26.0,        276.4),
             (-28.6,        276.7)]}

    keys = saa_models.keys()
    keys.sort()
    min_keys = min (keys)
    max_keys = max (keys)
    if model not in keys:
        raise KeyError, "model %d not found in keys (%d - %d)" % \
            (model, min_keys, max_keys)

    return saa_models[model]
