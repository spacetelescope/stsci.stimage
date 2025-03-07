include_directories(${STIMAGE_INCLUDE_DIR})

add_library(stimage STATIC
        immatch/lib/tolerance.c
        immatch/lib/triangles.c
        immatch/lib/triangles_vote.c
        immatch/geomap.c
        immatch/xyxymatch.c
        lib/error.c
        lib/lintransform.c
        lib/polynomial.c
        lib/util.c
        lib/xybbox.c
        lib/xycoincide.c
        lib/xysort.c
        surface/surface.c
        surface/vector.c
        surface/cholesky.c
        surface/fit.c
)

if (NOT MSVC)
    target_link_libraries(stimage PUBLIC m)
endif()