include_directories(${XY_COORD_MATCH_INCLUDE_DIR})

add_library(xy_coord_match STATIC
        immatch/lib/tolerance.c
        immatch/lib/triangles.c
        immatch/lib/triangles_vote.c
        immatch/xyxymatch.c
        lib/error.c
        lib/lintransform.c
        lib/util.c
        lib/xycoincide.c
        lib/xysort.c
)

if (NOT MSVC)
    target_link_libraries(xy_coord_match PUBLIC m)
endif()
