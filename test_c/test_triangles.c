#include <stdio.h>

#include "immatch/lib/triangles.h"
#include "lib/xysort.h"
#include "lib/xycoincide.h"
#include "test.h"

int main(int argc, char** argv) {
    #define ncoords 512
    coord_t data1[ncoords];
    coord_t data2[ncoords];
    const coord_t* ptr1[ncoords];
    const coord_t* ptr2[ncoords];
    size_t ncoord_matches = ncoords;
    const coord_t* ref_matches[ncoords];
    const coord_t* input_matches[ncoords];
    size_t ntriangles1;
    triangle_t* triangles1 = NULL;
    size_t ntriangles2;
    triangle_t* triangles2 = NULL;
    triangle_t* trans_triangles = NULL;
    triangle_t* tri = NULL;
    size_t ntriangle_matches;
    triangle_match_t* triangle_matches = NULL;
    size_t nunique;
    const double tolerance = 0.0001;
    const double max_ratio = 10.0;
    const size_t max_points = 30;
    const size_t nreject = 10;
    const double tol2 = tolerance*tolerance;
    double dist[3];
    stimage_error_t error;
    double last_ratio;
    int status = 1;

    size_t i = 0;
    size_t j = 0;

    stimage_error_init(&error);

    srand48(0);

    for (i = 0; i < ncoords; ++i) {
        data1[i].x = data2[i].x = drand48();
        data1[i].y = data2[i].y = drand48();
    }

    xysort(ncoords, data1, ptr1);
    xysort(ncoords, data2, ptr2);
    nunique = xycoincide(ncoords, ptr1, ptr1, tolerance);

    if (max_num_triangles(nunique, max_points, &ntriangles1, &error)) {
        goto exit;
    }
    ntriangles2 = ntriangles1;
    printf("Allocating room for %lu triangles\n", (long unsigned)ntriangles1);
    triangles1 = malloc(sizeof(triangle_t) * ntriangles1);
    if (triangles1 == NULL) {
        goto exit;
    }
    triangles2 = malloc(sizeof(triangle_t) * ntriangles2);
    if (triangles1 == NULL) {
        goto exit;
    }

    if (find_triangles(
            nunique, ptr1, &ntriangles1, triangles1, max_points,
            tolerance, max_ratio, &error)) {
        goto exit;
    }

    if (find_triangles(
            nunique, ptr2, &ntriangles2, triangles2, max_points,
            tolerance, max_ratio, &error)) {
        goto exit;
    }

    printf("Found %lu triangles\n", (unsigned long)ntriangles1);

    /* Print some random triangles, just for kicks */
    for (i = ntriangles1-10; i < ntriangles1; ++i) {
        tri = &triangles1[i];
        printf("Triangle %lu:\n", (unsigned long)i);

        printf("   (%.3f, %.3f)--(%.3f, %.3f)--(%.3f, %.3f)\n",
               tri->vertices[0]->x, tri->vertices[0]->y,
               tri->vertices[1]->x, tri->vertices[1]->y,
               tri->vertices[2]->x, tri->vertices[2]->y);
        printf("   ");
        for (j = 0; j < 3; ++j) {
            dist[j] = euclid_distance2(tri->vertices[j], tri->vertices[(j+1)%3]);
            printf("%f ", dist[j]);
        }
        printf("\n");
        printf("   log_perimeter:    %.3f\n", tri->log_perimeter);
        printf("   ratio:            %.3f\n", tri->ratio);
        printf("   cosine_v1:        %.3f\n", tri->cosine_v1);
        printf("   ratio_tolerance:  %.3f\n", tri->ratio_tolerance);
        printf("   cosine_tolerance: %.3f\n", tri->cosine_tolerance);
        printf("   sense:            %d\n", tri->sense);
        printf("\n");
    }

    last_ratio = triangles1[0].ratio;
    for (i = 1; i < ntriangles1; ++i) {
        tri = &triangles1[i];

        if (tri->ratio < last_ratio) {
            printf ("Ratios are not sorted\n");
            goto exit;
        }
        if (tri->ratio > max_ratio) {
            printf("Ratio larger than max_ratio\n");
            goto exit;
        }

        for (j = 0; j < 3; ++j) {
            dist[j] = euclid_distance2(tri->vertices[j], tri->vertices[(j+1)%3]);
            if (dist[j] <= tol2) {
                printf("Distances too short\n");
                goto exit;
            }
        }

        if (dist[0] > dist[1] || dist[1] > dist[2]) {
            printf("Distances in the wrong order\n");
            goto exit;
        }
    }

    /* Merging triangles with the same set should result in all exact matches */
    ntriangle_matches = ntriangles1;
    printf("Allocating room for %lu matches\n", (unsigned long)ntriangle_matches);
    triangle_matches = malloc(sizeof(triangle_match_t) * ntriangle_matches);
    if (triangle_matches == NULL) {
        goto exit;
    }

    if (merge_triangles(
            ntriangles1, triangles1, ntriangles2, triangles2,
            &ntriangle_matches, triangle_matches, &error)) {
        goto exit;
    }

    if (ntriangle_matches != ntriangles1) {
        printf("Found %lu instead of %lu when self-matching\n",
               (unsigned long)ntriangle_matches, (unsigned long)ntriangles1);
        goto exit;
    }

    if (reject_triangles(
            &ntriangle_matches, triangle_matches, nreject, &error)) {
        goto exit;
    }

    if (ntriangle_matches != ntriangles1) {
        printf("Found %lu instead of %lu after rejection\n",
               (unsigned long)ntriangle_matches, (unsigned long)ntriangles1);
        goto exit;
    }

    if (vote_triangle_matches(
            ncoords, data2,
            ncoords, data1,
            ntriangle_matches, triangle_matches,
            &ncoord_matches, ref_matches, input_matches,
            &error)) {
        goto exit;
    }

    if (ncoord_matches != max_points) {
        printf("Found %lu instead of %lu after voting\n",
               (unsigned long)ncoord_matches, (unsigned long)max_points);
        goto exit;
    }

    for (i = 0; i < ncoord_matches; ++i) {
        if (ref_matches[i]->x != input_matches[i]->x ||
            ref_matches[i]->y != input_matches[i]->y) {
            printf("Mismatched coordinates\n");
            goto exit;
        }

        /* Compare the indices -- they should also be the same */
        if (ref_matches[i] - data2 != input_matches[i] - data1) {
            printf("Mismatched pointers (%lu vs. %lu)\n",
                   ref_matches[i] - data1,
                   input_matches[i] - data2);
            goto exit;
        }
    }

    status = 0;

 exit:
    free(triangles1);
    free(triangles2);
    free(triangle_matches);
    free(trans_triangles);

    if (status) {
        if (error.message[0]) {
            printf(stimage_error_get_message(&error));
        }
    }

    return status;
}
