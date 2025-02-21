#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "surface/surface.h"
#include "test.h"

int main(int argv, char** argc) {
    surface_t surface;
    surface_t copy;
    bbox_t bbox;
    stimage_error_t error;

    int status = 1;

    stimage_error_init(&error);
    bbox_init(&bbox);

    surface_new(&surface);

    status = surface_init(
            &surface,
            surface_type_polynomial,
            2, 2, xterms_half, &bbox, &error);
    if (status != 0) goto exit;
    if (surface.matrix == NULL) goto exit;

    status = surface_copy(
            &surface,
            &copy,
            &error);

    if (status != 0) goto exit;
    if (copy.matrix == NULL) goto exit;
    if (copy.matrix == surface.matrix) goto exit;

    status = 0;

 exit:
    surface_free(&surface);
    surface_free(&copy);

    if (status) {
        if (error.message[0]) {
            printf(stimage_error_get_message(&error));
        }
    }

    return status;
}
