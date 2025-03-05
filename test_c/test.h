#ifndef STSCI_STIMAGE_TEST_H
#define STSCI_STIMAGE_TEST_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <unistd.h>
#include <float.h>

#define TEST_DATA_FMT "%lf"

extern inline const char *get_test_data_dir() {
    const char *datadir = getenv("STIMAGE_TEST_DATA");
    if (!datadir) {
        fprintf(stderr, "error: STIMAGE_TEST_DATA environment variable must be set\n");
        exit(1);
    }
    return datadir;
}

extern inline double iter_test_data(double **data) {
    double result = **data;
    (*data)++;
    return result;
}

extern inline void get_test_data(const char *filename, double **result, size_t nelem) {
    const char *datadir = get_test_data_dir();
    char *path = NULL;
    if (asprintf(&path, "%s/%s", datadir, filename) < 2) {
        perror("path string creation failed");
        exit(1);
    }

    *result = malloc(nelem * sizeof(**result));
    if (!*result) {
        perror("unable to allocate bytes for result array");
        exit(1);
    }

    FILE *fp = fopen(path, "r");
    if (!fp) {
        perror(path);
        exit(1);
    }

    char value_buf[255] = {0};
    size_t i = 0;
    while (i < nelem && fgets(value_buf, sizeof(value_buf) - 1, fp) != NULL) {
        // truncate line feeds
        value_buf[strcspn(value_buf, "\r\n")] = 0;

        char *value_ptr = NULL;
        double value = strtod(value_buf, &value_ptr);
        if (!value_ptr && !value) {
            fprintf(stderr, "invalid value in %s: '%s'\n", path, value_buf);
            fclose(fp);
            free(path);
            exit(1);
        }

        *(*result + i) = value;
        i++;
    }

    fclose(fp);
    free(path);
}

#endif // STSCI_STIMAGE_TEST_H
