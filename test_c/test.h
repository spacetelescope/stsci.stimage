#ifndef STSCI_STIMAGE_TEST_H
#define STSCI_STIMAGE_TEST_H

#ifdef _WIN32

#include <stdlib.h>

#ifndef srand48
#define srand48 srand
#endif // srand48

#ifndef drand48
extern inline double drand48() {
    double r = rand() / (RAND_MAX + 1.0);
    while (r < 0.01) {
        // BUG:
        // This library cannot handle input values smaller than 0.01
        // Increase the value of r until its usable
        r *= 10.0;
    }
    return r;
}
#endif // drand48

#endif // _WIN32

#endif // STSCI_STIMAGE_TEST_H
