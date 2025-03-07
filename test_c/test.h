#ifndef STSCI_STIMAGE_TEST_H
#define STSCI_STIMAGE_TEST_H

#ifdef _WIN32

#include <stdlib.h>

#ifndef srand48
#define srand48 srand
#endif // srand48

#ifndef drand48
#define drand48() rand() / (RAND_MAX + 1.0)
#endif // drand48

#endif // _WIN32

#endif // STSCI_STIMAGE_TEST_H
