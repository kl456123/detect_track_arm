#ifndef __DEFINE_H__
#define __DEFINE_H__
/* borrow code from MNN framework */
#include <stdio.h>


#define PRINT(format, ...) printf(format, ##__VA_ARGS__)
#define ERROR(format, ...) printf(format, ##__VA_ARGS__)

#ifdef DEBUG
#define ASSERT(x)                                            \
    {                                                            \
        int res = (x);                                           \
        if (!res) {                                              \
            ERROR("Error for %s, %d\n", __FILE__, __LINE__); \
            assert(res);                                         \
        }                                                        \
    }
#else
#define ASSERT(x)                                            \
    {                                                            \
        int res = (x);                                           \
        if (!res) {                                              \
            ERROR("Error for %s, %d\n", __FILE__, __LINE__); \
        }                                                        \
    }
#endif

#endif
