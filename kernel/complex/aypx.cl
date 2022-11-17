
#include "cmplx.h"

__kernel void aypx(__global cfloat *x,
                   __global cfloat *y,
                   __constant cfloat *a,
                   const int size) {
    const int __idx = get_global_id(0);
    if (__idx < size)
        for (int r = 0; r < N_RHS; r++)
            y[__idx + r * size] = cmul(a[r], y[__idx + r * size]) + x[__idx + r * size];
}
