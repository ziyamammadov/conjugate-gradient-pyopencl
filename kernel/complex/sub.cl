
#include "cmplx.h"

__kernel void sub(__global const cfloat *a,
                  __global const cfloat *b,
                  __global cfloat *result,
                  const int size) {
    const int __idx = get_global_id(0);
    if (__idx < size)
        for (int r = 0; r < N_RHS; r++)
            result[__idx + r * size] = csub(
                    a[__idx + r * size],
                    b[__idx + r * size]
                    );
}