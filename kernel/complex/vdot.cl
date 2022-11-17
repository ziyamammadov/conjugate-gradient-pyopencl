
#include "cmplx.h"

__kernel void vdot(__global const cfloat *a,
                   __global const cfloat *b,
                   __local cfloat *localSums,
                   __global cfloat *globalSums,
                   const int size) {

    const int __idx = get_global_id(0);
    const int __localIdx = get_local_id(0);

    if (__idx < size)
        for (int r = 0; r < N_RHS; r++)
            localSums[__localIdx + r * WG_SIZE] = cmul(a[__idx + r * size],
                                                       b[__idx + r * size]);
    else
        for (int r = 0; r < N_RHS; r++)
            localSums[__localIdx + r * WG_SIZE] = (cfloat) (0.0f, 0.0f);

    int mask = 1;
    for (int offset = 1; offset < WG_SIZE;) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if ((__localIdx & mask) == 0)
            for (int r = 0; r < N_RHS; r++)
                localSums[__localIdx + r * WG_SIZE] =
                        cadd(localSums[__localIdx + r * WG_SIZE],
                             localSums[__localIdx + r * WG_SIZE + offset]);
        offset <<= 1;
        mask <<= 1;
        mask += 1;
    }

    const int wgNum = get_group_id(0);
    const int wgCount = get_global_size(0) / WG_SIZE;
    if (__localIdx == 0) {
        for (int r = 0; r < N_RHS; r++) {
            globalSums[wgNum + r * wgCount] = localSums[r * WG_SIZE];
        }
    }
}