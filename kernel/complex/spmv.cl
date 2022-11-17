
#include "cmplx.h"

// Block-SpMV for tall & slim vector, e.g. multiple RHS
// One warp/wave/wavefront per row.
// Matrix format: CSR
__kernel void spmv(const int size,
                   __global const cfloat *aValues,
                   __global const int *aPointers,
                   __global const int *aCols,
                   __global const cfloat *x,
                   __global cfloat *y,
                   __local cfloat *localSums) {

    const int __idx = get_global_id(0);                   // global thread index
    const int __localIdx = get_local_id(0);               // local thread index
    const int threadLane = __localIdx & (WAVE_SIZE - 1);  // thread index within the wave
    const int waveId = __idx / WAVE_SIZE;                 // global wave index

    const int row_start = aPointers[waveId];
    const int row_end = aPointers[waveId + 1];

    // compute private sums
    cfloat sum[N_RHS] = {(cfloat) (0.0f, 0.0f)};
    for (int j = row_start + threadLane; j < row_end; j += WAVE_SIZE)
        for (int r = 0; r < N_RHS; r++)
            sum[r] = cadd(sum[r],
                          cmul(aValues[j],
                               x[aCols[j] + r * size]));

    // write to local array for other work-items to access
    for (int r = 0; r < N_RHS; r++)
        localSums[__localIdx + r * WG_SIZE] = sum[r];

    // reduce local sums to row sum
    int mask = 1;
    for (int offset = 1; offset < WAVE_SIZE;) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if ((threadLane & mask) == 0)
            for (int r = 0; r < N_RHS; r++)
                localSums[__localIdx + r * WG_SIZE] =
                        cadd(localSums[__localIdx + r * WG_SIZE],
                             localSums[__localIdx + r * WG_SIZE + offset]);
        offset <<= 1;
        mask <<= 1;
        mask += 1;
    }

    // first thread of wave writes warp result
    if (threadLane == 0 && waveId < size)
        for (int r = 0; r < N_RHS; r++)
            y[waveId + r * size] = localSums[__localIdx + r * WG_SIZE];
}