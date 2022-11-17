
__kernel void aypx(__global float *x,
                   __global float *y,
                   __constant float *a,
                   const int size) {
    const int __idx = get_global_id(0);
    if (__idx < size)
        for (int r = 0; r < N_RHS; r++)
            y[__idx + r * size] = y[__idx + r * size] * a[r] + x[__idx + r * size];
}
