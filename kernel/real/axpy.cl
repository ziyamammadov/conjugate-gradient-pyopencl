
__kernel void axpy(__global float *x,
                   __global float *y,
                   __constant float *a,
                   const int aSign,
                   const int size) {

    const int __idx = get_global_id(0);
    if (__idx < size) {
        if (aSign)
            for (int r = 0; r < N_RHS; r++)
                y[__idx + r * size] += a[r] * x[__idx + r * size];
        else
            for (int r = 0; r < N_RHS; r++)
                y[__idx + r * size] -= a[r] * x[__idx + r * size];
    }
}