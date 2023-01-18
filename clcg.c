#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include "clcg.h"
#include <math.h>
#include <time.h>

#include <bebop/smc/sparse_matrix.h>
#include <bebop/smc/sparse_matrix_ops.h>

#include <bebop/smc/csr_matrix.h>
/* BUFFERS
 * aVals     -- constant    (*float of size `aNZ`)
 * aPointers -- constant    (*int of size `aNZ`)
 * aCols     -- constant    (*int of size `aNZ`)
 * b         -- constant    (*float of size `size * nRHS`)
 * x         -- global      (*float of size `size * nRHS`)
 * r         -- global      (*float of size `size * nRHS`)
 * q         -- global      (*float of size `size * nRHS`)
 * d         -- global      (*float of size `size * nRHS`)
 * dotRes    -- global      (*float of size `size / LOCAL_SIZE * nRHS`)
 * dotLocal  -- local       (*float of size `LOCAL_SIZE * nRHS`)
 * mvLocal   -- local       (*float of size `LOCAL_SIZE * nRHS`)
 * alpha     -- constant    (*float of size `nRHS`)
 * beta      -- constant    (*float of size `nRHS`)
 */


/*  KERNELS
 * spmv     -- sparse matrix and dense vector dot product
 * axpy     -- dense vector y += scalar times dense vector x
 * aypx     -- dense vector y = scalar times dense vector y + dense vector x
 * sub      -- dense vector subtraction
 * vdot     -- dense vector dot product, requires CPU reduction
 */

#define LOCAL_SIZE 256
//#define LOCAL_SIZE 128
///#define LOCAL_SIZE 216

// used for spmv, multiple of LOCAL_SIZE
#define WAVE_SIZE 32
////#define WAVE_SIZE 36

#define MAX_ITERATIONS 10
#define MAX_SOURCE_SIZE (0x1000)

#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

void checkClSuccess(int ret, char *text) {
    if (ret != CL_SUCCESS) {
        printf("error -- %s = %d is not %d\n", text, ret, CL_SUCCESS);
    }
}

cl_program buildProgramAndKernels(cl_context ctx, cl_device_id *dId,
                                  cl_kernel *kSpmv, cl_kernel *kAxpy,
                                  cl_kernel *kAypx, cl_kernel *kSub,
                                  cl_kernel *kDot, int nRHS, int isComplex) {
    char *filenames[] = {"spmv.cl", "axpy.cl", "aypx.cl", "vdot.cl", "sub.cl"};
    char *sourceStrings[5];
    char fullPath[30];
    FILE *fp;
    for (int i = 0; i < 5; i++) {
        snprintf(fullPath, 30, "kernel/%s/%s",
                 isComplex ? "complex" : "real", filenames[i]);
        sourceStrings[i] = calloc(MAX_SOURCE_SIZE, sizeof(char));

        fp = fopen(fullPath, "r");
        fread(sourceStrings[i], 1, MAX_SOURCE_SIZE, fp);
        fclose(fp);
    }

    cl_int ret = CL_SUCCESS;
    cl_program program = clCreateProgramWithSource(ctx, 5,
                                                   (const char **) sourceStrings, NULL,
                                                   &ret);

    char programArguments[100];
    sprintf(programArguments, "%s-D N_RHS=%d -D WAVE_SIZE=%d -D WG_SIZE=%d",
            isComplex ? "-I kernel/complex/ " : "",
            nRHS, WAVE_SIZE, LOCAL_SIZE);
    ret = clBuildProgram(program, 1, dId, programArguments, NULL, NULL);
    checkClSuccess(ret, "clBuildProgram");
    if (ret == -11) {
        // Determine the size of the log
        size_t log_size;
        clGetProgramBuildInfo(program, *dId, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        // Allocate memory for the log
        char *log = (char *) malloc(log_size);
        clGetProgramBuildInfo(program, *dId, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("%s\n", log);
        exit(1);
    }
    *kSpmv = clCreateKernel(program, "spmv", &ret);
    checkClSuccess(ret, "clCreateKernel_spmv");
    *kAxpy = clCreateKernel(program, "axpy", &ret);
    checkClSuccess(ret, "clCreateKernel_axpy");
    *kAypx = clCreateKernel(program, "aypx", &ret);
    checkClSuccess(ret, "clCreateKernel_aypx");
    *kSub = clCreateKernel(program, "sub", &ret);
    checkClSuccess(ret, "clCreateKernel_sub");
    *kDot = clCreateKernel(program, "vdot", &ret);
    checkClSuccess(ret, "clCreateKernel_dot");

    return program;
}

float* cg(int size, int nonZeros, const float *aValues, const float *bValues, const int *aPointers,
        const int *aCols, float *x, int nRHS, int nIterations, int isComplex) {

    // printf("Size of nnz: %d\n",nonZeros);
    // for (int i=0;i < 9;i++) {
        // printf("A pointers: %d\n", aPointers[i]);
        // printf("A cols: %d\n", aCols[i]);
        // printf("Avalues: %f%+fi\n", crealf(aValues[i]), cimagf(aValues[i]));
        // printf("B: %f%+fi\n", crealf(bValues[i]), cimagf(bValues[i]));
    // }
    // amount of work-groups.
    // system size / work-group size rounded up
    if (size<LOCAL_SIZE) printf("********** WARNING '''''''''  size less than %d NOT SUPPORTED!!! ********* (size=%d)\n\n",LOCAL_SIZE,size);
    const unsigned int workGroups = 1 + ((size - 1) / LOCAL_SIZE);
    const size_t globalSize = workGroups == 1 // FIXME dot doesn't like localsize which is not a power of 2
                                ? size
                                : workGroups * LOCAL_SIZE;
    const size_t localSize = workGroups == 1
                             ? globalSize
                             : LOCAL_SIZE;

    const size_t rowsPerWg = (LOCAL_SIZE / WAVE_SIZE);
    const size_t spmvWorkGroups = 1 + ((size - 1) / rowsPerWg);
    const size_t spmvGlobalSize = spmvWorkGroups * LOCAL_SIZE;
    const size_t spmvLocalSize = LOCAL_SIZE;

    cl_int ret = CL_SUCCESS;
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    checkClSuccess(
            clGetPlatformIDs(1, &platform_id, &ret_num_platforms),
            "clGetPlatformIDs"
    );
    // for cpu, use CL_DEVICE_TYPE_DEFAULT
    checkClSuccess(
            clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices),
            "clGetDeviceIDs"
    );
    cl_context ctx = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    checkClSuccess(ret, "clCreateContext");

    size_t valSize;
    if (isComplex)
        valSize = sizeof(cl_float2);
    else
        valSize = sizeof(cl_float);
    //printf("valsize=%ld\n",valSize);
    //printf("nRHS=%d\n",nRHS);

    // Create Device memory buffers
    cl_mem dAValues = clCreateBuffer(ctx, CL_MEM_READ_ONLY, nonZeros * valSize, NULL, &ret);
    checkClSuccess(ret, "clCreateBuffer_dAValues");
    cl_mem dACols = clCreateBuffer(ctx, CL_MEM_READ_ONLY, nonZeros * sizeof(cl_int), NULL, &ret);
    checkClSuccess(ret, "clCreateBuffer_dACols");
    cl_mem dAPointers = clCreateBuffer(ctx, CL_MEM_READ_ONLY, (size + 1) * sizeof(cl_int), NULL, &ret);
    checkClSuccess(ret, "clCreateBuffer_dAPointers");
    cl_mem dB = clCreateBuffer(ctx, CL_MEM_READ_ONLY, nRHS * size * valSize, NULL, &ret);
    checkClSuccess(ret, "clCreateBuffer_dB");
    cl_mem dX = clCreateBuffer(ctx, CL_MEM_READ_WRITE, nRHS * size * valSize, NULL, &ret);
    checkClSuccess(ret, "clCreateBuffer_dX");
    cl_mem dR = clCreateBuffer(ctx, CL_MEM_READ_WRITE, nRHS * size * valSize, NULL, &ret);
    checkClSuccess(ret, "clCreateBuffer_dR");
    cl_mem dD = clCreateBuffer(ctx, CL_MEM_READ_WRITE, nRHS * size * valSize, NULL, &ret);
    checkClSuccess(ret, "clCreateBuffer_dD");
    cl_mem dQ = clCreateBuffer(ctx, CL_MEM_READ_WRITE, nRHS * size * valSize, NULL, &ret);
    checkClSuccess(ret, "clCreateBuffer_dQ");
    cl_mem dDotRes = clCreateBuffer(ctx, CL_MEM_READ_WRITE, nRHS * workGroups * valSize, NULL, &ret);
    checkClSuccess(ret, "clCreateBuffer_dDivRes");
    cl_mem dAxpyConst = clCreateBuffer(ctx, CL_MEM_READ_ONLY, nRHS * valSize, NULL, &ret);
    checkClSuccess(ret, "clCreateBuffer_dDivRes");

    cl_command_queue cq = clCreateCommandQueue(ctx, device_id, 0, &ret);
    checkClSuccess(ret, "clCreateCommandQueue");

    // Allocate Host memory buffers
    void *hDotRes = malloc(nRHS * workGroups * valSize);
    void *hDq = malloc(nRHS * valSize);
    void *hAlpha = malloc(nRHS * valSize);
    void *hBeta = malloc(nRHS * valSize);
    void *deltaNew = calloc(nRHS, valSize);
    //void *deltaNew = malloc(nRHS * valSize);
    void *deltaOld = malloc(nRHS * valSize);
    int negative = 0;
    int positive = 1;
    /// for debugging:
    ////void *alphas = malloc(nRHS * valSize * nIterations );
    ///

    // Copy over the constant and initial values
    checkClSuccess(clEnqueueWriteBuffer(cq, dAValues, CL_FALSE, 0, nonZeros * valSize, aValues, 0, NULL, NULL),
                   "clEnqueueWriteBuffer_aValues");
    checkClSuccess(clEnqueueWriteBuffer(cq, dAPointers, CL_FALSE, 0, (size + 1) * sizeof(cl_int), aPointers, 0, NULL, NULL),
                   "clEnqueueWriteBuffer_aPointers");
    checkClSuccess(clEnqueueWriteBuffer(cq, dACols, CL_FALSE, 0, nonZeros * sizeof(cl_int), aCols, 0, NULL, NULL),
                   "clEnqueueWriteBuffer_aCols");
    checkClSuccess(clEnqueueWriteBuffer(cq, dB, CL_FALSE, 0, nRHS * size * valSize, bValues, 0, NULL, NULL),
                   "clEnqueueWriteBuffer_b");
    checkClSuccess(clEnqueueWriteBuffer(cq, dX, CL_FALSE, 0, nRHS * size * valSize, x, 0, NULL, NULL),
                   "clEnqueueWriteBuffer_x");

    cl_kernel kSpmv, kAxpy, kAypx, kSub, kDot;
    cl_program program = buildProgramAndKernels(ctx, &device_id, &kSpmv, &kAxpy, &kAypx, &kSub, &kDot, nRHS, isComplex);

    // Set initial kernel arguments
    // spmv
    checkClSuccess(clSetKernelArg(kSpmv, 0, sizeof(cl_int), &size), "clSetKernelArg_spmv_size");
    checkClSuccess(clSetKernelArg(kSpmv, 1, sizeof(cl_mem), &dAValues), "clSetKernelArg_spmv_dAValues_outer");
    checkClSuccess(clSetKernelArg(kSpmv, 2, sizeof(cl_mem), &dAPointers), "clSetKernelArg_spmv_dAPointers_outer");
    checkClSuccess(clSetKernelArg(kSpmv, 3, sizeof(cl_mem), &dACols), "clSetKernelArg_spmv_dACols_outer");
    checkClSuccess(clSetKernelArg(kSpmv, 4, sizeof(cl_mem), &dX), "clSetKernelArg_spmv_dX_outer");
    checkClSuccess(clSetKernelArg(kSpmv, 5, sizeof(cl_mem), &dQ), "clSetKernelArg_spmv_dQ_outer");
    checkClSuccess(clSetKernelArg(kSpmv, 6, nRHS * spmvLocalSize * valSize, NULL), "clSetKernelArg_spmv_local_outer");

    // sub
    checkClSuccess(clSetKernelArg(kSub, 0, sizeof(cl_mem), &dB), "clSetKernelArg_sub_dB");
    // Buffer Q only used to avoid creating another buffer.
    checkClSuccess(clSetKernelArg(kSub, 1, sizeof(cl_mem), &dQ), "clSetKernelArg_sub_dQ");
    checkClSuccess(clSetKernelArg(kSub, 2, sizeof(cl_mem), &dR), "clSetKernelArg_sub_dR");
    checkClSuccess(clSetKernelArg(kSub, 3, sizeof(cl_int), &size), "clSetKernelArg_sub_size");

    // dot
    checkClSuccess(clSetKernelArg(kDot, 0, sizeof(cl_mem), &dR), "clSetKernelArg_dot_dR_1");
    checkClSuccess(clSetKernelArg(kDot, 1, sizeof(cl_mem), &dR), "clSetKernelArg_dot_dR_2");
    checkClSuccess(clSetKernelArg(kDot, 2, nRHS * localSize * valSize, NULL), "clSetKernelArg_dot_local");
    checkClSuccess(clSetKernelArg(kDot, 3, sizeof(cl_mem), &dDotRes), "clSetKernelArg_dot_result");
    checkClSuccess(clSetKernelArg(kDot, 4, sizeof(cl_int), &size), "clSetKernelArg_dot_size");

    // axpy & aypx
    checkClSuccess(clSetKernelArg(kAxpy, 2, sizeof(cl_mem), &dAxpyConst), "clSetKernelArg_alpha");
    checkClSuccess(clSetKernelArg(kAxpy, 4, sizeof(cl_int), &size), "clSetKernelArg_axpy_size");
    checkClSuccess(clSetKernelArg(kAypx, 0, sizeof(cl_mem), &dR), "clSetKernelArg_d");
    checkClSuccess(clSetKernelArg(kAypx, 1, sizeof(cl_mem), &dD), "clSetKernelArg_x");
    checkClSuccess(clSetKernelArg(kAypx, 2, sizeof(cl_mem), &dAxpyConst), "clSetKernelArg_beta");
    checkClSuccess(clSetKernelArg(kAypx, 3, sizeof(cl_int), &size), "clSetKernelArg_aypx_size");

    cl_event waitKSpmv, waitKSub, waitKDot, waitKAxpy, waitKAypx, waitCopy;

    // INITIALIZATION END
    // CG START

    // r = b - Ax
    //      y = A * x                   (spmv)
    checkClSuccess(clEnqueueNDRangeKernel(cq, kSpmv, 1, NULL, &spmvGlobalSize, &spmvLocalSize, 0, NULL, &waitKSpmv),
                   "clEnqueueNDRangeKernel_spmv_1");


    // r = b - y                   (sub)
    checkClSuccess(clEnqueueNDRangeKernel(cq, kSub, 1, NULL, &globalSize, &localSize, 1, &waitKSpmv, &waitKSub),
                   "clEnqueueNDRangeKernel_sub");

    // d = r
    checkClSuccess(clEnqueueCopyBuffer(cq, dR, dD, 0, 0, nRHS * size * valSize, 1, &waitKSub, &waitCopy),
                   "clEnqueueCopyBuffer_r->d");

    // deltaNew = r^T * r               (dot) reduce in cpu
    checkClSuccess(clEnqueueNDRangeKernel(cq, kDot, 1, NULL, &globalSize, &localSize, 1, &waitCopy, &waitKDot),
                   "clEnqueueNDRangeKernel_dot_1");
    checkClSuccess(clEnqueueReadBuffer(cq, dDotRes, CL_TRUE, 0, nRHS * workGroups * valSize, hDotRes, 1, &waitKDot, NULL),
                   "clEnqueueReadBuffer_dot_result");

    // printf("initial \t\tdelta {");
    for (int r = 0; r < nRHS; r++) {
        // Reduction
        for (int i = 0; i < workGroups; i++) {
            if (isComplex) ((float complex *) deltaNew)[r] += ((float complex *) hDotRes)[r * workGroups + i];
            else ((float *) deltaNew)[r] += ((float *) hDotRes)[r * workGroups + i];
        }
        //if (isComplex) printf("deltaNew= %6f %+.6fi\n",crealf(((float complex *) deltaNew)[r]),cimagf(((float complex *) deltaNew)[r]));
        if (isComplex) {
            ((float complex *) deltaOld)[r] = ((float complex *) deltaNew)[r];
            // printf(" %s%.4e%s%.4ei ",
                //    creal(((float complex *) deltaNew)[r]) < 0 ? "" : " ",
                //    creal(((float complex *) deltaNew)[r]),
                //    cimag(((float complex *) deltaNew)[r]) < 0 ? "" : "+",
                //    cimag(((float complex *) deltaNew)[r]));
        } else {
            ((float *) deltaOld)[r] = ((float *) deltaNew)[r];
            // printf(" %.5e ", ((float *) deltaNew)[r]);
        }
    }
    // printf("}\n");

    // While not converged
    int iteration = 0;
    while (iteration < nIterations) {
        // q = A * d                    (spmv)
        checkClSuccess(clSetKernelArg(kSpmv, 4, sizeof(cl_mem), &dD), "clSetKernelArg_spmv_dD");
        if (iteration == 0)
            checkClSuccess(clEnqueueNDRangeKernel(cq, kSpmv, 1, NULL, &spmvGlobalSize, &spmvLocalSize, 0, NULL, &waitKSpmv),
                           "clEnqueueNDRangeKernel_spmv_q=Ad");
        else
            checkClSuccess(clEnqueueNDRangeKernel(cq, kSpmv, 1, NULL, &spmvGlobalSize, &spmvLocalSize, 1, &waitKAypx, &waitKSpmv),
                           "clEnqueueNDRangeKernel_spmv_q=Ad");
    ///// Load to CPU the result of Ax op: q
        // alpha = deltaNew / (d * q)
        //      dq = d * q              (dot) reduce in cpu
        checkClSuccess(clSetKernelArg(kDot, 0, sizeof(cl_mem), &dD), "clSetKernelArg_dot_dD_1");
        checkClSuccess(clSetKernelArg(kDot, 1, sizeof(cl_mem), &dQ), "clSetKernelArg_dot_dQ_2");
    ///(void)sleep(0.0);
        checkClSuccess(clEnqueueNDRangeKernel(cq, kDot, 1, NULL, &globalSize, &localSize, 1, &waitKSpmv, &waitKDot),
                       "clEnqueueNDRangeKernel_dot_DQ");
        checkClSuccess(clEnqueueReadBuffer(cq, dDotRes, CL_TRUE, 0, nRHS * workGroups * valSize, hDotRes, 1, &waitKDot, NULL),
                       "clEnqueueReadBuffer_dot_DQresult");

        for (int r = 0; r < nRHS; r++) {
            if (isComplex) ((float complex *) hDq)[r] = 0.0f + 0.0f * I;
            else ((float *) hDq)[r] = 0.0f;
            // Reduction
            for (int i = 0; i < workGroups; i++) {
                if (isComplex) ((float complex *) hDq)[r] += ((float complex*) hDotRes)[r * workGroups + i];
                else ((float *) hDq)[r] += ((float *) hDotRes)[r * workGroups + i];
            }
            //      alpha = deltaNew / dq   in cpu
            if (isComplex) ((float complex *) hAlpha)[r] = ((float complex *) deltaNew)[r] / ((float complex *) hDq)[r];
            else ((float *) hAlpha)[r] = ((float *) deltaNew)[r] / ((float *) hDq)[r];
            ////
            ////if (isComplex) printf("deltaNew= %6f %+.6fi\n",crealf(((float complex *) deltaNew)[r]),cimagf(((float complex *) deltaNew)[r]));
            //if (isComplex) printf("%d hAlpha= %6f %+.6fi\n",iteration, crealf(((float complex *) hAlpha)[r]),cimagf(((float complex *) hAlpha)[r]));
            ////
            ////if (isComplex) ((float complex *)alphas)[iteration*nRHS+r]=((float complex *) hAlpha)[r];
        }
        checkClSuccess(clEnqueueWriteBuffer(cq, dAxpyConst, CL_TRUE, 0, nRHS * valSize, hAlpha, 0, NULL, NULL),
                       "clEnqueueWriteBuffer_axpyConst");

        // x = alpha * d + x            (axpy)
        checkClSuccess(clSetKernelArg(kAxpy, 0, sizeof(cl_mem), &dD), "clSetKernelArg_d");
        checkClSuccess(clSetKernelArg(kAxpy, 1, sizeof(cl_mem), &dX), "clSetKernelArg_x");
        checkClSuccess(clSetKernelArg(kAxpy, 3, sizeof(cl_int), &positive), "clSetKernelArg_axpySign1");
        checkClSuccess(clEnqueueNDRangeKernel(cq, kAxpy, 1, NULL, &globalSize, &localSize, 0, NULL, NULL),
                       "clEnqueueNDRangeKernel_axpy_alphaDX");

        // r = - alpha * q + r          (axpy)
        checkClSuccess(clSetKernelArg(kAxpy, 0, sizeof(cl_mem), &dQ), "clSetKernelArg_d");
        checkClSuccess(clSetKernelArg(kAxpy, 1, sizeof(cl_mem), &dR), "clSetKernelArg_x");
        checkClSuccess(clSetKernelArg(kAxpy, 3, sizeof(cl_int), &negative), "clSetKernelArg_axpySign2");
        checkClSuccess(clEnqueueNDRangeKernel(cq, kAxpy, 1, NULL, &globalSize, &localSize, 0, NULL, &waitKAxpy),
                       "clEnqueueNDRangeKernel_axpy_alphaQR");
        for (int r = 0; r < nRHS; r++) {
            // deltaOld = deltaNew
            if (isComplex) ((float complex *) deltaOld)[r] = ((float complex *) deltaNew)[r];
            else ((float *) deltaOld)[r] = ((float *) deltaNew)[r];

            if (isComplex) ((float complex *) deltaNew)[r] = 0.0f + 0.0f * I;
            else ((float *) deltaNew)[r] = 0.0f;
            // // Reduction
            // for (int i = 0; i < workGroups; i++) {
            //     if (isComplex) ((float complex *) deltaNew)[r] += ((float complex *) hDotRes)[r * workGroups + i];
            //     else ((float *) deltaNew)[r] += ((float *) hDotRes)[r * workGroups + i];
            // }
            // beta = deltaNew / deltaOld   in cpu
            // if (isComplex)
            //     ((float complex *) hBeta)[r] = ((float complex *) deltaNew)[r] / ((float complex *) deltaOld)[r];
            // else ((float *) hBeta)[r] = ((float *) deltaNew)[r] / ((float *) deltaOld)[r];
        }

        // deltaNew = r^T * r           (dot) reduce in cpu
        checkClSuccess(clSetKernelArg(kDot, 0, sizeof(cl_mem), &dR), "clSetKernelArg_dot_dD_1");
        checkClSuccess(clSetKernelArg(kDot, 1, sizeof(cl_mem), &dR), "clSetKernelArg_dot_dQ_2");
        checkClSuccess(clEnqueueNDRangeKernel(cq, kDot, 1, NULL, &globalSize, &localSize, 1, &waitKAxpy, &waitKDot),
                       "clEnqueueNDRangeKernel_dot_RR");
        checkClSuccess(clEnqueueReadBuffer(cq, dDotRes, CL_TRUE, 0, nRHS * workGroups * valSize, hDotRes, 1, &waitKDot, NULL),
                       "clEnqueueReadBuffer_dot_Rresult");

        for (int r = 0; r < nRHS; r++) {
            // // deltaOld = deltaNew
            // if (isComplex) ((float complex *) deltaOld)[r] = ((float complex *) deltaNew)[r];
            // else ((float *) deltaOld)[r] = ((float *) deltaNew)[r];

            // if (isComplex) ((float complex *) deltaNew)[r] = 0.0f + 0.0f * I;
            // else ((float *) deltaNew)[r] = 0.0f;
            // Reduction
            for (int i = 0; i < workGroups; i++) {
                if (isComplex) ((float complex *) deltaNew)[r] += ((float complex *) hDotRes)[r * workGroups + i];
                else ((float *) deltaNew)[r] += ((float *) hDotRes)[r * workGroups + i];
            }
            // beta = deltaNew / deltaOld   in cpu
            if (isComplex)
                ((float complex *) hBeta)[r] = ((float complex *) deltaNew)[r] / ((float complex *) deltaOld)[r];
            else ((float *) hBeta)[r] = ((float *) deltaNew)[r] / ((float *) deltaOld)[r];
        }

        // printf("iteration %d \tdelta {", iteration);
        // for (int r = 0; r < nRHS; r++) {
            // if (isComplex)
                // printf(" %s%.4e%s%.4ei ",
                    //    creal(((float complex *) deltaNew)[r]) < 0 ? "" : " ",
                    //    creal(((float complex *) deltaNew)[r]),
                    //    cimag(((float complex *) deltaNew)[r]) < 0 ? "" : "+",
                    //    cimag(((float complex *) deltaNew)[r]));
            // else
                // printf(" %.5e ", ((float *) deltaNew)[r]);
        // }
        // printf("}\n");
//        printf("   delta = %e\n", deltaNew[2]);
//        printf("      dq = %e\n", hDq[2]);
//        printf("   alpha = %e\n", hAlpha[2]);
//        printf("    beta = %e\n", hBeta[2]);

        checkClSuccess(clEnqueueWriteBuffer(cq, dAxpyConst, CL_TRUE, 0, nRHS * valSize, hBeta, 0, NULL, NULL),
                       "clEnqueueWriteBuffer_axpyConst_beta");

        // d = beta * d + r             (aypx)
        checkClSuccess(clEnqueueNDRangeKernel(cq, kAypx, 1, NULL, &globalSize, &localSize, 0, NULL, &waitKAypx),
                       "clEnqueueNDRangeKernel_aypx_alphaQR");

        iteration++;
    }
    ////for (int r = 0; r < nRHS; r++) {
    ////    for (iteration=0; iteration < nIterations; iteration++) {
    ////        if (isComplex) printf("%d: %d hAlpha= %6f %+.6fi\n",r,iteration,
    ////        crealf(((float complex *) alphas)[iteration*nRHS+r]),cimagf(((float complex *) alphas)[iteration*nRHS+r]));
    ////    }
    ////}
    checkClSuccess(clEnqueueReadBuffer(cq, dX, CL_TRUE, 0, nRHS * size * valSize, x, 0, NULL, NULL),
                   "clEnqueueReadBuffer_x_result");

    checkClSuccess(clFlush(cq), "flush_commandqueue");
    checkClSuccess(clFinish(cq), "finish_commandqueue");

    checkClSuccess(clReleaseMemObject(dR), "release_r");
    checkClSuccess(clReleaseMemObject(dQ), "release_q");
    checkClSuccess(clReleaseMemObject(dD), "release_d");
    checkClSuccess(clReleaseMemObject(dB), "release_b");
    checkClSuccess(clReleaseMemObject(dX), "release_x");
    checkClSuccess(clReleaseMemObject(dDotRes), "release_dotres");
    checkClSuccess(clReleaseMemObject(dAValues), "release_avalues");
    checkClSuccess(clReleaseMemObject(dACols), "release_acols");
    checkClSuccess(clReleaseMemObject(dAPointers), "release_arows");

    checkClSuccess(clReleaseCommandQueue(cq), "release_commandqueue");
    checkClSuccess(clReleaseProgram(program), "release_program");

    checkClSuccess(clReleaseKernel(kDot), "release_dot");
    checkClSuccess(clReleaseKernel(kSpmv), "release_spmv");
    checkClSuccess(clReleaseKernel(kAxpy), "release_axpy");
    checkClSuccess(clReleaseKernel(kSub), "release_sub");
    checkClSuccess(clReleaseKernel(kAypx), "release_aypx");

    checkClSuccess(clReleaseEvent(waitKDot), "release_dotevent");
    checkClSuccess(clReleaseEvent(waitKSpmv), "release_spmvevent");
    checkClSuccess(clReleaseEvent(waitKAxpy), "release_axpyevent");
    checkClSuccess(clReleaseEvent(waitKAypx), "release_aypxevent");
    checkClSuccess(clReleaseEvent(waitKSub), "release_subevent");
    checkClSuccess(clReleaseEvent(waitCopy), "release_copyevent");

    checkClSuccess(clReleaseContext(ctx), "release_context");
    checkClSuccess(clReleaseDevice(device_id), "release_device");
    // for (int i=0;i < 6;i++) {
    //     printf("%.1f%+.1fi\n", crealf(x[i]), cimagf(x[i]));
        // printf("%.1f\n", x[i]);
    // }
    free(hDotRes);
    return x;
}
