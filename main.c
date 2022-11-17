#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <complex.h>
#include "clcg.h"

#include <bebop/smc/sparse_matrix.h>
#include <bebop/smc/sparse_matrix_ops.h>

#include <bebop/smc/csr_matrix.h>

int main(int argc, char *argv[]) {

    if (argc != 5) {
        fprintf(stderr, "Usage: ./CG <input matrix file> <number of RHS> <is complex> <number of iterations>");
        exit(1);
    }

    struct sparse_matrix_t *mtx = load_sparse_matrix(MATRIX_MARKET, argv[1]);
    if (mtx == NULL) {
        printf("Could not read matrix\n");
        exit(1);
    }
    sparse_matrix_expand_symmetric_storage(mtx);

    if (sparse_matrix_convert(mtx, CSR)) {
        printf("Could not convert matrix to CSR\n");
        destroy_sparse_matrix(mtx);
        exit(1);
    }
    struct csr_matrix_t *a = mtx->repr;
    int nRHS = atoi(argv[2]);
    int isComplex = atoi(argv[3]);
    int nIterations = atoi(argv[4]);

    float complex *b = (float complex *) malloc(nRHS * a->n * sizeof(float complex));
    float complex *x = (float complex *) malloc(nRHS * a->n * sizeof(float complex));
    // initial values
    for (int r = 0; r < nRHS; r++) {
        for (int i = 0; i < a->n; i++) {
            x[r * a->n + i] = 0.0f + 0.0f * I;
            b[r * a->n + i] = ((r + 1) * 1.0f) + 0.0f * I;
        }
    }
    // Can't handle double precision yet
    float complex *aValues = malloc(a->nnz * sizeof(float complex));
    for (int i = 0; i < a->nnz; i++) aValues[i] = (float complex) ((double complex*) a->values)[i];
    cg(a->n, a->nnz, (const float *) aValues, a->rowptr, a->colidx, (const float *) b, (float *) x, nRHS,nIterations, isComplex);

    destroy_csr_matrix(a);
    free(aValues);
    return 0;
}