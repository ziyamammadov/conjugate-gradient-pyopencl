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

    if (argc != 2) {
        fprintf(stderr, "Usage: ./CG <input matrix file>");
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
//    print_csr_matrix_in_matrix_market_format(stdout, a);

    int nRHS = 4;
//
//    float *b = (float *) malloc(nRHS * a->n * sizeof(float));
//    float *x = (float *) malloc(nRHS * a->n * sizeof(float));
//    // initial values
//    for (int r = 0; r < nRHS; r++) {
//        for (int i = 0; i < a->n; i++) {
//            x[r * a->n + i] = 0.0f;
//            b[r * a->n + i] = (r + 1) * 1.0f;
//        }
//    }
//
//    // Can't handle double precision yet
//    float *aValues = malloc(a->nnz * sizeof(float));
//    for (int i = 0; i < a->nnz; i++) aValues[i] = (float) ((double*) a->values)[i];
//
//    cg(a->n, a->nnz, aValues, a->rowptr, a->colidx, b, x, nRHS, 0);

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

    cg(a->n, a->nnz, (const float *) aValues, a->rowptr, a->colidx, (const float *) b, (float *) x, nRHS, 1);

    destroy_csr_matrix(a);
    free(aValues);
    return 0;
}
