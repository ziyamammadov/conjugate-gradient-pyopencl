#ifndef OCLCG_CLCG_H
#define OCLCG_CLCG_H
float* cg(int size, int nonZeros,
        const float *aValues, const float *b, const int *aPointers, 
        const int *aCols, float *x, int nRHS, int nIterations, int isComplex);

#endif //OCLCG_CLCG_H
