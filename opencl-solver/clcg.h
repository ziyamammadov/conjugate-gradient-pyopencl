#ifndef OCLCG_CLCG_H
#define OCLCG_CLCG_H
void cg(int size, int nonZeros,
        const float *aValues, const int *aPointers, const int *aCols,
        const float *b, float *x, int nRHS, int isComplex);

#endif //OCLCG_CLCG_H
