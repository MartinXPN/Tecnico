#ifndef PDC_UTIL_H
#define PDC_UTIL_H


#include <cstdlib>
#include "matrix.h"

#define db(X) cerr << #X << ": " << X << endl
#define dba(X) cerr << #X << ": "; for(auto i: X)   cerr << i << " ";   cerr << endl
#define db2d(X) cerr << #X << ": \n"; for(auto x: X) { for( auto i : x )  cerr << i << " "; cerr << endl; }  cerr << endl
#define RAND01 ((double)random()) / ((double)RAND_MAX)


void random_fill_LR(Matrix& L, Matrix& R,int nU, int nI, int nF)
{
    srand(0);
    for(int i = 0; i < nU; i++)
        for(int j = 0; j < nF; j++)
            L[i][j] = RAND01 / (double) nF;
    for(int i = 0; i < nF; i++)
        for(int j = 0; j < nI; j++)
            R[i][j] = RAND01 / (double) nF;
}


#endif //PDC_UTIL_H
