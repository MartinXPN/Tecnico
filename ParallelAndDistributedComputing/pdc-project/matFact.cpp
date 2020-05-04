#pragma GCC optimize ("O1")
#pragma GCC optimize ("O2")
#pragma GCC optimize ("O3")
#pragma GCC push_options
#pragma GCC optimize ("unroll-loops")

#include <iostream>
#include <vector>
#include <string>
#include <cstdio>
#include <chrono>
#include <numeric>
#include <algorithm>
#include "matrix.h"
#include "util.h"

//#define VERBOSE
using namespace std;


Matrix factorize(const Matrix& L,
                 const Matrix& R,
                 const vector< vector <int> >& nonzeroRowsPos,
                 const vector< vector <double> >& nonzeroRowsVal,
                 const vector< vector <int> >& nonzeroColsPos,
                 const vector< vector <double> >& nonzeroColsVal,
                 const int numIterations,
                 const double learningRate) {

    /// tick will be 0 or 1 depending on the time (prev = t%2, cur = 1-prev)
    /// and both matrices are copied to timeL and timeR
    Matrix timeL[2] = {L, L};
    Matrix timeR[2] = {R, R};

    /// transposed versions of L and R matrices
    Matrix timeLT[2] = {timeL[0].transpose(), timeL[1].transpose()};
    Matrix timeRT[2] = {timeR[0].transpose(), timeR[1].transpose()};

    vector< vector <double> > D = nonzeroRowsVal;       /// difference matrix: D = B - A for all A != 0
    vector< vector <double> > DT = nonzeroColsVal;      /// D transposed

    const size_t rows = L.rows;
    const size_t nbFeatures = R.rows;
    const size_t columns = R.cols;


    for( int t=0; t < numIterations; ++t ) {
        const int prev = t % 2;
        const int cur = 1 - prev;

        /// compute B and B transposed with D and D transposed
        for( int i=0; i < rows; ++i )
            for( int j=0; j < nonzeroRowsPos[i].size(); ++j ) {
                int c = nonzeroRowsPos[i][j];
                double b = inner_product(timeL[prev][i], timeL[prev][i] + timeL[prev].cols, timeRT[prev][c], 0.);
                D[i][j] = b - nonzeroRowsVal[i][j];
            }

        for( int j=0; j < columns; ++j )
            for( int i=0; i < nonzeroColsPos[j].size(); ++i ) {
                int r = nonzeroColsPos[j][i];
                double bt = inner_product(timeL[prev][r], timeL[prev][r] + timeL[prev].cols, timeRT[prev][j], 0.);
                DT[j][i] = bt - nonzeroColsVal[j][i];
            }

        /// compute L(t)
        for( int i=0; i < rows; ++i )
            for( int k=0; k < nbFeatures; ++k ) {
                double sum = 0;
                for( int j=0; j < nonzeroRowsPos[i].size(); ++j )
                    sum += D[i][j] * timeR[prev][k][ nonzeroRowsPos[i][j] ];

                timeLT[cur][k][i] = timeL[cur][i][k] = timeL[prev][i][k] - learningRate * 2 * sum;
            }

        /// compute R(t)
        for( int k=0; k < nbFeatures; ++k )
            for( int j=0; j < columns; ++j ) {
                double sum = 0;
                for( int i=0; i < nonzeroColsPos[j].size(); ++i )
                    sum += DT[j][i] * timeLT[prev][k][ nonzeroColsPos[j][i] ];

                timeRT[cur][j][k] = timeR[cur][k][j] = timeR[prev][k][j] - learningRate * 2 * sum;
            }


        #ifdef VERBOSE
        cerr << "L at time " + to_string(t) << endl << timeL[cur] << endl;
        cerr << "R at time " + to_string(t) << endl << timeR[cur] << endl;
        #endif
    }

    return timeL[numIterations % 2] * timeR[numIterations % 2];
}


int main(int argc, char *argv[]) {
    std::cout.precision(6);
    std::cout << std::fixed;
    auto startTime = chrono::high_resolution_clock::now();

    string filename = argv[1];
    db(filename);
    freopen(argv[1], "r", stdin);

    // first line
    int numIterations;
    cin >> numIterations;
    db(numIterations);

    // second line
    double alpha;
    cin >> alpha;
    db(alpha);

    // third line
    size_t nbFeatures;
    cin >> nbFeatures;
    db(nbFeatures);

    // fourth line
    size_t rows, columns, nonZeroElements;
    cin >> rows >> columns >> nonZeroElements;
    db(rows);
    db(columns);
    db(nonZeroElements);

    // rest of the lines are the elements of the matrix
    vector< vector <int> > nonzeroRowsPos(rows);
    vector< vector <double> > nonzeroRowsVal(rows);
    vector< vector <int> > nonzeroColsPos(columns);
    vector< vector <double> > nonzeroColsVal(columns);
    for( int i=0; i < nonZeroElements; ++i ) {
        int r, c;
        double value;
        cin >> r >> c >> value;
        nonzeroRowsPos[r].push_back(c);
        nonzeroColsPos[c].push_back(r);
        nonzeroRowsVal[r].push_back(value);
        nonzeroColsVal[c].push_back(value);
    }


    // L, R matrices
    Matrix L(rows, nbFeatures);
    Matrix R(nbFeatures, columns);
    random_fill_LR(L, R, rows, columns, nbFeatures);

    #ifdef VERBOSE
    cerr << "Initial matrix L" << endl << L << endl;
    cerr << "Initial matrix R" << endl << R << endl;
    cerr << "Initial matrix B" << endl << L * R << endl;
    #endif

    // LR factorization
    auto B = factorize(L, R, nonzeroRowsPos, nonzeroRowsVal, nonzeroColsPos, nonzeroColsVal, numIterations, alpha);

    // Print recommendations
    for( int i=0; i < B.rows; ++i ) {
        for( const int col: nonzeroRowsPos[i] )
            B[i][col] = -1;
        cout << int(max_element(B[i], B[i] + B.cols) - B[i]) << endl;
    }

    auto finishTime = chrono::high_resolution_clock::now();
    cerr << "Execution time: " << chrono::duration_cast<std::chrono::milliseconds>(finishTime - startTime ).count() / 1000. << endl;
    return 0;
}
