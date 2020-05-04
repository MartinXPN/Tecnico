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
#include <set>
#include <mpi.h>
#include <cmath>
#include "matrix.h"
#include "util.h"

//#define VERBOSE
using namespace std;

inline void receive(Matrix& RT,
                    const vector <int>& offsets,
                    const vector <int>& blockLengths,
                    const vector <int>& workers,
                    vector <MPI_Request>& requests) {
    assert( offsets.size() == blockLengths.size() );
    assert( offsets.size() == workers.size() );
    assert( offsets.size() == requests.size() );

    for( int i=0; i < workers.size(); ++i ) {
        assert( blockLengths[i] > 0 );
        assert( offsets[i] >= 0 );
        MPI_Irecv(RT[0] + offsets[i], blockLengths[i], MPI_DOUBLE, workers[i], workers[i], MPI_COMM_WORLD, &requests[i]);
    }
}

inline void send(const Matrix& L,
                 const int wId,
                 const vector <MPI_Datatype>& sendTypes,
                 const vector <int>& workers,
                 vector <MPI_Request>& requests) {
    assert( workers.size() == sendTypes.size() );
    assert( workers.size() == requests.size() );

    for(int i=0; i < workers.size(); ++i ) {
        MPI_Isend(L[0], 1, sendTypes[i], workers[i], wId, MPI_COMM_WORLD, &requests[i]);
    }
}


void factorize(const int numIterations,
               const int wId, const int lWorkers, const int nbWorkers,
               const Matrix& L,
               const Matrix& RT,
               const Matrix& fullRT,
               const double learningRate,
               const vector< pair <int, int> >&ranges,
               const vector <int>& offsets,
               const vector <int>& blockLengths,
               const vector< vector<int> > &initialNonzeroPos,
               const vector< vector<int> > &nonzeroPos,
               const vector< vector<double> > &nonzeroVal,
               const vector <vector <int> >& toSend,
               const vector <MPI_Datatype>& sendTypes) {

    const pair <int, int> range = ranges[wId];
    const int start = range.first;
    const int end = range.second;

    Matrix timeL[2] = {L, L};
    Matrix timeRT[2] = {RT, RT};
    vector< vector <double> > D = nonzeroVal;

    /// compress offsets and block lengths to keep only needed ones
    vector <int> receiveOffsets, receiveBlockLengths, receiveWorkers;
    for(int w=0; w < offsets.size(); ++w )
        if(blockLengths[w] > 0 )
            receiveOffsets.push_back( offsets[w] ), receiveBlockLengths.push_back( blockLengths[w] ), receiveWorkers.push_back(w);
//    dba(receiveOffsets);
//    dba(receiveBlockLengths);
//    dba(receiveWorkers);

    /// compress send data
    vector <int> sendWorkers;
    vector <MPI_Datatype> compressedSendTypes;
    for( int w=0; w < toSend.size(); ++w )
        if( !toSend[w].empty() )
            compressedSendTypes.push_back( sendTypes[w] ), sendWorkers.push_back(w);
//    dba(sendWorkers);

    const int nbFeatures = RT.cols;
    vector<MPI_Request> receiveRequest[2] = {vector<MPI_Request>(receiveWorkers.size()), vector<MPI_Request>(receiveWorkers.size())};
    vector<MPI_Request> sendRequest[2] = {vector<MPI_Request>(sendWorkers.size()), vector<MPI_Request>(sendWorkers.size())};
    for( int t=0; t < numIterations; ++t ) {
        const int prev = t % 2;
        const int cur = 1 - prev;

        /// RT[cur] is not being used so we can initialize receive right here
        if( t != numIterations -1  )
            receive(timeRT[cur], receiveOffsets, receiveBlockLengths, receiveWorkers, receiveRequest[cur]);

        /// make sure to receive RT[prev] before proceeding
        if( t != 0 )
            MPI_Waitall(receiveRequest[prev].size(), receiveRequest[prev].data(), MPI_STATUSES_IGNORE);

        for( int i=start; i < end; ++i )
            for( int j=0; j < nonzeroPos[i].size(); ++j ) {
                int c = nonzeroPos[i][j];
                double b = inner_product(timeL[prev][i], timeL[prev][i] + timeL[prev].cols, timeRT[prev][c], 0.);
                D[i][j] = b - nonzeroVal[i][j];
            }

        /// make sure to send L[cur] before modifying
        if( t >= 2 )
            MPI_Waitall(sendRequest[cur].size(), sendRequest[cur].data(), MPI_STATUSES_IGNORE);

        for( int i=start; i < end; ++i )
            for( int k=0; k < nbFeatures; ++k ) {
                double sum = 0;
                for( int j=0; j < nonzeroPos[i].size(); ++j )
                    sum += D[i][j] * timeRT[prev][ nonzeroPos[i][j] ][k];

                timeL[cur][i][k] = timeL[prev][i][k] - learningRate * 2 * sum;
            }

//        if( wId < lWorkers )
//            cerr << "L at time " << t << endl << timeL[cur] << endl;
        if( t != numIterations - 1 )
            send(timeL[cur], wId, compressedSendTypes, sendWorkers, sendRequest[cur]);
    }


    Matrix& finalL = timeL[numIterations % 2];
//    db(wId);
//    db(finalL.rows);
//    db(finalL.cols);
//    cerr << "Final L: \n" << finalL;
    /// workers who have the final version of R[start; end) send their version to all workers that compute their part of L
    if( wId >= lWorkers ) {
        vector <MPI_Request> req(lWorkers);
        for( int w=0; w < lWorkers; ++w )
            MPI_Isend(finalL[start], (end - start) * nbFeatures, MPI_DOUBLE, w, wId, MPI_COMM_WORLD, &req[w]);

        MPI_Waitall(req.size(), req.data(), MPI_STATUSES_IGNORE);
        return;
    }

    /// wId who was computing L should receive the whole matrix R and then send the max elements retrieved to wId=0
    vector <MPI_Request > requestR(nbWorkers - lWorkers);
    cerr << wId << " requesting for matrix R..." << endl;
    for( int w=lWorkers; w < nbWorkers; ++w )
        MPI_Irecv(fullRT[ ranges[w].first ], (ranges[w].second - ranges[w].first) * nbFeatures, MPI_DOUBLE, w, w, MPI_COMM_WORLD, &requestR[w - lWorkers]);
    MPI_Waitall(requestR.size(), requestR.data(), MPI_STATUSES_IGNORE);

    cerr << "[" << wId << "]" << "Received full Matrix RT:" << fullRT.rows << " " << fullRT.cols << endl;


    /// if the wId=0 => it should accumulate and print the max elements
    /// otherwise the elements should be sent to wId=0
    vector <int> res( wId == 0 ? L.rows : end - start );
    vector <double> B( fullRT.rows );
    for( int r=start; r < end; ++r ) {
        for(int c=0; c < fullRT.rows; ++c ) {
            const double bij = inner_product(finalL[r], finalL[r] + finalL.cols, fullRT[c], 0. );
            B[c] = bij;
        }
        for( int c: initialNonzeroPos[r] )
            B[c] = -1;
//        cerr << wId << " " << r << ": "; dba(B);
        res[r - start] = max_element( B.begin(), B.end() ) - B.begin();
    }
//    cerr << wId << " => ";  dba(res);

    if( wId != 0 ) {
        assert( res.size() == end - start );
        MPI_Ssend(res.data(), res.size(), MPI_INT, 0, wId, MPI_COMM_WORLD);
        return;
    }

    /// wId = 0
    cerr << wId << " Requesting results from other nodes..." << endl;
    vector <MPI_Request> requestResults(lWorkers - 1);
    for( int w=1; w < lWorkers; ++w )
        MPI_Irecv(res.data()  + ranges[w].first, ranges[w].second - ranges[w].first, MPI_INT, w, w, MPI_COMM_WORLD, &requestResults[w - 1]);
    cerr << wId << " Waiting for the results from other nodes..." << endl;
    MPI_Waitall(requestResults.size(), requestResults.data(), MPI_STATUSES_IGNORE);

    for( int recommendationId: res )
        cout << recommendationId << endl;
}


int main(int argc, char *argv[]) {
    std::cout.precision(6);
    std::cout << std::fixed;
    std::cerr.precision(6);
    std::cerr << std::fixed;
    auto startTime = chrono::high_resolution_clock::now();
    MPI_Init(&argc, &argv);

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
    int nbFeatures;
    cin >> nbFeatures;
    db(nbFeatures);

    // fourth line
    int rows, columns, nonZeroElements;
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
    Matrix LT = L.transpose();
    Matrix RT = R.transpose();

    /// Each worker has to compute [start; end) rows
    int nbWorkers = 6;
    MPI_Comm_size(MPI_COMM_WORLD, &nbWorkers);
    db(nbWorkers);

    if( nbWorkers < 2 ) {
        cerr << "Need at least 2 workers" << endl;
        exit( 1 );
    }

    /// How mny workers should process L and R
    /// resources are allocated taking into account the size of the matrices
    int lWorkers = min(nbWorkers - 1, max( 1, (L.rows * nbWorkers) / (L.rows + RT.rows) ) );
    int rWorkers = nbWorkers - lWorkers;
    db(lWorkers);
    db(rWorkers);

    vector< pair <int, int>> range(nbWorkers);           /// which rows are going to be processed by the worker i
    for( int w=0; w < lWorkers; ++w )   range[w]            = {w * L.rows / lWorkers,   (w + 1) * L.rows / lWorkers};
    for( int w=0; w < rWorkers; ++w )   range[w + lWorkers] = {w * RT.rows / rWorkers,  (w + 1) * RT.rows / rWorkers};

    for( int w=0; w < nbWorkers; ++w )
        cout << "W" << w << " will process rows " << range[w].first << " - " << range[w].second << endl;

    vector<set <int>> needed(nbWorkers);                /// which rows are needed for that worker
    for( int w=0; w < lWorkers; ++w )           for( int r=range[w].first; r < range[w].second; ++r )   needed[w].insert( nonzeroRowsPos[r].begin(), nonzeroRowsPos[r].end() );
    for( int w=lWorkers; w < nbWorkers; ++w )   for( int r=range[w].first; r < range[w].second; ++r )   needed[w].insert( nonzeroColsPos[r].begin(), nonzeroColsPos[r].end() );

//    cerr << endl << endl;
//    for( int w=0; w < nbWorkers; ++w ) {
//        cerr << "W" << w << " will need rows "; dba(needed[w]);
//    }

    vector< vector< vector <int> >> toSend(nbWorkers, vector <vector <int> >(nbWorkers));
    for( int w1=0; w1 < nbWorkers; ++w1 )
        for( int w2=0; w2 < nbWorkers; ++w2 ) {
            if( (w1 < lWorkers && w2 < lWorkers ) || (w1 >= lWorkers && w2 >= lWorkers) )
                continue;

            for( int r : needed[w2] )
                if( range[w1].first <= r && r < range[w1].second )
                    // w2 needs to send row r to w2
                    toSend[w1][w2].push_back(r);
        }

//    for( int w1=0; w1 < nbWorkers; ++w1 )
//        for( int w2=0; w2 < nbWorkers; ++w2 ) {
//            cerr << w1 << " " << w2 << ": "; dba(toSend[w1][w2]);
//        }

    /// initialize MPI types for sending and receiving rows without looping through each of them
    vector< vector <MPI_Datatype> > sendTypes(nbWorkers, vector <MPI_Datatype>(nbWorkers));
    for( int w1=0; w1 < nbWorkers; ++w1 ) {
        for( int w2=0; w2 < nbWorkers; ++w2 ) {
            if( toSend[w1][w2].empty() )
                continue;

            vector <int> blockLengths, displacements;
            int start = 0, cur = 0;
            for( int i=1; i <= toSend[w1][w2].size(); ++i ) {
                if( i < toSend[w1][w2].size() && toSend[w1][w2][i] == toSend[w1][w2][cur] + 1 ) {
                    ++cur;
                    continue;
                }
                // note that both L and RT have .cols = nbFeatures
                blockLengths.push_back( (i - start) * nbFeatures );
                displacements.push_back( toSend[w1][w2][start] * nbFeatures );
                start = cur = i;
            }

            assert( !blockLengths.empty() );

//            cerr << w1 << " " << w2 << ":\n";
//            dba(blockLengths);
//            dba(displacements);
//            cerr << endl;
            MPI_Type_indexed(blockLengths.size(), &blockLengths[0], &displacements[0], MPI_DOUBLE, &sendTypes[w1][w2]);
            MPI_Type_commit(&sendTypes[w1][w2]);
        }
    }

    /// Compress matrix R to leave only needed rows for worker wId
    /// one very important observation here is that
    /// workers process contiguous rows => when compressing matrix RT we can be sure to receive contiguous
    /// rows from each worker and the rows won't be shuffled across results from different workers
    int wId = 5;
    MPI_Comm_rank(MPI_COMM_WORLD, &wId);
//    db(wId);
//    dba(needed[wId]);
    auto nonzeroPos = wId < lWorkers ? nonzeroRowsPos : nonzeroColsPos;
    auto nonzeroVal = wId < lWorkers ? nonzeroRowsVal : nonzeroColsVal;

    /// compress RT so that we have only the needed rows
    vector <int> rIds(wId < lWorkers ? RT.rows : L.rows, -1);
    {
        int i=0;
        for( int rId : needed[wId] )
            rIds[ rId ] = i++;

        for(auto & nonzeroPosRow : nonzeroPos)
            for(int &pos : nonzeroPosRow)
                pos = rIds[pos];
    }
//    dba(rIds);
//    db2d(nonzeroPos);
//    db2d(nonzeroVal);

    /// compute offsets for receiving compressed RT
    vector <int> offsets( nbWorkers, -1 );
    vector <int> blockLengths( nbWorkers, 0 );
    int nbCompressedRows = 0;
    for( int w=0, offset=0; w < nbWorkers; ++w )
        if( !toSend[w][wId].empty() ) {
            offsets[w] = offset;
            blockLengths[w] = toSend[w][wId].size() * nbFeatures;

            offset += blockLengths[w];
            nbCompressedRows += toSend[w][wId].size();
        }
//    dba(offsets);
//    dba(blockLengths);

    Matrix compressedRT(nbCompressedRows, nbFeatures);
    int rowId = 0;
    for( int w=0; w < nbWorkers; ++w ) {
        for( int r : toSend[w][wId] ) {
            for( int c=0; c < compressedRT.cols; ++c )
                compressedRT[rowId][c] = wId < lWorkers ? RT[r][c] : L[r][c];
            ++rowId;
        }
    }

//    cerr << "Initial RT:\n" << (wId < lWorkers ? RT : L) << endl;
//    cerr << "Compressed RT: \n" << compressedRT << endl;

    if( wId < lWorkers )    factorize(numIterations, wId, lWorkers, nbWorkers, L, compressedRT, RT, alpha, range, offsets, blockLengths, nonzeroRowsPos, nonzeroPos, nonzeroVal, toSend[wId], sendTypes[wId]);
    else                    factorize(numIterations, wId, lWorkers, nbWorkers, RT, compressedRT, L, alpha, range, offsets, blockLengths, nonzeroColsPos, nonzeroPos, nonzeroVal, toSend[wId], sendTypes[wId]);

    cerr << "Finalize: " << wId << endl;
    MPI_Finalize();
    auto finishTime = chrono::high_resolution_clock::now();
    cerr << "Execution time: " << chrono::duration_cast<std::chrono::milliseconds>(finishTime - startTime ).count() / 1000. << endl;
    return 0;
}
