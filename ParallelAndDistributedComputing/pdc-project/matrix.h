#ifndef PDC_MATRIX_H
#define PDC_MATRIX_H

#include <iostream>
#include <numeric>
#include <cassert>

bool bigEnough = true;

class Matrix {
private:
    double *data;
    bool keepData;

public:
    const int rows;
    const int cols;
    Matrix( int rows, int cols, double *data, bool keepData = false ) : rows{rows}, cols{cols}, data{data}, keepData{keepData} {}
    Matrix( int rows, int cols ) : Matrix( rows, cols, new double[rows * cols] ) {}
    Matrix( const Matrix& other ) : Matrix( other.rows, other.cols ) {
        std::copy( other.data, other.data + rows*cols, this->data );
    }
    Matrix( Matrix&& other ) noexcept : cols{other.cols}, rows{other.rows}, keepData{other.keepData} {
        if(this == &other)
            return;

        data = other.data;
        other.data = nullptr;
    }

    Matrix& operator = ( const Matrix &other ) {
        if( this == &other )
            return *this;

        assert(this->rows == other.rows && this->cols == other.cols);
        delete[] data;
        data = new double[rows * cols];
        std::copy(other.data, other.data + rows * cols, data);
        return *this;
    }

    Matrix& operator = ( Matrix &&other )  noexcept {
        if( this == &other )
            return *this;
        assert(this->rows == other.rows && this->cols == other.cols);
        delete[] data;

        data = other.data;
        other.data = nullptr;
        return *this;
    }

    inline double* operator []( int i ) const {
//        assert( i >= 0 && i < rows );
        return &data[i * cols];
    }
    friend std::ostream & operator << (std::ostream &out, const Matrix &matrix) {
        for( int i=0; i < matrix.rows; ++i, out << std::endl )
            for( int j=0; j < matrix.cols; ++j )
                out << matrix[i][j] << " ";
        return out;
    }


    ~Matrix() {
        if( !keepData )
            delete[] data;
    }


    void transpose(Matrix &res, const int block_size) const {
        #pragma omp parallel for schedule(static, 2) if (bigEnough)
        for( int i=0; i < rows; i += block_size )
            for( int j=0; j < cols; j += block_size )
                for( int bi=i; bi < i + block_size && bi < rows; ++bi )
                    for( int bj=j; bj < j + block_size && bj < cols; ++bj )
                        res[bj][bi] = (*this)[bi][bj];
    }

    Matrix transpose(const int block_size=16) const {
        Matrix res( cols, rows );
        transpose(res, block_size);
        return res;
    }


    Matrix operator * (const Matrix& B) {
        Matrix res(rows, B.cols);
        assert( cols == B.rows );

        // B transpose
        Matrix BT = B.transpose(16);

        // Set of inner products
        #pragma omp parallel for if (bigEnough)
        for ( int i = 0; i < rows; i++ )
            for ( int j = 0; j < B.cols; j++ )
                res[i][j] = std::inner_product( (*this)[i], (*this)[i] + cols, BT[j], 0. );

        return res;
    }
};

#endif //PDC_MATRIX_H
