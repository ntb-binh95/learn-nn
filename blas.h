#ifndef BLAS_H
#define BLAS_H

void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);

// void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
// {
//     int i;
//     for(i = 0; i < N; ++i) Y[i*INCY] += ALPHA*X[i*INCX];
// }

#endif