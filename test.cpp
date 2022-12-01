#include "gemm.h"
#include <iostream>
#include <iomanip>

float rand_uniform(float min, float max) {
    if (max < min) {
        float swap = min;
        min = max;
        max = swap;
    }
    return ((float)rand()/RAND_MAX * (max - min)) + min;
}

float * make_matrix(int rows, int cols) {
    float * mat = new float[rows * cols];
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = rand_uniform(0,1);
    }
    return mat;
}

void show_matrix(float* mat, int rows, int cols) {
    std::cout << std::setprecision(4);
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++) {
            std::cout << mat[i*cols + j] << "\t";
        }
        std::cout << std::endl;
    }

}

int main() {
    srand(32);
    int m = 5;
    int n = 1;
    int k = 3;
    std::cout << "W:" << std::endl;
    auto weight = make_matrix(k, m);
    show_matrix(weight, k,m);

    std::cout << "input:" << std::endl;
    auto input = make_matrix(k,n);
    show_matrix(input, k,n);

    std::cout << "output: " << std::endl;
    auto output = make_matrix(m, n);


    float * a = weight;
    float * b = input;
    float * c = output;

    gemm(1, 0, m, n, k, 1, a, m, b, n, 0, c, n);
    show_matrix(output, m, n);

    // for (int i = 0; i < 4; i++){
    //     std::cout << c[i] << std::endl;
    // }

    // float a = rand_uniform(-1, 1);
    // std::cout << a << std::endl;

    // // make a matrix
    // int rows = 3;
    // int cols = 4;
    // auto myMat = make_matrix(rows, cols);

    // show_matrix(myMat, rows, cols);
    
    return 0;
}