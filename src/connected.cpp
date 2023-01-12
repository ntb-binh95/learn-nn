#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cmath>

#include "random_func.h"
#include "log.h"
#include "connected.h"

#include "gemm.h"
#include "blas.h"

void Connected::random_initialize() {
    // srand(time(0)); use for different seeding
    weights = std::make_unique<float[]>(neuralSize * inputSize);   
    bias = std::make_unique<float[]>(neuralSize);   
    output = std::make_unique<float[]>(batch * neuralSize);   
    input = std::make_unique<float[]>(batch * inputSize);
    gradient = std::make_unique<float[]>(batch * neuralSize);
    update_weights = std::make_unique<float[]>(neuralSize*inputSize);
    update_bias = std::make_unique<float[]>(neuralSize);
    layer_delta = std::make_unique<float[]>(batch * neuralSize);

    float scale = sqrt(2./inputSize);
    for(int i = 0; i < inputSize * neuralSize; i++){
        weights[i] = scale * rand_uniform(-1, 1);
    }
    for(int i = 0; i < neuralSize; i++){
        bias[i] = 0;
    }
};

size_t Connected::get_size(){
    return neuralSize;
};

size_t Connected::get_input_size() {
    return  inputSize;
};

float Connected::sigmoid(float x) {
    return 1 / (1 + exp(-x));
};

float Connected::gradient_sigmoid(float x) {
    return x * (1-x);
};
        
void Connected::forward(float * inp){
    //fill the input
    for (int i = 0; i < batch * inputSize; i++) {
        input[i] = inp[i];
    }

    // calculate weight^T * input
    int m = batch;
    int n = neuralSize;
    int k = inputSize;
    float * a = input.get(); 
    float * b = weights.get(); //weights
    float * c = output.get();

    // LOG("Matrix A: ");
    // show_matrix(a, m, k);
    // LOG("Matrix B: ");
    // show_matrix(b, k, n);
    gemm(0,0,m,n,k,1,a,k,b,n,0,c,n);
    // LOG("Matrix C: ");
    // show_matrix(c, m, n);

    for(int b =0 ; b < batch; b++) {
        for (int i = 0; i < neuralSize; i++) {
            output[b * neuralSize + i] += bias[i]; // output = output + bias
            output[b * neuralSize + i] = sigmoid(output[b*neuralSize+i]); // output = activate(output)

            // compute gradient in advance
            gradient[b * neuralSize + i] = gradient_sigmoid(output[b * neuralSize + i]);

            // reassign output to input
            inp[b * neuralSize + i] = output[b * neuralSize + i];
        }
    }
};
void Connected::backward(float * delta) {
    // update = delta * gradient_sigmoid * input
    for(int i = 0; i < batch * neuralSize; i++) {
        delta[i] *= gradient[i];
        // LOG("delta backward: " << delta[i]);
    }

    backward_bias(delta);

    int m = inputSize;
    int k = batch; // add batch
    int n = neuralSize;
    float * a = input.get();
    float * b = delta;
    float * c = update_weights.get();

    gemm(1,0,m,n,k,1,a,m,b,n,0,c,n);

    // delta = delta * weight
    // update delta for backpropagate
    for (int i = 0; i < batch * neuralSize; i++) {
        layer_delta[i] = delta[i];
    }

    m = batch;
    k = neuralSize;
    n = inputSize; // add batch
    a = layer_delta.get();
    b = weights.get();
    c = delta;

    gemm(0,1,m,n,k,1,a,k,b,k,0,c,n);
};

void Connected::update() {
    float lr = 0.5f;
    // for (int i = 0; i < inputSize * neuralSize; i++) {
    //     weights[i] += -lr/batch * update_weights[i];
    // }
    axpy_cpu(inputSize * neuralSize, -lr/batch, update_weights.get(), 1, weights.get(), 1);

    // only 1 bias
    // for (int i = 0; i < neuralSize; i++){
    //     bias[i] += -lr/batch * update_bias[i];
    // }
    axpy_cpu(neuralSize, -lr/batch, update_bias.get(), 1, bias.get(), 1);
};

void Connected::save_weight(std::ofstream &weight_file) {
    weight_file.write(reinterpret_cast<const char *>(weights.get()), inputSize * neuralSize * sizeof(float));
    weight_file.write(reinterpret_cast<const char *>(bias.get()), neuralSize * sizeof(float));
};

void Connected::load_weight(std::ifstream &weight_file) {
    weight_file.read(reinterpret_cast<char *>(weights.get()), inputSize * neuralSize * sizeof(float));
    weight_file.read(reinterpret_cast<char *>(bias.get()), neuralSize * sizeof(float));
};

void Connected::backward_bias(float * delta) {
    for (int j = 0; j < neuralSize; j++){
        update_bias[j] = 0;
    }
    for(int b = 0; b < batch; b++) {
        for(int i = 0; i < neuralSize; i++){
            update_bias[i] += delta[b*neuralSize + i]; 
        }
    }
};