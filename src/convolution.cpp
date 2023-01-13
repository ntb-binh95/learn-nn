#include <memory>
#include <cmath>

#include "convolution.h"
#include "random_func.h"
#include "gemm.h"

void Convolution::random_initialize() {
    weights = std::make_unique<float[]>(kernelSize * kernelSize * in_channels * out_channels);
    bias = std::make_unique<float[]>(out_channels);
    update_weights = std::make_unique<float[]>(kernelSize * kernelSize * in_channels * out_channels);
    update_bias = std::make_unique<float[]>(out_channels);

    weightSize = kernelSize * kernelSize * in_channels * out_channels;
    biasSize = out_channels;
    float scale = sqrt(2./(kernelSize*kernelSize*out_channels));
    for(int i = 0; i < weightSize; ++i) {
        weights[i] = scale*rand_normal();
    }
    
    for(int i = 0; i < biasSize; i++) {
        bias[i] = 0;
    }

    outWidth = getOutputWidth(w, stride, pad);
    outHeight = getOutputWidth(h, stride, pad);

    outputSize = outWidth * outHeight * out_channels;
    inputSize = w * h * in_channels;

    output = std::make_unique<float[]>(batch * outputSize);   
    input = std::make_unique<float[]>(batch * inputSize);
    layer_delta = std::make_unique<float[]>(batch * outputSize);
};

int Convolution::getOutputWidth(int w, int s, int p) {
    return (w + 2*p - kernelSize) / s + 1;
};

int Convolution::getOutputHeight(int h, int s, int p) {
    return (h + 2*p - kernelSize) / s + 1;
}

void Convolution::forward(float * ws) {
    //fill the input
    for (int i = 0; i < batch * inputSize; i++) {
        input[i] = ws[i];
    }

    int m = out_channels;    
    int k = kernelSize * kernelSize * in_channels;
    int n = outWidth * outHeight;
    for(int i = 0; i < batch; i++) {
        float *a = weights.get();
        float *b = ws;
        float *c = output.get() + i*outputSize;
        float *im = input.get()+ i*inputSize;

        if(kernelSize == 1) {
            b = im;
        } else {
            im2col_cpu(im, in_channels, h, w, kernelSize, stride, pad, b);
            printf("debug \n");
        }
        // gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
    }
}

void Convolution::backward(float * delta) {

}

void Convolution::update() {

}

float Convolution::relu(float x) {
    return x * (x > 0);
}

float Convolution::gradient_relu(float x) {
    return (x > 0);
}

size_t Convolution::get_size(){
    return outputSize;
};

size_t Convolution::get_input_size() {
    return  inputSize;
};


void Convolution::save_weight(std::ofstream &weight_file) {

}
void Convolution::load_weight(std::ifstream &weight_file) {

}