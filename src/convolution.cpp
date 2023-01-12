#include <memory>
#include <cmath>

#include "convolution.h"
#include "random_func.h"

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

    int outWidth = getOutputWidth(w, stride, pad);
    int outHeight = getOutputWidth(h, stride, pad);
};

int Convolution::getOutputWidth(int w, int s, int p) {
    return (w + 2*p - kernelSize) / s + 1;
};

int Convolution::getOutputHeight(int h, int s, int p) {
    return (h + 2*p - kernelSize) / s + 1;
}