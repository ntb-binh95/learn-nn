#ifndef _CONVOLUTION_
#define _CONVOLUTION_

#include "layer.h"

class Convolution : public Layer {
    public:
        Convolution(int in_channels, int inWidth, int inHeight, int out_channels, int kernelSize, int batchSize, int pad, int stride)
        : Layer("convolution", CONVOLUTION), 
        in_channels{in_channels}, kernelSize{kernelSize}, out_channels{out_channels},
        pad{pad}, stride{stride}, w{inWidth}, h{inHeight}
        {
            batch = batchSize;
            random_initialize();
        };
        
        // size_t get_size() override;
        // size_t get_input_size() override;

    protected:
        // void forward(float * inp) override;
        // void backward(float * delta) override;
        // void update() override;
        // void save_weight(std::ofstream &weight_file) override;
        // void load_weight(std::ifstream &weight_file) override;

    private:
        int kernelSize;
        int in_channels, h, w;
        int out_channels;
        int pad, stride;
        int weightSize, biasSize;
        
        // std::unique_ptr<float[]> input;
        std::unique_ptr<float[]> weights;
        std::unique_ptr<float[]> bias;
        std::unique_ptr<float[]> update_weights;
        std::unique_ptr<float[]> update_bias;
        // std::unique_ptr<float[]> gradient;
        // std::unique_ptr<float[]> layer_delta;
        // float sigmoid(float x);
        // float gradient_sigmoid(float x);
        // void backward_bias(float * delta);
        void random_initialize();
        int getOutputWidth(int w, int s, int p);
        int getOutputHeight(int h, int s, int p);
};

#endif