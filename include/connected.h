#ifndef _CONNECTED_
#define _CONNECTED_

#include "layer.h"

class Connected : public Layer {
    public:
        Connected(int inputSize, int n, int batchSize) : Layer("connected", CONNECTED), 
        neuralSize{n}, inputSize{inputSize}{
            //random initialize weight and bias
            batch = batchSize;
            random_initialize();
        };
        
        size_t get_size() override;

        size_t get_input_size() override;

    protected:
        void forward(float * inp) override;
        void backward(float * delta) override;
        void update() override;
        void save_weight(std::ofstream &weight_file) override;
        void load_weight(std::ifstream &weight_file) override;

    private:
        int neuralSize;
        int inputSize;
        std::unique_ptr<float[]> output;
        std::unique_ptr<float[]> input;
        std::unique_ptr<float[]> weights;
        std::unique_ptr<float[]> bias;
        std::unique_ptr<float[]> update_weights;
        std::unique_ptr<float[]> update_bias;
        std::unique_ptr<float[]> gradient;
        std::unique_ptr<float[]> layer_delta;
        float sigmoid(float x);
        float gradient_sigmoid(float x);
        void backward_bias(float * delta);
        void random_initialize();
};

#endif