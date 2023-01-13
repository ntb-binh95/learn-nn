#ifndef _NETWORK_
#define _NETWORK_

#include "layer.h"
#include "log.h"
#include <iostream>

class Network {
    public:
        Network(int batchSize): batch{batchSize} {};
        void forward_net(float * inp);
        std::unique_ptr<int[]> predict(float * inp);
        void backward_net();
        void update_net();
        template <typename T>
        void add_layer(std::shared_ptr<T> &l) {
            if(!n) {
                inputSize = l->get_input_size();
                maxSize = inputSize;
                std::cout << "input size " << inputSize << std::endl;
            }
            outputSize = l->get_size();
            std::cout << " output size " << outputSize << std::endl;
            if (outputSize > maxSize) {
                std::cout << "max size " << maxSize << std::endl;
                maxSize = outputSize;
            }
            layers.push_back(l);
            n++;
        };

        float calc_loss(float* gt, int n);
        void build();
        void save_weights();
        void load_weights();
    
        std::vector<std::shared_ptr<Layer>> layers;
        size_t n = 0;
    private:
        size_t inputSize = 0;
        size_t outputSize = 0;
        size_t maxSize = 0;
        float err = 0.0f;
        int batch = 0;
        std::unique_ptr<float[]> delta;
        std::unique_ptr<float[]> output;
        std::unique_ptr<float[]> workspace;
        std::string saved_model = "my.weight";
};

#endif