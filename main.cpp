#include <iostream>
#include <vector>
#include <cmath>
#include <memory>

#include "gemm.h"

using namespace std;

#define LOG(x) cout << x << endl

typedef enum {CONNECTED, CONVOLUTION, BLANK} LAYER_TYPE;
class Layer {
    friend class Network;
    public:
        Layer(string name, LAYER_TYPE type): name{name}, type{type} {};
        string name = "blank";
        LAYER_TYPE type = BLANK;
    
    protected:
        virtual void forward(float * inp) = 0;
        virtual size_t get_size() = 0;
        virtual void backward(float * delta) = 0;
        virtual void update() = 0;
};

class Connected : public Layer {
    public:
        Connected(int n, int inputSize) : Layer("connected", CONNECTED), 
        neuralSize{n}, inputSize{inputSize} {
            weights = make_unique<float[]>( neuralSize * inputSize);   
            bias = make_unique<float[]>(1);   
            output = make_unique<float[]>(neuralSize);   
            input = make_unique<float[]>(inputSize);
        };

        void load_weight(float * w) {
            for (int i = 0; i < neuralSize*inputSize; i++) {
                weights[i] = w[i];
            }
        };

        void load_bias(float b) {
            bias[0] = b;
        };
        
        size_t get_size() override {
            return neuralSize;
        };

    protected:
        void forward(float * inp) override {
            //fill the input
            for (int i = 0; i < inputSize; i++) {
                input[i] = inp[i];
            }
            int m = 1;
            int n = neuralSize;
            int k = inputSize;
            float * a = input.get();
            float * b = weights.get();
            float * c = output.get();

            gemm(0,0,m,n,k,1,a,m,b,k,1,c,n);

            // computer gradient in advance
            gradient = make_unique<float[]>(neuralSize);
            for (int i = 0; i < neuralSize; i++) {
                output[i] += bias[0];
                output[i] = sigmoid(output[i]);

                // computer gradient of sigmoid function
                gradient[i] = gradient_sigmoid(output[i]);

                // reassign output to input
                inp[i] = output[i];
            }
        };
        void backward(float * delta) override {
            // update = delta * gradient_sigmoid * input
            update_weights = make_unique<float[]>(neuralSize*inputSize);
            for(int i = 0; i < neuralSize; i++) {
                delta[i] = gradient[i] * delta[i];
                // LOG("delta backward: " << delta[i]);
            }

            update_bias = make_unique<float[]>(1);
            backward_bias(delta);

            int m = neuralSize;
            int k = 1;
            int n = inputSize;
            float * a = input.get();
            float * b = delta;
            float * c = update_weights.get();

            gemm(1,0,m,n,k,1,a,m,b,k,0,c,n);
            // for(int i =0; i< inputSize * neuralSize; i++) {
                // LOG("update weights: " << -0.5f * update_weights[i] + weights[i]);
                // LOG("update weights: " << update_weights[i]);
            // }

            // delta = delta * weight
            // update delta
            auto layer_delta = make_unique<float[]>(neuralSize);
            for (int i = 0; i < neuralSize; i++) {
                layer_delta[i] = delta[i];
            }
            m = inputSize;
            k = neuralSize;
            n = 1;
            a = weights.get();
            b = layer_delta.get();
            c = delta;

            gemm(0,1,m,n,k,1,a,m,b,k,0,c,n);
            // for (int i = 0 ; i < inputSize; i ++) {
            //     LOG("update delta: " << delta[i]);
            // }
        };
        void update() override {
            for (int i = 0; i < inputSize * neuralSize; i++) {
                weights[i] += -0.5f * update_weights[i];
                // LOG("weight updated: " << weights[i]);
            }
            // only 1 bias
            bias[0] += -0.5f * update_bias[0];
        };

    private:
        size_t neuralSize;
        int inputSize;
        unique_ptr<float[]> output;
        unique_ptr<float[]> input;
        unique_ptr<float[]> weights;
        unique_ptr<float[]> bias;
        unique_ptr<float[]> update_weights;
        unique_ptr<float[]> update_bias;
        unique_ptr<float[]> gradient;
        float sigmoid(float x) {
            return 1 / (1 + exp(-x));
        };

        float gradient_sigmoid(float x) {
            return x * (1-x);
        };
        
        void backward_bias(float * delta) {
            for(int i = 0; i < neuralSize; i++){
                update_bias[0] += delta[i];
            }
        };
};

class Network {
    public:
        vector<shared_ptr<Layer>> layers;
        size_t n = 0;
        void forward_net(float * inp) {
            // unique_ptr<float[]> x{inp}; // unique_ptr use
            unique_ptr<float[]> x = make_unique<float[]>(2);// TODO: store inputSize=2
            for(int i = 0; i < 2; i++){
                x[i] = inp[i];
            }

            LOG(x[0] << " " << x[1]);
            for (int l = 0; l < n; l++) {
                layers[l]->forward(x.get());
            }
            // x.release(); // because the 'inp' is stack-controll pointer. Cause double free if unique_ptr free it.
            output = make_unique<float[]>(outputSize);
            for (int i = 0; i < outputSize; i++) {
                output[i] = x[i];
                LOG("output: " << output[i]);
            }
        };
        void backward_net() {
            for (int i = n-1; i > -1; i--) {
                layers[i]->backward(delta.get());
            }
        };
        void update_net() {
            for (int i = 0; i < n; i++) {
                layers[i]->update();
            }
        };
        template <typename T>
        void add_layer(shared_ptr<T> &l) {
            layers.push_back(l);
            n++;
            outputSize = l->get_size();
        };

        float calc_loss(float* gt, int n) {
            delta = make_unique<float[]>(outputSize);
            err = 0.0f;
            for (int i = 0; i < n; i++) {
                err += (1.f/2) * pow((gt[i] - output[i]), 2);
                delta[i] = output[i] - gt[i];
            }

            return err;
        };
    
    private:
        size_t outputSize;
        float err = 0.0f;
        unique_ptr<float[]> delta;
        unique_ptr<float[]> output;
};


int main(int argc, char** argv) {
    float * input = new float[2] {0.05, 0.1};
    // float input[] = {0.05, 0.1};
    float w1[] = {0.15, 0.25,
                  0.2, 0.3};
    float b1{0.35};
    float w2[] = {0.4, 0.5,
                  0.45, 0.55};
    float b2{0.6};

    float ground_truth[] = {0.01, 0.99};

    shared_ptr<Connected> conn1 = make_shared<Connected>(2,2);
    conn1->load_weight(w1);
    conn1->load_bias(b1);

    shared_ptr<Connected> conn2 = make_shared<Connected>(2,2);
    conn2->load_weight(w2);
    conn2->load_bias(b2);

    unique_ptr<Network> net = make_unique<Network>();
    net->add_layer(conn1);
    net->add_layer(conn2);

    int n_epochs = 10000;
    for (int e = 0; e < n_epochs; e++) {
        net->forward_net(input);

        float err = net->calc_loss(ground_truth, 2);
        LOG("Network error: " << err);

        net->backward_net();

        net->update_net();
    }

    net->forward_net(input);

}