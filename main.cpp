#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <iomanip>

#include "gemm.h"

using namespace std;

#define LOG(x) cout << x << endl

typedef enum {CONNECTED, CONVOLUTION, BLANK} LAYER_TYPE;

float rand_uniform(float min, float max) {
    if (max < min) {
        float swap = min;
        min = max;
        max = swap;
    }
    return ((float)rand()/RAND_MAX * (max - min)) + min;
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
class Layer {
    friend class Network;
    public:
        Layer(string name, LAYER_TYPE type): name{name}, type{type} {};
        string name = "blank";
        LAYER_TYPE type = BLANK;
    
    protected:
        virtual void forward(float * inp) = 0;
        virtual size_t get_size() = 0;
        virtual size_t get_input_size() = 0;
        virtual void backward(float * delta) = 0;
        virtual void update() = 0;
};

class Connected : public Layer {
    public:
        Connected(int inputSize, int n) : Layer("connected", CONNECTED), 
        neuralSize{n}, inputSize{inputSize} {
            weights = make_unique<float[]>(neuralSize * inputSize);   
            bias = make_unique<float[]>(1);   
            output = make_unique<float[]>(neuralSize);   
            input = make_unique<float[]>(inputSize);
            gradient = make_unique<float[]>(neuralSize);
            update_weights = make_unique<float[]>(neuralSize*inputSize);
            update_bias = make_unique<float[]>(1);
            // temp variable
            layer_delta = make_unique<float[]>(neuralSize);

            //random initialize weight and bias
            // srand(time(0)); use for different seeding
            float scale = sqrt(2./inputSize);
            for(int i = 0; i < inputSize * neuralSize; i++){
                weights[i] = scale * rand_uniform(-1, 1);
            }
            bias[0] = 0; // bias should be size of output for more general case.
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

        size_t get_input_size() override {
            return  inputSize;
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
            float * b = weights.get(); //weights
            float * c = output.get();

            // LOG("Matrix A: ");
            // show_matrix(a, m, k);
            // LOG("Matrix B: ");
            // show_matrix(b, k, n);
            gemm(0,0,m,n,k,1,a,k,b,n,0,c,n);
            // LOG("Matrix C: ");
            // show_matrix(c, m, n);

            // need to compute gradient in advance
            for (int i = 0; i < neuralSize; i++) {
                output[i] += bias[0];
                output[i] = sigmoid(output[i]);

                // compute gradient in advance
                gradient[i] = gradient_sigmoid(output[i]);

                // reassign output to input
                inp[i] = output[i];
            }
        };
        void backward(float * delta) override {
            // update = delta * gradient_sigmoid * input
            for(int i = 0; i < neuralSize; i++) {
                delta[i] = gradient[i] * delta[i];
                // LOG("delta backward: " << delta[i]);
            }

            backward_bias(delta);

            // int m = neuralSize;
            // int k = 1;
            // int n = inputSize;
            // float * a = input.get();
            // float * b = delta;
            // float * c = update_weights.get();

            // gemm(1,0,m,n,k,1,a,m,b,k,0,c,n);
            int m = inputSize;
            int k = 1;
            int n = neuralSize;
            float * a = input.get();
            float * b = delta;
            float * c = update_weights.get();

            gemm(0,1,m,n,k,1,a,k,b,k,0,c,n);


            // for(int i =0; i< inputSize * neuralSize; i++) {
                // LOG("update weights: " << -0.5f * update_weights[i] + weights[i]);
            // }

            // delta = delta * weight
            // update delta
            for (int i = 0; i < neuralSize; i++) {
                layer_delta[i] = delta[i];
            }
            // m = inputSize;
            // k = neuralSize;
            // n = 1;
            // a = weights.get();
            // b = layer_delta.get();
            // c = delta;

            // gemm(0,1,m,n,k,1,a,m,b,k,0,c,n);

            m = inputSize;
            k = neuralSize;
            n = 1;
            a = weights.get();
            b = layer_delta.get();
            c = delta;

            gemm(0,0,m,n,k,1,a,k,b,n,0,c,n);
            // for (int i = 0 ; i < inputSize; i ++) {
            //     LOG("update delta: " << delta[i]);
            // }
        };

        void update() override {
            float lr = 0.2f;
            for (int i = 0; i < inputSize * neuralSize; i++) {
                weights[i] += -lr * update_weights[i];
                // LOG("weight updated: " << weights[i]);
            }
            // only 1 bias
            bias[0] += -lr * update_bias[0];
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
        unique_ptr<float[]> layer_delta;
        float sigmoid(float x) {
            return 1 / (1 + exp(-x));
        };

        float gradient_sigmoid(float x) {
            return x * (1-x);
        };
        
        void backward_bias(float * delta) {
            update_bias[0] = 0;
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
            // copy input to the workspace
            for(int i = 0; i < inputSize; i++){
                workspace[i] = inp[i];
            }

            // loop over all layer in network
            for (int l = 0; l < n; l++) {
                layers[l]->forward(workspace.get());
            }

            // get output after forwarding
            for (int i = 0; i < outputSize; i++) {
                output[i] = workspace[i];
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
            if(!n) {
                inputSize = l->get_input_size();
                maxSize = inputSize;
            }
            outputSize = l->get_size();
            if (outputSize > maxSize) {
                maxSize = outputSize;
            }
            layers.push_back(l);
            n++;
        };

        float calc_loss(float* gt, int n) {
            err = 0.0f;
            for (int i = 0; i < n; i++) {
                err += (1.f/2) * pow((gt[i] - output[i]), 2);
                delta[i] = output[i] - gt[i];
            }

            return err;
        };

        void build() {
            LOG(maxSize << " " << outputSize);
            workspace = make_unique<float[]>(maxSize);
            output = make_unique<float[]>(outputSize);
            delta = make_unique<float[]>(maxSize);
        };
    
    private:
        size_t inputSize = 0;
        size_t outputSize = 0;
        size_t maxSize = 0;
        float err = 0.0f;
        unique_ptr<float[]> delta;
        unique_ptr<float[]> output;
        unique_ptr<float[]> workspace;
};


int main(int argc, char** argv) {
    /*TODO: 
    1. Add random to weight
    2. 
    */
    int inputSize = 5;
    int hiddenSize = 100;
    int outputSize = 5;

    float * input = new float[inputSize] {0.02, 0.3, 0.025, 0.05, 0.1};
    float * ground_truth = new float[outputSize] {0.02, 0.42, 0.3, 0.99, 0.1};

    shared_ptr<Connected> conn1 = make_shared<Connected>(inputSize,hiddenSize);
    // conn1->load_weight(w1);
    // conn1->load_bias(b1);

    shared_ptr<Connected> conn2 = make_shared<Connected>(hiddenSize,outputSize);
    // conn2->load_weight(w2);
    // conn2->load_bias(b2);

    unique_ptr<Network> net = make_unique<Network>();
    net->add_layer(conn1);
    net->add_layer(conn2);

    net->build();

    int n_epochs = 1000;
    for (int e = 0; e < n_epochs; e++) {
        LOG("Epoch " << e);
        net->forward_net(input);
        float err = net->calc_loss(ground_truth, outputSize);
        LOG("Network error: " << err);
        net->backward_net();
        net->update_net();
    }
    delete input;
    delete ground_truth;

    // net->forward_net(input);

}