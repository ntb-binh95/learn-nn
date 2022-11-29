#include <iostream>
#include <vector>

#include "gemm.h"
#include <math.h>

#define LOG(x) std::cout << x << std::endl;

typedef enum {CONNECTED, CONVOLUTION} LAYER_TYPE;
class layer {
    public:
        layer(std::string name, LAYER_TYPE type) : name{name}, type{type} {};
        std::string name="blank";
        LAYER_TYPE type=CONVOLUTION; 
        virtual void forward_net(float * inp, float * out) {};
        virtual size_t get_size() {};
        virtual void backward_net(float * delta) {};
        virtual void update_net() {};
};

class conn_layer : public layer{
    public:
        conn_layer(int n, int input_size) : layer{"connected", CONNECTED}, 
        neural_size{n}, input_size{input_size}{
            weights = (float *)calloc(n*input_size, sizeof(float));
            bias = (float *)calloc(1, sizeof(float));
            output = (float *)calloc(n, sizeof(float));
        };

        void forward_net(float * inp, float * out) override {
            input = (float *)calloc(2, sizeof(float));
            for (int i = 0; i < neural_size; i++){
                // LOG(inp[i]);
                input[i] = inp[i];
            }
            int m = 1;
            int n = neural_size;
            int k = input_size;
            float * a = inp;
            float * b = weights;
            float * c = out;

            gemm(0,0, m, n, k, 1, a, m, b, k, 1, c, n);
            for (int i = 0; i < input_size; i++) {
                out[i] += *bias;
                out[i] = sigmoid(out[i]);
                output[i] = out[i];
            }
        };

        void load_weight(float * w) {
            for(int i = 0; i < neural_size*input_size; i++) {
                weights[i] = w[i];
            } 
        };

        void load_bias(float *b){
            bias = b;
        };

        size_t get_size() override {
            return neural_size;
        };

        void backward_net(float * delta) override {
            float* gradient = (float *)calloc(neural_size, sizeof(float));
            // update = delta * gradient_sigmoid * input
            update_weights = (float *)calloc(neural_size * input_size, sizeof(float));
            gradient_sigmoid(gradient);
            for(int i = 0; i < neural_size; i++) {
                delta[i] = gradient[i] * delta[i];
                // LOG("delta backward: " << delta[i]);
            }

            int m = neural_size;
            int k = 1;
            int n = input_size;
            float * a = input;
            float * b = delta;
            float * c = update_weights;

            gemm(1,0,m,n,k,1,a,m,b,k,0,c,n);
            for(int i =0; i< input_size * neural_size; i++) {
                // LOG(-0.5f * update_weights[i] + weights[i]);
            }
            // delta = delta * weight
            // update delta
            float * layer_delta = (float *)calloc(neural_size, sizeof(float));
            for (int i = 0; i < neural_size; i++) {
                layer_delta[i] = delta[i];
            }
            m = 2;
            k = 2;
            n = 1;
            a = weights;
            b = layer_delta;
            c = delta;

            gemm(0,1,m,n,k,1,a,m,b,k,0,c,n);
            // for (int i = 0 ; i < input_size; i ++) {
            //     LOG("update delta: " << delta[i]);
            // }

        };

        float sigmoid(float x) {
            return 1 / (1 + exp(-x)); 
        };

        void update_net() override {
            for (int i = 0; i < input_size * neural_size; i++) {
                weights[i] += -0.5f * update_weights[i];
                // LOG("weight updated: " << weights[i]);
            }
        };

    private:
        size_t neural_size;
        int input_size;
        float * output;
        float * input;
        float * weights;
        float * bias;
        float * update_weights;
        void gradient_sigmoid(float * gradient) {
            for(int i = 0; i < neural_size; i++){
                gradient[i] = output[i] * (1 - output[i]);
            }
        };
};


class network {
    public:
        std::vector<layer*> layers;
        size_t n = 0; // number of layer
        void forward(float *inp, float *out) {
            LOG(inp[0] << " " << inp[1]);
            for (int i = 0; i < n; i ++ ) {
                output = (float *)calloc(workspace_size, sizeof(float));
                layers[i]->forward_net(inp, output);

                // update input
                for (int j = 0; j < workspace_size; j++) {
                    out[j] = output[j];
                    inp[j] = output[j];
                    LOG("output: " << output[j]);
                }
            }
        };
        void backward(){
            // for (int i = 0; i < output_size; i++) {
            //     LOG("delta: " << delta[i]);
            // }
            // layers.back()->backward_net(delta);
            for (int i = n-1; i > -1; i--) {
                layers[i]->backward_net(delta);
            }
        };
        void update() {
            for (int i = 0; i < n; i++) {
                layers[i]->update_net();
            }
        };
        float * output;
        size_t output_size;
        size_t workspace_size = 0;
        void add_layer(layer &l) {
            if (l.get_size() > workspace_size) {
                workspace_size = l.get_size();
            }
            n++;
            layers.push_back(&l);
        };

        float calc_loss(float* gt, float* pred, int n) {
            output_size = layers.back()->get_size();
            delta = (float *)calloc(output_size, sizeof(float));
            for (int i = 0; i < n; i++) {
                err += (1.f/2) * pow((gt[i] - pred[i]), 2);
                delta[i] = pred[i] - gt[i];
            }

            return err;
        };

    private:
        float err = 0.0f;
        float * delta;
};


int main() {
    float input[] = {0.05, 0.1};
    float w1[] = {0.15, 0.25,
                  0.2, 0.3};
    float b1{0.35};
    float w2[] = {0.4, 0.5,
                  0.45, 0.55};
    float b2{0.6};

    float ground_truth[] = {0.01, 0.99};

    conn_layer conn1{2, 2};
    conn1.load_weight(w1);
    conn1.load_bias(&b1);

    conn_layer conn2{2, 2};
    conn2.load_weight(w2);
    conn2.load_bias(&b2);

    network net;
    net.add_layer(conn1);
    net.add_layer(conn2);

    size_t output_size = 2;
    int n_epochs = 2;
    for (int e = 0; e < n_epochs; e++){
        float * output = (float *)calloc(output_size, sizeof(float));
        net.forward(input, output);
        // for (int i = 0; i < output_size; i++){
        //     LOG("output " << i<< ": " << output[i]);
        // }

        float err = net.calc_loss(ground_truth, output, output_size);
        LOG("Network error: " << err);

        net.backward();

        net.update();

    }
    return 0;
}