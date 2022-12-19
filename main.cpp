#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <iomanip>

#include "gemm.h"
#include "mnist.h"
#include "blas.h"
#include "progress_bar.h"

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
        int batch = 0;
};

class Connected : public Layer {
    public:
        Connected(int inputSize, int n, int batchSize) : Layer("connected", CONNECTED), 
        neuralSize{n}, inputSize{inputSize}{
            batch = batchSize;
            weights = make_unique<float[]>(neuralSize * inputSize);   
            bias = make_unique<float[]>(neuralSize);   
            output = make_unique<float[]>(batch * neuralSize);   
            input = make_unique<float[]>(batch * inputSize);
            gradient = make_unique<float[]>(batch * neuralSize);
            update_weights = make_unique<float[]>(neuralSize*inputSize);
            update_bias = make_unique<float[]>(neuralSize);
            // temp variable
            layer_delta = make_unique<float[]>(batch * neuralSize);

            //random initialize weight and bias
            // srand(time(0)); use for different seeding
            float scale = sqrt(2./inputSize);
            for(int i = 0; i < inputSize * neuralSize; i++){
                weights[i] = scale * rand_uniform(-1, 1);
            }
            for(int i = 0; i < neuralSize; i++){
                bias[i] = 0;
            }
            // bias[0] = 0; // bias should be size of output for more general case.
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
        void backward(float * delta) override {
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

        void update() override {
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

    private:
        int neuralSize;
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
            for (int j = 0; j < neuralSize; j++){
                update_bias[j] = 0;
            }
            for(int b = 0; b < batch; b++) {
                for(int i = 0; i < neuralSize; i++){
                    update_bias[i] += delta[b*neuralSize + i]; 
                }
            }
            
        };
};

class Network {
    public:
        Network(int batchSize): batch{batchSize} {};
        vector<shared_ptr<Layer>> layers;
        size_t n = 0;
        void forward_net(float * inp) {
            // copy input to the workspace
            for(int i = 0; i < batch * inputSize; i++){
                workspace[i] = inp[i];
            }

            // loop over all layer in network
            for (int l = 0; l < n; l++) {
                // LOG("Forwarding layer " << l << "!");
                layers[l]->forward(workspace.get());
            }

            // get output after forwarding 
            // cout << "output: " ;
            for (int i = 0; i < batch * outputSize; i++) {
                output[i] = workspace[i];
                // cout << output[i] << " ";
            }
            // cout << endl;
        };

        void predict(float * inp) {
            // copy input to the workspace
            for(int i = 0; i < batch * inputSize; i++){
                workspace[i] = inp[i];
            }

            // loop over all layer in network
            for (int l = 0; l < n; l++) {
                // LOG("Forwarding layer " << l << "!");
                layers[l]->forward(workspace.get());
            }

            // get output after forwarding 
            for (int i = 0; i < batch * outputSize; i++) {
                output[i] = workspace[i];
            }

            float max = 0;
            int max_index = 0;
            for (int b = 0; b < batch; b++) {
                for (int i = 0; i < outputSize; i++) {
                    if (max < output[b * outputSize + i]){
                        max = output[b * outputSize + i];
                        max_index = i;
                    }
                }
                LOG("Output : " << max_index);
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
            for(int b = 0; b < batch; b++){
                for (int i = 0; i < n; i++) {
                    err += (1.f/2) * pow((gt[b * outputSize + i] - output[b * outputSize + i]), 2) / batch;
                    delta[b * outputSize + i] = output[b * outputSize + i] - gt[b * outputSize + i];
                }
            }

            return err;
        };

        void build() {
            LOG(maxSize << " " << outputSize);
            workspace = make_unique<float[]>(batch * maxSize);
            output = make_unique<float[]>(batch * outputSize);
            delta = make_unique<float[]>(batch * maxSize);
        };
    
    private:
        size_t inputSize = 0;
        size_t outputSize = 0;
        size_t maxSize = 0;
        float err = 0.0f;
        int batch = 0;
        unique_ptr<float[]> delta;
        unique_ptr<float[]> output;
        unique_ptr<float[]> workspace;
};


int main(int argc, char** argv) {
    /*TODO: 
    1. Add batch processing (DONE)
    2. using axpy in blas (DONE)
    3. Implement predict
    4. Implement save weight and load weight function.
    */
    // Load data
    int batchSize = 8;
    mnist dataset{batchSize};
    size_t trainImageSize = dataset.read_training_images();
    size_t trainLabelSize = dataset.read_training_labels();
    assert(trainImageSize == trainLabelSize);

    // build model
    int inputSize = 784;
    int hiddenSize = 128;
    int outputSize = 10;

    shared_ptr<Connected> conn1 = make_shared<Connected>(inputSize,hiddenSize, batchSize);
    shared_ptr<Connected> conn2 = make_shared<Connected>(hiddenSize,outputSize, batchSize);
    unique_ptr<Network> net = make_unique<Network>(batchSize);
    net->add_layer(conn1);
    net->add_layer(conn2);

    net->build();

    int epochs = 5;
    for (int e = 0; e < epochs; e++){
        pBar bar;
        float avg_err = 0;
        LOG("Epoch: " << e);
        for (int d = 0; d < trainImageSize/batchSize; d++){
            batch_item batch = dataset.get_next_batch(); 
            unique_ptr<float[]> input = std::move(get<0>(batch));
            unique_ptr<float[]> ground_truth = std::move(get<1>(batch));
        
            net->forward_net(input.get());

            float err = net->calc_loss(ground_truth.get(), outputSize);
            // LOG("Network err " << err);
            net->backward_net();
            net->update_net();

            avg_err = (avg_err*d + err)/(d+1);
            if(d % (trainImageSize/batchSize/100) == 0){
                bar.update(1);
                bar.print();
                bar.update_err(avg_err);
            }
        }
        LOG("");
    }

    return 0;
}