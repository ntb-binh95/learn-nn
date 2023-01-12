#include <iostream>
#include <vector>
#include <memory>
#include <iomanip>

#include "gemm.h"
#include "mnist.h"
#include "blas.h"
#include "progress_bar.h"
#include "network.h"
#include "connected.h"
#include "log.h"

using namespace std;

void show_matrix(float* mat, int rows, int cols) {
    std::cout << std::setprecision(4);
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++) {
            std::cout << mat[i*cols + j] << "\t";
        }
        std::cout << std::endl;
    }

}

int main(int argc, char** argv) {
    /*TODO: 
    1. Add batch processing (DONE)
    2. using axpy in blas (DONE)
    3. Implement predict (inprogress)
    4. Implement save weight and load weight function. (Done)
    */

    // configuration parameter
    bool isTest = false;
    bool isTrain = false;

    // Load data
    int batchSize = 10;
    mnist dataset{"train",batchSize};
    int trainSize = dataset.get_dataset_size();

    // build model
    int inputSize = 784;
    int hiddenSize = 256;
    int outputSize = 10;

    shared_ptr<Connected> conn1 = make_shared<Connected>(inputSize,hiddenSize, batchSize);
    shared_ptr<Connected> conn2 = make_shared<Connected>(hiddenSize,outputSize, batchSize);
    unique_ptr<Network> net = make_unique<Network>(batchSize);
    net->add_layer<Connected>(conn1);
    net->add_layer<Connected>(conn2);

    net->build();

    // train model
    isTrain = true;
    if(isTrain) {
    int epochs = 6;
    for (int e = 0; e < epochs; e++){
        pBar bar;
        float avg_err = 0;
        LOG("Epoch: " << e);
        for (int d = 0; d < trainSize/batchSize; d++){
            batch_item batch = dataset.get_next_batch(); 
            unique_ptr<float[]> input = std::move(get<0>(batch));
            unique_ptr<float[]> ground_truth = std::move(get<1>(batch));
        
            net->forward_net(input.get());

            float err = net->calc_loss(ground_truth.get(), outputSize);
            // LOG("Network err " << err);
            net->backward_net();
            net->update_net();

            avg_err = (avg_err*d + err)/(d+1);
            if(d % (trainSize/batchSize/100) == 0){
                bar.update(1);
                bar.print();
                bar.update_err(avg_err);
            }
        }
        LOG("");
    }
    net->save_weights();
    }

    isTest = true;
    if (isTest) {
        net->load_weights();
        mnist test_set{"test", batchSize};
        size_t testSize = test_set.get_dataset_size();

        int true_count = 0;
        int num_batch = testSize/batchSize;
        for(int d = 0; d < testSize/batchSize; d++) {
            batch_item batch = dataset.get_next_batch(); 
            unique_ptr<float[]> input = std::move(get<0>(batch));
            unique_ptr<float[]> ground_truth = std::move(get<1>(batch));

            // LOG("batch " << d);
            unique_ptr<int[]> batch_predictions = net->predict(input.get());
            for(int i = 0; i < batchSize; i++){
                int gt_index = 0;
                for(int j = 0; j < 10; j++) {
                    if(ground_truth[i*10 + j] == 1) {
                        gt_index = j;
                        break;
                    }
                }
                // LOG("output each item " << batch_predictions[i] << " " << gt_index);
                if(batch_predictions[i] == gt_index) {
                    true_count++;
                }
            }
        }
        LOG("Accuracy: " << true_count * 100.f/ (num_batch * batchSize));
    }

    return 0;
}