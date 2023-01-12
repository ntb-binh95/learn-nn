#include <memory>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>

#include "network.h"
#include "log.h"

std::unique_ptr<int[]> Network::predict(float * inp) {
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

    std::unique_ptr<int[]> batch_index = std::make_unique<int[]>(batch);
    for (int b = 0; b < batch; b++) {
        float max = 0;
        int max_index = 0;
        for (int i = 0; i < outputSize; i++) {
            if (max < output[b * outputSize + i]){
                max = output[b * outputSize + i];
                max_index = i;
            }
        }
        // LOG("Output: " << max_index);
        batch_index[b] = max_index;
    }
    return batch_index;
};

void Network::forward_net(float * inp) {
    // copy input to the workspace
    for(int i = 0; i < batch * inputSize; i++){
        workspace[i] = inp[i];
    }

    // loop over all layer in network
    for (int l = 0; l < n; l++) {
        layers[l]->forward(workspace.get());
    }

    // get output after forwarding 
    for (int i = 0; i < batch * outputSize; i++) {
        output[i] = workspace[i];
    }
};

void Network::backward_net() {
    for (int i = n-1; i > -1; i--) {
        layers[i]->backward(delta.get());
    }
};
void Network::update_net() {
    for (int i = 0; i < n; i++) {
        layers[i]->update();
    }
};


void Network::build() {
    workspace = std::make_unique<float[]>(batch * maxSize);
    output = std::make_unique<float[]>(batch * outputSize);
    delta = std::make_unique<float[]>(batch * maxSize);
};

void Network::save_weights() {
    // save model
    LOG("Saving model to " << saved_model);
    int major = 0;
    int minor = 1;
    int revision = 0;
    LOG("Writing version " << major << " " << minor << " " << revision);

    std::ofstream out_weight;
    out_weight.open(saved_model, std::ios::binary | std::ios::out);
    out_weight.write(reinterpret_cast<const char *>(&major), sizeof(int));
    out_weight.write(reinterpret_cast<const char *>(&minor), sizeof(int));
    out_weight.write(reinterpret_cast<const char *>(&revision), sizeof(int));

    for (int i = 0; i < n; i++) {
        layers[i]->save_weight(out_weight);
    }

    out_weight.close();
}

float Network::calc_loss(float* gt, int n) {
    err = 0.0f;
    for(int b = 0; b < batch; b++){
        for (int i = 0; i < n; i++) {
            err += (1.f/2) * pow((gt[b * outputSize + i] - output[b * outputSize + i]), 2) / batch;
            delta[b * outputSize + i] = output[b * outputSize + i] - gt[b * outputSize + i];
        }
    }

    return err;
};

void Network::load_weights() {
    // load model
    int major = 0;
    int minor = 0;
    int revision = 0;

    std::ifstream in_weight;

    in_weight.open(saved_model, std::ios::binary | std::ios::in);
    in_weight.read(reinterpret_cast<char *>(&major), sizeof(int));
    in_weight.read(reinterpret_cast<char *>(&minor), sizeof(int));
    in_weight.read(reinterpret_cast<char *>(&revision), sizeof(int));
    LOG("Reading version " << major << "." << minor << "." << revision);

    for(int i = 0; i < n; i++) {
        layers[i]->load_weight(in_weight);
    }
    in_weight.close();
}