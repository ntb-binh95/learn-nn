#include <memory>
#include <vector>

#include "convolution.h"
#include "network.h"

using namespace std;

int main(int argc, char *argv[]) {
    int batchSize = 1;
    unique_ptr<Network> net = make_unique<Network>(batchSize);

    int c = 1; // in channel
    int w = 5, h = 5; // width, height
    int n = 1; // out channel
    int k =3; // kernel size
    int stride = 1;
    int pad = 0;

    shared_ptr<Convolution> conv1 = make_shared<Convolution>(c, w, h, n, k, batchSize, pad, stride);
    net->add_layer(conv1);

    net->build();

    unique_ptr<float[]> input = make_unique<float[]>(w * h * c);
    // for(int i = 0; i < w*h*c; i++) {
    //     LOG(input[i]);
    // }

    net->forward_net(input.get());
    LOG("End!!");
    return 0;
}