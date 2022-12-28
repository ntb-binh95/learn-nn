#ifndef __MINIST_H_
#define __MINIST_H_
#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <cassert>
#include <tuple>

typedef std::vector<uint8_t> image;
typedef std::tuple<std::unique_ptr<float[]>, std::unique_ptr<float[]>> batch_item;

class mnist {
    public:
        mnist(std::string datatype, int batch): datatype{datatype}, batch{batch} {
            if(datatype == "train") {
                int num_images = read_images("train-images.idx3-ubyte");
                int num_labels = read_labels("train-labels.idx1-ubyte");
                assert(num_images == num_labels);
            } else if (datatype == "test") {
                int num_images = read_images("t10k-images.idx3-ubyte");
                int num_labels = read_labels("t10k-labels.idx1-ubyte");
                assert(num_images == num_labels);
            } else {
                std::cout << "invalid data type: ""train"" or ""test""" << std::endl;
            }
        };
        size_t get_dataset_size();
        std::tuple<image, int> get_next_item();
        batch_item get_next_batch();

    private:
        int read_images(std::string path);
        int read_labels(std::string path);
        uint32_t be2le(uint32_t uVal);
        uint32_t read_header(const std::unique_ptr<char[]>& buffer, size_t position);
        std::vector<image> images;
        std::vector<uint8_t> labels;
        size_t index = 0;
        size_t dataset_size = 0;
        int batch;
        int imageSize = 0;
        std::string datatype = "train";
};

#endif