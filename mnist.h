#pragma once

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
        mnist(int batch): batch{batch} {};
        int read_training_images();
        int read_training_labels();
        std::tuple<image, int> get_next_item();
        batch_item get_next_batch();

    private:
        uint32_t be2le(uint32_t uVal);
        uint32_t read_header(const std::unique_ptr<char[]>& buffer, size_t position);
        std::vector<image> training_images;
        std::vector<uint8_t> training_labels;
        size_t index = 0;
        size_t dataset_size = 0;
        void set_training_size(size_t size);
        int batch;
        int imageSize = 0;
};