#pragma once

#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <cassert>
#include <tuple>

typedef std::vector<uint8_t> image;

class mnist {
    public:
        int read_training_images();
        int read_training_labels();
        std::tuple<image, int> get_next_item();

    private:
        uint32_t be2le(uint32_t uVal);
        uint32_t read_header(const std::unique_ptr<char[]>& buffer, size_t position);
        std::vector<image> training_images;
        std::vector<uint8_t> training_labels;
        size_t items = 0;
        size_t index = 0;
};