#include "mnist.h"

#define LOG(x) std::cout << x << std::endl


uint32_t mnist::be2le(uint32_t uVal) {
    return (uVal << 24) | (uVal >> 24) | (uVal & 0x00FF0000) >> 8 | (uVal & 0x0000FF00) << 8;
}

uint32_t mnist::read_header(const std::unique_ptr<char[]>& buffer, size_t position) {
    auto header = reinterpret_cast<uint32_t*>(buffer.get());

    auto value = *(header + position); // currenly value is in Big Endian format

    return be2le(value); 
};


int mnist::read_images(std::string path) {
    const std::string images_path = path;
    std::ifstream images_file;

    // std::ios::in: open for reading
    // std::ios::binary: operation performed in binary mode rather than text
    // std::ios::ate: (at end) the output position starts at the end of file
    images_file.open(images_path, std::ios::in | std::ios::binary |std::ios::ate);

    if (!images_file) {
        LOG("Error open file"); 
    }

    // get current position, since we seek from end, it return file size
    const auto filesize = images_file.tellg();     
    std::unique_ptr<char[]> buffer(new char[filesize]);

    // read the entire file at once
    images_file.seekg(0, std::ios::beg); //seek to the begin of file.
    images_file.read(buffer.get(), filesize); // buffer.get() return a pointer to the managed object
    images_file.close(); // close file

    if (!buffer) { return -1;} // return if cannot read buffer.

    unsigned magic = read_header(buffer, 0); // conver BE to LE representation

    if (magic != 0x803) {
        LOG("This is not training mnist dataset");
        return -1;
    } 
    unsigned data_size = read_header(buffer, 1);

    unsigned image_width = read_header(buffer, 2);
    unsigned image_height = read_header(buffer, 3);
    imageSize = image_width * image_height;

    assert(filesize==(image_height * image_width * data_size + 16));

    // cast to unsigned char is necessary cause signedness of char is platform-specific
    uint8_t *images_buffer = reinterpret_cast<uint8_t*>(buffer.get() + 16);

    for (int i = 0; i < data_size; i++){
        image img;
        for(int j = 0; j < image_height * image_width; j++) {
            auto pixel = *images_buffer++;
            img.push_back(static_cast<uint8_t>(pixel));
        }
        images.push_back(img);
    }
    if(data_size != dataset_size){
        dataset_size = data_size;
    }
    return images.size();
}

int mnist::read_labels(std::string path) {
    const std::string labels_path = path;
    std::ifstream labels_file;

    // std::ios::in: open for reading
    // std::ios::binary: operation performed in binary mode rather than text
    // std::ios::ate: (at end) the output position starts at the end of file
    labels_file.open(labels_path, std::ios::in | std::ios::binary |std::ios::ate);

    if (!labels_file) {
        LOG("Error open file"); 
    }

    // get current position, since we seek from end, it return file size
    const auto size = labels_file.tellg();     
    std::unique_ptr<char[]> buffer = std::make_unique<char[]>(size);

    // read the entire file at once
    labels_file.seekg(0, std::ios::beg); //seek to the begin of file.
    labels_file.read(buffer.get(), size); // buffer.get() return a pointer to the managed object
    labels_file.close(); // close file

    if (!buffer) { return -1;} // return if cannot read buffer.

    unsigned magic = read_header(buffer, 0); // conver BE to LE representation

    if (magic != 0x801) {
        LOG("This is not mnist label dataset file");
        return -1;
    } 
    unsigned label_size = read_header(buffer, 1);

    // cast to unsigned char is necessary cause signedness of char is platform-specific
    uint8_t *labels_buffer = reinterpret_cast<uint8_t*>(buffer.get() + 8);

    for (int i = 0; i <label_size; i++){
        uint8_t label = *labels_buffer++;
        labels.push_back(label);
    }
    if(label_size != dataset_size) {
        dataset_size = label_size;
    }
    return labels.size();
}

std::tuple<image, int> mnist::get_next_item() {
    int current_index = index++;
    if (index == dataset_size) {
        index = 0;
    }
    return std::make_tuple(images[current_index], labels[current_index]);
}

batch_item mnist::get_next_batch() {
    std::unique_ptr<float[]> input = std::make_unique<float[]>(imageSize * batch);
    std::unique_ptr<float[]> ground_truth = std::make_unique<float[]>(10 * batch);
    for(int i = 0; i < batch; i++) {
        auto item = get_next_item();
        image img = std::get<0>(item);
        int label = std::get<1>(item);
        for (int j = 0; j < imageSize; j++){
            input[i*imageSize + j] = img[j] / 255.0;
        }
        ground_truth[i*10 + label] = 1;
    }
    return std::make_tuple(std::move(input), std::move(ground_truth));
}

size_t mnist::get_dataset_size(){
    return dataset_size;
}