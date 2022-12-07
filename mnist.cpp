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


int mnist::read_training_images() {
    const std::string train_images_path = "train-images.idx3-ubyte";
    std::ifstream train_images_file;

    // std::ios::in: open for reading
    // std::ios::binary: operation performed in binary mode rather than text
    // std::ios::ate: (at end) the output position starts at the end of file
    train_images_file.open(train_images_path, std::ios::in | std::ios::binary |std::ios::ate);

    if (!train_images_file) {
        LOG("Error open file"); 
    }

    // get current position, since we seek from end, it return file size
    const auto size = train_images_file.tellg();     
    std::unique_ptr<char[]> buffer(new char[size]);

    // read the entire file at once
    train_images_file.seekg(0, std::ios::beg); //seek to the begin of file.
    train_images_file.read(buffer.get(), size); // buffer.get() return a pointer to the managed object
    train_images_file.close(); // close file

    if (!buffer) { return -1;} // return if cannot read buffer.

    unsigned magic = read_header(buffer, 0); // conver BE to LE representation

    if (magic != 0x803) {
        LOG("This is not training mnist dataset");
        return -1;
    } 
    unsigned train_size = read_header(buffer, 1);

    unsigned image_width = read_header(buffer, 2);
    unsigned image_height = read_header(buffer, 3);

    assert(size==(image_height * image_width * train_size + 16));

    // cast to unsigned char is necessary cause signedness of char is platform-specific
    uint8_t *images_buffer = reinterpret_cast<uint8_t*>(buffer.get() + 16);

    for (int i = 0; i < train_size; i++){
        image img;
        for(int j = 0; j < image_height * image_width; j++) {
            auto pixel = *images_buffer++;
            img.push_back(static_cast<uint8_t>(pixel));
        }
        training_images.push_back(img);
    }
    if(train_size != dataset_size){
        dataset_size = train_size;
    }
    return training_images.size();
}

int mnist::read_training_labels() {
    const std::string train_labels_path = "train-labels.idx1-ubyte";
    std::ifstream train_labels_file;

    // std::ios::in: open for reading
    // std::ios::binary: operation performed in binary mode rather than text
    // std::ios::ate: (at end) the output position starts at the end of file
    train_labels_file.open(train_labels_path, std::ios::in | std::ios::binary |std::ios::ate);

    if (!train_labels_file) {
        LOG("Error open file"); 
    }

    // get current position, since we seek from end, it return file size
    const auto size = train_labels_file.tellg();     
    std::unique_ptr<char[]> buffer(new char[size]);

    // read the entire file at once
    train_labels_file.seekg(0, std::ios::beg); //seek to the begin of file.
    train_labels_file.read(buffer.get(), size); // buffer.get() return a pointer to the managed object
    train_labels_file.close(); // close file

    if (!buffer) { return -1;} // return if cannot read buffer.

    unsigned magic = read_header(buffer, 0); // conver BE to LE representation

    if (magic != 0x801) {
        LOG("This is not training mnist dataset");
        return -1;
    } 
    unsigned train_size = read_header(buffer, 1);

    // cast to unsigned char is necessary cause signedness of char is platform-specific
    uint8_t *labels_buffer = reinterpret_cast<uint8_t*>(buffer.get() + 8);

    for (int i = 0; i < train_size; i++){
        uint8_t label = *labels_buffer++;
        training_labels.push_back(label);
    }
    if(train_size != dataset_size) {
        dataset_size = train_size;
    }
    return training_labels.size();
}

std::tuple<image, int> mnist::get_next_item() {
    int current_index = index++;
    if (index == dataset_size) {
        index = 0;
    }
    return std::make_tuple(training_images[current_index], training_labels[current_index]);
}