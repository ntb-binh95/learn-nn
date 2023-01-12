#ifndef _LAYER_
#define _LAYER_

#include <iostream>

typedef enum {CONNECTED, CONVOLUTION, BLANK} LAYER_TYPE;

class Layer {
    friend class Network;
    public:
        Layer(std::string name, LAYER_TYPE type): name{name}, type{type} {};
        std::string name = "blank";
        LAYER_TYPE type = BLANK;
    
    protected:
        virtual void forward(float * inp) = 0;
        virtual size_t get_size() = 0;
        virtual size_t get_input_size() = 0;
        virtual void backward(float * delta) = 0;
        virtual void update() = 0;
        virtual void save_weight(std::ofstream &weight_file) = 0;
        virtual void load_weight(std::ifstream &weight_file) = 0;
        int batch = 0;
};

#endif