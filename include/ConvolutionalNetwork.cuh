#pragma once

#include "Tensor3D.cuh"
#include "ConvolutionalLayer.cuh"
#include "ActivationLayer.cuh"
#include "PoolingLayer.cuh"
#include "MultiLayerPerceptron.cuh"
#include <vector>
#include <memory>
#include <string>

using namespace std;

class ConvolutionalNetwork {
    public:
        ConvolutionalNetwork(double learning_rate = 0.01, double momentum = 0.9, double weight_decay = 1e-4, int seed = 42);
        void add_convolutional_layer(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0);
        void add_ReLU_layer();
        void add_pooling_layer(int pool_size, int stride, PoolingType type = PoolingType::Max);
        void build(const vector<int> &input_shape, const vector<int> &mlp_hidden_sizes, int num_classes, shared_ptr<ActivationFunction> mlp_activation);
        
        Matrix forward(const Tensor3D &input);
        void backward(const Matrix &y_true);

        TrainHistory train(const vector<Tensor3D> &X_train, const Matrix &y_train, int epochs = 50, int batch_size = 32,
            const vector<Tensor3D> *X_value = nullptr, const Matrix *y_value = nullptr, bool verbose = true, int patience = 20);
        
        int predict(const Tensor3D &input);
        double accuracy(const vector<Tensor3D> &X, const Matrix &y);
        void summary() const;
    
    private:
        double learning_rate_, momentum_, weight_decay_;
        int seed_;
        bool built_;

        vector<ConvolutionalLayer> conv_layers_;
        vector<ReLUActivationLayer> activation_layers_;
        vector<PoolingLayer> pooling_layers_;

        enum class LayerType { Convolution, ReLU, Pooling };
        vector<pair<LayerType, int>> layer_order_;

        unique_ptr<MultiLayerPerceptron> mlp_classificator_;

        vector<Tensor3D> layer_inputs_;
        Tensor3D flatten_input_cache_;
        vector<Matrix> mlp_activations_;

        int flatten_size_;
};