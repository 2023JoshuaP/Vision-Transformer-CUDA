#pragma once

#include "Tensor3D.cuh"
#include <string>

using namespace std;

enum class PoolingType { Max, Min, Average };

class PoolingLayer {
public:
    PoolingLayer(int kernel, int stride, PoolingType type);
    ~PoolingLayer();

    PoolingLayer(const PoolingLayer&) = delete;
    PoolingLayer& operator=(const PoolingLayer&) = delete;

    Tensor3D forward(const Tensor3D& input);
    Tensor3D backward(const Tensor3D& gradient_output) const;

    static int output_size(int input_size, int kernel, int stride);
    string getTypeName() const;

private:
    int pool_size_, stride_;
    PoolingType type_;

    int* d_selected_indices_;
    int allocated_indices_size_;
    int input_height_, input_width_, input_channels_;
};