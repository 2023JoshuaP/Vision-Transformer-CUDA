#include "PoolingLayer.cuh"
#include <limits>
#include <stdexcept>
#include <iostream>

using namespace std;

__global__ void pool_forward_kernel(
    const double* input, double* output, int* selected_indices,
    int channels, int in_h, int in_w,
    int out_h, int out_w,
    int pool_size, int stride,
    PoolingType type
) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z * blockDim.z + threadIdx.z;

    if (w < out_w && h < out_h && c < channels) {
        int out_idx = (c * out_h + h) * out_w + w;
        
        if (type == PoolingType::Average) {
            double sum = 0.0;
            for (int ph = 0; ph < pool_size; ph++) {
                for (int pw = 0; pw < pool_size; pw++) {
                    int ih = h * stride + ph;
                    int iw = w * stride + pw;
                    sum += input[(c * in_h + ih) * in_w + iw];
                }
            }
            output[out_idx] = sum / (pool_size * pool_size);
        } else {
            bool isMax = (type == PoolingType::Max);
            double best_val = isMax ? -INFINITY : INFINITY;
            int best_ih = 0, best_iw = 0;

            for (int ph = 0; ph < pool_size; ph++) {
                for (int pw = 0; pw < pool_size; pw++) {
                    int ih = h * stride + ph;
                    int iw = w * stride + pw;
                    double val = input[(c * in_h + ih) * in_w + iw];
                    
                    bool is_better = isMax ? (val > best_val) : (val < best_val);
                    if (is_better) {
                        best_val = val;
                        best_ih = ih;
                        best_iw = iw;
                    }
                }
            }
            output[out_idx] = best_val;
            selected_indices[out_idx] = (c * in_h + best_ih) * in_w + best_iw;
        }
    }
}

__global__ void pool_backward_kernel(
    const double* grad_output, double* grad_input, const int* selected_indices,
    int channels, int in_h, int in_w,
    int out_h, int out_w,
    int pool_size, int stride,
    PoolingType type
) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z * blockDim.z + threadIdx.z;

    if (w < out_w && h < out_h && c < channels) {
        int out_idx = (c * out_h + h) * out_w + w;
        double grad_val = grad_output[out_idx];

        if (type == PoolingType::Average) {
            double dist = grad_val / (pool_size * pool_size);
            for (int ph = 0; ph < pool_size; ph++) {
                for (int pw = 0; pw < pool_size; pw++) {
                    int ih = h * stride + ph;
                    int iw = w * stride + pw;
                    atomicAdd(&grad_input[(c * in_h + ih) * in_w + iw], dist);
                }
            }
        } else {
            int in_idx = selected_indices[out_idx];
            atomicAdd(&grad_input[in_idx], grad_val);
        }
    }
}

PoolingLayer::PoolingLayer(int pool_size, int stride, PoolingType type) 
    : pool_size_(pool_size), stride_(stride), type_(type),
      d_selected_indices_(nullptr), allocated_indices_size_(0),
      input_height_(0), input_width_(0), input_channels_(0) {}

PoolingLayer::~PoolingLayer() {
    if (d_selected_indices_) {
        cudaFree(d_selected_indices_);
    }
}

PoolingLayer::PoolingLayer(PoolingLayer&& other) noexcept
    : pool_size_(other.pool_size_), stride_(other.stride_), type_(other.type_),
      d_selected_indices_(other.d_selected_indices_), allocated_indices_size_(other.allocated_indices_size_),
      input_height_(other.input_height_), input_width_(other.input_width_), input_channels_(other.input_channels_) {
    other.d_selected_indices_ = nullptr;
    other.allocated_indices_size_ = 0;
}

PoolingLayer& PoolingLayer::operator=(PoolingLayer&& other) noexcept {
    if (this != &other) {
        if (d_selected_indices_) cudaFree(d_selected_indices_);
        pool_size_ = other.pool_size_;
        stride_ = other.stride_;
        type_ = other.type_;
        d_selected_indices_ = other.d_selected_indices_;
        allocated_indices_size_ = other.allocated_indices_size_;
        input_height_ = other.input_height_;
        input_width_ = other.input_width_;
        input_channels_ = other.input_channels_;
        other.d_selected_indices_ = nullptr;
        other.allocated_indices_size_ = 0;
    }
    return *this;
}

int PoolingLayer::output_size(int input_size, int kernel, int stride) {
    return (input_size - kernel) / stride + 1;
}

string PoolingLayer::getTypeName() const {
    switch (type_) {
        case PoolingType::Max: return "MaxPool";
        case PoolingType::Min: return "MinPool";
        case PoolingType::Average: return "AvgPool";
        default: return "Unknown";
    }
}

Tensor3D PoolingLayer::forward(const Tensor3D& input) {
    input_height_ = input.height;
    input_width_ = input.width;
    input_channels_ = input.channels;

    int out_h = output_size(input_height_, pool_size_, stride_);
    int out_w = output_size(input_width_, pool_size_, stride_);

    if (out_h <= 0 || out_w <= 0) {
        throw invalid_argument("Invalid pooling parameters: output dimensions must be positive.");
    }

    Tensor3D output(input_channels_, out_h, out_w, 0.0);
    
    int required_size = input_channels_ * out_h * out_w;
    if (type_ != PoolingType::Average && required_size > allocated_indices_size_) {
        if (d_selected_indices_) cudaFree(d_selected_indices_);
        CUDA_CHECK(cudaMalloc(&d_selected_indices_, required_size * sizeof(int)));
        allocated_indices_size_ = required_size;
    }

    dim3 blockSize(16, 16, 1);
    dim3 numBlocks((out_w + blockSize.x - 1) / blockSize.x, 
                   (out_h + blockSize.y - 1) / blockSize.y,
                   (input_channels_ + blockSize.z - 1) / blockSize.z);

    pool_forward_kernel<<<numBlocks, blockSize>>>(
        input.d_data, output.d_data, d_selected_indices_,
        input_channels_, input_height_, input_width_,
        out_h, out_w, pool_size_, stride_, type_
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return output;
}

Tensor3D PoolingLayer::backward(const Tensor3D& gradient_output) const {
    Tensor3D gradient_input(input_channels_, input_height_, input_width_, 0.0);
    
    int out_h = gradient_output.height;
    int out_w = gradient_output.width;

    dim3 blockSize(16, 16, 1);
    dim3 numBlocks((out_w + blockSize.x - 1) / blockSize.x, 
                   (out_h + blockSize.y - 1) / blockSize.y,
                   (input_channels_ + blockSize.z - 1) / blockSize.z);

    pool_backward_kernel<<<numBlocks, blockSize>>>(
        gradient_output.d_data, gradient_input.d_data, d_selected_indices_,
        input_channels_, input_height_, input_width_,
        out_h, out_w, pool_size_, stride_, type_
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return gradient_input;
}