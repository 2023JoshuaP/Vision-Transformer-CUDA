#include "ConvolutionalLayer.cuh"
#include <cmath>
#include <random>
#include <stdexcept>
#include <iostream>

using namespace std;

// Forward Kernel
__global__ void conv_forward_kernel(
    const double* input, const double* kernels, const double* biases, double* output,
    int in_c, int in_h, int in_w,
    int out_c, int out_h, int out_w,
    int k_size, int stride, int padding
) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int f = blockIdx.z * blockDim.z + threadIdx.z;

    if (w < out_w && h < out_h && f < out_c) {
        double sum = biases[f];
        for (int c = 0; c < in_c; c++) {
            for (int kh = 0; kh < k_size; kh++) {
                for (int kw = 0; kw < k_size; kw++) {
                    int in_h_idx = h * stride - padding + kh;
                    int in_w_idx = w * stride - padding + kw;
                    
                    if (in_h_idx >= 0 && in_h_idx < in_h && in_w_idx >= 0 && in_w_idx < in_w) {
                        double in_val = input[(c * in_h + in_h_idx) * in_w + in_w_idx];
                        double k_val = kernels[((f * in_c + c) * k_size + kh) * k_size + kw];
                        sum += in_val * k_val;
                    }
                }
            }
        }
        output[(f * out_h + h) * out_w + w] = sum;
    }
}

// Backward Kernel (updates kernels and biases, computes input grad)
__global__ void conv_backward_weights_kernel(
    const double* input, const double* grad_output, double* kernels, double* biases,
    int in_c, int in_h, int in_w,
    int out_c, int out_h, int out_w,
    int k_size, int stride, int padding, double lr
) {
    // Thread computes gradient for one weight: (f, c, kh, kw)
    int f = blockIdx.z * blockDim.z + threadIdx.z;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int k_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (f < out_c && c < in_c && k_idx < k_size * k_size) {
        int kh = k_idx / k_size;
        int kw = k_idx % k_size;

        double weight_grad = 0.0;
        for (int h = 0; h < out_h; h++) {
            for (int w = 0; w < out_w; w++) {
                int in_h_idx = h * stride - padding + kh;
                int in_w_idx = w * stride - padding + kw;
                
                if (in_h_idx >= 0 && in_h_idx < in_h && in_w_idx >= 0 && in_w_idx < in_w) {
                    double go = grad_output[(f * out_h + h) * out_w + w];
                    double in_val = input[(c * in_h + in_h_idx) * in_w + in_w_idx];
                    weight_grad += go * in_val;
                }
            }
        }
        
        // Update weights
        int kernel_flat_idx = ((f * in_c + c) * k_size + kh) * k_size + kw;
        atomicAdd(&kernels[kernel_flat_idx], -lr * weight_grad);
        
        // Update biases (only one thread per filter should do this, say c=0, k_idx=0)
        if (c == 0 && k_idx == 0) {
            double bias_grad = 0.0;
            for (int h = 0; h < out_h; h++) {
                for (int w = 0; w < out_w; w++) {
                    bias_grad += grad_output[(f * out_h + h) * out_w + w];
                }
            }
            atomicAdd(&biases[f], -lr * bias_grad);
        }
    }
}

__global__ void conv_backward_input_kernel(
    const double* grad_output, const double* kernels, double* grad_input,
    int in_c, int in_h, int in_w,
    int out_c, int out_h, int out_w,
    int k_size, int stride, int padding
) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z * blockDim.z + threadIdx.z;

    if (w < in_w && h < in_h && c < in_c) {
        double grad = 0.0;
        for (int f = 0; f < out_c; f++) {
            for (int kh = 0; kh < k_size; kh++) {
                for (int kw = 0; kw < k_size; kw++) {
                    int out_h_idx = (h + padding - kh);
                    int out_w_idx = (w + padding - kw);
                    
                    if (out_h_idx % stride == 0 && out_w_idx % stride == 0) {
                        out_h_idx /= stride;
                        out_w_idx /= stride;
                        
                        if (out_h_idx >= 0 && out_h_idx < out_h && out_w_idx >= 0 && out_w_idx < out_w) {
                            double go = grad_output[(f * out_h + out_h_idx) * out_w + out_w_idx];
                            double k_val = kernels[((f * in_c + c) * k_size + kh) * k_size + kw];
                            grad += go * k_val;
                        }
                    }
                }
            }
        }
        grad_input[(c * in_h + h) * in_w + w] = grad;
    }
}

ConvolutionalLayer::ConvolutionalLayer(int input_channels, int output_channels, int kernel, int stride, int padding, int seed) 
    : input_channels_(input_channels), output_channels_(output_channels), kernel_(kernel), stride_(stride), padding_(padding) {
    
    int kernels_size = output_channels_ * input_channels_ * kernel_ * kernel_;
    CUDA_CHECK(cudaMalloc(&d_kernels_, kernels_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_biases_, output_channels_ * sizeof(double)));
    
    // Initialize weights on host and copy to device
    mt19937 rng(seed);
    double scale = sqrt(2.0 / (input_channels * kernel * kernel));
    normal_distribution<double> dist(0.0, scale);
    
    double* h_kernels = new double[kernels_size];
    for (int i = 0; i < kernels_size; i++) {
        h_kernels[i] = dist(rng);
    }
    CUDA_CHECK(cudaMemcpy(d_kernels_, h_kernels, kernels_size * sizeof(double), cudaMemcpyHostToDevice));
    delete[] h_kernels;
    
    // Initialize biases to 0
    double* h_biases = new double[output_channels_];
    for (int i = 0; i < output_channels_; i++) h_biases[i] = 0.0;
    CUDA_CHECK(cudaMemcpy(d_biases_, h_biases, output_channels_ * sizeof(double), cudaMemcpyHostToDevice));
    delete[] h_biases;
}

ConvolutionalLayer::~ConvolutionalLayer() {
    if (d_kernels_) cudaFree(d_kernels_);
    if (d_biases_) cudaFree(d_biases_);
}

int ConvolutionalLayer::size_out(int input_size, int kernel, int stride, int padding) {
    return (input_size - kernel + 2 * padding) / stride + 1;
}

Tensor3D ConvolutionalLayer::forward(const Tensor3D& input) {
    if (input.channels != input_channels_) {
        throw invalid_argument("Input channels do not match layer configuration.");
    }
    input_height_ = input.height;
    input_width_ = input.width;
    input_cache_ = input; // Uses the updated Tensor3D copy constructor

    int out_h = size_out(input.height, kernel_, stride_, padding_);
    int out_w = size_out(input.width, kernel_, stride_, padding_);
    Tensor3D output(output_channels_, out_h, out_w);

    dim3 blockSize(16, 16, 1);
    dim3 numBlocks((out_w + blockSize.x - 1) / blockSize.x, 
                   (out_h + blockSize.y - 1) / blockSize.y,
                   (output_channels_ + blockSize.z - 1) / blockSize.z);

    conv_forward_kernel<<<numBlocks, blockSize>>>(
        input.d_data, d_kernels_, d_biases_, output.d_data,
        input_channels_, input.height, input.width,
        output_channels_, out_h, out_w,
        kernel_, stride_, padding_
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return output;
}

Tensor3D ConvolutionalLayer::backward(const Tensor3D& gradient_output, double learning_rate) {
    int out_h = gradient_output.height;
    int out_w = gradient_output.width;

    Tensor3D gradient_input(input_channels_, input_height_, input_width_, 0.0);

    // Compute gradient w.r.t input
    dim3 blockSizeIn(16, 16, 1);
    dim3 numBlocksIn((input_width_ + blockSizeIn.x - 1) / blockSizeIn.x, 
                     (input_height_ + blockSizeIn.y - 1) / blockSizeIn.y,
                     (input_channels_ + blockSizeIn.z - 1) / blockSizeIn.z);

    conv_backward_input_kernel<<<numBlocksIn, blockSizeIn>>>(
        gradient_output.d_data, d_kernels_, gradient_input.d_data,
        input_channels_, input_height_, input_width_,
        output_channels_, out_h, out_w,
        kernel_, stride_, padding_
    );
    CUDA_CHECK(cudaGetLastError());
    
    // Compute gradient w.r.t weights and update them
    int k_size2 = kernel_ * kernel_;
    dim3 blockSizeW(k_size2, 1, 1);
    dim3 numBlocksW(1, input_channels_, output_channels_);

    conv_backward_weights_kernel<<<numBlocksW, blockSizeW>>>(
        input_cache_.d_data, gradient_output.d_data, d_kernels_, d_biases_,
        input_channels_, input_height_, input_width_,
        output_channels_, out_h, out_w,
        kernel_, stride_, padding_, learning_rate
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return gradient_input;
}