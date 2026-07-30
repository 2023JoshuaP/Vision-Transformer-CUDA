#include "ActivationLayer.cuh"
#include <iostream>

__global__ void relu_tensor_forward_kernel(const double* input, double* output, double* mask, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (input[idx] > 0) {
            output[idx] = input[idx];
            mask[idx] = 1.0;
        } else {
            output[idx] = 0.0;
            mask[idx] = 0.0;
        }
    }
}

__global__ void relu_tensor_backward_kernel(const double* grad_output, const double* mask, double* grad_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_input[idx] = mask[idx] * grad_output[idx];
    }
}

Tensor3D ReLUActivationLayer::forward(const Tensor3D& input) {
    int sz = input.size();
    Tensor3D output(input.channels, input.height, input.width);
    mask_ = Tensor3D(input.channels, input.height, input.width, 0.0);

    int blockSize = 256;
    int numBlocks = (sz + blockSize - 1) / blockSize;

    relu_tensor_forward_kernel<<<numBlocks, blockSize>>>(input.d_data, output.d_data, mask_.d_data, sz);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return output;
}

Tensor3D ReLUActivationLayer::backward(const Tensor3D& gradient_output) {
    int sz = gradient_output.size();
    Tensor3D gradient_input(gradient_output.channels, gradient_output.height, gradient_output.width);

    int blockSize = 256;
    int numBlocks = (sz + blockSize - 1) / blockSize;

    relu_tensor_backward_kernel<<<numBlocks, blockSize>>>(gradient_output.d_data, mask_.d_data, gradient_input.d_data, sz);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return gradient_input;
}