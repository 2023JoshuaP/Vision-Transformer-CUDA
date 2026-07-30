#include "Activation.cuh"
#include <vector>
#include <algorithm>
#include <cmath>

__global__ void kernel_sigmoid_fwd(const double *in, double *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = 1.0 / (1.0 + exp(-in[idx]));
    }
}

__global__ void kernel_sigmoid_deriv(const double *a, double *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * (1.0 - a[idx]);
    }
}

__global__ void kernel_relu_fwd(const double *in, double *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] > 0.0 ? in[idx] : 0.0;
    }
}

__global__ void kernel_relu_deriv(const double *a, double *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] > 0.0 ? 1.0 : 0.0;
    }
}

__global__ void kernel_softmax(const double *in, double *out, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const double *row_in = in + row * cols;
    double *row_out = out + row * cols;

    double row_max = row_in[0];
    for (int j = 1; j < cols; j++) {
        if (row_in[j] > row_max) row_max = row_in[j];
    }
    double sum_exp = 0.0;
    for (int j = 0; j < cols; j++) {
        row_out[j] = exp(row_in[j] - row_max);
        sum_exp += row_out[j];
    }

    for (int j = 0; j < cols; j++) {
        row_out[j] /= sum_exp;
    }
}

static inline int grid1d(int n, int block = 256) {
    return (n + block - 1) / block;
}

Matrix Sigmoid::forward(const Matrix &z) const {
    int n = z.rows * z.cols;
    Matrix res(z.rows, z.cols);
    kernel_sigmoid_fwd<<<grid1d(n), 256>>>(z.d_data, res.d_data, n);
    CUDA_CHECK(cudaGetLastError());
    return res;
}

Matrix Sigmoid::derivative(const Matrix &a) const {
    int n = a.rows * a.cols;
    Matrix res(a.rows, a.cols);
    kernel_sigmoid_deriv<<<grid1d(n), 256>>>(a.d_data, res.d_data, n);
    CUDA_CHECK(cudaGetLastError());
    return res;
}

Matrix ReLU::forward(const Matrix &z) const {
    int n = z.rows * z.cols;
    Matrix res(z.rows, z.cols);
    kernel_relu_fwd<<<grid1d(n), 256>>>(z.d_data, res.d_data, n);
    CUDA_CHECK(cudaGetLastError());
    return res;
}

Matrix ReLU::derivative(const Matrix &a) const {
    int n = a.rows * a.cols;
    Matrix res(a.rows, a.cols);
    kernel_relu_deriv<<<grid1d(n), 256>>>(a.d_data, res.d_data, n);
    CUDA_CHECK(cudaGetLastError());
    return res;
}

Matrix Softmax::forward(const Matrix &z) {
    Matrix res(z.rows, z.cols);
    kernel_softmax<<<z.rows, 1>>>(z.d_data, res.d_data, z.rows, z.cols);
    CUDA_CHECK(cudaGetLastError());
    return res;
}
