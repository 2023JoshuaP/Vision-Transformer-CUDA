#include "Tensor3D.cuh"
#include <iostream>

// CUDA Kernels for operations
__global__ void tensor_add_kernel(const double* a, const double* b, double* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

__global__ void tensor_sub_kernel(const double* a, const double* b, double* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] - b[idx];
    }
}

__global__ void tensor_mul_scalar_kernel(const double* a, double scalar, double* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] * scalar;
    }
}

__global__ void tensor_fill_kernel(double* d_data, double value, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_data[idx] = value;
    }
}

// Constructors & Destructor
Tensor3D::Tensor3D() : channels(0), height(0), width(0), d_data(nullptr), owns_data(false) {}

Tensor3D::Tensor3D(int c, int h, int w, double value) 
    : channels(c), height(h), width(w), owns_data(true) {
    int sz = size();
    if (sz > 0) {
        CUDA_CHECK(cudaMalloc(&d_data, sz * sizeof(double)));
        int blockSize = 256;
        int numBlocks = (sz + blockSize - 1) / blockSize;
        tensor_fill_kernel<<<numBlocks, blockSize>>>(d_data, value, sz);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        d_data = nullptr;
    }
}

Tensor3D::~Tensor3D() {
    if (owns_data && d_data != nullptr) {
        cudaFree(d_data);
        d_data = nullptr;
    }
}

// Copy Constructor
Tensor3D::Tensor3D(const Tensor3D &other) 
    : channels(other.channels), height(other.height), width(other.width), owns_data(true) {
    int sz = size();
    if (sz > 0) {
        CUDA_CHECK(cudaMalloc(&d_data, sz * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_data, other.d_data, sz * sizeof(double), cudaMemcpyDeviceToDevice));
    } else {
        d_data = nullptr;
    }
}

// Copy Assignment
Tensor3D &Tensor3D::operator=(const Tensor3D &other) {
    if (this != &other) {
        if (owns_data && d_data != nullptr) {
            cudaFree(d_data);
        }
        channels = other.channels;
        height = other.height;
        width = other.width;
        owns_data = true;
        int sz = size();
        if (sz > 0) {
            CUDA_CHECK(cudaMalloc(&d_data, sz * sizeof(double)));
            CUDA_CHECK(cudaMemcpy(d_data, other.d_data, sz * sizeof(double), cudaMemcpyDeviceToDevice));
        } else {
            d_data = nullptr;
        }
    }
    return *this;
}

// Move Constructor
Tensor3D::Tensor3D(Tensor3D &&other) noexcept
    : channels(other.channels), height(other.height), width(other.width), d_data(other.d_data), owns_data(other.owns_data) {
    other.d_data = nullptr;
    other.owns_data = false;
    other.channels = 0;
    other.height = 0;
    other.width = 0;
}

// Move Assignment
Tensor3D &Tensor3D::operator=(Tensor3D &&other) noexcept {
    if (this != &other) {
        if (owns_data && d_data != nullptr) {
            cudaFree(d_data);
        }
        channels = other.channels;
        height = other.height;
        width = other.width;
        d_data = other.d_data;
        owns_data = other.owns_data;
        
        other.d_data = nullptr;
        other.owns_data = false;
        other.channels = 0;
        other.height = 0;
        other.width = 0;
    }
    return *this;
}

// Host Transfers
void Tensor3D::toHost(double *host_buf) const {
    CUDA_CHECK(cudaMemcpy(host_buf, d_data, size() * sizeof(double), cudaMemcpyDeviceToHost));
}

void Tensor3D::fromHost(const double *host_buf) {
    CUDA_CHECK(cudaMemcpy(d_data, host_buf, size() * sizeof(double), cudaMemcpyHostToDevice));
}

// Flatten
Matrix Tensor3D::flatten() const {
    Matrix result(1, size());
    CUDA_CHECK(cudaMemcpy(result.d_data, d_data, size() * sizeof(double), cudaMemcpyDeviceToDevice));
    return result;
}

// Reconstruct
Tensor3D Tensor3D::reconstructureFlatMatrix(const Matrix &matrix, int channels, int height, int width) {
    Tensor3D result(channels, height, width);
    CUDA_CHECK(cudaMemcpy(result.d_data, matrix.d_data, result.size() * sizeof(double), cudaMemcpyDeviceToDevice));
    return result;
}

// Operations
Tensor3D Tensor3D::operator+(const Tensor3D &other) const {
    Tensor3D result(channels, height, width);
    int sz = size();
    int blockSize = 256;
    int numBlocks = (sz + blockSize - 1) / blockSize;
    tensor_add_kernel<<<numBlocks, blockSize>>>(d_data, other.d_data, result.d_data, sz);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return result;
}

Tensor3D Tensor3D::operator-(const Tensor3D &other) const {
    Tensor3D result(channels, height, width);
    int sz = size();
    int blockSize = 256;
    int numBlocks = (sz + blockSize - 1) / blockSize;
    tensor_sub_kernel<<<numBlocks, blockSize>>>(d_data, other.d_data, result.d_data, sz);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return result;
}

Tensor3D Tensor3D::operator*(double scalar) const {
    Tensor3D result(channels, height, width);
    int sz = size();
    int blockSize = 256;
    int numBlocks = (sz + blockSize - 1) / blockSize;
    tensor_mul_scalar_kernel<<<numBlocks, blockSize>>>(d_data, scalar, result.d_data, sz);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return result;
}

Tensor3D tensorRandom(int channels, int height, int width, double scale, mt19937 &rng) {
    Tensor3D result(channels, height, width);
    int sz = result.size();
    double* host_temp = new double[sz];
    normal_distribution<double> distance(0.0, scale);
    for (int i = 0; i < sz; i++) {
        host_temp[i] = distance(rng);
    }
    result.fromHost(host_temp);
    delete[] host_temp;
    return result;
}