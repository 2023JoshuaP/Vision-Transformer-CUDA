#include "Matrix.cuh"
#include <cstring>
#include <vector>
#include <algorithm>

__global__ void kernel_fill(double *data, int n, double val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = val;
}

__global__ void kernel_add(const double *a, const double *b, double *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

__global__ void kernel_sub(const double *a, const double *b, double *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] - b[idx];
}

__global__ void kernel_mul_scalar(const double *a, double s, double *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] * s;
}

__global__ void kernel_div_scalar(const double *a, double s, double *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] / s;
}

__global__ void kernel_hadamard(const double *a, const double *b, double *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] * b[idx];
}

__global__ void kernel_dot(const double *A, const double *B, double *C,
                           int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        double sum = 0.0;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void kernel_transpose(const double *src, double *dst, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int r = idx / cols;
        int c = idx % cols;
        dst[c * rows + r] = src[r * cols + c];
    }
}

__global__ void kernel_col_mean(const double *src, double *dst, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {
        double s = 0.0;
        for (int r = 0; r < rows; r++) {
            s += src[r * cols + col];
        }
        dst[col] = s / rows;
    }
}

__global__ void kernel_slice(const double *src, double *dst,
                             int start_row, int n_rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_rows * cols) {
        int r = idx / cols;
        int c = idx % cols;
        dst[r * cols + c] = src[(start_row + r) * cols + c];
    }
}

__global__ void kernel_reduce_sum(const double *data, double *partial, int n) {
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? data[idx] : 0.0;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) partial[blockIdx.x] = sdata[0];
}

__global__ void kernel_bias_add(double *z, const double *bias, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int c = idx % cols;
        z[idx] += bias[c];
    }
}

static inline int grid1d(int n, int block = 256) {
    return (n + block - 1) / block;
}

Matrix::Matrix() : rows(0), cols(0), d_data(nullptr), owns_data(true) {}

Matrix::Matrix(int r, int c, double val) : rows(r), cols(c), d_data(nullptr), owns_data(true) {
    if (r * c > 0) {
        CUDA_CHECK(cudaMalloc(&d_data, (size_t)r * c * sizeof(double)));
        kernel_fill<<<grid1d(r * c), 256>>>(d_data, r * c, val);
        CUDA_CHECK(cudaGetLastError());
    }
}

Matrix::~Matrix() {
    if (owns_data && d_data) {
        cudaFree(d_data);
    }
}

Matrix::Matrix(const Matrix &other) : rows(other.rows), cols(other.cols), d_data(nullptr), owns_data(true) {
    int n = rows * cols;
    if (n > 0) {
        CUDA_CHECK(cudaMalloc(&d_data, (size_t)n * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_data, other.d_data, (size_t)n * sizeof(double),
                              cudaMemcpyDeviceToDevice));
    }
}

Matrix::Matrix(Matrix &&other) noexcept : rows(other.rows), cols(other.cols), d_data(other.d_data), owns_data(other.owns_data) {
    other.d_data = nullptr;
    other.rows = other.cols = 0;
    other.owns_data = true;
}

Matrix &Matrix::operator=(const Matrix &other) {
    if (this == &other) return *this;
    if (owns_data && d_data) cudaFree(d_data);

    rows = other.rows;
    cols = other.cols;
    owns_data = true;
    int n = rows * cols;
    if (n > 0) {
        CUDA_CHECK(cudaMalloc(&d_data, (size_t)n * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_data, other.d_data, (size_t)n * sizeof(double),
                              cudaMemcpyDeviceToDevice));
    } else {
        d_data = nullptr;
    }
    return *this;
}

Matrix &Matrix::operator=(Matrix &&other) noexcept {
    if (this == &other) return *this;
    if (owns_data && d_data) cudaFree(d_data);

    rows = other.rows;
    cols = other.cols;
    d_data = other.d_data;
    owns_data = other.owns_data;
    other.d_data = nullptr;
    other.rows = other.cols = 0;
    other.owns_data = true;
    return *this;
}

double Matrix::at(int r, int c) const {
    double val;
    CUDA_CHECK(cudaMemcpy(&val, d_data + r * cols + c, sizeof(double),
                          cudaMemcpyDeviceToHost));
    return val;
}

void Matrix::set(int r, int c, double val) {
    CUDA_CHECK(cudaMemcpy(d_data + r * cols + c, &val, sizeof(double),
                          cudaMemcpyHostToDevice));
}

void Matrix::toHost(double *host_buf) const {
    CUDA_CHECK(cudaMemcpy(host_buf, d_data, (size_t)rows * cols * sizeof(double),
                          cudaMemcpyDeviceToHost));
}

void Matrix::fromHost(const double *host_buf) {
    CUDA_CHECK(cudaMemcpy(d_data, host_buf, (size_t)rows * cols * sizeof(double),
                          cudaMemcpyHostToDevice));
}

Matrix Matrix::operator+(const Matrix &other) const {
    Matrix res(rows, cols);
    int n = rows * cols;
    kernel_add<<<grid1d(n), 256>>>(d_data, other.d_data, res.d_data, n);
    CUDA_CHECK(cudaGetLastError());
    return res;
}

Matrix Matrix::operator-(const Matrix &other) const {
    Matrix res(rows, cols);
    int n = rows * cols;
    kernel_sub<<<grid1d(n), 256>>>(d_data, other.d_data, res.d_data, n);
    CUDA_CHECK(cudaGetLastError());
    return res;
}

Matrix Matrix::operator*(double s) const {
    Matrix res(rows, cols);
    int n = rows * cols;
    kernel_mul_scalar<<<grid1d(n), 256>>>(d_data, s, res.d_data, n);
    CUDA_CHECK(cudaGetLastError());
    return res;
}

Matrix Matrix::operator/(double s) const {
    Matrix res(rows, cols);
    int n = rows * cols;
    kernel_div_scalar<<<grid1d(n), 256>>>(d_data, s, res.d_data, n);
    CUDA_CHECK(cudaGetLastError());
    return res;
}

Matrix Matrix::hadamard(const Matrix &other) const {
    Matrix res(rows, cols);
    int n = rows * cols;
    kernel_hadamard<<<grid1d(n), 256>>>(d_data, other.d_data, res.d_data, n);
    CUDA_CHECK(cudaGetLastError());
    return res;
}

Matrix Matrix::dot(const Matrix &other) const {
    Matrix res(rows, other.cols, 0.0);
    dim3 block(16, 16);
    dim3 grid((other.cols + 15) / 16, (rows + 15) / 16);
    kernel_dot<<<grid, block>>>(d_data, other.d_data, res.d_data,
                                rows, cols, other.cols);
    CUDA_CHECK(cudaGetLastError());
    return res;
}

Matrix Matrix::transpose() const {
    Matrix res(cols, rows);
    int n = rows * cols;
    kernel_transpose<<<grid1d(n), 256>>>(d_data, res.d_data, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    return res;
}

Matrix Matrix::col_mean() const {
    Matrix res(1, cols, 0.0);
    kernel_col_mean<<<grid1d(cols), 256>>>(d_data, res.d_data, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    return res;
}

Matrix Matrix::slice(int start_row, int end_row) const {
    int n_rows = end_row - start_row;
    Matrix res(n_rows, cols);
    int n = n_rows * cols;
    kernel_slice<<<grid1d(n), 256>>>(d_data, res.d_data, start_row, n_rows, cols);
    CUDA_CHECK(cudaGetLastError());
    return res;
}

double Matrix::sum() const {
    int n = rows * cols;
    const int BLOCK = 256;
    int numBlocks = grid1d(n, BLOCK);

    double *d_partial;
    CUDA_CHECK(cudaMalloc(&d_partial, numBlocks * sizeof(double)));
    kernel_reduce_sum<<<numBlocks, BLOCK, BLOCK * sizeof(double)>>>(d_data, d_partial, n);
    CUDA_CHECK(cudaGetLastError());

    vector<double> h_partial(numBlocks);
    CUDA_CHECK(cudaMemcpy(h_partial.data(), d_partial, numBlocks * sizeof(double),
                          cudaMemcpyDeviceToHost));
    cudaFree(d_partial);

    double total = 0.0;
    for (double v : h_partial) total += v;
    return total;
}

double Matrix::mean() const {
    return sum() / (rows * cols);
}

Matrix gpu_zeros(int rows, int cols) {
    return Matrix(rows, cols, 0.0);
}

Matrix gpu_random(int rows, int cols, double scale, mt19937 &rng) {
    normal_distribution<double> dist(0.0, scale);
    int n = rows * cols;
    vector<double> h_data(n);
    for (double &val : h_data) {
        val = dist(rng);
    }

    Matrix res(rows, cols);
    res.fromHost(h_data.data());
    return res;
}
