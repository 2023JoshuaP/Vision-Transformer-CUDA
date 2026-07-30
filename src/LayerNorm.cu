#include "LayerNorm.cuh"
#include <cmath>
#include <utility>

static const int BLOCK_SIZE = 256;

// ============================================================================
// Kernel auxiliar: Llena un array con un valor constante
// ============================================================================
__global__ void layernorm_fill_kernel(double* data, int n, double val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = val;
}

// ============================================================================
// Kernel 1: Media de cada fila
// Un bloque por fila. Usa strided loop para soportar dim > BLOCK_SIZE.
// Reduccion con shared memory para sumar los parciales.
// ============================================================================
__global__ void layernorm_mean_kernel(const double* input, double* mean, int batch, int dim) {
    extern __shared__ double sdata[];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    double sum = 0.0;
    for (int i = tid; i < dim; i += blockDim.x) {
        sum += input[row * dim + i];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        mean[row] = sdata[0] / dim;
    }
}

// ============================================================================
// Kernel 2: Varianza de cada fila usando la media cacheada.
// Mismo patron que el kernel de media.
// ============================================================================
__global__ void layernorm_variance_kernel(const double* input, const double* mean,
                                          double* variance, int batch, int dim) {
    extern __shared__ double sdata[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    double m = mean[row];

    double sum = 0.0;
    for (int i = tid; i < dim; i += blockDim.x) {
        double diff = input[row * dim + i] - m;
        sum += diff * diff;
    }
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        variance[row] = sdata[0] / dim;
    }
}

// ============================================================================
// Kernel 3: Normalizacion elemento a elemento + aplicar gamma y beta.
// output[idx] = gamma[col] * (input[idx] - mean[row]) / sqrt(var[row] + eps) + beta[col]
// normalized_out[idx] = (input[idx] - mean[row]) / sqrt(var[row] + eps)  (valor crudo x̂)
// ============================================================================
__global__ void layernorm_normalize_kernel(const double* input, const double* mean,
                                           const double* variance, const double* gamma,
                                           const double* beta, double* output,
                                           double* normalized_out,
                                           int batch, int dim, double epsilon) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * dim;

    if (idx < total) {
        int row = idx / dim;
        int col = idx % dim;
        double inv_std = 1.0 / sqrt(variance[row] + epsilon);
        double normalized = (input[idx] - mean[row]) * inv_std;
        output[idx] = gamma[col] * normalized + beta[col];
        normalized_out[idx] = normalized;
    }
}

// ============================================================================
// Kernel 4: Backward pass.
// Un bloque por fila. Dos pasadas con shared memory:
//   Paso 1: Reducir para obtener sum(dx_hat) y sum(dx_hat * normalized) por fila.
//   Paso 2: Calcular grad_input usando la formula correcta de LayerNorm.
//           y acumular grad_gamma/grad_beta con atomicAdd.
//
// Formula del backward de LayerNorm (para cada fila):
//   dx_hat_i = dL/dy_i * gamma_i
//   dL/dx_i = (1/sigma) * (dx_hat_i - (1/D)*sum_j(dx_hat_j) - hat_x_i*(1/D)*sum_j(dx_hat_j*hat_x_j))
//   dL/dgamma_j = sum_i(dL/dy_i * hat_x_i)  (across batch)
//   dL/dbeta_j  = sum_i(dL/dy_i)             (across batch)
// ============================================================================
__global__ void layernorm_backward_kernel(const double* grad_output, const double* normalized,
                                          const double* gamma, const double* variance,
                                          double* grad_input, double* grad_gamma, double* grad_beta,
                                          int batch, int dim, double epsilon) {
    extern __shared__ double sdata[];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    // Paso 1: Calcular dx_hat y reducir sum(dx_hat) y sum(dx_hat * normalized)
    double local_dx_hat_sum = 0.0;
    double local_dx_hat_x_sum = 0.0;
    for (int i = tid; i < dim; i += blockDim.x) {
        int idx = row * dim + i;
        double dx_hat = grad_output[idx] * gamma[i];
        local_dx_hat_sum += dx_hat;
        local_dx_hat_x_sum += dx_hat * normalized[idx];
    }

    sdata[tid] = local_dx_hat_sum;
    sdata[blockDim.x + tid] = local_dx_hat_x_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
            sdata[blockDim.x + tid] += sdata[blockDim.x + tid + s];
        }
        __syncthreads();
    }

    __shared__ double s_dx_hat_sum;
    __shared__ double s_dx_hat_x_sum;
    if (tid == 0) {
        s_dx_hat_sum = sdata[0];
        s_dx_hat_x_sum = sdata[blockDim.x];
    }
    __syncthreads();

    // Paso 2: Calcular grad_input y acumular grad_gamma/grad_beta
    double inv_std = 1.0 / sqrt(variance[row] + epsilon);
    double inv_dim = 1.0 / dim;
    for (int i = tid; i < dim; i += blockDim.x) {
        int idx = row * dim + i;
        double dx_hat = grad_output[idx] * gamma[i];

        grad_input[idx] = inv_std * (dx_hat
                           - s_dx_hat_sum * inv_dim
                           - normalized[idx] * s_dx_hat_x_sum * inv_dim);

        atomicAdd(&grad_gamma[i], grad_output[idx] * normalized[idx]);
        atomicAdd(&grad_beta[i], grad_output[idx]);
    }
}

// ============================================================================
// Kernel auxiliar: Aplica SGD a gamma y beta.
// param_new[param] = param_new[param] - lr * grad[param]
// ============================================================================
__global__ void layernorm_update_params_kernel(double* param, const double* grad,
                                               int n, double learning_rate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        param[idx] -= learning_rate * grad[idx];
    }
}

// ============================================================================
// Constructor
// ============================================================================
LayerNorm::LayerNorm(int dim, double epsilon)
    : dim_(dim), epsilon_(epsilon), d_gamma_(nullptr), d_beta_(nullptr),
      d_mean_(nullptr), d_variance_(nullptr), cached_batch_size_(0) {
    CUDA_CHECK(cudaMalloc(&d_gamma_, dim * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_beta_, dim * sizeof(double)));

    // Inicializar gamma a 1.0 y beta a 0.0
    int grid = (dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
    layernorm_fill_kernel<<<grid, BLOCK_SIZE>>>(d_gamma_, dim, 1.0);
    CUDA_CHECK(cudaGetLastError());
    layernorm_fill_kernel<<<grid, BLOCK_SIZE>>>(d_beta_, dim, 0.0);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Destructor
// ============================================================================
LayerNorm::~LayerNorm() {
    if (d_gamma_) cudaFree(d_gamma_);
    if (d_beta_) cudaFree(d_beta_);
    if (d_mean_) cudaFree(d_mean_);
    if (d_variance_) cudaFree(d_variance_);
}

// ============================================================================
// Forward pass
// ============================================================================
Matrix LayerNorm::forward(const Matrix& input) {
    int batch = input.rows;
    int dim = input.cols;

    // 1. Alojar o redimensionar d_mean_ y d_variance_ si el batch cambio
    if (batch != cached_batch_size_) {
        if (d_mean_) cudaFree(d_mean_);
        if (d_variance_) cudaFree(d_variance_);
        CUDA_CHECK(cudaMalloc(&d_mean_, batch * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_variance_, batch * sizeof(double)));
        cached_batch_size_ = batch;
    } else {
        // Limpiar valores previos
        CUDA_CHECK(cudaMemset(d_mean_, 0, batch * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_variance_, 0, batch * sizeof(double)));
    }

    // 2. Calcular media de cada fila
    layernorm_mean_kernel<<<batch, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(
        input.d_data, d_mean_, batch, dim);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 3. Calcular varianza de cada fila
    layernorm_variance_kernel<<<batch, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(
        input.d_data, d_mean_, d_variance_, batch, dim);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 4. Normalizar y aplicar gamma/beta
    Matrix output(batch, dim);
    Matrix normalized_matrix(batch, dim);
    int total = batch * dim;
    int grid = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    layernorm_normalize_kernel<<<grid, BLOCK_SIZE>>>(
        input.d_data, d_mean_, d_variance_, d_gamma_, d_beta_, output.d_data,
        normalized_matrix.d_data, batch, dim, epsilon_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 5. Guardar en cache para el backward
    normalized_cache_ = std::move(normalized_matrix);

    return output;
}

// ============================================================================
// Backward pass
// ============================================================================
Matrix LayerNorm::backward(const Matrix& grad_output, double learning_rate) {
    int batch = grad_output.rows;
    int dim = grad_output.cols;

    // 1. Allocar buffers temporales para grad_gamma y grad_beta
    double* d_grad_gamma;
    double* d_grad_beta;
    CUDA_CHECK(cudaMalloc(&d_grad_gamma, dim * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_grad_beta, dim * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_grad_gamma, 0, dim * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_grad_beta, 0, dim * sizeof(double)));

    // 2. Buffer para grad_input
    Matrix grad_input(batch, dim);

    // 3. Lanzar kernel de backward (un bloque por fila, shared memory para dos reducciones)
    layernorm_backward_kernel<<<batch, BLOCK_SIZE, 2 * BLOCK_SIZE * sizeof(double)>>>(
        grad_output.d_data, normalized_cache_.d_data, d_gamma_, d_variance_,
        grad_input.d_data, d_grad_gamma, d_grad_beta,
        batch, dim, epsilon_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 4. Actualizar gamma y beta con SGD
    int param_grid = (dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
    layernorm_update_params_kernel<<<param_grid, BLOCK_SIZE>>>(
        d_gamma_, d_grad_gamma, dim, learning_rate);
    CUDA_CHECK(cudaGetLastError());
    layernorm_update_params_kernel<<<param_grid, BLOCK_SIZE>>>(
        d_beta_, d_grad_beta, dim, learning_rate);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 5. Liberar buffers temporales
    cudaFree(d_grad_gamma);
    cudaFree(d_grad_beta);

    return grad_input;
}
