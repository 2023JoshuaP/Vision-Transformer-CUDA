/**
 * Archivo: TransformerEncoder.cu
 * Descripcion: Bloque residual completo (Pre-Norm) de la arquitectura Transformer.
 * Rol en ViT: Empaqueta la Secuencia de Parches procesandola a traves de: 
 * (LayerNorm -> Atencion -> Add) -> (LayerNorm -> MLP -> Add). Estos bloques 
 * se apilan N veces en la arquitectura para construir la intuicion y contexto de la imagen.
 */

#include "TransformerEncoder.cuh"
#include <cmath>

static const int BLOCK_SIZE = 256;

__global__ void gelu_forward_kernel(const double* input, double* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        double x = input[idx];
        output[idx] = 0.5 * x * (1.0 + tanh(sqrt(2.0/M_PI) * (x + 0.044715 * x * x * x)));
    }
}

__global__ void gelu_backward_kernel(const double* grad_output, const double* input, double* grad_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        double x = input[idx];
        double tanh_term = tanh(sqrt(2.0/M_PI) * (x + 0.044715 * x * x * x));
        double sech_term = 1.0 - tanh_term * tanh_term;
        double derivative = 0.5 * (1.0 + tanh_term) + 0.5 * x * sech_term * sqrt(2.0/M_PI) * (1.0 + 3.0 * 0.044715 * x * x);
        grad_input[idx] = grad_output[idx] * derivative;
    }
}

TransformerEncoderBlock::TransformerEncoderBlock(int d_model, int num_heads, int mlp_dim, int seed)
    : d_model_(d_model), num_heads_(num_heads), mlp_dim_(mlp_dim),
      norm1_(d_model), norm2_(d_model), attention_(d_model, num_heads, seed) {
    
    mt19937 rng(seed);
    W1_ = gpu_random(d_model, mlp_dim, sqrt(2.0 / d_model), rng);
    b1_ = gpu_zeros(1, mlp_dim);

    W2_ = gpu_random(mlp_dim, d_model, sqrt(2.0 / mlp_dim), rng);
    b2_ = gpu_zeros(1, d_model);
}

Matrix TransformerEncoderBlock::forward(const Matrix& input) {
    input_cache_ = input;

    norm1_out_cache_ = norm1_.forward(input);
    Matrix attn_out = attention_.forward(norm1_out_cache_);

    after_attn_cache_ = input + attn_out;

    norm2_out_cache_ = norm2_.forward(after_attn_cache_);

    mlp_hidden_cache_ = norm2_out_cache_.dot(W1_);

    gelu_out_cache_ = Matrix(mlp_hidden_cache_.rows, mlp_hidden_cache_.cols);
    int total_gelu = mlp_hidden_cache_.rows * mlp_hidden_cache_.cols;
    int grid_gelu = (total_gelu + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gelu_forward_kernel<<<grid_gelu, BLOCK_SIZE>>>(mlp_hidden_cache_.d_data, gelu_out_cache_.d_data, total_gelu);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    Matrix mlp_out = gelu_out_cache_.dot(W2_);

    return after_attn_cache_ + mlp_out;
}

Matrix TransformerEncoderBlock::backward(const Matrix& grad_output, double learning_rate) {
    Matrix grad_mlp_out = grad_output;
    Matrix grad_residual2 = grad_output;

    Matrix grad_gelu = grad_mlp_out.dot(W2_.transpose());
    
    Matrix grad_W2 = gelu_out_cache_.transpose().dot(grad_mlp_out);
    W2_ = W2_ - grad_W2 * learning_rate;

    Matrix grad_hidden(grad_gelu.rows, grad_gelu.cols);
    int total_gelu = grad_gelu.rows * grad_gelu.cols;
    int grid_gelu = (total_gelu + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gelu_backward_kernel<<<grid_gelu, BLOCK_SIZE>>>(grad_gelu.d_data, mlp_hidden_cache_.d_data, grad_hidden.d_data, total_gelu);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    Matrix grad_norm2_out = grad_hidden.dot(W1_.transpose());
    
    Matrix grad_W1 = norm2_out_cache_.transpose().dot(grad_hidden);
    W1_ = W1_ - grad_W1 * learning_rate;

    Matrix grad_after_attn = norm2_.backward(grad_norm2_out, learning_rate);
    grad_after_attn = grad_after_attn + grad_residual2;

    Matrix grad_attn_out = grad_after_attn;
    Matrix grad_residual1 = grad_after_attn;

    Matrix grad_norm1_out = attention_.backward(grad_attn_out, learning_rate);

    Matrix grad_input = norm1_.backward(grad_norm1_out, learning_rate);
    grad_input = grad_input + grad_residual1;

    return grad_input;
}

