#include "TransformerEncoder.cuh"
#include <cmath>

// TODO: Kernel CUDA que aplica la funcion de activacion GELU:
// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// Es la activacion estandar del Transformer, mas suave que ReLU.
// __global__ void gelu_forward_kernel(...)

// TODO: Kernel CUDA para el backward de GELU.
// Calcula la derivada: dGELU/dx y la multiplica por el gradiente entrante.
// __global__ void gelu_backward_kernel(...)

TransformerEncoderBlock::TransformerEncoderBlock(int d_model, int num_heads, int mlp_dim, int seed)
    : d_model_(d_model), num_heads_(num_heads), mlp_dim_(mlp_dim),
      norm1_(d_model), norm2_(d_model), attention_(d_model, num_heads, seed) {
    // TODO:
    // 1. Inicializar W1_ (d_model, mlp_dim) con Xavier init.
    //    Inicializar b1_ (1, mlp_dim) a cero.
    // 2. Inicializar W2_ (mlp_dim, d_model) con Xavier init.
    //    Inicializar b2_ (1, d_model) a cero.
}

Matrix TransformerEncoderBlock::forward(const Matrix& input) {
    // TODO:
    // 1. input_cache_ = input
    //
    // 2. Pre-Norm + Atencion:
    //    norm1_out_cache_ = norm1_.forward(input)
    //    attn_out = attention_.forward(norm1_out_cache_)
    //
    // 3. Conexion residual 1:
    //    after_attn_cache_ = input + attn_out
    //
    // 4. Pre-Norm + MLP Feed-Forward:
    //    norm2_out_cache_ = norm2_.forward(after_attn_cache_)
    //    mlp_hidden_cache_ = norm2_out_cache_.dot(W1_) + b1_
    //    gelu_out_cache_ = GELU(mlp_hidden_cache_)  (lanzar gelu_forward_kernel)
    //    mlp_out = gelu_out_cache_.dot(W2_) + b2_
    //
    // 5. Conexion residual 2:
    //    output = after_attn_cache_ + mlp_out
    //
    // 6. Retornar output (seq_len, d_model).
    return input; // placeholder
}

Matrix TransformerEncoderBlock::backward(const Matrix& grad_output, double learning_rate) {
    // TODO:
    // 1. Conexion residual 2:
    //    grad_mlp_out = grad_output  (copia directa por la suma)
    //    grad_residual2 = grad_output (se acumula despues)
    //
    // 2. Backprop a traves del MLP:
    //    grad_gelu = grad_mlp_out.dot(W2_.transpose())
    //    grad_W2 = gelu_out_cache_.transpose().dot(grad_mlp_out)
    //    Actualizar W2_ y b2_.
    //
    //    grad_hidden = gelu_backward(grad_gelu, mlp_hidden_cache_)
    //    grad_norm2_out = grad_hidden.dot(W1_.transpose())
    //    grad_W1 = norm2_out_cache_.transpose().dot(grad_hidden)
    //    Actualizar W1_ y b1_.
    //
    // 3. Backprop a traves de LayerNorm2:
    //    grad_after_attn = norm2_.backward(grad_norm2_out, learning_rate)
    //    grad_after_attn = grad_after_attn + grad_residual2
    //
    // 4. Conexion residual 1:
    //    grad_attn_out = grad_after_attn (copia directa)
    //    grad_residual1 = grad_after_attn (se acumula despues)
    //
    // 5. Backprop a traves de la Atencion:
    //    grad_norm1_out = attention_.backward(grad_attn_out, learning_rate)
    //
    // 6. Backprop a traves de LayerNorm1:
    //    grad_input = norm1_.backward(grad_norm1_out, learning_rate)
    //    grad_input = grad_input + grad_residual1
    //
    // 7. Retornar grad_input (seq_len, d_model).
    return grad_output; // placeholder
}
