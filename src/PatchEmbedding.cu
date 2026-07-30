#include "PatchEmbedding.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// Kernel 1: extract_patches_kernel
// Cada hilo procesa un parche: copia C*ph*pw valores del feature map
// (C, H, W) a una fila de la matriz de salida (num_patches, patch_dim).
// ---------------------------------------------------------------------------
__global__ void extract_patches_kernel(const double* __restrict__ feature_map,
                                       double* __restrict__ patches,
                                       int C, int H, int W,
                                       int ph, int pw,
                                       int num_patches_h, int num_patches_w) {
    int patch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_patches = num_patches_h * num_patches_w;
    if (patch_idx >= total_patches) return;

    int pi = patch_idx / num_patches_w;  // fila de parche en la grilla
    int pj = patch_idx % num_patches_w;  // columna de parche en la grilla

    int origin_h = pi * ph;
    int origin_w = pj * pw;
    int patch_dim = C * ph * pw;

    double* dst = patches + patch_idx * patch_dim;

    for (int c = 0; c < C; ++c) {
        for (int i = 0; i < ph; ++i) {
            for (int j = 0; j < pw; ++j) {
                int flat_in_patch = c * ph * pw + i * pw + j;
                int src_idx = c * H * W + (origin_h + i) * W + (origin_w + j);
                dst[flat_in_patch] = feature_map[src_idx];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: reconstruct_patches_kernel
// Operacion inversa de extract_patches_kernel.
// Acumula (atomicAdd) los gradientes de (num_patches, patch_dim)
// de vuelta al tensor (C, H, W).
// ---------------------------------------------------------------------------
__global__ void reconstruct_patches_kernel(const double* __restrict__ grad_patches,
                                            double* __restrict__ grad_feature_map,
                                            int C, int H, int W,
                                            int ph, int pw,
                                            int num_patches_h, int num_patches_w) {
    int patch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_patches = num_patches_h * num_patches_w;
    if (patch_idx >= total_patches) return;

    int pi = patch_idx / num_patches_w;
    int pj = patch_idx % num_patches_w;

    int origin_h = pi * ph;
    int origin_w = pj * pw;
    int patch_dim = C * ph * pw;

    const double* src = grad_patches + patch_idx * patch_dim;

    for (int c = 0; c < C; ++c) {
        for (int i = 0; i < ph; ++i) {
            for (int j = 0; j < pw; ++j) {
                int flat_in_patch = c * ph * pw + i * pw + j;
                int dst_idx = c * H * W + (origin_h + i) * W + (origin_w + j);
                atomicAdd(&grad_feature_map[dst_idx], src[flat_in_patch]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: add_row_bias_kernel
// Suma el vector bias (1, d_model) a cada fila de mat (N, d_model).
// ---------------------------------------------------------------------------
__global__ void add_row_bias_kernel(double* __restrict__ mat,
                                    const double* __restrict__ bias,
                                    int N, int d_model) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * d_model) return;
    mat[idx] += bias[idx % d_model];
}

// ---------------------------------------------------------------------------
// Kernel 4: param_update_kernel   SGD: param -= lr * grad
// ---------------------------------------------------------------------------
__global__ void param_update_kernel(double* __restrict__ param,
                                    const double* __restrict__ grad,
                                    double lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    param[idx] -= lr * grad[idx];
}

// ===========================================================================
// Constructor / Destructor  (stubs — se implementan en el siguiente commit)
// ===========================================================================
PatchEmbedding::PatchEmbedding(int input_channels, int image_height, int image_width,
                               int d_model,
                               int cnn_out_channels, int cnn_kernel, int cnn_stride, int cnn_padding,
                               int pool_size, int pool_stride,
                               int patch_h, int patch_w,
                               int seed)
    : d_model_(d_model), patch_h_(patch_h), patch_w_(patch_w),
      conv_layer_(input_channels, cnn_out_channels, cnn_kernel, cnn_stride, cnn_padding, seed),
      relu_layer_(),
      pool_layer_(pool_size, pool_stride, PoolingType::Max),
      cnn_output_cache_(), patches_flat_cache_(),
      W_proj_(), b_proj_(), cls_token_(), pos_embeddings_()
{
    // 1. Dimensiones del feature map tras CNN + Pooling
    int conv_h = ConvolutionalLayer::size_out(image_height, cnn_kernel, cnn_stride, cnn_padding);
    int conv_w = ConvolutionalLayer::size_out(image_width,  cnn_kernel, cnn_stride, cnn_padding);
    feature_h_ = PoolingLayer::output_size(conv_h, pool_size, pool_stride);
    feature_w_ = PoolingLayer::output_size(conv_w, pool_size, pool_stride);
    feature_c_ = cnn_out_channels;

    // 2. Numero de parches y dimension aplanada de cada parche
    num_patches_ = (feature_h_ / patch_h_) * (feature_w_ / patch_w_);
    patch_dim_   = feature_c_ * patch_h_ * patch_w_;

    // 3. W_proj (patch_dim, d_model) Xavier; b_proj (1, d_model) = 0
    mt19937 rng(seed);
    double xavier_scale = sqrt(2.0 / (patch_dim_ + d_model_));
    W_proj_ = gpu_random(patch_dim_, d_model_, xavier_scale, rng);
    b_proj_ = gpu_zeros(1, d_model_);

    // 4. cls_token (1, d_model) y pos_embeddings (num_patches+1, d_model)
    //    con valores pequenos para romper simetria
    cls_token_      = gpu_random(1, d_model_, 0.02, rng);
    pos_embeddings_ = gpu_random(num_patches_ + 1, d_model_, 0.02, rng);
}

PatchEmbedding::~PatchEmbedding() {}

Matrix PatchEmbedding::forward(const Tensor3D& image) {
    // -----------------------------------------------------------------------
    // Paso 1: CNN -> ReLU -> Pooling
    // -----------------------------------------------------------------------
    Tensor3D conv_out = conv_layer_.forward(image);
    Tensor3D relu_out = relu_layer_.forward(conv_out);
    cnn_output_cache_ = pool_layer_.forward(relu_out);

    // -----------------------------------------------------------------------
    // Paso 2: Extraer y aplanar parches -> (num_patches, patch_dim)
    // -----------------------------------------------------------------------
    int num_patches_h = feature_h_ / patch_h_;
    int num_patches_w = feature_w_ / patch_w_;

    patches_flat_cache_ = Matrix(num_patches_, patch_dim_, 0.0);

    int threads = 256;
    int blocks  = (num_patches_ + threads - 1) / threads;
    extract_patches_kernel<<<blocks, threads>>>(
        cnn_output_cache_.d_data,
        patches_flat_cache_.d_data,
        feature_c_, feature_h_, feature_w_,
        patch_h_, patch_w_,
        num_patches_h, num_patches_w
    );
    cudaDeviceSynchronize();

    // -----------------------------------------------------------------------
    // Paso 3: Proyeccion lineal: (num_patches, patch_dim) @ W_proj + b_proj
    //         -> (num_patches, d_model)
    // -----------------------------------------------------------------------
    Matrix projected = patches_flat_cache_.dot(W_proj_);
    int total_proj   = num_patches_ * d_model_;
    int bblocks      = (total_proj + threads - 1) / threads;
    add_row_bias_kernel<<<bblocks, threads>>>(
        projected.d_data, b_proj_.d_data,
        num_patches_, d_model_
    );
    cudaDeviceSynchronize();

    // -----------------------------------------------------------------------
    // Paso 4: Prepend del token [CLS] -> (num_patches+1, d_model)
    //         fila 0 = cls_token_; filas 1..N = projected
    // -----------------------------------------------------------------------
    int seq_len = num_patches_ + 1;
    Matrix output(seq_len, d_model_, 0.0);

    cudaMemcpy(output.d_data,
               cls_token_.d_data,
               d_model_ * sizeof(double),
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(output.d_data + d_model_,
               projected.d_data,
               (size_t)num_patches_ * d_model_ * sizeof(double),
               cudaMemcpyDeviceToDevice);

    // -----------------------------------------------------------------------
    // Paso 5: Sumar positional embeddings element-wise
    // -----------------------------------------------------------------------
    output = output + pos_embeddings_;

    return output;
}

Tensor3D PatchEmbedding::backward(const Matrix& grad_output, double learning_rate) {
    return Tensor3D(); // placeholder
}
