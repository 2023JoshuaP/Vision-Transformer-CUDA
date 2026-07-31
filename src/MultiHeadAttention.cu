/**
 * Archivo: MultiHeadAttention.cu
 * Descripcion: Implementa el mecanismo central 'Scaled Dot-Product Attention' fraccionado.
 * Rol en ViT: Calcula la relevancia que tiene cada parche de la imagen respecto a todos 
 * los demas parches en paralelo (Self-Attention). Al dividir el calculo en multiples 'cabezas',
 * permite al modelo enfocarse simultaneamente en distintas caracteristicas espaciales.
 */

#include "MultiHeadAttention.cuh"
#include <cmath>

__global__ void softmax_kernel(const double* input, double* output, int rows, int cols) {
    int row = blockIdx.x; // Un bloque por fila
    int tid = threadIdx.x;
    
    if (row < rows) {
        double max_val = -INFINITY;
        for (int col = tid; col < cols; col += blockDim.x) {
            if (input[row * cols + col] > max_val) {
                max_val = input[row * cols + col];
            }
        }
        
        __shared__ double shared_max[256];
        shared_max[tid] = max_val;
        __syncthreads();
        
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                if (shared_max[tid + s] > shared_max[tid]) {
                    shared_max[tid] = shared_max[tid + s];
                }
            }
            __syncthreads();
        }
        max_val = shared_max[0];
        
        double sum_exp = 0.0;
        for (int col = tid; col < cols; col += blockDim.x) {
            double val = exp(input[row * cols + col] - max_val);
            output[row * cols + col] = val;
            sum_exp += val;
        }
        
        __shared__ double shared_sum[256];
        shared_sum[tid] = sum_exp;
        __syncthreads();
        
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared_sum[tid] += shared_sum[tid + s];
            }
            __syncthreads();
        }
        sum_exp = shared_sum[0];
        
        for (int col = tid; col < cols; col += blockDim.x) {
            output[row * cols + col] /= sum_exp;
        }
    }
}

__global__ void softmax_backward_kernel(const double* S, const double* dL_dS, double* dL_dX, int rows, int cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    if (row < rows) {
        double sum_dS_S = 0.0;
        for (int col = tid; col < cols; col += blockDim.x) {
            sum_dS_S += dL_dS[row * cols + col] * S[row * cols + col];
        }
        
        __shared__ double shared_sum[256];
        shared_sum[tid] = sum_dS_S;
        __syncthreads();
        
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared_sum[tid] += shared_sum[tid + s];
            }
            __syncthreads();
        }
        double total_sum = shared_sum[0];
        
        for (int col = tid; col < cols; col += blockDim.x) {
            dL_dX[row * cols + col] = S[row * cols + col] * (dL_dS[row * cols + col] - total_sum);
        }
    }
}

__global__ void add_bias_kernel(const double* input, const double* bias, double* output, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int col = idx % cols;
        output[idx] = input[idx] + bias[col];
    }
}

MultiHeadAttention::MultiHeadAttention(int d_model, int num_heads, int seed)
    : d_model_(d_model), num_heads_(num_heads), d_k_(d_model / num_heads) {
    if (d_model_ % num_heads_ != 0) {
        throw std::invalid_argument("d_model must be divisible by num_heads");
    }

    std::mt19937 rng(seed);
    double xavier_scale = sqrt(2.0 / (d_model + d_model));

    Wq_ = gpu_random(d_model, d_model, xavier_scale, rng);
    Wk_ = gpu_random(d_model, d_model, xavier_scale, rng);
    Wv_ = gpu_random(d_model, d_model, xavier_scale, rng);
    Wo_ = gpu_random(d_model, d_model, xavier_scale, rng);

    bq_ = gpu_zeros(1, d_model);
    bk_ = gpu_zeros(1, d_model);
    bv_ = gpu_zeros(1, d_model);
    bo_ = gpu_zeros(1, d_model);
}

MultiHeadAttention::~MultiHeadAttention() {}

static inline int grid1d(int n, int block = 256) {
    return (n + block - 1) / block;
}

Matrix MultiHeadAttention::forward(const Matrix& input) {
    input_cache_ = input;
    
    int seq_len = input.rows;
    
    Matrix Q = input.dot(Wq_);
    add_bias_kernel<<<grid1d(seq_len * d_model_), 256>>>(Q.d_data, bq_.d_data, Q.d_data, seq_len, d_model_);
    CUDA_CHECK(cudaGetLastError());
    Q_cache_ = Q;
    
    Matrix K = input.dot(Wk_);
    add_bias_kernel<<<grid1d(seq_len * d_model_), 256>>>(K.d_data, bk_.d_data, K.d_data, seq_len, d_model_);
    CUDA_CHECK(cudaGetLastError());
    K_cache_ = K;
    
    Matrix V = input.dot(Wv_);
    add_bias_kernel<<<grid1d(seq_len * d_model_), 256>>>(V.d_data, bv_.d_data, V.d_data, seq_len, d_model_);
    CUDA_CHECK(cudaGetLastError());
    V_cache_ = V;
    
    attention_weights_.clear();
    context_cache_.clear();
    
    Matrix concat;
    
    for (int h = 0; h < num_heads_; h++) {
        int start_col = h * d_k_;
        int end_col = start_col + d_k_;
        
        Matrix Q_h = Q.slice_cols(start_col, end_col);
        Matrix K_h = K.slice_cols(start_col, end_col);
        Matrix V_h = V.slice_cols(start_col, end_col);
        
        Matrix K_h_T = K_h.transpose();
        Matrix scores_h = Q_h.dot(K_h_T);
        scores_h = scores_h / sqrt((double)d_k_);
        
        Matrix weights_h(scores_h.rows, scores_h.cols);
        softmax_kernel<<<scores_h.rows, 256>>>(scores_h.d_data, weights_h.d_data, scores_h.rows, scores_h.cols);
        CUDA_CHECK(cudaGetLastError());
        
        attention_weights_.push_back(weights_h);
        
        Matrix context_h = weights_h.dot(V_h);
        context_cache_.push_back(context_h);
        
        if (h == 0) {
            concat = context_h;
        } else {
            concat = concat.concat_cols(context_h);
        }
    }
    
    Matrix output = concat.dot(Wo_);
    add_bias_kernel<<<grid1d(seq_len * d_model_), 256>>>(output.d_data, bo_.d_data, output.d_data, seq_len, d_model_);
    CUDA_CHECK(cudaGetLastError());
    return output;
}

Matrix MultiHeadAttention::backward(const Matrix& grad_output, double learning_rate) {
    int seq_len = grad_output.rows;

    Matrix concat_cache;
    for (int h = 0; h < num_heads_; h++) {
        if (h == 0) concat_cache = context_cache_[h];
        else concat_cache = concat_cache.concat_cols(context_cache_[h]);
    }
    
    Matrix grad_concat = grad_output.dot(Wo_.transpose());
    Matrix grad_Wo = concat_cache.transpose().dot(grad_output);
    Matrix grad_bo = grad_output.col_mean() * (double)seq_len;
    
    Wo_ = Wo_ - (grad_Wo * learning_rate);
    bo_ = bo_ - (grad_bo * learning_rate);
    
    Matrix grad_Q;
    Matrix grad_K;
    Matrix grad_V;
    
    for (int h = 0; h < num_heads_; h++) {
        int start_col = h * d_k_;
        int end_col = start_col + d_k_;
        
        Matrix grad_context_h = grad_concat.slice_cols(start_col, end_col);
        
        Matrix V_h = V_cache_.slice_cols(start_col, end_col);
        Matrix weights_h = attention_weights_[h];
        
        Matrix grad_weights_h = grad_context_h.dot(V_h.transpose());
        Matrix grad_V_h = weights_h.transpose().dot(grad_context_h);
        
        Matrix grad_scores_h(seq_len, seq_len);
        softmax_backward_kernel<<<seq_len, 256>>>(weights_h.d_data, grad_weights_h.d_data, grad_scores_h.d_data, seq_len, seq_len);
        CUDA_CHECK(cudaGetLastError());
        
        grad_scores_h = grad_scores_h / sqrt((double)d_k_);
        
        Matrix Q_h = Q_cache_.slice_cols(start_col, end_col);
        Matrix K_h = K_cache_.slice_cols(start_col, end_col);
        
        Matrix grad_Q_h = grad_scores_h.dot(K_h);
        Matrix grad_K_h = grad_scores_h.transpose().dot(Q_h);
        
        if (h == 0) {
            grad_Q = grad_Q_h;
            grad_K = grad_K_h;
            grad_V = grad_V_h;
        } else {
            grad_Q = grad_Q.concat_cols(grad_Q_h);
            grad_K = grad_K.concat_cols(grad_K_h);
            grad_V = grad_V.concat_cols(grad_V_h);
        }
    }
    
    Matrix grad_input_q = grad_Q.dot(Wq_.transpose());
    Matrix grad_Wq = input_cache_.transpose().dot(grad_Q);
    Matrix grad_bq = grad_Q.col_mean() * (double)seq_len;
    
    Matrix grad_input_k = grad_K.dot(Wk_.transpose());
    Matrix grad_Wk = input_cache_.transpose().dot(grad_K);
    Matrix grad_bk = grad_K.col_mean() * (double)seq_len;
    
    Matrix grad_input_v = grad_V.dot(Wv_.transpose());
    Matrix grad_Wv = input_cache_.transpose().dot(grad_V);
    Matrix grad_bv = grad_V.col_mean() * (double)seq_len;
    
    Wq_ = Wq_ - (grad_Wq * learning_rate);
    bq_ = bq_ - (grad_bq * learning_rate);
    
    Wk_ = Wk_ - (grad_Wk * learning_rate);
    bk_ = bk_ - (grad_bk * learning_rate);
    
    Wv_ = Wv_ - (grad_Wv * learning_rate);
    bv_ = bv_ - (grad_bv * learning_rate);
    
    Matrix grad_input = grad_input_q + grad_input_k;
    grad_input = grad_input + grad_input_v;
    
    return grad_input;
}
