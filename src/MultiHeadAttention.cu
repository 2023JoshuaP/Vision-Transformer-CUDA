#include "MultiHeadAttention.cuh"
#include <cmath>

__global__ void softmax_kernel(const double* input, double* output, int rows, int cols) {
    int row = blockIdx.x; // Un bloque por fila
    int tid = threadIdx.x;
    
    if (row < rows) {
        // Encontrar el maximo para estabilidad numerica
        double max_val = -INFINITY;
        for (int col = tid; col < cols; col += blockDim.x) {
            if (input[row * cols + col] > max_val) {
                max_val = input[row * cols + col];
            }
        }
        
        // Reduccion para el maximo
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
        
        // Calcular exponenciales y suma local
        double sum_exp = 0.0;
        for (int col = tid; col < cols; col += blockDim.x) {
            double val = exp(input[row * cols + col] - max_val);
            output[row * cols + col] = val; // Guardar temporalmente
            sum_exp += val;
        }
        
        // Reduccion para la suma
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
        
        // Dividir por la suma total
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

MultiHeadAttention::~MultiHeadAttention() {
    // Los objetos Matrix se liberan automaticamente con su destructor.
}

Matrix MultiHeadAttention::forward(const Matrix& input) {
    // TODO:
    // 1. Guardar input en input_cache_.
    //
    // 2. Proyectar la entrada:
    //    Q = input.dot(Wq) + bq   -> (seq_len, d_model)
    //    K = input.dot(Wk) + bk   -> (seq_len, d_model)
    //    V = input.dot(Wv) + bv   -> (seq_len, d_model)
    //
    // 3. Dividir Q, K, V en num_heads cabezas:
    //    Para cada cabeza h (0..num_heads-1):
    //      Q_h = columnas [h*d_k .. (h+1)*d_k) de Q  -> (seq_len, d_k)
    //      K_h = columnas [h*d_k .. (h+1)*d_k) de K  -> (seq_len, d_k)
    //      V_h = columnas [h*d_k .. (h+1)*d_k) de V  -> (seq_len, d_k)
    //
    // 4. Para cada cabeza, calcular atencion escalada:
    //    scores_h = Q_h.dot(K_h.transpose()) / sqrt(d_k)  -> (seq_len, seq_len)
    //    weights_h = softmax(scores_h)                      -> (seq_len, seq_len)
    //    context_h = weights_h.dot(V_h)                     -> (seq_len, d_k)
    //    Guardar weights_h y context_h en cache.
    //
    // 5. Concatenar los context de todas las cabezas:
    //    concat = [context_0 | context_1 | ... | context_{h-1}] -> (seq_len, d_model)
    //
    // 6. Proyectar la concatenacion:
    //    output = concat.dot(Wo) + bo  -> (seq_len, d_model)
    //
    // 7. Retornar output.
    return input; // placeholder
}

Matrix MultiHeadAttention::backward(const Matrix& grad_output, double learning_rate) {
    // TODO:
    // 1. Backprop a traves de la proyeccion de salida Wo:
    //    grad_concat = grad_output.dot(Wo.transpose())
    //    grad_Wo = concat_cache.transpose().dot(grad_output)
    //    Actualizar Wo y bo.
    //
    // 2. Dividir grad_concat en num_heads gradientes.
    //
    // 3. Para cada cabeza h:
    //    a. Backprop a traves de context = weights * V:
    //       grad_weights_h = grad_context_h.dot(V_h.transpose())
    //       grad_V_h = weights_h.transpose().dot(grad_context_h)
    //    b. Backprop a traves de softmax:
    //       grad_scores_h = softmax_backward(grad_weights_h, weights_h)
    //    c. Backprop a traves del escalado:
    //       grad_scores_h = grad_scores_h / sqrt(d_k)
    //    d. Backprop a traves de Q * K^T:
    //       grad_Q_h = grad_scores_h.dot(K_h)
    //       grad_K_h = grad_scores_h.transpose().dot(Q_h)
    //
    // 4. Reconstruir grad_Q, grad_K, grad_V concatenando las cabezas.
    //
    // 5. Backprop a traves de las proyecciones Wq, Wk, Wv:
    //    grad_input_q = grad_Q.dot(Wq.transpose())
    //    grad_Wq = input_cache.transpose().dot(grad_Q)
    //    (similar para K y V)
    //    Actualizar Wq, Wk, Wv y sus sesgos.
    //
    // 6. grad_input = grad_input_q + grad_input_k + grad_input_v
    //    Retornar grad_input.
    return grad_output; // placeholder
}
