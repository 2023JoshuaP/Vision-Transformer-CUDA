#include "LayerNorm.cuh"

// TODO: Kernel CUDA que calcula la media de cada fila de la matriz.
// Cada bloque de hilos procesa una fila, reduciendo todos sus
// elementos para obtener un unico valor de media.
// __global__ void layernorm_mean_kernel(...)

// TODO: Kernel CUDA que calcula la varianza de cada fila usando
// la media previamente calculada.
// __global__ void layernorm_variance_kernel(...)

// TODO: Kernel CUDA que aplica la normalizacion elemento a elemento:
// output[i] = gamma * (input[i] - mean) / sqrt(variance + epsilon) + beta
// __global__ void layernorm_normalize_kernel(...)

// TODO: Kernel CUDA para el backward que calcula los gradientes
// respecto a gamma, beta y la entrada simultaneamente.
// __global__ void layernorm_backward_kernel(...)

LayerNorm::LayerNorm(int dim, double epsilon) : dim_(dim), epsilon_(epsilon), cached_batch_size_(0) {
    // TODO: Alojar d_gamma_ y d_beta_ en GPU con cudaMalloc.
    // Inicializar gamma a 1.0 y beta a 0.0.
    // Inicializar d_mean_ y d_variance_ a nullptr (se alojan en forward).
}

LayerNorm::~LayerNorm() {
    // TODO: Liberar d_gamma_, d_beta_, d_mean_, d_variance_ con cudaFree.
}

Matrix LayerNorm::forward(const Matrix& input) {
    // TODO:
    // 1. Alojar o redimensionar d_mean_ y d_variance_ si el batch cambio.
    // 2. Lanzar layernorm_mean_kernel para calcular la media de cada fila.
    // 3. Lanzar layernorm_variance_kernel para calcular la varianza de cada fila.
    // 4. Lanzar layernorm_normalize_kernel para normalizar y aplicar gamma/beta.
    // 5. Guardar input y output normalizados en cache.
    // 6. Retornar la Matrix normalizada.
    return input; // placeholder
}

Matrix LayerNorm::backward(const Matrix& grad_output, double learning_rate) {
    // TODO:
    // 1. Lanzar layernorm_backward_kernel para calcular:
    //    - Gradiente respecto a gamma (sum de grad_output * normalized).
    //    - Gradiente respecto a beta (sum de grad_output).
    //    - Gradiente respecto a la entrada (formula completa de chain rule
    //      con la derivada de la normalizacion).
    // 2. Actualizar d_gamma_ y d_beta_ con learning_rate.
    // 3. Retornar gradiente respecto a la entrada.
    return grad_output; // placeholder
}
