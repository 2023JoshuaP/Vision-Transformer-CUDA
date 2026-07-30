#pragma once

#include "Matrix.cuh"

/*
 * LayerNorm: Normalizacion por capa (Layer Normalization).
 * 
 * A diferencia de Batch Normalization (que normaliza a traves del batch),
 * Layer Normalization normaliza a traves de las features de una sola muestra.
 * 
 * Para un vector x de dimension D:
 *   mu    = (1/D) * sum(x_i)
 *   sigma = sqrt((1/D) * sum((x_i - mu)^2) + epsilon)
 *   y_i   = gamma * (x_i - mu) / sigma + beta
 * 
 * Donde gamma y beta son parametros aprendibles de escala y desplazamiento.
 */
class LayerNorm {
public:
    /*
     * Constructor: inicializa los parametros gamma (a 1.0) y beta (a 0.0)
     * en la GPU para un vector de dimension `dim`.
     * epsilon es una constante pequena para estabilidad numerica.
     */
    LayerNorm(int dim, double epsilon = 1e-5);
    ~LayerNorm();

    LayerNorm(const LayerNorm&) = delete;
    LayerNorm& operator=(const LayerNorm&) = delete;

    /*
     * Forward: Recibe una Matrix de forma (batch, dim).
     * Normaliza cada fila independientemente usando media y varianza.
     * Aplica la transformacion afin con gamma y beta.
     * Guarda en cache los valores necesarios para el backward.
     * Retorna la Matrix normalizada de forma (batch, dim).
     */
    Matrix forward(const Matrix& input);

    /*
     * Backward: Recibe el gradiente de la salida (batch, dim).
     * Calcula los gradientes respecto a gamma, beta y la entrada.
     * Actualiza gamma y beta con el learning_rate proporcionado.
     * Retorna el gradiente respecto a la entrada (batch, dim).
     */
    Matrix backward(const Matrix& grad_output, double learning_rate);

private:
    int dim_;
    double epsilon_;

    double* d_gamma_;   // Parametro de escala, dimension (dim_)
    double* d_beta_;    // Parametro de desplazamiento, dimension (dim_)

    // Cache para el backward pass
    Matrix input_cache_;
    Matrix normalized_cache_;
    double* d_mean_;    // Media por fila, dimension (batch)
    double* d_variance_; // Varianza por fila, dimension (batch)
    int cached_batch_size_;
};
