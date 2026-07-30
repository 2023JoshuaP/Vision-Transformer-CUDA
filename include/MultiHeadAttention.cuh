#pragma once

#include "Matrix.cuh"

/*
 * MultiHeadAttention: Mecanismo de Autoatencion Multi-Cabeza.
 *
 * Este es el componente central del Transformer. Permite que cada posicion
 * (parche) de la secuencia "atienda" a todas las demas posiciones para
 * capturar relaciones globales en la imagen.
 *
 * Funcionamiento interno:
 *   1. La entrada X (seq_len, d_model) se proyecta linealmente en tres
 *      matrices: Query (Q), Key (K) y Value (V).
 *   2. Q, K y V se dividen en `num_heads` cabezas independientes.
 *   3. Para cada cabeza se calcula:
 *      Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
 *   4. Los resultados de todas las cabezas se concatenan y se proyectan
 *      a traves de una ultima capa lineal de salida (Wo).
 *
 * Parametros aprendibles:
 *   - Wq, Wk, Wv: matrices de proyeccion (d_model, d_model) cada una
 *   - Wo: matriz de proyeccion de salida (d_model, d_model)
 *   - bq, bk, bv, bo: sesgos correspondientes
 */
class MultiHeadAttention {
public:
    /*
     * Constructor: Inicializa las matrices de pesos Wq, Wk, Wv, Wo
     * y sus respectivos sesgos en la GPU.
     * d_model: dimension del modelo (tamano del embedding).
     * num_heads: cantidad de cabezas de atencion.
     * d_model debe ser divisible entre num_heads.
     * d_k = d_model / num_heads (dimension por cabeza).
     */
    MultiHeadAttention(int d_model, int num_heads, int seed = 42);
    ~MultiHeadAttention();

    MultiHeadAttention(const MultiHeadAttention&) = delete;
    MultiHeadAttention& operator=(const MultiHeadAttention&) = delete;

    /*
     * Forward: Recibe la entrada X de forma (seq_len, d_model).
     * 
     * Paso 1: Proyectar X en Q, K, V usando Wq, Wk, Wv.
     *         Q = X * Wq + bq   (seq_len, d_model)
     *         K = X * Wk + bk   (seq_len, d_model)
     *         V = X * Wv + bv   (seq_len, d_model)
     * 
     * Paso 2: Reshape de Q, K, V para dividir en cabezas.
     *         Se reorganizan de (seq_len, d_model) a (num_heads, seq_len, d_k).
     *
     * Paso 3: Para cada cabeza, computar atencion escalada:
     *         scores = Q_h * K_h^T / sqrt(d_k)   -> (seq_len, seq_len)
     *         weights = softmax(scores)            -> (seq_len, seq_len)
     *         context = weights * V_h              -> (seq_len, d_k)
     * 
     * Paso 4: Concatenar las cabezas y proyectar con Wo.
     *         output = concat(context_1, ..., context_h) * Wo + bo
     *
     * Guarda en cache Q, K, V, scores, weights para el backward.
     * Retorna la salida de forma (seq_len, d_model).
     */
    Matrix forward(const Matrix& input);

    /*
     * Backward: Recibe el gradiente de la salida (seq_len, d_model).
     *
     * Propaga los gradientes a traves de:
     *   1. La proyeccion de salida Wo (y actualiza Wo y bo).
     *   2. La operacion de atencion (softmax y producto Q*K^T).
     *   3. Las proyecciones Q, K, V (y actualiza Wq, Wk, Wv y sus sesgos).
     *
     * Retorna el gradiente respecto a la entrada X (seq_len, d_model).
     */
    Matrix backward(const Matrix& grad_output, double learning_rate);

private:
    int d_model_, num_heads_, d_k_;

    // Pesos de proyeccion (almacenados en GPU)
    Matrix Wq_, Wk_, Wv_, Wo_;  // Cada una de forma (d_model, d_model)
    Matrix bq_, bk_, bv_, bo_;  // Sesgos de forma (1, d_model)

    // Cache del forward para el backward
    Matrix input_cache_;
    Matrix Q_cache_, K_cache_, V_cache_;
    // attention_weights_: una por cabeza, de forma (seq_len, seq_len)
    // context_cache_: resultado de weights * V por cabeza
    vector<Matrix> attention_weights_;
    vector<Matrix> context_cache_;
};
