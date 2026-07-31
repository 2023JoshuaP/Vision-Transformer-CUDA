#pragma once

#include "Matrix.cuh"

class MultiHeadAttention {
    public:
        MultiHeadAttention(int d_model, int num_heads, int seed = 42);
        ~MultiHeadAttention();

        MultiHeadAttention(const MultiHeadAttention &) = delete;
        MultiHeadAttention &operator=(const MultiHeadAttention &) = delete;

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

        Matrix forward(const Matrix &input);

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

        Matrix backward(const Matrix &grad_output, double learning_rate);

    private:
        int d_model_, num_heads_, d_k_;

        Matrix Wq_, Wk_, Wv_, Wo_;
        Matrix bq_, bk_, bv_, bo_;

        Matrix input_cache_;
        Matrix Q_cache_, K_cache_, V_cache_;
        vector<Matrix> attention_weights_;
        vector<Matrix> context_cache_;
};
