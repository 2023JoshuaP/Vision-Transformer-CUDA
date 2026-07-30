#pragma once

#include "MultiHeadAttention.cuh"
#include "LayerNorm.cuh"
#include "MultiLayerPerceptron.cuh"

/*
 * TransformerEncoderBlock: Un bloque individual del codificador Transformer.
 *
 * Cada bloque sigue la estructura:
 *   1. Normalizacion (LayerNorm 1)
 *   2. Multi-Head Self-Attention
 *   3. Conexion Residual (sumar la entrada original)
 *   4. Normalizacion (LayerNorm 2)
 *   5. Feed-Forward Network (MLP con una capa oculta y activacion GELU)
 *   6. Conexion Residual (sumar la entrada del paso 4)
 *
 * Diagrama:
 *   x ──────────────────────┐
 *   │                       │
 *   └─> LayerNorm1 -> MHSA ─+─> z ──────────────────┐
 *                               │                     │
 *                               └─> LayerNorm2 -> MLP ─+─> salida
 *
 * Las conexiones residuales son fundamentales para que los gradientes
 * fluyan correctamente a traves de bloques profundos sin desvanecerse.
 */
class TransformerEncoderBlock {
public:
    /*
     * Constructor:
     *   - d_model: dimension del embedding (ancho de la secuencia).
     *   - num_heads: numero de cabezas de atencion.
     *   - mlp_dim: dimension de la capa oculta del MLP interno
     *     (tipicamente 4 * d_model).
     */
    TransformerEncoderBlock(int d_model, int num_heads, int mlp_dim, int seed = 42);

    /*
     * Forward: Recibe la secuencia X de forma (seq_len, d_model).
     *
     * Paso 1: norm1 = LayerNorm1(X)
     * Paso 2: attn_out = MultiHeadAttention(norm1)
     * Paso 3: z = X + attn_out              (conexion residual 1)
     * Paso 4: norm2 = LayerNorm2(z)
     * Paso 5: mlp_out = MLP(norm2)           (Linear -> GELU -> Linear)
     * Paso 6: output = z + mlp_out           (conexion residual 2)
     *
     * Guarda en cache X, z, norm1, norm2 para el backward.
     * Retorna la salida de forma (seq_len, d_model).
     */
    Matrix forward(const Matrix& input);

    /*
     * Backward: Recibe el gradiente de la salida (seq_len, d_model).
     *
     * Descompone los gradientes siguiendo el orden inverso:
     *   1. Gradiente a traves de la conexion residual 2.
     *   2. Backward del MLP y LayerNorm2.
     *   3. Gradiente a traves de la conexion residual 1.
     *   4. Backward de la MultiHeadAttention y LayerNorm1.
     *   5. Suma de gradientes residuales.
     *
     * Retorna el gradiente respecto a la entrada (seq_len, d_model).
     */
    Matrix backward(const Matrix& grad_output, double learning_rate);

private:
    int d_model_, num_heads_, mlp_dim_;

    LayerNorm norm1_;
    LayerNorm norm2_;
    MultiHeadAttention attention_;

    // MLP Feed-Forward interno: Linear(d_model, mlp_dim) -> GELU -> Linear(mlp_dim, d_model)
    // Se representara como dos matrices de pesos y sus sesgos
    Matrix W1_, b1_;  // Primera capa: (d_model, mlp_dim)
    Matrix W2_, b2_;  // Segunda capa: (mlp_dim, d_model)

    // Cache para backward
    Matrix input_cache_;       // Entrada original X
    Matrix after_attn_cache_;  // z = X + attn_out
    Matrix norm1_out_cache_;   // Salida de LayerNorm1
    Matrix norm2_out_cache_;   // Salida de LayerNorm2
    Matrix mlp_hidden_cache_;  // Salida de la primera capa del MLP (antes de GELU)
    Matrix gelu_out_cache_;    // Salida despues de GELU
};
