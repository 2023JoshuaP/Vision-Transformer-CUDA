/**
 * Archivo: TransformerEncoder.cuh
 * Descripcion: Bloque residual completo (Pre-Norm) de la arquitectura Transformer.
 * Rol en ViT: Empaqueta la Secuencia de Parches procesandola a traves de: 
 * (LayerNorm -> Atencion -> Add) -> (LayerNorm -> MLP -> Add). Estos bloques 
 * se apilan N veces en la arquitectura para construir la intuicion y contexto de la imagen.
 */

#pragma once

#include "MultiHeadAttention.cuh"
#include "LayerNorm.cuh"
#include "MultiLayerPerceptron.cuh"

class TransformerEncoderBlock {
public:
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

    Matrix W1_, b1_;
    Matrix W2_, b2_;

    Matrix input_cache_;
    Matrix after_attn_cache_;
    Matrix norm1_out_cache_;
    Matrix norm2_out_cache_;
    Matrix mlp_hidden_cache_;
    Matrix gelu_out_cache_;
};
