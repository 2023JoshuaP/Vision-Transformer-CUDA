#pragma once

#include "Matrix.cuh"
#include "Tensor3D.cuh"
#include "ConvolutionalLayer.cuh"
#include "PoolingLayer.cuh"
#include "ActivationLayer.cuh"

/*
 * PatchEmbedding: Convierte la imagen (o el feature map de la CNN) en una
 * secuencia de embeddings que el Transformer puede procesar.
 *
 * En el enfoque hibrido:
 *   1. La imagen original pasa por la CNN (ConvolutionalLayer + ReLU + Pooling)
 *      para obtener un feature map de forma (C_out, H_out, W_out).
 *   2. Ese feature map se divide en parches de tamano (patch_h x patch_w).
 *      Cada parche se aplana para obtener un vector de dimension
 *      C_out * patch_h * patch_w.
 *   3. Cada parche aplanado se proyecta linealmente a un embedding de
 *      dimension d_model mediante una multiplicacion de matrices.
 *   4. Se agrega un token especial [CLS] al inicio de la secuencia
 *      (vector aprendible de dimension d_model) cuya salida final
 *      representa la clasificacion global de toda la imagen.
 *   5. Se suman embeddings posicionales aprendibles a cada parche
 *      (incluido el [CLS]) para inyectar informacion de la posicion
 *      espacial original de cada parche en la imagen.
 *
 * Salida final: una Matrix de forma (num_patches + 1, d_model).
 */
class PatchEmbedding {
public:
    /*
     * Constructor:
     *   - input_channels: canales de la imagen original (ej. 3 para RGB).
     *   - image_height, image_width: dimensiones de la imagen de entrada.
     *   - d_model: dimension del embedding de salida para el Transformer.
     *   - cnn_out_channels: canales de salida de la CNN.
     *   - cnn_kernel, cnn_stride, cnn_padding: parametros de la ConvolutionalLayer.
     *   - pool_size, pool_stride: parametros de la PoolingLayer.
     *   - patch_h, patch_w: tamano de los parches sobre el feature map.
     *
     * Internamente construye la CNN, calcula las dimensiones del feature map
     * resultante, y crea la matriz de proyeccion lineal y los embeddings
     * posicionales en la GPU.
     */
    PatchEmbedding(int input_channels, int image_height, int image_width,
                   int d_model,
                   int cnn_out_channels, int cnn_kernel, int cnn_stride, int cnn_padding,
                   int pool_size, int pool_stride,
                   int patch_h, int patch_w,
                   int seed = 42);
    ~PatchEmbedding();

    PatchEmbedding(const PatchEmbedding&) = delete;
    PatchEmbedding& operator=(const PatchEmbedding&) = delete;

    /*
     * Forward: Recibe un Tensor3D de la imagen (channels, height, width).
     *
     * Paso 1: Pasar la imagen por la CNN (conv -> relu -> pool) para
     *         obtener el feature map.
     * Paso 2: Dividir el feature map en parches y aplanar cada uno
     *         en un vector de dimension (C_out * patch_h * patch_w).
     * Paso 3: Proyectar linealmente cada parche a dimension d_model
     *         mediante W_proj (patch_dim, d_model) + bias.
     * Paso 4: Prepend del token [CLS] al inicio de la secuencia.
     * Paso 5: Sumar los positional embeddings a toda la secuencia.
     *
     * Retorna Matrix de forma (num_patches + 1, d_model).
     */
    Matrix forward(const Tensor3D& image);

    /*
     * Backward: Recibe el gradiente (num_patches + 1, d_model).
     *
     * Propaga los gradientes a traves de:
     *   1. Los positional embeddings (los actualiza).
     *   2. El token [CLS] (lo actualiza).
     *   3. La proyeccion lineal W_proj (la actualiza).
     *   4. La reconstruccion de parches al feature map.
     *   5. La CNN en reversa (pool -> relu -> conv backward).
     *
     * Retorna el gradiente respecto a la imagen de entrada.
     */
    Tensor3D backward(const Matrix& grad_output, double learning_rate);

    int get_num_patches() const { return num_patches_; }
    int get_seq_length() const { return num_patches_ + 1; } // +1 por [CLS]

private:
    int d_model_;
    int patch_h_, patch_w_;
    int num_patches_;
    int patch_dim_;   // C_out * patch_h * patch_w

    // CNN como extractor de caracteristicas
    ConvolutionalLayer conv_layer_;
    ReLUActivationLayer relu_layer_;
    PoolingLayer pool_layer_;

    int feature_h_, feature_w_, feature_c_;  // Dimensiones del feature map

    // Proyeccion lineal: (patch_dim, d_model)
    Matrix W_proj_;
    Matrix b_proj_;    // (1, d_model)

    // Token [CLS]: vector aprendible (1, d_model)
    Matrix cls_token_;

    // Positional Embeddings: (num_patches + 1, d_model)
    Matrix pos_embeddings_;

    // Cache para backward
    Tensor3D cnn_output_cache_;
    Matrix patches_flat_cache_;
};
