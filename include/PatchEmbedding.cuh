/**
 * Archivo: PatchEmbedding.cuh
 * Descripcion: Convierte la vision por computadora en un problema secuencial de Procesamiento de Lenguaje.
 * Rol en ViT: Toma el mapa de caracteristicas de la CNN, lo recorta en 'parches', 
 * aplana los parches en vectores, inyecta el token de clasificacion [CLS] y suma las 
 * posiciones espaciales (Positional Encoding) para que el Transformer sepa el orden original.
 */

#pragma once

#include "ActivationLayer.cuh"
#include "ConvolutionalLayer.cuh"
#include "Matrix.cuh"
#include "PoolingLayer.cuh"
#include "Tensor3D.cuh"

class PatchEmbedding {
    public:
        PatchEmbedding(int input_channels, int image_height, int image_width,
                        int d_model, int cnn_out_channels, int cnn_kernel,
                        int cnn_stride, int cnn_padding, int pool_size,
                        int pool_stride, int patch_h, int patch_w, int seed = 42);
        ~PatchEmbedding();

        PatchEmbedding(const PatchEmbedding &) = delete;
        PatchEmbedding &operator=(const PatchEmbedding &) = delete;

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
        Matrix forward(const Tensor3D &image);

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
        Tensor3D backward(const Matrix &grad_output, double learning_rate);

        int get_num_patches() const { return num_patches_; }
        int get_seq_length() const { return num_patches_ + 1; }

    private:
        int d_model_;
        int patch_h_, patch_w_;
        int num_patches_;
        int patch_dim_;

        ConvolutionalLayer conv_layer_;
        ReLUActivationLayer relu_layer_;
        PoolingLayer pool_layer_;

        int feature_h_, feature_w_, feature_c_;

        Matrix W_proj_;
        Matrix b_proj_;

        Matrix cls_token_;

        Matrix pos_embeddings_;

        Tensor3D cnn_output_cache_;
        Matrix patches_flat_cache_;
};
