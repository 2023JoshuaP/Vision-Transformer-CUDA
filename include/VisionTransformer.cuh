/**
 * Archivo: VisionTransformer.cuh
 * Descripcion: Clase constructora e integradora (Orquestador Maestro) del modelo end-to-end.
 * Rol en ViT: Conecta la CNN, el PatchEmbedding, los bloques Transformer y la cabeza 
 * final de clasificacion. Controla todo el ciclo de vida del aprendizaje: calcula el Loss, 
 * dirige el Backward propagation por todos los bloques y actualiza los pesos de la VRAM.
 */

#pragma once

#include "PatchEmbedding.cuh"
#include "TransformerEncoder.cuh"
#include "LayerNorm.cuh"
#include "Matrix.cuh"
#include "Tensor3D.cuh"
#include <vector>

class VisionTransformer {
public:
    VisionTransformer(int input_channels, int image_height, int image_width,
                      int num_classes,
                      int d_model, int num_heads, int num_layers, int mlp_dim,
                      double learning_rate,
                      int cnn_out_channels, int cnn_kernel, int cnn_stride, int cnn_padding,
                      int pool_size, int pool_stride,
                      int patch_h, int patch_w,
                      int seed = 42);
    ~VisionTransformer();

    VisionTransformer(const VisionTransformer&) = delete;
    VisionTransformer& operator=(const VisionTransformer&) = delete;

    Matrix forward(const Tensor3D& image);
    void backward(const Matrix& y_true);

    int predict(const Tensor3D& image);
    double accuracy(const vector<Tensor3D>& X, const Matrix& y);

    TrainHistory train(const vector<Tensor3D>& X_train, const Matrix& y_train,
                       int epochs, int batch_size,
                       const vector<Tensor3D>* X_val = nullptr, const Matrix* y_val = nullptr,
                       bool verbose = true, int patience = 20);

    void summary() const;
    static double cross_entropy_loss(const Matrix& logits, const Matrix& y_true);

private:
    int num_classes_, d_model_, num_layers_;
    double learning_rate_;

    PatchEmbedding* patch_embedding_;
    vector<TransformerEncoderBlock*> encoder_blocks_;
    LayerNorm* final_norm_;
    Matrix W_head_;
    Matrix b_head_;

    Matrix encoder_output_cache_;
    Matrix cls_output_cache_;
    Matrix cls_normed_cache_;
    Matrix logits_cache_;
};
