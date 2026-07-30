#include "VisionTransformer.cuh"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cmath>

// TODO: Kernel CUDA que aplica Softmax sobre un vector 1D (los logits).
// Se reutiliza la logica del softmax del modulo de atencion.
// __global__ void softmax_logits_kernel(...)

VisionTransformer::VisionTransformer(int input_channels, int image_height, int image_width,
                                     int num_classes,
                                     int d_model, int num_heads, int num_layers, int mlp_dim,
                                     double learning_rate,
                                     int cnn_out_channels, int cnn_kernel, int cnn_stride, int cnn_padding,
                                     int pool_size, int pool_stride,
                                     int patch_h, int patch_w,
                                     int seed)
    : num_classes_(num_classes), d_model_(d_model), num_layers_(num_layers), learning_rate_(learning_rate) {
    // TODO:
    // 1. Crear el PatchEmbedding con los parametros de la CNN hibrida.
    //    patch_embedding_ = new PatchEmbedding(...);
    //
    // 2. Crear num_layers bloques de TransformerEncoderBlock:
    //    for (int i = 0; i < num_layers; i++)
    //        encoder_blocks_.push_back(new TransformerEncoderBlock(d_model, num_heads, mlp_dim, seed + i));
    //
    // 3. Crear el LayerNorm final:
    //    final_norm_ = new LayerNorm(d_model);
    //
    // 4. Crear la cabeza de clasificacion:
    //    W_head_ = gpu_random(d_model, num_classes, scale, rng);
    //    b_head_ = gpu_zeros(1, num_classes);
}

VisionTransformer::~VisionTransformer() {
    // TODO: Liberar patch_embedding_, cada encoder_block, y final_norm_.
}

Matrix VisionTransformer::forward(const Tensor3D& image) {
    // TODO:
    // 1. Convertir la imagen a una secuencia de embeddings:
    //    x = patch_embedding_->forward(image)  -> (seq_len, d_model)
    //
    // 2. Pasar por cada bloque del Transformer Encoder:
    //    for (int i = 0; i < num_layers_; i++)
    //        x = encoder_blocks_[i]->forward(x)
    //    encoder_output_cache_ = x
    //
    // 3. Extraer el token [CLS] (fila 0 de la secuencia):
    //    cls_output_cache_ = x.slice(0, 1)  -> (1, d_model)
    //
    // 4. Aplicar LayerNorm final:
    //    cls_normed_cache_ = final_norm_->forward(cls_output_cache_)
    //
    // 5. Pasar por la cabeza de clasificacion:
    //    logits_cache_ = cls_normed_cache_.dot(W_head_) + b_head_  -> (1, num_classes)
    //
    // 6. Retornar logits_cache_.
    return Matrix(); // placeholder
}

void VisionTransformer::backward(const Matrix& y_true) {
    // TODO:
    // 1. Calcular el gradiente de cross-entropy loss respecto a los logits:
    //    softmax_output = softmax(logits_cache_)
    //    grad_logits = softmax_output - y_true  -> (1, num_classes)
    //
    // 2. Backprop a traves de la cabeza de clasificacion:
    //    grad_cls_normed = grad_logits.dot(W_head_.transpose())  -> (1, d_model)
    //    grad_W_head = cls_normed_cache_.transpose().dot(grad_logits)
    //    Actualizar W_head_ y b_head_.
    //
    // 3. Backprop a traves del LayerNorm final:
    //    grad_cls = final_norm_->backward(grad_cls_normed, learning_rate_)
    //
    // 4. Reconstruir el gradiente para toda la secuencia:
    //    Crear grad_seq (seq_len, d_model) con ceros.
    //    Colocar grad_cls en la fila 0.
    //
    // 5. Backprop a traves de los bloques del Encoder (en orden inverso):
    //    for (int i = num_layers_ - 1; i >= 0; i--)
    //        grad_seq = encoder_blocks_[i]->backward(grad_seq, learning_rate_)
    //
    // 6. Backprop a traves del PatchEmbedding (incluida la CNN):
    //    patch_embedding_->backward(grad_seq, learning_rate_)
}

int VisionTransformer::predict(const Tensor3D& image) {
    // TODO:
    // 1. Ejecutar forward(image) para obtener logits.
    // 2. Encontrar el indice del valor maximo (argmax) en los logits.
    // 3. Retornar ese indice como la clase predicha.
    return 0; // placeholder
}

double VisionTransformer::accuracy(const vector<Tensor3D>& X, const Matrix& y) {
    // TODO:
    // 1. Iterar sobre todas las muestras de X.
    // 2. Para cada muestra, llamar predict() y comparar con la clase real.
    // 3. Contar las predicciones correctas.
    // 4. Retornar correctas / total.
    return 0.0; // placeholder
}

TrainHistory VisionTransformer::train(const vector<Tensor3D>& X_train, const Matrix& y_train,
                                      int epochs, int batch_size,
                                      const vector<Tensor3D>* X_val, const Matrix* y_val,
                                      bool verbose, int patience) {
    TrainHistory history;
    // TODO:
    // 1. Crear un vector de indices y un generador aleatorio para shuffle.
    //
    // 2. Para cada epoca:
    //    a. Shuffle de los indices.
    //    b. Iterar por mini-batches:
    //       - Para cada muestra del batch:
    //         * logits = forward(X_train[index])
    //         * loss += cross_entropy_loss(logits, y_true_sample)
    //         * backward(y_true_sample)
    //    c. Calcular loss promedio de la epoca.
    //    d. Si hay datos de validacion:
    //       * Calcular val_loss iterando X_val sin backward.
    //       * Early stopping: si val_loss no mejora por `patience` epocas, parar.
    //    e. Si verbose, imprimir progreso de la epoca.
    //
    // 3. Retornar history con train_losses y val_losses.
    return history; // placeholder
}

double VisionTransformer::cross_entropy_loss(const Matrix& logits, const Matrix& y_true) {
    // TODO:
    // 1. Aplicar softmax a los logits para obtener probabilidades.
    // 2. Calcular loss = -sum(y_true * log(softmax + epsilon)).
    // 3. Retornar el valor escalar de la perdida.
    return 0.0; // placeholder
}

void VisionTransformer::summary() const {
    // TODO:
    // 1. Imprimir la arquitectura del modelo capa por capa:
    //    - PatchEmbedding (parametros de CNN, num_patches, d_model)
    //    - Cada TransformerEncoderBlock (d_model, num_heads, mlp_dim)
    //    - LayerNorm final
    //    - Classification Head (d_model -> num_classes)
    // 2. Calcular y mostrar el numero total de parametros.
}
