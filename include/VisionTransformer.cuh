#pragma once

#include "PatchEmbedding.cuh"
#include "TransformerEncoder.cuh"
#include "LayerNorm.cuh"
#include "Matrix.cuh"
#include "Tensor3D.cuh"
#include <vector>

/*
 * VisionTransformer: El modelo completo del Vision Transformer Hibrido.
 *
 * Arquitectura de extremo a extremo:
 *
 *   Imagen (C, H, W)
 *       |
 *       v
 *   [PatchEmbedding]  --  CNN + division en parches + proyeccion + [CLS] + pos_emb
 *       |
 *       v
 *   (num_patches+1, d_model)
 *       |
 *       v
 *   [TransformerEncoderBlock x N]  --  N bloques apilados de (Atencion + MLP)
 *       |
 *       v
 *   (num_patches+1, d_model)
 *       |
 *       v
 *   [Extraer token CLS]  --  Se toma solo la fila 0 de la secuencia
 *       |
 *       v
 *   (1, d_model)
 *       |
 *       v
 *   [LayerNorm final]
 *       |
 *       v
 *   [Classification Head]  --  Una capa lineal (d_model, num_classes)
 *       |
 *       v
 *   (1, num_classes)  --  Logits de cada clase
 */
class VisionTransformer {
public:
    /*
     * Constructor:
     *   - input_channels: canales de la imagen (ej. 3 para RGB).
     *   - image_height, image_width: dimensiones de la imagen.
     *   - num_classes: numero de categorias de clasificacion.
     *   - d_model: dimension del embedding del Transformer.
     *   - num_heads: cabezas de atencion por bloque.
     *   - num_layers: cantidad de TransformerEncoderBlocks apilados.
     *   - mlp_dim: dimension oculta del MLP en cada bloque.
     *   - learning_rate: tasa de aprendizaje para todas las capas.
     *   - Parametros de la CNN hibrida (canales, kernel, stride, etc.)
     *   - patch_h, patch_w: tamano de los parches sobre el feature map.
     *
     * Construye internamente: PatchEmbedding, N bloques de TransformerEncoder,
     * un LayerNorm final y la cabeza de clasificacion.
     */
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

    /*
     * Forward: Recibe un Tensor3D de la imagen (channels, height, width).
     *
     * Paso 1: patch_emb = PatchEmbedding(imagen)     -> (seq_len, d_model)
     * Paso 2: x = TransformerEncoder_1(patch_emb)
     *         x = TransformerEncoder_2(x)
     *         ...
     *         x = TransformerEncoder_N(x)             -> (seq_len, d_model)
     * Paso 3: cls_out = x[0]  (extraer token [CLS])  -> (1, d_model)
     * Paso 4: cls_norm = LayerNorm(cls_out)           -> (1, d_model)
     * Paso 5: logits = cls_norm * W_head + b_head     -> (1, num_classes)
     *
     * Retorna los logits de forma (1, num_classes).
     */
    Matrix forward(const Tensor3D& image);

    /*
     * Backward: Recibe los labels verdaderos y_true de forma (1, num_classes)
     * en formato one-hot.
     *
     * Paso 1: Calcular gradiente de la funcion de perdida (cross-entropy)
     *         respecto a los logits.
     * Paso 2: Backprop a traves de la cabeza de clasificacion (W_head).
     * Paso 3: Backprop a traves del LayerNorm final.
     * Paso 4: Reconstruir el gradiente completo (seq_len, d_model)
     *         colocando el gradiente del CLS en la fila 0 y ceros en el resto.
     * Paso 5: Backprop a traves de cada TransformerEncoderBlock en reversa.
     * Paso 6: Backprop a traves del PatchEmbedding (incluyendo la CNN).
     */
    void backward(const Matrix& y_true);

    /*
     * Predict: Recibe una imagen y retorna el indice de la clase predicha
     * (argmax de los logits).
     */
    int predict(const Tensor3D& image);

    /*
     * Accuracy: Calcula la precision sobre un conjunto de datos completo.
     * Itera sobre todas las imagenes, predice la clase y compara con
     * los labels verdaderos.
     */
    double accuracy(const vector<Tensor3D>& X, const Matrix& y);

    /*
     * Train: Entrena el modelo sobre un conjunto de datos.
     * Implementa el ciclo de entrenamiento con:
     *   - Shuffle de datos por epoca.
     *   - Forward y backward por cada muestra del batch.
     *   - Calculo de loss y accuracy por epoca.
     *   - Validacion opcional con early stopping.
     *
     * Retorna un historial con las perdidas de entrenamiento y validacion.
     */
    TrainHistory train(const vector<Tensor3D>& X_train, const Matrix& y_train,
                       int epochs, int batch_size,
                       const vector<Tensor3D>* X_val = nullptr, const Matrix* y_val = nullptr,
                       bool verbose = true, int patience = 20);

    /*
     * Summary: Imprime en consola un resumen de la arquitectura del modelo,
     * incluyendo cada capa, sus dimensiones y la cantidad total de parametros.
     */
    void summary() const;

    /*
     * Cross-Entropy Loss: Calcula la perdida entre los logits predichos
     * y el label one-hot verdadero. Aplica softmax internamente sobre
     * los logits antes de calcular -sum(y_true * log(softmax(logits))).
     */
    static double cross_entropy_loss(const Matrix& logits, const Matrix& y_true);

private:
    int num_classes_, d_model_, num_layers_;
    double learning_rate_;

    // Componentes del modelo
    PatchEmbedding* patch_embedding_;
    vector<TransformerEncoderBlock*> encoder_blocks_;
    LayerNorm* final_norm_;

    // Cabeza de clasificacion: Linear(d_model, num_classes)
    Matrix W_head_;   // (d_model, num_classes)
    Matrix b_head_;   // (1, num_classes)

    // Cache para backward
    Matrix encoder_output_cache_;  // Salida del ultimo encoder (seq_len, d_model)
    Matrix cls_output_cache_;      // Token CLS extraido (1, d_model)
    Matrix cls_normed_cache_;      // Token CLS normalizado (1, d_model)
    Matrix logits_cache_;          // Logits finales (1, num_classes)
};
