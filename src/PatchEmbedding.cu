#include "PatchEmbedding.cuh"

// TODO: Kernel CUDA que extrae los parches del feature map y los aplana.
// Cada hilo se encarga de un parche: copia los valores del parche
// (C_out * patch_h * patch_w) a una fila de la matriz de salida.
// __global__ void extract_patches_kernel(...)

// TODO: Kernel CUDA inverso que redistribuye los gradientes de los parches
// aplanados de vuelta al feature map para el backward de la CNN.
// __global__ void reconstruct_patches_kernel(...)

PatchEmbedding::PatchEmbedding(int input_channels, int image_height, int image_width,
                               int d_model,
                               int cnn_out_channels, int cnn_kernel, int cnn_stride, int cnn_padding,
                               int pool_size, int pool_stride,
                               int patch_h, int patch_w,
                               int seed)
    : d_model_(d_model), patch_h_(patch_h), patch_w_(patch_w),
      conv_layer_(input_channels, cnn_out_channels, cnn_kernel, cnn_stride, cnn_padding, seed),
      relu_layer_(),
      pool_layer_(pool_size, pool_stride, PoolingType::Max) {
    // TODO:
    // 1. Calcular las dimensiones del feature map despues de la CNN:
    //    feature_h_ = PoolingLayer::output_size(
    //        ConvolutionalLayer::size_out(image_height, cnn_kernel, cnn_stride, cnn_padding),
    //        pool_size, pool_stride);
    //    feature_w_ = (similar para width)
    //    feature_c_ = cnn_out_channels;
    //
    // 2. Calcular num_patches_:
    //    num_patches_ = (feature_h_ / patch_h) * (feature_w_ / patch_w)
    //
    // 3. Calcular patch_dim_:
    //    patch_dim_ = feature_c_ * patch_h * patch_w
    //
    // 4. Crear W_proj_ (patch_dim_, d_model) con Xavier init en GPU.
    //    Crear b_proj_ (1, d_model) inicializado a cero.
    //
    // 5. Crear cls_token_ (1, d_model) con valores aleatorios pequenos.
    //
    // 6. Crear pos_embeddings_ (num_patches_ + 1, d_model) con valores
    //    aleatorios pequenos. El +1 es para la posicion del token [CLS].
}

PatchEmbedding::~PatchEmbedding() {
    // Los objetos Matrix/Tensor3D se liberan automaticamente con sus destructores.
}

Matrix PatchEmbedding::forward(const Tensor3D& image) {
    // TODO:
    // 1. Pasar la imagen por la CNN:
    //    conv_out = conv_layer_.forward(image)
    //    relu_out = relu_layer_.forward(conv_out)
    //    cnn_output_cache_ = pool_layer_.forward(relu_out)
    //
    // 2. Extraer parches del feature map:
    //    Lanzar extract_patches_kernel para copiar bloques del feature map
    //    a una matriz de forma (num_patches, patch_dim).
    //    Guardar en patches_flat_cache_.
    //
    // 3. Proyeccion lineal:
    //    projected = patches_flat_cache_.dot(W_proj_) + b_proj_
    //    -> (num_patches, d_model)
    //
    // 4. Prepend del token [CLS]:
    //    Crear una nueva Matrix (num_patches + 1, d_model).
    //    Fila 0 = cls_token_.
    //    Filas 1..num_patches = projected.
    //
    // 5. Sumar positional embeddings:
    //    output = output + pos_embeddings_
    //
    // 6. Retornar output (num_patches + 1, d_model).
    return Matrix(); // placeholder
}

Tensor3D PatchEmbedding::backward(const Matrix& grad_output, double learning_rate) {
    // TODO:
    // 1. Separar el gradiente del [CLS] (fila 0) del resto (filas 1..N):
    //    grad_cls = grad_output fila 0     -> (1, d_model)
    //    grad_patches = grad_output filas 1..N -> (num_patches, d_model)
    //
    // 2. Actualizar pos_embeddings_ con el gradiente completo.
    //    pos_embeddings_ -= learning_rate * grad_output
    //
    // 3. Actualizar cls_token_ con grad_cls.
    //    cls_token_ -= learning_rate * grad_cls
    //
    // 4. Backprop a traves de la proyeccion lineal:
    //    grad_flat = grad_patches.dot(W_proj_.transpose()) -> (num_patches, patch_dim)
    //    grad_W_proj = patches_flat_cache_.transpose().dot(grad_patches)
    //    Actualizar W_proj_ y b_proj_.
    //
    // 5. Reconstruir gradiente del feature map:
    //    Lanzar reconstruct_patches_kernel para redistribuir grad_flat
    //    de vuelta a un Tensor3D (feature_c, feature_h, feature_w).
    //
    // 6. Backprop a traves de la CNN:
    //    grad_pool = pool_layer_.backward(grad_feature_map)
    //    grad_relu = relu_layer_.backward(grad_pool)
    //    grad_image = conv_layer_.backward(grad_relu, learning_rate)
    //
    // 7. Retornar grad_image.
    return Tensor3D(); // placeholder
}
