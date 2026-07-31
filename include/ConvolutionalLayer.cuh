/**
 * Archivo: ConvolutionalLayer.cuh
 * Descripcion: Implementa la convolucion 2D altamente paralela usando CUDA.
 * Rol en ViT: Es la etapa frontal (Front-End) del Vision Transformer Hibrido.
 * Su funcion es extraer bordes, texturas y formas espaciales de la imagen cruda 28x28,
 * transformandola en mapas de caracteristicas ricos antes de dividirse en secuencias.
 */

#pragma once

#include "Tensor3D.cuh"

class ConvolutionalLayer {
public:
  ConvolutionalLayer(int input_channels, int output_channels, int kernel,
                     int stride = 1, int padding = 0, int seed = 42);
  ~ConvolutionalLayer();

  ConvolutionalLayer(const ConvolutionalLayer &) = delete;
  ConvolutionalLayer &operator=(const ConvolutionalLayer &) = delete;
  ConvolutionalLayer(ConvolutionalLayer &&other) noexcept;
  ConvolutionalLayer &operator=(ConvolutionalLayer &&other) noexcept;

  Tensor3D forward(const Tensor3D &input);
  Tensor3D backward(const Tensor3D &gradient_output, double learning_rate);

  static int size_out(int input_size, int kernel, int stride, int padding);

  int get_output_channels() const { return output_channels_; }
  int get_kernel() const { return kernel_; }
  int get_stride() const { return stride_; }
  int get_padding() const { return padding_; }

private:
  int input_channels_, output_channels_, kernel_, stride_, padding_;
  double *
      d_kernels_; /* shape: (output_channels, input_channels, kernel, kernel) */
  double *d_biases_; /* shape: (output_channels) */

  Tensor3D input_cache_;
  int input_height_, input_width_;
};