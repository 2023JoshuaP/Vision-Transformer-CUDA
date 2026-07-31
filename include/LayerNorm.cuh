#pragma once

#include "Matrix.cuh"

class LayerNorm {
public:
  LayerNorm(int dim, double epsilon = 1e-5);
  ~LayerNorm();

  LayerNorm(const LayerNorm &) = delete;
  LayerNorm &operator=(const LayerNorm &) = delete;

  /*
   * Forward: Recibe una Matrix de forma (batch, dim).
   * Normaliza cada fila independientemente usando media y varianza.
   * Aplica la transformacion afin con gamma y beta.
   * Guarda en cache los valores necesarios para el backward.
   * Retorna la Matrix normalizada de forma (batch, dim).
   */
  Matrix forward(const Matrix &input);

  /*
   * Backward: Recibe el gradiente de la salida (batch, dim).
   * Calcula los gradientes respecto a gamma, beta y la entrada.
   * Actualiza gamma y beta con el learning_rate proporcionado.
   * Retorna el gradiente respecto a la entrada (batch, dim).
   */
  Matrix backward(const Matrix &grad_output, double learning_rate);

private:
  int dim_;
  double epsilon_;

  double *d_gamma_;
  double *d_beta_;

  Matrix input_cache_;
  Matrix normalized_cache_;
  double *d_mean_;
  double *d_variance_;
  int cached_batch_size_;
};
