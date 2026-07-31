/**
 * Archivo: ActivationLayer.cuh
 * Descripcion: Implementa la capa de activacion no lineal ReLU en CUDA.
 * Rol en ViT: Despues de las operaciones de convolucion espacial, anade no linealidad
 * al mapa de caracteristicas apagando los valores negativos, permitiendo al modelo
 * aprender representaciones mas complejas antes del Patch Embedding.
 */

#pragma once
#include "Tensor3D.cuh"

class ReLUActivationLayer {
public:
    Tensor3D forward(const Tensor3D& input);
    Tensor3D backward(const Tensor3D& gradient_output);
private:
    Tensor3D mask_;
};