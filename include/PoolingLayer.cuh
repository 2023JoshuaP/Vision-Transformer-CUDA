/**
 * Archivo: PoolingLayer.cuh
 * Descripcion: Implementa Max Pooling 2D en CUDA para reduccion de dimensionalidad.
 * Rol en ViT: Filtra los rasgos mas dominantes del mapa convolucional y 
 * reduce la resolucion espacial, disminuyendo la carga computacional y la longitud 
 * total de la secuencia antes de inyectarse a los pesados bloques Transformer.
 */

#pragma once

#include "Tensor3D.cuh"
#include <string>

using namespace std;

enum class PoolingType { Max, Min, Average };

class PoolingLayer {
public:
    PoolingLayer(int kernel, int stride, PoolingType type);
    ~PoolingLayer();

    PoolingLayer(const PoolingLayer&) = delete;
    PoolingLayer& operator=(const PoolingLayer&) = delete;
    PoolingLayer(PoolingLayer&& other) noexcept;
    PoolingLayer& operator=(PoolingLayer&& other) noexcept;

    Tensor3D forward(const Tensor3D& input);
    Tensor3D backward(const Tensor3D& gradient_output) const;

    static int output_size(int input_size, int kernel, int stride);
    string getTypeName() const;

private:
    int pool_size_, stride_;
    PoolingType type_;

    int* d_selected_indices_;
    int allocated_indices_size_;
    int input_height_, input_width_, input_channels_;
};