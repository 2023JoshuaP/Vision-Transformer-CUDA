/**
 * Archivo: Tensor3D.cuh
 * Descripcion: Estructura algebraica de bajo nivel para arreglos volumetricos en VRAM.
 * Rol en ViT: Es el portador de imagenes original. Almacena las dimensiones de
 * (Canales, Altura, Anchura). Es crucial para pasar la data visual a traves de la CNN 
 * antes de que la imagen sea finalmente aplanada a una Matrix.
 */

#pragma once

#include "Matrix.cuh"
#include <random>

using namespace std;

struct Tensor3D {
    int channels, height, width;
    double *d_data;
    bool owns_data;

    Tensor3D();
    Tensor3D(int c, int h, int w, double value = 0.0);
    ~Tensor3D();

    Tensor3D(const Tensor3D &other);
    Tensor3D &operator=(const Tensor3D &other);

    Tensor3D(Tensor3D &&other) noexcept;
    Tensor3D &operator=(Tensor3D &&other) noexcept;

    void toHost(double *host_buf) const;
    void fromHost(const double *host_buf);

    Matrix flatten() const;
    static Tensor3D reconstructureFlatMatrix(const Matrix &matrix, int channels, int height, int width);

    int size() const {
        return channels * height * width;
    }

    Tensor3D operator+(const Tensor3D &other) const;
    Tensor3D operator-(const Tensor3D &other) const;
    Tensor3D operator*(double scalar) const;
};

Tensor3D tensorRandom(int channels, int height, int width, double scale, mt19937 &rng);