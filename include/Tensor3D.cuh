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

    // Copy semantics
    Tensor3D(const Tensor3D &other);
    Tensor3D &operator=(const Tensor3D &other);

    // Move semantics
    Tensor3D(Tensor3D &&other) noexcept;
    Tensor3D &operator=(Tensor3D &&other) noexcept;

    // Memory transfer
    void toHost(double *host_buf) const;
    void fromHost(const double *host_buf);

    // Flat Matrix conversion
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