/**
 * Archivo: Matrix.cuh
 * Descripcion: Estructura algebraica de bajo nivel alojada integramente en VRAM.
 * Rol en ViT: Es el pilar de datos 2D del modelo. Soporta multiplicaciones masivas (cuBLAS/dot-product),
 * transposiciones y cortes de tensores, permitiendo a las redes feed-forward y a la 
 * multiplicacion QKV procesar todos los parches de imagenes al mismo tiempo.
 */

#pragma once

#include <cstddef>
#include <cmath>
#include <random>
#include <stdexcept>
#include <cuda_runtime.h>

using namespace std;

struct Matrix {
    int rows, cols;
    double *d_data;
    bool owns_data;

    Matrix();
    Matrix(int r, int c, double val = 0.0);
    ~Matrix();

    Matrix(const Matrix &other);
    Matrix(Matrix &&other) noexcept;

    Matrix &operator=(const Matrix &other);
    Matrix &operator=(Matrix &&other) noexcept;
    double at(int r, int c) const;
    void set(int r, int c, double val);

    void toHost(double *host_buf) const;
    void fromHost(const double *host_buf);

    int size() const { return rows * cols; }
    Matrix operator+(const Matrix &other) const;
    Matrix operator-(const Matrix &other) const;
    Matrix operator*(double s) const;
    Matrix operator/(double s) const;

    Matrix hadamard(const Matrix &other) const;
    Matrix dot(const Matrix &other) const;
    Matrix transpose() const;

    Matrix col_mean() const;
    Matrix slice(int start_row, int end_row) const;
    Matrix slice_cols(int start_col, int end_col) const;
    Matrix concat_cols(const Matrix &other) const;

    double sum() const;
    double mean() const;
};

Matrix gpu_zeros(int rows, int cols);
Matrix gpu_random(int rows, int cols, double scale, mt19937 &rng);

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d — %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)
