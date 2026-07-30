#pragma once

#include "Matrix.cuh"
#include <string>

using namespace std;

struct ActivationFunction {
    virtual ~ActivationFunction() {}
    virtual Matrix forward(const Matrix &z) const = 0;
    virtual Matrix derivative(const Matrix &a) const = 0;
    virtual string name() const = 0;
};

struct Sigmoid : public ActivationFunction {
    Matrix forward(const Matrix &z) const override;
    Matrix derivative(const Matrix &a) const override;
    string name() const override {
        return "sigmoid";
    }
};

struct ReLU : public ActivationFunction {
    Matrix forward(const Matrix &z) const override;
    Matrix derivative(const Matrix &a) const override;
    string name() const override {
        return "ReLU";
    }
};

struct Softmax {
    static Matrix forward(const Matrix &z);
};
