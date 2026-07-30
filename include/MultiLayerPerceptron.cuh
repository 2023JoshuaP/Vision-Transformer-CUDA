#pragma once

#include "Matrix.cuh"
#include "Activation.cuh"
#include <vector>
#include <map>
#include <string>
#include <memory>
#include <random>

using namespace std;

struct TrainHistory {
    vector<double> train_losses;
    vector<double> val_losses;
};

class MultiLayerPerceptron {
    public:
        MultiLayerPerceptron(const vector<int> &layer_sizes, shared_ptr<ActivationFunction> activation, double learning_rate = 0.01, double momentum = 0.9, double weight_decay = 1e-4, int seed = 42);

        vector<Matrix> forward(const Matrix &input) const;
        void backward(const vector<Matrix> &activations, const Matrix &y_true);

        static double cross_entropy_loss(const Matrix &y_pred, const Matrix &y_true);
        static double mse_loss(const Matrix &y_pred, const Matrix &y_true);

        TrainHistory train(const Matrix &X, const Matrix &y, int epochs = 200, int batch_size = 32, const Matrix *X_val = nullptr, const Matrix *y_val = nullptr, bool verbose = false, int patience = 50);

        Matrix predict(const Matrix &X) const;
    
    private:
        vector<int> layer_sizes_;
        shared_ptr<ActivationFunction> activation_;
        double learning_rate_;
        double momentum_;
        double weight_decay_;
        int num_layers_;
        vector<Matrix> weights_;
        vector<Matrix> biases_;
        vector<Matrix> vel_weights_;
        vector<Matrix> vel_biases_;
        mt19937 rng_;

        void shuffle_data(Matrix &X, Matrix &y);
};
