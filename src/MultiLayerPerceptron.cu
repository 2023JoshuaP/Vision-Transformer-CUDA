#include "MultiLayerPerceptron.cuh"
#include <iostream>
#include <cmath>
#include <iomanip>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <vector>

__global__ void bias_add_kernel(double* activations, const double* biases, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int c = idx % cols;
        activations[idx] += biases[c];
    }
}

__global__ void sgd_momentun_kernel(double* W, double* V, const double* gradient, int n, double learning_rate, double momentum, double weight_decay) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double grad_reg = gradient[idx] + weight_decay * W[idx];
        V[idx] = momentum * V[idx] + grad_reg;
        W[idx] -= learning_rate * V[idx];
    }
}

__global__ void sgd_momentun_bias_kernel(double* B, double* V, const double* gradient, int n, double learning_rate, double momentum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        V[idx] = momentum * V[idx] + gradient[idx];
        B[idx] -= learning_rate * V[idx];
    }
}

__global__ void cross_entropy_kernel(const double* y_prediction, const double* y_true, double* loss_out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double p = y_prediction[idx];
        if (p < 1e-12) p = 1e-12;
        if (p > 1.0 - 1e-12) p = 1.0 - 1e-12;
        loss_out[idx] = -y_true[idx] * log(p);
    }
}

__global__ void mse_element_kernel(const double *y_prediction, const double *y_true, double *loss_out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double error = y_prediction[idx] - y_true[idx];
        loss_out[idx] = error * error;
    }
}

__global__ void mlp_reduce_sum_kernel(const double* data, double* partial, int n) {
    extern __shared__ double s_data[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    s_data[tid] = (idx < n) ? data[idx] : 0.0;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();    
    }

    if (tid == 0) {
        partial[blockIdx.x] = s_data[0];
    }
}

/* Helper function to calculate number of blocks */
static inline int grid1d(int n, int block = 256) {
    return (n + block - 1) / block;
}

/* Sum of vector on GPU */
static double sum_gpu(const double *d_data, int n) {
    const int block = 256;
    const int g = grid1d(n, block);
    double *d_partial;

    CUDA_CHECK(cudaMalloc(&d_partial, g * sizeof(double)));
    mlp_reduce_sum_kernel<<<g, block, block * sizeof(double)>>> (d_data, d_partial, n);
    CUDA_CHECK(cudaGetLastError());

    vector<double> h_partial(g);
    CUDA_CHECK(cudaMemcpy(h_partial.data(), d_partial, g * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_partial));

    double total = 0;
    for (double v : h_partial) {
        total += v;
    }

    return total;
}

MultiLayerPerceptron::MultiLayerPerceptron(const vector<int> &layer_sizes, shared_ptr<ActivationFunction> activation, double learning_rate, double momentum, double weight_decay, int seed) : layer_sizes_(layer_sizes), activation_(activation), learning_rate_(learning_rate), momentum_(momentum), weight_decay_(weight_decay), num_layers_(static_cast<int>(layer_sizes.size())), rng_(seed) {
    mt19937 init_rng(seed);
    for (int i = 0; i < num_layers_ - 1; i++) {
        int fan_in = layer_sizes_[i];
        int fan_out = layer_sizes_[i + 1];
        double scale = sqrt(2.0 / fan_in);
        weights_.push_back(gpu_random(fan_in, fan_out, scale, init_rng));
        biases_.push_back(gpu_zeros(1, fan_out));
        vel_weights_.push_back(gpu_zeros(fan_in, fan_out));
        vel_biases_.push_back(gpu_zeros(1, fan_out));
    }
}

vector<Matrix> MultiLayerPerceptron::forward(const Matrix &input) const {
    vector<Matrix> activations;
    activations.reserve(num_layers_);
    activations.push_back(input);

    int n_weight_layers = num_layers_ - 1;
    Matrix A = input;
    for (int i = 0; i < n_weight_layers; i++) {
        Matrix Z = A.dot(weights_[i]);
        int total = Z.rows * Z.cols;

        bias_add_kernel<<<grid1d(total), 256>>> (Z.d_data, biases_[i].d_data, Z.rows, Z.cols);
        CUDA_CHECK(cudaGetLastError());

        if (i == n_weight_layers - 1) {
            A = Softmax::forward(Z);
        }
        else {
            A = activation_->forward(Z);
        }

        activations.push_back(A);
    }

    return activations;
}

void MultiLayerPerceptron::backward(const vector<Matrix> &activations, const Matrix &y_true) {
    int n = num_layers_ - 1;
    int n_lay = num_layers_ - 1;

    vector<Matrix> dW(n_lay);
    vector<Matrix> dB(n_lay);

    Matrix delta = activations.back() - y_true;

    for (int i = n_lay - 1; i >= 0; i--) {
        dW[i] = activations[i].transpose().dot(delta) / n;
        dB[i] = delta.col_mean();

        if (i > 0) {
            Matrix dA = delta.dot(weights_[i].transpose());
            Matrix derivate = activation_->derivative(activations[i]);
            delta = dA.hadamard(derivate);
        }
    }

    for (int i = 0; i < n_lay; i++) {
        int nw = weights_[i].rows * weights_[i].cols;
        sgd_momentun_kernel<<<grid1d(nw), 256>>> (weights_[i].d_data, vel_weights_[i].d_data, dW[i].d_data, nw, learning_rate_, momentum_, weight_decay_);
        CUDA_CHECK(cudaGetLastError());

        int nb = biases_[i].cols;
        sgd_momentun_bias_kernel<<<grid1d(nb), 256>>> (biases_[i].d_data, vel_biases_[i].d_data, dB[i].d_data, nb, learning_rate_, momentum_);
        CUDA_CHECK(cudaGetLastError());
    }
}

double MultiLayerPerceptron::mse_loss(const Matrix &y_pred, const Matrix &y_true) {
    int n = y_pred.size();
    double *d_loss;

    CUDA_CHECK(cudaMalloc(&d_loss, n * sizeof(double)));
    mse_element_kernel<<<grid1d(n), 256>>> (y_pred.d_data, y_true.d_data, d_loss, n);
    CUDA_CHECK(cudaGetLastError());

    double total_loss = sum_gpu(d_loss, n);
    cudaFree(d_loss);

    return total_loss / n;
}

void MultiLayerPerceptron::shuffle_data(Matrix &X, Matrix &y) {
    int n = X.rows;
    vector<int> indexes(n);
    iota(indexes.begin(), indexes.end(), 0);
    shuffle(indexes.begin(), indexes.end(), rng_);

    vector<double> h_X(n * X.cols);
    vector<double> h_y(n * y.cols);
    X.toHost(h_X.data());
    y.toHost(h_y.data());
    vector<double> h_X_shuffled(n * X.cols);
    vector<double> h_y_shuffled(n * y.cols);
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < X.cols; j++) {
            h_X_shuffled[i * X.cols + j] = h_X[indexes[i] * X.cols + j];
        }
        for (int j = 0; j < y.cols; j++) {
            h_y_shuffled[i * y.cols + j] = h_y[indexes[i] * y.cols + j];
        }
    }

    X.fromHost(h_X_shuffled.data());
    y.fromHost(h_y_shuffled.data());
}

TrainHistory MultiLayerPerceptron::train(const Matrix &X, const Matrix &y, int epochs, int batch_size, const Matrix *X_val, const Matrix *y_val, bool verbose, int patience) {
    TrainHistory history;
    bool has_val = (X_val != nullptr && y_val != nullptr);

    Matrix X_train = X;
    Matrix y_train = y;

    int n = X_train.rows;
    int batches = (n + batch_size - 1) / batch_size;

    double best_value_loss = 1e18;
    int epochs_no_improve = 0;
    vector<Matrix> best_weights = weights_;
    vector<Matrix> best_biases = biases_;

    for (int epoch = 1; epoch <= epochs; epoch++) {
        shuffle_data(X_train, y_train);
        double epoch_loss = 0.0;

        for (int b = 0; b < batches; b++) {
            int start = b * batch_size;
            int end = min(start + batch_size, n);
            Matrix X_batch = X_train.slice(start, end);
            Matrix y_batch = y_train.slice(start, end);
            auto activations = forward(X_batch);
            double batch_loss = mse_loss(activations.back(), y_batch);
            epoch_loss += batch_loss;

            backward(activations, y_batch);
        }

        epoch_loss /= batches;
        history.train_losses.push_back(epoch_loss);
        double value_loss = 0.0;

        if (has_val) {
            Matrix value_predice = predict(*X_val);
            value_loss = mse_loss(value_predice, *y_val);
            history.val_losses.push_back(value_loss);

            if (value_loss < best_value_loss) {
                best_value_loss = value_loss;
                epochs_no_improve = 0;
                best_weights = weights_;
                best_biases = biases_;
            }
            else {
                epochs_no_improve++;
            }

            if (epochs_no_improve >= patience) {
                if (verbose) {
                    cout << "Early stopping at epoch " << epoch << " (best val loss: " << fixed << setprecision(4) << best_value_loss << ")" << endl;
                }
                weights_ = best_weights;
                biases_ = best_biases;
                break;
            }
        }

        if (verbose && (epoch % max(1, epochs / 10) == 0 || epoch == 1)) {
            cout << "Epoch " << epoch << "/" << epochs << " - Train Loss: " << fixed << setprecision(4) << epoch_loss;
            if (has_val) {
                cout << " - Val Loss: " << fixed << setprecision(4) << value_loss;
            }
            cout << endl;
        }
    }

    return history;
}

Matrix MultiLayerPerceptron::predict(const Matrix &X) const {
    return forward(X).back();
}