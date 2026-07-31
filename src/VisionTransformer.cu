#include "VisionTransformer.cuh"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cmath>

static const int BLOCK_SIZE = 256;

__global__ void softmax_logits_kernel(const double* input, double* output, int num_classes) {
    int row = blockIdx.x;
    const double* in_row = input + row * num_classes;
    double* out_row = output + row * num_classes;

    double max_val = in_row[0];
    for (int i = 1; i < num_classes; i++) {
        if (in_row[i] > max_val) max_val = in_row[i];
    }

    double sum = 0.0;
    for (int i = 0; i < num_classes; i++) {
        double e = exp(in_row[i] - max_val);
        out_row[i] = e;
        sum += e;
    }

    for (int i = 0; i < num_classes; i++) {
        out_row[i] /= sum;
    }
}

VisionTransformer::VisionTransformer(int input_channels, int image_height, int image_width,
                                     int num_classes,
                                     int d_model, int num_heads, int num_layers, int mlp_dim,
                                     double learning_rate,
                                     int cnn_out_channels, int cnn_kernel, int cnn_stride, int cnn_padding,
                                     int pool_size, int pool_stride,
                                     int patch_h, int patch_w,
                                     int seed)
    : num_classes_(num_classes), d_model_(d_model), num_layers_(num_layers), learning_rate_(learning_rate) {
    
    patch_embedding_ = new PatchEmbedding(input_channels, image_height, image_width, d_model,
                                          cnn_out_channels, cnn_kernel, cnn_stride, cnn_padding,
                                          pool_size, pool_stride, patch_h, patch_w, seed);

    for (int i = 0; i < num_layers; i++) {
        encoder_blocks_.push_back(new TransformerEncoderBlock(d_model, num_heads, mlp_dim, seed + i));
    }

    final_norm_ = new LayerNorm(d_model);

    mt19937 rng(seed);
    W_head_ = gpu_random(d_model, num_classes, sqrt(2.0 / d_model), rng);
    b_head_ = gpu_zeros(1, num_classes);
}

VisionTransformer::~VisionTransformer() {
    delete patch_embedding_;
    for (auto block : encoder_blocks_) {
        delete block;
    }
    delete final_norm_;
}

Matrix VisionTransformer::forward(const Tensor3D& image) {
    Matrix x = patch_embedding_->forward(image);

    for (int i = 0; i < num_layers_; i++) {
        x = encoder_blocks_[i]->forward(x);
    }
    encoder_output_cache_ = x;

    cls_output_cache_ = x.slice(0, 1);
    cls_normed_cache_ = final_norm_->forward(cls_output_cache_);

    logits_cache_ = cls_normed_cache_.dot(W_head_);
    return logits_cache_;
}

void VisionTransformer::backward(const Matrix& y_true) {
    Matrix softmax_output(logits_cache_.rows, logits_cache_.cols);
    softmax_logits_kernel<<<logits_cache_.rows, 1>>>(logits_cache_.d_data, softmax_output.d_data, num_classes_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    Matrix grad_logits = softmax_output - y_true;

    Matrix grad_cls_normed = grad_logits.dot(W_head_.transpose());
    Matrix grad_W_head = cls_normed_cache_.transpose().dot(grad_logits);
    W_head_ = W_head_ - grad_W_head * learning_rate_;

    Matrix grad_cls = final_norm_->backward(grad_cls_normed, learning_rate_);

    Matrix grad_seq = gpu_zeros(encoder_output_cache_.rows, encoder_output_cache_.cols);
    CUDA_CHECK(cudaMemcpy(grad_seq.d_data, grad_cls.d_data, d_model_ * sizeof(double), cudaMemcpyDeviceToDevice));

    for (int i = num_layers_ - 1; i >= 0; i--) {
        grad_seq = encoder_blocks_[i]->backward(grad_seq, learning_rate_);
    }

    patch_embedding_->backward(grad_seq, learning_rate_);
}

int VisionTransformer::predict(const Tensor3D& image) {
    Matrix output = forward(image);
    
    vector<double> h_out(num_classes_);
    output.toHost(h_out.data());

    int best_class = 0;
    double max_val = h_out[0];
    for (int i = 1; i < num_classes_; i++) {
        if (h_out[i] > max_val) {
            max_val = h_out[i];
            best_class = i;
        }
    }
    return best_class;
}

double VisionTransformer::accuracy(const vector<Tensor3D>& X, const Matrix& y) {
    int correct = 0;
    int total = X.size();

    vector<double> h_y(y.rows * y.cols);
    y.toHost(h_y.data());

    for (size_t i = 0; i < X.size(); i++) {
        int pred_class = predict(X[i]);
        
        int true_class = 0;
        double max_val = h_y[i * num_classes_];
        for (int j = 1; j < num_classes_; j++) {
            if (h_y[i * num_classes_ + j] > max_val) {
                max_val = h_y[i * num_classes_ + j];
                true_class = j;
            }
        }
        
        if (pred_class == true_class) {
            correct++;
        }
    }
    return static_cast<double>(correct) / total;
}

TrainHistory VisionTransformer::train(const vector<Tensor3D>& X_train, const Matrix& y_train,
                                      int epochs, int batch_size,
                                      const vector<Tensor3D>* X_val, const Matrix* y_val,
                                      bool verbose, int patience) {
    TrainHistory history;
    int n = X_train.size();
    bool has_val = (X_val != nullptr && y_val != nullptr);

    double best_val_loss = 1e18;
    int epochs_no_improve = 0;

    vector<int> indices(n);
    iota(indices.begin(), indices.end(), 0);
    mt19937 rng(42);

    for (int epoch = 1; epoch <= epochs; epoch++) {
        shuffle(indices.begin(), indices.end(), rng);
        double epoch_loss = 0.0;
        int train_correct = 0;

        for (int i = 0; i < n; i++) {
            int idx = indices[i];
            Matrix logits = forward(X_train[idx]);
            Matrix y_sample = y_train.slice(idx, idx + 1);

            epoch_loss += cross_entropy_loss(logits, y_sample);

            vector<double> h_logits(logits.cols);
            logits.toHost(h_logits.data());
            int pred_class = distance(h_logits.begin(), max_element(h_logits.begin(), h_logits.end()));
            
            vector<double> h_y(y_sample.cols);
            y_sample.toHost(h_y.data());
            int true_class = distance(h_y.begin(), max_element(h_y.begin(), h_y.end()));

            if (pred_class == true_class) train_correct++;

            backward(y_sample);
        }
        
        epoch_loss /= n;
        double train_acc = (double)train_correct / n * 100.0;
        history.train_losses.push_back(epoch_loss);

        double val_loss = 0.0;
        double val_acc = 0.0;
        if (has_val) {
            int n_val = X_val->size();
            int val_correct = 0;
            for (int i = 0; i < n_val; i++) {
                Matrix logits = forward((*X_val)[i]);
                Matrix y_sample = y_val->slice(i, i + 1);
                val_loss += cross_entropy_loss(logits, y_sample);

                vector<double> h_logits(logits.cols);
                logits.toHost(h_logits.data());
                int pred_class = distance(h_logits.begin(), max_element(h_logits.begin(), h_logits.end()));
                
                vector<double> h_y(y_sample.cols);
                y_sample.toHost(h_y.data());
                int true_class = distance(h_y.begin(), max_element(h_y.begin(), h_y.end()));

                if (pred_class == true_class) val_correct++;
            }
            val_loss /= n_val;
            val_acc = (double)val_correct / n_val * 100.0;
            history.val_losses.push_back(val_loss);

            if (val_loss < best_val_loss) {
                best_val_loss = val_loss;
                epochs_no_improve = 0;
            }
            else {
                epochs_no_improve++;
            }

            if (epochs_no_improve >= patience) {
                if (verbose) cout << "Early stopping at epoch " << epoch << endl;
                break;
            }
        }

        if (verbose) {
            cout << "Epoch " << epoch << "/" << epochs 
                 << " - Train Loss: " << fixed << setprecision(4) << epoch_loss 
                 << " - Train Acc: " << fixed << setprecision(2) << train_acc << "%";
            if (has_val) {
                cout << " - Val Loss: " << fixed << setprecision(4) << val_loss
                     << " - Val Acc: " << fixed << setprecision(2) << val_acc << "%";
            }
            cout << endl;
        }
    }
    return history;
}

double VisionTransformer::cross_entropy_loss(const Matrix& logits, const Matrix& y_true) {
    Matrix softmax_output(logits.rows, logits.cols);
    softmax_logits_kernel<<<logits.rows, 1>>>(logits.d_data, softmax_output.d_data, logits.cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    vector<double> h_soft(logits.rows * logits.cols);
    vector<double> h_true(logits.rows * logits.cols);
    softmax_output.toHost(h_soft.data());
    y_true.toHost(h_true.data());

    double loss = 0.0;
    for (int i = 0; i < logits.rows * logits.cols; i++) {
        loss -= h_true[i] * log(h_soft[i] + 1e-9);
    }
    return loss / logits.rows;
}

void VisionTransformer::summary() const {
    cout << "=========================================" << endl;
    cout << "Vision Transformer (Hybrid) Architecture" << endl;
    cout << "=========================================" << endl;
    cout << "Patch Embedding Output: (" << patch_embedding_->get_seq_length() << " x " << d_model_ << ")" << endl;
    cout << "Transformer Encoders: " << num_layers_ << " layers" << endl;
    cout << "  - d_model: " << d_model_ << endl;
    cout << "  - Heads:   " << num_layers_ << endl;
    cout << "  - MLP dim: 4x" << d_model_ << endl;
    cout << "Classification Head: " << num_classes_ << " classes" << endl;
    cout << "=========================================" << endl;
}
