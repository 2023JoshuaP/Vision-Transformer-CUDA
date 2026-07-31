#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "VisionTransformer.cuh"
#include "DataLoader.cuh"

using namespace std;

int main(int argc, char** argv) {
    cout << "Loading Full MNIST dataset (70,000 images)..." << endl;
    
    auto data1 = DataLoader::load_mnist_csv_to_tensor("data/mnist_train.csv", -1);
    auto data2 = DataLoader::load_mnist_csv_to_tensor("data/mnist_test.csv", -1);

    vector<Tensor3D> all_X;
    vector<int> all_labels;
    
    auto extract_labels = [](const Matrix& Y, vector<int>& labels) {
        vector<double> h_Y(Y.rows * Y.cols);
        Y.toHost(h_Y.data());
        for (int i = 0; i < Y.rows; i++) {
            int label = 0;
            double max_val = h_Y[i * Y.cols];
            for (int j = 1; j < Y.cols; j++) {
                if (h_Y[i * Y.cols + j] > max_val) {
                    max_val = h_Y[i * Y.cols + j];
                    label = j;
                }
            }
            labels.push_back(label);
        }
    };

    all_X.reserve(data1.first.size() + data2.first.size());
    for(auto& t : data1.first) all_X.push_back(move(t));
    for(auto& t : data2.first) all_X.push_back(move(t));

    extract_labels(data1.second, all_labels);
    extract_labels(data2.second, all_labels);

    int total_samples = all_X.size();
    cout << "Total samples loaded: " << total_samples << endl;

    vector<int> indices(total_samples);
    iota(indices.begin(), indices.end(), 0);
    mt19937 rng(42);
    shuffle(indices.begin(), indices.end(), rng);

    int num_train = total_samples * 0.70;
    int num_val = total_samples * 0.15;
    int num_test = total_samples - num_train - num_val;

    cout << "Splitting: Train=" << num_train << " | Val=" << num_val << " | Test=" << num_test << endl;

    vector<Tensor3D> X_train, X_val, X_test;
    vector<int> L_train, L_val, L_test;

    for (int i = 0; i < num_train; i++) {
        X_train.push_back(move(all_X[indices[i]]));
        L_train.push_back(all_labels[indices[i]]);
    }
    for (int i = num_train; i < num_train + num_val; i++) {
        X_val.push_back(move(all_X[indices[i]]));
        L_val.push_back(all_labels[indices[i]]);
    }
    for (int i = num_train + num_val; i < total_samples; i++) {
        X_test.push_back(move(all_X[indices[i]]));
        L_test.push_back(all_labels[indices[i]]);
    }

    auto make_onehot = [](const vector<int>& labels) {
        vector<double> h_Y(labels.size() * 10, 0.0);
        for (size_t i = 0; i < labels.size(); i++) {
            h_Y[i * 10 + labels[i]] = 1.0;
        }
        Matrix Y(labels.size(), 10);
        Y.fromHost(h_Y.data());
        return Y;
    };

    Matrix y_train = make_onehot(L_train);
    Matrix y_val = make_onehot(L_val);
    Matrix y_test = make_onehot(L_test);

    cout << "\nInicializando Vision Transformer..." << endl;
    VisionTransformer vit(
        1,
        28, 28,
        10,
        64,
        4,
        2,
        128,
        0.001,
        16, 3, 1, 1,
        2, 2,
        7, 7,
        42
    );

    vit.summary();

    cout << "\nIniciando entrenamiento..." << endl;
    TrainHistory history = vit.train(X_train, y_train, 5, 32, &X_val, &y_val, true, 2);

    cout << "\nEntrenamiento finalizado. Evaluando precision en conjunto de Test..." << endl;
    double acc = vit.accuracy(X_test, y_test);
    cout << "Test Accuracy final: " << acc * 100.0 << "%" << endl;

    cout << "\nGenerando predicciones sobre el conjunto Test..." << endl;
    
    int cols = 5;
    int rows = 5;
    int patch_w = 28;
    int patch_h = 28;
    int margin = 35;
    int scale = 4;
    int cell_w = patch_w * scale;
    int cell_h = patch_h * scale + margin;
    
    cv::Mat grid(rows * cell_h, cols * cell_w, CV_8UC3, cv::Scalar(255, 255, 255));

    int num_preds = min(25, (int)X_test.size());
    for (int i = 0; i < num_preds; i++) {
        int pred = vit.predict(X_test[i]);
        int true_val = L_test[i];
        
        cout << "Img " << i+1 << ": Prediccion = " << pred << " | Real = " << true_val;
        if (pred == true_val) cout << " [CORRECTO]" << endl;
        else cout << " [INCORRECTO]" << endl;

        vector<double> h_pixels(784);
        X_test[i].toHost(h_pixels.data());
        
        cv::Mat img(28, 28, CV_8UC1);
        for (int p = 0; p < 784; p++) {
            img.at<uint8_t>(p / 28, p % 28) = static_cast<uint8_t>(h_pixels[p] * 255.0);
        }
        
        cv::Mat color_img;
        cv::cvtColor(img, color_img, cv::COLOR_GRAY2BGR);
        cv::resize(color_img, color_img, cv::Size(cell_w, cell_h - margin), 0, 0, cv::INTER_NEAREST);
        
        int row = i / cols;
        int col = i % cols;
        
        cv::Rect roi(col * cell_w, row * cell_h + margin, cell_w, cell_h - margin);
        color_img.copyTo(grid(roi));
        
        string text = "P:" + to_string(pred) + " | R:" + to_string(true_val);
        cv::Scalar text_color = (pred == true_val) ? cv::Scalar(0, 200, 0) : cv::Scalar(0, 0, 200); // Verde o Rojo (BGR)
        cv::putText(grid, text, cv::Point(col * cell_w + 5, row * cell_h + margin - 10), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2);
    }

    cv::imwrite("resultados_prediccion.png", grid);
    cout << "\nPredicciones exportadas a 'resultados_prediccion.png' (Generado nativamente con OpenCV)!" << endl;

    return 0;
}
