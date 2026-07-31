/**
 * Archivo: DataLoader.cu
 * Descripcion: Gestiona la ingesta y preprocesamiento de datos desde el almacenamiento local.
 * Rol en ViT: Carga archivos CSV masivos (MNIST), normaliza los pixeles [0,1],
 * convierte las etiquetas a formato One-Hot y transfiere los batches enteros a la 
 * memoria VRAM para evitar embotellamientos (bottlenecks) durante el entrenamiento.
 */

#include "DataLoader.cuh"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <stdexcept>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <utility>
#include <map>
#include <vector>
#include <string>

using namespace std;
namespace fs = std::filesystem;

vector<string> DataLoader::CLASS_NAMES = {};
map<int, int> DataLoader::SYMBOL_ID_TO_INDEX = {};
int DataLoader::NUM_CLASSES = 0;

/* Load symbols from CSV file */
void DataLoader::load_symbols(const string &symbols_csv) {
    ifstream file(symbols_csv);
    if (!file.is_open()) {
        cerr << "Error: Could not open symbols CSV." << endl;
        return;
    }

    CLASS_NAMES.clear();
    SYMBOL_ID_TO_INDEX.clear();

    string line;
    getline(file, line);

    int index = 0;
    while (getline(file, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            continue;
        }

        stringstream ss(line);
        string symbol_id, latex_name;
        getline(ss, symbol_id, ',');
        getline(ss, latex_name, ',');

        int symbol_int_id = stoi(symbol_id);
        SYMBOL_ID_TO_INDEX[symbol_int_id] = index;
        CLASS_NAMES.push_back(latex_name);
        index++;
    }

    NUM_CLASSES = index;
    file.close();
}

/* Load data from CSV file */
pair<Matrix, Matrix> DataLoader::load_csv_data(const string &csv_path, const string &base_dir, int img_size) {
    if (NUM_CLASSES == 0) {
        cerr << "Could not load data, symbols not loaded." << endl;
        throw runtime_error("Symbols not loaded");
    }

    int flat_size = img_size * img_size;
    ifstream file(csv_path);
    if (!file.is_open()) {
        cerr << "Could not open csv file: " << csv_path << endl;
        throw runtime_error("Could not open csv file");
    }

    vector<vector<double>> rows_x;
    vector<int> labels;
    string line;
    getline(file, line);

    int loaded = 0, skipped = 0;
    while (getline(file, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            continue;
        }

        stringstream ss(line);
        string img_rel_path, symbol_id_str;
        getline(ss, img_rel_path, ',');
        getline(ss, symbol_id_str, ',');

        int symbol_id = stoi(symbol_id_str);
        auto it = SYMBOL_ID_TO_INDEX.find(symbol_id);
        if (it == SYMBOL_ID_TO_INDEX.end()) {
            skipped++;
            continue;
        }
        int label = it->second;

        fs::path full_path = fs::path(base_dir) / img_rel_path;
        cv::Mat image = cv::imread(full_path.string(), cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            skipped++;
            cerr << "Skipping ";
            continue;
        }

        if (image.rows != img_size || image.cols != img_size) {
            cv::resize(image, image, cv::Size(img_size, img_size), 0, 0, cv::INTER_AREA);
        }

        vector<double> flat_image(flat_size);
        for (int i = 0; i < img_size; i++) {
            for (int j = 0; j < img_size; j++) {
                flat_image[i * img_size + j] = static_cast<double>(image.at<uint8_t>(i, j)) / 255.0;
            }
        }

        rows_x.push_back(flat_image);
        labels.push_back(label);
        loaded++;


    }

    int N = static_cast<int>(rows_x.size());
    if (N == 0) {
        cerr << "No valid images found" << endl;
        throw runtime_error("No valid images found");
    }



    vector<double> h_X(N * flat_size);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < flat_size; j++) {
            h_X[i * flat_size + j] = rows_x[i][j];
        }
    }

    Matrix X(N, flat_size);
    X.fromHost(h_X.data());

    vector<double> h_Y(N * NUM_CLASSES, 0.0);
    for (int i = 0; i < N; i++) {
        h_Y[i * NUM_CLASSES + labels[i]] = 1.0;
    }

    Matrix Y(N, NUM_CLASSES);
    Y.fromHost(h_Y.data());

    return {move(X), move(Y)};
}

pair<vector<Tensor3D>, Matrix> DataLoader::load_mnist_csv_to_tensor(const string& csv_path, int limit) {
    ifstream file(csv_path);
    if (!file.is_open()) {
        cerr << "Could not open csv file: " << csv_path << endl;
        throw runtime_error("Could not open csv file");
    }

    vector<Tensor3D> images;
    vector<int> labels;

    string line;
    int count = 0;
    while (getline(file, line)) {
        if (limit > 0 && count >= limit) break;
        if (line.empty()) continue;

        stringstream ss(line);
        string token;
        getline(ss, token, ',');
        int label = stoi(token);
        labels.push_back(label);

        vector<double> img_data(28 * 28);
        int i = 0;
        while (getline(ss, token, ',') && i < 784) {
            img_data[i++] = stod(token) / 255.0;
        }

        Tensor3D t(1, 28, 28);
        t.fromHost(img_data.data());
        images.push_back(move(t));
        count++;
    }

    int N = labels.size();
    vector<double> h_Y(N * 10, 0.0);
    for (int i = 0; i < N; i++) {
        h_Y[i * 10 + labels[i]] = 1.0;
    }

    Matrix Y(N, 10);
    Y.fromHost(h_Y.data());

    return {move(images), move(Y)};
}

double DataLoader::accuracy(const Matrix& predictions, const Matrix& trues) {
    int total = predictions.rows;
    int cols = predictions.cols;

    vector<double> h_pred(total * cols);
    vector<double> h_trues(total * cols);
    predictions.toHost(h_pred.data());
    trues.toHost(h_trues.data());

    int correct = 0;
    for (int i = 0; i < total; i++) {
        int pred_label = 0;
        double best = h_pred[i * cols];
        for (int j = 1; j < cols; j++) {
            if (h_pred[i * cols + j] > best) {
                best = h_pred[i * cols + j];
                pred_label = j;
            }
        }

        int true_label = 0;
        for (int j = 1; j < cols; j++) {
            if (h_trues[i * cols + j] > h_trues[i * cols + true_label]) {
                true_label = j;
            }
        }
        if (pred_label == true_label) {
            correct++;
        }
    }

    return static_cast<double>(correct) / total * 100.0;
}