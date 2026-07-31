/**
 * Archivo: DataLoader.cuh
 * Descripcion: Gestiona la ingesta y preprocesamiento de datos desde el almacenamiento local.
 * Rol en ViT: Carga archivos CSV masivos (MNIST), normaliza los pixeles [0,1],
 * convierte las etiquetas a formato One-Hot y transfiere los batches enteros a la 
 * memoria VRAM para evitar embotellamientos (bottlenecks) durante el entrenamiento.
 */

#pragma once

#include "Matrix.cuh"
#include <vector>
#include <string>
#include <utility>
#include <map>
#include "Tensor3D.cuh"

using namespace std;

struct DataLoader {
    static vector<string> CLASS_NAMES;
    static map<int, int> SYMBOL_ID_TO_INDEX;
    static int NUM_CLASSES;
    static void load_symbols(const string& symbols_csv);
    static pair<Matrix, Matrix> load_csv_data(const string& csv_path, const string& base_dir, int img_size = 32);
    static pair<vector<Tensor3D>, Matrix> load_mnist_csv_to_tensor(const string& csv_path, int limit = -1);
    static double accuracy(const Matrix& predictions, const Matrix& trues);
};
