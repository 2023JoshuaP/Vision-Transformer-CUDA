#pragma once

#include "Matrix.cuh"
#include <vector>
#include <string>
#include <utility>
#include <map>

using namespace std;

struct DataLoader {
    static vector<string> CLASS_NAMES;
    static map<int, int> SYMBOL_ID_TO_INDEX;
    static int NUM_CLASSES;
    static void load_symbols(const string& symbols_csv);
    static pair<Matrix, Matrix> load_csv_data(const string& csv_path, const string& base_dir, int img_size = 32);
    static double accuracy(const Matrix& predictions, const Matrix& trues);
};
