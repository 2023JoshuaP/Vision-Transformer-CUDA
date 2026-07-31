import urllib.request
import gzip
import os
import struct

def download_file(url, filename):
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, filename)

def convert_to_csv(img_file, label_file, out_csv):
    print(f"Converting to {out_csv}...")
    with gzip.open(label_file, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = flbl.read()

    with gzip.open(img_file, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = fimg.read()

    with open(out_csv, 'w') as f:
        # Escribimos los datos: primera columna = label, resto = pixeles (0-255)
        for i in range(len(lbl)):
            label = lbl[i]
            pixels = img[i*rows*cols : (i+1)*rows*cols]
            pixels_str = ",".join(map(str, pixels))
            f.write(f"{label},{pixels_str}\n")
    print(f"Done! Saved {len(lbl)} records to {out_csv}")

if __name__ == '__main__':
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    
    files = {
        "train_img": "train-images-idx3-ubyte.gz",
        "train_lbl": "train-labels-idx1-ubyte.gz",
        "test_img": "t10k-images-idx3-ubyte.gz",
        "test_lbl": "t10k-labels-idx1-ubyte.gz"
    }

    os.makedirs("data", exist_ok=True)
    
    for k, v in files.items():
        if not os.path.exists(os.path.join("data", v)):
            download_file(base_url + v, os.path.join("data", v))
            
    convert_to_csv(os.path.join("data", files["train_img"]), 
                   os.path.join("data", files["train_lbl"]), 
                   "data/mnist_train.csv")
                   
    convert_to_csv(os.path.join("data", files["test_img"]), 
                   os.path.join("data", files["test_lbl"]), 
                   "data/mnist_test.csv")
