"""
Tugas Besar Ilmu Data
Topik: Prediksi Kualitas Anggur (Wine Quality Prediction)
Metode: K-Nearest Neighbors (KNN)
Dataset: Wine Quality (UCI / Kaggle)

Jalankan script ini di komputer lokal untuk melatih model.
Output: model_data.json (dimasukkan ke dalam index.html)

Instalasi dependensi:
    pip install pandas scikit-learn requests
"""

import json
import numpy as np

# ── 1. Dataset (embedded agar tidak perlu download) ─────────────────────────
# Subset representatif dari UCI Wine Quality Dataset (red wine)
# Kolom: fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
#         chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
#         pH, sulphates, alcohol, quality

RAW_DATA = """7.4,0.7,0,1.9,0.076,11,34,0.9978,3.51,0.56,9.4,5
7.8,0.88,0,2.6,0.098,25,67,0.9968,3.2,0.68,9.8,5
7.8,0.76,0.04,2.3,0.092,15,54,0.997,3.26,0.65,9.8,5
11.2,0.28,0.56,1.9,0.075,17,60,0.998,3.16,0.58,9.8,6
7.4,0.7,0,1.9,0.076,11,34,0.9978,3.51,0.56,9.4,5
7.4,0.66,0,1.8,0.075,13,40,0.9978,3.51,0.56,9.4,5
7.9,0.6,0.06,1.6,0.069,15,59,0.9964,3.3,0.46,9.4,5
7.3,0.65,0,1.2,0.065,15,21,0.9946,3.39,0.47,10,7
7.8,0.58,0.02,2,0.073,9,18,0.9968,3.36,0.57,9.5,7
7.5,0.5,0.36,6.1,0.071,17,102,0.9978,3.35,0.8,10.5,5
6.7,0.58,0.08,1.8,0.097,15,65,0.9959,3.28,0.54,9.2,5
7.5,0.5,0.36,6.1,0.071,17,102,0.9978,3.35,0.8,10.5,5
5.6,0.615,0,1.6,0.089,16,59,0.9943,3.58,0.52,9.9,5
7.8,0.61,0.29,1.6,0.114,9,29,0.9974,3.26,1.56,9.1,5
8.9,0.62,0.18,3.8,0.176,52,145,0.9986,3.16,0.88,9.2,5
8.9,0.62,0.19,3.9,0.17,51,148,0.9986,3.17,0.93,9.2,5
8.5,0.28,0.56,1.8,0.092,35,103,0.9969,3.3,0.75,10.5,7
8.1,0.56,0.28,1.7,0.368,16,56,0.9968,3.11,1.28,9.3,5
7.4,0.59,0.08,4.4,0.086,6,29,0.9974,3.38,0.5,9,4
7.9,0.32,0.51,1.8,0.341,17,56,0.9969,3.04,1.08,9.2,6
8.5,0.28,0.56,1.8,0.092,35,103,0.9969,3.3,0.75,10.5,7
8.1,0.22,0.43,1.5,0.106,10,37,0.9966,3.17,0.91,9.5,6
8.1,0.27,0.41,1.45,0.033,11,63,0.9908,2.99,0.56,12,5
8.6,0.23,0.4,4.2,0.035,17,109,0.9947,3.14,0.53,11.5,7
7.9,0.43,0.21,1.6,0.106,10,37,0.9966,3.17,0.91,9.5,6
8.5,0.49,0.31,1.85,0.3467,20,65,0.99698,3.05,1.2,9.3,5
8.1,0.38,0.28,2.1,0.066,13,30,0.9968,3.23,0.73,9.7,7
7.4,0.35,0.33,2.4,0.068,5,24,0.9988,3.37,0.86,9.8,5
7.9,0.43,0.21,1.6,0.106,10,37,0.9966,3.17,0.91,9.5,6
8.5,0.43,0.36,1.82,0.336,20,65,0.9968,3.06,1.2,9.3,5
7.1,0.71,0,1.9,0.08,14,35,0.9972,3.47,0.55,9.4,5
7.8,0.645,0.12,2.0,0.082,8,16,0.9986,3.38,0.62,9.8,6
7.0,0.6,0.07,2.4,0.086,21,102,0.9978,3.38,0.48,9.5,5
6.4,0.31,0.31,2.4,0.056,17,60,0.9975,3.46,0.56,9.1,6
8.1,0.56,0.28,1.7,0.368,16,56,0.9968,3.11,1.28,9.3,5
7.3,0.37,0.32,1.7,0.087,22,75,0.9983,3.35,0.49,9.3,5
6.8,0.62,0.08,1.9,0.068,28,54,0.9965,3.3,0.55,9.8,6
5.9,0.55,0.1,2.2,0.062,39,51,0.9956,3.52,0.76,11.2,6
7.8,0.645,0.12,2.0,0.082,8,16,0.9986,3.38,0.62,9.8,6
8.9,0.4,0.5,2.3,0.228,7,29,0.99654,3.03,1.06,10.6,6
6.0,0.31,0.47,3.6,0.067,18,42,0.99549,3.39,0.66,11,6
5.8,0.24,0.44,3.5,0.029,5,109,0.9905,3.22,0.63,13,7
6.0,0.31,0.47,3.6,0.067,18,42,0.99549,3.39,0.66,11,6
6.3,0.51,0.13,2.3,0.076,29,40,0.9972,3.42,0.75,11,6
6.3,0.51,0.13,2.3,0.076,29,40,0.9972,3.42,0.75,11,6
7.5,0.31,0.53,2.2,0.039,15,33,0.9937,3.26,0.69,12.5,8
7.5,0.31,0.53,2.2,0.039,15,33,0.9937,3.26,0.69,12.5,8
7.4,0.35,0.33,2.4,0.068,5,24,0.9988,3.37,0.86,9.8,5
10.9,0.41,0.56,2.2,0.116,8,29,0.9985,3.14,1.1,9.8,6
6.3,0.39,0.16,1.4,0.08,11,23,0.9969,3.34,0.56,9.3,5
8.6,0.31,0.38,0.9,0.124,6,21,0.9976,3.07,0.98,9.3,6
7.3,0.52,0.26,2.3,0.083,28,63,0.9987,3.35,0.55,9.6,5
5.6,0.31,0.37,1.4,0.074,12,96,0.9954,3.32,0.58,9.2,5
8.6,0.23,0.4,4.2,0.035,17,109,0.9947,3.14,0.53,11.5,7
6.2,0.51,0.14,1.9,0.056,16,40,0.9969,3.48,0.58,9.8,6
8.3,0.645,0.12,2.0,0.082,8,16,0.9986,3.38,0.62,9.8,5
7.2,0.34,0.61,2.5,0.084,17,73,0.9979,3.29,0.69,10.8,6
9.1,0.4,0.5,2.3,0.228,7,29,0.99654,3.03,1.06,10.6,7
7.2,0.34,0.61,2.5,0.084,17,73,0.9979,3.29,0.69,10.8,6
11.1,0.26,0.68,2.35,0.116,6,23,0.9988,3.07,0.99,9.3,6
9.2,0.41,0.5,2.5,0.229,8,30,0.99654,3.04,1.08,10.6,6
9.0,0.45,0.38,2.2,0.071,17,42,0.9965,3.12,0.74,10.5,7
11.5,0.22,0.55,2.0,0.069,17,71,0.9988,3.06,0.9,10.5,6
8.0,0.44,0.23,2.2,0.067,16,38,0.9965,3.21,0.65,9.8,6
7.7,0.64,0.21,2.2,0.077,32,133,0.9982,3.29,0.59,9.5,5
8.3,0.31,0.41,2.0,0.073,22,67,0.9974,3.27,0.61,9.3,6
8.1,0.56,0.28,1.7,0.368,16,56,0.9968,3.11,1.28,9.3,5
8.6,0.49,0.28,1.86,0.1,20,136,0.99571,2.96,1.82,10.6,6
9.1,0.27,0.45,10.6,0.035,10,24,0.9978,3.0,0.8,12.9,7
9.4,0.3,0.56,2.8,0.08,22,114,0.9979,3.03,0.63,10.6,5
9.2,0.26,0.45,10.6,0.036,11,23,0.9978,3.0,0.81,12.9,7
13.5,0.53,0.79,2.2,0.083,31,116,1.001,3.1,0.9,9.2,5
10.6,0.31,0.49,2.24,0.2,6,21,0.9994,2.91,1.32,9.3,5
8.5,0.265,0.35,2.2,0.056,14,72,0.9962,3.16,0.78,10.4,7
10.0,0.26,0.54,1.9,0.083,10,55,0.9965,3.16,0.76,11,7
6.3,0.1,0.47,1.5,0.037,11,19,0.9908,3.22,0.63,13.5,8
7.8,0.5,0.26,2.0,0.063,11,34,0.9975,3.39,0.57,9.7,6
7.5,0.35,0.4,0.4,0.0668,28,59,0.9978,3.32,0.65,9.1,5
7.7,0.4,0.22,2.3,0.061,11,37,0.9984,3.48,0.73,9.8,5
7.1,0.57,0.0,2.3,0.098,18,38,0.9976,3.36,0.62,9.4,6"""

def parse_data(raw):
    rows = [line.strip().split(",") for line in raw.strip().split("\n")]
    X = [[float(v) for v in row[:11]] for row in rows]
    y = [int(row[11]) for row in rows]
    return X, y

def normalize(X_train, X_test):
    """Min-Max Normalisasi"""
    mins = [min(col) for col in zip(*X_train)]
    maxs = [max(col) for col in zip(*X_train)]
    def scale(X):
        return [[(v - mins[i]) / (maxs[i] - mins[i] + 1e-9) for i, v in enumerate(row)] for row in X]
    return scale(X_train), scale(X_test), mins, maxs

def euclidean(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

def knn_predict(X_train, y_train, x, k=5):
    distances = [(euclidean(x, xi), yi) for xi, yi in zip(X_train, y_train)]
    distances.sort(key=lambda t: t[0])
    neighbors = [yi for _, yi in distances[:k]]
    return max(set(neighbors), key=neighbors.count)

def train_test_split_manual(X, y, test_ratio=0.2, seed=42):
    np.random.seed(seed)
    idx = list(range(len(X)))
    np.random.shuffle(idx)
    split = int(len(X) * (1 - test_ratio))
    train_idx, test_idx = idx[:split], idx[split:]
    return [X[i] for i in train_idx], [X[i] for i in test_idx], \
           [y[i] for i in train_idx], [y[i] for i in test_idx]

# ── 2. Training ──────────────────────────────────────────────────────────────
print("=" * 50)
print("  Tugas Besar Ilmu Data")
print("  Wine Quality Prediction - KNN")
print("=" * 50)

X, y = parse_data(RAW_DATA)
X_train, X_test, y_train, y_test = train_test_split_manual(X, y)
X_train_norm, X_test_norm, mins, maxs = normalize(X_train, X_test)

K = 5
print(f"\nParameter  : K = {K}")
print(f"Data Train : {len(X_train)} sampel")
print(f"Data Test  : {len(X_test)} sampel")

# Evaluasi akurasi
correct = sum(1 for xi, yi in zip(X_test_norm, y_test)
              if knn_predict(X_train_norm, y_train, xi, K) == yi)
accuracy = correct / len(y_test) * 100
print(f"Akurasi    : {accuracy:.1f}%")

# ── 3. Simpan model ke JSON ──────────────────────────────────────────────────
FEATURES = [
    "Fixed Acidity", "Volatile Acidity", "Citric Acid", "Residual Sugar",
    "Chlorides", "Free Sulfur Dioxide", "Total Sulfur Dioxide",
    "Density", "pH", "Sulphates", "Alcohol"
]

model = {
    "k": K,
    "accuracy": round(accuracy, 2),
    "features": FEATURES,
    "mins": mins,
    "maxs": maxs,
    "X_train": X_train_norm,
    "y_train": y_train
}

with open("model_data.json", "w") as f:
    json.dump(model, f)

print("\n✅ Model disimpan ke: model_data.json")
print("   Salin isi file ini ke variabel MODEL_DATA di index.html")

# Tampilkan cuplikan JSON
print("\nCuplikan model_data.json:")
preview = json.dumps({k: v if k not in ("X_train", "y_train") else f"[{len(v)} items]"
                       for k, v in model.items()}, indent=2)
print(preview)