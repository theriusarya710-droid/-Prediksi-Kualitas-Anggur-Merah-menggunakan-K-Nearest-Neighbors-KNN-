# 🍷 Wine Quality Prediction — Tugas Besar Ilmu Data

**Prediksi kualitas anggur merah menggunakan algoritma K-Nearest Neighbors (KNN)**

> **Demo live:** `https://<username>.github.io/<repo-name>/`

---

## 📋 Deskripsi Proyek

| Item | Detail |
|------|--------|
| **Topik** | Prediksi Kualitas Anggur Merah |
| **Metode** | K-Nearest Neighbors (KNN), K=5 |
| **Dataset** | UCI Wine Quality Dataset (Red Wine) |
| **Fitur** | 11 parameter kimiawi |
| **Target** | Skor kualitas anggur (3–8) |
| **Akurasi** | 68,75% pada data uji |

---

## 🗂️ Struktur File

```
├── index.html        ← Website (1 halaman, model ter-embed)
├── train_model.py    ← Script training model di komputer lokal
├── model_data.json   ← Output training (sudah di-embed ke HTML)
└── README.md         ← Dokumentasi ini
```

---

## 🚀 Cara Deploy ke GitHub Pages

### Langkah 1 — Buat Repository GitHub
1. Buka [github.com](https://github.com) → **New repository**
2. Nama repo: `wine-quality-knn` (atau sesuai keinginan)
3. Set ke **Public**
4. Klik **Create repository**

### Langkah 2 — Upload File
**Via GitHub Web (cara mudah):**
1. Di halaman repo → klik **"uploading an existing file"**
2. Drag & drop file: `index.html`, `train_model.py`, `model_data.json`, `README.md`
3. Klik **Commit changes**

**Via Git (cara developer):**
```bash
git init
git add .
git commit -m "Initial commit: Wine Quality KNN"
git branch -M main
git remote add origin https://github.com/<username>/<repo-name>.git
git push -u origin main
```

### Langkah 3 — Aktifkan GitHub Pages
1. Di repo → **Settings** → **Pages** (menu kiri)
2. **Source:** Deploy from a branch
3. **Branch:** `main` / `(root)`
4. Klik **Save**
5. Tunggu 1–2 menit → URL muncul di bagian atas halaman Pages

### Langkah 4 — Akses Website
```
https://<username>.github.io/<repo-name>/
```

---

## 🧪 Cara Training Ulang Model

```bash
# Install dependensi
pip install numpy

# Jalankan training
python train_model.py

# Output: model_data.json
# Salin isi JSON ke variabel MODEL di index.html
```

---

## 📊 Tentang Algoritma KNN

**K-Nearest Neighbors** adalah algoritma klasifikasi non-parametrik yang:
1. Menghitung **jarak Euclidean** antara data baru dengan semua data training
2. Memilih **K tetangga terdekat** (K=5)
3. Menentukan kelas berdasarkan **voting mayoritas** dari tetangga

**Normalisasi Min-Max** digunakan agar semua fitur berada dalam skala [0, 1]:

```
x_norm = (x - x_min) / (x_max - x_min)
```

---

## 📁 Fitur Dataset

| No | Fitur | Satuan | Keterangan |
|----|-------|--------|-----------|
| 1 | Fixed Acidity | g/L | Asam tartrat |
| 2 | Volatile Acidity | g/L | Asam asetat (cuka) |
| 3 | Citric Acid | g/L | Asam sitrat |
| 4 | Residual Sugar | g/L | Sisa gula setelah fermentasi |
| 5 | Chlorides | g/L | Kandungan garam |
| 6 | Free Sulfur Dioxide | mg/L | SO₂ bebas |
| 7 | Total Sulfur Dioxide | mg/L | Total SO₂ |
| 8 | Density | g/cm³ | Massa jenis |
| 9 | pH | — | Tingkat keasaman |
| 10 | Sulphates | g/L | Aditif antimikroba |
| 11 | Alcohol | % vol | Kadar alkohol |

**Target:** Skor kualitas 3–8 (semakin tinggi = semakin baik)

---

*Tugas Besar Ilmu Data — Dataset: UCI Wine Quality (Red Wine)*