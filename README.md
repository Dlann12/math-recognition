# Handwritten Digit & Math Recognition

Aplikasi Deep Learning Computer Vision berbasis **TensorFlow/Keras** dan **Streamlit** untuk pengenalan angka tulisan tangan (*Handwritten Digit Recognition*) dan kalkulator ekspresi matematika tulisan tangan (*Handwritten Math Expression Recognition*).

Dilengkapi dengan kanvas gambar interaktif, preprocessing adaptif berbasis *Center of Mass* (standar MNIST), evaluasi model komprehensif (akurasi data uji **99.04%**), serta parser aritmatika aman berbasis AST (*Abstract Syntax Tree*) tanpa eksekusi `eval()` mentah.

---

## 🌟 Fitur Utama

### 1. Handwritten Digit Recognition (MVP)
- **Kanvas Gambar Interaktif:** Menggambar satu angka (0–9) menggunakan mouse atau layar sentuh dengan kontrol ketebalan garis (*stroke*).
- **Pengunggahan Gambar:** Mendukung unggahan file format PNG, JPG, dan JPEG.
- **Preprocessing Adaptif:** 
  - Penyesuaian polaritas otomatis (mengubah latar putih/garis hitam kanvas menjadi standar MNIST: latar hitam/garis putih).
  - *Bounding box crop* & *aspect-ratio preserved scaling* (20×20).
  - Pemusatan gambar menggunakan *Center of Mass* (28×28).
  - Normalisasi piksel `[0.0, 1.0]`.
- **Hasil Prediksi Detail:**
  - Angka hasil prediksi dan *Confidence Score* (%).
  - Peringkat Top-3 prediksi beserta persentase probabilitas.
  - Grafik distribusi probabilitas untuk seluruh 10 kelas angka (0–9).
  - Tampilan visual hasil preprocessing yang diterima oleh model.
  - Penanganan kanvas kosong atau coretan tidak valid.

### 2. Model Evaluation Dashboard
- **Metrik Pengujian pada 10.000 Data Test MNIST:**
  - **Akurasi Pengujian:** **99.04%** (melampaui target PRD 98%).
  - **Test Loss:** **0.0286**.
  - **Weighted Precision:** **99.04%** | **Weighted F1-Score:** **99.04%**.
- **Confusion Matrix:** Visualisasi matriks kesalahan beresolusi tinggi pada seluruh kelas angka.
- **Tabel Metrik per Kelas:** Presisi, recall, f1-score, dan jumlah sampel (*support*) untuk setiap digit 0–9.
- **Galeri Analisis Kesalahan (*Error Analysis*):** Contoh prediksi benar dan contoh prediksi salah pada data uji.

### 3. Handwritten Math Recognition & Calculator (Fitur Lanjutan)
- **Kanvas Ekspresi Matematika:** Menulis persamaan matematika satu baris (misal: `12 + 7 =` atau `45 * 2`).
- **Segmentasi Karakter Otomatis:** Memisahkan karakter dan simbol dari kiri ke kanan menggunakan analisis kontur dan penggabungan komponen vertikal (seperti tanda `=` dan `+`).
- **Model Math CNN (15 Kelas):** Mengklasifikasikan angka `0–9` serta operator `+`, `-`, `*`, `/`, dan `=`.
- **Fitur Koreksi Karakter:** Memungkinkan pengguna mengedit atau memvalidasi ekspresi sebelum dihitung.
- **Safe Math Evaluation:** Perhitungan ekspresi aritmatika menggunakan parser AST Python yang aman (menangani prioritas operator, tanda kurung, bilangan desimal/negatif, dan pembagian dengan nol) tanpa risiko injeksi kode.

---

## 📁 Struktur Folder Proyek

```text
handwritten-math-recognition/
├── app.py                     # Aplikasi web Streamlit (3 Halaman)
├── requirements.txt           # Daftar dependensi proyek
├── README.md                  # Dokumentasi proyek
├── .gitignore                 # Konfigurasi pengecualian Git
│
├── models/
│   ├── digit_cnn.keras        # Model CNN 10 kelas (MNIST Akurasi: 99.04%)
│   └── math_cnn.keras         # Model CNN 15 kelas (Angka + Operator Matematika)
│
├── data/
│   ├── raw/                   # Penyimpanan dataset mentah
│   └── processed/             # Penyimpanan dataset yang telah diproses
│
├── notebooks/
│   └── exploratory_analysis.ipynb  # Notebook EDA & Visualisasi Evaluasi
│
├── src/
│   ├── __init__.py
│   ├── train.py               # Script pelatihan model Digit CNN & Math CNN
│   ├── evaluate.py            # Script evaluasi model & generate laporan
│   ├── preprocess.py          # Pipeline preprocessing & segmentasi karakter
│   ├── predict.py             # Modul inferensi DigitPredictor & MathPredictor
│   ├── canvas.py              # Utilitas kanvas & visualisasi bounding box
│   └── math_parser.py         # Parser AST aritmatika aman (FR-11 & FR-12)
│
├── reports/
│   ├── classification_report.txt   # Laporan klasifikasi per kelas
│   ├── confusion_matrix.png        # Gambar plot Confusion Matrix
│   └── evaluation_metrics.json     # Metrik evaluasi tersimpan
│
└── tests/
    ├── __init__.py
    ├── test_preprocessing.py   # Pengujian unit preprocessing & segmentasi
    ├── test_prediction.py      # Pengujian unit inferensi model
    └── test_math_parser.py     # Pengujian unit kalkulator aritmatika aman
```

---

## 🚀 Panduan Instalasi & Penggunaan

### 1. Persiapan Environment Python

Pastikan Python 3.10+ telah terpasang. Buat dan aktifkan virtual environment:

```bash
# Buat virtual environment
python3 -m venv venv

# Aktifkan virtual environment
# Linux / macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependensi
pip install -r requirements.txt
```

### 2. Melatih Model (Opsional, Model Sudah Tersedia)

Model yang telah dilatih sudah tersimpan di direktori `models/`. Jika ingin melatih ulang:

```bash
# Melatih model Digit CNN (MNIST 10-Class)
python src/train.py --model-type digit --epochs 10

# Melatih model Math CNN (15-Class)
python src/train.py --model-type math --epochs 10

# Melatih kedua model sekaligus
python src/train.py --model-type all --epochs 10
```

### 3. Menjalankan Evaluasi Model

Untuk menguji performa model pada data uji MNIST dan membuat laporan klasifikasi serta Confusion Matrix:

```bash
python src/evaluate.py
```
Hasil evaluasi akan otomatis disimpan ke folder `reports/`.

### 4. Menjalankan Aplikasi Web Streamlit

Jalankan server aplikasi web Streamlit:

```bash
streamlit run app.py
```

Secara default, aplikasi dapat diakses di browser melalui:  
👉 **`http://localhost:8501`** (atau port lain yang ditentukan, misalnya `http://localhost:8505`).

---

## 🧪 Menjalankan Pengujian Otomatis (Pytest)

Seluruh komponen telah dilengkapi unit test komprehensif. Jalankan perintah berikut:

```bash
PYTHONPATH=. pytest tests/ -v
```

Hasil pengujian mencakup:
- `tests/test_math_parser.py`: Operasi aritmatika, prioritas operator, tanda kurung, desimal, penanganan pembagian nol, dan pengujian keamanan injeksi kode.
- `tests/test_preprocessing.py`: Deteksi gambar kosong, normalisasi polaritas latar, pemusatan *Center of Mass*, dan segmentasi karakter.
- `tests/test_prediction.py`: Pengujian inferensi model, format probabilitas, top-3 prediksi, dan pengujian akurasi batch data uji.

---

## 🔒 Keamanan Evaluasi Matematika (FR-11 & FR-12)

Aplikasi memenuhi persyaratan PRD **FR-11** dan **FR-12**:
- **Tidak menggunakan `eval()` bawaan Python** pada string input pengguna.
- Menggunakan parsing berbasis AST (*Abstract Syntax Tree*) dengan whitelist ketat:
  - Hanya mengizinkan operator: `ast.Add`, `ast.Sub`, `ast.Mult`, `ast.Div`, `ast.FloorDiv`, `ast.Pow`, `ast.USub`, `ast.UAdd`.
  - Hanya mengizinkan konstanta numerik (`int` dan `float`).
  - Menolak fungsi bawaan, modul, pemanggilan metode, variabel, atau string yang berpotensi membahayakan.
  - Menangani error sintaks dan `ZeroDivisionError` secara ramah pengguna.

---

## 📊 Hasil Evaluasi Model

| Metrik | Nilai | Keterangan |
|---|---|---|
| **Test Accuracy** | **99.04%** | Target PRD: minimal 98% |
| **Test Loss** | **0.0286** | Cross-entropy loss |
| **Macro Precision** | **99.05%** | Rata-rata presisi 10 kelas |
| **Macro Recall** | **99.02%** | Rata-rata recall 10 kelas |
| **Macro F1-Score** | **99.04%** | F1-Score seimbang |

File laporan lengkap tersedia di [reports/classification_report.txt](file:///home/kerja/Documents/math-recognition/reports/classification_report.txt) dan visualisasi di [reports/confusion_matrix.png](file:///home/kerja/Documents/math-recognition/reports/confusion_matrix.png).
