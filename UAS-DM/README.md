# 📊 Prediksi Risiko Penyakit Jantung Menggunakan Algoritma Klasifikasi

## 📋 Deskripsi Proyek
Proyek ini bertujuan untuk memprediksi risiko penyakit jantung menggunakan algoritma klasifikasi. Dataset yang digunakan berisi informasi tentang pasien, termasuk usia, jenis kelamin, tekanan darah, kolesterol, dan variabel lainnya.

## 🚀 Cara Menjalankan Proyek

1. **Persiapan Lingkungan**
   - Pastikan Anda memiliki Python terinstal di sistem Anda.
   - Instal pustaka yang diperlukan dengan menjalankan perintah berikut:
     ```bash
     pip install pandas scikit-learn seaborn matplotlib
     ```

2. **Mengunduh Dataset**
   - Unduh file `Heart_Disease_Classification_Dataset.csv` dan simpan di direktori yang sama dengan skrip Python.

3. **Menjalankan Skrip**
   - Buka terminal atau command prompt.
   - Navigasikan ke direktori tempat Anda menyimpan skrip `Final.py`.
   - Jalankan skrip dengan perintah:
     ```bash
     python Final.py
     ```

4. **Hasil**
   - Skrip akan menampilkan beberapa informasi, termasuk:
     - Lima baris pertama dari dataset.
     - Informasi tentang nilai yang hilang.
     - Hasil evaluasi model, termasuk akurasi, presisi, recall, dan F1-score.
     - Visualisasi matriks kebingungan dan distribusi usia terhadap risiko penyakit jantung.

## 📊 Penjelasan Proses

- **Pengumpulan Data**: Mengimpor dataset dan menampilkan beberapa baris pertama untuk memahami strukturnya.
- **Preprocessing Data**: Memeriksa dan menghapus nilai yang hilang, serta menormalkan fitur kolesterol dan tekanan darah.
- **Pemilihan Algoritma Klasifikasi**: Memisahkan fitur dan target, serta membagi data menjadi data pelatihan dan pengujian.
- **Pelatihan Model**: Membuat dan melatih model KNN (K-Nearest Neighbors).
- **Evaluasi Model**: Melakukan prediksi dan menghitung metrik evaluasi.
- **Visualisasi Hasil**: Menampilkan hasil dalam bentuk visual untuk analisis lebih lanjut.

## 📈 Contoh Data
Berikut adalah contoh beberapa baris dari dataset:
```
age: 61, sex: 0, cp: 0, trestbps: 130, chol: 330, fbs: 0, restecg: 0, thalach: 169, exang: 0, oldpeak: 0, slope: 2, ca: 0, thal: 2, target: 0
```
