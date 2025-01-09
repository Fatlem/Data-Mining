#!/usr/bin/env python
# coding: utf-8

# ## "Prediksi Risiko Penyakit Jantung Menggunakan Algoritma Klasifikasi"

# >- ###### NAMA : FAATHIR EL TASLEEM
# >- ###### NIM : A11.2022.14624
# >- ###### KELOMPOK : A11.4504

# ### Ringkasan dan Permasalahan Project
# - Ringkasan: Proyek ini bertujuan untuk mengembangkan model klasifikasi yang dapat memprediksi risiko penyakit jantung berdasarkan data medis pasien.
# - Permasalahan: Penyakit jantung merupakan salah satu penyebab utama kematian di seluruh dunia. Oleh karena itu, penting untuk memiliki alat yang dapat membantu dalam mendeteksi risiko penyakit jantung lebih awal.
# - Tujuan yang Akan Dicapai:
# Membangun model yang akurat untuk memprediksi risiko penyakit jantung.
# Mengidentifikasi faktor-faktor yang berkontribusi terhadap risiko penyakit jantung.
# - Model / Alur Penyelesaian:
#   - Pengumpulan Data: Mengumpulkan dataset yang relevan.
#   - Preprocessing Data: Membersihkan dan menyiapkan data untuk analisis.
#   - Pemilihan Algoritma: Memilih algoritma klasifikasi yang sesuai (misalnya, KNN).
#   - Pelatihan Model: Melatih model menggunakan data pelatihan.
#   - Evaluasi Model: Mengukur performa model menggunakan metrik evaluasi.

# ### Penjelasan Dataset, EDA, dan Proses Features Dataset
# - Dataset: Dataset berisi informasi medis pasien yang mencakup berbagai fitur seperti usia, jenis kelamin, tekanan darah, kolesterol, dan hasil diagnosis (target).
# - Exploratory Data Analysis (EDA):
#     - Memeriksa nilai yang hilang dan melakukan pembersihan data.
#     - Normalisasi fitur seperti kolesterol dan tekanan darah untuk memastikan skala yang konsisten.
# - Proses Features Dataset:
#     - Memisahkan fitur (X) dan target (y).
#     - Menggunakan teknik seperti MinMaxScaler untuk menormalkan data.

# ### Proses Learning / Modeling
# - Model yang Digunakan: KNeighborsClassifier (KNN) dengan 5 tetangga terdekat.
# - Langkah-langkah:
#     - Memisahkan dataset menjadi data pelatihan dan pengujian (80% pelatihan, 20% pengujian).
#     - Melatih model KNN menggunakan data pelatihan.

# ### Performa Model
# - Metrik Evaluasi:
#     - Akurasi: Mengukur seberapa banyak prediksi yang benar dibandingkan dengan total prediksi.
#     - Presisi: Mengukur proporsi prediksi positif yang benar.
#     - Recall: Mengukur proporsi kasus positif yang terdeteksi.
# F1-Score: Harmonik rata-rata dari presisi dan recall.
#     - Matriks Kebingungan: Menampilkan jumlah prediksi benar dan salah untuk analisis lebih lanjut.

# ##### 1. Pengumpulan Data

# In[1]:


import pandas as pd

# Mengunduh dan membaca dataset
data = pd.read_csv('Heart_Disease_Classification_Dataset.csv')  # Ganti dengan path file Anda

# Menampilkan beberapa baris pertama dari dataset
print(data.head())


# - Kode ini mengimpor pustaka pandas dan membaca dataset dari file CSV. Kemudian, menampilkan lima baris pertama dari dataset untuk memahami strukturnya.

# ##### 2. Preprocessing Data

# In[2]:


# Memeriksa nilai yang hilang
print(data.isnull().sum())

# Menghapus baris dengan nilai yang hilang (jika ada)
data.dropna(inplace=True)

# Normalisasi fitur (contoh: kolesterol dan tekanan darah)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
data[['chol', 'trestbps']] = scaler.fit_transform(data[['chol', 'trestbps']])

# Menampilkan informasi dataset setelah preprocessing
print(data.info())


# - Memeriksa apakah ada nilai yang hilang dalam dataset. Jika ada, baris tersebut dihapus. Kemudian, fitur kolesterol dan tekanan darah dinormalisasi ke rentang 0-1 menggunakan MinMaxScaler. Terakhir, informasi dataset ditampilkan untuk memastikan tidak ada nilai yang hilang.

# ##### 3. Pemilihan Algoritma Klasifikasi

# In[3]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Memisahkan fitur dan target
X = data.drop('target', axis=1)  # Fitur
y = data['target']  # Target

# Membagi data menjadi data pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# - Memisahkan dataset menjadi fitur (X) dan target (y). Kemudian, data dibagi menjadi data pelatihan dan pengujian dengan proporsi 80% untuk pelatihan dan 20% untuk pengujian.

# ##### 4. Pelatihan Model

# In[4]:


# Membuat model KNN
knn = KNeighborsClassifier(n_neighbors=5)

# Melatih model
knn.fit(X_train, y_train)


# - Membuat model KNN dengan 5 tetangga terdekat dan melatih model menggunakan data pelatihan.

# ##### 5. Evaluasi Model

# In[5]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Melakukan prediksi
y_pred = knn.predict(X_test)

# Menghitung metrik evaluasi
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Menampilkan hasil evaluasi
print(f'Akurasi: {accuracy:.2f}')
print(f'Presisi: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-Score: {f1:.2f}')

# Menampilkan matriks kebingungan
conf_matrix = confusion_matrix(y_test, y_pred)
print('Matriks Kebingungan:')
print(conf_matrix)


# - Melakukan prediksi pada data pengujian dan menghitung metrik evaluasi seperti akurasi, presisi, recall, dan F1-score. Hasil evaluasi ditampilkan, bersama dengan matriks kebingungan yang menunjukkan jumlah prediksi benar dan salah.

# ##### 6. Interpretasi dan Analisis Hasil

# In[6]:


import seaborn as sns
import matplotlib.pyplot as plt

# Visualisasi matriks kebingungan
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Matriks Kebingungan')
plt.show()


# - Menggunakan seaborn untuk membuat heatmap dari matriks kebingungan, yang membantu dalam visualisasi hasil prediksi model.

# ##### 7. Visualisasi dan Kesimpulan
# Menyajikan hasil dalam bentuk visual dan menarik kesimpulan terkait prediksi dan variabel penting.

# In[8]:


# Visualisasi distribusi fitur
plt.figure(figsize=(12, 8))
sns.countplot(data=data, x='age', hue='target')
plt.title('Distribusi Usia terhadap Risiko Penyakit Jantung')
plt.xlabel('Usia')
plt.ylabel('Jumlah Pasien')
plt.legend(title='Target', loc='upper right', labels=['No Disease', 'Disease'])
plt.show()


# - Membuat visualisasi distribusi usia terhadap risiko penyakit jantung, menunjukkan jumlah pasien berdasarkan usia dan status penyakit jantung.

# ### Diskusi Hasil dan Kesimpulan
# - Visualisasi Hasil: Menggunakan heatmap untuk matriks kebingungan dan grafik distribusi usia terhadap risiko penyakit jantung.-  
# - Kesimpulan:
# >- Model KNN dapat digunakan untuk memprediksi risiko penyakit jantung dengan akurasi yang memadai.
# >- Faktor-faktor seperti usia dan kolesterol memiliki pengaruh signifikan terhadap risiko penyakit jantung.

# In[ ]:




