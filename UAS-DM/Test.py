#!/usr/bin/env python
# coding: utf-8

# #### "Prediksi Risiko Penyakit Jantung Menggunakan Algoritma Klasifikasi"

# ##### 1. Pengumpulan Data

# In[2]:


import pandas as pd

# Mengunduh dan membaca dataset
data = pd.read_csv('Heart_Disease_Classification_Dataset.csv')  # Ganti dengan path file Anda

# Menampilkan beberapa baris pertama dari dataset
print(data.head())


# - Kode ini mengimpor pustaka pandas dan membaca dataset dari file CSV. Kemudian, menampilkan lima baris pertama dari dataset untuk memahami strukturnya.

# ##### 2. Preprocessing Data

# In[3]:


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

# In[4]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Memisahkan fitur dan target
X = data.drop('target', axis=1)  # Fitur
y = data['target']  # Target

# Membagi data menjadi data pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# - Memisahkan dataset menjadi fitur (X) dan target (y). Kemudian, data dibagi menjadi data pelatihan dan pengujian dengan proporsi 80% untuk pelatihan dan 20% untuk pengujian.

# ##### 4. Pelatihan Model

# In[5]:


# Membuat model KNN
knn = KNeighborsClassifier(n_neighbors=5)

# Melatih model
knn.fit(X_train, y_train)


# - Membuat model KNN dengan 5 tetangga terdekat dan melatih model menggunakan data pelatihan.

# ##### 5. Evaluasi Model

# In[6]:


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
