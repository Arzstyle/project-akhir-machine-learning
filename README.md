# 📉 Customer Churn Prediction with Logistic Regression  
**Pemodelan Prediksi Churn dengan Regresi Logistik: Studi Kasus pada Pelanggan Berbasis Langganan**

Python • Pandas • Scikit-learn • Matplotlib • Seaborn • Streamlit-ready

---

## 🔗 Demo Aplikasi (Coming Soon)  
Aplikasi Streamlit akan segera tersedia. Saat ini, model dan pipeline sudah disiapkan dan siap untuk diintegrasikan ke antarmuka interaktif.

---

## 👥 Tim Pengembang  
Kelas: TI23A - Proyek Akhir Mata Kuliah Machine Learning

| Nama Lengkap         | NIM          | Peran                                      |
|----------------------|--------------|---------------------------------------------|
| Muhamad Akbar Rizky S          | 20230040   | Pengembang utama, dokumentasi & model ML    |
| Tresna Gunawan          | 20230040236   | dokumentasi & model ML    |
| Inda Fadila Nur Hawa          | 20230040236   | dokumentasi & model ML    |

---

## 📋 Table of Contents  
- 🎯 [Deskripsi Proyek](#deskripsi-proyek)  
- 📊 [Dataset](#dataset)  
- 🛠️ [Instalasi & Cara Menjalankan](#instalasi--cara-menjalankan)  
- 💡 [Metodologi](#metodologi)  
- 📈 [Visualisasi & Evaluasi](#visualisasi--evaluasi)  
- 🧠 [Analisis Fitur](#analisis-fitur)  
- 💾 [Penyimpanan Model](#penyimpanan-model)  
- 🏁 [Kesimpulan](#kesimpulan)  
- 📞 [Kontak](#kontak)

---

## 🎯 Deskripsi Proyek  
Industri telekomunikasi memiliki tingkat churn pelanggan sebesar 15–25% per tahun. Karena biaya akuisisi pelanggan baru jauh lebih tinggi dibanding mempertahankan pelanggan lama, prediksi churn menjadi aspek krusial dalam strategi bisnis.

Proyek ini bertujuan:
- Membangun model **Logistic Regression** untuk memprediksi churn
- Melakukan evaluasi performa model menggunakan metrik-metrik ML
- Mengidentifikasi **fitur utama** yang memengaruhi keputusan pelanggan untuk berhenti berlangganan

---

## 📊 Dataset  

Dataset digunakan dalam format CSV (`data-customer-churn.csv`) dan memuat informasi pelanggan berlangganan:

| Kolom           | Deskripsi                                    |
|------------------|----------------------------------------------|
| `gender`         | Jenis kelamin pelanggan                      |
| `SeniorCitizen`  | Apakah pelanggan adalah warga senior         |
| `tenure`         | Lama berlangganan (bulan)                    |
| `MonthlyCharges` | Biaya bulanan (dalam IDR)                    |
| `TotalCharges`   | Total biaya selama berlangganan              |
| `Contract`       | Jenis kontrak: bulanan, tahunan, dll         |
| `Churn`          | Status churn (Yes/No) sebagai target          |

### 🔎 Penanganan Data:
- Nilai kosong pada `TotalCharges` dibersihkan dan dikonversi
- Kolom `customerID` dihapus (tidak relevan untuk prediksi)
- Target `Churn` dikonversi menjadi 0 (No) dan 1 (Yes)

---

## 💡 Metodologi  

### 📦 Langkah-langkah yang Dilakukan:
1. **Preprocessing Data**  
   - Menghapus kolom `customerID` karena tidak relevan untuk prediksi  
   - Membersihkan nilai kosong pada `TotalCharges`  
   - Mengonversi variabel target `Churn` menjadi numerik (1 untuk "Yes", 0 untuk "No")  
   - Memisahkan fitur menjadi numerik dan kategorikal  
   - Menerapkan `StandardScaler` untuk fitur numerik dan `OneHotEncoder` untuk fitur kategorikal  

2. **Train-Test Split**  
   - Data dibagi menjadi 80% pelatihan dan 20% pengujian  
   - Stratifikasi dilakukan berdasarkan target (`Churn`) untuk menjaga distribusi kelas  

3. **Model Logistic Regression**  
   - Menggunakan solver `liblinear`  
   - Konfigurasi `class_weight='balanced'` untuk mengatasi imbalance dataset  

4. **Pipeline**  
   - Dibuat pipeline otomatis yang menggabungkan preprocessing dan model  
   - Menggunakan `ColumnTransformer` dan `Pipeline` dari `scikit-learn`  

5. **Evaluasi Model**  
   - Metode evaluasi: `Confusion Matrix`, `Accuracy`, `Precision`, `Recall`, `F1-Score`, `ROC AUC`  
   - Visualisasi hasil evaluasi: heatmap dan kurva ROC

---

## 📈 Visualisasi & Evaluasi  

### 📊 Visualisasi Utama:
- **Distribusi Target (Churn vs Tidak Churn)** menggunakan bar chart  
- **Confusion Matrix** dalam bentuk heatmap  
- **ROC Curve** untuk mengevaluasi kemampuan klasifikasi model  

### 📉 Hasil Evaluasi:
Metrik evaluasi berikut ditampilkan secara terpisah untuk model default dan model dengan class_weight:
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- ROC AUC  

Kurva ROC memperlihatkan keseimbangan antara true positive rate dan false positive rate, serta AUC menunjukkan seberapa baik model membedakan antara churn dan tidak churn.

---

## 🧠 Analisis Fitur  

### 🔍 Feature Importance:
- Koefisien model Logistic Regression digunakan untuk menentukan pengaruh setiap fitur  
- Fitur yang paling memengaruhi churn (positif dan negatif) ditampilkan dalam grafik  
- Visualisasi `Top 15 Feature Importance` menggunakan seaborn dengan palet warna `tab10`  
- Interpretasi:
  - Koefisien positif tinggi → fitur meningkatkan kemungkinan churn  
  - Koefisien negatif tinggi → fitur menurunkan kemungkinan churn  

---

## 💾 Penyimpanan Model  

Model akhir disimpan menggunakan `joblib` agar bisa digunakan di aplikasi Streamlit nantinya:
```python
joblib.dump(model_pipeline, "logistic_regression_churn_model.pkl")


## 🛠️ Instalasi & Cara Menjalankan  

### 📋 Prasyarat  
- Python >= 3.8  
- VS Code atau Google Colab  
- Paket: pandas, numpy, sklearn, matplotlib, seaborn, joblib  

### 🔧 Jalankan Notebook / Script  
```bash
# Jika menjalankan file .py
python machine_learning_project_sesi_akhir_finalisasi.py

# Jika dalam Colab
upload notebook dan dataset, lalu run semua cell
