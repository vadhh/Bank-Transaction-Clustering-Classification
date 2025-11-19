# Hybrid ML: Analisis Transaksi Bank untuk Segmentasi Pelanggan

Proyek ini mengimplementasikan alur kerja Machine Learning yang menggabungkan metode **Unsupervised Learning (K-Means Clustering)** dan **Supervised Learning (Classification)**. Tujuannya adalah untuk mengidentifikasi segmen pelanggan (clustering) dari data transaksi tanpa label dan kemudian membangun model klasifikasi untuk memprediksi segmen baru.

Proyek ini diselesaikan sebagai bagian dari **Submission Akhir** kelas **Belajar Machine Learning untuk Pemula (BMLP)**.

---

## 🛠️ Ringkasan Metode dan Alur Kerja

| Tahap | Metode Utama | Tujuan |
| :--- | :--- | :--- |
| **Clustering (Unsupervised)** | K-Means, Elbow Method, PCA (Opsional) | Mengelompokkan data transaksi menjadi `k` segmen pelanggan dan membuat kolom `Target`. |
| **Classification (Supervised)** | Decision Tree, Hyperparameter Tuning (GridSearch) | Melatih model untuk memprediksi `Target` (segmen) berdasarkan fitur transaksi. |

---

## 📂 Struktur Repositori
├── [Clustering]_Submission_Akhir_BMLP_Afridho_Tattaq_Tavadhu.ipynb # Notebook untuk Clustering dan Preprocessing 
├── [Klasifikasi]_Submission_Akhir_BMLP_Afridho_Tattaq_Tavadhu.ipynb # Notebook untuk Klasifikasi dan Tuning 
├── data_clustering.csv # Hasil ekspor data setelah Clustering (dengan kolom 'Target') 
├── model_clustering.h5 # Model K-Means yang sudah dilatih 
├── PCA_model_clustering.h5 # Model K-Means yang dilatih pada data hasil PCA (Opsional Advanced) 
├── decision_tree_model.h5 # Model Decision Tree Dasar 
└── tuning_classification.h5 # Model Decision Tree Terbaik setelah Hyperparameter Tuning

---

## 📊 Hasil dan Analisis Clustering (Unsupervised Learning)

### Pembersihan dan Pra-pemrosesan Data

1.  **Penanganan Missing Values & Duplikat:**
    * Duplikat baris dihapus (`df.drop_duplicates(inplace=True)`).
    * Nilai yang hilang pada fitur numerik (`CustomerAge`, `TransactionDuration`, `LoginAttempts`, `AccountBalance`) diisi dengan **median**.
    * Nilai yang hilang pada fitur kategorikal (`TransactionAmount`, `TransactionType`, `Location`, `Channel`, `CustomerOccupation`) diisi dengan **modus** (implisit melalui `LabelEncoder` atau `fillna` median/modus pada data *scaled*).
2.  **Feature Engineering & Encoding:**
    * Kolom ID dan Tanggal (`TransactionID`, `AccountID`, `DeviceID`, `IP Address`, `MerchantID`, `TransactionDate`, `PreviousTransactionDate`) di-*drop* karena kardinalitas tinggi/tidak relevan untuk *clustering*.
    * Fitur kategorikal tersisa di-*encode* menggunakan **LabelEncoder**.
3.  **Scaling & Outlier Handling (Advanced):**
    * Data dinormalisasi menggunakan **StandardScaler**.
    * Outlier ditangani dengan metode **IQR Capping** untuk menjaga distribusi namun membatasi nilai ekstrem agar tidak mendominasi proses *clustering*.

### Penentuan Jumlah Cluster (Elbow Method)

* **Metode yang Digunakan:** Elbow Method (menggunakan `KElbowVisualizer`).
* **Alasan Penggunaan:** Untuk mengidentifikasi jumlah cluster optimal (`k`) di mana penambahan cluster tidak lagi memberikan penurunan distorsi yang signifikan.
* **Hasil yang Didapat:** Nilai `k` optimal yang direkomendasikan adalah **7**.

### Evaluasi dan Interpretasi Cluster

* **Algoritma:** K-Means Clustering (`n_clusters=7`).
* **Metode Evaluasi (Skilled):**
    * **Silhouette Score:** **0.1473** (Skor ini menunjukkan adanya struktur cluster, meskipun batas-batasnya mungkin tidak sangat jelas, yang umum terjadi pada data transaksi nyata).
* **Interpretasi Hasil (Berdasarkan `df_unscaled.groupby('Target').mean()`):**

| Cluster | Nama Segmen | Ciri Khas Utama (Setelah Scaling/Z-Score) |
| :---: | :--- | :--- |
| **0** | **Pelanggan Konvensional Usia Matang** | `Channel` **Sangat Rendah** (-1.22), `CustomerAge` **Tinggi** (0.57). |
| **1** | **Pelanggan Transaksi Berdurasi Panjang** | `TransactionDuration` **Sangat Tinggi** (1.76). |
| **2** | **Pelanggan Konvensional Lokasi Spesifik (B)** | `Location` **Sangat Rendah** (-0.93), `Channel` **Sedang-Tinggi** (0.67). |
| **3** | **Pelanggan Muda Saldo Rendah (A)** | `CustomerAge` **Sangat Rendah** (-1.20), `AccountBalance` **Sangat Rendah** (-0.93). |
| **4** | **Pelanggan Umum Lokasi Spesifik (A)** | `Location` **Tinggi** (0.93), `Channel` **Sedang-Tinggi** (0.69). |
| **5** | **Pelanggan Muda Lokasi Spesifik (B)** | `CustomerAge` **Sangat Rendah** (-1.21), `CustomerOccupation` **Sangat Tinggi** (1.29). |
| **6** | **Nasabah Prioritas Saldo Tinggi** | `AccountBalance` **Sangat Tinggi** (1.55), `CustomerOccupation` **Sangat Rendah** (-1.24). |

---

## 🎯 Hasil Klasifikasi (Supervised Learning)

### Model Dasar (Decision Tree Classifier)

* **Algoritma:** Decision Tree Classifier (`random_state=42`).
* **Tujuan:** Mengukur kinerja model dasar dalam memprediksi label `Target` yang dihasilkan dari *clustering*.

| Metrik | Precision (Weighted Avg) | Recall (Weighted Avg) | F1-Score (Weighted Avg) | Accuracy |
| :---: | :---: | :---: | :---: | :---: |
| **Hasil Dasar** | 0.94 | 0.94 | 0.94 | **0.94** |

### Hyperparameter Tuning (Advanced)

* **Algoritma:** Decision Tree Classifier (Dipilih untuk *tuning*).
* **Metode:** GridSearchCV (Menggunakan 3-fold Cross-Validation).
* **Parameter yang Diuji:** `criterion`, `max_depth`, `min_samples_split`, `min_samples_leaf`.
* **Parameter Terbaik (`grid_search.best_params_`):**
    ```json
    {
      "criterion": "entropy",
      "max_depth": null,
      "min_samples_leaf": 4,
      "min_samples_split": 2
    }
    ```
* **Evaluasi Model Terbaik:**

| Metrik | Precision (Weighted Avg) | Recall (Weighted Avg) | F1-Score (Weighted Avg) | Accuracy |
| :---: | :---: | :---: | :---: | :---: |
| **Hasil Tuning** | 0.94 | 0.94 | 0.94 | **0.94** |

### Kesimpulan Klasifikasi

Model **Decision Tree Tuned** menunjukkan kinerja yang sangat tinggi (Akurasi **0.94**) dalam memprediksi segmen pelanggan yang telah dibuat oleh model *clustering*. Hal ini mengindikasikan bahwa fitur-fitur transaksi yang tersisa memiliki kekuatan diskriminatif yang kuat untuk membedakan antara segmen-segmen tersebut.

---
