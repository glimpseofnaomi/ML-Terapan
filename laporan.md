# Laporan Proyek Machine Learning - Naomi Sitanggang

## Domain Proyek

Air bersih merupakan kebutuhan dasar manusia yang sangat krusial untuk kesehatan dan kehidupan sehari-hari. Namun, kualitas air di berbagai wilayah sering kali tidak terjamin akibat pencemaran oleh aktivitas industri, pertanian, maupun limbah rumah tangga. Oleh karena itu, penting untuk melakukan pengawasan kualitas air secara berkala untuk memastikan kelayakan konsumsi (potabilitas) air oleh masyarakat.

Proses konvensional dalam pengujian kualitas air biasanya memerlukan analisis laboratorium terhadap parameter-parameter fisikokimia seperti pH, kadar klorin, kandungan logam berat, dan lainnya. Meskipun akurat, pendekatan ini umumnya membutuhkan waktu lama, biaya tinggi, dan tenaga ahli (WHO, 2017). Seiring perkembangan teknologi, pendekatan berbasis _machine learning_ (ML) menjadi alternatif yang menjanjikan karena mampu memproses data dalam jumlah besar secara efisien, cepat, dan otomatis, serta memberikan prediksi yang akurat berdasarkan pola historis data.

Beberapa algoritma ML seperti Random Forest, XGBoost, Support Vector Machine (SVM), Decision Tree, dan Logistic Regression telah digunakan secara luas dalam berbagai studi untuk klasifikasi kualitas air dan prediksi potabilitas (Gazzaz et al., 2022). Algoritma-algoritma ini bekerja dengan mengenali hubungan kompleks antara parameter-parameter air dan label kelayakan (_potable_ atau _not potable_), sehingga dapat membantu otoritas pengelola air dalam mengambil keputusan yang cepat dan tepat.

Beberapa studi telah menunjukkan efektivitas _machine learning_ dalam klasifikasi potabilitas air. Misalnya, penelitian oleh Shah et al. (2021) menunjukkan bahwa model Random Forest dapat mengklasifikasikan kelayakan air dengan akurasi di atas 80%, bahkan tanpa menggunakan data mikrobiologis. Selain itu, Malakar et al. (2020) juga menegaskan bahwa penerapan model ML dapat menjadi solusi yang efektif untuk skala besar dalam pemantauan kualitas air di wilayah terpencil.

Dalam proyek ini, dilakukan pengembangan model klasifikasi potabilitas air berbasis data fisikokimia menggunakan beberapa algoritma _machine learning_, dengan tujuan mengevaluasi dan membandingkan performa masing-masing model, serta mengidentifikasi variabel-variabel yang paling berkontribusi terhadap kelayakan air minum. Harapannya, pendekatan ini dapat menjadi solusi pendukung dalam pengawasan kualitas air yang lebih efisien dan berbasis data.



## Business Understanding

### Problem Statements

1. Bagaimana cara memprediksi kelayakan air (_potable_ atau tidak) secara efisien?
2. Seberapa besar pengaruh variabel-variabel fisikokimia terhadap potabilitas air?
3. Algoritma _machine learning_ apa yang memiliki performa terbaik untuk klasifikasi potabilitas air?
4. Bagaimana cara meningkatkan performa model klasifikasi potabilitas air melalui teknik seperti _hyperparameter tuning_?

### Goals

1. Mengembangkan model klasifikasi berbasis _machine learning_ untuk memprediksi potabilitas air menggunakan parameter-parameter fisikokimia.
2. Mengidentifikasi variabel-variabel yang paling berpengaruh dalam menentukan apakah air layak minum atau tidak.
3. Membandingkan performa beberapa algoritma klasifikasi (Logistic Regression, Decision Tree, Random Forest, XGBoost, dan SVM).
4. Mengoptimalkan model terbaik menggunakan GridSearchCV untuk meningkatkan akurasi dan efisiensi model.

### Solution Statements
- Membangun dan melatih lima algoritma _machine learning_: Logistic Regression, Decision Tree, Random Forest, XGBoost, dan SVM untuk mengklasifikasikan potabilitas air.
- Melakukan evaluasi model berdasarkan evaluasi metrik, precision, recall, dan F1-score untuk mengukur kinerja model secara menyeluruh.
- Melakukan _feature importance analysis_ pada model terbaik untuk memahami kontribusi masing-masing parameter terhadap klasifikasi.
- Menerapkan teknik _hyperparameter tuning_ (GridSearchCV) pada model Random Forest untuk meningkatkan performa lebih lanjut.
- Menyediakan rekomendasi model terbaik yang dapat digunakan untuk pemantauan kualitas air secara otomatis dan efisien.

## Data Understanding

Proyek ini menggunakan data yang diambil dari situs Kaggle dengan judul  [Water Quality and Potability](https://www.kaggle.com/datasets/uom190346a/water-quality-and-potability/data). Dataset tersebut memuat sebanyak 3.276 entri yang berisi informasi mengenai berbagai parameter kimia dan fisika air. Dataset ini dipilih karena mampu memberikan gambaran yang komprehensif tentang faktor-faktor yang memengaruhi potabilitas air.

### Variabel-variabel pada dataset:

| Variabel         | Keterangan                                |
| -------------    | ----------------------------------------- |
| `pH`             | Tingkat pH air                            |
| `Hardness`       | Kesadahan air, ukuran kandungan mineral   |
| `Solids`         | Total padatan terlarut dalam air          |
| `Chloramines`    | Konsentrasi kloramina dalam air           |
| `Sulfate`        | Konsentrasi sulfat dalam air              |
| `Conductivity`   | Konduktivitas listrik air                 |
| `Organic carbon` | Kandungan karbon organik dalam air        |
| `Trihalometana`  | Konsentrasi trihalometana dalam air       |
| `Turbidity`      | Tingkat kekeruhan, ukuran kejernihan air  |
| `Potability`     | Variabel target; menunjukkan ketersediaan air dengan nilai 1 (dapat diminum) dan 0 (tidak dapat diminum)|

### Exploratory data analysis

#### 1. Struktur data
  | No | Variabel        | Tipe Data |
  |----|-------------- |-----------|
  | 0 | ph             | float64   |
  | 1 | Hardness       | float64   |
  | 2 | Solids         | float64   |
  | 3 | Chloramines    | float64   |
  | 4 | Sulfate        | float64   |
  | 5 | Conductivity   | float64   |
  | 5 | Organic carbon | float64   |
  | 7 | Trihalomethanes| float64   |
  | 8 | Turbidity      | float64   |
  | 9 | Potability     | int64     |

Tipe data yang dimiliki dari semua variabel merupakan tipe data numerik

#### 2. Mendeteksi missing values
  | No | Variabel         | Jumlah missing value |
  |----|--------------    |----------------------|
  | 0 | ph             | 491   |
  | 1 | Hardness       | 0     |
  | 2 | Solids         | 0     |
  | 3 | Chloramines    | 0     |
  | 4 | Sulfate        | 781   |
  | 5 | Conductivity   | 0     |
  | 5 | Organic carbon | 0     |
  | 7 | Trihalomethanes| 162   |
  | 8 | Turbidity      | 0     |
  | 9 | Potability     | 0     |

Dari hasil pendeteksian _missing values_ terlihat ada 3 variabel yang memiliki _missing values_ yaitu pH sebanyak 491, Sulfate sebanyak 781, dan Trihalomethanes sebanyak sebanyak 162.

#### 3. Mendeteksi data duplikat
Data tidak memiliki nilai duplikat

#### 4. Analisis distribusi dan korelasi
- Statistika deskriptif

|      | pH | Hardness | Solids | Chloramines | Sulfate | Conductivity | Organic_carbon | Trihalomethanes | Turbidity | Potability |
|------|----------|--------|-------------|---------|--------------|----------------|-----------------|-----------|----------|-----|
| count| 2785.0 | 3276.0 | 3276.0   | 3276.0| 2495.0 | 3276.0          | 3276.0         | 3114.0      | 3276.0       | 3276.0   |
| mean | 7.0    | 196.3  | 22014.00 | 7.1   | 333.7  | 426.2           | 14.2           | 66.3        | 3.9          | 0.3      |
| std  | 1.5    | 32.8   | 8768.5   | 1.5   | 41.4   | 80.8            | 3.3            | 16.1        | 0.7          | 0.4      |
| min  | 0.0    | 47.4   | 320.9    | 0.3   | 129.0  | 181.4           | 2.2            | 0.7         | 1.4          | 0.0      |
| 25%  | 6.0    | 176.8  | 15666.6  | 6.1   | 307.6  | 365.7           | 12.0           | 55.8        | 3.4          | 0.0      |
| 50%  | 7.0    | 196.9  | 20927.8  | 7.1   | 333.0  | 421.8           | 14.2           | 66.6        | 3.9          | 0.0      |
| 75%  | 8.0    | 216.6  | 27332.7  | 8.1   | 359.9  | 481.7           | 16.5           | 77.3        | 4.5          | 1.0      |
| max  | 14.0   | 323.1  | 61227.1  | 13.1  | 481.0  | 753.3           | 28.3           | 124.0       | 6.7          | 1.0      |

   Berdasarkan statistik deskriptif, variabel seperti ph memiliki nilai minimum 0 yang tidak realistis, menunjukkan kemungkinan data tidak valid. Proporsi data Potability menunjukkan ketidakseimbangan kelas (sekitar 39% layak konsumsi). Beberapa variabel seperti Solids dan Trihalomethanes memiliki rentang nilai luas, mengindikasikan kemungkinan _outlier_. Maka, diperlukan penanganan terhadap data hilang dan _outlier_ sebelum pemodelan.

- Distribusi variabel 

  ![Distribusi Label](https://drive.google.com/uc?export=view&id=1xjeODXx-uc5LvhTeYBj83q1gCEsjiKia)

  Berdasarkan grafik histogram menunjukkan bahwa sebagian besar variabel terdistribusi normal, meskipun ada sedikit kemencengan pada variabel seperti Solids dan Trihalomethanes. 

- Distribusi persentase variabel Potability

  ![Distribusi Label](https://drive.google.com/uc?export=view&id=1Ibng71hKaEUXtmOzkzFg18XDi_8K_WkK)

  Berdasarkan grafik _pie chart_ terlihat bahwa persentase ketersediaan air yang tidak dapat diminum memiliki persentase lebih besar yaitu 61% dibandingkan ketersediaan air yang tidak dapat diminum yaitu 39%.

- Distribusi variabel Potability pada variabel non target

  ![Distribusi Label](https://drive.google.com/uc?export=view&id=1yRz2O5cuzlXZDXBScpvngKz0Mu9ES187)

  Pada grafik, terlihat bahwa variabel-variabel yang memengaruhi status dapat diminum dan tidak dapat diminum tidak jauh berbeda satu sama lain.

- Distribusi pairplot

  ![Distribusi Label](https://drive.google.com/uc?export=view&id=1e2yipzuLm_3qWO7jQyd92s7qM2Z2u253)

  Berdasarkan visualisasi _pairplot_ di atas, terlihat bahwa Beberapa distribusi variabel tampak tidak simetris dan cenderung _skewed_ yang mengindikasikan perlunya transformasi seperti _log transform_ pada tahap praproses untuk meningkatkan performa model.

- Boxplot

  ![Distribusi Label](https://drive.google.com/uc?export=view&id=1cDJvmU2C1KcGEXDjsOiCssHM0tFr769J)

  Berdasarkan visualisasi _boxplot_ di atas terlihat bahwa variabel seperti Solids dan Trihalomethanes memiliki sebaran data yang cukup luas dan _outlier_ ekstrem. Meskipun distribusi data secara umum tampak simetris pada beberapa variabel, keberadaan _outlier_ dapat memengaruhi kinerja model _machine learning_ sehingga perlu dilakukan penanganan.

- Heatmap korelasi

  ![Distribusi Label](https://drive.google.com/uc?export=view&id=1lH_kZM3gyQY96od2hGPF1d_hM1_qHL6V)

  Berdasarkan hasil _heatmap_ korelasi antar variabel di atas, dapat disimpulkan bahwa hubungan antar variabel dalam dataset ini cenderung lemah. Korelasi paling besar secara absolut adalah antara Solids dan Sulfate (-0.17), tetapi nilainya tetap tergolong rendah. Hal ini menunjukkan bahwa pendekatan klasifikasi yang digunakan tidak bisa hanya mengandalkan hubungan linier antar variabel, melainkan perlu model yang mampu menangkap hubungan non-linier, seperti Random Forest atau XGBoost.

## Data Preparation

Pada tahap ini, dilakukan beberapa teknik untuk menyiapkan data sebelum masuk ke proses pemodelan. Urutan dan penjelasan tiap langkah sebagai berikut:

1. Menangani missing values
   Setelah melakukan pengecekan _missing values_, terdeteksi terdapat _missing values_ pada 3 variabel. Hal ini perlu diatasi dengan mengisi _missing values_ pada variabel tersebut menggunakan nilai median karena data yang dimiliki terindikasi _skewed_. Tujuan mengisi _missing values_ ini agar model tidak _error_ saat pelatihan dan dapat meningkatkan kualitas data.

2. Menangani skewness
   _Skewness_ dapat dideteksi dengan meilihat _boxplot_ dan juga distribusi histogram, untuk mengatasi masalah _skewness_ dilakukan _log transform_. Tujuan dilakukannya _log transform_ adalah untuk mengurangi pengaruh _outlier_ ekstrem dan membantu algoritma bekerja lebih stabil dan akurat.


3. Standarisasi data 
   Semua variabel numerik dinormalisasi menggunakan StandardScaler agar memiliki skala yang sama. Tujuan melakukan standarisasi data supaya model bisa belajar lebih efisien, terutama pada algoritma seperti SVM yang sensitif terhadap perbedaan skala fitur.

4. Menangani imbalance data
   Terdapat pembagian kelas yang tidak merata pada variabel target untuk menagatasi masalah ini dilakukan SMOTE. Tujuan dilakukan SMOTE untuk membantu model belajar pola dari kedua kelas secara merata.

5. Membagi data
   Dataset dibagi menjadi 80% data latih dan 20% data uji.Tujuan melakukan pembagian data ini agar model memiliki cukup data untuk belajar pola secara optimal, sekaligus dapat dievaluasi performanya pada data yang belum pernah dilihat. Rasio ini umum digunakan karena memberikan keseimbangan antara akurasi pelatihan dan kemampuan generalisasi, sehingga model tidak _overfitting_ dan tetap andal saat digunakan pada data baru.

## Modeling

Pada tahap modeling, digunakan lima algoritma untuk membangun dan membandingkan performa model klasifikasi, yaitu Logistic Regression, Decision Tree, Random Forest, XGBoost, dan Support Vector Machine (SVM)

1. **Logistic Regression**
   Logistic Regression memiliki kelebihan berupa kesederhanaan, interpretabilitas tinggi, dan efisien untuk dataset linier. Namun, kekurangannya adalah performa yang rendah ketika terdapat relasi non-linier atau fitur yang saling berinteraksi secara kompleks. Model ini digunakan sebagai _baseline_ karena cepat dan ringan untuk pelatihan awal.
   Parameter yang digunakan :  
   - `max_iter=10000` : Mengatur jumlah maksimum iterasi agar konvergen
   - `random_state=42` : Untuk menjaga konsistensi hasil antar running
   

2. **Decision Tree**
   Decision Tree memiliki kelebihan mudah dipahami dan diinterpretasikan, serta mampu menangani fitur kategorikal dan numerik tanpa perlu skala. Kelemahannya adalah cenderung *overfitting* terhadap data train, terutama jika tidak dilakukan pruning. Parameter yang digunakan: 
   - `random_state=42`:  Untuk menjaga konsistensi hasil antar running


3. **Random Forest**
   Random Forest memiliki kelebihan untuk mengatasi _overfitting_ dari Decision Tree dengan menggabungkan banyak pohon (_ensemble_). Model ini lebih stabil dan akurat dibanding pohon tunggal, tetapi kekurangannya adalah kurang interpretatif dan lebih mahal secara komputasi. Model ini ditingkatkan dari Decision Tree untuk hasil prediksi yang lebih konsisten.
   Parameter sebelum tuning:
   - `random_state=42`:  Untuk menjaga konsistensi hasil antar running

   Parameter setelah tuning:
   - `n_estimators`: [100, 200,300] : Jumlah pohon pada ensemble
   - `max_depth`:  [None, 10, 20, 30]  :  Kedalaman maksimum pohon
   - `min_samples_split`:  [2, 5, 10]  :  Minimum jumlah sampel untuk split
   - `min_samples_leaf`:   [1, 2, 4]  :  Minimum sampel di daun


4. **XGBoost**
   XGBoost adalah model _boosting_ yang sangat kuat, dengan keunggulan dalam menangani data tidak seimbang, performa tinggi, dan memiliki banyak opsi regularisasi untuk menghindari _overfitting_. Kekurangannya adalah kompleksitas tinggi dan waktu pelatihan yang relatif lebih lama. Model ini diuji sebagai kandidat model terbaik karena kemampuannya dalam menangani dataset kompleks. Parameter yang digunakan :
   - `use_label_encoder=False`:  Menonaktifkan label encoder bawaan untuk menghindari warning
   - `eval_metric='logloss'`:  Metrik evaluasi untuk klasifikasi biner
   - `random_state=42`:  Untuk hasil konsisten
   
5. **Support Vector Machine (SVM)**
   SVM efektif untuk dataset berdimensi tinggi dan mampu menemukan _hyperplane_ optimal. Kelemahannya adalah sensitif terhadap skala fitur dan kurang efisien pada dataset besar. SVM digunakan untuk menguji performa pada margin yang maksimal dalam pemisahan kelas. Parameter yang digunakan :
   - `probability=True'`:  Agar model bisa menghasilkan probabilitas untuk klasifikasi
   - `random_state=42`:  Untuk hasil konsisten


### Hyperparameter tuning
Dari semua model yang digunakan, hanya Random Forest yang dilakukan hyperparameter tuning secara eksplisit menggunakan GridSearchCV, sementara model lain menggunakan parameter default atau _baseline_ untuk pembandingan awal. Hal ini karena Random Forest merupakan model terbaik karena memiliki nilai metrik evaluasi tertinggi.


## Evaluation

### Metrik evaluasi yang digunakan:

- **Presisi**

  $\displaystyle \text{Presisi} = \frac{TP}{TP + FP}$

  Presisi menunjukkan seberapa tepat model saat menyatakan air sebagai “potable”. Dalam konteks ini, presisi tinggi berarti ketika model mengatakan air tersebut layak minum, kemungkinan besar itu benar. Ini penting untuk menghindari risiko kesehatan akibat salah memprediksi air tercemar sebagai aman (_false positive_).

- **Recall**

  $\displaystyle \text{Recall} = \frac{TP}{TP + FN}$

  Recall mengukur seberapa banyak air yang benar-benar layak minum yang berhasil dikenali oleh model. Jika recall rendah, model melewatkan banyak air bersih yang seharusnya diklasifikasikan sebagai “potable”, yang dapat berdampak pada efisiensi pemanfaatan sumber air bersih.

- **F1-Score**

  $\displaystyle F1 = 2 \times \frac{\text{Presisi} \times \text{Recall}}{\text{Presisi} + \text{Recall}}$

  F1-score adalah rata-rata harmonik antara presisi dan recall, dan sangat berguna jika kita ingin menyeimbangkan antara tidak memberikan air tercemar kepada publik (tinggi presisi) dan tidak membuang air bersih secara sia-sia (tinggi recall). 

- **Akurasi**

  $\displaystyle \text{Akurasi} = \frac{TP + TN}{TP + TN + FP + FN}$

  Akurasi mengukur proporsi total prediksi yang benar dari seluruh data. Jika model memiliki akurasi tinggi, itu berarti secara umum model mampu memprediksi kelayakan air dengan baik.


### Hasil evaluasi sebelum tuning

| Model                    | Accuracy | Precision | Recall | F1-Score |
| ------------------------ | -------- | --------- | ------ | -------- |
| Logistic Regression      | 0.5150     | 0.5150      | 0.5150   | 0.5150   |
| Decision Tree            | 0.6462     | 0.6470      | 0.6462   | 0.6458   |
| Random Forest            | 0.7275     | 0.7279      | 0.7275   | 0.7274   |
| XGBoost                  | 0.6725     | 0.6725      | 0.6725   | 0.6725   |
| SVM                      | 0.6475     | 0.6475      | 0.6475   | 0.6475   |

Model Random Forest memberikan performa terbaik di antara semua model sebelum _tuning_, dengan nilai akurasi sebesar 0.7275, nilai presisi sebesar 0.7279, nilai recall sebesar 0.7275 , dan F1-score sebesar 0.7274. Hal ini menunjukkan bahwa model ini cukup andal dalam mengklasifikasikan air sebagai layak atau tidak layak minum secara seimbang. Model lain seperti Decision Tree, XGBoost, dan SVM menunjukkan performa yang cukup kompetitif, tetapi masih berada di bawah Random Forest. Sementara itu, Logistic Regression menunjukkan performa paling rendah yaitu hanya 0.5150 pada seluruh metrik, menandakan kurang cocok untuk dataset ini.

### Hasil evaluasi setelah tuning

| Model                            | Accuracy   | Precision | Recall | F1-Score   |
| -------------------------------- | ---------- | --------- | ------ | ---------- |
| Random Forest  (Tuned)           | 0.7288     | 0.7288    | 0.7288 | 0.7287     |

Setelah dilakukan _hyperparameter tuning_ dengan GridSearchCV, performa Random Forest meningkat menjadi 0.7288 di semua metrik evaluasi kecuali F1-score. Meskipun peningkatannya tidak signifikan secara numerik, hal ini mengindikasikan bahwa proses _tuning_ tetap memberikan perbaikan kecil dan menjaga konsistensi performa model.

### Feature importance

![Distribusi Label](https://drive.google.com/uc?export=view&id=1kmkl3hBggoFutbsWiDOUHqydKgdI70Gx)

Hasil analisis _feature importance_ dari model Random Forest mengindikasikan bahwa pH, Sulfate, dan Hardness merupakan variabel yang paling signifikan dalam menentukan kelayakan air untuk dikonsumsi.

## Kesimpulan

Proyek ini berhasil menunjukkan bahwa pendekatan _machine learning_ berbasis variabel-variabel fisikokimia seperti pH, Sulfate, dan Hardness dapat digunakan secara efektif untuk memprediksi potabilitas air atau kelayakan air untuk dikonsumsi. Beberapa algoritma klasifikasi seperti Logistic Regression, Decision Tree, Random Forest, XGBoost, dan SVM telah diuji dalam eksperimen ini.

Hasil evaluasi menunjukkan bahwa model Random Forest memberikan performa terbaik dengan nilai  presisi, recall, dan F1-score tertinggi diantara model lain dan nilai akurasi tertinggi sebesar 0.7275, yang kemudian meningkat menjadi 0.7288 setelah dilakukan proses _hyperparameter tuning_ menggunakan GridSearchCV. Model ini mampu memberikan keseimbangan yang baik antara presisi dan recall, yang sangat penting dalam konteks klasifikasi air layak dan tidak layak minum untuk mencegah risiko kesehatan.

Dibandingkan model lain, Random Forest terbukti unggul dalam mengenali pola dari variabel-variabel penting air, sementara XGBoost dan SVM menunjukkan performa yang cukup baik namun masih berada di bawah Random Forest. Logistic Regression menjadi model dengan performa terendah dalam kasus ini.

Secara keseluruhan, penerapan _machine learning_ dalam klasifikasi potabilitas air menunjukkan potensi besar sebagai sistem pendukung pengambilan keputusan yang otomatis, efisien, dan andal untuk pemantauan kualitas air. Hal ini dapat berkontribusi secara signifikan dalam mendukung upaya menjaga kesehatan masyarakat dan pengelolaan sumber daya air yang berkelanjutan.

---

**Referensi:**

World Health Organization (WHO). (2017). _Guidelines for drinking-water quality: Fourth edition incorporating the first addendum_. World Health Organization.
https://www.who.int/publications/i/item/9789241549950

Gazzaz, N. M., Alomari, O. A., Khan, M. A., & Alotaibi, R. M.(2022). Water quality classification using machine learning algorithms. _Environmental Technology & Innovation_, 27, 102410.
https://www.sciencedirect.com/science/article/abs/pii/S2214714422003646

Shah, S. A., Singh, R., & Fatima, N. (2021). Prediction of quality of water according to a Random Forest classifier. _International Journal of Advanced Computer Science and Applications_, 13(6), 105–110.
https://thesai.org/Downloads/Volume13No6/Paper_105-Prediction_of_Quality_of_Water_According_to_a_Random_Forest_Classifier.pdf

Malakar, S., Banerjee, S., & Majumder, A. (2020). Efficient water quality prediction using supervised machine learning. _International Journal of Scientific & Technology Research_, 9(4), 1234–1238.
https://www.researchgate.net/publication/336808732_Efficient_Water_Quality_Prediction_Using_Supervised_Machine_Learning

