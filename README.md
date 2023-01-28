# Prediksi Harga Tiket Pesawat di India - Abdul Aziz Munawar

## Domain Proyek

**Latar Belakang**  
Pesawat merupakan salah satu moda transportasi yang dapat digunakan untuk bepergian, baik untuk urusan bisnis, liburan maupun urusan lainnya. Untuk dapat menggunakan jasa moda transportasi ini, seseorang harus memiliki tiket pesawat.

Harga tiket yang terlalu mahal akan membuat seseorang berpikir berkali-kali untuk menggunakan jasa moda transportasi pesawat. Sebaliknya, harga tiket pesawat yang terlalu murah, akan menyebabkan perusahaan maskapai penerbangan tidak dapat memaksimalkan keuntungan bagi perusahaan, bahkan jika tidak ditangani secara serius, biaya operasional tidak akan sebanding dengan keuntungan yang didapatkan, hal ini akan menyebabkan kerugian bagi perusahaan maskapai penerbangan.

**Oleh sebab itu, diperlukan suatu aplikasi yang dapat memprediksi secara akurat, harga tiket pesawat yang ideal untuk ditawarkan kepada pelanggan, sehingga harga tiket akan seimbang (tidak terlalu mahal, juga tidak terlalu murah).**

Pada kasus ini, aplikasi *machine learning* secara spesifik akan memprediksi harga ideal untuk tiket, sehingga dapat dijadikan acuan oleh maskapai penerbangan di India, dalam menetapkan keputusan harga tiket bagi calon pengguna jasa layanan pesawatnya.

**Alasan Penting Yang Mendasari Proyek Ini**:
- Alasan penting yang mendasari bahwa permasalahan harga tiket harus diselesaikan, yaitu sebagai berikut:
    - harga tiket yang tidak terlalu mahal dapat menyebabkan seseorang untuk berpikir beberapa kali untuk menggunakan jasa moda transportasi melalui pesawat yang disediakan perusahaan x.
    - harga tiket yang terlalu murah dapat menyebabkan keuntungan perusahaan x tidak optimal, bahkan dapat menyebabkan kerugian. Alasannya karena, biaya operasional tidak sebanding dengan keuntungan yang didapatkan.
    - untuk menyelesaikan permasalahan tersebut, maka akan dibuat aplikasi yang dapat memprediksi harga ideal untuk tiket pesawat penerbangan (dalam kasus ini, prediksi spesifik hanya akan menampilkan harga ideal pesawat maskapai penerbangan di India).
    - aplikasi ini akan memanfaatkan teknologi machine learning serta bahasa pemrograman Python dalam membuat prediksi harga ideal untuk menjadi bahan keputusan bagi maskapai penerbangan dalam menentukan harga tiket pesawat.
- Hasil riset terkait:
    [Predicting Flight Prices in India](https://www.researchgate.net/profile/Tarun-Devireddy/publication/337821411_Predicting_Flight_Prices_in_India/links/5debfba992851c83646b669a/Predicting-Flight-Prices-in-India.pdf)
    [Understanding Customer Perception While Booking Flight Tickets](http://www.solidstatetechnology.us/index.php/JSST/article/view/5721)
    [Airline Fare Prediction Using Machine Learning Algorithms](https://ieeexplore.ieee.org/abstract/document/9716563)

## Business Understanding
Maskapai penerbangan x merupakan salah satu perusahaan maskapai yang menyediakan moda transportasi pesawat udara di Negara India. Untuk memaksimalkan keuntungan perusahaan, maka perusahaan harus mengetahui harga tiket ideal untuk diterapkan pada layanan jasa penerbangannya.

Harga tiket ideal adalah harga yang tidak terlalu mahal maupun tidak terlalu murah. Jika harga tiket terlalu mahal, maka akan membuat calon pengguna layanan berpikir beberapa kali untuk menggunakan layanan penerbangan di maskapai x. Namun jika terlalu murah, maka keuntungan perusahaan tidak akan maksimal, bahkan akan menderita kerugian, karena biaya operasional tidak sebanding dengan pendapatan. 

Oleh sebab itu, maka perlu dibuat aplikasi yang dapat memprediksi harga ideal tiket pesawat di India, untuk menjadi bahan pengambilan keputusan bagi pimpinan dalam menentukan harga layanan.

Dengan menerapkan harga tiket yang ideal, maka pengguna layanan akan merasa senang dan berpotensi untuk menambah pelanggan yang ingin menggunakan layanan penerbangan melalui maskapai penerbangan x.

### Problem Statements

Berdasarkan penjelasan yang telah disampaikan sebelumnya, maka problem statements (rumusan masalah), yaitu sebagai berikut:
- Apa faktor-faktor yang dapat mempengaruhi harga tiket pesawat?  
- Berapa harga ideal tiket pesawat untuk diterapkan di maskapai penerbangan x?

### Goals

Tujuan yang ingin dicapai dari pembuatan aplikasi prediksi harga tiket pesawat di India ini, yaitu sebagai berikut:
- Mengetahui faktor-faktor yang mempengaruhi harga tiket pesawat?
- Membuat aplikasi yang dapat memprediksi harga tiket pesawat secara akurat, sebagai bahan pengambilan keputusan dalam penerapan harga tiket ideal untuk diterapkan dimaskapai penerbangan x.

    ### Solution statements
    - Solusi yang dapat dilakukan untuk menangani permasalahan sebagaimana terdapat dalam problem statements, yaitu dengan membuat aplikasi prediksi harga tiket pesawat. Adapun aplikasi tersebut dibuat dengan menerapkan teknologi machine learning serta bahasa pemrograman python.
    - Algoritma machine learning yang akan digunakan, yaitu Random Forest dan Boosting Algorithm.
    - Untuk mengukur keakuratan/keidealan prediksi harga tiket pesawat yang dilakukan oleh aplikasi yang dibuat, maka metrik yang digunakan adalah Mean Squared Error (MSE). 
    - Mean Squared Error (MSE) adalah mengukur harga hasil akurasi prediksi terhadap hargaa ideal.
    - Mengajukan 2 atau lebih solution statement. Misalnya, menggunakan dua atau lebih algoritma untuk mencapai solusi yang diinginkan atau melakukan improvement pada baseline model dengan hyperparameter tuning.
    - Solusi yang diberikan harus dapat terukur dengan metrik evaluasi.

## Data Understanding
Data yang digunakan adalah dataset yang bersumber dari situs Kaggle yang berisi dataset terkait tiket pesawat di maskapai penerbangan India. Dataset sebagaimana dimaksud dapat didownload pada link berikut ini: [Flight Price Prediction](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction)

### Variabel-variabel yang terdapat dalam dataset Flight Price Prediction:
- airline = Nama maskapai penerbangan.
- source_city = Kota awal pemberangkatan.
- departure_time = Waktu pemberangkatan.
- stops = jumlah transit selama perjalanan.
- arrival_time = Waktu tiba di kota tujuan.
- destination_city = Kota tujuan penerbangan.
- class = Kelas penerbangan.
- duration = Waktu yang dibutuhkan untuk tiba di kota tujuan.
- days_left = Jarak hari pemesanan tiket dengan hari penerbangan.
- price = harga tiket.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Untuk memahami dataset, langkah-langkah yang dilakukan, yaitu sebagai berikut:
    - Melakukan load dataset kedalam google colaboratory.
    - Melakukan Exploratory data analysis untuk memahami makna-makna variabel yang terdapat dalam dataset.
    - menggunakan teknik visualisasi data kategorik dan non kategorik dengan menggunakan library seaborn.
    - Memvisualisasikan data dengan menggunakan boxplot untuk mencari outlier.
    - Menggunakan IQR (Interquartile Range) untuk mengeliminasi outlier.
    - Melakukan univariative analysis untuk memahami sebaran data variabel.
    - Melakukan multivariative analysis untuk memahami korelasi variabel kategorikal dan numberikak terhadap variabel price. 

## *Data Preparation*
Teknik *data preparation* yang dilakukan, yaitu sebagai berikut:
1. Mengubah dataset *flight price prediction* menjadi *dataframe* dengan menggunakan *pandas*.
2. Analisis awal untuk membuang variabel yang sangat tidak relevan untuk prediksi data.
3. Melakukan *exploratory data analysis* untuk memahami variabel-variabel yang terdapat dalam dataset.
4. Memvisualisasikan data dengan menggunakan *boxplot* untuk mencari *outlier*.
5. Menggunakan IQR *(Interquartile Range)* untuk mengeliminasi outlier.
6. Melakukan *univariative analysis* untuk memahami sebaran data variabel.
7. Melakukan *multivariative analysis* untuk memahami korelasi variabel kategorikal dan numberikal terhadap variabel *price*.  
8. Membuat *correlation matrix* untuk fitur numerik.
9. Mengeliminasi variabel numerik yang memiliki korelasi rendah terhadap variabel *price*.

**Proses *Data Preparation***: 
- Proses data preparation dilakukan melalui langkah-langkah, yaitu sebagai berikut: Melakukan *load* data pada *google colaboratory*, kemudian melakukan analisis awal terkait variabel yang sangat tidak relevan untuk diproses lebih lanjut. Selanjutnya, memahami makna-makna variabel dengan menerapkan *Exploratory Data Analysis*, kemudian melakukan visualisasi data untuk mencari outlier dengan menggunakan *boxplot* dari *library seaborn*. Selanjutnya, menerapkan metode IQR untuk mengeliminasi outlier, kemudian menggunakan *univariative analysis* serta *multivariative analysis*. Selanjutnya membuat *correlation matrix*, kemudian membuang variabel numberik yang memiliki korelasi rendah terhadap variabel *price.*
- Data preparation diperlukan agar data yang akan diproses oleh algoritma *machine learning* bebas dari *outlier* dan variabel-variabel yang digunakan untuk algoritma adalah variabel yang memiliki korelasi tinggi terhadap penentuan prediksi harga tiket pesawat.
- Pembuatan aplikasi ini menggunakan IQR *(Interquartile Range)* untuk mengeliminasi *outlier* yang terdapat dalam dataset *flight price prediction*.

## Modeling
- Model *machine learning* yang digunakan adalah *random forest* dan *boosting algorithm*.
- *Random forest* adalah model *machine learning* yang menerapkan gabungan model *machine learning decision tree*.
- Untuk menggunakan model *Random Forest* menggunakan *function RandomForestRegressor* yang merupakan bagian dari *library sklearn.ensemble*.
- Parameter yang digunakan dalam model *Random Forest*, yaitu sebagai berikut:
    - *n\_estimators* = jumlah pohon keputusan *(decision tree)* yang akan dibuat pada model *Random Forest* yang digunakan. Pada model ini n_estimators yang di buat, yaitu 100.
    - *max_depth* = maksimal kedalaman dari decision tree yang akan dibuat. Pada model Random Forest ini kedalaman yang dibuat sampai 64 level.
    - *max_features* = feature maksimal yang digunakan ketika melakukan split. Parameter yang di input adalah sqrt.
    - *random_state* = mengatur status random dari model *Random Forest*. Pada model ini *random state* yang digunakan adalah 1.
    - *n\_jobs* = mengatur sistem paralel dan penggunaan prosesor dalam proses decision tree. Pada model *Random Forest* ini, paramater di isi -1 agar seluruh prosesor digunakan.
    - *verbose* = mengatur tampilan pada saat proses *training*. Nilai yang di input adalah 2, sehingga hasil dari *training* ditampilkan setiap langkah.
    - *warm_start* = mengatur apakah *weight* hasil pelatihan sebelumnya akan digunakan lagi pada *training* baru atau tidak. Nilai yang di input adalah *True*.
    - *RF.fit(X_train, y_train)* = menentukan data yang akan digunakan pada proses *training* model *Random Forest*.
    - *models.loc* = mengakses kolom dan baris dari *dataframe* yang digunakan untuk proses *training* model *Random Forest*
    - *mean_absolute_error* = metrik yang digunakan untuk mengukur akurasi model yang telah dilatih.
 - Demi mendapatkan hasil yang terbaik, selain menggunakan random forest, digunakan algoritma lain, yaitu ***boosting algorithm*** sebagai algoritma pembanding untuk mengukur manakah algoritma yang lebih baik diantara keduanya dalam menghasilkan prediksi harga tiket pesawat.
 - *Boosting algorithm* adalah salah satu algoritma *ensemble learning* yang proses laltihannya dilakukan secara sekuensial serta iteratif.
 -  Parameter yang digunakan dalam *boosting algorithm*, yaitu sebagai berikut:
	 - *AdaBoostRegressor* = function yang digunakan untuk melakukan proses training model dengan menggunakan *Boosting Algorithm*. Function ini berada pada library / modul *sklearn.ensemble*.
	 - *learning_rate*= parameter yang digunakan untuk mengatur proses training dari algoritma ini. Pada model ini, paramater di isi 1.0.
	 - *random_state* = mengatur status random dari model *Boosting Algorithm*. Pada model ini *random state* yang digunakan adalah 1.
	 - *boosting.fit(X_train, y_train)* = load data yang akan digunakan dalam *training* model *Boosting Algorithm*.
	 - *models.loc['train_mse','Boosting']* = mengakses kolom dan baris dari *dataframe* yang digunakan untuk proses *training* model *Boosting Algorithm*.
	 - *mean_absolute_error* = metrik yang digunakan untuk mengukur akurasi model yang telah dilatih.
	 
**Kelebihan dan Kekurangan Random Forest dan Boosting Algorithm** 
- Setelah melakukan training menggunakan *Random Forest* dan *Boosting Algorithm*, maka dapat disimpulkan kelebihan dan kekurangan masing-masing, yaitu sebagai berikut:
- Kelebihan algoritma *Random Forest*, yaitu dapat hasil prediksi masih andal meskipun ada *noise* maupun *missing value* pada dataset yang digunakan.
- Kekurangan *Random Forest* , yaitu untuk mendapatkan prediksi yang akurat, *tuning* parameter harus dilakukan secara tepat.
- Kelebihan *Boosting Algorithm*, yaitu memori untuk proses latihan model relatif lebih kecil dibandingkan dengan Random Forest. 
- Kekurangan *Boosting Algorithm*, yaitu sensitif pada *noise* dan *missing value*, apabila dibandingkan dengan *Random Forest*.
- Berdasarkan hasil training model, maka ditetapkan bahwa algoritma yang terbaik diantara *Random Forest* dan *Boosting Algorithm* dalam memprediksi harga tiket, yaitu algoritma *Random Forest*.
- Alasannya, karena nilai *Mean Absolute Error (MAE)* yang dihasilkan *Random Forest* lebih baik dari *Boosting Algorithm*.

## *Evaluation*
Metrik yang digunakan untuk mengukur hasil *training* adalah *mean absolute error (MAE)*. Berdasarkan hasil training, bahwa model *Random Forest* menghasilan nilai MAE pada saat *training* = 2009.8582782434323 dan pada saat tes = 2218.153604433864. Ketika dianalisis, nilai MAE tersebut kurang dari 10%, sehingga model sudah dapat dikatakan menghasilkan nilai yang baik *(good fit)*.

**Cara Kerja Metrik Mean Absolute Error**: 
- *Mean Absolute Error* adalah metrik statistik yang digunakan untuk mengukur keakuratan dari prediksi nilai yang bersifat kontinyu.
- Semakin kecil nilai MAE, maka akan semakin baik pula model tersebut dalam melakukan prediksi nilai.