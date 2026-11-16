Abstrak

Diabetes melitus merupakan penyakit metabolik kronis yang deteksi dininya berperan penting dalam mencegah komplikasi. Repositori ini memuat implementasi klasifikasi biner status diabetes (positif/negatif) menggunakan algoritma Support Vector Machine (SVM) pada Pima Indians Diabetes Dataset yang terdiri atas 768 observasi, 8 atribut klinis, dan 1 label (Outcome). Tahapan penelitian meliputi pemuatan data, pemisahan atribut dan label, standarisasi fitur dengan Z-score (StandardScaler), pembagian data latih dan data uji dengan proporsi 67:33 (random_state=42), pelatihan model SVC dengan parameter bawaan (kernel RBF), serta evaluasi menggunakan akurasi. Model memperoleh akurasi sebesar 0,7559 (75,59%) pada data uji. Dokumen ini menguraikan latar belakang, landasan teori, metodologi, cara menjalankan program, hasil, keterbatasan, dan saran pengembangan.

Kata kunci: diabetes, klasifikasi, support vector machine, standarisasi, Pima Indians Diabetes, scikit-learn.

Daftar Isi
Pendahuluan
Tinjauan Pustaka
Dataset
Metodologi
Implementasi dan Cara Menjalankan
Hasil dan Pembahasan
Keterbatasan dan Saran Pengembangan
Kesimpulan
Referensi
Penulis dan Lisensi
1. Pendahuluan
1.1 Latar Belakang

Diabetes melitus ditandai dengan kadar glukosa darah yang tinggi (hiperglikemia) akibat gangguan sekresi insulin, kerja insulin, atau keduanya. International Diabetes Federation (2021) melaporkan bahwa sekitar 537 juta orang dewasa di dunia hidup dengan diabetes pada tahun 2021. Deteksi dini penting karena penyakit ini kerap tidak disadari hingga komplikasi muncul.

Pendekatan machine learning banyak dimanfaatkan untuk membantu klasifikasi pasien diabetes. Purbolaksono et al. (2021) membandingkan SVM dan Modified Balanced Random Forest untuk mengklasifikasikan data pasien diabetes dan melaporkan performa maksimum SVM sebesar 87,94%. Rahmawati et al. (2024) meneliti prediksi diabetes pada data pasien rumah sakit umum daerah di Indonesia dan menemukan bahwa SVM yang dikombinasikan dengan SMOTE menghasilkan akurasi tertinggi, yaitu 95,8%. Kedua penelitian tersebut menunjukkan bahwa SVM merupakan kandidat algoritma yang layak untuk klasifikasi diabetes.

SVM bekerja berdasarkan jarak antardata sehingga peka terhadap perbedaan skala antarfitur. Karena itu, praktik umum pada klasifikasi diabetes adalah menormalisasi atau menstandarisasi fitur sebelum pelatihan (Kurniawan & Megawaty, 2025; Pratama & Salam, 2025). Repositori ini menerapkan alur sederhana tersebut: standarisasi Z-score diikuti klasifikasi SVM.

1.2 Rumusan Masalah
Bagaimana menerapkan algoritma SVM untuk mengklasifikasikan status diabetes pada Pima Indians Diabetes Dataset?
Seberapa besar akurasi model SVM berparameter bawaan setelah fitur distandarisasi?
1.3 Tujuan
Membangun model klasifikasi diabetes berbasis SVM menggunakan pustaka scikit-learn.
Mengukur kinerja model pada data uji menggunakan metrik akurasi.
1.4 Manfaat

Repositori ini dapat menjadi acuan pembelajaran bagi pemula mengenai alur dasar klasifikasi: pemuatan data, preprocessing, pembagian data, pelatihan, dan evaluasi.

Catatan: Program ini bertujuan edukatif dan penelitian. Hasilnya bukan alat diagnosis medis.

2. Tinjauan Pustaka
2.1 Diabetes Melitus

Diabetes melitus adalah penyakit metabolik kronis dengan karakteristik hiperglikemia. Pada dataset yang digunakan, indikator klinis seperti kadar glukosa plasma, indeks massa tubuh (BMI), tekanan darah, dan riwayat keluarga dicatat sebagai atribut prediktor (Smith et al., 1988).

2.2 Support Vector Machine (SVM)

SVM diperkenalkan oleh Cortes dan Vapnik (1995) sebagai metode klasifikasi yang mencari hyperplane pemisah dengan margin maksimum antara dua kelas. Untuk label $y_i \in \{-1, +1\}$, SVM dengan margin lunak (soft margin) menyelesaikan:

$$\min_{w,b,\xi} \; \frac{1}{2}\lVert w \rVert^2 + C \sum_{i=1}^{n} \xi_i
\quad \text{dengan syarat} \quad
y_i (w \cdot \phi(x_i) + b) \ge 1 - \xi_i,\; \xi_i \ge 0$$

Parameter $C$ mengatur kompromi antara lebar margin dan kesalahan klasifikasi pada data latih. Pada data yang tidak dapat dipisahkan secara linear, SVM memetakan data ke ruang berdimensi lebih tinggi melalui fungsi kernel.

2.3 Kernel Radial Basis Function (RBF)

Kelas SVC pada scikit-learn menggunakan kernel RBF secara bawaan:

$$K(x, x') = \exp\left(-\gamma \lVert x - x' \rVert^2\right)$$

Nilai $\gamma$ menentukan seberapa jauh pengaruh satu data latih. Pada pengaturan bawaan, $C = 1{,}0$ dan gamma='scale', yaitu $\gamma = 1 / (n_{fitur} \cdot \mathrm{Var}(X))$ (Pedregosa et al., 2011). Hsu et al. (2003) menyarankan kernel RBF sebagai pilihan awal yang wajar serta menyarankan penskalaan data sebelum pelatihan SVM.

2.4 Standarisasi Z-score

Standarisasi mengubah setiap fitur sehingga memiliki rerata 0 dan simpangan baku 1:

$$z = \frac{x - \mu}{\sigma}$$

Pratama dan Salam (2025) menunjukkan bahwa atribut pada Pima Indians Diabetes Dataset memiliki rentang nilai yang sangat berbeda (misalnya Insulin hingga 846 sedangkan DiabetesPedigreeFunction di bawah 2,5), sehingga fitur berskala besar cenderung mendominasi perhitungan berbasis jarak. Kurniawan dan Megawaty (2025) menerapkan standarisasi Z-score pada fitur numerik karena SVM dan KNN peka terhadap perbedaan magnitudo.

2.5 Metrik Evaluasi

Akurasi dihitung sebagai proporsi prediksi yang benar terhadap seluruh data uji:

$$\text{Akurasi} = \frac{TP + TN}{TP + TN + FP + FN}$$

Metode clf.score(X_test, y_test) pada klasifikator scikit-learn mengembalikan nilai akurasi rerata pada data uji.

2.6 Penelitian Terkait
Peneliti	Fokus Penelitian	Temuan Utama
Purbolaksono et al. (2021)	Perbandingan SVM dan Modified Balanced Random Forest pada data pasien diabetes	Performa maksimum 87,94% (SVM) dan 97,8% (MBRF)
Rahmawati et al. (2024)	SVM, Logistic Regression, dan Naïve Bayes dengan SMOTE/ADASYN pada 657 data pasien RSUD	SVM + SMOTE mencapai akurasi 95,8% dan AUC 99,1
Pratama & Salam (2025)	Perbandingan teknik normalisasi pada KNN untuk Pima Indians Diabetes	Min-Max Scaling terbaik dengan akurasi 0,8117; Z-score juga memberi hasil baik
Kurniawan & Megawaty (2025)	LR, RF, SVM, dan KNN pada 100.000 data diabetes (Kaggle)	Random Forest terbaik (96,88%); SVM 89,59% dengan waktu latih paling lama
3. Dataset

Dataset yang digunakan adalah Pima Indians Diabetes Dataset (berkas diabetes.csv), yang berasal dari National Institute of Diabetes and Digestive and Kidney Diseases dan dapat diunduh dari Kaggle atau UCI Machine Learning Repository. Seluruh pasien berjenis kelamin perempuan, berusia minimal 21 tahun, dan berasal dari keturunan Pima Indian (Smith et al., 1988; Pratama & Salam, 2025).

Berdasarkan keluaran df.info() pada notebook, dataset terdiri atas 768 baris dan 9 kolom tanpa nilai kosong (non-null).

No	Atribut	Tipe Data	Deskripsi
1	Pregnancies	int64	Jumlah kehamilan
2	Glucose	int64	Konsentrasi glukosa plasma
3	BloodPressure	int64	Tekanan darah diastolik (mm Hg)
4	SkinThickness	int64	Ketebalan lipatan kulit trisep (mm)
5	Insulin	int64	Kadar insulin serum
6	BMI	float64	Indeks massa tubuh
7	DiabetesPedigreeFunction	float64	Skor riwayat diabetes keluarga
8	Age	int64	Usia (tahun)
9	Outcome	int64	Label: 1 = diabetes, 0 = tidak diabetes

Kedelapan atribut pertama berperan sebagai fitur (X) dan kolom Outcome sebagai target (y). Menurut dokumentasi dataset, label terdiri atas 500 observasi kelas 0 dan 268 observasi kelas 1, sehingga distribusi kelas tidak seimbang.

4. Metodologi
4.1 Alur Penelitian
Muat diabetes.csv
Inspeksi datahead dan info
Pisahkan fitur Xdan label y
StandarisasiStandardScaler
Split data67% latih : 33% uji
Latih modelSVC
Evaluasiakurasi
4.2 Tahapan dan Kode

Setiap tahapan di bawah ini identik dengan sel pada notebook SVM_Klasifikasi_DIabetes.ipynb.

Tahap 1 — Pemuatan data. Berkas CSV dimuat ke dalam DataFrame pandas.

python
import pandas as pd
df = pd.read_csv('diabetes.csv')

Tahap 2 — Inspeksi data. Lima baris pertama ditampilkan, kemudian struktur data diperiksa (jumlah baris, kolom, tipe data, dan nilai kosong).

python
df.head()
df.info()

Tahap 3 — Pemisahan fitur dan label. Delapan kolom pertama menjadi fitur, kolom Outcome menjadi label.

python
# memisahkan atribut pada dataset dan menyimpannya pada sebuah variabel
X = df[df.columns[:8]]
# memisahkan label pada dataset dan menyimpannya pada sebuah variabel
y = df['Outcome']

Tahap 4 — Standarisasi fitur. StandardScaler mengubah setiap fitur menjadi berrerata 0 dan bersimpangan baku 1 (lihat Subbab 2.4).

python
from sklearn.preprocessing import StandardScaler

# standarisasi nilai-nilai dari dataset
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

Tahap 5 — Pembagian data latih dan data uji. Sebanyak 33% data dialokasikan sebagai data uji, dengan random_state=42 agar hasil dapat direproduksi. Dari 768 data, diperoleh sekitar 514 data latih dan 254 data uji.

python
from sklearn.model_selection import train_test_split

# memisahkan data untuk training dan testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

Tahap 6 — Pelatihan model. Objek SVC dibuat dengan parameter bawaan (kernel RBF, $C = 1{,}0$, gamma='scale') lalu dilatih menggunakan data latih.

python
from sklearn.svm import SVC

# membuat objek SVC dan memanggil fungsi fit untuk melatih model
clf = SVC()
clf.fit(X_train, y_train)

Tahap 7 — Evaluasi. Akurasi dihitung pada data uji.

python
# Menampilkan skor akurasi prediksi
clf.score(X_test, y_test)
5. Implementasi dan Cara Menjalankan
5.1 Struktur Repositori
text
.
├── SVM_Klasifikasi_DIabetes.ipynb   # notebook utama
├── diabetes.csv                     # dataset (unduh terpisah, lihat Bab 3)
└── README.md
5.2 Prasyarat
Python 3.x
Pustaka: pandas, scikit-learn, dan jupyter (atau Jupyter Lab / Google Colab)
5.3 Langkah Menjalankan
Klon repositori
bash
   git clone https://github.com/<username>/<nama-repositori>.git
   cd <nama-repositori>
(Opsional) Buat virtual environment
bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
Pasang dependensi
bash
   pip install pandas scikit-learn jupyter
Siapkan dataset. Unduh diabetes.csv (lihat Bab 3) dan letakkan satu folder dengan notebook, karena kode membaca berkas dengan jalur relatif 'diabetes.csv'.
Jalankan notebook
bash
   jupyter notebook SVM_Klasifikasi_DIabetes.ipynb

Jalankan seluruh sel secara berurutan (Run All).

Alternatif Google Colab: unggah notebook dan diabetes.csv ke sesi Colab, lalu jalankan semua sel. pandas dan scikit-learn sudah tersedia secara bawaan.

6. Hasil dan Pembahasan
6.1 Hasil Eksplorasi Data

Lima baris pertama data (df.head()):

Pregnancies	Glucose	BloodPressure	SkinThickness	Insulin	BMI	DiabetesPedigreeFunction	Age	Outcome
6	148	72	35	0	33.6	0.627	50	1
1	85	66	29	0	26.6	0.351	31	0
8	183	64	0	0	23.3	0.672	32	1
1	89	66	23	94	28.1	0.167	21	0
0	137	40	35	168	43.1	2.288	33	1

Keluaran df.info() menunjukkan 768 entri (indeks 0–767) dengan 9 kolom, seluruhnya berstatus 768 non-null, tujuh kolom bertipe int64 dan dua kolom (BMI, DiabetesPedigreeFunction) bertipe float64.

6.2 Hasil Klasifikasi
Komponen	Nilai
Algoritma	SVC() (kernel RBF, $C=1{,}0$, gamma='scale')
Pra-pemrosesan	StandardScaler (Z-score)
Pembagian data	67% latih : 33% uji, random_state=42
Akurasi data uji	0,7559 (75,59%)

Nilai akurasi 0,7559055… setara dengan 192 prediksi benar dari 254 data uji.

6.3 Pembahasan
Interpretasi. Sekitar tiga dari empat pasien pada data uji diklasifikasikan dengan benar. Karena kelas negatif mendominasi dataset (±65%), klasifikator naif yang selalu memprediksi "tidak diabetes" akan memperoleh akurasi sekitar 65%. Dengan demikian, model ini memberikan peningkatan yang nyata di atas garis dasar tersebut, namun belum tinggi.
Perbandingan dengan literatur. Akurasi ini lebih rendah daripada performa SVM maksimum pada Purbolaksono et al. (2021) dan hasil SVM + SMOTE pada Rahmawati et al. (2024), serta sedikit di bawah akurasi KNN dengan Z-score pada Pratama dan Salam (2025) sebesar 0,7922 (K = 7). Perbandingan ini bersifat indikatif karena perbedaan dataset, proporsi pembagian data, teknik preprocessing, pemilihan fitur, dan penyetelan parameter.
Peran standarisasi. Notebook ini tidak membandingkan model dengan dan tanpa standarisasi, sehingga kontribusi langsung standarisasi terhadap akurasi tidak dapat dikuantifikasi dari kode yang ada. Landasan penggunaannya bersumber dari literatur (Hsu et al., 2003; Pratama & Salam, 2025).
Parameter bawaan. Nilai $C$, $\gamma$, dan jenis kernel belum disetel, sehingga terdapat ruang peningkatan kinerja melalui optimasi hiperparameter.
7. Keterbatasan dan Saran Pengembangan
7.1 Keterbatasan Implementasi Saat Ini
Potensi kebocoran data (data leakage). StandardScaler dilatih (fit) pada seluruh data sebelum pembagian latih-uji, sehingga rerata dan simpangan baku data uji ikut memengaruhi proses standarisasi. Secara metodologis, scaler sebaiknya dilatih hanya pada data latih. Praktik serupa ditekankan pada penelitian terkait, misalnya SMOTE yang hanya diterapkan pada data latih setelah pembagian data (Kurniawan & Megawaty, 2025).
Nilai nol yang tidak wajar. Pada dataset ini, atribut seperti Glucose, BloodPressure, SkinThickness, Insulin, dan BMI memiliki nilai minimum 0 (Pratama & Salam, 2025), yang secara klinis tidak masuk akal dan umumnya merepresentasikan data hilang. Contohnya terlihat pada kolom Insulin dan SkinThickness di Subbab 6.1. Kode belum menangani hal ini.
Satu metrik evaluasi. Hanya akurasi yang dilaporkan, padahal kelas tidak seimbang. Presisi, recall, F1-score, matriks konfusi, dan AUC diperlukan untuk menilai kemampuan mendeteksi kasus positif.
Satu kali pembagian data. Hasil bergantung pada satu random_state tanpa validasi silang, dan pembagian tidak menggunakan stratifikasi kelas.
Cakupan populasi. Dataset hanya mencakup perempuan keturunan Pima Indian berusia ≥ 21 tahun, sehingga generalisasi ke populasi lain perlu kehati-hatian.
7.2 Saran Pengembangan
Menggunakan Pipeline (scaler + SVC) agar standarisasi hanya mempelajari data latih.
Melakukan penyetelan hiperparameter ($C$, $\gamma$, kernel) dengan GridSearchCV dan validasi silang berstrata (StratifiedKFold).
Menangani nilai nol yang tidak wajar melalui imputasi.
Mengatasi ketidakseimbangan kelas, misalnya dengan SMOTE atau ADASYN pada data latih (Rahmawati et al., 2024).
Menambahkan metrik presisi, recall, F1-score, matriks konfusi, dan kurva ROC-AUC.
Membandingkan SVM dengan algoritma lain seperti KNN, Logistic Regression, dan Random Forest (Kurniawan & Megawaty, 2025).
8. Kesimpulan

Penelitian ini menerapkan algoritma SVM (SVC dengan parameter bawaan) pada Pima Indians Diabetes Dataset setelah fitur distandarisasi menggunakan Z-score. Dengan pembagian data 67:33, model mencapai akurasi 75,59% pada data uji. Hasil ini menunjukkan bahwa alur sederhana SVM dan standarisasi mampu mengklasifikasikan status diabetes di atas garis dasar proporsi kelas mayoritas, tetapi masih dapat ditingkatkan melalui perbaikan prosedur preprocessing (menghindari kebocoran data dan menangani nilai nol), penyetelan hiperparameter, penanganan ketidakseimbangan kelas, serta evaluasi dengan metrik yang lebih lengkap.

9. Referensi
9.1 Jurnal Nasional Terakreditasi SINTA
No	Referensi	Akreditasi
1	Pratama, Y. D., & Salam, A. (2025). Comparison of data normalization techniques on KNN classification performance for Pima Indians Diabetes dataset. Journal of Applied Informatics and Computing, 9(3), 693–706. https://doi.org/10.30871/jaic.v9i3.9353	SINTA 3
2	Kurniawan, M. F., & Megawaty, D. A. (2025). Comparison of logistic regression, random forest, support vector machine (SVM) and K-nearest neighbor (KNN) algorithms in diabetes prediction. Journal of Applied Informatics and Computing, 9(5), 2154–2162. https://jurnal.polibatam.ac.id/index.php/JAIC/article/download/9815/3041	SINTA 3
3	Purbolaksono, M. D., Tantowi, M. I., Hidayat, A. I., & Adiwijaya. (2021). Perbandingan Support Vector Machine dan Modified Balanced Random Forest dalam deteksi pasien penyakit diabetes. Jurnal RESTI (Rekayasa Sistem dan Teknologi Informasi), 5(2), 393–399. https://doi.org/10.29207/resti.v5i2.3008	SINTA 2
4	Rahmawati, S., Wibowo, A., & Masruriyah, A. F. N. (2024). Improving diabetes prediction accuracy in Indonesia: A comparative analysis of SVM, logistic regression, and naive Bayes with SMOTE and ADASYN. Jurnal RESTI (Rekayasa Sistem dan Teknologi Informasi), 8(5), 607–614. https://doi.org/10.29207/resti.v8i5.5980	SINTA 2

Status akreditasi mengacu pada informasi resmi pengelola jurnal (JAIC: SK Dirjen No. 79/E/KPT/2023; RESTI: SK Dirjen No. 158/E/KPT/2021). Periksa status terkini pada SINTA sebelum penggunaan formal.

9.2 Referensi Internasional dan Dokumentasi
Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine Learning, 20(3), 273–297. https://doi.org/10.1007/BF00994018
Hsu, C.-W., Chang, C.-C., & Lin, C.-J. (2003). A practical guide to support vector classification. Department of Computer Science, National Taiwan University.
International Diabetes Federation. (2021). IDF Diabetes Atlas (10th ed.). https://diabetesatlas.org
Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., … Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.
Smith, J. W., Everhart, J. E., Dickson, W. C., Knowler, W. C., & Johannes, R. S. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. Proceedings of the Annual Symposium on Computer Application in Medical Care, 261–265.
