# Perbandingan RNN dan LSTM untuk Peramalan Suhu Per Jam (Jena Climate)

Proyek ini membandingkan dua arsitektur jaringan saraf rekuren, **SimpleRNN** dan **LSTM**, untuk meramalkan suhu udara satu jam ke depan memakai data cuaca multivariat Jena Climate (2009–2016). Hasil kedua model dibandingkan dengan baseline *naive* dan diuji pada beberapa panjang jendela (window) serta beberapa seed acak.

## Pertanyaan Penelitian

1. Apakah RNN dan LSTM menghasilkan peramalan suhu per jam yang lebih akurat daripada baseline naive?
2. Apakah LSTM lebih akurat daripada SimpleRNN, dan bagaimana pengaruh panjang window (12, 24, 48 jam) terhadap selisih keduanya?

## Dataset

- **Sumber:** Jena Climate, data stasiun cuaca Max Planck Institute for Biogeochemistry, Jena, Jerman. Tersedia di Kaggle: <https://www.kaggle.com/datasets/mnassrib/jena-climate>
- **Ukuran:** 420.551 baris, 15 kolom, dicatat setiap 10 menit (1 Januari 2009 sampai 1 Januari 2017).
- **Variabel:** tekanan (`p`), suhu (`T`), suhu potensial (`Tpot`), titik embun (`Tdew`), kelembapan relatif (`rh`), tekanan uap (`VPmax`, `VPact`, `VPdef`), kelembapan spesifik (`sh`), konsentrasi uap air (`H2OC`), kerapatan udara (`rho`), kecepatan angin (`wv`, `max. wv`), dan arah angin (`wd`).
- **Variabel target:** `T (degC)` pada satu jam setelah jendela input.
- File data **tidak disertakan** di repositori ini. Unduh dari tautan di atas dan letakkan di folder `data/`. Periksa lisensi pada halaman dataset sebelum menggunakannya kembali.

## Metodologi

1. **Pemeriksaan data (EDA).** Ukuran data, tipe, statistik deskriptif, distribusi, boxplot, korelasi, pola harian, bulanan, dan tahunan.
2. **Pembersihan data.**
   - Timestamp ganda (327 baris) dibuang.
   - `Date Time` diubah ke datetime, diurutkan, dan dijadikan index.
   - Nilai `-9999` pada `wv` (18 baris) dan `max. wv` (20 baris) adalah penanda data salah. Nilai itu diganti `NaN` lalu diinterpolasi berdasarkan waktu.
   - **Nilai ekstrem tidak dibuang atau di-clip.** Suhu −23 °C sampai 37 °C adalah kejadian cuaca nyata, dan metode IQR akan memotong data musiman yang valid.
3. **Rekayasa fitur.** Arah angin diubah menjadi komponen vektor `Wx` dan `Wy` agar sudut 0° dan 360° tidak diperlakukan berjauhan. Kolom `wv`, `max. wv`, dan `wd` kemudian dibuang. Hasil akhir: 13 fitur.
4. **Agregasi per jam.** Data 10 menit dirata-ratakan menjadi 70.129 jam.
5. **Pembagian data berdasarkan waktu** (tanpa pengacakan): train 70% (2009-01-01 sampai 2014-08-08), validasi 15% (sampai 2015-10-20), uji 15% (sampai 2017-01-01).
6. **Standarisasi** memakai rata-rata dan simpangan baku data train saja, untuk mencegah kebocoran data.
7. **Windowing.** Input berupa 24 jam terakhir (13 fitur), label berupa suhu 1 jam berikutnya. Windowing dilakukan setelah pembagian data.
8. **Model.** Arsitektur identik kecuali jenis lapis rekurennya:
   - `SimpleRNN(64, tanh)` + `Dense(1)` (5.057 parameter)
   - `LSTM(64)` + `Dense(1)` (20.033 parameter)
   - Optimizer Adam (learning rate 0,001), loss MSE, batch 256, maksimum 30 epoch, early stopping (patience 5).
9. **Evaluasi.** RMSE, MAE (dalam °C), dan R² pada data uji, dibandingkan dengan baseline naive (suhu jam terakhir dipakai sebagai prediksi).
10. **Eksperimen tambahan.** Panjang window 12, 24, dan 48 jam, masing-masing dengan 3 seed (1, 2, 3).

## Hasil

### Data uji (satu kali training, seed 1, window 24 jam)

| Model | RMSE (°C) | MAE (°C) | R² | Parameter |
|---|---|---|---|---|
| Naive | 0,9009 | 0,6425 | 0,9866 | 0 |
| RNN | 0,5460 | 0,3817 | 0,9951 | 5.057 |
| LSTM | 0,5287 | 0,3672 | 0,9954 | 20.033 |

Dibandingkan naive, RMSE RNN turun sekitar 39% dan LSTM sekitar 41%. RMSE pada train, validasi, dan uji berdekatan (LSTM: 0,547, 0,563, 0,529), sehingga tidak ada indikasi overfitting.

### Pengaruh panjang window (rata-rata ± simpangan baku dari 3 seed, RMSE dalam °C)

| Window | LSTM | RNN | Naive |
|---|---|---|---|
| 12 jam | 0,5609 ± 0,0086 | 0,5783 ± 0,0074 | 0,9004 |
| 24 jam | 0,5359 ± 0,0066 | 0,5550 ± 0,0087 | 0,9009 |
| 48 jam | 0,5340 ± 0,0076 | 0,5616 ± 0,0096 | 0,9010 |

### Visualisasi

| | |
|---|---|
| ![Pola musiman](images/Pola%20Musiman.png) | ![Suhu harian per tahun](images/Daily%20temperature%20by%20year.png) |
| ![Kurva loss](images/Kurva%20train%20loss%20dan%20validation%20loss.png) | ![RMSE vs window](images/RMSE%20vs%20panjang%20window.png) |
| ![Prediksi vs aktual](images/Prediksi%20vs%20aktual.png) | ![Scatter prediksi vs aktual](images/Sccaterplot%20Prediksi%20vs%20aktual.png) |

## Pembahasan

- RNN maupun LSTM jelas lebih baik daripada baseline naive pada seluruh window yang diuji.
- LSTM lebih akurat daripada RNN pada ketiga window. Selisihnya sekitar 0,017 sampai 0,028 °C, atau sekitar 2 sampai 4 kali galat baku selisihnya. RMSE LSTM membaik dari window 12 ke 24 jam lalu mendatar di 48 jam (satu siklus harian sudah tercakup di window 24 jam), sedangkan RMSE RNN tidak membaik di window 48 jam. Pola ini konsisten dengan teori *vanishing gradient* pada RNN, tetapi efeknya kecil.
- R² yang sangat tinggi tidak menjadi bukti utama, karena baseline naive pun sudah mencapai 0,9866 akibat dominasi siklus harian dan musiman. Ukuran yang lebih informatif adalah penurunan RMSE terhadap naive.

## Keterbatasan

- Jumlah seed hanya 3, sehingga perbedaan RNN dan LSTM bersifat indikatif, belum konklusif.
- Hanya horizon 1 jam yang diuji. Pada horizon pendek, naive sudah sangat kuat, dan perbedaan antar model lebih jelas pada horizon lebih panjang.
- Hiperparameter (64 unit, 30 epoch, learning rate 0,001) tidak dituning.
- Satu lokasi dan satu kali pembagian waktu; tidak ada validasi silang berbasis waktu.
- Seluruh 13 fitur dipakai, termasuk fitur yang redundan secara fisik (`Tpot`, `sh`, `H2OC`, dan lainnya). Ablasi fitur belum dilakukan.
- Nilai tekanan minimum (913,6 mbar) belum diperiksa apakah kejadian cuaca nyata atau galat sensor.

## Pengembangan Lanjutan

- Menambah seed dan panjang window (96, 168 jam), serta horizon multi-langkah.
- Ablasi fitur (set fitur ringkas vs lengkap).
- Menambahkan model MLP dan CNN-LSTM sebagai pembanding.

## Struktur Repositori

```
.
├── Model.ipynb          # notebook utama
├── data/                # letakkan jena_climate_2009_2016.csv di sini (tidak di-commit)
├── images/              # gambar hasil
├── requirements.txt
├── .gitignore
└── README.md
```

## Cara Menjalankan

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook Model.ipynb
```

Pastikan `data/jena_climate_2009_2016.csv` sudah ada. Jalankan seluruh sel secara berurutan dari atas (Restart & Run All). Bagian eksperimen seed dan window memerlukan waktu paling lama. Catatan: TensorFlow ≥ 2.11 di Windows native hanya berjalan di CPU. Gunakan WSL2 atau Google Colab bila ingin GPU.

## Penulis

Fatta — Program Studi Statistika, FMIPA, Universitas Negeri Padang