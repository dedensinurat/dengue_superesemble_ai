# Batasan Metodologi & Keterbatasan Sistem (LIMITATIONS)

Dokumen ini menguraikan asumsi dasar, faktor-faktor biologis/lingkungan yang tidak dimodelkan, serta panduan kapan prakiraan DengueCast tidak boleh dipercaya.

---

## 1. Penafian Utama (Disclaimer)

> **PEMBERITAHUAN**: DengueCast adalah prototipe riset sistem peramalan probabilistik yang terinspirasi oleh arsitektur D-MOSS (Colón-González et al. 2021; Campbell et al. 2026). Sistem ini **BUKAN ALAT KEPUTUSAN KLINIS ATAU KEBIJAKAN OPERASIONAL LANGSUNG**. Angka kasus yang dimodelkan adalah kasus DBD berdiagnosis klinis yang dilaporkan secara rutin, bukan kasus terkonfirmasi laboratorium secara lengkap.

---

## 2. Asumsi Dasar Pemodelan

1. **Stasionaritas Musiman Relatif**: Hubungan fungsional antara anomali iklim (hujan, suhu, kelembaban, ENSO) dan dinamika vektor *Aedes aegypti* diasumsikan mengikuti pola empiris yang dapat dipelajari dari riwayat 5–10 tahun sebelumnya.
2. **Ketergantungan Spasial per Kelompok Wilayah**: Mengingat panjang deret waktu yang terbatas di beberapa wilayah, parameter interaksi iklim dikelompokkan berdasarkan gugus kepulauan (*region groups*: Sumatra, Jawa-Bali, Kalimantan, Sulawesi, Nusa Tenggara, Maluku, Papua).
3. **Penyedia Iklim Masa Depan**: Akurasi prakiraan lead 1–6 bulan sangat bergantung pada kualitas penyedia iklim (`ClimateProvider`). Jika menggunakan klimatologi (Tahap A), anomali iklim ekstrem masa depan tidak tertangkap dalam fitur iklim.

---

## 3. Faktor-Faktor yang Tidak Dimodelkan (Unmodeled Factors)

Sistem DengueCast sengaja membatasi ruang lingkup pada variabel yang tersedia secara operasional dan konsisten. Faktor-faktor berikut tidak tercakup dalam model:

1. **Transisi Serotipe Virus Dengue (DENV 1–4)**:
   Pergantian serotipe dominan dapat memicu lonjakan kasus akibat minimnya imunitas populasi terhadap serotipe baru. Data surveilans serotipe rutin tidak tersedia secara agregat bulanan.
2. **Dinamika Imunitas Populasi Mikro & Imunitas Silang Heterolog**:
   Kekebalan sementara silang antar-serotipe pasca-infeksi primer yang berlangsung 6–12 bulan tidak dimodelkan secara eksplisit secara mekanistik.
3. **Intervensi Pengendalian Vektor**:
   Tindakan pengendalian seperti pengasapan (*fogging* insektisida masif), Gerakan 1 Rumah 1 Jumantik (Pemberantasan Sarang Nyamuk/PSN), dan pelepasan nyamuk ber-*Wolbachia* (seperti di Yogyakarta, Semarang, Bandung) mereduksi transmisi secara lokal di luar ekspektasi model iklim.
4. **Mobilitas Manusia & Arus Urbanisasi**:
   Pola mudik hari raya, pergerakan komuter harian antarkota, dan pariwisata antarwilayah mempercepat introduksi virus ke populasi rentan.
5. **Kualitas dan Kelengkapan Pelaporan**:
   Perbedaan kapasitas fasilitas kesehatan, khususnya di wilayah Indonesia Timur, menyebabkan disparitas angka pelaporan (*under-reporting*) yang signifikan.

---

## 4. Distorsi Periode Pandemi COVID-19 (2020–2021)

Selama pandemi COVID-19, sistem pelaporan DBD mengalami disrupsi ganda:
- Masyarakat enggan mengunjungi faskes karena kekhawatiran penularan virus SARS-CoV-2.
- Pengalihan fokus tenaga kesehatan dan laboratorium ke pelaporan COVID-19.
- Kebijakan pembatasan sosial berskala besar (PSBB) mengubah pola mobilitas dan pajanan gigitan nyamuk di pemukiman.

DengueCast menyediakan opsi analisis sensitivitas (`exclude_periods: ["2020-01:2021-12"]`) untuk memverifikasi apakah bobot model dan estimasi parameter stabil saat periode disrupsi ini dikeluarkan dari pelatihan.

---

## 5. Gerbang Kelayakan: Provinsi 'Kurang Andal'

Studi Djaafara et al. (2026) menunjukkan bahwa 16 dari 34 provinsi di Indonesia memiliki koherensi fase iklim-DBD yang rendah ($< 0.70$).
- Pada provinsi-provinsi ini, DengueCast secara otomatis menyematkan label **Kurang Andal** (`reliable = False`) jika skor CRPSS median $\le 0$ atau batas bawah bootstrap interval kredibel $\le 0$.
- **Rekomendasi**: Untuk wilayah yang ditandai kurang andal, pemangku kepentingan disarankan mengutamakan pemantauan baseline epidemiologis historis (saluran endemik) daripada bertumpu pada sinyal iklim.

---

## 6. Kapan Prakiraan Tidak Boleh Dipercaya?

Pengguna dilarang mengambil tindakan berbasis prakiraan dalam kondisi:
1. **Lead Time Jauh ($h \ge 4$) pada Musim Transisi**: Ketidakpastian prakiraan iklim pada lead 4–6 bulan sangat tinggi; lebar interval kredibel 95% melebar tajam.
2. **Terjadi Bencana Alam / Gempa / Banjir Bandang**: Kerusakan infrastruktur memicu pengungsian dan anomali sanitasi yang tidak mencerminkan pola musiman normal.
3. **Intervensi Berskala Besar Baru Saja Dilakukan**: Jika program Wolbachia baru saja diterapkan secara luas di suatu wilayah, efikasi penularan vektor telah berubah secara struktural.
