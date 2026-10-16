# PRD — DengueCast: Sistem Peramalan DBD Probabilistik

**Versi**: 1.0  
**Tanggal**: Oktober 2026  
**Status**: Draft Awal

---

## Ringkasan

DengueCast adalah sistem peramalan probabilistik kasus Demam Berdarah Dengue (DBD) bulanan per wilayah administrasi untuk 1 sampai 6 bulan ke depan, terinspirasi sistem D-MOSS (Colón-González dkk., 2021; Campbell dkk., 2026). Sistem ini adalah prototipe riset dan **bukan alat keputusan klinis atau kebijakan**.

---

## Latar Belakang

Pengendalian dengue sering bersifat reaktif karena surveilans pasif dan terlambat. Pengambil keputusan membutuhkan prakiraan dini dengan ketidakpastian yang eksplisit agar dapat merencanakan intervensi dan alokasi sumber daya secara lebih efektif.

---

## Tujuan

| Kode | Tujuan |
|------|--------|
| G1 | Prakiraan 1–6 bulan per wilayah beserta interval ketidakpastian |
| G2 | Probabilitas melewati ambang KLB yang dapat dipilih pengguna |
| G3 | Evaluasi jujur terhadap baseline, tanpa kebocoran data |
| G4 | Pipeline bulanan yang dapat diulang dan dilacak |
| G5 | Dashboard sederhana untuk non-statistisi |
| G6 | Menandai provinsi yang prakiraannya layak dan yang kurang andal *(ID)* |

---

## Bukan Tujuan

- Keputusan klinis individual
- Model deep learning
- Prediksi tingkat individu
- Resolusi di bawah unit administrasi data kasus
- Sistem multi-tenant produksi

---

## Persona Pengguna

| Persona | Kebutuhan Utama |
|---------|----------------|
| **Epidemiolog Dinkes** | Membaca peta risiko per wilayah, memilih ambang KLB |
| **Data Scientist** | Mengganti data dan model lewat konfigurasi dan adapter |
| **Manajer Program** | Alokasi anggaran dan sumber daya berdasarkan tingkat risiko |

---

## User Stories

| Kode | Story |
|------|-------|
| US1 | Sebagai epidemiolog, saya melihat peta tingkat risiko per wilayah untuk bulan target dan lead yang saya pilih. |
| US2 | Saya memilih ambang KLB (mean+1SD, mean+2SD, P75, P95). |
| US3 | Saya melihat kasus historis 24 bulan dan interval prakiraan per wilayah. |
| US4 | Sebagai data scientist, saya mengganti sumber data lewat konfigurasi dan adapter. |
| US5 | Saya menjalankan evaluasi dan melihat skor per lead, bulan, wilayah. |
| US6 | Saya membandingkan model dengan baseline. |
| US7 | Saya mengunduh prakiraan sebagai CSV. |
| US8 | Saya melihat jejak run: versi model, hash data, konfigurasi, tanggal. |

---

## Persyaratan Fungsional

| Kode | Persyaratan |
|------|-------------|
| FR1 | Ingest dan validasi data (skema, duplikat, bulan hilang, nilai tak wajar) |
| FR2 | Agregasi ke wilayah berbobot populasi |
| FR3 | Fitur sesuai MODEL_SPEC C.4 |
| FR4 | Baseline B1 dan B2 |
| FR5 | NB-GLM dan rekonstruksi kasus ber-lag |
| FR6 | Kandidat dan seleksi berbasis CRPS |
| FR7 | Superensemble berbobot CRPS |
| FR8 | Prakiraan 1–6 bulan beserta sampel |
| FR9 | Ambang KLB, probabilitas, tingkat risiko |
| FR10 | Evaluasi expanding-window |
| FR11 | Laporan evaluasi (tabel dan grafik) |
| FR12 | API baca prakiraan dan evaluasi |
| FR13 | Dashboard |
| FR14 | CLI untuk setiap tahap |
| FR15 | Mode sintetis end-to-end |
| FR16 | Adapter BMKG Data Online (Excel/CSV, penanganan kode 8888 dan 9999) *(ID)* |
| FR17 | Adapter iklim grid (ERA5-Land, NASA POWER) *(ID)* |
| FR18 | Adapter kasus (OpenDengue, CSV Dinkes) *(ID)* |
| FR19 | Digitasi gambar dan tabel PDF BMKG sebagai jalur cadangan berlabel *(ID)* |
| FR20 | Gerbang kelayakan per provinsi *(ID)* |
| FR21 | Analisis sensitivitas tanpa periode 2020–2021 *(ID)* |

---

## Persyaratan Non-Fungsional

| Kode | Persyaratan |
|------|-------------|
| NFR1 | Reprodusibel (seed, versi, hash data dan konfigurasi) |
| NFR2 | Bebas kebocoran data, dibuktikan tes otomatis |
| NFR3 | Demo sintetis (63 wilayah, 12 tahun) selesai < 10 menit di laptop biasa |
| NFR4 | Modular lewat antarmuka (`ClimateProvider`, `ForecastModel`) |
| NFR5 | Cakupan tes ≥ 80% untuk modul inti |
| NFR6 | Data sensitif tidak masuk repo |
| NFR7 | Data sintetis berlabel jelas di API dan UI |
| NFR8 | Dokumentasi keterbatasan yang jujur |

---

## Kriteria Penerimaan

| Kode | Kriteria |
|------|----------|
| AC1 | Pada data sintetis dengan sinyal iklim yang diketahui, CRPSS superensemble > 0 terhadap B1 pada lead 1–3 |
| AC2 | Cakupan interval 95% antara 0.90 dan 1.00 |
| AC3 | Tes kebocoran lulus |
| AC4 | Pipeline CLI end-to-end sukses |
| AC5 | API dan dashboard menampilkan prakiraan |
| AC6 | Semua tes hijau, ruff dan mypy bersih |
| AC7 | Adapter BMKG Data Online mengubah berkas contoh menjadi tabel bulanan dengan penanganan 8888 dan 9999 yang teruji *(ID)* |
| AC8 | Digitizer gambar memulihkan kelas pada gambar uji sintetis dan menolak georeferensi yang buruk *(ID)* |

---

## Risiko dan Mitigasi

| Risiko | Mitigasi |
|--------|----------|
| Data kasus sedikit atau berisik | Beri peringatan, tolak latih bila di bawah ambang minimum |
| Keterlambatan pelaporan | Tandai bulan terakhir tidak lengkap |
| Batas wilayah berubah | Bekukan ke satu tahun acuan |
| Model tidak mengalahkan baseline di lead panjang | Laporkan apa adanya |
| Overfitting karena banyak kandidat | Validasi expanding-window dan evaluasi prospektif terpisah |
| Stasiun BMKG tidak merata dan berlubang *(ID)* | Iklim grid sebagai sumber utama |
| Gambar BMKG kehilangan presisi *(ID)* | Label, galat kuantisasi, validasi |
| Kasus DBD klinis dan pelaporan wilayah timur lebih rendah *(ID)* | Peringatan per provinsi |
| Syarat penggunaan data BMKG dapat berubah *(ID)* | Cek manual dan catat di metadata |

---

## Asumsi

- Data kasus bulanan per wilayah tersedia minimal 8 tahun.
- Wilayah punya ID dan GeoJSON.
- Sumber utama iklim berbentuk grid (ERA5-Land atau NASA POWER); data stasiun BMKG Data Online untuk validasi; gambar atau PDF BMKG hanya sebagai jalur cadangan. *(ID)*

---

## Milestone

| Kode | Milestone |
|------|-----------|
| M0 | Aturan agen dan PRD |
| M1 | Kerangka repo |
| M2 | Data dan generator sintetis |
| M3 | Fitur dan model |
| M4 | Ensemble dan evaluasi |
| M5 | Pipeline, API, dashboard |
| M6 | Data nyata (Tahap A) |
| M7 | Prakiraan iklim musiman (Tahap B) |

---

## Pertanyaan Terbuka

- [ ] Wilayah target: provinsi atau kabupaten/kota? *(ID)*
- [ ] Definisi kasus: suspek atau terkonfirmasi laboratorium?
- [ ] Lisensi dan akses data (Kemenkes atau Dinkes)
- [ ] Periode data kasus yang tersedia *(ID)*
- [ ] Izin akses data Kemenkes atau Dinkes *(ID)*

---

**(ID)** = perubahan dari patch P1b untuk Indonesia
