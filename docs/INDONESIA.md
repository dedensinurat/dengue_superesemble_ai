# INDONESIA.md — Panduan Data dan Penyesuaian untuk Indonesia

> Simpan sebagai `docs/INDONESIA.md`. Baca sebelum P1.
> Bila bertabrakan dengan prompt lain, dokumen ini yang berlaku.

---

## A2. Penyesuaian untuk Data Indonesia (BMKG)

Data BMKG tidak seluruhnya berupa gambar: data iklim pengamatan stasiun tersedia sebagai tabel, sedangkan prakiraan hujan bulanan dan prakiraan musim berupa PDF atau PNG.

### Apa yang Disediakan BMKG

| Jenis Data | Format | Dipakai Untuk | Catatan |
|------------|--------|---------------|---------|
| Data pengamatan harian per stasiun (Data Online BMKG) | Tabel (Excel atau CSV) | Fitur iklim observasi | Contoh kolom: `Tn`, `Tx`, `Tavg`, `RH_avg`, `RR`, `ss`, `ff_x`, `ddd_x`, `ff_avg`, `ddd_car`. Kode `8888` = tidak terukur, `9999` = tidak ada data |
| Data iklim olahan (harian, dasarian, bulanan, tahunan) | CSV, Level 2 | Fitur iklim observasi | Level 2 dapat diakses bebas; data mentah Level 1 hanya lewat kerja sama BMKG |
| Data model iklim regional seluruh Indonesia, bulanan | NetCDF, Level 2 | Fitur iklim berbentuk grid | Cara akses perlu ditanyakan ke BMKG |
| Prakiraan curah hujan bulanan Indonesia | PDF atau PNG, Level 3 | Prakiraan hujan kasar | Peta kelas hujan, perlu digitasi |
| Prakiraan musim hujan dan kemarau | PDF, Level 3 | Penanda musim | Kategori, bukan angka kontinu |

> Kebijakan data bertanggal 2014 (Perka BMKG 20/2014). Syarat dan biaya akses terkini harus dicek di portal Data Online BMKG.

---

### Data Kasus DBD

Data kasus provinsi bulanan 2010–2024 tersedia di **OpenDengue** (hingga Oktober 2024). Data yang lebih rinci milik Kemenkes dan butuh izin ([Djaafara dkk., 2026](https://doi.org/10.1371/journal.pntd.0014135)).

- Kasusnya DBD berdiagnosis **klinis**; konfirmasi laboratorium jarang.
- Penelitian itu memakai 34 provinsi dengan batas sebelum 2022.
- Empat provinsi pemekaran Papua dikecualikan karena data hilang.
- Unit analisis prototipe adalah **provinsi**; kabupaten/kota hanya bila ada data Dinkes.

---

### Keputusan Sumber Data

| Peran | Sumber Utama | Cadangan / Pelengkap |
|-------|-------------|----------------------|
| Kasus | OpenDengue (provinsi, bulanan) atau CSV Dinkes | — |
| Iklim observasi | ERA5-Land bulanan berbobot populasi, atau NASA POWER | Stasiun BMKG Data Online (validasi dan koreksi) |
| ENSO dan IOD | ONI dan DMI | — |
| Prakiraan iklim 1–6 bulan | Tahap A: klimatologi + persistensi ONI/DMI. Tahap B: prakiraan musiman global | Peta prakiraan hujan BMKG hasil digitasi |
| Gambar dan PDF BMKG | Hanya jalur cadangan (P3d) | — |

**Alasan**: sebaran stasiun mungkin tidak merata; gambar hanya memberi kelas, bukan angka; grid memberi cakupan penuh. Penelitian pada provinsi Indonesia juga memakai ERA5-Land berbobot populasi.

---

### Temuan Indonesia yang Mengubah Desain

1. Hubungan iklim dan DBD berbeda antar provinsi → efek musim per wilayah, interaksi iklim × kelompok wilayah, gerbang kelayakan per provinsi.
2. Hujan lebih berguna daripada suhu → uji jendela lag 0–2 dan 0–4.
3. Kelembaban mengikuti hujan dengan jeda ~1 bulan → filter kolinearitas tetap aktif; RH dihitung dari T dan titik embun.
4. ONI berkorelasi dengan tahun puncak (rho 0.83); DMI lebih kuat di Maluku dan Nusa Tenggara → fitur ONI dan DMI, serta persistensi indeks untuk horizon panjang.
5. 16 provinsi punya koherensi rendah (< 0.70) → tandai sebagai 'kurang andal'.
6. Kasus berdiagnosis klinis; 2020–2021 terdampak COVID-19 → analisis sensitivitas tanpa 2020–2021.

---

### Prompt yang Berubah karena Indonesia

| Prompt | Perubahan |
|--------|-----------|
| P0b (baru) | Aturan tambahan untuk data BMKG dan Indonesia |
| P1b (baru) | Patch PRD |
| P2 | Dependensi diperluas |
| P3 | Direvisi: hanya antarmuka dan registry adapter |
| P3a (baru) | Adapter BMKG Data Online (stasiun) |
| P3b (baru) | Iklim grid: ERA5-Land dan NASA POWER |
| P3c (baru) | Kasus DBD, batas wilayah, dan populasi Indonesia |
| P3d (baru) | Digitasi gambar dan tabel PDF BMKG (jalur cadangan) |
| P3e (baru) | Tabel acuan wilayah, ONI, dan DMI |
| P4b, P5b, P7b (baru) | Patch fitur, model, dan evaluasi |
| P9 dan P10 | Tambahkan field `reliable` dan banner 'kurang andal' |

---

### Yang Belum Diverifikasi

1. Syarat, biaya, dan format unduhan Data Online BMKG saat ini.
2. Nama kolom persis pada berkas Anda sendiri, dan struktur tabel PDF Prakiraan Musim.
3. Nama kolom berkas OpenDengue dan nama parameter API NASA POWER.

> Prompt di bawah memerintahkan agen membaca berkas contoh Anda dan dokumentasi resmi lebih dulu, bukan menebak.

---

### Sumber

- [Peraturan Kepala BMKG 20/2014 (Kebijakan Data)](https://jdih.bmkg.go.id/storage/common/dokumen/KEBIJAKAN%20PENGELOLAAN%20DATA%20METEOROLOGI,%20KLIMATOLOGI,%20DAN%20GEOFISIKA%20DILINGKUNGAN%20BMKG.pdf)
- [Contoh data stasiun BMKG Sleman (IEEE DataPort)](https://ieee-dataport.org/documents/weather-data-measured-bmkg-sleman-station-yogyakarta-indonesia-2004-2024)
- [Djaafara dkk. (2026), PLOS NTD](https://doi.org/10.1371/journal.pntd.0014135)
