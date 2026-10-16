# DengueCast — Sistem Peramalan DBD Probabilistik

> **⚠️ DISCLAIMER**: DengueCast adalah prototipe riset yang terinspirasi sistem D-MOSS.
> Sistem ini **BUKAN** alat keputusan klinis atau kebijakan.
> Data default adalah **data sintetis** — bukan data DBD nyata.

---

## Tujuan

DengueCast meramalkan kasus Demam Berdarah Dengue (DBD) secara probabilistik per wilayah administrasi, untuk 1–6 bulan ke depan. Sistem menghasilkan interval ketidakpastian dan probabilitas melewati ambang Kejadian Luar Biasa (KLB).

---

## Instalasi

```bash
# 1. Clone repo
git clone <url-repo> dengue-forecast
cd dengue-forecast

# 2. Install (Python 3.11+)
make setup

# 3. Verifikasi
make lint
make test
```

---

## Demo 5 Menit (Data Sintetis)

```bash
# Generate data sintetis dan jalankan pipeline lengkap
make demo

# Jalankan API
make api
# → buka http://localhost:8000/docs

# Jalankan dashboard
make dashboard
# → buka http://localhost:8501
```

---

## Arsitektur Singkat

```
Data (kasus, iklim, ENSO, populasi, GeoJSON)
  │
  ▼ [ingest adapters]
Lapisan Data (Parquet + DuckDB)
  │
  ▼ [features/]
Rekayasa Fitur (lag, rolling, z-score, endemic channel)
  │
  ▼ [models/ + ensemble/]
NB-GLM Kandidat → Seleksi CRPS → Superensemble
  │
  ▼ [evaluation/]
Evaluasi Expanding-Window (CRPS, AUC, reliabilitas)
  │
  ▼ [api/ + dashboard/]
REST API (FastAPI) + Dashboard (Streamlit)
```

Detail lengkap: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

---

## Cara Mengganti Data Sintetis dengan Data Nyata

### Langkah 1: Siapkan data kasus
```bash
# Unduh data dari OpenDengue
denguecast data load-cases --source opendengue --path data/raw/opendengue.csv
```

### Langkah 2: Siapkan iklim (ERA5-Land direkomendasikan)
Unduh NetCDF dari Copernicus CDS, lalu:
```bash
denguecast data load --source era5land --path data/raw/era5land/
```

Lihat panduan lengkap: [`docs/DATA_GUIDE.md`](docs/DATA_GUIDE.md)

### Langkah 3: Adapter BMKG (opsional, stasiun)
```bash
denguecast data bmkg-load --dir data/raw/bmkg/ --stations data/ref/stations.csv
```

### Langkah 4: Indeks ENSO & IOD
```bash
denguecast data load-indices --oni data/raw/oni.txt --dmi data/raw/dmi.txt
```

### Langkah 5: Validasi sebelum melatih
```bash
# WAJIB: periksa laporan kualitas sebelum melanjutkan
denguecast data validate
```

### Langkah 6: Jalankan pipeline
```bash
# Nonaktifkan mode sintetis di config/default.yaml: synthetic: false
denguecast run --origin 2024-10
```

---

## Batasan dan Disclaimer

Lihat [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) untuk daftar lengkap. Ringkasan:

- Sistem ini **prototipe riset**, bukan produk siap pakai.
- Data default adalah **sintetis** — tidak mencerminkan situasi DBD nyata.
- Model tidak memodelkan serotipe, imunitas populasi, mobilitas, atau intervensi pengendalian.
- Provinsi dengan koherensi rendah ditandai sebagai **kurang andal**.
- Klaim performa pada data nyata hanya valid setelah **uji prospektif** yang dibekukan sebelum nilai aktual tersedia.
- Kasus adalah DBD berdiagnosis **klinis** — bukan konfirmasi laboratorium.

---

## Referensi

- Colón-González FJ, dkk. (2021). Probabilistic seasonal dengue forecasting in Vietnam. *PLOS NTD*.
- Campbell AM, dkk. (2026). D-MOSS system. *(referensi diperbarui sesuai versi publikasi)*.
- Djaafara BA, dkk. (2026). Climate-dengue associations in Indonesia. *PLOS NTD*. https://doi.org/10.1371/journal.pntd.0014135

---

## Struktur Repo

```
dengue-forecast/
├── AGENTS.md               ← aturan kerja agen
├── config/default.yaml     ← SEMUA parameter di sini
├── data/                   ← gitignored
├── docs/
│   ├── PRD.md              ← kebutuhan produk
│   ├── MODEL_SPEC.md       ← spesifikasi model
│   ├── ARCHITECTURE.md     ← arsitektur dan desain
│   ├── INDONESIA.md        ← panduan data Indonesia
│   ├── DATA_GUIDE.md       ← cara menyiapkan data nyata
│   └── LIMITATIONS.md      ← batasan sistem
├── src/denguecast/         ← kode utama
├── apps/dashboard/         ← Streamlit dashboard
└── tests/                  ← pytest
```

---

*Versi: 0.1.0 | Python 3.11+ | Lisensi: MIT*
