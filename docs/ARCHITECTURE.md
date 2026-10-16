# ARCHITECTURE.md — Arsitektur DengueCast

---

## Diagram Alur

```
INPUT ──────────────────────────────────────────────────────────
  Kasus DBD (OpenDengue / CSV Dinkes)
  Iklim Observasi (ERA5-Land / NASA POWER / BMKG Stasiun)
  ENSO & DMI (ONI / DMI indices)
  Populasi (BPS Sensus 2010 & 2020)
  GeoJSON Batas Wilayah (34 provinsi, sebelum 2022)
         │
         ▼
LAPISAN DATA ───────────────────────────────────────────────────
  src/denguecast/data/
  ├── schema.py     → validasi skema & rentang nilai
  ├── validate.py   → laporan kualitas (bulan hilang, outlier, dll)
  ├── store.py      → Parquet I/O + DuckDB registry + hash jejak
  ├── synthetic.py  → generator data sintetis (63 wilayah, 12 tahun)
  └── ingest/       → adapter sumber data (CLI: denguecast data load)
      ├── cases.py           → CSV generik kasus
      ├── opendengue.py      → OpenDengue CSV (P3c)
      ├── bmkg_dataonline.py → BMKG stasiun Excel/CSV (P3a)
      ├── era5land.py        → ERA5-Land NetCDF grid (P3b)
      ├── nasa_power.py      → NASA POWER API (P3b, opsional jaringan)
      ├── indices.py         → ONI & DMI + persistensi prakiraan (P3e)
      ├── bmkg_image.py      → digitizer gambar BMKG (P3d, cadangan)
      ├── bmkg_pdf.py        → ekstraktor tabel PDF BMKG (P3d, cadangan)
      ├── population.py      → interpolasi populasi BPS
      ├── geo.py             → GeoJSON + adjacency shapely
      ├── landcover.py       → tutupan lahan (opsional)
      └── climate_obs.py     → adapter iklim generik
         │
         ▼
LAPISAN IKLIM PRAKIRAAN ────────────────────────────────────────
  src/denguecast/climate/
  ├── provider.py           → ABC ClimateProvider (antarmuka utama)
  ├── climatology.py        → ClimatologyProvider (MVP, Tahap A)
  ├── synthetic_ensemble.py → SyntheticEnsembleProvider (20 anggota, tes)
  └── seasonal.py           → SeasonalForecastProvider (stub, Tahap B)
         │
         ▼
REKAYASA FITUR ─────────────────────────────────────────────────
  src/denguecast/features/
  ├── aggregate.py   → agregasi berbobot populasi, DTR
  ├── lags.py        → rolling mean lag per wilayah (anti-bocor)
  ├── build.py       → build_training_frame, build_forecast_frame
  └── thresholds.py  → endemic channel, risk_level, exceedance_prob
         │
         ▼
MODEL ──────────────────────────────────────────────────────────
  src/denguecast/models/
  ├── base.py               → ABC ForecastModel + ModelSpec
  ├── baseline.py           → B1 SeasonalExpandingMean, B2 SeasonalNB
  ├── nb_glm.py             → NBGLM (statsmodels NegativeBinomial)
  ├── recursive_cases.py    → ReconstructLag (h≥2, MODEL_SPEC C.5)
  ├── hierarchical_bayes.py → HierarchicalBayes (stub, Tahap B, PyMC)
  └── registry.py           → build_model dari ModelSpec
         │
         ▼
ENSEMBLE & SELEKSI ─────────────────────────────────────────────
  src/denguecast/ensemble/
  ├── candidates.py → generate_candidates (maks 128 spesifikasi)
  ├── selection.py  → score_candidates (expanding CV + joblib paralel)
  └── stacking.py   → compute_weights (CRPS²) + Superensemble (linear pool)
         │
         ▼
EVALUASI ───────────────────────────────────────────────────────
  src/denguecast/evaluation/
  ├── splits.py        → ExpandingWindowSplitter
  ├── scores.py        → crps_samples, crpss, coverage, interval_width
  ├── classification.py → brier, AUC, contingency table, sensitivity, PPV
  ├── reliability.py   → reliability diagram table
  └── report.py        → eval_scores Parquet + Markdown/HTML report
         │
         ▼
PRAKIRAAN & PIPELINE ───────────────────────────────────────────
  src/denguecast/forecast/
  ├── run.py     → run_forecast (orchestrates h=1..6, all regions)
  └── outputs.py → format_forecasts → forecasts + ensemble_weights tables

  src/denguecast/pipelines/
  └── monthly_run.py → denguecast run --origin YYYY-MM
         │
         ▼
OUTPUT ─────────────────────────────────────────────────────────
  data/processed/denguecast.duckdb
  ├── cases, population, climate_obs, enso, landcover
  ├── forecasts (dengan reliable, synthetic flags)
  ├── ensemble_weights
  ├── eval_scores
  └── province_eligibility

         │                    │
         ▼                    ▼
  REST API (FastAPI)    Dashboard (Streamlit)
  src/denguecast/api/   apps/dashboard/
  /v1/forecasts         Peta risiko + fan chart
  /v1/evaluation        Evaluasi + model info
  /v1/download          Download CSV
```

---

## Tanggung Jawab Tiap Modul

| Modul | Tanggung Jawab |
|-------|----------------|
| `config.py` | Satu sumber kebenaran parameter, tanpa angka ajaib di kode |
| `data/schema.py` | Validasi skema dan rentang nilai semua tabel |
| `data/validate.py` | Laporan kualitas data, flag keterlambatan pelaporan |
| `data/store.py` | Parquet I/O, DuckDB, hash jejak run |
| `data/synthetic.py` | Generator deterministik, koefisien ground-truth tersimpan |
| `data/ingest/*.py` | Adapter sumber data — nama kolom sumber HANYA di sini |
| `climate/provider.py` | ABC anti-kebocoran: `get()` dilarang akses bulan > origin |
| `features/build.py` | Frame latih dan prakiraan, z-score dari data latih |
| `features/thresholds.py` | Endemic channel dari data sebelum origin |
| `models/base.py` | Antarmuka seragam semua model |
| `models/nb_glm.py` | NB-GLM + sampling MVN ketidakpastian |
| `models/recursive_cases.py` | Rekonstruksi ylag untuk h≥2 |
| `ensemble/` | Enumerasi, seleksi, bobot CRPS, linear pool |
| `evaluation/` | Expanding-window CV, CRPS, AUC, laporan |
| `pipelines/monthly_run.py` | Pipeline end-to-end idempoten |
| `api/` | FastAPI baca-saja: prakiraan, evaluasi, unduhan |
| `apps/dashboard/` | Streamlit + plotly: peta, fan chart, evaluasi |

---

## Daftar Dependensi Diizinkan

### Inti
```
pandas numpy pyarrow duckdb scipy statsmodels scikit-learn
geopandas shapely pyyaml pydantic pydantic-settings
typer fastapi uvicorn streamlit plotly joblib
openpyxl  # BMKG Excel
```

### Extras Opsional
```
xarray rasterio rasterstats  # ERA5-Land grid
httpx cdsapi                 # NASA POWER, Copernicus
pdfplumber                   # BMKG PDF
pillow scikit-image          # BMKG gambar
pymc                         # Tahap B hierarkis
```

### Dev
```
pytest pytest-cov hypothesis ruff mypy httpx
```

**Tidak ada dependensi lain tanpa persetujuan eksplisit.**

---

## Keputusan Desain

1. **Python 3.11 (bukan R)**: ekosistem data lebih luas; lebih mudah dikerjakan agen koding. Tahap B bisa menjembatani ke R-INLA bila diperlukan.
2. **DuckDB + Parquet (bukan database berat)**: cukup untuk prototipe satu mesin; mudah direproduksi.
3. **ABC/Protocol ketat untuk ClimateProvider dan ForecastModel**: memastikan setiap implementasi baru mematuhi kontrak anti-kebocoran.
4. **Expanding-window (bukan k-fold)**: menghormati urutan waktu dan mensimulasikan kondisi produksi nyata.
5. **Linear pool (bukan stacking ML)**: transparan, dapat diaudit, tidak perlu data holdout tambahan.
6. **Data sintetis sebagai default**: tidak ada risiko data nyata masuk repo secara tidak sengaja.
7. **Satu file konfigurasi YAML**: semua parameter terpusat; tidak ada angka ajaib tersebar di kode.
8. **Nama kolom kanonik di luar adapter**: `tmin`, `tmax`, `tmean`, `humidity`, `precip` — bukan `Tn`, `Tx`, dll.
