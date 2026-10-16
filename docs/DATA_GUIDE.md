# Panduan Data DengueCast (DATA_GUIDE)

Dokumen ini memuat spesifikasi teknis format data, cara penyiapan, serta tata cara mengganti data sintetis dengan data observasi riil (BMKG, ERA5-Land, OpenDengue, Kemenkes).

---

## 1. Spesifikasi dan Format Tabel

Seluruh tabel data disimpan dalam format Apache Parquet pada direktori `data/processed/` dan didaftarkan sebagai tabel relasional DuckDB (`data/processed/denguecast.duckdb`).

### A. Tabel Kasus DBD (`cases`)
Memuat jumlah kasus DBD bulanan per wilayah administrasi.
- **Definisi Kasus**: DBD berdiagnosis klinis (ICD-10 A90/A91).
- **Format CSV**:
```csv
region_id,month,cases,definition,synthetic
ID-JK,2018-01,142,clinical,false
ID-JK,2018-02,185,clinical,false
ID-JB,2018-01,420,clinical,false
```
- **Aturan**:
  - `region_id`: Kode wilayah standar (ISO 3166-2 atau BPS).
  - `month`: Format ISO `YYYY-MM`.
  - `cases`: Integer $\ge 0$.
  - `synthetic`: Boolean (`true` jika data buatan, `false` jika data observasi).

### B. Tabel Populasi (`population`)
Memuat proyeksi populasi tahunan per wilayah untuk offset log-populasi $\log(\text{pop})$.
```csv
region_id,year,population
ID-JK,2018,10450000
ID-JK,2019,10560000
ID-JB,2018,48200000
```

### C. Tabel Iklim Observasi (`climate_obs`)
Memuat variabel iklim bulanan per wilayah hasil agregasi spasial berbobot populasi.
```csv
region_id,month,tmin,tmax,tmean,dtr,precip,humidity,wind,synthetic
ID-JK,2018-01,24.1,31.5,27.2,7.4,320.5,84.2,2.1,false
ID-JK,2018-02,23.8,30.8,26.7,7.0,290.0,86.1,2.3,false
```
- Variabel utama:
  - `tmin`, `tmax`, `tmean`: Suhu udara minimum, maksimum, dan rata-rata (°C).
  - `dtr`: Diurnal Temperature Range ($T_{max} - T_{min}$).
  - `precip`: Akumulasi curah hujan bulanan (mm).
  - `humidity`: Kelembaban relatif rata-rata (%).
  - `wind`: Kecepatan angin rata-rata (m/s).

### D. Tabel Indeks Iklim Global (`enso`)
Memuat indeks anomali sirkulasi laut-atmosfer global bulanan.
```csv
month,nino34,oni,dmi,synthetic
2018-01,-0.82,-0.75,0.12,false
2018-02,-0.71,-0.68,0.08,false
```
- `oni`: Oceanic Niño Index (anomali SST 3-bulan berjalan Niño 3.4 dari NOAA CPC).
- `dmi`: Dipole Mode Index (anomali gradien suhu Samudra Hindia / IOD dari BoM/NOAA).

---

## 2. Penurunan Iklim Berbobot Populasi (ERA5-Land & CHIRPS)

Untuk wilayah administrasi luas (provinsi/kabupaten), penggunaan rata-rata aritmetika grid iklim tidak akurat karena sebagian besar area berupa hutan/pegunungan tak berpenghuni di mana penularan DBD tidak terjadi.

DengueCast menerapkan pembobotan spasial berbobot kepadatan penduduk:
$$X_{region, t} = \frac{\sum_{p \in region} X_{p, t} \cdot W_p}{\sum_{p \in region} W_p}$$
di mana $W_p$ adalah kepadatan populasi pada grid sel $p$ (misal dari *WorldPop* atau *GPWv4*).

### Langkah Ekstraksi:
1. Unduh data bulanan ERA5-Land (suhu 2m, titik embun 2m, kecepatan angin 10m).
2. Unduh raster grid kepadatan populasi (WorldPop 1 km atau 10 km).
3. Jalankan modul `denguecast.data.ingest.grid_climate`:
```python
from denguecast.data.ingest.grid_climate import extract_population_weighted_climate

climate_monthly = extract_population_weighted_climate(
    nc_path="data/raw/era5_land_monthly.nc",
    pop_raster_path="data/raw/worldpop_indonesia.tif",
    geojson_path="data/ref/indonesia_provinces.geojson",
)
```

---

## 3. Penanganan Data Stasiun BMKG Data Online

Data harian stasiun meteorologi BMKG diunduh manual melalui portal `dataonline.bmkg.go.id`.

### A. Kode Khusus BMKG:
- `8888`: Nilai tidak terukur (misal alat penakar hujan rusak atau hujan sangat tipis/trace). Diubah menjadi `NaN`.
- `9999`: Data tidak tersedia/tidak dilakukan pengamatan. Diubah menjadi `NaN`.

### B. Validasi dan Ambang Hari Valid:
Agregasi harian ke bulanan mengharuskan fraksi hari valid minimal 80% (`min_valid_days_frac: 0.8`). Bulan yang memiliki $< 80\%$ pengamatan valid otomatis ditetapkan sebagai `NaN` untuk mencegah bias estimasi curah hujan akumulatif.

---

## 4. Pembekuan Batas Wilayah & Penyelarasan Nama

Perubahan pemekaran wilayah (seperti 4 provinsi baru di Papua pada tahun 2022) dapat merusak kontinuitas data historis.
- Rujukan batas: Batas 34 Provinsi Indonesia (2010–2021).
- Penyelarasan alias otomatis diatur dalam `denguecast.data.ingest.indonesia_cases`:
  - `DKI Jakarta` $\rightarrow$ `ID-JK`
  - `Jawa Barat` $\rightarrow$ `ID-JB`
  - `Daerah Istimewa Yogyakarta` / `DI Yogyakarta` $\rightarrow$ `ID-YO`

---

## 5. Keterlambatan Pelaporan (Reporting Delays)

Data kasus pada 1–2 bulan terakhir sebelum waktu peramalan ($t$) sering kali belum lengkap terlaporkan dari fasilitas kesehatan perifer ke Dinas Kesehatan.
- Modul `validate.flag_possibly_incomplete` otomatis memberi tanda peringatan `possibly_incomplete = True` pada data bulan terakhir.
- Pipeline bulanan menyediakan opsi untuk memangkas atau mendiskon bobot observasi yang belum lengkap.
