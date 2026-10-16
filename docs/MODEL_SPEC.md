# MODEL_SPEC.md — Spesifikasi Model DengueCast

> Dokumen ini adalah sumber kebenaran teknis model. Simpan di `docs/MODEL_SPEC.md`.
> Bagian C.13 berlaku di atas bagian lain bila bertabrakan (khusus konteks Indonesia).

---

## C.1 Alur Sistem

```
Kasus bulanan ───┐
Iklim observasi ─┼─> agregasi wilayah ─> fitur (lag, rata-rata bergerak) ─┐
ENSO, populasi ──┘                                                        │
Prakiraan iklim (atau klimatologi) ─> fitur bulan target ─────────────────┤
                                                                          v
      kandidat NB-GLM (banyak set fitur) + baseline musiman
                     │  expanding-window, skor CRPS
                     v
      pilih K terbaik + baseline ─> bobot 1/CRPS² ─> linear pool 1000 sampel
                     v
 kuantil (2.5/50/97.5), probabilitas melewati ambang KLB, tingkat risiko
                     v
            Parquet/DuckDB ─> API ─> dashboard
```

---

## C.2 Notasi

- `i` = wilayah
- `t` = bulan
- `o` = origin (bulan terakhir yang punya data kasus)
- `h` = lead 1 sampai 6; bulan target `t = o + h`
- `Y` = kasus; `P` = populasi; `inc` = 100.000 × Y / P

---

## C.3 Model Inti (Tahap A — nb_glm)

```
Y[i,t] ~ NegativeBinomial(rata-rata mu[i,t], dispersi alpha)
log mu[i,t] = a_i                       (efek tetap wilayah)
            + m_bulan(t)                (efek tetap bulan kalender)
            + log P[i,t]                (offset populasi)
            + rho * log(Ylag[i,t] + 1)  (kasus bulan sebelumnya)
            + sum_k beta_k * X_k[i,t]   (fitur iklim dan lingkungan)
```

**Ketidakpastian**: ambil 1000 sampel beta ~ MVN(beta_hat, Cov), lalu Y ~ NB(mu(beta), alpha).

**Antarmuka seragam**: `predict_samples(X, n_samples, seed) -> array[n_rows, n_samples]`

Model Tahap B (opsional) menambah efek acak wilayah, wilayah×bulan (AR1), tahun, dan spasial (ICAR/BYM2) seperti D-MOSS, dengan antarmuka yang sama.

---

## C.4 Fitur (X)

| Fitur | Definisi |
|-------|----------|
| `ylag` | log(kasus bulan sebelumnya + 1). h=1 memakai observasi bulan o; h≥2 memakai rekonstruksi (C.5) |
| `tmean_l02`, `tmin_l02`, `tmax_l02` | Rata-rata suhu bulan t, t−1, t−2 |
| `dtr_l02` | dtr = tmax − tmin (harian, lalu dirata-rata bulanan), rata-rata bulan t, t−1, t−2 |
| `humidity_l02`, `precip_l02` | Rata-rata bulan t, t−1, t−2 |
| `wind_l0` | Bulan t, tanpa lag |
| `nino34_l03` | Rata-rata anomali Niño 3.4 bulan t sampai t−3 |
| `urban_pct`, `periurban_pct` | Tahunan, opsional |

**Aturan fitur**:

1. z-score memakai statistik data latih saja.
2. Dalam satu model, |Spearman| antar prediktor harus < 0.7. `tmin` dan `tmax` tidak dipakai bersama `tmean` atau `dtr`.
3. Rata-rata bergerak untuk bulan target memakai observasi untuk bulan ≤ o dan nilai dari `ClimateProvider` untuk bulan > o. Observasi bulan > o **tidak boleh** dipakai.

---

## C.5 Rekonstruksi Kasus ber-Lag (h ≥ 2)

Model M0 (NB-GLM: wilayah + bulan + log(Ylag+1), tanpa iklim) memprediksi rata-rata kasus bulan o+1 dari Y[o]; lalu bulan o+2 dari prediksi bulan o+1; dan seterusnya sampai h=6. Deret rekonstruksi yang sama dipakai semua model kandidat.

Seperti D-MOSS, ketidakpastian langkah ini tidak diteruskan secara default. Sediakan opsi `propagate_lag_uncertainty` (default `false`) yang mengambil sampel lintasan.

---

## C.6 ClimateProvider

**Antarmuka**: `get(region_ids, origin, target_months)` mengembalikan tabel `[region_id, target, member, variabel_iklim]`.

| Provider | Keterangan |
|----------|------------|
| `ClimatologyProvider` (MVP) | Rata-rata bulan kalender 10 tahun terakhir sebelum origin, satu anggota |
| `SyntheticEnsembleProvider` | Klimatologi + derau terkorelasi, 20 anggota, untuk uji |
| `SeasonalForecastProvider` (Tahap B) | Prakiraan iklim musiman nyata beserta hindcast |

Model dijalankan per anggota ensemble, lalu semua sampel digabung.

---

## C.7 Kandidat dan Seleksi

Kandidat = ylag + efek bulan + subset grup `{suhu, dtr, humidity, precip, wind, nino, landcover}`.

- Maksimal 128 kandidat (setara jumlah spesifikasi D-MOSS)
- Buang yang melanggar filter kolinearitas
- Skor kandidat = rata-rata CRPS expanding-window (C.9) pada lead 1 sampai 6
- Pilih **K=4** terbaik, lalu tambahkan baseline

---

## C.8 Superensemble

Bobot: `w_i = (1/CRPS_i²) / Σ_j (1/CRPS_j²)`

Linear pool: ambil `round(w_i × N)` sampel dari tiap model, N=1000.

Bobot dihitung ulang pada tiap prakiraan baru.

**Keluaran**: `mean`, `q025`, `q50`, `q975`, `p_exceed` per ambang, `risk_level`.

---

## C.9 Validasi (Expanding-Window)

- Data latih awal sampai bulan T0 (default 5 tahun pertama)
- Untuk tiap origin `o` dari T0 sampai akhir data − 1:
  - Latih hanya dengan data ≤ o
  - Prediksi h=1..6 dengan iklim dari `ClimateProvider`
  - Observasi iklim bulan > o **dilarang**
- Tes otomatis kebocoran **wajib** ada

---

## C.10 Ambang KLB dan Tingkat Risiko

**Endemic channel** per wilayah × bulan kalender dari insidens 5 tahun sebelum origin (opsi: buang tahun KLB).

**Ambang**: `mean+1SD`, `mean+2SD`, persentil 75, persentil 95.

`p_exceed` = proporsi sampel insidens yang melampaui ambang.

**Tingkat risiko**:
- Rendah: < mean
- Sedang: mean sampai +1SD
- Tinggi: +1SD sampai +2SD
- Sangat Tinggi: > +2SD

---

## C.11 Metrik

- **CRPS**: `mean|X − y| − 0.5 × mean|X − X'|`; CRPSS = 1 − CRPS_model / CRPS_baseline
- **Point**: MAE, RMSE, bias
- **Interval**: cakupan 95% (target 0.95 ± 0.05), lebar interval
- **Klasifikasi**: Brier, AUC, sensitivitas, PPV, akurasi pada cutoff 0.25, 0.5, 0.75; tabel reliabilitas
- **Stratifikasi**: semua dipecah per lead, bulan target, dan wilayah

**Baseline**:
- B1: Rata-rata musiman yang diperbarui per wilayah-bulan
- B2: NB musiman tanpa iklim dan tanpa ylag
- B3: Persistensi (opsional)

---

## C.12 Skema Tabel (Parquet atau DuckDB)

```sql
cases(region_id, month, cases, definition)
population(region_id, year, population)
climate_obs(region_id, month, tmean, tmin, tmax, dtr, precip, humidity, wind)
enso(month, nino34_anom)
landcover(region_id, year, urban_pct, periurban_pct)
forecasts(run_id, region_id, origin, target, lead, mean, q025, q50, q975,
          p_exceed_m1sd, p_exceed_m2sd, p_exceed_p75, p_exceed_p95,
          risk_level, model_version, synthetic)
ensemble_weights(run_id, model_id, features, weight, crps)
eval_scores(run_id, region_id, lead, target_month, metric, value)
```

---

## C.13 Penyesuaian untuk Indonesia

> Bagian ini **menggantikan** C.3 sampai C.11 bila bertentangan.

### Rumus RH dari T dan Td

```
RH = 100 × exp(17.625 × Td / (243.04 + Td))
         / exp(17.625 × T  / (243.04 + T))
```

di mana T = suhu udara (°C), Td = suhu titik embun (°C).

### Aturan Spesifik Indonesia

1. **Unit analisis**: provinsi (34, batas sebelum 2022). Kolom `region_group`: Sumatra (10 provinsi), Java-Bali (7), Kalimantan (5), Sulawesi (6), Nusa Tenggara (2), Maluku (2), Papua (2).

2. **Fitur tambahan**: `oni_l03` dan `dmi_l03` (rata-rata bulan t sampai t−3).

3. **Jendela lag**: `climate_window = 3` (lag 0–2) atau `5` (lag 0–4); keduanya diuji sebagai grup kandidat terpisah.

4. **Musim**: efek bulan × wilayah (34 × 12 = 408 parameter). Bila observasi per wilayah < 72 bulan, pakai bulan × `region_group`.

5. **Interaksi**: koefisien `precip` dan `humidity` boleh berbeda per `region_group` (opsi `climate_by_group`).

6. **Gerbang kelayakan**: provinsi layak bila CRPSS median pada lead 1–3 terhadap B1 > 0 dan batas bawah interval bootstrap > 0; selain itu `reliable = false` pada semua keluaran.

7. **COVID-19**: opsi `exclude_periods` (default 2020–2021); laporkan skor dengan dan tanpa periode itu.

8. **Prakiraan iklim Tahap A**: `ClimatologyProvider` untuk iklim lokal; ONI dan DMI dipertahankan pada nilai bulan origin (persistensi). Asumsi ini **wajib** dicatat di metadata keluaran.

9. **Skema tambahan**:
   - `climate_obs(source, n_stations, valid_frac, quant_error)`
   - `regions(region_id, name, region_group, lat, lon)`
   - `forecasts.reliable`
   - `province_eligibility(run_id, region_id, crpss_median, ci_low, ci_high, eligible)`
