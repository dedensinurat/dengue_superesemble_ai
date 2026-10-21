# DengueCast — Laporan Evaluasi Model Peramalan DBD Probabilistik

> **Pemberitahuan Penting**: DengueCast adalah prototipe riset untuk peramalan DBD probabilistik. Bukan alat keputusan klinis atau kebijakan. Data kasus berupa DBD berdiagnosis klinis.

## 1. Ringkasan Kinerja Keseluruhan per Model

| model_id | CRPS | CRPSS_vs_B1 | MAE | RMSE | Cakupan_95 | Lebar_Interval |
| --- | --- | --- | --- | --- | --- | --- |
| B1_seasonal_mean | 3985.681 | 0.000 | 5033.248 | 7054.779 | 0.869 | 23730.014 |
| B2_seasonal_nb | 3969.980 | 0.011 | 5262.181 | 7342.718 | 0.830 | 21198.728 |
| nbglm_eval | 4044.263 | 0.117 | 5718.336 | 8026.955 | 0.860 | 24699.757 |

## 2. Kinerja per Lead Horizon (1 - 6 Bulan)

| model_id | lead | CRPS | CRPSS_Median | Cakupan_95 |
| --- | --- | --- | --- | --- |
| B1_seasonal_mean | 1 | 4042.481 | 0.000 | 0.854 |
| B1_seasonal_mean | 2 | 4486.717 | 0.000 | 0.813 |
| B1_seasonal_mean | 3 | 4211.098 | 0.000 | 0.832 |
| B1_seasonal_mean | 4 | 4202.061 | 0.000 | 0.867 |
| B1_seasonal_mean | 5 | 3748.703 | 0.000 | 0.895 |
| B1_seasonal_mean | 6 | 3223.023 | 0.000 | 0.952 |
| B2_seasonal_nb | 1 | 4181.660 | 0.010 | 0.819 |
| B2_seasonal_nb | 2 | 4428.100 | 0.012 | 0.784 |
| B2_seasonal_nb | 3 | 4195.311 | 0.006 | 0.803 |
| B2_seasonal_nb | 4 | 4083.241 | 0.007 | 0.813 |
| B2_seasonal_nb | 5 | 3728.848 | -0.004 | 0.851 |
| B2_seasonal_nb | 6 | 3202.722 | 0.043 | 0.911 |
| nbglm_eval | 1 | 4164.105 | 0.089 | 0.838 |
| nbglm_eval | 2 | 4801.179 | 0.104 | 0.794 |
| nbglm_eval | 3 | 4346.794 | 0.123 | 0.825 |
| nbglm_eval | 4 | 4008.579 | 0.151 | 0.870 |
| nbglm_eval | 5 | 3616.337 | 0.140 | 0.886 |
| nbglm_eval | 6 | 3338.896 | 0.128 | 0.952 |

## 3. Gerbang Kelayakan Provinsi (P7b)

Provinsi dinyatakan **layak** jika median CRPSS pada lead 1-3 > 0 dan batas bawah 95% interval bootstrap > 0. Provinsi di luar kriteria ini ditandai sebagai **kurang andal** (`reliable = false`).

**Ringkasan Status**: 17 dari 63 provinsi dinyatakan **layak**.

| region_id | crpss_median | ci_low | ci_high | eligible | status |
| --- | --- | --- | --- | --- | --- |
| REG_53 | 0.429 | 0.015 | 0.533 | True | layak |
| REG_61 | 0.406 | 0.176 | 0.545 | True | layak |
| REG_16 | 0.372 | 0.076 | 0.457 | True | layak |
| REG_06 | 0.312 | 0.105 | 0.349 | True | layak |
| REG_55 | 0.303 | -0.018 | 0.522 | False | kurang andal |
| REG_57 | 0.301 | 0.090 | 0.464 | True | layak |
| REG_56 | 0.300 | 0.093 | 0.410 | True | layak |
| REG_18 | 0.252 | 0.171 | 0.504 | True | layak |
| REG_13 | 0.233 | -0.098 | 0.352 | False | kurang andal |
| REG_50 | 0.228 | -0.060 | 0.520 | False | kurang andal |
| REG_48 | 0.213 | -0.018 | 0.336 | False | kurang andal |
| REG_21 | 0.210 | -0.001 | 0.439 | False | kurang andal |
| REG_43 | 0.208 | 0.017 | 0.494 | True | layak |
| REG_34 | 0.196 | -0.419 | 0.328 | False | kurang andal |
| REG_46 | 0.180 | 0.053 | 0.316 | True | layak |
| REG_22 | 0.179 | -0.147 | 0.380 | False | kurang andal |
| REG_08 | 0.175 | 0.120 | 0.333 | True | layak |
| REG_35 | 0.175 | 0.017 | 0.375 | True | layak |
| REG_02 | 0.175 | -0.012 | 0.334 | False | kurang andal |
| REG_30 | 0.161 | -0.430 | 0.333 | False | kurang andal |
| REG_54 | 0.159 | 0.008 | 0.397 | True | layak |
| REG_20 | 0.159 | 0.025 | 0.261 | True | layak |
| REG_59 | 0.158 | 0.059 | 0.379 | True | layak |
| REG_11 | 0.156 | -0.344 | 0.308 | False | kurang andal |
| REG_14 | 0.155 | -0.948 | 0.421 | False | kurang andal |
| REG_26 | 0.149 | -0.448 | 0.266 | False | kurang andal |
| REG_05 | 0.148 | 0.050 | 0.374 | True | layak |
| REG_25 | 0.141 | -0.029 | 0.402 | False | kurang andal |
| REG_24 | 0.133 | -0.077 | 0.250 | False | kurang andal |
| REG_33 | 0.130 | 0.053 | 0.224 | True | layak |
| REG_15 | 0.125 | -0.261 | 0.327 | False | kurang andal |
| REG_29 | 0.121 | 0.029 | 0.283 | True | layak |
| REG_12 | 0.111 | -0.319 | 0.298 | False | kurang andal |
| REG_37 | 0.107 | -0.044 | 0.264 | False | kurang andal |
| REG_28 | 0.097 | -0.352 | 0.142 | False | kurang andal |
| REG_44 | 0.096 | -0.408 | 0.268 | False | kurang andal |
| REG_03 | 0.089 | -0.418 | 0.315 | False | kurang andal |
| REG_27 | 0.079 | -0.371 | 0.199 | False | kurang andal |
| REG_01 | 0.071 | -0.631 | 0.224 | False | kurang andal |
| REG_04 | 0.067 | -0.676 | 0.517 | False | kurang andal |
| REG_38 | 0.066 | -0.198 | 0.147 | False | kurang andal |
| REG_58 | 0.065 | -0.302 | 0.372 | False | kurang andal |
| REG_17 | 0.063 | -0.776 | 0.135 | False | kurang andal |
| REG_10 | 0.060 | -0.204 | 0.183 | False | kurang andal |
| REG_07 | 0.039 | -0.340 | 0.196 | False | kurang andal |
| REG_32 | 0.038 | -0.544 | 0.245 | False | kurang andal |
| REG_52 | 0.036 | -0.114 | 0.180 | False | kurang andal |
| REG_51 | 0.004 | -0.344 | 0.249 | False | kurang andal |
| REG_63 | -0.001 | -0.652 | 0.237 | False | kurang andal |
| REG_40 | -0.001 | -0.618 | 0.132 | False | kurang andal |
| REG_39 | -0.017 | -0.168 | 0.445 | False | kurang andal |
| REG_45 | -0.044 | -0.312 | 0.147 | False | kurang andal |
| REG_47 | -0.045 | -1.139 | 0.172 | False | kurang andal |
| REG_49 | -0.055 | -0.283 | 0.158 | False | kurang andal |
| REG_42 | -0.056 | -0.902 | 0.411 | False | kurang andal |
| REG_09 | -0.063 | -0.308 | 0.142 | False | kurang andal |
| REG_19 | -0.115 | -0.647 | 0.180 | False | kurang andal |
| REG_36 | -0.184 | -0.627 | -0.058 | False | kurang andal |
| REG_31 | -0.203 | -0.976 | 0.105 | False | kurang andal |
| REG_62 | -0.240 | -0.666 | 0.130 | False | kurang andal |
| REG_60 | -0.289 | -0.922 | -0.019 | False | kurang andal |
| REG_41 | -0.305 | -0.396 | -0.179 | False | kurang andal |
| REG_23 | -0.434 | -0.731 | -0.086 | False | kurang andal |

## 4. Analisis Sensitivitas: Pengaruh Pandemi COVID-19 (2020-2021)

| Kondisi | Mean CRPS | Median CRPSS | Cakupan 95% |
|---|---|---|---|
| Lengkap (Semua Tahun) | 3998.968 | 0.000 | 0.853 |
| Tanpa 2020-2021 (COVID) | 3491.988 | 0.000 | 0.845 |
