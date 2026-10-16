# DengueCast — Laporan Evaluasi Model Peramalan DBD Probabilistik

> **Pemberitahuan Penting**: DengueCast adalah prototipe riset untuk peramalan DBD probabilistik. Bukan alat keputusan klinis atau kebijakan. Data kasus berupa DBD berdiagnosis klinis.

## 1. Ringkasan Kinerja Keseluruhan per Model

| model_id | CRPS | CRPSS_vs_B1 | MAE | RMSE | Cakupan_95 | Lebar_Interval |
| --- | --- | --- | --- | --- | --- | --- |
| B1_seasonal_mean | 3348.050 | 0.000 | 4389.035 | 6247.907 | 0.889 | 23637.493 |
| B2_seasonal_nb | 3363.418 | 0.009 | 4633.117 | 6610.685 | 0.844 | 20075.341 |
| nbglm_eval | 3438.065 | 0.130 | 5000.485 | 7345.915 | 0.875 | 23631.987 |

## 2. Kinerja per Lead Horizon (1 - 6 Bulan)

| model_id | lead | CRPS | CRPSS_Median | Cakupan_95 |
| --- | --- | --- | --- | --- |
| B1_seasonal_mean | 1 | 3317.422 | 0.000 | 0.883 |
| B1_seasonal_mean | 2 | 3500.368 | 0.000 | 0.848 |
| B1_seasonal_mean | 3 | 3519.326 | 0.000 | 0.854 |
| B1_seasonal_mean | 4 | 3619.942 | 0.000 | 0.860 |
| B1_seasonal_mean | 5 | 3420.094 | 0.000 | 0.917 |
| B1_seasonal_mean | 6 | 2711.149 | 0.000 | 0.971 |
| B2_seasonal_nb | 1 | 3342.429 | 0.026 | 0.835 |
| B2_seasonal_nb | 2 | 3541.168 | 0.012 | 0.806 |
| B2_seasonal_nb | 3 | 3711.575 | 0.003 | 0.810 |
| B2_seasonal_nb | 4 | 3554.034 | -0.001 | 0.825 |
| B2_seasonal_nb | 5 | 3371.635 | 0.005 | 0.870 |
| B2_seasonal_nb | 6 | 2659.667 | 0.018 | 0.917 |
| nbglm_eval | 1 | 3937.314 | 0.106 | 0.963 |
| nbglm_eval | 2 | 3375.594 | 0.126 | 0.848 |
| nbglm_eval | 3 | 3681.451 | 0.131 | 0.838 |
| nbglm_eval | 4 | 3436.393 | 0.133 | 0.848 |
| nbglm_eval | 5 | 3105.222 | 0.152 | 0.889 |
| nbglm_eval | 6 | 3172.416 | 0.117 | 0.913 |

## 3. Gerbang Kelayakan Provinsi (P7b)

Provinsi dinyatakan **layak** jika median CRPSS pada lead 1-3 > 0 dan batas bawah 95% interval bootstrap > 0. Provinsi di luar kriteria ini ditandai sebagai **kurang andal** (`reliable = false`).

**Ringkasan Status**: 17 dari 63 provinsi dinyatakan **layak**.

| region_id | crpss_median | ci_low | ci_high | eligible | status |
| --- | --- | --- | --- | --- | --- |
| REG_59 | 0.454 | 0.142 | 0.517 | True | layak |
| REG_25 | 0.359 | 0.312 | 0.493 | True | layak |
| REG_48 | 0.354 | 0.076 | 0.476 | True | layak |
| REG_32 | 0.338 | 0.030 | 0.460 | True | layak |
| REG_56 | 0.333 | 0.071 | 0.414 | True | layak |
| REG_13 | 0.327 | 0.085 | 0.497 | True | layak |
| REG_19 | 0.296 | 0.209 | 0.391 | True | layak |
| REG_17 | 0.285 | 0.006 | 0.391 | True | layak |
| REG_22 | 0.271 | 0.138 | 0.355 | True | layak |
| REG_41 | 0.269 | 0.073 | 0.400 | True | layak |
| REG_44 | 0.262 | 0.190 | 0.406 | True | layak |
| REG_36 | 0.245 | 0.116 | 0.288 | True | layak |
| REG_63 | 0.215 | -0.143 | 0.243 | False | kurang andal |
| REG_07 | 0.207 | -0.145 | 0.266 | False | kurang andal |
| REG_39 | 0.201 | -0.321 | 0.299 | False | kurang andal |
| REG_42 | 0.194 | 0.127 | 0.414 | True | layak |
| REG_10 | 0.194 | -0.169 | 0.281 | False | kurang andal |
| REG_60 | 0.187 | -0.009 | 0.317 | False | kurang andal |
| REG_14 | 0.184 | -0.074 | 0.251 | False | kurang andal |
| REG_49 | 0.183 | 0.057 | 0.266 | True | layak |
| REG_40 | 0.178 | -0.006 | 0.296 | False | kurang andal |
| REG_38 | 0.178 | -0.067 | 0.481 | False | kurang andal |
| REG_11 | 0.170 | -0.162 | 0.250 | False | kurang andal |
| REG_03 | 0.155 | 0.080 | 0.261 | True | layak |
| REG_18 | 0.150 | 0.052 | 0.318 | True | layak |
| REG_09 | 0.145 | -0.355 | 0.268 | False | kurang andal |
| REG_54 | 0.145 | -0.427 | 0.364 | False | kurang andal |
| REG_29 | 0.145 | -0.006 | 0.375 | False | kurang andal |
| REG_52 | 0.139 | -0.065 | 0.466 | False | kurang andal |
| REG_06 | 0.132 | -0.065 | 0.192 | False | kurang andal |
| REG_27 | 0.130 | -0.028 | 0.320 | False | kurang andal |
| REG_50 | 0.127 | -0.467 | 0.224 | False | kurang andal |
| REG_33 | 0.126 | 0.104 | 0.335 | True | layak |
| REG_04 | 0.121 | -0.275 | 0.394 | False | kurang andal |
| REG_43 | 0.109 | -0.429 | 0.303 | False | kurang andal |
| REG_61 | 0.097 | -0.057 | 0.234 | False | kurang andal |
| REG_20 | 0.096 | -0.352 | 0.211 | False | kurang andal |
| REG_21 | 0.094 | -0.183 | 0.151 | False | kurang andal |
| REG_47 | 0.091 | -0.152 | 0.313 | False | kurang andal |
| REG_30 | 0.082 | -0.120 | 0.356 | False | kurang andal |
| REG_57 | 0.081 | -0.342 | 0.410 | False | kurang andal |
| REG_15 | 0.077 | -0.174 | 0.185 | False | kurang andal |
| REG_01 | 0.077 | -0.492 | 0.195 | False | kurang andal |
| REG_45 | 0.066 | -0.319 | 0.172 | False | kurang andal |
| REG_26 | 0.054 | -0.138 | 0.508 | False | kurang andal |
| REG_28 | 0.040 | -0.057 | 0.328 | False | kurang andal |
| REG_08 | 0.018 | -0.241 | 0.131 | False | kurang andal |
| REG_46 | 0.017 | -0.302 | 0.167 | False | kurang andal |
| REG_35 | 0.016 | -0.031 | 0.184 | False | kurang andal |
| REG_37 | 0.014 | -0.110 | 0.317 | False | kurang andal |
| REG_58 | -0.015 | -0.254 | 0.150 | False | kurang andal |
| REG_24 | -0.015 | -0.827 | 0.338 | False | kurang andal |
| REG_12 | -0.022 | -0.248 | 0.150 | False | kurang andal |
| REG_51 | -0.031 | -0.206 | 0.297 | False | kurang andal |
| REG_23 | -0.081 | -0.538 | 0.060 | False | kurang andal |
| REG_34 | -0.134 | -0.870 | 0.181 | False | kurang andal |
| REG_55 | -0.185 | -0.471 | -0.014 | False | kurang andal |
| REG_53 | -0.190 | -0.921 | 0.310 | False | kurang andal |
| REG_31 | -0.250 | -1.165 | -0.113 | False | kurang andal |
| REG_02 | -0.270 | -1.062 | 0.250 | False | kurang andal |
| REG_62 | -0.274 | -0.590 | 0.159 | False | kurang andal |
| REG_05 | -0.301 | -0.685 | 0.003 | False | kurang andal |
| REG_16 | -0.395 | -0.627 | 0.011 | False | kurang andal |

## 4. Analisis Sensitivitas: Pengaruh Pandemi COVID-19 (2020-2021)

| Kondisi | Mean CRPS | Median CRPSS | Cakupan 95% |
|---|---|---|---|
| Lengkap (Semua Tahun) | 3380.625 | 0.000 | 0.869 |
| Tanpa 2020-2021 (COVID) | 2986.845 | 0.000 | 0.870 |
