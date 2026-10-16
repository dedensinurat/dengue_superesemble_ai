# AGENTS.md — Aturan Kerja Agen DengueCast

Dokumen ini adalah sumber kebenaran untuk aturan kerja agen selama seluruh proyek DengueCast.
Agen **wajib** membaca dan mematuhi semua aturan berikut sebelum mengerjakan tugas apa pun.

---

## Aturan Utama (P0)

1. **Sumber kebenaran**: `docs/PRD.md` dan `docs/MODEL_SPEC.md`. Jika instruksi prompt bertabrakan dengan keduanya, tanya pengguna lebih dulu.
2. **Bahasa**: kode, nama variabel, dan docstring dalam bahasa **Inggris**; dokumen dan README dalam bahasa **Indonesia**.
3. **Kualitas kode**: Python 3.11, type hints penuh, `ruff` dan `mypy` bersih, `pytest` hijau **sebelum** menyatakan selesai.
4. **Konfigurasi**: semua parameter lewat `config/default.yaml` (pydantic). Tidak ada angka ajaib (*magic numbers*) di kode.
5. **DILARANG kebocoran data masa depan**: fitur prakiraan tidak boleh memakai data iklim observasi bulan > `origin`. Tulis tes yang membuktikannya.
6. **Jangan mengarang data nyata**. Data sintetis wajib berlabel `synthetic=True` (atau `synthetic` di kolom/metadata) di semua keluaran dan UI.
7. **Dependensi terkontrol**: jangan menambah dependensi di luar daftar di `docs/ARCHITECTURE.md` tanpa bertanya ke pengguna terlebih dahulu.
8. **Tes wajib**: setiap fungsi publik punya tes. Fungsi yang melibatkan keacakan menerima parameter `seed`.
9. **Ritual awal dan akhir tugas**:
   - *Mulai*: tulis asumsi dan rencana singkat.
   - *Akhir*: tulis ringkasan, daftar file yang diubah, perintah verifikasi, dan hal yang belum selesai.
10. **Perubahan terfokus**: kecil dan spesifik. Jangan merombak modul yang tidak diminta.
11. **Cek dokumentasi resmi**: jika ragu tentang API pustaka, periksa dokumentasi yang terinstal (`help()`, `inspect`) alih-alih menebak.
12. **Jangan melemahkan tes**: jika tes gagal dan tes-nya yang salah, jelaskan alasannya lebih dulu sebelum mengubah tes.

---

## Aturan Tambahan — Data Indonesia & BMKG (P0b)

13. **Prioritas spesifikasi Indonesia**: `docs/MODEL_SPEC.md` bagian C.13 dan `docs/INDONESIA.md` berlaku di atas bagian lain bila bertabrakan.
14. **Penggunaan data BMKG**:
    - Pakai hanya berkas yang diunduh sendiri dari portal resmi dan disimpan di `data/raw/` (gitignored).
    - Dilarang *scraping*, dilarang mengakali login, dilarang menyimpan kredensial.
    - Catat tanggal unduh, nama stasiun, dan parameter di metadata.
    - Syarat penggunaan data BMKG wajib dicek sendiri oleh pengguna.
15. **Kode penanda BMKG**:
    - `8888` = tidak terukur; `9999` = tidak ada data.
    - Keduanya adalah **penanda**, bukan angka. **JANGAN** pernah dirata-ratakan.
    - Perlakuan tiap kode diatur di `config/default.yaml` dan wajib didokumentasikan.
16. **Data digitasi gambar BMKG**:
    - Selalu berlabel `source=bmkg_image` beserta ketidakpastian kuantisasi.
    - Tidak pernah dipakai sebagai target latih.
    - Wajib divalidasi terhadap data stasiun atau grid sebelum dipakai.
17. **Tidak ada klaim kausalitas**: jangan menyimpulkan bahwa iklim *menyebabkan* DBD dari koefisien model; laporkan sebagai **asosiasi**.
18. **Nama kolom kanonik**: nama kolom sumber (mis. `Tn`, `Tx`, `RH_avg`, `RR`) hanya boleh muncul di lapisan adapter. Bagian lain memakai nama kanonik: `tmin`, `tmax`, `tmean`, `humidity`, `precip`.
19. **Verifikasi format berkas**: bila format berkas, nama kolom, atau parameter API tidak pasti, baca berkas contoh pengguna atau dokumentasi resmi lebih dulu. **Jangan menebak.**

---

## Catatan Penting

- Dokumen ini bersifat *append-only*: aturan baru ditambahkan, aturan lama tidak dihapus.
- Urutan prompt: **P0 → P0b → P1 → P1b → P2 → P3 → P3a–P3e → P4 → P4b → P5 → P5b → P6 → P7 → P7b → P8 → P9 → P10 → P11**.
- Satu prompt = satu sesi kerja = satu commit.
- Jangan melanjutkan ke prompt berikutnya sebelum kriteria 'Selesai bila' dari prompt sebelumnya terpenuhi.
