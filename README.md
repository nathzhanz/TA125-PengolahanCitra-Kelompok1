# TA Pengolahan Citra 125 — Perbandingan Mean, Gaussian, dan Median Filter untuk Reduksi Noise

Tugas Akhir mata kuliah **Pengolahan Citra** — Sistem dan Teknologi Informasi, Fakultas Teknik, Universitas Negeri Jakarta.
Dosen pengampu: Murien Nugraheni, S.T., M.Cs. · Kelompok 1.

Eksperimen terkontrol pada **104 citra** (BSD68, Set12, Kodak24) × **6 kondisi noise** (Gaussian σ = 10/25/50; *salt-and-pepper* 5/15/30%) × **3 filter** (mean, Gaussian, median) × **4 ukuran kernel** (3, 5, 7, 9).
Metrik: PSNR, SSIM, MAE, F1 tepi (Canny); uji Wilcoxon berpasangan; analisis kegagalan, waktu proses, dan pengaruh tingkat detail. Seed tetap (2026) sehingga hasil dapat direproduksi.

## Tautan

| Isi | Tautan |
|---|---|
| Notebook Google Colab | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XPJYI7uEGybDegiFA3G5xrfFeFUsYZYh?usp=sharing) |
| Google Drive (laporan, presentasi, hasil) | https://drive.google.com/drive/folders/1DdpsvcNxsteuKPIEeJVzMRbmm_lZ8iDZ?usp=drive_link |
| Repositori GitHub | https://github.com/nathzhanz/TA125-PengolahanCitra-Kelompok1 |

## Struktur repositori

```
.
├── README.md
├── requirements.txt
├── unduh_dataset.py      # mengunduh 104 citra uji ke data/
├── run_experiment.py     # eksperimen utama, waktu proses, sweep sigma, verifikasi manual vs OpenCV
├── analyze.py            # ringkasan, uji Wilcoxon, analisis kegagalan, citra berwarna
├── buat_gambar.py        # seluruh gambar/grafik hasil (figures/)
├── buat_xlsx.py          # dokumentasi hasil ke satu berkas Excel
├── src/
│   └── citra.py          # pustaka inti: prapemrosesan, noise, filter, metrik
├── notebook/
│   └── TA_PengolahanCitra125.ipynb   # versi Colab (semua dalam satu notebook)
├── results/              # CSV hasil + dokumentasi_hasil_eksperimen.xlsx
├── figures/              # gambar hasil (PNG)
└── laporan/              # laporan akhir (PDF)
```

Folder `data/` dibuat otomatis oleh `unduh_dataset.py` dan tidak disimpan di repositori.

## Cara menjalankan

**Di Google Colab (tanpa memasang apa pun).** Klik tombol *Open in Colab* di atas, lalu **Runtime → Run all** (sekitar 6–10 menit). Di akhir, izinkan akses Google Drive agar hasil tersalin ke *My Drive/TA_PengolahanCitra125*.

**Di komputer sendiri.** Python 3.10 atau lebih baru.

```bash
pip install -r requirements.txt
python unduh_dataset.py      # 104 citra -> data/
python run_experiment.py     # ± 5–8 menit pada CPU tunggal
python analyze.py
python buat_gambar.py
python buat_xlsx.py
```

Keluaran ada di `results/` dan `figures/`. Angka PSNR/SSIM identik pada setiap eksekusi; waktu proses dapat berbeda antarmesin.

## Sumber dataset

- Kodak24: Eastman Kodak Company, *Kodak Lossless True Color Image Suite* — https://r0k.us/graphics/kodak/
- BSD68: subset uji BSDS300, Martin dkk. (ICCV 2001) — https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/
- Set12, BSD68, Kodak24 (salinan uji denoising): https://github.com/cszn/DnCNN dan https://github.com/cszn/FFDNet

Citra dipakai hanya untuk keperluan pembelajaran/penelitian non-komersial dan tidak disertakan di repositori ini.

## Lisensi

Kode: MIT (lihat `LICENSE`). Dataset mengikuti lisensi/ketentuan sumber masing-masing.
