"""
run_experiment.py - Menjalankan seluruh eksperimen Tugas Akhir Pengolahan Citra 125.

Keluaran (folder results/):
  metadata_dataset.csv   : metadata tiap citra (sumber, dimensi, kanal, detail)
  hasil_utama.csv        : PSNR/SSIM/MAE/F1-tepi tiap (citra, noise, metode, kernel)
  waktu_proses.csv       : waktu proses per metode/kernel pada citra asli
  waktu_skala.csv        : waktu proses terhadap ukuran citra dan ukuran kernel
  sweep_sigma.csv        : pengaruh sigma Gaussian
  verifikasi_manual.csv  : beda implementasi dari nol vs OpenCV
Semua acak memakai seed tetap sehingga hasil dapat direproduksi.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import citra as c  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)

SEED = 2026
METODE = ["Mean", "Gaussian", "Median"]
KERNEL = [3, 5, 7, 9]

t0 = time.time()
daftar = c.daftar_citra(DATA)
print(f"Jumlah citra: {len(daftar)}", flush=True)

# ------------------------------------------------------------------ metadata
meta_rows, gray_cache = [], []
for idx, (kel, nama, path) in enumerate(daftar):
    asli, m = c.muat_asli(path)
    g = c.ke_grayscale(asli)
    gray_cache.append(g)
    meta_rows.append({
        "id": idx, "kelompok": kel, "berkas": nama, **m,
        "intensitas_rerata": round(float(g.mean()), 2),
        "kontras_std": round(float(g.std()), 2),
        "detail_sobel": round(c.kekuatan_detail(g), 2),
    })
meta = pd.DataFrame(meta_rows)
meta.to_csv(os.path.join(OUT, "metadata_dataset.csv"), index=False)
print(meta.groupby("kelompok").size().to_dict(), flush=True)

# ------------------------------------------------------------- eksperimen utama
rows = []
for idx, (kel, nama, path) in enumerate(daftar):
    ref = gray_cache[idx]
    amb = c.ambang_canny(ref)
    for ci, (kode, jenis, par) in enumerate(c.KONDISI_NOISE):
        rng = np.random.default_rng(SEED * 1000 + idx * 10 + ci)
        noisy = c.beri_noise(ref, kode, rng)
        base = dict(id=idx, kelompok=kel, berkas=nama, noise=kode, jenis=jenis, par=par)
        p, r_, f = c.f1_tepi(ref, noisy, amb)
        rows.append({**base, "metode": "Tanpa filter", "kernel": 0,
                     "psnr": c.psnr(ref, noisy), "ssim": c.ssim(ref, noisy),
                     "mae": c.mae(ref, noisy), "tepi_presisi": p, "tepi_recall": r_, "tepi_f1": f})
        for met in METODE:
            for k in KERNEL:
                out = c.filter_cv(noisy, met, k)
                p, r_, f = c.f1_tepi(ref, out, amb)
                rows.append({**base, "metode": met, "kernel": k,
                             "psnr": c.psnr(ref, out), "ssim": c.ssim(ref, out),
                             "mae": c.mae(ref, out), "tepi_presisi": p, "tepi_recall": r_,
                             "tepi_f1": f})
    if idx % 10 == 0:
        print(f"  utama: {idx + 1}/{len(daftar)}  ({time.time() - t0:.0f} dtk)", flush=True)
hasil = pd.DataFrame(rows)
hasil.to_csv(os.path.join(OUT, "hasil_utama.csv"), index=False)

# ------------------------------------------------------------------ waktu proses
# Dijalankan berurutan (tanpa paralelisme) pada kondisi G25.
wrows = []
for idx, (kel, nama, path) in enumerate(daftar):
    ref = gray_cache[idx]
    noisy = c.beri_noise(ref, "G25", np.random.default_rng(SEED * 1000 + idx * 10 + 1))
    for met in METODE:
        for k in KERNEL:
            wrows.append({"id": idx, "kelompok": kel, "metode": met, "kernel": k,
                          "mp": ref.size / 1e6,
                          "waktu_ms": c.waktu_ms(lambda: c.filter_cv(noisy, met, k), 3)})
pd.DataFrame(wrows).to_csv(os.path.join(OUT, "waktu_proses.csv"), index=False)
print(f"waktu selesai ({time.time() - t0:.0f} dtk)", flush=True)

# --------------------------------------------------- skala waktu (ukuran & kernel)
# Citra dasar: Kodak24 kodim01 (grayscale), di-resize ke berbagai sisi.
base = gray_cache[[i for i, d in enumerate(daftar) if d[1] == "kodim01.png"][0]]
srows = []
for sisi in [128, 256, 512, 1024, 2048]:
    im = c.ubah_ukuran(base, sisi)
    for met in METODE:
        for k in [3, 5, 7, 9, 11, 15]:
            srows.append({"implementasi": "OpenCV", "sisi": sisi, "metode": met, "kernel": k,
                          "waktu_ms": c.waktu_ms(lambda: c.filter_cv(im, met, k), 3)})
im = c.ubah_ukuran(base, 128)
for met in METODE:
    for k in [3, 5, 7, 9, 11]:
        srows.append({"implementasi": "Manual (NumPy)", "sisi": 128, "metode": met, "kernel": k,
                      "waktu_ms": c.waktu_ms(lambda: c.filter_manual(im, met, k), 2)})
pd.DataFrame(srows).to_csv(os.path.join(OUT, "waktu_skala.csv"), index=False)
print(f"skala selesai ({time.time() - t0:.0f} dtk)", flush=True)

# --------------------------------------------------------------- sweep sigma
srows = []
for idx, (kel, nama, path) in enumerate(daftar):
    ref = gray_cache[idx]
    noisy = c.beri_noise(ref, "G25", np.random.default_rng(SEED * 1000 + idx * 10 + 1))
    import cv2
    for sg in [0.5, 0.8, 1.0, 1.4, 2.0, 3.0]:
        out = cv2.GaussianBlur(noisy, (15, 15), sg, borderType=cv2.BORDER_REFLECT_101)
        srows.append({"id": idx, "sigma": sg, "psnr": c.psnr(ref, out), "ssim": c.ssim(ref, out)})
pd.DataFrame(srows).to_csv(os.path.join(OUT, "sweep_sigma.csv"), index=False)

# ------------------------------------------------- verifikasi implementasi manual
vrows = []
pilih = [0, 20, 68, 70, 80, 95]            # campuran BSD68, Set12, Kodak24
for idx in pilih:
    ref = c.ubah_ukuran(gray_cache[idx], 128)
    noisy = c.beri_noise(ref, "G25", np.random.default_rng(SEED + idx))
    for met in METODE:
        for k in [3, 5, 7]:
            a = c.filter_cv(noisy, met, k).astype(int)
            b = c.filter_manual(noisy, met, k).astype(int)
            vrows.append({"id": idx, "metode": met, "kernel": k,
                          "beda_maks": int(np.abs(a - b).max()),
                          "beda_rerata": float(np.abs(a - b).mean())})
pd.DataFrame(vrows).to_csv(os.path.join(OUT, "verifikasi_manual.csv"), index=False)
print(f"SELESAI total {time.time() - t0:.0f} dtk", flush=True)
