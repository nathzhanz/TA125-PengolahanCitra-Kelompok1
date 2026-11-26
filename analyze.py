"""
analyze.py - Meringkas hasil eksperimen, uji statistik, analisis kegagalan,
dan eksperimen tambahan pada citra berwarna (Kodak24).
Membaca results/*.csv, menulis results/ringkasan_*.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import citra as c  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results")
H = pd.read_csv(os.path.join(OUT, "hasil_utama.csv"))
M = pd.read_csv(os.path.join(OUT, "metadata_dataset.csv"))
KODE = [k for k, _, _ in c.KONDISI_NOISE]

# ---------------------------------------------------------------- ringkasan umum
agg = (H.groupby(["noise", "metode", "kernel"])
         .agg(psnr=("psnr", "mean"), psnr_sd=("psnr", "std"),
              ssim=("ssim", "mean"), ssim_sd=("ssim", "std"),
              mae=("mae", "mean"), tepi_f1=("tepi_f1", "mean"),
              tepi_presisi=("tepi_presisi", "mean"), tepi_recall=("tepi_recall", "mean"),
              n=("psnr", "size"))
         .reset_index())
agg["noise"] = pd.Categorical(agg["noise"], KODE, ordered=True)
agg = agg.sort_values(["noise", "metode", "kernel"])
agg.to_csv(os.path.join(OUT, "ringkasan_semua_konfigurasi.csv"), index=False)

# konfigurasi terbaik (berdasar PSNR rerata) per kondisi x metode
idx = agg[agg.metode != "Tanpa filter"].groupby(["noise", "metode"], observed=True)["psnr"].idxmax()
best = agg.loc[idx].copy()
best["noise"] = best["noise"].astype(str)
base = agg[agg.metode == "Tanpa filter"].assign(noise=lambda d: d.noise.astype(str)).set_index("noise")
best["psnr_noisy"] = best["noise"].astype(str).map(base["psnr"]).astype(float)
best["ssim_noisy"] = best["noise"].astype(str).map(base["ssim"]).astype(float)
best["tepi_f1_noisy"] = best["noise"].astype(str).map(base["tepi_f1"]).astype(float)
best["gain_psnr"] = best["psnr"] - best["psnr_noisy"]
best.to_csv(os.path.join(OUT, "ringkasan_terbaik.csv"), index=False)

# ----------------------------------------------------------- uji statistik berpasangan
rows = []
for kode in KODE:
    sub = H[H.noise == kode]
    bm = {m: int(best[(best.noise == kode) & (best.metode == m)].kernel.iloc[0])
          for m in ["Mean", "Gaussian", "Median"]}
    get = {m: sub[(sub.metode == m) & (sub.kernel == bm[m])].sort_values("id")["psnr"].values
           for m in bm}
    for a, b in [("Median", "Gaussian"), ("Median", "Mean"), ("Gaussian", "Mean")]:
        d = get[a] - get[b]
        try:
            p = stats.wilcoxon(get[a], get[b]).pvalue
        except ValueError:
            p = np.nan
        rows.append({"noise": kode, "A": f"{a} (k={bm[a]})", "B": f"{b} (k={bm[b]})",
                     "selisih_psnr_rerata": d.mean(), "A_lebih_baik_pada_%": 100 * (d > 0).mean(),
                     "p_wilcoxon": p})
pd.DataFrame(rows).to_csv(os.path.join(OUT, "uji_wilcoxon.csv"), index=False)

# ---------------------------------------------------- analisis kegagalan per citra
gagal = []
for kode in KODE:
    sub = H[H.noise == kode]
    noisy = sub[sub.metode == "Tanpa filter"].set_index("id")["psnr"]
    for m in ["Mean", "Gaussian", "Median"]:
        k = int(best[(best.noise == kode) & (best.metode == m)].kernel.iloc[0])
        s = sub[(sub.metode == m) & (sub.kernel == k)].set_index("id")["psnr"]
        gain = s - noisy
        gagal.append({"noise": kode, "metode": m, "kernel_terbaik": k,
                      "gain_rerata": gain.mean(), "gain_min": gain.min(), "gain_maks": gain.max(),
                      "citra_gagal": int((gain < 0).sum()),
                      "persen_gagal": 100 * (gain < 0).mean()})
pd.DataFrame(gagal).to_csv(os.path.join(OUT, "ringkasan_kegagalan.csv"), index=False)

# kegagalan pada kernel tetap k=5 (konfigurasi umum, bukan terbaik)
g5 = []
for kode in KODE:
    sub = H[H.noise == kode]
    noisy = sub[sub.metode == "Tanpa filter"].set_index("id")["psnr"]
    for m in ["Mean", "Gaussian", "Median"]:
        s = sub[(sub.metode == m) & (sub.kernel == 5)].set_index("id")["psnr"]
        gain = s - noisy
        g5.append({"noise": kode, "metode": m, "kernel": 5, "citra_gagal": int((gain < 0).sum()),
                   "persen_gagal": 100 * (gain < 0).mean(), "gain_rerata": gain.mean()})
pd.DataFrame(g5).to_csv(os.path.join(OUT, "kegagalan_k5.csv"), index=False)

# ------------------------------------------- pengaruh tingkat detail (tertil Sobel)
M["tingkat_detail"] = pd.qcut(M["detail_sobel"], 3, labels=["Rendah", "Sedang", "Tinggi"])
Hd = H.merge(M[["id", "tingkat_detail", "detail_sobel"]], on="id")
rows = []
for kode in KODE:
    for tl in ["Rendah", "Sedang", "Tinggi"]:
        sub = Hd[(Hd.noise == kode) & (Hd.tingkat_detail == tl)]
        for m in ["Mean", "Gaussian", "Median"]:
            for k in [3, 5, 7, 9]:
                s = sub[(sub.metode == m) & (sub.kernel == k)]
                rows.append({"noise": kode, "detail": tl, "metode": m, "kernel": k,
                             "psnr": s.psnr.mean(), "ssim": s.ssim.mean()})
D = pd.DataFrame(rows)
D.to_csv(os.path.join(OUT, "ringkasan_detail.csv"), index=False)
iD = D.groupby(["noise", "detail", "metode"])["psnr"].idxmax()
D.loc[iD].to_csv(os.path.join(OUT, "ringkasan_detail_terbaik.csv"), index=False)

# korelasi detail citra vs gain (konfigurasi terbaik)
rows = []
for kode in KODE:
    sub = Hd[Hd.noise == kode]
    noisy = sub[sub.metode == "Tanpa filter"].set_index("id")["psnr"]
    for m in ["Mean", "Gaussian", "Median"]:
        k = int(best[(best.noise == kode) & (best.metode == m)].kernel.iloc[0])
        s = sub[(sub.metode == m) & (sub.kernel == k)].set_index("id")["psnr"]
        gain = (s - noisy)
        det = M.set_index("id").loc[gain.index, "detail_sobel"]
        rho, p = stats.spearmanr(det, gain)
        rows.append({"noise": kode, "metode": m, "spearman_rho": rho, "p": p})
pd.DataFrame(rows).to_csv(os.path.join(OUT, "korelasi_detail_gain.csv"), index=False)

# ---------------------------------------------- eksperimen tambahan: citra warna RGB
SEED = 2026
rows = []
kod = M[M.kelompok == "Kodak24"]
for _, r in kod.iterrows():
    asli, _ = c.muat_asli(os.path.join(ROOT, "data", "Kodak24", r["berkas"]))
    for ci, kode in enumerate(["G25", "SP15"]):
        rng = np.random.default_rng(SEED * 7 + int(r["id"]) * 10 + ci)
        noisy = np.stack([c.beri_noise(asli[..., ch], kode, rng) for ch in range(3)], axis=-1)
        rows.append({"berkas": r["berkas"], "noise": kode, "metode": "Tanpa filter", "kernel": 0,
                     "psnr_rgb": c.psnr(asli, noisy), "ssim_rgb": float(
                         c.structural_similarity(asli, noisy, data_range=255, channel_axis=2))})
        for m in ["Mean", "Gaussian", "Median"]:
            out = np.stack([c.filter_cv(noisy[..., ch], m, 5) for ch in range(3)], axis=-1)
            rows.append({"berkas": r["berkas"], "noise": kode, "metode": m, "kernel": 5,
                         "psnr_rgb": c.psnr(asli, out), "ssim_rgb": float(
                             c.structural_similarity(asli, out, data_range=255, channel_axis=2))})
W = pd.DataFrame(rows)
W.to_csv(os.path.join(OUT, "hasil_warna.csv"), index=False)
W.groupby(["noise", "metode"])[["psnr_rgb", "ssim_rgb"]].mean().reset_index().to_csv(
    os.path.join(OUT, "ringkasan_warna.csv"), index=False)

# --------------------------------------------------------------------- cetak ringkas
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
print("== TERBAIK ==")
print(best[["noise", "metode", "kernel", "psnr", "psnr_sd", "ssim", "tepi_f1", "psnr_noisy",
            "gain_psnr"]].round(3).to_string(index=False))
print("\n== BASELINE ==")
print(base[["psnr", "ssim", "mae", "tepi_f1"]].round(3))
print("\n== WILCOXON ==")
print(pd.read_csv(os.path.join(OUT, "uji_wilcoxon.csv")).round(4).to_string(index=False))
print("\n== KEGAGALAN (terbaik) ==")
print(pd.read_csv(os.path.join(OUT, "ringkasan_kegagalan.csv")).round(2).to_string(index=False))
print("\n== KEGAGALAN k=5 ==")
print(pd.read_csv(os.path.join(OUT, "kegagalan_k5.csv")).round(2).to_string(index=False))
print("\n== KORELASI ==")
print(pd.read_csv(os.path.join(OUT, "korelasi_detail_gain.csv")).round(3).to_string(index=False))
print("\n== WARNA ==")
print(pd.read_csv(os.path.join(OUT, "ringkasan_warna.csv")).round(3).to_string(index=False))
