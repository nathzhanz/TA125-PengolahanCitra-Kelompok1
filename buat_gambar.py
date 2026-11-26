"""
buat_gambar.py - Membuat seluruh gambar/grafik laporan ke folder figures/.
Jalankan setelah run_experiment.py dan analyze.py.
"""
import os
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
import citra as c  # noqa: E402

FIG = os.path.join(ROOT, "figures")
RES = os.path.join(ROOT, "results")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titlesize": 10, "axes.titleweight": "bold",
    "savefig.dpi": 200, "savefig.bbox": "tight", "figure.facecolor": "white",
})
COL = {"Mean": "#D9822B", "Gaussian": "#1F7A99", "Median": "#7A4FA3", "Tanpa filter": "#9AA0A8"}
METODE = ["Mean", "Gaussian", "Median"]
KODE = [k for k, _, _ in c.KONDISI_NOISE]
NAMA = {"G10": "Gaussian σ=10", "G25": "Gaussian σ=25", "G50": "Gaussian σ=50",
        "SP05": "Salt-pepper 5%", "SP15": "Salt-pepper 15%", "SP30": "Salt-pepper 30%"}
SEED = 2026


def idn(x, nd=2):
    return f"{x:.{nd}f}".replace(".", ",")


FMT = FuncFormatter(lambda v, p: f"{v:g}".replace(".", ","))


def simpan(fig, nama):
    fig.savefig(os.path.join(FIG, nama))
    plt.close(fig)
    print("  ->", nama, flush=True)


H = pd.read_csv(os.path.join(RES, "hasil_utama.csv"))
M = pd.read_csv(os.path.join(RES, "metadata_dataset.csv"))
BEST = pd.read_csv(os.path.join(RES, "ringkasan_terbaik.csv"))
AGG = pd.read_csv(os.path.join(RES, "ringkasan_semua_konfigurasi.csv"))
DAFTAR = c.daftar_citra(os.path.join(ROOT, "data"))
GRAY = {}


def gray(i):
    if i not in GRAY:
        asli, _ = c.muat_asli(DAFTAR[i][2])
        GRAY[i] = c.ke_grayscale(asli)
    return GRAY[i]


def id_of(nama):
    return [i for i, d in enumerate(DAFTAR) if d[1] == nama][0]


def noisy_of(i, kode):
    ci = KODE.index(kode)
    return c.beri_noise(gray(i), kode, np.random.default_rng(SEED * 1000 + i * 10 + ci))


def bestk(kode, met):
    return int(BEST[(BEST.noise == kode) & (BEST.metode == met)].kernel.iloc[0])


# =============================================================== 1. contoh dataset
def fig_dataset():
    pilih = ["test001.png", "test020.png", "test045.png", "test062.png",
             "01.png", "09.png", "kodim01.png", "kodim20.png"]
    fig, ax = plt.subplots(2, 4, figsize=(9, 5))
    for a, nm in zip(ax.ravel(), pilih):
        i = id_of(nm)
        asli, meta = c.muat_asli(DAFTAR[i][2])
        a.imshow(asli, cmap="gray", vmin=0, vmax=255) if asli.ndim == 2 else a.imshow(asli)
        a.set_title(f"{DAFTAR[i][0]} / {nm}", fontsize=8, fontweight="normal")
        a.set_xlabel(f"{meta['lebar']}×{meta['tinggi']} px, {meta['kanal']} kanal", fontsize=7)
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
    fig.tight_layout()
    simpan(fig, "fig01_contoh_dataset.png")


# ============================================================ 2. statistik dataset
def fig_statistik():
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.1))
    warna = {"BSD68": "#1F7A99", "Set12": "#D9822B", "Kodak24": "#7A4FA3"}
    for kel, g in M.groupby("kelompok"):
        ax[0].scatter(g.lebar + np.random.default_rng(1).normal(0, 3, len(g)), g.tinggi,
                      s=18, alpha=.7, label=f"{kel} (n={len(g)})", color=warna[kel])
        ax[1].hist(g.detail_sobel, bins=np.linspace(0, 180, 19), alpha=.65, color=warna[kel],
                   label=kel)
        ax[2].scatter(g.intensitas_rerata, g.kontras_std, s=18, alpha=.7, color=warna[kel])
    ax[0].set_xlabel("Lebar (piksel)"); ax[0].set_ylabel("Tinggi (piksel)")
    ax[0].set_title("(a) Resolusi citra"); ax[0].legend(fontsize=7, frameon=False)
    ax[1].set_xlabel("Rerata magnitudo gradien Sobel"); ax[1].set_ylabel("Jumlah citra")
    ax[1].set_title("(b) Tingkat detail"); ax[1].legend(fontsize=7, frameon=False)
    ax[2].set_xlabel("Intensitas rerata (0–255)"); ax[2].set_ylabel("Simpangan baku intensitas")
    ax[2].set_title("(c) Kecerahan dan kontras")
    for a in ax:
        a.xaxis.set_major_formatter(FMT); a.yaxis.set_major_formatter(FMT)
    fig.tight_layout()
    simpan(fig, "fig02_statistik_dataset.png")


# ================================================================ 3. prapemrosesan
def fig_praproses():
    i = id_of("kodim01.png")
    asli, meta = c.muat_asli(DAFTAR[i][2])
    g = c.ke_grayscale(asli)
    fig = plt.figure(figsize=(10, 5.4))
    gs = fig.add_gridspec(2, 4, hspace=.35, wspace=.25)
    ax = [fig.add_subplot(gs[0, j]) for j in range(4)]
    ax[0].imshow(asli); ax[0].set_title("Citra RGB asli")
    for j, (nm, cm) in enumerate([("Kanal R", "Reds"), ("Kanal G", "Greens"), ("Kanal B", "Blues")]):
        ax[j + 1].imshow(asli[..., j], cmap=cm, vmin=0, vmax=255); ax[j + 1].set_title(nm)
    for a in ax:
        a.axis("off")
    a_g = fig.add_subplot(gs[1, 0]); a_g.imshow(g, cmap="gray", vmin=0, vmax=255)
    a_g.set_title("Grayscale (BT.601)"); a_g.axis("off")
    a_h = fig.add_subplot(gs[1, 1:3])
    for j, cl in enumerate(["#C0392B", "#2E8B57", "#2C6FBB"]):
        a_h.hist(asli[..., j].ravel(), bins=64, range=(0, 255), histtype="step", color=cl,
                 lw=1.2, label="RGB"[j])
    a_h.hist(g.ravel(), bins=64, range=(0, 255), histtype="step", color="k", lw=1.6,
             label="Gray")
    a_h.set_title("Histogram kanal dan grayscale"); a_h.set_xlabel("Intensitas (0–255)")
    a_h.set_ylabel("Jumlah piksel"); a_h.legend(frameon=False, ncol=4, fontsize=8)
    a_n = fig.add_subplot(gs[1, 3])
    x = c.normalisasi(g)
    a_n.hist(x.ravel(), bins=64, range=(0, 1), color="#9AA0A8")
    a_n.set_title("Setelah normalisasi [0,1]"); a_n.set_xlabel("Intensitas ternormalisasi")
    a_n.xaxis.set_major_formatter(FMT)
    fig.suptitle(f"Prapemrosesan pada kodim01.png ({meta['lebar']}×{meta['tinggi']} px, "
                 f"{meta['kanal']} kanal, {meta['dtype']})", fontsize=10, y=.98)
    simpan(fig, "fig03_praproses.png")


# ==================================================================== 4. pipeline
def fig_pipeline():
    kot = ["Citra masukan\n\n104 citra\n(BSD68, Set12,\nKodak24)",
           "Prapemrosesan\n\nRGB → grayscale\nnormalisasi [0,1]\nciti acuan bersih",
           "Injeksi noise\n\nGaussian\nσ = 10, 25, 50\nSalt-pepper\np = 5, 15, 30%",
           "Filtering\n\nMean\nGaussian\nMedian\nk = 3, 5, 7, 9",
           "Evaluasi\n\nPSNR, SSIM, MAE\nF1 tepi (Canny)\nwaktu proses",
           "Analisis\n\nuji Wilcoxon\nanalisis kegagalan\npengaruh detail"]
    kot[1] = kot[1].replace("citi acuan bersih", "acuan bersih")
    war = ["#E9EEF2", "#D8ECF3", "#F7E3CF", "#E4D8F0", "#D9EFE0", "#F3DCDC"]
    fig, ax = plt.subplots(figsize=(10.6, 3.0))
    ax.set_xlim(0, 10.5); ax.set_ylim(0, 2.3); ax.axis("off")
    w, gap = 1.5, 0.2
    for n, (t, wr) in enumerate(zip(kot, war)):
        x0 = 0.1 + n * (w + gap)
        ax.add_patch(FancyBboxPatch((x0, .5), w, 1.3, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=wr, ec="#44505C", lw=1))
        ax.text(x0 + w / 2, 1.15, t, ha="center", va="center", fontsize=7.2, linespacing=1.25)
        ax.text(x0 + w / 2, 2.0, f"Tahap {n + 1}", ha="center", fontsize=8, color="#44505C",
                fontweight="bold")
        if n < 5:
            ax.add_patch(FancyArrowPatch((x0 + w + .02, 1.15), (x0 + w + gap - .02, 1.15),
                                         arrowstyle="-|>", mutation_scale=9, color="#44505C"))
    ax.text(5.25, .18, "Citra bersih (grayscale) menjadi acuan untuk seluruh metrik; seed acak tetap "
            "sehingga hasil dapat direproduksi.", ha="center", fontsize=7.4,
            color="#44505C", style="italic")
    simpan(fig, "fig04_pipeline.png")


# ================================================================ 5. contoh noise
def fig_noise():
    i = id_of("01.png")
    fig, ax = plt.subplots(2, 4, figsize=(9.2, 4.9))
    ref = gray(i)
    ax[0, 0].imshow(ref, cmap="gray", vmin=0, vmax=255); ax[0, 0].set_title("Citra bersih")
    for a, kode in zip(list(ax.ravel()[1:7]), KODE):
        n = noisy_of(i, kode)
        a.imshow(n, cmap="gray", vmin=0, vmax=255)
        a.set_title(f"{NAMA[kode]}\nPSNR {idn(c.psnr(ref, n))} dB", fontsize=8, fontweight="normal")
    for a in ax.ravel()[:7]:
        a.axis("off")
    h = ax[1, 3]
    for kode, cl in [("G25", COL["Gaussian"]), ("SP15", COL["Median"])]:
        e = noisy_of(i, kode).astype(int) - ref.astype(int)
        h.hist(e.ravel(), bins=np.arange(-255, 256, 5), color=cl, alpha=.7, label=NAMA[kode])
    h.set_yscale("log"); h.set_xlabel("Galat (noisy − bersih)"); h.set_ylabel("Jumlah piksel (log)")
    h.set_title("Distribusi galat", fontsize=9); h.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    simpan(fig, "fig05_contoh_noise.png")


# ================================================================ 6. kualitatif
def fig_kualitatif(nama_citra, kode, k, zoom, fname):
    i = id_of(nama_citra)
    ref = gray(i)
    n = noisy_of(i, kode)
    hasil = [("Citra ber-noise", n)] + [(f"{m} k={k}", c.filter_cv(n, m, k)) for m in METODE]
    y0, y1, x0, x1 = zoom
    fig, ax = plt.subplots(2, 4, figsize=(10, 5.6),
                           gridspec_kw={"height_ratios": [1.0, 1.0]})
    for j, (jud, im) in enumerate(hasil):
        ax[0, j].imshow(im, cmap="gray", vmin=0, vmax=255)
        ax[0, j].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#E53935", lw=1.3))
        ax[0, j].set_title(f"{jud}\nPSNR {idn(c.psnr(ref, im))} dB · SSIM {idn(c.ssim(ref, im), 3)}",
                           fontsize=8, fontweight="normal")
        ax[1, j].imshow(im[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax[0, j].axis("off"); ax[1, j].axis("off")
    fig.suptitle(f"{DAFTAR[i][0]}/{nama_citra} · {NAMA[kode]} · kotak merah diperbesar di baris bawah",
                 fontsize=9, y=.99)
    fig.tight_layout()
    simpan(fig, fname)


# ============================================================ 7-10. metrik per kondisi
def fig_bar(metrik, judul, ylabel, fname, ylim=None):
    fig, ax = plt.subplots(figsize=(10, 3.6))
    x = np.arange(len(KODE)); w = 0.2
    series = ["Tanpa filter"] + METODE
    for s_i, m in enumerate(series):
        mean, sd, kk = [], [], []
        for kode in KODE:
            if m == "Tanpa filter":
                sub = H[(H.noise == kode) & (H.metode == m)]
                kk.append(None)
            else:
                k = bestk(kode, m); kk.append(k)
                sub = H[(H.noise == kode) & (H.metode == m) & (H.kernel == k)]
            mean.append(sub[metrik].mean()); sd.append(sub[metrik].std())
        pos = x + (s_i - 1.5) * w
        ax.bar(pos, mean, w, yerr=sd, color=COL[m], label=m, capsize=2, error_kw={"lw": .8})
        for xp, mv, k in zip(pos, mean, kk):
            if k:
                ax.text(xp, mv * 0.5 if metrik != "psnr" else mv * 0.5, f"k={k}", ha="center",
                        va="center", fontsize=6.5, color="white", rotation=90)
    ax.set_xticks(x); ax.set_xticklabels([NAMA[k] for k in KODE], fontsize=8)
    ax.set_ylabel(ylabel); ax.set_title(judul)
    if ylim:
        ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(FMT)
    ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    simpan(fig, fname)


def fig_heatmap():
    fig, ax = plt.subplots(2, 3, figsize=(10, 4.6))
    for a, kode in zip(ax.ravel(), KODE):
        sub = AGG[(AGG.noise == kode) & (AGG.metode.isin(METODE))]
        pv = sub.pivot(index="metode", columns="kernel", values="psnr").loc[METODE]
        a.imshow(pv.values, cmap="YlGnBu", aspect="auto")
        for r in range(3):
            for q in range(pv.shape[1]):
                a.text(q, r, idn(pv.values[r, q], 1), ha="center", va="center", fontsize=8,
                       color="white" if pv.values[r, q] > pv.values.mean() + 0.6 * pv.values.std()
                       else "#222")
        r_b, q_b = np.unravel_index(pv.values.argmax(), pv.values.shape)
        a.add_patch(Rectangle((q_b - .5, r_b - .5), 1, 1, fill=False, ec="#E53935", lw=2))
        a.set_xticks(range(pv.shape[1])); a.set_xticklabels([f"k={k}" for k in pv.columns], fontsize=8)
        a.set_yticks(range(3)); a.set_yticklabels(METODE, fontsize=8)
        a.set_title(NAMA[kode], fontsize=9)
        for s in a.spines.values():
            s.set_visible(False)
    fig.suptitle("PSNR rerata (dB) per metode dan ukuran kernel; kotak merah = terbaik", fontsize=10)
    fig.tight_layout()
    simpan(fig, "fig09_heatmap_psnr.png")


def fig_garis_kernel():
    fig, ax = plt.subplots(2, 3, figsize=(10, 5), sharex=True)
    for a, kode in zip(ax.ravel(), KODE):
        base = AGG[(AGG.noise == kode) & (AGG.metode == "Tanpa filter")].psnr.iloc[0]
        for m in METODE:
            s = AGG[(AGG.noise == kode) & (AGG.metode == m)].sort_values("kernel")
            a.plot(s.kernel, s.psnr, "-o", color=COL[m], label=m, ms=4)
        a.axhline(base, color=COL["Tanpa filter"], ls="--", lw=1, label="Tanpa filter")
        a.set_title(NAMA[kode], fontsize=9); a.set_xticks([3, 5, 7, 9])
        a.yaxis.set_major_formatter(FMT)
    for a in ax[1]:
        a.set_xlabel("Ukuran kernel k")
    for a in ax[:, 0]:
        a.set_ylabel("PSNR rerata (dB)")
    ax[0, 0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    simpan(fig, "fig10_pengaruh_kernel.png")


# ===================================================================== waktu
def fig_waktu():
    S = pd.read_csv(os.path.join(RES, "waktu_skala.csv"))
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.3))
    s = S[(S.implementasi == "OpenCV") & (S.sisi == 1024)]
    for m in METODE:
        q = s[s.metode == m].sort_values("kernel")
        ax[0].plot(q.kernel, q.waktu_ms, "-o", color=COL[m], label=m, ms=4)
    ax[0].set_yscale("log"); ax[0].set_xlabel("Ukuran kernel k"); ax[0].set_ylabel("Waktu (ms, log)")
    ax[0].set_title("(a) OpenCV, citra 1024×1024"); ax[0].legend(frameon=False, fontsize=8)
    s = S[(S.implementasi == "OpenCV") & (S.kernel == 5)]
    for m in METODE:
        q = s[s.metode == m].sort_values("sisi")
        ax[1].plot(q.sisi, q.waktu_ms, "-o", color=COL[m], label=m, ms=4)
    ax[1].set_xscale("log", base=2); ax[1].set_yscale("log")
    ax[1].set_xlabel("Sisi citra (piksel)"); ax[1].set_title("(b) OpenCV, k=5")
    s = S[(S.implementasi == "Manual (NumPy)")]
    for m in METODE:
        q = s[s.metode == m].sort_values("kernel")
        ax[2].plot(q.kernel, q.waktu_ms, "-s", color=COL[m], label=m, ms=4)
    ax[2].set_xlabel("Ukuran kernel k"); ax[2].set_ylabel("Waktu (ms)")
    ax[2].set_title("(c) Implementasi manual NumPy, 128×128")
    for a in ax:
        a.grid(alpha=.25, lw=.5)
    fig.tight_layout()
    simpan(fig, "fig11_waktu_proses.png")


def fig_sigma():
    S = pd.read_csv(os.path.join(RES, "sweep_sigma.csv"))
    g = S.groupby("sigma").agg(psnr=("psnr", "mean"), sd=("psnr", "std"), ssim=("ssim", "mean"))
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax[0].errorbar(g.index, g.psnr, yerr=g.sd, fmt="-o", color=COL["Gaussian"], capsize=3, ms=4)
    ax[0].set_xlabel("σ Gaussian (kernel 15×15)"); ax[0].set_ylabel("PSNR rerata (dB)")
    ax[1].plot(g.index, g.ssim, "-o", color=COL["Gaussian"], ms=4)
    ax[1].set_xlabel("σ Gaussian (kernel 15×15)"); ax[1].set_ylabel("SSIM rerata")
    for a in ax:
        a.xaxis.set_major_formatter(FMT); a.yaxis.set_major_formatter(FMT); a.grid(alpha=.25)
    fig.suptitle("Pengaruh σ pada noise Gaussian σ=25 (104 citra)", fontsize=10)
    fig.tight_layout()
    simpan(fig, "fig12_sweep_sigma.png")


# ==================================================================== analisis
def fig_detail():
    Hd = H.merge(M[["id", "detail_sobel"]], on="id")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5), sharey=False)
    for a, kode in zip(ax, ["G25", "SP15"]):
        sub = Hd[Hd.noise == kode]
        nz = sub[sub.metode == "Tanpa filter"].set_index("id")["psnr"]
        for m in METODE:
            k = bestk(kode, m)
            s = sub[(sub.metode == m) & (sub.kernel == k)].set_index("id")
            gain = s.psnr - nz
            rho = stats.spearmanr(s.detail_sobel, gain)[0]
            a.scatter(s.detail_sobel, gain, s=14, alpha=.7, color=COL[m],
                      label=f"{m} k={k} (ρ={idn(rho)})")
        a.axhline(0, color="k", lw=.8)
        a.set_xlabel("Tingkat detail citra (rerata gradien Sobel)")
        a.set_ylabel("Peningkatan PSNR (dB)"); a.set_title(NAMA[kode])
        a.legend(frameon=False, fontsize=7)
        a.xaxis.set_major_formatter(FMT); a.yaxis.set_major_formatter(FMT)
    fig.tight_layout()
    simpan(fig, "fig13_detail_vs_gain.png")


def fig_tertil():
    D = pd.read_csv(os.path.join(RES, "ringkasan_detail_terbaik.csv"))
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.3))
    for a, kode in zip(ax, ["G25", "SP15", "SP30"]):
        x = np.arange(3); w = .26
        for j, m in enumerate(METODE):
            v = [D[(D.noise == kode) & (D.metode == m) & (D.detail == t)].psnr.iloc[0]
                 for t in ["Rendah", "Sedang", "Tinggi"]]
            a.bar(x + (j - 1) * w, v, w, color=COL[m], label=m)
        a.set_xticks(x); a.set_xticklabels(["Detail rendah", "Detail sedang", "Detail tinggi"], fontsize=8)
        a.set_title(NAMA[kode], fontsize=9); a.set_ylabel("PSNR terbaik (dB)")
        a.yaxis.set_major_formatter(FMT)
    ax[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    simpan(fig, "fig14_per_tingkat_detail.png")


def fig_kegagalan():
    # (a) kasus terburuk Gaussian k=3 pada G10
    sub = H[(H.noise == "G10")]
    nz = sub[sub.metode == "Tanpa filter"].set_index("id")["psnr"]
    g3 = sub[(sub.metode == "Gaussian") & (sub.kernel == 3)].set_index("id")["psnr"] - nz
    i = int(g3.idxmin())
    ref, n = gray(i), noisy_of(i, "G10")
    hasil = [("Citra ber-noise", n)] + [(f"{m} k=3", c.filter_cv(n, m, 3)) for m in METODE]
    H_, W_ = ref.shape
    y0, x0 = H_ // 2 - 40, W_ // 2 - 40
    fig, ax = plt.subplots(2, 4, figsize=(10, 5.4))
    for j, (jud, im) in enumerate(hasil):
        ax[0, j].imshow(im, cmap="gray", vmin=0, vmax=255)
        ax[0, j].add_patch(Rectangle((x0, y0), 80, 80, fill=False, ec="#E53935", lw=1.3))
        ax[0, j].set_title(f"{jud}\nPSNR {idn(c.psnr(ref, im))} dB", fontsize=8, fontweight="normal")
        ax[1, j].imshow(im[y0:y0 + 80, x0:x0 + 80], cmap="gray", vmin=0, vmax=255,
                        interpolation="nearest")
        ax[0, j].axis("off"); ax[1, j].axis("off")
    fig.suptitle(f"Kegagalan 1: {DAFTAR[i][0]}/{DAFTAR[i][1]} (detail tinggi) · {NAMA['G10']} — "
                 f"filter menurunkan PSNR", fontsize=9, y=.99)
    fig.tight_layout()
    simpan(fig, "fig15_kegagalan_detail.png")
    # (b) median pada SP30, k=3..9
    i = id_of("02.png"); ref = gray(i); n = noisy_of(i, "SP30")
    hasil = [("Citra ber-noise", n)] + [(f"Median k={k}", c.filter_cv(n, "Median", k)) for k in [3, 5, 9]]
    fig, ax = plt.subplots(1, 4, figsize=(10, 2.9))
    for a, (jud, im) in zip(ax, hasil):
        a.imshow(im, cmap="gray", vmin=0, vmax=255)
        a.set_title(f"{jud}\nPSNR {idn(c.psnr(ref, im))} dB", fontsize=8, fontweight="normal")
        a.axis("off")
    fig.suptitle(f"Kegagalan 2: Median pada {NAMA['SP30']} — k kecil menyisakan bintik, k besar mengaburkan detail",
                 fontsize=9)
    fig.tight_layout()
    simpan(fig, "fig16_kegagalan_median_sp30.png")


# ===================================================================== warna
def fig_warna():
    i = id_of("kodim01.png")
    asli, _ = c.muat_asli(DAFTAR[i][2])
    fig, ax = plt.subplots(2, 5, figsize=(11.5, 4.6))
    for ci, kode in enumerate(["G25", "SP15"]):
        rng = np.random.default_rng(SEED * 7 + i * 10 + ci)
        n = np.stack([c.beri_noise(asli[..., ch], kode, rng) for ch in range(3)], axis=-1)
        row = [("RGB asli", asli), (f"Ber-noise {NAMA[kode]}", n)] + [
            (f"{m} k=5", np.stack([c.filter_cv(n[..., ch], m, 5) for ch in range(3)], -1))
            for m in METODE]
        for j, (jud, im) in enumerate(row):
            ax[ci, j].imshow(im[100:300, 150:350], interpolation="nearest")
            ttl = jud if j == 0 else f"{jud}\nPSNR {idn(c.psnr(asli, im))} dB"
            ax[ci, j].set_title(ttl, fontsize=7.5, fontweight="normal")
            ax[ci, j].axis("off")
    fig.suptitle("Filtering per kanal pada citra berwarna (kodim01, potongan 200×200 piksel)",
                 fontsize=9.5, y=1.0)
    fig.tight_layout()
    simpan(fig, "fig17_citra_warna.png")


if __name__ == "__main__":
    fig_dataset(); fig_statistik(); fig_praproses(); fig_pipeline(); fig_noise()
    fig_kualitatif("01.png", "G25", 3, (20, 100, 60, 140), "fig06a_kualitatif_G25_cameraman.png")
    fig_kualitatif("01.png", "SP15", 3, (20, 100, 60, 140), "fig06b_kualitatif_SP15_cameraman.png")
    fig_kualitatif("09.png", "G25", 5, (300, 400, 340, 440), "fig06c_kualitatif_G25_barbara.png")
    fig_kualitatif("09.png", "SP15", 5, (300, 400, 340, 440), "fig06d_kualitatif_SP15_barbara.png")
    fig_bar("psnr", "PSNR rerata pada konfigurasi k terbaik per metode (garis galat = simpangan baku antar citra)",
            "PSNR (dB)", "fig07_psnr.png", ylim=(0, 36))
    fig_bar("ssim", "SSIM rerata pada konfigurasi k terbaik per metode", "SSIM", "fig08_ssim.png",
            ylim=(0, 1.05))
    fig_bar("tepi_f1", "F1 peta tepi (Canny) pada konfigurasi k terbaik per metode", "F1 tepi",
            "fig08b_tepi_f1.png", ylim=(0, 1.1))
    fig_heatmap(); fig_garis_kernel(); fig_waktu(); fig_sigma(); fig_detail(); fig_tertil()
    fig_kegagalan(); fig_warna()
    print("Semua gambar selesai.")
