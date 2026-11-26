"""
citra.py - Pustaka inti Tugas Akhir Pengolahan Citra 125
Perbandingan Mean, Gaussian, dan Median Filter untuk Reduksi Noise.

Isi:
  - pemuatan dataset + metadata
  - prapemrosesan (RGB -> grayscale, normalisasi, resize)
  - pembangkit noise (Gaussian aditif, salt-and-pepper)
  - filter: implementasi OpenCV (cepat) dan implementasi dari nol (NumPy)
  - metrik: PSNR, SSIM, MAE, F1 tepi (Canny)
"""
import glob
import os
import time

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# --------------------------------------------------------------------------
# 1. Dataset
# --------------------------------------------------------------------------
SUMBER = {
    "BSD68": "Berkeley Segmentation Dataset (BSDS300), subset uji 68 citra (Martin dkk., 2001)",
    "Set12": "Set12, 12 citra uji baku denoising (Zhang dkk., 2017)",
    "Kodak24": "Kodak Lossless True Color Image Suite, 24 citra (Eastman Kodak)",
}


def daftar_citra(root):
    """Mengembalikan daftar (kelompok, nama_berkas, path) terurut."""
    hasil = []
    for kel in ["BSD68", "Set12", "Kodak24"]:
        for p in sorted(glob.glob(os.path.join(root, kel, "*.png"))):
            hasil.append((kel, os.path.basename(p), p))
    return hasil


def muat_asli(path):
    """Membaca citra apa adanya (tanpa konversi) beserta metadata."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Gagal membaca {path}")
    meta = {
        "tinggi": img.shape[0],
        "lebar": img.shape[1],
        "kanal": 1 if img.ndim == 2 else img.shape[2],
        "dtype": str(img.dtype),
        "ukuran_berkas_kb": round(os.path.getsize(path) / 1024, 1),
    }
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, meta


# --------------------------------------------------------------------------
# 2. Prapemrosesan
# --------------------------------------------------------------------------
def ke_grayscale(img):
    """RGB -> grayscale (bobot ITU-R BT.601: 0,299 R + 0,587 G + 0,114 B)."""
    if img.ndim == 2:
        return img.copy()
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def normalisasi(gray_u8):
    """uint8 [0,255] -> float32 [0,1]."""
    return gray_u8.astype(np.float32) / 255.0


def denormalisasi(x):
    """float [0,1] -> uint8 (clip + pembulatan)."""
    return np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8)


def ubah_ukuran(img, sisi):
    """Resize ke sisi x sisi (INTER_AREA untuk mengecil, INTER_LINEAR untuk membesar)."""
    h, w = img.shape[:2]
    inter = cv2.INTER_AREA if sisi < min(h, w) else cv2.INTER_LINEAR
    return cv2.resize(img, (sisi, sisi), interpolation=inter)


# --------------------------------------------------------------------------
# 3. Pembangkit noise
# --------------------------------------------------------------------------
def noise_gaussian(gray_u8, sigma, rng):
    """Noise Gaussian aditif, rerata 0, simpangan baku `sigma` (skala 0-255)."""
    x = normalisasi(gray_u8)
    x = x + rng.normal(0.0, sigma / 255.0, size=x.shape).astype(np.float32)
    return denormalisasi(x)


def noise_salt_pepper(gray_u8, kerapatan, rng):
    """Salt-and-pepper: `kerapatan` = proporsi piksel rusak (separuh 0, separuh 255)."""
    out = gray_u8.copy()
    r = rng.random(out.shape)
    out[r < kerapatan / 2] = 0
    out[(r >= kerapatan / 2) & (r < kerapatan)] = 255
    return out


KONDISI_NOISE = [
    ("G10", "Gaussian", 10),
    ("G25", "Gaussian", 25),
    ("G50", "Gaussian", 50),
    ("SP05", "Salt-pepper", 0.05),
    ("SP15", "Salt-pepper", 0.15),
    ("SP30", "Salt-pepper", 0.30),
]


def beri_noise(gray_u8, kode, rng):
    for k, jenis, par in KONDISI_NOISE:
        if k == kode:
            if jenis == "Gaussian":
                return noise_gaussian(gray_u8, par, rng)
            return noise_salt_pepper(gray_u8, par, rng)
    raise KeyError(kode)


# --------------------------------------------------------------------------
# 4. Filter - implementasi OpenCV
# --------------------------------------------------------------------------
def filter_cv(img, metode, k):
    if metode == "Mean":
        return cv2.blur(img, (k, k), borderType=cv2.BORDER_REFLECT_101)
    if metode == "Gaussian":
        # sigma=0 -> OpenCV memakai sigma = 0.3*((k-1)*0.5 - 1) + 0.8
        return cv2.GaussianBlur(img, (k, k), 0, borderType=cv2.BORDER_REFLECT_101)
    if metode == "Median":
        return cv2.medianBlur(img, k)
    raise KeyError(metode)


def sigma_otomatis(k):
    """Rumus sigma OpenCV untuk k > 7 (saat sigma=0): 0.3*((k-1)*0.5 - 1) + 0.8."""
    return 0.3 * ((k - 1) * 0.5 - 1) + 0.8


def kernel_gaussian_cv(k):
    """Kernel 1D yang benar-benar dipakai OpenCV saat sigma=0.
    Untuk k = 3, 5, 7 OpenCV memakai tabel koefisien tetap (mendekati binomial);
    untuk k > 7 memakai fungsi Gaussian dengan sigma_otomatis(k)."""
    return cv2.getGaussianKernel(k, 0).ravel()


def sigma_efektif(k):
    """Simpangan baku efektif kernel 1D yang dipakai OpenCV (sigma=0)."""
    w = kernel_gaussian_cv(k)
    x = np.arange(k) - (k - 1) / 2.0
    return float(np.sqrt(np.sum(w * x ** 2)))


# --------------------------------------------------------------------------
# 5. Filter - implementasi dari nol (NumPy), untuk verifikasi
# --------------------------------------------------------------------------
def kernel_gaussian(k, sigma):
    ax = np.arange(k) - (k - 1) / 2.0
    g = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    K = np.outer(g, g)
    return K / K.sum()


def filter_manual(img_u8, metode, k, sigma=None):
    """Konvolusi/rank-filter 2D manual.
    Perilaku batas disamakan dengan OpenCV agar dapat dibandingkan piksel demi piksel:
      Mean/Gaussian -> pantulan (BORDER_REFLECT_101 = np.pad 'reflect')
      Median        -> replikasi tepi (cv2.medianBlur hanya mendukung BORDER_REPLICATE)
    Gaussian tanpa `sigma` memakai kernel 1D yang sama dengan OpenCV (sigma=0)."""
    r = k // 2
    mode = "edge" if metode == "Median" else "reflect"
    pad = np.pad(img_u8.astype(np.float64), r, mode=mode)
    jendela = sliding_window_view(pad, (k, k))          # (H, W, k, k)
    if metode == "Mean":
        out = jendela.mean(axis=(2, 3))
    elif metode == "Gaussian":
        if sigma:
            K = kernel_gaussian(k, sigma)
        else:
            g1 = kernel_gaussian_cv(k)
            K = np.outer(g1, g1)
        out = np.einsum("hwij,ij->hw", jendela, K)
    elif metode == "Median":
        out = np.median(jendela.reshape(*jendela.shape[:2], -1), axis=2)
    else:
        raise KeyError(metode)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# 6. Metrik evaluasi
# --------------------------------------------------------------------------
def psnr(ref, uji):
    return float(peak_signal_noise_ratio(ref, uji, data_range=255))


def ssim(ref, uji):
    return float(structural_similarity(ref, uji, data_range=255))


def mae(ref, uji):
    return float(np.mean(np.abs(ref.astype(np.int16) - uji.astype(np.int16))))


def ambang_canny(ref):
    """Ambang Canny otomatis dari citra bersih: 0,66*median dan 1,33*median intensitas."""
    v = float(np.median(ref))
    return int(max(0, 0.66 * v)), int(min(255, 1.33 * v))


def f1_tepi(ref, uji, ambang):
    """F1 peta tepi Canny(uji) terhadap Canny(ref), toleransi 1 piksel."""
    lo, hi = ambang
    e_ref = cv2.Canny(ref, lo, hi) > 0
    e_uji = cv2.Canny(uji, lo, hi) > 0
    if e_ref.sum() == 0 or e_uji.sum() == 0:
        return 0.0, 0.0, 0.0
    se = np.ones((3, 3), np.uint8)
    d_ref = cv2.dilate(e_ref.astype(np.uint8), se) > 0
    d_uji = cv2.dilate(e_uji.astype(np.uint8), se) > 0
    presisi = (e_uji & d_ref).sum() / e_uji.sum()
    recall = (e_ref & d_uji).sum() / e_ref.sum()
    f1 = 0.0 if presisi + recall == 0 else 2 * presisi * recall / (presisi + recall)
    return float(presisi), float(recall), float(f1)


def kekuatan_detail(gray_u8):
    """Rerata magnitudo gradien Sobel (indikator tingkat detail/tekstur citra)."""
    gx = cv2.Sobel(gray_u8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_u8, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(np.hypot(gx, gy)))


def waktu_ms(fungsi, ulang=5):
    """Median waktu eksekusi (ms) dari `ulang` pengulangan."""
    t = []
    for _ in range(ulang):
        a = time.perf_counter()
        fungsi()
        t.append((time.perf_counter() - a) * 1000.0)
    return float(np.median(t))
