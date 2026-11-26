"""buat_xlsx.py - Dokumentasi hasil eksperimen ke satu berkas Excel (gaya konsisten)."""
import os
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OUT = os.path.join(R, "dokumentasi_hasil_eksperimen.xlsx")
HEAD = PatternFill("solid", fgColor="1F4E5F"); ZEBRA = PatternFill("solid", fgColor="F2F6F8")
THIN = Side(style="thin", color="C9D3D9")
BORD = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
rd = lambda n: pd.read_csv(os.path.join(R, n))

wb = Workbook(); wb.remove(wb.active)

def sheet(nama, df, judul, catatan=None, fmt=None, lebar=None):
    ws = wb.create_sheet(nama)
    ws["A1"] = judul; ws["A1"].font = Font(name="Arial", size=13, bold=True, color="1F4E5F")
    r0 = 3
    if catatan:
        ws["A2"] = catatan; ws["A2"].font = Font(name="Arial", size=9, italic=True, color="555555")
    for j, col in enumerate(df.columns, 1):
        c = ws.cell(r0, j, col); c.fill = HEAD; c.border = BORD
        c.font = Font(name="Arial", size=10, bold=True, color="FFFFFF")
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for i, row in enumerate(df.itertuples(index=False), r0 + 1):
        for j, v in enumerate(row, 1):
            if hasattr(v, "item"):
                v = v.item()
            c = ws.cell(i, j, v); c.border = BORD; c.font = Font(name="Arial", size=10)
            if (i - r0) % 2 == 0:
                c.fill = ZEBRA
            if isinstance(v, float):
                c.number_format = (fmt or {}).get(df.columns[j - 1], "0.000")
                c.alignment = Alignment(horizontal="right")
    for j, col in enumerate(df.columns, 1):
        w = (lebar or {}).get(col, max(10, min(26, len(str(col)) + 4)))
        ws.column_dimensions[get_column_letter(j)].width = w
    ws.freeze_panes = ws.cell(r0 + 1, 1)
    ws.row_dimensions[r0].height = 30
    return ws

# 1. Ringkasan terbaik
b = rd("ringkasan_terbaik.csv")[["noise", "metode", "kernel", "psnr", "psnr_sd", "ssim", "mae", "tepi_f1",
                                  "psnr_noisy", "ssim_noisy", "tepi_f1_noisy", "gain_psnr"]]
b.columns = ["Noise", "Metode", "k terbaik", "PSNR (dB)", "SD PSNR", "SSIM", "MAE", "F1 tepi",
             "PSNR tanpa filter", "SSIM tanpa filter", "F1 tepi tanpa filter", "Peningkatan PSNR (dB)"]
sheet("Ringkasan terbaik", b, "Konfigurasi terbaik (PSNR) per kondisi noise dan metode",
      "Rerata atas 104 citra; k terbaik dipilih dari {3,5,7,9} berdasarkan PSNR rerata.",
      {"PSNR (dB)": "0.00", "SD PSNR": "0.00", "PSNR tanpa filter": "0.00", "Peningkatan PSNR (dB)": "0.00"})
# 2. Semua konfigurasi
a = rd("ringkasan_semua_konfigurasi.csv")[["noise", "metode", "kernel", "psnr", "psnr_sd", "ssim", "ssim_sd", "mae",
                                            "tepi_presisi", "tepi_recall", "tepi_f1", "n"]]
a.columns = ["Noise", "Metode", "k", "PSNR", "SD PSNR", "SSIM", "SD SSIM", "MAE", "Presisi tepi", "Recall tepi", "F1 tepi", "n citra"]
sheet("Semua konfigurasi", a, "Seluruh konfigurasi (6 noise × 3 metode × 4 kernel + baseline)", None, {"PSNR": "0.00", "SD PSNR": "0.00"})
# 3. Wilcoxon
w = rd("uji_wilcoxon.csv"); w.columns = ["Noise", "Metode A", "Metode B", "Selisih PSNR rerata (A−B)", "A lebih baik pada (% citra)", "p (Wilcoxon)"]
sheet("Uji Wilcoxon", w, "Uji peringkat bertanda Wilcoxon berpasangan pada konfigurasi k terbaik", "n = 104 pasangan citra.",
      {"Selisih PSNR rerata (A−B)": "0.00", "A lebih baik pada (% citra)": "0.0", "p (Wilcoxon)": "0.0E+00"}, {"Metode A": 18, "Metode B": 18, "Selisih PSNR rerata (A−B)": 18, "A lebih baik pada (% citra)": 18})
# 4. Kegagalan
k = rd("ringkasan_kegagalan.csv"); k.columns = ["Noise", "Metode", "k terbaik", "Peningkatan PSNR rerata", "Peningkatan min", "Peningkatan maks", "Citra gagal", "% citra gagal"]
sheet("Kegagalan (k terbaik)", k, "Citra yang PSNR-nya turun setelah filtering (konfigurasi k terbaik)", None, {"Peningkatan PSNR rerata": "0.00", "Peningkatan min": "0.00", "Peningkatan maks": "0.00", "% citra gagal": "0.0"})
k5 = rd("kegagalan_k5.csv"); k5.columns = ["Noise", "Metode", "k", "Citra gagal", "% citra gagal", "Peningkatan PSNR rerata"]
sheet("Kegagalan (k=5)", k5, "Citra yang PSNR-nya turun pada kernel tetap k = 5", None, {"% citra gagal": "0.0", "Peningkatan PSNR rerata": "0.00"})
# 5. Waktu
wp = rd("waktu_proses.csv"); wp["ms_per_mp"] = wp.waktu_ms / wp.mp
wt = wp.groupby(["metode", "kernel"]).agg(waktu_ms=("waktu_ms", "mean"), ms_per_mp=("ms_per_mp", "mean")).reset_index()
wt.columns = ["Metode", "k", "Waktu rerata per citra (ms)", "Waktu per megapiksel (ms/MP)"]
sheet("Waktu proses", wt, "Waktu proses OpenCV pada 104 citra asli (CPU tunggal)", "Median dari 3 pengulangan per citra, lalu dirata-rata.", {"Waktu rerata per citra (ms)": "0.000", "Waktu per megapiksel (ms/MP)": "0.00"}, {"Waktu rerata per citra (ms)": 24, "Waktu per megapiksel (ms/MP)": 26})
ws_ = rd("waktu_skala.csv"); ws_.columns = ["Implementasi", "Sisi citra (px)", "Metode", "k", "Waktu (ms)"]
sheet("Skala waktu", ws_, "Waktu proses terhadap ukuran citra dan ukuran kernel", None, {"Waktu (ms)": "0.000"}, {"Implementasi": 18})
# 6. Verifikasi & sigma & warna & detail
v = rd("verifikasi_manual.csv").groupby(["metode", "kernel"]).agg(beda_maks=("beda_maks", "max"), beda_rerata=("beda_rerata", "mean")).reset_index()
v.columns = ["Metode", "k", "Selisih maksimum (tingkat abu)", "Selisih rerata (tingkat abu)"]
sheet("Verifikasi manual", v, "Implementasi manual NumPy vs OpenCV (6 citra uji, 128×128, noise G25)", None, {"Selisih rerata (tingkat abu)": "0.0000"}, {"Selisih maksimum (tingkat abu)": 28, "Selisih rerata (tingkat abu)": 28})
s = rd("sweep_sigma.csv").groupby("sigma").agg(psnr=("psnr", "mean"), sd=("psnr", "std"), ssim=("ssim", "mean")).reset_index()
s.columns = ["σ", "PSNR rerata (dB)", "SD PSNR", "SSIM rerata"]
sheet("Sweep sigma", s, "Pengaruh σ Gaussian (kernel 15×15) pada noise Gaussian σ=25", None, {"PSNR rerata (dB)": "0.00", "SD PSNR": "0.00"})
wr = rd("ringkasan_warna.csv"); wr.columns = ["Noise", "Metode", "PSNR RGB (dB)", "SSIM RGB"]
sheet("Citra warna", wr, "Filtering per kanal pada 24 citra Kodak (k=5)", None, {"PSNR RGB (dB)": "0.00"})
dt = rd("ringkasan_detail_terbaik.csv"); dt.columns = ["Noise", "Tingkat detail", "Metode", "k terbaik", "PSNR (dB)", "SSIM"]
sheet("Per tingkat detail", dt, "Konfigurasi terbaik per tertil tingkat detail citra", None, {"PSNR (dB)": "0.00"})
kr = rd("korelasi_detail_gain.csv"); kr.columns = ["Noise", "Metode", "Spearman ρ (detail vs peningkatan PSNR)", "p"]
sheet("Korelasi detail", kr, "Korelasi peringkat Spearman antara tingkat detail dan peningkatan PSNR", None, {"p": "0.0E+00"}, {"Spearman ρ (detail vs peningkatan PSNR)": 36})
m = rd("metadata_dataset.csv"); m.columns = ["ID", "Kelompok", "Berkas", "Tinggi (px)", "Lebar (px)", "Kanal", "Tipe data", "Ukuran berkas (KB)", "Intensitas rerata", "SD intensitas", "Detail (Sobel)"]
sheet("Metadata dataset", m, "Metadata 104 citra yang dipakai", "Sumber: BSD68 (Berkeley BSDS300), Set12 (citra uji baku denoising), Kodak24 (Eastman Kodak); lihat Daftar Pustaka laporan.", {"Ukuran berkas (KB)": "0.0"})
wb.save(OUT); print("OK", OUT)
