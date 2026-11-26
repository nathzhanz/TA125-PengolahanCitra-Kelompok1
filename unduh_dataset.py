"""
unduh_dataset.py - Mengunduh dataset publik yang dipakai pada proyek ini.

Sumber (semua citra dipakai hanya untuk keperluan pembelajaran/penelitian non-komersial):
  BSD68   : subset uji 68 citra dari Berkeley Segmentation Dataset (BSDS300).
            Martin dkk., ICCV 2001. https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/
  Set12   : 12 citra uji baku denoising (Zhang dkk., IEEE TIP 2017).
  Kodak24 : Kodak Lossless True Color Image Suite (Eastman Kodak), 24 citra; versi
            potongan 500x500 piksel yang dipakai pada benchmark denoising.
Salinan berkas diambil dari repositori uji milik penulis DnCNN/FFDNet:
  https://github.com/cszn/DnCNN  dan  https://github.com/cszn/FFDNet
Penggunaan:  python unduh_dataset.py
"""
import os
import urllib.request

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
BASE = "https://raw.githubusercontent.com/cszn"
DAFTAR = [
    ("BSD68", f"{BASE}/DnCNN/master/testsets/BSD68/test{{:03d}}.png", range(1, 69)),
    ("Set12", f"{BASE}/DnCNN/master/testsets/Set12/{{:02d}}.png", range(1, 13)),
    ("Kodak24", f"{BASE}/FFDNet/master/testsets/Kodak24/kodim{{:02d}}.png", range(1, 25)),
]

for kel, pola, rentang in DAFTAR:
    os.makedirs(os.path.join(ROOT, kel), exist_ok=True)
    for i in rentang:
        url = pola.format(i)
        tujuan = os.path.join(ROOT, kel, os.path.basename(url))
        if os.path.exists(tujuan):
            continue
        urllib.request.urlretrieve(url, tujuan)
    print(kel, "->", len(os.listdir(os.path.join(ROOT, kel))), "berkas")
print("Selesai. Total harus 104 citra (68 + 12 + 24).")
