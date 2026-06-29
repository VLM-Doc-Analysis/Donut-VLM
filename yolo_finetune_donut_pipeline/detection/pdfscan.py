# raw_pdf 소스 품질 검토 — 벡터 vs 스캔(유효 DPI) 분류 (고DPI 파이프라인 §7)
# 사용: conda activate donut_vml; python yolo_finetune_donut_pipeline/detection/pdfscan.py
import fitz, glob, os, statistics as st
from collections import Counter
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..")))  # detection→pipeline→repo
cat = Counter(); dpis = []; vec = []
for p in sorted(glob.glob("data/raw_pdf/*.pdf")):
    d = fitz.open(p); pg = d[0]
    Win = pg.rect.width / 72.0
    paths = len(pg.get_drawings())
    maxw = 0
    for im in pg.get_images(full=True):
        try:
            info = d.extract_image(im[0])
            if info["width"] > maxw: maxw = info["width"]
        except Exception:
            pass
    scan_dpi = maxw / Win if Win > 0 else 0
    if paths >= 300:
        cat["벡터(CAD, DPI무관)"] += 1; vec.append(paths)
    elif maxw > 0:
        if scan_dpi >= 550: b = "스캔 >=550 DPI"
        elif scan_dpi >= 350: b = "스캔 350-550"
        elif scan_dpi >= 250: b = "스캔 250-350"
        else: b = "스캔 <250"
        cat[b] += 1; dpis.append(scan_dpi)
    else:
        cat["불명"] += 1
tot = sum(cat.values())
print(f"총 {tot}개")
print("=== 분류 (유효 스캔 DPI 기준) ===")
for k, v in cat.most_common():
    print(f"  {k:<20} {v:>3}  ({v/tot*100:.0f}%)")
if dpis:
    print(f"\n래스터 유효 DPI: min/median/max = {min(dpis):.0f}/{st.median(dpis):.0f}/{max(dpis):.0f}")
vcount = cat["벡터(CAD, DPI무관)"]
print(f"\n벡터(600DPI 효과 O): {vcount} ({vcount/tot*100:.0f}%)")
raster = tot - vcount - cat.get("불명", 0)
print(f"래스터(스캔, 효과는 유효DPI에 종속): {raster}")
