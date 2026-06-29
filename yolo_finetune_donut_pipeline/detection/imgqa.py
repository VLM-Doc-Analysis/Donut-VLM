# Element 크롭 이미지 품질 측정 (이미지품질 분석 §재현)
# 사용: conda activate donut_vml; python yolo_finetune_donut_pipeline/detection/imgqa.py
import os, glob, json
import numpy as np, cv2
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # detection→pipeline→repo
from collections import defaultdict
import statistics as st

rows = []  # (cls, w, h, sharp, contrast, ink, ar, border_ink)
for f in glob.glob("data/elements/labels/*.json"):
    g = json.load(open(f, encoding="utf-8")); cls = next(iter(g)); stem = os.path.basename(f)[:-5]
    ip = None
    for ext in (".png", ".jpg", ".jpeg"):
        p = f"data/elements/images/{stem}{ext}"
        if os.path.exists(p): ip = p; break
    if not ip: continue
    im = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
    if im is None: continue
    h, w = im.shape
    sharp = cv2.Laplacian(im, cv2.CV_64F).var()
    contrast = float(im.std())
    # Otsu 전경(잉크) 비율
    _, bw = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = float((bw > 0).mean())
    # 경계 잉크(잘림 proxy): 테두리 2px 밴드의 잉크 비율
    band = np.concatenate([bw[:2].ravel(), bw[-2:].ravel(), bw[:, :2].ravel(), bw[:, -2:].ravel()])
    border = float((band > 0).mean())
    rows.append((cls, w, h, sharp, contrast, ink, w / h, border))

def pct(a, p): return float(np.percentile(a, p))
allh = [r[2] for r in rows]; allw = [r[1] for r in rows]
print(f"총 {len(rows)} 크롭")
print(f"높이 분포: {sorted(set(allh))[:8]}...  (p10/50/90 = {pct(allh,10):.0f}/{pct(allh,50):.0f}/{pct(allh,90):.0f})")
print(f"폭   분포: p10/50/90 = {pct(allw,10):.0f}/{pct(allw,50):.0f}/{pct(allw,90):.0f}  max={max(allw)}")
print()
hdr = f"{'class':<18}{'n':>5}{'h_med':>6}{'sharp_med':>10}{'contr_med':>10}{'ink%_med':>9}{'ar_med':>7}{'border%':>8}"
print(hdr); print("-"*len(hdr))
byc = defaultdict(list)
for r in rows: byc[r[0]].append(r)
for cls in sorted(byc, key=lambda c: -len(byc[c])):
    g = byc[cls]
    print(f"{cls:<18}{len(g):>5}{st.median(r[2] for r in g):>6.0f}{st.median(r[3] for r in g):>10.0f}"
          f"{st.median(r[4] for r in g):>10.1f}{st.median(r[5] for r in g)*100:>9.1f}{st.median(r[6] for r in g):>7.1f}"
          f"{st.median(r[7] for r in g)*100:>8.1f}")
print()
# 임계 기반 문제 플래그(전체)
sharp_all = [r[3] for r in rows]; ink_all = [r[5] for r in rows]; ar_all = [r[6] for r in rows]; bd_all=[r[7] for r in rows]
blur = sum(1 for s in sharp_all if s < 200)
lowc = sum(1 for r in rows if r[4] < 35)
sparse = sum(1 for i in ink_all if i < 0.03)
dense = sum(1 for i in ink_all if i > 0.30)
wide = sum(1 for a in ar_all if a > 5)
trunc = sum(1 for b in bd_all if b > 0.25)
N = len(rows)
print("=== 문제 플래그(전체 대비 %) ===")
print(f"  블러(Laplacian<200): {blur} ({blur/N*100:.1f}%)")
print(f"  저대비(std<35):       {lowc} ({lowc/N*100:.1f}%)")
print(f"  희박(잉크<3%):        {sparse} ({sparse/N*100:.1f}%)  ← 정렬불량/잘림 의심")
print(f"  과밀(잉크>30%):       {dense} ({dense/N*100:.1f}%)  ← 노이즈/다중요소 의심")
print(f"  초광폭(ar>5):         {wide} ({wide/N*100:.1f}%)  ← 다중요소 의심")
print(f"  경계잉크>25%:         {trunc} ({trunc/N*100:.1f}%)  ← 잘림(truncation) 의심")
