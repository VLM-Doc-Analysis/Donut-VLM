# 고DPI element 크롭 추출 — raw_pdf 적응형 재래스터 + GT 박스(view→element) rectify
# 사용: conda activate donut_vml; python yolo_finetune_donut_pipeline/detection/extract_hidpi_crops.py
#   [--dpi N]        고정 DPI 강제(기본=적응형: 벡터 900/스캔 native)
#   [--vector-only]  벡터 PDF 만 (A/B 짝 생성용)
#   [--out DIR]      출력 디렉터리(기본 data/elements_hidpi)
# 300/900 A/B 짝 예: --dpi 300 --vector-only --out data/elements_hidpi_300  /  --dpi 900 --vector-only --out data/elements_hidpi_900
import fitz, numpy as np, cv2, os, re, glob, json, sys, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
os.chdir(HERE); sys.path.insert(0, HERE)
from crop_utils import rectify_obb

ap = argparse.ArgumentParser()
ap.add_argument("--dpi", type=int, default=0, help="고정 DPI(0=적응형)")
ap.add_argument("--vector-only", action="store_true")
ap.add_argument("--out", default="data/elements_hidpi")
A = ap.parse_args()
FORCE_DPI = A.dpi; VECTOR_ONLY = A.vector_only

ELEM_NAMES = {0: "Dimension", 1: "GD&T_FCF", 2: "Datum", 3: "Surface_Roughness", 4: "Section", 5: "Hole_Callout"}
VLAB = "view/datasets/view/labels"; ELAB = "element/datasets/element/labels"
OUT = A.out if os.path.isabs(A.out) else os.path.join(REPO, A.out)
for sub in ("images", "labels"):
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)

def adaptive_dpi(pdf):
    pg = fitz.open(pdf)[0]; Win = pg.rect.width / 72.0
    if len(pg.get_drawings()) >= 300:
        return 900, "vector"
    maxw = 0
    for im in pg.get_images(full=True):
        try: maxw = max(maxw, fitz.open(pdf).extract_image(im[0])["width"])
        except Exception: pass
    native = maxw / Win if Win > 0 else 300
    return int(min(900, max(150, native))), "scan"

def render(pdf, dpi):
    pg = fitz.open(pdf)[0]; m = fitz.Matrix(dpi/72, dpi/72); pix = pg.get_pixmap(matrix=m, alpha=False)
    a = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
    return cv2.cvtColor(a, cv2.COLOR_RGB2BGR)

manifest = []; n_crop = 0; n_pdf = 0
efiles = sorted(glob.glob(f"{ELAB}/**/*_p*_v*.txt", recursive=True))
stems = sorted(set(re.match(r"(.+)_p\d+_v\d+$", os.path.basename(f)[:-4]).group(1) for f in efiles))
for stem in stems:
    pdf = os.path.join(REPO, "data/raw_pdf", stem + ".pdf")
    vfile = glob.glob(f"{VLAB}/**/{stem}_p1.txt", recursive=True)
    if not os.path.exists(pdf) or not vfile:
        continue
    dpi, kind = adaptive_dpi(pdf)
    if VECTOR_ONLY and kind != "vector":
        continue
    if FORCE_DPI:
        dpi = FORCE_DPI
    page = render(pdf, dpi); H, W = page.shape[:2]
    views = [list(map(float, l.split()[1:5])) for l in open(vfile[0]) if l.split() and l.split()[0] == "0"]
    n_pdf += 1
    for ef in sorted(glob.glob(f"{ELAB}/**/{stem}_p1_v*.txt", recursive=True)):
        vi = int(re.search(r"_v(\d+)", os.path.basename(ef)).group(1))
        if vi - 1 >= len(views): continue
        cx, cy, w, h = views[vi-1]
        x0, y0 = max(0, int((cx-w/2)*W)), max(0, int((cy-h/2)*H)); x1, y1 = int((cx+w/2)*W), int((cy+h/2)*H)
        view = page[y0:y1, x0:x1]; vh, vw = view.shape[:2]
        if vh < 4 or vw < 4: continue
        for ei, line in enumerate(open(ef)):
            p = line.split()
            if len(p) < 9: continue
            cls = ELEM_NAMES.get(int(p[0]), "value")
            quad = np.array([[float(p[1+2*k])*vw, float(p[2+2*k])*vh] for k in range(4)], dtype=np.float32)
            try: crop = rectify_obb(view, quad)
            except Exception: continue
            name = f"{stem}_p1_v{vi}_e{ei}_{cls}".replace("&", "and")
            crop.save(os.path.join(OUT, "images", name + ".png"))
            json.dump({cls: ""}, open(os.path.join(OUT, "labels", name + ".json"), "w", encoding="utf-8"), ensure_ascii=False)
            manifest.append({"crop": name + ".png", "class": cls, "dpi": dpi, "kind": kind, "value": "", "status": "pending"})
            n_crop += 1
open(os.path.join(OUT, "values_hidpi.jsonl"), "w", encoding="utf-8").write("\n".join(json.dumps(m, ensure_ascii=False) for m in manifest))
from collections import Counter
print(f"PDF {n_pdf} | 크롭 {n_crop} → {OUT}")
print("DPI 분포:", dict(Counter((m['dpi'], m['kind']) for m in manifest)))
print("클래스 분포:", dict(Counter(m['class'] for m in manifest)))
