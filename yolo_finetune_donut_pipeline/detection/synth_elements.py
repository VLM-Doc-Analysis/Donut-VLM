"""
synth_elements.py — Donut 요소 인식용 **합성 학습 데이터** 생성기.

왜 필요한가
-----------
Donut 값 인식의 근본 병목은 **데이터 규모**입니다(실라벨 ~1,975장 vs 논문 ~11,000장, 65% Dimension 편중).
손라벨링은 느리고, 모델은 "글자를 읽는 법"을 처음부터 배워야 하는데 표본이 너무 적습니다.
합성 데이터는 **공짜 라벨**로 이 격차를 메웁니다 — 치수/공차/데이텀/거칠기 문자열을 직접 렌더링해
정렬된 요소 크롭과 똑같은 모양(흰 배경·검은 글자·약한 노이즈/회전)으로 수천~수만 장을 찍어냅니다.

출력 포맷 (cvat_to_donut.py / DonutDataset 과 동일)
--------------------------------------------------
    <out>/images/<stem>.png
    <out>/labels/<stem>.json   →  {"<Class>": "<value>"}   (예: {"Dimension": "Ø65"})

클래스명은 detection/element.yaml = parse_to_schema 분기 키와 일치해야 함:
    Dimension · GD&T_FCF · Datum · Surface_Roughness · Section · Hole_Callout

사용 예
-------
    # 클래스 분포를 (실데이터의 역수로) 균형 맞춰 8000장 생성
    python detection/synth_elements.py --n 8000 --out data/elements_synth --seed 42

    # 그 뒤 실데이터와 합쳐 split → 재학습:
    #   data/elements_synth 와 data/elements 를 한 폴더로 모으거나,
    #   donut_training_elements.ipynb 의 split 셀이 두 소스를 모두 읽도록 경로 지정.

주의
----
- 합성은 실분포를 완벽히 대체하지 못합니다(폰트·레이아웃·노이즈 한정). **실데이터와 섞어** 쓰고,
  실검증셋(data/processed_elements/val)으로만 성능을 측정하세요. 합성으로만 평가하면 낙관 편향.
- 기호 폰트 커버리지: DejaVuSans 는 Ø ± ° ⊥ ∥ ∠ ⌀ ⏥ ◎ √ 를 지원. NFKC 로 분해되는 Ⓜ/Ⓛ 등
  원문자 수정자는 학습 토크나이저가 어차피 배제하므로 여기서도 쓰지 않습니다.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ── 폰트: matplotlib 동봉 DejaVuSans (kardi_env 에 항상 존재, 공학 기호 커버) ──
try:
    import matplotlib
    _FONT_DIR = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
    _FONT_FILES = [_FONT_DIR / "DejaVuSans.ttf", _FONT_DIR / "DejaVuSans-Bold.ttf"]
except Exception:                                   # 폴백: 시스템 기본
    _FONT_FILES = [Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")]
_FONT_FILES = [f for f in _FONT_FILES if Path(f).exists()] or [None]


# ── 클래스별 값 생성기 (실 도면 분포를 모사) ───────────────────────────────
def _num(lo, hi, dec):
    """lo~hi 범위 숫자를 dec 자리로. dec=0 이면 정수."""
    v = random.uniform(lo, hi)
    return str(int(round(v))) if dec == 0 else f"{v:.{dec}f}"


def gen_dimension():
    base = random.choice([
        lambda: f"Ø{_num(2, 200, random.choice([0, 0, 1]))}",        # 지름
        lambda: f"R{_num(1, 80, random.choice([0, 1]))}",            # 반지름
        lambda: _num(1, 300, random.choice([0, 0, 1, 2])),           # 일반 치수
        lambda: f"{_num(5, 60, 1)}°",                                # 각도
        lambda: f"M{random.choice([3,4,5,6,8,10,12,16,20])}",        # 나사 호칭
        lambda: f".{random.randint(1,99):02d} x 45°",                # 모따기
    ])()
    # 수량 접두 (N× / NX)
    if random.random() < 0.18:
        base = f"{random.randint(2, 8)}{random.choice(['X', '×', 'X '])}{base}"
    # 공차 접미
    r = random.random()
    if r < 0.18:
        base += f"±{_num(0.01, 0.5, random.choice([2, 3]))}"          # 대칭
    elif r < 0.28:
        base += f"+{_num(0.01, 0.3, 2)} -{_num(0.01, 0.3, 2)}"        # 비대칭
    return base


_GDT_SYM = ["⊥", "∥", "∠", "⌀", "○", "◎", "⏥", "/", "⌖", "↗"]


def gen_gdt():
    sym = random.choice(_GDT_SYM)
    tol = _num(0.001, 0.2, random.choice([3, 3, 2]))
    s = f"{sym}{tol}"
    n_dat = random.choice([0, 1, 1, 2])
    if n_dat:
        dats = random.sample(list("ABCD"), n_dat)
        s += " " + " ".join(dats)
    return s


def gen_datum():
    return random.choice(list("ABCDEFG")) + (str(random.randint(1, 3)) if random.random() < 0.1 else "")


def gen_roughness():
    return f"Ra {random.choice(['0.4','0.8','1.6','3.2','6.3','12.5','0.2','0.1'])}"


def gen_section():
    c = random.choice(list("ABCDEF"))
    return f"{c}-{c}"


def gen_hole():
    base = random.choice([
        lambda: f"Ø{_num(3, 30, random.choice([0, 1]))} THRU",
        lambda: f"M{random.choice([3,4,5,6,8,10])}X{random.choice(['0.5','0.7','1.0','1.25','1.5'])}",
        lambda: f"Ø{_num(4, 20, 1)} ⏥ .{random.randint(10,90)}",     # 카운터보어
    ])()
    if random.random() < 0.4:
        base = f"{random.randint(2, 12)}X {base}"
    return base


GENERATORS = {
    "Dimension":         gen_dimension,
    "GD&T_FCF":          gen_gdt,
    "Datum":             gen_datum,
    "Surface_Roughness": gen_roughness,
    "Section":           gen_section,
    "Hole_Callout":      gen_hole,
}

# 실데이터가 Dimension 에 편중(65%)되어 있으므로, 합성은 **소수 클래스를 더 많이** 찍어 균형 보강.
DEFAULT_WEIGHTS = {
    "Dimension":         0.30,
    "GD&T_FCF":          0.20,
    "Datum":             0.12,
    "Surface_Roughness": 0.16,
    "Section":           0.08,
    "Hole_Callout":      0.14,
}


# ── 렌더링: 정렬(rectify)된 요소 크롭 모양으로 ─────────────────────────────
def render(text: str, vertical: bool = False) -> Image.Image:
    font_path = random.choice(_FONT_FILES)
    size = random.randint(28, 46)
    font = (ImageFont.truetype(str(font_path), size) if font_path
            else ImageFont.load_default())

    # 글자 크기 측정
    tmp = Image.new("L", (4, 4), 255)
    l, t, r, b = ImageDraw.Draw(tmp).textbbox((0, 0), text, font=font)
    tw, th = r - l, b - t
    pad_x, pad_y = random.randint(6, 16), random.randint(5, 12)
    W, H = tw + 2 * pad_x, th + 2 * pad_y

    bg = random.randint(245, 255)                      # 살짝 회색빛 배경도 허용
    img = Image.new("L", (W, H), bg)
    d = ImageDraw.Draw(img)
    fill = random.randint(0, 45)                        # 검정~짙은 회색 글자
    d.text((pad_x - l, pad_y - t), text, font=font, fill=fill)

    # 약한 증강: 회전(정렬 잔여각 ±2°), 블러, 가우시안 노이즈
    if random.random() < 0.5:
        img = img.rotate(random.uniform(-2.0, 2.0), expand=True,
                         fillcolor=bg, resample=Image.BILINEAR)
    if random.random() < 0.35:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 0.8)))
    if random.random() < 0.5:
        import numpy as np
        a = np.asarray(img).astype("int16")
        a += np.random.normal(0, random.uniform(3, 9), a.shape).astype("int16")
        img = Image.fromarray(a.clip(0, 255).astype("uint8"))

    # 세로 치수 모사: 90° 세워 저장(파이프라인 rectify 가 다시 눕히는 케이스 대비)
    if vertical:
        img = img.rotate(90, expand=True, fillcolor=bg)
    return img.convert("RGB")


def main():
    ap = argparse.ArgumentParser(description="Donut 요소 인식용 합성 데이터 생성")
    ap.add_argument("--n", type=int, default=8000, help="생성할 총 샘플 수")
    ap.add_argument("--out", type=str, default="data/elements_synth", help="출력 루트")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vertical-ratio", type=float, default=0.0,
                    help="세로로 세운 크롭 비율(파이프라인이 rectify 로 다시 눕히는 경우만 >0 권장)")
    args = ap.parse_args()

    random.seed(args.seed)
    try:
        import numpy as np
        np.random.seed(args.seed)
    except ImportError:
        pass

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    classes = list(GENERATORS)
    weights = [DEFAULT_WEIGHTS[c] for c in classes]
    per_class = {c: 0 for c in classes}

    for i in range(args.n):
        cls = random.choices(classes, weights=weights, k=1)[0]
        value = GENERATORS[cls]()
        vertical = random.random() < args.vertical_ratio
        img = render(value, vertical=vertical)

        stem = f"synth_{i:06d}_{cls.replace('&', 'and').replace('/', '_')}"
        img.save(out / "images" / f"{stem}.png")
        json.dump({cls: value}, open(out / "labels" / f"{stem}.json", "w", encoding="utf-8"),
                  ensure_ascii=False)
        per_class[cls] += 1

    print(f"생성 완료: {args.n}장 → {out}")
    for c in classes:
        print(f"  {c:<20} {per_class[c]:>6}  ({per_class[c]/args.n*100:4.1f}%)")
    print("\n다음 단계: 이 합성셋을 실데이터(data/elements)와 **섞어** split → 재학습.")
    print("          성능 측정은 반드시 실검증셋(data/processed_elements/val)으로만.")


if __name__ == "__main__":
    main()
