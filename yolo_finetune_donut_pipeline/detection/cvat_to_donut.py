"""
cvat_to_donut.py — CVAT element 라벨(회전 박스 + value attribute) → Donut element 데이터셋.

Stage 2 에서 CVAT 로 element 를 라벨링할 때, 각 회전 박스에 `value` **text attribute** 로
요소값(예: "Ø65", "Ra 1.6")을 함께 입력해 두면, 한 번의 라벨링 패스로
⟨YOLO-OBB 검출 라벨⟩ + ⟨Donut 값 라벨⟩ 을 동시에 얻을 수 있습니다.

이 스크립트는 **CVAT "CVAT for images 1.1"** XML export 를 읽어서,
각 회전 박스를 (crop_utils.rectify_obb 로) 수평 정렬 크롭한 뒤 Donut 학습쌍을 만듭니다:

    ../data/elements/images/<viewstem>__<i>__<type>.png   # 정렬된 element 크롭
    ../data/elements/labels/<viewstem>__<i>__<type>.json  # {"<type>": "<value>"}
    (data/ 와 checkpoints_elements/ 는 프로젝트 루트 donut_vml/ 에 공유됨)

→ 이후 donut_training_elements.ipynb 의 split 셀이 train/val 로 나눕니다.

사용법 (yolo_finetune_donut_pipeline/ 에서):
    python detection/cvat_to_donut.py \
        --xml detection/element/cvat_export/annotations.xml \
        --images ../data/view_crops

주의: rectify 는 추론(crop_utils)과 동일 함수를 써야 학습/추론 크롭 형태가 일치합니다.
"""
from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

# 같은 detection/ 폴더의 crop_utils 사용 (추론과 동일한 정렬 로직)
from crop_utils import rectify_obb, load_bgr

VALUE_ATTR = "value"   # CVAT 라벨에 정의한 text attribute 이름


def rotated_corners(xtl, ytl, xbr, ybr, rotation_deg):
    """CVAT 박스(축정렬 + center 기준 회전) → 4개 코너 픽셀 좌표.

    CVAT 의 rotation 은 박스 중심 기준 시계방향(이미지 y-down 좌표) 각도(deg).
    """
    cx, cy = (xtl + xbr) / 2.0, (ytl + ybr) / 2.0
    corners = [(xtl, ytl), (xbr, ytl), (xbr, ybr), (xtl, ybr)]
    th = math.radians(rotation_deg)
    cos, sin = math.cos(th), math.sin(th)
    out = []
    for x, y in corners:
        dx, dy = x - cx, y - cy
        out.append([cx + dx * cos - dy * sin, cy + dx * sin + dy * cos])
    return out


def convert(xml_path: Path, images_dir: Path, out_root: Path):
    img_out = out_root / "images"
    lbl_out = out_root / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(xml_path)
    root = tree.getroot()
    n_imgs = n_boxes = n_skipped = 0

    for image in root.findall("image"):
        name = image.get("name")
        stem = Path(name).stem
        # CVAT name 이 경로를 포함할 수 있으므로 파일명만으로도 찾아봄
        src = images_dir / name
        if not src.exists():
            cand = list(images_dir.glob(f"{stem}.*"))
            if not cand:
                print(f"  [skip] 이미지 없음: {name}")
                continue
            src = cand[0]
        n_imgs += 1
        src_bgr = load_bgr(str(src))   # 박스마다 재디코딩하지 않도록 원본을 1회만 읽음

        for i, box in enumerate(image.findall("box")):
            label = box.get("label")
            xtl, ytl = float(box.get("xtl")), float(box.get("ytl"))
            xbr, ybr = float(box.get("xbr")), float(box.get("ybr"))
            rot = float(box.get("rotation", "0"))
            quad = rotated_corners(xtl, ytl, xbr, ybr, rot)

            value = ""
            for attr in box.findall("attribute"):
                if attr.get("name") == VALUE_ATTR:
                    value = (attr.text or "").strip()
            if not value:
                # 값 미입력 박스는 검출(YOLO-OBB export)용으로만 쓰고 Donut 라벨은 건너뜀.
                # (빈 라벨을 학습에 넣으면 Donut 이 "빈 출력"을 배워 인식이 망가짐)
                n_skipped += 1
                continue

            crop = rectify_obb(src_bgr, quad, pad=2)
            base = f"{stem}__{i:03d}__{label}"
            crop.save(img_out / f"{base}.png")
            json.dump({label: value}, open(lbl_out / f"{base}.json", "w", encoding="utf-8"),
                      ensure_ascii=False)
            n_boxes += 1

    print(f"이미지 {n_imgs} · element 크롭 {n_boxes}개 생성 → {out_root}")
    if n_skipped:
        print(f"  (값(value) 미입력 {n_skipped}개 — Donut 라벨 건너뜀. CVAT 에서 채우면 데이터 늘어남)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True, help="CVAT for images 1.1 export XML")
    ap.add_argument("--images", default="../data/view_crops", help="박스가 그려진 원본(view 크롭) 폴더")
    ap.add_argument("--out", default="../data/elements", help="출력 루트(images/, labels/)")
    a = ap.parse_args()
    convert(Path(a.xml), Path(a.images), Path(a.out))


if __name__ == "__main__":
    main()
