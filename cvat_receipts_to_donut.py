"""
cvat_receipts_to_donut.py — CVAT 영수증 라벨("CVAT for images 1.1" XML) → Donut JSON.

Donut 은 OCR-free 라 학습에 박스가 필요 없고 **JSON 정답만** 필요하다. CVAT 에서는
박스에 text attribute 로 값을 입력해 라벨링하고, 이 스크립트가 그 export 를
Donut 학습용 JSON 으로 변환한다.

라벨 스키마 (CVAT Projects → 라벨 → Raw 탭에 등록):
    store_name (rectangle) → attr: text
    total      (rectangle) → attr: text
    item       (rectangle) → attrs: name, price   (메뉴 한 줄 = 박스 1개)

출력: <out>/<이미지stem>.json  +  이미지 복사 (stem 일치)
→ 이후 donut_training.ipynb 의 "[선택] 로컬 데이터셋 준비" 셀이 train/val 로 분할.

사용법 (프로젝트 루트 donut_vml/ 에서):
    python cvat_receipts_to_donut.py \
        --xml annotations.xml \
        --images <CVAT에 올린 원본 이미지 폴더> \
        --out data/raw

생성되는 JSON 예시:
    {
      "store_name": "스타벅스",
      "total": "12500",
      "items": [
        {"name": "아메리카노", "price": "4500"},
        {"name": "카페라떼", "price": "8000"}
      ]
    }
"""
from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# 평면 필드(영수증당 1개) 라벨과 그 값이 담긴 text attribute 이름
FLAT_LABELS = ("store_name", "total")
ITEM_LABEL = "item"          # 메뉴 한 줄(중첩 객체) 라벨
TEXT_ATTR = "text"           # 평면 필드 값 attribute 이름


def attrs_of(box) -> dict:
    """<box> 하위 <attribute name=..>값</attribute> 들을 dict 로 수집."""
    return {a.get("name"): (a.text or "").strip() for a in box.findall("attribute")}


def ytop(box) -> float:
    """정렬용 박스 상단 y좌표 (item 을 읽기 순서로 정렬하기 위함)."""
    return float(box.get("ytl"))


def convert(xml_path: Path, images_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    root = ET.parse(xml_path).getroot()
    n_img = n_skip = n_noimg = 0

    for image in root.findall("image"):
        fname = image.get("name")              # 예: 영수증_001.jpg
        record: dict = {}
        items: list[tuple[float, dict]] = []

        for b in image.findall("box"):
            label = b.get("label")
            a = attrs_of(b)
            if label in FLAT_LABELS:
                record[label] = a.get(TEXT_ATTR, "")
            elif label == ITEM_LABEL:
                items.append((ytop(b), {"name": a.get("name", ""),
                                        "price": a.get("price", "")}))

        # item 은 위→아래(읽기 순서)로 정렬
        if items:
            record["items"] = [obj for _, obj in sorted(items, key=lambda t: t[0])]

        if not record:                          # 박스가 하나도 없는 이미지는 건너뜀
            n_skip += 1
            continue

        stem = Path(fname).stem
        # 1) JSON 정답 저장 (한글 보존)
        with open(out_dir / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        # 2) 이미지도 같은 폴더로 복사 (stem 이 일치해야 DonutDataset 이 쌍으로 인식)
        src_img = images_dir / fname
        if src_img.exists():
            shutil.copy2(src_img, out_dir / fname)
        else:
            print(f"  ⚠ 이미지 없음(복사 생략): {src_img}")
            n_noimg += 1
        n_img += 1

    print(f"변환 완료: {n_img}장 → {out_dir}")
    print(f"  라벨 없는 이미지 건너뜀: {n_skip}장 / 이미지 못 찾음: {n_noimg}장")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CVAT 영수증 라벨 XML → Donut JSON 변환")
    ap.add_argument("--xml", required=True, type=Path,
                    help='CVAT "CVAT for images 1.1" export 의 annotations.xml')
    ap.add_argument("--images", required=True, type=Path,
                    help="CVAT 에 업로드한 원본 이미지 폴더")
    ap.add_argument("--out", type=Path, default=Path("data/raw"),
                    help="출력 폴더 (기본 data/raw)")
    args = ap.parse_args()
    convert(args.xml, args.images, args.out)
