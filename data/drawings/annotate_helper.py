"""
도면 어노테이션 헬퍼 스크립트

사용법:
  1. data/drawings/images/ 에 PNG 이미지를 넣습니다
  2. 이 스크립트를 실행하면 각 이미지에 대한 JSON 템플릿을 생성합니다
  3. 생성된 JSON 파일을 수동으로 편집하여 실제 값을 채웁니다

  python3 annotate_helper.py
"""
import json
from pathlib import Path

IMAGES_DIR = Path(__file__).parent / 'images'
LABELS_DIR = Path(__file__).parent / 'labels'
LABELS_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 도면 라벨 스키마 (실제 50개 도면 라벨에서 도출 — data/drawings/labels/*.json)
#
#   [고정 코어]  아래 키는 전(全) 도면 공통 → 항상 존재. 도면에 표기 없으면 값은 "N/A".
#   [선택 확장]  도면에 해당 표기가 있을 때만 키 추가. 반드시 아래 *카탈로그의 이름 그대로*
#                써서 키 난립을 막을 것 (자유 작명 금지).
#
#   Donut 주의: 학습 빌드 단계는 이 TEMPLATE 가 아니라 *실제 라벨*의 키를 special token 으로
#   등록한다. 그래서 선택 키 이름이 들쭉날쭉하면 토큰이 폭증한다 → 카탈로그 고정.
#   task token 은 <s_drawing>(필드명 아님), 표제란 최상위 필드명은 title_block (충돌 방지).
# ─────────────────────────────────────────────────────────────────────────────
TEMPLATE = {
    "title_block": {            # 표제란 — 6개 키 전 도면 공통(없으면 N/A)
        "title": "",            # 부품명            예) FLANGE CIRCULAR PLAIN
        "Rev": "",              # 개정 번호(Revision)  예) 0
        "Drawing_no": "",       # 도면 번호         예) A14-942596-8
        "LIC_Material": "",     # LIC. Material
        "Material": "",         # 재질              예) SUS316L
        "Material_std": ""      # 재질 규격         예) ISO 6162-1:2012
    },
    "dimensions": {             # 치수 — 핵심 3개는 항상, 그 외는 아래 [선택 치수 카탈로그]
        "outer_diameter": "",   # 외경 (mm)        — 전 도면 공통
        "bore_diameter": "",    # 내경 (mm)        — 전 도면 공통
        "thickness": ""         # 두께 (mm)        — 전 도면 공통
        # ── 선택 치수 카탈로그 (해당 표기 있을 때만, 이름 그대로 추가) ──
        # "pcd"           : 볼트원 PCD (볼트홀이 한 종류일 때만; 두 종류면 bolt_holes 에)
        # "pcd_outer"     : 외측 볼트원 PCD        # "pcd_inner"   : 내측 볼트원 PCD
        # "pilot_od"      : 파일럿 외경            # "pilot_id"    : 파일럿 내경
        # "mid_diameter"  : 중간경                # "inner_bore"  : 추가 내경 단(段)
        # "hub_od"        : 허브 외경             # "hub_height"  : 허브 높이
        # "boss_diameter" : 보스 외경             # "boss_height" : 보스 높이
        # "slip_on_id"    : 슬립온 내경           # "width"       : 폭
        # "total_height"  : 전체 높이             # "fit"         : 끼워맞춤 공차(예 H7)
    },
    "bolt_holes": {             # 볼트홀 — 외측/내측 2조 항상 존재(없으면 각 값 N/A)
        "outer": {"count": "", "diameter": "", "pcd": ""},  # 통과홀: 개수/직경/PCD
        "inner": {"count": "", "size": "", "pcd": ""}       # 탭홀:  개수/나사규격/PCD
    },
    "surface_finish": {
        "general": "",      # 일반 표면 거칠기  예) 6.3
        "machined": ""      # 가공면 거칠기     예) 3.2
    },
    "gdt": {                    # 기하공차 — 표기 있을 때만 값(예 flatness "1.0"), 없으면 N/A
        "flatness": "",         # 평면도
        "perpendicularity": "", # 직각도
        "concentricity": ""     # 동심도
    },
    "threads": [],              # 나사 목록  예) ["M16"], ["PF3/8","PF1/2"], ["4xM6 x9/14.1"]
    "date": ""                  # 도면 날짜  예) 2025-03-12 (없으면 N/A)
    # ── 선택 최상위 키 (해당 시에만 추가) ──
    # "notes"   : []   자유 주석/제작 조건  예) ["6x M6 threaded holes"]
    # "variants": ""   다(多)규격 카탈로그 도면  예) "multiple DN sizes — see BOM table in drawing"
}

image_files = sorted(IMAGES_DIR.glob('*.png')) + sorted(IMAGES_DIR.glob('*.jpg'))
new_count = 0

for img_path in image_files:
    label_path = LABELS_DIR / (img_path.stem + '.json')
    if not label_path.exists():
        with open(label_path, 'w', encoding='utf-8') as f:
            json.dump(TEMPLATE, f, ensure_ascii=False, indent=2)
        print(f"템플릿 생성: {label_path.name}")
        new_count += 1
    else:
        print(f"이미 존재: {label_path.name}")

print(f"\n총 {len(image_files)}개 이미지 / {new_count}개 템플릿 신규 생성")
print(f"labels/ 폴더의 JSON 파일을 편집하여 값을 채워주세요.")
