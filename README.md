# Donut-VLM — 기계도면 정보 추출

한국 기계도면(표제란 · 치수 · 공차 · GD&T · 표면거칠기 · 볼트홀)을 **구조화 JSON** 으로 추출하는
[Donut](https://github.com/clovaai/donut)(Document Understanding Transformer) 파인튜닝 프로젝트입니다.
산출물은 **Jupyter 노트북 모음**이며, 각 노트북은 위 → 아래 순서로 실행하도록 작성돼 있습니다.

## 두 가지 접근

| 접근 | 진입점 | 설명 |
|---|---|---|
| **① Whole-page** | `donut_training_drawings.ipynb` | 도면 한 장 → JSON 직접 추출(베이스라인). 단순하지만 해상도·데이터 한계로 과적합되기 쉬움 |
| **② YOLO → Donut (권장)** | `yolo_finetune_donut_pipeline/` | 요소를 검출해 **크롭**한 뒤 작은 크롭을 Donut 으로 읽음 → ①의 해상도/도메인 한계를 완화 |

> **왜 ②인가**: 도면 전체를 `1280×960` 으로 줄이면 치수 글자가 뭉개집니다. 요소를 먼저 검출·정렬해
> 작은 크롭으로 인식하면 가독성이 크게 좋아집니다. (베이스라인의 한계 분석은
> `donut_training_drawings.ipynb` 마지막 마크다운 셀 참고)

## YOLO → Donut 파이프라인

```
PDF ──rasterize(300 DPI)──▶ page.png
   ──view 검출 (YOLOv11, AABB)──▶ view 크롭
   ──element 검출 (YOLOv11-OBB)──▶ 정렬(rectify)된 element 크롭 (+ 타입)
   ──element Donut 인식──▶ 값 ("Ø65", "Ra 1.6" …)
   ──조립──▶ { "views": [ { "elements": [ {type, value, box} ] } ] }
```

| 단계 | 위치 | 산출물 |
|---|---|---|
| 0. 래스터화 | `detection/rasterize_pdf.ipynb` | `data/drawings_pages/*.png` |
| 1. View 검출 학습 | `detection/view/train_view.ipynb` + `view.yaml` | `…/runs/view/weights/best.pt` |
| 2. Element 검출 학습 | `detection/element/train_element.ipynb` + `element.yaml` | `…/runs/element/weights/best.pt` |
| 3. Element Donut 학습 | `donut_training_elements.ipynb` | `checkpoints_elements/final` |
| 4. End-to-end | `pipeline_drawing.ipynb` | `result.json` |

공용 헬퍼 (`detection/`): `crop_utils.py`(크롭·OBB 정렬), `donut_utils.py`(토큰 I/O), `cvat_to_donut.py`(CVAT → Donut 데이터 변환).
단계별 상세 계획은 [`yolo_finetune_donut_pipeline/PLAN.md`](yolo_finetune_donut_pipeline/PLAN.md),
환경 셋업은 [`yolo_finetune_donut_pipeline/SETUP_GUIDE.md`](yolo_finetune_donut_pipeline/SETUP_GUIDE.md) 참고.

## 환경

- **권장: conda `kardi_env`** — 파이프라인 전 단계(YOLO + Donut)를 **단일 커널**로 실행
  (torch 2.11 · transformers 4.57 · ultralytics · timm · PyMuPDF · opencv 등). 커널 전환 불필요.
- 원 학습 노트북(`donut_training*.ipynb`)은 `torch_211_env`(transformers 5.3)에서도 동작.
- 설치: `pip install -r requirements.txt` (**transformers ≥ 4.45 필수** — `evaluation_strategy`→`eval_strategy`)
- 각 노트북 첫 코드 셀이 버전과 `torch.cuda.is_available()` 를 출력 — `False` 면 환경 설정을 점검하세요.

## 빠른 시작 (② 파이프라인)

1. **래스터화** — PDF 를 `data/raw_pdf/` 에 두고 `rasterize_pdf.ipynb` 실행 → 페이지 PNG
2. **검출 학습** — CVAT 로 view / element 라벨링 → `train_view.ipynb`, `train_element.ipynb` 학습
3. **인식 학습** — element 박스에 `value` **텍스트 속성**(예: "Ø65") 입력 → `cvat_to_donut.py` 로 Donut
   데이터 생성 → `donut_training_elements.ipynb` 학습
   - `class`(박스 종류, JSON 키) 와 `value`(읽은 값, JSON 값)는 **다른 것**입니다 → `{"Dimension": "Ø65"}`
4. **통합** — `pipeline_drawing.ipynb` 로 PDF → JSON end-to-end 실행

## 데이터 레이아웃

대용량이라 `data/` · `checkpoints*/` · `runs/` · `datasets/` 는 **커밋하지 않습니다**(.gitignore).

- 로컬 Donut 포맷: `<root>/images/*.png` + `<root>/labels/<stem>.json` (stem 이 매칭되는 쌍만 사용)
- **라벨 품질 = 모델 품질**: 모든 라벨 JSON 의 키 구조를 일관되게 유지하세요.

## 참고 — CORD 레퍼런스

`donut_training.ipynb`(영수증 CORD-v2 원본 파이프라인)와 `donut_CORD_*_test.ipynb`(추론 데모)는
도면 파인튜닝의 출발점이 된 레퍼런스 노트북입니다.
