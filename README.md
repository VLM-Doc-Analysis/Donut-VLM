# Donut-VLM — 기계도면 정보 추출

한국 기계도면(표제란 · 치수 · 공차 · GD&T · 표면거칠기 · 볼트홀)을 **구조화 JSON** 으로 추출하는
[Donut](https://github.com/clovaai/donut)(Document Understanding Transformer) 파인튜닝 프로젝트입니다.
산출물은 **Jupyter 노트북 모음**이며, 각 노트북은 위 → 아래 순서로 실행하도록 작성돼 있습니다.
각 코드 셀 위에는 **`🔹 역할` 한 줄 요약 마크다운**이 있어, 셀을 펼치지 않아도 전체 흐름을 파악할 수 있습니다.

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
환경 셋업은 [`yolo_finetune_donut_pipeline/SETUP_GUIDE.md`](yolo_finetune_donut_pipeline/SETUP_GUIDE.md),
CVAT element 라벨링(`value` 속성 정의 → export → 변환)은
[`yolo_finetune_donut_pipeline/CVAT_LABELING_GUIDE.md`](yolo_finetune_donut_pipeline/CVAT_LABELING_GUIDE.md) 참고.

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

대용량 산출물(`data/` · `checkpoints*/` · `output/` · `datasets/`)은 **Git LFS** 로 추적·커밋합니다.
가중치(`*.safetensors` · `*.pt` · `*.pth`)는 `.gitattributes` 의 LFS 필터를 통해 포인터로 저장됩니다.
(`runs/` · `__pycache__/` · `.venv/` 등은 여전히 `.gitignore` 제외. `donut_base_test.ipynb` 는 `.gitignore` 항목이 주석처리돼 **추적·커밋됨**)

- **클론 후**: `git lfs install` → `git lfs pull` 로 실제 바이너리를 내려받습니다(미설치 시 포인터만 받음).
- **주의**: LFS 저장 용량은 십수 GB 규모라 GitHub LFS 무료 한도(스토리지·대역폭 각 1 GB/월)를 초과합니다 — 유료 데이터 플랜 필요.
- 로컬 Donut 포맷: `<root>/images/*.png` + `<root>/labels/<stem>.json` (stem 이 매칭되는 쌍만 사용)
- **라벨 품질 = 모델 품질**: 모든 라벨 JSON 의 키 구조를 일관되게 유지하세요.

## 참고 — CORD 레퍼런스

`donut_training.ipynb`(영수증 CORD-v2 원본 파이프라인)와 `donut_CORD_*_test.ipynb`(추론 데모)는
도면 파인튜닝의 출발점이 된 레퍼런스 노트북입니다.

---

## 📚 문서 목록 (가이드 인덱스)

### 1) 개념·학습 가이드 — "어떻게 동작하나"

| 문서 | 내용 |
|---|---|
| [`Donut_CORD_v2_파인튜닝_가이드.md`](Donut_CORD_v2_파인튜닝_가이드.md) | donut-base → CORD-v2 파인튜닝 전 과정(토큰 변환·특수토큰 등록·Teacher Forcing·CFG·추론). **다른 태스크 이식법** 포함 (절차편) |
| [`Donut_파인튜닝_학습메커니즘_가이드.md`](Donut_파인튜닝_학습메커니즘_가이드.md) | 학습이 "이미지→JSON 토큰열" 매핑을 **어떻게** 만드나 (원리편): Teacher Forcing·Loss·cross-attention 정렬·노출편향·추론 흐름 |
| [`CORD_v2_토큰_종류_가이드.md`](CORD_v2_토큰_종류_가이드.md) | Donut이 출력하는 토큰 종류(구조/특수 + 필드) — 실제 영수증 예시로 토큰열→JSON |
| [`Element_Donut_구조화스키마_수작업annotation_가이드.md`](yolo_finetune_donut_pipeline/Element_Donut_구조화스키마_수작업annotation_가이드.md) | 값을 의미 필드로 분해하는 구조화 스키마(질 레버) + 기호 표현(토크나이저) |
| [`Element_Donut_토크나이저_기호추가_가이드.md`](yolo_finetune_donut_pipeline/Element_Donut_토크나이저_기호추가_가이드.md) | 공학 기호(Ø·⊥·± …) 토큰 추가 절차·NFKC 함정 |

### 2) 계획·절차 — "어떻게 작업하나"

| 문서 | 내용 |
|---|---|
| [`PLAN.md`](yolo_finetune_donut_pipeline/PLAN.md) | YOLO→Donut 파이프라인 단계별 계획(전체 설계) |
| [`SETUP_GUIDE.md`](yolo_finetune_donut_pipeline/SETUP_GUIDE.md) | 환경 셋업(PDF 래스터화·CVAT·학습) |
| [`CVAT_LABELING_GUIDE.md`](yolo_finetune_donut_pipeline/CVAT_LABELING_GUIDE.md) | CVAT 라벨링 조작법(`value` 속성 → export → 변환) |
| [`Element_Donut_데이터라벨링_작업계획.md`](yolo_finetune_donut_pipeline/Element_Donut_데이터라벨링_작업계획.md) | 실 라벨 ≥1만 확충(양 레버) — 소싱·모델보조·배치·QA·일정 |

### 3) 분석·평가 리포트 — "지금 어디에 있나"

| 문서 | 내용 |
|---|---|
| [`YOLO_검출모델_평가리포트.md`](yolo_finetune_donut_pipeline/detection/YOLO_검출모델_평가리포트.md) | View/Element YOLO 검출 성능(mAP) 평가 |
| [`Element_Donut_평가리포트.md`](yolo_finetune_donut_pipeline/Element_Donut_평가리포트.md) | Element Donut 성능 평가(Leaf-Match 등) |
| [`Element_Donut_근본원인_분석.md`](yolo_finetune_donut_pipeline/Element_Donut_근본원인_분석.md) | 저성능 근본 원인(데이터 규모·from-scratch OCR) |
| [`Element_Donut_추론실패_사례분석.md`](yolo_finetune_donut_pipeline/Element_Donut_추론실패_사례분석.md) | 추론 실패 사례 분석(degenerate 생성 등) |
| [`Element_Donut_성능미달_원인_및_해결방안.md`](yolo_finetune_donut_pipeline/Element_Donut_성능미달_원인_및_해결방안.md) | 성능미달 원인 종합 + 해결 로드맵 |

> **읽는 순서 추천**: 처음이면 **1) 개념 가이드**(파인튜닝 → 토큰 → 구조화 스키마) → 작업 들어가면 **2) 계획·절차** → 막히면 **3) 분석 리포트**.
