# YOLO → Donut 파이프라인 계획 (도면 요소 추출)

도면 PDF → 구조화 JSON. **검출(YOLO)** 로 영역을 찾아 크롭하고, **인식(Donut)** 으로 값을 읽는
2단 구조. 한 장의 도면을 뷰(view) 단위로 나누고, 각 뷰 안의 **치수/공차/GD&T/거칠기** 요소를
회전 박스로 검출·정렬한 뒤 Donut 으로 값을 읽어 JSON 으로 조립한다.

## 전체 아키텍처

```
[파인튜닝] Donut base (naver-clova-ix/donut-base, 범용 문서 모델)
   ──요소 크롭 + 값 라벨로 파인튜닝 (task token <s_element>)──▶ checkpoints_elements/final
                                                              │ (도면 전용 요소 인식 모델)
[추론]                                                        ▼
PDF ──래스터화──▶ page.png (300 DPI)
   ──뷰 YOLO-det ──▶ 뷰 크롭들
   ──요소 YOLO-OBB──▶ 정렬(rectify)된 요소 크롭들 (+ 타입)
   ──요소 크롭별 Donut 인식──▶ 요소값 텍스트("Ø65", "Ra 1.6" 등)
   ──조립──▶ { "views": [ { "elements": [ {type, value, box} ] } ] }
```

**왜 2단인가** — 전체 도면 한 장을 Donut 에 그대로 넣으면 작은 글자(치수·공차·GD&T·거칠기)의
해상도가 부족하다. YOLO 로 작은 영역만 잘라 정렬해서 넣어야 Donut 이 안정적으로 읽는다.

**라벨링 도구**: CVAT (자체 호스팅, 온프레미스)
**커널**: 전 단계를 **단일 커널 `donut_vml`** 로 실행한다. `donut_vml` 에 YOLO(ultralytics)와
Donut(transformers/torch) 의존성이 **모두 설치돼 있어 커널 전환·디스크 핸드오프가 불필요**하다.
- 핵심 구성: torch 2.11.0(CUDA 12.8/cu128) · transformers 5.12.1 · timm · sentencepiece · opencv · PyMuPDF · ultralytics
- 학습도 동일 커널: accelerate·datasets 포함, 평가는 자체 `compute_leaf_match`(외부 `evaluate` 불필요), `report_to=none`.
- 버전 메모: 과거엔 Donut 을 `torch_211_env`(transformers 5.3.0)에서 돌렸으나, 5.3.0 으로 저장한
  체크포인트도 통합 커널 `donut_vml`(당시 transformers 4.57.6)에서 정상 로드·추론됨을 확인. 굳이 두 환경을 나눌 이유가 없어 통합.

## 클래스 정의

| 검출기 | 박스 타입 | 클래스 (실제 YAML 기준) |
|---|---|---|
| 뷰 (`view.yaml`) | AABB (축정렬 사각형) | `0:view`, `1:title_block`, `2:notes` |
| 요소 (`element.yaml`) | OBB (회전 사각형, 4점) | `0:Dimension` 치수, `1:GD&T_FCF` 기하공차 FCF, `2:Datum` 데이텀, `3:Surface_Roughness` 거칠기, `4:Section` 단면, `5:Hole_Callout` 홀 콜아웃 |

> 클래스명은 `detection/{view,element}.yaml` 이 단일 진실원본(source of truth). 위 표는 그에 맞춰 동기화됨.
> (Donut 학습 `parse_to_schema` 는 이 클래스명 문자열을 그대로 분기 키로 쓰므로 변경 시 양쪽을 함께 수정.)

## 단계별 계획

### 0단계 — PDF 재래스터화  (`detection/rasterize_pdf.ipynb`, donut_vml)
- `data/raw_pdf/*.pdf` (50장) → `data/drawings_pages/<stem>.png`
- 엔진: PyMuPDF(`fitz`) 우선, 없으면 `pdf2image`(poppler)
- **300 DPI 이상**으로 재생성 — 요소 크롭 1개당 충분한 픽셀 확보가 목적
  (기존 2526×1785 PNG 는 작은 글자에 해상도 부족)

### 1단계 — 뷰 검출  (`detection/view/train_view.ipynb`, YOLOv11 AABB)
1. `data/drawings_pages/*.png` 50장을 **CVAT(사각형)** 으로 라벨링
2. CVAT → **"Ultralytics YOLO Detection"** 포맷으로 내보내기 → `detection/view/cvat_export/`
3. 분할 셀로 `datasets/view/{train,val}` 구성 (45/5, seed 42)
4. 학습 → `detection/view/runs/view/weights/best.pt`
- 라벨 포맷: `<class> <cx> <cy> <w> <h>` (0~1 정규화)

### 2단계 — 요소 검출  (`detection/element/train_element.ipynb`, YOLOv11-OBB)
1. `data/view_crops/*.png` 를 **CVAT(회전 사각형)** 으로 라벨링
   - **효율화 핵심**: CVAT 라벨에 `value` **텍스트 속성(text attribute)** 을 정의해, 박스를 그릴 때
     값(예: "Ø65", "Ra 1.6")을 같이 입력 → **한 번의 패스로 ⟨YOLO-OBB 검출 라벨⟩ + ⟨Donut 값 라벨⟩ 동시 확보**
2. CVAT → **"Ultralytics YOLO Oriented Bounding Boxes"** 포맷으로 내보내기
3. 학습 → `detection/element/runs/element/weights/best.pt`
- 라벨 포맷: `<class> <x1> <y1> ... <x4> <y4>` (4점, 0~1 정규화)

### 3단계 — Donut 요소 파인튜닝  (`donut_training_elements_flat.ipynb`, donut_vml)

범용 문서 모델 `naver-clova-ix/donut-base` 를 **도면 요소 전용 인식 태스크**로 바꾸는 단계.
베이스 모델은 도면 치수/공차 기호(Ø, Ra, ± 등)를 모르므로, 정렬된 요소 크롭 → `{"<type>":"<value>"}`
JSON 을 출력하도록 파인튜닝한다.

> 📌 **노트북 변형 · 현행 성능 (2026-07-02 재학습)**: 이 단계의 노트북은 구조화 방식·데이터에 따라 **3종**.
>
> | 노트북 | 방식 | 체크포인트 | field-F1 |
> |:--|:--|:--|--:|
> | `donut_training_elements_flat.ipynb` (기준) | 값 통짜 + 사후 정규식 | `checkpoints_elements/final` | **0.613** (val 197) |
> | `donut_training_elements_paper.ipynb` | 구조화 JSON 직접 생성 · U+XXXX | `checkpoints_elements_paper/final` | **0.618** (val 197) |
> | `donut_training_elements_paper_hidpy.ipynb` | 〃 · 고DPI 데이터 · 글리프 토큰 | `checkpoints_elements_paper_hidpi/final` | **0.888** (검수 val 98) |
>
> 같은 데이터에서 paper ≈ flat(GD&T 는 paper 우위), 고DPI 데이터에선 0.888 — 상세·채점 수정 내역은
> [`Element_Donut_평가리포트.md`](Element_Donut_평가리포트.md) §0-3. 아래 서술은 기준(flat) 노트북 기준.

**모델 구조** (HF `VisionEncoderDecoderModel`)
- 인코더: Swin Transformer — 크롭 이미지를 시각 특징으로 인코딩.
- 디코더: BART/mBART — 특징을 받아 타깃을 **토큰 시퀀스**로 생성. 위치 임베딩 한계 768 이 `max_length` 상한
  (요소 크롭은 값이 짧아 128 토큰 내외면 충분).

**왜 토큰 시퀀스인가 — JSON ↔ 토큰 변환**
- `json2token`: dict → XML 식 토큰. 예) `{"dimension":"Ø65"}` → `<s_dimension>Ø65</s_dimension>`.
- 타깃 시퀀스 = `task_prompt(<s_element>) + json2token(정답) + <eos>`.
- 추론은 역변환 `token2json` 으로 다시 dict 복원 (4단계 `read_value()` 가 사용).

**특수 토큰 등록** (가장 중요)
- `<s_element>` (task token) **+ 라벨에 등장하는 모든 타입 토큰**(`<s_dimension>`/`<s_tolerance>`/`<s_gdt>`/`<s_roughness>`)을
  `add_special_tokens` 로 토크나이저에 추가하고 디코더 임베딩을 리사이즈.
- 등록 키는 **템플릿이 아니라 실제 라벨 JSON 에서 읽음** → 라벨에 없는 타입은 모델이 못 내보냄.
- `task_prompt(<s_element>)` 는 디코더의 **시작 토큰**이기도 함 → `decoder_start_token_id`/`pad_token_id` 연결.

**학습 방식 — Teacher Forcing**
- 라벨을 `max_length` 로 토크나이즈·패딩, **패드 위치는 `-100`** 으로 `CrossEntropyLoss` 가 무시.
- `Seq2SeqTrainer` 사용. `save_total_limit=3`, `load_best_model_at_end=True`(best=최저 `eval_loss`).
- 평가: `compute_leaf_match` — 예측/정답 JSON 을 leaf 로 펴서 일치율("Leaf-Match Score") 보고.

**데이터 준비**: `detection/cvat_to_donut.py`
- CVAT "CVAT for images 1.1" XML 내보내기를 읽어 각 회전 박스를 `rectify_obb` 로 정렬 크롭.
- 출력: `data/elements/images/<viewstem>__<i>__<type>.png` + `labels/....json` (`{"<type>":"<value>"}`).
- **추론과 동일한 정렬(rectify) 함수**를 써서 학습/추론 크롭 형태를 일치시킨다 (불일치 시 인식 급락).
- 분할 셀: `data/elements/{images,labels}` → `data/processed_elements/{train,val}`.

**핵심 설정 (CFG)**
- `task_prompt = <s_element>`, `max_length` ≈ 128 (요소 값은 짧음), 이미지 해상도는 크롭 종횡비에 맞춤.
- 데이터가 적으므로 epoch 多 + 작은 batch + grad-accum 으로 유효 배치 확보. VRAM 부족 시 batch↓ / grad-accum↑.
- 출력: 최종 모델 **+ 프로세서**를 `checkpoints_elements/final` 에 함께 저장 (추론은 둘을 같은 경로에서 로드).

**⚠️ 파인튜닝 주의사항 (하드웬)**
- **`bf16` 사용, `fp16` 금지** — fp16 은 Donut 에서 수치 불안정 → 깨진/0점 출력.
- **필드(타입) 토큰 등록** — 위 "특수 토큰 등록" 누락 시 디코더가 키를 못 만듦.
- **task token ≠ 필드명** — task 는 `<s_element>`, 필드는 `dimension` 등으로 분리(충돌 방지).
- **`token2json` 전 BOS + task_prompt 제거** — 남은 special token 이 정규식 파싱을 깸.

### 4단계 — 전체 연결(end-to-end) 파이프라인  (`pipeline_drawing.ipynb`)
**단일 커널 `donut_vml` 로 끝까지 실행**한다 (커널 전환 불필요). 파트 A→B 를 이어서 돌리면 되고,
`meta.json`/크롭은 디버깅용 중간 산출물로 남길 뿐 핸드오프 목적은 아니다.
- **파트 A** — 검출·크롭
  1. PDF → `page.png` (fitz, 300 DPI)
  2. 뷰 YOLO(AABB) 추론 → `crop_aabb` 로 뷰 크롭 (`imgsz=1280, conf=0.25, pad=4`)
  3. 각 뷰 크롭 → 요소 YOLO-OBB 추론 → `rectify_obb` 로 정렬 크롭 (`imgsz=1024, conf=0.25, pad=2`)
  4. 크롭 + `meta.json` 을 `data/_pipeline_tmp/` 에 저장
- **파트 B** — Donut 값 추론 + 조립
  1. `checkpoints_elements/final` 로드, task token `<s_element>`
  2. (메모리상 records 또는 `meta.json`)의 요소 크롭마다 `read_value()` 로 값 추론
  3. YOLO 가 분류한 타입으로 보정 (Donut 키가 비거나 다르면 YOLO 타입 사용)
  4. 최종 JSON → `data/_pipeline_tmp/result.json`

최종 출력 형태:
```json
{ "source": "...", "views": [
  { "view_index": 0, "box": [...],
    "elements": [ {"type": "dimension", "value": "Ø65", "conf": 0.97, "box": [...]} ] } ] }
```

## 공용 유틸 (`detection/crop_utils.py`)
- `load_bgr` / `to_pil` — 경로/PIL/ndarray ↔ OpenCV BGR ↔ Donut 입력 PIL 통일
- `crop_aabb(image, xyxy, pad)` — 뷰(AABB) 축정렬 크롭
- `rectify_obb(image, quad, pad)` — 요소(OBB) 원근 변환(perspective warp)으로 수평 정렬 크롭
  (가로폭 < 세로높이면 90° 회전 → 치수 텍스트를 가로로 눕힘)
- `save_crops_from_result(result, ...)` — ultralytics 결과(AABB/OBB 자동 판별) → 크롭 PNG + 메타 리스트
  - 파일명 규칙: `{stem}__{idx:03d}__{classname}.png`

## 디렉터리 구조

```
yolo_finetune_donut_pipeline/
├─ PLAN.md                          # (이 문서)
├─ pipeline_drawing.ipynb           # 4단계 전체 연결
├─ donut_training_elements_flat.ipynb         # 3단계 Donut 파인튜닝 (flat, 기준)
├─ donut_training_elements_paper.ipynb        # 3단계 변형 — 논문식 구조화 JSON (U+XXXX)
├─ donut_training_elements_paper_hidpy.ipynb  # 3단계 변형 — 논문식 · 고DPI (글리프 토큰)
└─ detection/
   ├─ rasterize_pdf.ipynb           # 0단계
   ├─ crop_utils.py                 # 공용 크롭/정렬
   ├─ cvat_to_donut.py              # CVAT XML → Donut 요소 데이터셋
   ├─ view/    (view.yaml, train_view.ipynb, datasets/, runs/, cvat_export/)
   └─ element/ (element.yaml, train_element.ipynb, datasets/, runs/, cvat_export/)

# 프로젝트 루트(donut_vml/) 공유:
#   data/raw_pdf, data/drawings_pages, data/view_crops,
#   data/elements, data/processed_elements, data/_pipeline_tmp
#   data/elements_hidpi, data/processed_elements_hidpi   # 고DPI 재취득 (hidpi 노트북)
#   data/elements_synth                                  # 합성 크롭 (synth_elements.py, train 병합)
#   checkpoints_elements/final, checkpoints_elements_paper/final, checkpoints_elements_paper_hidpi/final
```

## 실행 순서 요약
1. 0단계: `rasterize_pdf.ipynb` (donut_vml) — PDF→PNG 300DPI
2. CVAT 로 뷰 라벨링 → 1단계: `train_view.ipynb` (donut_vml)
3. 뷰 크롭 생성 → CVAT 로 요소 라벨링(+ `value` 속성) → 2단계: `train_element.ipynb` (donut_vml)
4. `cvat_to_donut.py` 로 Donut 데이터셋 생성 → 3단계: `donut_training_elements_flat.ipynb` (donut_vml)
5. 4단계: `pipeline_drawing.ipynb` — 파트 A→B 를 **단일 커널 donut_vml** 로 연속 실행 (커널 전환 없음)

> 전 단계 단일 커널 `donut_vml`. 다른 환경 불필요.
