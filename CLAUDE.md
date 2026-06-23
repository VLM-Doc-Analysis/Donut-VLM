# CLAUDE.md

Donut(Document Understanding Transformer) 파인튜닝 프로젝트. 산출물은 **Jupyter 노트북 모음**
— 설치형 패키지·`src/`·테스트·엔트리포인트 스크립트는 없다. 모든 로직은 노트북 셀에 있고
**위→아래로 순서대로** 실행하도록 작성됨. 모든 노트북은 한국어 설명 마크다운/주석으로 파이프라인을
가르치는 톤이므로, 변경 시 **한국어 주석과 교육적 톤을 유지**할 것.

## 구성

**학습**
- `donut_training.ipynb` — 원본 파이프라인, 영수증 도메인(CORD-v2). HF 데이터셋 또는 로컬 데이터 지원.
- `donut_training_drawings.ipynb` — 같은 파이프라인을 커스텀 도메인(**한국 기계도면 표제란/치수/볼트홀 추출**)으로
  이식. 베이스 모델 + 로컬 모드, task token `<s_drawing>`, `max_length=768`, `checkpoints_drawings/` 에 저장.
  **현재 개발 중인 커스텀 태스크.**

**추론/데모** (파인튜닝된 모델 로드, 학습 흐름과 무관)
- `donut_base_test.ipynb` — 파인튜닝 전 `naver-clova-ix/donut-base` 동작 확인 (gitignored)
- `donut_CORD_v2_fine_tunned_test.ipynb` — `VisionEncoderDecoderModel` 수동 추론(레퍼런스)
- `donut_CORD_v2_fine_tunned_Pipeline_test.ipynb` — 동일 태스크를 HF `pipeline()` API 로
- `donut_CORD_v3_fine_tunned_AutoModel_test.ipynb` — 동일 태스크를 `AutoModel`/`AutoProcessor` 로
- `donut_CORD_v2_fine_tunned_test_kardi.ipynb` — 로컬 이미지 추론. **`kardi_env`** 커널 사용.

**YOLO → Donut 파이프라인** (`yolo_finetune_donut_pipeline/`)
- 도면을 뷰/요소(치수·공차·GD&T·거칠기)로 **검출(YOLO)** 후 크롭 → **인식(Donut)** 하는 2단 파이프라인.
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
- **요소 Donut 파인튜닝**(`donut_training_elements.ipynb`): 아래 "아키텍처"의 파인튜닝 메커니즘(json2token,
  특수 토큰 등록, Teacher Forcing, bf16 등)을 **요소 크롭 단위**로 적용. 도면 전체가 아니라 정렬된 요소
  크롭 1개 → `{"<type>":"<value>"}` 를 학습. task token 은 `<s_element>`, 타입 토큰(`<s_dimension>` 등)은
  라벨에서 자동 등록. 데이터는 `detection/cvat_to_donut.py` 로 CVAT export 에서 생성 → `checkpoints_elements/final` 저장.
- 전체 단계별 계획은 **[`yolo_finetune_donut_pipeline/PLAN.md`](yolo_finetune_donut_pipeline/PLAN.md)** 참고.

## 환경 & 실행

- 환경: 원본 CORD/도면 학습 노트북은 `torch_211_env` conda (PyTorch 2.11.0+cu130, transformers 5.3.0), 동명 Jupyter 커널.
  `_kardi` 노트북은 `kardi_env`. VS Code Jupyter 에서 개발.
- **YOLO → Donut 파이프라인은 전 단계 단일 커널 `kardi_env`** 로 실행 (검출+인식+학습 의존성 모두 설치됨:
  torch 2.11.0 · transformers 4.57.6 · ultralytics · timm · sentencepiece · opencv · PyMuPDF · accelerate · datasets).
  커널 전환·디스크 핸드오프 불필요. (5.3.0 으로 저장한 Donut 체크포인트도 4.57.6 에서 정상 로드 확인.)
- 의존성: `pip install -r requirements.txt`. **transformers ≥ 4.45 필수** (`evaluation_strategy`→`eval_strategy` 개명).
- 셀을 순서대로 실행. Step 0 이 버전과 `torch.cuda.is_available()` 출력 — `False` 면 환경 오설정.
- 노트북은 `os.environ['TQDM_NOTEBOOK']='false'` + 일반 `tqdm`(`tqdm.auto` 아님)을 의도적으로 사용
  — `tqdm.auto` 는 VS Code Jupyter 에서 ipywidget 오류. 그대로 둘 것.

## 아키텍처

모델은 HF `VisionEncoderDecoderModel`:
- **인코더**: Swin Transformer (이미지→특징). 해상도는 `CFG.model.image_size`(`[h,w]`, 기본 `[1280,960]`).
- **디코더**: BART/mBART (토큰 시퀀스 생성). `donut-base` 디코더 위치 임베딩 한계 **768** 이 `max_length` 상한.

**핵심: JSON ↔ 토큰 변환** (학습 노트북에 정의, 테스트 노트북에 재구현)
- `json2token` — dict/list → XML 식 토큰: `{"total":"12500"}` → `<s_total>12500</s_total>`,
  리스트는 `<sep/>` 로 결합, 키는 **역정렬**(결정적 순서).
- `token2json` — `<s_key>...</s_key>` 재귀 정규식으로 역변환.
- 타깃 시퀀스 = `task_prompt + json2token(gt) + eos`. **`task_prompt`(예 `<s_drawing>`)는 디코더 시작 토큰**이기도 함
  — `build_model_and_processor` 가 task_prompt + **라벨의 모든 필드 토큰**을 `add_special_tokens` 로 추가하고
  임베딩 리사이즈 + `decoder_start_token_id`/`pad_token_id` 연결. **태스크를 바꾸면 `task_prompt` 도 바꿀 것.**
- 학습은 **Teacher Forcing**: 라벨을 `max_length`(CORD 512/도면 768)로 패딩, 패드 위치는 `-100` 으로
  `CrossEntropyLoss` 무시. 평가 `compute_leaf_match` 는 leaf 경로 일치율("Leaf-Match Score") 보고.

### 커스텀 파인튜닝 주의사항 (하드웬)
- **`bf16` 사용, `fp16` 금지** — fp16 은 Donut 에서 수치 불안정 → 깨진/0점 출력.
- **필드 토큰 등록** — 모든 JSON 키를 special token 으로 추가해야 함(빌드 단계는 템플릿이 아니라
  **실제 라벨**의 키를 읽음), 아니면 디코더가 못 내보냄.
- **task token ≠ 필드명** — 도면 task 는 `<s_drawing>`, 최상위 필드는 `title_block` (충돌 방지).
- **`token2json` 전 BOS+task_prompt 제거** — 남은 special token 이 정규식 파싱을 깸
  (drawings 노트북의 `output_to_json`/`strip_tags` 가 처리).

## 설정 (CFG)

Step 1 의 단일 `CFG` 딕셔너리로 구동. 실제로 만지는 값:
- `data.dataset_name` — HF 데이터셋명 또는 **`None`(로컬 데이터, `local_train_dir`/`local_val_dir`)**. 도면은 로컬 모드.
- `data.task_prompt` — task token 과 일치(`<s_cord-v2>` / `<s_drawing>`).
- `model.max_length` — 512(CORD) / 768(도면, 디코더 한계).
- `training.num_epochs`/`learning_rate`/`batch_size`/`gradient_accumulation_steps` —
  유효 배치 = batch_size × grad_accum (CORD 2×8=16; 도면 2×2=4, ~50장이라 100 epoch). VRAM 부족 시 batch_size↓ + grad_accum↑.
- `DonutDataset` 는 모드 자동 감지: HF split(`ground_truth`/`gt_parse`) 또는 로컬 디렉터리 경로.

## 데이터 레이아웃

대용량 산출물(`data/`·`checkpoints*/`·`output/`·`datasets/`)은 **Git LFS** 로 추적·커밋한다.
가중치(`*.safetensors`·`*.pt`·`*.pth`)는 `.gitattributes` 의 LFS 필터로 포인터화되어 저장됨.
클론 후 `git lfs install && git lfs pull` 로 실제 바이너리를 받는다(미설치 시 포인터만). LFS 용량이
십수 GB 라 GitHub LFS 무료 한도(각 1 GB/월)를 초과 — 유료 데이터 플랜 필요. `runs/`·`__pycache__/`·
`.venv/`·`donut_base_test.ipynb` 등은 여전히 `.gitignore` 제외.

**로컬 데이터셋 포맷**: `<root>/images/*.{png,jpg,...}` + `<root>/labels/<같은-stem>.json`. stem 이 매칭되는 쌍만 사용.
- `data/raw/` — 영수증 원본. CORD 노트북의 "[선택] 로컬 데이터셋 준비" 셀이 `data/processed/{train,val}/...` 로 분할(`VAL_RATIO=0.1`, seed 42).
- `data/drawings/` — 도면 도메인. `images/`+`labels/`+`annotate_helper.py`. 헬퍼가 `images/` 스캔 후 라벨 없는 이미지에
  빈 JSON 템플릿(`TEMPLATE`: title_block/dimensions/bolt_holes/surface_finish/gdt/threads/date)을 생성 →
  **값은 손으로 채우고** 이미지와 대조 검수(라벨 품질 = 모델 품질).
- `data/processed_drawings/{train,val}/` — 도면 노트북 분할 셀 출력(`VAL_RATIO=0.1`, `DROP_EMPTY=False`), 도면 학습 입력.
- `data/raw_pdf/` — 원본 PDF(래스터화 전). `test_data/` — 임시 추론 입력.

## 체크포인트

`Seq2SeqTrainer` 가 `output_dir` 에 `save_total_limit=3`, `load_best_model_at_end=True`(best=최저 `eval_loss`)로 저장.
학습 셀은 추가로 최종 모델 **+ 프로세서**를 `<output_dir>/final/` 에 저장 — 추론은 둘을 같은 경로에서 로드해야 함.
- CORD → `checkpoints/`, 도면 → `checkpoints_drawings/`.
