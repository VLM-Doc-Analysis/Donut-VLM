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
- `donut_CORD_v2_fine_tunned_test_kardi.ipynb` — 로컬 이미지 추론. (파일명의 `kardi` 는 과거 `kardi_env`
  잔재 — 현재는 단일 커널 `donut_vml` 로 실행.)

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
- **요소 Donut 파인튜닝** — 아키텍처의 파인튜닝 메커니즘(json2token, 특수 토큰 등록, Teacher Forcing, bf16 등)을
  **요소 크롭 단위**(도면 전체 아님, 정렬된 요소 크롭 1개)로 적용. **구조화 방식·데이터가 다른 세 노트북**이 있다:
  - **`donut_training_elements_flat.ipynb`** (flat) — 모델은 **통짜 값**(`<s_value>…</s_value>`)만 생성(OCR 집중),
    구조 분해(quantity/nominalValue/tolerance…)는 **사후 정규식** `parse_to_schema` → `checkpoints_elements/final`.
  - **`donut_training_elements_paper.ipynb`** (논문식, Khan et al.) — 모델이 **카테고리별 구조화 JSON 을 직접 생성**.
    `parse_to_schema` 로 **학습 타깃**을 만들고, 추론은 `token2json` + **타입 조건부 디코딩**(YOLO 타입의 필드 태그만
    허용, `suppress_tokens`) → `checkpoints_elements_paper/final`.
  - **`donut_training_elements_paper_hidpy.ipynb`** (논문식 · 고DPI) — paper 와 동일 파이프라인, 데이터만
    고DPI 재취득 크롭(`data/elements_hidpi`, 검수 val 98 = `val_ids.txt` 고정 분리) → `checkpoints_elements_paper_hidpi/final`.
  - **핵심 차이**: `parse_to_schema` 의 위치 — flat=**모델 출력 사후처리** ↔ paper=**학습 타깃 전처리**.
    공통: task token `<s_element>`, 데이터는 `detection/cvat_to_donut.py` 로 CVAT export 에서 생성(`{"<type>":"<value>"}`).
    기호(Ø,⊥,±,°)는 paper 계열의 `USE_UNICODE_SYMBOLS` 토글로 — **paper 기본 `True`(U+XXXX ASCII 인코딩, 논문 재현)**,
    **hidpi 기본 `False`(글리프 토큰 = 기호를 문자 1글자당 토큰 1개로 등록)**, flat 은 글리프 토큰 고정.
    ⚠️ U+XXXX 모드는 디토크나이즈 공백 보정(`decode_symbols`)이 없으면 채점이 크게 왜곡된다(평가리포트 §0-3).
  - **선택 기준(2026-07-02 재학습 기준)**: 동일 데이터(val 197)에서 **paper(0.618) ≈ flat(0.613)** 이고 GD&T 는
    paper 우위(0.488 vs 0.423) — 과거 "flat 이 우위" 결론은 채점 아티팩트였다. 고DPI 검수 val 98 에선
    hidpi 구조화가 **0.888** 로 최고. 상세: `yolo_finetune_donut_pipeline/Element_Donut_평가리포트.md` §0-3.
- 전체 단계별 계획은 **[`yolo_finetune_donut_pipeline/PLAN.md`](yolo_finetune_donut_pipeline/PLAN.md)** 참고.

## 환경 & 실행

- 환경: 단일 conda 환경 **`donut_vml`** (동명 Jupyter 커널, VS Code Jupyter 에서 개발).
  **YOLO 검출 + Donut 인식/학습 의존성이 모두 설치돼 전 단계를 단일 커널로 실행** — 커널 전환·디스크 핸드오프 불필요.
  검증 하드웨어/버전: **RTX 5090**(Blackwell, sm_120, 32GB, driver 580.159.03) ·
  **PyTorch 2.11.0+cu128**(CUDA 12.8 — `torch.cuda.get_arch_list()` 에 `sm_120` 포함 → 네이티브 실행, "no kernel image" 없음) ·
  **transformers 5.12.1** · timm 1.0.27 · sentencepiece 0.2.1 · accelerate 1.14.0 · datasets 5.0.0 ·
  ultralytics 8.4.80 · opencv(cv2) 4.13.0 · PyMuPDF(fitz) 1.27.2.3.
  (과거 문서/파일명의 `torch_211_env`·`kardi_env`·cu130·transformers 4.57/5.3 은 더 이상 존재하지 않는 잔재.)
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

대용량 산출물(`data/`·`checkpoints*/`·`output/`·`datasets/`)은 **로컬에서 노트북/스크립트로 생성**하며
**git 에 커밋하지 않는다**(LFS 비용 회피). 가중치(`*.safetensors`·`*.pt`·`*.pth`)·데이터셋 이미지
(`*.png`·`*.jpg`·`*.jpeg`)·PDF 는 `.gitignore` 로 스테이징에서 제외되고, `.gitattributes` 의 LFS 필터는
과거 호환·문서이미지 예외 처리 용도로만 남겨둔다. 따라서 **클론 후 모델·데이터는 직접 학습/재생성**해야 한다.
`runs/`·`__pycache__/`·`.venv/`·`donut_base_test.ipynb` 등도 `.gitignore` 제외.

> ⚠️ **LFS 커밋 금지(사용자 지침).** `git add` 시 LFS 대상(체크포인트·`*.safetensors`·데이터 이미지·PDF)이
> 딸려오지 않게 **경로를 명시**해 add 하고, 이미 추적 중인 LFS 파일이라도 **새로 커밋하지 말 것**.
> 커밋 가능한 것은 `.md`·`.py`·`.ipynb`(노트북은 LFS 아님)와 아래 "문서용 이미지" 예외뿐.

### 이미지 LFS 정책 (중요)
**`*.png`/`*.jpg`/`*.jpeg` 는 전역 LFS 대상이지만, 문서·노트북·README 에 임베드되어 GitHub 에서
렌더돼야 하는 이미지는 `.gitattributes` 에서 `-filter -diff -merge` 로 LFS 를 해제해 일반 git blob 으로 둔다.**
- **이유**: GitHub 노트북(.ipynb) 뷰어는 상대경로 이미지를 `raw.githubusercontent.com` 에서 가져오는데,
  이 호스트는 LFS 파일에 대해 **포인터 텍스트(~131B)만 반환**해 이미지가 깨진다.
  (`.md` 는 `github.com/.../raw/` → media 리다이렉트라 LFS 도 렌더되지만, LFS 한도 초과 시 깨질 수 있어 동일하게 blob 권장.)
- **검증**: 반드시 `raw.githubusercontent.com/<owner>/<repo>/main/<path>` 로 확인 —
  `content_type` 이 `image/*` 이고 크기가 실제 이미지면 OK, `text/plain`·131B 면 LFS 포인터(=실패).
  (`github.com/.../raw/` 는 media 로 리다이렉트돼 "되는 것처럼" 보이니 검증 호스트로 쓰지 말 것.)
- **현재 blob 으로 둔 문서용 이미지**(`.gitattributes` 참고): `assets/*.png|*.jpg`(README 용),
  `SROIE_donut/assets/donut.png`, `**/assets/donut_architecture.jpg`, `**/hftuner/assets/*.png`,
  `**/detection/report_assets/*.png`, `test_data/CORD_Test_Data.png`.
- **새 문서용 이미지 추가 시**: `.gitattributes` 에 해당 경로의 LFS 해제 규칙을 넣고
  `git add --renormalize <경로>` 로 포인터→실제 blob 전환 후 커밋. **데이터셋 크롭 이미지는 커밋하지 않음(`.gitignore` 제외).**

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
