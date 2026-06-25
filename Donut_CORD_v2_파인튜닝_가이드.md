# donut-base → CORD-v2 파인튜닝 가이드 (영수증 파싱 모델 만들기)

> 범용 `donut-base` 를 **CORD-v2(영수증)** 로 추가 학습해 **영수증 → JSON 변환 전용 모델**을 만드는 전체 과정.
> 실제 구현: [`donut_training.ipynb`](donut_training.ipynb) · 추론 확인: [`donut_CORD_v2_fine_tunned_test.ipynb`](donut_CORD_v2_fine_tunned_test.ipynb)
> 출력 토큰 구조: [`CORD_v2_토큰_종류_가이드.md`](CORD_v2_토큰_종류_가이드.md)

---

## 0. 큰 그림

```
donut-base (이미지에서 글자 '읽기'만 아는 범용 모델)
   │
   └─ CORD-v2 (영수증 800장 + 정답 JSON) 로 추가 학습
   │
   ▼
영수증 파싱 전용 Donut
   입력:  영수증 이미지 + 태스크 토큰(<s_cord-v2>)
   출력:  gt_parse JSON 을 '토큰 시퀀스'로 생성  →  token2json 으로 dict 복원
```

모델 구조 = `VisionEncoderDecoderModel`:
- **인코더 = Swin Transformer** — 이미지 → 시각 특징
- **디코더 = mBART** — 특징을 받아 **토큰 시퀀스** 생성 (위치임베딩 한계 **768** = `max_length` 상한)

---

## 1. 데이터 — CORD-v2

- `naver-clova-ix/cord-v2` : **train 800 / val 100 / test 100**.
- 각 샘플 `ground_truth` 중 **`gt_parse`** 만 학습 정답으로 사용(`menu / sub_total / total / void_menu` 구조).
  `valid_line`·`roi` 등은 다른 OCR용이라 **무시**.

```python
dataset = load_dataset("naver-clova-ix/cord-v2")
gt = json.loads(sample["ground_truth"])["gt_parse"]   # ← 이 dict 가 학습 정답
```

---

## 2. 핵심 메커니즘 — JSON ↔ 토큰 변환

Donut은 JSON을 **XML 스타일 토큰 시퀀스**로 생성하도록 학습한다.

```python
json2token({"menu": {"nm": "치킨", "price": "12000"}})
# → "<s_menu><s_nm>치킨</s_nm><s_price>12000</s_price></s_menu>"
#   · 리스트는 <sep/> 로 결합
#   · 키는 역정렬해 항상 같은 순서(결정적)
```

**학습 타깃 시퀀스**:
```
target = task_prompt("<s_cord-v2>") + json2token(gt_parse) + eos("</s>")
```
추론 때는 역변환 `token2json` 으로 다시 dict 복원.

---

## 3. 특수 토큰 등록 ⭐ (가장 중요 — 빠뜨리면 안 됨)

디코더는 **vocab에 있는 토큰만** 생성할 수 있다. 그래서 학습 전에 반드시:

```python
# ① task token + 라벨에 등장하는 모든 필드 토큰 수집
new_tokens = ["<s_cord-v2>", "<s_menu>","</s_menu>", "<s_nm>","</s_nm>", ...]
processor.tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})

# ② 디코더 임베딩 행렬을 늘어난 vocab 크기에 맞춰 리사이즈
model.decoder.resize_token_embeddings(len(processor.tokenizer))

# ③ 시작/패딩 토큰 연결
model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids("<s_cord-v2>")
model.config.pad_token_id           = tokenizer.pad_token_id
```

- **필드 토큰은 템플릿이 아니라 '실제 라벨'에서 수집** — 라벨에 없는 키는 모델이 못 뱉는다.
- `<s_cord-v2>` 는 **디코더의 시작 토큰**이기도 하다(태스크 지정 + 생성 시작 신호).

> 📝 **메모**: 키 토큰은 `add_special_tokens`(이 프로젝트 방식)·`add_tokens` **둘 다 가능**(끝에 `resize_token_embeddings`만 하면 됨). 값 내용(숫자·기호)은 `skip_special_tokens=True`에도 **살아남아야** 하므로 `add_tokens`로 등록한다. — 참고로 공개 CORD-v2 체크포인트는 키 토큰도 `add_tokens`라 `all_special_tokens`엔 안 보인다(둘 다 정상 작동).

---

## 4. Dataset — Teacher Forcing 라벨 만들기

```python
class DonutDataset:
    def __getitem__(self, idx):
        pixel_values = processor(image, return_tensors="pt").pixel_values   # 이미지 전처리
        target = task_prompt + json2token(gt_parse) + eos
        labels = tokenizer(target, max_length=512, padding="max_length",
                           truncation=True).input_ids
        labels[labels == pad_token_id] = -100    # ★ 패딩 위치는 loss 에서 제외
        return {"pixel_values": pixel_values, "labels": labels}
```

- **Teacher Forcing**: 정답 토큰을 한 칸씩 밀어 디코더 입력으로 주고, 다음 토큰을 맞히게 학습.
- **`-100`**: `CrossEntropyLoss` 가 자동으로 무시 → 패딩 자리에서 loss 계산 안 함.

---

## 5. 학습 설정 (CFG) — CORD-v2 실제 값

| 항목 | 값 | 의미 |
|---|---|---|
| `pretrained_model_name` | `naver-clova-ix/donut-base` | 출발점 |
| `task_prompt` | `<s_cord-v2>` | 태스크 토큰 |
| `max_length` | **512** | 디코더 생성 최대 토큰 (상한 768 = 위치임베딩 한계) |
| `image_size` | `[1280, 960]` | 인코더 입력 해상도 |
| `num_epochs` | 30 | 전체 반복 횟수 |
| `batch_size × grad_accum` | **2 × 8 = 16** | 유효 배치 (VRAM 부족 시 batch↓·accum↑) |
| `learning_rate` / `warmup` | 3e-5 / 300 | |
| 정밀도 | fp16(원본) → **bf16 권장** | ⚠️ 아래 주의 |

```python
args = Seq2SeqTrainingArguments(
    output_dir="checkpoints", num_train_epochs=30,
    per_device_train_batch_size=2, gradient_accumulation_steps=8,
    learning_rate=3e-5, warmup_steps=300, weight_decay=0.01,
    bf16=True,                          # ← 권장 (fp16 대신)
    eval_strategy="steps", eval_steps=1000, save_steps=1000,
    save_total_limit=3, load_best_model_at_end=True,
    metric_for_best_model="eval_loss", greater_is_better=False,
)
trainer = Seq2SeqTrainer(model, args, train_dataset=train_ds, eval_dataset=val_ds)
```

---

## 6. 평가 — Leaf-Match Score

`compute_leaf_match`: 예측 JSON과 정답 JSON을 **leaf 경로**로 펼쳐(`menu/nm`, `total/total_price` …) **일치율**을 보고한다. (exact 보다 부분 진전을 보기 좋음)

---

## 7. 학습 실행 & 저장

```python
trainer.train()
trainer.save_model("checkpoints/final")
processor.save_pretrained("checkpoints/final")   # ★ 모델 + 프로세서 같은 경로에 함께 저장
```
추론 때는 **둘을 같은 경로에서** 로드해야 토크나이저(필드 토큰 포함)가 일치한다.

---

## 8. 추론 (만든 모델 확인)

```python
proc  = DonutProcessor.from_pretrained("checkpoints/final", use_fast=False)
model = VisionEncoderDecoderModel.from_pretrained("checkpoints/final").eval()

dec = proc.tokenizer("<s_cord-v2>", add_special_tokens=False, return_tensors="pt").input_ids
out = model.generate(pixel_values, decoder_input_ids=dec,
                     max_length=512, eos_token_id=proc.tokenizer.eos_token_id,
                     bad_words_ids=[[proc.tokenizer.unk_token_id]])
seq    = proc.batch_decode(out)[0]
result = token2json(seq)     # → {"menu":{...}, "total":{...}}
```
(이게 `donut_CORD_v2_fine_tunned_test.ipynb` 의 추론 흐름)

---

## ⚠️ 꼭 지킬 4가지 (프로젝트 하드원 교훈)

| # | 규칙 | 안 지키면 |
|---|---|---|
| 1 | **필드 토큰 등록 필수** (실제 라벨의 모든 키 → `add_special_tokens` → `resize_token_embeddings`) | 디코더가 키를 못 만듦 |
| 2 | **task token ≠ 필드명** (`<s_cord-v2>` vs `menu` 분리) | 토큰 충돌 |
| 3 | **`bf16` 권장, `fp16` 주의** (fp16은 Donut에서 수치 불안정 → 깨진 출력) | 0점/깨진 출력 |
| 4 | **`token2json` 전 BOS·task 토큰 제거** | 정규식 파싱 깨져 점수 0 |

> 메모: `max_length` 상한은 디코더 위치임베딩 **768**. CORD는 512로 충분.

---

## 9. 실행 순서 요약 (= `donut_training.ipynb`)

```
Step 0  환경 확인 (torch / cuda)
Step 1  CFG 설정 (dataset_name="naver-clova-ix/cord-v2", task_prompt="<s_cord-v2>")
Step 2  build_model_and_processor  (필드 토큰 등록 + 임베딩 리사이즈)
Step 3  DonutDataset 구성  (target = task + json2token(gt) + eos, pad→-100)
Step 4  Seq2SeqTrainer.train()
Step 5  checkpoints/final 에 모델 + 프로세서 저장
Step 6  추론으로 확인  (generate → token2json)
```

---

## 10. 다른 태스크로 이식하기 (영수증 → 내 도메인)

### 왜 "3가지만" 바꾸면 되나

Donut 학습 기계는 **도메인을 모른다.** 하는 일은 늘 똑같다:

```
이미지 → (인코더) 특징 → (디코더) <s_키>값</s_키> 토큰열 → token2json → JSON
```

`json2token` / 토큰 등록 / Teacher Forcing / Trainer / 평가는 **키가 `menu`든 `dimension`이든 상관없이** 똑같이 작동한다.
그래서 바꾸는 건 **"무엇을 읽을지"** 세 가지뿐이다.

### 바꿀 것 ① — 데이터셋 (어떤 이미지+정답으로 배울지)

| | CORD-v2 (HF) | 내 도메인 (로컬) |
|---|---|---|
| `data.dataset_name` | `"naver-clova-ix/cord-v2"` | **`None`** |
| 데이터 위치 | 자동 다운로드 | `local_train_dir` / `local_val_dir` 지정 |

```python
"data": {
    "dataset_name": None,                               # ← HF 대신 로컬 모드
    "local_train_dir": "data/processed_drawings/train",
    "local_val_dir":   "data/processed_drawings/val",
}
```
- 로컬 포맷: `<root>/images/*.png` + `<root>/labels/<같은이름>.json` (stem이 맞는 쌍만 사용).
- `DonutDataset`이 **HF/로컬 모드를 자동 감지**하므로 코드는 안 고쳐도 된다.

### 바꿀 것 ② — `task_prompt` (무슨 과제인지 알리는 새 시작 토큰)

```python
"task_prompt": "<s_drawing>",   # CORD: <s_cord-v2> → 도면: <s_drawing> (요소: <s_element>)
"max_length": 768,              # JSON 길이에 맞게 (도면 768, 요소 128, ≤768 상한)
```
- 이 토큰은 **새로 만든 것**이라 토크나이저에 등록돼야 한다 — §3의 `add_special_tokens`가 자동 처리.
- ⚠️ **task token ≠ 최상위 필드명.** 도면은 task=`<s_drawing>`, 최상위 필드=`title_block`으로 **분리**(충돌 방지).
- 추론할 때도 **같은 새 토큰**을 디코더 시작으로 줘야 한다.

### 바꿀 것 ③ — 라벨 JSON 스키마 (어떤 키/구조로 뽑을지)

내 도메인 정답 JSON 구조를 정의한다. 이게 곧 **모델이 배울 필드 토큰**이 된다.

```jsonc
// CORD-v2
{ "menu": {"nm":"치킨","price":"12000"}, "total": {"total_price":"12000"} }

// 도면(예)
{ "title_block": {"part_no":"A-1370","material":"SS400"},
  "dimensions": [ {"value":"Ø65"}, {"value":"R15"} ] }
```
→ 학습 시작 때 이 라벨들에서 키(`title_block`,`part_no`,`dimensions`,`value` …)를 모아 토큰으로 등록한다.
**스키마에 쓴 키 = 라벨에 실제로 있는 키** 여야 한다(없으면 모델이 못 뱉음).

### 안 바꾸는 것 (그대로 재사용)

| 그대로 | 역할 |
|---|---|
| `json2token` / `token2json` | JSON ↔ 토큰 변환 |
| `build_model_and_processor` | 라벨에서 키 수집 → 토큰 등록 → 임베딩 리사이즈 |
| `DonutDataset` | target = `task + json2token(gt) + eos`, pad→`-100` |
| `Seq2SeqTrainer` 설정 | 학습 루프 |
| `compute_leaf_match` / field-F1 | 평가 |

### 이식 체크리스트

- [ ] 데이터를 `images/*.png` + `labels/*.json`(같은 stem)로 준비
- [ ] `CFG.data.dataset_name=None` + `local_train_dir`/`local_val_dir` 지정
- [ ] `CFG.data.task_prompt`를 **새 토큰**으로 (예 `<s_drawing>`)
- [ ] `CFG.model.max_length`를 JSON 길이에 맞게 (≤768)
- [ ] 라벨 JSON 스키마 확정 — **최상위 필드명 ≠ task token**
- [ ] 학습 → `checkpoints_<도메인>/final` 에 모델+프로세서 저장
- [ ] 추론 시 task_prompt를 **같은 새 토큰**으로

### 실수하기 쉬운 점 (gotcha)

1. **task_prompt만 바꾸고 추론에선 옛 토큰 사용** → 디코더 시작이 안 맞아 출력이 깨진다.
2. **task token = 필드명** 으로 둠 → 토큰 충돌.
3. **라벨에 없는 키를 스키마에 기대** → 등록 안 돼 못 뱉음.
4. **fp16 사용** → 커스텀 도메인에서 수치 불안정. **bf16 권장.**
5. **max_length 768 초과** → 디코더 위치임베딩 한계로 뒤가 잘림.
6. **`token2json` 전 BOS·새 task 토큰 미제거** → 정규식 파싱이 깨져 점수 0.

### 실제 사례 (이 저장소 — 같은 기계에 3가지만 바꿔 끼움)

| 노트북 | task_prompt | max_length | 데이터 | 스키마 |
|---|---|---|---|---|
| `donut_training.ipynb` | `<s_cord-v2>` | 512 | HF cord-v2 | `menu`/`total`… |
| `donut_training_drawings.ipynb` | `<s_drawing>` | 768 | 로컬 도면 | `title_block`/`dimensions`… |
| `donut_training_elements.ipynb` | `<s_element>` | 128 | 로컬 요소 크롭 | `dimension`/`gdt`… (구조화) |

---

> **한 줄 요약**: donut-base를 CORD-v2로 파인튜닝 = ① 정답 JSON을 `task + <s_키>값</s_키> + eos` 토큰열로 바꾸고,
> ② 그 키 토큰들을 vocab에 등록한 뒤, ③ Teacher Forcing으로 학습 → ④ 추론은 `token2json`으로 JSON 복원.
