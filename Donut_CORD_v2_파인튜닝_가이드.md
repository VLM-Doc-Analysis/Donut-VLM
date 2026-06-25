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

## 10. 다른 태스크로 이식하기

이 파이프라인을 **다른 도메인**으로 옮길 때 바꿀 것은 **단 3가지**뿐이다:

| 바꾸는 것 | CORD-v2 | 예: 도면 |
|---|---|---|
| 데이터셋 | `naver-clova-ix/cord-v2` | 로컬 도면 크롭+라벨 |
| `task_prompt` (새 토큰) | `<s_cord-v2>` | `<s_drawing>` / `<s_element>` |
| 라벨 JSON 스키마 | `menu/total/...` | `title_block` / `dimension/...` |

나머지(`json2token`, 토큰 등록, Teacher Forcing, Trainer, 평가)는 **그대로**.
→ 실제로 `donut_training_drawings.ipynb` · `donut_training_elements.ipynb` 가 이 방식으로 이식한 것이다.

---

> **한 줄 요약**: donut-base를 CORD-v2로 파인튜닝 = ① 정답 JSON을 `task + <s_키>값</s_키> + eos` 토큰열로 바꾸고,
> ② 그 키 토큰들을 vocab에 등록한 뒤, ③ Teacher Forcing으로 학습 → ④ 추론은 `token2json`으로 JSON 복원.
