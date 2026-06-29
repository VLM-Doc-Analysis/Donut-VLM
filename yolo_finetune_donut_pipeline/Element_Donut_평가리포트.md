# Element Donut 인식모델 평가 리포트

> 대상 체크포인트: `checkpoints_elements/final` (best = step 450 / epoch 4)
> 평가일: 2026-06-24 · 환경: `donut_vml` (평가 당시 스택: torch 2.11.0+cu130 · transformers 4.57.6 · CUDA — **현재는 cu128 · transformers 5.12.1**)
> 평가셋: `data/processed_elements/val` (197장) · 지표: Leaf-Match Score (노트북 Step 5 동일 로직)
>
> ⚠️ **이 0.51% 는 "기호 토큰 추가 前" 베이스라인이다.** 이후 토크나이저 확장·재학습으로 타입붕괴가 풀리고
> (타입 91%), 값-전용 Donut 8.1% · TrOCR ~15% 로 올라섰다(→ [`성능미달_원인_및_해결방안`](Element_Donut_성능미달_원인_및_해결방안.md) §1).
> **0.51% 를 현재 성능으로 인용하지 말 것.** 병목은 **데이터 양(논문 ~11–13k 크롭 vs 우리 ~2k, ≈1/7) + 품질**
> 둘 다(논문 §3 확정 — *Multi-Stage* ~13,000→0.963, *From Drawings* 11,469→0.935).

---

## 0. 최신 재평가 (2026-06-29 · field-F1)

> 현행 코드(견고 디코딩 + field-level 지표)로 **현재 체크포인트들**을 val 197장에서 재평가한 결과.
> 지표: **Field-F1**(논문과 동일 계열, P=R=F 근사) · charsim(값 글자유사도) · exact(구조화 dict 필드별 완전일치, 공백무시).

| 모델 (체크포인트) | Field-F1 | charsim | exact | Halluc |
|:--|--:|--:|--:|--:|
| **flat** (`checkpoints_elements/final`, 값+정규식) | **0.444** | **0.733** | 41.1% | 0.556 |
| paper 구조화 · token-mode (`checkpoints_elements_paper/final`) | 0.312 | 0.572 | 28.4% | 0.689 |
| paper 구조화 · **U+XXXX** (`checkpoints_elements_paper_uxxxx/final`) | 0.349 | 0.598 | 31.5% | 0.651 |

<sub>구조화 두 모델은 타입 조건부 디코딩(typecond) ON 기준. typecond OFF 는 paper 0.299 / U+XXXX 0.344 로 거의 동일(미미한 차이).</sub>

### 클래스별 (flat = 최고 성능본)

| class | F1 | charsim | exact | n |
|:--|--:|--:|--:|--:|
| Dimension | 0.415 | 0.704 | 40.0% | 140 |
| Surface_Roughness | 0.476 | 0.785 | 47.6% | 21 |
| GD&T_FCF | 0.460 | 0.831 | 18.8% | 16 |
| Datum | 0.636 | 0.818 | 63.6% | 11 |
| Hole_Callout | 0.556 | 0.790 | 55.6% | 9 |

### 핵심 발견
1. **이 리포트의 옛 0.51%(§1~)는 무효** — 현 모델은 "사용 불가"가 아니라 **flat Field-F1 0.444 · charsim 0.73**(값을 평균 73% 글자 수준으로 읽음, 전 클래스 작동).
2. **flat(0.444) > 구조화(0.31~0.35)** — 소량 데이터에선 "값+정규식"이 "구조 직접생성"보다 안전(문서 결론과 일치).
3. **U+XXXX 인코딩이 구조화 모델 개선**: token-mode 0.312 → U+XXXX 0.349 (+3.7pt). 특히 **Surface_Roughness 0.476→0.714(+24pt)**. (단, 두 런이 완전 통제된 A/B 는 아님 — 시사적)
4. **GD&T 는 어떤 방식으로도 미해결**(구조화 F1~0.09·exact 0%) → 희소(16장)+복잡 구조의 라벨/크롭 **품질** 문제. 기호 인코딩·typecond 로 안 풀림.

### 재현
```bash
conda activate donut_vml
cd yolo_finetune_donut_pipeline
# 평가 스크립트(노트북 helper 재사용): 체크포인트 dir + [typecond]
python eval_fieldf1.py ../checkpoints_elements/final
python eval_fieldf1.py ../checkpoints_elements_paper_uxxxx/final typecond
```

> 결론: 같은 데이터에서 flat>구조화, U+XXXX>token(방식도 영향). 그러나 논문(0.93–0.96) 대비 근본 격차는 **데이터 양(≈1/7) + 품질** 둘 다 — GD&T 등 희소 클래스 확충·정제가 핵심 레버.

---

## 0-1. 고DPI 부트스트랩 런 (2026-06-29) — ⚠️ 미검증(순환)

고DPI 재취득 파이프라인(`extract_hidpi_crops`→`prefill`→`train_dpi_ab`)으로 첫 학습. **벡터 900/스캔 native** 크롭 651개(제외 41 후), `image_size 768`, flat.

| 구성 | val GT | Field-F1 | charsim | exact |
|:--|:--|--:|--:|--:|
| 순환(random split) | pre-fill(모델 초안) | 0.835 | 0.919 | 83.5% |
| held-out(val 학습제외) | "수용" 100개(편집 4 / 초안수용 96) | 0.890 | 0.937 | 89.0% |

> ⚠️ **두 수치 모두 신뢰 불가(순환).** val GT 가 사실상 **모델 초안(pre-fill)** 이다(held-out 도 96/100 이 미편집 수용). 0.835→0.890 차이는 *순환 제거가 아니라* split·변동.
> - 진짜 비순환 수치를 얻으려면 **val 값을 이미지 대조해 직접 교정/확정**해야 함(`review_val.html` → Enter/수정).
> - 이 런이 확인한 것: **고DPI 파이프라인·image_size 768 학습이 정상 작동**(plumbing + 부트스트랩). **품질 레버(고DPI가 정확도를 올리나)는 미확정.**
> - GD&T 는 로컬 소스 3개뿐이라 이 셋으로 평가 불가 → Phase 2(소스 확충) 필요.

---

## 1. 한 줄 결론 (2026-06-24 구버전 baseline — §0 으로 대체됨)

> ⚠️ 아래는 **기호 토큰 추가 前** 베이스라인 진단이다. 현재 성능은 위 **§0** 을 볼 것.

**(당시) 모델은 사실상 실패(사용 불가) 상태. Leaf-Match 0.51%로 거의 무작위 수준이며 재학습 필요.**

---

## 2. 종합 지표 (검증셋 197장)

| 지표 | 값 | 의미 |
|---|---|---|
| **Leaf-Match Score** | **0.51%** | 정답 leaf(필드 값) 일치율 평균 |
| **Exact-Match Rate** | **0.51%** | 예측 JSON 이 정답과 완전히 동일한 비율 |

> 두 지표가 같다는 건, 부분적으로라도 맞은 샘플이 거의 없다는 뜻(맞으면 통째로 맞고, 대부분 통째로 틀림).

### 클래스별 분해

| class | n | leaf% | exact% |
|---|--:|--:|--:|
| Dimension | 140 | 0.71 | 0.71 |
| Surface_Roughness | 21 | 0.00 | 0.00 |
| GD&T_FCF | 16 | 0.00 | 0.00 |
| Datum | 11 | 0.00 | 0.00 |
| Hole_Callout | 9 | 0.00 | 0.00 |

- 다수 클래스(**Dimension, 전체의 ~65%**)만 0.71%, **나머지 4개 클래스는 전부 0%**.
- 모델이 소수 클래스를 전혀 처리하지 못함 → 다수 클래스로 쏠림(majority collapse) 패턴.

---

## 3. 실패 양상 (raw 출력 증거)

`generate()` 원본 디코딩(특수 토큰 포함):

```
GT : {'Datum': '⊥ A'}       RAW : <s_element><s_Dimension> A</s>
GT : {'Dimension': '30°'}    RAW : <s_element><s_Dimension> 30</s_Dimension></s_Dimension>…(무한 반복)…</s>
GT : {'Dimension': '90±0.1'} RAW : <s_element><s_Dimension> 900<unk></s>
GT : {'Dimension': '25'}     RAW : <s_element><s_Dimension> 25</s>           (닫힘 태그 없음)
GT : {'Datum': 'A'}          RAW : <s_element><s_Dimension> A</s_Dimension><unk>A</s>
```

확인된 네 가지 고장:

1. **타입 토큰 붕괴(type collapse)**
   입력 종류와 무관하게 거의 항상 `<s_Dimension>` 을 출력. → Datum/GD&T/Surface/Hole 전멸.
   (특수 토큰 자체는 `added_tokens.json` 에 5종 모두 정상 등록되어 있음 — 등록 문제가 아니라 학습 실패)

2. **디코더 퇴화 반복(degenerate repetition)**
   `</s_Dimension></s_Dimension>…` 무한 반복 → 닫힘 태그 누락/중복으로 `token2json` 파싱 실패.

3. **기호 손실(tokenizer OOV)**
   `±`, `°`, `⊥`, `⏥`, `Ⓜ` 등이 `<unk>` 로 나오거나 사라짐.
   base 모델(`donut-base`, mBART SentencePiece)의 어휘에 **도면 기호가 없어 출력 자체가 불가능**.

4. **값도 부분적으로만**
   숫자 일부는 읽지만(`25`, `30`) 부호·기호가 붙은 값은 거의 다 틀림(`30°→30`, `90±0.1→900<unk>`).

> 참고: 디코딩에 `no_repeat_ngram_size=3`, `repetition_penalty=2.0` 을 적용해 반복을 억제한 재평가에서도
> 점수는 **0.51% 그대로**. 즉 반복은 부차적 증상이고, 핵심 원인은 타입붕괴·기호·값 오류.

---

## 4. 원인 분석 — 학습 곡선

`checkpoints_elements/checkpoint-700/trainer_state.json` 기준:

```
eval_loss : 4.97(ep0.5) → 1.62(ep1.3) → 1.02(ep4, 최저) → 1.05 → 1.07 ↑   [조기종료 ep6]
train_loss: 10.7 → … → 0.31(ep6)
best_metric(eval_loss) = 1.0174  @ step 450 (= final 에 저장된 모델)
```

- **과적합(overfitting)**: train 0.31 vs eval 1.02 — epoch 4 이후 eval 발산.
  best 체크포인트(step 450)가 `final/` 에 저장됐지만 그조차 eval_loss ~1.0 으로 **미수렴**.
- **데이터 불균형 + 소규모**: 학습 1778장 중 65%가 Dimension → 모델이 "Dimension 만 찍기"로 도피.
- **설계상 난이도**: 타입은 YOLO 가 이미 알고 있는데 Donut 에게도 타입 생성을 시켜 collapse 를 자초.

### 데이터 분포 (참고)

| class | train+val 합계 | 비율 |
|---|--:|--:|
| Dimension | 1284 | 65.0% |
| Surface_Roughness | 292 | 14.8% |
| GD&T_FCF | 180 | 9.1% |
| Datum | 149 | 7.5% |
| Hole_Callout | 70 | 3.5% |

---

## 5. 개선 권장 (영향 큰 순서)

1. **토크나이저에 도면 기호 추가** — `Ø ± ° ⊥ ∥ ⌀ ⏥ Ⓜ √ ⊿` 등을 토큰으로 등록.
   현재는 `<unk>` 라 정답을 **출력 자체가 불가능**(정확도 상한이 막혀 있음). → 최우선.
2. **타입 생성 제거 / 프롬프트화** — 라벨을 `{value}` 만 두거나,
   YOLO 가 준 type 토큰(`<s_Dimension>` 등)을 **디코더 시작 프롬프트로 강제 입력** → 타입붕괴 원천 차단.
3. **불균형 보정** — 소수 클래스 oversampling 또는 클래스 균등 샘플링.
4. **데이터 확충·정규화** — 1778장은 부족. label 표기 일관화(예: `30°` vs `30`, 공백/부호 규칙 통일).
5. **(보조) 추론 디코딩** — `no_repeat_ngram_size`, `repetition_penalty` 로 반복 루프 방지(근본 해결 아님).

---

## 6. 재현 방법

```bash
conda activate donut_vml
# 노트북: donut_training_elements_flat.ipynb → Step 5b (cell 23) 실행
#   - 체크포인트: checkpoints_elements/final
#   - 평가셋    : data/processed_elements/val (197장)
#   - 지표      : Leaf-Match Score (= 정답 leaf 일치율 평균)
```

> 본 리포트의 수치는 노트북 Step 5 와 동일한 `token2json` / `compute_leaf_match` 로직으로 산출했으며,
> 클래스별 분해·Exact-Match·raw 출력 덤프를 보강해 진단한 결과입니다.
