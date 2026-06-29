# Element Donut 인식모델 평가 리포트

> 대상 체크포인트: `checkpoints_elements/final` (best = step 450 / epoch 4)
> 평가일: 2026-06-24 · 환경: `donut_vml` (평가 당시 스택: torch 2.11.0+cu130 · transformers 4.57.6 · CUDA — **현재는 cu128 · transformers 5.12.1**)
> 평가셋: `data/processed_elements/val` (197장) · 지표: Leaf-Match Score (노트북 Step 5 동일 로직)
>
> ⚠️ **이 0.51% 는 "기호 토큰 추가 前" 베이스라인이다.** 이후 토크나이저 확장·재학습으로 타입붕괴가 풀리고
> (타입 91%), 값-전용 Donut 8.1% · TrOCR ~15% 로 올라섰다(→ [`성능미달_원인_및_해결방안`](Element_Donut_성능미달_원인_및_해결방안.md) §1).
> **0.51% 를 현재 성능으로 인용하지 말 것.** 또한 "데이터 양 부족" 진단은 2026-06-29 재검증으로
> "양이 아니라 품질·학습설정"으로 정정됐다(논문 B 는 1,406 크롭으로 F1 0.963).

---

## 1. 한 줄 결론

**현재 모델은 사실상 실패(사용 불가) 상태입니다. Leaf-Match 0.51%로 거의 무작위 수준이며, 재학습이 필요합니다.**

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
