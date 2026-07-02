# Element Donut — 토크나이저에 도면 기호 추가 가이드

> 대상: `donut_training_elements_flat.ipynb` · Step 2 `build_model_and_processor` (cell id `cell-model-code`)
> 배경: `checkpoints_elements/final` 의 Leaf-Match 0.51% 실패 원인 중 하나가 **값(value) 문자의 토크나이저 미수용**.
> base 모델 `naver-clova-ix/donut-base` (XLM-R SentencePiece, **NFKC 정규화** 적용)
> 작성/검증일: 2026-06-24 · 환경: `donut_vml`

---

## 0. 한 줄 요약

도면 값에 쓰이는 **기호(`⊥ ∡ ∥ …`)와 숫자 `0-9`** 가 base 토크나이저 어휘에 없어 `<unk>` 로 깨졌다.
→ 이 글자들을 **`add_tokens`(일반 토큰)** 로 등록하고 임베딩을 확장해 해결. 값 `<unk>` 비율 **19.3% → 0.05%**.

> ✅ **후속 보강 (2026-07-02)** — 본 가이드의 원리는 그대로 유효하며, 두 가지가 추가됐다:
> 1. **스캔 범위를 합성 라벨까지 확대** — train 에 병합되는 `elements_synth` 라벨을 스캔에서 빼먹으면
>    합성 전용 기호(`∥ ⌖ ⏥` 등)가 `<unk>` 로 학습된다(실제 발생했던 버그). 지금은 세 노트북 모두
>    `scan_dirs` 에 synth 포함(OOV 0/4,475 확인).
> 2. **added token 의 디토크나이즈 공백** — 숫자 0-9 를 added token 으로 등록하면 decode 시 토큰 경계에
>    공백이 삽입된다(§5 의 `'M6 X1 .0'` 이 그 흔적). 글리프 모드에선 공백무시 채점으로 무해하지만,
>    **U+XXXX 모드에선 `U+00 D8` 처럼 hex 가 쪼개져 글리프 복원이 전부 실패**한다 — `decode_symbols` 의
>    공백 보정 필수. 상세: [`평가리포트`](Element_Donut_평가리포트.md) §0-3.

---

## 1. 문제 진단

검증/학습 라벨의 모든 값을 base 토크나이저로 인코딩 → **381/1975 (19.3%)** 가 `<unk>` 포함.

```
GT : {'Dimension':'90±0.1'}  RAW : <s_element><s_Dimension> 900<unk></s>   ← 값이 깨짐
GT : {'Datum':'⊥ A'}         RAW : <s_element><s_Dimension> A</s>          ← ⊥ 소실
```

어휘에 없는 글자(단독 인코딩 기준):

| 글자 | 의미 | 값에 등장 |
|---|---|--:|
| `⊥` | 직각도(perpendicularity) | 179 |
| `∡` | 경사도(angularity) | 54 |
| `∥` | 평행도(parallelism) | 14 |
| `⏥` | 평면도(flatness) | 5 |
| `△ ⟂ ⌒ ⊘ ↓ ━` | 기타 GD&T/방향 | 1~2 |
| `�(U+FFFD)`, `️(U+FE0F)`, `Т(키릴)` | **라벨 깨짐/노이즈** | 1~2 |

---

## 2. 핵심 개념: `add_tokens` vs `add_special_tokens`

| | 용도 | 디코딩(`skip_special_tokens=True`) |
|---|---|---|
| `add_special_tokens` | **구조** 토큰 `<s_Dimension>` `<sep/>` | **사라짐** |
| `add_tokens` | **내용(값)** 글자 `⊥ ∡ ±` `0-9` | **유지됨** ✅ |

- 구조 토큰(`<s_{key}>`)은 기존 코드가 이미 `add_special_tokens` 로 처리 → 그대로 둠.
- **값 문자는 출력 내용**이므로 반드시 `add_tokens`. (`add_special_tokens` 로 넣으면 디코딩에서 날아감)
- 어휘가 늘었으니 마지막에 `model.decoder.resize_token_embeddings(len(tokenizer))` 한 번.

---

## 3. ⚠️ 발견한 숨은 지뢰 2가지 (가장 중요)

### 지뢰 ① 문자열 중간의 숫자 `1` 이 OOV

donut-base 어휘엔 `▁1`(공백 뒤 형태)만 있고 **문자열 중간의 맨 `1` 조각이 없다.**

```
"0.1"     → ['▁0', '.', '<unk>']     # '.' 다음 '1' 이 unk
"M6X1.0"  → ['▁M','6','X','<unk>','.','0']
"1"       → ['▁1']                    # 단독이면 정상 → 단독 검사로는 못 잡음!
"01"      → ['▁0','1']                # 이마저 정상 → OOV 가 '컨텍스트 의존'
```

- 치수값(숫자)이 핵심인 태스크에서 **치명적**. 19% 손상의 큰 축.
- **글자 단독 검사로 절대 안 잡힌다** → 숫자는 `list("0123456789")` 로 **명시적·통째 등록**.

### 지뢰 ② 원형 재질조건 기호의 NFKC 충돌

`Ⓜ Ⓛ Ⓟ Ⓢ Ⓕ Ⓣ` 는 NFKC 정규화로 **평문 `M L P S F T` 로 분해**된다.

```
add_tokens(["Ⓜ","Ⓛ",...]) 후:
  "TOP" → "ⓉOⓅ"     # 평문 T,P 가 원형기호로 하이재킹됨
  "SPF" → "ⓈⓅⒻ"     # 텍스트 전체 오염
```

- base 토크나이저가 입력을 NFKC 정규화하므로, **정규화 결과가 흔한 글자와 겹치면 그 평문까지 오염**.
- 게다가 base 가 이미 `Ⓜ→M` 로 정규화 → 원형기호는 어차피 복원 불가.
- **해결**: NFKC 안정성 가드 — `unicodedata.normalize("NFKC", c) == c` 인 글자만 추가.
  (원형 재질조건 집합이 자동 제외됨. 꼭 보존해야 하면 라벨에 `(M)` 같은 텍스트로 표기 = 데이터 차원)

---

## 4. 실제 적용한 코드 (Step 2 빌드 셀)

`add_special_tokens(...)` 블록 **바로 뒤, resize 직전**에 삽입:

```python
    # ── 값에 등장하는 도면/GD&T 기호를 '일반' 토큰으로 등록 ───────
    GDNT_SYMBOLS = [
        # 기하공차 기호
        "⊥", "∥", "∠", "∡", "⟂", "○", "◎", "⌒", "⌓", "⏤", "⏥",
        "⌭", "⌯", "⌖", "⌰", "△", "↗",
        # 치수 / 한정자
        "Ø", "⌀", "±", "°", "√", "∞", "×",
        # ※ 재질조건 원형기호 Ⓜ Ⓛ Ⓟ Ⓢ Ⓕ Ⓣ 는 일부러 제외(지뢰 ② — NFKC 충돌)
        # 가공 기호
        "⌴", "⌵", "↧", "⊘", "↓",
    ]
    # 지뢰 ① — 숫자는 컨텍스트 의존 OOV 라 통째로 명시 등록
    ASCII_DIGITS = list("0123456789")

    # 데이터에서 OOV 글자 자동 수집(미드-스트링 '0c0' 컨텍스트로 판정)
    unk_id = processor.tokenizer.unk_token_id
    def _is_oov(ch):
        return unk_id in processor.tokenizer.encode("0" + ch + "0", add_special_tokens=False)
    label_chars = set()
    for _sd in (cfg["data"]["local_train_dir"], cfg["data"]["local_val_dir"]):
        for _f in (Path(_sd) / "labels").glob("*.json"):
            for _v in json.load(open(_f, encoding="utf-8")).values():
                label_chars.update(str(_v))
    auto_oov = [c for c in label_chars
                if not c.isspace() and _is_oov(c)
                and c not in {"�", "️"}]   # 깨진 문자·변이선택자 제외

    # 안전장치: NFKC 로 모양이 바뀌는 글자는 제외(지뢰 ②)
    import unicodedata as _ud
    value_tokens = [c for c in dict.fromkeys(GDNT_SYMBOLS + ASCII_DIGITS + auto_oov)
                    if _ud.normalize("NFKC", c) == c]
    n_val_added = processor.tokenizer.add_tokens(value_tokens)   # ← 일반 토큰

    # 특수(구조) 토큰 또는 값 기호 중 하나라도 추가됐으면 임베딩 확장
    if num_added or n_val_added:
        model.decoder.resize_token_embeddings(len(processor.tokenizer))
```

---

## 5. 검증 결과

```
special token 추가: 12개 (task 1 + 필드 키 5종)
값 기호 토큰 추가 : 41개 (자동탐지 OOV 11종 포함)

값 <unk> : 1/1975 (0.05%)          # 잔존 1건 = '…\nⓂ️'(U+FE0F) 라벨 노이즈
ASCII 무오염: 'TOP'→'TOP'  'SPF'→'SPF'  'M6X1.0'→'M6 X1 .0'
기호/숫자 유지: '⊥ A'  '∡ 6 N'  'R0.2±0.1'   # <unk> 없음
```

(공백은 SentencePiece `▁` 마커 흔적일 뿐, 글자는 모두 보존)

---

## 6. 적용 후 주의사항

1. **재학습 필수** — 새로 추가된 토큰의 임베딩은 랜덤 초기화. 추가만으론 못 읽고, 학습해야 의미를 배운다.
2. **체크포인트에 토크나이저 동봉** — 학습 셀이 `processor` 를 `final/` 에 저장(`save_pretrained`)하므로
   확장된 어휘가 추론에 그대로 로드된다. 추론 때 base 토크나이저를 따로 부르지 말 것.
3. **노이즈 글자는 추가가 아니라 정제** — `�(U+FFFD)`·`️(U+FE0F)`·`Т(키릴 T)`·`━` 는 라벨 입력 오류.
   토큰으로 넣지 말고 `values.jsonl`(또는 CVAT)에서 올바른 기호로 고치는 게 정답.
4. 토크나이저만으로 정확도가 보장되진 않는다 — 0.51% 실패엔 **타입 토큰 붕괴·디코더 반복·과적합**도
   있었다. 함께 개선해야 한다. → [`Element_Donut_평가리포트.md`](Element_Donut_평가리포트.md) 참고.

---

## 7. 재현(자동 OOV 점검) 스니펫

```python
# base 토크나이저가 어떤 값 글자를 못 받는지 빠르게 점검
from transformers import DonutProcessor
import glob, json
tok = DonutProcessor.from_pretrained("naver-clova-ix/donut-base").tokenizer
unk = tok.unk_token_id
vals = [str(v) for f in glob.glob("../data/elements/labels/*.json")
                for v in json.load(open(f, encoding="utf-8")).values()]
broken = [v for v in vals if unk in tok.encode(v, add_special_tokens=False)]
print(f"<unk> 포함 값: {len(broken)}/{len(vals)}")
```
