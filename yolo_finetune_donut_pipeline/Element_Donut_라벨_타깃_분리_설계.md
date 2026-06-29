# Element Donut — 라벨 ↔ 학습 타깃 분리 설계 (스키마 정책)

> 핵심 원칙: **정답(라벨)은 가장 풍부한 형태로 1회 저장하고, 모델 학습 타깃은 거기서 코드로 파생(데이터 규모에 맞춰 토글)한다.**
> 작성: 2026-06-29 · 관련: [`구조화스키마_수작업annotation_가이드`](Element_Donut_구조화스키마_수작업annotation_가이드.md) · [`성능미달_원인_및_해결방안`](Element_Donut_성능미달_원인_및_해결방안.md) · [`논문성능_달성가능성_분석`](Element_Donut_논문성능_달성가능성_분석.md)

---

## 0. 한 줄 원칙

**라벨 = raw + 구조화 필드(Unicode 표준)  ·  학습 타깃 = `TARGET_MODE` 토글(flat ↔ structured).**

라벨링은 비싸고 1회성이므로 **최대 정보**로 저장하고, 학습 타깃은 데이터 규모에 따라 코드로 바꾼다(재라벨 0).

---

## 1. 왜 이렇게 하나

> **변환은 "풍부→단순"만 무손실, "단순→풍부"는 추측이 필요해 손실난다.**

- **구조화 필드로 라벨** → flat(통짜 값)은 언제든 자동 생성(무손실).
- **flat로 라벨** → 구조(필드)는 `parse_to_schema` 같은 **정규식 추측**으로만 복원 → 오류·손실 (우리가 GD&T 에서 겪은 그 문제).
- 게다가 **최적 타깃 복잡도는 데이터에 종속**(소량=flat 유리, 대량=structured 유리 — 측정: flat 0.444 > structured 0.349). 재라벨 없이 타깃만 바꾸려면 → 라벨은 풍부하게.

---

## 2. 데이터 흐름

```
[사람 1회 라벨링 — 디스크]            ← 가장 풍부한 형태
  {raw, fields(Unicode 표준)}
        │
        ├─(TARGET_MODE=flat)       → json2token({value: raw})   → 모델 학습(OCR 집중)
        └─(TARGET_MODE=structured) → json2token(fields)         → 모델 학습(구조 직접생성)
                                          │
[추론] 모델출력 → token2json → 구조 dict → Unicode 디코드(글리프)
[평가] 라벨의 fields 와 field-F1 비교
```

---

## 3. 라벨 파일 형태 (예시)

**GD&T 크롭 "⊥ Ø0.12(M) A B"**
```json
{
  "class": "GD&T_FCF",
  "raw": "⊥ Ø0.12(M) A B",
  "fields": {
    "geometricCharacteristic": "U+27C2",
    "tolerance": "U+2300 0.12 (M)",
    "datumReference": ["A", "B"]
  }
}
```
**Dimension 크롭 "8X Ø6.5±0.1"**
```json
{
  "class": "Dimension",
  "raw": "8X Ø6.5±0.1",
  "fields": {"quantity":"8","nominalValue":"U+2300 6.5","upperLimit":"+0.1","lowerLimit":"-0.1"}
}
```
- `raw` = 읽은 그대로(빠른 입력) → **flat 타깃**용.
- `fields` = 의미 단위 분해 + **기호는 표준 Unicode 코드**(⊥/⟂/⟵ 혼용 방지 = "reliable supervision") → **structured 타깃**용.

---

## 4. 학습 타깃 두 형태 (같은 라벨에서 파생)

| 모드 | 타깃 토큰열 | 모델 역할 |
|:--|:--|:--|
| **flat** | `<s_value>⊥ Ø0.12(M) A B</s_value>` | OCR(이미지→값)만, 구조 분해는 사후 |
| **structured** | `<s_geometricCharacteristic>U+27C2</s_geometricCharacteristic><s_tolerance>U+2300 0.12 (M)</s_tolerance><s_datumReference>A<sep/>B</s_datumReference>` | 구조를 직접 생성 |

---

## 5. 왜 raw + fields 둘 다 저장?

- flat 타깃은 `raw` 에서, structured 타깃은 `fields` 에서 **각각 무손실로** 생성 → **추측·정규식 불필요.**
- 현재 우리 라벨은 `{"GD&T_FCF":"⊥0.01A"}`(raw 만)이라, 구조화하려면 정규식이 필드 경계를 *추측* → 깨짐. **fields 를 사람이 직접 달면 그 손실이 사라진다.**

---

## 6. 토글 메커니즘 (노트북 골격 존재)

- `DonutDataset.__getitem__` 가 `TARGET_MODE` 로 분기:
  - `flat` → `json2token({"value": label["raw"]})`
  - `structured` → `json2token(label["fields"])`
- `USE_UNICODE_SYMBOLS` 토글(구현됨)로 Unicode 인코딩 on/off.
- **라벨 파일은 그대로, `TARGET_MODE` 한 줄만 바꿔** 소량→flat / 대량→structured 전환.

---

## 7. 규모별 운영 정책

| 단계 | 라벨(디스크) | 학습 타깃 | 근거 |
|:--|:--|:--|:--|
| 지금 ~수천 | raw + fields | **flat** | 소량엔 단순 타깃 우위(0.444 > 0.349 측정) |
| ~1만+ clean | raw + fields | **structured** | 대량엔 정교 스키마가 reliable supervision(논문 0.93–0.96) |

**라벨은 처음부터 풍부하게 → 전환 비용 0.** 데이터가 커지면 재라벨 없이 토글만.

---

## 8. 비용과 완화

- **비용**: 라벨링 시 필드 분해까지 → 크롭당 작업량↑(flat 대비).
- **완화**: 현 Donut 으로 `raw` pre-fill → 사람이 `fields` 분해(자주 쓰는 패턴은 규칙 보조), GD&T 만 수작업.
- **이득**: ① 무손실 토글, ② 정규식 추측 제거, ③ Unicode 표준화로 깨끗한 감독, ④ field-F1 평가 직접 가능.

---

## 9. 한 줄 요약

> **"라벨은 정교하게(raw + 구조화 필드 + Unicode), 학습 타깃 복잡도는 데이터에 맞춰 토글."**
> 비싼 라벨링을 1회만 하고도 데이터가 커질 때 스키마를 공짜로 전환 — 정규식 추측 손실 제거 + 표준 감독.
