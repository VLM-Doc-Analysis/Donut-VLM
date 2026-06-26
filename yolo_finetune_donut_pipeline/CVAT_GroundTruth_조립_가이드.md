# CVAT로 Ground Truth(gt_parse) 만들기 가이드

> **무엇을 다루나**: Donut 학습 정답인 **구조화 JSON**(CORD의 `gt_parse` 같은)을 CVAT로 어떻게 만드는가.
> 핵심은 **"CVAT는 평면(박스+속성)만 만들고, 중첩 JSON 구조는 변환 스크립트가 조립한다"**.
>
> 관련 문서: CVAT 조작법 → [`CVAT_LABELING_GUIDE.md`](CVAT_LABELING_GUIDE.md) · 변환 스크립트 → [`detection/cvat_to_donut.py`](detection/cvat_to_donut.py) · 필드 분해 → [`Element_Donut_구조화스키마_수작업annotation_가이드.md`](Element_Donut_구조화스키마_수작업annotation_가이드.md)

> ⚠️ **CORD-v2 자체는 CVAT로 안 만든다** — HF 데이터셋에 `gt_parse`가 이미 들어있다. CVAT는 **내 도메인(도면/요소)의 정답을 직접 만들 때** 쓴다. 이 문서는 "CORD 같은 구조화 JSON을 CVAT로 만들려면?"을 일반화해 설명한다.

---

## 0. 큰 그림 — CVAT는 "재료", 구조는 "스크립트가 조립"

```
CVAT (박스마다: class label + text 속성)
   │ export (CVAT for images 1.1 XML / YOLO)
   ▼
변환 스크립트 (cvat_to_donut.py 류)   ← 평면 라벨 → 중첩 JSON 조립
   ▼
gt_parse: { "menu":{…}, "total":{…} }   (= 모델 학습 정답)
```

CVAT는 **nested JSON을 직접 못 만든다.** 박스에 **"라벨(=키) + 속성(=값)"** 만 달고,
**계층·리스트 구조는 변환 코드가** 만든다.

---

## 1. 매핑 — CVAT 개념 ↔ JSON

| CVAT | → | JSON |
|---|---|---|
| **label (class)** | → | JSON **키** (`menu`, `total`, `Dimension` …) |
| **text attribute** | → | **값 / 필드** (`nm`, `price`, `value`, `nominalValue` …) |
| **박스 위치** | → | 크롭·정렬(요소) 또는 그룹 구분 |
| **같은 라벨 박스 여러 개** | → | **리스트** (`menu: [ {…}, {…} ]`) |

---

## 2. CORD 예시를 CVAT로 만든다면 (4단계)

목표:
```json
{ "menu": {"nm":"치킨","price":"12000"}, "total": {"total_price":"12000"} }
```

**① 라벨 스키마 정의** (CVAT Projects → 라벨 편집)

| 라벨(class) | text 속성 |
|---|---|
| `menu` | `nm`, `price` |
| `total` | `total_price` |

**② 박스 그리고 속성 입력**
- 메뉴 줄(치킨 12000) 주위에 `menu` 박스 → `nm="치킨"`, `price="12000"`
- 합계 영역에 `total` 박스 → `total_price="12000"`

**③ export** → "CVAT for images 1.1" XML

**④ 변환 스크립트가 조립**
```python
# 박스마다 {label: {attr: value, …}} 추출 → 키별로 묶기
{ "menu": {"nm":"치킨","price":"12000"}, "total": {"total_price":"12000"} }
```
→ `menu` 박스가 여러 개면 `"menu": [ {…}, {…} ]` 리스트로.

---

## 3. 이 프로젝트의 실제 워크플로 (요소 파이프라인)

프로젝트는 이미 이 패턴을 쓴다 — [`CVAT_LABELING_GUIDE.md`](CVAT_LABELING_GUIDE.md) + [`cvat_to_donut.py`](detection/cvat_to_donut.py):

| 단계 | 내용 |
|---|---|
| 라벨 | element 박스에 **class**(`Dimension`/`GD&T_FCF`…) + **`value` 텍스트 속성**(`"Ø65"`) |
| export | CVAT XML |
| 변환 | `cvat_to_donut.py` → `{"<class>": "<value>"}` per 크롭 (예: `{"Dimension":"Ø65"}`) |

> 요소는 **한 박스 = 한 `{class:value}`** 라 구조가 단순. CORD처럼 nested면 변환 스크립트에 조립 로직을 더한다.

---

## 4. 값을 "필드로 분해"해 라벨하려면 (구조화 gt_parse)

값을 통짜가 아니라 의미 필드로 쪼개면 **부분점수·다운스트림에 유리**(통짜 `{"value":…}` 지양).
CVAT 라벨에 **여러 text 속성**을 정의한다:

| 라벨 | 속성 (구조 필드) |
|---|---|
| `Dimension` | `quantity`, `nominalValue`, `upperLimit`, `lowerLimit` |
| `GD&T_FCF` | `geometricCharacteristic`, `tolerance`, `datumReference` |

→ 박스마다 필드별로 입력 → 변환 스크립트가
`{"Dimension": {"nominalValue":"Ø65","upperLimit":"+0.1", …}}` 로 조립.
(자세히: [`Element_Donut_구조화스키마_수작업annotation_가이드.md`](Element_Donut_구조화스키마_수작업annotation_가이드.md))

---

## 5. 계층·리스트는 어떻게? — CVAT의 한계와 해법

CVAT는 평면이라 **중첩은 변환 코드가 만든다**:

| 만들 구조 | 해법 |
|---|---|
| **리스트** (`menu:[…]`) | 같은 라벨 박스 여러 개 → 스크립트가 모음 |
| **그룹/최상위 키** (`menu` vs `total`) | 라벨 종류로 구분 |
| **부모-자식 중첩** (메뉴-옵션) | 박스 **포함관계**나 CVAT **`group_id`** 로 묶어 스크립트가 중첩 구성 |

> 즉 "**어떤 박스가 어떤 키/그룹에 속하는가**"는 CVAT의 라벨·group_id로 표시하고,
> 실제 **중첩 dict / list 조립**은 변환 스크립트 로직이 담당한다.

---

## 6. 체크리스트

- [ ] JSON 스키마 먼저 확정 (키 = CVAT 라벨, 값/필드 = text 속성)
- [ ] CVAT 라벨에 속성 정의 (단일 `value` 또는 분해 필드)
- [ ] 박스 그리고 속성 입력 (라벨 품질 = 모델 품질)
- [ ] export (CVAT for images 1.1 XML)
- [ ] 변환 스크립트로 `{키: 값/필드}` 조립 (+ 리스트/중첩 로직)
- [ ] 결과 JSON을 `labels/<stem>.json` 으로 저장 → 학습 입력

---

## 한 줄 요약

> **CVAT에선 "박스마다 라벨(=키) + 텍스트 속성(=값/필드)"만 단다.
> `gt_parse` 같은 중첩 구조는 export 후 변환 스크립트(`cvat_to_donut.py`)가 키별로 묶고 리스트·중첩으로 조립한다.**
> (CORD 자체는 데이터셋 제공이라 CVAT 불필요 — CVAT는 내 도면/요소 라벨용)
