# CVAT Element 라벨링 가이드 (`value` 속성 → Donut 학습 데이터)

Stage 2(Element 검출/인식) 단계에서 **CVAT로 element 박스를 그리고, 각 박스에 `value` 텍스트 속성을
입력**하면 — 한 번의 라벨링으로 **YOLO-OBB 검출 라벨**과 **Donut 값 인식 라벨**을 동시에 얻습니다.
이 문서는 그 전체 절차(속성 정의 → 입력 → export → 변환)를 정리합니다.

> 핵심 한 줄: CVAT 박스의 **`label`(=class)** + **`value`(=읽은 값)** → `{"<class>": "<value>"}` JSON.
> 예) `Dimension` 박스 + `value="Ø9 ±0.2"` → `{"Dimension": "Ø9 ±0.2"}`

---

## 0. Element 클래스 (6개)

`detection/element/element.yaml` 기준 — CVAT 라벨 이름은 이것과 **정확히** 일치해야 합니다.

| index | class 이름 | 의미 |
|---|---|---|
| 0 | `Dimension` | 치수 |
| 1 | `GD&T_FCF` | 기하공차 FCF (Feature Control Frame) |
| 2 | `Datum` | 데이텀 |
| 3 | `Surface_Roughness` | 표면 거칠기 |
| 4 | `Section` | 단면 표시 |
| 5 | `Hole_Callout` | 홀 콜아웃 |

---

## 1. 라벨에 `value` Text 속성 정의

> ⚠️ CVAT는 라벨 스키마에 속성을 **미리 선언해야** 박스에 입력칸이 생깁니다.
> 정의 안 하면 "텍스트 속성을 못 찾는" 상태가 됩니다.

### 어디서?
**Projects → 해당 프로젝트 → 라벨 편집** (프로젝트 레벨에서 고치면 모든 task/job에 적용됨).

### 방법 A: Constructor (권장 — 기존 박스 보존)

1. 라벨 에디터에서 상단 **`Constructor`** 탭 선택.
2. 각 라벨(예: `Dimension`)의 **연필(Edit)** 아이콘 클릭.
3. **`Add an attribute`** 클릭 → 아래대로 설정:

   | 필드 | 값 | 이유 |
   |---|---|---|
   | **Name** | `value` | 변환 스크립트가 `VALUE_ATTR = "value"`로 **이 이름만** 읽음 (대소문자·철자 정확히) |
   | **Type** | **`Text`** | "Ø9 ±0.2" 같은 자유 입력. Select/Radio면 미리 정한 보기만 선택됨 |
   | **Mutable** | ✅ 체크 | 객체마다 값이 다름 |
   | **Default value** | **(비움)** | 기본값 넣으면 새 박스마다 자동 입력돼 방해 + 빈 값으로 "미입력 박스" 구분이 안 됨 |

4. **6개 클래스 전부**에 반복 → `Submit`.

### 방법 B: Raw (일괄 — `id` 보존 주의)

`Raw` 탭에서 각 라벨에 아래 속성 블록을 추가. **기존 라벨의 `id`는 절대 삭제 금지**(삭제 시 그 라벨의 박스 소실). 새 속성의 `id`는 `-1`로 두면 CVAT가 발급:

```json
{ "name": "value", "id": -1, "mutable": true, "input_type": "text", "default_value": "", "values": [] }
```

### 정의 확인
저장 후 기존 박스를 클릭 → 우측 **Objects** 패널에서 펼쳤을 때 빈 `value` 입력칸이 보이면 성공. ✅

---

## 2. 박스 그리기 + 값 입력

1. **회전 박스(OBB)**: rectangle을 그린 뒤 위쪽 **회전 핸들**을 돌려 도면 요소에 정렬.
   (변환 스크립트가 `rotation` 값을 읽어 `rectify_obb`로 수평 정렬함)
2. **class 지정**: 6개 중 해당 element 타입 선택.
3. **value 입력**: 우측 **Objects** 패널에서 객체를 펼쳐 `value` 칸에 읽은 값 입력 (예: `Ø9 ±0.2`, `Ra 1.6`, `20`).
4. **빠른 입력 팁**: 우상단 워크스페이스를 **Standard → Attribute annotation** 으로 바꾸면
   박스를 하나씩 넘기며 `value`만 연속으로 입력 가능.

---

## 3. Export (⭐ 포맷이 핵심)

값 텍스트는 **`CVAT for images 1.1` (XML)** 포맷에만 보존됩니다. YOLO export는 텍스트를 버립니다.

### 절차
어노테이션 화면 좌상단 **Menu (☰) → `Export job dataset`** (또는 task ⋮ → `Export task dataset`)

| 설정 | 값 |
|---|---|
| **Export format** | **`CVAT for images 1.1`** |
| **Save images** | 원본이 `../data/view_crops`에 있으면 해제 가능 |

→ `annotations.xml`이 담긴 zip 다운로드 (바로 안 받아지면 상단 **Requests** 탭에서 확인).

> 검출(YOLO) 학습용은 **별도로** `Ultralytics YOLO Oriented Bounding Boxes` 포맷으로도 export.
> (두 export는 용도가 다르며 서로 대체 불가)

| Export 포맷 | 용도 | 산출물 |
|---|---|---|
| **CVAT for images 1.1** | Donut **값** 라벨 | `annotations.xml` (`value` 포함) |
| **Ultralytics YOLO OBB** | YOLO **검출** 학습 | `.txt` (4점 좌표, 값 없음) |

---

## 4. `cvat_to_donut.py` 적용

다운받은 `annotations.xml`을 `detection/element/cvat_export/`에 두고 실행:

```bash
cd yolo_finetune_donut_pipeline
python detection/cvat_to_donut.py \
    --xml detection/element/cvat_export/annotations.xml \
    --images ../data/view_crops \
    --out ../data/elements          # (기본값)
```

### 동작
- XML의 각 `<box>` → `label`(class)과 `value` 속성을 읽음.
- `value`가 **비면 건너뜀**(빈 라벨은 Donut 인식을 망가뜨림) → skip 개수 출력.
- 회전 박스를 `rectify_obb`로 **수평 정렬해 크롭** → 추론과 동일한 형태 보장.

---

## 5. 최종 산출물

`data/elements/` 아래에 **정렬된 크롭 PNG + `{class: value}` JSON 쌍**:

```
data/elements/
├── images/
│   └── <stem>__<i>__<class>.png    # 예: FCF_001__004__Dimension.png
└── labels/
    └── <stem>__<i>__<class>.json   # 예: {"Dimension": "Ø9 ±0.2"}
```

- 이미지·라벨은 같은 stem으로 **1:1 매칭**.
- 이후 **`donut_training_elements_flat.ipynb`** 의 split 셀이 `train/val`로 나눠 Donut 파인튜닝 →
  `checkpoints_elements/final` 모델 생성.

---

## 6. 전체 흐름

```
CVAT box (label = class, attribute "value" = 읽은 값)
   │  Export: "CVAT for images 1.1"  → annotations.xml
   ▼
cvat_to_donut.py
   │  rectify_obb 크롭 + json.dump({label: value})
   ▼
data/elements/{images,labels}/  ({"<class>": "<value>"})
   │  json2token:  {"Dimension":"Ø9 ±0.2"} → <s_Dimension>Ø9 ±0.2</s_Dimension>
   ▼
donut_training_elements_flat.ipynb  → checkpoints_elements/final
   │  (추론 시 token2json 으로 다시 {"<class>": "<value>"} 복원)
```

---

## 체크리스트

- [ ] 6개 클래스 **모두**에 `value` 속성 정의 (Name=`value`, Type=**Text**, Default 비움)
- [ ] 회전 박스(OBB)로 그리고 class 지정
- [ ] 각 박스 `value`에 읽은 값 입력 (Attribute annotation 모드 활용)
- [ ] **`CVAT for images 1.1`** 포맷으로 export → `annotations.xml`
- [ ] (검출용) `Ultralytics YOLO OBB` 포맷으로도 export
- [ ] `cvat_to_donut.py` 실행 → skip 개수 확인 (값 미입력 박스)
- [ ] `data/elements/{images,labels}/` 쌍 생성 확인 → `donut_training_elements_flat.ipynb` 학습

---

## 자주 막히는 부분

| 증상 | 원인 / 해결 |
|---|---|
| 박스에 텍스트 입력칸이 없음 | 라벨에 `value` 속성 미정의 → §1 |
| 임의 텍스트가 안 들어감 (보기만 선택됨) | 속성 Type이 Select/Radio → **Text**로 변경 |
| `cvat_to_donut.py`가 값을 못 읽음 | 속성 이름이 `value`가 아님 / YOLO 포맷으로 export함 → CVAT XML로 export |
| 변환 결과가 0개 | 모든 `value`가 비어 있음(전부 skip) → CVAT에서 값 입력 후 재export |
