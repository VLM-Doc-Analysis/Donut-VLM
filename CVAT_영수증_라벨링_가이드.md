# CVAT 영수증 라벨링 가이드

`donut_training.ipynb`(영수증 도메인, CORD 스타일)의 학습 데이터를 **CVAT**로 라벨링해
`data/raw/*.json` 정답을 만드는 방법.

> 도면/요소 파이프라인의 [`yolo_finetune_donut_pipeline/CVAT_LABELING_GUIDE.md`](yolo_finetune_donut_pipeline/CVAT_LABELING_GUIDE.md)
> 와 흐름이 거의 같다(박스 + text attribute → 변환 스크립트). 영수증은 **중첩 리스트(items)**가
> 있다는 점만 다르다.

---

## 0. 큰 그림

Donut 은 **OCR-free** 라 학습에 박스가 필요 없고 **JSON 정답만** 필요하다. 하지만 CVAT 의
기본 단위는 박스/태그다. 그래서 박스에 **값을 text attribute 로 입력**해 라벨링하고,
변환 스크립트가 이를 Donut JSON 으로 바꾼다.

```
CVAT(박스 + 텍스트 속성) → "CVAT for images 1.1" XML export
        → python cvat_receipts_to_donut.py → data/raw/*.json
        → 노트북 "[선택] 로컬 데이터셋 준비" 셀 → data/processed/{train,val}/
```

박스 위치는 정확할 필요 없다(값을 어디서 읽었는지 표시용). **값(attribute)만 정확하면 된다.**

---

## 1. 라벨(스키마) 정의

CVAT **Projects → 프로젝트 생성 → 라벨 편집 → Raw** 탭에 아래 JSON 을 붙여넣는다.

```json
[
  { "name": "store_name", "type": "rectangle",
    "attributes": [ {"name": "text", "input_type": "text", "mutable": true, "values": [""]} ] },
  { "name": "total", "type": "rectangle",
    "attributes": [ {"name": "text", "input_type": "text", "mutable": true, "values": [""]} ] },
  { "name": "item", "type": "rectangle",
    "attributes": [
      {"name": "name",  "input_type": "text", "mutable": true, "values": [""]},
      {"name": "price", "input_type": "text", "mutable": true, "values": [""]}
    ] }
]
```

| 라벨 | 개수 | 속성 | 매핑되는 JSON |
|---|---|---|---|
| `store_name` | 영수증당 1개 | `text` | `"store_name": "..."` |
| `total` | 영수증당 1개 | `text` | `"total": "..."` |
| `item` | 메뉴 줄당 1개 | `name`, `price` | `"items": [{"name":..,"price":..}]` |

**`item` 은 한 줄을 감싸는 박스 1개에 `name`·`price` 두 값을 함께 입력** → 별도 그룹핑 없이
중첩 객체로 바로 매핑된다. 품명과 가격이 가로로 떨어져 있어도 둘을 감싸는 박스 하나면 된다.

> **필드를 추가하려면**: 라벨을 더 만들고(예: `date`, `card_no`) 변환 스크립트
> `cvat_receipts_to_donut.py` 의 `FLAT_LABELS` 에 이름을 추가하면 된다.

---

## 2. 이미지 업로드 & 어노테이션

1. **Tasks → 새 Task** 생성, 위 프로젝트 연결, `영수증_001.jpg …` 이미지 업로드.
2. 각 이미지에서:
   - 상호 영역 → `store_name` 박스, `text = "스타벅스"`
   - 합계 영역 → `total` 박스, `text = "12500"`
   - 메뉴 줄마다 → `item` 박스, `name = "아메리카노"`, `price = "4500"` …
3. 한국어 입력은 정상 동작한다.
4. item 박스는 **위에서 아래 순서**로 그리면 좋지만, 변환 스크립트가 **y좌표로 자동 정렬**하므로
   순서는 신경 쓰지 않아도 읽기 순서가 보존된다.

> **라벨 품질 = 모델 품질.** 값 오타·누락이 그대로 학습된다. 이미지와 대조 검수할 것.

---

## 3. Export

Task → **Export annotations → "CVAT for images 1.1"** → `annotations.xml` 다운로드.

---

## 4. 변환 (XML → `data/raw/`)

프로젝트 루트 `donut_vml/` 에서:

```bash
python cvat_receipts_to_donut.py \
    --xml annotations.xml \
    --images <CVAT에 올린 원본 이미지 폴더> \
    --out data/raw
```

결과: `data/raw/영수증_001.json`(+ 이미지 복사). 예시 구조 그대로 생성된다.

```json
{
  "store_name": "스타벅스",
  "total": "12500",
  "items": [
    {"name": "아메리카노", "price": "4500"},
    {"name": "카페라떼", "price": "8000"}
  ]
}
```

---

## 5. 노트북에서 학습 데이터로

`donut_training.ipynb` 의 **"[선택] 로컬 데이터셋 준비"** 셀을 실행하면:

- `data/raw/` → `data/processed/{train,val}/` 분할 (`VAL_RATIO=0.1`, seed 42)
- `CFG["data"]["dataset_name"] = None` 으로 로컬 모드 자동 전환

그 뒤 **`CFG["data"]["task_prompt"]` 를 내 태스크 토큰으로 직접 변경**한다(예: `<s_receipt>`).

> ⚠️ **task token ≠ 필드명** (CLAUDE.md 규칙). 최상위 필드가 `store_name`/`total`/`items` 이므로
> task token 은 그와 겹치지 않는 `<s_receipt>` 등으로 둔다. 필드 토큰(`<s_store_name>` 등)은
> `build_model_and_processor` 가 **라벨에서 자동 수집**해 등록하므로 따로 손댈 필요 없다.

---

## 부록: 필드를 늘릴 때 변환 스크립트 수정 지점

`cvat_receipts_to_donut.py`:

```python
FLAT_LABELS = ("store_name", "total")   # ← 평면 필드 추가 시 여기에 라벨명 추가
ITEM_LABEL  = "item"                     # 중첩 리스트 라벨
TEXT_ATTR   = "text"                     # 평면 필드 값 attribute 이름
```

리스트 항목의 하위 키(`name`/`price`)를 바꾸려면 `convert()` 의 `items.append(...)` 부분을
함께 수정한다.
