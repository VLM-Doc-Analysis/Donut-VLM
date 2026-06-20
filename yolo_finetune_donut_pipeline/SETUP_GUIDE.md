# YOLO → Donut 파이프라인 셋업 가이드

`pipeline_drawing.ipynb` 실행 시 발생하는
`FileNotFoundError: detection/view/runs/view/weights/best.pt` 의 원인과 해결 절차 정리.

---

## 원인 진단

`best.pt` 는 **YOLO 학습이 끝나야 생기는 결과물**입니다. 파일 경로 문제가 아니라,
view / element 모델을 **아직 한 번도 학습하지 않은 상태**라서 발생하는 에러입니다.

| 확인 항목 | 상태 |
| :-- | :-- |
| `detection/view/runs/` 폴더 | ❌ 없음 (학습 미실행) |
| `datasets/view/images/{train,val}` | ❌ 비어 있음 |
| `detection/view/cvat_export/` (원본 라벨) | ❌ 없음 |
| `train_view.ipynb`, `view.yaml` | ✅ 있음 |
| `data/raw_pdf/*.pdf` (원본 도면) | ✅ 50개 있음 |
| `data/drawings_pages/*.png` (렌더된 페이지) | ❌ 비어 있음 |

> ⚠️ `datasets/view/images/` 안에 보이는 `train`, `val` 은 **빈 폴더 이름**일 뿐
> 실제 이미지가 아닙니다.

즉, 의존성 사슬이 다음과 같이 끊어져 있습니다:

```
PNG 페이지 없음 → CVAT 라벨 없음 → YOLO 학습 불가 → best.pt 없음 → 파이프라인 에러
```

---

## 전체 로드맵

```
[지금] data/raw_pdf/ 50개
   │ ① PDF → PNG 렌더
   ▼
data/drawings_pages/*.png
   │ ② CVAT 업로드 → view / title_block 라벨링 → export
   ▼
detection/view/cvat_export/
   │ ③ train_view.ipynb (split → train)
   ▼
detection/view/runs/view/weights/best.pt   ← 파이프라인이 찾던 파일
   │ ④ element 모델도 동일하게
   ▼
pipeline_drawing.ipynb 정상 동작 ✅
```

---

## 단계별 해결 절차

### ① PDF → PNG 렌더링
`data/raw_pdf/` 의 PDF를 페이지 PNG로 렌더해 `data/drawings_pages/` 에 저장.
**라벨링·추론과 동일한 해상도(`imgsz=1280`, `fitz` zoom/DPI)** 로 맞춰야 박스 좌표가 어긋나지 않음.

### ② CVAT 라벨링
1. `drawings_pages/*.png` 를 CVAT에 업로드
2. rectangle 로 `view` / `title_block` 영역 라벨링
3. **"Ultralytics YOLO Detection"** 포맷으로 export → 압축 해제 후 아래 구조로 배치:

```
detection/view/cvat_export/
├── images/   # 원본 PNG들
└── labels/   # <stem>.txt들 (각 줄: <class> <cx> <cy> <w> <h>, 0~1 정규화)
```

> `title_block` 이 불필요하면 `view` 한 클래스만 라벨링해도 됨 (`view.yaml` 참고).

### ③ view 모델 학습 — `train_view.ipynb`
**반드시 `detection/view/` 디렉토리에서** 셀을 순서대로 실행:

- **cell 2** — `cvat_export/` 를 90:10 으로 분리해 `datasets/view/` 채움
- **cell 3** — 학습 실행 → `runs/view/weights/best.pt` 생성

```python
model = YOLO("yolo11n.pt")          # 사전학습 시드 (자동 다운로드)
model.train(
    data="view.yaml", epochs=200, imgsz=1280, batch=4,
    project="runs", name="view", exist_ok=True, seed=42,
    # 도면은 방향 고정 → 반전·회전·모자이크 OFF, 밝기만 약하게
    fliplr=0.0, flipud=0.0, degrees=0.0, mosaic=0.0,
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.3, translate=0.05, scale=0.2,
)
```

### ④ element 모델 학습 — `detection/element/train_element.ipynb`
view 와 동일한 절차로 실행 → `detection/element/runs/element/weights/best.pt` 생성.
이게 있어야 파이프라인의 `ELEM_PT` 도 채워짐.

이 두 학습이 끝나면 `pipeline_drawing.ipynb` 의 `YOLO(VIEW_PT)` 가 정상 동작합니다.

---

## ⚠️ 핵심 체크포인트

- 파이프라인 경로 `detection/view/runs/view/weights/best.pt` **자체는 맞음**.
- 단, `train_view.ipynb` 는 **반드시 `detection/view/` 에서 실행**해야 함.
  cell 1 의 `assert (HERE / "view.yaml").exists()` 가 cwd 를 강제하며,
  그래야 `runs/` 가 그 위치에 생성되어 파이프라인 경로와 일치함.
- PDF → PNG 렌더 해상도는 학습/추론 전 단계에서 **항상 동일하게** 유지할 것.
