# YOLO 검출 모델 평가 리포트

> 도면 → (뷰 검출 → 요소 검출) → Donut 인식 2단 파이프라인 중,
> **검출(YOLO) 단계 두 모델**의 학습·평가 결과 정리.

대상 모델
- **View 모델** — `detect` 방식, 뷰/표제란/노트 검출 (`detection/view/runs/detect/runs/view`)
- **Element 모델** — `OBB`(회전 박스) 방식, 치수·공차 등 요소 검출 (`detection/element/runs/obb/runs/element`)

---

## 1. 핵심 결론 (요약)

| 지표 | **View (detect)** | **Element (OBB)** |
|---|---|---|
| mAP@0.5 | **0.994** | 0.918 |
| mAP@0.5:0.95 | **0.916** | 0.764 |
| Precision | 0.989 | 0.923 |
| Recall | 0.981 | 0.907 |
| 학습 epoch | 157 | 42 |

- **View 모델은 사실상 완성형.** 큰 영역을 거의 완벽히, 박스도 정밀하게 검출.
- **파이프라인의 약한 고리는 Element 모델.** 탐지(mAP@0.5 0.92)는 좋으나 위치 정밀도(mAP@0.5:0.95 0.76)가 낮음.
- 따라서 **개선 노력은 Element OBB 쪽에 집중**하는 것이 효율적.

---

## 2. 배경 개념

### 2.1 confusion matrix의 `background` 클래스

검출 모델의 혼동행렬에는 데이터 라벨에 없는 **`background`(객체 없음)** 행/열이 자동 추가된다.
분류와 달리 검출은 "객체가 있다/없다"까지 판단해야 하므로, 분류엔 없는 두 종류의 오류를 표현하기 위함이다.

| 위치 | 의미 |
|---|---|
| **background 열** (True=background) | **오탐 (False Positive)** — 없는 객체를 검출 |
| **background 행** (Pred=background) | **누락 (False Negative)** — 있는 객체를 놓침 |

> 참고: 일반 **분류(classification)** 혼동행렬에는 background가 없다(sklearn 등).
> **객체 검출(YOLO·Detectron2 등)** 에서는 오탐·누락 진단을 위해 background를 포함하는 것이 표준이다.

### 2.2 두 가지 mAP

| 지표 | IoU 기준 | 성격 |
|---|---|---|
| **mAP@0.5** (mAP50) | 0.5 하나 | **느슨** — "객체를 찾았나?" (탐지 여부) |
| **mAP@0.5:0.95** (mAP50-95) | 0.5~0.95, 0.05 간격 10개 평균 | **엄격** — 박스 위치·크기 정밀도까지. COCO 표준 |

- mAP50-95는 항상 mAP50보다 낮다.
- **두 값의 격차가 크면** = "객체는 잘 찾지만 박스가 헐겁다(위치 정밀도 낮음)".

---

## 3. View 모델 (detect)

![View 학습 곡선](view/runs/detect/runs/view/training_curves.png)

### 최종 수치 (epoch 157)

| 항목 | 값 | best |
|---|---|---|
| Precision | 0.989 | — |
| Recall | 0.981 | — |
| mAP@0.5 | 0.994 | 0.995 (ep32) |
| mAP@0.5:0.95 | 0.916 | 0.924 (ep107) |
| Train loss | box 0.17 · cls 0.21 · dfl 0.80 | — |
| Val loss | box 0.42 · cls 0.34 · dfl 0.99 | box 최저 0.414 (ep155) |

### 해석
- **약 20 epoch 만에 P·R 0.98+ 도달**, 이후 끝까지 안정적으로 유지.
- mAP50은 ep32에 이미 최고치 → 이후는 위치 정밀도(mAP50-95)만 천천히 다듬은 셈.
- val box loss가 ep155에 최저 → **과적합 징후 없음**.
- 큰 영역(뷰/표제란/노트) 검출이라 OBB 없이 detect로 충분, 성능 매우 우수.

---

## 4. Element 모델 (OBB)

![Element 학습 곡선](element/runs/obb/runs/element/training_curves.png)

### 최종 수치 (epoch 42)

| 항목 | 값 | best |
|---|---|---|
| Precision | 0.923 | 0.938 (ep41) |
| Recall | 0.907 | 0.913 |
| mAP@0.5 | 0.918 | 0.924 (ep19) |
| mAP@0.5:0.95 | 0.764 | 0.773 (ep22) |
| Train loss | box 0.56 · cls 0.45 · dfl 1.18 · angle 0.008 | — |
| Val loss | box 0.70 · cls 0.57 · dfl 1.28 · angle 0.013 | — |

### 해석
- **epoch 8~9에서 급상승 후 약 20 epoch에서 사실상 수렴.** best가 19~22 epoch에 나옴.
- **angle loss가 거의 0(0.008)** → 회전각 추정은 매우 잘 학습됨.
- 두 mAP의 격차(~0.15)가 View보다 큼 → **작은 회전 요소의 박스 정밀도가 약점.**
- train(box 0.56)↔val(box 0.70) 갭 존재하나, 검증셋이 작아 생기는 경미한 일반화 갭이며 발산은 아님.

---

## 5. 종합 진단 및 개선 방향

| 구분 | 진단 |
|---|---|
| 학습 건강성 | 두 모델 모두 loss 정상 수렴, 과적합 징후 미미, ~20 epoch 수렴 |
| 강점 | View 검출 거의 완벽 / Element 탐지율(mAP@0.5)도 양호 |
| 약점 | **Element의 위치 정밀도(mAP@0.5:0.95 0.76)** — 파이프라인 전체 병목 |

### Element 모델 개선 후보
1. **데이터 증강** — 작은 객체 스케일/모자이크 조정
2. **입력 해상도 ↑** — 작은 요소 디테일 확보
3. **라벨 박스 정밀도 점검** — OBB 라벨의 fit 품질이 mAP50-95 상한을 결정
4. 학습이 ~20 epoch에 수렴하므로, 무작정 epoch를 늘리기보다 위 데이터·해상도 개선이 효과적

---

## 6. 산출물 경로

| 파일 | 설명 |
|---|---|
| `view/runs/detect/runs/view/training_curves.png` | View 모델 loss·P/R·mAP 종합 |
| `view/runs/detect/runs/view/mAP_curve.png` | View mAP 곡선 (선택) |
| `element/runs/obb/runs/element/training_curves.png` | Element 모델 loss·P/R·mAP 종합 |
| `element/runs/obb/runs/element/mAP_curve.png` | Element mAP 곡선 |
| `*/confusion_matrix.png` | 각 모델 혼동행렬 (background 포함) |
