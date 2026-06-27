# Donut 문서 파싱 파인튜닝 (RTX 5090 로컬 적응판)

원본 Colab 노트북을 로컬 RTX 5090 / conda `donut_vml` 환경에 맞게 수정한 모음입니다.

- **태스크**: Donut 으로 문서(영수증) 구조 파싱 (SROIE). `<parsing>` task token, `{"menu":[...], "total":{...}}` 형태 JSON 출력. `max_length=128`, 이미지 `[720, 960]`.
- **의존 라이브러리**: [`hftuner/clovaai-donut`](https://github.com/hftuner/clovaai-donut) — 커스텀 `DonutModel`/`DataProcessor`. 첫 코드 셀이 이 폴더에 클론 후 `sys.path` 에 추가.

## 파일

| 파일 | 설명 |
|---|---|
| `source_colab.ipynb` | 원본 Colab 노트북 (수정 없이 보존) |
| `donut_parsing_5090.ipynb` | **RTX 5090 적응판 — 이걸 실행** |
| `hftuner/` | 첫 셀 실행 시 자동 생성되는 라이브러리 클론 (gitignore 처리) |

## 원본 대비 변경점

- `fp16=True` → **`bf16=True`** (Donut 은 fp16 수치 불안정; 5090 bf16 지원)
- `/content/...` Colab 경로 → **로컬 경로**, `hftuner` 를 폴더에 클론 후 `sys.path` 추가
- `push_to_hub=True` / 모델카드 푸시 **제거** → **로컬 저장만**
- `report_to="tensorboard"` → `"none"`
- 추론 dtype fp16 → bf16, **`pixel_values` 를 `model.dtype` 으로 캐스팅** (bf16 모델 ↔ float 입력
  mismatch `RuntimeError: Input type (float) and bias type (BFloat16)...` 선제 차단)
- 맨 앞에 위젯 렌더 픽스 셀(`TQDM_NOTEBOOK=false`) 추가

## 실행

1. `conda activate donut_vml`
2. `donut_parsing_5090.ipynb` 를 위에서 아래로 실행 (VS Code Jupyter).
   - 첫 실행 시 `hftuner` 클론 + SROIE 데이터셋 다운로드.
3. 학습(기본 `max_steps=100` 데모) 결과는 `donut-base-finetuned-sroie/` 에 로컬 저장.
   전체 학습하려면 학습 인자 셀의 `max_steps=100` 줄을 주석 처리.

## 주의

- `hftuner` 는 transformers 4.56.x 가정이나 현재 env(5.12.1)에서 import/모델로드 검증됨
  (분류 노트북과 동일 라이브러리). 학습/generate 중 5.x API 차로 막히면 `transformers==4.56.1`
  로 맞추세요(별도 env 권장).
- GPU 여유가 적으면 batch 1 유지, 여유 있으면 `per_device_train_batch_size` 를 2~4 로.
