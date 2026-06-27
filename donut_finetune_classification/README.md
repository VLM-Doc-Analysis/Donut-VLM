# Donut 문서 분류 파인튜닝 (RTX 5090 로컬 적응판)

원본 Colab 노트북을 로컬 RTX 5090 / conda `donut_vml` 환경에 맞게 수정한 모음입니다.

- **태스크**: Donut 으로 문서 이미지 분류 (RVL-CDIP 16 클래스). `<classification>` task token, `{"class": "..."}` 출력.
- **의존 라이브러리**: [`hftuner/clovaai-donut`](https://github.com/hftuner/clovaai-donut) — 커스텀 `DonutModel`/`DataProcessor`. 첫 코드 셀이 이 폴더에 클론하고 `sys.path` 에 추가합니다.

## 파일

| 파일 | 설명 |
|---|---|
| `source_colab.ipynb` | 원본 Colab 노트북 (수정 없이 보존) |
| `donut_classification_5090.ipynb` | **RTX 5090 적응판 — 이걸 실행** |
| `hftuner/` | 첫 셀 실행 시 자동 생성되는 라이브러리 클론 (gitignore 권장) |

## 원본 대비 변경점

- `fp16=True` → **`bf16=True`** — Donut 은 fp16 에서 수치 불안정. 5090 은 bf16 지원.
- `/content/...` Colab 경로 → **로컬 경로**, `!pip` 핀 설치 → 주석(이미 설치됨)
- `huggingface_hub.login()` / `push_to_hub=True` / 모델카드 푸시 **제거** → **로컬 저장만**
- `report_to="tensorboard"` → `"none"`, `per_device_eval_batch_size` 8 → 4 (960×1280 + generate 메모리)
- 추론 dtype fp16 → bf16, 평가 `test.py` 의 모델/경로 로컬화

## 실행

1. conda 환경 활성화: `conda activate donut_vml`
2. `donut_classification_5090.ipynb` 를 위에서 아래로 실행 (VS Code Jupyter).
   - 첫 실행 시 `hftuner` 클론 + RVL-CDIP 데이터셋(대용량) 다운로드가 일어납니다.
3. 학습 결과는 `donut-classification-turbo/` 에 로컬 저장됩니다.

## 호환성 (검증 결과)

원본 `hftuner` 는 transformers 4.56.x 를 가정하지만, **현재 env(transformers 5.12.1)에서 검증 완료**:

- ✅ `from hftuner.donut import DataProcessor, DonutModel` import + 인스턴스화
- ✅ `DonutProcessor.from_pretrained` / `DonutModel.from_pretrained('hf-tuner/donut-classification-turbo')` (200.3M)
- ✅ cell 25 의 MBart 내부 경로 `model.decoder.model.decoder.embed_positions.weight` (shape (10,1024)) — 5.x 에서도 유지

`DonutModel` 이 `VisionEncoderDecoderModel` 의 얇은 서브클래스라 메이저 버전 차에도 안전했습니다.
아직 **전체 학습(Trainer.train)·generate 까지는 미검증** — 실행 중 5.x API 차로 막히면 첫 코드 셀의
`%pip install "transformers==4.56.1"` 주석을 푸세요(가급적 별도 env — 도면 학습 노트북과 충돌 방지).

> 참고: `CLAUDE.md` 에는 transformers 4.57.6 으로 적혀 있으나 실제 `donut_vml` env 는 **5.12.1** 입니다.

## 기타

- GPU 여유가 적으면 `per_device_train_batch_size` 를 1 로 낮추고 `gradient_accumulation_steps` 를 올리세요.
