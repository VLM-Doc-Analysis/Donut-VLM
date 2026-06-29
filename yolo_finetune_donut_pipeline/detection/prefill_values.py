# 값 라벨 pre-fill — 현 Donut 으로 고DPI 크롭 값 초안 생성(사람 교정용 워크리스트)
# 사용: conda activate donut_vml; python yolo_finetune_donut_pipeline/detection/prefill_values.py
# 입력: data/elements_hidpi/  (extract_hidpi_crops.py 산출)  ·  체크포인트: checkpoints_elements/final
# 출력: 각 라벨 {class: value_pred} 갱신 + values_hidpi.jsonl (status="pending", 사람이 교정)
import os, sys, json, glob, re
import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
os.chdir(REPO)
# 헬퍼(token2json, _flatten) — paper 노트북 cell 11 재사용
nb = json.load(open("yolo_finetune_donut_pipeline/donut_training_elements_paper.ipynb", encoding="utf-8"))
for c in nb["cells"]:
    if "공통: 스키마" in "".join(c["source"]):
        exec("".join(c["source"]).replace("print(", "#print("), globals()); break

CKPT = "checkpoints_elements/final"
HID = "data/elements_hidpi"
device = "cuda" if torch.cuda.is_available() else "cpu"
proc = DonutProcessor.from_pretrained(CKPT, backend="pil")
model = VisionEncoderDecoderModel.from_pretrained(CKPT).to(device).eval()
model.config.tie_word_embeddings = True; model.decoder.config.tie_word_embeddings = True; model.tie_weights()
task = "<s_element>"
di = proc.tokenizer(task, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

def read_value(ip):
    pv = proc(Image.open(ip).convert("RGB"), return_tensors="pt").pixel_values.to(device)
    out = model.generate(pv, decoder_input_ids=di, max_length=128,
        pad_token_id=proc.tokenizer.pad_token_id, eos_token_id=proc.tokenizer.eos_token_id,
        use_cache=True, no_repeat_ngram_size=3, repetition_penalty=1.5)
    seq = proc.batch_decode(out, skip_special_tokens=False)[0]
    for t in (proc.tokenizer.eos_token, proc.tokenizer.pad_token, proc.tokenizer.bos_token, task):
        if t: seq = seq.replace(t, "")
    p = token2json(seq.strip())
    if isinstance(p, dict):
        return str(p.get("value")) if "value" in p else "".join(str(v) for v in _flatten(p).values())
    return str(p)

man = []
with torch.inference_mode():
    for lp in sorted(glob.glob(f"{HID}/labels/*.json")):
        name = os.path.basename(lp)[:-5]; ip = f"{HID}/images/{name}.png"
        if not os.path.exists(ip): continue
        cls = next(iter(json.load(open(lp, encoding="utf-8"))))
        val = _norm_value(read_value(ip))
        json.dump({cls: val}, open(lp, "w", encoding="utf-8"), ensure_ascii=False)
        man.append({"crop": name + ".png", "class": cls, "value": val, "status": "pending"})
open(f"{HID}/values_hidpi.jsonl", "w", encoding="utf-8").write("\n".join(json.dumps(m, ensure_ascii=False) for m in man))
print(f"pre-fill 완료: {len(man)} 크롭 → {HID}/values_hidpi.jsonl (status=pending, 사람 교정 대기)")
print("샘플:")
for m in man[:8]: print(f"  [{m['class']:<16}] {m['value']!r}  ({m['crop']})")
