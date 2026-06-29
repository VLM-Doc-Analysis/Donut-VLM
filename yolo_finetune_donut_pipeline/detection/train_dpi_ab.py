# 고DPI 품질 A/B 학습+평가 — 크롭 디렉터리 + image_size 받아 flat Donut 학습 후 field-F1
# 사용: conda activate donut_vml
#   python yolo_finetune_donut_pipeline/detection/train_dpi_ab.py --crops data/elements_hidpi_900 --image-size 768 --out checkpoints_ab_900
#   python yolo_finetune_donut_pipeline/detection/train_dpi_ab.py --crops data/elements_hidpi_300 --image-size 768 --out checkpoints_ab_300
# ⚠️ 라벨은 '검수 완료'(reviewed) 여야 의미 있음. pre-fill 상태로 돌리면 plumbing 확인용일 뿐(수치 무의미).
import os, sys, json, glob, argparse, random
import numpy as np, torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from transformers import (DonutProcessor, VisionEncoderDecoderModel,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments)
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
os.chdir(REPO)
# 헬퍼(json2token/token2json/parse_to_schema/field_prf1/_flatten/value_charsim/_norm_value) 재사용
nb = json.load(open("yolo_finetune_donut_pipeline/donut_training_elements_paper.ipynb", encoding="utf-8"))
for c in nb["cells"]:
    if "공통: 스키마" in "".join(c["source"]):
        exec("".join(c["source"]).replace("print(", "#print("), globals()); break
USE_UNICODE_SYMBOLS = False   # flat A/B: 기호 토큰 등록(플랫 노트북 방식, 최고 성능본)

ap = argparse.ArgumentParser()
ap.add_argument("--crops", required=True); ap.add_argument("--image-size", type=int, default=768)
ap.add_argument("--out", required=True); ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--max-steps", type=int, default=-1, help=">0 면 smoke 테스트")
ap.add_argument("--val-ratio", type=float, default=0.15); ap.add_argument("--seed", type=int, default=42)
A = ap.parse_args()
random.seed(A.seed); np.random.seed(A.seed); torch.manual_seed(A.seed)
TASK = "<s_element>"; ISZ = A.image_size
device = "cuda" if torch.cuda.is_available() else "cpu"

# ── 샘플 수집 + split ──
root = A.crops if os.path.isabs(A.crops) else os.path.join(REPO, A.crops)
pairs = []
for lp in sorted(glob.glob(f"{root}/labels/*.json")):
    ip = f"{root}/images/{Path(lp).stem}.png"
    if not os.path.exists(ip): continue
    g = json.load(open(lp, encoding="utf-8")); cls = next(iter(g)); val = str(g[cls])
    if val.strip() == "": continue   # 미라벨 제외
    pairs.append((ip, cls, val))
random.shuffle(pairs); nv = max(1, int(len(pairs)*A.val_ratio))
val_p, train_p = pairs[:nv], pairs[nv:]
print(f"crops={root} | train {len(train_p)} / val {len(val_p)} | image_size {ISZ}")

# ── 프로세서/모델(flat: <s_value> + 기호 토큰) ──
proc = DonutProcessor.from_pretrained("naver-clova-ix/donut-base", backend="pil")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
proc.image_processor.size = {"height": ISZ, "width": ISZ}
proc.image_processor.do_align_long_axis = False; proc.image_processor.do_thumbnail = True; proc.image_processor.do_pad = True
model.config.encoder.image_size = [ISZ, ISZ]; model.config.decoder.max_length = 128
spec = [TASK, "<sep/>", "<s_value>", "</s_value>"]
na = proc.tokenizer.add_special_tokens({"additional_special_tokens": spec})
import unicodedata
chars = set()
for _, _, v in pairs: chars.update(v)
syms = sorted(c for c in chars if ord(c) > 127 and unicodedata.normalize("NFKC", c) == c)
ns = proc.tokenizer.add_tokens(syms); nd = proc.tokenizer.add_tokens(list("0123456789"))
if na or ns or nd: model.decoder.resize_token_embeddings(len(proc.tokenizer), mean_resizing=False)
model.config.pad_token_id = proc.tokenizer.pad_token_id
model.config.decoder_start_token_id = proc.tokenizer.convert_tokens_to_ids(TASK)
model = model.to(device)
print(f"토큰 추가: 구조 {na} + 기호 {ns} + 숫자 {nd}")

class DS(Dataset):
    def __init__(self, items, aug=False): self.items = items; self.aug = aug
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        ip, cls, val = self.items[i]; im = Image.open(ip).convert("RGB")
        tgt = TASK + json2token({"value": val}) + proc.tokenizer.eos_token
        pv = proc(im, return_tensors="pt").pixel_values.squeeze(0)
        lab = proc.tokenizer(tgt, add_special_tokens=False, max_length=128, padding="max_length",
                             truncation=True, return_tensors="pt").input_ids.squeeze(0)
        lab[lab == proc.tokenizer.pad_token_id] = -100
        return {"pixel_values": pv, "labels": lab}

args = Seq2SeqTrainingArguments(
    output_dir=A.out, num_train_epochs=A.epochs, max_steps=A.max_steps,
    per_device_train_batch_size=4, per_device_eval_batch_size=4, gradient_accumulation_steps=2,
    learning_rate=3e-5, warmup_steps=30, weight_decay=0.01, bf16=True, fp16=False,
    eval_strategy="no", save_strategy="no", logging_steps=20, dataloader_num_workers=4,
    report_to=["none"], predict_with_generate=False)
trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=DS(train_p, True))
trainer.train()
final = os.path.join(A.out, "final"); trainer.save_model(final); proc.save_pretrained(final)

# ── field-F1 평가(val) ──
model.eval(); di = proc.tokenizer(TASK, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
def read_val(ip):
    pv = proc(Image.open(ip).convert("RGB"), return_tensors="pt").pixel_values.to(device)
    out = model.generate(pv, decoder_input_ids=di, max_length=128, pad_token_id=proc.tokenizer.pad_token_id,
        eos_token_id=proc.tokenizer.eos_token_id, use_cache=True, no_repeat_ngram_size=3, repetition_penalty=1.5)
    seq = proc.batch_decode(out, skip_special_tokens=False)[0]
    for t in (proc.tokenizer.eos_token, proc.tokenizer.pad_token, proc.tokenizer.bos_token, TASK):
        if t: seq = seq.replace(t, "")
    p = token2json(seq.strip())
    return str(p.get("value")) if isinstance(p, dict) and "value" in p else (str(p) if not isinstance(p, dict) else "")
F = CS = EX = 0.0
with torch.inference_mode():
    for ip, cls, gt in val_p:
        pv = parse_to_schema(cls, _norm_value(read_val(ip))); gv = parse_to_schema(cls, gt)
        _, _, f, _ = field_prf1(pv, gv); F += f; EX += (_flatten(pv) == _flatten(gv))
        CS += value_charsim("".join(map(str, _flatten(pv).values())), "".join(map(str, _flatten(gv).values())))
n = len(val_p) or 1
print(f"\n=== {A.crops} | image_size {ISZ} | val {len(val_p)} ===")
print(f"  Field-F1 {F/n:.3f}  | charsim {CS/n:.3f}  | exact {EX/n*100:.1f}%")
