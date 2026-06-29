# Element Donut field-F1 평가 유틸 (평가리포트 §0 재현용)
# 사용: conda activate donut_vml; cd yolo_finetune_donut_pipeline
#       python eval_fieldf1.py ../checkpoints_elements/final [typecond]
# 노트북(donut_training_elements_paper.ipynb) cell 11 헬퍼를 재사용해 로직 일치 보장.
import sys, json, glob, os
from pathlib import Path
import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
from collections import defaultdict

CKPT = sys.argv[1]
TYPECOND = len(sys.argv) > 2 and sys.argv[2] == "typecond"

# 노트북 paper cell 11 헬퍼 그대로 로드(로직 일치 보장)
nb = json.load(open("donut_training_elements_paper.ipynb", encoding="utf-8"))
for c in nb["cells"]:
    s = "".join(c["source"])
    if "공통: 스키마" in s:
        exec(s.replace("print(", "#print("), globals()); break

device = "cuda" if torch.cuda.is_available() else "cpu"
proc = DonutProcessor.from_pretrained(CKPT, backend="pil")
model = VisionEncoderDecoderModel.from_pretrained(CKPT).to(device).eval()
model.config.tie_word_embeddings = True
model.decoder.config.tie_word_embeddings = True
model.tie_weights()
task = "<s_element>"
dec = proc.tokenizer(task, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

val = "../data/processed_elements/val"
pairs = []
for ip in sorted(glob.glob(val + "/images/*.png")):
    lp = val + "/labels/" + os.path.splitext(os.path.basename(ip))[0] + ".json"
    if os.path.exists(lp):
        g = json.load(open(lp, encoding="utf-8")); k = next(iter(g)); pairs.append((ip, k, str(g[k])))

def to_struct(seq, cls):
    for t in (proc.tokenizer.eos_token, proc.tokenizer.pad_token, proc.tokenizer.bos_token, task):
        if t: seq = seq.replace(t, "")
    p = token2json(seq.strip())
    p = decode_tree(p) if isinstance(p, dict) else {}
    # flat 체크포인트({value:..}) → 타입별 구조화 재파싱 (구조화 체크포인트는 그대로)
    if isinstance(p, dict) and set(p.keys()) == {"value"} and cls != "Hole_Callout":
        p = parse_to_schema(cls, _norm_value(str(p["value"])))
    return p if isinstance(p, dict) else {}

def leaf(d): return "".join(str(x) for x in _flatten(d).values())

P = R = F = H = EX = CS = 0.0
per = defaultdict(lambda: {"n": 0, "f": 0.0, "ex": 0, "cs": 0.0})
with torch.inference_mode():
    for ip, c, v in pairs:
        pv = proc(Image.open(ip).convert("RGB"), return_tensors="pt").pixel_values.to(device)
        extra = {}
        if TYPECOND:
            sup = forbidden_tag_ids(proc.tokenizer, c)
            if sup: extra["suppress_tokens"] = sup
        out = model.generate(pv, decoder_input_ids=dec, max_length=128,
            pad_token_id=proc.tokenizer.pad_token_id, eos_token_id=proc.tokenizer.eos_token_id,
            use_cache=True, no_repeat_ngram_size=3, repetition_penalty=1.5, **extra)
        pred = to_struct(proc.batch_decode(out, skip_special_tokens=False)[0], c)
        gts = parse_to_schema(c, v)
        p, r, f, h = field_prf1(pred, gts); ex = (_flatten(pred) == _flatten(gts))
        cs = value_charsim(leaf(pred), leaf(gts))
        P += p; R += r; F += f; H += h; EX += ex; CS += cs
        d = per[c]; d["n"] += 1; d["f"] += f; d["ex"] += ex; d["cs"] += cs
n = len(pairs)
print(f"\n=== {CKPT}  (typecond={TYPECOND}, n={n}) ===")
print(f"  Field-F1 {F/n:.3f}  (P {P/n:.3f} / R {R/n:.3f} / Halluc {H/n:.3f})  |  charsim {CS/n:.3f}  |  exact {EX/n*100:.1f}%")
for c in sorted(per, key=lambda c: -per[c]["n"]):
    d = per[c]; print(f"    {c:<18} F1 {d['f']/d['n']:.3f} / charsim {d['cs']/d['n']:.3f} / exact {d['ex']/d['n']*100:5.1f}%  (n={d['n']})")
