# 검수 결과 반영 — review.html 의 reviewed.jsonl → 라벨/매니페스트 갱신
# 사용: conda activate donut_vml
#   python yolo_finetune_donut_pipeline/detection/ingest_reviewed.py <reviewed.jsonl 경로>
# 동작: status=ok → 라벨 {class: value} 갱신(검수본). status=bad → 라벨 삭제 + _excluded.txt 기록.
#       values_hidpi.jsonl 의 status 갱신.
import os, sys, json, glob
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
HID = os.path.join(REPO, "data/elements_hidpi")

rev_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HID, "reviewed.jsonl")
assert os.path.exists(rev_path), f"reviewed.jsonl 없음: {rev_path}"
rev = {}
for ln in open(rev_path, encoding="utf-8"):
    ln = ln.strip()
    if ln:
        d = json.loads(ln); rev[d["crop"]] = d

ok = bad = 0; excluded = []
for crop, d in rev.items():
    stem = os.path.splitext(crop)[0]; lp = os.path.join(HID, "labels", stem + ".json")
    if d["status"] == "ok":
        json.dump({d["class"]: d["value"]}, open(lp, "w", encoding="utf-8"), ensure_ascii=False); ok += 1
    elif d["status"] == "bad":
        if os.path.exists(lp): os.remove(lp)
        ip = os.path.join(HID, "images", crop)
        if os.path.exists(ip): os.remove(ip)
        excluded.append(crop); bad += 1

if excluded:
    open(os.path.join(HID, "_excluded.txt"), "a", encoding="utf-8").write("\n".join(excluded) + "\n")

# 매니페스트 status 갱신
man = []
for ln in open(os.path.join(HID, "values_hidpi.jsonl"), encoding="utf-8"):
    ln = ln.strip()
    if not ln: continue
    m = json.loads(ln); r = rev.get(m["crop"])
    if r and r["status"] == "bad": continue  # 제외분은 매니페스트에서도 제거
    if r and r["status"] == "ok":
        m["class"] = r["class"]; m["value"] = r["value"]; m["status"] = "reviewed"
    man.append(m)
open(os.path.join(HID, "values_hidpi.jsonl"), "w", encoding="utf-8").write(
    "\n".join(json.dumps(m, ensure_ascii=False) for m in man))

print(f"반영 완료: 검수 ok {ok} → 라벨 갱신 / bad {bad} → 제외(이미지·라벨 삭제, _excluded.txt 기록)")
print(f"매니페스트 남은 항목: {len(man)}")
