# 검수용 held-out val 부분집합 생성 — 층화 샘플 + 전용 검수 HTML
# 사용: conda activate donut_vml; python yolo_finetune_donut_pipeline/detection/make_val_review.py [--n 100] [--seed 42]
# 출력: data/elements_hidpi/val_ids.txt      (held-out crop id 목록 — 학습 제외용)
#        data/elements_hidpi/review_val.html  (이 부분집합만 검수 → 💾 저장 시 reviewed_val.jsonl)
# 이후: 사람이 review_val.html 값 확정/교정 → reviewed_val.jsonl
#        → ingest_reviewed.py 로 val 라벨=GT 반영
#        → train_dpi_ab.py --val-ids val_ids.txt 로 '비순환' field-F1
import os, json, argparse, random
from collections import defaultdict
import make_label_review as M

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=100, help="val 부분집합 크기(층화)")
ap.add_argument("--seed", type=int, default=42)
A = ap.parse_args()
HID = M.HID
random.seed(A.seed)

# 클래스별 그룹화(값 있는 것만)
by = defaultdict(list)
for ln in open(os.path.join(HID, "values_hidpi.jsonl"), encoding="utf-8"):
    ln = ln.strip()
    if not ln: continue
    d = json.loads(ln)
    if str(d.get("value", "")).strip() == "": continue
    by[d.get("class", "value")].append(d["crop"])

total = sum(len(v) for v in by.values())
# 클래스 비율대로 배분(희소 클래스는 최소 보장), 단 보유량 초과 금지
picked = []
for cls, crops in by.items():
    random.shuffle(crops)
    share = max(2, round(A.n * len(crops) / total))   # 최소 2개 보장
    picked += crops[:min(share, len(crops))]
random.shuffle(picked); picked = picked[:A.n]
picked_set = set(picked)

open(os.path.join(HID, "val_ids.txt"), "w", encoding="utf-8").write("\n".join(sorted(picked_set)))
rows = M.build_rows(os.path.join(HID, "values_hidpi.jsonl"), only=picked_set)
out, n, sz = M.build_html(rows, os.path.join(HID, "review_val.html"),
                          save_name="reviewed_val.jsonl", storage_key="elemReviewVal_v1", title="(held-out val)")
from collections import Counter
cc = Counter(r["cls"] for r in rows)
print(f"val 부분집합 {n}개 → {out} ({sz//1024} KB) · val_ids.txt 저장")
print("클래스 분포:", dict(cc.most_common()))
print("→ review_server.py 로 review_val.html 검수 → 💾 (reviewed_val.jsonl) → ingest → train_dpi_ab --val-ids")
