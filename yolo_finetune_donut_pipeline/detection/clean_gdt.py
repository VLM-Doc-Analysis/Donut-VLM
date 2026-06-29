# GD&T 라벨/크롭 품질 정제 — 자동 정규화 + 수동 작업 분류
# 사용: python yolo_finetune_donut_pipeline/detection/clean_gdt.py [--apply]
# 상세: ../Element_Donut_GDT_품질정제_작업목록.md
import json, glob, os, re, sys
APPLY = "--apply" in sys.argv
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))   # detection → pipeline → repo
ROOTS = [os.path.join(REPO, p) for p in ("data/elements", "data/processed_elements/train", "data/processed_elements/val")]
GDT = {"GD&T_FCF", "GD&T", "GDT"}
GEOM = set("⊥∥∠⊕◎○⌭⌒⌓⏥⏤⌯↗⌰⟂⊙=")      # 유효 FCF 선두 기하기호
LEADOK = GEOM | set("Ø∅⌀")                  # Ø 리딩(원통공차)도 허용

def normalize(v):
    o = v
    v = re.sub(r"(?<=\d)·(?=\d)", ".", v)        # 0·02 → 0.02
    v = re.sub(r"(?<![\d.])0-(?=\d)", "0.", v)   # 0-02 → 0.02 (범위 10-31·나사 1/2-13·datum A-B 보존)
    v = v.replace("⟂", "⊥")            # ⟂ → ⊥ (perpendicularity 통일)
    v = v.replace("←", "↗").replace("⟵", "↗")  # ⟵/← → ↗ (runout 오독)
    return v, v != o

def lead(v):
    s = v.lstrip()
    return s[0] if s else ""

# 분류
cats = {"reclass_Ra": [], "normalized": [], "rotated_suspect": [], "multi_element": [], "ambiguous": [], "clean": []}
changes = {}   # stem -> (oldkey,oldval) -> (newkey,newval)
seen = {}
for r in ROOTS:
    for f in sorted(glob.glob(r + "/labels/*.json")):
        g = json.load(open(f, encoding="utf-8")); k = next(iter(g))
        if k not in GDT: continue
        stem = os.path.basename(f)[:-5]; v = str(g[k])
        newk, newv = k, v
        # 1) Ra → Surface_Roughness 재분류
        if re.match(r"^\s*Ra\b|^\s*Ra[\d.]", v):
            newk = "Surface_Roughness"
            if stem not in seen: cats["reclass_Ra"].append((stem, v))
        else:
            nv, ch = normalize(v); newv = nv
            if ch and stem not in seen: cats["normalized"].append((stem, v, nv))
            # 분류(리포트용; 정규화 후 기준)
            if stem not in seen:
                if "\n" in nv: cats["multi_element"].append((stem, nv))
                elif lead(nv) not in LEADOK and not re.match(r"^\s*\d+[xX×]", nv):
                    cats["rotated_suspect"].append((stem, nv))
                elif "±" in nv or re.search(r"=.*-$", nv):
                    cats["ambiguous"].append((stem, nv))
                else:
                    cats["clean"].append((stem, nv))
        if (newk, newv) != (k, v):
            if APPLY:
                json.dump({newk: newv}, open(f, "w", encoding="utf-8"), ensure_ascii=False)
            changes[f] = (k, v, newk, newv)
        seen[stem] = True

n = len(seen)
print(f"=== GD&T 품질 정제 {'(APPLY)' if APPLY else '(DRY-RUN)'} — 고유 {n}건 ===\n")
print(f"[자동수정] Ra 재분류 {len(cats['reclass_Ra'])} · 정규화 {len(cats['normalized'])}  → 적용 파일 {len(changes)}개")
for stem, v in cats["reclass_Ra"]: print(f"   reclass→Surface  {v!r:20} {stem}")
for stem, ov, nv in cats["normalized"]: print(f"   normalize  {ov!r:18} → {nv!r:18} {stem}")
print(f"\n[수동 필요] 회전/오독 의심 {len(cats['rotated_suspect'])} · 다중요소 {len(cats['multi_element'])} · 모호기호 {len(cats['ambiguous'])}")
print("\n-- 회전/오독 의심(선두가 기하기호 아님) --")
for stem, v in cats["rotated_suspect"]: print(f"   {v!r:28} {stem}")
print("\n-- 다중요소 크롭(\\n 포함 → 재크롭 필요) --")
for stem, v in cats["multi_element"]: print(f"   {v!r:40} {stem}")
print("\n-- 모호 기호(±/= 등) --")
for stem, v in cats["ambiguous"]: print(f"   {v!r:28} {stem}")
print(f"\n[정상] {len(cats['clean'])}건")
