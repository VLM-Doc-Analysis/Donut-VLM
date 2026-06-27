"""노트북에서 GitHub 렌더를 깨뜨리는 위젯 메타데이터/출력을 제거.

원인: 노트북 실행 시 'metadata.widgets' 가 'state' 키 없이 저장되면
GitHub/nbconvert 가 "Invalid Notebook: the 'state' key is missing from
'metadata.widgets'" 로 렌더 거부. 이 스크립트가 그 잔재를 정리한다.

제거 대상 (출력 텍스트/이미지 등 *유효한 출력은 보존*):
  · 노트북/셀의 metadata.widgets
  · application/vnd.jupyter.widget-view+json 출력 (렌더 안 되는 위젯 진행바)

사용:
  python3 scripts/clean_ipynb_widgets.py <a.ipynb> [b.ipynb ...]
  (인자 없으면 git 스테이징된 .ipynb 자동 처리)
"""
import json
import subprocess
import sys


def clean(path):
    with open(path, encoding="utf-8") as fh:
        nb = json.load(fh)
    changed = False

    if nb.get("metadata", {}).pop("widgets", None) is not None:
        changed = True

    for cell in nb.get("cells", []):
        if cell.get("metadata", {}).pop("widgets", None) is not None:
            changed = True
        if cell.get("cell_type") == "code" and cell.get("outputs"):
            kept = []
            for out in cell["outputs"]:
                data = out.get("data", {}) if isinstance(out, dict) else {}
                if "application/vnd.jupyter.widget-view+json" in data:
                    changed = True
                    continue  # 위젯 출력 제거
                kept.append(out)
            cell["outputs"] = kept

    if changed:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(nb, fh, ensure_ascii=False, indent=1)
            fh.write("\n")
        print(f"[clean-ipynb] stripped widget metadata: {path}")
    return changed


def staged_notebooks():
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        capture_output=True, text=True,
    ).stdout
    return [f for f in out.splitlines() if f.endswith(".ipynb")]


def main(argv):
    paths = argv or staged_notebooks()
    any_changed = False
    for p in paths:
        if not p.endswith(".ipynb"):
            continue
        try:
            any_changed |= clean(p)
        except FileNotFoundError:
            pass  # 삭제된 파일 등
        except Exception as e:
            print(f"[clean-ipynb] skip {p}: {e}", file=sys.stderr)
    return 0 if not any_changed else 0  # 항상 0 (커밋 차단 안 함, 정리만)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
