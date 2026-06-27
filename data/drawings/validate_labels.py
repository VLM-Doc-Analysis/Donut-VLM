"""
도면 라벨(JSON) 검수 스크립트

annotate_helper.py 의 TEMPLATE 스키마와 *카탈로그 고정* 규칙에 맞춰
labels/*.json 을 점검한다. 라벨 품질 = 모델 품질이므로, 학습 빌드 전에 돌려서
키 누락 / 미입력(빈 문자열) / 카탈로그 外 자유 작명(키 난립) 을 잡아낸다.

사용법:
  python3 validate_labels.py            # 검수 후 리포트
  python3 validate_labels.py --strict   # 경고(WARN)도 실패로 취급 (종료코드 1)

심각도
  [ERROR] 학습을 깨뜨릴 수 있음 — 코어 키 누락 / 구조 오류 / 타입 오류 / 미입력 잔존
  [WARN ] special token 난립 유발 — 카탈로그 外 키, 스키마와 다른 하위 키 이름
"""
import json
import sys
from pathlib import Path

BASE = Path(__file__).parent
IMAGES_DIR = BASE / 'images'
LABELS_DIR = BASE / 'labels'

# ── 스키마 정의 (annotate_helper.py 와 동기화할 것) ──────────────────────────
# 고정 코어 — 전 도면 공통, 항상 존재. 값이 없으면 "N/A" 로 채움(빈 "" 는 미입력).
TOP_CORE = {'title_block', 'dimensions', 'bolt_holes', 'surface_finish', 'gdt', 'threads', 'date'}
TITLE_BLOCK_KEYS = {'title', 'Rev', 'Drawing_no', 'LIC_Material', 'Material', 'Material_std'}
DIM_CORE = {'outer_diameter', 'bore_diameter', 'thickness'}
SF_KEYS = {'general', 'machined'}
GDT_KEYS = {'flatness', 'perpendicularity', 'concentricity'}
BOLT_OUTER_KEYS = {'count', 'diameter', 'pcd'}   # 통과홀
BOLT_INNER_KEYS = {'count', 'size', 'pcd'}        # 탭홀 — 'size'(나사규격), 'diameter' 아님

# 선택 확장 — 도면에 해당 표기가 있을 때만 추가. 반드시 카탈로그 이름 그대로(자유 작명 금지).
DIM_CATALOG = {
    'pcd', 'pcd_outer', 'pcd_inner', 'pilot_od', 'pilot_id', 'mid_diameter',
    'inner_bore', 'hub_od', 'hub_height', 'boss_diameter', 'boss_height',
    'slip_on_id', 'width', 'total_height', 'fit',
}
TOP_OPTIONAL = {'notes', 'variants'}   # notes: list, variants: str


def validate(path):
    """파일 1개 검수 → (errors, warns) 메시지 리스트."""
    errs, warns = [], []
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except Exception as e:
        return [f'JSON 파싱 실패: {e}'], []
    if not isinstance(data, dict):
        return ['최상위가 object(dict) 가 아님'], []

    # 1) 미입력(빈 문자열) 잔존 — 템플릿 그대로 방치된 값
    def find_empty(o, p=''):
        if isinstance(o, dict):
            for k, v in o.items():
                find_empty(v, f'{p}.{k}')
        elif isinstance(o, list):
            for i, v in enumerate(o):
                find_empty(v, f'{p}[{i}]')
        elif isinstance(o, str) and o.strip() == '':
            errs.append(f'미입력(빈 문자열): {p.lstrip(".")}')
    find_empty(data)

    # 2) 최상위 코어 키 존재
    for k in TOP_CORE:
        if k not in data:
            errs.append(f'코어 최상위 키 누락: {k}')
    # 3) 카탈로그 外 최상위 키 (난립)
    for k in data:
        if k not in TOP_CORE and k not in TOP_OPTIONAL:
            warns.append(f'카탈로그 外 최상위 키: {k}  (자유 작명 금지)')

    # 4) 섹션별 하위 키 / 타입
    def check_section(name, allowed, *, exact=False):
        sec = data.get(name)
        if sec is None:
            return  # 누락은 위에서 ERROR 처리
        if not isinstance(sec, dict):
            errs.append(f'{name} 은 object 여야 함 (현재 {type(sec).__name__})')
            return
        for k in allowed:
            if k not in sec:
                errs.append(f'{name}.{k} 누락')
        if exact:
            for k in sec:
                if k not in allowed:
                    warns.append(f'{name}.{k} — 스키마 外 키')

    check_section('title_block', TITLE_BLOCK_KEYS, exact=True)
    check_section('surface_finish', SF_KEYS, exact=True)
    if 'gdt' in data:
        check_section('gdt', GDT_KEYS, exact=True)

    # dimensions — 코어 3키 필수 + 나머지는 카탈로그 안에서만
    dims = data.get('dimensions')
    if isinstance(dims, dict):
        for k in DIM_CORE:
            if k not in dims:
                errs.append(f'dimensions.{k} 누락 (코어)')
        for k in dims:
            if k not in DIM_CORE and k not in DIM_CATALOG:
                warns.append(f'dimensions.{k} — 카탈로그 外 (자유 작명 금지)')
    elif dims is not None:
        errs.append('dimensions 은 object 여야 함')

    # bolt_holes — outer/inner 2조, 각 하위 키 고정
    bh = data.get('bolt_holes')
    if isinstance(bh, dict):
        for grp, keys in (('outer', BOLT_OUTER_KEYS), ('inner', BOLT_INNER_KEYS)):
            g = bh.get(grp)
            if not isinstance(g, dict):
                errs.append(f'bolt_holes.{grp} 누락 또는 object 아님')
                continue
            for k in keys:
                if k not in g:
                    errs.append(f'bolt_holes.{grp}.{k} 누락')
            for k in g:
                if k not in keys:
                    warns.append(f'bolt_holes.{grp}.{k} — 스키마 外 키 '
                                 f'(inner 은 size, outer 는 diameter)')
    elif bh is not None:
        errs.append('bolt_holes 은 object 여야 함')

    # 5) 타입 검사 (리스트/문자열)
    if 'threads' in data and not isinstance(data['threads'], list):
        errs.append('threads 는 list 여야 함')
    if 'notes' in data and not isinstance(data['notes'], list):
        errs.append('notes 는 list 여야 함')
    if 'variants' in data and not isinstance(data['variants'], str):
        errs.append('variants 는 string 여야 함')
    if 'date' in data and not isinstance(data['date'], str):
        errs.append('date 는 string 여야 함')

    return errs, warns


def main():
    strict = '--strict' in sys.argv
    labels = sorted(LABELS_DIR.glob('*.json'))
    img_stems = {p.stem for p in IMAGES_DIR.glob('*.png')} | {p.stem for p in IMAGES_DIR.glob('*.jpg')}
    lbl_stems = {p.stem for p in labels}

    # 페어링 검사
    pair_errs = []
    for s in sorted(img_stems - lbl_stems):
        pair_errs.append(f'[PAIR ] 라벨 없는 이미지: {s}')
    for s in sorted(lbl_stems - img_stems):
        pair_errs.append(f'[PAIR ] 이미지 없는 라벨: {s}.json')
    for m in pair_errs:
        print(m)

    n_err_files = n_warn_files = 0
    total_err = total_warn = 0
    for path in labels:
        errs, warns = validate(path)
        total_err += len(errs)
        total_warn += len(warns)
        if errs:
            n_err_files += 1
        if warns:
            n_warn_files += 1
        if errs or warns:
            print(f'\n■ {path.name}')
            for e in errs:
                print(f'   [ERROR] {e}')
            for w in warns:
                print(f'   [WARN ] {w}')

    print('\n' + '─' * 60)
    print(f'검수 대상: {len(labels)}개 라벨 / {len(img_stems)}개 이미지')
    print(f'페어 불일치: {len(pair_errs)}건')
    print(f'ERROR: {total_err}건 ({n_err_files}개 파일) | WARN: {total_warn}건 ({n_warn_files}개 파일)')

    failed = bool(pair_errs) or total_err > 0 or (strict and total_warn > 0)
    if failed:
        print('결과: ❌ 실패 — 위 항목을 수정하세요.')
        sys.exit(1)
    print('결과: ✅ 통과')


if __name__ == '__main__':
    main()
