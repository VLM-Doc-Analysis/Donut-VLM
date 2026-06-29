"""
donut_utils.py — Donut 토큰 I/O 공용 유틸리티 모듈.

crop_utils.py 와 동일하게 노트북 옆에 두는 .py 헬퍼 패턴입니다. 단일 커널(donut_vml)에서
검출·인식을 모두 돌리는 파이프라인 노트북들이 import 해서 씁니다:

    import sys; sys.path.append("detection")
    from donut_utils import json2token, token2json, decode_donut_output

- json2token / token2json : JSON ↔ Donut XML 식 토큰 시퀀스 상호 변환.
- decode_donut_output     : generate() 디코딩 문자열에서 특수 토큰을 떼고 dict 로 파싱.

※ 학습 노트북(donut_training_elements_flat.ipynb)은 이 변환을 **교육 목적으로 셀에 직접** 정의해
  설명하므로 그쪽 인라인 정의는 그대로 둡니다(CLAUDE.md: 교육 인라인 톤 유지). 이 모듈은
  end-to-end 파이프라인 runner 가 같은 로직을 재정의(드리프트 위험)하지 않도록 한 곳에 모은
  **런타임 단일 소스**입니다.

  ⚠️ 미러링 계약 — 아래 함수들은 학습 노트북의 동명 인라인 정의와 **반드시 동일**해야 합니다.
     한쪽만 고치면 학습 시퀀스와 추론 디코딩이 어긋나 점수가 무너집니다. 둘을 함께 수정할 것:
       json2token · token2json · encode_symbols · decode_symbols · encode_tree · decode_tree
     (parse_to_schema 등 도메인 스키마 함수는 학습 전용이라 여기 두지 않음 — 추론은 token2json
      라운드트립만 쓰므로 불필요.)
"""
from __future__ import annotations

import re


# ── 기호 ↔ code-point 인코딩 ───────────────────────────────────────────
# 학습(donut_training_elements_flat.ipynb)은 값의 공학 기호(Ø,⊥,±,° …)를 " U+XXXX " ASCII
# 토큰으로 바꿔(encode_symbols) 토크나이저 OOV/NFKC 문제를 없앤 뒤 학습합니다. 따라서
# 추론 디코딩에서 **반드시 역변환(decode_symbols)** 해야 "U+00D8 65" 가 아니라 "Ø65" 가 나옵니다.
# (학습 노트북과 동일 정의 — 드리프트 방지용 단일 소스.)
def _keep_native(c: str) -> bool:   # CJK/한글/가나/전각 — 다국어 토크나이저가 처리하므로 인코딩 안 함
    o = ord(c)
    return (0xAC00 <= o <= 0xD7A3) or (0x1100 <= o <= 0x11FF) or (0x3130 <= o <= 0x318F) \
        or (0x4E00 <= o <= 0x9FFF) or (0x3040 <= o <= 0x30FF) or (0xFF00 <= o <= 0xFFEF)


def encode_symbols(s) -> str:
    """공학 기호(Ø,⊥,±,° …)만 " U+XXXX " 로. ASCII·CJK/한글은 그대로."""
    return "".join(c if (ord(c) <= 127 or _keep_native(c)) else f" U+{ord(c):04X} " for c in str(s))


def decode_symbols(s) -> str:
    """" U+XXXX " → 실제 글리프 (encode_symbols 의 역)."""
    return re.sub(r"\s*U\+([0-9A-Fa-f]{4,6})\s*", lambda m: chr(int(m.group(1), 16)), str(s))


def _walk(o, fn):
    if isinstance(o, dict):
        return {k: _walk(v, fn) for k, v in o.items()}
    if isinstance(o, list):
        return [_walk(v, fn) for v in o]
    return fn(str(o))


def encode_tree(o):
    """dict/list 의 모든 leaf 값에 encode_symbols 적용."""
    return _walk(o, encode_symbols)


def decode_tree(o):
    """dict/list 의 모든 leaf 값에 decode_symbols 적용."""
    return _walk(o, decode_symbols)


def json2token(obj, sort_keys: bool = True) -> str:
    """dict/list → Donut 토큰 시퀀스 변환.

    Donut의 정답은 JSON을 XML 스타일 토큰으로 표현합니다.
    예) {"total": "12500"} → <s_total>12500</s_total>
        {"items": [{"nm": "A"}]} → <s_items><s_nm>A</s_nm></s_items>

    sort_keys=True : 키를 역순 정렬하여 항상 동일한 순서를 보장합니다.
    """
    if isinstance(obj, dict):
        output = ""
        keys = sorted(obj.keys(), reverse=True) if sort_keys else obj.keys()
        for k in keys:
            output += f"<s_{k}>" + json2token(obj[k], sort_keys) + f"</s_{k}>"
        return output
    elif isinstance(obj, list):
        # 리스트 항목은 <sep/> 토큰으로 구분
        return "<sep/>".join([json2token(v, sort_keys) for v in obj])
    else:
        return str(obj)


def token2json(tokens: str):
    """모델이 생성한 토큰 시퀀스를 Python dict로 역변환 (json2token의 역과정).

    예) "<s_total>12500</s_total>" → {"total": "12500"}
    정규표현식으로 <s_key>...</s_key> 패턴을 찾아 재귀적으로 파싱합니다.
    파싱할 구조가 없으면 입력 문자열을 그대로(strip) 반환합니다.
    """
    output = {}
    while tokens:
        start = re.search(r"<s_(.+?)>", tokens)
        if not start:
            break
        key = start.group(1)
        end_pat = f"</s_{key}>"
        end_pos = tokens.find(end_pat, start.end())
        if end_pos == -1:
            break
        value = tokens[start.end():end_pos]
        # 값 안에 중첩 태그가 있으면 재귀 호출로 파싱
        output[key] = token2json(value) if "<s_" in value else value.strip()
        tokens = tokens[end_pos + len(end_pat):]
    return output if output else tokens.strip()


def decode_donut_output(sequence: str, tokenizer, task_prompt: str):
    """Donut generate() 디코딩 문자열 → dict(또는 str).

    특수 토큰(eos/pad/bos)과 task 토큰을 먼저 제거 → token2json 으로 파싱 → decode_tree 로
    " U+XXXX " 기호를 실제 글리프로 복원합니다(학습이 encode_symbols 로 인코딩해 학습했으므로 필수).
    ★ task 토큰을 안 떼면 닫힘 태그 없는 키로 오인돼 정규식 파싱이 깨지고 점수가 0 이 됩니다
      (MEMORY: "token2json 전 BOS+task 제거"). strip 대상 토큰 집합을 여기 한 곳에서 관리합니다.
    """
    for t in (tokenizer.eos_token, tokenizer.pad_token, tokenizer.bos_token, task_prompt):
        if t:
            sequence = sequence.replace(t, "")
    parsed = token2json(sequence.strip())
    # 학습 때 기호를 U+XXXX 로 인코딩했으므로 추론에서 반드시 역변환 (안 하면 "Ø65"→"U+00D8 65")
    return decode_tree(parsed) if isinstance(parsed, dict) else {"value": decode_symbols(str(parsed))}
