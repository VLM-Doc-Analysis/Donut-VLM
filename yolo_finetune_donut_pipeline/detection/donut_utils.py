"""
donut_utils.py — Donut 토큰 I/O 공용 유틸리티 모듈.

crop_utils.py 와 동일하게 노트북 옆에 두는 .py 헬퍼 패턴입니다. 단일 커널(kardi_env)에서
검출·인식을 모두 돌리는 파이프라인 노트북들이 import 해서 씁니다:

    import sys; sys.path.append("detection")
    from donut_utils import json2token, token2json, decode_donut_output

- json2token / token2json : JSON ↔ Donut XML 식 토큰 시퀀스 상호 변환.
- decode_donut_output     : generate() 디코딩 문자열에서 특수 토큰을 떼고 dict 로 파싱.

※ 학습 노트북(donut_training_elements.ipynb)은 이 변환을 **교육 목적으로 셀에 직접** 정의해
  설명하므로 그쪽 인라인 정의는 그대로 둡니다. 이 모듈은 end-to-end 파이프라인 runner 가
  같은 로직을 재정의(드리프트 위험)하지 않도록 한 곳에 모은 것입니다.
"""
from __future__ import annotations

import re


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

    특수 토큰(eos/pad/bos)과 task 토큰을 먼저 제거한 뒤 token2json 으로 파싱합니다.
    ★ task 토큰을 안 떼면 닫힘 태그 없는 키로 오인돼 정규식 파싱이 깨지고 점수가 0 이 됩니다
      (MEMORY: "token2json 전 BOS+task 제거"). strip 대상 토큰 집합을 여기 한 곳에서 관리합니다.
    """
    for t in (tokenizer.eos_token, tokenizer.pad_token, tokenizer.bos_token, task_prompt):
        if t:
            sequence = sequence.replace(t, "")
    return token2json(sequence.strip())
