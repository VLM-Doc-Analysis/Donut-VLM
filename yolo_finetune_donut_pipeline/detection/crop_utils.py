"""
crop_utils.py — 검출 박스를 크롭/정렬하는 공용 유틸리티 모듈.

도면 파이프라인의 노트북들이 import 해서 사용합니다 (annotate_helper.py 와 동일하게
노트북 옆에 두는 .py 헬퍼 패턴):

    from crop_utils import crop_aabb, rectify_obb, save_crops_from_result

- View 검출(YOLOv11, AABB) → crop_aabb 로 축정렬 사각형 크롭
- Element 검출(YOLOv11-OBB, 회전 박스) → rectify_obb 로 회전 박스를 수평으로 펴서 크롭
  (Donut 은 글자가 수평으로 정렬된 크롭을 입력으로 받아야 인식이 안정적입니다)

의존: numpy, opencv-python(cv2), Pillow. (kardi_env 에 모두 존재)
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# ──────────────────────────────────────────────────────────────────────────
# 이미지 로딩 유틸
# ──────────────────────────────────────────────────────────────────────────
def load_bgr(image):
    """경로/PIL/np.ndarray 무엇이 와도 OpenCV BGR ndarray 로 통일."""
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {image}")
        return img
    if isinstance(image, Image.Image):
        return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    if isinstance(image, np.ndarray):
        return image
    raise TypeError(f"지원하지 않는 이미지 타입: {type(image)}")


def to_pil(bgr: np.ndarray) -> Image.Image:
    """OpenCV BGR ndarray → PIL RGB (Donut 입력용)."""
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


# ──────────────────────────────────────────────────────────────────────────
# AABB (축정렬) 크롭 — View 검출 결과용
# ──────────────────────────────────────────────────────────────────────────
def crop_aabb(image, xyxy, pad: int = 0) -> Image.Image:
    """축정렬 박스 [x1,y1,x2,y2] (픽셀 좌표)로 크롭한 PIL 이미지를 반환.

    pad: 박스 바깥으로 여유 픽셀(경계 글자 잘림 방지). 이미지 경계로 클램프.
    """
    bgr = load_bgr(image)
    h, w = bgr.shape[:2]
    # x1<x2/y1<y2 보장(좌표가 뒤집혀 와도 안전)하도록 정렬 후 pad 적용·경계 클램프
    lo_x, hi_x = sorted((float(xyxy[0]), float(xyxy[2])))
    lo_y, hi_y = sorted((float(xyxy[1]), float(xyxy[3])))
    x1 = int(max(0, lo_x - pad))
    x2 = int(min(w, hi_x + pad))
    y1 = int(max(0, lo_y - pad))
    y2 = int(min(h, hi_y + pad))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"빈 크롭 영역: {xyxy}")
    return to_pil(bgr[y1:y2, x1:x2])


# ──────────────────────────────────────────────────────────────────────────
# OBB (회전) 정렬 크롭 — Element 검출 결과용
# ──────────────────────────────────────────────────────────────────────────
def _order_quad(pts: np.ndarray) -> np.ndarray:
    """4점을 [좌상, 우상, 우하, 좌하] 순서로 정렬."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype=np.float32,
    )


def rectify_obb(image, quad, pad: int = 2, min_side: int = 8) -> Image.Image:
    """회전 박스(4점, 8개 픽셀 좌표)를 perspective warp 로 수평 직사각형으로 펴서 반환.

    quad: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] 또는 길이 8 시퀀스 (픽셀 좌표).
    회전 박스의 긴 변이 가로가 되도록(가로폭 >= 세로높이) 필요 시 90° 회전.
    """
    bgr = load_bgr(image)
    quad = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    q = _order_quad(quad)

    # 변 길이로 목적지 사각형 크기 산출
    width = int(round(max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3]))))
    height = int(round(max(np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1]))))
    width = max(width, min_side) + 2 * pad
    height = max(height, min_side) + 2 * pad

    dst = np.array(
        [[pad, pad], [width - 1 - pad, pad],
         [width - 1 - pad, height - 1 - pad], [pad, height - 1 - pad]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(q, dst)
    warped = cv2.warpPerspective(bgr, M, (width, height), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)

    # 치수 텍스트는 대개 가로로 길다 → 세로가 더 길면 90° 회전해 가로로 눕힘
    if warped.shape[0] > warped.shape[1]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return to_pil(warped)


# ──────────────────────────────────────────────────────────────────────────
# Ultralytics 결과 → 크롭 파일 저장 헬퍼
# ──────────────────────────────────────────────────────────────────────────
def save_crops_from_result(result, out_dir, stem, names=None, pad=2):
    """하나의 ultralytics Result 에서 모든 박스를 크롭해 PNG 로 저장.

    AABB(result.boxes) 와 OBB(result.obb) 를 자동 판별:
      - boxes  → crop_aabb (xyxy)
      - obb    → rectify_obb (xyxyxyxy 4점)

    저장 파일명: {stem}__{idx:03d}__{classname}.png
    반환: [{"path", "cls", "name", "conf", "xy"}] 메타 리스트 (Donut/조립 단계에서 사용).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img = result.orig_img  # BGR ndarray
    names = names or getattr(result, "names", {})
    meta = []

    obb = getattr(result, "obb", None)
    if obb is not None and obb.xyxyxyxy is not None and len(obb) > 0:
        polys = obb.xyxyxyxy.cpu().numpy()         # (N,4,2)
        clss = obb.cls.cpu().numpy().astype(int)
        confs = obb.conf.cpu().numpy()
        for i, (poly, c, cf) in enumerate(zip(polys, clss, confs)):
            crop = rectify_obb(img, poly, pad=pad)
            name = names.get(int(c), str(int(c)))
            p = out_dir / f"{stem}__{i:03d}__{name}.png"
            crop.save(p)
            meta.append({"path": str(p), "cls": int(c), "name": name,
                         "conf": float(cf), "xy": poly.tolist()})
        return meta

    boxes = getattr(result, "boxes", None)
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        for i, (b, c, cf) in enumerate(zip(xyxy, clss, confs)):
            crop = crop_aabb(img, b, pad=pad)
            name = names.get(int(c), str(int(c)))
            p = out_dir / f"{stem}__{i:03d}__{name}.png"
            crop.save(p)
            meta.append({"path": str(p), "cls": int(c), "name": name,
                         "conf": float(cf), "xy": b.tolist()})
    return meta
