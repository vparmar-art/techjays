import base64
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import fitz
import numpy as np
import requests
from django.conf import settings


MARKER_FIRST_14_ONLY = True
MIN_SEGMENT_LENGTH_PX = 22.0
DEDUPE_BIN_PX = 10
TILE_SIZE_PX = 1024
TILE_OVERLAP_PX = 224
MIN_AXIS_RATIO = 2.2
MIN_CANDIDATE_LENGTH_PX = 42.0
MIN_CANDIDATE_EVIDENCE = 0.45
MAX_CANDIDATES_FOR_AI = 320
EVIDENCE_DARK_THRESHOLD = 178
EVIDENCE_BAND_RADIUS_PX = 2
EVIDENCE_STRONG_THRESHOLD = 0.58
CLIPPED_OUTSIDE_RATIO_THRESHOLD = 0.20
GOOD_STRONG_RATIO_THRESHOLD = 0.70
WARN_STRONG_RATIO_THRESHOLD = 0.45
GOOD_CLIPPED_RATIO_THRESHOLD = 0.15
WARN_CLIPPED_RATIO_THRESHOLD = 0.35
GOOD_MARKER_COVERED_RATIO_THRESHOLD = 0.80
WARN_MARKER_COVERED_RATIO_THRESHOLD = 0.50
MARKER_COVERAGE_RADIUS_PX = 30
GOOD_BOUNDARY_COMPLETE_RATIO_THRESHOLD = 0.80
WARN_BOUNDARY_COMPLETE_RATIO_THRESHOLD = 0.50
MARKER_DEDUPE_CENTER_PX = 24.0
MARKER_DEDUPE_IOU_THRESHOLD = 0.25
SIZE_MARKER_MIN_RADIUS_PX = 500.0
SIZE_MARKER_RADIUS_DIAG_SCALE = 1.2
SIZE_MARKER_MIN_LINE_LEN_PX = 60.0
MARKER_PICK_LOCAL_MAX_CANDIDATES = 60
GOOD_LOCAL_NEAR_RATIO_THRESHOLD = 0.80
WARN_LOCAL_NEAR_RATIO_THRESHOLD = 0.50
LSD_BOX_MIN_EDGE_LENGTH_PX = 18.0
LSD_BOX_MIN_WIDTH_PX = 18
LSD_BOX_MIN_HEIGHT_PX = 14
LSD_BOX_MAX_WIDTH_PX = 2200
LSD_BOX_MAX_HEIGHT_PX = 700
LSD_BOX_LINE_COORD_TOL = 4
LSD_BOX_LINE_SPAN_TOL = 8
LSD_BOX_MATCH_PAD_PX = 8

GEMINI_BOUNDARY_SYSTEM_PROMPT = (
    "You are an HVAC drafting analyst. "
    "Analyze this mechanical floor-plan crop and identify ducts around only 14-inch diameter markers (14\"⌀ / 14\"Ø / 14\"ø / Ø14). "
    "For each valid marker, return duct boundary paths. "
    "Coordinates must be normalized to 0..1000 where [0,0] is top-left of the crop and [1000,1000] is bottom-right. "
    "Return strict JSON only with no markdown or prose. "
    "Do not guess uncertain duct boundaries."
)

GEMINI_BOUNDARY_USER_PROMPT = (
    "Find every duct associated with a 14-inch diameter marker. "
    "Return marker text, marker bbox/center, and 1-2 boundary polylines for each duct. "
    "Exclude title block, notes table, room boundaries, grid lines, and leader lines. "
    "If uncertain for a marker, skip it."
)

SIZE_MARKER_SYSTEM_PROMPT = (
    "You are reading an HVAC floor-plan crop. "
    "Find ONLY duct size text markers that indicate 14-inch diameter, such as 14\"⌀, 14 Ø, 14\"ø, or Ø14. "
    "Return strict JSON only with schema: "
    "{\"size_markers\":[{\"text\":\"14\\\"⌀\",\"bbox\":[x1,y1,x2,y2],\"confidence\":0.0}]}. "
    "bbox coordinates must be normalized 0..1000. "
    "Do not return notes-table text, title-block text, or unrelated labels. "
    "Skip uncertain detections instead of guessing."
)

SIZE_MARKER_USER_PROMPT = (
    "Detect only 14-inch diameter duct markers in this crop. "
    "Return only true marker text, not notes or legends, and output JSON only."
)

SIZE_MARKER_TEXT_RE = re.compile(r'(?:\b14\s*(?:"|”|″|in)?\s*[⌀Øø]|\b[⌀Øø]\s*14\b)', re.IGNORECASE)
SIZE_MARKER_NUM_TOKEN_RE = re.compile(r'^14(?:"|”|″|in)?$', re.IGNORECASE)
SIZE_MARKER_SYM_TOKEN_RE = re.compile(r'^[⌀Øø]$')
SIZE_MARKER_ZEROISH_TOKEN_RE = re.compile(r'^[0Oo]$')
SIZE_MARKER_QUOTE_TOKEN_RE = re.compile(r'^(?:"|”|″)$')


def _response_error_detail(response: requests.Response) -> str:
    try:
        payload = response.json()
    except Exception:
        payload = None

    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if message:
                return str(message)
            return json.dumps(error)
        message = payload.get("message")
        if message:
            return str(message)

    text = response.text.strip()
    return text[:500] if text else f"HTTP {response.status_code}"


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()

    def _normalize_parsed_json(parsed: Any) -> dict[str, Any]:
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            # Some models return top-level array despite schema instructions.
            return {"duct_segments": parsed}
        raise ValueError("Model response JSON must be an object or array.")

    try:
        return _normalize_parsed_json(json.loads(text))
    except Exception:
        pass

    fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
    if fenced:
        fenced_text = fenced.group(1).strip()
        try:
            return _normalize_parsed_json(json.loads(fenced_text))
        except Exception:
            pass

    obj_start = text.find("{")
    obj_end = text.rfind("}")
    if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
        try:
            return _normalize_parsed_json(json.loads(text[obj_start : obj_end + 1]))
        except Exception:
            pass

    arr_start = text.find("[")
    arr_end = text.rfind("]")
    if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
        return _normalize_parsed_json(json.loads(text[arr_start : arr_end + 1]))

    raise ValueError("No JSON object found in model response.")


def _clip(val: float, lo: float, hi: float) -> int:
    return int(max(lo, min(hi, round(val))))


def _line_length(line: list[int]) -> float:
    x1, y1, x2, y2 = line
    return float(math.hypot(x2 - x1, y2 - y1))


def _normalize_confidence(value: Any, default: float = 0.55) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = default
    return round(max(0.0, min(1.0, confidence)), 3)


def _canonical_line_key(line: list[int], bin_px: int = DEDUPE_BIN_PX) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = line
    if (x1 > x2) or (x1 == x2 and y1 > y2):
        x1, y1, x2, y2 = x2, y2, x1, y1
    return (
        int(round(x1 / bin_px)),
        int(round(y1 / bin_px)),
        int(round(x2 / bin_px)),
        int(round(y2 / bin_px)),
    )


def _tile_positions(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    step = max(64, tile - overlap)
    max_start = max(0, length - tile)
    starts = list(range(0, max_start + 1, step))
    if starts[-1] != max_start:
        starts.append(max_start)
    return sorted(set(starts))


def _extract_line_candidates_tiled(plan_crop_bgr: np.ndarray) -> list[dict[str, Any]]:
    crop_h, crop_w = plan_crop_bgr.shape[:2]
    gray = cv2.cvtColor(plan_crop_bgr, cv2.COLOR_BGR2GRAY)
    dark_mask = cv2.inRange(gray, 0, EVIDENCE_DARK_THRESHOLD)

    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    x_starts = _tile_positions(crop_w, TILE_SIZE_PX, TILE_OVERLAP_PX)
    y_starts = _tile_positions(crop_h, TILE_SIZE_PX, TILE_OVERLAP_PX)

    candidates: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, int]] = set()

    for ty in y_starts:
        for tx in x_starts:
            x2 = min(crop_w, tx + TILE_SIZE_PX)
            y2 = min(crop_h, ty + TILE_SIZE_PX)
            tile_gray = gray[ty:y2, tx:x2]
            detected = lsd.detect(tile_gray)
            if not detected:
                continue
            raw_lines = detected[0]
            if raw_lines is None:
                continue

            for row in raw_lines:
                vals = row[0] if hasattr(row, "__len__") and len(row) == 1 else row
                try:
                    x1f, y1f, x2f, y2f = [float(v) for v in vals]
                except Exception:
                    continue

                dx = abs(x2f - x1f)
                dy = abs(y2f - y1f)
                axis_ratio = max(dx, dy) / max(1.0, min(dx, dy))
                if axis_ratio < MIN_AXIS_RATIO:
                    continue

                if dx >= dy:
                    y_mid = (y1f + y2f) / 2.0
                    x_lo, x_hi = sorted((x1f, x2f))
                    line = [
                        _clip(x_lo + tx, 0, crop_w - 1),
                        _clip(y_mid + ty, 0, crop_h - 1),
                        _clip(x_hi + tx, 0, crop_w - 1),
                        _clip(y_mid + ty, 0, crop_h - 1),
                    ]
                    orientation = "h"
                else:
                    x_mid = (x1f + x2f) / 2.0
                    y_lo, y_hi = sorted((y1f, y2f))
                    line = [
                        _clip(x_mid + tx, 0, crop_w - 1),
                        _clip(y_lo + ty, 0, crop_h - 1),
                        _clip(x_mid + tx, 0, crop_w - 1),
                        _clip(y_hi + ty, 0, crop_h - 1),
                    ]
                    orientation = "v"

                length_px = _line_length(line)
                if length_px < MIN_CANDIDATE_LENGTH_PX:
                    continue

                key = _canonical_line_key(line, bin_px=8)
                if key in seen:
                    continue

                evidence = _line_evidence_score(dark_mask, line, radius=1)
                if evidence < MIN_CANDIDATE_EVIDENCE:
                    continue

                seen.add(key)
                candidates.append(
                    {
                        "line": line,
                        "orientation": orientation,
                        "length_px": round(length_px, 1),
                        "evidence": round(float(evidence), 3),
                    }
                )

    candidates.sort(key=lambda item: (item["evidence"], item["length_px"]), reverse=True)
    limited = candidates[:MAX_CANDIDATES_FOR_AI]

    renumbered: list[dict[str, Any]] = []
    for idx, cand in enumerate(limited):
        renumbered.append(
            {
                "id": f"C{idx + 1}",
                "line": cand["line"],
                "orientation": cand["orientation"],
                "length_px": cand["length_px"],
                "evidence": cand["evidence"],
            }
        )
    return renumbered


def _extract_lsd_raw_lines_tiled(plan_crop_bgr: np.ndarray) -> list[dict[str, Any]]:
    crop_h, crop_w = plan_crop_bgr.shape[:2]
    gray = cv2.cvtColor(plan_crop_bgr, cv2.COLOR_BGR2GRAY)

    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    x_starts = _tile_positions(crop_w, TILE_SIZE_PX, TILE_OVERLAP_PX)
    y_starts = _tile_positions(crop_h, TILE_SIZE_PX, TILE_OVERLAP_PX)

    raw_lines: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, int]] = set()

    for ty in y_starts:
        for tx in x_starts:
            x2 = min(crop_w, tx + TILE_SIZE_PX)
            y2 = min(crop_h, ty + TILE_SIZE_PX)
            tile_gray = gray[ty:y2, tx:x2]
            detected = lsd.detect(tile_gray)
            if not detected:
                continue
            lsd_rows = detected[0]
            if lsd_rows is None:
                continue

            for row in lsd_rows:
                vals = row[0] if hasattr(row, "__len__") and len(row) == 1 else row
                try:
                    x1f, y1f, x2f, y2f = [float(v) for v in vals]
                except Exception:
                    continue

                line = [
                    _clip(x1f + tx, 0, crop_w - 1),
                    _clip(y1f + ty, 0, crop_h - 1),
                    _clip(x2f + tx, 0, crop_w - 1),
                    _clip(y2f + ty, 0, crop_h - 1),
                ]
                length_px = _line_length(line)
                if length_px < 10.0:
                    continue

                key = _canonical_line_key(line, bin_px=4)
                if key in seen:
                    continue
                seen.add(key)

                dx = abs(line[2] - line[0])
                dy = abs(line[3] - line[1])
                orientation = "h" if dx >= dy else "v"

                raw_lines.append(
                    {
                        "id": f"R{len(raw_lines) + 1}",
                        "line": line,
                        "orientation": orientation,
                        "length_px": round(length_px, 1),
                    }
                )

    raw_lines.sort(key=lambda item: item["length_px"], reverse=True)
    for idx, item in enumerate(raw_lines):
        item["id"] = f"R{idx + 1}"
    return raw_lines


def _normalize_axis_line(raw_line: list[int], min_len: float = LSD_BOX_MIN_EDGE_LENGTH_PX) -> dict[str, Any] | None:
    if not isinstance(raw_line, list) or len(raw_line) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in raw_line]
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    if max(dx, dy) < min_len:
        return None
    axis_ratio = max(dx, dy) / max(1.0, min(dx, dy))
    if axis_ratio < 2.5:
        return None

    if dx >= dy:
        y = int(round((y1 + y2) / 2.0))
        xa = int(round(min(x1, x2)))
        xb = int(round(max(x1, x2)))
        if xb - xa < min_len:
            return None
        return {"orientation": "h", "x1": xa, "x2": xb, "y": y, "length_px": float(xb - xa)}

    x = int(round((x1 + x2) / 2.0))
    ya = int(round(min(y1, y2)))
    yb = int(round(max(y1, y2)))
    if yb - ya < min_len:
        return None
    return {"orientation": "v", "x": x, "y1": ya, "y2": yb, "length_px": float(yb - ya)}


def _merge_axis_segments(segments: list[dict[str, Any]], orientation: str) -> list[dict[str, Any]]:
    if not segments:
        return []

    merged: list[dict[str, Any]] = []
    if orientation == "h":
        segments_sorted = sorted(segments, key=lambda s: (s["y"], s["x1"]))
    else:
        segments_sorted = sorted(segments, key=lambda s: (s["x"], s["y1"]))

    for seg in segments_sorted:
        matched_idx = -1
        for idx in range(len(merged) - 1, -1, -1):
            cur = merged[idx]
            if orientation == "h":
                if abs(seg["y"] - cur["y"]) > LSD_BOX_LINE_COORD_TOL:
                    continue
                if seg["x1"] > cur["x2"] + LSD_BOX_LINE_SPAN_TOL or seg["x2"] < cur["x1"] - LSD_BOX_LINE_SPAN_TOL:
                    continue
                matched_idx = idx
                break
            else:
                if abs(seg["x"] - cur["x"]) > LSD_BOX_LINE_COORD_TOL:
                    continue
                if seg["y1"] > cur["y2"] + LSD_BOX_LINE_SPAN_TOL or seg["y2"] < cur["y1"] - LSD_BOX_LINE_SPAN_TOL:
                    continue
                matched_idx = idx
                break

        if matched_idx < 0:
            merged.append(dict(seg))
            continue

        cur = merged[matched_idx]
        if orientation == "h":
            cur["x1"] = min(cur["x1"], seg["x1"])
            cur["x2"] = max(cur["x2"], seg["x2"])
            cur["y"] = int(round((cur["y"] + seg["y"]) / 2.0))
            cur["length_px"] = float(cur["x2"] - cur["x1"])
        else:
            cur["y1"] = min(cur["y1"], seg["y1"])
            cur["y2"] = max(cur["y2"], seg["y2"])
            cur["x"] = int(round((cur["x"] + seg["x"]) / 2.0))
            cur["length_px"] = float(cur["y2"] - cur["y1"])

    return merged


def _extract_axis_lines_for_boxes(raw_lines: list[dict[str, Any]], width: int, height: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    axis_h: list[dict[str, Any]] = []
    axis_v: list[dict[str, Any]] = []
    for row in raw_lines:
        norm = _normalize_axis_line(row.get("line", []))
        if not norm:
            continue
        if norm["orientation"] == "h":
            norm["y"] = _clip(norm["y"], 0, max(0, height - 1))
            norm["x1"] = _clip(norm["x1"], 0, max(0, width - 1))
            norm["x2"] = _clip(norm["x2"], 0, max(0, width - 1))
            if norm["x2"] - norm["x1"] >= LSD_BOX_MIN_EDGE_LENGTH_PX:
                axis_h.append(norm)
        else:
            norm["x"] = _clip(norm["x"], 0, max(0, width - 1))
            norm["y1"] = _clip(norm["y1"], 0, max(0, height - 1))
            norm["y2"] = _clip(norm["y2"], 0, max(0, height - 1))
            if norm["y2"] - norm["y1"] >= LSD_BOX_MIN_EDGE_LENGTH_PX:
                axis_v.append(norm)

    h_lines = _merge_axis_segments(axis_h, "h")
    v_lines = _merge_axis_segments(axis_v, "v")
    h_lines = sorted(h_lines, key=lambda s: float(s["length_px"]), reverse=True)[:520]
    v_lines = sorted(v_lines, key=lambda s: float(s["length_px"]), reverse=True)[:420]
    return h_lines, v_lines


def _extract_lsd_box_candidates(raw_lines: list[dict[str, Any]], width: int, height: int) -> list[dict[str, Any]]:
    h_lines, v_lines = _extract_axis_lines_for_boxes(raw_lines, width, height)

    raw_boxes: list[dict[str, Any]] = []
    x_tol = 6
    y_tol = 6

    for i, left in enumerate(v_lines):
        for right in v_lines[i + 1 : i + 22]:
            x1 = int(left["x"])
            x2 = int(right["x"])
            if x2 <= x1:
                continue
            width_px = x2 - x1
            if width_px < LSD_BOX_MIN_WIDTH_PX or width_px > LSD_BOX_MAX_WIDTH_PX:
                continue

            overlap_y1 = int(max(left["y1"], right["y1"]))
            overlap_y2 = int(min(left["y2"], right["y2"]))
            if overlap_y2 - overlap_y1 < LSD_BOX_MIN_HEIGHT_PX:
                continue

            crossing = [
                h
                for h in h_lines
                if (overlap_y1 - y_tol) <= h["y"] <= (overlap_y2 + y_tol)
                and h["x1"] <= (x1 + x_tol)
                and h["x2"] >= (x2 - x_tol)
            ]
            if len(crossing) < 2:
                continue

            top = min(crossing, key=lambda h: abs(h["y"] - overlap_y1))
            bottom = min(crossing, key=lambda h: abs(h["y"] - overlap_y2))
            y1 = int(top["y"])
            y2 = int(bottom["y"])
            if y2 <= y1:
                continue
            height_px = y2 - y1
            if height_px < LSD_BOX_MIN_HEIGHT_PX or height_px > LSD_BOX_MAX_HEIGHT_PX:
                continue
            if abs(y1 - overlap_y1) > 40 or abs(y2 - overlap_y2) > 40:
                continue

            score = float(left["length_px"] + right["length_px"] + top["length_px"] + bottom["length_px"])
            raw_boxes.append(
                {
                    "id": f"B{len(raw_boxes) + 1}",
                    "bbox": [x1, y1, x2, y2],
                    "width_px": width_px,
                    "height_px": height_px,
                    "area": int(width_px * height_px),
                    "score": round(score, 2),
                }
            )

    raw_boxes.sort(key=lambda b: (float(b.get("score", 0.0)), float(b.get("area", 0))), reverse=True)
    deduped: list[dict[str, Any]] = []
    for box in raw_boxes:
        bbox = box["bbox"]
        is_dup = False
        for kept in deduped:
            iou = _bbox_iou(bbox, kept["bbox"])
            if iou >= 0.82:
                is_dup = True
                break
        if not is_dup:
            deduped.append(box)

    for idx, box in enumerate(deduped):
        box["id"] = f"B{idx + 1}"
    return deduped


def _find_local_lsd_box_for_marker(
    marker: dict[str, Any],
    h_lines: list[dict[str, Any]],
    v_lines: list[dict[str, Any]],
) -> list[int] | None:
    cx, cy = marker.get("center", [0, 0])

    crossing_h = [h for h in h_lines if h["x1"] <= (cx + 8) and h["x2"] >= (cx - 8)]
    top_candidates = [h for h in crossing_h if h["y"] < cy]
    bottom_candidates = [h for h in crossing_h if h["y"] > cy]
    top_candidates.sort(key=lambda h: abs(cy - h["y"]))
    bottom_candidates.sort(key=lambda h: abs(h["y"] - cy))
    if not top_candidates or not bottom_candidates:
        return None

    best_bbox: list[int] | None = None
    best_score = 1e18
    for top in top_candidates[:22]:
        for bottom in bottom_candidates[:22]:
            y1 = int(top["y"])
            y2 = int(bottom["y"])
            if y2 <= y1:
                continue
            height_px = y2 - y1
            if height_px < LSD_BOX_MIN_HEIGHT_PX or height_px > 380:
                continue

            left_candidates = [
                v
                for v in v_lines
                if v["x"] < cx
                and v["y1"] <= (y1 + 20)
                and v["y2"] >= (y2 - 20)
            ]
            right_candidates = [
                v
                for v in v_lines
                if v["x"] > cx
                and v["y1"] <= (y1 + 20)
                and v["y2"] >= (y2 - 20)
            ]
            if not left_candidates or not right_candidates:
                continue

            left = min(left_candidates, key=lambda v: abs(cx - v["x"]))
            right = min(right_candidates, key=lambda v: abs(v["x"] - cx))
            x1 = int(left["x"])
            x2 = int(right["x"])
            if x2 <= x1:
                continue
            width_px = x2 - x1
            if width_px < LSD_BOX_MIN_WIDTH_PX or width_px > LSD_BOX_MAX_WIDTH_PX:
                continue

            center_margin = min(cx - x1, x2 - cx, cy - y1, y2 - cy)
            if center_margin < -4:
                continue

            area = width_px * height_px
            score = area + (abs(cy - y1) + abs(y2 - cy)) * 10 + abs(cx - x1 - width_px / 2.0) * 3
            if score < best_score:
                best_score = score
                best_bbox = [x1, y1, x2, y2]
    return best_bbox


def _augment_boxes_with_marker_local_candidates(
    box_candidates: list[dict[str, Any]],
    markers: list[dict[str, Any]],
    raw_lines: list[dict[str, Any]],
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    h_lines, v_lines = _extract_axis_lines_for_boxes(raw_lines, width, height)
    merged = [dict(item) for item in box_candidates]
    for marker in markers:
        local_bbox = _find_local_lsd_box_for_marker(marker, h_lines, v_lines)
        if not local_bbox:
            continue
        duplicate = False
        for existing in merged:
            if _bbox_iou(local_bbox, existing.get("bbox", [0, 0, 0, 0])) >= 0.8:
                duplicate = True
                break
        if duplicate:
            continue
        x1, y1, x2, y2 = local_bbox
        merged.append(
            {
                "id": f"B{len(merged) + 1}",
                "bbox": local_bbox,
                "width_px": int(x2 - x1),
                "height_px": int(y2 - y1),
                "area": int((x2 - x1) * (y2 - y1)),
                "score": 0.0,
                "source": "marker_local",
                "marker_id_hint": marker.get("id", ""),
            }
        )
    for idx, box in enumerate(merged):
        box["id"] = f"B{idx + 1}"
    return merged


def _point_in_bbox(point: list[int], bbox: list[int], pad: int = 0) -> bool:
    if len(point) != 2 or len(bbox) != 4:
        return False
    x, y = point
    x1, y1, x2, y2 = bbox
    return (x1 - pad) <= x <= (x2 + pad) and (y1 - pad) <= y <= (y2 + pad)


def _match_markers_to_lsd_boxes(
    markers: list[dict[str, Any]],
    box_candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    issues: list[str] = []
    matches: list[dict[str, Any]] = []

    for marker in markers:
        marker_id = str(marker.get("id", ""))
        marker_bbox = marker.get("bbox", [0, 0, 0, 0])
        marker_center = marker.get("center", [0, 0])
        candidate_rows: list[dict[str, Any]] = []
        for box in box_candidates:
            bbox = box.get("bbox", [0, 0, 0, 0])
            center_inside = _point_in_bbox(marker_center, bbox, pad=LSD_BOX_MATCH_PAD_PX)
            overlap = _bbox_iou(marker_bbox, bbox)
            if not center_inside and overlap <= 0.0:
                continue
            area = max(1, int((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])))
            candidate_rows.append(
                {
                    "box": box,
                    "area": area,
                    "overlap": overlap,
                    "center_inside": center_inside,
                }
            )

        if not candidate_rows:
            issues.append(f"{marker_id}: no matching LSD box for marker '{marker.get('text', '')}'.")
            continue

        candidate_rows.sort(key=lambda row: (0 if row["center_inside"] else 1, row["area"], -row["overlap"]))
        chosen = candidate_rows[0]["box"]
        matches.append(
            {
                "id": f"MB{len(matches) + 1}",
                "marker_id": marker_id,
                "marker_text": marker.get("text", ""),
                "marker_bbox": marker_bbox,
                "marker_center": marker_center,
                "marker_confidence": marker.get("confidence", 0.0),
                "box_id": chosen.get("id", ""),
                "bbox": chosen.get("bbox", [0, 0, 0, 0]),
                "score": chosen.get("score", 0.0),
            }
        )
    return matches, issues


def _build_segments_from_marker_boxes(
    marker_boxes: list[dict[str, Any]],
    width: int,
    height: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    duct_segments: list[dict[str, Any]] = []
    crop_line_segments: list[dict[str, Any]] = []
    line_index = 0

    for idx, item in enumerate(marker_boxes):
        x1, y1, x2, y2 = item.get("bbox", [0, 0, 0, 0])
        duct_id = f"D{idx + 1}"
        marker_id = str(item.get("marker_id", ""))

        corners = [[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]]
        path = []
        for px, py in corners:
            path.append(
                [
                    _clip((px / max(1, width - 1)) * 1000.0, 0, 1000),
                    _clip((py / max(1, height - 1)) * 1000.0, 0, 1000),
                ]
            )

        duct_segments.append(
            {
                "id": duct_id,
                "type": "marker_box",
                "confidence": _normalize_confidence(item.get("marker_confidence", 0.9), default=0.9),
                "path": path,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "point_count": len(path),
                "segment_count": 4,
                "source_candidate_id": item.get("box_id", ""),
                "source_marker_ids": [marker_id] if marker_id else [],
                "seed_distance_px": None,
                "graph_depth": 0,
                "selection_reason": "lsd_marker_box_match",
            }
        )

        edges = [
            [x1, y1, x2, y1],
            [x2, y1, x2, y2],
            [x2, y2, x1, y2],
            [x1, y2, x1, y1],
        ]
        for edge_idx, edge in enumerate(edges):
            line_index += 1
            crop_line_segments.append(
                {
                    "id": f"L{line_index}",
                    "duct_id": duct_id,
                    "duct_type": "marker_box",
                    "confidence": _normalize_confidence(item.get("marker_confidence", 0.9), default=0.9),
                    "line": [int(edge[0]), int(edge[1]), int(edge[2]), int(edge[3])],
                    "length_px": round(_line_length([int(edge[0]), int(edge[1]), int(edge[2]), int(edge[3])]), 1),
                    "source_candidate_id": item.get("box_id", ""),
                    "source_marker_ids": [marker_id] if marker_id else [],
                    "seed_distance_px": None,
                    "graph_depth": 0,
                    "marker_id": marker_id,
                    "boundary_id": f"box_edge_{edge_idx + 1}",
                    "segment_index": edge_idx,
                    "selection_reason": "lsd_marker_box_match",
                }
            )

    return duct_segments, crop_line_segments


def _normalize_marker_bbox(raw_bbox: Any, width: int, height: int) -> list[int] | None:
    if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in raw_bbox]
    except (TypeError, ValueError):
        return None

    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1000.0:
        x1 = (x1 / 1000.0) * (width - 1)
        x2 = (x2 / 1000.0) * (width - 1)
        y1 = (y1 / 1000.0) * (height - 1)
        y2 = (y2 / 1000.0) * (height - 1)

    x1i = _clip(min(x1, x2), 0, width - 1)
    y1i = _clip(min(y1, y2), 0, height - 1)
    x2i = _clip(max(x1, x2), 0, width - 1)
    y2i = _clip(max(y1, y2), 0, height - 1)
    if x2i <= x1i or y2i <= y1i:
        return None
    return [x1i, y1i, x2i, y2i]


def _normalize_size_markers(payload: dict[str, Any], width: int, height: int) -> tuple[list[dict[str, Any]], list[str]]:
    issues: list[str] = []
    raw = payload.get("size_markers", payload.get("markers", [])) if isinstance(payload, dict) else []
    if not isinstance(raw, list):
        return [], ["size_markers missing or invalid."]

    markers: list[dict[str, Any]] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            issues.append(f"size_markers[{idx}] is not an object.")
            continue
        text = str(item.get("text", "")).strip()
        if not text or not SIZE_MARKER_TEXT_RE.search(text):
            continue
        bbox = _normalize_marker_bbox(item.get("bbox"), width, height)
        if not bbox:
            issues.append(f"size_markers[{idx}] invalid bbox.")
            continue
        x1, y1, x2, y2 = bbox
        bbox_diag_px = round(math.hypot(x2 - x1, y2 - y1), 2)
        markers.append(
            {
                "id": f"M{len(markers) + 1}",
                "text": text,
                "bbox": bbox,
                "center": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
                "bbox_diag_px": bbox_diag_px,
                "local_radius_px": round(max(SIZE_MARKER_MIN_RADIUS_PX, SIZE_MARKER_RADIUS_DIAG_SCALE * bbox_diag_px), 2),
                "confidence": _normalize_confidence(item.get("confidence", 0.7), default=0.7),
            }
        )
    markers = _dedupe_size_markers(markers)
    for idx, marker in enumerate(markers):
        marker["id"] = f"M{idx + 1}"
    return markers, issues


def _bbox_iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = float(iw * ih)
    if inter <= 0:
        return 0.0
    a_area = float(max(1, (ax2 - ax1) * (ay2 - ay1)))
    b_area = float(max(1, (bx2 - bx1) * (by2 - by1)))
    return inter / max(1.0, a_area + b_area - inter)


def _dedupe_size_markers(markers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not markers:
        return []
    ranked = sorted(markers, key=lambda m: float(m.get("confidence", 0.0)), reverse=True)
    kept: list[dict[str, Any]] = []
    for marker in ranked:
        mx, my = marker["center"]
        duplicate = False
        for prev in kept:
            px, py = prev["center"]
            center_dist = math.hypot(mx - px, my - py)
            iou = _bbox_iou(marker["bbox"], prev["bbox"])
            if center_dist <= MARKER_DEDUPE_CENTER_PX and iou >= MARKER_DEDUPE_IOU_THRESHOLD:
                duplicate = True
                break
            if center_dist <= 5.0:
                duplicate = True
                break
        if not duplicate:
            kept.append(marker)
    return kept


def _vision_vertices_to_bbox(raw_vertices: Any, width: int, height: int) -> list[int] | None:
    if not isinstance(raw_vertices, list) or not raw_vertices:
        return None

    xs: list[float] = []
    ys: list[float] = []
    for vertex in raw_vertices:
        if not isinstance(vertex, dict):
            continue
        try:
            x = float(vertex.get("x", 0.0))
            y = float(vertex.get("y", 0.0))
        except (TypeError, ValueError):
            continue
        xs.append(x)
        ys.append(y)

    if not xs or not ys:
        return None

    x1 = _clip(min(xs), 0, width - 1)
    y1 = _clip(min(ys), 0, height - 1)
    x2 = _clip(max(xs), 0, width - 1)
    y2 = _clip(max(ys), 0, height - 1)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _bbox_to_norm_1000(bbox: list[int], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = bbox
    den_w = max(1, width - 1)
    den_h = max(1, height - 1)
    return [
        round((x1 / den_w) * 1000.0, 3),
        round((y1 / den_h) * 1000.0, 3),
        round((x2 / den_w) * 1000.0, 3),
        round((y2 / den_h) * 1000.0, 3),
    ]


def _bbox_union(a: list[int], b: list[int]) -> list[int]:
    return [
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3]),
    ]


def _bbox_y_overlap_ratio(a: list[int], b: list[int]) -> float:
    ay1, ay2 = a[1], a[3]
    by1, by2 = b[1], b[3]
    overlap = max(0, min(ay2, by2) - max(ay1, by1))
    ah = max(1, ay2 - ay1)
    bh = max(1, by2 - by1)
    return float(overlap) / float(max(1, min(ah, bh)))


def _normalize_marker_token_text(text: str) -> str:
    compact = str(text).strip()
    compact = compact.replace("”", '"').replace("″", '"')
    compact = compact.replace("’", "'").replace("‘", "'")
    compact = re.sub(r"\s+", "", compact)
    return compact


def _extract_vision_word_tokens(raw_response: dict[str, Any], width: int, height: int) -> list[dict[str, Any]]:
    responses = raw_response.get("responses", []) if isinstance(raw_response, dict) else []
    first = responses[0] if isinstance(responses, list) and responses else {}
    full = first.get("fullTextAnnotation", {}) if isinstance(first, dict) else {}
    pages = full.get("pages", []) if isinstance(full, dict) else []

    tokens: list[dict[str, Any]] = []
    for p_idx, page in enumerate(pages):
        if not isinstance(page, dict):
            continue
        blocks = page.get("blocks", [])
        for b_idx, block in enumerate(blocks):
            if not isinstance(block, dict):
                continue
            paragraphs = block.get("paragraphs", [])
            for para_idx, para in enumerate(paragraphs):
                if not isinstance(para, dict):
                    continue
                words = para.get("words", [])
                for w_idx, word in enumerate(words):
                    if not isinstance(word, dict):
                        continue
                    symbols = word.get("symbols", [])
                    if not isinstance(symbols, list):
                        symbols = []
                    text = "".join(str(sym.get("text", "")) for sym in symbols).strip()
                    if not text:
                        continue
                    bbox = _vision_vertices_to_bbox(word.get("boundingBox", {}).get("vertices", []), width, height)
                    if not bbox:
                        continue
                    x1, y1, x2, y2 = bbox
                    tokens.append(
                        {
                            "id": f"T{len(tokens) + 1}",
                            "text": text,
                            "text_compact": _normalize_marker_token_text(text),
                            "bbox": bbox,
                            "center": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
                            "width_px": int(x2 - x1),
                            "height_px": int(y2 - y1),
                            "confidence": _normalize_confidence(word.get("confidence", 0.7), default=0.7),
                            "line_hint": f"{p_idx}:{b_idx}:{para_idx}",
                            "word_index": int(w_idx),
                        }
                    )

    if tokens:
        return tokens

    text_annotations = first.get("textAnnotations", []) if isinstance(first, dict) else []
    if isinstance(text_annotations, list):
        for ann_idx, ann in enumerate(text_annotations[1:], start=1):
            if not isinstance(ann, dict):
                continue
            text = str(ann.get("description", "")).strip()
            if not text:
                continue
            bbox = _vision_vertices_to_bbox(ann.get("boundingPoly", {}).get("vertices", []), width, height)
            if not bbox:
                continue
            x1, y1, x2, y2 = bbox
            tokens.append(
                {
                    "id": f"T{len(tokens) + 1}",
                    "text": text,
                    "text_compact": _normalize_marker_token_text(text),
                    "bbox": bbox,
                    "center": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
                    "width_px": int(x2 - x1),
                    "height_px": int(y2 - y1),
                    "confidence": 0.7,
                    "line_hint": "fallback",
                    "word_index": int(ann_idx),
                }
            )
    return tokens


def _extract_vision_size_markers(tokens: list[dict[str, Any]], width: int, height: int) -> tuple[dict[str, Any], list[str]]:
    issues: list[str] = []
    token_debug: list[dict[str, Any]] = []
    marker_rows: list[dict[str, Any]] = []

    number_tokens: list[dict[str, Any]] = []
    symbol_tokens: list[dict[str, Any]] = []
    quote_tokens: list[dict[str, Any]] = []
    direct_token_ids: set[str] = set()
    paired_token_ids: set[str] = set()

    for token in tokens:
        compact = str(token.get("text_compact", ""))
        bbox = token.get("bbox", [0, 0, 0, 0])
        is_direct = bool(SIZE_MARKER_TEXT_RE.search(compact))
        is_number = bool(SIZE_MARKER_NUM_TOKEN_RE.fullmatch(compact))
        is_symbol = bool(SIZE_MARKER_SYM_TOKEN_RE.fullmatch(compact))
        is_zeroish = bool(SIZE_MARKER_ZEROISH_TOKEN_RE.fullmatch(compact))
        is_quote = bool(SIZE_MARKER_QUOTE_TOKEN_RE.fullmatch(compact))

        if is_direct:
            direct_token_ids.add(str(token.get("id", "")))
            marker_rows.append(
                {
                    "text": str(token.get("text", "14\"Ø")),
                    "bbox": bbox,
                    "confidence": _normalize_confidence(token.get("confidence", 0.7), default=0.7),
                    "source": "direct",
                    "token_ids": [str(token.get("id", ""))],
                }
            )
        if is_number:
            number_tokens.append(token)
        if is_symbol or is_zeroish:
            symbol_tokens.append(token)
        if is_quote:
            quote_tokens.append(token)

        token_debug.append(
            {
                "id": token.get("id", ""),
                "text": token.get("text", ""),
                "text_compact": compact,
                "bbox": bbox,
                "center": token.get("center", [0, 0]),
                "confidence": token.get("confidence", 0.0),
                "is_direct_match": is_direct,
                "is_number_token": is_number,
                "is_symbol_token": is_symbol,
                "is_zeroish_token": is_zeroish,
                "is_quote_token": is_quote,
            }
        )

    for number in number_tokens:
        best_sym: dict[str, Any] | None = None
        best_score = 1e9
        n_bbox = number.get("bbox", [0, 0, 0, 0])
        n_center = number.get("center", [0, 0])
        n_h = max(1, int(number.get("height_px", 0)))
        n_w = max(1, int(number.get("width_px", 0)))
        for symbol in symbol_tokens:
            s_bbox = symbol.get("bbox", [0, 0, 0, 0])
            s_center = symbol.get("center", [0, 0])
            s_h = max(1, int(symbol.get("height_px", 0)))
            s_w = max(1, int(symbol.get("width_px", 0)))

            y_delta = abs(float(n_center[1]) - float(s_center[1]))
            if y_delta > max(14.0, 0.65 * float(max(n_h, s_h))):
                continue
            if _bbox_y_overlap_ratio(n_bbox, s_bbox) < 0.40:
                continue

            if s_bbox[0] >= n_bbox[2]:
                gap = float(s_bbox[0] - n_bbox[2])
            elif n_bbox[0] >= s_bbox[2]:
                gap = float(n_bbox[0] - s_bbox[2])
            else:
                gap = 0.0

            max_gap = max(26.0, 2.2 * float(max(n_h, s_h)))
            if gap > max_gap:
                continue

            x_delta = abs(float(n_center[0]) - float(s_center[0]))
            if x_delta > max(90.0, 6.0 * float(max(n_h, s_h))):
                continue

            score = (gap * 3.0) + (y_delta * 2.0) + (x_delta * 0.03) - (
                float(number.get("confidence", 0.0)) + float(symbol.get("confidence", 0.0))
            )
            if score < best_score:
                best_score = score
                best_sym = symbol

        if not best_sym:
            continue

        n_id = str(number.get("id", ""))
        s_id = str(best_sym.get("id", ""))
        paired_token_ids.add(n_id)
        paired_token_ids.add(s_id)

        marker_rows.append(
            {
                "text": '14"Ø',
                "bbox": _bbox_union(number.get("bbox", [0, 0, 0, 0]), best_sym.get("bbox", [0, 0, 0, 0])),
                "confidence": _normalize_confidence(
                    (float(number.get("confidence", 0.7)) + float(best_sym.get("confidence", 0.7))) / 2.0,
                    default=0.7,
                ),
                "source": "paired",
                "token_ids": [n_id, s_id],
            }
        )

    for number in number_tokens:
        n_compact = str(number.get("text_compact", ""))
        if '"' in n_compact:
            continue

        n_bbox = number.get("bbox", [0, 0, 0, 0])
        n_center = number.get("center", [0, 0])
        n_h = max(1, int(number.get("height_px", 0)))
        best_quote: dict[str, Any] | None = None
        best_gap = 1e9

        for quote in quote_tokens:
            q_bbox = quote.get("bbox", [0, 0, 0, 0])
            q_center = quote.get("center", [0, 0])
            y_delta = abs(float(n_center[1]) - float(q_center[1]))
            if y_delta > max(12.0, 0.6 * float(n_h)):
                continue
            if _bbox_y_overlap_ratio(n_bbox, q_bbox) < 0.40:
                continue

            if q_bbox[0] >= n_bbox[2]:
                gap = float(q_bbox[0] - n_bbox[2])
            elif n_bbox[0] >= q_bbox[2]:
                gap = float(n_bbox[0] - q_bbox[2])
            else:
                gap = 0.0

            if gap > 14.0:
                continue
            if gap < best_gap:
                best_gap = gap
                best_quote = quote

        if not best_quote:
            continue

        n_id = str(number.get("id", ""))
        q_id = str(best_quote.get("id", ""))
        paired_token_ids.add(n_id)
        paired_token_ids.add(q_id)

        marker_rows.append(
            {
                "text": '14"Ø',
                "bbox": _bbox_union(n_bbox, best_quote.get("bbox", [0, 0, 0, 0])),
                "confidence": _normalize_confidence(
                    (float(number.get("confidence", 0.7)) + float(best_quote.get("confidence", 0.7))) / 2.0,
                    default=0.7,
                ),
                "source": "paired_quote",
                "token_ids": [n_id, q_id],
            }
        )

    internal_markers: list[dict[str, Any]] = []
    for row in marker_rows:
        bbox = row.get("bbox", [])
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        if x2 <= x1 or y2 <= y1:
            continue
        bbox_diag_px = round(math.hypot(x2 - x1, y2 - y1), 2)
        internal_markers.append(
            {
                "id": f"M{len(internal_markers) + 1}",
                "text": str(row.get("text", '14"Ø')).strip() or '14"Ø',
                "bbox": [x1, y1, x2, y2],
                "center": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
                "bbox_diag_px": bbox_diag_px,
                "local_radius_px": round(max(SIZE_MARKER_MIN_RADIUS_PX, SIZE_MARKER_RADIUS_DIAG_SCALE * bbox_diag_px), 2),
                "confidence": _normalize_confidence(row.get("confidence", 0.7), default=0.7),
                "source": row.get("source", ""),
                "token_ids": row.get("token_ids", []),
            }
        )

    internal_markers = _dedupe_size_markers(internal_markers)
    for idx, marker in enumerate(internal_markers):
        marker["id"] = f"M{idx + 1}"

    size_markers_payload: list[dict[str, Any]] = []
    for marker in internal_markers:
        size_markers_payload.append(
            {
                "text": marker["text"],
                "bbox": _bbox_to_norm_1000(marker["bbox"], width, height),
                "confidence": marker["confidence"],
            }
        )

    paired_or_direct = direct_token_ids | paired_token_ids
    for token_row in token_debug:
        token_row["used_for_marker"] = str(token_row.get("id", "")) in paired_or_direct

    if not tokens:
        issues.append("Vision OCR returned no word tokens in the plan crop.")
    if tokens and not size_markers_payload:
        issues.append("Vision OCR found text but no valid 14-inch markers (14\"⌀ / 14\"Ø / 14\"ø / Ø14).")

    return {
        "size_markers": size_markers_payload,
        "vision_tokens": token_debug,
        "vision_marker_candidates_px": internal_markers,
    }, issues


def _normalize_norm_point(raw_point: Any, width: int, height: int) -> list[int] | None:
    if not isinstance(raw_point, list) or len(raw_point) != 2:
        return None
    try:
        x_raw = float(raw_point[0])
        y_raw = float(raw_point[1])
    except (TypeError, ValueError):
        return None
    if not (0.0 <= x_raw <= 1000.0 and 0.0 <= y_raw <= 1000.0):
        return None
    return [
        _clip((x_raw / 1000.0) * (width - 1), 0, width - 1),
        _clip((y_raw / 1000.0) * (height - 1), 0, height - 1),
    ]


def _normalize_marker_center(raw_center: Any, bbox: list[int], width: int, height: int) -> list[int]:
    center = _normalize_norm_point(raw_center, width, height)
    if center:
        return center
    x1, y1, x2, y2 = bbox
    return [int((x1 + x2) / 2), int((y1 + y2) / 2)]


def _normalize_boundary_path(raw_path: Any, width: int, height: int) -> list[list[int]] | None:
    if not isinstance(raw_path, list) or len(raw_path) < 2:
        return None
    points_px: list[list[int]] = []
    for point in raw_path:
        point_px = _normalize_norm_point(point, width, height)
        if not point_px:
            return None
        if not points_px or point_px != points_px[-1]:
            points_px.append(point_px)
    if len(points_px) < 2:
        return None
    total_len = 0.0
    for idx in range(len(points_px) - 1):
        x1, y1 = points_px[idx]
        x2, y2 = points_px[idx + 1]
        total_len += float(math.hypot(x2 - x1, y2 - y1))
    if total_len <= 1.0:
        return None
    return points_px


def _normalize_gemini_duct_boundaries(payload: dict[str, Any], width: int, height: int) -> tuple[list[dict[str, Any]], list[str]]:
    issues: list[str] = []
    raw_ducts = payload.get("ducts", []) if isinstance(payload, dict) else []
    if not isinstance(raw_ducts, list):
        return [], ["ducts missing or invalid."]

    ducts: list[dict[str, Any]] = []
    for idx, raw_duct in enumerate(raw_ducts):
        if not isinstance(raw_duct, dict):
            issues.append(f"ducts[{idx}] is not an object.")
            continue

        marker_raw = raw_duct.get("marker")
        if not isinstance(marker_raw, dict):
            issues.append(f"ducts[{idx}] marker missing or invalid.")
            continue

        marker_text = str(marker_raw.get("text", "")).strip()
        if not marker_text or not SIZE_MARKER_TEXT_RE.search(marker_text):
            continue

        marker_bbox = _normalize_marker_bbox(marker_raw.get("bbox"), width, height)
        if not marker_bbox:
            issues.append(f"ducts[{idx}] marker bbox invalid.")
            continue
        marker_center = _normalize_marker_center(marker_raw.get("center"), marker_bbox, width, height)
        marker_conf = _normalize_confidence(marker_raw.get("confidence", 0.7), default=0.7)

        raw_boundaries = raw_duct.get("boundaries")
        if not isinstance(raw_boundaries, list) or not (1 <= len(raw_boundaries) <= 2):
            issues.append(f"ducts[{idx}] must include 1 or 2 boundaries.")
            continue

        boundaries: list[dict[str, Any]] = []
        boundary_parse_failed = False
        for bidx, raw_boundary in enumerate(raw_boundaries):
            if not isinstance(raw_boundary, dict):
                issues.append(f"ducts[{idx}].boundaries[{bidx}] is not an object.")
                boundary_parse_failed = True
                break
            raw_path = raw_boundary.get("path")
            path_px = _normalize_boundary_path(raw_path, width, height)
            if not path_px:
                issues.append(f"ducts[{idx}].boundaries[{bidx}] path invalid.")
                boundary_parse_failed = True
                break

            path_norm: list[list[int]] = []
            for point in raw_path:
                path_norm.append([_clip(float(point[0]), 0, 1000), _clip(float(point[1]), 0, 1000)])

            boundary_id = str(raw_boundary.get("id", f"B{bidx + 1}")).strip() or f"B{bidx + 1}"
            boundaries.append(
                {
                    "id": boundary_id,
                    "confidence": _normalize_confidence(raw_boundary.get("confidence", 0.7), default=0.7),
                    "path": path_norm,
                    "path_px": path_px,
                }
            )
        if boundary_parse_failed or not boundaries:
            issues.append(f"ducts[{idx}] skipped due to malformed boundaries.")
            continue

        marker_bbox_diag = round(math.hypot(marker_bbox[2] - marker_bbox[0], marker_bbox[3] - marker_bbox[1]), 2)
        duct_id = str(raw_duct.get("id", f"D{len(ducts) + 1}")).strip() or f"D{len(ducts) + 1}"
        ducts.append(
            {
                "id": duct_id,
                "type": "other",
                "confidence": _normalize_confidence(raw_duct.get("confidence", 0.7), default=0.7),
                "marker": {
                    "id": f"M{len(ducts) + 1}",
                    "text": marker_text,
                    "bbox": marker_bbox,
                    "center": marker_center,
                    "bbox_diag_px": marker_bbox_diag,
                    "local_radius_px": round(max(70.0, marker_bbox_diag * 1.2), 2),
                    "confidence": marker_conf,
                },
                "boundaries": boundaries,
                "valid_boundary_count": len(boundaries),
            }
        )

    return ducts, issues


def _point_to_segment_distance(px: float, py: float, line: list[int]) -> float:
    x1, y1, x2, y2 = [float(v) for v in line]
    dx = x2 - x1
    dy = y2 - y1
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / ((dx * dx) + (dy * dy))
    t = max(0.0, min(1.0, t))
    cx = x1 + (t * dx)
    cy = y1 + (t * dy)
    return math.hypot(px - cx, py - cy)


def _candidate_rows_for_marker_picker(candidates: list[dict[str, Any]], width: int, height: int) -> str:
    rows: list[dict[str, Any]] = []
    for cand in candidates:
        x1, y1, x2, y2 = cand["line"]
        rows.append(
            {
                "id": cand["id"],
                "line": [
                    _clip((x1 / max(1, width - 1)) * 1000.0, 0, 1000),
                    _clip((y1 / max(1, height - 1)) * 1000.0, 0, 1000),
                    _clip((x2 / max(1, width - 1)) * 1000.0, 0, 1000),
                    _clip((y2 / max(1, height - 1)) * 1000.0, 0, 1000),
                ],
                "orientation": cand.get("orientation", "h"),
                "length_px": cand.get("length_px", 0.0),
                "evidence": cand.get("evidence", 0.0),
                "distance_to_marker_px": round(float(cand.get("distance_to_marker_px", 0.0)), 2),
            }
        )
    return json.dumps(rows, separators=(",", ":"))


def _normalize_marker_picker_selection(
    payload: dict[str, Any],
    local_candidates: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    issues: list[str] = []
    raw = payload.get("selected_lines", payload.get("selected_candidates", payload.get("selections", [])))
    if isinstance(raw, dict):
        raw = raw.get("items", [])
    if not isinstance(raw, list):
        return [], ["selected_lines missing or invalid."]

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for idx, item in enumerate(raw):
        line_id = ""
        confidence_raw: Any = 0.7
        if isinstance(item, str):
            line_id = item.strip()
        elif isinstance(item, dict):
            line_id = str(item.get("id") or item.get("candidate_id") or "").strip()
            confidence_raw = item.get("confidence", item.get("score", 0.7))
        else:
            issues.append(f"selected_lines[{idx}] invalid item type.")
            continue

        if not line_id:
            issues.append(f"selected_lines[{idx}] missing id.")
            continue
        if line_id in seen:
            continue
        cand = local_candidates.get(line_id)
        if not cand:
            issues.append(f"selected_lines[{idx}] id '{line_id}' is outside local pool.")
            continue
        seen.add(line_id)
        selected.append(
            {
                "candidate_id": line_id,
                "line": cand["line"],
                "length_px": cand["length_px"],
                "evidence": cand.get("evidence", 0.0),
                "duct_type": "other",
                "confidence": _normalize_confidence(confidence_raw, default=0.7),
                "seed_distance_px": float(cand.get("distance_to_marker_px", 0.0)),
            }
        )
    return selected, issues


def _select_marker_candidates_with_gemini(
    image_b64: str,
    width: int,
    height: int,
    candidates: list[dict[str, Any]],
    markers: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    list[str],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    issues: list[str] = []
    if not markers or not candidates:
        return (
            [],
            {"marker_count": len(markers), "seed_count": 0, "seed_added_count": 0, "per_marker": []},
            {"selection_mode": "gemini_per_marker_local", "calls": 0, "success_count": 0, "failed_count": 0, "far_rejected_count": 0, "parse_issues": []},
            issues,
            [],
            [],
        )

    selected_by_id: dict[str, dict[str, Any]] = {}
    per_marker_meta: list[dict[str, Any]] = []
    picker_raw_runs: list[dict[str, Any]] = []
    picker_parsed_runs: list[dict[str, Any]] = []
    parse_issues: list[str] = []
    success_count = 0
    failed_count = 0
    marker_radius_by_id = {str(m["id"]): float(m.get("local_radius_px", SIZE_MARKER_MIN_RADIUS_PX)) for m in markers}

    for marker in markers:
        marker_id = str(marker["id"])
        mx, my = marker["center"]
        radius = float(marker.get("local_radius_px", SIZE_MARKER_MIN_RADIUS_PX))

        nearby: list[dict[str, Any]] = []
        for cand in candidates:
            length_px = float(cand.get("length_px", 0.0))
            if length_px < SIZE_MARKER_MIN_LINE_LEN_PX:
                continue
            dist = _point_to_segment_distance(float(mx), float(my), cand["line"])
            if dist > radius:
                continue
            nearby.append(
                {
                    **cand,
                    "distance_to_marker_px": round(float(dist), 2),
                }
            )

        nearby.sort(key=lambda c: (float(c["distance_to_marker_px"]), -float(c.get("evidence", 0.0)), -float(c.get("length_px", 0.0))))
        local_pool = nearby[:MARKER_PICK_LOCAL_MAX_CANDIDATES]
        local_by_id = {str(c["id"]): c for c in local_pool}

        picked_ids: list[str] = []
        marker_issues: list[str] = []
        if not local_pool:
            marker_issues.append("No local candidates near this marker.")
            issues.append(f"{marker_id}: no local candidate lines within marker radius.")
            failed_count += 1
            picker_raw_runs.append(
                {
                    "marker_id": marker_id,
                    "ok": False,
                    "http_status": None,
                    "error": "No local candidates near this marker.",
                    "raw_response": {},
                }
            )
            picker_parsed_runs.append(
                {
                    "marker_id": marker_id,
                    "ok": False,
                    "parsed_json": {},
                    "selected_candidate_ids": [],
                    "issues": marker_issues,
                }
            )
        else:
            candidate_rows_json = _candidate_rows_for_marker_picker(local_pool, width, height)
            picker_result = _call_gemini_marker_line_picker(image_b64, width, height, marker, candidate_rows_json)
            picker_raw_runs.append(
                {
                    "marker_id": marker_id,
                    "ok": picker_result.get("ok", False),
                    "http_status": picker_result.get("http_status"),
                    "error": picker_result.get("error", ""),
                    "raw_response": picker_result.get("raw_response", {}),
                }
            )

            if not picker_result.get("ok", False):
                failed_count += 1
                err = str(picker_result.get("error", "")).strip() or "Marker picker call failed."
                issues.append(f"{marker_id}: {err}")
                marker_issues.append(err)
                picker_parsed_runs.append(
                    {
                        "marker_id": marker_id,
                        "ok": False,
                        "parsed_json": picker_result.get("parsed_json", {}),
                        "selected_candidate_ids": [],
                        "issues": marker_issues,
                    }
                )
            else:
                success_count += 1
                picked, norm_issues = _normalize_marker_picker_selection(picker_result.get("parsed_json", {}), local_by_id)
                marker_issues.extend(norm_issues)
                if norm_issues:
                    parse_issues.extend([f"{marker_id}: {msg}" for msg in norm_issues])

                for item in picked:
                    cid = str(item["candidate_id"])
                    picked_ids.append(cid)
                    existing = selected_by_id.get(cid)
                    if not existing:
                        selected_by_id[cid] = {
                            **item,
                            "source_marker_ids": [marker_id],
                            "seed_distance_px": item["seed_distance_px"],
                            "graph_depth": 0,
                            "selection_reason": "gemini_marker_pick",
                            "marker_hit_count": 1,
                        }
                        continue
                    existing_markers = set(str(v) for v in existing.get("source_marker_ids", []))
                    existing_markers.add(marker_id)
                    existing["source_marker_ids"] = sorted(existing_markers)
                    existing["seed_distance_px"] = round(min(float(existing.get("seed_distance_px", 1e9)), float(item.get("seed_distance_px", 0.0))), 2)
                    existing["confidence"] = round(max(float(existing.get("confidence", 0.7)), float(item.get("confidence", 0.7))), 3)
                    existing["marker_hit_count"] = int(existing.get("marker_hit_count", 1)) + 1

                picker_parsed_runs.append(
                    {
                        "marker_id": marker_id,
                        "ok": True,
                        "parsed_json": picker_result.get("parsed_json", {}),
                        "selected_candidate_ids": sorted(set(picked_ids)),
                        "issues": marker_issues,
                    }
                )

        per_marker_meta.append(
            {
                "marker_id": marker_id,
                "radius_px": round(radius, 2),
                "request_candidate_ids": [str(c["id"]) for c in local_pool],
                "request_candidate_count": len(local_pool),
                "selected_candidate_ids": sorted(set(picked_ids)),
                "selected_count": len(sorted(set(picked_ids))),
            }
        )

    far_rejected_count = 0
    for cid, item in list(selected_by_id.items()):
        hits = int(item.get("marker_hit_count", 1))
        if hits > 1:
            continue
        source_marker_ids = [str(v) for v in item.get("source_marker_ids", [])]
        if not source_marker_ids:
            selected_by_id.pop(cid, None)
            far_rejected_count += 1
            continue
        mid = source_marker_ids[0]
        radius = float(marker_radius_by_id.get(mid, SIZE_MARKER_MIN_RADIUS_PX))
        if float(item.get("seed_distance_px", 0.0)) > radius:
            selected_by_id.pop(cid, None)
            far_rejected_count += 1

    selected: list[dict[str, Any]] = []
    for cid in sorted(selected_by_id, key=lambda k: (selected_by_id[k]["line"][1], selected_by_id[k]["line"][0])):
        item = dict(selected_by_id[cid])
        item["source_marker_ids"] = sorted(set(str(v) for v in item.get("source_marker_ids", [])))
        item["seed_distance_px"] = round(float(item.get("seed_distance_px", 0.0)), 2)
        item["confidence"] = round(float(item.get("confidence", 0.7)), 3)
        item.pop("marker_hit_count", None)
        selected.append(item)

    marker_seeding_meta = {
        "marker_count": len(markers),
        "seed_count": len(selected),
        "seed_added_count": len(selected),
        "per_marker": per_marker_meta,
    }
    picker_meta = {
        "selection_mode": "gemini_per_marker_local",
        "calls": len(markers),
        "success_count": success_count,
        "failed_count": failed_count,
        "far_rejected_count": far_rejected_count,
        "parse_issues": parse_issues,
    }
    return selected, marker_seeding_meta, picker_meta, issues, picker_raw_runs, picker_parsed_runs


def _build_segments_from_selected_candidates(
    selected_candidates: list[dict[str, Any]],
    width: int,
    height: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    duct_segments: list[dict[str, Any]] = []
    crop_line_segments: list[dict[str, Any]] = []

    for idx, item in enumerate(selected_candidates):
        x1, y1, x2, y2 = item["line"]
        duct_id = f"D{idx + 1}"
        line_id = f"L{idx + 1}"

        path = [
            [
                _clip((x1 / max(1, width - 1)) * 1000.0, 0, 1000),
                _clip((y1 / max(1, height - 1)) * 1000.0, 0, 1000),
            ],
            [
                _clip((x2 / max(1, width - 1)) * 1000.0, 0, 1000),
                _clip((y2 / max(1, height - 1)) * 1000.0, 0, 1000),
            ],
        ]

        duct_segments.append(
            {
                "id": duct_id,
                "type": item["duct_type"],
                "confidence": item["confidence"],
                "path": path,
                "point_count": 2,
                "segment_count": 1,
                "source_candidate_id": item["candidate_id"],
                "source_marker_ids": item.get("source_marker_ids", []),
                "seed_distance_px": item.get("seed_distance_px"),
                "graph_depth": item.get("graph_depth", 0),
                "selection_reason": item.get("selection_reason", ""),
            }
        )

        crop_line_segments.append(
            {
                "id": line_id,
                "duct_id": duct_id,
                "duct_type": item["duct_type"],
                "confidence": item["confidence"],
                "line": [x1, y1, x2, y2],
                "length_px": item["length_px"],
                "source_candidate_id": item["candidate_id"],
                "source_marker_ids": item.get("source_marker_ids", []),
                "seed_distance_px": item.get("seed_distance_px"),
                "graph_depth": item.get("graph_depth", 0),
                "selection_reason": item.get("selection_reason", ""),
            }
        )

    return duct_segments, crop_line_segments


def _build_segments_from_lsd_raw_lines(
    raw_lines: list[dict[str, Any]],
    width: int,
    height: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    duct_segments: list[dict[str, Any]] = []
    crop_line_segments: list[dict[str, Any]] = []

    for idx, item in enumerate(raw_lines):
        x1, y1, x2, y2 = item["line"]
        duct_id = f"D{idx + 1}"
        line_id = f"L{idx + 1}"
        path = [
            [
                _clip((x1 / max(1, width - 1)) * 1000.0, 0, 1000),
                _clip((y1 / max(1, height - 1)) * 1000.0, 0, 1000),
            ],
            [
                _clip((x2 / max(1, width - 1)) * 1000.0, 0, 1000),
                _clip((y2 / max(1, height - 1)) * 1000.0, 0, 1000),
            ],
        ]

        duct_segments.append(
            {
                "id": duct_id,
                "type": "raw_lsd",
                "confidence": 1.0,
                "path": path,
                "point_count": 2,
                "segment_count": 1,
                "source_candidate_id": item.get("id", ""),
                "source_marker_ids": [],
                "seed_distance_px": None,
                "graph_depth": 0,
                "selection_reason": "lsd_debug_raw",
            }
        )

        crop_line_segments.append(
            {
                "id": line_id,
                "duct_id": duct_id,
                "duct_type": "raw_lsd",
                "confidence": 1.0,
                "line": [x1, y1, x2, y2],
                "length_px": float(item.get("length_px", _line_length(item["line"]))),
                "source_candidate_id": item.get("id", ""),
                "source_marker_ids": [],
                "seed_distance_px": None,
                "graph_depth": 0,
                "selection_reason": "lsd_debug_raw",
            }
        )

    return duct_segments, crop_line_segments


def _build_segments_from_gemini_ducts(
    ducts: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    normalized_ducts: list[dict[str, Any]] = []
    crop_line_segments: list[dict[str, Any]] = []
    markers: list[dict[str, Any]] = []
    line_index = 0

    for d_idx, duct in enumerate(ducts):
        duct_id = str(duct.get("id", f"D{d_idx + 1}")).strip() or f"D{d_idx + 1}"
        marker = dict(duct.get("marker", {}))
        marker_id = str(marker.get("id", f"M{d_idx + 1}")).strip() or f"M{d_idx + 1}"
        marker["id"] = marker_id

        boundaries_out: list[dict[str, Any]] = []
        for b_idx, boundary in enumerate(duct.get("boundaries", [])):
            boundary_id = str(boundary.get("id", f"B{b_idx + 1}")).strip() or f"B{b_idx + 1}"
            path_norm = boundary.get("path", [])
            path_px = boundary.get("path_px", [])
            if not isinstance(path_px, list) or len(path_px) < 2:
                continue

            boundaries_out.append(
                {
                    "id": boundary_id,
                    "confidence": _normalize_confidence(boundary.get("confidence", 0.7), default=0.7),
                    "path": path_norm,
                }
            )

            for seg_idx in range(len(path_px) - 1):
                x1, y1 = path_px[seg_idx]
                x2, y2 = path_px[seg_idx + 1]
                line = [int(x1), int(y1), int(x2), int(y2)]
                length_px = _line_length(line)
                if length_px < MIN_SEGMENT_LENGTH_PX:
                    continue
                line_index += 1
                crop_line_segments.append(
                    {
                        "id": f"L{line_index}",
                        "duct_id": duct_id,
                        "duct_type": "other",
                        "confidence": _normalize_confidence(boundary.get("confidence", duct.get("confidence", 0.7)), default=0.7),
                        "line": line,
                        "length_px": round(length_px, 1),
                        "source_candidate_id": "",
                        "source_marker_ids": [marker_id],
                        "seed_distance_px": None,
                        "graph_depth": 0,
                        "marker_id": marker_id,
                        "boundary_id": boundary_id,
                        "segment_index": seg_idx,
                        "selection_reason": "gemini_boundary_direct",
                    }
                )

        if not boundaries_out:
            continue

        normalized_ducts.append(
            {
                "id": duct_id,
                "type": "other",
                "confidence": _normalize_confidence(duct.get("confidence", 0.7), default=0.7),
                "marker": marker,
                "boundaries": boundaries_out,
                "valid_boundary_count": len(boundaries_out),
            }
        )
        markers.append(marker)

    return normalized_ducts, crop_line_segments, markers


def _project_line_segments_to_full_image(
    crop_line_segments: list[dict[str, Any]],
    roi: tuple[int, int, int, int],
    full_width: int,
    full_height: int,
) -> list[dict[str, Any]]:
    x_off, y_off, _, _ = roi
    projected: list[dict[str, Any]] = []

    for item in crop_line_segments:
        x1, y1, x2, y2 = item["line"]
        line = [
            _clip(x1 + x_off, 0, full_width - 1),
            _clip(y1 + y_off, 0, full_height - 1),
            _clip(x2 + x_off, 0, full_width - 1),
            _clip(y2 + y_off, 0, full_height - 1),
        ]
        length_px = _line_length(line)
        if length_px < MIN_SEGMENT_LENGTH_PX:
            continue

        projected.append(
            {
                "id": item["id"],
                "duct_id": item["duct_id"],
                "duct_type": item["duct_type"],
                "confidence": item["confidence"],
                "line": line,
                "length_px": round(length_px, 1),
                "source_candidate_id": item.get("source_candidate_id", ""),
                "source_marker_ids": item.get("source_marker_ids", []),
                "seed_distance_px": item.get("seed_distance_px"),
                "graph_depth": item.get("graph_depth", 0),
                "marker_id": item.get("marker_id", ""),
                "boundary_id": item.get("boundary_id", ""),
                "segment_index": item.get("segment_index"),
                "selection_reason": item.get("selection_reason", ""),
            }
        )

    return projected


def _project_bbox_to_full_image(bbox: list[int], roi: tuple[int, int, int, int], full_width: int, full_height: int) -> list[int]:
    x_off, y_off, _, _ = roi
    x1, y1, x2, y2 = bbox
    return [
        _clip(x1 + x_off, 0, full_width - 1),
        _clip(y1 + y_off, 0, full_height - 1),
        _clip(x2 + x_off, 0, full_width - 1),
        _clip(y2 + y_off, 0, full_height - 1),
    ]


def _verify_marker_box_matches(
    image_bgr: np.ndarray,
    line_segments: list[dict[str, Any]],
    roi: tuple[int, int, int, int],
    provider_error: str,
    normalization_issues: list[str],
    markers: list[dict[str, Any]],
    marker_boxes: list[dict[str, Any]],
) -> dict[str, Any]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    dark_mask = cv2.inRange(gray, 0, EVIDENCE_DARK_THRESHOLD)

    per_segment: list[dict[str, Any]] = []
    evidence_sum = 0.0
    strong_count = 0
    clipped_count = 0
    for seg in line_segments:
        evidence_score = _line_evidence_score(dark_mask, seg["line"])
        outside_ratio = _outside_plan_ratio(seg["line"], roi)
        is_strong = evidence_score >= EVIDENCE_STRONG_THRESHOLD
        is_clipped = outside_ratio > CLIPPED_OUTSIDE_RATIO_THRESHOLD
        if is_strong:
            strong_count += 1
        if is_clipped:
            clipped_count += 1
        evidence_sum += evidence_score
        per_segment.append(
            {
                "line_id": seg.get("id", ""),
                "duct_id": seg.get("duct_id", ""),
                "evidence_score": round(evidence_score, 3),
                "outside_plan_ratio": round(outside_ratio, 3),
                "is_strong_evidence": is_strong,
                "is_clipped": is_clipped,
            }
        )

    total = len(line_segments)
    strong_ratio = (strong_count / total) if total else 0.0
    clipped_ratio = (clipped_count / total) if total else 0.0
    avg_evidence = (evidence_sum / total) if total else 0.0

    marker_count = len(markers)
    matched_marker_ids = {str(item.get("marker_id", "")) for item in marker_boxes if item.get("marker_id")}
    marker_covered_count = sum(1 for marker in markers if str(marker.get("id", "")) in matched_marker_ids)
    marker_covered_ratio = (marker_covered_count / marker_count) if marker_count else 0.0

    if marker_count == 0:
        status = "warn"
    elif marker_covered_ratio >= 0.8 and strong_ratio >= 0.35 and clipped_ratio <= WARN_CLIPPED_RATIO_THRESHOLD:
        status = "good"
    elif marker_covered_ratio >= 0.5:
        status = "warn"
    else:
        status = "bad"

    reasons: list[str] = []
    if provider_error:
        reasons.append(f"Provider warning: {provider_error}")
    if marker_count == 0:
        reasons.append("No valid 14-inch markers were detected.")
    if marker_count > 0 and marker_covered_count < marker_count:
        reasons.append(f"Matched {marker_covered_count}/{marker_count} markers to LSD boxes.")
    if total == 0:
        reasons.append("No matched marker boxes were converted to drawable segments.")
    for issue in normalization_issues[:8]:
        reasons.append(f"Parse warning: {issue}")

    per_marker: list[dict[str, Any]] = []
    for marker in markers:
        marker_id = str(marker.get("id", ""))
        per_marker.append(
            {
                "marker_id": marker_id,
                "center": marker.get("center", [0, 0]),
                "covered": marker_id in matched_marker_ids,
            }
        )

    return {
        "status": status,
        "mode": "marker_box_match",
        "attempts": 1,
        "thresholds": {
            "strong_evidence_score": EVIDENCE_STRONG_THRESHOLD,
            "clipped_outside_ratio": CLIPPED_OUTSIDE_RATIO_THRESHOLD,
            "good_marker_covered_ratio": 0.8,
            "warn_marker_covered_ratio": 0.5,
        },
        "metrics": {
            "total_line_segments": total,
            "strong_evidence_count": strong_count,
            "clipped_count": clipped_count,
            "strong_evidence_ratio": round(strong_ratio, 4),
            "clipped_ratio": round(clipped_ratio, 4),
            "avg_evidence_score": round(avg_evidence, 4),
            "marker_count": marker_count,
            "marker_covered_count": marker_covered_count,
            "marker_covered_ratio": round(marker_covered_ratio, 4),
            "duct_count": len(marker_boxes),
            "boundary_complete_count": len(marker_boxes),
            "boundary_complete_ratio": round(marker_covered_ratio, 4),
        },
        "reasons": reasons,
        "per_segment": per_segment,
        "per_marker": per_marker,
    }


def _line_evidence_score(dark_mask: np.ndarray, line: list[int], radius: int = EVIDENCE_BAND_RADIUS_PX) -> float:
    x1, y1, x2, y2 = line
    length = max(1.0, _line_length(line))
    samples = max(12, int(length / 8.0))
    h, w = dark_mask.shape[:2]

    hits = 0
    for i in range(samples + 1):
        t = i / samples
        x = int(round((1.0 - t) * x1 + (t * x2)))
        y = int(round((1.0 - t) * y1 + (t * y2)))
        xa = max(0, x - radius)
        xb = min(w, x + radius + 1)
        ya = max(0, y - radius)
        yb = min(h, y + radius + 1)
        if xa >= xb or ya >= yb:
            continue
        if np.any(dark_mask[ya:yb, xa:xb] > 0):
            hits += 1

    return hits / float(samples + 1)


def _outside_plan_ratio(line: list[int], roi: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = line
    rx1, ry1, rx2, ry2 = roi
    length = max(1.0, _line_length(line))
    samples = max(12, int(length / 8.0))

    outside = 0
    for i in range(samples + 1):
        t = i / samples
        x = int(round((1.0 - t) * x1 + (t * x2)))
        y = int(round((1.0 - t) * y1 + (t * y2)))
        if x < rx1 or x >= rx2 or y < ry1 or y >= ry2:
            outside += 1

    return outside / float(samples + 1)


def _verify_line_segments(
    image_bgr: np.ndarray,
    line_segments: list[dict[str, Any]],
    roi: tuple[int, int, int, int],
    provider_error: str,
    normalization_issues: list[str],
    markers: list[dict[str, Any]],
    ducts: list[dict[str, Any]],
) -> dict[str, Any]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    dark_mask = cv2.inRange(gray, 0, EVIDENCE_DARK_THRESHOLD)

    per_segment: list[dict[str, Any]] = []
    evidence_sum = 0.0
    strong_count = 0
    clipped_count = 0

    for seg in line_segments:
        evidence_score = _line_evidence_score(dark_mask, seg["line"])
        outside_ratio = _outside_plan_ratio(seg["line"], roi)
        is_strong = evidence_score >= EVIDENCE_STRONG_THRESHOLD
        is_clipped = outside_ratio > CLIPPED_OUTSIDE_RATIO_THRESHOLD

        if is_strong:
            strong_count += 1
        if is_clipped:
            clipped_count += 1
        evidence_sum += evidence_score

        per_segment.append(
            {
                "line_id": seg["id"],
                "duct_id": seg["duct_id"],
                "evidence_score": round(evidence_score, 3),
                "outside_plan_ratio": round(outside_ratio, 3),
                "is_strong_evidence": is_strong,
                "is_clipped": is_clipped,
            }
        )

    total = len(line_segments)
    strong_ratio = (strong_count / total) if total else 0.0
    clipped_ratio = (clipped_count / total) if total else 0.0
    avg_evidence = (evidence_sum / total) if total else 0.0
    marker_count = len(markers)
    marker_covered_count = 0
    per_marker: list[dict[str, Any]] = []

    for marker in markers:
        mx, my = marker["center"]
        nearest = 1e9
        for seg in line_segments:
            nearest = min(nearest, _point_to_segment_distance(float(mx), float(my), seg["line"]))
        covered = nearest <= MARKER_COVERAGE_RADIUS_PX
        if covered:
            marker_covered_count += 1
        per_marker.append(
            {
                "marker_id": marker["id"],
                "center": marker["center"],
                "nearest_segment_distance_px": round(nearest, 2) if nearest < 1e9 else None,
                "covered": covered,
            }
        )

    marker_covered_ratio = (marker_covered_count / marker_count) if marker_count else 0.0
    duct_count = len(ducts)
    boundary_complete_count = 0
    for duct in ducts:
        if int(duct.get("valid_boundary_count", 0)) >= 2:
            boundary_complete_count += 1
    boundary_complete_ratio = (boundary_complete_count / duct_count) if duct_count else 0.0

    if total == 0:
        status = "warn"
    elif (
        strong_ratio >= GOOD_STRONG_RATIO_THRESHOLD
        and marker_covered_ratio >= GOOD_MARKER_COVERED_RATIO_THRESHOLD
        and boundary_complete_ratio >= GOOD_BOUNDARY_COMPLETE_RATIO_THRESHOLD
        and clipped_ratio <= GOOD_CLIPPED_RATIO_THRESHOLD
    ):
        status = "good"
    elif (
        strong_ratio >= WARN_STRONG_RATIO_THRESHOLD
        and marker_covered_ratio >= WARN_MARKER_COVERED_RATIO_THRESHOLD
        and boundary_complete_ratio >= WARN_BOUNDARY_COMPLETE_RATIO_THRESHOLD
        and clipped_ratio <= WARN_CLIPPED_RATIO_THRESHOLD
    ):
        status = "warn"
    else:
        status = "bad"

    reasons: list[str] = []
    if provider_error:
        reasons.append(f"Provider warning: {provider_error}")
    if total == 0:
        reasons.append("No valid line segments were produced after parsing and filtering.")
    if total > 0 and strong_ratio < WARN_STRONG_RATIO_THRESHOLD:
        reasons.append(
            f"Only {round(strong_ratio * 100, 1)}% of segments align with drawing linework (low evidence)."
        )
    if total > 0 and clipped_ratio > WARN_CLIPPED_RATIO_THRESHOLD:
        reasons.append(
            f"{round(clipped_ratio * 100, 1)}% of segments lie outside the plan ROI (high clipping)."
        )
    if marker_count > 0 and marker_covered_ratio < WARN_MARKER_COVERED_RATIO_THRESHOLD:
        reasons.append(
            f"Only {round(marker_covered_ratio * 100, 1)}% of detected 14-inch markers are covered by nearby lines."
        )
    if duct_count > 0 and boundary_complete_ratio < WARN_BOUNDARY_COMPLETE_RATIO_THRESHOLD:
        reasons.append(
            f"Only {round(boundary_complete_ratio * 100, 1)}% of ducts include two valid boundary polylines."
        )
    if marker_count == 0:
        reasons.append("No valid 14-inch markers were detected, so no marker-anchored coverage is available.")

    for issue in normalization_issues[:8]:
        reasons.append(f"Parse warning: {issue}")

    return {
        "status": status,
        "mode": "single_call",
        "attempts": 1,
        "thresholds": {
            "strong_evidence_score": EVIDENCE_STRONG_THRESHOLD,
            "clipped_outside_ratio": CLIPPED_OUTSIDE_RATIO_THRESHOLD,
            "good_strong_ratio": GOOD_STRONG_RATIO_THRESHOLD,
            "warn_strong_ratio": WARN_STRONG_RATIO_THRESHOLD,
            "good_clipped_ratio": GOOD_CLIPPED_RATIO_THRESHOLD,
            "warn_clipped_ratio": WARN_CLIPPED_RATIO_THRESHOLD,
            "marker_coverage_radius_px": MARKER_COVERAGE_RADIUS_PX,
            "good_marker_covered_ratio": GOOD_MARKER_COVERED_RATIO_THRESHOLD,
            "warn_marker_covered_ratio": WARN_MARKER_COVERED_RATIO_THRESHOLD,
            "good_boundary_complete_ratio": GOOD_BOUNDARY_COMPLETE_RATIO_THRESHOLD,
            "warn_boundary_complete_ratio": WARN_BOUNDARY_COMPLETE_RATIO_THRESHOLD,
        },
        "metrics": {
            "total_line_segments": total,
            "strong_evidence_count": strong_count,
            "clipped_count": clipped_count,
            "strong_evidence_ratio": round(strong_ratio, 4),
            "clipped_ratio": round(clipped_ratio, 4),
            "avg_evidence_score": round(avg_evidence, 4),
            "marker_count": marker_count,
            "marker_covered_count": marker_covered_count,
            "marker_covered_ratio": round(marker_covered_ratio, 4),
            "duct_count": duct_count,
            "boundary_complete_count": boundary_complete_count,
            "boundary_complete_ratio": round(boundary_complete_ratio, 4),
        },
        "reasons": reasons,
        "per_segment": per_segment,
        "per_marker": per_marker,
    }


def _draw_overlay(image_bgr: np.ndarray, line_segments: list[dict[str, Any]]) -> np.ndarray:
    overlay = image_bgr.copy()
    for seg in line_segments:
        x1, y1, x2, y2 = seg["line"]
        cv2.line(overlay, (x1, y1), (x2, y2), (255, 0, 0), 3, cv2.LINE_AA)
    return overlay


def _draw_marker_boxes_overlay(
    image_bgr: np.ndarray,
    marker_boxes: list[dict[str, Any]],
    detected_markers: list[dict[str, Any]],
) -> np.ndarray:
    overlay = image_bgr.copy()

    # Draw matched LSD boxes in blue.
    for box in marker_boxes:
        x1, y1, x2, y2 = box.get("bbox", [0, 0, 0, 0])
        box_id = str(box.get("box_id", box.get("id", "BOX"))).strip() or "BOX"
        cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2, cv2.LINE_AA)
        text_y = max(16, int(y2) + 16)
        cv2.putText(overlay, box_id, (int(x1), text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 0, 0), 1, cv2.LINE_AA)

    # Draw all detected 14"Ø markers in red for verification.
    for marker in detected_markers:
        x1, y1, x2, y2 = marker.get("bbox", [0, 0, 0, 0])
        label = str(marker.get("text", "14\"Ø")).strip() or "14\"Ø"
        marker_id = str(marker.get("id", "M?")).strip() or "M?"
        cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2, cv2.LINE_AA)
        text_y = max(16, int(y1) - 6)
        cv2.putText(
            overlay,
            f"{marker_id}:{label}",
            (int(x1), text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    return overlay


def _render_pdf_first_page(pdf_bytes: bytes) -> np.ndarray:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if doc.page_count < 1:
        raise ValueError("PDF has no pages.")
    page = doc.load_page(0)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def _estimate_plan_roi(image_bgr: np.ndarray) -> tuple[int, int, int, int]:
    h, w = image_bgr.shape[:2]
    x1 = int(w * 0.10)
    x2 = int(w * 0.82)
    y1 = int(h * 0.16)
    y2 = int(h * 0.62)
    x1 = max(0, min(x1, w - 20))
    y1 = max(0, min(y1, h - 20))
    x2 = max(x1 + 20, min(x2, w))
    y2 = max(y1 + 20, min(y2, h))
    return x1, y1, x2, y2


def _call_gemini_duct_boundaries(image_b64: str, width: int, height: int) -> dict[str, Any]:
    key = settings.GEMINI_API_KEY
    if not key:
        return {
            "ok": False,
            "provider": "gemini",
            "http_status": None,
            "error": "GEMINI_API_KEY is missing in .env",
            "raw_response": {},
            "model_text": "",
            "parsed_json": {},
        }

    model = settings.GEMINI_MODEL
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    payload = {
        "system_instruction": {"parts": [{"text": GEMINI_BOUNDARY_SYSTEM_PROMPT}]},
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            f"{GEMINI_BOUNDARY_USER_PROMPT} Image size is {width}x{height} pixels. "
                            "Return strict JSON only."
                        )
                    },
                    {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}},
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "ducts": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "id": {"type": "STRING"},
                                "marker": {
                                    "type": "OBJECT",
                                    "properties": {
                                        "text": {"type": "STRING"},
                                        "bbox": {
                                            "type": "ARRAY",
                                            "minItems": 4,
                                            "maxItems": 4,
                                            "items": {"type": "NUMBER"},
                                        },
                                        "center": {
                                            "type": "ARRAY",
                                            "minItems": 2,
                                            "maxItems": 2,
                                            "items": {"type": "NUMBER"},
                                        },
                                        "confidence": {"type": "NUMBER"},
                                    },
                                    "required": ["text", "bbox", "center", "confidence"],
                                },
                                "boundaries": {
                                    "type": "ARRAY",
                                    "minItems": 1,
                                    "maxItems": 2,
                                    "items": {
                                        "type": "OBJECT",
                                        "properties": {
                                            "id": {"type": "STRING"},
                                            "path": {
                                                "type": "ARRAY",
                                                "minItems": 2,
                                                "items": {
                                                    "type": "ARRAY",
                                                    "minItems": 2,
                                                    "maxItems": 2,
                                                    "items": {"type": "NUMBER"},
                                                },
                                            },
                                            "confidence": {"type": "NUMBER"},
                                        },
                                        "required": ["id", "path", "confidence"],
                                    },
                                },
                                "confidence": {"type": "NUMBER"},
                            },
                            "required": ["id", "marker", "boundaries", "confidence"],
                        },
                    }
                },
                "required": ["ducts"],
            },
        },
    }

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=(settings.AI_CONNECT_TIMEOUT_SECONDS, settings.AI_REQUEST_TIMEOUT_SECONDS),
        )
    except Exception as exc:
        return {
            "ok": False,
            "provider": "gemini",
            "http_status": None,
            "error": f"Gemini duct-boundary request failed: {exc}",
            "raw_response": {},
            "model_text": "",
            "parsed_json": {},
        }

    try:
        raw_response: Any = response.json()
    except Exception:
        raw_response = {"raw_text": response.text}

    if response.status_code >= 400:
        detail = _response_error_detail(response)
        return {
            "ok": False,
            "provider": "gemini",
            "http_status": response.status_code,
            "error": f"Gemini duct-boundary API error ({response.status_code}): {detail}",
            "raw_response": raw_response,
            "model_text": "",
            "parsed_json": {},
        }

    model_text = ""
    if isinstance(raw_response, dict):
        try:
            model_text = raw_response["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            model_text = ""

    if not model_text:
        return {
            "ok": False,
            "provider": "gemini",
            "http_status": response.status_code,
            "error": "Gemini duct-boundary response missing model text.",
            "raw_response": raw_response,
            "model_text": "",
            "parsed_json": {},
        }

    try:
        parsed_json = _extract_json(model_text)
    except Exception as exc:
        return {
            "ok": False,
            "provider": "gemini",
            "http_status": response.status_code,
            "error": f"Gemini duct-boundary JSON parse failed: {exc}",
            "raw_response": raw_response,
            "model_text": model_text,
            "parsed_json": {},
        }

    return {
        "ok": True,
        "provider": "gemini",
        "http_status": response.status_code,
        "error": "",
        "raw_response": raw_response,
        "model_text": model_text,
        "parsed_json": parsed_json,
    }


def _call_gemini_size_markers(image_b64: str, width: int, height: int) -> dict[str, Any]:
    key = settings.GEMINI_API_KEY
    if not key:
        return {
            "ok": False,
            "provider": "gemini",
            "http_status": None,
            "error": "GEMINI_API_KEY is missing in .env",
            "raw_response": {},
            "model_text": "",
            "parsed_json": {},
        }

    model = settings.GEMINI_MODEL
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    payload = {
        "system_instruction": {"parts": [{"text": SIZE_MARKER_SYSTEM_PROMPT}]},
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            f"{SIZE_MARKER_USER_PROMPT} Image size is {width}x{height} pixels. "
                            "Return strict JSON only."
                        )
                    },
                    {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}},
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "size_markers": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "text": {"type": "STRING"},
                                "bbox": {
                                    "type": "ARRAY",
                                    "minItems": 4,
                                    "maxItems": 4,
                                    "items": {"type": "NUMBER"},
                                },
                                "confidence": {"type": "NUMBER"},
                            },
                            "required": ["text", "bbox", "confidence"],
                        },
                    }
                },
                "required": ["size_markers"],
            },
        },
    }

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=(settings.AI_CONNECT_TIMEOUT_SECONDS, settings.AI_REQUEST_TIMEOUT_SECONDS),
        )
    except Exception as exc:
        return {
            "ok": False,
            "provider": "gemini",
            "http_status": None,
            "error": f"Gemini marker request failed: {exc}",
            "raw_response": {},
            "model_text": "",
            "parsed_json": {},
        }

    try:
        raw_response: Any = response.json()
    except Exception:
        raw_response = {"raw_text": response.text}

    if response.status_code >= 400:
        detail = _response_error_detail(response)
        return {
            "ok": False,
            "provider": "gemini",
            "http_status": response.status_code,
            "error": f"Gemini marker API error ({response.status_code}): {detail}",
            "raw_response": raw_response,
            "model_text": "",
            "parsed_json": {},
        }

    model_text = ""
    if isinstance(raw_response, dict):
        try:
            model_text = raw_response["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            model_text = ""

    if not model_text:
        return {
            "ok": False,
            "provider": "gemini",
            "http_status": response.status_code,
            "error": "Gemini marker response missing model text.",
            "raw_response": raw_response,
            "model_text": "",
            "parsed_json": {},
        }

    try:
        parsed_json = _extract_json(model_text)
    except Exception as exc:
        return {
            "ok": False,
            "provider": "gemini",
            "http_status": response.status_code,
            "error": f"Gemini marker JSON parse failed: {exc}",
            "raw_response": raw_response,
            "model_text": model_text,
            "parsed_json": {},
        }

    return {
        "ok": True,
        "provider": "gemini",
        "http_status": response.status_code,
        "error": "",
        "raw_response": raw_response,
        "model_text": model_text,
        "parsed_json": parsed_json,
    }


def _call_google_vision_size_markers(image_b64: str, width: int, height: int) -> dict[str, Any]:
    key = settings.GEMINI_API_KEY
    if not key:
        return {
            "ok": False,
            "provider": "google_vision",
            "http_status": None,
            "error": "GEMINI_API_KEY is missing in .env",
            "raw_response": {},
            "model_text": "",
            "parsed_json": {"size_markers": [], "vision_tokens": [], "issues": ["Missing API key."]},
        }

    url = f"https://vision.googleapis.com/v1/images:annotate?key={key}"
    payload = {
        "requests": [
            {
                "image": {"content": image_b64},
                "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
            }
        ]
    }

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=(settings.AI_CONNECT_TIMEOUT_SECONDS, settings.AI_REQUEST_TIMEOUT_SECONDS),
        )
    except Exception as exc:
        return {
            "ok": False,
            "provider": "google_vision",
            "http_status": None,
            "error": f"Google Vision marker request failed: {exc}",
            "raw_response": {},
            "model_text": "",
            "parsed_json": {"size_markers": [], "vision_tokens": [], "issues": [f"Request failed: {exc}"]},
        }

    try:
        raw_response: Any = response.json()
    except Exception:
        raw_response = {"raw_text": response.text}

    if response.status_code >= 400:
        detail = _response_error_detail(response)
        return {
            "ok": False,
            "provider": "google_vision",
            "http_status": response.status_code,
            "error": f"Google Vision API error ({response.status_code}): {detail}",
            "raw_response": raw_response,
            "model_text": "",
            "parsed_json": {"size_markers": [], "vision_tokens": [], "issues": [detail]},
        }

    response_error = ""
    if isinstance(raw_response, dict):
        responses = raw_response.get("responses", [])
        if isinstance(responses, list) and responses and isinstance(responses[0], dict):
            error_obj = responses[0].get("error")
            if isinstance(error_obj, dict):
                response_error = str(error_obj.get("message", "")).strip()
                if not response_error:
                    response_error = json.dumps(error_obj)

    if response_error:
        return {
            "ok": False,
            "provider": "google_vision",
            "http_status": response.status_code,
            "error": f"Google Vision response error: {response_error}",
            "raw_response": raw_response,
            "model_text": "",
            "parsed_json": {"size_markers": [], "vision_tokens": [], "issues": [response_error]},
        }

    tokens = _extract_vision_word_tokens(raw_response if isinstance(raw_response, dict) else {}, width, height)
    parsed_json, extraction_issues = _extract_vision_size_markers(tokens, width, height)
    parsed_json["issues"] = extraction_issues

    return {
        "ok": True,
        "provider": "google_vision",
        "http_status": response.status_code,
        "error": "",
        "raw_response": raw_response,
        "model_text": (
            f"vision_tokens={len(tokens)}, size_markers={len(parsed_json.get('size_markers', []))}, "
            f"issues={len(extraction_issues)}"
        ),
        "parsed_json": parsed_json,
    }


def _call_gemini_marker_line_picker(
    image_b64: str,
    width: int,
    height: int,
    marker: dict[str, Any],
    candidate_rows_json: str,
) -> dict[str, Any]:
    key = settings.GEMINI_API_KEY
    if not key:
        return {
            "ok": False,
            "provider": "gemini",
            "http_status": None,
            "error": "GEMINI_API_KEY is missing in .env",
            "raw_response": {},
            "model_text": "",
            "parsed_json": {},
        }

    model = settings.GEMINI_MODEL
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    marker_center = marker.get("center", [0, 0])
    marker_bbox = marker.get("bbox", [0, 0, 0, 0])
    payload = {
        "system_instruction": {"parts": [{"text": MARKER_PICK_SYSTEM_PROMPT}]},
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            f"{MARKER_PICK_USER_PROMPT} "
                            f"Image size is {width}x{height} pixels. "
                            f"Marker ID: {marker.get('id', '')}, text: {marker.get('text', '')}, "
                            f"marker center px: {marker_center}, marker bbox px: {marker_bbox}. "
                            f"Candidate lines (normalized 0..1000): {candidate_rows_json}. "
                            "Return strict JSON only."
                        )
                    },
                    {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}},
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "selected_lines": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "id": {"type": "STRING"},
                                "confidence": {"type": "NUMBER"},
                            },
                            "required": ["id", "confidence"],
                        },
                    }
                },
                "required": ["selected_lines"],
            },
        },
    }

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=(settings.AI_CONNECT_TIMEOUT_SECONDS, settings.AI_REQUEST_TIMEOUT_SECONDS),
        )
    except Exception as exc:
        return {
            "ok": False,
            "provider": "gemini",
            "http_status": None,
            "error": f"Gemini marker-line picker request failed: {exc}",
            "raw_response": {},
            "model_text": "",
            "parsed_json": {},
        }

    try:
        raw_response: Any = response.json()
    except Exception:
        raw_response = {"raw_text": response.text}

    if response.status_code >= 400:
        detail = _response_error_detail(response)
        return {
            "ok": False,
            "provider": "gemini",
            "http_status": response.status_code,
            "error": f"Gemini marker-line picker API error ({response.status_code}): {detail}",
            "raw_response": raw_response,
            "model_text": "",
            "parsed_json": {},
        }

    model_text = ""
    if isinstance(raw_response, dict):
        try:
            model_text = raw_response["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            model_text = ""

    if not model_text:
        return {
            "ok": False,
            "provider": "gemini",
            "http_status": response.status_code,
            "error": "Gemini marker-line picker response missing model text.",
            "raw_response": raw_response,
            "model_text": "",
            "parsed_json": {},
        }

    try:
        parsed_json = _extract_json(model_text)
    except Exception as exc:
        return {
            "ok": False,
            "provider": "gemini",
            "http_status": response.status_code,
            "error": f"Gemini marker-line picker JSON parse failed: {exc}",
            "raw_response": raw_response,
            "model_text": model_text,
            "parsed_json": {},
        }

    return {
        "ok": True,
        "provider": "gemini",
        "http_status": response.status_code,
        "error": "",
        "raw_response": raw_response,
        "model_text": model_text,
        "parsed_json": parsed_json,
    }


def run_detection(pdf_bytes: bytes, provider: str, media_root: Path) -> dict[str, Any]:
    media_root.mkdir(parents=True, exist_ok=True)
    run_id = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_{uuid4().hex[:6]}'
    run_dir = media_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    original = _render_pdf_first_page(pdf_bytes)
    h, w = original.shape[:2]
    cv2.imwrite(str(run_dir / "page.png"), original)

    roi = _estimate_plan_roi(original)
    rx1, ry1, rx2, ry2 = roi
    plan_crop = original[ry1:ry2, rx1:rx2]
    cv2.imwrite(str(run_dir / "plan_crop.png"), plan_crop)

    detection_mode = str(getattr(settings, "DETECTION_MODE", "gemini_global_marker_boundaries")).strip().lower()

    if detection_mode == "lsd_14_marker_boxes":
        raw_candidates = _extract_lsd_raw_lines_tiled(plan_crop)
        candidate_lines_rel = f"runs/{run_id}/candidate_lines.json"
        with open(run_dir / "candidate_lines.json", "w", encoding="utf-8") as handle:
            json.dump(raw_candidates, handle, indent=2)
        with open(run_dir / "candidate_lines_raw.json", "w", encoding="utf-8") as handle:
            json.dump(raw_candidates, handle, indent=2)

        box_candidates = _extract_lsd_box_candidates(raw_candidates, plan_crop.shape[1], plan_crop.shape[0])
        lsd_box_candidates_rel = f"runs/{run_id}/lsd_box_candidates.json"

        ok, encoded = cv2.imencode(".jpg", plan_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise RuntimeError("Failed to encode page image for AI request.")
        image_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")

        marker_result = _call_google_vision_size_markers(image_b64, plan_crop.shape[1], plan_crop.shape[0])
        parsed_marker_payload = marker_result.get("parsed_json", {}) if isinstance(marker_result.get("parsed_json", {}), dict) else {}
        marker_extraction_issues = parsed_marker_payload.get("issues", [])
        marker_extraction_issues = [str(msg) for msg in marker_extraction_issues] if isinstance(marker_extraction_issues, list) else []
        size_markers, marker_issues = _normalize_size_markers(
            parsed_marker_payload,
            plan_crop.shape[1],
            plan_crop.shape[0],
        )
        marker_issues = marker_extraction_issues + marker_issues
        if not size_markers:
            marker_issues.append("No 14-inch diameter markers (14\"⌀) detected.")

        box_candidates = _augment_boxes_with_marker_local_candidates(
            box_candidates,
            size_markers,
            raw_candidates,
            plan_crop.shape[1],
            plan_crop.shape[0],
        )
        with open(run_dir / "lsd_box_candidates.json", "w", encoding="utf-8") as handle:
            json.dump(box_candidates, handle, indent=2)

        marker_boxes_crop, match_issues = _match_markers_to_lsd_boxes(size_markers, box_candidates)
        normalization_issues = marker_issues + match_issues

        duct_segments, crop_line_segments = _build_segments_from_marker_boxes(
            marker_boxes_crop,
            plan_crop.shape[1],
            plan_crop.shape[0],
        )
        line_segments = _project_line_segments_to_full_image(crop_line_segments, roi, w, h)

        full_markers: list[dict[str, Any]] = []
        for marker in size_markers:
            mx, my = marker["center"]
            bx1, by1, bx2, by2 = marker["bbox"]
            full_markers.append(
                {
                    "id": marker["id"],
                    "text": marker["text"],
                    "center": [mx + rx1, my + ry1],
                    "bbox": [bx1 + rx1, by1 + ry1, bx2 + rx1, by2 + ry1],
                    "local_radius_px": float(marker.get("local_radius_px", 70.0)),
                    "confidence": marker.get("confidence", 0.0),
                }
            )

        full_markers_by_id = {str(marker["id"]): marker for marker in full_markers}
        marker_boxes: list[dict[str, Any]] = []
        for item in marker_boxes_crop:
            marker_id = str(item.get("marker_id", ""))
            bbox_crop = item.get("bbox", [0, 0, 0, 0])
            bbox_full = _project_bbox_to_full_image(bbox_crop, roi, w, h)
            marker_full = full_markers_by_id.get(marker_id, {})
            row = {
                **item,
                "bbox_crop": bbox_crop,
                "bbox": bbox_full,
                "marker_bbox_full": marker_full.get("bbox", item.get("marker_bbox", [])),
                "marker_center_full": marker_full.get("center", item.get("marker_center", [])),
            }
            marker_boxes.append(row)

        marker_box_matches_rel = f"runs/{run_id}/marker_box_matches.json"
        with open(run_dir / "marker_box_matches.json", "w", encoding="utf-8") as handle:
            json.dump({"matches": marker_boxes, "issues": normalization_issues}, handle, indent=2)

        marker_raw_rel = f"runs/{run_id}/size_markers_raw.json"
        with open(run_dir / "size_markers_raw.json", "w", encoding="utf-8") as handle:
            json.dump(marker_result.get("raw_response", {}), handle, indent=2, default=str)

        marker_norm_rel = f"runs/{run_id}/size_markers.json"
        with open(run_dir / "size_markers.json", "w", encoding="utf-8") as handle:
            json.dump({"markers": size_markers, "issues": normalization_issues, "error": marker_result.get("error", "")}, handle, indent=2)

        vision_tokens_rel = f"runs/{run_id}/vision_tokens.json"
        with open(run_dir / "vision_tokens.json", "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "tokens": parsed_marker_payload.get("vision_tokens", []),
                    "marker_candidates_px": parsed_marker_payload.get("vision_marker_candidates_px", []),
                    "issues": marker_extraction_issues,
                },
                handle,
                indent=2,
            )

        gemini_boundaries_raw_rel = f"runs/{run_id}/gemini_boundaries_raw.json"
        with open(run_dir / "gemini_boundaries_raw.json", "w", encoding="utf-8") as handle:
            json.dump({}, handle, indent=2)

        gemini_boundaries_parsed_rel = f"runs/{run_id}/gemini_boundaries_parsed.json"
        with open(run_dir / "gemini_boundaries_parsed.json", "w", encoding="utf-8") as handle:
            json.dump({"ducts": [], "issues": ["Gemini boundary mode disabled in lsd_14_marker_boxes."]}, handle, indent=2)

        provider_runs: list[dict[str, Any]] = [{"run_index": 1, **marker_result}]
        provider_result = provider_runs[0]
        provider_error_summary = str(marker_result.get("error", "")).strip()

        raw_provider_response_rel = f"runs/{run_id}/provider_raw_response.json"
        with open(run_dir / "provider_raw_response.json", "w", encoding="utf-8") as handle:
            json.dump(
                [
                    {
                        "run_index": r["run_index"],
                        "ok": r.get("ok", False),
                        "http_status": r.get("http_status"),
                        "error": r.get("error", ""),
                        "raw_response": r.get("raw_response", {}),
                    }
                    for r in provider_runs
                ],
                handle,
                indent=2,
                default=str,
            )

        provider_text_rel = f"runs/{run_id}/provider_model_text.txt"
        with open(run_dir / "provider_model_text.txt", "w", encoding="utf-8") as handle:
            for r in provider_runs:
                handle.write(f"===== RUN {r['run_index']} =====\n")
                handle.write(str(r.get("model_text", "")))
                handle.write("\n\n")

        provider_parsed_rel = f"runs/{run_id}/provider_parsed.json"
        with open(run_dir / "provider_parsed.json", "w", encoding="utf-8") as handle:
            json.dump(
                [
                    {
                        "run_index": r["run_index"],
                        "ok": r.get("ok", False),
                        "parsed_json": r.get("parsed_json", {}),
                    }
                    for r in provider_runs
                ],
                handle,
                indent=2,
                default=str,
            )

        normalization_meta = {
            "mode": "lsd_14_marker_boxes",
            "marker_first_mode": MARKER_FIRST_14_ONLY,
            "input_duct_segments": len(marker_boxes),
            "accepted_duct_segments": len(duct_segments),
            "accepted_line_segments": len(crop_line_segments),
            "invalid_items": 0,
            "rejected_paths": 0,
            "short_or_duplicate_lines": 0,
            "issues": normalization_issues,
        }

        verification = _verify_marker_box_matches(
            original,
            line_segments,
            roi,
            provider_error_summary,
            normalization_meta.get("issues", []),
            full_markers,
            marker_boxes,
        )

        overlay = _draw_marker_boxes_overlay(original, marker_boxes, full_markers)
        cv2.imwrite(str(run_dir / "overlay.png"), overlay)
        marker_box_overlay_rel = f"runs/{run_id}/marker_box_overlay.png"
        cv2.imwrite(str(run_dir / "marker_box_overlay.png"), overlay)

        attempts = [
            {
                "provider": "google_vision",
                "run_index": 1,
                "status": "ok" if marker_result.get("ok") else "warn",
                "selected_count": str(len(marker_boxes)),
                "detail": marker_result.get("error") or "marker detection success",
            }
        ]

        warning_reasons = verification.get("reasons", [])
        result = {
            "run_id": run_id,
            "provider_requested": (provider or "").strip().lower() or "gemini",
            "provider_used": "google_vision",
            "provider_http_status": provider_result.get("http_status"),
            "provider_error": provider_error_summary,
            "marker_first_mode": MARKER_FIRST_14_ONLY,
            "detection_mode": detection_mode,
            "selection_mode": "lsd_14_marker_boxes",
            "lsd_enabled": True,
            "gemini_enabled": False,
            "image_width": w,
            "image_height": h,
            "detection_count": len(line_segments),
            "duct_segment_count": len(duct_segments),
            "line_segment_count": len(line_segments),
            "raw_candidate_line_count": len(raw_candidates),
            "centerline_candidate_count": 0,
            "candidate_line_count": len(raw_candidates),
            "selected_raw_candidate_count": len(marker_boxes),
            "selected_candidate_count": len(marker_boxes),
            "size_marker_count": len(size_markers),
            "size_marker_seed_added_count": len(marker_boxes),
            "expansion_added_count": 0,
            "merged_centerline_count": 0,
            "merged_pairs_count": 0,
            "plan_roi": {"x1": rx1, "y1": ry1, "x2": rx2, "y2": ry2},
            "coordinate_space": {
                "normalized": "0..1000 relative to plan_crop.png",
                "pixel_line_segments": "full page image coordinates",
            },
            "marker_boxes": marker_boxes,
            "marker_seeding": {
                "disabled": True,
                "reason": "lsd_14_marker_boxes uses text-in-box matching, not line seeding.",
                "marker_count": len(size_markers),
                "seed_count": len(marker_boxes),
                "seed_added_count": len(marker_boxes),
                "per_marker": [],
            },
            "gemini_marker_picking": {
                "disabled": True,
                "reason": "Marker text detection in this mode is handled by Google Vision OCR only.",
                "selection_mode": "lsd_14_marker_boxes",
                "calls": 0,
                "success_count": 0,
                "failed_count": 0,
                "far_rejected_count": 0,
                "parse_issues": [],
                "per_marker": [],
            },
            "graph_expansion": {
                "disabled": True,
                "reason": "Not used in lsd_14_marker_boxes mode.",
                "expanded_count": 0,
                "component_count": 0,
                "kept_component_count": 0,
                "pruned_count": 0,
            },
            "ducts": duct_segments,
            "duct_segments": duct_segments,
            "line_segments": line_segments,
            "duct_lines": line_segments,
            "verification": verification,
            "normalization": normalization_meta,
            "raw_provider_response_path": raw_provider_response_rel,
            "warnings": warning_reasons,
            "attempts": attempts,
            "files": {
                "page": f"runs/{run_id}/page.png",
                "plan_crop": f"runs/{run_id}/plan_crop.png",
                "overlay": f"runs/{run_id}/overlay.png",
                "marker_box_overlay": marker_box_overlay_rel,
                "lsd_box_candidates": lsd_box_candidates_rel,
                "marker_box_matches": marker_box_matches_rel,
                "json": f"runs/{run_id}/result.json",
                "candidate_lines": candidate_lines_rel,
                "candidate_lines_raw": f"runs/{run_id}/candidate_lines_raw.json",
                "size_markers_raw": marker_raw_rel,
                "size_markers": marker_norm_rel,
                "vision_tokens": vision_tokens_rel,
                "gemini_boundaries_raw": gemini_boundaries_raw_rel,
                "gemini_boundaries_parsed": gemini_boundaries_parsed_rel,
                "raw_provider_response": raw_provider_response_rel,
                "provider_text": provider_text_rel,
                "provider_parsed": provider_parsed_rel,
            },
        }

        with open(run_dir / "result.json", "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, default=str)
        return result

    if detection_mode == "lsd_debug":
        raw_candidates = _extract_lsd_raw_lines_tiled(plan_crop)
        candidate_lines_rel = f"runs/{run_id}/candidate_lines.json"
        with open(run_dir / "candidate_lines.json", "w", encoding="utf-8") as handle:
            json.dump(raw_candidates, handle, indent=2)
        with open(run_dir / "candidate_lines_raw.json", "w", encoding="utf-8") as handle:
            json.dump(raw_candidates, handle, indent=2)

        lsd_raw_lines_rel = f"runs/{run_id}/lsd_raw_lines.json"
        with open(run_dir / "lsd_raw_lines.json", "w", encoding="utf-8") as handle:
            json.dump(raw_candidates, handle, indent=2)

        duct_segments, crop_line_segments = _build_segments_from_lsd_raw_lines(
            raw_candidates,
            plan_crop.shape[1],
            plan_crop.shape[0],
        )
        line_segments = _project_line_segments_to_full_image(crop_line_segments, roi, w, h)

        marker_raw_rel = f"runs/{run_id}/size_markers_raw.json"
        with open(run_dir / "size_markers_raw.json", "w", encoding="utf-8") as handle:
            json.dump({}, handle, indent=2)

        marker_norm_rel = f"runs/{run_id}/size_markers.json"
        with open(run_dir / "size_markers.json", "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "markers": [],
                    "issues": ["LSD debug mode enabled; Gemini marker detection skipped."],
                    "error": "gemini_disabled_for_debug",
                },
                handle,
                indent=2,
            )

        gemini_boundaries_raw_rel = f"runs/{run_id}/gemini_boundaries_raw.json"
        with open(run_dir / "gemini_boundaries_raw.json", "w", encoding="utf-8") as handle:
            json.dump({}, handle, indent=2)

        gemini_boundaries_parsed_rel = f"runs/{run_id}/gemini_boundaries_parsed.json"
        with open(run_dir / "gemini_boundaries_parsed.json", "w", encoding="utf-8") as handle:
            json.dump({"ducts": [], "issues": ["Gemini disabled in lsd_debug mode."]}, handle, indent=2)

        provider_error_summary = "gemini_disabled_for_debug"
        raw_provider_response_rel = f"runs/{run_id}/provider_raw_response.json"
        with open(run_dir / "provider_raw_response.json", "w", encoding="utf-8") as handle:
            json.dump([], handle, indent=2)

        provider_text_rel = f"runs/{run_id}/provider_model_text.txt"
        with open(run_dir / "provider_model_text.txt", "w", encoding="utf-8") as handle:
            handle.write("gemini_disabled_for_debug\n")

        provider_parsed_rel = f"runs/{run_id}/provider_parsed.json"
        with open(run_dir / "provider_parsed.json", "w", encoding="utf-8") as handle:
            json.dump([], handle, indent=2)

        debug_warning = "LSD debug mode: geometric line extraction only (not duct classification)."
        normalization_meta = {
            "mode": "lsd_debug_raw_roi",
            "marker_first_mode": MARKER_FIRST_14_ONLY,
            "input_duct_segments": len(duct_segments),
            "accepted_duct_segments": len(duct_segments),
            "accepted_line_segments": len(crop_line_segments),
            "invalid_items": 0,
            "rejected_paths": 0,
            "short_or_duplicate_lines": 0,
            "issues": [debug_warning],
        }

        verification = _verify_line_segments(
            original,
            line_segments,
            roi,
            provider_error_summary,
            normalization_meta.get("issues", []),
            [],
            [],
        )
        verification["status"] = "warn"
        verification.setdefault("reasons", [])
        if debug_warning not in verification["reasons"]:
            verification["reasons"].insert(0, debug_warning)

        overlay = _draw_overlay(original, line_segments)
        cv2.imwrite(str(run_dir / "overlay.png"), overlay)
        lsd_raw_overlay_rel = f"runs/{run_id}/lsd_raw_overlay.png"
        cv2.imwrite(str(run_dir / "lsd_raw_overlay.png"), overlay)

        warning_reasons = verification.get("reasons", [])
        attempts = [
            {
                "provider": "none",
                "run_index": 1,
                "status": "skipped",
                "selected_count": str(len(line_segments)),
                "detail": "Gemini disabled for lsd_debug mode.",
            }
        ]

        result = {
            "run_id": run_id,
            "provider_requested": (provider or "").strip().lower() or "none",
            "provider_used": "none",
            "provider_http_status": None,
            "provider_error": provider_error_summary,
            "marker_first_mode": MARKER_FIRST_14_ONLY,
            "detection_mode": detection_mode,
            "selection_mode": "lsd_debug_raw_roi",
            "lsd_enabled": True,
            "gemini_enabled": False,
            "image_width": w,
            "image_height": h,
            "detection_count": len(line_segments),
            "duct_segment_count": len(duct_segments),
            "line_segment_count": len(line_segments),
            "raw_candidate_line_count": len(raw_candidates),
            "centerline_candidate_count": 0,
            "candidate_line_count": len(raw_candidates),
            "selected_raw_candidate_count": len(raw_candidates),
            "selected_candidate_count": len(line_segments),
            "size_marker_count": 0,
            "size_marker_seed_added_count": 0,
            "expansion_added_count": 0,
            "merged_centerline_count": 0,
            "merged_pairs_count": 0,
            "plan_roi": {"x1": rx1, "y1": ry1, "x2": rx2, "y2": ry2},
            "coordinate_space": {
                "normalized": "0..1000 relative to plan_crop.png",
                "pixel_line_segments": "full page image coordinates",
            },
            "marker_seeding": {
                "disabled": True,
                "reason": "LSD debug mode does not use marker seeding.",
                "marker_count": 0,
                "seed_count": 0,
                "seed_added_count": 0,
                "per_marker": [],
            },
            "gemini_marker_picking": {
                "disabled": True,
                "reason": "Gemini disabled in lsd_debug mode.",
                "selection_mode": "lsd_debug_raw_roi",
                "calls": 0,
                "success_count": 0,
                "failed_count": 0,
                "far_rejected_count": 0,
                "parse_issues": [],
                "per_marker": [],
            },
            "graph_expansion": {
                "disabled": True,
                "reason": "LSD debug mode does not run graph expansion.",
                "expanded_count": 0,
                "component_count": 0,
                "kept_component_count": 0,
                "pruned_count": 0,
            },
            "ducts": duct_segments,
            "duct_segments": duct_segments,
            "line_segments": line_segments,
            "duct_lines": line_segments,
            "verification": verification,
            "normalization": normalization_meta,
            "raw_provider_response_path": raw_provider_response_rel,
            "warnings": warning_reasons,
            "attempts": attempts,
            "files": {
                "page": f"runs/{run_id}/page.png",
                "plan_crop": f"runs/{run_id}/plan_crop.png",
                "overlay": f"runs/{run_id}/overlay.png",
                "lsd_raw_overlay": lsd_raw_overlay_rel,
                "lsd_raw_lines": lsd_raw_lines_rel,
                "json": f"runs/{run_id}/result.json",
                "candidate_lines": candidate_lines_rel,
                "candidate_lines_raw": f"runs/{run_id}/candidate_lines_raw.json",
                "size_markers_raw": marker_raw_rel,
                "size_markers": marker_norm_rel,
                "gemini_boundaries_raw": gemini_boundaries_raw_rel,
                "gemini_boundaries_parsed": gemini_boundaries_parsed_rel,
                "raw_provider_response": raw_provider_response_rel,
                "provider_text": provider_text_rel,
                "provider_parsed": provider_parsed_rel,
            },
        }

        with open(run_dir / "result.json", "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, default=str)

        return result

    ok, encoded = cv2.imencode(".jpg", plan_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("Failed to encode page image for AI request.")
    image_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")

    # LSD is disabled for this Gemini-only boundary detection mode.
    raw_candidates: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    candidate_lines_rel = f"runs/{run_id}/candidate_lines.json"
    with open(run_dir / "candidate_lines.json", "w", encoding="utf-8") as handle:
        json.dump(candidates, handle, indent=2)
    with open(run_dir / "candidate_lines_raw.json", "w", encoding="utf-8") as handle:
        json.dump(raw_candidates, handle, indent=2)

    provider_requested = (provider or "").strip().lower()
    provider_used = "gemini"
    provider_selection_warning = ""
    if provider_requested and provider_requested != "gemini":
        provider_selection_warning = "Gemini-only boundary mode only supports Gemini. Request was forced to gemini."
    elif not provider_requested:
        provider_selection_warning = "No provider selected; defaulted to gemini."

    boundary_result: dict[str, Any] = {
        "ok": False,
        "http_status": None,
        "error": "",
        "raw_response": {},
        "parsed_json": {},
    }
    boundary_issues: list[str] = []
    boundary_result = _call_gemini_duct_boundaries(image_b64, plan_crop.shape[1], plan_crop.shape[0])
    normalized_ducts, boundary_issues = _normalize_gemini_duct_boundaries(
        boundary_result.get("parsed_json", {}),
        plan_crop.shape[1],
        plan_crop.shape[0],
    )
    if not normalized_ducts:
        boundary_issues.append("No valid 14-inch ducts with boundary paths were detected.")

    duct_segments, crop_line_segments, crop_markers = _build_segments_from_gemini_ducts(normalized_ducts)

    line_segments = _project_line_segments_to_full_image(crop_line_segments, roi, w, h)

    size_markers: list[dict[str, Any]] = []
    for marker in crop_markers:
        size_markers.append(
            {
                "id": marker.get("id"),
                "text": marker.get("text", ""),
                "bbox": marker.get("bbox", []),
                "center": marker.get("center", []),
                "bbox_diag_px": marker.get("bbox_diag_px", 0.0),
                "local_radius_px": marker.get("local_radius_px", 70.0),
                "confidence": marker.get("confidence", 0.0),
            }
        )

    marker_raw_rel = f"runs/{run_id}/size_markers_raw.json"
    with open(run_dir / "size_markers_raw.json", "w", encoding="utf-8") as handle:
        json.dump(boundary_result.get("raw_response", {}), handle, indent=2, default=str)

    marker_norm_rel = f"runs/{run_id}/size_markers.json"
    with open(run_dir / "size_markers.json", "w", encoding="utf-8") as handle:
        json.dump({"markers": size_markers, "issues": boundary_issues, "error": boundary_result.get("error", "")}, handle, indent=2)

    gemini_boundaries_raw_rel = f"runs/{run_id}/gemini_boundaries_raw.json"
    with open(run_dir / "gemini_boundaries_raw.json", "w", encoding="utf-8") as handle:
        json.dump(boundary_result.get("raw_response", {}), handle, indent=2, default=str)

    gemini_boundaries_parsed_rel = f"runs/{run_id}/gemini_boundaries_parsed.json"
    with open(run_dir / "gemini_boundaries_parsed.json", "w", encoding="utf-8") as handle:
        json.dump({"ducts": normalized_ducts, "issues": boundary_issues}, handle, indent=2, default=str)

    provider_runs: list[dict[str, Any]] = [{"run_index": 1, **boundary_result}]
    provider_result = provider_runs[0]
    provider_error_summary = str(boundary_result.get("error", "")).strip()

    raw_provider_response_rel = f"runs/{run_id}/provider_raw_response.json"
    raw_provider_response_path = run_dir / "provider_raw_response.json"
    with open(raw_provider_response_path, "w", encoding="utf-8") as handle:
        json.dump(
            [
                {
                    "run_index": r["run_index"],
                    "ok": r.get("ok", False),
                    "http_status": r.get("http_status"),
                    "error": r.get("error", ""),
                    "raw_response": r.get("raw_response", {}),
                }
                for r in provider_runs
            ],
            handle,
            indent=2,
            default=str,
        )

    provider_text_rel = f"runs/{run_id}/provider_model_text.txt"
    provider_text_path = run_dir / "provider_model_text.txt"
    with open(provider_text_path, "w", encoding="utf-8") as handle:
        for r in provider_runs:
            handle.write(f"===== RUN {r['run_index']} =====\n")
            handle.write(str(r.get("model_text", "")))
            handle.write("\n\n")

    provider_parsed_rel = f"runs/{run_id}/provider_parsed.json"
    provider_parsed_path = run_dir / "provider_parsed.json"
    with open(provider_parsed_path, "w", encoding="utf-8") as handle:
        json.dump(
            [
                {
                    "run_index": r["run_index"],
                    "ok": r.get("ok", False),
                    "parsed_json": r.get("parsed_json", {}),
                }
                for r in provider_runs
            ],
            handle,
            indent=2,
            default=str,
        )

    attempts: list[dict[str, Any]] = [
        {
            "provider": provider_used,
            "run_index": 1,
            "status": "ok" if boundary_result.get("ok") else "warn",
            "selected_count": str(len(line_segments)),
            "detail": boundary_result.get("error") or "gemini boundary detection success",
        }
    ]

    normalization_meta = {
        "mode": "gemini_global_marker_boundaries",
        "marker_first_mode": MARKER_FIRST_14_ONLY,
        "input_duct_segments": len(normalized_ducts),
        "accepted_duct_segments": len(duct_segments),
        "accepted_line_segments": len(crop_line_segments),
        "invalid_items": 0,
        "rejected_paths": 0,
        "short_or_duplicate_lines": 0,
        "issues": boundary_issues,
    }

    full_markers: list[dict[str, Any]] = []
    for marker in crop_markers:
        mx, my = marker["center"]
        bx1, by1, bx2, by2 = marker["bbox"]
        full_markers.append(
            {
                "id": marker["id"],
                "text": marker["text"],
                "center": [mx + rx1, my + ry1],
                "bbox": [bx1 + rx1, by1 + ry1, bx2 + rx1, by2 + ry1],
                "local_radius_px": float(marker.get("local_radius_px", 70.0)),
                "confidence": marker.get("confidence", 0.0),
            }
        )

    verification = _verify_line_segments(
        original,
        line_segments,
        roi,
        provider_error_summary,
        normalization_meta.get("issues", []),
        full_markers,
        duct_segments,
    )

    if provider_selection_warning:
        verification["reasons"].insert(0, provider_selection_warning)

    overlay = _draw_overlay(original, line_segments)
    cv2.imwrite(str(run_dir / "overlay.png"), overlay)

    warning_reasons = verification.get("reasons", [])
    result = {
        "run_id": run_id,
        "provider_requested": provider_requested or "gemini",
        "provider_used": provider_used,
        "provider_http_status": provider_result.get("http_status"),
        "provider_error": provider_error_summary,
        "marker_first_mode": MARKER_FIRST_14_ONLY,
        "detection_mode": detection_mode,
        "lsd_enabled": False,
        "gemini_enabled": True,
        "image_width": w,
        "image_height": h,
        "detection_count": len(line_segments),
        "duct_segment_count": len(duct_segments),
        "line_segment_count": len(line_segments),
        "raw_candidate_line_count": 0,
        "centerline_candidate_count": 0,
        "candidate_line_count": 0,
        "selected_raw_candidate_count": 0,
        "selected_candidate_count": len(line_segments),
        "size_marker_count": len(size_markers),
        "size_marker_seed_added_count": 0,
        "selection_mode": "gemini_global_marker_boundaries",
        "expansion_added_count": 0,
        "merged_centerline_count": 0,
        "merged_pairs_count": 0,
        "plan_roi": {"x1": rx1, "y1": ry1, "x2": rx2, "y2": ry2},
        "coordinate_space": {
            "normalized": "0..1000 relative to plan_crop.png",
            "pixel_line_segments": "full page image coordinates",
        },
        "marker_seeding": {
            "disabled": True,
            "reason": "gemini_global_marker_boundaries mode does not use LSD candidate seeding",
            "marker_count": len(size_markers),
            "seed_count": 0,
            "seed_added_count": 0,
            "per_marker": [],
        },
        "gemini_marker_picking": {
            "disabled": True,
            "reason": "replaced by one global Gemini boundary extraction call",
            "selection_mode": "gemini_global_marker_boundaries",
            "calls": 0,
            "success_count": 0,
            "failed_count": 0,
            "far_rejected_count": 0,
            "parse_issues": [],
            "per_marker": [],
        },
        "graph_expansion": {
            "disabled": True,
            "reason": "replaced by gemini_global_marker_boundaries",
            "expanded_count": 0,
            "component_count": 0,
            "kept_component_count": 0,
            "pruned_count": 0,
        },
        "ducts": duct_segments,
        "duct_segments": duct_segments,
        "line_segments": line_segments,
        "duct_lines": line_segments,
        "verification": verification,
        "normalization": normalization_meta,
        "raw_provider_response_path": raw_provider_response_rel,
        "warnings": warning_reasons,
        "attempts": attempts,
        "files": {
            "page": f"runs/{run_id}/page.png",
            "plan_crop": f"runs/{run_id}/plan_crop.png",
            "overlay": f"runs/{run_id}/overlay.png",
            "json": f"runs/{run_id}/result.json",
            "candidate_lines": candidate_lines_rel,
            "candidate_lines_raw": f"runs/{run_id}/candidate_lines_raw.json",
            "size_markers_raw": marker_raw_rel,
            "size_markers": marker_norm_rel,
            "gemini_boundaries_raw": gemini_boundaries_raw_rel,
            "gemini_boundaries_parsed": gemini_boundaries_parsed_rel,
            "raw_provider_response": raw_provider_response_rel,
            "provider_text": provider_text_rel,
            "provider_parsed": provider_parsed_rel,
        },
    }

    with open(run_dir / "result.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, default=str)

    return result
