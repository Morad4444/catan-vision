from __future__ import annotations

import argparse
import collections
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

try:
    from pygrabber.dshow_graph import FilterGraph
except ImportError:
    FilterGraph = None


# ── HSV colour ranges  (H: 0-180, S: 0-255, V: 0-255) ────────────────────────

# Die body colours
_YELLOW_BODY = [((22, 60, 120), (34, 255, 255))]
_RED_BODY    = [((0, 120, 70), (20, 255, 255))]
_WHITE_BODY  = [((0, 0, 170), (180, 50, 255))]

# Pip colours (yellow die → red pips; red die → yellow pips)
_RED_PIP    = [((0, 100, 70), (10, 255, 255)), ((160, 100, 70), (180, 255, 255))]
_YELLOW_PIP = [((15, 0, 0), (35, 255, 255))]

# Symbol colours for the white die
_GREEN_SYM  = [((40, 60, 60),  (80, 255, 255))]
_BLUE_SYM   = [((90, 80, 60),  (130, 255, 255))]
_YELLOW_SYM = [((10, 60, 60),  (35, 255, 255))]
_BLACK_SYM  = [((0, 0, 0),     (180, 255, 70))]

# Brown tray range (tune if lighting changes)
_TRAY_BROWN = [((1, 30, 10), (18, 200, 80))]
_TRAY_CTRL_WINDOW = "Tray HSV Controls"

# Flat side of the tray ROI: "bottom" | "top" | "left" | "right"
# The circle will be clipped by a chord at this side.
_TRAY_FLAT_SIDE = "bottom"
# Scale applied to the detected tray radius so the ROI tracks the tray interior, not the outer rim.
_TRAY_RADIUS_SCALE = 0.91
# How far the chord cuts in from the edge of the inscribed circle (0.0 = no cut, 1.0 = half)
_TRAY_FLAT_DEPTH = 0.24

# Debug HSV picker state
_DEBUG_HSV_STATE = {
    "current_frame_hsv": None,
    "last_clicked_hsv": None,
    "hover_hsv": None,
    "hover_pos": None,
    "enable": False
}


def _noop(_: int) -> None:
    pass


def _make_flat_circle_mask(
    shape: tuple[int, int],
    cx: int,
    cy: int,
    r: int,
    flat_side: str = _TRAY_FLAT_SIDE,
    flat_depth: float = _TRAY_FLAT_DEPTH,
) -> np.ndarray:
    """Return a mask that is a filled circle with one side cut flat (D-shape)."""
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    cut = max(0, min(r, int(r * flat_depth)))
    if cut == 0:
        return mask

    if flat_side == "bottom":
        y_chord = cy + r - cut
        half_width = int(np.sqrt(max(0, r * r - (y_chord - cy) ** 2)))
        pts = np.array([
            [cx - half_width, y_chord],
            [cx + half_width, y_chord],
            [cx + half_width, cy + r],
            [cx - half_width, cy + r],
        ], dtype=np.int32)
    elif flat_side == "top":
        y_chord = cy - r + cut
        half_width = int(np.sqrt(max(0, r * r - (y_chord - cy) ** 2)))
        pts = np.array([
            [cx - half_width, cy - r],
            [cx + half_width, cy - r],
            [cx + half_width, y_chord],
            [cx - half_width, y_chord],
        ], dtype=np.int32)
    elif flat_side == "left":
        x_chord = cx - r + cut
        half_height = int(np.sqrt(max(0, r * r - (x_chord - cx) ** 2)))
        pts = np.array([
            [cx - r, cy - half_height],
            [x_chord, cy - half_height],
            [x_chord, cy + half_height],
            [cx - r, cy + half_height],
        ], dtype=np.int32)
    else:  # right
        x_chord = cx + r - cut
        half_height = int(np.sqrt(max(0, r * r - (x_chord - cx) ** 2)))
        pts = np.array([
            [x_chord, cy - half_height],
            [cx + r, cy - half_height],
            [cx + r, cy + half_height],
            [x_chord, cy + half_height],
        ], dtype=np.int32)

    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    cv2.fillPoly(mask, [pts], 0)
    return mask


def _on_mouse_click(event: int, x: int, y: int, flags: int, param: tuple) -> None:
    """Mouse callback to read HSV value at cursor position (hover + click)."""
    if not _DEBUG_HSV_STATE["enable"]:
        return

    hsv_frame = _DEBUG_HSV_STATE["current_frame_hsv"]
    if hsv_frame is None:
        return

    h_img, w_img = hsv_frame.shape[:2]
    if x < 0 or x >= w_img or y < 0 or y >= h_img:
        return

    h_val = int(hsv_frame[y, x, 0])
    s_val = int(hsv_frame[y, x, 1])
    v_val = int(hsv_frame[y, x, 2])

    # Always update live hover readout
    _DEBUG_HSV_STATE["hover_hsv"] = (h_val, s_val, v_val)
    _DEBUG_HSV_STATE["hover_pos"] = (x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        _DEBUG_HSV_STATE["last_clicked_hsv"] = (h_val, s_val, v_val)
        print(f"Clicked HSV at ({x}, {y}): H={h_val}, S={s_val}, V={v_val}")


def _init_tray_hsv_controls() -> None:
    cv2.namedWindow(_TRAY_CTRL_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(_TRAY_CTRL_WINDOW, 420, 220)
    lo, hi = _TRAY_BROWN[0]
    cv2.createTrackbar("H min", _TRAY_CTRL_WINDOW, lo[0], 180, _noop)
    cv2.createTrackbar("H max", _TRAY_CTRL_WINDOW, hi[0], 180, _noop)
    cv2.createTrackbar("S min", _TRAY_CTRL_WINDOW, lo[1], 255, _noop)
    cv2.createTrackbar("S max", _TRAY_CTRL_WINDOW, hi[1], 255, _noop)
    cv2.createTrackbar("V min", _TRAY_CTRL_WINDOW, lo[2], 255, _noop)
    cv2.createTrackbar("V max", _TRAY_CTRL_WINDOW, hi[2], 255, _noop)


def _read_tray_hsv_controls() -> list[tuple[tuple[int, int, int], tuple[int, int, int]]]:
    try:
        if cv2.getWindowProperty(_TRAY_CTRL_WINDOW, cv2.WND_PROP_VISIBLE) < 0:
            _init_tray_hsv_controls()

        h_min = cv2.getTrackbarPos("H min", _TRAY_CTRL_WINDOW)
        h_max = cv2.getTrackbarPos("H max", _TRAY_CTRL_WINDOW)
        s_min = cv2.getTrackbarPos("S min", _TRAY_CTRL_WINDOW)
        s_max = cv2.getTrackbarPos("S max", _TRAY_CTRL_WINDOW)
        v_min = cv2.getTrackbarPos("V min", _TRAY_CTRL_WINDOW)
        v_max = cv2.getTrackbarPos("V max", _TRAY_CTRL_WINDOW)
    except cv2.error:
        return list(_TRAY_BROWN)

    lo = (min(h_min, h_max), min(s_min, s_max), min(v_min, v_max))
    hi = (max(h_min, h_max), max(s_min, s_max), max(v_min, v_max))
    return [(lo, hi)]

# One entry per physical die; drives both detection and analysis
_DIE_TYPES: list[dict] = [
    {"label": "Yellow", "body": _YELLOW_BODY, "pip_ranges": _RED_PIP,    "symbol": False, "color": (0, 200, 255)},
    {"label": "Red",    "body": _RED_BODY,    "pip_ranges": _YELLOW_PIP, "symbol": False, "color": (0, 0, 220)},
    {"label": "White",  "body": _WHITE_BODY,  "pip_ranges": None,        "symbol": True,  "color": (200, 200, 200)},
]


# ── Dataclass ─────────────────────────────────────────────────────────────────

@dataclass
class DieDetection:
    label: str                    # "Yellow" | "Red" | "White"
    contour: np.ndarray
    center: tuple[int, int]
    area: float
    value: int | str | None       # 1-6 for pipped dice; colour name for white die
    best_guess: int | str | None = field(default=None)  # most common value over recent frames


class ValueHistory:
    """Tracks per-die value detections over a rolling window and returns the mode."""

    def __init__(self, window: int = 30) -> None:
        self._window = window
        # label -> deque of recent values (None entries excluded from vote)
        self._history: dict[str, collections.deque] = {}

    def update(self, label: str, value: int | str | None) -> int | str | None:
        """Record a new observation and return the current best guess."""
        if label not in self._history:
            self._history[label] = collections.deque(maxlen=self._window)
        if value is not None:
            self._history[label].append(value)
        buf = self._history[label]
        if not buf:
            return None
        counter = collections.Counter(buf)
        return counter.most_common(1)[0][0]

    def reset(self, label: str) -> None:
        """Clear history for a die (e.g. when it leaves the frame)."""
        self._history.pop(label, None)


# ── Colour helpers ────────────────────────────────────────────────────────────

def _hsv_mask(hsv: np.ndarray, ranges: list[tuple]) -> np.ndarray:
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
    return mask


def _center_roi(img: np.ndarray, margin_frac: float = 0.15) -> np.ndarray:
    """Crop away the border fraction on all four sides."""
    m = int(img.shape[0] * margin_frac)
    h, w = img.shape[:2]
    return img[m:h - m, m:w - m]


def _detect_tray_circle(frame: np.ndarray) -> tuple[int, int, int] | None:
    """Return (cx, cy, r) of the round tray interior if found."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    tray_mask = _hsv_mask(hsv, _TRAY_BROWN)
    tray_mask = cv2.morphologyEx(tray_mask, cv2.MORPH_CLOSE, np.ones((9, 9), dtype=np.uint8), iterations=2)
    tray_mask = cv2.morphologyEx(tray_mask, cv2.MORPH_OPEN, np.ones((5, 5), dtype=np.uint8), iterations=1)

    contours, _ = cv2.findContours(tray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h, w = frame.shape[:2]
    frame_area = float(h * w)
    best: tuple[int, int, int] | None = None
    best_score = -1.0

    for c in contours:
        area = cv2.contourArea(c)
        if area < frame_area * 0.08:
            continue
        p = cv2.arcLength(c, True)
        if p <= 0:
            continue
        circularity = 4.0 * np.pi * area / (p * p)
        if circularity < 0.45:
            continue

        (cx, cy), r = cv2.minEnclosingCircle(c)
        if r < min(h, w) * 0.18:
            continue

        score = area * circularity
        if score > best_score:
            best_score = score
            best = (int(cx), int(cy), int(r))

    return best


# ── Pip / symbol analysis ─────────────────────────────────────────────────────

def _count_circular_blobs(binary: np.ndarray) -> int | None:
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total = binary.shape[0] * binary.shape[1]
    count = 0
    for c in contours:
        a = cv2.contourArea(c)
        if a < total * 0.003 or a > total * 0.12:
            continue
        p = cv2.arcLength(c, True)
        if p > 0 and (4 * np.pi * a / p ** 2) >= 0.35:
            count += 1
    return count if 1 <= count <= 6 else None


def count_colored_pips(warped: np.ndarray, pip_ranges: list[tuple]) -> int | None:
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    return _count_circular_blobs(_center_roi(_hsv_mask(hsv, pip_ranges)))


def detect_white_die_symbol(warped: np.ndarray) -> str | None:
    """Return the dominant symbol colour visible on the white die face."""
    hsv_roi = _center_roi(cv2.cvtColor(warped, cv2.COLOR_BGR2HSV))
    min_px = int(hsv_roi.shape[0] * hsv_roi.shape[1] * 0.025)
    scores = {
        "green":  cv2.countNonZero(_hsv_mask(hsv_roi, _GREEN_SYM)),
        "blue":   cv2.countNonZero(_hsv_mask(hsv_roi, _BLUE_SYM)),
        "yellow": cv2.countNonZero(_hsv_mask(hsv_roi, _YELLOW_SYM)),
        "black":  cv2.countNonZero(_hsv_mask(hsv_roi, _BLACK_SYM)),
    }
    valid = {k: v for k, v in scores.items() if v >= min_px}
    return max(valid, key=valid.get) if valid else None


# ── Geometric helpers ─────────────────────────────────────────────────────────

def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    sums, diffs = pts.sum(axis=1), np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(sums)]
    rect[2] = pts[np.argmax(sums)]
    rect[1] = pts[np.argmin(diffs)]
    rect[3] = pts[np.argmax(diffs)]
    return rect


def _warp_die(frame: np.ndarray, corners: np.ndarray, size: int = 220) -> np.ndarray:
    ordered = _order_points(corners.astype(np.float32))
    dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=np.float32)
    return cv2.warpPerspective(frame, cv2.getPerspectiveTransform(ordered, dst), (size, size))


def _contour_center(contour: np.ndarray) -> tuple[int, int]:
    M = cv2.moments(contour)
    if M["m00"] == 0:
        x, y, w, h = cv2.boundingRect(contour)
        return x + w // 2, y + h // 2
    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])


# ── Detector ──────────────────────────────────────────────────────────────────

class DiceDetector:
    _CLOSE = np.ones((7, 7), dtype=np.uint8)
    _OPEN  = np.ones((5, 5), dtype=np.uint8)
    _MIN_AREA_FRAC = 0.003
    _MAX_AREA_FRAC = 0.04
    _MAX_ASPECT_RATIO = 1.55
    _MIN_RECT_FILL = 0.58
    _MAX_SQUARE_SHAPE_SCORE = 0.16
    _TRAY_KEEP_FRAMES = 45

    def __init__(self) -> None:
        self.last_tray_circle: tuple[int, int, int] | None = None
        self.last_tray_mask: np.ndarray | None = None
        self.tray_brown_ranges: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = list(_TRAY_BROWN)
        self._tray_miss_count = 0

    def _tray_mask(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        brown_mask = _hsv_mask(hsv, self.tray_brown_ranges)
        brown_mask = cv2.morphologyEx(brown_mask, cv2.MORPH_CLOSE, np.ones((9, 9), dtype=np.uint8), iterations=2)
        brown_mask = cv2.morphologyEx(brown_mask, cv2.MORPH_OPEN, np.ones((5, 5), dtype=np.uint8), iterations=1)

        contours, _ = cv2.findContours(brown_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > (h * w * 0.05):
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(mask, [largest], -1, 255, thickness=-1)
                self.last_tray_mask = mask
                self.last_tray_circle = None
                self._tray_miss_count = 0
                return mask

        if self.last_tray_mask is not None:
            self._tray_miss_count += 1
            if self._tray_miss_count <= self._TRAY_KEEP_FRAMES:
                return self.last_tray_mask
            self.last_tray_mask = None

        self.last_tray_circle = None
        return np.zeros((h, w), dtype=np.uint8)

    def _is_die_like_contour(self, contour: np.ndarray, frame_area: float) -> bool:
        area = cv2.contourArea(contour)
        if area < frame_area * self._MIN_AREA_FRAC or area > frame_area * self._MAX_AREA_FRAC:
            return False

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            return False

        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        if len(approx) < 4 or len(approx) > 8:
            return False

        rect = cv2.minAreaRect(contour)
        w_r, h_r = rect[1]
        if w_r == 0 or h_r == 0:
            return False

        aspect_ratio = max(w_r, h_r) / min(w_r, h_r)
        if aspect_ratio > self._MAX_ASPECT_RATIO:
            return False

        rect_area = w_r * h_r
        fill_ratio = area / max(rect_area, 1.0)
        if fill_ratio < self._MIN_RECT_FILL:
            return False

        # Compare contour against its best-fit rectangle to reject irregular large planes.
        square_contour = cv2.boxPoints(rect).reshape((-1, 1, 2)).astype(np.float32)
        shape_score = cv2.matchShapes(contour, square_contour, cv2.CONTOURS_MATCH_I1, 0.0)
        if shape_score > self._MAX_SQUARE_SHAPE_SCORE:
            return False

        return True

    def _best_contour(self, frame: np.ndarray, body_ranges: list[tuple]) -> np.ndarray | None:
        """Find the largest blob matching the body colour that looks like a die face."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = _hsv_mask(hsv, body_ranges)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._CLOSE, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._OPEN,  iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_area = frame.shape[0] * frame.shape[1]
        best, best_area = None, 0.0
        for c in contours:
            if not self._is_die_like_contour(c, frame_area):
                continue

            a = cv2.contourArea(c)
            if a > best_area:
                best_area, best = a, c
        return best

    def detect(self, frame: np.ndarray, use_roi: bool = True) -> list[DieDetection]:
        results: list[DieDetection] = []
        if use_roi:
            tray_mask = self._tray_mask(frame)
            masked_frame = cv2.bitwise_and(frame, frame, mask=tray_mask)
        else:
            masked_frame = frame

        for die_type in _DIE_TYPES:
            contour = self._best_contour(masked_frame, die_type["body"])
            if contour is None:
                continue

            corners = cv2.boxPoints(cv2.minAreaRect(contour.astype(np.float32))).astype(np.int32)
            warped = _warp_die(frame, corners)

            value: int | str | None
            if die_type["symbol"]:
                value = detect_white_die_symbol(warped)
            else:
                value = count_colored_pips(warped, die_type["pip_ranges"])

            results.append(DieDetection(
                label=die_type["label"],
                contour=contour,
                center=_contour_center(contour),
                area=cv2.contourArea(contour),
                value=value,
            ))

        results.sort(key=lambda d: (d.center[1], d.center[0]))
        return results


# ── Overlay ───────────────────────────────────────────────────────────────────

def draw_detections(
    frame: np.ndarray,
    detections: list[DieDetection],
    tray_circle: tuple[int, int, int] | None = None,
    tray_mask: np.ndarray | None = None,
    use_roi: bool = True,
) -> np.ndarray:
    if use_roi and tray_mask is not None:
        out = cv2.bitwise_and(frame, frame, mask=tray_mask)
    else:
        out = frame.copy()

    found_labels = {d.label for d in detections}

    for det in detections:
        color = next(t["color"] for t in _DIE_TYPES if t["label"] == det.label)
        cv2.polylines(out, [cv2.convexHull(det.contour)], True, color, 2)

        v = det.value if det.value is not None else "?"
        bg = det.best_guess if det.best_guess is not None else "?"
        live_label = f"{det.label}: {v}"
        best_label = f"  best: {bg}"
        org_live = (det.center[0] - 55, det.center[1] - 10)
        org_best = (det.center[0] - 55, det.center[1] + 18)
        cv2.putText(out, live_label, org_live, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (10, 10, 10), 3, cv2.LINE_AA)
        cv2.putText(out, live_label, org_live, cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
        cv2.putText(out, best_label, org_best, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (10, 10, 10), 3, cv2.LINE_AA)
        cv2.putText(out, best_label, org_best, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    missing = [t["label"] for t in _DIE_TYPES if t["label"] not in found_labels]
    status = "Detected: " + (", ".join(found_labels) if found_labels else "none")
    if missing:
        status += "  |  Not found: " + ", ".join(missing)
    cv2.putText(out, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (10, 10, 10), 3, cv2.LINE_AA)
    cv2.putText(out, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return out


# ── Camera helpers ────────────────────────────────────────────────────────────

def camera_name_matches(device_name: str, preferred_name: str) -> bool:
    normalized = device_name.casefold()
    aliases = {
        "logitech": {"logitech", "logi"},
        "logi":     {"logitech", "logi"},
        "webcam":   {"webcam", "camera", "cam"},
        "camera":   {"webcam", "camera", "cam"},
    }
    for token in preferred_name.casefold().split():
        group = aliases.get(token, {token})
        if not any(a in normalized for a in group):
            return False
    return True


def find_preferred_camera_index(preferred_name: str) -> int | None:
    if platform.system() != "Windows" or FilterGraph is None:
        return None
    try:
        devices = FilterGraph().get_input_devices()
    except Exception:
        return None
    for i, name in enumerate(devices):
        if camera_name_matches(name, preferred_name):
            return i
    for i, name in enumerate(devices):
        n = name.casefold()
        if "hd usb camera" in n:
            return i
    return None


def open_camera(
    camera_index: int,
    width: int,
    height: int,
    preferred_name: str | None = None,
    exposure: float | None = -3.0,
) -> tuple[cv2.VideoCapture, int]:
    idx = camera_index
    if preferred_name and camera_index == 0:
        preferred = find_preferred_camera_index(preferred_name)
        if preferred is not None:
            idx = preferred
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # On many Windows webcams, manual exposure requires setting AUTO_EXPOSURE first.
    # DSHOW commonly uses 0.25 = manual and exposure values around -4 .. -10.
    if exposure is not None:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure))

    return cap, idx


def set_manual_exposure(cap: cv2.VideoCapture, exposure: float) -> None:
    """Switch camera to manual exposure mode and apply exposure value."""
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure))


def set_auto_exposure(cap: cv2.VideoCapture) -> None:
    """Switch camera to auto exposure mode (best-effort across webcam drivers)."""
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detect three unique dice from a USB webcam.")
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--camera-name",  type=str, default="HD USB Camera")
    p.add_argument("--width",        type=int, default=1280)
    p.add_argument("--height",       type=int, default=720)
    p.add_argument(
        "--exposure",
        type=float,
        default=-3.0,
        help="Manual camera exposure (commonly -4 to -10 on Windows webcams).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    detector = DiceDetector()
    history = ValueHistory(window=30)  # ~1 s at 30 fps
    capture, selected_index = open_camera(
        args.camera_index,
        args.width,
        args.height,
        args.camera_name,
        args.exposure,
    )

    if not capture.isOpened():
        raise RuntimeError("Could not open the webcam. Check the USB connection and camera index.")

    print(f"Using camera index {selected_index}.")
    print(f"Requested manual exposure: {args.exposure}")
    print("Press 'q' to quit.")
    print("Press 'd' to toggle HSV debug mode (click pixels to see HSV).")
    print("Press 'r' to toggle tray ROI masking.")
    print("Press 'e' to step exposure darker (-0.5).")
    print("Press 'w' to step exposure brighter (+0.5).")
    _init_tray_hsv_controls()
    mouse_callback_registered = False
    roi_enabled = True
    current_exposure = round(float(args.exposure) * 2.0) / 2.0
    current_exposure = max(-10.0, min(-1.0, current_exposure))
    set_manual_exposure(capture, current_exposure)

    while True:
        ok, frame = capture.read()
        if not ok:
            print("Could not read a frame from the webcam.")
            break
        detector.tray_brown_ranges = _read_tray_hsv_controls()
        detections = detector.detect(frame, use_roi=roi_enabled)

        # Update best-guess history; clear history for dice not seen this frame
        detected_labels = {d.label for d in detections}
        for die_type in _DIE_TYPES:
            if die_type["label"] not in detected_labels:
                history.reset(die_type["label"])
        for det in detections:
            det.best_guess = history.update(det.label, det.value)

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        _DEBUG_HSV_STATE["current_frame_hsv"] = hsv_frame

        display = draw_detections(frame, detections, detector.last_tray_circle, detector.last_tray_mask, use_roi=roi_enabled)

        # ROI status indicator
        roi_text = "ROI: ON" if roi_enabled else "ROI: OFF"
        roi_color = (0, 220, 0) if roi_enabled else (0, 80, 220)
        cv2.putText(display, roi_text, (display.shape[1] - 130, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (10, 10, 10), 3, cv2.LINE_AA)
        cv2.putText(display, roi_text, (display.shape[1] - 130, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, roi_color, 2, cv2.LINE_AA)

        exp_text = f"EXP: {current_exposure:.1f}  (E darker, W brighter)"
        exp_color = (0, 220, 0)
        cv2.putText(display, exp_text, (display.shape[1] - 250, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (10, 10, 10), 3, cv2.LINE_AA)
        cv2.putText(display, exp_text, (display.shape[1] - 250, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.58, exp_color, 2, cv2.LINE_AA)
        
        # Draw live HSV overlay when debug mode is on
        if _DEBUG_HSV_STATE["enable"]:
            img_h = display.shape[0]
            # Live hover readout near the cursor
            hover = _DEBUG_HSV_STATE["hover_hsv"]
            pos   = _DEBUG_HSV_STATE["hover_pos"]
            if hover and pos:
                h_v, s_v, v_v = hover
                px, py = pos
                tip = f"H={h_v}  S={s_v}  V={v_v}"
                tx = min(px + 12, display.shape[1] - 160)
                ty = max(py - 12, 20)
                cv2.putText(display, tip, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0),   2, cv2.LINE_AA)
                cv2.putText(display, tip, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.drawMarker(display, (px, py), (0, 255, 255), cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)
            # Pinned last-clicked value at bottom
            if _DEBUG_HSV_STATE["last_clicked_hsv"]:
                h_v, s_v, v_v = _DEBUG_HSV_STATE["last_clicked_hsv"]
                text = f"Pinned (click): H={h_v}, S={s_v}, V={v_v}"
            else:
                text = "HSV debug ON — hover to read, click to pin"
            cv2.putText(display, text, (20, img_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0),     2, cv2.LINE_AA)
            cv2.putText(display, text, (20, img_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)
        
        cv2.imshow("Dice Detection", display)
        
        # Register mouse callback after first window creation
        if not mouse_callback_registered:
            cv2.setMouseCallback("Dice Detection", _on_mouse_click)
            mouse_callback_registered = True
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            roi_enabled = not roi_enabled
            print(f"Tray ROI: {'ON' if roi_enabled else 'OFF'}")
        elif key == ord("d"):
            _DEBUG_HSV_STATE["enable"] = not _DEBUG_HSV_STATE["enable"]
            print(f"HSV debug mode: {'ON (click on pixels)' if _DEBUG_HSV_STATE['enable'] else 'OFF'}")
            # Clear HSV display when toggling off
            if not _DEBUG_HSV_STATE["enable"]:
                _DEBUG_HSV_STATE["last_clicked_hsv"] = None
                _DEBUG_HSV_STATE["hover_hsv"] = None
                _DEBUG_HSV_STATE["hover_pos"] = None
        elif key == ord("e"):
            current_exposure -= 0.5
            if current_exposure < -10.0:
                current_exposure = -1.0
            set_manual_exposure(capture, current_exposure)
            print(f"Exposure set to {current_exposure:.1f}")
        elif key == ord("w"):
            current_exposure += 0.5
            if current_exposure > -1.0:
                current_exposure = -10.0
            set_manual_exposure(capture, current_exposure)
            print(f"Exposure set to {current_exposure:.1f}")

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()