from __future__ import annotations

import argparse
import platform
from dataclasses import dataclass

import cv2
import numpy as np

try:
    from pygrabber.dshow_graph import FilterGraph
except ImportError:
    FilterGraph = None


# ── HSV colour ranges  (H: 0-180, S: 0-255, V: 0-255) ────────────────────────

# Die body colours
_YELLOW_BODY = [((15, 100, 100), (35, 255, 255))]
_RED_BODY    = [((0, 120, 70), (10, 255, 255)), ((160, 120, 70), (180, 255, 255))]
_WHITE_BODY  = [((0, 0, 170), (180, 50, 255))]

# Pip colours (yellow die → red pips; red die → yellow pips)
_RED_PIP    = [((0, 100, 70), (10, 255, 255)), ((160, 100, 70), (180, 255, 255))]
_YELLOW_PIP = [((15, 100, 100), (35, 255, 255))]

# Symbol colours for the white die
_GREEN_SYM  = [((40, 60, 60),  (80, 255, 255))]
_BLUE_SYM   = [((90, 80, 60),  (130, 255, 255))]
_YELLOW_SYM = [((15, 80, 80),  (35, 255, 255))]
_BLACK_SYM  = [((0, 0, 0),     (180, 255, 70))]

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

    def detect(self, frame: np.ndarray) -> list[DieDetection]:
        results: list[DieDetection] = []
        for die_type in _DIE_TYPES:
            contour = self._best_contour(frame, die_type["body"])
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

def draw_detections(frame: np.ndarray, detections: list[DieDetection]) -> np.ndarray:
    out = frame.copy()
    found_labels = {d.label for d in detections}

    for det in detections:
        color = next(t["color"] for t in _DIE_TYPES if t["label"] == det.label)
        cv2.polylines(out, [cv2.convexHull(det.contour)], True, color, 2)

        v = det.value if det.value is not None else "?"
        label = f"{det.label}: {v}"
        org = (det.center[0] - 55, det.center[1])
        cv2.putText(out, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 10), 3, cv2.LINE_AA)
        cv2.putText(out, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

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
        if "c270" in n and ("logitech" in n or "logi" in n):
            return i
    return None


def open_camera(camera_index: int, width: int, height: int, preferred_name: str | None = None) -> tuple[cv2.VideoCapture, int]:
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
    return cap, idx


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detect three unique dice from a USB webcam.")
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--camera-name",  type=str, default="Logitech C270")
    p.add_argument("--width",        type=int, default=1280)
    p.add_argument("--height",       type=int, default=720)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    detector = DiceDetector()
    capture, selected_index = open_camera(args.camera_index, args.width, args.height, args.camera_name)

    if not capture.isOpened():
        raise RuntimeError("Could not open the webcam. Check the USB connection and camera index.")

    print(f"Using camera index {selected_index}.")
    print("Press 'q' to quit.")

    while True:
        ok, frame = capture.read()
        if not ok:
            print("Could not read a frame from the webcam.")
            break
        cv2.imshow("Dice Detection", draw_detections(frame, detector.detect(frame)))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()