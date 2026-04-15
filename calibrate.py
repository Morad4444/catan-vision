from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from app import open_camera

# ── Config ────────────────────────────────────────────────────────────────────

SAVE_ROOT = Path(__file__).parent / "calibration"

_DIE_KEYS: dict[int, str] = {
    ord("1"): "yellow",
    ord("2"): "red",
    ord("3"): "white",
}

_WHITE_FACE_KEYS: dict[int, str] = {
    ord("g"): "green",
    ord("b"): "blue",
    ord("y"): "yellow_sym",
    ord("k"): "black",
}

# ── Mutable state ─────────────────────────────────────────────────────────────

_mouse_pos: tuple[int, int] = (0, 0)
_current_die: str = "yellow"
_current_white_face: str = "green"
_saved_counts: dict[str, int] = {"yellow": 0, "red": 0, "white": 0}
_flash_frames: int = 0          # countdown for on-screen SAVED flash
_flash_path: str = ""
_frame_counter: int = 0


def _mouse_callback(event: int, x: int, y: int, flags: int, param: object) -> None:
    global _mouse_pos
    _mouse_pos = (x, y)


# ── Save helper ───────────────────────────────────────────────────────────────

def _save_frame(frame: np.ndarray) -> Path | None:
    global _flash_frames, _flash_path
    if _current_die == "white":
        folder = SAVE_ROOT / "white" / _current_white_face
    else:
        folder = SAVE_ROOT / _current_die
    folder.mkdir(parents=True, exist_ok=True)
    idx = _saved_counts[_current_die]
    path = folder / f"{idx:04d}.jpg"
    # cv2.imwrite cannot handle non-ASCII paths on Windows; encode to buffer instead
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        print(f"ERROR: cv2.imencode failed for {path}")
        return None
    path.write_bytes(buf.tobytes())
    _saved_counts[_current_die] += 1
    _flash_frames = 45   # show flash for ~1.5 s at 30 fps
    _flash_path = str(path.relative_to(Path(__file__).parent))
    return path


# ── Overlay ───────────────────────────────────────────────────────────────────

def _draw_overlay(frame: np.ndarray) -> np.ndarray:
    global _flash_frames, _frame_counter
    out = frame.copy()
    h, w = out.shape[:2]

    # Frame counter (proves the feed is live)
    cv2.putText(out, f"#{_frame_counter}", (w - 90, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1, cv2.LINE_AA)
    _frame_counter += 1

    # Clamp cursor to frame
    cx = max(0, min(_mouse_pos[0], w - 1))
    cy = max(0, min(_mouse_pos[1], h - 1))

    # HSV value at cursor pixel
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, sat, val = (int(v) for v in hsv_frame[cy, cx])

    # Colour swatch matching the HSV under cursor
    swatch_bgr = cv2.cvtColor(
        np.array([[[hue, sat, val]]], dtype=np.uint8), cv2.COLOR_HSV2BGR
    )[0, 0].tolist()

    # ── Status bar background ─────────────────────────────────────────────────
    bar_h = 85
    cv2.rectangle(out, (0, 0), (w, bar_h), (25, 25, 25), -1)

    # Line 1 – selected die
    die_label = _current_die.capitalize()
    if _current_die == "white":
        die_label += f"  /  face symbol: {_current_white_face}"
    counts = "   ".join(
        f"{k[0].upper()}:{v}" for k, v in _saved_counts.items()
    )
    cv2.putText(
        out, f"Die: {die_label}   |   Saved — {counts}",
        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
    )

    # Line 2 – HSV readout + colour swatch
    hsv_text = f"HSV at cursor:  H={hue}  S={sat}  V={val}"
    cv2.putText(
        out, hsv_text,
        (10, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 180), 1, cv2.LINE_AA,
    )
    cv2.rectangle(out, (w - 50, 5), (w - 5, 55), swatch_bgr, -1)
    cv2.rectangle(out, (w - 50, 5), (w - 5, 55), (200, 200, 200), 1)

    # Line 3 – key hints
    hint = "1=Yellow  2=Red  3=White  |  g/b/y/k = white face symbol  |  Space=save  q=quit"
    cv2.putText(
        out, hint,
        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1, cv2.LINE_AA,
    )

    # ── Crosshair at cursor ───────────────────────────────────────────────────
    cv2.drawMarker(out, (cx, cy), (0, 230, 180), cv2.MARKER_CROSS, 24, 1, cv2.LINE_AA)

    # ── SAVED flash ──────────────────────────────────────────────────────────
    if _flash_frames > 0:
        alpha = min(1.0, _flash_frames / 15.0)   # fade out in last 15 frames
        overlay = out.copy()
        cv2.rectangle(overlay, (w // 2 - 220, h // 2 - 45), (w // 2 + 220, h // 2 + 45), (0, 180, 0), -1)
        cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
        cv2.putText(out, f"SAVED  {_flash_path}",
                    (w // 2 - 210, h // 2 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        _flash_frames -= 1

    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    global _current_die, _current_white_face

    capture, cam_idx = open_camera(0, 1280, 720, "Logitech C270")
    if not capture.isOpened():
        raise RuntimeError("Could not open the webcam.")

    print(f"Using camera index {cam_idx}.")
    print()
    print("  Hover over any area to read its HSV values in the status bar.")
    print("  Use those values to refine the HSV ranges in app.py.")
    print()
    print("  Keys:")
    print("    1 / 2 / 3  — select Yellow / Red / White die")
    print("    g / b / y / k  — label white die face (green / blue / yellow / black symbol)")
    print("    Space      — save current frame")
    print("    q          — quit")
    print()
    print(f"  Images are saved under:  {SAVE_ROOT}")
    print()
    print("  >>> Click the camera window first, then press keys. <<<")

    window = "Calibration  —  hover to read HSV  |  CLICK HERE first"  
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, _mouse_callback)

    while True:
        ok, frame = capture.read()
        if not ok:
            print("Could not read frame from webcam.")
            break

        cv2.imshow(window, _draw_overlay(frame))
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key in _DIE_KEYS:
            _current_die = _DIE_KEYS[key]
            print(f"Selected die:  {_current_die}")
        elif key in _WHITE_FACE_KEYS:
            _current_white_face = _WHITE_FACE_KEYS[key]
            if _current_die == "white":
                print(f"White die face symbol colour:  {_current_white_face}")
        elif key == ord(" "):
            saved_path = _save_frame(frame)
            if saved_path:
                print(f"Saved  →  {saved_path}")
            else:
                print("Save FAILED — check folder permissions.")

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
