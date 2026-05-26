from __future__ import annotations

import traceback
from pathlib import Path

import cv2
import numpy as np

from config import CAMERA_INDEX, WINDOW_NAME
from utils import put_lines

from board_detection import (
    detect_board_contour,
    approximate_polygon,
    polygon_to_points,
    order_hexagon_points,
    set_debug_prefix,
    draw_contour,
    generate_catan_tile_centers_from_hex,
)


SIDEBAR_WIDTH = 430


def open_camera(index: int):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)

    if not cap.isOpened():
        cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {index}")

    return cap


def process_board_geometry(image_bgr, prefix="hsv_debug"):
    set_debug_prefix(prefix)

    contour = detect_board_contour(image_bgr)
    polygon = approximate_polygon(contour, image_bgr)
    points = polygon_to_points(polygon)
    ordered_points = order_hexagon_points(points, image_bgr)

    centers = generate_catan_tile_centers_from_hex(
        ordered_points,
        image_bgr,
    )

    return {
        "contour": contour,
        "ordered_points": ordered_points,
        "centers": centers,
    }


def crop_board_view(image_bgr, ordered_points, margin=180, target_height=700):
    pts = np.asarray(ordered_points, dtype=np.float32)

    h, w = image_bgr.shape[:2]

    x1 = int(np.floor(np.min(pts[:, 0]) - margin))
    y1 = int(np.floor(np.min(pts[:, 1]) - margin))
    x2 = int(np.ceil(np.max(pts[:, 0]) + margin))
    y2 = int(np.ceil(np.max(pts[:, 1]) + margin))

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    crop = image_bgr[y1:y2, x1:x2].copy()

    if crop.size == 0:
        return image_bgr

    shifted_pts = pts.copy()
    shifted_pts[:, 0] -= x1
    shifted_pts[:, 1] -= y1

    mask = np.zeros(crop.shape[:2], dtype=np.uint8)

    cv2.fillConvexPoly(
        mask,
        np.round(shifted_pts).astype(np.int32),
        255,
    )

    crop = cv2.bitwise_and(crop, crop, mask=mask)

    ch, cw = crop.shape[:2]
    scale = target_height / float(ch)
    target_width = int(round(cw * scale))

    return cv2.resize(
        crop,
        (target_width, target_height),
        interpolation=cv2.INTER_LINEAR,
    )


def make_display_canvas(board_img, lines):
    h, w = board_img.shape[:2]

    canvas = np.zeros((h, w + SIDEBAR_WIDTH, 3), dtype=np.uint8)
    canvas[:, SIDEBAR_WIDTH:] = board_img

    panel = canvas[:, :SIDEBAR_WIDTH]

    put_lines(
        panel,
        lines,
        origin=(18, 36),
        line_height=34,
        scale=0.85,
        thickness=2,
        color=(255, 255, 255),
        bg=False,
    )

    return canvas


def crop_tile(image_bgr, x, y, size=42):
    h, w = image_bgr.shape[:2]

    x1 = max(0, int(x - size))
    y1 = max(0, int(y - size))
    x2 = min(w, int(x + size))
    y2 = min(h, int(y + size))

    return image_bgr[y1:y2, x1:x2].copy()


def build_inner_mask(h, w, radius_scale=0.30):
    yy, xx = np.mgrid[0:h, 0:w]

    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    r = min(h, w) * radius_scale

    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r


def extract_hsv_features(tile_patch_bgr):
    if tile_patch_bgr is None or tile_patch_bgr.size == 0:
        return {
            "h": 0.0,
            "s": 0.0,
            "v": 0.0,
            "green_frac": 0.0,
            "yellow_frac": 0.0,
            "red_frac": 0.0,
            "blue_frac": 0.0,
            "low_sat_frac": 0.0,
        }

    hsv = cv2.cvtColor(tile_patch_bgr, cv2.COLOR_BGR2HSV)
    h_img, w_img = hsv.shape[:2]

    mask = build_inner_mask(h_img, w_img, radius_scale=0.30)

    vals = hsv[mask]

    if len(vals) == 0:
        vals = hsv.reshape(-1, 3)

    vals = vals.astype(np.float32)

    v_lo = np.percentile(vals[:, 2], 8)
    v_hi = np.percentile(vals[:, 2], 92)

    keep = (vals[:, 2] >= v_lo) & (vals[:, 2] <= v_hi)
    vals = vals[keep] if np.count_nonzero(keep) >= 20 else vals

    h_med = float(np.median(vals[:, 0]))
    s_med = float(np.median(vals[:, 1]))
    v_med = float(np.median(vals[:, 2]))

    sat_mask = vals[:, 1] >= 45
    sat_vals = vals[sat_mask] if np.count_nonzero(sat_mask) >= 20 else vals

    hue = sat_vals[:, 0]
    sat = sat_vals[:, 1]

    green_frac = float(np.mean((hue >= 35) & (hue <= 85)))
    yellow_frac = float(np.mean((hue >= 16) & (hue <= 38)))
    red_frac = float(np.mean((hue <= 15) | (hue >= 170)))
    blue_frac = float(np.mean((hue >= 95) & (hue <= 130)))
    low_sat_frac = float(np.mean(sat < 60))

    return {
        "h": h_med,
        "s": s_med,
        "v": v_med,
        "green_frac": green_frac,
        "yellow_frac": yellow_frac,
        "red_frac": red_frac,
        "blue_frac": blue_frac,
        "low_sat_frac": low_sat_frac,
    }


def compute_all_hsv_features(image_bgr, centers, crop_size=42):
    features = [None] * len(centers)

    for tile_id, x, y in centers:
        tile_patch = crop_tile(image_bgr, x, y, size=crop_size)
        features[tile_id] = extract_hsv_features(tile_patch)

    return features


def draw_hsv_values_on_board(image_bgr, centers, features):
    img = image_bgr.copy()

    for tile_id, x, y in centers:
        f = features[tile_id]

        lines = [
            f"{tile_id}",
            f"H:{int(round(f['h']))}",
            f"S:{int(round(f['s']))}",
            f"V:{int(round(f['v']))}",
        ]

        cv2.circle(img, (x, y), 4, (0, 0, 255), -1)

        start_x = int(x - 22)
        start_y = int(y - 22)

        for i, txt in enumerate(lines):
            yy = start_y + i * 13

            cv2.putText(
                img,
                txt,
                (start_x, yy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.putText(
                img,
                txt,
                (start_x, yy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    return img


def print_hsv_table(centers, features):
    print("\n================ HSV TABLE ================")

    for tile_id, _, _ in centers:
        f = features[tile_id]

        print(
            f"Tile {tile_id:02d}: "
            f"H={f['h']:.1f}, "
            f"S={f['s']:.1f}, "
            f"V={f['v']:.1f}, "
            f"green={f['green_frac']:.2f}, "
            f"yellow={f['yellow_frac']:.2f}, "
            f"red={f['red_frac']:.2f}, "
            f"blue={f['blue_frac']:.2f}, "
            f"lowSat={f['low_sat_frac']:.2f}"
        )

    print("===========================================\n")


def main():
    cap = open_camera(CAMERA_INDEX)

    geom = None
    features = None
    last_message = "press B to capture board"

    print("Controls:")
    print("  B = capture / recapture board")
    print("  P = print HSV table")
    print("  Q or ESC = quit")
    print("")
    print("Important: click the OpenCV window before pressing keys.")

    while True:
        ok, frame = cap.read()

        if not ok:
            continue

        key = cv2.waitKey(1) & 0xFF
        key_chr = chr(key).lower() if key not in (255, 0) else ""

        try:
            if key_chr == "b":
                geom = process_board_geometry(frame, prefix="hsv_manual_capture")
                features = compute_all_hsv_features(frame, geom["centers"])

                last_message = "board captured, HSV shown"
                print("B pressed: board captured.")

            if geom is None:
                preview = frame.copy()

                put_lines(
                    preview,
                    [
                        "HSV Debug",
                        "Press B to capture board.",
                        "Then HSV values will appear.",
                        "Press P to print table.",
                        "",
                        f"Status: {last_message}",
                    ],
                    origin=(20, 40),
                    line_height=30,
                    scale=0.8,
                    thickness=2,
                )

                cv2.imshow(WINDOW_NAME + " - HSV Debug", preview)

                if key_chr == "q" or key == 27:
                    break

                continue

            centers = geom["centers"]

            if key_chr == "p":
                features = compute_all_hsv_features(frame, centers)
                print_hsv_table(centers, features)
                last_message = "HSV table printed"

            features = compute_all_hsv_features(frame, centers)

            overlay = frame.copy()

            overlay = draw_hsv_values_on_board(
                overlay,
                centers,
                features,
            )

            overlay = draw_contour(
                overlay,
                geom["ordered_points"],
            )

            zoomed_overlay = crop_board_view(
                overlay,
                geom["ordered_points"],
                margin=180,
                target_height=700,
            )

            lines = [
                "HSV Debug:",
                "Board geometry cached.",
                "No warp / no rotation.",
                "HSV values shown on board.",
                "",
                "B = recapture board",
                "P = print HSV table",
                "Q = quit",
                "",
                f"Status: {last_message}",
            ]

            canvas = make_display_canvas(zoomed_overlay, lines)

            cv2.imshow(WINDOW_NAME + " - HSV Debug", canvas)

        except Exception as exc:
            error_frame = frame.copy()

            put_lines(
                error_frame,
                [
                    "Runtime error",
                    str(exc),
                    "",
                    "Press B to recapture board.",
                    "Check terminal traceback.",
                ],
                origin=(20, 40),
                line_height=30,
                scale=0.8,
                thickness=2,
            )

            cv2.imshow(WINDOW_NAME + " - HSV Debug", error_frame)
            traceback.print_exc()

        if key_chr == "q" or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise