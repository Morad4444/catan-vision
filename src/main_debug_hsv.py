from __future__ import annotations

import traceback
import time
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
    normalize_hexagon,
)
from chip_detection import (
    detect_chips,
    assign_chips_to_tiles,
    draw_chips,
    save_chip_debug_patches,
)
from number_detection_debug import (
    generate_random_number_layout,
    create_manual_board_state,
    analyze_chip_identities,
    detect_pair_swaps,
    apply_detected_swaps,
    print_swap_detected,
    refresh_pending_reference_edges,
)
from tile_classification_hsv_debug import classify_resources, draw_tile_hsv_values


SIDEBAR_WIDTH = 430
CHIP_DEBUG_DIR = Path("data/output/chip_debug_hsv")
STATE_DIR = Path("data/output/board_state_hsv_debug")


def open_camera(index: int):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {index}")
    return cap


def crop_live_roi(frame):
    h, w = frame.shape[:2]
    return frame.copy(), (0, 0, w, h)


def process_board_geometry(image_bgr, prefix="live_hsv_debug"):
    set_debug_prefix(prefix)
    contour = detect_board_contour(image_bgr)
    polygon = approximate_polygon(contour, image_bgr)
    points = polygon_to_points(polygon)
    ordered_points = order_hexagon_points(points, image_bgr)
    normalized_img, normalized_points = normalize_hexagon(image_bgr, ordered_points)
    centers = generate_catan_tile_centers_from_hex(normalized_points, normalized_img)

    return {
        "contour": contour,
        "ordered_points": ordered_points,
        "normalized_img": normalized_img,
        "normalized_points": normalized_points,
        "centers": centers,
    }


def polygon_mean_point(points):
    pts = np.asarray(points, dtype=np.float32)
    return np.mean(pts, axis=0)


def polygon_average_corner_shift(poly_a, poly_b):
    a = np.asarray(poly_a, dtype=np.float32)
    b = np.asarray(poly_b, dtype=np.float32)
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


def accept_new_geometry(new_geom, last_good_geom, max_center_shift=55.0, max_corner_shift=70.0):
    if last_good_geom is None:
        return True

    prev_poly = last_good_geom["ordered_points"]
    new_poly = new_geom["ordered_points"]

    prev_center = polygon_mean_point(prev_poly)
    new_center = polygon_mean_point(new_poly)

    center_shift = float(np.linalg.norm(new_center - prev_center))
    corner_shift = polygon_average_corner_shift(prev_poly, new_poly)

    return center_shift <= max_center_shift and corner_shift <= max_corner_shift


def stabilize_geometry(
    frame,
    last_good_geom,
    last_good_resource_labels,
    last_good_resource_features,
    last_good_number_map,
):
    roi, roi_box = crop_live_roi(frame)

    try:
        candidate = process_board_geometry(roi)

        if accept_new_geometry(candidate, last_good_geom):
            centers = candidate["centers"]
            resource_labels, _, resource_features = classify_resources(
                candidate["normalized_img"],
                centers,
            )

            number_map = last_good_number_map
            if number_map is None:
                number_map = generate_random_number_layout(centers, resource_labels)

            return {
                "geom": candidate,
                "resource_labels": resource_labels,
                "resource_features": resource_features,
                "number_map": number_map,
                "status": "fresh",
                "roi_box": roi_box,
            }

        if last_good_geom is not None:
            return {
                "geom": last_good_geom,
                "resource_labels": last_good_resource_labels,
                "resource_features": last_good_resource_features,
                "number_map": last_good_number_map,
                "status": "held_last_good",
                "roi_box": roi_box,
            }

        raise RuntimeError("Board jump rejected and no previous stable board exists.")

    except Exception as exc:
        if last_good_geom is not None:
            return {
                "geom": last_good_geom,
                "resource_labels": last_good_resource_labels,
                "resource_features": last_good_resource_features,
                "number_map": last_good_number_map,
                "status": f"fallback_after_error: {exc}",
                "roi_box": roi_box,
            }
        raise


def make_display_canvas(board_img, lines, status_color=(255, 255, 255)):
    h, w = board_img.shape[:2]

    canvas = np.zeros((h, w + SIDEBAR_WIDTH, 3), dtype=np.uint8)
    canvas[:, SIDEBAR_WIDTH:] = board_img

    cv2.line(
        canvas,
        (SIDEBAR_WIDTH - 1, 0),
        (SIDEBAR_WIDTH - 1, h - 1),
        (80, 80, 80),
        1,
    )

    panel = canvas[:, :SIDEBAR_WIDTH]

    put_lines(
        panel,
        lines,
        origin=(18, 36),
        line_height=34,
        scale=0.85,
        thickness=2,
        color=status_color,
        bg=False,
    )

    return canvas


def print_hsv_table(centers, features):
    print("\nPer-tile HSV values:")

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


def main():
    CHIP_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    cap = open_camera(CAMERA_INDEX)

    last_good_geom = None
    last_good_resource_labels = None
    last_good_resource_features = None

    number_map = None
    board_state = None

    consecutive_good_frames = 0
    min_good_frames = 5
    last_message = "waiting"

    monitor_mode = False
    monitor_start_time = 0.0
    monitor_warmup_seconds = 1.5
    last_swap_tiles = []

    print("Controls:")
    print("  R = reroll legal numbers")
    print("  P = print HSV table to terminal")
    print("  N = capture chips and start Dice-score monitoring")
    print("  Q or ESC = quit")
    print("")
    print("Using number_detection_debug.py")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        key = cv2.waitKey(1) & 0xFF

        try:
            stable = stabilize_geometry(
                frame=frame,
                last_good_geom=last_good_geom,
                last_good_resource_labels=last_good_resource_labels,
                last_good_resource_features=last_good_resource_features,
                last_good_number_map=number_map,
            )

            geom = stable["geom"]
            normalized = geom["normalized_img"]
            centers = geom["centers"]
            resource_labels = stable["resource_labels"]
            resource_features = stable["resource_features"]
            number_map = stable["number_map"]

            if stable["status"] == "fresh":
                consecutive_good_frames += 1
                last_good_geom = geom
                last_good_resource_labels = resource_labels
                last_good_resource_features = resource_features
            else:
                consecutive_good_frames = max(0, consecutive_good_frames - 1)

            overlay = draw_tile_hsv_values(
                normalized,
                centers,
                resource_features,
                numbers=number_map,
            )

            if (
                monitor_mode
                and resource_labels is not None
                and number_map is not None
                and board_state is not None
            ):
                chips = detect_chips(
                    normalized,
                    centers,
                    resource_labels=resource_labels,
                )

                assignments = assign_chips_to_tiles(chips, resource_labels)
                overlay = draw_chips(overlay, chips)

                if time.time() - monitor_start_time >= monitor_warmup_seconds:
                    identity_report = analyze_chip_identities(
                        assignments,
                        board_state,
                        debug_dir=STATE_DIR,
                    )

                    swaps = detect_pair_swaps(
                        identity_report,
                        board_state,
                        debug_dir=STATE_DIR,
                    )

                    applied = apply_detected_swaps(
                        board_state,
                        swaps,
                        STATE_DIR,
                    )

                    refresh_done = refresh_pending_reference_edges(
                        assignments,
                        board_state,
                        STATE_DIR,
                    )

                    if applied:
                        number_map = dict(board_state["number_map"])
                        last_swap_tiles = [
                            (sw["tile_a"], sw["tile_b"])
                            for sw in applied
                        ]
                        print_swap_detected(applied)
                        last_message = "swap detected"

                    elif refresh_done:
                        last_message = "references refreshed"
                        last_swap_tiles = []

                    else:
                        last_message = "monitoring Dice scores"
                        last_swap_tiles = []

                else:
                    last_message = "monitor warmup"

            if last_swap_tiles:
                for a, b in last_swap_tiles:
                    for tile_id in (a, b):
                        _, x, y = centers[tile_id]
                        cv2.circle(overlay, (x, y), 42, (0, 255, 255), 3)

            overlay = draw_contour(overlay, geom["normalized_points"])

            lines = [
                "HSV + Dice debug preview:",
                "Shows H, S, V for each tile.",
                "Uses number_detection_debug.",
                "",
                "Press N after chips are placed.",
                "Press R to reroll numbers.",
                "Press P to print HSV table.",
                "",
                f"Stable frames: {consecutive_good_frames}/{min_good_frames}",
                f"Board source: {stable['status']}",
                f"Mode: {'monitor' if monitor_mode else 'preview'}",
                f"Status: {last_message}",
            ]

            canvas = make_display_canvas(overlay, lines)
            cv2.imshow(WINDOW_NAME + " - HSV Debug", canvas)

            if key == ord("r"):
                number_map = generate_random_number_layout(centers, resource_labels)
                board_state = None
                monitor_mode = False
                last_swap_tiles = []
                last_message = "numbers rerolled"

            elif key == ord("p"):
                print_hsv_table(centers, resource_features)
                last_message = "printed HSV table to terminal"

            elif key == ord("n"):
                if resource_labels is None or number_map is None:
                    last_message = "wait for board first"

                elif consecutive_good_frames < min_good_frames:
                    last_message = "wait for stable board first"

                else:
                    chips = detect_chips(
                        normalized,
                        centers,
                        resource_labels=resource_labels,
                    )

                    assignments = assign_chips_to_tiles(chips, resource_labels)

                    valid_chip_count = sum(
                        1 for item in assignments
                        if item.get("chip_patch") is not None
                    )

                    if valid_chip_count < 10:
                        last_message = f"not enough chips detected ({valid_chip_count})"

                    else:
                        save_chip_debug_patches(chips, CHIP_DEBUG_DIR)

                        board_state = create_manual_board_state(
                            assignments,
                            number_map,
                            STATE_DIR,
                        )

                        monitor_mode = True
                        monitor_start_time = time.time()
                        last_swap_tiles = []
                        last_message = "chip references captured"

        except Exception as exc:
            error_frame = frame.copy()

            put_lines(
                error_frame,
                [
                    "Board not ready",
                    str(exc),
                    "Move camera closer.",
                    "Keep full board visible.",
                    "Reduce shadows and reflections.",
                ],
                origin=(20, 40),
                line_height=30,
                scale=0.8,
                thickness=2,
            )

            cv2.imshow(WINDOW_NAME + " - HSV Debug", error_frame)

        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise