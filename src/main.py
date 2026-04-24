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

from tile_classification import classify_resources, draw_tile_labels

from chip_detection import (
    detect_chips,
    assign_chips_to_tiles,
    draw_chips,
    save_chip_debug_patches,
)

from number_detection import (
    generate_random_number_layout,
    create_manual_board_state,
    analyze_chip_identities,
    detect_pair_swaps,
    apply_detected_swaps,
    print_swap_detected,
    refresh_pending_reference_edges,
)

from piece_detection import (
    detect_settlements,
    summarize_settlement_changes,
    draw_detected_houses,
)


SIDEBAR_WIDTH = 430

CHIP_DEBUG_DIR = Path("data/output/chip_debug_live")
STATE_DIR = Path("data/output/board_state_live")


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


def process_board_geometry(image_bgr, prefix="live_preview"):
    set_debug_prefix(prefix)

    contour = detect_board_contour(image_bgr)
    polygon = approximate_polygon(contour, image_bgr)
    points = polygon_to_points(polygon)
    ordered_points = order_hexagon_points(points, image_bgr)

    normalized_img, normalized_points = normalize_hexagon(
        image_bgr,
        ordered_points,
    )

    centers = generate_catan_tile_centers_from_hex(
        normalized_points,
        normalized_img,
    )

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


def accept_new_geometry(
    new_geom,
    last_good_geom,
    max_center_shift=55.0,
    max_corner_shift=70.0,
):
    if last_good_geom is None:
        return True

    prev_poly = last_good_geom["ordered_points"]
    new_poly = new_geom["ordered_points"]

    prev_center = polygon_mean_point(prev_poly)
    new_center = polygon_mean_point(new_poly)

    center_shift = float(np.linalg.norm(new_center - prev_center))
    corner_shift = polygon_average_corner_shift(prev_poly, new_poly)

    return center_shift <= max_center_shift and corner_shift <= max_corner_shift


def stabilize_geometry(frame, last_good_geom):
    roi, roi_box = crop_live_roi(frame)

    try:
        candidate = process_board_geometry(roi)

        if accept_new_geometry(candidate, last_good_geom):
            return {
                "geom": candidate,
                "status": "fresh",
                "roi_box": roi_box,
            }

        if last_good_geom is not None:
            return {
                "geom": last_good_geom,
                "status": "held_last_good",
                "roi_box": roi_box,
            }

        raise RuntimeError("Board jump rejected and no previous stable board exists.")

    except Exception as exc:
        if last_good_geom is not None:
            return {
                "geom": last_good_geom,
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


def main():
    CHIP_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    cap = open_camera(CAMERA_INDEX)

    last_good_geom = None

    frozen_labels = None
    number_map = None
    board_state = None

    previous_houses = None
    current_houses = []
    settlement_changes = {
        "new": [],
        "kept": [],
        "removed": [],
    }

    consecutive_good_frames = 0
    min_good_frames = 5
    last_message = "waiting"

    monitor_mode = False
    monitor_start_time = 0.0
    monitor_warmup_seconds = 1.5

    last_swap_tiles = []

    print("Controls:")
    print("  R = re-detect resources + reroll legal numbers")
    print("  N = capture chips + settlements and start monitoring")
    print("  Q or ESC = quit")

    while True:
        ok, frame = cap.read()

        if not ok:
            continue

        key = cv2.waitKey(1) & 0xFF

        try:
            stable = stabilize_geometry(
                frame=frame,
                last_good_geom=last_good_geom,
            )

            geom = stable["geom"]
            normalized = geom["normalized_img"]
            centers = geom["centers"]

            if stable["status"] == "fresh":
                consecutive_good_frames += 1
                last_good_geom = geom
            else:
                consecutive_good_frames = max(0, consecutive_good_frames - 1)

            if frozen_labels is None and consecutive_good_frames >= min_good_frames:
                frozen_labels, _, _ = classify_resources(
                    normalized,
                    centers,
                )

                number_map = generate_random_number_layout(
                    centers,
                    frozen_labels,
                )

                last_message = "resources locked"

            overlay = normalized.copy()

            if frozen_labels is not None and number_map is not None:
                overlay = draw_tile_labels(
                    overlay,
                    centers,
                    frozen_labels,
                    numbers=number_map,
                )

            if (
                monitor_mode
                and frozen_labels is not None
                and number_map is not None
                and board_state is not None
            ):
                chips = detect_chips(
                    normalized,
                    centers,
                    resource_labels=frozen_labels,
                )

                assignments = assign_chips_to_tiles(
                    chips,
                    frozen_labels,
                )

                overlay = draw_chips(
                    overlay,
                    chips,
                )

                current_houses, _, _ = detect_settlements(
                    normalized,
                    centers,
                )

                settlement_changes = summarize_settlement_changes(
                    previous_houses,
                    current_houses,
                )

                overlay = draw_detected_houses(
                    overlay,
                    current_houses,
                    new_houses=settlement_changes["new"],
                )

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

                        last_message = ", ".join(
                            [
                                f"swap {sw['number_a']}<->{sw['number_b']}"
                                for sw in applied
                            ]
                        )

                        print_swap_detected(applied)

                    else:
                        if refresh_done:
                            last_message = "chip references refreshed"
                        elif settlement_changes["new"]:
                            last_message = f"new settlement: {len(settlement_changes['new'])}"
                        else:
                            last_message = "monitoring"

                        last_swap_tiles = []

                else:
                    last_message = "monitor warmup"

            if last_swap_tiles:
                for a, b in last_swap_tiles:
                    for tile_id in (a, b):
                        _, x, y = centers[tile_id]
                        cv2.circle(
                            overlay,
                            (x, y),
                            42,
                            (0, 255, 255),
                            3,
                        )

            overlay = draw_contour(
                overlay,
                geom["normalized_points"],
            )

            lines = [
                "Catan monitor:",
                "Resources are fixed.",
                "Numbers are generated legally.",
                "",
                "Press N after chips/pieces placed.",
                "Press R to reset board state.",
                "",
                f"Stable frames: {consecutive_good_frames}/{min_good_frames}",
                f"Board source: {stable['status']}",
                f"Mode: {'monitor' if monitor_mode else 'preview'}",
                f"Settlements: {len(current_houses)}",
                f"New settlements: {len(settlement_changes['new'])}",
                f"Removed settlements: {len(settlement_changes['removed'])}",
                f"Status: {last_message}",
            ]

            canvas = make_display_canvas(overlay, lines)
            cv2.imshow(WINDOW_NAME, canvas)

            if key == ord("r"):
                if consecutive_good_frames >= min_good_frames:
                    frozen_labels, _, _ = classify_resources(
                        normalized,
                        centers,
                    )

                    number_map = generate_random_number_layout(
                        centers,
                        frozen_labels,
                    )

                    board_state = None
                    monitor_mode = False
                    last_swap_tiles = []

                    previous_houses = None
                    current_houses = []
                    settlement_changes = {
                        "new": [],
                        "kept": [],
                        "removed": [],
                    }

                    last_message = "resources re-detected and numbers rerolled"

                else:
                    last_message = "wait for stable board first"

            elif key == ord("n"):
                if frozen_labels is None or number_map is None:
                    last_message = "wait for board/resources first"

                elif consecutive_good_frames < min_good_frames:
                    last_message = "wait for stable board first"

                else:
                    chips = detect_chips(
                        normalized,
                        centers,
                        resource_labels=frozen_labels,
                    )

                    assignments = assign_chips_to_tiles(
                        chips,
                        frozen_labels,
                    )

                    valid_chip_count = sum(
                        1
                        for item in assignments
                        if item.get("chip_patch") is not None
                    )

                    current_houses, _, _ = detect_settlements(
                        normalized,
                        centers,
                    )

                    previous_houses = current_houses
                    settlement_changes = {
                        "new": [],
                        "kept": current_houses,
                        "removed": [],
                    }

                    if valid_chip_count < 10:
                        last_message = f"not enough chips detected yet ({valid_chip_count})"

                    else:
                        save_chip_debug_patches(
                            chips,
                            CHIP_DEBUG_DIR,
                        )

                        board_state = create_manual_board_state(
                            assignments,
                            number_map,
                            STATE_DIR,
                        )

                        monitor_mode = True
                        monitor_start_time = time.time()
                        last_swap_tiles = []

                        last_message = (
                            f"chip refs + {len(current_houses)} settlements captured"
                        )

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

            cv2.imshow(WINDOW_NAME, error_frame)

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