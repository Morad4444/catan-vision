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
)

from tile_classification import (
    classify_resources,
    draw_tile_labels,
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


SIDEBAR_WIDTH = 430

CHIP_DEBUG_DIR = Path("data/output/chip_swap_debug")
STATE_DIR = Path("data/output/board_state_chip_swap_debug")


def open_camera(index: int):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)

    if not cap.isOpened():
        cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {index}")

    return cap


def process_board_geometry(image_bgr, prefix="chip_swap_debug"):
    """
    Detect board once on original frame.
    No warp / no normalize_hexagon.
    """
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
    """
    Only for display.
    Detection and tracking still use the full camera frame.
    """
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


def capture_board_and_resources(frame):
    geom = process_board_geometry(frame, prefix="chip_swap_capture")
    centers = geom["centers"]

    resource_labels, _, resource_features = classify_resources(
        frame,
        centers,
    )

    number_map = generate_random_number_layout(
        centers,
        resource_labels,
    )

    return geom, resource_labels, resource_features, number_map


def print_resource_table(centers, resource_labels, number_map):
    print("\n================ BOARD RESOURCE STATE ================")

    for tile_id, _, _ in centers:
        label = resource_labels[tile_id]
        num = number_map.get(tile_id, "-")
        print(f"tile {tile_id:02d}: {label:7s} number={num}")

    print("======================================================\n")


def start_chip_monitor(frame, centers, resource_labels, number_map):
    chips = detect_chips(
        frame,
        centers,
        resource_labels=resource_labels,
    )

    assignments = assign_chips_to_tiles(
        chips,
        resource_labels,
    )

    valid_chip_count = sum(
        1
        for item in assignments
        if item.get("chip_patch") is not None
    )

    save_chip_debug_patches(
        chips,
        CHIP_DEBUG_DIR,
    )

    print(f"N pressed: chips detected = {valid_chip_count}")
    print(f"assignments = {len(assignments)}")
    print(f"chip debug saved to: {CHIP_DEBUG_DIR}")

    if valid_chip_count < 10:
        return None, chips, assignments, f"not enough chips detected ({valid_chip_count})"

    board_state = create_manual_board_state(
        assignments,
        number_map,
        STATE_DIR,
    )

    print(f"chip references saved to: {STATE_DIR}")
    print("monitor started. Scores will print every 2 seconds.")
    print("Swap two chips now. Avoid locked numbers: 2, 6, 8, 12.")

    return board_state, chips, assignments, f"monitor started: {valid_chip_count} chips"


def main():
    CHIP_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    cap = open_camera(CAMERA_INDEX)

    window_title = WINDOW_NAME + " - Chip Swap Debug"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

    geom = None
    resource_labels = None
    resource_features = None
    number_map = None
    board_state = None

    monitor_mode = False
    monitor_start_time = 0.0
    monitor_warmup_seconds = 1.5

    last_message = "press B to capture board"
    last_key = "none"
    last_swap_tiles = []

    print("Controls:")
    print("  B = capture board + classify resources + generate numbers")
    print("  N = capture chip references + start score monitor")
    print("  R = re-capture resources/numbers using cached board")
    print("  S = stop monitor")
    print("  Q or ESC = quit")
    print("")
    print("Important:")
    print("  1) Click the OpenCV window before pressing keys.")
    print("  2) Press B first.")
    print("  3) Put chips on board, then press N.")
    print("  4) Scores print every 2 seconds after N.")
    print("  5) This file imports tile_classification.py, not tile_classification_hsv_debug.py.")
    print("")

    while True:
        ok, frame = cap.read()

        if not ok:
            continue

        key = cv2.waitKey(1) & 0xFF

        key_chr = ""
        if key not in (255, 0):
            try:
                key_chr = chr(key).lower()
                last_key = key_chr
                print(f"KEY pressed: {key_chr}")
            except Exception:
                last_key = str(key)
                print(f"KEY pressed code: {key}")

        try:
            if key_chr == "q" or key == 27:
                break

            if key_chr == "b":
                geom, resource_labels, resource_features, number_map = capture_board_and_resources(frame)

                board_state = None
                monitor_mode = False
                last_swap_tiles = []

                print("B pressed: board captured, resources classified, numbers generated.")
                print_resource_table(geom["centers"], resource_labels, number_map)

                last_message = "board captured - press N after chips are placed"

            if geom is None:
                preview = frame.copy()

                put_lines(
                    preview,
                    [
                        "Chip swap debug",
                        "",
                        "No board captured yet.",
                        "Click this window.",
                        "Press B to capture board.",
                        "",
                        f"Last key: {last_key}",
                        f"Status: {last_message}",
                    ],
                    origin=(20, 40),
                    line_height=30,
                    scale=0.8,
                    thickness=2,
                )

                cv2.imshow(window_title, preview)
                continue

            centers = geom["centers"]
            overlay = frame.copy()

            if key_chr == "r":
                resource_labels, _, resource_features = classify_resources(
                    frame,
                    centers,
                )

                number_map = generate_random_number_layout(
                    centers,
                    resource_labels,
                )

                board_state = None
                monitor_mode = False
                last_swap_tiles = []

                print("R pressed: resources re-classified and numbers regenerated.")
                print_resource_table(centers, resource_labels, number_map)

                last_message = "resources/numbers reset - press N again"

            elif key_chr == "s":
                board_state = None
                monitor_mode = False
                last_swap_tiles = []
                last_message = "monitor stopped"
                print("S pressed: monitor stopped.")

            elif key_chr == "n":
                if resource_labels is None or number_map is None:
                    print("N ignored: press B first.")
                    last_message = "press B first"

                else:
                    board_state, chips, assignments, msg = start_chip_monitor(
                        frame,
                        centers,
                        resource_labels,
                        number_map,
                    )

                    last_message = msg

                    if board_state is not None:
                        monitor_mode = True
                        monitor_start_time = time.time()
                        last_swap_tiles = []
                    else:
                        monitor_mode = False

            if resource_labels is not None and number_map is not None:
                overlay = draw_tile_labels(
                    overlay,
                    centers,
                    resource_labels,
                    numbers=number_map,
                )

            if (
                monitor_mode
                and board_state is not None
                and resource_labels is not None
                and number_map is not None
            ):
                chips = detect_chips(
                    frame,
                    centers,
                    resource_labels=resource_labels,
                )

                assignments = assign_chips_to_tiles(
                    chips,
                    resource_labels,
                )

                overlay = draw_chips(
                    overlay,
                    chips,
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

                        print_swap_detected(applied)
                        last_message = "swap detected"

                    elif refresh_done:
                        last_message = "reference refreshed"
                        last_swap_tiles = []

                    else:
                        last_message = "monitoring chip scores"
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
                geom["ordered_points"],
            )

            zoomed_overlay = crop_board_view(
                overlay,
                geom["ordered_points"],
                margin=180,
                target_height=700,
            )

            lines = [
                "Chip Swap Debug:",
                "Uses GOOD classifier.",
                "Scores print every 2 sec.",
                "",
                "B = capture board",
                "N = capture refs/start",
                "R = reset resources",
                "S = stop monitor",
                "Q = quit",
                "",
                f"Last key: {last_key}",
                f"Mode: {'monitor' if monitor_mode else 'preview'}",
                f"Status: {last_message}",
            ]

            canvas = make_display_canvas(zoomed_overlay, lines)
            cv2.imshow(window_title, canvas)

        except Exception:
            traceback.print_exc()

            error_frame = frame.copy()

            put_lines(
                error_frame,
                [
                    "Runtime error.",
                    "See terminal traceback.",
                    "",
                    "Usually fix:",
                    "1) Press B first.",
                    "2) Check classifier files.",
                    "3) Check chip_detection.py.",
                ],
                origin=(20, 40),
                line_height=30,
                scale=0.8,
                thickness=2,
            )

            cv2.imshow(window_title, error_frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise