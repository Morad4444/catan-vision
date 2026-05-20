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

from tile_classification_hsv_debug import (
    classify_resources,
    draw_tile_hsv_values,
)


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


def process_board_geometry(image_bgr, prefix="hsv_debug_cached"):
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


def print_hsv_table(centers, features, labels=None):
    print("\n================ HSV TABLE ================")

    for tile_id, _, _ in centers:
        f = features[tile_id]

        label_txt = ""
        if labels is not None:
            label_txt = f" label={labels[tile_id]}"

        print(
            f"Tile {tile_id:02d}: "
            f"H={f['h']:.1f}, "
            f"S={f['s']:.1f}, "
            f"V={f['v']:.1f}, "
            f"green={f.get('green_frac', 0.0):.2f}, "
            f"yellow={f.get('yellow_frac', 0.0):.2f}, "
            f"red={f.get('red_frac', 0.0):.2f}, "
            f"blue={f.get('blue_frac', 0.0):.2f}, "
            f"lowSat={f.get('low_sat_frac', 0.0):.2f}"
            f"{label_txt}"
        )

    print("===========================================\n")


def handle_capture_board(frame):
    geom = process_board_geometry(frame, prefix="hsv_manual_capture")
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


def main():
    CHIP_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    cap = open_camera(CAMERA_INDEX)

    cv2.namedWindow(WINDOW_NAME + " - HSV Debug", cv2.WINDOW_NORMAL)

    geom = None
    resource_labels = None
    resource_features = None
    number_map = None
    board_state = None

    monitor_mode = False
    monitor_start_time = 0.0
    monitor_warmup_seconds = 1.5

    last_message = "press B to capture board"
    last_swap_tiles = []

    print("Controls:")
    print("  B = capture / recapture board geometry")
    print("  P = print HSV table")
    print("  R = re-detect HSV + reroll legal numbers")
    print("  N = capture chips and start Dice-score monitoring")
    print("  Q or ESC = quit")
    print("")
    print("Important: click the OpenCV window before pressing keys.")
    print("This version prints every key received by OpenCV.")
    print("Using number_detection_debug.py")

    while True:
        ok, frame = cap.read()

        if not ok:
            continue

        key = cv2.waitKey(1) & 0xFF

        key_chr = ""
        if key not in (255, 0):
            try:
                key_chr = chr(key).lower()
                print(f"KEY pressed: {key_chr}")
            except Exception:
                print(f"KEY pressed code: {key}")

        try:
            # --------------------------------------------------
            # Quit
            # --------------------------------------------------
            if key_chr == "q" or key == 27:
                break

            # --------------------------------------------------
            # Board capture
            # --------------------------------------------------
            if key_chr == "b":
                geom, resource_labels, resource_features, number_map = handle_capture_board(frame)

                board_state = None
                monitor_mode = False
                last_swap_tiles = []
                last_message = "board captured - HSV detected"

                print("B pressed: board captured and HSV values detected.")

            # --------------------------------------------------
            # If no board yet, only B/Q are useful
            # --------------------------------------------------
            if geom is None:
                error_frame = frame.copy()

                put_lines(
                    error_frame,
                    [
                        "No board geometry yet.",
                        "Click this window.",
                        "Press B to capture board.",
                        "Then press N to start chip score debug.",
                        "",
                        f"Last key: {key_chr if key_chr else 'none'}",
                        f"Status: {last_message}",
                    ],
                    origin=(20, 40),
                    line_height=30,
                    scale=0.8,
                    thickness=2,
                )

                cv2.imshow(WINDOW_NAME + " - HSV Debug", error_frame)
                continue

            centers = geom["centers"]
            overlay = frame.copy()

            # --------------------------------------------------
            # Re-detect HSV
            # --------------------------------------------------
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

                last_message = "HSV re-detected and numbers rerolled"
                print("R pressed: HSV re-detected and numbers rerolled.")

            # --------------------------------------------------
            # Print HSV table
            # --------------------------------------------------
            elif key_chr == "p":
                if resource_features is None:
                    last_message = "press B first"
                    print("P ignored: press B first.")
                else:
                    print_hsv_table(
                        centers,
                        resource_features,
                        labels=resource_labels,
                    )
                    last_message = "HSV table printed to terminal"

            # --------------------------------------------------
            # Start chip debug
            # --------------------------------------------------
            elif key_chr == "n":
                print("N pressed")

                if resource_labels is None or number_map is None:
                    last_message = "press B first"
                    print("N ignored: no resource labels or number map yet.")

                else:
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

                    print(f"chips detected on N: {valid_chip_count}")

                    save_chip_debug_patches(
                        chips,
                        CHIP_DEBUG_DIR,
                    )

                    if valid_chip_count < 10:
                        last_message = f"N worked, but not enough chips detected ({valid_chip_count})"
                        print(last_message)

                    else:
                        board_state = create_manual_board_state(
                            assignments,
                            number_map,
                            STATE_DIR,
                        )

                        monitor_mode = True
                        monitor_start_time = time.time()
                        last_swap_tiles = []

                        last_message = f"monitor started: {valid_chip_count} chips"
                        print(last_message)

            # --------------------------------------------------
            # Draw HSV values
            # --------------------------------------------------
            if resource_features is not None:
                overlay = draw_tile_hsv_values(
                    overlay,
                    centers,
                    resource_features,
                    numbers=number_map,
                )

            # --------------------------------------------------
            # Monitoring
            # --------------------------------------------------
            if (
                monitor_mode
                and resource_labels is not None
                and number_map is not None
                and board_state is not None
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
                        last_message = "references refreshed"
                        last_swap_tiles = []

                    else:
                        last_message = "monitoring Dice scores"
                        last_swap_tiles = []

                else:
                    last_message = "monitor warmup"

            # --------------------------------------------------
            # Draw swap highlight
            # --------------------------------------------------
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
                "HSV + chip-score debug:",
                "Click window before keys.",
                "",
                "B = capture board",
                "P = print HSV table",
                "R = re-detect HSV",
                "N = start chip scores",
                "Q = quit",
                "",
                f"Last key: {key_chr if key_chr else 'none'}",
                f"Mode: {'monitor' if monitor_mode else 'preview'}",
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
                    "Check terminal for traceback.",
                ],
                origin=(20, 40),
                line_height=30,
                scale=0.8,
                thickness=2,
            )

            cv2.imshow(WINDOW_NAME + " - HSV Debug", error_frame)
            traceback.print_exc()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise