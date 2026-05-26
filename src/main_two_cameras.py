from __future__ import annotations

import traceback
import time
from pathlib import Path

import cv2
import numpy as np

from config import WINDOW_NAME
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

from dice_detection import (
    DiceDetector,
    ValueHistory,
    open_camera as open_dice_camera,
    draw_detections,
)

from game_logic import get_resource_payout_message


# -------------------------------------------------------------------
# Camera settings
# -------------------------------------------------------------------

BOARD_CAMERA_INDEX = 0
DICE_CAMERA_INDEX = 1

BOARD_WIDTH = 1280
BOARD_HEIGHT = 720

DICE_WIDTH = 1280
DICE_HEIGHT = 720
DICE_EXPOSURE = -6.0
DICE_CAMERA_NAME = None  # Example: "HD USB Camera"


# -------------------------------------------------------------------
# Output folders
# -------------------------------------------------------------------

SIDEBAR_WIDTH = 430

CHIP_DEBUG_DIR = Path("data/output/chip_debug_two_cameras")
STATE_DIR = Path("data/output/board_state_two_cameras")


# -------------------------------------------------------------------
# Camera helpers
# -------------------------------------------------------------------

def open_board_camera(index: int, width: int = BOARD_WIDTH, height: int = BOARD_HEIGHT):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)

    if not cap.isOpened():
        cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open board camera index {index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return cap


# -------------------------------------------------------------------
# Board processing
# -------------------------------------------------------------------

def process_board_geometry(image_bgr, prefix="two_camera_board"):
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


def capture_board_and_resources(frame):
    geom = process_board_geometry(frame, prefix="two_camera_capture")
    centers = geom["centers"]

    resource_labels, _, _ = classify_resources(
        frame,
        centers,
    )

    number_map = generate_random_number_layout(
        centers,
        resource_labels,
    )

    return geom, resource_labels, number_map


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


# -------------------------------------------------------------------
# Dice processing
# -------------------------------------------------------------------

def update_dice_state(detections, history: ValueHistory):
    detected_labels = {d.label for d in detections}

    for label in ("Yellow", "Red", "White"):
        if label not in detected_labels:
            history.reset(label)

    values = {}

    for det in detections:
        det.best_guess = history.update(det.label, det.value)
        values[det.label] = det.best_guess

    yellow = values.get("Yellow")
    red = values.get("Red")

    dice_sum = None

    if isinstance(yellow, int) and isinstance(red, int):
        dice_sum = yellow + red

    return {
        "Yellow": yellow,
        "Red": red,
        "White": values.get("White"),
        "sum": dice_sum,
    }


def draw_dice_panel(panel, dice_state, payout_message):
    h, w = panel.shape[:2]

    cv2.rectangle(panel, (18, h - 250), (w - 18, h - 20), (20, 20, 20), -1)
    cv2.rectangle(panel, (18, h - 250), (w - 18, h - 20), (255, 255, 255), 1)

    yellow = dice_state.get("Yellow")
    red = dice_state.get("Red")
    white = dice_state.get("White")
    dice_sum = dice_state.get("sum")

    lines = [
        "Dice camera:",
        f"Yellow: {yellow if yellow is not None else '?'}",
        f"Red:    {red if red is not None else '?'}",
        f"White:  {white if white is not None else '?'}",
        f"SUM:    {dice_sum if dice_sum is not None else '?'}",
    ]

    put_lines(
        panel,
        lines,
        origin=(34, h - 218),
        line_height=26,
        scale=0.68,
        thickness=2,
        color=(255, 255, 255),
        bg=False,
    )

    # Big dice sum on the right side of the dice panel
    if dice_sum is not None:
        sum_text = str(dice_sum)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 2.6
        thickness = 6

        (tw, th), _ = cv2.getTextSize(sum_text, font, scale, thickness)
        x = int(w - 80 - tw / 2)
        y = int(h - 145 + th / 2)

        cv2.putText(
            panel,
            sum_text,
            (x, y),
            font,
            scale,
            (0, 0, 0),
            thickness + 4,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            sum_text,
            (x, y),
            font,
            scale,
            (0, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    # Colored payout text.
    # Example:
    #   "red gets 1 Brick | blue gets 1 Grain"
    payout_y = h - 84

    if payout_message and payout_message != "No resources produced.":
        payout_parts = payout_message.split("|")

        color_map = {
            "red": (0, 0, 255),
            "blue": (255, 0, 0),
            "orange": (0, 165, 255),
            "white": (255, 255, 255),
            "green": (0, 255, 0),
        }

        for i, part in enumerate(payout_parts[:5]):
            txt = part.strip()
            txt_lower = txt.lower()

            color = (255, 255, 255)

            for player_name, player_color in color_map.items():
                if txt_lower.startswith(player_name):
                    color = player_color
                    break

            y = payout_y + i * 25

            cv2.putText(
                panel,
                txt,
                (34, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )

            cv2.putText(
                panel,
                txt,
                (34, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                color,
                2,
                cv2.LINE_AA,
            )

    else:
        cv2.putText(
            panel,
            "No resources produced.",
            (34, payout_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )

        cv2.putText(
            panel,
            "No resources produced.",
            (34, payout_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (180, 180, 180),
            2,
            cv2.LINE_AA,
        )

# -------------------------------------------------------------------
# Display
# -------------------------------------------------------------------

def make_board_display(board_img, lines, dice_state, payout_message):
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
        line_height=30,
        scale=0.70,
        thickness=2,
        color=(255, 255, 255),
        bg=False,
    )

    draw_dice_panel(panel, dice_state, payout_message)

    return canvas


# -------------------------------------------------------------------
# Game logic wrapper
# -------------------------------------------------------------------

def compute_payout_message(dice_state, number_map, resource_labels, current_houses):
    if (
        dice_state is None
        or number_map is None
        or resource_labels is None
        or current_houses is None
    ):
        return "No resources produced."

    return get_resource_payout_message(
        dice_state.get("sum"),
        number_map,
        resource_labels,
        current_houses,
    )


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    CHIP_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    board_cap = open_board_camera(BOARD_CAMERA_INDEX)

    dice_cap, selected_dice_index = open_dice_camera(
        DICE_CAMERA_INDEX,
        DICE_WIDTH,
        DICE_HEIGHT,
        preferred_name=DICE_CAMERA_NAME,
        exposure=DICE_EXPOSURE,
    )

    if not dice_cap.isOpened():
        raise RuntimeError(f"Could not open dice camera index {DICE_CAMERA_INDEX}")

    dice_detector = DiceDetector()
    dice_history = ValueHistory(window=30)

    window_board = WINDOW_NAME + " - Two Cameras"
    window_dice = "Dice Camera"

    cv2.namedWindow(window_board, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_dice, cv2.WINDOW_NORMAL)

    geom = None
    resource_labels = None
    number_map = None
    board_state = None

    previous_houses = None
    current_houses = []
    settlement_changes = {"new": [], "kept": [], "removed": []}

    monitor_mode = False
    monitor_start_time = 0.0
    monitor_warmup_seconds = 1.5

    last_message = "press B to capture board"
    last_key = "none"
    last_swap_tiles = []
    payout_message = "No resources produced."

    dice_state = {
        "Yellow": None,
        "Red": None,
        "White": None,
        "sum": None,
    }

    roi_enabled = True

    print("Two-camera Catan system")
    print(f"Board camera index: {BOARD_CAMERA_INDEX}")
    print(f"Dice camera index:  {selected_dice_index}")
    print("")
    print("Controls:")
    print("  B = capture/recapture board from board camera")
    print("  N = capture chip references and start board monitoring")
    print("  R = reset resources/numbers using cached board")
    print("  S = stop board monitor")
    print("  D = toggle dice tray ROI")
    print("  Q or ESC = quit")
    print("")
    print("Dice sum and resource payout are shown in the left panel next to the board.")

    while True:
        ok_board, board_frame = board_cap.read()
        ok_dice, dice_frame = dice_cap.read()

        if not ok_board:
            print("Could not read board camera frame.")
            continue

        if not ok_dice:
            print("Could not read dice camera frame.")
            dice_frame = np.zeros_like(board_frame)

        # ---------------- Dice camera ----------------
        try:
            dice_detections = dice_detector.detect(dice_frame, use_roi=roi_enabled)
            dice_state = update_dice_state(dice_detections, dice_history)

            dice_display = draw_detections(
                dice_frame,
                dice_detections,
                dice_detector.last_tray_circle,
                dice_detector.last_tray_mask,
                use_roi=roi_enabled,
            )

            sum_txt = f"SUM: {dice_state['sum'] if dice_state['sum'] is not None else '?'}"
            cv2.putText(dice_display, sum_txt, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(dice_display, sum_txt, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)

            cv2.imshow(window_dice, dice_display)

        except Exception:
            traceback.print_exc()

        key = cv2.waitKey(1) & 0xFF

        key_chr = ""
        if key not in (255, 0):
            try:
                key_chr = chr(key).lower()
                last_key = key_chr
                print(f"KEY pressed: {key_chr}")
            except Exception:
                last_key = str(key)

        try:
            if key_chr == "q" or key == 27:
                break

            if key_chr == "d":
                roi_enabled = not roi_enabled
                print(f"Dice ROI: {'ON' if roi_enabled else 'OFF'}")

            if key_chr == "b":
                geom, resource_labels, number_map = capture_board_and_resources(board_frame)

                board_state = None
                monitor_mode = False
                last_swap_tiles = []
                previous_houses = None

                # Detect houses immediately after board capture too
                current_houses, _, _ = detect_settlements(
                    board_frame,
                    geom["centers"],
                )

                settlement_changes = {
                    "new": current_houses,
                    "kept": [],
                    "removed": [],
                }

                payout_message = compute_payout_message(
                    dice_state,
                    number_map,
                    resource_labels,
                    current_houses,
                )

                last_message = "board captured - press N after chips are placed"
                print("B pressed: board captured and resources classified.")

            if geom is None:
                preview = board_frame.copy()

                put_lines(
                    preview,
                    [
                        "Two-camera Catan",
                        "",
                        "No board captured yet.",
                        "Press B to capture board.",
                        "",
                        f"Dice sum: {dice_state['sum'] if dice_state['sum'] is not None else '?'}",
                        f"Payout: {payout_message}",
                        f"Last key: {last_key}",
                        f"Status: {last_message}",
                    ],
                    origin=(20, 40),
                    line_height=30,
                    scale=0.8,
                    thickness=2,
                )

                cv2.imshow(window_board, preview)
                continue

            centers = geom["centers"]
            overlay = board_frame.copy()

            if key_chr == "r":
                resource_labels, _, _ = classify_resources(
                    board_frame,
                    centers,
                )

                number_map = generate_random_number_layout(
                    centers,
                    resource_labels,
                )

                board_state = None
                monitor_mode = False
                last_swap_tiles = []
                previous_houses = None

                current_houses, _, _ = detect_settlements(
                    board_frame,
                    centers,
                )

                settlement_changes = {
                    "new": current_houses,
                    "kept": [],
                    "removed": [],
                }

                payout_message = compute_payout_message(
                    dice_state,
                    number_map,
                    resource_labels,
                    current_houses,
                )

                last_message = "resources/numbers reset - press N again"
                print("R pressed: resources/numbers reset.")

            elif key_chr == "s":
                board_state = None
                monitor_mode = False
                last_swap_tiles = []
                last_message = "monitor stopped"
                print("S pressed: board monitor stopped.")

            elif key_chr == "n":
                if resource_labels is None or number_map is None:
                    last_message = "press B first"
                    print("N ignored: press B first.")

                else:
                    chips = detect_chips(
                        board_frame,
                        centers,
                        resource_labels=resource_labels,
                    )

                    assignments = assign_chips_to_tiles(
                        chips,
                        resource_labels,
                    )

                    valid_chip_count = sum(
                        1 for item in assignments
                        if item.get("chip_patch") is not None
                    )

                    save_chip_debug_patches(
                        chips,
                        CHIP_DEBUG_DIR,
                    )

                    print(f"N pressed: chips detected = {valid_chip_count}")

                    current_houses, _, _ = detect_settlements(
                        board_frame,
                        centers,
                    )

                    previous_houses = current_houses
                    settlement_changes = {
                        "new": [],
                        "kept": current_houses,
                        "removed": [],
                    }

                    payout_message = compute_payout_message(
                        dice_state,
                        number_map,
                        resource_labels,
                        current_houses,
                    )

                    if valid_chip_count < 10:
                        board_state = None
                        monitor_mode = False
                        last_message = f"not enough chips detected ({valid_chip_count})"
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

            if resource_labels is not None and number_map is not None:
                overlay = draw_tile_labels(
                    overlay,
                    centers,
                    resource_labels,
                    numbers=number_map,
                )

            # Always update payout if we know the board and houses.
            if resource_labels is not None and number_map is not None:
                payout_message = compute_payout_message(
                    dice_state,
                    number_map,
                    resource_labels,
                    current_houses,
                )

            if (
                monitor_mode
                and board_state is not None
                and resource_labels is not None
                and number_map is not None
            ):
                chips = detect_chips(
                    board_frame,
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

                current_houses, _, _ = detect_settlements(
                    board_frame,
                    centers,
                )

                settlement_changes = summarize_settlement_changes(
                    previous_houses,
                    current_houses,
                )

                payout_message = compute_payout_message(
                    dice_state,
                    number_map,
                    resource_labels,
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

                        payout_message = compute_payout_message(
                            dice_state,
                            number_map,
                            resource_labels,
                            current_houses,
                        )

                        print_swap_detected(applied)
                        last_message = "swap detected"

                    elif refresh_done:
                        last_message = "reference refreshed"
                        last_swap_tiles = []

                    elif settlement_changes["new"]:
                        last_message = f"new settlement: {len(settlement_changes['new'])}"
                        last_swap_tiles = []

                    else:
                        last_message = "monitoring"
                        last_swap_tiles = []

                else:
                    last_message = "monitor warmup"

            else:
                # Even outside monitor mode, show settlement overlay if available.
                if current_houses:
                    overlay = draw_detected_houses(
                        overlay,
                        current_houses,
                        new_houses=settlement_changes.get("new", []),
                    )

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
                "Two-camera Catan:",
                "Board + dice active.",
                "",
                "B = capture board",
                "N = start chip monitor",
                "R = reset resources",
                "S = stop monitor",
                "D = dice ROI toggle",
                "Q = quit",
                "",
                f"Dice SUM: {dice_state['sum'] if dice_state['sum'] is not None else '?'}",
                f"Yellow: {dice_state['Yellow'] if dice_state['Yellow'] is not None else '?'}",
                f"Red:    {dice_state['Red'] if dice_state['Red'] is not None else '?'}",
                f"White:  {dice_state['White'] if dice_state['White'] is not None else '?'}",
                "",
                f"Mode: {'monitor' if monitor_mode else 'preview'}",
                f"Houses: {len(current_houses)}",
                f"Status: {last_message}",
            ]

            canvas = make_board_display(
                zoomed_overlay,
                lines,
                dice_state,
                payout_message,
            )

            cv2.imshow(window_board, canvas)

        except Exception:
            traceback.print_exc()

            error_frame = board_frame.copy()

            put_lines(
                error_frame,
                [
                    "Runtime error.",
                    "See terminal traceback.",
                    "",
                    "Press B to recapture board.",
                ],
                origin=(20, 40),
                line_height=30,
                scale=0.8,
                thickness=2,
            )

            cv2.imshow(window_board, error_frame)

    board_cap.release()
    dice_cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise