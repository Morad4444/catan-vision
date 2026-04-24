from __future__ import annotations

from pathlib import Path
import random
import time
import json
import shutil
import cv2
import numpy as np

NUMBER_POOL = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12]
LOCKED_NUMBERS = {2, 6, 8, 12}
EDGE_SIZE = 112

# --------------------------------------------------
# board neighbors
# --------------------------------------------------

TILE_NEIGHBORS = {
    0: [1, 3, 4],
    1: [0, 2, 4, 5],
    2: [1, 5, 6],
    3: [0, 4, 7, 8],
    4: [0, 1, 3, 5, 8, 9],
    5: [1, 2, 4, 6, 9, 10],
    6: [2, 5, 10, 11],
    7: [3, 8, 12],
    8: [3, 4, 7, 9, 12, 13],
    9: [4, 5, 8, 10, 13, 14],
    10: [5, 6, 9, 11, 14, 15],
    11: [6, 10, 15],
    12: [7, 8, 13, 16],
    13: [8, 9, 12, 14, 16, 17],
    14: [9, 10, 13, 15, 17, 18],
    15: [10, 11, 14, 18],
    16: [12, 13, 17],
    17: [13, 14, 16, 18],
    18: [14, 15, 17],
}

_last_print_time = 0.0


# --------------------------------------------------
# legal number generation
# --------------------------------------------------

def _can_place_special(number_map, tile_id, number):
    if number not in (6, 8):
        return True
    for nb in TILE_NEIGHBORS.get(tile_id, []):
        if number_map.get(nb) in (6, 8):
            return False
    return True


def generate_random_number_layout(centers, resource_labels, max_attempts=5000):
    non_desert_tiles = [t for t, _, _ in centers if resource_labels[t] != "Desert"]

    rng = random.Random()

    for _ in range(max_attempts):
        pool = NUMBER_POOL[:]
        rng.shuffle(pool)

        specials = [x for x in pool if x in (6, 8)]
        others = [x for x in pool if x not in (6, 8)]

        number_map = {}
        shuffled = non_desert_tiles[:]
        rng.shuffle(shuffled)

        idx = 0
        for tile in shuffled:
            if idx >= len(specials):
                break
            n = specials[idx]
            if _can_place_special(number_map, tile, n):
                number_map[tile] = n
                idx += 1

        if idx != len(specials):
            continue

        remain = [t for t in non_desert_tiles if t not in number_map]
        rng.shuffle(remain)

        for tile, n in zip(remain, others):
            number_map[tile] = n

        return number_map

    raise RuntimeError("Could not generate legal layout")


# --------------------------------------------------
# folders
# --------------------------------------------------

def _temps_dir(base):
    return Path(base) / "chips_temps"


def _current_dir(base):
    return Path(base) / "chips_current"


def _edge_path(folder, tile_id):
    return folder / f"tile_{tile_id}_edges.png"


# --------------------------------------------------
# image prep
# --------------------------------------------------

def _blank():
    return np.zeros((EDGE_SIZE, EDGE_SIZE), dtype=np.uint8)


def _center_crop(img):
    h, w = img.shape[:2]
    s = min(h, w)
    x = (w - s) // 2
    y = (h - s) // 2
    return img[y:y+s, x:x+s]


def _inner_mask(shape, scale=0.60):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cx = w // 2
    cy = h // 2
    r = int(round(min(h, w) * 0.5 * scale))
    cv2.circle(mask, (cx, cy), r, 255, -1)
    return mask


def preprocess_chip_edges(chip_patch_bgr):
    if chip_patch_bgr is None or chip_patch_bgr.size == 0:
        return None

    img = _center_crop(chip_patch_bgr)
    img = cv2.resize(img, (EDGE_SIZE, EDGE_SIZE))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mask = _inner_mask(gray.shape, 0.60)
    gray = cv2.bitwise_and(gray, gray, mask=mask)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 50, 120)
    edges = cv2.bitwise_and(edges, edges, mask=mask)

    return edges


# --------------------------------------------------
# Dice score
# --------------------------------------------------

def dice_similarity(a, b):
    if a is None or b is None:
        return -1.0

    if a.shape != b.shape:
        return -1.0

    aa = (a > 0).astype(np.uint8)
    bb = (b > 0).astype(np.uint8)

    inter = np.sum((aa == 1) & (bb == 1))
    sa = np.sum(aa)
    sb = np.sum(bb)

    if sa + sb == 0:
        return 1.0

    return float((2.0 * inter) / (sa + sb))


# --------------------------------------------------
# save edges
# --------------------------------------------------

def save_reference_edges(assignments, base_dir):
    folder = _temps_dir(base_dir)
    folder.mkdir(parents=True, exist_ok=True)

    for item in assignments:
        tile = int(item["tile_id"])
        img = preprocess_chip_edges(item.get("chip_patch"))
        if img is None:
            img = _blank()
        cv2.imwrite(str(_edge_path(folder, tile)), img)


def save_current_edges(assignments, base_dir):
    folder = _current_dir(base_dir)
    folder.mkdir(parents=True, exist_ok=True)

    for item in assignments:
        tile = int(item["tile_id"])
        img = preprocess_chip_edges(item.get("chip_patch"))
        if img is None:
            img = _blank()
        cv2.imwrite(str(_edge_path(folder, tile)), img)


# --------------------------------------------------
# state
# --------------------------------------------------

def create_manual_board_state(assignments, number_map, save_dir=None):
    state = {
        "created_at": time.time(),
        "number_map": {int(k): int(v) for k, v in number_map.items()},
        "last_swap_time": 0.0,
    }

    if save_dir is not None:
        save_reference_edges(assignments, save_dir)

    return state


# --------------------------------------------------
# analysis
# --------------------------------------------------

def analyze_chip_identities(assignments, state, debug_dir=None):
    global _last_print_time

    base = Path(debug_dir)
    save_current_edges(assignments, base)

    temps = _temps_dir(base)
    curr = _current_dir(base)

    report = []

    text_parts = []

    for item in assignments:
        tile = int(item["tile_id"])

        a = cv2.imread(str(_edge_path(curr, tile)), 0)
        b = cv2.imread(str(_edge_path(temps, tile)), 0)

        if a is None:
            a = _blank()
        if b is None:
            b = _blank()

        score = dice_similarity(a, b)

        report.append({
            "tile_id": tile,
            "self_score": score,
            "current_number": state["number_map"][tile],
        })

        text_parts.append(f"tile {tile}: {score:.3f}")

    now = time.time()
    if now - _last_print_time >= 2.0:
        print(" | ".join(text_parts))
        _last_print_time = now

    return report


# --------------------------------------------------
# detect swaps
# --------------------------------------------------

def detect_pair_swaps(
    identity_report,
    state,
    debug_dir,
    mismatch_threshold=0.39,
    cooldown_seconds=2.0,
    **kwargs
):
    now = time.time()

    if now - state.get("last_swap_time", 0.0) < cooldown_seconds:
        return []

    changed = [r for r in identity_report if r["self_score"] < mismatch_threshold]

    if len(changed) != 2:
        return []

    a = changed[0]["tile_id"]
    b = changed[1]["tile_id"]

    na = state["number_map"][a]
    nb = state["number_map"][b]

    if na == nb:
        return []

    if na in LOCKED_NUMBERS or nb in LOCKED_NUMBERS:
        return []

    return [{
        "tile_a": a,
        "tile_b": b,
        "number_a": na,
        "number_b": nb,
    }]


# --------------------------------------------------
# apply swap
# --------------------------------------------------

def apply_detected_swaps(state, swaps, base_dir):
    applied = []

    for sw in swaps:
        a = sw["tile_a"]
        b = sw["tile_b"]

        state["number_map"][a], state["number_map"][b] = (
            state["number_map"][b],
            state["number_map"][a],
        )

        applied.append(sw)

    if applied:
        state["last_swap_time"] = time.time()

    return applied


# --------------------------------------------------
# keep compatible
# --------------------------------------------------

def refresh_pending_reference_edges(*args, **kwargs):
    return False


def print_swap_detected(swaps):
    for sw in swaps:
        print(f"swap detected: {sw['number_a']} <-> {sw['number_b']}")