from __future__ import annotations

import cv2
import numpy as np


RESOURCE_COUNTS = {
    "Ore": 3,
    "Brick": 3,
    "Grain": 4,
    "Lumber": 4,
    "Wool": 4,
    "Desert": 1,
}

RESOURCE_COLORS_BGR = {
    "Ore": (190, 120, 80),
    "Brick": (40, 90, 200),
    "Grain": (40, 200, 240),
    "Lumber": (60, 160, 60),
    "Wool": (120, 220, 140),
    "Desert": (180, 180, 180),
}


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


def extract_tile_hsv(tile_patch):
    hsv = cv2.cvtColor(tile_patch, cv2.COLOR_BGR2HSV)

    h, w = hsv.shape[:2]

    mask = build_inner_mask(h, w, radius_scale=0.30)

    vals = hsv[mask]

    if len(vals) == 0:
        vals = hsv.reshape(-1, 3)

    vals = vals.astype(np.float32)

    # remove glare/shadows
    v_lo = np.percentile(vals[:, 2], 8)
    v_hi = np.percentile(vals[:, 2], 92)

    keep = (vals[:, 2] >= v_lo) & (vals[:, 2] <= v_hi)

    core = vals[keep] if np.count_nonzero(keep) >= 20 else vals

    h_val = float(np.median(core[:, 0]))
    s_val = float(np.median(core[:, 1]))
    v_val = float(np.median(core[:, 2]))

    return {
        "h": h_val,
        "s": s_val,
        "v": v_val,
    }


def classify_hsv(h, s, v):
    # ------------------------------------------------------------
    # WOOL
    # H: 39-40
    # S: 176-190
    # V: 145-154
    # ------------------------------------------------------------
    if (
        37 <= h <= 42
        and 150 <= s <= 230
        and 138 <= v <= 160
    ):
        return "Wool"

    # ------------------------------------------------------------
    # LUMBER
    # H: 37-39
    # S: 144-148
    # V: 79-99
    # ------------------------------------------------------------
    if (
        35 <= h <= 41
        and 130 <= s <= 220
        and 70 <= v <= 110
    ):
        return "Lumber"

    # ------------------------------------------------------------
    # ORE
    # H: 19-25
    # S: 40-60
    # V: 98-153
    # ------------------------------------------------------------
    if (
        18 <= h <= 27
        and 30 <= s <= 75
        and 105 <= v <= 140
    ):
        return "Ore"

    # ------------------------------------------------------------
    # BRICK
    # H: 16-17
    # S: 191-214
    # V: 123-135
    # ------------------------------------------------------------
    if (
        14 <= h <= 19
        and 175 <= s <= 225
        and 110 <= v <= 160
    ):
        return "Brick"

    # ------------------------------------------------------------
    # GRAIN
    # H: 22-23
    # S: 135-204
    # V: 164-190
    # ------------------------------------------------------------
    if (
        20 <= h <= 25
        and 200 <= s <= 230
        and 155 <= v <= 190
    ):
        return "Grain"

    # ------------------------------------------------------------
    # DESERT
    # H: 23
    # S: 139
    # V: 154
    # ------------------------------------------------------------
    if (
        20 <= h <= 26
        and 110 <= s <= 155
        and 140 <= v <= 170
    ):
        return "Desert"

    return None


def hsv_distance(h, s, v, target):
    return (
        abs(h - target["h"]) * 2.0
        + abs(s - target["s"]) * 0.35
        + abs(v - target["v"]) * 0.25
    )


REFERENCE_VALUES = {
    "Wool": {"h": 40, "s": 180, "v": 150},
    "Lumber": {"h": 38, "s": 145, "v": 90},
    "Ore": {"h": 22, "s": 50, "v": 125},
    "Brick": {"h": 16, "s": 200, "v": 130},
    "Grain": {"h": 23, "s": 180, "v": 175},
    "Desert": {"h": 23, "s": 139, "v": 154},
}


DESERT_TILE_ID = 9


def classify_resources(image_bgr, centers, crop_size=42):
    n = len(centers)

    labels = [None] * n
    features = [None] * n

    assigned_counts = {k: 0 for k in RESOURCE_COUNTS}
    assigned_counts["Desert"] = 1
    missing_tiles = []

    for tile_id, x, y in centers:
        tile_patch = crop_tile(
            image_bgr,
            x,
            y,
            size=crop_size,
        )

        hsv = extract_tile_hsv(tile_patch)

        features[tile_id] = hsv

        if tile_id == DESERT_TILE_ID:
            labels[tile_id] = "Desert"
            continue

        label = classify_hsv(
            hsv["h"],
            hsv["s"],
            hsv["v"],
        )

        if label == "Desert":
            label = None

        labels[tile_id] = label

        if label is None:
            missing_tiles.append(tile_id)
        else:
            assigned_counts[label] += 1

    remaining = {
        k: RESOURCE_COUNTS[k] - assigned_counts[k]
        for k in RESOURCE_COUNTS
    }

    # fallback classification
    for tile_id in missing_tiles:
        hsv = features[tile_id]

        best_label = None
        best_score = 1e9

        for label, remaining_count in remaining.items():
            if remaining_count <= 0:
                continue

            score = hsv_distance(
                hsv["h"],
                hsv["s"],
                hsv["v"],
                REFERENCE_VALUES[label],
            )

            if score < best_score:
                best_score = score
                best_label = label

        if best_label is None:
            continue

        labels[tile_id] = best_label
        remaining[best_label] -= 1

    return labels, labels, features


def draw_tile_labels(
    image_bgr,
    centers,
    labels,
    numbers=None,
):
    img = image_bgr.copy()

    number_scale = 0.50
    number_thickness_outer = 2
    number_thickness_inner = 2

    label_scale = 0.32
    label_thickness = 2

    for tile_id, x, y in centers:
        label = labels[tile_id]

        color = RESOURCE_COLORS_BGR.get(
            label,
            (255, 255, 255),
        )

        if numbers is not None and tile_id in numbers:
            txt = str(numbers[tile_id])

            (tw, th), _ = cv2.getTextSize(
                txt,
                cv2.FONT_HERSHEY_SIMPLEX,
                number_scale,
                number_thickness_outer,
            )

            tx = int(x - tw / 2)
            ty = int(y - 5)

            cv2.putText(
                img,
                txt,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                number_scale,
                (255, 255, 255),
                number_thickness_outer,
                cv2.LINE_AA,
            )

            cv2.putText(
                img,
                txt,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                number_scale,
                (0, 0, 0),
                number_thickness_inner,
                cv2.LINE_AA,
            )

        (lw, _), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            label_scale,
            label_thickness,
        )

        lx = int(x - lw / 2)
        ly = int(y + 16)

        cv2.putText(
            img,
            label,
            (lx, ly),
            cv2.FONT_HERSHEY_SIMPLEX,
            label_scale,
            color,
            label_thickness,
            cv2.LINE_AA,
        )

    return img