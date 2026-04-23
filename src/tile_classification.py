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

# Target HSV centers based on your live screenshots
RESOURCE_HSV_CENTERS = {
    "Ore":    {"h": 121.0, "s": 100.0, "v": 70.0},
    "Brick":  {"h": 10.0,  "s": 82.0,  "v": 82.0},
    "Grain":  {"h": 22.0,  "s": 92.0,  "v": 120.0},
    "Lumber": {"h": 88.0,  "s": 82.0,  "v": 52.0},
    "Wool":   {"h": 72.0,  "s": 100.0, "v": 95.0},
    "Desert": {"h": 20.0,  "s": 55.0,  "v": 118.0},
}


def crop_tile(image_bgr, x, y, size=42):
    h, w = image_bgr.shape[:2]
    x1 = max(0, x - size)
    y1 = max(0, y - size)
    x2 = min(w, x + size)
    y2 = min(h, y + size)
    return image_bgr[y1:y2, x1:x2].copy()


def _build_inner_mask(h: int, w: int, radius_scale: float = 0.30) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    r = min(h, w) * radius_scale
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r


def extract_tile_hsv(tile_patch: np.ndarray) -> dict:
    hsv = cv2.cvtColor(tile_patch, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]

    mask = _build_inner_mask(h, w, radius_scale=0.30)
    vals = hsv[mask]
    if len(vals) == 0:
        return {"h": 0.0, "s": 0.0, "v": 0.0}

    vals = vals.astype(np.float32)

    v_lo = np.percentile(vals[:, 2], 8)
    v_hi = np.percentile(vals[:, 2], 92)
    keep = (vals[:, 2] >= v_lo) & (vals[:, 2] <= v_hi)
    core = vals[keep] if np.count_nonzero(keep) >= 20 else vals

    h_med = float(np.median(core[:, 0]))
    s_med = float(np.median(core[:, 1]))
    v_med = float(np.median(core[:, 2]))

    return {"h": h_med, "s": s_med, "v": v_med}


def classify_hsv_simple(h: float, s: float, v: float) -> str | None:
    """
    Rule-based HSV classification using the ranges visible in your screenshot.
    Returns the resource name, or None if no rule matches.
    """

    # Ore: blue / purple mountains
    if 100 <= h <= 112 and 30 <= s <= 60 and 130 <= v <= 160:
        return "Ore"

    # Lumber: darker green forest
    elif 68 <= h <= 90 and 40 <= s <= 80 and 100 <= v <= 120:
        return "Lumber"

    # Wool: brighter green pasture
    elif 55 <= h <= 66 and 60 <= s <= 85 and 128 <= v <= 155:
        return "Wool"

    # Grain: yellow fields
    elif 24 <= h <= 38 and 40 <= s <= 75 and 140 <= v <= 160:
        return "Grain"

    # Desert: beige / pale yellow, slightly darker than grain
    elif 24 <= h <= 34 and 60 <= s <= 80 and 128 <= v <= 140:
        return "Desert"

    # Brick: brown / reddish hills
    elif 45 <= h <= 95 and 25 <= s <= 40 and 115 <= v <= 130:
        return "Brick"

    return None


def hsv_distance_to_label(h: float, s: float, v: float, label: str) -> float:
    target = RESOURCE_HSV_CENTERS[label]

    dh = abs(h - target["h"])
    ds = abs(s - target["s"])
    dv = abs(v - target["v"])

    # Weighted distance
    return 2.0 * dh + 0.35 * ds + 0.25 * dv


def classify_resources(image_bgr: np.ndarray, centers: list, crop_size: int = 42):
    """
    Returns:
        labels: final labels
        raw_labels: direct rule labels, may contain None
        features: per-tile HSV
    """
    n = len(centers)
    raw_labels: list[str | None] = [None] * n
    features: list[dict] = [None] * n  # type: ignore[assignment]

    assigned_counts = {name: 0 for name in RESOURCE_COUNTS}
    missing_tile_ids: list[int] = []

    for tile_id, x, y in centers:
        tile_patch = crop_tile(image_bgr, x, y, size=crop_size)
        hsv = extract_tile_hsv(tile_patch)
        features[tile_id] = hsv

        label = classify_hsv_simple(hsv["h"], hsv["s"], hsv["v"])
        raw_labels[tile_id] = label

        if label is None:
            missing_tile_ids.append(tile_id)
        else:
            assigned_counts[label] += 1

    remaining_counts = {
        name: RESOURCE_COUNTS[name] - assigned_counts[name]
        for name in RESOURCE_COUNTS
    }

    final_labels = raw_labels[:]

    # Fallback score assignment only for missing tiles
    for tile_id in missing_tile_ids:
        hsv = features[tile_id]
        h, s, v = hsv["h"], hsv["s"], hsv["v"]

        best_label = None
        best_score = float("inf")

        for label, count_left in remaining_counts.items():
            if count_left <= 0:
                continue

            score = hsv_distance_to_label(h, s, v, label)
            if score < best_score:
                best_score = score
                best_label = label

        if best_label is None:
            # safety fallback
            for label, count_left in remaining_counts.items():
                if count_left > 0:
                    best_label = label
                    break

        final_labels[tile_id] = best_label
        remaining_counts[best_label] -= 1  # type: ignore[index]

    return final_labels, raw_labels, features


def draw_tile_labels(
    image_bgr: np.ndarray,
    centers: list,
    labels: list[str],
    numbers: dict | None = None,
):
    img = image_bgr.copy()

    for tile_id, x, y in centers:
        label = labels[tile_id]
        color = RESOURCE_COLORS_BGR.get(label, (255, 255, 255))

        if numbers is not None and tile_id in numbers:
            txt = str(numbers[tile_id])
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.95, 3)
            tx = int(x - tw / 2)
            ty = int(y + th / 2 - 6)
            cv2.putText(img, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(img, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 0), 1, cv2.LINE_AA)

        label_y = int(y + 34)
        if label == "Desert":
            label_y = int(y + 28)

        cv2.putText(
            img,
            label,
            (int(x - 40), label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            color,
            2,
            cv2.LINE_AA,
        )

    return img