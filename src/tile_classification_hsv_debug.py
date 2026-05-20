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
    x1 = max(0, x - size)
    y1 = max(0, y - size)
    x2 = min(w, x + size)
    y2 = min(h, y + size)
    return image_bgr[y1:y2, x1:x2].copy()


def build_inner_mask(h, w, radius_scale=0.30):
    yy, xx = np.mgrid[0:h, 0:w]
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    r = min(h, w) * radius_scale
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r


def extract_tile_hsv(tile_patch: np.ndarray) -> dict:
    hsv = cv2.cvtColor(tile_patch, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]

    mask = build_inner_mask(h, w, radius_scale=0.30)
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
    Simple HSV rules tuned for the current board color ranges.
    """

    # Desert: pale sand with low saturation
    if 10 <= h <= 35 and s <= 110 and v >= 100:
        return "Desert"

    # Ore: greyish stone with high hue and moderate saturation/value
    if 105 <= h <= 135 and s <= 140 and v <= 110:
        return "Ore"

    # Brick: brown / orange hills
    if 0 <= h <= 22 and s >= 55 and 70 <= v <= 145:
        return "Brick"

    # Grain: yellow fields
    if 15 <= h <= 35 and s >= 60 and v >= 105:
        return "Grain"

    # Wool: bright pastel green pasture
    if 55 <= h <= 85 and s >= 70 and v >= 90:
        return "Wool"

    # Lumber: darker forest green
    if 75 <= h <= 105 and s >= 45 and v <= 100:
        return "Lumber"

    return None


def hsv_distance_to_label(h: float, s: float, v: float, label: str) -> float:
    target = {
        "Ore":    {"h": 121.0, "s": 100.0, "v": 70.0},
        "Brick":  {"h": 10.0,  "s": 82.0,  "v": 82.0},
        "Grain":  {"h": 22.0,  "s": 92.0,  "v": 120.0},
        "Lumber": {"h": 88.0,  "s": 82.0,  "v": 52.0},
        "Wool":   {"h": 72.0,  "s": 100.0, "v": 95.0},
        "Desert": {"h": 20.0,  "s": 55.0,  "v": 118.0},
    }[label]
    dh = abs(h - target["h"])
    ds = abs(s - target["s"])
    dv = abs(v - target["v"])
    return 2.0 * dh + 0.35 * ds + 0.25 * dv


def robust_tile_features(tile_patch):
    hsv = cv2.cvtColor(tile_patch, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]

    mask = build_inner_mask(h, w, radius_scale=0.30)
    vals = hsv[mask]
    if len(vals) == 0:
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

    green_frac = float(np.mean((hue >= 40) & (hue <= 95)))
    yellow_frac = float(np.mean((hue >= 12) & (hue <= 40)))
    red_frac = float(np.mean((hue <= 18) | (hue >= 170)))
    blue_frac = float(np.mean((hue >= 95) & (hue <= 145)))
    brown_frac = float(np.mean((hue <= 25) & (sat >= 50) & (sat <= 160) & (vals[:, 2] <= 150)))
    low_sat_frac = float(np.mean(vals[:, 1] < 80))

    return {
        "h": h_med,
        "s": s_med,
        "v": v_med,
        "green_frac": green_frac,
        "yellow_frac": yellow_frac,
        "red_frac": red_frac,
        "blue_frac": blue_frac,
        "brown_frac": brown_frac,
        "low_sat_frac": low_sat_frac,
    }


def score_tile(tile_patch):
    f = robust_tile_features(tile_patch)
    h = f["h"]
    s = f["s"]
    v = f["v"]

    scores = {
        "Ore": -999.0,
        "Brick": -999.0,
        "Grain": -999.0,
        "Lumber": -999.0,
        "Wool": -999.0,
        "Desert": -999.0,
    }

    scores["Desert"] = (
        55.0 * f["low_sat_frac"]
        + 24.0 * f["yellow_frac"]
        + 18.0 * f["brown_frac"]
        - 0.12 * abs(h - 20.0)
        - 0.05 * abs(s - 55.0)
        - 0.05 * abs(v - 118.0)
        + (12.0 if 10 <= h <= 35 and s <= 110 and v >= 100 else 0.0)
    )

    scores["Ore"] = (
        40.0 * f["low_sat_frac"]
        + 12.0 * f["blue_frac"]
        - 0.12 * abs(h - 121.0)
        - 0.04 * abs(s - 100.0)
        - 0.20 * abs(v - 70.0)
        + (12.0 if 105 <= h <= 135 and s <= 140 and v <= 110 else 0.0)
    )

    scores["Brick"] = (
        36.0 * f["red_frac"]
        + 18.0 * f["yellow_frac"]
        + 12.0 * f["brown_frac"]
        - 0.10 * abs(h - 10.0)
        - 0.05 * abs(s - 82.0)
        - 0.05 * abs(v - 82.0)
        + (12.0 if 0 <= h <= 22 and s >= 55 and 70 <= v <= 145 else 0.0)
    )

    scores["Grain"] = (
        45.0 * f["yellow_frac"]
        + 10.0 * f["brown_frac"]
        - 0.12 * abs(h - 22.0)
        - 0.05 * abs(s - 92.0)
        - 0.05 * abs(v - 120.0)
        + (12.0 if 15 <= h <= 35 and s >= 60 and v >= 105 else 0.0)
    )

    scores["Lumber"] = (
        42.0 * f["green_frac"]
        - 0.12 * abs(h - 88.0)
        - 0.05 * abs(s - 82.0)
        - 0.08 * abs(v - 52.0)
        + (12.0 if 75 <= h <= 105 and s >= 45 and v <= 100 else 0.0)
    )

    scores["Wool"] = (
        42.0 * f["green_frac"]
        + 8.0 * f["yellow_frac"]
        - 0.12 * abs(h - 72.0)
        - 0.05 * abs(s - 100.0)
        - 0.08 * abs(v - 95.0)
        + (12.0 if 55 <= h <= 85 and s >= 70 and v >= 90 else 0.0)
    )

    return scores


def assign_resources_with_counts(all_scores):
    remaining = RESOURCE_COUNTS.copy()
    labels = [None] * len(all_scores)

    candidates = []
    for tile_idx, scores in enumerate(all_scores):
        for name, score in scores.items():
            candidates.append((float(score), tile_idx, name))

    candidates.sort(reverse=True, key=lambda x: x[0])

    used_tiles = set()
    for score, tile_idx, name in candidates:
        if tile_idx in used_tiles:
            continue
        if remaining[name] <= 0:
            continue
        labels[tile_idx] = name
        remaining[name] -= 1
        used_tiles.add(tile_idx)

    left = []
    for name, count in remaining.items():
        left.extend([name] * count)

    for i in range(len(labels)):
        if labels[i] is None and left:
            labels[i] = left.pop(0)

    return labels


def classify_resources(image_bgr, centers, crop_size=36):
    all_scores = []
    all_features = []
    raw_labels: list[str | None] = [None] * len(centers)

    for tile_id, x, y in centers:
        tile_patch = crop_tile(image_bgr, x, y, size=crop_size)
        hsv = extract_tile_hsv(tile_patch)
        all_features.append(hsv)

        raw_label = classify_hsv_simple(hsv["h"], hsv["s"], hsv["v"])
        if raw_label is None:
            distances = {
                label: hsv_distance_to_label(hsv["h"], hsv["s"], hsv["v"], label)
                for label in RESOURCE_COUNTS
            }
            raw_label = min(distances, key=distances.get)

        raw_labels[tile_id] = raw_label
        all_scores.append(score_tile(tile_patch))

    labels = assign_resources_with_counts(all_scores)
    return labels, raw_labels, all_features


def draw_tile_labels(image_bgr, centers, labels, numbers=None):
    img = image_bgr.copy()

    for tile_id, x, y in centers:
        label = labels[tile_id]
        color = RESOURCE_COLORS_BGR.get(label, (255, 255, 255))

        if numbers is not None and tile_id in numbers:
            txt = str(numbers[tile_id])
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.95, 3)
            tx = int(x - tw / 2)
            ty = int(y + th / 2 - 6)
            cv2.putText(
                img, txt, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 3, cv2.LINE_AA
            )
            cv2.putText(
                img, txt, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 0), 1, cv2.LINE_AA
            )

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


def draw_tile_hsv_values(image_bgr, centers, features, numbers=None):
    img = image_bgr.copy()

    for tile_id, x, y in centers:
        f = features[tile_id]
        h_txt = f"H:{int(round(f['h']))}"
        s_txt = f"S:{int(round(f['s']))}"
        v_txt = f"V:{int(round(f['v']))}"

        if numbers is not None and tile_id in numbers:
            num_txt = str(numbers[tile_id])
            (tw, th), _ = cv2.getTextSize(num_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.95, 3)
            tx = int(x - tw / 2)
            ty = int(y + th / 2 - 6)
            cv2.putText(img, num_txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(img, num_txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 0), 1, cv2.LINE_AA)

        lines = [h_txt, s_txt, v_txt]
        start_y = int(y + 26)
        for i, txt in enumerate(lines):
            yy = start_y + i * 18
            cv2.putText(img, txt, (int(x - 34), yy), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, txt, (int(x - 34), yy), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

    return img
