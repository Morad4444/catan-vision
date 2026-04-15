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


# Tuned for your board images
RESOURCE_PROTOTYPES = {
    "Ore":    np.array([108,  70, 105], dtype=np.float32),
    "Brick":  np.array([ 12, 150, 115], dtype=np.float32),
    "Grain":  np.array([ 24, 150, 185], dtype=np.float32),
    "Lumber": np.array([ 55, 115,  85], dtype=np.float32),
    "Wool":   np.array([ 42,  85, 170], dtype=np.float32),
    "Desert": np.array([ 22,  40, 175], dtype=np.float32),
}


def crop_tile(image_bgr, x, y, size=50):
    h, w = image_bgr.shape[:2]
    x1 = max(0, x - size)
    y1 = max(0, y - size)
    x2 = min(w, x + size)
    y2 = min(h, y + size)
    return image_bgr[y1:y2, x1:x2].copy()


def circular_mean_hsv(tile_patch):
    hsv = cv2.cvtColor(tile_patch, cv2.COLOR_BGR2HSV)

    h, w = hsv.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # slightly larger than before, but still avoids hex border
    radius = int(min(h, w) * 0.40)
    cv2.circle(mask, (w // 2, h // 2), radius, 255, -1)

    vals = hsv[mask == 255]
    if len(vals) == 0:
        return np.array([0, 0, 0], dtype=np.float32)

    return np.mean(vals, axis=0).astype(np.float32)


def score_tile(tile_patch):
    hsv_mean = circular_mean_hsv(tile_patch)
    h, s, v = hsv_mean

    scores = {}

    for name, proto in RESOURCE_PROTOTYPES.items():
        ph, ps, pv = proto

        dh = min(abs(h - ph), 180 - abs(h - ph))
        ds = abs(s - ps)
        dv = abs(v - pv)

        # hue is most important, then saturation, then value
        dist = 2.8 * dh + 0.9 * ds + 0.5 * dv
        scores[name] = -float(dist)

    # Desert should be low saturation and relatively bright
    if s < 60:
        scores["Desert"] += 18.0

    # Wool should be greener/brighter than Lumber
    if 35 <= h <= 50 and s < 115 and v > 140:
        scores["Wool"] += 8.0

    # Lumber should be darker and usually more saturated green
    if 45 <= h <= 65 and v < 120:
        scores["Lumber"] += 8.0

    # Grain tends toward yellow
    if 18 <= h <= 30 and s > 110:
        scores["Grain"] += 8.0

    # Brick tends toward orange-brown, darker than grain
    if 5 <= h <= 18 and v < 150:
        scores["Brick"] += 8.0

    # Ore tends blue-gray / low-ish saturation
    if 95 <= h <= 125:
        scores["Ore"] += 8.0

    return scores


def assign_resources_with_counts(all_scores):
    remaining = RESOURCE_COUNTS.copy()
    labels = [None] * len(all_scores)

    candidates = []
    for tile_idx, scores in enumerate(all_scores):
        for name, score in scores.items():
            candidates.append((score, tile_idx, name))

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

        if len(used_tiles) == len(all_scores):
            break

    # fill any leftover slots
    left = []
    for name, count in remaining.items():
        left.extend([name] * count)

    for i in range(len(labels)):
        if labels[i] is None and left:
            labels[i] = left.pop(0)

    return labels


def draw_tile_labels(image_bgr, centers, labels):
    img = image_bgr.copy()

    for (tile_id, x, y), label in zip(centers, labels):
        cv2.putText(
            img,
            label,
            (x - 32, y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return img