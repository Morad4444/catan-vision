from __future__ import annotations

import cv2
import numpy as np


HOUSE_COLOR_BGR = {
    "orange": (0, 165, 255),
    "blue": (255, 0, 0),
    "red": (0, 0, 255),
}


def estimate_tile_size_from_centers(centers):
    pts = np.array([(x, y) for _, x, y in centers], dtype=np.float32)
    nearest = []
    for i in range(len(pts)):
        dists = []
        for j in range(len(pts)):
            if i == j:
                continue
            dists.append(np.linalg.norm(pts[i] - pts[j]))
        nearest.append(min(dists))
    median_nn = float(np.median(nearest))
    return median_nn / np.sqrt(3.0)


def regular_hexagon_points(side_length, center, rotation_rad=0.0):
    a = float(side_length)
    h = a * np.sqrt(3.0) / 2.0
    pts = np.array([
        [-a / 2.0, -h],
        [a / 2.0, -h],
        [a, 0.0],
        [a / 2.0, h],
        [-a / 2.0, h],
        [-a, 0.0],
    ], dtype=np.float32)
    if rotation_rad != 0.0:
        c = np.cos(rotation_rad)
        s = np.sin(rotation_rad)
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        pts = pts.dot(rot.T)
    return pts + np.array(center, dtype=np.float32)


def generate_tile_corners_from_centers(centers, tile_size):
    corners = []
    rotation_rad = np.pi / 6
    for tile_id, cx, cy in centers:
        center = np.array([cx, cy], dtype=np.float32)
        tile_corners = regular_hexagon_points(tile_size, center, rotation_rad)
        corners.append(tile_corners)
    return corners


def analyze_corner_colors(image_bgr, corners, patch_size=9, core_quantile=75):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    samples = []
    for tile_idx, tile_corners in enumerate(corners):
        for corner_idx, (x, y) in enumerate(tile_corners):
            x, y = int(round(x)), int(round(y))
            h, w = image_bgr.shape[:2]
            x1 = max(0, x - patch_size // 2)
            y1 = max(0, y - patch_size // 2)
            x2 = min(w, x + patch_size // 2)
            y2 = min(h, y + patch_size // 2)
            patch = hsv[y1:y2, x1:x2]
            if patch.size == 0:
                continue
            flat = patch.reshape(-1, 3).astype(np.float32)
            mean_hsv = np.mean(flat, axis=0)
            saturation_floor = np.percentile(flat[:, 1], core_quantile)
            core_pixels = flat[flat[:, 1] >= saturation_floor]
            if len(core_pixels) == 0:
                core_pixels = flat
            peak_count = min(10, len(flat))
            peak_pixels = flat[np.argsort(flat[:, 1])[-peak_count:]]
            samples.append({
                "tile_id": tile_idx,
                "corner_id": corner_idx,
                "label": f"T{tile_idx}C{corner_idx}",
                "point": np.array([x, y], dtype=np.float32),
                "mean_hsv": mean_hsv,
                "core_hsv": np.mean(core_pixels, axis=0),
                "peak_hsv": np.mean(peak_pixels, axis=0),
            })
    return samples


def _group_corner_samples(samples, merge_distance=4.0):
    groups = []
    for sample in samples:
        point = sample["point"]
        group = None
        for existing in groups:
            if np.linalg.norm(point - existing["point"]) <= merge_distance:
                group = existing
                break
        if group is None:
            group = {"point": point.copy(), "samples": []}
            groups.append(group)
        group["samples"].append(sample)

    grouped = []
    for group in groups:
        samples_in_group = group["samples"]
        point = np.mean([s["point"] for s in samples_in_group], axis=0)
        peak_hsv = np.mean([s["peak_hsv"] for s in samples_in_group], axis=0)
        labels = sorted(s["label"] for s in samples_in_group)
        grouped.append({
            "point": point.astype(np.float32),
            "peak_hsv": peak_hsv.astype(np.float32),
            "labels": labels,
            "primary_label": labels[0],
        })
    return grouped


def classify_house_color(peak_hsv, sat_threshold=95):
    h, s, v = [float(x) for x in peak_hsv]
    if s < sat_threshold or v < 50:
        return None
    if 95 <= h <= 135:
        return "blue"
    if h <= 12 or h >= 170:
        return "red"
    if 6 <= h <= 28:
        return "orange"
    return None


def detect_houses_from_corner_hsv(samples, merge_distance=4.0, sat_threshold=95):
    grouped = _group_corner_samples(samples, merge_distance=merge_distance)
    detected = []
    for group in grouped:
        color = classify_house_color(group["peak_hsv"], sat_threshold=sat_threshold)
        if color is None:
            continue
        x, y = np.round(group["point"]).astype(int)
        item = dict(group)
        item["color"] = color
        item["x"] = int(x)
        item["y"] = int(y)
        detected.append(item)
    detected.sort(key=lambda g: (g["y"], g["x"]))
    thresholds = {"house_saturation_threshold": sat_threshold}
    return detected, thresholds


def detect_settlements(image_bgr, centers):
    tile_size = estimate_tile_size_from_centers(centers)
    tile_corners = generate_tile_corners_from_centers(centers, tile_size)
    corner_samples = analyze_corner_colors(image_bgr, tile_corners)
    detected_houses, thresholds = detect_houses_from_corner_hsv(corner_samples)
    return detected_houses, thresholds, tile_corners


def summarize_settlement_changes(previous_houses, current_houses, move_tolerance=14):
    if previous_houses is None:
        return {"new": current_houses, "kept": [], "removed": []}

    def matched(a, b):
        if a["color"] != b["color"]:
            return False
        return np.hypot(a["x"] - b["x"], a["y"] - b["y"]) <= move_tolerance

    kept = []
    new_items = []
    removed = []
    used_prev = set()

    for cur in current_houses:
        hit = None
        for i, prev in enumerate(previous_houses):
            if i in used_prev:
                continue
            if matched(prev, cur):
                hit = i
                break
        if hit is None:
            new_items.append(cur)
        else:
            used_prev.add(hit)
            kept.append(cur)

    for i, prev in enumerate(previous_houses):
        if i not in used_prev:
            removed.append(prev)

    return {"new": new_items, "kept": kept, "removed": removed}


def draw_detected_houses(image_bgr, detected_houses, new_houses=None):
    img = image_bgr.copy()
    new_houses = new_houses or []
    new_keys = {(h["color"], h["x"], h["y"]) for h in new_houses}

    for house in detected_houses:
        x, y = house["x"], house["y"]
        color = HOUSE_COLOR_BGR[house["color"]]
        cv2.circle(img, (x, y), 10, color, -1)
        outline = (0, 255, 255) if (house["color"], x, y) in new_keys else (255, 255, 255)
        thickness = 3 if (house["color"], x, y) in new_keys else 2
        cv2.circle(img, (x, y), 16, outline, thickness)
        cv2.putText(img, house["color"], (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return img