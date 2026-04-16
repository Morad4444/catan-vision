import cv2
import numpy as np


HOUSE_COLOR_BGR = {
    "orange": (0, 165, 255),
    "blue": (255, 0, 0),
    "red": (0, 0, 255),
}


def estimate_tile_size_from_centers(centers):
    """
    Estimate tile side length from nearest center spacing.
    """
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
    tile_size = median_nn / np.sqrt(3.0)  # Exact for hexagon grid
    return tile_size


def regular_hexagon_points(side_length, center, rotation_rad=0.0):
    """Return regular hexagon vertices in the order expected by order_hexagon_points."""
    a = float(side_length)
    h = a * np.sqrt(3.0) / 2.0
    pts = np.array(
        [
            [-a / 2.0, -h],
            [a / 2.0, -h],
            [a, 0.0],
            [a / 2.0, h],
            [-a / 2.0, h],
            [-a, 0.0],
        ],
        dtype=np.float32,
    )

    if rotation_rad != 0.0:
        c = np.cos(rotation_rad)
        s = np.sin(rotation_rad)
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        pts = pts.dot(rot.T)

    return pts + np.array(center, dtype=np.float32)


def generate_tile_corners_from_centers(centers, tile_size):
    """
    Generate corners for each of the 19 tiles.
    Returns a list of lists: for each tile, a list of 6 (x,y) corner points.
    """
    corners = []
    rotation_rad = np.pi / 6  # 30 degrees rotation to match Catan tile orientation
    for tile_id, cx, cy in centers:
        center = np.array([cx, cy], dtype=np.float32)
        tile_corners = regular_hexagon_points(tile_size, center, rotation_rad)
        corners.append(tile_corners)
    return corners


def analyze_corner_colors(image_bgr, corners, patch_size=9, core_quantile=75, verbose=False):
    """
    Analyze HSV colors at each corner point.
    corners: list of lists, each sublist has 6 (x,y) points for one tile.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    samples = []

    for tile_idx, tile_corners in enumerate(corners):
        if verbose:
            print(f"\nTile {tile_idx} corners:")
        for corner_idx, (x, y) in enumerate(tile_corners):
            x, y = int(round(x)), int(round(y))
            h, w = image_bgr.shape[:2]

            # Crop small patch around corner
            x1 = max(0, x - patch_size // 2)
            y1 = max(0, y - patch_size // 2)
            x2 = min(w, x + patch_size // 2)
            y2 = min(h, y + patch_size // 2)

            patch = hsv[y1:y2, x1:x2]
            if patch.size == 0:
                if verbose:
                    print(f"  Corner {corner_idx}: no patch")
                continue

            flat = patch.reshape(-1, 3).astype(np.float32)
            mean_hsv = np.mean(flat, axis=0)
            saturation_floor = np.percentile(flat[:, 1], core_quantile)
            core_pixels = flat[flat[:, 1] >= saturation_floor]
            if len(core_pixels) == 0:
                core_pixels = flat
            core_hsv = np.mean(core_pixels, axis=0)
            peak_count = min(10, len(flat))
            peak_pixels = flat[np.argsort(flat[:, 1])[-peak_count:]]
            peak_hsv = np.mean(peak_pixels, axis=0)

            sample = {
                "tile_id": tile_idx,
                "corner_id": corner_idx,
                "label": f"T{tile_idx}C{corner_idx}",
                "point": np.array([x, y], dtype=np.float32),
                "mean_hsv": mean_hsv,
                "core_hsv": core_hsv,
                "peak_hsv": peak_hsv,
            }
            samples.append(sample)

            if verbose:
                h_val, s_val, v_val = mean_hsv
                print(f"  Corner {corner_idx}: H={h_val:.1f}, S={s_val:.1f}, V={v_val:.1f}")

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
            group = {
                "point": point.copy(),
                "samples": [],
            }
            groups.append(group)

        group["samples"].append(sample)

    grouped = []
    for group in groups:
        samples_in_group = group["samples"]
        point = np.mean([s["point"] for s in samples_in_group], axis=0)
        mean_hsv = np.mean([s["mean_hsv"] for s in samples_in_group], axis=0)
        core_hsv = np.mean([s["core_hsv"] for s in samples_in_group], axis=0)
        peak_hsv = np.mean([s["peak_hsv"] for s in samples_in_group], axis=0)
        labels = sorted(s["label"] for s in samples_in_group)

        grouped.append(
            {
                "point": point.astype(np.float32),
                "mean_hsv": mean_hsv.astype(np.float32),
                "core_hsv": core_hsv.astype(np.float32),
                "peak_hsv": peak_hsv.astype(np.float32),
                "labels": labels,
                "primary_label": labels[0],
            }
        )

    return grouped


def _threshold_between(sorted_values, lower_count):
    if len(sorted_values) <= lower_count:
        return float(sorted_values[-1])
    return 0.5 * (sorted_values[lower_count - 1] + sorted_values[lower_count])


def detect_houses_from_corner_hsv(samples, expected_counts=None):
    if expected_counts is None:
        expected_counts = {"orange": 3, "blue": 1, "red": 1}

    total_expected = sum(expected_counts.values())
    grouped = _group_corner_samples(samples)
    if len(grouped) < total_expected:
        raise RuntimeError(
            f"Only found {len(grouped)} unique corner groups, expected at least {total_expected}."
        )

    by_saturation = sorted(grouped, key=lambda g: float(g["peak_hsv"][1]), reverse=True)
    candidate_groups = by_saturation[:total_expected]
    saturation_values = [float(g["peak_hsv"][1]) for g in by_saturation]
    sat_threshold = _threshold_between(saturation_values, total_expected)

    detected = [dict(group) for group in candidate_groups]

    by_hue = sorted(detected, key=lambda g: float(g["peak_hsv"][0]))
    blue_count = expected_counts.get("blue", 0)
    warm_groups = by_hue
    blue_threshold = None

    if blue_count:
        split_index = len(by_hue) - blue_count
        blue_threshold = _threshold_between(
            [float(g["peak_hsv"][0]) for g in by_hue],
            split_index,
        )
        warm_groups = by_hue[:split_index]
        for group in by_hue[split_index:]:
            group["color"] = "blue"

    red_count = expected_counts.get("red", 0)
    red_threshold = None
    orange_groups = warm_groups

    if red_count:
        orange_start = red_count
        red_threshold = _threshold_between(
            [float(g["peak_hsv"][0]) for g in warm_groups],
            orange_start,
        )
        for group in warm_groups[:red_count]:
            group["color"] = "red"
        orange_groups = warm_groups[red_count:]

    for group in orange_groups:
        group["color"] = "orange"

    for group in detected:
        x, y = np.round(group["point"]).astype(int)
        group["x"] = int(x)
        group["y"] = int(y)

    detected.sort(key=lambda g: (g["y"], g["x"]))

    thresholds = {
        "house_saturation_threshold": sat_threshold,
        "blue_hue_threshold": blue_threshold,
        "red_hue_threshold": red_threshold,
    }

    return detected, thresholds


def print_detected_houses(detected_houses, thresholds):
    print("\nDetected houses:")
    if not detected_houses:
        print("  no houses detected")
        return

    print(
        "  thresholds: "
        f"S>={thresholds['house_saturation_threshold']:.1f}, "
        f"blue_h>{thresholds['blue_hue_threshold']:.1f}, "
        f"red_h<={thresholds['red_hue_threshold']:.1f}"
    )

    for house in detected_houses:
        h_val, s_val, v_val = house["peak_hsv"]
        labels = ", ".join(house["labels"])
        print(
            f"  {house['color']:>6}: {labels} "
            f"at ({house['x']}, {house['y']}) "
            f"H={h_val:.1f} S={s_val:.1f} V={v_val:.1f}"
        )


def print_detected_house_points(detected_houses):
    print("\nDetected house points:")
    if not detected_houses:
        print("  none")
        return

    for house in detected_houses:
        labels = ", ".join(house["labels"])
        print(f"  {house['color']:>6}: {labels}")


def draw_detected_houses(image_bgr, detected_houses):
    img = image_bgr.copy()

    if not detected_houses:
        cv2.putText(
            img,
            "No houses detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return img

    for house in detected_houses:
        x, y = house["x"], house["y"]
        color = HOUSE_COLOR_BGR[house["color"]]
        cv2.circle(img, (x, y), 10, color, -1)
        cv2.circle(img, (x, y), 16, (255, 255, 255), 2)
        cv2.putText(
            img,
            ", ".join(house["labels"]),
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    return img


def draw_corner_analysis(image_bgr, corners, detected_houses=None, patch_size=10):
    """
    Draw the corners and their analysis ROIs on the image.
    Returns the annotated image.
    """
    img = image_bgr.copy()
    h, w = img.shape[:2]
    house_lookup = {}

    if detected_houses is not None:
        for house in detected_houses:
            key = tuple(np.round(house["point"]).astype(int))
            house_lookup[key] = house

    for tile_idx, tile_corners in enumerate(corners):
        for corner_idx, (x, y) in enumerate(tile_corners):
            x, y = int(round(x)), int(round(y))
            key = (x, y)
            house = house_lookup.get(key)

            # Draw the corner point
            color = (0, 255, 0)
            radius = 5
            if house is not None:
                color = HOUSE_COLOR_BGR[house["color"]]
                radius = 7
            cv2.circle(img, (x, y), radius, color, -1)

            # Draw the ROI rectangle
            x1 = max(0, x - patch_size // 2)
            y1 = max(0, y - patch_size // 2)
            x2 = min(w, x + patch_size // 2)
            y2 = min(h, y + patch_size // 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)  # Blue rectangle for ROI

            # Label the corner
            cv2.putText(
                img,
                f"T{tile_idx}C{corner_idx}",
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color if house is not None else (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return img


