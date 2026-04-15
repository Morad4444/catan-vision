import cv2
import numpy as np
from pathlib import Path


def estimate_tile_size_from_centers(centers):
    """
    Estimate tile size from nearest-neighbor spacing between tile centers.
    centers: list of (tile_id, x, y)
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
    tile_size = median_nn / 1.5
    return tile_size


def crop_center_patch(image_bgr, center_x, center_y, half_size):
    """
    Crop a square patch centered at (center_x, center_y).
    Returns:
        patch, x1, y1
    where (x1, y1) is the top-left corner in the original image.
    """
    h, w = image_bgr.shape[:2]
    half_size = int(round(half_size))

    x1 = max(0, int(center_x - half_size))
    y1 = max(0, int(center_y - half_size))
    x2 = min(w, int(center_x + half_size))
    y2 = min(h, int(center_y + half_size))

    patch = image_bgr[y1:y2, x1:x2].copy()
    return patch, x1, y1


def crop_tile_patch(image_bgr, center_x, center_y, tile_size, margin_factor=0.60):
    """
    Crop the tile patch around the tile center.
    This is saved as tile_*.png
    """
    half = tile_size * margin_factor
    return crop_center_patch(image_bgr, center_x, center_y, half)


def detect_chip_in_tile_patch(tile_patch_bgr):
    """
    Detect the white chip inside one tile patch using HoughCircles.

    Returns:
        best_circle = (x, y, r) in tile-patch coordinates, or None
        tile_patch_detected = tile patch with circle drawn if found
        chip_patch = cropped chip image from detected circle, or None
    """
    img = tile_patch_bgr.copy()
    h, w = img.shape[:2]

    if h == 0 or w == 0:
        return None, img, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(18, min(h, w) // 4),
        param1=100,
        param2=18,
        minRadius=max(10, int(min(h, w) * 0.10)),
        maxRadius=max(20, int(min(h, w) * 0.28)),
    )

    best_circle = None
    best_score = -1e9

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    patch_cx = w / 2.0
    patch_cy = h / 2.0

    if circles is not None:
        circles = np.round(circles[0]).astype(int)

        for (x, y, r) in circles:
            if r <= 0:
                continue

            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (x, y), r, 255, -1)

            inside = mask == 255
            if np.count_nonzero(inside) == 0:
                continue

            mean_v = float(np.mean(v[inside]))
            mean_s = float(np.mean(s[inside]))

            # Prefer bright, low-saturation, centered circles
            score = mean_v - 0.8 * mean_s

            dist = np.sqrt((x - patch_cx) ** 2 + (y - patch_cy) ** 2)
            score -= 0.35 * dist

            if score > best_score:
                best_score = score
                best_circle = (x, y, r)

    tile_patch_detected = img.copy()
    chip_patch = None

    if best_circle is not None:
        x, y, r = best_circle

        cv2.circle(tile_patch_detected, (x, y), r, (0, 255, 0), 2)
        cv2.circle(tile_patch_detected, (x, y), 2, (0, 0, 255), -1)

        inner_r = max(1, int(round(r * 0.78)))
        x1 = max(0, x - inner_r)
        x2 = min(w, x + inner_r)
        y1 = max(0, y - inner_r)
        y2 = min(h, y + inner_r)

        chip_patch = img[y1:y2, x1:x2].copy()

    return best_circle, tile_patch_detected, chip_patch


def detect_chips(image_bgr, centers, tile_margin_factor=0.60):
    """
    1) Take a fixed tile crop around each tile center -> tile_*.png
    2) Run Hough circle detection inside that tile patch
    3) Save drawn result as tile_*_detected.png
    4) Save chip-only crop as tile_*_chip.png

    Returns one result per tile center.
    """
    tile_size = estimate_tile_size_from_centers(centers)
    detections = []

    detected_rs = []

    for tile_id, tx, ty in centers:
        tile_patch, x_offset, y_offset = crop_tile_patch(
            image_bgr,
            tx,
            ty,
            tile_size,
            margin_factor=tile_margin_factor,
        )

        best_circle, tile_patch_detected, chip_patch = detect_chip_in_tile_patch(tile_patch)

        if best_circle is None:
            fallback_r = int(round(tile_size * 0.22))
            detections.append(
                {
                    "tile_id": tile_id,
                    "tile_x": tx,
                    "tile_y": ty,
                    "chip_x": int(tx),
                    "chip_y": int(ty),
                    "chip_r_detected": fallback_r,
                    "chip_r": int(round(fallback_r * 0.78)),
                    "detected": False,
                    "tile_patch": tile_patch,
                    "chip_patch": None,
                    "tile_patch_detected": tile_patch_detected,
                }
            )
            continue

        local_x, local_y, r = best_circle
        chip_x = int(x_offset + local_x)
        chip_y = int(y_offset + local_y)
        chip_r = int(round(r * 0.78))

        detected_rs.append(r)

        detections.append(
            {
                "tile_id": tile_id,
                "tile_x": tx,
                "tile_y": ty,
                "chip_x": chip_x,
                "chip_y": chip_y,
                "chip_r_detected": int(r),
                "chip_r": chip_r,
                "detected": True,
                "tile_patch": tile_patch,
                "chip_patch": chip_patch,
                "tile_patch_detected": tile_patch_detected,
            }
        )

    # Optional normalization of displayed radii so overlays look consistent
    if detected_rs:
        avg_r = int(round(np.mean(detected_rs)))
        avg_inner = int(round(avg_r * 0.78))

        for item in detections:
            if item["detected"]:
                item["chip_r_detected"] = avg_r
                item["chip_r"] = avg_inner

    return detections


def save_chip_debug_patches(chips, save_dir):
    """
    Save:
      - tile_*.png          = centered tile crop
      - tile_*_detected.png = centered tile crop with detected circle
      - tile_*_chip.png     = chip-only crop from detected circle
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for item in chips:
        tile_id = item["tile_id"]

        tile_patch = item.get("tile_patch", None)
        chip_patch = item.get("chip_patch", None)
        tile_patch_detected = item.get("tile_patch_detected", None)

        if tile_patch is not None:
            cv2.imwrite(str(save_dir / f"tile_{tile_id}.png"), tile_patch)

        if tile_patch_detected is not None:
            cv2.imwrite(str(save_dir / f"tile_{tile_id}_detected.png"), tile_patch_detected)

        if chip_patch is not None:
            cv2.imwrite(str(save_dir / f"tile_{tile_id}_chip.png"), chip_patch)


def assign_chips_to_tiles(chips, labels):
    """
    Keep only chips on non-desert tiles.
    """
    assignments = []

    for chip, label in zip(chips, labels):
        if label == "Desert":
            continue

        item = dict(chip)
        item["label"] = label
        assignments.append(item)

    return assignments


def draw_chips(image_bgr, chips):
    """
    Draw chip detections for all tiles.
    Green = detected circle
    Yellow = inner crop radius
    Orange = fallback when not detected
    """
    img = image_bgr.copy()

    for item in chips:
        tile_id = item["tile_id"]
        x = item["chip_x"]
        y = item["chip_y"]
        r_detected = item["chip_r_detected"]
        r_inner = item["chip_r"]

        color = (0, 255, 0) if item["detected"] else (0, 128, 255)

        cv2.circle(img, (x, y), r_detected, color, 2)
        cv2.circle(img, (x, y), r_inner, (0, 255, 255), 1)
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

        cv2.putText(
            img,
            str(tile_id),
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return img


def draw_chip_tile_assignments(image_bgr, assignments):
    """
    Draw final chip assignments on non-desert tiles only.
    """
    img = image_bgr.copy()

    for item in assignments:
        tile_id = item["tile_id"]
        x = item["chip_x"]
        y = item["chip_y"]
        r_detected = item["chip_r_detected"]
        r_inner = item["chip_r"]

        color = (0, 255, 0) if item["detected"] else (0, 128, 255)

        cv2.circle(img, (x, y), r_detected, color, 2)
        cv2.circle(img, (x, y), r_inner, (0, 255, 255), 1)
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

        cv2.putText(
            img,
            f"T{tile_id}",
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return img