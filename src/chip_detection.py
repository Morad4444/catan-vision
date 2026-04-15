import cv2
import numpy as np
from pathlib import Path


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
    tile_size = median_nn / 1.5
    return tile_size


def crop_tile_patch(image_bgr, center_x, center_y, tile_size, margin_factor=0.95):
    """
    Crop one local tile patch around a tile center.
    This is saved as tile_*.png
    """
    h, w = image_bgr.shape[:2]
    half = int(round(tile_size * margin_factor))

    x1 = max(0, int(center_x - half))
    y1 = max(0, int(center_y - half))
    x2 = min(w, int(center_x + half))
    y2 = min(h, int(center_y + half))

    patch = image_bgr[y1:y2, x1:x2].copy()
    return patch, x1, y1


def detect_white_chip(img_rgb):
    """
    Detect the best white chip inside one tile patch.
    Returns:
        result_img_rgb, chip_crop_rgb, best_circle
    where best_circle = (x, y, r) in tile-patch coordinates.
    """
    img = img_rgb.copy()
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(20, min(h, w) // 3),
        param1=100,
        param2=22,
        minRadius=max(12, int(min(h, w) * 0.12)),
        maxRadius=max(25, int(min(h, w) * 0.33)),
    )

    best_circle = None
    best_score = -1e9

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        for (x, y, r) in circles:
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (x, y), r, 255, -1)

            inside = mask == 255
            if np.count_nonzero(inside) == 0:
                continue

            brightness = float(np.mean(v[inside]))
            saturation = float(np.mean(s[inside]))

            score = brightness - saturation

            dist_to_center = np.sqrt((x - w / 2.0) ** 2 + (y - h / 2.0) ** 2)
            score -= 0.5 * dist_to_center

            if score > best_score:
                best_score = score
                best_circle = (x, y, r)

    result_img = img.copy()
    chip_crop = None

    if best_circle is not None:
        x, y, r = best_circle

        cv2.circle(result_img, (x, y), r, (0, 255, 0), 2)
        cv2.circle(result_img, (x, y), 2, (255, 0, 0), 3)

        x1 = max(0, x - r)
        x2 = min(w, x + r)
        y1 = max(0, y - r)
        y2 = min(h, y + r)
        chip_crop = img[y1:y2, x1:x2].copy()

    return result_img, chip_crop, best_circle


def detect_chip_near_center(image_bgr, center_x, center_y, tile_size):
    """
    Two-stage chip detection:
    1) crop tile patch around tile center
    2) run detect_white_chip on tile patch
    Returns global coordinates:
        (chip_x, chip_y, r_detected, r_inner, tile_patch, chip_patch, tile_patch_detected)
    or None if not found.
    """
    tile_patch_bgr, x_offset, y_offset = crop_tile_patch(
        image_bgr,
        center_x,
        center_y,
        tile_size,
        margin_factor=0.95,
    )

    tile_patch_rgb = cv2.cvtColor(tile_patch_bgr, cv2.COLOR_BGR2RGB)
    tile_patch_detected_rgb, chip_patch_rgb, best_circle = detect_white_chip(tile_patch_rgb)

    if best_circle is None:
        return None

    local_x, local_y, r_detected = best_circle
    r_inner = max(1, int(round(r_detected * 0.68)))

    # small refinement toward tile-patch center
    patch_h, patch_w = tile_patch_bgr.shape[:2]
    patch_cx = patch_w // 2
    patch_cy = patch_h // 2

    dx = local_x - patch_cx
    dy = local_y - patch_cy

    max_shift = int(0.25 * r_detected)
    dx = max(-max_shift, min(max_shift, dx))
    dy = max(-max_shift, min(max_shift, dy))

    refined_local_x = int(round(patch_cx + 0.5 * dx))
    refined_local_y = int(round(patch_cy + 0.5 * dy))

    chip_x = int(x_offset + refined_local_x)
    chip_y = int(y_offset + refined_local_y)

    tile_patch_detected_bgr = cv2.cvtColor(tile_patch_detected_rgb, cv2.COLOR_RGB2BGR)
    chip_patch_bgr = None
    if chip_patch_rgb is not None:
        chip_patch_bgr = cv2.cvtColor(chip_patch_rgb, cv2.COLOR_RGB2BGR)

    return (
        chip_x,
        chip_y,
        int(r_detected),
        r_inner,
        tile_patch_bgr,
        chip_patch_bgr,
        tile_patch_detected_bgr,
    )


def detect_chips(image_bgr, centers):
    """
    Detect chips near the tile centers.
    Returns one result per tile center.
    """
    tile_size = estimate_tile_size_from_centers(centers)
    detections = []

    outer_rs = []
    inner_rs = []

    for tile_id, tx, ty in centers:
        result = detect_chip_near_center(image_bgr, tx, ty, tile_size)

        if result is None:
            detections.append(
                {
                    "tile_id": tile_id,
                    "tile_x": tx,
                    "tile_y": ty,
                    "chip_x": tx,
                    "chip_y": ty,
                    "chip_r_detected": int(round(tile_size * 0.42)),
                    "chip_r": int(round(tile_size * 0.33)),
                    "detected": False,
                    "tile_patch": None,
                    "chip_patch": None,
                    "tile_patch_detected": None,
                }
            )
            continue

        x, y, r_detected, r_inner, tile_patch, chip_patch, tile_patch_detected = result

        outer_rs.append(r_detected)
        inner_rs.append(r_inner)

        detections.append(
            {
                "tile_id": tile_id,
                "tile_x": tx,
                "tile_y": ty,
                "chip_x": x,
                "chip_y": y,
                "chip_r_detected": r_detected,
                "chip_r": r_inner,
                "detected": True,
                "tile_patch": tile_patch,
                "chip_patch": chip_patch,
                "tile_patch_detected": tile_patch_detected,
            }
        )

    if outer_rs:
        avg_outer = int(round(np.mean(outer_rs)))
        avg_inner = int(round(np.mean(inner_rs)))

        for item in detections:
            if item["detected"]:
                item["chip_r_detected"] = avg_outer
                item["chip_r"] = avg_inner

    return detections


def save_chip_debug_patches(chips, save_dir):
    """
    Save outputs:
      - tile_*.png          = original tile patch
      - tile_*_detected.png = tile patch with detected circle drawn
      - tile_*_chip.png     = cropped chip only
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
    Draw all local chip detections.
    Green = detected outer circle
    Yellow = inner crop radius
    Orange = fallback circle when not detected
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

        cv2.circle(img, (x, y), r_detected, (0, 255, 0), 2)
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