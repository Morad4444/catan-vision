from pathlib import Path
import cv2
import numpy as np


VALID_NUMBERS = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]

CATAN_NUMBER_COUNTS = {
    2: 1,
    3: 2,
    4: 2,
    5: 2,
    6: 2,
    8: 2,
    9: 2,
    10: 2,
    11: 2,
    12: 1,
}


def center_crop_square(image):
    h, w = image.shape[:2]
    side = min(h, w)
    x1 = (w - side) // 2
    y1 = (h - side) // 2
    return image[y1:y1 + side, x1:x1 + side].copy()


def make_circular_mask(shape, radius_scale=0.90):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cx = w // 2
    cy = h // 2
    r = int(round(min(h, w) * 0.5 * radius_scale))
    cv2.circle(mask, (cx, cy), r, 255, -1)
    return mask


def preprocess_chip_base(image_bgr, output_size=96):
    """
    Base preprocessing of chip/token image.
    Returns a normalized grayscale token image.
    """
    if image_bgr is None or image_bgr.size == 0:
        return np.zeros((output_size, output_size), dtype=np.uint8)

    patch = center_crop_square(image_bgr)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

    mask = make_circular_mask(gray.shape, radius_scale=0.90)
    gray = cv2.bitwise_and(gray, gray, mask=mask)

    # Normalize only inside token area
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.resize(gray, (output_size, output_size), interpolation=cv2.INTER_LINEAR)

    return gray


def extract_symbol_mask(image_bgr, output_size=96):
    """
    Extract dark printed ink from bright chip.
    Returns:
      binary_mask: white symbol on black background
      digit_mask: symbol part without most dots
      pip_count: detected count of lower dots
      pip_centers_y_mean: mean y of detected lower dots or None
    """
    gray = preprocess_chip_base(image_bgr, output_size=output_size)

    # Dark ink on bright token -> invert threshold
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        8,
    )

    # keep only near token area
    circle_mask = make_circular_mask(binary.shape, radius_scale=0.90)
    binary = cv2.bitwise_and(binary, binary, mask=circle_mask)

    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    h, w = binary.shape[:2]
    img_center_y = h / 2.0

    digit_mask = np.zeros_like(binary)
    pip_components = []

    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        cx, cy = centroids[i]

        if area < 6:
            continue

        # likely pip: small, compact, and lower than main digit center
        if area <= 120 and cw <= 18 and ch <= 18 and cy > img_center_y * 0.85:
            pip_components.append((i, area, cx, cy))
            continue

        # everything else goes to digit mask
        digit_mask[labels == i] = 255

    # Keep only lower pips; dots are under the number
    pip_components = [p for p in pip_components if p[3] > img_center_y * 0.85]
    pip_count = len(pip_components)
    pip_centers_y_mean = float(np.mean([p[3] for p in pip_components])) if pip_components else None

    return binary, digit_mask, pip_count, pip_centers_y_mean


def tight_symbol_crop(binary_mask, output_size=64, fill_ratio=0.72):
    """
    Crop tightly to foreground and place on centered square canvas.
    """
    ys, xs = np.where(binary_mask > 0)
    canvas = np.zeros((output_size, output_size), dtype=np.uint8)

    if len(xs) == 0 or len(ys) == 0:
        return canvas

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    cropped = binary_mask[y1:y2 + 1, x1:x2 + 1]
    ch, cw = cropped.shape[:2]
    if ch == 0 or cw == 0:
        return canvas

    scale = min((output_size * fill_ratio) / cw, (output_size * fill_ratio) / ch)
    new_w = max(1, int(round(cw * scale)))
    new_h = max(1, int(round(ch * scale)))

    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    ox = (output_size - new_w) // 2
    oy = (output_size - new_h) // 2
    canvas[oy:oy + new_h, ox:ox + new_w] = resized

    return canvas


def preprocess_chip_or_template(image_bgr, output_size=64):
    """
    Final comparison image:
    only the symbol, not the full token.
    """
    _, digit_mask, _, _ = extract_symbol_mask(image_bgr, output_size=96)
    return tight_symbol_crop(digit_mask, output_size=output_size, fill_ratio=0.72)


def load_number_templates(template_dir, output_size=64):
    template_dir = Path(template_dir)
    templates = {}

    for number in VALID_NUMBERS:
        path = template_dir / f"{number}.png"
        if not path.exists():
            continue

        image = cv2.imread(str(path))
        if image is None:
            continue

        symbol = preprocess_chip_or_template(image, output_size=output_size)
        _, _, pip_count, pip_y = extract_symbol_mask(image, output_size=96)

        templates[number] = {
            "symbol": symbol,
            "pip_count": pip_count,
            "pip_y": pip_y,
        }

    if not templates:
        raise RuntimeError(f"No valid templates found in {template_dir}")

    return templates


def rotate_image_keep_size(image, angle_deg):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def template_score(chip_symbol, template_symbol):
    chip_f = chip_symbol.astype(np.float32) / 255.0
    templ_f = template_symbol.astype(np.float32) / 255.0

    num = float(np.sum(chip_f * templ_f))
    den = float(np.sqrt(np.sum(chip_f * chip_f) * np.sum(templ_f * templ_f)) + 1e-8)
    return num / den


def apply_number_priors(number, base_score, pip_count):
    """
    Adjust score with Catan-specific priors.
    Dots are under the number and are especially useful for 6 and 9.
    """
    score = float(base_score)

    if pip_count > 0:
        # Standard Catan token probabilities
        if number in (6, 8):
            expected = 5
        elif number in (5, 9):
            expected = 4
        elif number in (4, 10):
            expected = 3
        elif number in (3, 11):
            expected = 2
        elif number in (2, 12):
            expected = 1
        else:
            expected = 0

        score -= 0.08 * abs(pip_count - expected)

    return score


def recognize_one_chip(chip_patch_bgr, templates, rotations=range(-60, 61, 10)):
    """
    Compare one chip patch to all templates using symbol-only matching.
    """
    _, digit_mask, pip_count, pip_y = extract_symbol_mask(chip_patch_bgr, output_size=96)
    chip_symbol = tight_symbol_crop(digit_mask, output_size=64, fill_ratio=0.72)

    best_number = None
    best_score = -1e9
    best_rotation = 0
    best_per_number = {}

    for angle in rotations:
        rotated_chip = rotate_image_keep_size(chip_symbol, angle)

        for number, templ in templates.items():
            score = template_score(rotated_chip, templ["symbol"])
            score = apply_number_priors(number, score, pip_count)

            # Extra tie-breaker for 6 vs 9:
            # user rule: dots are under the number, so if there are lower dots and
            # the shape is otherwise ambiguous, prefer the upright interpretation
            if pip_count >= 4 and number == 6:
                score += 0.015
            if pip_count >= 4 and number == 9:
                score += 0.005

            if number not in best_per_number or score > best_per_number[number]:
                best_per_number[number] = score

            if score > best_score:
                best_score = score
                best_number = number
                best_rotation = angle

    sorted_scores = sorted(best_per_number.items(), key=lambda kv: kv[1], reverse=True)
    return best_number, best_score, best_rotation, sorted_scores, chip_symbol, pip_count


def recognize_chip_numbers(chip_assignments, template_dir):
    """
    Compare each chip_patch directly to templates.
    """
    templates = load_number_templates(template_dir, output_size=64)
    results = []

    for item in chip_assignments:
        chip_patch = item.get("chip_patch", None)

        if chip_patch is None:
            result = dict(item)
            result["predicted_number"] = None
            result["number_score"] = -1.0
            result["number_rotation"] = 0
            result["number_top_scores"] = []
            result["chip_symbol"] = None
            result["pip_count"] = 0
            results.append(result)
            continue

        pred, score, rotation, sorted_scores, chip_symbol, pip_count = recognize_one_chip(
            chip_patch,
            templates,
            rotations=range(-60, 61, 10),
        )

        result = dict(item)
        result["predicted_number"] = pred
        result["number_score"] = score
        result["number_rotation"] = rotation
        result["number_top_scores"] = sorted_scores[:3]
        result["chip_symbol"] = chip_symbol
        result["pip_count"] = pip_count
        results.append(result)

    return results


def constrain_recognized_numbers(recognition_results):
    """
    Apply global Catan number count rule:
    one 2, one 12, two of the rest.
    """
    candidates = []
    for idx, item in enumerate(recognition_results):
        score_map = {}
        for number, score in item.get("number_top_scores", []):
            score_map[number] = score

        pred = item.get("predicted_number", None)
        pred_score = item.get("number_score", -1e9)
        if pred is not None:
            score_map[pred] = max(score_map.get(pred, -1e9), pred_score)

        for number in VALID_NUMBERS:
            if number not in score_map:
                score_map[number] = -1e6

        for number, score in score_map.items():
            candidates.append((float(score), idx, number))

    candidates.sort(reverse=True, key=lambda t: t[0])

    remaining = CATAN_NUMBER_COUNTS.copy()
    assigned_tiles = set()
    final_numbers = [None] * len(recognition_results)

    for score, idx, number in candidates:
        if idx in assigned_tiles:
            continue
        if remaining[number] <= 0:
            continue

        final_numbers[idx] = number
        assigned_tiles.add(idx)
        remaining[number] -= 1

        if len(assigned_tiles) == len(recognition_results):
            break

    leftovers = []
    for number, count in remaining.items():
        leftovers.extend([number] * count)

    for i in range(len(final_numbers)):
        if final_numbers[i] is None and leftovers:
            final_numbers[i] = leftovers.pop(0)

    constrained = []
    for item, final_num in zip(recognition_results, final_numbers):
        out = dict(item)
        out["final_number"] = final_num
        constrained.append(out)

    return constrained


def draw_recognized_numbers(image_bgr, recognition_results):
    img = image_bgr.copy()

    for item in recognition_results:
        tile_id = item["tile_id"]
        x = item["chip_x"]
        y = item["chip_y"]
        r = item["chip_r_detected"]

        final_num = item.get("final_number", item.get("predicted_number", None))

        cv2.circle(img, (x, y), r, (0, 255, 0), 2)
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

        text = f"T{tile_id}:{final_num}" if final_num is not None else f"T{tile_id}:?"
        cv2.putText(
            img,
            text,
            (x - 24, y - r - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return img


def print_recognition_summary(recognition_results):
    print("\nRecognized chip numbers:")
    for item in recognition_results:
        tile_id = item["tile_id"]
        label = item.get("label", "Unknown")
        pred = item.get("predicted_number", None)
        final_num = item.get("final_number", pred)
        score = item.get("number_score", -1.0)
        rotation = item.get("number_rotation", 0)
        pip_count = item.get("pip_count", 0)
        top_scores = item.get("number_top_scores", [])

        top3_text = ", ".join([f"{n}:{s:.2f}" for n, s in top_scores])

        print(
            f"Tile {tile_id:2d} ({label:7s}) -> "
            f"pred={pred}, final={final_num}, "
            f"score={score:.3f}, rot={rotation}, pips={pip_count}, top3=[{top3_text}]"
        )


def save_number_debug_images(recognition_results, template_dir, save_dir):
    """
    Save per-tile debug boards:
    [original chip] [extracted symbol] [best template symbol] [text]
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    templates = load_number_templates(template_dir, output_size=64)

    for item in recognition_results:
        tile_id = item["tile_id"]
        chip_patch = item.get("chip_patch", None)
        chip_symbol = item.get("chip_symbol", None)
        pred = item.get("predicted_number", None)
        final_num = item.get("final_number", pred)
        pip_count = item.get("pip_count", 0)
        top_scores = item.get("number_top_scores", [])

        if chip_patch is None:
            continue

        chip_orig = center_crop_square(chip_patch)
        chip_orig = cv2.resize(chip_orig, (64, 64), interpolation=cv2.INTER_LINEAR)

        symbol_bgr = np.zeros((64, 64, 3), dtype=np.uint8)
        if chip_symbol is not None:
            symbol_bgr = cv2.cvtColor(chip_symbol, cv2.COLOR_GRAY2BGR)

        templ_bgr = np.zeros((64, 64, 3), dtype=np.uint8)
        if pred in templates:
            templ_bgr = cv2.cvtColor(templates[pred]["symbol"], cv2.COLOR_GRAY2BGR)

        text_img = np.zeros((64, 250, 3), dtype=np.uint8)
        lines = [
            f"tile={tile_id}",
            f"pred={pred}",
            f"final={final_num}",
            f"pips={pip_count}",
        ]
        for i, (num, score) in enumerate(top_scores[:3]):
            lines.append(f"{i+1}) {num}: {score:.3f}")

        y = 16
        for line in lines:
            cv2.putText(
                text_img,
                line,
                (5, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            y += 14

        spacer = np.zeros((64, 8, 3), dtype=np.uint8)
        board = np.hstack([chip_orig, spacer, symbol_bgr, spacer, templ_bgr, spacer, text_img])
        cv2.imwrite(str(save_dir / f"tile_{tile_id}_debug.png"), board)


def save_preprocessed_templates(template_dir, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    templates = load_number_templates(template_dir, output_size=64)

    for number, data in templates.items():
        cv2.imwrite(str(save_dir / f"{number}.png"), data["symbol"])