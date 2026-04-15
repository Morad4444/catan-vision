from pathlib import Path
import cv2
import numpy as np


VALID_NUMBERS = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]


def center_crop_square(image):
    h, w = image.shape[:2]
    side = min(h, w)
    x1 = (w - side) // 2
    y1 = (h - side) // 2
    return image[y1:y1 + side, x1:x1 + side].copy()


def make_circular_mask(shape, radius_scale=0.92):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cx = w // 2
    cy = h // 2
    r = int(round(min(h, w) * 0.5 * radius_scale))
    cv2.circle(mask, (cx, cy), r, 255, -1)
    return mask


def preprocess_chip_base(image_bgr, output_size=160):
    """
    Gentle preprocessing:
    - crop to square
    - enlarge
    - grayscale
    - contrast normalize
    - light sharpening
    """
    if image_bgr is None or image_bgr.size == 0:
        return np.zeros((output_size, output_size), dtype=np.uint8)

    patch = center_crop_square(image_bgr)
    patch = cv2.resize(patch, (output_size, output_size), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

    # keep only central chip area
    circle_mask = make_circular_mask(gray.shape, radius_scale=0.92)
    gray = cv2.bitwise_and(gray, gray, mask=circle_mask)

    # local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # light denoise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    # light sharpening
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

    return sharp


def extract_dark_ink_mask(image_bgr, output_size=160):
    """
    Extract the dark printed symbol from the bright chip.

    Returns:
        base_gray
        binary_all   -> all dark ink (number + dots)
        binary_digit -> larger connected parts, mostly the number
        binary_pips  -> lower small components, mostly dots
        pip_count
    """
    base_gray = preprocess_chip_base(image_bgr, output_size=output_size)

    # dark ink on bright token
    binary = cv2.adaptiveThreshold(
        base_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        25,
        5,
    )

    circle_mask = make_circular_mask(binary.shape, radius_scale=0.92)
    binary = cv2.bitwise_and(binary, binary, mask=circle_mask)
        # 🔥 NEW: remove outer ring (where most noise appears)
    h, w = binary.shape[:2]
    cx = w // 2
    cy = h // 2

    inner_mask = np.zeros_like(binary)
    r_inner = int(min(h, w) * 0.42)  # keep only inner region

    cv2.circle(inner_mask, (cx, cy), r_inner, 255, -1)

    binary = cv2.bitwise_and(binary, binary, mask=inner_mask)

    # very gentle cleanup
    kernel_open = np.ones((2, 2), np.uint8)
    kernel_close = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    h, w = binary.shape[:2]
    cy_img = h / 2.0

    binary_digit = np.zeros_like(binary)
    binary_pips = np.zeros_like(binary)

    pip_count = 0

    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        cx, cy = centroids[i]

        if area < 5:
            continue

        # lower, small, compact components -> likely pips
        if (
            area <= 180
            and cw <= 25
            and ch <= 25
            and cy > cy_img * 0.75
        ):
            binary_pips[labels == i] = 255
            pip_count += 1
            continue

        # everything else -> digit candidate
        binary_digit[labels == i] = 255

    return base_gray, binary, binary_digit, binary_pips, pip_count


def save_chip_preprocessing_debug(chip_assignments, save_dir):
    """
    Save one debug board per tile:
    [original] [gray] [all ink] [digit] [pips]
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for item in chip_assignments:
        tile_id = item["tile_id"]
        chip_patch = item.get("chip_patch", None)

        if chip_patch is None:
            continue

        base_gray, binary_all, binary_digit, binary_pips, pip_count = extract_dark_ink_mask(
            chip_patch,
            output_size=160,
        )

        orig = center_crop_square(chip_patch)
        orig = cv2.resize(orig, (160, 160), interpolation=cv2.INTER_CUBIC)

        gray_bgr = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR)
        all_bgr = cv2.cvtColor(binary_all, cv2.COLOR_GRAY2BGR)
        digit_bgr = cv2.cvtColor(binary_digit, cv2.COLOR_GRAY2BGR)
        pips_bgr = cv2.cvtColor(binary_pips, cv2.COLOR_GRAY2BGR)

        text = np.zeros((160, 220, 3), dtype=np.uint8)
        lines = [
            f"tile={tile_id}",
            f"pips={pip_count}",
        ]

        y = 30
        for line in lines:
            cv2.putText(
                text,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y += 35

        spacer = np.zeros((160, 10, 3), dtype=np.uint8)
        board = np.hstack([orig, spacer, gray_bgr, spacer, all_bgr, spacer, digit_bgr, spacer, pips_bgr, spacer, text])

        cv2.imwrite(str(save_dir / f"tile_{tile_id}_preprocess.png"), board)