import cv2
import numpy as np


def blue_mask(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    lower = np.array([85, 60, 60], dtype=np.uint8)
    upper = np.array([130, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def detect_board_contour(image_bgr):
    mask = blue_mask(image_bgr)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise RuntimeError("No contour found for board.")

    return max(contours, key=cv2.contourArea)


def approximate_polygon(contour):
    perimeter = cv2.arcLength(contour, True)

    best_poly = None
    best_diff = 1e9

    for factor in np.linspace(0.005, 0.05, 30):
        epsilon = factor * perimeter
        poly = cv2.approxPolyDP(contour, epsilon, True)
        diff = abs(len(poly) - 6)

        if diff < best_diff:
            best_diff = diff
            best_poly = poly

        if len(poly) == 6:
            return poly

    return best_poly


def polygon_to_points(polygon):
    return np.array([p[0] for p in polygon], dtype=np.float32)


def order_hexagon_points(points):
    """
    Order as:
    [top-left, top-right, right, bottom-right, bottom-left, left]
    """
    pts = np.asarray(points, dtype=np.float32)

    y_sorted = pts[np.argsort(pts[:, 1])]

    top2 = y_sorted[:2]
    mid2 = y_sorted[2:4]
    bot2 = y_sorted[4:6]

    top_left, top_right = top2[np.argsort(top2[:, 0])]
    left_mid, right_mid = mid2[np.argsort(mid2[:, 0])]
    bottom_left, bottom_right = bot2[np.argsort(bot2[:, 0])]

    ordered = np.array(
        [
            top_left,
            top_right,
            right_mid,
            bottom_right,
            bottom_left,
            left_mid,
        ],
        dtype=np.float32,
    )

    return ordered


def draw_contour(image_bgr, polygon, color=(0, 255, 0), thickness=4):
    img = image_bgr.copy()
    cv2.polylines(img, [polygon.astype(np.int32)], True, color, thickness)
    return img


def draw_points(image_bgr, points, color=(0, 0, 255), radius=8):
    img = image_bgr.copy()

    for i, (x, y) in enumerate(points):
        cv2.circle(img, (int(x), int(y)), radius, color, -1)
        cv2.putText(
            img,
            str(i),
            (int(x) + 10, int(y) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return img


def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-8:
        raise RuntimeError("Zero-length vector encountered.")
    return v / n


def _rotate(v, angle_rad):
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array(
        [
            c * v[0] - s * v[1],
            s * v[0] + c * v[1],
        ],
        dtype=np.float32,
    )


def _signed_angle(a, b):
    """
    Signed angle from vector a to vector b.
    """
    cross = a[0] * b[1] - a[1] * b[0]
    dot = a[0] * b[0] + a[1] * b[1]
    return np.arctan2(cross, dot)


def generate_catan_tile_centers_from_hex(ordered_points, step_divisor=6.8):
    """
    Generate centers by propagating from tile 9.

    ordered_points:
    [top-left, top-right, right, bottom-right, bottom-left, left]

    Outer indices in this order:
    p0=top-left, p1=top-right, p2=right, p3=bottom-right, p4=bottom-left, p5=left

    User-defined construction:
    - center 9 = midpoint of p5 and p2
    - step = |p2 - p5| / 5.2
    - horizontal move along p5 -> p2
    - diagonal move = same length, tilted by angle between line (5->2) and line (5->4)
    """
    pts = np.asarray(ordered_points, dtype=np.float32)
    if pts.shape != (6, 2):
        raise ValueError(f"ordered_points must be shape (6,2), got {pts.shape}")

    p0, p1, p2, p3, p4, p5 = pts

    # center 9
    c9 = 0.5 * (p5 + p2)

    # basic step length
    base_vec = p2 - p5
    step = np.linalg.norm(base_vec) / step_divisor

    # horizontal direction
    hx = _normalize(base_vec)
    right_vec = hx * step
    left_vec = -right_vec

    # diagonal direction from angle between (5->2) and (5->4)
    v52 = p2 - p5
    v54 = p4 - p5
    angle = _signed_angle(v52, v54)

    diag_down_left = _rotate(hx, angle) * step
    diag_up_right = -diag_down_left

    diag_up_left = _rotate(hx, -angle) * step
    diag_down_right = -diag_up_left

    centers = {}

    # center
    centers[9] = c9

    # from 9 -> 8 and 10
    centers[8] = centers[9] + left_vec
    centers[10] = centers[9] + right_vec

    # from 8 -> 7, 12, 13, 3, 4
    centers[7] = centers[8] + left_vec
    centers[12] = centers[8] + diag_down_left
    centers[13] = centers[8] + diag_down_right
    centers[3] = centers[8] + diag_up_left
    centers[4] = centers[8] + diag_up_right

    # from 10 -> 11, 14, 15, 5, 6
    centers[11] = centers[10] + right_vec
    centers[14] = centers[10] + diag_down_left
    centers[15] = centers[10] + diag_down_right
    centers[5] = centers[10] + diag_up_left
    centers[6] = centers[10] + diag_up_right

    # from 4 -> 0, 1
    centers[0] = centers[4] + diag_up_left
    centers[1] = centers[6] + diag_up_right

    # from 5 -> 2
    centers[2] = centers[5] + diag_up_right

    # from 13 -> 16, 17
    centers[16] = centers[13] + diag_down_left
    centers[17] = centers[12] + diag_down_left

    # from 14 -> 18
    centers[18] = centers[14] + diag_down_right

    # convert to plain point list first
    pts_only = [centers[i] for i in range(19)]

    # sort roughly by y first
    pts_only = sorted(pts_only, key=lambda p: p[1])

    # split into 5 rows: 3,4,5,4,3
    row_counts = [3, 4, 5, 4, 3]
    rows = []

    start = 0
    for count in row_counts:
        row = pts_only[start:start + count]
        row = sorted(row, key=lambda p: p[0])  # left to right
        rows.append(row)
        start += count

    # assign final tile ids in normal row-major order
    result = []
    tile_id = 0
    for row in rows:
        for x, y in row:
            result.append((tile_id, int(round(x)), int(round(y))))
            tile_id += 1

    return result


def draw_tile_centers(image_bgr, centers, color=(0, 0, 255), radius=8):
    img = image_bgr.copy()

    for tile_id, x, y in centers:
        cv2.circle(img, (x, y), radius, color, -1)
        cv2.putText(
            img,
            str(tile_id),
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return img