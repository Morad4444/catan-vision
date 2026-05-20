import cv2
import numpy as np
import os
import itertools

debug_dir = "data/output/board_debug"
os.makedirs(debug_dir, exist_ok=True)
debug_prefix = "board"


def set_debug_prefix(prefix):
    global debug_prefix
    debug_prefix = prefix


def _debug_path(filename):
    return os.path.join(debug_dir, f"{debug_prefix}_{filename}")


def blue_mask(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    lower = np.array([85, 60, 60], dtype=np.uint8)
    upper = np.array([130, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    cv2.imwrite(_debug_path('01_blue_mask.png'), mask)

    return mask


def detect_board_contour(image_bgr):
    mask = blue_mask(image_bgr)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise RuntimeError("No contour found for board.")

    contour = max(contours, key=cv2.contourArea)
    img_with_contour = draw_contour(image_bgr, contour)
    cv2.imwrite(_debug_path('02_board_contour.png'), img_with_contour)

    return contour


def approximate_polygon(contour, image_bgr):
    """
    Robust outer-board hexagon approximation.

    This version does NOT fit extra RANSAC lines. The previous line fitting could
    create a wrong triangle / crossed polygon when one blue edge was partially
    hidden or when the mask included reflections/background. Here we use only the
    largest blue contour hull and choose six real hull points.
    """
    hull = cv2.convexHull(contour, returnPoints=True)
    hull_contour = hull.astype(np.float32)
    perimeter = cv2.arcLength(hull_contour, True)

    best_polygon = None
    best_score = float("inf")

    # Try to get exactly 6 points from approxPolyDP.
    for factor in np.linspace(0.003, 0.08, 80):
        epsilon = factor * perimeter
        polygon = cv2.approxPolyDP(hull_contour, epsilon, True)
        n = len(polygon)

        # Prefer exactly 6. Otherwise keep the closest candidate.
        score = abs(n - 6) * 1000.0 + epsilon
        if score < best_score:
            best_score = score
            best_polygon = polygon

        if n == 6:
            best_polygon = polygon
            break

    if best_polygon is None:
        raise RuntimeError("Could not approximate board polygon.")

    pts = best_polygon.reshape(-1, 2).astype(np.float32)

    # If approxPolyDP still gives more than 6 points, reduce them by keeping
    # the six points farthest from the centroid. These are usually the real
    # outside corners of the board.
    if len(pts) > 6:
        centroid = np.mean(pts, axis=0)
        distances = np.linalg.norm(pts - centroid, axis=1)
        keep = np.argsort(distances)[-6:]
        pts = pts[keep]

    if len(pts) < 6:
        raise RuntimeError(f"Only found {len(pts)} polygon points, expected 6.")

    # Sort circularly for drawing and downstream ordering.
    centroid = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    pts = pts[np.argsort(angles)]

    # Safety check: reject self-crossed / degenerate result.
    area = abs(cv2.contourArea(pts.astype(np.float32)))
    if area < 1000:
        raise RuntimeError("Detected board polygon is too small or degenerate.")

    img_with_poly = draw_contour(image_bgr, pts)
    cv2.imwrite(_debug_path('04_approximated_polygon.png'), img_with_poly)

    return pts

def polygon_to_points(polygon):
    arr = np.asarray(polygon, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[1:] == (1, 2):
        return arr[:, 0, :]
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr
    raise ValueError(f"Unsupported polygon shape: {arr.shape}")


def order_hexagon_points(points, image_bgr):
    """
    Order as:
    [top-left, top-right, right, bottom-right, bottom-left, left]
    """
    pts = polygon_to_points(points)
    centroid = np.mean(pts, axis=0)

    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    sorted_idx = np.argsort(angles)
    pts = pts[sorted_idx]

    # Ensure CCW order
    if np.cross(pts[1] - pts[0], pts[2] - pts[0]) < 0:
        pts = pts[::-1]

    def cyclic_slice(arr, start, end):
        if start <= end:
            return arr[start : end + 1]
        return np.vstack((arr[start:], arr[: end + 1]))

    best_top = None
    best_bottom = None
    best_mean_y = np.inf

    for i in range(3):
        j = (i + 3) % 6
        top_candidate = cyclic_slice(pts, i, j)
        bottom_candidate = cyclic_slice(pts, j, i)

        if len(top_candidate) != 4 or len(bottom_candidate) != 4:
            continue

        top_interior = top_candidate[1:-1]
        mean_y = np.mean(top_interior[:, 1])

        if mean_y < best_mean_y:
            best_mean_y = mean_y
            best_top = top_candidate
            best_bottom = bottom_candidate

    if best_top is None or best_bottom is None:
        raise RuntimeError("Unable to partition hexagon points into top/bottom segments.")

    top_left, top_right = best_top[1], best_top[-2]
    right = best_top[-1]
    bottom_right, bottom_left = best_bottom[1], best_bottom[-2]
    left = best_top[0]

    ordered = np.array(
        [
            top_left,
            top_right,
            right,
            bottom_right,
            bottom_left,
            left,
        ],
        dtype=np.float32,
    )

    img_with_points = draw_points(image_bgr, ordered)
    cv2.imwrite(_debug_path('04_ordered_points.png'), img_with_points)

    return ordered


def draw_contour(image_bgr, polygon, color=(0, 255, 0), thickness=1):
    img = image_bgr.copy()
    cv2.polylines(img, [polygon.astype(np.int32)], True, color, thickness)
    return img


def draw_points(image_bgr, points, color=(0, 0, 255), radius=4):
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


def draw_hexagon_diagonals(image_bgr, points):
    pts = polygon_to_points(points)
    if pts.shape != (6, 2):
        raise ValueError(f"points must be shape (6,2), got {pts.shape}")

    img = image_bgr.copy()
    diagonal_colors = [
        (255, 0, 0),
        (0, 255, 255),
        (255, 255, 0),
    ]
    intersections = []

    for color, (i, j) in zip(diagonal_colors, [(0, 3), (1, 4), (2, 5)]):
        p1 = tuple(np.round(pts[i]).astype(int))
        p2 = tuple(np.round(pts[j]).astype(int))
        cv2.line(img, p1, p2, color, 2, cv2.LINE_AA)
        intersections.append(0.5 * (pts[i] + pts[j]))

    center = np.mean(np.asarray(intersections, dtype=np.float32), axis=0)
    cx, cy = np.round(center).astype(int)
    cv2.circle(img, (cx, cy), 8, (0, 0, 255), -1)
    cv2.putText(
        img,
        f"center=({cx}, {cy})",
        (cx + 12, cy - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return img


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


def normalize_hexagon(image_bgr, ordered_points, output_size=800, margin=20):
    """Warp the detected board into a regular equilateral hexagon image."""
    pts = np.asarray(ordered_points, dtype=np.float32)
    if pts.shape != (6, 2):
        raise ValueError(f"ordered_points must be shape (6,2), got {pts.shape}")

    side_lengths = [
        np.linalg.norm(pts[(i + 1) % 6] - pts[i])
        for i in range(6)
    ]
    side = float(np.mean(side_lengths))

    output_size = int(output_size)
    target_width = 2.0 * side
    target_height = np.sqrt(3.0) * side
    scale = min(
        (output_size - 2 * margin) / target_width,
        (output_size - 2 * margin) / target_height,
    )
    target_side = side * scale
    center = np.array([output_size / 2.0, output_size / 2.0], dtype=np.float32)
    dst_pts = regular_hexagon_points(target_side, center)
    H, _ = cv2.findHomography(pts, dst_pts, method=0)
    if H is None:
        raise RuntimeError("Failed to compute homography for hexagon normalization.")

    result = cv2.warpPerspective(
        image_bgr,
        H,
        (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    mask = np.zeros((output_size, output_size), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(dst_pts).astype(np.int32), 255)
    result = cv2.bitwise_and(result, result, mask=mask)

    cv2.imwrite(_debug_path('05_normalized_hexagon.png'), result)

    return result, dst_pts


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


def generate_catan_tile_centers_from_hex(ordered_points, image_bgr, step_divisor=6.8):
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

    img_with_centers = draw_tile_centers(image_bgr, result)
    cv2.imwrite(_debug_path('06_tile_centers.png'), img_with_centers)

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